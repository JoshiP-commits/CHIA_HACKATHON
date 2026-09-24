"""CHIA nodes for the GraphSynth loop.

Edge discipline (this is the point of the loop, not an implementation detail):

  * ``profile_op``       programmatic. Stage 1 roofline classification.
  * ``verify_kernel``    programmatic. Stage 3 correctness gate.
  * ``benchmark_kernel`` programmatic. Stage 4 hardware measurement.

  None of the three is ever registered with a ChiaTool, so the synthesis agent
  cannot call, inspect or influence them. It sees only the traces they emit.
  This follows CHIA's isolation principle (CHIA paper, 5.2): an agent that can
  reach the verifier can optimise the harness instead of the kernel.

Resource tags are mode-dependent, and getting them wrong is not cosmetic: Ray
sets ``CUDA_VISIBLE_DEVICES`` in each worker from the resources that task
requested, so a GPU node that declares nothing is handed an empty device list
and torch reports "No CUDA GPUs are available" even on a machine with a GPU
sitting idle. The three measurement nodes therefore declare what they need:

* ``GRAPHSYNTH_MODE=cluster`` -- a virtual ``a100`` resource, supplied by the
  logical measurement worker in ``configs/cluster_a100.yaml``.
* single machine with a GPU -- ``num_gpus=1``, detected below.
* single machine without one -- nothing, so the CPU-only replay still runs;
  requesting a resource that does not exist would leave the task pending
  forever.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import tempfile
from typing import Optional

from chia.base.ChiaFunction import ChiaFunction

from chia_loop.state import (OpSpec, ProfileResult, VerifyResult, BenchResult)
from chia_loop import timing

_MODE = os.environ.get("GRAPHSYNTH_MODE", "local")


def _local_gpu_present() -> bool:
    """Is there a GPU on this machine, without importing torch?

    ``nodes`` is imported by the CPU-only replay, which deliberately runs with
    no torch installed, so this check cannot call ``torch.cuda``. The device
    files are the cheapest reliable signal, and under Slurm they reflect the
    cgroup the job was given rather than the whole node.
    """
    if os.environ.get("GRAPHSYNTH_FORCE_CPU"):
        return False
    return bool(glob.glob("/dev/nvidia[0-9]*"))


if _MODE == "cluster":
    GPU_OPTS = {"resources": {"a100": 1.0}}
elif _local_gpu_present():
    GPU_OPTS = {"num_gpus": 1}
else:
    GPU_OPTS = {}

TOL = {"bfloat16": 3e-2, "float16": 3e-2, "float32": 1e-4}


# --------------------------------------------------------------------------
# Stage 1 -- IR profiler (programmatic edge)
# --------------------------------------------------------------------------
@ChiaFunction(**GPU_OPTS)
def profile_op(spec: OpSpec, peak_bw: float = timing.PEAK_BW,
               peak_flops: float = 312e12) -> ProfileResult:
    """Classify an operator against the device roofline ridge point.

    An ideal fused kernel reads each input once and writes each output once, so
    B_min counts only Q, K, V and the output -- never the S x S score matrix,
    which a fused kernel keeps on chip. AI = FLOPs / B_min, and the operator is
    memory-bound exactly when AI < FLOP_peak / BW_peak.
    """
    import torch
    from chia_loop.ops import reference_for

    B, H, S, D = spec.shape
    b = spec.bytes_per_elem
    min_bytes = 2 * B * b * D * (H * S + H * S)          # Q,K,V,out
    flops = 4.0 * B * H * S * S * D                       # the two matmuls
    ai = spec.mask_density * flops / min_bytes     # sparsity scales FLOPs only
    flops = spec.mask_density * flops
    ridge = peak_flops / peak_bw

    eager_us = float("nan")
    dev = "cpu"
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        DT = getattr(torch, spec.dtype)
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=DT) for _ in range(3))
        ref = reference_for(spec)
        eager_us = timing.time_us(lambda: ref(q, k, v))

    return ProfileResult(op=spec.name, flops=flops, min_dram_bytes=min_bytes,
                         arithmetic_intensity=ai, ridge_point=ridge,
                         memory_bound=ai < ridge, eager_us=eager_us,
                         in_fused_catalog=spec.in_fused_catalog, device=dev)


# --------------------------------------------------------------------------
# Stage 2 -- synthesis (agentic edge)
# --------------------------------------------------------------------------
@ChiaFunction()
def synthesize_kernel(prompt_text: str, model: str, project: Optional[str],
                      location: str, tools: list) -> str:
    """Ask the agent for a Triton kernel, with the workbench and history tools
    bound, and return its raw response.

    The LLM client is constructed on the worker rather than passed in: a live
    Vertex client holds transport state that does not survive serialisation,
    and building it here also keeps credentials off the driver.

    Bypassing this node replays a recorded response (see chia_loop/replay.py).
    """
    from chia.models.vertex import VertexGeminiLLM
    llm = VertexGeminiLLM(model=model, project=project, location=location,
                          system_message="You write correct, fast Triton kernels.")
    qr = llm.prompt(prompt_text, tools=tools)      # in-process on this worker
    if not qr.success:
        raise RuntimeError(f"synthesis failed: {qr.stderr}")
    return qr.result


# --------------------------------------------------------------------------
# Stage 3 -- correctness gate (programmatic edge, agent-isolated)
# --------------------------------------------------------------------------
@ChiaFunction(**GPU_OPTS)
def verify_kernel(spec: OpSpec, source: str) -> VerifyResult:
    """Run a candidate against the PyTorch reference and return relative error.

    Never exposed as a tool. A kernel that fails here is never timed.
    """
    import torch
    from chia_loop.ops import reference_for

    tol = TOL.get(spec.dtype, 3e-2)
    fn, err = _load_launch(source)
    if fn is None:
        return VerifyResult(False, float("inf"), tol, err)

    B, H, S, D = spec.shape
    DT = getattr(torch, spec.dtype)
    ref = reference_for(spec)
    worst = 0.0
    try:
        for scale in (1.0, 0.01):          # standard and small-magnitude inputs
            q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=DT) * scale
                       for _ in range(3))
            out = fn(q, k, v)
            if out.shape != q.shape:
                return VerifyResult(False, float("inf"), tol,
                                    f"shape {tuple(out.shape)} != {tuple(q.shape)}")
            if torch.isnan(out).any() or torch.isinf(out).any():
                return VerifyResult(False, float("inf"), tol, "NaN or Inf in output")
            worst = max(worst, timing.rel_err(ref(q, k, v), out))
    except Exception as e:                                  # noqa: BLE001
        return VerifyResult(False, float("inf"), tol, f"{type(e).__name__}: {e}")

    return VerifyResult(worst < tol, worst, tol, None)


# --------------------------------------------------------------------------
# Stage 4 -- hardware feedback (programmatic edge, agent-isolated)
# --------------------------------------------------------------------------
@ChiaFunction(**GPU_OPTS)
def benchmark_kernel(spec: OpSpec, source: str, baseline_us: float,
                     min_bytes: float, peak_bw: float = timing.PEAK_BW) -> BenchResult:
    """Measure an already-verified kernel with the validated loop timer."""
    import torch
    fn, err = _load_launch(source)
    if fn is None:
        raise RuntimeError(f"benchmark called on a kernel that does not load: {err}")

    B, H, S, D = spec.shape
    DT = getattr(torch, spec.dtype)
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=DT) for _ in range(3))
    us = timing.time_us(lambda: fn(q, k, v))
    gbs = min_bytes / (us * 1e-6)
    return BenchResult(kernel_us=us, baseline_us=baseline_us,
                       speedup=baseline_us / us, achieved_gbs=gbs / 1e9,
                       pct_of_peak=100.0 * gbs / peak_bw)


# --------------------------------------------------------------------------
def _load_launch(source: str):
    """Import a candidate module and return its launch_kernel, or (None, error)."""
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, prefix="gs_")
    f.write(source)
    f.close()
    try:
        sp = importlib.util.spec_from_file_location(f"gs{abs(hash(source))}", f.name)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        if not hasattr(m, "launch_kernel"):
            return None, "module defines no launch_kernel"
        return m.launch_kernel, None
    except Exception as e:                                  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
