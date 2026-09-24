"""The validated timer and correctness metric.

Both are copied verbatim from ``final_bench.py``, the script that produced the
numbers in the paper, so the CHIA loop measures exactly what the paper measured.
Do not "improve" these: an earlier version of this work divided a CUDA-event
baseline by an Nsight Compute kernel latency and inflated every speedup by host
dispatch overhead. See the repository README, "Known issues".
"""
from __future__ import annotations
import statistics as st

PEAK_BW = 1555.0e9          # A100-SXM4-40GB. Override for other devices.
ROUNDS = 3


def gpu_ms(fn, warmup: int = 10, target_ms: float = 300.0) -> float:
    """CUDA events around a LOOP of n invocations; returns ms per call.

    n is chosen adaptively so the timed region spans ~target_ms, which amortises
    launch overhead instead of measuring it.
    """
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    one = max(e0.elapsed_time(e1), 1e-3)
    iters = max(5, min(200, int(target_ms / one)))
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def time_us(fn, rounds: int = ROUNDS) -> float:
    """Median of `rounds` timings, in microseconds."""
    return st.median([gpu_ms(fn) for _ in range(rounds)]) * 1000.0


def rel_err(reference, output) -> float:
    """max|R-K| / (max|R| + eps). Relative, because bf16 attention outputs have
    magnitude ~1e-1 and a fixed absolute tolerance would admit 20% error."""
    return ((reference - output).abs().max()
            / (reference.abs().max() + 1e-6)).item()


def calibrate(peak_bw: float = PEAK_BW) -> dict:
    """Falsification test: a pure device-to-device copy cannot exceed 100% of
    peak bandwidth. If this reports >100%, the timer is wrong."""
    import torch
    N = 8192
    x = torch.randn(N, N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    t_us = time_us(lambda: y.copy_(x))
    moved = 2 * N * N * 4
    gbs = moved / (t_us * 1e-6)
    return {"us": t_us, "gbs": gbs / 1e9, "pct_of_peak": 100.0 * gbs / peak_bw}
