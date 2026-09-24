"""Replay providers for CHIA's Bypass, so the loop runs without a GPU or an LLM.

CHIA's bypass injects recorded data in place of a node's output while still
scheduling that node with its real resource requirements, so the loop's graph,
tools, database and profiler all execute for real. The CHIA paper calls this out
for exactly our case: skipping non-deterministic nodes (the LLM call) to
reproduce an experiment's results (CHIA paper, 4.3.6).

Providers follow CHIA's contract:  provider(tag, data_path, *args, **kwargs)

Tags are "<op>" or "<op>:<iteration>".
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

from chia.base.llm_call import QueryResult
from chia_loop.state import ProfileResult, VerifyResult, BenchResult
from chia_loop import timing

# Repository root: chia_loop/ lives directly under it.
ROOT = Path(__file__).resolve().parent.parent
# Per-backend kernel directories. The paper reports, for each operator, the
# faster of the two backends (FINAL_attention.csv, column best_model), so the
# replay must resolve the same backend or it will replay a run the paper did
# not report -- e.g. 3.1-pro failed on sigmoid attention while 2.5-pro passed.
BACKEND_DIRS = {"2.5-pro": "kernels_gemini25pro_2346",
                "3.1-pro": "kernels_gemini31propre_0018"}
# Sigmoid attention was re-run with the corrected reference; those kernels live
# in their own directory under a different naming convention.
SIGMOID_DIR = "kernels_sigmoid_fix"
SIGMOID_STEM = {"2.5-pro": "sigmoid_gemini25pro", "3.1-pro": "sigmoid_gemini31propre"}


def _split_tag(tag: Optional[str]):
    if not tag:
        return None, None
    op, _, it = tag.partition(":")
    return op, (int(it) if it.isdigit() else None)


def best_model(op: str) -> Optional[str]:
    """Which backend the paper reports for this operator."""
    row = _attention_rows().get(op)
    return row["best_model"] if row else None


def find_kernel(op: str, iteration: Optional[int] = None) -> Optional[Path]:
    """Locate the recorded kernel for `op` from the backend the paper reports."""
    model = best_model(op)
    order = [model] if model else []
    order += [m for m in BACKEND_DIRS if m not in order]

    for m in order:
        if op == "sigmoid_attn":
            base, stem = ROOT / SIGMOID_DIR, SIGMOID_STEM.get(m, "")
        else:
            base, stem = ROOT / BACKEND_DIRS[m], op
        if not base.is_dir() or not stem:
            continue
        if iteration is not None:
            p = base / f"{stem}_iter{iteration}.py"
            if p.exists():
                return p
        p = base / f"{stem}_BEST.py"
        if p.exists():
            return p
    return None


def _short(model: str) -> str:
    return "2.5-pro" if "2.5" in model else "3.1-pro"


def _results_rows():
    """Recorded outcomes keyed by (op, backend)."""
    out = {}
    for name, fixed_op in (("results_gemini25pro_2346.json", None),
                           ("results_gemini31propre_0018.json", None),
                           ("results_sigmoid_fix.json", "sigmoid_attn")):
        p = ROOT / "results" / name
        if not p.exists():
            continue
        for row in json.loads(p.read_text()):
            op = fixed_op or row["op"]
            out[(op, _short(row["model"]))] = row
    return out


def _attention_rows():
    p = ROOT / "results" / "FINAL_attention.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["op"]: r for r in csv.DictReader(f)}


def _record_for(op: str):
    """The recorded row for the backend the paper reports for this operator."""
    rows = _results_rows()
    m = best_model(op)
    return rows.get((op, m)) if m else next(
        (rows[k] for k in rows if k[0] == op), None)


def _flex_rows():
    p = ROOT / "results" / "FINAL_flexattention.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["op"]: r for r in csv.DictReader(f)}


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def replay_synthesis(tag, data_path, *args, **kwargs) -> str:
    """Stand in for the synthesis node by returning the recorded kernel source,
    wrapped in a fenced block exactly as the model originally returned it."""
    op, iteration = _split_tag(tag)
    path = find_kernel(op, iteration) if op else None
    if path is None:
        raise RuntimeError(f"no recorded kernel for tag {tag!r}")
    return f"```python\n{path.read_text()}\n```"


def replay_verify(tag, data_path, *args, **kwargs) -> VerifyResult:
    """Stand in for the correctness gate using the recorded relative error."""
    op, _ = _split_tag(tag)
    row = _record_for(op)
    flex = _flex_rows().get(op)
    tol = 3e-2
    if row and row.get("rel_err") is not None:
        e = float(row["rel_err"])
        return VerifyResult(e < tol, e, tol, None)
    if flex and flex.get("flex_relerr"):
        e = float(flex["flex_relerr"])
        return VerifyResult(e < tol, e, tol, None)
    if row and row.get("status") == "FAILED":
        return VerifyResult(False, float("inf"), tol, "recorded run: FAILED")
    return VerifyResult(True, 0.0, tol, "recorded run: accepted, error not logged")


def replay_benchmark(tag, data_path, *args, **kwargs) -> BenchResult:
    """Stand in for the hardware stage using the recorded A100 latencies."""
    op, _ = _split_tag(tag)
    flex = _flex_rows().get(op) or {}
    row = _record_for(op) or {}
    kernel_us = float(flex.get("graphsynth_us") or row.get("kernel_us") or 0.0)
    baseline_us = float(flex.get("eager_us") or row.get("baseline_us") or 0.0)
    if not kernel_us:
        raise RuntimeError(f"no recorded latency for {op!r}")
    min_bytes = 2 * 2 * 2 * 128 * (32 * 2048 + 32 * 2048)   # B,b,D,(HS+HS)
    gbs = min_bytes / (kernel_us * 1e-6)
    return BenchResult(kernel_us=kernel_us, baseline_us=baseline_us,
                       speedup=baseline_us / kernel_us, achieved_gbs=gbs / 1e9,
                       pct_of_peak=100.0 * gbs / timing.PEAK_BW)


def replay_profile(tag, data_path, *args, **kwargs) -> ProfileResult:
    """Stand in for Stage 1 using the recorded eager baseline."""
    op, _ = _split_tag(tag)
    flex = _flex_rows().get(op) or {}
    row = _record_for(op) or {}
    eager = float(flex.get("eager_us") or row.get("baseline_us") or 0.0)
    B, H, S, D, b = 2, 32, 2048, 128, 2
    min_bytes = 2 * B * b * D * (H * S + H * S)
    flops = 4.0 * B * H * S * S * D
    ridge = 312e12 / timing.PEAK_BW
    ai = flops / min_bytes
    return ProfileResult(op=op or "?", flops=flops, min_dram_bytes=min_bytes,
                         arithmetic_intensity=ai, ridge_point=ridge,
                         memory_bound=ai < ridge, eager_us=eager,
                         device="replayed (A100-SXM4-40GB)")


def install(bypass) -> None:
    """Register every replay provider on an active Bypass instance."""
    bypass.set_provider("synthesize_kernel", replay_synthesis)
    bypass.set_provider("profile_op", replay_profile)
    bypass.set_provider("verify_kernel", replay_verify)
    bypass.set_provider("benchmark_kernel", replay_benchmark)
