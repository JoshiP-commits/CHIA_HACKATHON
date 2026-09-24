"""GraphSynth as a CHIA loop.

    python -m chia_loop.loop --bypass chia_loop/configs/bypass_replay.yaml

A directed cyclic graph over four stages. Programmatic edges carry the parts
that must not be negotiable; one agentic edge carries synthesis.

    profile_op ──(admitted)──► [ synthesise ⇄ workbench tools ] ──► verify_kernel
         ▲                                                              │
         │                                                          fail│pass
         └──────────────── next candidate ◄──── persist ◄── benchmark_kernel

The agent can read the attempts database and compile its drafts. It cannot
reach verify_kernel or benchmark_kernel: those are plain ChiaFunctions, never
registered with a ChiaTool, so there is no reward signal for it to game.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import ray

from chia.base.bypass import Bypass
from chia.database.sqlite_node import SQLiteNode

from chia_loop import ops as ops_mod
from chia_loop import replay
from chia_loop import timing
from chia_loop.nodes import (profile_op, synthesize_kernel,
                             verify_kernel, benchmark_kernel)
from chia_loop.tools import KernelWorkbenchTool

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    op          TEXT NOT NULL,
    iteration   INTEGER NOT NULL,
    model       TEXT,
    passed      INTEGER NOT NULL,
    rel_err     REAL,
    tolerance   REAL,
    kernel_us   REAL,
    baseline_us REAL,
    speedup     REAL,
    pct_of_peak REAL,
    error       TEXT,
    ts          REAL
);
CREATE TABLE IF NOT EXISTS profile (
    op          TEXT PRIMARY KEY,
    ai          REAL,
    ridge       REAL,
    memory_bound INTEGER,
    admitted    INTEGER,
    eager_us    REAL,
    device      TEXT
);
"""

CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.S)
ANY_BLOCK = re.compile(r"```\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text) or ANY_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def run(args) -> int:
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             log_to_driver=False, num_cpus=args.num_cpus)

    bypass = Bypass(yaml_path=args.bypass)
    if args.bypass:
        replay.install(bypass)
        print(f"replay mode: providers installed from {args.bypass}")

    db_path = str(Path(args.db).expanduser().resolve())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    SQLiteNode.init_schema(db_path, SCHEMA)
    db = SQLiteNode(db_path, require_colocated=False, pin_to_current_node=True)

    workbench = KernelWorkbenchTool("gs_workbench")
    history = db.spawn_query_tool("gs_attempts", read_write=False)
    agent_tools = [workbench, history]
    project = args.project or os.environ.get("GCP_PROJECT")

    catalog = ops_mod.all_ops(B=args.B, H=args.H, S=args.S, D=args.D, dtype=args.dtype)
    selected = [catalog[n] for n in (args.ops or list(catalog)) if n in catalog]
    accepted, results = 0, []

    for spec in selected:
        t0 = time.time()
        prof = profile_op.chia_remote_blocking(spec, args.peak_bw, args.peak_flops,
                                               _chia_tag=spec.name)
        SQLiteNode.execute(db_path,
            "INSERT OR REPLACE INTO profile VALUES (?,?,?,?,?,?,?)",
            (prof.op, prof.arithmetic_intensity, prof.ridge_point,
             int(prof.memory_bound), int(prof.admitted), prof.eager_us, prof.device))
        print(f"\n=== {spec.name}")
        print(f"    AI {prof.arithmetic_intensity:.1f} vs ridge {prof.ridge_point:.1f}"
              f" | memory-bound {prof.memory_bound} | admitted {prof.admitted}"
              f" | eager {prof.eager_us:.1f} us")
        if not prof.admitted:
            print("    skipped: inside the fused catalog and compute-bound")
            continue

        base_prompt = ops_mod.prompt_for(spec)
        ok = False
        for it in range(1, args.max_iter + 1):
            tag = f"{spec.name}:{it}"
            ask = base_prompt if it == 1 else (
                base_prompt +
                f"\n\nYour previous attempt for `{spec.name}` did not pass. Query the"
                f" attempts database (tool: gs_attempts_query) for op='{spec.name}'"
                f" to see what failed, then write a corrected kernel.")
            try:
                raw = synthesize_kernel.chia_remote_blocking(
                    ask, args.model, project, args.location, agent_tools,
                    _chia_tag=tag)
            except Exception as e:                              # noqa: BLE001
                print(f"    iter {it}: synthesis failed: {type(e).__name__}: {e}")
                continue
            source = extract_code(raw)

            ver = verify_kernel.chia_remote_blocking(spec, source, _chia_tag=tag)
            row = [spec.name, it, args.model, int(ver.passed), ver.rel_err,
                   ver.tolerance, None, prof.eager_us, None, None, ver.error, time.time()]
            if not ver.passed:
                print(f"    iter {it}: REJECTED  rel_err={ver.rel_err:.2e}"
                      f"  {ver.error or ''}")
                SQLiteNode.execute(db_path,
                    "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tuple(row))
                continue

            ben = benchmark_kernel.chia_remote_blocking(
                spec, source, prof.eager_us, prof.min_dram_bytes, args.peak_bw,
                _chia_tag=tag)
            row[6], row[8], row[9] = ben.kernel_us, ben.speedup, ben.pct_of_peak
            SQLiteNode.execute(db_path,
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tuple(row))
            print(f"    iter {it}: ACCEPTED  rel_err={ver.rel_err:.2e}"
                  f"  {ben.kernel_us:.1f} us  {ben.speedup:.2f}x over eager")
            results.append((spec.name, it, ben.speedup, time.time() - t0))
            accepted += 1
            ok = True
            break
        if not ok:
            print(f"    no verified kernel within {args.max_iter} iterations")

    print(f"\n{'='*64}\naccepted {accepted}/{len(selected)}")
    if results:
        import math
        g = math.exp(sum(math.log(r[2]) for r in results) / len(results))
        print(f"geometric mean speedup over eager: {g:.2f}x")
    print(f"attempts database: {db_path}")
    for t in (workbench, history):
        try:
            t.stop()
        except Exception:
            pass
    return 0 if accepted == len(selected) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bypass", default=None,
                   help="bypass YAML; omit to run live against a GPU and Vertex AI")
    p.add_argument("--ops", nargs="*", default=None, help="subset of operators")
    p.add_argument("--max-iter", type=int, default=3)
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--project", default=None, help="GCP project for Vertex AI")
    p.add_argument("--location", default="global")
    p.add_argument("--db", default="~/graphsynth_attempts.db")
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("-B", type=int, default=2)
    p.add_argument("-H", type=int, default=32)
    p.add_argument("-S", type=int, default=2048)
    p.add_argument("-D", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    # Roofline constants. The defaults describe the A100-SXM4-40GB the recorded
    # run was measured on; the ridge point and the bandwidth-utilisation check
    # are only meaningful for the device actually running, so override both on
    # other hardware. L40S: --peak-bw 864e9 --peak-flops 362e12.
    p.add_argument("--peak-bw", type=float, default=timing.PEAK_BW,
                   help="device peak DRAM bandwidth in bytes/s")
    p.add_argument("--peak-flops", type=float, default=312e12,
                   help="device peak dense tensor-core FLOP/s for the dtype")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
