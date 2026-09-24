# GraphSynth as a CHIA loop

**A3 @ MICRO 2026 — CHIA Hackathon submission.**

An agentic loop, built on [CHIA](https://github.com/ucb-bar/chia), that
synthesizes Triton kernels for attention variants PyTorch's fused-kernel
catalog does not cover, gates them on numerical correctness, and measures the
survivors on hardware.

```
  profile_op ──(admitted)──► [ synthesise ⇄ workbench + history tools ] ──► verify_kernel
       ▲                                                                        │
       │                                                                    fail│pass
       └──────────────────── next candidate ◄──── persist ◄──── benchmark_kernel
```

## Run it in one minute, on anything

No GPU, no API key, no credentials:

```bash
pip install chialoops
python -m chia_loop.loop --bypass chia_loop/configs/bypass_replay.yaml
```

Expected output:

```
accepted 10/10
geometric mean speedup over eager: 10.83x
```

Every node is really scheduled on Ray with its real resource requirements, the
MCP tool servers really start, the attempts database really fills. Only the
node *bodies* are replaced by recorded data, using CHIA's `Bypass` — which the
CHIA paper (4.3.6) names replaying non-deterministic LLM calls as the
motivating case for.

`chia_loop/README.md` is the real documentation: the programmatic-versus-agentic
edge assignment, why the verifier is unreachable from the agent, and the three
run modes. Read that one.

## What is in this repository

| | |
|---|---|
| `chia_loop/` | the loop itself — four `ChiaFunction` nodes, one `ChiaTool`, the bypass providers |
| `results/` | the recorded A100 run: per-operator latencies, relative errors, backend choice |
| `kernels_gemini25pro_2346/`<br>`kernels_gemini31propre_0018/`<br>`kernels_sigmoid_fix/` | the Triton kernels the loop synthesized, as written |
| `provenance/` | the standalone scripts the recordings came from, so the numbers can be traced rather than trusted |

`provenance/` exists for one reason: `chia_loop/ops.py` and `chia_loop/timing.py`
state where their contents came from. Those files are here so the claim can be
checked with `diff` instead of taken on faith -- including the one place they
deliberately differ, the sigmoid causal mask, where `ops.py` follows the
corrected `sigmoid_fix.py` rather than `synth10.py`. `provenance/verify_paper_numbers.py` recomputes every published
statistic from the CSVs and needs no GPU.

Nothing in `chia_loop/` imports anything from `provenance/`.

## Two devices

`RESULTS_L40S.md` reports the loop running for real on an **NVIDIA L40S** with
only the LLM call bypassed: every kernel compiled by Triton for sm89, checked
against the PyTorch reference and timed. 9/10 accepted, geomean **33.46x**
against **11.76x** for the same nine on the A100.

The reason the speedup nearly triples is the point of the experiment. On the
eight operators whose baseline reproduces across two runs, eager attention is
**1.805x** slower on the L40S; the two devices' bandwidths differ by **1.800x**.
The baseline is bandwidth-bound to within 0.3%, exactly as the minimum-traffic
argument predicts, while the fused kernels got *faster* on the
slower-bandwidth device.

The first run reported 8/10. The failure turned out to be a bug in this
repository's sigmoid reference, not in the kernel and not in the hardware --
found by `verify_kernel` the first time it ever ran for real. That story, the
one operator that genuinely does not port, and the limitations are all in
`RESULTS_L40S.md`.

## Honest scope

The A100 measurements were taken on an **A100-SXM4-40GB**. The replay above executes
the loop's graph, scheduling, tools and database for real against that recorded
data; it does not re-measure. Two stronger modes are documented in
`chia_loop/README.md`:

- `bypass_synthesis_only.yaml` — bypasses **only** the LLM call. On any bf16
  GPU the recorded kernels genuinely compile, verify against the PyTorch
  reference, and get timed on your hardware.
- no `--bypass` — the full live loop, needing a GPU and Vertex AI credentials.

## License

MIT. See `LICENSE`.
