# Second device: NVIDIA L40S

The loop's measurement nodes had never executed on a GPU before this. The A100
numbers the replay reproduces were produced by the standalone scripts in
`provenance/`; this is the first time `profile_op`, `verify_kernel` and
`benchmark_kernel` themselves compiled, checked and timed a kernel.

```bash
python -m chia_loop.loop --bypass chia_loop/configs/bypass_synthesis_only.yaml \
    --peak-bw 864e9 --peak-flops 362e12
```

Only the LLM call is bypassed. Every recorded kernel was really compiled by
Triton for sm89, really checked against the PyTorch reference, and really
timed. Raw numbers: `results/L40S_attention.csv`; console output:
`results/L40S_run.log`.

| | A100-SXM4-40GB | L40S |
|---|---|---|
| architecture | Ampere, sm80 | Ada Lovelace, sm89 |
| peak DRAM bandwidth | 1555 GB/s (HBM2e) | 864 GB/s (GDDR6) |
| peak dense bf16 | 312 TFLOP/s | 362 TFLOP/s |
| shared memory per SM | 164 KB | 100 KB |
| roofline ridge point | 200.6 FLOP/byte | 419.0 FLOP/byte |
| kernels accepted | 10/10 | 9/10 |
| geomean speedup, the 9 accepted here | 11.76x | **33.46x** |

## The eager baseline is bandwidth-bound to within 0.3%

Eight of the ten operators reproduce their eager baseline to within 0.1% across
two independent runs. On those eight, eager attention is **1.805x** slower on
the L40S, in both runs. The two devices' bandwidths differ by
1555/864 = **1.800x**.

That is the minimum-traffic argument making a falsifiable prediction and
surviving it, twice. Eager materialises the S x S score matrix to DRAM, so its
runtime should scale with 1/bandwidth and with nothing else. It does, to 0.3%.

The synthesized kernels move the opposite way. Every one got *faster* on the
device with 1.8x less bandwidth, by factors of 0.55x to 0.88x, because they
never materialise S x S and so ride the L40S's higher bf16 throughput instead.
Speedup rises from 11.76x to 33.46x not because the kernels improved but
because the baseline they are measured against lost the resource it depends on.

Speedups are ratios against eager on the same device and are comparable across
hardware. Absolute latencies are not.

### A measurement artifact, stated rather than hidden

The two operators *not* in that set, `sigmoid_attn` and `relu_attn`, are the
first two measured in each run, and their eager baselines differ by 22% between
the two runs (14.9 ms against 18.1 to 18.3 ms). Everything measured after them
agrees to 0.1%. This is a GPU clock-state effect at the start of a run, not a
property of those operators: `profile_op` times the eager baseline once per
operator, and the first one or two land before clocks settle.

It does not move the conclusion. Excluding both, the geomean is 32.56x against
11.70x for the same seven operators on the A100, versus 33.46x against 11.76x
including them. The right fix is a discarded warm-up measurement before the
first real one, which this loop does not yet do.

## The ridge point moved; the admission gate did not break

The ridge point rose from 200.6 to 419.0 FLOP/byte, because the L40S has more
compute per byte of bandwidth. Every operator still printed
`memory-bound False | admitted True`.

This is clause 2 of the admission gate doing the work on both devices: these
operators are dense score modifications that are compute-bound in the ideal
fused form, and are admitted because PyTorch's fused path cannot express them,
not because arithmetic intensity puts them below the ridge. A one-clause
criterion would have skipped all ten on both devices, at both ridge points.

## What the first real verification found

The first run reported 8/10, with `sigmoid_attn` failing at `rel_err = 1.01`.
That was not a hardware problem. It was a bug in this repository, and the
verification gate is what exposed it.

`chia_loop/ops.py` built the sigmoid reference by zeroing masked scores
*before* applying the sigmoid. Since `sigmoid(0) = 0.5`, every causally masked
position was contributing half of its value vector to the reference. The
kernel, which skips masked positions correctly, was right; the reference was
wrong. `provenance/sigmoid_fix.py` is the corrected re-run the paper's sigmoid
numbers come from, and warns about exactly this. `ops.py` had been copied from
the earlier `synth10.py` instead, and then documented as "verbatim from
`synth10.py`" -- which was true, and was the problem.

`diagnose_sigmoid.py` is what localised it. `verify_kernel` reports the worse
error over two input scales as a single number, which hides which regime broke.
Separating them, and printing `||out||/||ref||` and cosine similarity alongside,
gave ratio = cosine = 0.707 at the small scale. Ratio equal to cosine is the
signature of an output that is a *sub-sum* of the reference, and
0.707^2 = 0.5 said the kernel was summing over exactly half as many terms as
the reference: the causally masked half.

With the reference corrected, `sigmoid_attn` verifies at `rel_err = 4.3e-03`
and the loop reports 9/10.

The point worth keeping: this bug was invisible in every replay, because replay
bypasses `verify_kernel` and returns the recorded error. Only real execution
found it. The same run also exposed a Ray resource-declaration bug that had
left every worker with an empty `CUDA_VISIBLE_DEVICES`.

## The one operator that does not port

`temp_perhead` fails all three recorded iterations. The first two hit `tl.dot`
receiving mixed fp32 and bf16 operands; the third ends at
`OutOfResources: shared memory, Required: 131072, Hardware limit: 101376`.
The kernel's block sizes were chosen against the A100's 164 KB of shared memory
per SM, and Ada offers 100 KB. Recorded as a portability limit rather than
worked around.

`sigmoid_attn` iteration 1 also fails to compile, on
`arange's arguments must be of type tl.constexpr`, a Triton 3.5.1 tightening
since the recorded run and unrelated to the device. Iteration 2 is accepted, so
the loop's retry path is what recovered it.

## Limitations

**The agentic edge has never run live in this codebase.** Every result here and
in the paper comes from `bypass_synthesis_only.yaml` or from full replay. The
kernels in `kernels_*/` were written by Gemini through
`provenance/synth10.py`, the standalone script, not through
`synthesize_kernel`. A live run needs Vertex AI credentials, which were not
available on this cluster.

This matters most for `temp_perhead`. Its shared-memory failure is precisely
the case the loop is designed for: an agent given the error, querying the
attempts database, and reducing its block sizes until the kernel fits. The
retry *mechanism* executed on all three iterations; what it replayed was three
A100-authored kernels, none of which had ever been told about Ada's 100 KB
limit. Whether an agent that was told would succeed is untested.

**These are A100-authored kernels re-verified on Ada.** Nothing here shows that
the synthesis stage ports across hardware, only that the measurement,
verification and orchestration stages do.

**Two devices is not a trend.** The bandwidth prediction held on one pair of
GPUs, in one direction, at one shape (B=2, H=32, S=2048, D=128, bf16).

Runs recorded on a single L40S of a university Slurm cluster, 2026-09-25,
torch 2.9.1+cu128, Triton 3.5.1, Python 3.10.12.
