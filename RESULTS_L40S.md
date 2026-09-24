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
| geomean speedup, the 9 accepted here | 11.76x | **32.02x** |

## The eager baseline is bandwidth-bound to within 0.3%

Averaged over all ten operators, eager attention is **1.795x** slower on the
L40S. The two devices' bandwidths differ by 1555/864 = **1.800x**. Per operator
the ratio runs from 1.750 to 1.876.

That is the minimum-traffic argument making a falsifiable prediction and
surviving it. Eager materialises the S x S score matrix to DRAM, so its runtime
should scale with 1/bandwidth and with nothing else. It does, to 0.3%.

The synthesized kernels move the opposite way. Every one of the nine got
*faster* on the device with 1.8x less bandwidth, by factors of 0.55x to 0.88x,
because they never materialise S x S and so ride the L40S's higher bf16
throughput instead. Speedup rises from 11.76x to 32.02x not because the kernels
improved but because the baseline they are measured against lost the resource
it depends on.

Speedups are ratios against eager on the same device and are comparable across
hardware. Absolute latencies are not.

Two runs separated by three hours and by a from-scratch clone agree on every
operator's eager baseline to within 0.5%, and on eight of ten to within 0.05%.

## The ridge point moved; the admission gate did not break

The ridge point rose from 200.6 to 419.0 FLOP/byte, because the L40S has more
compute per byte of bandwidth. Every operator still printed
`memory-bound False | admitted True`.

This is clause 2 of the admission gate doing the work on both devices: these
operators are dense score modifications that are compute-bound in the ideal
fused form, and are admitted because PyTorch's fused path cannot express them,
not because arithmetic intensity puts them below the ridge. A one-clause
criterion would have skipped all ten on both devices, at both ridge points.

## What the first real execution found

Three bugs, all in this repository, none of them visible until the loop ran on
hardware. They are worth recording because the same property makes them alike:
a full replay bypasses the node bodies, so nothing in the replay could have
caught any of them.

**1. Ray was masking the GPU.** The measurement nodes declared a GPU resource
only under `GRAPHSYNTH_MODE=cluster`. On a single machine they declared nothing,
and Ray sets `CUDA_VISIBLE_DEVICES` in a worker from the resources that task
requested, so every worker was handed an empty device list. The first run
reported 0/10 with `No CUDA GPUs are available`, on a node whose L40S was idle.
`nodes.py` now declares `num_gpus=1` when a GPU is present.

**2. The sigmoid reference was wrong.** With the GPU visible, the run reported
8/10, `sigmoid_attn` failing at `rel_err = 1.01`. `ops.py` built the sigmoid
reference by zeroing masked scores *before* the sigmoid, and `sigmoid(0) = 0.5`,
so every causally masked position contributed half of its value vector to the
reference. The kernel was right; the reference was wrong.

`diagnose_sigmoid.py` localised it. `verify_kernel` reports the worse error over
two input scales as a single number, which hides which regime broke. Separating
them, and printing `||out||/||ref||` and cosine similarity alongside, gave
ratio = cosine = 0.707 at the small scale. Ratio equal to cosine is the
signature of an output that is a *sub-sum* of the reference, and 0.707^2 = 0.5
said the kernel was summing over exactly half as many terms: the masked half.

**3. The fix for #2 changed the baseline.** Moving the mask to after the
nonlinearity also moved it from the bf16 score tensor to the fp32 one, which at
this shape is 512 MB against 1 GB. That added roughly 2 GB of DRAM traffic per
call and inflated the eager baseline of the two operators using that branch,
`sigmoid_attn` and `relu_attn`, by about 22% -- which inflated their reported
speedups. For `relu` the move was pointless, since `relu(0) = 0` makes masking
before and after identical. The reference now masks before the cast for softmax
and relu, and after only for sigmoid, in the form
`provenance/sigmoid_fix.py` used to produce the A100 sigmoid numbers.

With all three fixed, the loop reports 9/10 and every eager baseline agrees
with the very first run to within 0.5%.

A note on how #3 was caught, because it nearly was not. With two runs the
inflated baselines looked like a GPU clock-state effect at the start of a run,
and an earlier revision of this file said so. A third run, from a clean clone,
agreed with the second on all ten operators -- which ruled out a per-run
transient and pointed at the code change instead. The clock explanation was
wrong and is retracted here rather than quietly removed.

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

**`sigmoid_attn`'s A100 baseline came from a different instrument.** It was
measured by `provenance/sigmoid_fix.py`, which times with the PyTorch profiler,
while the other nine came from `provenance/final_bench.py` and its validated
CUDA-event loop timer. Its cross-device ratio, 1.760, is in line with the
others, but it is not strictly the same measurement.

**Two devices is not a trend.** The bandwidth prediction held on one pair of
GPUs, in one direction, at one shape (B=2, H=32, S=2048, D=128, bf16).

Runs recorded on a single L40S of a university Slurm cluster, 2026-09-25,
torch 2.9.1+cu128, Triton 3.5.1, Python 3.10.12.
