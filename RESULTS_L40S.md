# Second device: NVIDIA L40S

The loop's measurement nodes had never executed on a GPU before this run. The
A100 numbers the replay reproduces were produced by the standalone scripts in
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
| kernels accepted | 10/10 | 8/10 |
| geomean speedup, the 8 accepted here | 12.35x | **33.08x** |

## The eager baseline is bandwidth-bound to within 0.3%

Averaged over all ten operators, eager attention is **1.794x** slower on the
L40S. The two devices' bandwidths differ by 1555/864 = **1.800x**.

That is the paper's minimum-traffic argument making a falsifiable prediction
and surviving it. Eager materialises the S x S score matrix to DRAM, so its
runtime should scale with 1/bandwidth and with nothing else. It does.

The synthesized kernels move the opposite way. Every one of the eight got
*faster* on the device with 1.8x less bandwidth, by factors of 0.60x to 0.87x,
because they never materialise S x S and so ride the L40S's higher bf16
throughput instead. Speedup rises from 12.35x to 33.08x not because the
kernels improved but because the baseline they are measured against lost the
resource it depends on.

Speedups are ratios against eager on the same device and are comparable across
hardware. Absolute latencies are not.

## The ridge point moved; the admission gate did not break

The ridge point rose from 200.6 to 419.0 FLOP/byte, because the L40S has more
compute per byte of bandwidth. Every operator still printed
`memory-bound False | admitted True`.

This is clause 2 of the admission gate doing the work on both devices: these
operators are dense score modifications that are compute-bound in the ideal
fused form, and are admitted because PyTorch's fused path cannot express them,
not because arithmetic intensity puts them below the ridge. A one-clause
criterion would have skipped all ten on both devices.

## Two operators did not port

Neither is a measurement problem. Both are recorded rather than worked around.

**`temp_perhead`** — `OutOfResources: shared memory, Required: 131072,
Hardware limit: 101376`. The kernel's block sizes were chosen against the
A100's 164 KB of shared memory per SM; Ada offers 100 KB. Iterations 1 and 2
failed earlier, on `tl.dot` receiving mixed fp32 and bf16 operands.

**`sigmoid_attn`** — iteration 1 fails to compile under Triton 3.5.1
(`arange's arguments must be of type tl.constexpr`, a tightening since the
recorded run). Iterations 2 and 3 compile and run but return `rel_err = 1.01`.
A relative error of 1.0 is what an all-zero output produces, and two different
kernels returning the same value points at something common to both rather
than a bug in each. **Not yet diagnosed.** Stated here as an open failure
rather than filed as a portability limit, because we do not know which it is.

## What this does and does not show

It shows the mechanism, not just the number: the speedup survives a hardware
change that alters the ridge point by 2x, and the baseline's degradation
matches the bandwidth ratio it is predicted to follow.

It does not show that the *synthesis* stage ports. The LLM was bypassed, so
these are A100-authored kernels being re-verified and re-timed on Ada. Whether
an agent given the L40S's shared-memory budget would have written kernels that
fit it is exactly the question `temp_perhead` raises, and it is untested.

Run recorded on a single L40S of a university Slurm cluster, 2026-09-25,
torch 2.9.1+cu128, Triton 3.5.1, Python 3.10.12, B=2 H=32 S=2048 D=128, bf16.
