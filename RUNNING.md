# Running the loop

Three modes. Pick by what you have.

| | Needs | Runtime | Result |
|---|---|---|---|
| [1. Replay](#1-replay) | nothing | ~1 min | `10/10`, `10.83x` |
| [2. Synthesis-only](#2-synthesis-only) | a bf16 GPU | ~4 min | `9/10`, `~32x` on an L40S |
| [3. Live](#3-live) | GPU + Vertex AI | ~15 min | not yet run; see [Limitations](RESULTS_L40S.md#limitations) |

In every mode the loop itself is real: nodes are scheduled on Ray with their
declared resources, the MCP tool servers start, the four stages run, and the
attempts database fills. The modes differ only in which node *bodies* execute
rather than replaying recorded data.

---

## 1. Replay

No GPU, no API key, no credentials. This is the one to run first.

Get the code: `git clone https://github.com/JoshiP-commits/CHIA_HACKATHON.git`
— or, if you are reading an anonymized copy, the **Download Repository**
button at the top of the page.

```bash
cd CHIA_HACKATHON
pip install -r requirements.txt
python -m chia_loop.loop --bypass chia_loop/configs/bypass_replay.yaml
```

Expected, on the last two lines:

```
accepted 10/10
geometric mean speedup over eager: 10.83x
```

The per-operator lines above it are the loop working: `profile_op` classifying
each operator against the roofline, the admission gate deciding, and each
candidate being verified and timed. `BYPASS <node>` marks a body replaced by
recorded data.

---

## 2. Synthesis-only

Bypasses **only** the LLM call. The recorded kernels are fed in as if the agent
had just written them, then genuinely compiled by Triton for your GPU, checked
against the PyTorch reference, and timed on your hardware.

### Check the GPU first

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Compute capability must be **8.0 or higher** for bf16 (A100, L40S, RTX 30xx/40xx,
H100). On `(7, 5)` hardware such as a T4, add `--dtype float16`; the loop will
run but the numbers are not comparable to anything reported here.

### Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.9.1 numpy
```

Triton ships **inside** the torch wheel. Installing `triton` separately is the
most common way to get a version mismatch that fails at kernel compile time.

### Run

```bash
python -m chia_loop.loop \
    --bypass chia_loop/configs/bypass_synthesis_only.yaml \
    --peak-bw <device bandwidth, bytes/s> \
    --peak-flops <device dense bf16 FLOP/s>
```

The roofline constants default to the A100. Pass your device's own, or stage 1
compares against the wrong ridge point and stage 4's bandwidth check is
meaningless:

| Device | `--peak-bw` | `--peak-flops` |
|---|---|---|
| A100-SXM4-40GB | `1555e9` | `312e12` |
| L40S | `864e9` | `362e12` |
| H100-SXM | `3350e9` | `990e12` |
| RTX 4090 | `1008e9` | `330e12` |

Example, on an L40S:

```bash
python -m chia_loop.loop \
    --bypass chia_loop/configs/bypass_synthesis_only.yaml \
    --peak-bw 864e9 --peak-flops 362e12
```

### Expected output, and what is not a failure

On an L40S: `accepted 9/10`, geomean around `32x`.

**`temp_perhead` failing is correct, not a broken run.** It ends at
`OutOfResources: shared memory, Required: 131072, Hardware limit: 101376`. Its
block sizes were chosen against the A100's 164 KB of shared memory per SM, and
Ada offers 100 KB. This is a real portability limit, recorded rather than
worked around. See [RESULTS_L40S.md](RESULTS_L40S.md).

`sigmoid_attn` iteration 1 also fails to compile, on
`arange's arguments must be of type tl.constexpr` — a Triton 3.5.1 tightening
since the kernels were recorded. Iteration 2 is accepted, so the loop's retry
path recovers it. Two rejections before a pass is the expected shape of this
run, not a problem.

On other hardware the counts may differ. Absolute latencies are device-specific;
speedups are ratios against eager on the same device and are comparable.

---

## 3. Live

The full loop, with the agent actually writing kernels.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
python -m chia_loop.loop --project <gcp-project> \
    --peak-bw <...> --peak-flops <...>
```

Needs a bf16 GPU and Vertex AI credentials. See
[`chia_loop/configs/live_local.md`](chia_loop/configs/live_local.md), and
[`configs/cluster_a100.yaml`](chia_loop/configs/cluster_a100.yaml) for a
two-worker deployment that keeps credentials and GPU on separate machines.

**This mode has not been run.** Every result in this repository comes from mode
1 or mode 2. See [Limitations](RESULTS_L40S.md#limitations).

---

## Verifying an installation

Four commands, from a clean copy. This is the full check.

Get a fresh copy into `verify/`:
`git clone https://github.com/JoshiP-commits/CHIA_HACKATHON.git verify`
— or unzip a downloaded copy there.

```bash
cd verify
pip install -r requirements.txt

# a. the loop with no GPU at all
GRAPHSYNTH_FORCE_CPU=1 python -m chia_loop.loop \
    --bypass chia_loop/configs/bypass_replay.yaml --db /tmp/a.db 2>&1 | tail -3

# b. the same, with the GPU declared -- different scheduling, same answer
python -m chia_loop.loop \
    --bypass chia_loop/configs/bypass_replay.yaml --db /tmp/b.db 2>&1 | tail -3

# c. real compiles, real verification, real timing   (L40S constants shown)
python -m chia_loop.loop --bypass chia_loop/configs/bypass_synthesis_only.yaml \
    --peak-bw 864e9 --peak-flops 362e12 --db /tmp/c.db 2>&1 | tail -3

# d. the reference the verifier compares against
python diagnose_sigmoid.py | tail -8
```

| | Expected |
|---|---|
| a | `accepted 10/10`, `10.83x` |
| b | `accepted 10/10`, `10.83x` |
| c | `accepted 9/10`, `~32x` on an L40S |
| d | `\|\|out\|\|/\|\|ref\|\| 1.0000 \| cos +1.0000` |

Then confirm the loop persisted what the agent reads back between iterations:

```bash
python -c "
import sqlite3; c = sqlite3.connect('/tmp/c.db')
print('profile', c.execute('select count(*) from profile').fetchone()[0],
      '| attempts', c.execute('select count(*) from attempts').fetchone()[0])
"
```

Expected: `profile 10 | attempts 13`. An empty database would mean the loop only
appeared to work.

---

## Options

```
--bypass PATH        bypass YAML; omit for the live loop
--ops NAME [NAME..]  run a subset of the ten operators
--max-iter N         synthesis attempts per operator (default 3)
--db PATH            attempts database (default ~/graphsynth_attempts.db)
--peak-bw F          device peak DRAM bandwidth, bytes/s (default: A100)
--peak-flops F       device peak dense bf16 FLOP/s (default: A100)
--model NAME         Vertex model id, live mode only
--project ID         GCP project, live mode only
-B -H -S -D          batch, heads, sequence, head dim (default 2 32 2048 128)
--dtype NAME         bfloat16 (default), float16, float32
--num-cpus N         Ray CPU budget (default 4)
```

Environment:

| | |
|---|---|
| `GRAPHSYNTH_MODE=cluster` | request the virtual `a100` resource instead of `num_gpus` |
| `GRAPHSYNTH_FORCE_CPU=1` | declare no GPU even if one is present |
| `GRAPHSYNTH_SHOW_WARNINGS=1` | restore the two silenced dependency warnings |

---

## Troubleshooting

**`No CUDA GPUs are available` on a machine that has one.** Ray sets
`CUDA_VISIBLE_DEVICES` in each worker from the resources that task requested.
The loop detects a local GPU and declares `num_gpus=1`; if detection fails,
`GRAPHSYNTH_MODE=cluster` with a `chia up` cluster is the supported path. Do
**not** set `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`, which Ray's own warning
suggests: it would stop Ray clearing the variable for tasks that requested no
GPU, and `synthesize_kernel` requesting no GPU is how this loop keeps the agent
off the measurement hardware.

**`out of resource: shared memory`.** The kernel needs more shared memory per SM
than your GPU has. Expected for `temp_perhead` on Ada; see above.

**`arange's arguments must be of type tl.constexpr`.** A Triton version newer
than the one the kernel was recorded under. The loop moves to the next
iteration.

**Ray cannot write its temp directory** on a shared filesystem:
`export RAY_TMPDIR=/fast/local/path/ray`.

**Slow kernel compiles on a cluster.** Triton's cache defaults to `~/.triton`,
usually NFS. Point it at local disk: `export TRITON_CACHE_DIR=/tmp/$USER/triton`.
