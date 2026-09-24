# Running live on one GPU machine

No cluster needed — CHIA runs on a local Ray instance.

```bash
pip install chialoops torch==2.9.1
gcloud auth application-default login          # Stage 2 only
export GCP_PROJECT=<your-project>

python -m chia_loop.loop --ops bias_block_diagonal --max-iter 3
```

Resource tags are off by default so tasks schedule on any worker. On a
provisioned cluster set `GRAPHSYNTH_MODE=cluster` to request the virtual
`a100` resource declared in `cluster_a100.yaml`.

## Smaller GPUs

The recorded kernels are bf16 at `[2,32,2048,128]`. bf16 needs sm80 or newer
(A100, L4, A10, RTX 30-series and later). On an sm75 card such as a T4, run a
reduced shape in fp16:

```bash
python -m chia_loop.loop -B 1 -H 8 -S 512 -D 64 --dtype float16
```

Those numbers are not comparable to the paper's; they demonstrate the loop,
not the result.
