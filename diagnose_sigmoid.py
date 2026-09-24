"""Why does sigmoid_attn return rel_err = 1.01 on the L40S?

`verify_kernel` takes the *worst* relative error over two input scales, 1.0 and
0.01, and reports one number. That collapses two different questions into one.
This script separates them, for every recorded sigmoid kernel, and prints
enough of the output's shape to tell a wrong kernel from a wrong reference.

A relative error near 1.0 has two very different causes:

  * the kernel returns approximately zero -- ||out|| collapses, ||ref|| does not;
  * the kernel returns something of the right magnitude but uncorrelated.

The ratio ||out|| / ||ref|| and the cosine similarity separate those, and doing
it per scale says whether the failure is in the small-magnitude regime, where
sigmoid(x) -> 0.5 for every score and attention degenerates into an unweighted
sum, which is exactly where a kernel carrying a softmax habit such as
subtracting a row maximum would diverge from the reference.

    python diagnose_sigmoid.py
"""
from __future__ import annotations

import sys

import torch

from chia_loop import ops as ops_mod
from chia_loop import replay
from chia_loop.nodes import _load_launch


def report(tag: str, path, spec) -> None:
    src = path.read_text()
    fn, err = _load_launch(src)
    print(f"\n--- {tag}: {path.name}")
    if fn is None:
        print(f"    does not load: {err}")
        return

    ref = ops_mod.reference_for(spec)
    B, H, S, D = spec.shape
    DT = getattr(torch, spec.dtype)
    torch.manual_seed(0)

    for scale in (1.0, 0.01):
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=DT) * scale
                   for _ in range(3))
        try:
            out = fn(q, k, v)
        except Exception as e:                                  # noqa: BLE001
            print(f"    scale {scale:<5}: raised {type(e).__name__}: {e}")
            continue
        r = ref(q, k, v)
        o32, r32 = out.float(), r.float()
        rel = ((r32 - o32).abs().max() / (r32.abs().max() + 1e-6)).item()
        n_o, n_r = o32.norm().item(), r32.norm().item()
        cos = torch.nn.functional.cosine_similarity(
            o32.flatten(), r32.flatten(), dim=0).item()
        print(f"    scale {scale:<5}: rel_err {rel:9.3e} | "
              f"||out||/||ref|| {n_o / (n_r + 1e-12):7.4f} | cos {cos:+.4f} | "
              f"nan {int(torch.isnan(o32).sum())} inf {int(torch.isinf(o32).sum())}")


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    print(torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0), "| torch", torch.__version__)

    spec = ops_mod.all_ops(B=2, H=32, S=2048, D=128, dtype="bfloat16")["sigmoid_attn"]
    print(f"best_model for sigmoid_attn: {replay.best_model('sigmoid_attn')!r}")

    for it in (1, 2, 3):
        p = replay.find_kernel("sigmoid_attn", it)
        if p is None:
            print(f"\n--- iter {it}: no recorded kernel")
            continue
        report(f"iter {it}", p, spec)

    print("\nReading of the numbers:")
    print("  ||out||/||ref|| near 0      -> the kernel returns ~zero")
    print("  ratio near 1 but cos near 0 -> right magnitude, wrong values")
    print("  fails only at scale 0.01    -> the small-score regime, not the kernel as such")
    return 0


if __name__ == "__main__":
    sys.exit(main())
