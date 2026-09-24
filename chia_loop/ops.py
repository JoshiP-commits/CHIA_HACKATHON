"""The ten attention variants, with their reference implementations.

Semantics and prompt constraints are taken verbatim from ``synth10.py`` so the
CHIA loop synthesizes against the same specification the paper evaluated.

One deliberate departure. ``synth10.py`` builds the sigmoid reference by
zeroing masked scores *before* the sigmoid, which leaves every masked
position contributing 0.5 of its value vector, since sigmoid(0) = 0.5.
``provenance/sigmoid_fix.py`` is the corrected re-run that the paper's
sigmoid numbers come from, and it masks after the nonlinearity. The
reference below follows the corrected version, not ``synth10.py``.

torch is imported lazily: in replay (bypass) mode the loop runs with no torch
and no GPU, so nothing here may import torch at module scope.
"""
from __future__ import annotations
import math
from typing import Callable, Dict
from chia_loop.state import OpSpec

PROMPT_PREAMBLE = """
You are an expert GPU kernel engineer writing Triton for an NVIDIA A100 (sm80).

HARD REQUIREMENTS
- Return ONE ```python code block, nothing else.
- @triton.jit kernel named exactly `graphsynth_kernel`.
- `def launch_kernel(q, k, v):` taking three bfloat16 tensors [2,32,2048,128],
  returning ONE bfloat16 tensor of the same shape.
- FlashAttention-style tiling: loop over KV blocks, running max + running sum,
  fp32 accumulators. NEVER materialise the [S,S] score matrix in DRAM.
- Do NOT use `continue` or `break` in the loop. Bound the loop range on the host
  or use tl.where.

TRITON API CONSTRAINTS -- violating any of these fails:
- `tl.math.tanh` does NOT exist. Use 2.0*tl.sigmoid(2.0*x) - 1.0
- `tl.dot()` has NO `trans_b`. Load K with swapped strides instead.
- Python ints have no `.to()`. Compute 1/sqrt(D) on the host, pass as a float.
- Use `tl.where(cond,a,b)` for masking; no boolean indexing.
- Init running max with `tl.full([BLOCK_M], -1e9, tl.float32)`, not -inf.
- Head index is available as a program id -- derive per-head constants from it.

OPERATOR (scale = 1/sqrt(128)):
"""

SEMANTICS: Dict[str, str] = {
 "sigmoid_attn":
   "s=(q@k^T)*scale; causal mask j<=i (masked entries contribute 0);\n"
   "p=sigmoid(s) elementwise (NO softmax, no running max needed -- much simpler);\n"
   "out=p@v",
 "relu_attn":
   "s=(q@k^T)*scale; causal mask j<=i (masked contribute 0);\n"
   "p=relu(s) elementwise (NO softmax); out=p@v",
 "temp_perhead":
   "s=(q@k^T)*scale; multiply s by a PER-HEAD temperature t_h = 0.5 + h/32 where\n"
   "h is the head index (derive it from the program id, do not pass a tensor);\n"
   "causal mask j<=i; p=softmax(s); out=p@v",
 "bias_alibi":
   "s=(q@k^T)*scale; add ALiBi bias slope_h*(j-i) where slope_h = h+1 and h is the\n"
   "head index (compute it from the program id -- do NOT load a bias tensor, that\n"
   "would be 1.07 GB of avoidable DRAM traffic); causal mask j<=i;\n"
   "p=softmax(s); out=p@v",
 "bias_dilated":
   "s=(q@k^T)*scale; query i attends to key j only if (j<=i) AND ((i-j) % 4 == 0);\n"
   "p=softmax(s); out=p@v. Compute the stride test arithmetically in the kernel.",
 "bias_local":
   "s=(q@k^T)*scale; query i attends to key j only if abs(i-j) < 256 (BIDIRECTIONAL\n"
   "band, not causal); p=softmax(s); out=p@v. Only KV blocks overlapping the band\n"
   "need visiting -- bound the loop range on the host accordingly.",
 "bias_block_diagonal":
   "s=(q@k^T)*scale; tokens are packed documents of length 512: doc(x)=x//512.\n"
   "query i attends to key j only if (doc(i)==doc(j)) AND (j<=i);\n"
   "p=softmax(s); out=p@v. Only the diagonal document block needs visiting.",
 "bias_prefix_lm":
   "s=(q@k^T)*scale; prefix-LM mask: query i attends to key j if (j<=i) OR (j<256)\n"
   "-- the first 256 tokens are visible to everyone; p=softmax(s); out=p@v",
 "softcap_gemma2":
   "s=(q@k^T)*scale; s=tanh(s/30)*30 (Gemma-2 logit soft-capping, applied BEFORE\n"
   "the running-max update); causal mask j<=i; p=softmax(s); out=p@v",
 "bias_sliding_window":
   "s=(q@k^T)*scale; query i attends to key j only if (j<=i) AND (i-j < 256);\n"
   "p=softmax(s); out=p@v. Only ~12% of the matrix is unmasked -- bound the KV\n"
   "loop range on the host so masked blocks are never visited.",
}


def all_ops(**shape) -> Dict[str, OpSpec]:
    """The ten variants outside PyTorch's fused-kernel catalog."""
    return {n: OpSpec(name=n, semantics=s, **shape) for n, s in SEMANTICS.items()}


def prompt_for(spec: OpSpec) -> str:
    return PROMPT_PREAMBLE + spec.semantics


def reference_for(spec: OpSpec) -> Callable:
    """Build the PyTorch reference. Imports torch lazily."""
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DT = getattr(torch, spec.dtype)
    H, S, D = spec.H, spec.S, spec.D
    name = spec.name

    def idx():
        ar = torch.arange(S, device=dev)
        return ar.view(-1, 1), ar.view(1, -1)

    def base_scores(q, k):
        return (q @ k.transpose(-2, -1)) / math.sqrt(D)

    def mk_ref(mod=None, maskfn=None, norm="softmax"):
        def ref(q, k, v):
            s = base_scores(q, k)
            if mod is not None:
                s = mod(s)
            i, j = idx()
            ok = maskfn(i, j) if maskfn else (j <= i)
            # The mask has to be applied on the side of the nonlinearity where
            # "masked" actually means "contributes nothing". For softmax that is
            # before, with -inf. For sigmoid it is AFTER: sigmoid(0) = 0.5, so
            # zeroing the score first leaves every masked position contributing
            # half of its value vector. relu is indifferent, since relu(0) = 0,
            # but it is masked after as well so that all three read alike.
            if norm == "softmax":
                s = s.masked_fill(~ok, float("-inf"))
                p = torch.softmax(s.float(), -1)
            elif norm == "sigmoid":
                p = torch.sigmoid(s.float()).masked_fill(~ok, 0.0)
            else:
                p = F.relu(s.float()).masked_fill(~ok, 0.0)
            return p.to(DT) @ v
        return ref

    if name == "sigmoid_attn":
        return mk_ref(norm="sigmoid")
    if name == "relu_attn":
        return mk_ref(norm="relu")
    if name == "temp_perhead":
        temp = (0.5 + torch.arange(H, device=dev, dtype=DT) / H).view(1, H, 1, 1)
        return mk_ref(mod=lambda s: s * temp)
    if name == "bias_alibi":
        slope = torch.arange(1, H + 1, device=dev, dtype=torch.float32).view(1, H, 1, 1)
        def ref_alibi(q, k, v):
            s = base_scores(q, k)
            i, j = idx()
            s = s + (slope * (j - i).float()).to(DT)
            return torch.softmax(s.masked_fill(j > i, float("-inf")).float(), -1).to(DT) @ v
        return ref_alibi
    if name == "bias_dilated":
        return mk_ref(maskfn=lambda i, j: (j <= i) & (((i - j) % 4) == 0))
    if name == "bias_local":
        return mk_ref(maskfn=lambda i, j: (i - j).abs() < 256)
    if name == "bias_block_diagonal":
        return mk_ref(maskfn=lambda i, j: ((i // 512) == (j // 512)) & (j <= i))
    if name == "bias_prefix_lm":
        return mk_ref(maskfn=lambda i, j: (j <= i) | (j < 256))
    if name == "softcap_gemma2":
        return mk_ref(mod=lambda s: torch.tanh(s / 30) * 30)
    if name == "bias_sliding_window":
        return mk_ref(maskfn=lambda i, j: (j <= i) & ((i - j) < 256))
    raise KeyError(name)
