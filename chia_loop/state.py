"""Typed state passed along CHIA edges.

Mirrors the pattern of chia.chipyard.state_def: every edge in the loop carries a
dataclass, not a loose dict, so a node's contract is visible at its signature.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class OpSpec:
    """One attention variant to synthesize a kernel for."""
    name: str
    semantics: str                 # natural-language operator description
    B: int = 2
    H: int = 32
    S: int = 2048
    D: int = 128
    dtype: str = "bfloat16"
    mask_density: float = 1.0      # unmasked fraction of the S x S score grid
    in_fused_catalog: bool = False # PyTorch's fused path cannot express these

    @property
    def shape(self) -> tuple:
        return (self.B, self.H, self.S, self.D)

    @property
    def bytes_per_elem(self) -> int:
        return 4 if self.dtype == "float32" else 2


@dataclass
class ProfileResult:
    """Stage 1 output: where this operator sits against the device roofline."""
    op: str
    flops: float
    min_dram_bytes: float
    arithmetic_intensity: float
    ridge_point: float
    memory_bound: bool
    eager_us: float
    in_fused_catalog: bool = False
    compiled_us: Optional[float] = None
    device: str = ""

    @property
    def admitted(self) -> bool:
        """Stage 1 admission, two clauses.

        An operator is admitted when either

          (a) it is memory-bound: AI < ridge point. Sparsity moves an operator
              along this axis, since a mask of density d scales the FLOP count
              while leaving B_min unchanged; or

          (b) PyTorch's fused path cannot express it. The *ideal* fused kernel
              for a dense score modification sits above the ridge, but no such
              kernel exists here -- eager materialises the S x S score matrix,
              so the implementation that actually runs is bandwidth-bound even
              though the ideal one would not be.

        Clause (b) is why dense variants such as ALiBi and logit soft-capping
        are admitted despite AI > ridge. Reporting only clause (a) would
        contradict the operators this loop actually optimises.
        """
        return self.memory_bound or not self.in_fused_catalog


@dataclass
class KernelCandidate:
    op: str
    iteration: int
    source: str
    model: str
    origin: str = "live"           # "live" | "replay"


@dataclass
class VerifyResult:
    """Stage 3 output. Produced ONLY by the programmatic gate, never by a tool."""
    passed: bool
    rel_err: float
    tolerance: float
    error: Optional[str] = None


@dataclass
class BenchResult:
    """Stage 4 output. Produced ONLY programmatically."""
    kernel_us: float
    baseline_us: float
    speedup: float
    achieved_gbs: float
    pct_of_peak: float


@dataclass
class Attempt:
    """One row of the attempts database."""
    op: str
    iteration: int
    model: str
    passed: bool
    rel_err: float
    kernel_us: Optional[float]
    speedup: Optional[float]
    error: Optional[str]

    def as_row(self) -> tuple:
        return (self.op, self.iteration, self.model, int(self.passed),
                self.rel_err, self.kernel_us, self.speedup, self.error)
