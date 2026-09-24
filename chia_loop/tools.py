"""The agentic edge: the only surface the synthesis agent can touch.

The agent gets a scratchpad (read/write the candidate source) and a compile
check. It deliberately does NOT get the correctness gate or the benchmark --
those stay programmatic nodes in nodes.py. ``smoke_compile`` reports whether the
code builds, never whether it is right or how fast it is, so there is no reward
signal here to optimise against.

Docstrings matter: CHIA passes the docstring of each registered function to the
model as the tool description.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from chia.base.tools.ChiaTool import ChiaTool


class KernelWorkbenchTool(ChiaTool):
    """Scratchpad plus a compile check for the kernel-synthesis agent."""

    def __init__(self, name: str = "kernel_workbench",
                 task_options: Optional[Dict] = None,
                 shape=(1, 2, 256, 64), dtype: str = "bfloat16",
                 logging_level: int = logging.INFO):
        self._drafts: Dict[str, str] = {}
        self._smoke_shape = shape          # tiny; a build check, not a benchmark
        self._smoke_dtype = dtype
        super().__init__(name, task_options=task_options,
                         logging_level=logging_level)

    def setup(self):
        self.mcp.add_tool(self.read_kernel, name=f"{self.name}_read_kernel")
        self.mcp.add_tool(self.write_kernel, name=f"{self.name}_write_kernel")
        self.mcp.add_tool(self.smoke_compile, name=f"{self.name}_smoke_compile")

    # -- tools exposed to the agent ----------------------------------------
    def read_kernel(self, op: str) -> str:
        """Return the current draft Triton source for `op`, or an empty string
        if nothing has been written yet."""
        return self._drafts.get(op, "")

    def write_kernel(self, op: str, source: str) -> str:
        """Store `source` as the current draft Triton kernel for `op`.

        The source must define a @triton.jit kernel named graphsynth_kernel and
        a function launch_kernel(q, k, v) returning one tensor of q's shape.
        """
        self._drafts[op] = source
        return f"stored {len(source)} chars for {op}"

    def smoke_compile(self, op: str) -> str:
        """Try to import the current draft for `op` and run it once on a tiny
        shape, to surface Python and Triton compilation errors early.

        This is a BUILD CHECK ONLY. It does not verify numerical correctness and
        does not measure performance; a kernel that passes here can still be
        wrong or slow. Correctness and timing are decided later by the loop,
        outside your control.
        """
        source = self._drafts.get(op)
        if not source:
            return "no draft written yet for this op -- call write_kernel first"

        from chia_loop.nodes import _load_launch
        fn, err = _load_launch(source)
        if fn is None:
            return f"FAILED TO LOAD: {err}"

        try:
            import torch
        except ImportError:
            return "loaded and defines launch_kernel (torch unavailable: no run check)"

        if not torch.cuda.is_available():
            return "loaded and defines launch_kernel (no GPU here: no run check)"

        B, H, S, D = self._smoke_shape
        DT = getattr(torch, self._smoke_dtype)
        try:
            q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=DT) for _ in range(3))
            out = fn(q, k, v)
            torch.cuda.synchronize()
        except Exception as e:                              # noqa: BLE001
            return f"COMPILED BUT RAISED on shape {self._smoke_shape}: {type(e).__name__}: {e}"

        if out.shape != q.shape:
            return (f"ran, but returned shape {tuple(out.shape)}; "
                    f"expected {tuple(q.shape)}")
        return f"ran on tiny shape {self._smoke_shape} and returned the right shape"

    # -- loop-side accessor (not a tool) -----------------------------------
    def draft(self, op: str) -> str:
        """Read a draft from the orchestrator. Not registered as a tool."""
        return self._drafts.get(op, "")
