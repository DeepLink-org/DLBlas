# Copyright (c) 2026, cute_gemm team.

"""MoE expert projections as fused grouped GEMM primitives.

Each op is one expert-grouped GEMM with the activation fused into the epilogue.
Routing, aggregation, weight gradients, and autograd ownership remain with the
caller.
"""

from .interface import moe_down_proj_dact, moe_up_proj_act

__all__ = ["moe_down_proj_dact", "moe_up_proj_act"]
