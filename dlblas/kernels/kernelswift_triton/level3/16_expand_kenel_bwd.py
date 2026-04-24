# model.py

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _sum_mhc_reduce_kernel(
    o_ptr, out_ptr,
    N0, N1, M, H,
    stride0, stride1, stride2, stride3,
    ostride0, ostride1, ostride2,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)  # over N0 * N1
    pid_h = tl.program_id(1)  # over tiles of H

    n0 = pid_n // N1
    n1 = pid_n % N1

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    tl.multiple_of(h_offsets, 8)
    h_mask = h_offsets < H

    base_n = n0 * stride0 + n1 * stride1

    # Precompute base pointer for this (n0, n1) and H tile
    base_ptr_h = o_ptr + base_n + h_offsets * stride3

    # Accumulator in FP32 for improved numerical stability
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Fast paths for small/common M to reduce loop overhead and improve ILP
    if M == 4:
        v0 = tl.load(base_ptr_h + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v1 = tl.load(base_ptr_h + 1 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v2 = tl.load(base_ptr_h + 2 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v3 = tl.load(base_ptr_h + 3 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        acc = (v0.to(tl.float32) + v1.to(tl.float32)) + (v2.to(tl.float32) + v3.to(tl.float32))
    elif M == 3:
        v0 = tl.load(base_ptr_h + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v1 = tl.load(base_ptr_h + 1 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v2 = tl.load(base_ptr_h + 2 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        acc = (v0.to(tl.float32) + v1.to(tl.float32)) + v2.to(tl.float32)
    elif M == 2:
        v0 = tl.load(base_ptr_h + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        v1 = tl.load(base_ptr_h + 1 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        acc = v0.to(tl.float32) + v1.to(tl.float32)
    elif M == 1:
        v0 = tl.load(base_ptr_h + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
        acc = v0.to(tl.float32)
    else:
        # General path with improved ILP: two independent accumulators and wider unroll
        acc0 = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_H], dtype=tl.float32)
        m = 0
        # Unroll by 8
        while m + 7 < M:
            ptr = base_ptr_h + m * stride2
            v0 = tl.load(ptr + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v1 = tl.load(ptr + 1 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v2 = tl.load(ptr + 2 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v3 = tl.load(ptr + 3 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v4 = tl.load(ptr + 4 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v5 = tl.load(ptr + 5 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v6 = tl.load(ptr + 6 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v7 = tl.load(ptr + 7 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            acc0 += v0.to(tl.float32) + v2.to(tl.float32) + v4.to(tl.float32) + v6.to(tl.float32)
            acc1 += v1.to(tl.float32) + v3.to(tl.float32) + v5.to(tl.float32) + v7.to(tl.float32)
            m += 8

        # Handle remaining blocks of 4
        while m + 3 < M:
            ptr = base_ptr_h + m * stride2
            v0 = tl.load(ptr + 0 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v1 = tl.load(ptr + 1 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v2 = tl.load(ptr + 2 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            v3 = tl.load(ptr + 3 * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            acc0 += v0.to(tl.float32) + v2.to(tl.float32)
            acc1 += v1.to(tl.float32) + v3.to(tl.float32)
            m += 4

        # Tail
        while m < M:
            v = tl.load(base_ptr_h + m * stride2, mask=h_mask, other=0.0, cache_modifier=".cg")
            acc0 += v.to(tl.float32)
            m += 1

        acc = acc0 + acc1

    out_ptr_tile = out_ptr + n0 * ostride0 + n1 * ostride1 + h_offsets * ostride2
    tl.store(out_ptr_tile, acc, mask=h_mask)


class ModelNew(nn.Module):
    """
    Model that simulates the backward of expand_to_mhc operation.
    It reduces (sums) along the broadcasted dimension.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, o_grad: torch.Tensor) -> torch.Tensor:
        """
        Simulated backward of expand operation.

        Args:
            o_grad (torch.Tensor): Gradient tensor of shape
                                  (n0, n1, mhc_mult, h)

        Returns:
            torch.Tensor: Reduced gradient of shape (n0, n1, h)
        """
        # Fallback to PyTorch if not on CUDA or unsupported dtype
        if (not o_grad.is_cuda) or (o_grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            return o_grad.sum(dim=-2)

        n0, n1, mhc_mult, h = o_grad.shape
        out = torch.empty((n0, n1, h), device=o_grad.device, dtype=o_grad.dtype)

        BLOCK_H = 256
        grid = (n0 * n1, triton.cdiv(h, BLOCK_H))

        _sum_mhc_reduce_kernel[grid](
            o_grad, out,
            n0, n1, mhc_mult, h,
            o_grad.stride(0), o_grad.stride(1), o_grad.stride(2), o_grad.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return out


# ----------------------------
# Test input configuration
# ----------------------------
batch_n0 = 2
batch_n1 = 1024
mhc_mult = 4
hidden_dim = 1280


def get_inputs():
    o_grad = torch.randn(batch_n0, batch_n1, mhc_mult, hidden_dim)
    return [o_grad]


def get_init_inputs():
    return []