import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_backward_kernel(
    input_ptr,            # float* (n0, n1, k)
    mhc_scale_ptr,        # float* (1,)
    mhc_base_ptr,         # float* (k,)
    grad_out_ptr,         # float* (n0, n1, k)
    grad_input_ptr,       # float* (n0, n1, k)
    grad_scale_ptr,       # float* (1,)
    grad_base_ptr,        # float* (k,)
    n0, n1, k,            # int sizes
    stride_in_0, stride_in_1, stride_in_2,   # input strides
    stride_go_0, stride_go_1, stride_go_2,   # grad_out strides
    stride_gi_0, stride_gi_1, stride_gi_2,   # grad_input strides
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    M = n0 * n1
    mask_m = offs_m < M
    mask_k = offs_k < k
    mask = mask_m[:, None] & mask_k[None, :]

    # Decompose offs_m -> (i0, i1)
    i0 = offs_m // n1
    i1 = offs_m - i0 * n1  # cheaper than modulo

    # Compute pointers
    ptr_in = input_ptr + (i0[:, None] * stride_in_0 + i1[:, None] * stride_in_1 + offs_k[None, :] * stride_in_2)
    ptr_go = grad_out_ptr + (i0[:, None] * stride_go_0 + i1[:, None] * stride_go_1 + offs_k[None, :] * stride_go_2)
    ptr_gi = grad_input_ptr + (i0[:, None] * stride_gi_0 + i1[:, None] * stride_gi_1 + offs_k[None, :] * stride_gi_2)

    # Loads with cache/eviction hints
    x = tl.load(ptr_in, mask=mask, other=0.0, cache_modifier=".cg")
    go = tl.load(ptr_go, mask=mask, other=0.0, cache_modifier=".cg")
    scale = tl.load(mhc_scale_ptr, eviction_policy="evict_last")
    base_k = tl.load(mhc_base_ptr + offs_k, mask=mask_k, other=0.0, eviction_policy="evict_last")

    # z = x * scale + base
    z = x * scale + base_k[None, :]

    # sigmoid and grad_z (use s - s*s for derivative)
    s = tl.sigmoid(z)
    t = s - s * s
    grad_z = go * t

    # grad_input_mix
    gi = grad_z * scale
    tl.store(ptr_gi, gi, mask=mask)

    # grad_mhc_base: sum over m for each k, then atomic add
    sum_m = tl.sum(grad_z, axis=0)
    tl.atomic_add(grad_base_ptr + offs_k, sum_m, mask=mask_k)

    # grad_mhc_scale: sum over all elements of grad_z * x
    prod = grad_z * x
    tile_row_sum = tl.sum(prod, axis=1)          # [BLOCK_M]
    tile_sum = tl.sum(tile_row_sum, axis=0)      # scalar
    tl.atomic_add(grad_scale_ptr, tile_sum)


class ModelNew(nn.Module):
    """
    Model that computes manual backward of mhc_head_compute_mix using a fused Triton kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        """
        Manual backward computation.

        Args:
            input_mix: (n0, n1, mhc_mult)
            mhc_scale: (1,)
            mhc_base: (mhc_mult,)
            grad_out: same shape as input_mix

        Returns:
            grad_input_mix, grad_mhc_scale, grad_mhc_base
        """

        # Prepare outputs
        grad_input_mix = torch.empty_like(input_mix)
        grad_mhc_scale = torch.zeros_like(mhc_scale)
        grad_mhc_base = torch.zeros_like(mhc_base)

        n0, n1, k = input_mix.shape

        # Strides (in elements)
        s_in0, s_in1, s_in2 = input_mix.stride()
        s_go0, s_go1, s_go2 = grad_out.stride()
        s_gi0, s_gi1, s_gi2 = grad_input_mix.stride()

        # Launch Triton kernel
        BLOCK_M = 128
        BLOCK_K = 32
        grid = (triton.cdiv(n0 * n1, BLOCK_M), triton.cdiv(k, BLOCK_K))
        fused_backward_kernel[grid](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input_mix,
            grad_mhc_scale,
            grad_mhc_base,
            n0, n1, k,
            s_in0, s_in1, s_in2,
            s_go0, s_go1, s_go2,
            s_gi0, s_gi1, s_gi2,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        return grad_input_mix, grad_mhc_scale.view(1), grad_mhc_base

batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)
    mhc_scale = torch.randn(1, dtype=torch.float32)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32)
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)

    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []