import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def fused_backward_kernel(
    input_ptr,           # *f32, shape [Ni, Mh]
    grad_out_ptr,        # *f32, shape [Ni, Mh]
    mhc_scale_ptr,       # *f32, shape [1]
    mhc_base_ptr,        # *f32, shape [Mh]
    grad_input_ptr,      # *f32, shape [Ni, Mh]
    grad_mhc_base_ptr,   # *f32, shape [Mh]
    grad_mhc_scale_ptr,  # *f32, shape [1]
    Ni: tl.constexpr,    # int
    Mh: tl.constexpr,    # int
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Hints for better codegen/vectorization
    tl.multiple_of(n_offsets, BLOCK_N)
    tl.multiple_of(m_offsets, BLOCK_M)

    n_mask = n_offsets < Ni
    m_mask = m_offsets < Mh
    mask = n_mask[:, None] & m_mask[None, :]

    # Compute base pointers for 2D tile
    base_ptrs = n_offsets[:, None] * Mh + m_offsets[None, :]

    # Loads with cache hints: stream x/g, keep base in cache (reused across rows)
    x = tl.load(input_ptr + base_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
    g = tl.load(grad_out_ptr + base_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
    s = tl.load(mhc_scale_ptr)  # scalar
    b_m = tl.load(mhc_base_ptr + m_offsets, mask=m_mask, other=0.0, cache_modifier=".ca")

    # Broadcast mhc_base across rows and compute sigmoid and its derivative
    z = tl.fma(x, s, b_m[None, :])
    sig = tl.sigmoid(z)
    one_minus_sig = 1.0 - sig
    gz = g * sig * one_minus_sig

    # grad_input_mix
    grad_input = gz * s
    tl.store(grad_input_ptr + base_ptrs, grad_input, mask=mask)

    # Partial reductions for grad_mhc_base (sum over n for each m)
    partial_base = tl.sum(gz, axis=0)  # [BLOCK_M]
    tl.atomic_add(grad_mhc_base_ptr + m_offsets, partial_base, mask=m_mask)

    # Partial reduction for grad_mhc_scale (sum over all n and m)
    partial_scale_rows = tl.sum(gz * x, axis=1)  # [BLOCK_N]
    partial_scale = tl.sum(partial_scale_rows, axis=0)
    tl.atomic_add(grad_mhc_scale_ptr, partial_scale)


class ModelNew(nn.Module):
    """
    Model that computes manual backward of mhc_head_compute_mix using a fused Triton kernel when available.
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
        # Fallback to PyTorch if inputs are not CUDA tensors or not float32
        if (
            (not input_mix.is_cuda)
            or (not grad_out.is_cuda)
            or (not mhc_scale.is_cuda)
            or (not mhc_base.is_cuda)
            or (input_mix.dtype != torch.float32)
            or (grad_out.dtype != torch.float32)
            or (mhc_scale.dtype != torch.float32)
            or (mhc_base.dtype != torch.float32)
        ):
            z = input_mix * mhc_scale + mhc_base
            sigmoid = torch.sigmoid(z)
            grad_z = grad_out * sigmoid * (1 - sigmoid)
            grad_input_mix = grad_z * mhc_scale
            grad_mhc_base = grad_z.sum(dim=(0, 1), keepdim=True).view(-1)
            grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2), keepdim=True).view(1)
            return grad_input_mix, grad_mhc_scale, grad_mhc_base

        # Ensure contiguous memory layout
        n0, n1, mh = input_mix.shape
        Ni = n0 * n1
        Mh = mh

        x2d = input_mix.reshape(Ni, Mh).contiguous()
        g2d = grad_out.reshape(Ni, Mh).contiguous()

        grad_input_mix = torch.empty_like(input_mix)
        grad_input_2d = grad_input_mix.view(Ni, Mh)

        grad_mhc_base = torch.zeros(Mh, device=input_mix.device, dtype=input_mix.dtype)
        grad_mhc_scale = torch.zeros(1, device=input_mix.device, dtype=input_mix.dtype)

        # Launch Triton kernel
        BLOCK_N = 128
        BLOCK_M = 32
        grid = (triton.cdiv(Ni, BLOCK_N), triton.cdiv(Mh, BLOCK_M))
        fused_backward_kernel[grid](
            x2d,
            g2d,
            mhc_scale,
            mhc_base,
            grad_input_2d,
            grad_mhc_base,
            grad_mhc_scale,
            Ni,
            Mh,
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=2,
        )

        return grad_input_mix, grad_mhc_scale, grad_mhc_base


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