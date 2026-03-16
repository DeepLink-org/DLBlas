import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def fused_bias_act_gn_kernel(
    x_ptr,            # [N, C]
    extra_bias_ptr,   # [C]
    gamma_ptr,        # [C] groupnorm weight
    beta_ptr,         # [C] groupnorm bias
    out_ptr,          # [N, C]
    N, C, G,          # ints
    GROUP_SIZE,       # int = C // G
    eps,              # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    base = n * C
    g_off = g * GROUP_SIZE

    # Fast path: whole group fits in one block
    if GROUP_SIZE <= BLOCK_SIZE:
        c = g_off + offs
        mask_c = offs < GROUP_SIZE
        mask = mask_c & (n < N)
        idx = base + c

        # Load input and per-channel extra bias
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
        v = x + b

        # Hardtanh: clamp to [-1, 1]
        v = tl.minimum(tl.maximum(v, -1.0), 1.0)

        # Mish: v * tanh(softplus(v)), compute in fp32 for stability
        vf = v.to(tl.float32)
        abs_v = tl.abs(vf)
        m = tl.maximum(vf, 0.0)
        # stable softplus: m + log1p(exp(-|v|))
        softplus = m + libdevice.log1p(tl.exp(-abs_v))
        tanh_sp = libdevice.tanh(softplus)
        mish = vf * tanh_sp

        # GroupNorm stats over channels within the group for this sample
        mish_masked = tl.where(mask, mish, 0.0)
        sum1 = tl.sum(mish_masked, axis=0)
        sum2 = tl.sum(mish_masked * mish_masked, axis=0)
        gs = tl.full((), GROUP_SIZE, dtype=tl.float32)
        mean = sum1 / gs
        var = sum2 / gs - mean * mean
        inv_std = tl.rsqrt(var + eps)

        y = (mish - mean) * inv_std

        # Affine: per-channel gamma/beta
        gamma = tl.load(gamma_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
        y = y * gamma + beta

        tl.store(out_ptr + idx, y.to(x.dtype), mask=mask)
    else:
        # General path: process group in tiles
        rsum = tl.zeros((), dtype=tl.float32)
        rsq = tl.zeros((), dtype=tl.float32)

        # Pass 1: compute mean/var
        for start in range(0, GROUP_SIZE, BLOCK_SIZE):
            idx_in_group = start + offs
            mask_c = idx_in_group < GROUP_SIZE
            c = g_off + idx_in_group
            mask = mask_c & (n < N)
            idx = base + c

            x = tl.load(x_ptr + idx, mask=mask, other=0.0)
            b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
            v = x + b
            v = tl.minimum(tl.maximum(v, -1.0), 1.0)

            vf = v.to(tl.float32)
            abs_v = tl.abs(vf)
            m = tl.maximum(vf, 0.0)
            softplus = m + libdevice.log1p(tl.exp(-abs_v))
            mish = vf * libdevice.tanh(softplus)

            mish_masked = tl.where(mask, mish, 0.0)
            rsum += tl.sum(mish_masked, axis=0)
            rsq += tl.sum(mish_masked * mish_masked, axis=0)

        gs = tl.full((), GROUP_SIZE, dtype=tl.float32)
        mean = rsum / gs
        var = rsq / gs - mean * mean
        inv_std = tl.rsqrt(var + eps)

        # Pass 2: normalize + affine + store
        for start in range(0, GROUP_SIZE, BLOCK_SIZE):
            idx_in_group = start + offs
            mask_c = idx_in_group < GROUP_SIZE
            c = g_off + idx_in_group
            mask = mask_c & (n < N)
            idx = base + c

            x = tl.load(x_ptr + idx, mask=mask, other=0.0)
            b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
            v = x + b
            v = tl.minimum(tl.maximum(v, -1.0), 1.0)

            vf = v.to(tl.float32)
            abs_v = tl.abs(vf)
            m = tl.maximum(vf, 0.0)
            softplus = m + libdevice.log1p(tl.exp(-abs_v))
            mish = vf * libdevice.tanh(softplus)

            y = (mish - mean) * inv_std

            gamma = tl.load(gamma_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
            y = y * gamma + beta

            tl.store(out_ptr + idx, y.to(x.dtype), mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def _fused_post_gemm(self, y: torch.Tensor) -> torch.Tensor | None:
        # Fused BiasAdd -> Hardtanh -> Mish -> GroupNorm using Triton
        N, C = y.shape
        G = self.groupnorm.num_groups
        # Fallback if shape not compatible
        if (C % G) != 0:
            return None
        GROUP_SIZE = C // G

        # Ensure contiguity
        y_in = y.contiguous()
        extra_bias = self.bias.contiguous()
        gamma = self.groupnorm.weight.contiguous()
        beta = self.groupnorm.bias.contiguous()
        eps = float(self.groupnorm.eps)

        out = torch.empty_like(y_in)

        # Choose an efficient block size (power-of-two, capped)
        BLOCK_SIZE = _next_power_of_2(GROUP_SIZE)
        BLOCK_SIZE = min(max(BLOCK_SIZE, 32), 256)
        # Prefer fewer warps for small groups to reduce overhead
        if BLOCK_SIZE <= 32:
            num_warps = 1
        elif BLOCK_SIZE <= 64:
            num_warps = 2
        else:
            num_warps = 4

        grid = (N * G,)

        fused_bias_act_gn_kernel[grid](
            y_in, extra_bias, gamma, beta, out,
            N, C, G, GROUP_SIZE, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return out

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # GEMM via cuBLAS/ATen for peak performance
        y = self.gemm(x)

        # Use Triton fused epilogue on CUDA when safe.
        use_triton = y.is_cuda and self.groupnorm.affine
        if use_triton:
            fused = self._fused_post_gemm(y)
            if fused is not None:
                return fused

        # Fallback: exact PyTorch reference
        y = y + self.bias
        y = self.hardtanh(y)
        y = self.mish(y)
        y = self.groupnorm(y)
        return y


batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]