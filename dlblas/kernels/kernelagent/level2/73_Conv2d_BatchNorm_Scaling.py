import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, y_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    s = tl.full([1], scale, x.dtype)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def _scale_triton(x: torch.Tensor, scale: float) -> torch.Tensor:
    # CPU fallback to preserve functionality across devices
    if not x.is_cuda:
        return x * scale
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    # Larger tile to reduce grid size; tuned for Hopper/H200 bandwidth
    BLOCK_SIZE = 16384
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _scale_kernel[grid](
        x_contig, y, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2
    )
    return y


@triton.jit
def _bn_fuse_params_kernel(
    mean_ptr, var_ptr, gamma_ptr, beta_ptr, convb_ptr,
    g_out_ptr, b_out_ptr,
    eps, scale, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 8)
    tl.max_contiguous(offs, 8)
    mask = offs < n_elements

    m = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    v = tl.load(var_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    cb = tl.load(convb_ptr + offs, mask=mask, other=0.0)

    eps_t = tl.full([1], eps, dtype=v.dtype)
    s_t = tl.full([1], scale, dtype=g.dtype)

    std = tl.sqrt(v + eps_t)
    g_ch = (g * s_t) / std
    b_ch = b * s_t + (cb - m) * g_ch

    tl.store(g_out_ptr + offs, g_ch, mask=mask)
    tl.store(b_out_ptr + offs, b_ch, mask=mask)


def _bn_fuse_params_triton(
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    conv_bias: torch.Tensor,
    eps: float,
    scale: float,
):
    # Fallback for CPU or very small tensors
    if not (running_mean.is_cuda and running_var.is_cuda and gamma.is_cuda and beta.is_cuda and conv_bias.is_cuda):
        std = (running_var + eps).sqrt()
        g = (gamma * scale) / std
        b = beta * scale + (conv_bias - running_mean) * g
        return g, b
    C = running_mean.numel()
    rm = running_mean.contiguous()
    rv = running_var.contiguous()
    ga = gamma.contiguous()
    be = beta.contiguous()
    cb = conv_bias.contiguous()
    g_out = torch.empty_like(ga)
    b_out = torch.empty_like(be)
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(C, meta["BLOCK_SIZE"]),)
    _bn_fuse_params_kernel[grid](
        rm, rv, ga, be, cb,
        g_out, b_out,
        float(eps), float(scale), C,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=1
    )
    return g_out, b_out


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    Optimizations:
      - Fuse the final scalar multiplication into BatchNorm's affine parameters during training mode.
      - In eval mode with tracked running stats, fold Conv2d + BatchNorm2d + scaling into a single Conv2d
        by precomputing per-channel fused scale/bias on GPU via a tiny Triton kernel.
      - Fallback Triton kernel for full-tensor scaling remains available but is avoided in common paths.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        # Enable cuDNN benchmarking for potential conv speed-ups (does not change numerics)
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        # Fast eval path: fuse conv+bn+scale into a single conv when using running stats.
        if (not self.bn.training) and self.bn.track_running_stats:
            W = self.conv.weight
            B = self.conv.bias
            device = W.device
            dtype = W.dtype
            C = self.bn.num_features

            # Prepare BN parameters (affine or not) in Conv dtype
            if self.bn.affine:
                gamma = self.bn.weight.to(dtype=dtype)
                beta = self.bn.bias.to(dtype=dtype)
            else:
                gamma = torch.ones(C, device=device, dtype=dtype)
                beta = torch.zeros(C, device=device, dtype=dtype)

            running_mean = self.bn.running_mean.to(dtype=dtype)
            running_var = self.bn.running_var.to(dtype=dtype)
            conv_bias = B if B is not None else torch.zeros(C, device=device, dtype=dtype)

            # Compute per-channel fused scale and bias with Triton
            g, b = _bn_fuse_params_triton(
                running_mean, running_var, gamma, beta, conv_bias, self.bn.eps, float(self.scaling_factor)
            )

            # Fold into conv weights/bias and run a single conv
            W_fused = W * g.view(-1, 1, 1, 1)
            y = F.conv2d(
                x, W_fused, b, stride=self.conv.stride, padding=self.conv.padding,
                dilation=self.conv.dilation, groups=self.conv.groups
            )
            return y

        # Training or non-tracked path: fold final scaling into BN's affine or functional weight.
        x = self.conv(x)
        s = float(self.scaling_factor)

        # Derive exact BatchNorm semantics
        training_flag = self.bn.training or not self.bn.track_running_stats
        running_mean = self.bn.running_mean if self.bn.track_running_stats else None
        running_var = self.bn.running_var if self.bn.track_running_stats else None

        if self.bn.affine:
            # Fuse s into affine parameters to avoid an extra elementwise pass.
            fused_weight = self.bn.weight * s
            fused_bias = self.bn.bias * s
            x = F.batch_norm(
                x,
                running_mean=running_mean,
                running_var=running_var,
                weight=fused_weight,
                bias=fused_bias,
                training=training_flag,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
            return x
        else:
            # No affine parameters: pass synthetic weight/bias to F.batch_norm to fuse scaling.
            C = self.bn.num_features
            device = x.device
            dtype = x.dtype
            fused_weight = torch.full((C,), s, device=device, dtype=dtype)
            fused_bias = torch.zeros((C,), device=device, dtype=dtype)
            x = F.batch_norm(
                x,
                running_mean=running_mean,
                running_var=running_var,
                weight=fused_weight,
                bias=fused_bias,
                training=training_flag,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
            return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]