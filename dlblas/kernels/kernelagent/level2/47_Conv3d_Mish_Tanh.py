import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _mish_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements

    # Load and upcast to fp32 for numerical stability
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute tanh(softplus(x)) in a log-free, stable way:
    # exp(2*softplus(x)) = exp(2*max(x,0)) * (1 + exp(-|x|))^2
    ax = tl.abs(x_f32)
    e_neg_ax = tl.exp(-ax)
    one_plus = 1.0 + e_neg_ax
    e2max = tl.exp(2.0 * tl.maximum(x_f32, 0.0))
    e2s = e2max * one_plus * one_plus
    tanh_sp = 1.0 - 2.0 / (1.0 + e2s)

    # mish(x) = x * tanh(softplus(x))
    mish = x_f32 * tanh_sp

    # tanh(mish) using stable formulation: tanh(a) = sign(a) * (1 - 2 / (1 + exp(2|a|)))
    am = tl.abs(mish)
    e2a = tl.exp(2.0 * am)
    sign = tl.where(mish >= 0.0, 1.0, -1.0)
    out_f32 = sign * (1.0 - 2.0 / (1.0 + e2a))

    # Downcast and store
    out = out_f32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    # Fused activation: y = tanh(mish(x)) with stable softplus
    if not x.is_cuda:
        # CPU fallback to exact PyTorch ops
        return torch.tanh(torch.nn.functional.mish(x))
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _mish_tanh_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        # Fused Triton kernel for Mish + Tanh
        x = fused_mish_tanh(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]