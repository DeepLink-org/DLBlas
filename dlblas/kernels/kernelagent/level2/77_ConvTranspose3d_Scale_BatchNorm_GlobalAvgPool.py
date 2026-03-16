import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=5),
    ],
    key=["L"],
)
@triton.jit
def _global_avg_pool3d_ncdhw_kernel(x_ptr, y_ptr, M, L: tl.constexpr, BLOCK: tl.constexpr):
    # One program per (n, c) pair; reduce over contiguous DHW block of length L.
    pid = tl.program_id(axis=0)
    valid_pid = pid < M

    base = pid * L
    offs = tl.arange(0, BLOCK)

    # Software pipelined reduction with fp32 accumulation
    total = 0.0
    # Preload first chunk
    idx0 = offs
    mask0 = valid_pid & (idx0 < L)
    vals0 = tl.load(x_ptr + base + idx0, mask=mask0, other=0.0).to(tl.float32)

    for start in range(BLOCK, L, BLOCK):
        idx1 = start + offs
        mask1 = valid_pid & (idx1 < L)
        vals1 = tl.load(x_ptr + base + idx1, mask=mask1, other=0.0).to(tl.float32)
        # Reduce previously loaded chunk while the next is in flight
        total += tl.sum(vals0, axis=0)
        vals0 = vals1

    # Final chunk reduction
    total += tl.sum(vals0, axis=0)

    mean = total * (1.0 / L)
    tl.store(y_ptr + pid, mean, mask=valid_pid)


def global_avg_pool3d_triton(x: torch.Tensor) -> torch.Tensor:
    # x is expected to be [N, C, D, H, W] and contiguous in memory (NCDHW)
    assert x.ndim == 5, "Input must be NCDHW"
    N, C, D, H, W = x.shape
    M = N * C
    L = D * H * W

    # Ensure contiguous for correct linearization over DHW
    x_contig = x.contiguous()
    y = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    # Flatten views for kernel pointers
    x_flat = x_contig.view(-1)
    y_flat = y.view(-1)

    grid = (M,)
    _global_avg_pool3d_ncdhw_kernel[grid](x_flat, y_flat, M, L=L)
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        # Keep for API compatibility; pooling may use Triton kernel on CUDA
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # Fuse the scalar scaling into the ConvTranspose3d to remove an extra tensor-wide multiply.
        w = self.conv_transpose.weight * self.scale_factor
        b = None if self.conv_transpose.bias is None else (self.conv_transpose.bias * self.scale_factor)
        x = F.conv_transpose3d(
            x,
            w,
            bias=b,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation,
        )

        # If BatchNorm is in eval mode and tracking running stats, we can safely commute
        # BN and GlobalAvgPool: Avg(BN(x)) == BN(Avg(x)). This avoids full-tensor BN work.
        use_commute = (not self.batch_norm.training) and getattr(self.batch_norm, "track_running_stats", True)
        if use_commute:
            # Global average pool first (fast Triton on CUDA, fallback on CPU)
            if x.is_cuda:
                x = global_avg_pool3d_triton(x)
            else:
                x = self.global_avg_pool(x)
            # Apply BN using running statistics and affine params on the pooled tensor
            dtype = x.dtype
            rm = self.batch_norm.running_mean.to(dtype).view(1, -1, 1, 1, 1)
            rv = self.batch_norm.running_var.to(dtype).view(1, -1, 1, 1, 1)
            inv_std = torch.rsqrt(rv + self.batch_norm.eps)

            if self.batch_norm.affine:
                weight = self.batch_norm.weight.to(dtype).view(1, -1, 1, 1, 1)
                bias = self.batch_norm.bias.to(dtype).view(1, -1, 1, 1, 1)
            else:
                weight = torch.ones(1, x.size(1), 1, 1, 1, device=x.device, dtype=dtype)
                bias = torch.zeros(1, x.size(1), 1, 1, 1, device=x.device, dtype=dtype)

            x = (x - rm) * (weight * inv_std) + bias
            return x
        else:
            # Training mode or not tracking running stats: follow original ordering
            x = self.batch_norm(x)
            if x.is_cuda:
                x = global_avg_pool3d_triton(x)
            else:
                x = self.global_avg_pool(x)
            return x


batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]