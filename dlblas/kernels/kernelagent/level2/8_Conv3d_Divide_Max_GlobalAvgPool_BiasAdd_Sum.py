import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _reduce_bcdhw_to_b_kernel(
    x_ptr,         # *float32/float16/bfloat16, tensor [B, C, D, H, W] flattened per-batch slice
    bias_ptr,      # *float32, tensor [C, 1, 1, 1]
    out_ptr,       # *float32/float16/bfloat16, tensor [B]
    stride_b,      # int, stride for batch dim of x in elements
    N,             # int, total elements per batch slice = C*D*H*W
    C,             # int, channels
    V,             # int, spatial volume = D*H*W
    bias_stride_c, # int, stride for channel dim of bias
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = x_ptr + pid * stride_b

    offs = tl.arange(0, BLOCK_SIZE)

    # Software-pipelined reduction over N elements for improved latency hiding
    acc = tl.zeros((), dtype=tl.float32)
    i = 0
    idx = i + offs
    mask = idx < N
    vals = tl.load(base + idx, mask=mask, other=0.0)
    i += BLOCK_SIZE
    while i < N:
        idx_n = i + offs
        mask_n = idx_n < N
        vals_n = tl.load(base + idx_n, mask=mask_n, other=0.0)
        # accumulate current tile while prefetching next
        acc += tl.sum(vals.to(tl.float32), axis=0)
        vals = vals_n
        mask = mask_n
        i += BLOCK_SIZE
    acc += tl.sum(vals.to(tl.float32), axis=0)

    # Sum bias over channels: sum_c bias[c]
    acc_b = tl.zeros((), dtype=tl.float32)
    j = 0
    idxb = j + offs
    maskb = idxb < C
    bvals = tl.load(bias_ptr + idxb * bias_stride_c, mask=maskb, other=0.0)
    j += BLOCK_SIZE
    while j < C:
        idxb_n = j + offs
        maskb_n = idxb_n < C
        bvals_n = tl.load(bias_ptr + idxb_n * bias_stride_c, mask=maskb_n, other=0.0)
        acc_b += tl.sum(bvals.to(tl.float32), axis=0)
        bvals = bvals_n
        maskb = maskb_n
        j += BLOCK_SIZE
    acc_b += tl.sum(bvals.to(tl.float32), axis=0)

    # Compute mean over spatial volume and add bias sum
    Vf = tl.full((), V, dtype=tl.float32)
    out_val = acc / Vf + acc_b
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    Optimized: folds division into convolution weights/bias, and fuses
    global average pooling + bias add + channel-sum into a single Triton reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # Fold division by constant into convolution weights/bias for fewer global memory ops
        w = self.conv.weight
        b = self.conv.bias
        x = F.conv3d(
            x,
            w / self.divisor,
            None if b is None else b / self.divisor,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        x = self.max_pool(x)

        # Triton fused reduction if on CUDA and summing over channels (dim=1)
        if x.is_cuda and self.sum_dim == 1:
            x = x.contiguous()
            bias = self.bias.contiguous()
            B, C, D, H, W = x.shape
            out = torch.empty((B, 1, 1, 1), device=x.device, dtype=x.dtype)

            grid = (B,)
            N = C * D * H * W
            V = D * H * W
            # Kernel computes: for each batch b -> sum_{c,d,h,w} x[b,c,d,h,w]/(D*H*W) + sum_c bias[c]
            _reduce_bcdhw_to_b_kernel[grid](
                x,
                bias,
                out.view(B),
                x.stride(0),
                N,
                C,
                V,
                bias.stride(0),
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=4,
            )
            return out

        # Fallback path (CPU or non-standard reduction dim): exact original semantics
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]