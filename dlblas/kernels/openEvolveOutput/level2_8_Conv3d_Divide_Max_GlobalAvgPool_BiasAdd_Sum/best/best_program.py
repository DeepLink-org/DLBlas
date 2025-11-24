# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _global_avg_pool3d(
    x_ptr,
    out_ptr,
    batch_size,
    C,
    D, H, W,
    x_stride_b, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    if pid_b >= batch_size or pid_c >= C:
        return

    total = 0.0
    numel = D * H * W
    for d in range(D):
        for h in range(H):
            for w in range(W):
                offset = pid_b * x_stride_b + pid_c * x_stride_c + d * x_stride_d + h * x_stride_h + w * x_stride_w
                total += tl.load(x_ptr + offset)
    
    avg = total / numel
    out_index = pid_b * C + pid_c
    tl.store(out_ptr + out_index, avg)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        
        batch, C, D, H, W = x.shape
        avg_out = torch.empty((batch, C), device=x.device, dtype=x.dtype)
        grid = (batch, C)
        _global_avg_pool3d[grid](
            x, avg_out,
            batch, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            BLOCK_SIZE=1
        )
        
        bias_reshaped = self.bias.view(C)
        x = avg_out + bias_reshaped
        x = x.sum(dim=self.sum_dim, keepdim=True)
        x = x.view(batch, 1, 1, 1)
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
# =================== EVOLVE-BLOCK-END ===================