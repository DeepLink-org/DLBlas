# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsumexp_triton_kernel(
    input_ptr, 
    output_ptr,
    batch_size,
    height,
    width,
    n_channels,
    stride_batch,
    stride_channel,
    stride_height,
    stride_width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_spatial = batch_size * height * width
    if pid >= total_spatial:
        return

    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width

    base_offset = batch_idx * stride_batch + h * stride_height + w * stride_width
    max_val = -float('inf')
    total = 0.0

    for c in range(0, n_channels, BLOCK_SIZE):
        channel_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < n_channels
        offsets = base_offset + channel_offsets * stride_channel
        data = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
        chunk_max = tl.max(data, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    for c in range(0, n_channels, BLOCK_SIZE):
        channel_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < n_channels
        offsets = base_offset + channel_offsets * stride_channel
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        exp_data = tl.exp(data - max_val)
        chunk_sum = tl.sum(exp_data, axis=0)
        total += chunk_sum

    result = tl.log(total) + max_val
    tl.store(output_ptr + pid, result)

def triton_logsumexp(x: torch.Tensor):
    batch_size, n_channels, height, width = x.shape
    total_spatial = batch_size * height * width
    x_contig = x.contiguous()
    output = torch.empty(batch_size, height, width, device=x.device, dtype=x.dtype)
    output_1d = output.view(-1)

    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    grid = (total_spatial,)
    
    logsumexp_triton_kernel[grid](
        x_contig,
        output_1d,
        batch_size,
        height,
        width,
        n_channels,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        x_contig.stride(3),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output.unsqueeze(1)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        x_tanh = self.tanh(x_norm)
        x_hard_swish = self.hard_swish(x_tanh)
        x_res = x_conv + x_hard_swish
        x_logsumexp = triton_logsumexp(x_res)
        return x_logsumexp

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
# =================== EVOLVE-BLOCK-END ===================