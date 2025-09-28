# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    @triton.jit
    def _fused_kernel(
        input_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        height,
        width,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE: tl.constexpr,
        channels: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= batch_size:
            return
        
        spatial_size = height * width
        batch_offset = pid * stride_b
        channel_vals = tl.zeros((channels,), dtype=tl.float32)

        # Process each channel sequentially
        for c in tl.static_range(channels):
            accum = 0.0
            # Block processing of spatial dimensions
            for i in range(0, spatial_size, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < spatial_size
                h = offsets // width
                w = offsets % width
                ptr = input_ptr + batch_offset + c * stride_c + h * stride_h + w * stride_w
                data = tl.load(ptr, mask=mask, other=0.0)
                accum += tl.sum(data, axis=0)
            
            channel_avg = accum / spatial_size
            bias_val = tl.load(bias_ptr + c)
            channel_vals += tl.where(tl.arange(0, channels) == c, channel_avg + bias_val, 0.0)

        # Log-sum-exp with numerical stability
        max_val = tl.max(channel_vals, axis=0)
        exp_vals = tl.exp(channel_vals - max_val)
        exp_sum = tl.sum(exp_vals, axis=0)
        result = max_val + tl.log(exp_sum)
        tl.store(output_ptr + pid, result * 10.0)

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        x = x.contiguous()
        output = torch.empty(B, device=x.device, dtype=torch.float32)
        
        grid = (B,)
        self._fused_kernel[grid](
            x, 
            self.bias.reshape(-1), 
            output,
            B,
            H, 
            W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_SIZE=128,
            channels=C
        )
        return output.unsqueeze(1)

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================