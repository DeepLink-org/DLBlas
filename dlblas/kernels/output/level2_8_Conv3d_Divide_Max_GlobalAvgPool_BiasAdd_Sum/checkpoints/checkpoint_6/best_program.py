# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # Store parameters for convolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        # Initialize convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
    def forward(self, x):
        # Use Triton for fused convolution and division
        x = self.triton_conv_forward(x, self.weight, self.divisor)
        
        # Continue with PyTorch for remaining operations
        x = nn.functional.max_pool3d(x, self.pool_size)
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_D': 4, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
            triton.Config({'BLOCK_D': 4, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=8),
            triton.Config({'BLOCK_D': 4, 'BLOCK_H': 32, 'BLOCK_W': 16}, num_warps=4),
            triton.Config({'BLOCK_D': 4, 'BLOCK_H': 32, 'BLOCK_W': 16}, num_warps=8),
        ],
        key=['output_depth', 'output_height', 'output_width'],
    )
    @triton.jit
    def _conv3d_forward_kernel(
        x_ptr, weight_ptr, output_ptr,
        # Tensor dimensions
        batch_size, in_channels, input_depth, input_height, input_width,
        out_channels, kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        divisor,
        # Strides
        x_batch_stride, x_channel_stride, x_d_stride, x_h_stride, x_w_stride,
        weight_oc_stride, weight_ic_stride, weight_d_stride, weight_h_stride, weight_w_stride,
        output_batch_stride, output_oc_stride, output_d_stride, output_h_stride, output_w_stride,
        BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_batch = tl.program_id(0)
        pid_oc = tl.program_id(1)
        pid_d = tl.program_id(2)
        pid_h = tl.program_id(3)
        pid_w = tl.program_id(4)
        
        # Output block indices
        d0 = pid_d * BLOCK_D
        h0 = pid_h * BLOCK_H
        w0 = pid_w * BLOCK_W
        
        # Offsets in output block
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        w_offsets = w0 + tl.arange(0, BLOCK_W)
        
        # Boundary check masks
        d_mask = d_offsets < output_depth
        h_mask = h_offsets < output_height
        w_mask = w_offsets < output_width
        block_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
        
        # Initialize output block
        output_block = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
        
        # Loop over input channels and kernel dimensions
        for ic in range(0, in_channels):
            for kd in range(0, kernel_depth):
                for kh in range(0, kernel_height):
                    for kw in range(0, kernel_width):
                        # Input indices (accounting for padding)
                        input_d = d0 * stride_d + kd - padding_d
                        input_h = h0 * stride_h + kh - padding_h
                        input_w = w0 * stride_w + kw - padding_w
                        
                        # Check input boundaries
                        input_d_valid = (input_d >= 0) & (input_d < input_depth)
                        input_h_valid = (input_h >= 0) & (input_h < input_height)
                        input_w_valid = (input_w >= 0) & (input_w < input_width)
                        valid_mask = input_d_valid & input_h_valid & input_w_valid
                        
                        # Load input block
                        input_ptr_offset = (
                            pid_batch * x_batch_stride +
                            ic * x_channel_stride +
                            input_d * x_d_stride +
                            input_h * x_h_stride +
                            input_w * x_w_stride
                        )
                        input_block = tl.load(
                            x_ptr + input_ptr_offset + 
                            tl.arange(0, BLOCK_D)[:, None, None] * x_d_stride +
                            tl.arange(0, BLOCK_H)[None, :, None] * x_h_stride +
                            tl.arange(0, BLOCK_W)[None, None, :] * x_w_stride,
                            mask=block_mask & valid_mask,
                            other=0.0
                        )
                        
                        # Load weight value
                        weight_val = tl.load(
                            weight_ptr +
                            pid_oc * weight_oc_stride +
                            ic * weight_ic_stride +
                            kd * weight_d_stride +
                            kh * weight_h_stride +
                            kw * weight_w_stride
                        )
                        
                        # Accumulate
                        output_block += input_block * weight_val
        
        # Apply divisor
        output_block = output_block / divisor
        
        # Store output block
        output_ptr_offset = (
            pid_batch * output_batch_stride +
            pid_oc * output_oc_stride +
            d0 * output_d_stride +
            h0 * output_h_stride +
            w0 * output_w_stride
        )
        tl.store(
            output_ptr + output_ptr_offset +
            tl.arange(0, BLOCK_D)[:, None, None] * output_d_stride +
            tl.arange(0, BLOCK_H)[None, :, None] * output_h_stride +
            tl.arange(0, BLOCK_W)[None, None, :] * output_w_stride,
            output_block,
            mask=block_mask
        )
    
    def triton_conv_forward(self, x, weight, divisor):
        # Compute output dimensions
        _, _, D_in, H_in, W_in = x.shape
        K_D, K_H, K_W = self.kernel_size
        D_out = D_in - K_D + 1
        H_out = H_in - K_H + 1
        W_out = W_in - K_W + 1
        
        # Preallocate output tensor
        output = torch.empty(
            (x.size(0), self.out_channels, D_out, H_out, W_out),
            device=x.device,
            dtype=x.dtype
        )
        
        # Launch kernel with appropriate grid
        grid = (
            x.size(0),  # batch size
            self.out_channels,  # output channels
            triton.cdiv(D_out, 4),  # adjust based on block size
            triton.cdiv(H_out, 16),
            triton.cdiv(W_out, 16),
        )
        
        # Call Triton kernel
        ModelNew._conv3d_forward_kernel[grid](
            x, weight, output,
            # Tensor dimensions
            x.size(0), self.in_channels, D_in, H_in, W_in,
            self.out_channels, K_D, K_H, K_W,
            D_out, H_out, W_out,
            1, 1, 1,  # strides
            0, 0, 0,  # padding
            divisor,
            # Tensor strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            BLOCK_D=4, BLOCK_H=16, BLOCK_W=16
        )
        return output

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