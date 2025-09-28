# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        
        # Precompute output dimensions
        self.input_dim = (32, 32, 32)
        self.output_dim = (
            (self.input_dim[0] - 1) * stride - 2 * padding + kernel_size,
            (self.input_dim[1] - 1) * stride - 2 * padding + kernel_size,
            (self.input_dim[2] - 1) * stride - 2 * padding + kernel_size
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Optimized Triton kernel for transposed convolution
        B, C_in, D_in, H_in, W_in = x.shape
        C_out = self.conv_transpose.out_channels
        D_out, H_out, W_out = self.output_dim
        
        # Allocate output tensor
        conv_output = torch.empty((B, C_out, D_out, H_out, W_out), 
                                 device=x.device, dtype=x.dtype)
        
        # Precompute strides for coalesced memory access
        grid = (B * C_out * D_out, H_out)
        
        # Launch optimized Triton kernel
        self._transposed_conv3d_kernel[grid](
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias,
            conv_output,
            C_in, C_out, 
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            self.kernel_size, self.stride, self.padding,
            BLOCK_SIZE=32
        )
        
        # Continue with remaining operations
        x = self.batch_norm(conv_output)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x

    @triton.jit
    def _transposed_conv3d_kernel(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        in_chans, out_chans,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size: tl.constexpr, 
        stride: tl.constexpr, 
        padding: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        # 3D grid: [batch*out_channels*depth, height]
        pid = tl.program_id(0)
        h_out = tl.program_id(1)
        w_range = tl.arange(0, BLOCK_SIZE)
        
        # Decompose first dimension
        batch_idx = pid // (out_chans * D_out)
        chan_idx = (pid % (out_chans * D_out)) // D_out
        d_out = pid % D_out
        
        # Initialize output block
        output_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Loop over kernel dimensions and input channels
        for c_in in range(in_chans):
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Compute input indices
                        d_in = (d_out - kd + padding) / stride
                        h_in = (h_out - kh + padding) / stride
                        
                        # Only process integer positions
                        if d_in == tl.math.floor(d_in) and h_in == tl.math.floor(h_in):
                            d_in = tl.math.floor(d_in)
                            h_in = tl.math.floor(h_in)
                            
                            # Calculate w_in positions for the entire block
                            w_in = (w_range - kw + padding) / stride
                            w_in_floor = tl.math.floor(w_in)
                            
                            # Check valid positions
                            valid = (
                                (d_in >= 0) & (d_in < D_in) & 
                                (h_in >= 0) & (h_in < H_in) & 
                                (w_in_floor >= 0) & (w_in_floor < W_in) &
                                (w_in == w_in_floor)
                            )
                            
                            # Load input if valid
                            input_idx = (
                                batch_idx * in_chans * D_in * H_in * W_in +
                                c_in * D_in * H_in * W_in +
                                d_in * H_in * W_in +
                                h_in * W_in +
                                w_in_floor
                            )
                            x_val = tl.load(x_ptr + input_idx, mask=valid, other=0.0)
                            
                            # Load weight
                            weight_idx = (
                                chan_idx * in_chans * kernel_size**3 +
                                c_in * kernel_size**3 +
                                kd * kernel_size**2 +
                                kh * kernel_size +
                                kw
                            )
                            w_val = tl.load(weight_ptr + weight_idx)
                            
                            # Accumulate
                            output_block += tl.where(valid, x_val * w_val, 0.0)
        
        # Add bias
        bias = tl.load(bias_ptr + chan_idx)
        output_block += bias
        
        # Store output block
        output_idx = (
            batch_idx * out_chans * D_out * H_out * W_out +
            chan_idx * D_out * H_out * W_out +
            d_out * H_out * W_out +
            h_out * W_out +
            w_range
        )
        tl.store(output_ptr + output_idx, output_block, mask=w_range < W_out)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================