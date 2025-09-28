# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels // groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, width, height = x.shape
        kernel_d, kernel_w, kernel_h = self.kernel_size
        
        # Calculate output dimensions
        depth_out = (depth - 1) * self.stride[0] - 2 * self.padding[0] + kernel_d + self.output_padding[0]
        width_out = (width - 1) * self.stride[1] - 2 * self.padding[1] + kernel_w + self.output_padding[1]
        height_out = (height - 1) * self.stride[2] - 2 * self.padding[2] + kernel_h + self.output_padding[2]
        
        # Create output tensor
        output = torch.zeros(
            batch_size, 
            self.out_channels, 
            depth_out, 
            width_out, 
            height_out, 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Flatten tensors for kernel processing
        x_flat = x.reshape(-1)
        weight_flat = self.weight.reshape(-1)
        output_flat = output.reshape(-1)
        
        # Calculate total output elements
        total_elements = output_flat.numel()
        
        # Grid configuration - one thread per output element
        grid = (total_elements,)
        
        # Launch kernel
        self._conv_transpose3d_kernel[grid](
            x_flat, weight_flat, output_flat,
            batch_size, self.in_channels, self.out_channels,
            depth, width, height,
            depth_out, width_out, height_out,
            kernel_d, kernel_w, kernel_h,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            total_elements
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output

    @triton.jit
    def _conv_transpose3d_kernel(
        x_ptr, weight_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        depth, width, height,
        depth_out, width_out, height_out,
        kernel_d, kernel_w, kernel_h,
        stride_d, stride_w, stride_h,
        padding_d, padding_w, padding_h,
        total_elements
    ):
        pid = tl.program_id(0)
        if pid >= total_elements:
            return
        
        # Convert flat index to 5D output tensor indices
        h_out = pid % height_out
        w_out = (pid // height_out) % width_out
        d_out = (pid // (height_out * width_out)) % depth_out
        c_out = (pid // (height_out * width_out * depth_out)) % out_channels
        b_out = pid // (height_out * width_out * depth_out * out_channels)
        
        # Initialize accumulator
        acc = 0.0
        
        # Compute input indices
        d_in = (d_out + padding_d) / stride_d
        w_in = (w_out + padding_w) / stride_w
        h_in = (h_out + padding_h) / stride_h
        
        # Check if input indices are integers
        d_in_int = tl.abs(d_in - tl.math.floor(d_in)) < 1e-5
        w_in_int = tl.abs(w_in - tl.math.floor(w_in)) < 1e-5
        h_in_int = tl.abs(h_in - tl.math.floor(h_in)) < 1e-5
        
        # Convert to integers for valid positions
        d_in_val = tl.math.floor(d_in).to(tl.int32)
        w_in_val = tl.math.floor(w_in).to(tl.int32)
        h_in_val = tl.math.floor(h_in).to(tl.int32)
        
        # Check input bounds
        in_bounds = (d_in_val >= 0) & (d_in_val < depth) & \
                   (w_in_val >= 0) & (w_in_val < width) & \
                   (h_in_val >= 0) & (h_in_val < height)
        
        # Compute kernel indices
        k_d = (d_out + padding_d) - d_in_val * stride_d
        k_w = (w_out + padding_w) - w_in_val * stride_w
        k_h = (h_out + padding_h) - h_in_val * stride_h
        
        # Check kernel bounds
        kernel_bounds = (k_d >= 0) & (k_d < kernel_d) & \
                       (k_w >= 0) & (k_w < kernel_w) & \
                       (k_h >= 0) & (k_h < kernel_h)
        
        # Combine conditions for valid computation
        valid = d_in_int & w_in_int & h_in_int & in_bounds & kernel_bounds
        
        if valid:
            # Loop over input channels
            for c_in in range(0, in_channels):
                # Compute flat indices
                x_idx = b_out * in_channels * depth * width * height + \
                        c_in * depth * width * height + \
                        d_in_val * width * height + \
                        w_in_val * height + \
                        h_in_val
                
                weight_idx = c_in * out_channels * kernel_d * kernel_w * kernel_h + \
                             c_out * kernel_d * kernel_w * kernel_h + \
                             k_d * kernel_w * kernel_h + \
                             k_w * kernel_h + \
                             k_h
                
                # Load values
                x_val = tl.load(x_ptr + x_idx)
                w_val = tl.load(weight_ptr + weight_idx)
                acc += x_val * w_val
        
        # Store result
        tl.store(output_ptr + pid, acc)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================