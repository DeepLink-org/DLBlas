# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl
import math

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight initialization
        self.weight = torch.nn.Parameter(torch.empty(
            out_channels, 
            in_channels, 
            kernel_size, 
            kernel_size
        ))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    @triton.jit
    def _transposed_conv2d_kernel(
        # Pointers to matrices
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        # Tensor dimensions
        batch, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        # Kernel parameters
        stride, padding, dilation, kernel_size,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid = tl.num_programs(0)
        num_output_elements = batch * out_channels * height_out * width_out
        output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = output_idx < num_output_elements
        
        # Decompose output index into components
        idx_b = output_idx // (out_channels * height_out * width_out)
        idx_oc = (output_idx % (out_channels * height_out * width_out)) // (height_out * width_out)
        idx_h = (output_idx % (height_out * width_out)) // width_out
        idx_w = output_idx % width_out
        
        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Loop over kernel positions
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                # Calculate input coordinates
                h_in = (idx_h + padding - kh * dilation) / stride
                w_in = (idx_w + padding - kw * dilation) / stride
                
                # Check integer positions and boundaries
                cond_h = (h_in >= 0) & (h_in < height_in) & (tl.abs(h_in - tl.math.floor(h_in + 0.5)) < 1e-5)
                cond_w = (w_in >= 0) & (w_in < width_in) & (tl.abs(w_in - tl.math.floor(w_in + 0.5)) < 1e-5)
                cond = cond_h & cond_w
                h_in_int = tl.math.floor(h_in + 0.5).to(tl.int32)
                w_in_int = tl.math.floor(w_in + 0.5).to(tl.int32)
                
                # Loop over input channels in blocks
                for ic_block in range(0, in_channels, BLOCK_SIZE_IC):
                    ic_offsets = ic_block + tl.arange(0, BLOCK_SIZE_IC)
                    ic_mask = ic_offsets < in_channels
                    
                    # Reshape indices for 2D broadcasting
                    idx_b_2d = tl.reshape(idx_b, (BLOCK_SIZE, 1))
                    idx_oc_2d = tl.reshape(idx_oc, (BLOCK_SIZE, 1))
                    h_in_int_2d = tl.reshape(h_in_int, (BLOCK_SIZE, 1))
                    w_in_int_2d = tl.reshape(w_in_int, (BLOCK_SIZE, 1))
                    cond_2d = tl.reshape(cond, (BLOCK_SIZE, 1))
                    mask_2d = tl.reshape(mask, (BLOCK_SIZE, 1))
                    ic_offsets_2d = tl.reshape(ic_offsets, (1, BLOCK_SIZE_IC))
                    ic_mask_2d = tl.reshape(ic_mask, (1, BLOCK_SIZE_IC))
                    
                    # Calculate input pointer offsets with 2D shape
                    input_offsets = (
                        idx_b_2d * (in_channels * height_in * width_in) +
                        ic_offsets_2d * (height_in * width_in) +
                        h_in_int_2d * width_in +
                        w_in_int_2d
                    )
                    
                    # Calculate weight pointer offsets with 2D shape
                    weight_offsets = (
                        idx_oc_2d * (in_channels * kernel_size * kernel_size) +
                        ic_offsets_2d * (kernel_size * kernel_size) +
                        kh * kernel_size +
                        kw
                    )
                    
                    # Load input and weight values with 2D masks
                    x_val = tl.load(
                        x_ptr + input_offsets,
                        mask=cond_2d & mask_2d & ic_mask_2d,
                        other=0.0
                    )
                    w_val = tl.load(
                        weight_ptr + weight_offsets,
                        mask=ic_mask_2d,
                        other=0.0
                    )
                    
                    # Accumulate with reduction over input channels
                    product = x_val * w_val
                    accumulator += tl.sum(product, axis=1)
        
        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + idx_oc, mask=mask, other=0.0)
            accumulator += bias_val
        
        # Store results
        tl.store(output_ptr + output_idx, accumulator, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height_in, width_in = x.shape
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output = torch.empty((batch, self.out_channels, height_out, width_out), device=x.device, dtype=x.dtype)
        
        # Flatten spatial dimensions for kernel
        x_flat = x.contiguous()
        weight_flat = self.weight.contiguous()
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Calculate grid size
        num_output_elements = output.numel()
        grid = lambda meta: (triton.cdiv(num_output_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        self._transposed_conv2d_kernel[grid](
            x_flat, weight_flat, bias_ptr, output,
            batch, self.in_channels, self.out_channels,
            height_in, width_in,
            height_out, width_out,
            self.stride, self.padding, self.dilation, self.kernel_size,
            BLOCK_SIZE=1024, BLOCK_SIZE_IC=32
        )
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================