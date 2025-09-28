# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    height,
    width,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    out_h,
    out_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Compute output element indices
    total_outputs = out_channels * out_h * out_w
    oc = pid // (out_h * out_w)
    pos = pid % (out_h * out_w)
    oh = pos // out_w
    ow = pos % out_w
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    
    # Loop over kernel positions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input positions with dilation
            ih = oh * stride + kh * dilation - padding
            iw = ow * stride + kw * dilation - padding
            
            # Check input boundaries
            within_bounds = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
            
            # Process input channels in blocks
            for ic_block in range(0, in_channels, BLOCK_SIZE_C):
                ic_offsets = ic_block + tl.arange(0, BLOCK_SIZE_C)
                mask = ic_offsets < in_channels
                
                # Load input block if within bounds
                if within_bounds:
                    x_ptrs = x_ptr + (ih * width + iw) * in_channels + ic_offsets
                    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
                else:
                    x_vals = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
                
                # Load weight block
                w_ptrs = weight_ptr + (oc * in_channels * kernel_size * kernel_size + 
                                      (kh * kernel_size + kw) * in_channels + ic_offsets)
                w_vals = tl.load(w_ptrs, mask=mask, other=0.0)
                
                # Accumulate
                acc += x_vals * w_vals
    
    # Reduce and store
    result = tl.sum(acc, axis=0)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc)
        result += bias
        
    tl.store(output_ptr + pid, result)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        if groups != 1:
            raise ValueError("Only groups=1 is supported")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        out_h = (x.shape[2] + 2 * self.padding - 
                 self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (x.shape[3] + 2 * self.padding - 
                 self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Prepare output tensor
        output = torch.empty(
            x.shape[0], self.out_channels, out_h, out_w,
            device=x.device, dtype=x.dtype
        )
        
        # Process each batch element
        for b in range(x.shape[0]):
            # Flatten spatial dimensions for kernel
            flat_output = output[b].view(self.out_channels, out_h * out_w)
            
            # Launch kernel
            grid = (self.out_channels * out_h * out_w,)
            conv2d_kernel[grid](
                x[b].contiguous(),
                self.weight.contiguous(),
                self.bias.contiguous() if self.bias is not None else None,
                flat_output,
                self.in_channels,
                x.shape[2],
                x.shape[3],
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                out_h,
                out_w,
                BLOCK_SIZE_C=32,
                BLOCK_SIZE_H=1,
                BLOCK_SIZE_W=1,
            )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128  # Asymmetric input

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================