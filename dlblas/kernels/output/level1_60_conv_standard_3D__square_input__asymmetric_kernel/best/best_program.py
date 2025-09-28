# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Tensor dimensions
    B, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    # Strides
    stride_bx, stride_ix, stride_dx, stride_hx, stride_wx,
    stride_ow, stride_oc, stride_od, stride_oh,
    # Meta-parameters
    BLOCK_OC: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program indices
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    pid_ow = tl.program_id(4)
    
    # Create ranges for blocking
    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    d_offsets = pid_od * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(IC):
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    # Calculate input positions
                    d_pos = d_offsets + kd
                    h_pos = h_offsets + kh
                    w_pos = w_offsets + kw
                    
                    # Boundary check mask
                    d_mask = (d_pos < D) & (d_pos >= 0)
                    h_mask = (h_pos < H) & (h_pos >= 0)
                    w_mask = (w_pos < W) & (w_pos >= 0)
                    mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
                    
                    # Load input block
                    input_offsets = pid_b * stride_bx + ic * stride_ix + d_pos * stride_dx + h_pos * stride_hx + w_pos * stride_wx
                    input_val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
                    
                    # Load weights
                    weight_offsets = oc_offsets[:, None, None, None] * stride_oc + ic * stride_ix + kd * stride_dx + kh * stride_hx + kw * stride_wx
                    weight_val = tl.load(weight_ptr + weight_offsets)
                    
                    # Update accumulator
                    acc += input_val * weight_val
    
    # Prepare output offsets
    output_offsets = (
        pid_b * stride_bx + 
        oc_offsets[:, None, None, None] * stride_oc + 
        d_offsets * stride_od + 
        h_offsets * stride_oh + 
        w_offsets * stride_ow
    )
    
    # Store output
    output_mask = (
        (oc_offsets < OC)[:, None, None, None] &
        (d_offsets < OD)[None, :, None, None] &
        (h_offsets < OH)[None, None, :, None] &
        (w_offsets < OW)[None, None, None, :]
    )
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        k_w, k_h, k_d = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, k_d, k_h, k_w))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, depth, height, width = x.shape
        k_d, k_h, k_w = self.kernel_size
        
        # Calculate output dimensions
        out_d = (depth - k_d) // self.stride + 1
        out_h = (height - k_h) // self.stride + 1
        out_w = (width - k_w) // self.stride + 1
        
        # Allocate output tensor
        output = torch.empty(
            batch_size, self.out_channels, out_d, out_h, out_w,
            device=x.device, dtype=x.dtype
        )
        
        # Get strides
        stride_bx, stride_ix, stride_dx, stride_hx, stride_wx = x.stride()
        _, stride_oc, stride_od, stride_oh, stride_ow = output.stride()
        
        # Configure kernel grid
        grid = lambda meta: (
            batch_size,
            triton.cdiv(self.out_channels, meta['BLOCK_OC']),
            triton.cdiv(out_d, meta['BLOCK_D']),
            triton.cdiv(out_h, meta['BLOCK_H']),
            triton.cdiv(out_w, meta['BLOCK_W']),
        )
        
        # Launch kernel
        conv3d_kernel[grid](
            x, self.weight, output,
            batch_size, in_channels, depth, height, width,
            self.out_channels, k_d, k_h, k_w,
            out_d, out_h, out_w,
            stride_bx, stride_ix, stride_dx, stride_hx, stride_wx,
            stride_ow, stride_oc, stride_od, stride_oh,
            BLOCK_OC=32,
            BLOCK_D=4,
            BLOCK_H=8,
            BLOCK_W=8,
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda')
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================