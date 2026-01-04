# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K = self.kernel_size
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        
        # Convert to FP16 for Tensor Core acceleration
        x_fp16 = x.to(torch.float16)
        weight_fp16 = self.weight.view(self.out_channels, -1).to(torch.float16)
        
        # Precompute flattened output dimensions
        M = B * H_out * W_out
        N = self.out_channels
        K_total = self.in_channels * K * K
        
        # Allocate output tensor (FP32 for accumulation)
        output = torch.empty(B, N, H_out, W_out, device=x.device, dtype=torch.float32)
        
        # Launch Triton kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
        _conv2d_forward[grid](
            x_fp16, weight_fp16, output,
            # Tensor dimensions
            B, C, H, W, N, K,
            # Convolution parameters
            self.stride, self.padding, self.dilation,
            H_out, W_out,
            # Tensor strides
            x_fp16.stride(0), x_fp16.stride(1), x_fp16.stride(2), x_fp16.stride(3),
            weight_fp16.stride(0), weight_fp16.stride(1),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            # Kernel parameters
            M, K_total,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias[None, :, None, None]
            
        return output

@triton.jit
def _conv2d_forward(
    # Pointers to tensors
    input_ptr, weight_ptr, output_ptr,
    # Tensor dimensions
    B, C, H, W, F, K,
    # Convolution parameters
    stride, padding, dilation,
    H_out, W_out,
    # Input tensor strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    # Weight tensor strides
    weight_out_channel_stride, weight_kernel_stride,
    # Output tensor strides
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    # Flattened dimensions
    M, K_total,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges for block processing
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for output positions
    m_mask = rm < M
    # Compute batch and spatial indices
    batch_idx = rm // (H_out * W_out)
    spatial_idx = rm % (H_out * W_out)
    h_out = spatial_idx // W_out
    w_out = spatial_idx % W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over kernel blocks
    for k in range(0, tl.cdiv(K_total, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = rk < K_total
        
        # Map kernel index to 3D coordinates
        kernel_idx = rk
        c = kernel_idx // (K * K)
        kh = (kernel_idx % (K * K)) // K
        kw = kernel_idx % K
        
        # Compute input positions
        h_in = h_out[:, None] * stride + kh[None, :] * dilation - padding
        w_in = w_out[:, None] * stride + kw[None, :] * dilation - padding
        
        # Create input pointer mask
        in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        full_mask = m_mask[:, None] & k_mask[None, :] & in_bounds
        
        # Compute input pointers
        input_offsets = (
            batch_idx[:, None] * input_batch_stride +
            c[None, :] * input_channel_stride +
            h_in * input_height_stride +
            w_in * input_width_stride
        )
        input_block = tl.load(input_ptr + input_offsets, mask=full_mask, other=0.0)
        
        # Load weight block
        weight_block = tl.load(
            weight_ptr + rn[None, :] * weight_out_channel_stride + rk[:, None] * weight_kernel_stride,
            mask=k_mask[:, None] & (rn[None, :] < F),
            other=0.0
        )
        
        # Accumulate matrix product
        input_block = input_block.to(tl.float16)
        weight_block = weight_block.to(tl.float16)
        acc += tl.dot(input_block, weight_block)
    
    # Compute output pointers
    output_offsets = (
        batch_idx[:, None] * output_batch_stride +
        rn[None, :] * output_channel_stride +
        h_out[:, None] * output_height_stride +
        w_out[:, None] * output_width_stride
    )
    tl.store(output_ptr + output_offsets, acc, mask=m_mask[:, None] & (rn[None, :] < F))

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================