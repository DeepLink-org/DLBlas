# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    @triton.jit
    def _conv3d_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        bias_ptr,
        # Tensor dimensions
        B, IC, D, H, W,
        OC, K, 
        # Convolution parameters
        stride, padding, dilation,
        # Output dimensions
        OD, OH, OW,
        # Tensor strides
        x_stride_b, x_stride_ic, x_stride_d, x_stride_h, x_stride_w,
        w_stride_oc, w_stride_ic, w_stride_k0, w_stride_k1, w_stride_k2,
        out_stride_b, out_stride_oc, out_stride_d, out_stride_h, out_stride_w,
        # Blocking
        BLOCK_SIZE: tl.constexpr,
        total_elements: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        # Convert linear index to 5D output indices
        idx = tl.load(tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE, mask=mask)
        b = idx // (OC * OD * OH * OW)
        residual = idx % (OC * OD * OH * OW)
        oc = residual // (OD * OH * OW)
        residual = residual % (OD * OH * OW)
        od = residual // (OH * OW)
        residual = residual % (OH * OW)
        oh = residual // OW
        ow = residual % OW
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Loop over kernel dimensions and input channels
        for kd in range(K):
            for kh in range(K):
                for kw in range(K):
                    for ic in range(IC):
                        # Calculate input positions with dilation and padding
                        id = od * stride - padding + kd * dilation
                        ih = oh * stride - padding + kh * dilation
                        iw = ow * stride - padding + kw * dilation
                        
                        # Boundary check
                        in_bounds = (id >= 0) & (id < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        
                        if in_bounds:
                            # Calculate memory offsets
                            x_offset = b * x_stride_b + ic * x_stride_ic + id * x_stride_d + ih * x_stride_h + iw * x_stride_w
                            w_offset = oc * w_stride_oc + ic * w_stride_ic + kd * w_stride_k0 + kh * w_stride_k1 + kw * w_stride_k2
                            
                            # Load and accumulate
                            x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                            w_val = tl.load(weight_ptr + w_offset, mask=mask, other=0.0)
                            acc += x_val * w_val
        
        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
            acc += bias_val
        
        # Store result
        out_offset = b * out_stride_b + oc * out_stride_oc + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
        tl.store(output_ptr + out_offset, acc, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        D, H, W = x.shape[2:]
        OD = (D + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        OH = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        OW = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(
            x.shape[0], self.out_channels, OD, OH, OW,
            device=x.device, dtype=x.dtype
        )
        
        # Prepare tensor pointers
        total_elements = output.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        # Launch kernel
        self._conv3d_kernel[grid](
            x, self.weight, output, self.bias,
            # Tensor dimensions
            x.shape[0], self.in_channels, D, H, W,
            self.out_channels, self.kernel_size,
            # Convolution parameters
            self.stride, self.padding, self.dilation,
            # Output dimensions
            OD, OH, OW,
            # Tensor strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3), self.weight.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            # Blocking
            BLOCK_SIZE,
            total_elements
        )
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================