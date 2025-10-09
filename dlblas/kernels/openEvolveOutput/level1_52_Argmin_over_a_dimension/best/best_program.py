# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _argmin_kernel(
    x_ptr,
    output_ptr,
    B, 
    M, 
    N,
    stride_b,
    stride_m,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid % N
    b = pid // N
    
    base_offset = b * stride_b + n * stride_n
    offsets = base_offset + tl.arange(0, BLOCK_SIZE) * stride_m
    mask = tl.arange(0, BLOCK_SIZE) < M
    
    vec = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    min_index = tl.argmin(vec, axis=0)
    tl.store(output_ptr + b * N + n, min_index.to(tl.int64))

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.dim() == 3:
            B, M, N = x.shape
            if M <= 256:
                x = x.contiguous()
                output = torch.empty(B, N, dtype=torch.int64, device=x.device)
                grid = (B * N,)
                _argmin_kernel[grid](
                    x, output, 
                    B, M, N,
                    x.stride(0), x.stride(1), x.stride(2),
                    BLOCK_SIZE=256
                )
                return output
        return torch.argmin(x, dim=self.dim)

batch_size = 16
dim1 = 256
dim2 = 256
dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================