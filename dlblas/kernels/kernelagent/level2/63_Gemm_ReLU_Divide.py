import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _relu_divide_inplace_kernel(x_ptr, n_elements, divisor, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Coalesced load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ReLU + scale by reciprocal (avoid per-element division)
    inv = 1.0 / divisor
    x = tl.where(x > 0, x * inv, 0.0)

    # Coalesced store
    tl.store(x_ptr + offsets, x, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    Uses cuBLAS for GEMM and selectively applies a Triton fused kernel for large tensors to reduce overhead.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        # Compute linear layer using optimized GEMM (cuBLAS via PyTorch)
        y = self.linear(x)

        # Use Triton only when profitable to avoid launch overhead on small tensors
        if y.is_cuda:
            n_elements = y.numel()
            use_triton = n_elements >= 1_000_000  # Heuristic threshold

            if use_triton:
                y = y.contiguous()

                def grid(meta):
                    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                _relu_divide_inplace_kernel[grid](
                    y, n_elements, float(self.divisor),
                    BLOCK_SIZE=4096, num_warps=4, num_stages=2
                )
                return y
            else:
                # Fast in-place fallback minimizes overhead for smaller tensors
                y.relu_()
                y.div_(self.divisor)
                return y
        else:
            # CPU fallback
            y = torch.relu(y)
            y = y / self.divisor
            return y


batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]