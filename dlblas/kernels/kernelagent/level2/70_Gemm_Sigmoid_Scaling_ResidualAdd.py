import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sigmoid_scale_residual_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Streaming load to reduce L1 pollution for this purely streaming op
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    # Compute in fp32 for stability/consistency with PyTorch
    x_fp32 = x.to(tl.float32)

    # Sigmoid and fused epilogue: y = x + scale * sigmoid(x)
    s = tl.sigmoid(x_fp32)
    y = x_fp32 + s * scale

    # Cast back to original dtype before store; streaming write
    y_cast = y.to(x.dtype)
    tl.store(out_ptr + offsets, y_cast, mask=mask, eviction_policy="evict_first")


class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # GEMM via cuBLAS; fuse post-ops (sigmoid, scale, residual add) efficiently.
        x = self.gemm(x)
        if x.is_cuda:
            # For smaller tensors, the overhead of a custom kernel can dominate.
            # Use efficient PyTorch fused expression in that regime.
            n_elements = x.numel()
            SMALL_THRESHOLD = 131072  # tuned for H100/H200 elementwise epilogues
            if n_elements < SMALL_THRESHOLD:
                # y = x + scale * sigmoid(x) using a single fused add
                return torch.add(x, torch.sigmoid(x), alpha=self.scaling_factor)

            # Otherwise, run the Triton fused epilogue
            if not x.is_contiguous():
                x = x.contiguous()

            # Heuristic tuning for H100/H200: prefer larger blocks and modest warps
            if n_elements >= (1 << 22):      # >= 4M elements
                BLOCK_SIZE = 16384
                NUM_WARPS = 8
                NUM_STAGES = 2
            elif n_elements >= (1 << 18):    # >= 262k elements
                BLOCK_SIZE = 8192
                NUM_WARPS = 4
                NUM_STAGES = 1
            else:                             # >= 131k elements (since small handled above)
                BLOCK_SIZE = 8192
                NUM_WARPS = 4
                NUM_STAGES = 1

            grid = lambda META: (triton.cdiv(n_elements, BLOCK_SIZE),)
            _sigmoid_scale_residual_kernel[grid](
                x, x, n_elements, self.scaling_factor,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS, num_stages=NUM_STAGES
            )
            return x
        else:
            # CPU fallback preserves original semantics
            original_x = x
            x = torch.sigmoid(x)
            x = x * self.scaling_factor
            x = x + original_x
            return x


batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]