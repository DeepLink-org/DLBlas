import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def expand_to_mhc_kernel(
    x_ptr, y_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_y_b, stride_y_s, stride_y_m, stride_y_h,
    BLOCK_H: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Map program ids to (b, s, m)
    b = pid_bs // S
    s = pid_bs - b * S
    m = pid_m

    # Offsets along hidden dimension
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    tl.max_contiguous(h_offsets, BLOCK_H)
    tl.multiple_of(h_offsets, 16)
    h_mask = h_offsets < H

    # Precompute base pointers for (b, s, m)
    base_x = b.to(tl.int64) * stride_x_b + s.to(tl.int64) * stride_x_s
    base_y = (b.to(tl.int64) * stride_y_b +
              s.to(tl.int64) * stride_y_s +
              m.to(tl.int64) * stride_y_m)

    # Compute input/output pointers
    x_ptrs = x_ptr + base_x + h_offsets.to(tl.int64) * stride_x_h
    y_ptrs = y_ptr + base_y + h_offsets.to(tl.int64) * stride_y_h

    # Hints for locality/coalescing
    tl.max_contiguous(x_ptrs, BLOCK_H)
    tl.max_contiguous(y_ptrs, BLOCK_H)

    # Load once and store; use L2 cache to improve reuse across M CTAs
    x_vals = tl.load(x_ptrs, mask=h_mask, other=0, cache_modifier=".cg")
    tl.store(y_ptrs, x_vals, mask=h_mask)


class ModelNew(nn.Module):
    """
    算子：expand_to_mhc
    功能：在输入张量的倒数第二维插入一个新维度，并将其扩展 mhc_mult 倍。
    输入形状：(..., hidden_size)
    输出形状：(..., mhc_mult, hidden_size)
    """
    def __init__(self, mhc_mult: int):
        super(ModelNew, self).__init__()
        self.mhc_mult = mhc_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行维度扩展操作。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (batch, seq_len, hidden_dim)。

        Returns:
            torch.Tensor: 扩展后的张量，形状为 (batch, seq_len, mhc_mult, hidden_dim)。
        """
        # Fast path: Triton kernel for CUDA tensors with 3D shape (B, S, H)
        if  x.ndim == 3:
            B, S, H = x.shape
            M = self.mhc_mult
            # Allocate contiguous output
            y = torch.empty((B, S, M, H), dtype=x.dtype, device=x.device)

            # Get strides in element units
            sx_b, sx_s, sx_h = x.stride()
            sy_b, sy_s, sy_m, sy_h = y.stride()

            # Larger tile and more warps for better bandwidth utilization
            BLOCK_H = 1024
            grid = (B * S, triton.cdiv(H, BLOCK_H), M)

            expand_to_mhc_kernel[grid](
                x, y,
                B, S, M, H,
                sx_b, sx_s, sx_h,
                sy_b, sy_s, sy_m, sy_h,
                BLOCK_H=BLOCK_H,
                num_warps=8,
                num_stages=2,
            )
            return y
        else:
            # Fallback to reference implementation for non-CUDA or non-3D cases
            original_shape = x.shape
            return x.unsqueeze(-2).expand(*original_shape[:-1], self.mhc_mult, original_shape[-1]).contiguous()


def get_init_inputs():
    """
    提供 Model 类初始化所需的参数。
    mhc_mult: 扩展倍数，参考原代码通常为 2, 4, 8 等。
    """
    return [4]  # 示例：mhc_mult = 4

def get_inputs():
    """
    提供 Model forward 函数所需的输入张量。
    参考原测试用例：n0=1, n1=1024, h=1280
    """
    # 创建一个形状为 (1, 1024, 1280) 的随机张量
    batch_size = 1
    seq_len = 1024
    hidden_dim = 1280
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return [x]