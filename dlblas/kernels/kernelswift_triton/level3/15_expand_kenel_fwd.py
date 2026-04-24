import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def expand_to_mhc_kernel(
    x_ptr,  # *x* pointer, shape (L, H), contiguous
    y_ptr,  # *y* pointer, shape (L, M, H), contiguous
    L,      # total leading elements collapsed
    M,      # mhc_mult
    H,      # hidden size
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Program IDs along (L, ceil_div(M, BLOCK_M), ceil_div(H, BLOCK_H))
    pid_l = tl.program_id(0)
    pid_mb = tl.program_id(1)
    pid_ht = tl.program_id(2)

    # Offsets along H and M for this program
    offs_h = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_m = pid_mb * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Load a contiguous tile of H from x for this l
    x_row_ptr = x_ptr + pid_l * H + offs_h
    vals = tl.load(x_row_ptr, mask=mask_h, other=0)

    # Prepare a 2D tile pointer for y with shape (BLOCK_M, BLOCK_H)
    # y index: ((l * M + m) * H + h)
    y_row_offsets = (pid_l * M + offs_m) * H
    y_tile_ptrs = y_ptr + y_row_offsets[:, None] + offs_h[None, :]

    # Broadcast vals across the M dimension and store once
    tl.store(y_tile_ptrs, vals[None, :], mask=mask_m[:, None] & mask_h[None, :])


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
        original_shape = x.shape
        M = self.mhc_mult
        H = original_shape[-1]

        # Fallback to reference path for CPU or degenerate cases
        if (not x.is_cuda) or (H == 0) or (M == 0) or (x.numel() == 0):
            return x.unsqueeze(-2).expand(*original_shape[:-1], M, H).contiguous()

        # Ensure contiguous and flatten leading dims into L
        x_contig = x.contiguous()
        L = 1
        for d in original_shape[:-1]:
            L *= d
        x_flat = x_contig.view(L, H)

        # Allocate output tensor as (L, M, H), then reshape to final
        y_flat = torch.empty((L, M, H), dtype=x.dtype, device=x.device)

        # Kernel launch configuration tuning
        # Larger BLOCK_H improves bandwidth by reducing grid overhead; choose based on H
        if H >= 512:
            BLOCK_H = 512
            num_warps = 8
        elif H >= 256:
            BLOCK_H = 256
            num_warps = 8
        elif H >= 128:
            BLOCK_H = 128
            num_warps = 4
        else:
            BLOCK_H = 64
            num_warps = 2

        # Replicate multiple M slices per program; choose a small power of two for occupancy
        if M >= 16:
            BLOCK_M = 16
        elif M >= 8:
            BLOCK_M = 8
        elif M >= 4:
            BLOCK_M = 4
        elif M >= 2:
            BLOCK_M = 2
        else:
            BLOCK_M = 1

        grid = (L, (M + BLOCK_M - 1) // BLOCK_M, (H + BLOCK_H - 1) // BLOCK_H)

        expand_to_mhc_kernel[grid](
            x_flat, y_flat,
            L, M, H,
            BLOCK_H=BLOCK_H,
            BLOCK_M=BLOCK_M,
            num_warps=num_warps,
        )

        # Reshape back to (..., M, H)
        return y_flat.view(*original_shape[:-1], M, H)


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
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    return [x]