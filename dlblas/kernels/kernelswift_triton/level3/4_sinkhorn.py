import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sinkhorn_kernel(
    x_ptr, y_ptr,
    N0, N1, MH,
    s0, s1, s2, s3,
    eps,
    REPEAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i0 = pid // N1
    i1 = pid - i0 * N1
    base_off = i0 * s0 + i1 * s1

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask_r = rows < MH
    mask_c = cols < MH
    mask = mask_r[:, None] & mask_c[None, :]

    offs = base_off + rows[:, None] * s2 + cols[None, :] * s3
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # 1) softmax along last dim (-1): row-wise softmax (stable)
    row_max = tl.max(tl.where(mask, x, -1.0e30), axis=1)
    x_center = x - row_max[:, None]
    ex = tl.exp(x_center)
    ex = tl.where(mask, ex, 0.0)
    row_sum = tl.sum(ex, axis=1)
    denom_row = tl.where(mask_r, row_sum, 1.0)
    y = ex / denom_row[:, None]
    # + eps on valid elements only
    y = tl.where(mask, y + eps, 0.0)

    # 2) column-normalize: divide by sum over rows (-2) + eps
    col_sum = tl.sum(y, axis=0)
    denom_col = tl.where(mask_c, col_sum + eps, 1.0)
    y = y / denom_col[None, :]

    # 3) repeat-1 times: row-normalize then column-normalize
    for _ in range(REPEAT - 1):
        # row-normalize: divide by sum over columns (-1) + eps
        rsum = tl.sum(y, axis=1)
        drow = tl.where(mask_r, rsum + eps, 1.0)
        y = y / drow[:, None]
        # column-normalize: divide by sum over rows (-2) + eps
        csum = tl.sum(y, axis=0)
        dcol = tl.where(mask_c, csum + eps, 1.0)
        y = y / dcol[None, :]

    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Triton implementation of sinkhorn_normalize.
    Iteratively normalizes a matrix to be doubly stochastic:
      1. softmax(x, dim=-1) + eps
      2. column-normalize: x / (x.sum(-2) + eps)
      3. repeat (row-normalize then column-normalize) for repeat-1 iterations
    """
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., mhc, mhc] float32
        Returns:
            [..., mhc, mhc] float32  (doubly stochastic)
        """
        # CPU fallback for safety
        if not x.is_cuda:
            y = x.softmax(-1) + self.eps
            y = y / (y.sum(-2, keepdim=True) + self.eps)
            for _ in range(self.repeat - 1):
                y = y / (y.sum(-1, keepdim=True) + self.eps)
                y = y / (y.sum(-2, keepdim=True) + self.eps)
            return y

        # Expect shape [n0, n1, mhc, mhc]
        assert x.dim() == 4, "Expected input of shape [n0, n1, mhc, mhc]"
        n0, n1, mhc, mhc2 = x.shape
        assert mhc == mhc2, "Last two dimensions must be equal"

        y = torch.empty_like(x)
        s0, s1, s2, s3 = x.stride()
        grid = (n0 * n1,)

        # Tuned for small tiles: use a single warp for minimal overhead
        BLOCK = 16

        sinkhorn_kernel[grid](
            x, y,
            n0, n1, mhc,
            s0, s1, s2, s3,
            self.eps,
            REPEAT=self.repeat,
            BLOCK=BLOCK,
            num_warps=1,
        )
        return y


n0 = 1
n1 = 1024
mhc = 4
def get_inputs():
    x = torch.randn(n0, n1, mhc, mhc)
    return [x]
def get_init_inputs():
    return []