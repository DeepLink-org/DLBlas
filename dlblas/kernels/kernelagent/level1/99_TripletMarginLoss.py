import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _triplet_margin_row_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    B, D,
    stride_a0, stride_a1,
    stride_p0, stride_p1,
    stride_n0, stride_n1,
    eps, margin,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = row < B

    a_row_ptr = anchor_ptr + row * stride_a0
    p_row_ptr = pos_ptr + row * stride_p0
    n_row_ptr = neg_ptr + row * stride_n0

    acc_ap = tl.zeros((), dtype=tl.float32)
    acc_an = tl.zeros((), dtype=tl.float32)

    for i in tl.static_range(N_ITERS):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = valid_row & (offs < D)

        a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offs * stride_p1, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offs * stride_n1, mask=mask, other=0.0)

        da = a - p
        dn = a - n

        acc_ap += tl.sum(da * da, axis=0)
        acc_an += tl.sum(dn * dn, axis=0)

    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)

    tl.store(out_ptr + row, loss, mask=valid_row)


def _triplet_margin_loss_triton(anchor: torch.Tensor,
                                positive: torch.Tensor,
                                negative: torch.Tensor,
                                margin: float = 1.0,
                                eps: float = 1e-6) -> torch.Tensor:
    # Fallback for non-CUDA tensors
    if anchor.device.type != "cuda":
        d_p = torch.nn.functional.pairwise_distance(anchor, positive, p=2, eps=eps)
        d_n = torch.nn.functional.pairwise_distance(anchor, negative, p=2, eps=eps)
        return torch.clamp(d_p - d_n + margin, min=0.0).mean()

    # Ensure same shapes and contiguous memory
    a = anchor.contiguous()
    p = positive.contiguous()
    n = negative.contiguous()

    # Use float32 for stable, numerically-consistent compute
    if a.dtype != torch.float32:
        a = a.float()
    if p.dtype != torch.float32:
        p = p.float()
    if n.dtype != torch.float32:
        n = n.float()

    B, D = a.shape
    out = torch.empty(B, device=a.device, dtype=torch.float32)

    # Heuristic tuning for better throughput on Hopper/H200
    if D >= 2048:
        BLOCK_SIZE = 1024
        num_warps = 8
        num_stages = 4
    elif D >= 1024:
        BLOCK_SIZE = 512
        num_warps = 8
        num_stages = 4
    elif D >= 512:
        BLOCK_SIZE = 256
        num_warps = 4
        num_stages = 3
    else:
        BLOCK_SIZE = 128
        num_warps = 4
        num_stages = 2

    N_ITERS = triton.cdiv(D, BLOCK_SIZE)

    grid = (B,)
    _triplet_margin_row_kernel[grid](
        a, p, n, out,
        B, D,
        a.stride(0), a.stride(1),
        p.stride(0), p.stride(1),
        n.stride(0), n.stride(1),
        eps, float(margin),
        BLOCK_SIZE=BLOCK_SIZE,
        N_ITERS=N_ITERS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.mean()


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = float(margin)
        self.eps = 1e-6

    def forward(self, anchor, positive, negative):
        return _triplet_margin_loss_triton(anchor, positive, negative, self.margin, self.eps)


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin