import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _cosine_similarity_rows_kernel(
    x_ptr, y_ptr, out_ptr,
    B, D,
    stride_xb, stride_xd,
    stride_yb, stride_yd,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_mask = pid < B

    offs = tl.arange(0, BLOCK_SIZE)
    mask = (offs < D) & row_mask

    x_row_ptr = x_ptr + pid * stride_xb
    y_row_ptr = y_ptr + pid * stride_yb

    x = tl.load(x_row_ptr + offs * stride_xd, mask=mask, other=0.0)
    y = tl.load(y_row_ptr + offs * stride_yd, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)

    dot = tl.sum(x32 * y32, axis=0)
    nx2 = tl.sum(x32 * x32, axis=0)
    ny2 = tl.sum(y32 * y32, axis=0)

    denom = tl.sqrt(nx2) * tl.sqrt(ny2)
    denom = tl.maximum(denom, EPS)
    loss = 1.0 - (dot / denom)

    tl.store(out_ptr + pid, loss, mask=row_mask)


class ModelNew(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Fallback to PyTorch if Triton not available or inputs not CUDA / 2D / matching
        if (
            not TRITON_AVAILABLE
            or (not predictions.is_cuda)
            or (not targets.is_cuda)
            or predictions.ndim != 2
            or targets.ndim != 2
            or predictions.shape != targets.shape
        ):
            cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
            return torch.mean(1 - cosine_sim)

        x = predictions.contiguous()
        y = targets.contiguous()
        B, D = x.shape

        # Output per-row loss in fp32 for robust reductions
        out = torch.empty(B, device=x.device, dtype=torch.float32)

        sx0, sx1 = x.stride()
        sy0, sy1 = y.stride()

        grid = (B,)
        _cosine_similarity_rows_kernel[grid](
            x, y, out,
            B, D,
            sx0, sx1,
            sy0, sy1,
            EPS=1e-8,
            BLOCK_SIZE=D,
        )

        return out.mean()


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []