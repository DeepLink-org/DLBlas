import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _smooth_l1_mean_atomic_kernel(
    pred_ptr, tgt_ptr, out_mean_ptr,
    n_elements,
    inv_n,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Process the block in smaller chunks to reduce register pressure and increase occupancy
    CHUNK: tl.constexpr = 1024
    inv_beta = 1.0 / beta
    acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, BLOCK_SIZE, CHUNK):
        offsets = block_start + off + tl.arange(0, CHUNK)
        mask = offsets < n_elements

        tl.multiple_of(offsets, CHUNK)
        tl.max_contiguous(offsets, CHUNK)

        # Load and compute in fp32 for stability
        p = tl.load(pred_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(tgt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        d = p - t
        ad = tl.abs(d)

        # Huber / Smooth L1 with beta
        small = 0.5 * (d * d) * inv_beta
        large = ad - 0.5 * beta
        loss = tl.where(ad < beta, small, large)
        loss = tl.where(mask, loss, 0.0)

        acc += tl.sum(loss, axis=0)

    # Accumulate mean contribution atomically (only once per program)
    tl.atomic_add(out_mean_ptr, acc * inv_n)


def smooth_l1_loss_triton(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    # Fallback if tensors are not CUDA tensors or have zero elements
    if (not predictions.is_cuda) or (not targets.is_cuda) or predictions.numel() == 0:
        return torch.nn.functional.smooth_l1_loss(predictions, targets, beta=beta)

    assert predictions.shape == targets.shape, "predictions and targets must have the same shape"
    preds = predictions.contiguous()
    tgts = targets.contiguous()
    device = preds.device
    n_elements = preds.numel()

    # Output accumulator (fp32 for numerical stability)
    out_mean = torch.zeros(1, device=device, dtype=torch.float32)
    inv_n = 1.0 / n_elements

    # Use a moderate tile and chunked compute for high occupancy on Hopper-class GPUs
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _smooth_l1_mean_atomic_kernel[grid](
        preds, tgts, out_mean,
        n_elements,
        inv_n,
        beta,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    # Match PyTorch dtype
    return out_mean[0].to(predictions.dtype)


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton implementation when possible for speed, fallback otherwise
        return smooth_l1_loss_triton(predictions, targets, beta=1.0)


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    # Generate CUDA tensors to exercise the Triton kernel on GPU
    return [torch.randn(batch_size, *input_shape, device='cuda'),
            torch.randn(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    return []
