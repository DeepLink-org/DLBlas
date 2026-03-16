import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=8),
    ],
    key=["HW"],
)
@triton.jit
def _instance_norm2d_kernel(
    x_ptr,
    y_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # Base pointers for this (n, c) plane
    x_base = x_ptr + n * stride_n + c * stride_c
    y_base = y_ptr + n * stride_n + c * stride_c

    idx_vec = tl.arange(0, BLOCK_SIZE)

    # ------------------ Pass 1: compute mean and variance (pipelined loads) ------------------
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Preload first tile
    off0 = 0
    idx0 = off0 + idx_vec
    mask0 = idx0 < HW
    x0 = tl.load(x_base + idx0, mask=mask0, other=0.0)
    xf0 = x0.to(tl.float32)

    # Main loop with software prefetch
    for off in tl.static_range(BLOCK_SIZE, HW, BLOCK_SIZE):
        idx1 = off + idx_vec
        mask1 = idx1 < HW
        x1 = tl.load(x_base + idx1, mask=mask1, other=0.0)

        # Reduce current buffered tile
        sum_val += tl.sum(xf0, axis=0)
        sumsq_val += tl.sum(xf0 * xf0, axis=0)

        # Advance pipeline
        idx0, mask0, x0 = idx1, mask1, x1
        xf0 = x0.to(tl.float32)

    # Epilogue for the last buffered tile
    sum_val += tl.sum(xf0, axis=0)
    sumsq_val += tl.sum(xf0 * xf0, axis=0)

    inv_hw = 1.0 / tl.full((), HW, dtype=tl.float32)
    mean = sum_val * inv_hw
    var = sumsq_val * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    # ------------------ Pass 2: normalize and store (pipelined loads/stores) ------------------
    off0 = 0
    idx0 = off0 + idx_vec
    mask0 = idx0 < HW
    x0 = tl.load(x_base + idx0, mask=mask0, other=0.0)
    xf0 = x0.to(tl.float32)

    for off in tl.static_range(BLOCK_SIZE, HW, BLOCK_SIZE):
        idx1 = off + idx_vec
        mask1 = idx1 < HW
        x1 = tl.load(x_base + idx1, mask=mask1, other=0.0)

        y0 = (xf0 - mean) * rstd
        tl.store(y_base + idx0, y0.to(x0.dtype), mask=mask0)

        # Advance pipeline
        idx0, mask0, x0 = idx1, mask1, x1
        xf0 = x0.to(tl.float32)

    # Store last buffered tile
    y_last = (xf0 - mean) * rstd
    tl.store(y_base + idx0, y_last.to(x0.dtype), mask=mask0)


class ModelNew(nn.Module):
    """
    Instance Normalization implemented via a Triton kernel (affine=False, track_running_stats=False).
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Fallback for CPU or non-CUDA tensors; matches PyTorch defaults
        self._fallback = nn.InstanceNorm2d(
            num_features=num_features, affine=False, track_running_stats=False, eps=self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch on CPU to preserve semantics
        if not x.is_cuda:
            return self._fallback(x)

        assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
        N, C, H, W = x.shape
        assert C == self.num_features, f"Expected C == num_features ({self.num_features}), got {C}"

        x = x.contiguous()
        y = torch.empty_like(x)

        stride_n, stride_c, stride_h, stride_w = x.stride()
        HW = H * W
        grid = (N * C,)

        _instance_norm2d_kernel[grid](
            x, y,
            N, C, H, W,
            stride_n, stride_c, stride_h, stride_w,
            self.eps,
            HW=HW,
        )
        return y


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]