import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bn_row_reduce_nhw_store(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    N, H, W,
    stride_n, stride_c, stride_h, stride_w,
    NH,  # number of (n,h) rows per channel
    BLOCK_W: tl.constexpr,
    NUM_W_CHUNKS: tl.constexpr,
):
    # program ids
    c = tl.program_id(0)
    nh = tl.program_id(1)
    # derive n, h from nh
    n = nh // H
    h = nh - n * H

    # base pointer for this row
    row_base = x_ptr + n * stride_n + c * stride_c + h * stride_h

    # accumulate along W with vector accumulation to reduce number of tl.sum ops
    offs_w = tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, BLOCK_W)
    acc_vec_sum = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc_vec_sumsq = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for cw in tl.static_range(0, NUM_W_CHUNKS):
        w_idx = cw * BLOCK_W + offs_w
        mask = w_idx < W
        vals = tl.load(row_base + w_idx * stride_w, mask=mask, other=0.0)
        acc_vec_sum += vals
        acc_vec_sumsq += vals * vals

    # reduce vectors to scalars once
    acc_sum = tl.sum(acc_vec_sum, axis=0)
    acc_sumsq = tl.sum(acc_vec_sumsq, axis=0)

    # store partial reductions per (c, nh), no atomics
    base_idx = c * NH + nh
    tl.store(partial_sum_ptr + base_idx, acc_sum)
    tl.store(partial_sumsq_ptr + base_idx, acc_sumsq)


@triton.jit
def _bn_finalize_params(
    partial_sum_ptr,
    partial_sumsq_ptr,
    scale_ptr,
    shift_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    NH,      # number of (n,h) rows per channel
    M,       # total elements per-channel = N*H*W
    eps,     # epsilon
    exp_avg_factor,  # exponential_average_factor
    use_batch_stats,  # 1 if using batch stats, 0 if using running stats
    do_update,        # 1 to update running stats (training & tracking), else 0
    affine_flag,      # 1 if affine, else 0
    BLOCK_NH: tl.constexpr,
    NUM_NH_CHUNKS: tl.constexpr,
):
    c = tl.program_id(0)

    # compute mean/var either from batch partial sums or from running stats
    if use_batch_stats:
        offs = tl.arange(0, BLOCK_NH)
        acc_sum = 0.0
        acc_sumsq = 0.0
        for chunk in tl.static_range(0, NUM_NH_CHUNKS):
            idx = chunk * BLOCK_NH + offs
            mask = idx < NH
            base = c * NH + idx
            psum = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
            psumsq = tl.load(partial_sumsq_ptr + base, mask=mask, other=0.0)
            acc_sum += tl.sum(psum, axis=0)
            acc_sumsq += tl.sum(psumsq, axis=0)
        mean = acc_sum / M
        var = acc_sumsq / M - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = 1.0 / tl.sqrt(var + eps)

        # optionally update running stats
        if do_update:
            rm = tl.load(running_mean_ptr + c)
            rv = tl.load(running_var_ptr + c)
            one_minus = 1.0 - exp_avg_factor
            rm = rm * one_minus + mean * exp_avg_factor
            rv = rv * one_minus + var * exp_avg_factor
            tl.store(running_mean_ptr + c, rm)
            tl.store(running_var_ptr + c, rv)

        mean_use = mean
        invstd_use = invstd
    else:
        # evaluation using running stats
        rm = tl.load(running_mean_ptr + c)
        rv = tl.load(running_var_ptr + c)
        mean_use = rm
        invstd_use = 1.0 / tl.sqrt(rv + eps)

    if affine_flag:
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        scale = invstd_use * w
        shift = b - mean_use * scale
    else:
        scale = invstd_use
        shift = -mean_use * invstd_use

    tl.store(scale_ptr + c, scale)
    tl.store(shift_ptr + c, shift)


@triton.jit
def _bn_apply_nhw(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    N, H, W,
    stride_nx, stride_cx, stride_hx, stride_wx,
    stride_ny, stride_cy, stride_hy, stride_wy,
    BLOCK_W: tl.constexpr,
    NUM_W_CHUNKS: tl.constexpr,
):
    c = tl.program_id(0)
    nh = tl.program_id(1)
    n = nh // H
    h = nh - n * H

    scale_c = tl.load(scale_ptr + c)
    shift_c = tl.load(shift_ptr + c)

    x_row = x_ptr + n * stride_nx + c * stride_cx + h * stride_hx
    y_row = y_ptr + n * stride_ny + c * stride_cy + h * stride_hy

    offs_w = tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, BLOCK_W)
    for cw in tl.static_range(0, NUM_W_CHUNKS):
        w_idx = cw * BLOCK_W + offs_w
        mask = w_idx < W
        x = tl.load(x_row + w_idx * stride_wx, mask=mask, other=0.0)
        y = x * scale_c + shift_c
        tl.store(y_row + w_idx * stride_wy, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs Batch Normalization using Triton-optimized kernels.
    Falls back to PyTorch for non-CUDA / unsupported dtypes / non-contiguous inputs.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback for unsupported cases
        if (not x.is_cuda) or (x.dtype != torch.float32) or (not x.is_contiguous()):
            return self.bn(x)

        bn = self.bn
        N, C, H, W = x.shape
        eps = bn.eps

        # Compute exponential_average_factor following PyTorch semantics
        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum

        if bn.training and bn.track_running_stats:
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)
                if bn.momentum is None:
                    exponential_average_factor = 1.0 / float(bn.num_batches_tracked.item())
            else:
                exponential_average_factor = 1.0

        # use_batch_stats matches PyTorch: training or not tracking
        use_batch_stats = bn.training or (not bn.track_running_stats)
        update_stats = bn.training and bn.track_running_stats

        x_fp32 = x  # already float32 & contiguous per guard
        stride_n, stride_c, stride_h, stride_w = x_fp32.stride()
        # Triton uses element-wise strides
        NH = N * H
        M = N * H * W

        BLOCK_W = 256
        NUM_W_CHUNKS = (W + BLOCK_W - 1) // BLOCK_W
        # Larger block for NH reduction to cut down loop trips in finalize
        BLOCK_NH = 2048
        NUM_NH_CHUNKS = (NH + BLOCK_NH - 1) // BLOCK_NH

        # Prepare scale/shift buffers
        scale = torch.empty(C, device=x.device, dtype=torch.float32)
        shift = torch.empty(C, device=x.device, dtype=torch.float32)

        # Running stats pointers (may be dummy if not tracked)
        if bn.track_running_stats and (bn.running_mean is not None) and (bn.running_var is not None):
            running_mean = bn.running_mean
            running_var = bn.running_var
        else:
            # dummy tensors to satisfy kernel signature; not used unless needed
            running_mean = torch.zeros(C, device=x.device, dtype=torch.float32)
            running_var = torch.ones(C, device=x.device, dtype=torch.float32)

        affine_flag = int(bn.affine and (bn.weight is not None) and (bn.bias is not None))
        if affine_flag:
            weight = bn.weight.to(dtype=torch.float32)
            bias = bn.bias.to(dtype=torch.float32)
        else:
            # dummy
            weight = torch.empty(1, device=x.device, dtype=torch.float32)
            bias = torch.empty(1, device=x.device, dtype=torch.float32)

        # If we use batch stats, first compute per-(c, nh) partial reductions without atomics
        if use_batch_stats:
            partial_sum = torch.empty((C, NH), device=x.device, dtype=torch.float32)
            partial_sumsq = torch.empty((C, NH), device=x.device, dtype=torch.float32)
            grid_rows = (C, NH)
            _bn_row_reduce_nhw_store[grid_rows](
                x_fp32,
                partial_sum, partial_sumsq,
                N, H, W,
                stride_n, stride_c, stride_h, stride_w,
                NH,
                BLOCK_W=BLOCK_W,
                NUM_W_CHUNKS=NUM_W_CHUNKS,
                num_warps=2,
                num_stages=2,
            )
        else:
            # allocate dummy to satisfy kernel signature
            partial_sum = torch.empty(1, device=x.device, dtype=torch.float32)
            partial_sumsq = torch.empty(1, device=x.device, dtype=torch.float32)

        # Finalize params (mean/var/scale/shift), and optionally update running stats
        _bn_finalize_params[(C,)](
            partial_sum, partial_sumsq,
            scale, shift,
            running_mean, running_var,
            weight, bias,
            NH, M, eps, exponential_average_factor,
            int(use_batch_stats), int(update_stats), affine_flag,
            BLOCK_NH=BLOCK_NH,
            NUM_NH_CHUNKS=NUM_NH_CHUNKS,
            num_warps=4,
            num_stages=2,
        )

        # Apply normalization + affine with Triton
        y = torch.empty_like(x_fp32)
        grid_apply = (C, NH)
        _bn_apply_nhw[grid_apply](
            x_fp32, y,
            scale, shift,
            N, H, W,
            stride_n, stride_c, stride_h, stride_w,
            stride_n, stride_c, stride_h, stride_w,
            BLOCK_W=BLOCK_W,
            NUM_W_CHUNKS=NUM_W_CHUNKS,
            num_warps=4,
            num_stages=2,
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