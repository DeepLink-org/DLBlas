"""
Ascend-optimized fused Triton implementation of 8_torch.Model.

Fuses in one kernel per (n0, n1) token row:
  RMS-normalized projection  ->  split (scale/base/sigmoid)  ->  Sinkhorn  ->  apply_mix

Default: force Triton on NPU. Set force_triton=False or OP8_FORCE_TRITON=0 for PyTorch path.

Reference: 8_torch.py
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.runtime.driver as driver


def _num_vectorcores() -> int:
    device = torch.npu.current_device()
    return int(driver.active.utils.get_device_properties(device)["num_vectorcore"])


def _phase1_mixes_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor:
    """Batched RMS + projection (matches 8_big_fuse._mhc_pre_norm_fn)."""
    n0, n1 = residual.shape[:2]
    x = residual.reshape(n0 * n1, -1).float()
    mixes = x @ fn.T
    sqrsum = (x * x).sum(dim=1, keepdim=True)
    return mixes * torch.rsqrt(sqrsum / x.shape[-1] + rms_eps)


@triton.jit
def _mhc_tail_fuse_kernel(
    residual_ptr,
    base_ptr,
    mixes_scratch_ptr,
    post_ptr,
    comb_ptr,
    layer_input_ptr,
    n0,
    n1,
    mhc_pre_eps,
    sinkhorn_eps,
    post_mult,
    s0,
    s1,
    s2,
    stride_res_n0,
    stride_res_n1,
    stride_res_mhc,
    stride_res_h,
    stride_post_n0,
    stride_post_n1,
    stride_post_mhc,
    stride_comb_n0,
    stride_comb_n1,
    stride_comb_h,
    stride_comb_w,
    stride_out_n0,
    stride_out_n1,
    stride_out_h,
    MHC: tl.constexpr,
    H: tl.constexpr,
    MIX_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    n_bn = n0 * n1
    r = tl.arange(0, MHC)
    c = tl.arange(0, MHC)
    row_i = r[:, None]
    col_j = c[None, :]
    mask_hc_2d = (row_i < MHC) & (col_j < MHC)

    # Map grid to Vector Core count with 1D stride loop over blocks
    # Ensures coreDim overflow avoidance and UB-bound tile reuse
    for bn in range(pid, n_bn, NUM_CORES):
        b_idx = bn // n1
        n_idx = bn - b_idx * n1
        res_bn = residual_ptr + b_idx * stride_res_n0 + n_idx * stride_res_n1
        mix_row = mixes_scratch_ptr + bn * MIX_DIM

        col = tl.arange(0, MHC)
        mask_m = col < MHC

        x_pre = tl.load(mix_row + col, mask=mask_m, other=0.0) * s0
        x_pre = x_pre + tl.load(base_ptr + col, mask=mask_m, other=0.0)
        pre_val = tl.sigmoid(x_pre) + mhc_pre_eps

        x_post = tl.load(mix_row + MHC + col, mask=mask_m, other=0.0) * s1
        x_post = x_post + tl.load(base_ptr + MHC + col, mask=mask_m, other=0.0)
        post_val = tl.sigmoid(x_post) * post_mult

        flat_ij = row_i * MHC + col_j
        raw_off = 2 * MHC + flat_ij
        x_raw = tl.load(mix_row + raw_off, mask=mask_hc_2d, other=0.0)
        b_comb = tl.load(base_ptr + raw_off, mask=mask_hc_2d, other=0.0)
        comb = x_raw * s2 + b_comb
        tl.store(mix_row + col, pre_val, mask=mask_m)

        # --- Phase 3: Sinkhorn (matches 8_big_fuse._sinkhorn_normalize) ---
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        row_sum = tl.sum(comb, axis=1)
        comb = comb / row_sum[:, None] + sinkhorn_eps
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + sinkhorn_eps)

        for _ in tl.static_range(SINKHORN_ITERS - 1):
            row_sum = tl.sum(comb, axis=1)
            comb = comb / (row_sum[:, None] + sinkhorn_eps)
            col_sum = tl.sum(comb, axis=0)
            comb = comb / (col_sum[None, :] + sinkhorn_eps)

        post_bn = post_ptr + b_idx * stride_post_n0 + n_idx * stride_post_n1
        comb_bn = comb_ptr + b_idx * stride_comb_n0 + n_idx * stride_comb_n1
        out_bn = layer_input_ptr + b_idx * stride_out_n0 + n_idx * stride_out_n1

        tl.store(post_bn + col * stride_post_mhc, post_val, mask=mask_m)
        tl.store(
            comb_bn + row_i * stride_comb_h + col_j * stride_comb_w,
            comb,
            mask=mask_hc_2d,
        )

        w0 = tl.load(mix_row + 0, mask=True, other=0.0)
        w1 = tl.load(mix_row + 1, mask=True, other=0.0)
        w2 = tl.load(mix_row + 2, mask=True, other=0.0)
        w3 = tl.load(mix_row + 3, mask=True, other=0.0)

        num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
        for hb in range(num_h_blocks):
            h_start = hb * BLOCK_H
            h_offs = h_start + tl.arange(0, BLOCK_H)
            mask_h = h_offs < H
            x0 = tl.load(
                res_bn + 0 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x1 = tl.load(
                res_bn + 1 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x2 = tl.load(
                res_bn + 2 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x3 = tl.load(
                res_bn + 3 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            acc_h = (
                x0.to(tl.float32) * w0
                + x1.to(tl.float32) * w1
                + x2.to(tl.float32) * w2
                + x3.to(tl.float32) * w3
            )
            tl.store(
                out_bn + h_offs * stride_out_h,
                acc_h.to(tl.bfloat16),
                mask=mask_h,
            )


@triton.jit
def _mhc_big_fuse_kernel(
    residual_ptr,
    fn_ptr,
    base_ptr,
    mixes_scratch_ptr,
    post_ptr,
    comb_ptr,
    layer_input_ptr,
    n0,
    n1,
    rms_eps,
    mhc_pre_eps,
    sinkhorn_eps,
    post_mult,
    s0,
    s1,
    s2,
    stride_res_n0,
    stride_res_n1,
    stride_res_mhc,
    stride_res_h,
    stride_fn_n,
    stride_fn_k,
    stride_post_n0,
    stride_post_n1,
    stride_post_mhc,
    stride_post_one,
    stride_comb_n0,
    stride_comb_n1,
    stride_comb_h,
    stride_comb_w,
    stride_out_n0,
    stride_out_n1,
    stride_out_h,
    MHC: tl.constexpr,
    H: tl.constexpr,
    RGS: tl.constexpr,
    MIX_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    n_bn = n0 * n1
    r = tl.arange(0, MHC)
    c = tl.arange(0, MHC)
    row_i = r[:, None]
    col_j = c[None, :]
    mask_hc_2d = (row_i < MHC) & (col_j < MHC)

    # Map grid to Vector Core count with 1D stride loop over blocks
    # Ensures coreDim overflow avoidance and UB-bound tile reuse
    for bn in range(pid, n_bn, NUM_CORES):
        b_idx = bn // n1
        n_idx = bn - b_idx * n1
        res_bn = residual_ptr + b_idx * stride_res_n0 + n_idx * stride_res_n1
        res_flat = res_bn  # [MHC, H] flattened as RGS contiguous in last two dims

        # --- Phase 1: K-tiled RMS + mixes = (x @ fn.T) * rsqrt(mean(x^2)+eps) ---
        acc_mix = tl.zeros((MIX_DIM,), dtype=tl.float32)
        sqrsum = tl.zeros((1,), dtype=tl.float32)
        n_k_blocks = (RGS + BLOCK_K - 1) // BLOCK_K

        for kb in range(n_k_blocks):
            k_start = kb * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_offs < RGS

            mhc_idx = k_offs // H
            h_idx = k_offs - mhc_idx * H
            x_k = tl.load(
                res_flat + mhc_idx * stride_res_mhc + h_idx * stride_res_h,
                mask=mask_k,
                other=0.0,
            )
            x_k = x_k.to(tl.float32)
            sqrsum += tl.sum(x_k * x_k)

            j_offs = tl.arange(0, MIX_DIM)
            w = tl.load(
                fn_ptr + j_offs[:, None] * stride_fn_n + k_offs[None, :] * stride_fn_k,
                mask=mask_k[None, :],
                other=0.0,
            )
            acc_mix += tl.sum(w.to(tl.float32) * x_k[None, :], axis=1)

        rms_inv = tl.rsqrt(sqrsum / RGS + rms_eps)
        acc_mix = acc_mix * rms_inv

        mix_row = mixes_scratch_ptr + bn * MIX_DIM
        j_all = tl.arange(0, MIX_DIM)
        tl.store(mix_row + j_all, acc_mix, mask=True)

        # --- Phase 2: split (scale + base + sigmoid) ---
        col = tl.arange(0, MHC)
        mask_m = col < MHC

        x_pre = tl.load(mix_row + col, mask=mask_m, other=0.0) * s0
        x_pre = x_pre + tl.load(base_ptr + col, mask=mask_m, other=0.0)
        pre_val = tl.sigmoid(x_pre) + mhc_pre_eps

        x_post = tl.load(mix_row + MHC + col, mask=mask_m, other=0.0) * s1
        x_post = x_post + tl.load(base_ptr + MHC + col, mask=mask_m, other=0.0)
        post_val = tl.sigmoid(x_post) * post_mult

        flat_ij = row_i * MHC + col_j
        raw_off = 2 * MHC + flat_ij
        x_raw = tl.load(mix_row + raw_off, mask=mask_hc_2d, other=0.0)
        b_comb = tl.load(base_ptr + raw_off, mask=mask_hc_2d, other=0.0)
        comb = x_raw * s2 + b_comb
        tl.store(mix_row + col, pre_val, mask=mask_m)

        # --- Phase 3: Sinkhorn (matches 8_big_fuse._sinkhorn_normalize) ---
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        row_sum = tl.sum(comb, axis=1)
        comb = comb / row_sum[:, None] + sinkhorn_eps
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + sinkhorn_eps)

        for _ in tl.static_range(SINKHORN_ITERS - 1):
            row_sum = tl.sum(comb, axis=1)
            comb = comb / (row_sum[:, None] + sinkhorn_eps)
            col_sum = tl.sum(comb, axis=0)
            comb = comb / (col_sum[None, :] + sinkhorn_eps)

        # --- Phase 4: layer_input = (residual * pre_mix).sum(-2) -> bf16 ---
        post_bn = post_ptr + b_idx * stride_post_n0 + n_idx * stride_post_n1
        comb_bn = comb_ptr + b_idx * stride_comb_n0 + n_idx * stride_comb_n1
        out_bn = layer_input_ptr + b_idx * stride_out_n0 + n_idx * stride_out_n1

        tl.store(
            post_bn + col * stride_post_mhc,
            post_val,
            mask=mask_m,
        )

        tl.store(
            comb_bn + row_i * stride_comb_h + col_j * stride_comb_w,
            comb,
            mask=mask_hc_2d,
        )

        w0 = tl.load(mix_row + 0, mask=True, other=0.0)
        w1 = tl.load(mix_row + 1, mask=True, other=0.0)
        w2 = tl.load(mix_row + 2, mask=True, other=0.0)
        w3 = tl.load(mix_row + 3, mask=True, other=0.0)

        num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
        for hb in range(num_h_blocks):
            h_start = hb * BLOCK_H
            h_offs = h_start + tl.arange(0, BLOCK_H)
            mask_h = h_offs < H
            x0 = tl.load(
                res_bn + 0 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x1 = tl.load(
                res_bn + 1 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x2 = tl.load(
                res_bn + 2 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            x3 = tl.load(
                res_bn + 3 * stride_res_mhc + h_offs * stride_res_h,
                mask=mask_h,
                other=0.0,
            )
            acc_h = (
                x0.to(tl.float32) * w0
                + x1.to(tl.float32) * w1
                + x2.to(tl.float32) * w2
                + x3.to(tl.float32) * w3
            )
            tl.store(
                out_bn + h_offs * stride_out_h,
                acc_h.to(tl.bfloat16),
                mask=mask_h,
            )


def _mhc_big_fuse_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference path (8_torch.Model semantics)."""
    n0, n1 = residual.shape[:2]
    x = residual.flatten(2, 3).float().reshape(n0 * n1, -1)
    mixes = x @ fn.T
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * (sqrsum / x.shape[-1] + rms_eps).rsqrt()
    mixes = mixes.view(n0, n1, -1)

    a, b = mixes.shape[:2]
    scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ]
    )
    input_mixes = mixes * scale + mhc_base
    pre_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_mix = (
        input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value
    ).unsqueeze(-1)
    comb_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    comb_mix = comb_mix.softmax(-1) + sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + sinkhorn_eps)

    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, comb_mix, layer_input


def mhc_mhc_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int = 4,
    hidden_size: int = 1280,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    block_k: int = 128,
    block_h: int = 128,
    torch_phase1: bool = True,
    *,
    force_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused MHC pre pipeline; outputs match 8_torch.Model.forward.

    Default (force_triton=True): launch Triton on NPU. Set force_triton=False or
    OP8_FORCE_TRITON=0 for PyTorch reference path.

    When torch_phase1=True (default), Phase-1 batched GEMM runs on NPU via Torch;
    Triton fuses Phase 2–4 only.
    """
    env_force = os.environ.get("OP8_FORCE_TRITON", "1").lower()
    use_triton = force_triton and env_force not in ("0", "false", "no")
    if not use_triton:
        return _mhc_big_fuse_torch(
            residual,
            fn,
            mhc_scale,
            mhc_base,
            mhc_mult,
            hidden_size,
            rms_eps,
            mhc_pre_eps,
            sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
        )

    if residual.device.type != "npu":
        raise RuntimeError(
            f"mhc_mhc_triton requires NPU when force_triton=True, got {residual.device}"
        )

    if residual.dim() != 4:
        raise ValueError(f"residual must be 4D, got {residual.dim()}D")
    n0, n1, mhc, h = residual.shape
    if mhc != mhc_mult or h != hidden_size:
        raise ValueError(f"shape mismatch: got ({mhc}, {h}), expected ({mhc_mult}, {hidden_size})")

    mix_dim = mhc_mult * 2 + mhc_mult * mhc_mult
    rgs = mhc_mult * hidden_size
    if fn.shape != (mix_dim, rgs):
        raise ValueError(f"fn shape {tuple(fn.shape)} != ({mix_dim}, {rgs})")
    if mhc_base.numel() != mix_dim:
        raise ValueError(f"mhc_base len {mhc_base.numel()} != {mix_dim}")

    res_c = residual.contiguous()
    fn_c = fn.contiguous()
    base_c = mhc_base.reshape(-1).contiguous().to(torch.float32)
    if fn_c.dtype != torch.float32:
        fn_c = fn_c.to(torch.float32)

    post = torch.empty((n0, n1, mhc_mult, 1), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n0, n1, mhc_mult, mhc_mult), device=residual.device, dtype=torch.float32)
    layer_input = torch.empty((n0, n1, hidden_size), device=residual.device, dtype=torch.bfloat16)

    s0 = float(mhc_scale[0].item())
    s1 = float(mhc_scale[1].item())
    s2 = float(mhc_scale[2].item())
    num_cores = _num_vectorcores()
    if torch_phase1:
        mixes_scratch = _phase1_mixes_torch(res_c, fn_c, rms_eps)
    else:
        mixes_scratch = torch.empty((n0 * n1, mix_dim), device=residual.device, dtype=torch.float32)

    if torch_phase1:
        # Launch with grid mapped to Vector Core count using 1D stride loop
        _mhc_tail_fuse_kernel[(num_cores,)](
            res_c,
            base_c,
            mixes_scratch,
            post,
            comb,
            layer_input,
            n0,
            n1,
            mhc_pre_eps,
            sinkhorn_eps,
            mhc_post_mult_value,
            s0,
            s1,
            s2,
            res_c.stride(0),
            res_c.stride(1),
            res_c.stride(2),
            res_c.stride(3),
            post.stride(0),
            post.stride(1),
            post.stride(2),
            comb.stride(0),
            comb.stride(1),
            comb.stride(2),
            comb.stride(3),
            layer_input.stride(0),
            layer_input.stride(1),
            layer_input.stride(2),
            MHC=mhc_mult,
            H=hidden_size,
            MIX_DIM=mix_dim,
            BLOCK_H=block_h,
            SINKHORN_ITERS=sinkhorn_repeat,
            NUM_CORES=num_cores,
        )
        return post, comb, layer_input

    # Launch with grid mapped to Vector Core count using 1D stride loop
    _mhc_big_fuse_kernel[(num_cores,)](
        res_c,
        fn_c,
        base_c,
        mixes_scratch,
        post,
        comb,
        layer_input,
        n0,
        n1,
        rms_eps,
        mhc_pre_eps,
        sinkhorn_eps,
        mhc_post_mult_value,
        s0,
        s1,
        s2,
        res_c.stride(0),
        res_c.stride(1),
        res_c.stride(2),
        res_c.stride(3),
        fn_c.stride(0),
        fn_c.stride(1),
        post.stride(0),
        post.stride(1),
        post.stride(2),
        post.stride(3),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        comb.stride(3),
        layer_input.stride(0),
        layer_input.stride(1),
        layer_input.stride(2),
        MHC=mhc_mult,
        H=hidden_size,
        RGS=rgs,
        MIX_DIM=mix_dim,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        SINKHORN_ITERS=sinkhorn_repeat,
        NUM_CORES=num_cores,
    )
    return post, comb, layer_input


class ModelTriton(nn.Module):
    """Drop-in module using fused Triton forward on NPU (force Triton by default)."""

    def __init__(
        self,
        mhc_mult: int,
        hidden_size: int,
        rms_eps: float = 1e-6,
        mhc_pre_eps: float = 1e-6,
        mhc_sinkhorn_eps: float = 1e-6,
        mhc_post_mult_value: float = 1.0,
        sinkhorn_repeat: int = 10,
        *,
        force_triton: bool = True,
    ):
        super().__init__()
        self.mhc_mult = mhc_mult
        self.hidden_size = hidden_size
        self.rms_eps = rms_eps
        self.mhc_pre_eps = mhc_pre_eps
        self.mhc_sinkhorn_eps = mhc_sinkhorn_eps
        self.mhc_post_mult_value = mhc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat
        self.force_triton = force_triton

        mix_dim = mhc_mult * 2 + mhc_mult * mhc_mult
        self.fn = nn.Parameter(torch.randn(mix_dim, mhc_mult * hidden_size) * 1e-4)
        self.mhc_scale = nn.Parameter(torch.randn(3) * 0.1)
        self.mhc_base = nn.Parameter(torch.randn(mix_dim) * 0.1)

    def forward(
        self,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_mhc_triton(
            residual,
            self.fn,
            self.mhc_scale,
            self.mhc_base,
            mhc_mult=self.mhc_mult,
            hidden_size=self.hidden_size,
            rms_eps=self.rms_eps,
            mhc_pre_eps=self.mhc_pre_eps,
            sinkhorn_eps=self.mhc_sinkhorn_eps,
            mhc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.sinkhorn_repeat,
            force_triton=self.force_triton,
        )


ModelNew = ModelTriton

n1 = 512
mhc_mult = 4
hidden_size = 1280


def get_inputs():
    residual = torch.randn(1, n1, mhc_mult, hidden_size).bfloat16()
    return [residual]


def get_init_inputs():
    return [mhc_mult, hidden_size]


def _accuracy_smoke() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "mhc_torch_ori", Path(__file__).parent / "8_torch.py"
    )
    assert spec and spec.loader
    ori = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ori)

    device = "npu"
    torch.manual_seed(0)
    residual = ori.get_inputs()[0].to(device)
    init = ori.get_init_inputs()

    ref_m = ori.Model(*init).to(device)
    tri_m = ModelTriton(*init).to(device)
    tri_m.fn.data.copy_(ref_m.fn.data)
    tri_m.mhc_scale.data.copy_(ref_m.mhc_scale.data)
    tri_m.mhc_base.data.copy_(ref_m.mhc_base.data)

    with torch.no_grad():
        r_post, r_comb, r_layer = ref_m(residual)
        t_post, t_comb, t_layer = tri_m(residual)

    for name, ref, got in [
        ("post_mix", r_post, t_post),
        ("comb_mix", r_comb, t_comb),
        ("layer_input", r_layer, t_layer),
    ]:
        err = (got.float() - ref.float()).abs().max().item()
        rtol, atol = (2e-2, 2e-2) if name == "layer_input" else (1e-5, 1e-5)
        torch.testing.assert_close(got.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        print(f"{name}: ok max_err={err:.3e}")


if __name__ == "__main__":
    _accuracy_smoke()
