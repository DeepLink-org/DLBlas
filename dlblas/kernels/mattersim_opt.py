# Copyright (c) 2025, DeepLink.
"""MatterSim optimized Triton kernels extracted into dlBLAS.

The kernels in this file come from the MatterSim optimized LAMMPS path.  They are
kept independent from MatterSim model classes so they can be unit-tested and
benchmarked as dlBLAS primitives.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ACT_NONE = 0
ACT_SIGMOID = 1
ACT_SWISH = 2


def _act_code(activation: Optional[str]) -> int:
    if activation is None or activation == "none":
        return ACT_NONE
    if activation == "sigmoid":
        return ACT_SIGMOID
    if activation == "swish" or activation == "silu":
        return ACT_SWISH
    raise ValueError(f"unsupported activation: {activation}")


def _apply_activation(x: torch.Tensor, activation: Optional[str]) -> torch.Tensor:
    code = _act_code(activation)
    if code == ACT_NONE:
        return x
    if code == ACT_SIGMOID:
        return torch.sigmoid(x)
    return F.silu(x)


@triton.jit
def _linear_bias_act_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    ACT: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    if ACT == 1:
        acc = tl.sigmoid(acc)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _addmm_sigmoid_mul_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    preact_ptr,
    sigmoid_ptr,
    swish_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]
    sig = tl.sigmoid(acc)
    swish = acc * sig

    offsets = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(preact_ptr + offsets, acc, mask=out_mask)
    tl.store(sigmoid_ptr + offsets, sig, mask=out_mask)
    tl.store(swish_ptr + offsets, swish, mask=out_mask)


@triton.jit
def _addmm_sigmoid_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    sigmoid_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sm,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    sig = tl.sigmoid(acc + bias[None, :])
    out_ptrs = sigmoid_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    tl.store(out_ptrs, sig, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _two_path_two_layer_kernel(
    x_ptr,
    w1g_ptr,
    b1g_ptr,
    w2g_ptr,
    b2g_ptr,
    w1s_ptr,
    b1s_ptr,
    w2s_ptr,
    b2s_ptr,
    y_ptr,
    M,
    K0,
    N1,
    N2,
    stride_xm,
    stride_xk,
    stride_w1g_n,
    stride_w1g_k,
    stride_w2g_n,
    stride_w2g_k,
    stride_w1s_n,
    stride_w1s_k,
    stride_w2s_n,
    stride_w2s_k,
    stride_ym,
    stride_yn,
    ACT1_G: tl.constexpr,
    ACT2_G: tl.constexpr,
    ACT1_S: tl.constexpr,
    ACT2_S: tl.constexpr,
    USE_B1G: tl.constexpr,
    USE_B2G: tl.constexpr,
    USE_B1S: tl.constexpr,
    USE_B2S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_K0: tl.constexpr,
    BLOCK_N1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n2 = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    mask_m = offs_m < M
    mask_n2 = offs_n2 < N2
    acc_g = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

    for n1 in range(0, N1, BLOCK_N1):
        offs_n1 = n1 + tl.arange(0, BLOCK_N1)
        mask_n1 = offs_n1 < N1
        h_g = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)
        h_s = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)

        for k0 in range(0, K0, BLOCK_K0):
            offs_k0 = k0 + tl.arange(0, BLOCK_K0)
            mask_k = offs_k0 < K0
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k0[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w1g_ptrs = (
                w1g_ptr
                + offs_n1[:, None] * stride_w1g_n
                + offs_k0[None, :] * stride_w1g_k
            )
            w1s_ptrs = (
                w1s_ptr
                + offs_n1[:, None] * stride_w1s_n
                + offs_k0[None, :] * stride_w1s_k
            )
            w1g = tl.load(w1g_ptrs, mask=mask_n1[:, None] & mask_k[None, :], other=0.0)
            w1s = tl.load(w1s_ptrs, mask=mask_n1[:, None] & mask_k[None, :], other=0.0)
            h_g += tl.dot(x, tl.trans(w1g))
            h_s += tl.dot(x, tl.trans(w1s))

        if USE_B1G:
            b1g = tl.load(b1g_ptr + offs_n1, mask=mask_n1, other=0.0)
            h_g += b1g[None, :]
        if USE_B1S:
            b1s = tl.load(b1s_ptr + offs_n1, mask=mask_n1, other=0.0)
            h_s += b1s[None, :]

        if ACT1_G == 1:
            h_g = tl.sigmoid(h_g)
        elif ACT1_G == 2:
            h_g = h_g * tl.sigmoid(h_g)
        if ACT1_S == 1:
            h_s = tl.sigmoid(h_s)
        elif ACT1_S == 2:
            h_s = h_s * tl.sigmoid(h_s)

        w2g_ptrs = (
            w2g_ptr + offs_n2[:, None] * stride_w2g_n + offs_n1[None, :] * stride_w2g_k
        )
        w2s_ptrs = (
            w2s_ptr + offs_n2[:, None] * stride_w2s_n + offs_n1[None, :] * stride_w2s_k
        )
        w2g = tl.load(w2g_ptrs, mask=mask_n2[:, None] & mask_n1[None, :], other=0.0)
        w2s = tl.load(w2s_ptrs, mask=mask_n2[:, None] & mask_n1[None, :], other=0.0)
        acc_g += tl.dot(h_g.to(w2g.dtype), tl.trans(w2g))
        acc_s += tl.dot(h_s.to(w2s.dtype), tl.trans(w2s))

    if USE_B2G:
        b2g = tl.load(b2g_ptr + offs_n2, mask=mask_n2, other=0.0)
        acc_g += b2g[None, :]
    if USE_B2S:
        b2s = tl.load(b2s_ptr + offs_n2, mask=mask_n2, other=0.0)
        acc_s += b2s[None, :]

    if ACT2_G == 1:
        acc_g = tl.sigmoid(acc_g)
    elif ACT2_G == 2:
        acc_g = acc_g * tl.sigmoid(acc_g)
    if ACT2_S == 1:
        acc_s = tl.sigmoid(acc_s)
    elif ACT2_S == 2:
        acc_s = acc_s * tl.sigmoid(acc_s)

    y = acc_g * acc_s
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n2[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n2[None, :])


@triton.jit
def _fused_mul5_sigmoid_bwd_kernel(
    out_mul5_ptr,
    out_s_bwd_ptr,
    t1_ptr,
    a1_ptr,
    s1_ptr,
    s3_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    t1 = tl.load(t1_ptr + offs, mask=mask, other=0.0)
    a1 = tl.load(a1_ptr + offs, mask=mask, other=0.0)
    s1 = tl.load(s1_ptr + offs, mask=mask, other=0.0)
    s3 = tl.load(s3_ptr + offs, mask=mask, other=0.0)
    mul5 = t1 * s3
    sbwd = t1 * (a1 * s1) * (s3 - s3 * s3)
    tl.store(out_mul5_ptr + offs, mul5, mask=mask)
    tl.store(out_s_bwd_ptr + offs, sbwd, mask=mask)


@triton.jit
def _fused_add_swish_bwd_kernel(
    out_ptr,
    x_ptr,
    a_ptr,
    s_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0)
    out = x * (s + a * (s - s * s))
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _scatter_add_2d_dim0_tile_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    T,
    D,
    stride_src0,
    stride_src1,
    stride_out0,
    stride_out1,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < T
    col_mask = cols < D
    dst = tl.load(index_ptr + rows, mask=row_mask, other=0)
    vals = tl.load(
        src_ptr + rows[:, None] * stride_src0 + cols[None, :] * stride_src1,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.atomic_add(
        out_ptr + dst[:, None] * stride_out0 + cols[None, :] * stride_out1,
        vals,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _offset_three_body_kernel(
    triple_offsets,
    bond_offsets,
    num_three_body,
    three_body_indices,
    block_size: tl.constexpr,
):
    graph_id = tl.program_id(0)
    block_id = tl.program_id(1)
    local = block_id * block_size + tl.arange(0, block_size)
    ntrip = tl.load(num_three_body + graph_id)
    mask = local < ntrip
    triple_start = tl.load(triple_offsets + graph_id)
    bond_bias = tl.load(bond_offsets + graph_id)
    row = triple_start + local
    base = row * 2
    col0 = tl.load(three_body_indices + base, mask=mask, other=0)
    col1 = tl.load(three_body_indices + base + 1, mask=mask, other=0)
    tl.store(three_body_indices + base, col0 + bond_bias, mask=mask)
    tl.store(three_body_indices + base + 1, col1 + bond_bias, mask=mask)


@triton.jit
def _fill_threebody_kernel(
    orig_idx,
    k_per_atom,
    bond_offsets,
    triple_offsets,
    out,
    block_size: tl.constexpr,
):
    atom_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * block_size + tl.arange(0, block_size)
    k = tl.load(k_per_atom + atom_id)
    total = k * (k - 1)
    active = offsets < total
    denom = tl.maximum(k - 1, 1)
    j_local = offsets // denom
    l_raw = offsets - j_local * denom
    l_local = tl.where(l_raw < j_local, l_raw, l_raw + 1)
    bond_start = tl.load(bond_offsets + atom_id)
    triple_start = tl.load(triple_offsets + atom_id)
    j_orig = tl.load(orig_idx + bond_start + j_local, mask=active, other=0)
    l_orig = tl.load(orig_idx + bond_start + l_local, mask=active, other=0)
    out_base = (triple_start + offsets) * 2
    tl.store(out + out_base, j_orig, mask=active)
    tl.store(out + out_base + 1, l_orig, mask=active)


def _matmul_config(M: int, N: int, K: int):
    block_m = 32 if M <= 32 else 64
    block_n = 32 if N <= 32 else (64 if N <= 64 else 128)
    block_k = 16 if K <= 16 else (32 if K <= 32 else 64)
    tile_elems = block_m * block_n
    if tile_elems <= 1024:
        num_warps = 1
        num_stages = 1 if K <= block_k else 2
    elif tile_elems <= 4096:
        num_warps = 4
        num_stages = 2
    else:
        num_warps = 8
        num_stages = 3
    return block_m, block_n, block_k, num_warps, num_stages


def linear_bias_activation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "none",
) -> torch.Tensor:
    """Compute ``activation(x @ weight.T + bias)``.

    ``weight`` is expected in PyTorch linear layout ``[N, K]``.
    """
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must both be rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1]")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias must have shape [weight.shape[0]]")
    if not x.is_cuda:
        return _apply_activation(F.linear(x, weight, bias), activation)

    x = x.contiguous()
    weight = weight.contiguous()
    bias_arg = bias.contiguous() if bias is not None else weight
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    block_m, block_n, block_k, num_warps, num_stages = _matmul_config(M, N, K)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _linear_bias_act_kernel[grid](
        x,
        weight,
        bias_arg,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        ACT=_act_code(activation),
        USE_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def addmm_sigmoid_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute linear preactivation, sigmoid, and swish outputs."""
    if not x.is_cuda:
        preact = F.linear(x, weight, bias)
        sig = torch.sigmoid(preact)
        return preact, sig, preact * sig

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    preact = torch.empty((M, N), device=x.device, dtype=x.dtype)
    sigmoid = torch.empty_like(preact)
    swish = torch.empty_like(preact)
    block_m, block_n, block_k, num_warps, num_stages = _matmul_config(M, N, K)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _addmm_sigmoid_mul_kernel[grid](
        x,
        weight,
        bias,
        preact,
        sigmoid,
        swish,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        preact.stride(0),
        preact.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return preact, sigmoid, swish


def addmm_sigmoid(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Compute ``sigmoid(x @ weight.T + bias)``."""
    if not x.is_cuda:
        return torch.sigmoid(F.linear(x, weight, bias))

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    block_m, block_n, block_k, num_warps, num_stages = _matmul_config(M, N, K)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _addmm_sigmoid_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def two_path_two_layer_mlp(
    x: torch.Tensor,
    w1g: torch.Tensor,
    b1g: Optional[torch.Tensor],
    w2g: torch.Tensor,
    b2g: Optional[torch.Tensor],
    w1s: torch.Tensor,
    b1s: Optional[torch.Tensor],
    w2s: torch.Tensor,
    b2s: Optional[torch.Tensor],
    act1_g: Optional[str] = "swish",
    act2_g: Optional[str] = "swish",
    act1_s: Optional[str] = "swish",
    act2_s: Optional[str] = "sigmoid",
) -> torch.Tensor:
    """Compute two two-layer MLP branches and multiply their outputs."""

    def branch(inp, w1, b1, w2, b2, act1, act2):
        hidden = _apply_activation(F.linear(inp, w1, b1), act1)
        return _apply_activation(F.linear(hidden, w2, b2), act2)

    if not x.is_cuda:
        return branch(x, w1g, b1g, w2g, b2g, act1_g, act2_g) * branch(
            x, w1s, b1s, w2s, b2s, act1_s, act2_s
        )

    tensors = [x, w1g, w2g, w1s, w2s]
    tensors += [t for t in (b1g, b2g, b1s, b2s) if t is not None]
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all inputs must be on CUDA when x is CUDA")

    x = x.contiguous()
    w1g = w1g.contiguous()
    w2g = w2g.contiguous()
    w1s = w1s.contiguous()
    w2s = w2s.contiguous()
    b1g_arg = b1g.contiguous() if b1g is not None else w1g
    b2g_arg = b2g.contiguous() if b2g is not None else w2g
    b1s_arg = b1s.contiguous() if b1s is not None else w1s
    b2s_arg = b2s.contiguous() if b2s is not None else w2s
    M, K0 = x.shape
    N1 = w1g.shape[0]
    N2 = w2g.shape[0]
    y = torch.empty((M, N2), device=x.device, dtype=x.dtype)
    block_m = 32 if M >= 32 else 16
    block_n2 = 32 if N2 <= 32 else (128 if N2 <= 128 else 64)
    block_n1 = 16 if N1 <= 16 else 32
    block_k0 = 16 if K0 <= 16 else (32 if K0 <= 32 else 64)
    if block_n2 >= 128 or block_k0 >= 64:
        num_warps = 4
        num_stages = 3
    else:
        num_warps = 2 if block_m * block_n2 >= 1024 else 1
        num_stages = 1 if K0 <= block_k0 else 2
    grid = (triton.cdiv(M, block_m), triton.cdiv(N2, block_n2))
    _two_path_two_layer_kernel[grid](
        x,
        w1g,
        b1g_arg,
        w2g,
        b2g_arg,
        w1s,
        b1s_arg,
        w2s,
        b2s_arg,
        y,
        M,
        K0,
        N1,
        N2,
        x.stride(0),
        x.stride(1),
        w1g.stride(0),
        w1g.stride(1),
        w2g.stride(0),
        w2g.stride(1),
        w1s.stride(0),
        w1s.stride(1),
        w2s.stride(0),
        w2s.stride(1),
        y.stride(0),
        y.stride(1),
        ACT1_G=_act_code(act1_g),
        ACT2_G=_act_code(act2_g),
        ACT1_S=_act_code(act1_s),
        ACT2_S=_act_code(act2_s),
        USE_B1G=b1g is not None,
        USE_B2G=b2g is not None,
        USE_B1S=b1s is not None,
        USE_B2S=b2s is not None,
        BLOCK_M=block_m,
        BLOCK_N2=block_n2,
        BLOCK_K0=block_k0,
        BLOCK_N1=block_n1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def fused_mul5_sigmoid_bwd(
    tangents_1: torch.Tensor,
    addmm_1: torch.Tensor,
    sigmoid_1: torch.Tensor,
    sigmoid_3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MatterSim optimized helper for two elementwise backward intermediates."""
    if not tangents_1.is_cuda:
        mul5 = tangents_1 * sigmoid_3
        sbwd = tangents_1 * (addmm_1 * sigmoid_1) * sigmoid_3 * (1 - sigmoid_3)
        return mul5, sbwd

    t1 = tangents_1.contiguous()
    a1 = addmm_1.contiguous()
    s1 = sigmoid_1.contiguous()
    s3 = sigmoid_3.contiguous()
    out_mul5 = torch.empty_like(t1)
    out_bwd = torch.empty_like(t1)
    numel = t1.numel()
    block = 1024
    grid = (triton.cdiv(numel, block),)
    _fused_mul5_sigmoid_bwd_kernel[grid](
        out_mul5, out_bwd, t1, a1, s1, s3, numel, BLOCK_SIZE=block
    )
    return out_mul5, out_bwd


def fused_add_swish_bwd(
    x: torch.Tensor, addmm: torch.Tensor, sigmoid: torch.Tensor
) -> torch.Tensor:
    """Compute ``x * (sigmoid + addmm * sigmoid * (1 - sigmoid))``."""
    if not x.is_cuda:
        return x * (sigmoid + addmm * sigmoid * (1 - sigmoid))

    x_c = x.contiguous()
    a_c = addmm.contiguous()
    s_c = sigmoid.contiguous()
    out = torch.empty_like(x_c)
    numel = x_c.numel()
    block = 1024
    grid = (triton.cdiv(numel, block),)
    _fused_add_swish_bwd_kernel[grid](out, x_c, a_c, s_c, numel, BLOCK_SIZE=block)
    return out


def scatter_add_2d_dim0(
    src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None
) -> torch.Tensor:
    """Compute ``out[index[i]] += src[i]`` for ``src`` shaped ``[T, D]``."""
    if src.dim() != 2 or index.dim() != 1:
        raise ValueError("src must be [T, D] and index must be [T]")
    if src.shape[0] != index.shape[0]:
        raise ValueError("src.shape[0] must match index.shape[0]")
    if src.shape[0] == 0:
        out_rows = 0 if dim_size is None else int(dim_size)
        return torch.zeros((out_rows, src.shape[1]), device=src.device, dtype=src.dtype)
    out_rows = int(dim_size) if dim_size is not None else int(index.max().item()) + 1
    if not src.is_cuda:
        out = torch.zeros((out_rows, src.shape[1]), dtype=src.dtype, device=src.device)
        return out.scatter_add_(0, index.view(-1, 1).expand_as(src), src)

    src_c = src.contiguous()
    index_c = index.contiguous()
    out = torch.zeros((out_rows, src.shape[1]), dtype=src.dtype, device=src.device)
    T, D = src.shape
    if D <= 32:
        block_t, block_d, num_warps = 8, 32, 2
    elif D <= 64:
        block_t, block_d, num_warps = 4, 64, 2
    elif D <= 128:
        block_t, block_d, num_warps = 4, 128, 4
    else:
        block_t, block_d, num_warps = 8, 128, 4
    grid = (triton.cdiv(T, block_t), triton.cdiv(D, block_d))
    _scatter_add_2d_dim0_tile_kernel[grid](
        src_c,
        index_c,
        out,
        T,
        D,
        src_c.stride(0),
        src_c.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def offset_three_body_indices(
    three_body_indices: torch.Tensor,
    num_three_body: torch.Tensor,
    num_bonds: torch.Tensor,
) -> torch.Tensor:
    """Offset per-graph triplet bond indices into a flattened bond index space."""
    num_three_body = num_three_body.view(-1)
    num_bonds = num_bonds.view(-1)
    if num_bonds.numel() <= 1 or three_body_indices.numel() == 0:
        return three_body_indices
    bond_offsets = torch.cumsum(num_bonds, dim=0) - num_bonds
    if not three_body_indices.is_cuda:
        return three_body_indices + torch.repeat_interleave(
            bond_offsets,
            num_three_body,
            output_size=three_body_indices.shape[0],
        ).unsqueeze(-1)

    triple_offsets = torch.cumsum(num_three_body, dim=0) - num_three_body
    out = three_body_indices.clone()
    block_size = 256
    max_triples = int(num_three_body.max().item())
    grid = (num_bonds.numel(), triton.cdiv(max_triples, block_size))
    _offset_three_body_kernel[grid](
        triple_offsets,
        bond_offsets,
        num_three_body,
        out,
        block_size=block_size,
    )
    return out


def fill_threebody_indices(
    orig_idx: torch.Tensor, k_per_atom: torch.Tensor
) -> torch.Tensor:
    """Build ordered ``(j, l)`` edge-index pairs for each center atom.

    ``orig_idx`` must be grouped by center atom, and ``k_per_atom`` gives the
    number of valid bonds for each center atom.
    """
    k_per_atom = k_per_atom.view(-1).to(torch.long)
    n_triple = k_per_atom * (k_per_atom - 1)
    total = int(n_triple.sum().item())
    if total == 0:
        return torch.empty((0, 2), device=orig_idx.device, dtype=torch.long)

    if not orig_idx.is_cuda:
        pieces = []
        cursor = 0
        for k in k_per_atom.tolist():
            local = orig_idx[cursor : cursor + k]
            for j in range(k):
                for l in range(k):
                    if j != l:
                        pieces.append(torch.stack((local[j], local[l])))
            cursor += k
        return torch.stack(pieces, dim=0).to(torch.long)

    triple_offsets = torch.empty(
        (k_per_atom.numel() + 1,), dtype=torch.long, device=orig_idx.device
    )
    triple_offsets[0] = 0
    triple_offsets[1:] = torch.cumsum(n_triple, dim=0)
    bond_offsets = torch.empty(
        (k_per_atom.numel() + 1,), dtype=torch.long, device=orig_idx.device
    )
    bond_offsets[0] = 0
    bond_offsets[1:] = torch.cumsum(k_per_atom, dim=0)
    out = torch.empty((total, 2), dtype=torch.long, device=orig_idx.device)
    max_triples = int(n_triple.max().item())
    block_size = 256
    grid = (k_per_atom.numel(), triton.cdiv(max_triples, block_size))
    _fill_threebody_kernel[grid](
        orig_idx.to(torch.long).contiguous(),
        k_per_atom.contiguous(),
        bond_offsets,
        triple_offsets,
        out,
        block_size=block_size,
    )
    return out
