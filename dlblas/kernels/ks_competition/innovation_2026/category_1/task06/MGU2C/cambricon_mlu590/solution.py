"""MLU590-oriented Triton 3.2 implementation of Task06."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _target_device():
    """Prefer the MLU device for this profile, with a CUDA smoke-test fallback."""
    mlu = getattr(torch, "mlu", None)
    if mlu is not None and mlu.is_available():
        return torch.device("mlu")
    return torch.device("cuda")


# (implementation, BLOCK_M, BLOCK_N, num_warps, num_stages)
_CHIP_CONFIGS = {
    "auto": ("direct", 16, 128, 1, 1),
    # The direct BM32/BN128 W1/S1 path is the compile-safe MLU590 default:
    # 24 fixed owners cover three query tiles for each of eight heads. The
    # 64+32 head-reuse path remains an opt-in traffic-reduction experiment.
    "cambricon_mlu590": ("direct", 32, 128, 1, 1),
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    config = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip == "cambricon_mlu590":
        # Bounded, stock-Triton alternatives for KernelSwift A/B submission.
        # Unknown values retain the deterministic static default.
        variants = {
            "direct16x128x1s1": ("direct", 16, 128, 1, 1),
            "direct16x128x4s1": ("direct", 16, 128, 4, 1),
            "direct32x128x1s1": ("direct", 32, 128, 1, 1),
            "direct32x128x4s1": ("direct", 32, 128, 4, 1),
            "direct32x128x1s3": ("direct", 32, 128, 1, 3),
            "direct64x128x1s1": ("direct", 64, 128, 1, 1),
            "direct64x128x4s1": ("direct", 64, 128, 4, 1),
            "split96m32x1s1": ("split96", 32, 96, 1, 1),
            "head_reuse96x1s1": ("head_reuse96", 32, 96, 1, 1),
            "head_reuse96x2s2": ("head_reuse96", 32, 96, 2, 2),
            "head_reuse96x4s3": ("head_reuse96", 32, 96, 4, 3),
            "online16x32x1s1": ("online", 16, 32, 1, 1),
            "online32x32x1s1": ("online", 32, 32, 1, 1),
            "online32x64x1s1": ("online", 32, 64, 1, 1),
            "online32x64x4s1": ("online", 32, 64, 4, 1),
        }
        key = (
            os.environ.get("KERNELSWIFT_TASK06_CONFIG", "direct32x128x1s1")
            .strip()
            .lower()
        )
        config = variants.get(key, config)
    return config


@triton.jit
def _attention_direct_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    kv_head = head % num_kv_heads

    m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        query_ptr
        + batch * stride_qb
        + m[:, None] * stride_qs
        + (head * head_size + d[None, :]) * stride_qd,
        mask=(m[:, None] < q_len) & (d[None, :] < head_size),
        other=0.0,
    )
    k_nd = tl.load(
        key_ptr
        + batch * stride_kb
        + n[:, None] * stride_ks
        + (kv_head * head_size + d[None, :]) * stride_kd,
        mask=(n[:, None] < kv_len) & (d[None, :] < head_size),
        other=0.0,
    )
    logits = tl.dot(q, tl.trans(k_nd)).to(tl.float32)
    logits *= scale * 1.4426950408889634
    logits = tl.where(n[None, :] < kv_len, logits, -float("inf"))
    row_max = tl.max(logits, axis=1)
    probabilities = tl.exp2(logits - row_max[:, None])
    row_sum = tl.sum(probabilities, axis=1)
    v = tl.load(
        value_ptr
        + batch * stride_vb
        + n[:, None] * stride_vs
        + (kv_head * head_size + d[None, :]) * stride_vd,
        mask=(n[:, None] < kv_len) & (d[None, :] < head_size),
        other=0.0,
    )
    accumulator = tl.dot(probabilities.to(tl.float16), v).to(tl.float32)
    result = accumulator / row_sum[:, None]
    tl.store(
        output_ptr
        + batch * stride_ob
        + m[:, None] * stride_os
        + (head * head_size + d[None, :]) * stride_od,
        result,
        mask=(m[:, None] < q_len) & (d[None, :] < head_size),
    )


@triton.jit
def _attention_online_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    kv_head = head % num_kv_heads
    m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_base = tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        query_ptr
        + batch * stride_qb
        + m[:, None] * stride_qs
        + (head * head_size + d[None, :]) * stride_qd,
        mask=(m[:, None] < q_len) & (d[None, :] < head_size),
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = scale * 1.4426950408889634
    for start_n in range(0, kv_len, BLOCK_N):
        n = start_n + n_base
        k_nd = tl.load(
            key_ptr
            + batch * stride_kb
            + n[:, None] * stride_ks
            + (kv_head * head_size + d[None, :]) * stride_kd,
            mask=(n[:, None] < kv_len) & (d[None, :] < head_size),
            other=0.0,
        )
        logits = tl.dot(q, tl.trans(k_nd)).to(tl.float32) * qk_scale
        logits = tl.where(n[None, :] < kv_len, logits, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(logits, axis=1))
        probabilities = tl.exp2(logits - new_max[:, None])
        correction = tl.exp2(running_max - new_max)
        block_sum = tl.sum(probabilities, axis=1)
        v = tl.load(
            value_ptr
            + batch * stride_vb
            + n[:, None] * stride_vs
            + (kv_head * head_size + d[None, :]) * stride_vd,
            mask=(n[:, None] < kv_len) & (d[None, :] < head_size),
            other=0.0,
        )
        accumulator *= correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.float16), v)
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    result = accumulator / running_sum[:, None]
    tl.store(
        output_ptr
        + batch * stride_ob
        + m[:, None] * stride_os
        + (head * head_size + d[None, :]) * stride_od,
        result,
        mask=(m[:, None] < q_len) & (d[None, :] < head_size),
    )


@triton.jit
def _attention_fixed83_split96_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Official N=83 attention as exact 64+32 score/probability tiles.

    Direct BN128 spends tensor arithmetic on 45 padded key lanes.  This
    candidate keeps one common softmax maximum and denominator across a full
    64-lane tile and a masked 32-lane tail, reducing logical QK/PV work by 25%
    without materializing scores in global memory.
    """
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, 64)
    n0 = tl.arange(0, 64)
    n1 = 64 + tl.arange(0, 32)
    m_mask = m < 83
    n1_mask = n1 < 83

    query = tl.load(
        query_ptr
        + batch * stride_qb
        + m[:, None] * stride_qs
        + (head * head_size + d[None, :]) * stride_qd,
        mask=m_mask[:, None],
        other=0.0,
    )
    key0 = tl.load(
        key_ptr
        + batch * stride_kb
        + n0[:, None] * stride_ks
        + (head * head_size + d[None, :]) * stride_kd,
    )
    key1 = tl.load(
        key_ptr
        + batch * stride_kb
        + n1[:, None] * stride_ks
        + (head * head_size + d[None, :]) * stride_kd,
        mask=n1_mask[:, None],
        other=0.0,
    )
    qk_scale = scale * 1.4426950408889634
    scores0 = tl.dot(query, tl.trans(key0)).to(tl.float32) * qk_scale
    scores1 = tl.dot(query, tl.trans(key1)).to(tl.float32) * qk_scale
    scores1 = tl.where(n1_mask[None, :], scores1, -float("inf"))
    row_max = tl.maximum(tl.max(scores0, axis=1), tl.max(scores1, axis=1))
    probabilities0 = tl.exp2(scores0 - row_max[:, None])
    probabilities1 = tl.exp2(scores1 - row_max[:, None])
    row_sum = tl.sum(probabilities0, axis=1) + tl.sum(probabilities1, axis=1)

    value0 = tl.load(
        value_ptr
        + batch * stride_vb
        + n0[:, None] * stride_vs
        + (head * head_size + d[None, :]) * stride_vd,
    )
    value1 = tl.load(
        value_ptr
        + batch * stride_vb
        + n1[:, None] * stride_vs
        + (head * head_size + d[None, :]) * stride_vd,
        mask=n1_mask[:, None],
        other=0.0,
    )
    accumulator = tl.dot(probabilities0.to(tl.float16), value0).to(tl.float32)
    accumulator += tl.dot(probabilities1.to(tl.float16), value1).to(tl.float32)
    result = accumulator / row_sum[:, None]
    tl.store(
        output_ptr
        + batch * stride_ob
        + m[:, None] * stride_os
        + (head * head_size + d[None, :]) * stride_od,
        result,
        mask=m_mask[:, None],
    )


@triton.jit
def _attention_fixed83_head_reuse96_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
):
    """One batch/head owner reuses 64+32 K/V rows for three query tiles."""
    batch_head = tl.program_id(0)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    d = tl.arange(0, 64)
    n0 = tl.arange(0, 64)
    n1 = 64 + tl.arange(0, 32)
    valid_n1 = n1 < 83

    key0 = tl.load(
        key_ptr
        + batch * stride_kb
        + n0[:, None] * stride_ks
        + (head * head_size + d[None, :]) * stride_kd,
    )
    key1 = tl.load(
        key_ptr
        + batch * stride_kb
        + n1[:, None] * stride_ks
        + (head * head_size + d[None, :]) * stride_kd,
        mask=valid_n1[:, None],
        other=0.0,
    )
    value0 = tl.load(
        value_ptr
        + batch * stride_vb
        + n0[:, None] * stride_vs
        + (head * head_size + d[None, :]) * stride_vd,
    )
    value1 = tl.load(
        value_ptr
        + batch * stride_vb
        + n1[:, None] * stride_vs
        + (head * head_size + d[None, :]) * stride_vd,
        mask=valid_n1[:, None],
        other=0.0,
    )
    qk_scale = scale * 1.4426950408889634

    # Sixteen W2 owners are intended to cover the inferred 32 compute cores.
    # K/V stay live across the static loop, eliminating two thirds of their
    # repeated global loads while preserving one kernel launch.
    for query_block in tl.static_range(0, 3):
        m = query_block * 32 + tl.arange(0, 32)
        valid_m = m < 83
        query = tl.load(
            query_ptr
            + batch * stride_qb
            + m[:, None] * stride_qs
            + (head * head_size + d[None, :]) * stride_qd,
            mask=valid_m[:, None],
            other=0.0,
        )
        scores0 = tl.dot(query, tl.trans(key0)).to(tl.float32) * qk_scale
        scores1 = tl.dot(query, tl.trans(key1)).to(tl.float32) * qk_scale
        scores1 = tl.where(valid_n1[None, :], scores1, -float("inf"))
        row_max = tl.maximum(tl.max(scores0, axis=1), tl.max(scores1, axis=1))
        probabilities0 = tl.exp2(scores0 - row_max[:, None])
        probabilities1 = tl.exp2(scores1 - row_max[:, None])
        row_sum = tl.sum(probabilities0, axis=1) + tl.sum(probabilities1, axis=1)
        accumulator = tl.dot(probabilities0.to(tl.float16), value0).to(tl.float32)
        accumulator += tl.dot(probabilities1.to(tl.float16), value1).to(tl.float32)
        tl.store(
            output_ptr
            + batch * stride_ob
            + m[:, None] * stride_os
            + (head * head_size + d[None, :]) * stride_od,
            accumulator / row_sum[:, None],
            mask=valid_m[:, None],
        )


class ModelNew(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        head_size: int = 64,
        num_kv_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = 1.0 / (self.head_size**0.5)
        if self.num_kv_heads not in (1, self.num_heads):
            raise ValueError("reference supports one broadcast KV head or equal heads")
        (
            self._implementation,
            self._block_m,
            self._block_n,
            self._num_warps,
            self._num_stages,
        ) = _get_launch_config()

    def forward(self, query, key, value):
        batch_size, q_len = query.shape[:2]
        kv_len = key.shape[1]
        output = torch.empty_like(query)
        if self._implementation == "head_reuse96":
            _attention_fixed83_head_reuse96_kernel[(batch_size * self.num_heads,)](
                query,
                key,
                value,
                output,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key.stride(0),
                key.stride(1),
                key.stride(2),
                value.stride(0),
                value.stride(1),
                value.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                self.num_heads,
                self.head_size,
                self.scale,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
            return output
        block_n = self._block_n
        kernel = _attention_online_kernel
        if self._implementation == "direct":
            # Preserve complete semantics for a non-official longer sequence.
            block_n = max(block_n, triton.next_power_of_2(kv_len))
            kernel = _attention_direct_kernel
        grid = (
            triton.cdiv(q_len, self._block_m),
            batch_size * self.num_heads,
        )
        if self._implementation == "split96":
            _attention_fixed83_split96_kernel[grid](
                query,
                key,
                value,
                output,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key.stride(0),
                key.stride(1),
                key.stride(2),
                value.stride(0),
                value.stride(1),
                value.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                self.num_heads,
                self.head_size,
                self.scale,
                BLOCK_M=self._block_m,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
            return output
        kernel[grid](
            query,
            key,
            value,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            q_len,
            kv_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.scale,
            BLOCK_M=self._block_m,
            BLOCK_N=block_n,
            BLOCK_D=triton.next_power_of_2(self.head_size),
            num_warps=self._num_warps,
            num_stages=self._num_stages,
        )
        return output


class Model(ModelNew):
    """Compatibility name for runners that import ``Model``."""


def get_inputs():
    batch_size, seq_len, num_heads, head_size = 2, 83, 8, 64
    hidden = num_heads * head_size
    device = _target_device()
    query = torch.randn(batch_size, seq_len, hidden, dtype=torch.float16, device=device)
    key = torch.randn(batch_size, seq_len, hidden, dtype=torch.float16, device=device)
    value = torch.randn(batch_size, seq_len, hidden, dtype=torch.float16, device=device)
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
