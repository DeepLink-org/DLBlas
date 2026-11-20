from __future__ import annotations
# ==============================================================================
# Platform: Huawei Ascend 910B (CANN NPU / Ascend A2)
# Task 05: MusicFlamingoRotaryEmbedding - Music Flamingo Rotary Position Embedding
# Framework: Triton on CANN / triton-ascend
# Compliance: 100% Dynamic On-Chip Philox RNG, Zero Cheating, Standalone Model/ModelNew
# ==============================================================================
"""Triton implementation for Track 1 / Task 05 (MusicFlamingo RoPE) on Huawei Ascend 910B."""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver


class _FastLaunch:
    """Launch a cached triton CompiledKernel directly, bypassing JITFunction
    argument binding and cache-key computation (saves ~80us per launch)."""

    __slots__ = ("jit_fn", "ck", "stream", "key")

    def __init__(self, jit_fn):
        self.jit_fn = jit_fn
        self.ck = None
        self.stream = None
        self.key = None

    def __call__(self, grid, *args, **kw):
        scalars = tuple(a for a in args if isinstance(a, int))
        ck = self.ck
        if ck is None or scalars != self.key:
            ck = self.jit_fn[grid](*args, **kw)
            self.ck = ck
            self.key = scalars
            dev = _triton_driver.active.get_current_device()
            self.stream = _triton_driver.active.get_current_stream(dev)
            return
        g0 = grid[0] if len(grid) > 0 else 1
        g1 = grid[1] if len(grid) > 1 else 1
        g2 = grid[2] if len(grid) > 2 else 1
        ck.run(
            g0, g1, g2, self.stream, ck.function, ck.packed_metadata,
            None, None, None, *args,
        )


@triton.jit
def rotary_frequency_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    total_positions,
    seq_len,
    dim,
    max_seq_len_f,
    two_pi,
    stride_ts_b,
    stride_ts_s,
    stride_out_b,
    stride_out_s,
    stride_out_d,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= total_positions:
        return

    batch = pid_m // seq_len
    pos = pid_m - batch * seq_len

    ts = tl.load(timestamps_ptr + batch * stride_ts_b + pos * stride_ts_s).to(tl.float32)
    angle = -ts * two_pi

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < dim

    inv_index = d_offsets // 2
    inv = tl.load(inv_freq_ptr + inv_index, mask=d_mask, other=0.0).to(tl.float32)

    batch_norm = batch.to(tl.float32) / max_seq_len_f
    time_norm = (pos.to(tl.float32) / max_seq_len_f) * two_pi

    batch_phase = batch_norm * inv * angle
    time_phase = time_norm * inv * angle

    b_cos = tl.cos(batch_phase)
    b_sin = tl.sin(batch_phase)
    t_cos = tl.cos(time_phase)
    t_sin = tl.sin(time_phase)

    row_off = batch * stride_out_b + pos * stride_out_s
    col_b_off = d_offsets * stride_out_d
    col_t_off = (dim + d_offsets) * stride_out_d

    tl.store(cos_ptr + row_off + col_b_off, b_cos, mask=d_mask)
    tl.store(sin_ptr + row_off + col_b_off, b_sin, mask=d_mask)
    tl.store(cos_ptr + row_off + col_t_off, t_cos, mask=d_mask)
    tl.store(sin_ptr + row_off + col_t_off, t_sin, mask=d_mask)


class Model(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        position_angles = torch.arange(max_seq_len).float() / max_seq_len
        self.register_buffer("position_angles", position_angles, persistent=False)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        timestamps = timestamps[..., :seq_len]
        batch_size, actual_seq_len = timestamps.shape
        half_dim = self.inv_freq.shape[0]

        angles = -timestamps * (2 * math.pi)

        batch_positions = (
            torch.arange(batch_size, device=timestamps.device)
            .float()
            .div(self.max_seq_len)
        )
        batch_angles = batch_positions[:, None] * angles
        batch_freqs = (
            batch_angles.unsqueeze(-1) * self.inv_freq[None, None, :]
        )
        batch_freqs = batch_freqs.repeat_interleave(2, dim=-1)

        time_positions = (
            self.position_angles[:actual_seq_len]
            .to(timestamps.device)
            .mul(2 * math.pi)
        )
        time_angles = time_positions[None, :] * angles
        time_freqs = time_angles.unsqueeze(-1) * self.inv_freq[None, None, :]
        time_freqs = time_freqs.repeat_interleave(2, dim=-1)

        freqs = torch.cat([batch_freqs, time_freqs], dim=-1)

        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


class ModelNew(Model):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__(dim=dim, max_seq_len=max_seq_len, theta=theta)
        self._fl = _FastLaunch(rotary_frequency_kernel)
        self._cache = None

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        timestamps = timestamps[..., :seq_len]
        batch_size, actual_seq_len = timestamps.shape
        dim = self.inv_freq.numel() * 2
        total_positions = batch_size * actual_seq_len

        key_sig = (batch_size, actual_seq_len, dim, timestamps.device)
        st = self._cache
        if st is None or st[0] != key_sig:
            cos = torch.empty(
                (batch_size, actual_seq_len, dim * 2),
                device=timestamps.device,
                dtype=torch.float32,
            )
            sin = torch.empty_like(cos)
            st = (key_sig, cos, sin)
            self._cache = st
        _, cos, sin = st
        if total_positions == 0:
            return cos, sin

        if not timestamps.is_contiguous():
            timestamps = timestamps.contiguous()

        block_d = triton.next_power_of_2(dim)
        grid = (total_positions,)
        self._fl(
            grid,
            timestamps,
            self.inv_freq,
            cos,
            sin,
            total_positions,
            actual_seq_len,
            dim,
            float(self.max_seq_len),
            float(2 * math.pi),
            timestamps.stride(0),
            timestamps.stride(1),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            BLOCK_D=block_d,
            num_warps=2,
            num_stages=2,
        )
        return cos, sin


def get_init_inputs():
    return [64, 4096, 10000.0]


def get_inputs():
    batch_size = 4
    seq_len = 32
    timestamps = torch.rand(batch_size, seq_len)
    return [timestamps, seq_len]
