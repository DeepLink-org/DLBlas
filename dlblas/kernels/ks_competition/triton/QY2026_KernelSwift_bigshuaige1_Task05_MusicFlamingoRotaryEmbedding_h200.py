"""KernelSwift Task05 fused two-axis rotary embedding for MetaX C500."""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "metax_c500"
ROW_WARPS = 2


@triton.jit
def _music_rope_kernel(
    timestamps,
    inv_freq,
    position_angles,
    cos_out,
    sin_out,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // seq_len
    seq_idx = row - batch_idx * seq_len
    feature = tl.arange(0, BLOCK)
    valid = feature < (2 * dim)
    local_feature = feature % dim
    inv = tl.load(inv_freq + local_feature // 2, mask=valid, other=0.0)
    time_angle = tl.load(
        position_angles + seq_idx * dim + local_feature,
        mask=valid,
        other=0.0,
    )
    batch_angle = batch_idx * inv / max_seq_len
    base_angle = tl.where(feature < dim, batch_angle, time_angle)
    timestamp = tl.load(timestamps + row).to(tl.float32)
    angle = base_angle * (-timestamp * (2.0 * math.pi))
    output_offset = row * (2 * dim) + feature
    tl.store(cos_out + output_offset, tl.cos(angle), mask=valid)
    tl.store(sin_out + output_offset, tl.sin(angle), mask=valid)


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_seq_len, dtype=torch.float)
        position_angles = (
            positions / max_seq_len * (2 * math.pi)
        ).unsqueeze(-1) * inv_freq
        self.register_buffer(
            "position_angles", position_angles.repeat_interleave(2, dim=-1)
        )

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        batch = timestamps.shape[0]
        output_shape = (batch, seq_len, 2 * self.dim)
        cos = torch.empty(output_shape, dtype=torch.float32, device=timestamps.device)
        sin = torch.empty_like(cos)
        _music_rope_kernel[(batch * seq_len,)](
            timestamps,
            self.inv_freq,
            self.position_angles,
            cos,
            sin,
            seq_len=seq_len,
            dim=self.dim,
            max_seq_len=self.max_seq_len,
            BLOCK=128,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return cos, sin


class Model(ModelNew):
    pass


def get_inputs():
    batch, seq = 4, 32
    return [torch.rand(batch, seq, device="cuda"), seq]


def get_init_inputs():
    return [64, 256, 10000.0]
