import math
from functools import lru_cache
import torch_npu  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        x_f32 = x_f32 * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + self.eps)
        return (x_f32 * self.weight).to(dtype)


@lru_cache(16)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_complex = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    y.copy_(x_complex)
    return y


class Model(nn.Module):
    def __init__(
        self,
        max_batch_size: int = 4,
        max_seq_len: int = 256,
        dim: int = 512,
        head_dim: int = 128,
        rope_head_dim: int = 64,
        compress_ratio: int = 4,
        norm_eps: float = 1e-6,
    ):
        super(Model, self).__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coeff = 1 + int(self.overlap)

        self.ape = nn.Parameter(torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32))
        self.wkv = Linear(dim, coeff * head_dim, dtype=torch.float32)
        self.wgate = Linear(dim, coeff * head_dim, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, norm_eps)

        self.register_buffer(
            "kv_state",
            torch.zeros(max_batch_size, coeff * compress_ratio, coeff * head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (max_batch_size, coeff * compress_ratio, coeff * head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, max_seq_len // compress_ratio, head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(rope_head_dim, max_seq_len),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ape, mean=0.0, std=0.02)
        self.wkv.reset_parameters()
        self.wgate.reset_parameters()
        nn.init.ones_(self.norm.weight)

    def overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        batch_size, num_windows, ratio, _ = tensor.shape
        output = tensor.new_full((batch_size, num_windows, 2 * ratio, self.head_dim), fill_value)
        output[:, :, ratio:] = tensor[:, :, :, self.head_dim :]
        output[:, 1:, :ratio] = tensor[:, :-1, :, : self.head_dim]
        return output

    def reset_runtime_state(self) -> None:
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))
        self.kv_cache.zero_()

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        batch_size, seqlen, _ = x.shape
        ratio = self.compress_ratio
        head_dim = self.head_dim
        rope_head_dim = self.rope_head_dim
        overlap = self.overlap
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            if overlap and cutoff >= ratio:
                self.kv_state[:batch_size, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:batch_size, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape

            if remainder > 0:
                kv, self.kv_state[:batch_size, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:batch_size, offset : offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]

            if not should_compress:
                return None

            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            slot = start_pos % ratio
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[slot]

            if overlap:
                self.kv_state[:batch_size, ratio + slot] = kv.squeeze(1)
                self.score_state[:batch_size, ratio + slot] = score.squeeze(1)
                if not should_compress:
                    return None
                merged_kv = torch.cat(
                    [self.kv_state[:batch_size, :ratio, :head_dim], self.kv_state[:batch_size, ratio:, head_dim:]],
                    dim=1,
                )
                merged_score = torch.cat(
                    [
                        self.score_state[:batch_size, :ratio, :head_dim],
                        self.score_state[:batch_size, ratio:, head_dim:],
                    ],
                    dim=1,
                )
                kv = (merged_kv * merged_score.softmax(dim=1)).sum(dim=1, keepdim=True)
                self.kv_state[:batch_size, :ratio] = self.kv_state[:batch_size, ratio:]
                self.score_state[:batch_size, :ratio] = self.score_state[:batch_size, ratio:]
            else:
                self.kv_state[:batch_size, slot] = kv.squeeze(1)
                self.score_state[:batch_size, slot] = score.squeeze(1)
                if not should_compress:
                    return None
                kv = (
                    self.kv_state[:batch_size, :ratio]
                    * self.score_state[:batch_size, :ratio].softmax(dim=1)
                ).sum(dim=1, keepdim=True)

        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio].to(kv.device)
            self.kv_cache[:batch_size, : seqlen // ratio] = kv
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0).to(kv.device)
            self.kv_cache[:batch_size, start_pos // ratio] = kv.squeeze(1)
        apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis)
        return kv


def generate_test_data(params: dict) -> tuple[torch.Tensor, int]:
    batch_size = params["batch_size"]
    seq_len = params["seq_len"]
    dim = params["dim"]
    start_pos = params["start_pos"]
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device="cpu")
    return x, start_pos


def test_kv_compress():
    return Model(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {"batch_size": 1, "seq_len": 12, "dim": 448, "start_pos": 0}
    return list(generate_test_data(params))


def get_init_inputs():
    return [1, 256, 448, 32, 4, 4, 1e-6]
