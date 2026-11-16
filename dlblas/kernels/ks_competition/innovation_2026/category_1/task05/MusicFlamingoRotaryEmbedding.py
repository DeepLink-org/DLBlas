import math
import torch
import torch.nn as nn


class Model(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.
    Returns (cos, sin) where cos/sin combines batch and time frequencies."""

    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
        base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim
        ))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        batch_positions = torch.arange(
            timestamps.shape[0], device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        batch_positions = batch_positions / self.max_seq_len
        batch_freqs = batch_positions.unsqueeze(-1) * self.inv_freq
        batch_freqs = batch_freqs.repeat_interleave(2, dim=-1)

        batch_freqs = batch_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        batch_freqs, time_freqs = torch.broadcast_tensors(batch_freqs, time_freqs)
        freqs = torch.cat((batch_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * math.pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


def get_inputs():
    # timestamps: [batch_size, seq_len] — normalized song timestamps per time step
    B, SEQ = 4, 32
    timestamps = torch.rand(B, SEQ, device="cpu")
    return [timestamps, SEQ]


def get_init_inputs():
    return [64, 256, 10000.0]


if __name__ == "__main__":
    model = Model(*get_init_inputs()).eval()
    with torch.no_grad():
        cos, sin = model(*get_inputs())
    print(cos.shape, sin.shape)
