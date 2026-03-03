import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import triton
import triton.language as tl
from dlblas.kernels.engram import EngramTri


device = "cuda"


class EngramPt(nn.Module):
    def __init__(
        self,
        engram_hidden_size: int,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()

        self.value_proj = nn.Linear(engram_hidden_size, hidden_size, device=device)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, hidden_size, device=device)
                for _ in range(hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(hidden_size, device=device) for _ in range(hc_mult)]
        )
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(hidden_size, device=device) for _ in range(hc_mult)]
        )
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            device=device,
        )

        self.norms = nn.ModuleList(
            [
                nn.RMSNorm(hidden_size, eps=norm_eps, device=device)
                for _ in range(hc_mult)
            ]
        )

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, embeddings, hidden_states):
        gates = []
        for hc_idx in range(hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)

        # output = value + self.short_conv(value)
        normed_chunks = []
        B, T, G, C = value.shape
        for i in range(G):
            chunk = value[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        output = value + y
        # output = value + self.short_conv(value)
        return output



# ----------------------------------------------------------------------
#  Test utilities (unchanged)
# ----------------------------------------------------------------------
engram_hidden_size = 1024
hidden_size = 1024
kernel_size = 4
dilation = 3
hc_mult = 4


def generate_test_data(engram_hidden_size, hidden_size, kernel_size, dilation, hc_mult):
    min_val = -14.0328
    max_val = 13.9169
    shape = (1, 14, 4, 1024)
    hidden_states = (
        torch.rand(shape, dtype=torch.float32, device=device) * (max_val - min_val)
        + min_val
    )

    min_val = -4.0709
    max_val = 4.3762
    shape = (1, 14, 1024)
    embeddings = (
        torch.rand(shape, dtype=torch.float32, device=device) * (max_val - min_val)
        + min_val
    )

    return [hidden_states, embeddings]


def get_inputs():
    hidden_states, embeddings = generate_test_data(
        engram_hidden_size, hidden_size, kernel_size, dilation, hc_mult
    )
    return [embeddings, hidden_states]


def get_init_inputs():
    return [engram_hidden_size, hidden_size, kernel_size, dilation, hc_mult]

def test_engram():
    embeddings_pt, hidden_states_pt = get_inputs()
    embeddings_tri = embeddings_pt.clone()
    hidden_states_tri = hidden_states_pt.clone()

    torch.manual_seed(41)
    engram_tri = EngramTri(engram_hidden_size, hidden_size, kernel_size, dilation, hc_mult)
    hidden_states_tri = (
        engram_tri(embeddings=embeddings_tri, hidden_states=hidden_states_tri)
        + hidden_states_tri
    )
    print(hidden_states_tri)

    torch.manual_seed(41)
    engram_pt = EngramPt(engram_hidden_size, hidden_size, kernel_size, dilation, hc_mult)
    hidden_states_pt = (
        engram_pt(embeddings=embeddings_pt, hidden_states=hidden_states_pt)
        + hidden_states_pt
    )
    print(hidden_states_pt)

    assert torch.allclose(hidden_states_tri, hidden_states_pt, rtol=1e-3, atol=1e-3)
    print("✅ Forward Complete!")
