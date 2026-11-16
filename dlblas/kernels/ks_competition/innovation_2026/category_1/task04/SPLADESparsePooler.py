import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # run MLM head
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        # SPLADE activation: log(1 + relu(logits))
        x = torch.log1p(F.relu(x))
        # pool per sequence
        result = []
        offset = 0
        for L in seq_lens.tolist():
            chunk = x[offset:offset + L]   # [L, vocab]
            if self.pooling == "max":
                result.append(chunk.max(dim=0).values)
            else:
                result.append(chunk.sum(dim=0))
            offset += L
        return result


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cpu")
    hidden_states = torch.randn(83, 768, device="cpu")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]


if __name__ == "__main__":
    model = Model(*get_init_inputs()).eval()
    with torch.no_grad():
        out = model(*get_inputs())
    for o in out:
        print(o.shape)
