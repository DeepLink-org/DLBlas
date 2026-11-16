import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """forward_native → _forward_sdpa with cu_seqlens=None.
        Inputs: [bsz, seq_len, num_heads * head_size]
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        # Reshape to [bsz, seq, num_heads, head_size]
        q = query.view(bsz, q_len, self.num_heads, self.head_size).transpose(1, 2)
        k = key.view(bsz, kv_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = value.view(bsz, kv_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        # scaled_dot_product_attention: [bsz, num_heads, seq, head_size]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # Reshape back to [bsz, seq, num_heads * head_size]
        return out.transpose(1, 2).reshape(bsz, q_len, -1)


def get_inputs():
    # query: [bsz, q_len, num_heads * head_size], float16
    # key:   [bsz, kv_len, num_kv_heads * head_size], float16
    # value: [bsz, kv_len, num_kv_heads * head_size], float16
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cpu")
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cpu")
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cpu")
    return [query, key, value]


def get_init_inputs():
    # num_heads, head_size, num_kv_heads
    return [8, 64, 8]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape"):
                print(o.shape)
    else:
        print(out.shape)
