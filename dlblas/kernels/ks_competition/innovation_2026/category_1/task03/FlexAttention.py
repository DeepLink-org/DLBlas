import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # query: [num_tokens, num_heads, head_size]
        # key/value: [num_tokens, num_kv_heads, head_size]
        num_tokens = query.shape[0]
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                             is_causal=True)
        return out.squeeze(0).transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_size)


def get_inputs():
    # query: [num_tokens, num_heads, head_size], float16
    # key:   [num_tokens, num_kv_heads, head_size], float16
    # value: [num_tokens, num_kv_heads, head_size], float16
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cpu")
    key   = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cpu")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cpu")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]


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
