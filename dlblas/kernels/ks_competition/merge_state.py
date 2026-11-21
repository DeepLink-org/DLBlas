import torch
import triton
import triton.language as tl


def state_merge_torch(o, m, d, other_o, other_m, other_d):
    """
    数值稳定的softmax状态合并
    """
    m_max = torch.maximum(m, other_m)
    d = d * torch.exp2(m - m_max) + other_d * torch.exp2(other_m - m_max)
    o = o * torch.exp2(m - m_max) + other_o * torch.exp2(other_m - m_max)
    return o, m_max, d


def state_normalize_torch(o, m, d):
    """
    归一化状态
    """
    o = o / d
    return o, m, d


def state_get_lse_torch(o, m, d):
    """
    获取log-sum-exp
    """
    return m + torch.log2(d)


def _merge_normalize_fused_torch(v_a, s_a, v_b, s_b):
    """
    Fused, numerically-stable PyTorch implementation that matches the initial semantics.
    Optimized to minimize intermediate tensors and memory traffic on NPU.
    """
    # s_merged via base-2 logaddexp for numerical stability
    s_merged = torch.logaddexp2(s_a, s_b)  # [S, H], fp32

    # Weight for v_a: w_a = 2^(s_a - s_merged) in [0, 1], broadcasting over head_dim
    w_a = (s_a - s_merged).exp2_().unsqueeze(-1)  # [S, H, 1], fp32

    # Merge vectors in fp32 using a single-pass lerp in-place on vb32:
    va32 = v_a.to(torch.float32)
    vb32 = v_b.to(torch.float32)
    vb32.lerp_(va32, w_a)  # vb = vb + (va - vb) * w_a

    return vb32, s_merged


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, v_a, s_a, v_b, s_b):
        """
        Args:
            v_a: [seq_len, num_heads, head_dim] float16
            s_a: [seq_len, num_heads] float32
            v_b: [seq_len, num_heads, head_dim] float16
            s_b: [seq_len, num_heads] float32
        Returns:
            v_merged: [seq_len, num_heads, head_dim] float32
            s_merged: [seq_len, num_heads] float32
        """
        return _merge_normalize_fused_torch(v_a, s_a, v_b, s_b)


# Below are helper utilities for local testing parity with the initial code.
seq_len = 2048
num_heads = 32
head_dim = 128


def get_inputs():
    va = torch.randn(seq_len, num_heads, head_dim, device="npu").half()
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32, device="npu")
    vb = torch.randn(seq_len, num_heads, head_dim, device="npu").half()
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32, device="npu")
    return [va, sa, vb, sb]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)