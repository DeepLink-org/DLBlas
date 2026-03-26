import math
import torch
import torch.nn as nn
from torch import Tensor


def flash_attention_varlen_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_start_loc: Tensor,
    kv_seqlens: Tensor,
    causal: bool = True,
) -> Tensor:
    """Naive varlen attention reference.

    Shapes:
      q: (Tq, Hq, Dk)
      k: (Tk, Hk, Dk)
      v: (Tk, Hk, Dv)
      q_start_loc, q_seqlens, kv_start_loc, kv_seqlens: (B,)
    Returns:
      out: (Tq, Hq, Dv)
    """
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    total_q, num_heads_q, head_dim_k = q.shape
    total_k, num_heads_k, head_dim_k2 = k.shape
    assert head_dim_k == head_dim_k2
    head_dim_v = v.size(-1)

    out = q.new_empty(total_q, num_heads_q, head_dim_v)

    group = num_heads_q // num_heads_k
    assert group * num_heads_k == num_heads_q

    for b in range(q_seqlens.numel()):
        q_start = int(q_start_loc[b].item())
        qs = int(q_seqlens[b].item())
        k_start = int(kv_start_loc[b].item())
        ks = int(kv_seqlens[b].item())

        q_b = q[q_start:q_start + qs]  # (qs, Hq, Dk)
        k_b = k[k_start:k_start + ks]  # (ks, Hk, Dk)
        v_b = v[k_start:k_start + ks]  # (ks, Hk, Dv)

        # Expand K/V across groups to Hq heads
        # k: (Hq, Dk, ks)
        k_hd = k_b.permute(1, 2, 0)  # (Hk, Dk, ks)
        k_hd = k_hd.unsqueeze(1).expand(-1, group, -1, -1).reshape(num_heads_q, head_dim_k, ks)
        # v: (Hq, ks, Dv)
        v_hkv = v_b.transpose(0, 1)  # (Hk, ks, Dv)
        v_hkv = v_hkv.unsqueeze(1).expand(-1, group, -1, -1).reshape(num_heads_q, ks, head_dim_v)

        # q: (Hq, qs, Dk)
        q_hsd = q_b.transpose(0, 1)

        # Build causal mask if needed
        if causal:
            history_len = ks - qs
            # mask shape: (qs, ks) with True where valid
            q_pos = torch.arange(qs, device=q.device)[:, None]
            k_pos = torch.arange(ks, device=q.device)[None, :]
            valid = k_pos <= (history_len + q_pos)
            bias = torch.where(valid, torch.zeros((), device=q.device, dtype=torch.float32), torch.full((), -1e30))
            bias = bias[None, :, :]  # (1, qs, ks), broadcast over heads
        else:
            bias = 0.0

        qk = torch.matmul(q_hsd, k_hd) / math.sqrt(head_dim_k)
        qk = qk.to(torch.float32) + bias
        attn = torch.softmax(qk, dim=-1, dtype=torch.float32)
        attn = attn.to(q.dtype)
        o_hsd = torch.matmul(attn, v_hkv)  # (Hq, qs, Dv)
        out[q_start:q_start + qs] = o_hsd.transpose(0, 1)

    return out


class Model(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_start_loc: Tensor,
        q_seqlens: Tensor,
        kv_start_loc: Tensor,
        kv_seqlens: Tensor,
    ) -> Tensor:
        return flash_attention_varlen_ref(q, k, v, q_start_loc, q_seqlens, kv_start_loc, kv_seqlens, causal=self.causal)


dtype = torch.float16


def _conti_input(data, seqlens):
    data = [x[:l] for x, l in zip(data, seqlens)]
    data = torch.cat(data, dim=0)
    return data


def get_inputs():
    num_heads_q = 8
    num_heads_k = 2
    head_dim_k = 32
    head_dim_v = 32

    q_seqlens = torch.tensor([30, 50, 70, 90])
    history_lens = torch.tensor([50, 40, 30, 20])
    kv_seqlens = q_seqlens + history_lens

    batch_size = q_seqlens.numel()
    max_q = int(q_seqlens.max().item())
    max_kv = int(kv_seqlens.max().item())

    batched_q = torch.rand(batch_size, max_q, num_heads_q, head_dim_k, dtype=dtype)
    batched_k = torch.rand(batch_size, max_kv, num_heads_k, head_dim_k, dtype=dtype)
    batched_v = torch.rand(batch_size, max_kv, num_heads_k, head_dim_v, dtype=dtype)

    conti_q = _conti_input(batched_q, q_seqlens)
    conti_k = _conti_input(batched_k, kv_seqlens)
    conti_v = _conti_input(batched_v, kv_seqlens)

    q_start_loc = q_seqlens.cumsum(0) - q_seqlens
    kv_start_loc = kv_seqlens.cumsum(0) - kv_seqlens

    return [conti_q, conti_k, conti_v, q_start_loc, q_seqlens, kv_start_loc, kv_seqlens]


def get_init_inputs():
    return []
