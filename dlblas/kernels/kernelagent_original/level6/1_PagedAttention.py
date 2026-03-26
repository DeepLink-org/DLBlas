import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conti_input(data, seq_lens):
    data = [x[:l] for x, l in zip(data, seq_lens)]
    data = torch.cat(data, dim=0)
    return data

def _naive_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class Model(nn.Module):
    def __init__(self, dtype=torch.float16, device='cpu'):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def forward(self, batched_q, batched_kv, seq_lens, history_lens):
        mask = self._mask(seq_lens, history_lens)
        return _conti_input(_naive_attention(batched_q, batched_kv, mask), seq_lens)

    def _make_bias(self, seq_lens, history_lens, neg_val):
        full_seq_lens = seq_lens + history_lens
        max_seq_len = seq_lens.max().item()
        max_full_len = full_seq_lens.max().item()
        seq_ranges = [torch.arange(max_seq_len) for _ in seq_lens]
        for r, l in zip(seq_ranges, seq_lens):
            r[l:] = -max_full_len
        seq_ranges = torch.stack(seq_ranges, dim=0).to(self.device)
        kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
        kv_ranges = torch.stack(kv_ranges, 0).to(self.device)
        mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None]
        return mask.float() * neg_val

    def _mask(self, seq_lens, history_lens):
        neg_val = -1e30
        return self._make_bias(seq_lens, history_lens, neg_val)


device = 'cpu'
dtype = torch.float16
feat_dim = 16
feat_dim_v = 16
num_heads_q = 4
num_heads_k = 2
seq_lens = torch.tensor([128], device=device)
history_lens = torch.tensor([128], device=device)

def _batched_q(seq_lens, num_heads_q, feat_dim, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    max_seq_len = seq_lens.max().item()
    return torch.randn(batch_size, max_seq_len, num_heads_q, feat_dim, dtype=dtype, device=device)


def _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    full_seq_lens = seq_lens + history_lens
    max_seq_len = full_seq_lens.max().item()
    k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device=device)
    v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device=device)
    return k, v

def get_inputs():
    batched_q = _batched_q(seq_lens, num_heads_q, feat_dim, dtype)
    batched_kv = _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype)
    return (batched_q, batched_kv, seq_lens, history_lens)

def get_init_inputs():
    return (dtype, device)