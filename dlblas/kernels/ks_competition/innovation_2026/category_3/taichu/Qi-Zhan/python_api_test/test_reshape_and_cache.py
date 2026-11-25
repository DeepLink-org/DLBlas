#!/usr/bin/env python3
# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import math
import torch
import tecoops


# ========== 你的 Python 参考实现 ==========

def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping):
    """CPU 参考: 将 key/value 按 slot_mapping 写入 paged cache"""
    num_blocks, block_size, num_kv_heads, head_size = key_cache.size()
    key_cache = key_cache.view(num_blocks * block_size, num_kv_heads, head_size)
    value_cache = value_cache.view(num_blocks * block_size, num_kv_heads, head_size)
    for i in range(slot_mapping.numel()):
        slot_idx = int(slot_mapping[i])
        if slot_idx == -1:
            continue
        else:
            key_cache[slot_idx] = key[i]
            value_cache[slot_idx] = value[i]


def cdiv(x, y):
    return (x + y - 1) // y


def flash_attn_varlen_func(
    q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
    cu_seqlens_k=None, seqused_k=None, softmax_scale=None,
    causal=False, window_size=None, block_table=None,
    return_softmax_lse=False, out=None,
):
    """CPU 参考: paged flash attention"""
    assert cu_seqlens_k is None
    assert seqused_k is not None
    assert causal
    assert block_table is not None
    assert not return_softmax_lse

    softmax_scale = 1 / math.sqrt(q.size(-1)) if softmax_scale is None else softmax_scale

    _, num_heads, _ = q.size()
    num_blocks, block_size, num_kv_heads, head_size = k.size()

    for i in range(cu_seqlens_q.numel() - 1):
        L = int(cu_seqlens_q[i + 1]) - int(cu_seqlens_q[i])
        S = int(seqused_k[i])
        block_ids = block_table[i, :cdiv(S, block_size)]

        out_ = out[int(cu_seqlens_q[i]):int(cu_seqlens_q[i + 1])]
        q_ = q[int(cu_seqlens_q[i]):int(cu_seqlens_q[i + 1])]
        k_ = k.index_select(0, block_ids).view(-1, num_kv_heads, head_size)[:S]
        v_ = v.index_select(0, block_ids).view(-1, num_kv_heads, head_size)[:S]

        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if causal:
            attn_mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril(diagonal=0).logical_not()[-L:]
            attn_bias = attn_bias.masked_fill_(attn_mask, float("-inf"))

        q_ = q_.permute(1, 0, 2)
        k_ = k_.permute(1, 2, 0).repeat_interleave(num_heads // num_kv_heads, 0)
        p = torch.bmm(q_, k_) * softmax_scale + attn_bias
        p = torch.softmax(p, dim=-1)
        v_ = v_.permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0)
        o = torch.bmm(p, v_).permute(1, 0, 2)
        out_.copy_(o, non_blocking=True)


# ========== 辅助: SDAA <-> 参考 layout 互转 ==========
# SDAA kernel: [B, H, S, D]    参考: [B, S, H, D]

def sdaa_to_cpu(kc_sdaa):
    """SDAA layout -> 参考 layout"""
    return kc_sdaa.permute(0, 2, 1, 3).contiguous()


# ========== Test: reshape_and_cache 精度对拍（大规模） ==========

CONFIGS = [
    # (name, hidden_size, num_q_heads, num_kv_heads, head_dim)
    ("4096-32q-8kv", 4096, 32, 8, 128),
    ("2048-16q-4kv", 2048, 16, 4, 128),
    ("1536-16q-2kv", 1536, 16, 2, 128),
]


def check_reshape_and_cache_config(name, hidden, q_heads, kv_heads, head_dim):
    print(f"  Config: {name} (hidden={hidden}, kv_heads={kv_heads}, head_dim={head_dim})")

    num_tokens = 512
    block_size = 16
    num_blocks = 64  # ceil(512/16) * 1.5

    # 同一组输入 (fp16)
    key = torch.randn(num_tokens, kv_heads, head_dim, dtype=torch.float16)
    value = torch.randn(num_tokens, kv_heads, head_dim, dtype=torch.float16)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    assert slot_mapping.max() < num_blocks * block_size

    # ---- CPU 参考 ----
    kc_cpu = torch.zeros(num_blocks, block_size, kv_heads, head_dim, dtype=torch.float16)
    vc_cpu = torch.zeros(num_blocks, block_size, kv_heads, head_dim, dtype=torch.float16)
    reshape_and_cache_flash(key, value, kc_cpu, vc_cpu, slot_mapping)

    # ---- SDAA kernel ----
    kc_sdaa = torch.zeros(num_blocks, kv_heads, block_size, head_dim, device='sdaa', dtype=torch.float16)
    vc_sdaa = torch.zeros(num_blocks, kv_heads, block_size, head_dim, device='sdaa', dtype=torch.float16)
    tecoops.reshape_and_cache(key.to('sdaa'), value.to('sdaa'), slot_mapping.to('sdaa'), kc_sdaa, vc_sdaa)

    # ---- 比对 ----
    key_ok = torch.allclose(sdaa_to_cpu(kc_sdaa.cpu()), kc_cpu, atol=0.01)
    val_ok = torch.allclose(sdaa_to_cpu(vc_sdaa.cpu()), vc_cpu, atol=0.01)
    print(f"    key_cache:  {'OK' if key_ok else 'MISMATCH'}")
    print(f"    value_cache: {'OK' if val_ok else 'MISMATCH'}")
    return key_ok and val_ok


def test_reshape_and_cache():
    print("=" * 60)
    print("Test: reshape_and_cache (SDAA vs CPU reference)")
    print("=" * 60)

    all_ok = True
    for name, hidden, q_heads, kv_heads, head_dim in CONFIGS:
        ok = check_reshape_and_cache_config(name, hidden, q_heads, kv_heads, head_dim)
        all_ok = all_ok and ok

    assert all_ok, "FAILED"
    print("  All configs PASSED\n")


# ========== 主入口 ==========

if __name__ == "__main__":
    test_reshape_and_cache()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("=" * 50)
