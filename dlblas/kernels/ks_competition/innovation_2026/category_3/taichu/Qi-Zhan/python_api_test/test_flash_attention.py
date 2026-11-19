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

import os
import math
import random
import torch
import torch_sdaa
import tecoops
import numpy as np


# ========================================================================
# CPU 参考实现 (来自 zoo baseline)
# ========================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "fa_pt")

def cdiv(x, y):
    return (x + y - 1) // y


def flash_attn_varlen_func_ref(
    q,           # [total_q_tokens, num_heads, head_size]
    k_cache,     # [num_blocks, block_size, num_kv_heads, head_size]
    v_cache,     # [num_blocks, block_size, num_kv_heads, head_size]
    q_seq_lens,  # [batch_size]  每个 batch 的 q 序列长度
    kv_seq_lens, # [batch_size]  每个 batch 的 kv 序列长度
    block_table, # [batch_size, block_table_dim]
    causal=True,
    softmax_scale=None,
):
    """CPU 参考: flash_attn_varlen_func

    与 test/zoo/teco/flash_attention/flash_attention.py 中的实现一致。
    """
    _, num_heads, head_size = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    batch_size = len(q_seq_lens)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(head_size))

    cu_seqlens_q = [0]
    for l in q_seq_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + l)

    out = torch.zeros_like(q)

    for i in range(batch_size):
        L = int(q_seq_lens[i])
        S = int(kv_seq_lens[i])

        block_ids = block_table[i, :cdiv(S, block_size)].to(torch.long)
        k_ = k_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_size)[:S]
        v_ = v_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_size)[:S]

        q_start = int(cu_seqlens_q[i])
        q_end = int(cu_seqlens_q[i + 1])
        q_ = q[q_start:q_end]
        out_ = out[q_start:q_end]

        # Causal mask
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if causal:
            attn_mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril(diagonal=0).logical_not()[-L:]
            attn_bias = attn_bias.masked_fill_(attn_mask, float("-inf"))

        # Q: [L, num_heads, head_size] -> [num_heads, L, head_size]
        q_t = q_.permute(1, 0, 2)
        # K: [S, num_kv_heads, head_size] -> [num_kv_heads, head_size, S]
        k_t = k_.permute(1, 2, 0).repeat_interleave(num_heads // num_kv_heads, 0)

        # P = softmax(Q @ K^T * scale + bias)
        p = torch.matmul(q_t, k_t) * softmax_scale
        p = p + attn_bias
        p = torch.softmax(p, dim=-1)

        # V: [S, num_kv_heads, head_size] -> [num_kv_heads, S, head_size]
        v_t = v_.permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0)

        # O = P @ V: [num_heads, L, head_size] -> [L, num_heads, head_size]
        o = torch.matmul(p, v_t).permute(1, 0, 2)
        out_.copy_(o)

    return out


# ========================================================================
# Layout 转换
# ========================================================================

def sdaa_kv_to_ref_layout(kv_sdaa):
    """SDAA kernel layout -> CPU 参考 layout

    SDAA: [num_blocks, num_kv_heads, block_size, head_size]
    参考:  [num_blocks, block_size, num_kv_heads, head_size]
    """
    return kv_sdaa.permute(0, 2, 1, 3).contiguous()


# ========================================================================
# 测试辅助: 随机生成 FA 输入
# ========================================================================

def _make_random_fa_inputs(
    q_seq_lens,      # list[int], per-batch query seq lens
    kv_seq_lens,     # list[int], per-batch kv seq lens
    num_heads=8,
    num_kv_heads=4,
    head_size=128,
    block_size=32,
    max_blocks=64,
    dtype=torch.float16,
):
    batch_size = len(q_seq_lens)
    total_q = sum(q_seq_lens)

    q = torch.randn(total_q, num_heads, head_size, dtype=dtype)

    k_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_size, dtype=dtype)
    v_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_size, dtype=dtype)

    # Block table: 随机分配 block
    max_blocks_per_seq = max(cdiv(s, block_size) for s in kv_seq_lens)
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32)
    available = list(range(max_blocks))
    random.shuffle(available)
    idx = 0
    for i in range(batch_size):
        nb = cdiv(kv_seq_lens[i], block_size)
        for j in range(nb):
            block_table[i, j] = available[idx % max_blocks]
            idx += 1

    return q, k_cache, v_cache, block_table


def _run_one_fa_test(
    q, k_cache, v_cache,
    q_seq_lens, kv_seq_lens, block_table,
    causal=True,
    softmax_scale=None,
    atol=2e-2,
    label="",
):
    batch_size = len(q_seq_lens)
    head_size = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(head_size))
    max_seqlen_q = max(q_seq_lens)
    max_seqlen_k = max(kv_seq_lens)

    # ---- CPU 参考 (float32 精度) ----
    k_ref = sdaa_kv_to_ref_layout(k_cache).float()
    v_ref = sdaa_kv_to_ref_layout(v_cache).float()
    q_f32 = q.float()

    out_ref = flash_attn_varlen_func_ref(
        q_f32, k_ref, v_ref,
        torch.tensor(q_seq_lens),
        torch.tensor(kv_seq_lens),
        block_table,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    # ---- SDAA kernel ----
    q_dev = q.contiguous().to("sdaa")
    k_dev = k_cache.contiguous().to("sdaa")
    v_dev = v_cache.contiguous().to("sdaa")
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32)
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_seq_lens[i]
    cu_seqlens_q = cu_seqlens_q.to("sdaa")
    seqused_k = torch.tensor(kv_seq_lens, dtype=torch.int32).to("sdaa")
    bt_dev = block_table.to("sdaa")
    out_dev = torch.empty_like(q_dev)

    tecoops.flash_attn_varlen_func(
        q_dev, k_dev, v_dev,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        torch.Tensor(),       # cu_seqlens_k (unused)
        seqused_k,
        softmax_scale,
        causal,
        torch.Tensor(),       # window_size (unused)
        bt_dev,
        False,                # return_softmax_lse
        out_dev,
    )

    # ---- 比对 ----
    out_dev_cpu = out_dev.cpu().float()

    # 检查 NaN
    nan_dev = torch.isnan(out_dev_cpu).any().item()
    nan_ref = torch.isnan(out_ref).any().item()
    has_nan = nan_dev or nan_ref
    if nan_dev:
        nan_count = torch.isnan(out_dev_cpu).sum().item()
        nan_mask = torch.isnan(out_dev_cpu)
        print(f"  [{label:30s}] *** SDAA output has {nan_count} NaN values, shape={list(out_dev_cpu.shape)} ***")
        # 打印前 10 个 NaN 位置 (flatten index -> multi-dim index)
        flat_idx = nan_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
        print(f"  [{label:30s}]     NaN at flattened indices: {flat_idx[:10].tolist()}")
        # 如果 NaN 很多，只打印第一个和最后一个的位置范围
        if len(flat_idx) > 10:
            print(f"  [{label:30s}]     ... and {len(flat_idx) - 10} more NaN positions"
                  f" (first at [{0}], last at [{flat_idx[-1]}])")
        # 打印一个 NaN 所在行的数据快照 (前 5 列)
        if out_dev_cpu.dim() >= 2:
            first_nan_row = flat_idx[0].item() // out_dev_cpu.size(-1)
            row_idx = first_nan_row // out_dev_cpu.size(1) if out_dev_cpu.dim() == 3 else first_nan_row
            head_idx = (first_nan_row // out_dev_cpu.size(-1)) % out_dev_cpu.size(1) if out_dev_cpu.dim() == 3 else 0
            if out_dev_cpu.dim() == 3 and row_idx < out_dev_cpu.size(0) and head_idx < out_dev_cpu.size(1):
                print(f"  [{label:30s}]     sample NaN row [batch={row_idx}, head={head_idx}]: "
                      f"{out_dev_cpu[row_idx, head_idx, :8].tolist()} (first 8 cols)")
    if nan_ref:
        nan_count = torch.isnan(out_ref).sum().item()
        print(f"  [{label:30s}] *** CPU ref has {nan_count} NaN values ***")
    if has_nan:
        print(f"  [{label:30s}] max_err=NaN  mean_err=NaN  FAILED (NaN detected)")
        return False, float("nan")

    max_err = (out_dev_cpu - out_ref).abs().max().item()
    mean_err = (out_dev_cpu - out_ref).abs().mean().item()

    passed = max_err < atol
    status = "PASSED" if passed else "FAILED"
    print(f"  [{label:30s}] max_err={max_err:.6e}  mean_err={mean_err:.6e}  {status}")

    return passed, max_err


# ========================================================================
# Test 用例
# ========================================================================

def test_prefill():
    """prefill: L == S (全量计算)"""
    print("-" * 60)
    print("prefill 测试 (L == S)")
    all_ok = True

    cases = [
        # (name, H, KV, HS, BS, q_lens, kv_lens)
        ("prefill_1x64",   8, 4, 128, 32, [64],    [64]),
        ("prefill_1x128",  8, 4, 128, 32, [128],   [128]),
        ("prefill_2x64_128", 8, 4, 128, 32, [64, 128], [64, 128]),
    ]

    for name, H, KV, HS, BS, q_lens, kv_lens in cases:
        q, kc, vc, bt = _make_random_fa_inputs(q_lens, kv_lens, H, KV, HS, BS)
        ok, _ = _run_one_fa_test(q, kc, vc, q_lens, kv_lens, bt, label=name)
        all_ok = all_ok and ok

    return all_ok


def test_decode():
    """decode: L=1, S 较大"""
    print("-" * 60)
    print("decode 测试 (L=1)")
    all_ok = True

    cases = [
        # (name, H, KV, HS, BS, q_lens, kv_lens)
        ("decode_1x128",  8, 4, 128, 32, [1],     [128]),
        ("decode_1x256",  8, 4, 128, 32, [1],     [256]),
        ("decode_2x128",  8, 4, 128, 32, [1, 1],  [128, 64]),
    ]

    for name, H, KV, HS, BS, q_lens, kv_lens in cases:
        q, kc, vc, bt = _make_random_fa_inputs(q_lens, kv_lens, H, KV, HS, BS)
        ok, _ = _run_one_fa_test(q, kc, vc, q_lens, kv_lens, bt, label=name)
        all_ok = all_ok and ok

    return all_ok


def test_chunked_prefill():
    """chunked prefill: L < S"""
    print("-" * 60)
    print("chunked prefill 测试 (L < S)")
    all_ok = True

    # NOTE: current kernel only supports block_size=32
    cases = [
        ("chunked_32x64",   8, 4, 128, 32, [32],    [64]),
        ("chunked_64x128",  8, 4, 128, 32, [64],    [128]),
        ("chunked_2b_32x64", 8, 4, 128, 32, [32, 16], [64, 32]),
    ]

    for name, H, KV, HS, BS, q_lens, kv_lens in cases:
        q, kc, vc, bt = _make_random_fa_inputs(q_lens, kv_lens, H, KV, HS, BS)
        ok, _ = _run_one_fa_test(q, kc, vc, q_lens, kv_lens, bt, label=name)
        all_ok = all_ok and ok

    return all_ok


def test_gqa():
    """GQA: num_heads > num_kv_heads"""
    print("-" * 60)
    print("GQA 测试 (num_heads > num_kv_heads)")
    all_ok = True

    cases = [
        # 常见的 GQA 比例
        ("GQA_32q_8kv_prefill",   32, 8,  128, 32, [64],    [64]),
        ("GQA_16q_4kv_decode",    16, 4,  128, 32, [1],     [128]),
        ("GQA_8q_2kv_chunked",     8, 2,  128, 32, [32],    [64]),
        ("GQA_2q_1kv_decode_2b",   2, 1,  128, 32, [1, 1],  [64, 128]),
    ]

    for name, H, KV, HS, BS, q_lens, kv_lens in cases:
        q, kc, vc, bt = _make_random_fa_inputs(q_lens, kv_lens, H, KV, HS, BS)
        ok, _ = _run_one_fa_test(q, kc, vc, q_lens, kv_lens, bt, label=name)
        all_ok = all_ok and ok

    return all_ok


def test_multi_batch():
    """multi-batch: 混合 decode + prefill"""
    print("-" * 60)
    print("multi-batch 测试 (mixed decode+prefill)")
    all_ok = True

    # 第一个 batch decode (L=1), 第二个 prefill (L=32)
    cases = [
        ("mix_decode_prefill", 8, 4, 128, 32, [1, 32], [128, 64]),
    ]

    for name, H, KV, HS, BS, q_lens, kv_lens in cases:
        q, kc, vc, bt = _make_random_fa_inputs(q_lens, kv_lens, H, KV, HS, BS)
        ok, _ = _run_one_fa_test(q, kc, vc, q_lens, kv_lens, bt, label=name)
        all_ok = all_ok and ok

    return all_ok

# ========================================================================
# 主入口
# ========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("flash_attn_varlen_func 精度测试 (SDAA vs CPU reference)")
    print("=" * 60)

    if not torch.sdaa.is_available():
        print("WARNING: SDAA not available, tests will fail!\n")

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    tests = [
        ("test_prefill",  test_prefill),
        ("test_decode",  test_decode),
        ("test_chunked_prefill",  test_chunked_prefill),
        ("test_gqa",  test_gqa),
        ("test_multi_batch",  test_multi_batch),
    ]

    all_passed = True
    for name, fn in tests:
        ok = fn()
        all_passed = all_passed and ok

    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
