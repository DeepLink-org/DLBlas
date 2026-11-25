# BSD 3-Clause License
#
# Copyright (c) 2024, Tecorigin Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# encoding:utf-8
'''
FlashAttention CPU reference implementation.
Follows flash_attn_varlen_func semantics.

Inputs (from prototxt):
    - blockTable [batch_size, block_table_dim] int32
    - qData [total_q_tokens, local_head_num, head_size] half
    - kCache [max_block_num, local_kv_head_num, block_size, head_size] half
    - vCache [max_block_num, local_kv_head_num, block_size, head_size] half

Proto params:
    - max_seqlen_q, max_seqlen_k
    - q_seq_lens (repeated int32, length=batch_size)
    - kv_seq_lens (repeated int32, length=batch_size)

Output:
    - oData [total_q_tokens, local_head_num, head_size] half
'''

import os
import sys
import json
import math
import torch
import numpy as np
sys.path.append("../zoo/teco/")
sys.path.append("../")
from executor import *


def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    if param_path == "":
        print("The path of prototxt file is empty.")
        return False
    if len(input_lists) != 6:
        print("The number of input data is wrong (expected 6: blockTable, qData, kCache, vCache, q_seq_lens, kv_seq_lens).")
        return False
    if len(reuse_lists) != 0:
        print("The number of reuse data is wrong.")
        return False
    if len(output_lists) != 1:
        print("The number of output data is wrong (expected 1: oData).")
        return False
    return True


def cdiv(x, y):
    return (x + y - 1) // y

def flash_attn_varlen_func(
    q,           # [total_q_tokens, num_heads, head_size]
    k_cache,     # [num_blocks, block_size, num_kv_heads, head_size]
    v_cache,     # [num_blocks, block_size, num_kv_heads, head_size]
    q_seq_lens,  # [batch_size]
    kv_seq_lens, # [batch_size]
    block_table, # [batch_size, block_table_dim]
    causal=True,
    softmax_scale=None,
):
    _, num_heads, head_size = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
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

        # Causal mask bias: [q_seq_len, kv_seq_len]
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if causal:
            attn_mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril(diagonal=0).logical_not()[-L:]
            attn_bias = attn_bias.masked_fill_(attn_mask, float("-inf"))

        # Q: [L, num_heads, head_size] -> [num_heads, L, head_size]
        q_t = q_.permute(1, 0, 2)
        # K: [S, num_kv_heads, head_size] -> [num_kv_heads, head_size, S]
        # GQA repeat: num_kv_heads -> num_heads
        k_t = k_.permute(1, 2, 0).repeat_interleave(num_heads // num_kv_heads, 0)

        # P = softmax(Q @ K^T * scale + bias)
        p = torch.matmul(q_t, k_t) * softmax_scale 
        p = p + attn_bias  # [num_heads, L, S]
        p = torch.softmax(p, dim=-1)

        # V: [S, num_kv_heads, head_size] -> [num_kv_heads, S, head_size]
        # GQA repeat: num_kv_heads -> num_heads
        v_t = v_.permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0)

        # O = P @ V : [num_heads, L, head_size] -> [L, num_heads, head_size]
        o = torch.matmul(p, v_t).permute(1, 0, 2)
        out_.copy_(o)

    return out


def test_flash_attention(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return
    if device == "cuda":
        is_avail, used_device = is_device_available(device)
        if not is_avail:
            return

    params = read_prototxt(param_path)
    input_params = params["input"]
    output_params = params["output"]

    # Read seq_lens from input tensors (input[4], input[5])
    q_seq_lens_t = to_tensor(input_lists[4], input_params[4], device=device)
    kv_seq_lens_t = to_tensor(input_lists[5], input_params[5], device=device)
    q_seq_lens = q_seq_lens_t.cpu().numpy().astype(int).tolist()
    kv_seq_lens = kv_seq_lens_t.cpu().numpy().astype(int).tolist()

    # Read input tensors
    block_table = to_tensor(input_lists[0], input_params[0], device=device)  # [batch_size, block_table_dim] int32
    q_data = to_tensor(input_lists[1], input_params[1], device=device)       # [total_q_tokens, num_heads, head_size] half
    k_cache = to_tensor(input_lists[2], input_params[2], device=device)      # [num_blocks, num_kv_heads, block_size, head_size] half
    v_cache = to_tensor(input_lists[3], input_params[3], device=device)      # [num_blocks, num_kv_heads, block_size, head_size] half

    q_data = q_data.to(torch.float32)
    k_cache = k_cache.to(torch.float32)
    v_cache = v_cache.to(torch.float32)

    # flash_attn_varlen_func expects cache as [num_blocks, block_size, num_kv_heads, head_size]
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()

    out = flash_attn_varlen_func(
        q_data, k_cache, v_cache,
        q_seq_lens, kv_seq_lens,
        block_table,
        causal=True,
        softmax_scale=1.0 / math.sqrt(float(q_data.shape[2])),
    )

    with open(output_lists[0], "wb") as f:
        save_tensor(f, out, output_params["dtype"])


def parse_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params


if __name__ == "__main__":
    params = parse_params(sys.argv[1])
    device = sys.argv[2]
    test_flash_attention(params["param_path"], params["input_lists"],
                         params["reuse_lists"], params["output_lists"], device)
