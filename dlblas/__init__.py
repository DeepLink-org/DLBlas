# Copyright (c) 2025, DeepLink.
from typing import Tuple

import torch
from torch import Tensor

# this import all kernels dynamically
import dlblas.kernels  # noqa
from dlblas.utils import get_op

__version__ = '0.0.1'


# output: l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx
def topk_gating(
    logits: Tensor,
    k: int,
    capacity_factor: float = 1.0,
    min_capacity: int = 2,
    higher_precision: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    op = get_op('topk_gating', (logits, k, capacity_factor, min_capacity, higher_precision))
    return op(logits, k, capacity_factor, min_capacity, higher_precision)


def layernorm_gated(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    op = get_op(
        'layernorm_gated',
        (x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm),
    )
    return op(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    op = get_op('selective_state_update', (state, x, dt, A, B, C, D, z, dt_bias, dt_softplus))
    return op(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)


def matmul(a: Tensor, b: Tensor, activation=''):
    if activation == 'leaky_relu':
        op = get_op('matmul_leaky_relu', (a, b, activation))
        return op(a, b, activation)
    elif activation == '':
        op = get_op('matmul', (a, b))
        return op(a, b)
    else:
        raise f"matmul_{activation} not impl."


def paged_attention(
    query: Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: Tensor,  # [num_seqs]
    block_tables: Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
):
    op = get_op(
        'paged_attention',
        (
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            attn_scale,
            max_context_len,
        ),
    )
    return op(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
    )


def selective_scan(u, delta, A, B, C, D, initial_state):
    op = get_op('selective_scan', (u, delta, A, B, C, D, initial_state))
    return op(u, delta, A, B, C, D, initial_state)


def add_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-6,
):
    op = get_op('add_rms_norm', (hidden_states, weight, eps, residual))
    return op(hidden_states, weight, eps, residual)


def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    op = get_op('rms_norm', (hidden_states, weight, eps))
    return op(hidden_states, weight, eps)


def fill_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indices: torch.Tensor,
):
    op = get_op('fill_kv_cache', (key, value, key_cache, value_cache, kv_indices))
    return op(key, value, key_cache, value_cache, kv_indices)


def partial_rotary_emb(q, k_pe, kv, cos, sin):
    op = get_op('partial_rotary_emb', (q, k_pe, kv, cos, sin))
    return op(q, k_pe, kv, cos, sin)


def fused_rotary_and_fa(q, k, v, cos, sin):
    op = get_op('fused_rotary_and_fa', (q, k, v, cos, sin))
    return op(q, k, v, cos, sin)


def flash_attention_v2(q, k, v):
    op = get_op('flash_attention_v2', (q, k, v))
    return op(q, k, v)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids_1d):
    op = get_op('apply_rotary_pos_emb', (q, k, cos, sin, position_ids_1d))
    return op(q, k, cos, sin, position_ids_1d)


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    n_share_experts_fusion=0,
    routed_scaling_factor=0,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select exerpt groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limitted for now.
    # for non-supported case, we suggestion to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # n_share_experts_fusion: if > 0, the last expert will be replaced with a round-robin shared expert
    # routed_scaling_factor: if > 0, the last expert will be scaled by this factor
    return torch.ops._DLBLAS.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
    )


def moe_sum(input, output):
    return torch.ops._DLBLAS.moe_sum.default(input, output)
