# Copyright (c) 2025, DeepLink.
from typing import Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

import dlblas
from dlblas.op_registry import op_registry
from dlblas.utils import ChoiceSpace, SymVar, Tensor, register_dlblas_op


def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


class TopKGatingFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                logits: torch.Tensor,
                k: int,
                capacity_factor: float = 1.0,
                probs_policy: bool = False,
                min_capacity: int = 2,
                higher_precision: bool = False):
        # compute the capacity
        capacity = _capacity(logits, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)).item()
        scores, masks, masked_gates, topk_indices, topk_values = dlblas._topk_gating_fwd_part1(logits, k)
        if probs_policy == False:
            mask_with_capacity, tokens_per_expert_before_capacity, aggregated_probs_per_expert, aux_loss_per_expert = dlblas._topk_gating_fwd_part2_position(
                scores, masks, k, capacity, moe_aux_loss_coeff=1e-3)
        else:
            mask_with_capacity, tokens_per_expert_before_capacity, aggregated_probs_per_expert, aux_loss_per_expert = dlblas._topk_gating_fwd_part2_probs(
                scores, masks, masked_gates, k, capacity, moe_aux_loss_coeff=1e-3)

        final_indices, final_probs = dlblas._topk_gating_fwd_part3(logits, mask_with_capacity, topk_indices,
                                                                   topk_values, k, capacity)

        aux_loss = torch.sum(aux_loss_per_expert)
        ctx.save_for_backward(tokens_per_expert_before_capacity, scores, torch.tensor(k, dtype=torch.int64))
        # return aux_loss, *part3_res
        return aux_loss, final_probs, final_indices, masked_gates, mask_with_capacity

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_l_aux = grad_outputs[0]
        tokens_per_expert_before_capacity, scores, k = ctx.saved_tensors
        # print("TopKGatingFunc-tokens_per_expert_before_capacity:\n", tokens_per_expert_before_capacity)
        # print("TopKGatingFunc-scores:\n", scores)
        # print("TopKGatingFunc-grad_l_aux:", grad_l_aux)
        grad_logits = dlblas._topk_gating_bwd(tokens_per_expert_before_capacity, scores, grad_l_aux, k.item())
        return grad_logits, None, None, None, None, None


def call(logits: torch.Tensor,
         k: int,
         capacity_factor: float = 1.0,
         probs_policy: bool = False,
         min_capacity: int = 2,
         enable_token_rearrange_opt: bool = False):
    return TopKGatingFunc.apply(logits, k, capacity_factor, probs_policy, min_capacity, enable_token_rearrange_opt)


def bench_fn(logits: torch.Tensor,
             k: int,
             capacity_factor: float = 1.0,
             probs_policy: bool = False,
             min_capacity: int = 2,
             enable_token_rearrange_opt: bool = False):
    fn = lambda: call(logits, k, capacity_factor, drop_policy, probs_policy, enable_token_rearrange_opt)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = 'topk_gating'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        capacity_factor = SymVar('capacity_factor')
        drop_policy = SymVar('drop_policy')
        min_capacity = SymVar('min_capacity')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)

        # space = ChoiceSpace([])
        register_dlblas_op(name, None,
                           (logits, torch.SymInt, torch.SymFloat, torch.SymBool, torch.SymInt, torch.SymBool), call,
                           bench_fn, call)
