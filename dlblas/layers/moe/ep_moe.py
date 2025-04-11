from typing import List

import torch
import torch.distributed as dist
from dlblas.kernels.moe import (grouped_gemm_triton, quant_fp8, silu_and_mul_masked_post_quant_fwd,
                                                  silu_and_mul_triton_kernel, renormalize, quant_fp8)
from dlblas.layers.moe.token_dispatcher import DeepEPTokenDispatcherLowLatency, TokenDispatcherBuilder
import deep_gemm
from dlblas.utils.logger import get_logger

logger = get_logger(__name__)


class DeepEPExpertsGroupedGEMM:
    """MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-
    ai/DeepEP/tree/main)"""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        block_shape: list[int],
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_shape = block_shape
        self.use_fp8_w8a8 = True

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, gate_up_weight: torch.Tensor,
                gate_up_scale: torch.Tensor, gate_down_weight: torch.Tensor, gate_down_scale: torch.Tensor):
        seg_indptr_cur_rank = torch.cat([
            torch.zeros(1, device=tokens_per_expert.device, dtype=tokens_per_expert.dtype),
            torch.cumsum(tokens_per_expert, dim=0),
        ])
        reorder_topk_ids = torch.repeat_interleave(tokens_per_expert)
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            gate_up_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.shape[0] > 0:
            input, input_scale = quant_fp8(hidden_states, 128, dtype=gate_up_weight.dtype)
            gateup_output = grouped_gemm_triton(
                a=input,
                b=gate_up_weight,
                c=gateup_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=input_scale,
                scale_b=gate_up_scale,
                block_shape=self.block_shape,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        silu_and_mul_triton_kernel[(gateup_output.shape[0], )](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            None,
            0,
            self.num_experts_per_partition - 1,
            BLOCK_SIZE=512,
        )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            gate_down_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_input, down_input_scale = quant_fp8(down_input, 128, dtype=gate_down_weight.dtype)
            down_output = grouped_gemm_triton(
                a=down_input,
                b=gate_down_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=down_input_scale,
                scale_b=gate_down_scale,
                block_shape=self.block_shape,
            )
        return down_output


class DeepEPExpertsDeepGEMM:
    def __init__(self, num_experts: int, ep_size: int, block_size: int, out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_size = block_size
        self.use_fp8_w8a8 = True
        self.out_dtype = out_dtype

    def forward(
        self,
        hidden_states_fp8,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_down_weight: torch.Tensor,
        gate_down_scale: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ):

        gate_up_weight_fp8 = (gate_up_weight, gate_up_scale)
        gate_down_weight_fp8 = (gate_down_weight, gate_down_scale)
        assert (hidden_states_fp8[0].size(0) % 4 == 0), f'TMA alignment error: {hidden_states_fp8[0].size(0)}'
        num_groups, m, k = hidden_states_fp8[0].size()
        n = gate_up_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty((num_groups, m, n), device=hidden_states_fp8[0].device, dtype=self.out_dtype)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(hidden_states_fp8, gate_up_weight_fp8,
                                                                              gateup_output, masked_m, expected_m)
        down_input = torch.empty((
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2,
        ),
                                 device=gateup_output.device,
                                 dtype=gate_down_weight.dtype)

        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // self.block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            self.block_size,
            masked_m,
        )
        n = gate_down_weight.size(1)
        down_input_fp8 = (
            down_input,
            deep_gemm.get_col_major_tma_aligned_tensor(down_input_scale),
        )
        down_output = torch.empty((num_groups, m, n), device=down_input.device, dtype=self.out_dtype)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(down_input_fp8, gate_down_weight_fp8,
                                                                              down_output, masked_m, expected_m)
        return down_output


class FusedMoENormal:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 num_experts: int,
                 hidden_dim: int,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.experts = DeepEPExpertsGroupedGEMM(num_experts, ep_size, [block_size, block_size])
        self.token_dispatcher = TokenDispatcherBuilder.build(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, recv_topk_ids, recv_topk_weights, tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            expert_list,
        )
        out_states = self.experts.forward(recv_hidden_states, tokens_per_expert, up_weights, up_scale, down_weights,
                                          down_scale)
        out_states = self.token_dispatcher.combine(out_states)
        return out_states


class FusedMoELowLatency:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 num_experts: int,
                 hidden_dim: int,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.experts = DeepEPExpertsDeepGEMM(num_experts, ep_size, block_size, out_dtype)
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            self.num_experts,
        )
        out_states = self.experts.forward(recv_hidden_states, up_weights, up_scale, down_weights, down_scale, masked_m,
                                          expected_m)
        out_states = self.token_dispatcher.combine(out_states, topk_idx, topk_weights)
        return out_states


class FusedMoEBlockedF8Impl:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 top_k: int,
                 num_experts: int,
                 hidden_dim: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                is_decoding: bool = False,
                expert_list: List[int] = None):
        """forward."""
        topk_weights = renormalize(topk_weights, self.renormalize)
        moe = None
        if is_decoding is False:
            moe = FusedMoENormal(self.ep_size, self.ep_group, self.num_experts, self.hidden_dim, self.block_size,
                                 self.out_dtype)
        else:
            moe = FusedMoELowLatency(self.ep_size, self.ep_group, self.num_experts, self.hidden_dim, self.block_size,
                                     self.out_dtype)
        out_states = moe.forward(hidden_states, topk_weights, topk_ids, gate_up_weights, gate_up_scale, down_weights,
                                 down_scale, expert_list)
        return out_states
