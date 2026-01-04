# Copyright (c) 2025, DeepLink.
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from dlblas.kernels.fp8 import per_token_group_quant_fp8
from dlblas.kernels.fused_moe_v3 import fused_moe_v3
from dlblas.kernels.moe import quant_fp8, silu_and_mul_masked_post_quant_fwd
from dlblas.layers.moe.experts_distribution_recorder import ExpertsDistributionRecorder
from dlblas.layers.moe.kernels.blocked_fp8_fused_moe import dlblas_fused_moe_blocked_fp8
from dlblas.layers.moe.token_dispatcher import (
    DeepEPTokenDispatcherLowLatency,
    DeepEPTokenDispatcherNormal,
)
from dlblas.utils.logger import get_logger
from dlblas.utils.utils import DisposibleTensor

try:
    import deep_gemm

    use_deep_gemm = True
except ImportError:
    use_deep_gemm = False

logger = get_logger(__name__)

enable_moe_load_stats = (
    os.environ.get("DLBLAS_MOE_LOAD_STATS", "false").lower() == "true"
)


class FusedMoENormal:
    recorder = ExpertsDistributionRecorder(output_dir="/tmp/dlblas/prefill_moe_stats")

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int = 0,
        block_size: int = 128,
        top_k: int = 8,
        out_dtype: torch.dtype = torch.bfloat16,
        chunk_size: Optional[int] = 32 * 1024,
        expert_alignment: int = 128,
    ):
        self.layer_index = layer_index
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size
        self.num_local_experts = num_experts // ep_size
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            expert_alignment=expert_alignment,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: List[int] = None,
    ):
        """forward."""
        if enable_moe_load_stats:
            FusedMoENormal.recorder.record(topk_ids, self.layer_index, self.num_experts)
        hs_quant, hs_scale = per_token_group_quant_fp8(hidden_states, self.block_size)
        hidden_states = None
        x, recv_topk_ids, recv_topk_weights, recv_tokens_per_expert = (
            self.token_dispatcher.dispatch(
                (hs_quant, hs_scale),
                topk_ids,
                topk_weights,
                expert_list,
            )
        )
        topk_ids, topk_weights = None, None
        out_states = fused_moe_v3(
            x,
            recv_topk_ids,
            recv_topk_weights,
            (up_weights, up_scale),
            (down_weights, down_scale),
            recv_tokens_per_expert,
        )
        out_states = self.token_dispatcher.combine(out_states)
        return out_states

    def capture(self):
        return self.token_dispatcher.buffer_normal.capture()

    def wait(self, event):
        self.token_dispatcher.release()
        event.current_stream_wait()

    def dispatch_async(
        self,
        x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        previous_event=None,
        async_finish=True,
    ):
        if isinstance(x, torch.Tensor):
            hs_quant, hs_scale = per_token_group_quant_fp8(x, self.block_size)
            x = None
        else:
            hs_quant = x[0]
            hs_scale = x[1]
        return self.token_dispatcher.dispatch_normal_async(
            (hs_quant, hs_scale),
            topk_idx,
            topk_weights,
            num_experts,
            previous_event,
            async_finish,
        )

    def combine_async(
        self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True
    ):
        return self.token_dispatcher.combine_normal_async(
            x, handle, previous_event, async_finish
        )

    def release(self):
        return self.token_dispatcher.release()

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        return fused_moe_v3(
            state["recv_hidden_states"],
            state["recv_topk_idx"],
            state["recv_topk_weights"],
            (up_weight, up_scale),
            (down_weight, down_scale),
            state["recv_tokens_per_expert"],
        )

    def per_token_group_quant_fp8(self, x: torch.Tensor):
        return per_token_group_quant_fp8(x, self.block_size)


class FusedMoELowLatency:
    recorder = ExpertsDistributionRecorder(output_dir="/tmp/dlblas/decode_moe_stats")

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int,
        block_size: int = 128,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_experts = num_experts
        self.layer_index = layer_index
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def deepgemm_grouped_fp8_nt_masked(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
        w_tuple: Tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        assert use_deep_gemm, "Please install deep_gemm"
        if hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked"):
            return deep_gemm.m_grouped_fp8_gemm_nt_masked(
                input_tuple, w_tuple, out, masked_m, expected_m
            )
        if hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_masked"):
            return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                input_tuple, w_tuple, out, masked_m, expected_m
            )
        raise RuntimeError("deep_gemm version mismatch")

    def experts(
        self,
        hidden_states_fp8,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_down_weight: torch.Tensor,
        gate_down_scale: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        # modify from sglang
        gate_up_weight_fp8 = (gate_up_weight, gate_up_scale)
        gate_down_weight_fp8 = (gate_down_weight, gate_down_scale)
        num_groups, m, k = hidden_states_fp8[0].shape
        n = gate_up_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=self.out_dtype
        )
        self.deepgemm_grouped_fp8_nt_masked(
            [DisposibleTensor.maybe_unwrap(x) for x in hidden_states_fp8],
            gate_up_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
        )
        DisposibleTensor.maybe_dispose(hidden_states_fp8[0])
        DisposibleTensor.maybe_dispose(hidden_states_fp8[1])
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=gate_down_weight.dtype,
        )
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
        del gateup_output
        n = gate_down_weight.size(1)
        down_input_fp8 = (down_input, down_input_scale)
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=self.out_dtype
        )
        self.deepgemm_grouped_fp8_nt_masked(
            down_input_fp8, gate_down_weight_fp8, down_output, masked_m, expected_m
        )
        return down_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: List[int] = None,
    ):
        """forward."""
        if enable_moe_load_stats:
            FusedMoELowLatency.recorder.record(
                topk_ids, self.layer_index, self.num_experts
            )

        recv_hidden_states, topk_idx, topk_weights, masked_m, expected_m = (
            self.token_dispatcher.dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                self.num_experts,
            )
        )
        hidden_states = None
        out_states = self.experts(
            recv_hidden_states,
            up_weights,
            up_scale,
            down_weights,
            down_scale,
            masked_m,
            expected_m,
        )
        out_states = self.token_dispatcher.combine(out_states, topk_idx, topk_weights)
        return out_states

    def wait(self, event):
        event.current_stream_wait()

    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = True,
        async_finish: bool = True,
    ):
        return self.token_dispatcher.dispatch_async(
            hidden_states, topk_idx, num_experts, use_fp8, async_finish
        )

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        async_finish: bool,
    ):
        return self.token_dispatcher.combine_async(
            hidden_states, topk_idx, topk_weights, handle, async_finish
        )

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        recv_hidden_states = state["recv_hidden_states"]
        masked_m = state["recv_expert_count"]
        hidden_shape = state["raw_hidden_shape"]
        topk_idx = state["topk_idx"]
        expected_m = (
            hidden_shape[0]
            * self.token_dispatcher.buffer_low_latency.group_size
            * topk_idx.shape[1]
            + self.token_dispatcher.num_experts
        ) // self.token_dispatcher.num_experts
        return self.experts(
            recv_hidden_states,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            masked_m,
            expected_m,
        )


class FusedMoEBlockedF8Impl:

    def __init__(
        self,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        top_k: int,
        layer_index: int,
        num_experts: int,
        hidden_dim: int,
        renormalize: bool = False,
        block_size: int = 128,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.top_k = top_k
        self.layer_index = layer_index
        self.hidden_dim = hidden_dim
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype


def build_deepep_moe(
    low_latency_mode: bool,
    ep_size: int,
    ep_group: dist.ProcessGroup,
    num_experts: int,
    hidden_dim: int,
    block_size: int,
    top_k: int,
    out_dtype: torch.dtype,
    layer_idx: int = 0,
    chunk_size: Optional[int] = 32 * 1024,
    expert_alignment: int = 128,
):
    if low_latency_mode:
        return FusedMoELowLatency(
            ep_size=ep_size,
            ep_group=ep_group,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            layer_index=layer_idx,
            block_size=block_size,
            out_dtype=out_dtype,
        )
    else:
        return FusedMoENormal(
            ep_size=ep_size,
            ep_group=ep_group,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            layer_index=layer_idx,
            block_size=block_size,
            top_k=top_k,
            out_dtype=out_dtype,
            chunk_size=chunk_size,
            expert_alignment=expert_alignment,
        )


class DlblasTritonFusedMoEBlockedF8Impl(FusedMoEBlockedF8Impl):
    """triton fused moe blocked f8 implementation."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        block_size: int = 128,
        out_dtype: torch.dtype = torch.float16,
        ep_size: int = 1,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.ep_size = ep_size

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_weights: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: List[int] = None,
    ):
        """forward."""
        input_size = hidden_states.shape
        hidden_states = hidden_states.flatten(0, -2)
        input_quant, input_scale = quant_fp8(
            hidden_states, self.block_size, dtype=gate_up_weights.dtype
        )

        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        output = dlblas_fused_moe_blocked_fp8(
            input_quant,
            input_scale,
            gate_up_weights,
            gate_up_scale,
            down_weights,
            down_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk=self.top_k,
            out_dtype=hidden_states.dtype,
            expert_offset=expert_offset,
            num_experts=num_experts,
            renormalize=self.renormalize,
            ep_size=self.ep_size,
        )
        output = output.unflatten(0, input_size[:-1])
        return output
