# Copyright (c) 2025, DeepLink.
from typing import List, Optional, Tuple
import os

from dlblas.kernels.fused_moe_v2 import fused_moe
enable_eplb = os.environ.get('EPLB_ENABLED', '0') == '1'

import deep_gemm
import torch
import torch.distributed as dist

from dlblas.kernels.moe import (grouped_gemm_triton, quant_fp8, renormalize, silu_and_mul_masked_post_quant_fwd,
                                silu_and_mul_triton_kernel, map_logic_to_physical_idx_hash_random)
from dlblas.layers.moe.kernels.blocked_fp8_fused_moe import dlblas_fused_moe_blocked_fp8
from dlblas.layers.moe.token_dispatcher import DeepEPTokenDispatcherLowLatency, DeepEPTokenDispatcherNormal
from dlblas.utils.logger import get_logger

logger = get_logger(__name__)

class FusedExperts:
    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        block_shape: list[int],
        chunk_size: Optional[int] = 32 * 1024,
    ):
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.block_shape = block_shape
        self.experts_map = torch.arange(num_experts, device="cuda", dtype=torch.int32)
        self.experts_map[self.experts_map >= self.num_local_experts] = -1
        self.chunk_size = chunk_size

    def forward(self, hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                up_weights: torch.Tensor,
                up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None,):
        out_states = fused_moe(hidden_states,
                              up_weights,
                              down_weights,
                              topk_weights,
                              topk_ids,
                              inplace=True,
                              global_num_experts=self.num_experts,
                              num_local_experts=self.num_local_experts,
                              expert_map=self.experts_map,
                              use_fp8_w8a8=True,
                              w1_scale=up_scale,
                              w2_scale=down_scale,
                              block_shape=self.block_shape,
                              chunk_size=self.chunk_size)
        return out_states


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
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(hidden_states_fp8, gate_up_weight_fp8, gateup_output, masked_m,
                                                        expected_m)
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
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(down_input_fp8, gate_down_weight_fp8, down_output, masked_m,
                                                        expected_m)
        return down_output


class FusedMoENormal:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 num_experts: int,
                 hidden_dim: int,
                 layer_index: int = 0,
                 block_size: int = 128,
                 top_k: int = 8,
                 out_dtype: torch.dtype = torch.bfloat16,
                 chunk_size: Optional[int] = 32 * 1024,
                 ):
        self.layer_index = layer_index
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size
        self.num_local_experts = num_experts // ep_size
        self.experts = FusedExperts(
            num_experts=num_experts, 
            num_local_experts=self.num_local_experts,
            top_k=self.top_k, 
            block_shape=[block_size, block_size],
            chunk_size=chunk_size,
            )
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            layer_index=layer_index,
        )
        
    def balanced_packing(self, weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_layers, num_groups = weight.shape
        assert num_groups % num_packs == 0
        groups_per_pack = num_groups // num_packs

        if groups_per_pack == 1:
            pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
            rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
            return pack_index, rank_in_pack

        indices = weight.float().sort(-1, descending=True).indices.cpu()
        pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
        rank_in_pack = torch.full_like(pack_index, fill_value=-1)
        for i in range(num_layers):
            pack_weights = [0] * num_packs
            pack_items = [0] * num_packs
            for group in indices[i]:
                pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack), 
                        key=pack_weights.__getitem__)
                assert pack_items[pack] < groups_per_pack
                pack_index[i, group] = pack
                rank_in_pack[i, group] = pack_items[pack]
                pack_weights[pack] += weight[i, group]
                pack_items[pack] += 1
        return pack_index, rank_in_pack


    def replicate_experts(self, weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        n, num_log = weight.shape
        num_redundant = num_phy - num_log
        assert num_redundant >= 0
        device = weight.device
        phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
        rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        arangen = torch.arange(n, dtype=torch.int64, device=device)
        for i in range(num_log, num_phy):
            redundant_indices = (weight / logcnt).max(dim=-1).indices
            phy2log[:, i] = redundant_indices
            rank[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
        return phy2log, rank, logcnt


    def rebalance_experts_hierarchical(self, weight: torch.Tensor, num_physical_experts: int, num_groups: int, num_nodes: int, num_gpus: int):

        num_layers, num_logical_experts = weight.shape
        assert num_logical_experts % num_groups == 0
        group_size = num_logical_experts // num_groups 
        assert num_groups % num_nodes == 0
        groups_per_node = num_groups // num_nodes
        assert num_gpus % num_nodes == 0
        assert num_physical_experts % num_gpus == 0
        phy_experts_per_gpu = num_physical_experts // num_gpus

        def inverse(perm: torch.Tensor) -> torch.Tensor:
            inv = torch.empty_like(perm)
            inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
            return inv

        tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
        group_pack_index, group_rank_in_pack = self.balanced_packing(tokens_per_group, num_nodes) 
        log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) + 
                    torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
        mlog2log = inverse(log2mlog)
        tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
        phy2mlog, phyrank, mlogcnt = self.replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)  
        tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
        pack_index, rank_in_pack = self.balanced_packing(tokens_per_phy, num_gpus // num_nodes)
        phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
        pphy2phy = inverse(phy2pphy)
        pphy2mlog = phy2mlog.gather(-1, pphy2phy) # [num_layers * num_nodes, num_log_per_nodes]
        pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + 
                    torch.arange(0, num_logical_experts, num_logical_experts // num_nodes).view(1, -1, 1)).flatten(-2)
        pphy2log = mlog2log.gather(-1, pphy2mlog)
        pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
        logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
        return pphy2log, pphyrank, logcnt

    def rebalance_experts(self, weight: torch.Tensor, num_replicas: int, num_groups: int,
                        num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        num_layers, num_logical_experts = weight.shape
        weight = weight.float().cpu()
        if num_groups % num_nodes == 0:
            # use hierarchical load-balance policy
            phy2log, phyrank, logcnt = self.rebalance_experts_hierarchical(weight, num_replicas, 
                                                                    num_groups, num_nodes, num_gpus)
        else:
            # use global load-balance policy
            phy2log, phyrank, logcnt = self.replicate_experts(weight, num_replicas)
        maxlogcnt = logcnt.max().item()
        log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt), 
                                        -1, dtype=torch.int64, device=logcnt.device)
        log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank, 
                torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
        return phy2log, log2phy, logcnt
    
    def ep_expert_list(self, world_size: int, rank: int, num_groups: int=None, num_nodes: int=None, weight: torch.Tensor=None):
        """experts list of current rank."""
        if enable_eplb:
            self.num_groups = num_groups
            self.num_nodes = num_nodes
            self.num_gpus = world_size
            # 调用 rebalance_experts 函数获取映射信息
            phy2log, log2phy, logcnt = self.rebalance_experts(weight, self.num_experts, self.num_groups, self.num_nodes, self.num_gpus)
            self.phy2log = phy2log[0].to('cuda')
            self.log2phy = log2phy[0].to('cuda')
            self.logcnt = logcnt[0].to('cuda')
            # 计算每个 rank 对应的专家数量
            expert_per_rank = (self.num_experts + world_size - 1) // world_size
            first_expert = rank * expert_per_rank
            last_expert = min(first_expert + expert_per_rank, self.num_experts)
            # if(rank == 0):
            # print("first_expert = ", first_expert)
            # print("last_expert = ", last_expert)

            # 获取 phy2log 的切片数据
            sliced_phy2log = self.phy2log[first_expert:last_expert].tolist()
            # if(rank == 0):
            # print("sliced_phy2log = ", sliced_phy2log)

            return sliced_phy2log
        else:
            num_experts = self.num_experts
            expert_per_rank = (num_experts + world_size - 1) // world_size
            first_expert = rank * expert_per_rank
            last_expert = min(first_expert + expert_per_rank, num_experts)
            return list(range(first_expert, last_expert))


    def forward(self,
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
        if enable_eplb:
            topk_ids = map_logic_to_physical_idx_hash_random(topk_ids, self.log2phy, self.logcnt)
        else:
            topk_ids = topk_ids
        recv_hidden_states, recv_topk_ids, recv_topk_weights, tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            expert_list,
        )
        out_states = self.experts.forward(recv_hidden_states, recv_topk_weights, recv_topk_ids, up_weights, up_scale,
                                       down_weights, down_scale)
        out_states = self.token_dispatcher.combine(out_states)
        return out_states

    def capture(self):
        return self.token_dispatcher.buffer_normal.capture()

    def wait(self, event):
        self.token_dispatcher.release()
        event.current_stream_wait()

    def dispatch_async(self,
                       x: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       num_experts: Optional[int] = None,
                       previous_event=None,
                       async_finish=True):
        return self.token_dispatcher.dispatch_normal_async(x, topk_idx, topk_weights, num_experts, previous_event,
                                                           async_finish)

    def combine_async(self, x: torch.Tensor, handle: tuple, previous_event=None, async_finish=True):
        return self.token_dispatcher.combine_normal_async(x, handle, previous_event, async_finish)

    def release(self):
        return self.token_dispatcher.release()

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        return self.experts.forward(state['recv_hidden_states'], state['recv_topk_weights'], state['recv_topk_idx'], up_weight, up_scale, down_weight, down_scale)


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
        return self.token_dispatcher.dispatch_async(hidden_states, topk_idx, num_experts, use_fp8, async_finish)

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        async_finish: bool,
    ):
        return self.token_dispatcher.combine_async(hidden_states, topk_idx, topk_weights, handle, async_finish)

    def fusedmoe_forward(self, state, up_weight, up_scale, down_weight, down_scale):
        recv_hidden_states = state['recv_hidden_states']
        masked_m = state['recv_expert_count']
        hidden_shape = state['raw_hidden_shape']
        topk_idx = state['topk_idx']
        expected_m = (hidden_shape[0] * self.token_dispatcher.buffer_low_latency.group_size * topk_idx.shape[1] +
                      self.token_dispatcher.num_experts) // self.token_dispatcher.num_experts
        return self.experts.forward(recv_hidden_states, up_weight, up_scale, down_weight, down_scale, masked_m,
                                    expected_m)


class FusedMoEBlockedF8Impl:

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 top_k: int,
                 layer_index: int,
                 num_experts: int,
                 hidden_dim: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.top_k = top_k
        self.layer_index = layer_index
        self.hidden_dim = hidden_dim
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype


def build_deepep_moe(low_latency_mode: bool, ep_size:int, ep_group: dist.ProcessGroup, num_experts:int, hidden_dim:int, block_size:int, top_k:int, out_dtype: torch.dtype,chunk_size: Optional[int] = 32 * 1024):
    if low_latency_mode:
        return FusedMoELowLatency(ep_size=ep_size, ep_group=ep_group, num_experts=num_experts, hidden_dim=hidden_dim, block_size=block_size, out_dtype=out_dtype)
    else:
        return FusedMoENormal(ep_size=ep_size, ep_group=ep_group, num_experts=num_experts, hidden_dim=hidden_dim, block_size=block_size, top_k=top_k, out_dtype=out_dtype, chunk_size=chunk_size)
        
        
class DlblasTritonFusedMoEBlockedF8Impl(FusedMoEBlockedF8Impl):
    """triton fused moe blocked f8 implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.float16,
                 ep_size: int = 1):
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


    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        input_size = hidden_states.shape
        hidden_states = hidden_states.flatten(0, -2)
        input_quant, input_scale = quant_fp8(hidden_states, self.block_size, dtype=gate_up_weights.dtype)

        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        output = dlblas_fused_moe_blocked_fp8(input_quant,
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
                                              ep_size=self.ep_size)
        output = output.unflatten(0, input_size[:-1])
        return output
