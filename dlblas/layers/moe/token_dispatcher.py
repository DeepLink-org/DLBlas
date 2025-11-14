# Copyright (c) 2025, DeepLink.
try:
    # deepep 1.2.1+9af0e0d
    from deep_ep import Buffer
    use_deepep = True
except ImportError:
    use_deepep = False

import os
from typing import List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.distributed as dist

from dlblas.layers.moe.token_dispatcher_base import TokenDispatcherBase
from dlblas.utils.utils import DisposibleTensor


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"


class DeepEPBuffer:
    _buffer_normal = None
    _buffer_low_latency = None
    _buffer_common = None
    _deepep_mode = DeepEPMode.AUTO
    _deepep_sms = Buffer.num_sms
    _num_max_dispatch_tokens_per_rank = 128
    _allow_mnnvl = True
    _latest_mode = DeepEPMode.AUTO
    _hidden_size = -1
    _num_experts = -1

    @classmethod
    def update_parameters(cls, hidden_size, num_experts):
        cls._hidden_size = hidden_size
        cls._num_experts = num_experts
        cls._deepep_sms = int(os.getenv('DEEPEP_SMS', Buffer.num_sms))
        cls._num_max_dispatch_tokens_per_rank = int(os.getenv('DEEPEP_MAX_TOKENS_PER_RANK', 128))
        allow_mnnvl = int(os.getenv('DEEPEP_ENABLE_MNNVL', 1))
        if allow_mnnvl == 0:
            cls._allow_mnnvl = False
        env_deepep_mode = os.getenv('DEEPEP_MODE', "auto").strip().lower()
        if env_deepep_mode == "normal":
            cls._deepep_mode = DeepEPMode.NORMAL
        if env_deepep_mode == "low_latency":
            cls._deepep_mode = DeepEPMode.LOW_LATENCY

    @classmethod
    def set_deepep_mode(cls, mode):
        low_latency_buffer_cleaned = False
        if cls._deepep_mode == DeepEPMode.AUTO and mode == DeepEPMode.LOW_LATENCY and cls._latest_mode == DeepEPMode.NORMAL:
            cls.clean_low_latency_buffer(cls._buffer_common)
            low_latency_buffer_cleaned = True
        cls._latest_mode = mode
        return cls._latest_mode, low_latency_buffer_cleaned

    @classmethod
    def clean_low_latency_buffer(cls, buffer = None):
        if buffer is None:
            buffer = cls._buffer_common
        if isinstance(buffer, Buffer):
            buffer.clean_low_latency_buffer(
                cls._num_max_dispatch_tokens_per_rank,
                cls._hidden_size,
                cls._num_experts,
            )

    @classmethod
    def get_buffer_common(
        cls,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        hidden_bytes: int,
    ):
        if cls._buffer_common is not None:
            return cls._buffer_common
        else:
            cls.update_parameters(hidden_size, num_experts)

        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
                Buffer.get_dispatch_config(group.size()),
                Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
            assert cls._deepep_mode == DeepEPMode.AUTO
            assert cls._num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    cls._num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )
            num_qps_per_rank = max(cls._deepep_sms, num_experts // group.size())

        # Note: deepep 1.2.1 support set deep_ep.Config for Buffer:
        # struct Config {int num_sms;
        #    int num_max_nvl_chunked_send_tokens;
        #    int num_max_nvl_chunked_recv_tokens;
        #    int num_max_rdma_chunked_send_tokens;
        #    int num_max_rdma_chunked_recv_tokens;
        cls._buffer_common = Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_mnnvl=cls._allow_mnnvl
        )
        cls._buffer_common.set_num_sms(cls._deepep_sms)
        return cls._buffer_common

    @classmethod
    def get_buffer_normal(cls, group: dist.ProcessGroup, hidden_bytes: int):
        """Copy from DeepEP example usage in model inference prefilling.

        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
        """
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
                Buffer.get_dispatch_config(group.size()),
                Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

        if (_buffer_normal is None or _buffer_normal.group != group or _buffer_normal.num_nvl_bytes < num_nvl_bytes
                or _buffer_normal.num_rdma_bytes < num_rdma_bytes):
            cls._buffer_normal = Buffer(group, num_nvl_bytes, num_rdma_bytes)
        return cls._buffer_normal

    @classmethod
    def get_buffer_low_latency(
        cls, 
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ):
        """Copy from DeepEP example usage in model inference decoding.

        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """

        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(),
                                                            num_experts)

        if (_buffer_low_latency is None or _buffer_low_latency.group != group or not _buffer_low_latency.low_latency_mode
                or _buffer_low_latency.num_rdma_bytes < num_rdma_bytes):
            assert num_experts % group.size(
            ) == 0, f'num_experts: {num_experts} must be divisible by ep_size: {group.size()}'
            cls._buffer_low_latency = Buffer(
                group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=max(num_experts // group.size(), Buffer.num_sms // 2),
            )
        return cls._buffer_low_latency


class DeepEPTokenDispatcherNormal(TokenDispatcherBase):
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
    ):
        self.dispatch_count = 0
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        self.num_max_dispatch_tokens_per_rank = DeepEPBuffer._num_max_dispatch_tokens_per_rank
        # Handle used for combine operation
        self.handle = None
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = DeepEPBuffer.get_buffer_common(self.group,
                                               self.num_max_dispatch_tokens_per_rank,
                                               self.hidden_size,
                                               self.num_experts,
                                               hidden_bytes=self.hidden_size * self.params_bytes)
        # self.buffer_normal = DeepEPBuffer.get_buffer_normal(self.group,
        #                                        hidden_bytes=self.hidden_size * self.params_bytes)

    def get_buffer(self):
        return self.buffer_normal

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: List[int] = None,
        previous_event=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, x_scales = x if isinstance(x, tuple) else (x, None)
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        (
            x,
            topk_idx,
            topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.dispatch_normal(x, topk_idx, topk_weights, self.num_experts, previous_event)

        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        return x, topk_idx, topk_weights, recv_tokens_per_expert

    def dispatch_normal(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.get_buffer().get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.get_buffer().dispatch(x,
                                        topk_idx=topk_idx,
                                        topk_weights=topk_weights.to(torch.float32),
                                        num_tokens_per_rank=num_tokens_per_rank,
                                        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                                        is_token_in_rank=is_token_in_rank,
                                        num_tokens_per_expert=num_tokens_per_expert,
                                        previous_event=previous_event,
                                        async_finish=False,
                                        allocate_on_comm_stream=False,
                                        expert_alignment=128) # Note: expert_alignment = 128 if deepgemm else 1

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def dispatch_normal_async(self,
                              x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                              topk_idx: torch.Tensor,
                              topk_weights: torch.Tensor,
                              num_experts: Optional[int] = None,
                              previous_event=None,
                              async_finish=True):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.get_buffer().get_dispatch_layout(
            topk_idx,
            num_experts=self.num_experts if num_experts is None else num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.get_buffer().dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
            expert_alignment=128,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_normal_async(self, x: torch.Tensor, handle: Tuple, previous_event=None, async_finish=True):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined_x, event

    def release(self):
        self.handle = None
        self.topk_idx = None
        self.topk_weights = None
        return True


class DeepEPTokenDispatcherLowLatency(TokenDispatcherBase):

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        return_recv_hook: bool = False,
    ):
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        self.handle = None
        self.num_max_dispatch_tokens_per_rank = DeepEPBuffer._num_max_dispatch_tokens_per_rank
        self.buffer_low_latency = DeepEPBuffer.get_buffer_common(self.group,
                                                    self.num_max_dispatch_tokens_per_rank,
                                                    self.hidden_size,
                                                    self.num_experts,
                                                    hidden_bytes=self.hidden_size * self.params_bytes)
        # self.buffer_low_latency = DeepEPBuffer.get_buffer_low_latency(self.group,
        #                                             self.num_max_dispatch_tokens_per_rank,
        #                                             self.hidden_size,
        #                                             self.num_experts)
        self.return_recv_hook = return_recv_hook

    def get_buffer(self):
        return self.buffer_low_latency

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        use_fp8: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_experts is not None and self.num_experts is not None:
            assert self.num_experts == num_experts
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (hidden_states.shape[0] * self.get_buffer().group_size * topk_idx.shape[1] +
                      num_experts) // num_experts

        packed_recv_hidden, masked_m, self.handle, event, hook = (self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=use_fp8,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        packed_recv_hidden = [DisposibleTensor(x) for x in packed_recv_hidden]
        return (
            packed_recv_hidden,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
        )

    # TODO: add use_ue8m0 and use_nvfp4 with round_scale support
    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = True,
        async_finish: bool = True,
    ):
        assert topk_idx.dtype == torch.int64
        if num_experts is not None and self.num_experts is not None:
            assert self.num_experts == num_experts
        recv_hidden_states, recv_expert_count, handle, event, hook = (self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
            use_fp8=use_fp8,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        ))
        recv_hidden_states = [DisposibleTensor(x) for x in recv_hidden_states]
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        combined_hidden_states, event, hook = (self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights.to(torch.float32),
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        ))
        hook() if self.return_recv_hook else event.current_stream_wait()
        return combined_hidden_states

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
        async_finish: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert topk_idx.dtype == torch.int64
        assert topk_weights.dtype == torch.float32
        combined_hidden_states, event, hook = self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        )
        return combined_hidden_states, event, hook
