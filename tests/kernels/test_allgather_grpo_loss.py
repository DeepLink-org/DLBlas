# Copyright (c) 2025, DeepLink.
import os
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from contextlib import nullcontext
from typing import Callable, List, Optional
from liger_kernel.chunked_loss.fused_linear_rlhf import LigerFusedLinearRLHFBase

from dlblas.kernels.allgather_grpo_loss import grpo_loss


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        log_probs,
        attention_mask,
        rewards,
        ref_log_probs=None,
        beta=0.1,
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation."""
        # Get chosen token probabilities
        chosen_tokens = log_probs.argmax(dim=-1)  # (batch_size, seq_len)
        chosen_token_logprobs = log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, seq_len)

        # Get reference model probabilities
        if ref_log_probs is not None:
            with torch.no_grad():
                ref_token_logprobs = ref_log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            ref_token_logprobs = chosen_token_logprobs.detach()

        # Compute advantages per batch entry in a grouped fashion
        mean_grouped_rewards = rewards.mean()  # [batch_size,]
        std_grouped_rewards = rewards.std()  # [batch_size,]

        # Calculate advantages using the same epsilon as in GRPOTrainer
        eps = 1e-4
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)

        # Compute policy gradient loss with importance sampling ratio
        ratio = torch.exp(chosen_token_logprobs - chosen_token_logprobs.detach())
        policy_loss = -ratio * advantages.unsqueeze(1)

        # Compute KL penalty
        kl_div = (
            torch.exp(ref_token_logprobs - chosen_token_logprobs) - (ref_token_logprobs - chosen_token_logprobs) - 1.0
        )

        # Combine losses
        per_token_loss = policy_loss + beta * kl_div

        # Apply masking and normalize
        masked_loss = per_token_loss * attention_mask
        seq_lengths = attention_mask.sum()
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        loss = masked_loss.sum() / seq_lengths
        # Calculate metrics
        metrics = (
            chosen_token_logprobs.mean(),  # mean log prob
            chosen_token_logprobs.std(),  # std log prob
            log_probs.mean(),  # mean all log probs
            ((kl_div * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)).mean(),  # mean KL div
        )

        return loss, metrics

    def forward(
        cls,
        ctx,
        _input,
        weight,
        attention_mask,
        rewards,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
        num_generations=1,
    ):
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            attention_mask=attention_mask,
            rewards=rewards,
            bias=bias,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            num_generations=num_generations,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        grads = LigerFusedLinearRLHFBase.backward(ctx, grad_output)
        return (
            *grads[:5],  # grad_input, grad_weight, grad_attention_mask, grad_rewards, grad_bias
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_num_generations
        )

class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = True,
        num_generations: int = 1,
    ):
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.num_generations = num_generations

    def forward(
        self,
        _input,
        lin_weight,
        attention_mask,
        rewards,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            attention_mask,
            rewards,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.compiled,
            self.use_ref_model,
            self.num_generations,
        )


def benchmark_with_event(
    target_fn: Callable[[None], None],
    warmup_iters: int = 10,
    benchmark_iters: int = 20,
    profile_ranks: Optional[List[int]] = None,
    flush_l2: bool = False,
    cuda_graph: bool = False,
) -> float:
    if cuda_graph:
        target_fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            target_fn()
        target_fn = lambda: g.replay()

    if "BENCHMARK_ITERS" in os.environ:
        benchmark_iters = int(os.environ["BENCHMARK_ITERS"])

    rank = dist.get_rank() if dist.is_initialized() else 0
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        target_fn()

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()

    begin_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    if rank in profile_ranks:
        try:
            from trace_handler import trace_handler
        except ImportError:
            trace_handler = None

        if "NO_TRACE" in os.environ:
            trace_handler = None

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            begin_events[i].record()
            target_fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]
    return torch.tensor(latencies).median().item() * 1000


def all_gather_grpo_loss_cublas(log_probs_shard, attention_mask, rewards, V, liger, ref_log_probs=None):
    torch.distributed.all_gather_into_tensor(log_probs, log_probs_shard)
    return liger.rlhf_loss_fn(log_probs.view(1, -1, V), attention_mask, rewards, ref_log_probs=ref_log_probs),


def all_gather_grpo_loss_triton(log_probs_shard, attention_mask, rewards, B, T, V, loss, ref_log_probs=None):
    torch.distributed.all_gather_into_tensor(log_probs, log_probs_shard)
    return grpo_loss(log_probs_shard, log_probs.view(1, -1, V), attention_mask, rewards, B, T, V, loss,
                     BLOCK_SIZE_T, ref_log_probs_out=ref_log_probs, grpo_only=True)

B = 8
T = 4096
H = 1024
V = 4096
BLOCK_SIZE_T = 4
scalar = 1.0
num_generations = 1
dtype = torch.bfloat16

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.manual_seed(42)  # + rank)
dist.init_process_group("nccl")

_input = torch.randn(B * num_generations, T, H, device='cuda', dtype=dtype) * scalar
_input1 = torch.randn(B * num_generations, T, H, device='cuda', dtype=dtype) * scalar
chunks = max(1, _input.shape[0] // num_generations)
_attention_mask = torch.ones(B * num_generations, T, device='cuda')
_attention_mask = torch.chunk(_attention_mask, chunks=chunks, dim=0)
attention_masks = zip(_attention_mask)
attention_mask = list(attention_masks)[0][0]

weight = torch.randn(V, H, device='cuda', dtype=dtype)
weight1 = torch.randn(V, H, device='cuda', dtype=dtype)
_input_chunk = torch.chunk(_input, chunks=chunks, dim=0)
input_chunks = zip(_input_chunk)
input_chunk = list(input_chunks)[0][0]
_input_chunk1 = torch.chunk(_input1, chunks=chunks, dim=0)
input_chunks1 = zip(_input_chunk1)
input_chunk1 = list(input_chunks1)[0][0]

logits = torch.matmul(input_chunk, weight.t())
log_probs = F.log_softmax(logits.float(), dim=-1)
logits1 = torch.matmul(input_chunk1, weight1.t())
log_probs1 = F.log_softmax(logits1.float(), dim=-1)
rewards = torch.rand(B * num_generations, device='cuda', dtype=dtype)
st_log_probs = log_probs.clone()
st_log_probs1 = log_probs1.clone()
st_rewards = rewards.clone()
st_attention_mask = attention_mask.clone()

chunks = log_probs.chunk(world_size, dim=1)

log_probs_shard = symm_mem.empty(
            1, T // world_size, V, dtype=torch.float32, device=device
            ).copy_(chunks[rank])
loss = torch.zeros((T // BLOCK_SIZE_T,), dtype=torch.float32, device='cuda')

tri_loss = grpo_loss(log_probs_shard, log_probs, attention_mask, rewards, B, T, V, loss,
                            BLOCK_SIZE_T, ref_log_probs_out=log_probs1)

liger = LigerFusedLinearGRPOFunction()

if rank == 0:
    ref_loss, _ = liger.rlhf_loss_fn(st_log_probs, st_attention_mask, st_rewards, ref_log_probs=st_log_probs1)
    print("reference result: ", ref_loss)
    print("triton result: ", tri_loss)

lat_cublas_nccl = benchmark_with_event(
    lambda: all_gather_grpo_loss_cublas(log_probs_shard, st_attention_mask, st_rewards, V, liger, ref_log_probs=st_log_probs1),
    flush_l2=True,
)

lat_triton_nccl = benchmark_with_event(
    lambda: all_gather_grpo_loss_triton(log_probs_shard, attention_mask, rewards, B, T, V, loss, ref_log_probs=log_probs1),
    flush_l2=True,
)

lat_cublas_grpo = benchmark_with_event(
    lambda: liger.rlhf_loss_fn(st_log_probs, st_attention_mask, st_rewards, ref_log_probs=st_log_probs1),
    flush_l2=True,
)

lat_triton_grpo = benchmark_with_event(
    lambda: grpo_loss(log_probs_shard, log_probs, attention_mask, rewards, B, T, V, loss,
                     BLOCK_SIZE_T, ref_log_probs_out=log_probs1, grpo_only=True),
    flush_l2=True,
)

lat_triton_fused = benchmark_with_event(
    lambda: grpo_loss(log_probs_shard, log_probs, attention_mask, rewards, B, T, V, loss,
                     BLOCK_SIZE_T, ref_log_probs_out=log_probs1),
    flush_l2=True,
)

if rank == 0:
    print(f"cublas grpo only:\t{round(lat_cublas_grpo)} us")
    print(f"triton grpo only:\t{round(lat_triton_grpo)} us")
    print(f"cublas nccl:\t{round(lat_cublas_nccl)} us")
    print(f"triton nccl:\t{round(lat_triton_nccl)} us")
    print(f"triton fused:\t{round(lat_triton_fused)} us")
    print(f"triton speedup:\t{lat_cublas_grpo / lat_triton_grpo:.02f}x")
    print(f"communication speedup:\t{lat_triton_nccl / lat_triton_fused:.02f}x")
    print(f"total speedup:\t{lat_cublas_nccl / lat_triton_fused:.02f}x")

dist.destroy_process_group()
