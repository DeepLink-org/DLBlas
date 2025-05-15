# Copyright (c) 2025, DeepLink.
import os
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from triton import Config
from contextlib import nullcontext
from typing import Callable, List, Optional


@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def wait_signal(addr, flat_tid):
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .pred  %p<1>;

                wait_block:
                    ld.global.relaxed.gpu.u32 $0, [$1];
                    setp.eq.u32 %p0, $0, 1;
                    @!%p0 bra wait_block;
            }
            """,
            "=r, l",
            [addr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def grpo_loss_kernel(
    log_probs_shard,
    log_probs,
    ref_log_probs,
    log_probs_detached,
    attention_mask,
    rewards,
    beta,
    loss,
    progress_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    V: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    COMM_BLOCK_SIZE_T: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    # current index of T block
    flat_tid = get_flat_tid()
    pid = tl.program_id(axis=0)
    num_pid = tl.cdiv(T, BLOCK_SIZE_T)

    # wait for data reached
    NUM_PID_T_PER_COMM_BLOCK = COMM_BLOCK_SIZE_T // BLOCK_SIZE_T
    NUM_COMM_BLOCKS = T // COMM_BLOCK_SIZE_T
    NUM_COMM_BLOCKS_PER_RANK = NUM_COMM_BLOCKS // WORLD_SIZE

    pid_t = (pid + NUM_PID_T_PER_COMM_BLOCK * RANK) % num_pid
    comm_block_id = pid_t // NUM_PID_T_PER_COMM_BLOCK
    offs_probs_dim1 = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    if comm_block_id // NUM_COMM_BLOCKS_PER_RANK == RANK:
        offs_probs_dim1 = offs_probs_dim1 % COMM_BLOCK_SIZE_T
        log_probs_ptr = log_probs_shard
    else:
        wait_signal((progress_ptr + comm_block_id).to(tl.uint64), flat_tid)
        log_probs_ptr = log_probs

    # load data and calculate
    offs_probs_dim2 = tl.arange(0, V)
    d_log_probs = tl.load(log_probs_ptr + offs_probs_dim1[:, None] * V + offs_probs_dim2[None, :])
    chosen_tokens = tl.argmax(d_log_probs, axis=-1)

    # gather for chosen_token_logprobs
    chosen_token_logprobs = tl.gather(d_log_probs, tl.expand_dims(chosen_tokens, -1), axis=-1)
    chosen_token_logprobs = tl.ravel(chosen_token_logprobs)

    # load and gather for ref_token_logprobs
    offs_ref_dim1 = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    d_ref_log_probs = tl.load(ref_log_probs + offs_ref_dim1[:, None] * V + offs_probs_dim2[None, :])
    ref_token_logprobs = tl.gather(d_ref_log_probs, tl.expand_dims(chosen_tokens, -1), axis=-1)
    ref_token_logprobs = tl.ravel(ref_token_logprobs)

    # mean
    offs_rewards = tl.arange(0, B)
    d_rewards = tl.load(rewards + offs_rewards)
    sum_grouped_rewards = tl.sum(d_rewards.to(tl.float32))
    mean_grouped_rewards = sum_grouped_rewards / B
    mean_grouped_rewards = mean_grouped_rewards.to(tl.bfloat16)

    # std
    sub_grouped_rewards = d_rewards.to(tl.float32) - mean_grouped_rewards.to(tl.float32)
    square_grouped_rewards = sub_grouped_rewards * sub_grouped_rewards
    square_grouped_rewards = tl.sum(square_grouped_rewards)
    square_grouped_rewards /= B - 1
    std_grouped_rewards = tl.sqrt(square_grouped_rewards).to(tl.bfloat16)

    # calculate advantages
    offs_detach_dim1 = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    d_log_probs_detached = tl.load(log_probs_detached + offs_detach_dim1[:, None] * V + offs_probs_dim2[None, :])
    detach_token_logprobs = tl.gather(d_log_probs_detached, tl.expand_dims(chosen_tokens, -1), axis=-1)
    detach_token_logprobs = tl.ravel(detach_token_logprobs)

    advantages = (d_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    advantages = advantages.to(tl.bfloat16)
    ratio = tl.exp(chosen_token_logprobs - detach_token_logprobs)

    # calculate policy_loss
    policy_loss = -ratio * advantages[:, None]

    # calculate kl_div
    sub_chosen = ref_token_logprobs - chosen_token_logprobs
    kl_div = tl.exp(sub_chosen) - sub_chosen - 1.0

    # calculate per_token_loss
    offs_attn_mask = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    d_attention_mask = tl.load(attention_mask + offs_attn_mask)
    per_token_loss = policy_loss + beta * kl_div
    masked_loss = per_token_loss * d_attention_mask

    # output loss value
    d_loss = tl.sum(masked_loss)
    off_loss = pid_t + tl.arange(0, 1)
    tl.store(loss + off_loss, d_loss)


def all_gather_with_progress(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
):
    assert inp.is_contiguous()
    symm_mem_hdl = symm_mem.rendezvous(inp, group=dist.group.WORLD)
    assert symm_mem_hdl is not None

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    assert inp.numel() % splits_per_rank == 0
    assert progress.numel() == world_size * splits_per_rank

    # check output shape
    output_shape = list(inp.shape)
    output_shape[1] *= world_size
    assert list(output.shape) == output_shape, (list(output.shape), output_shape)

    # assign chunks and sync
    chunks = output.chunk(world_size * splits_per_rank, dim=1)
    for step in range(0, world_size):
        src_rank = (rank + step + 1) % world_size
        for split_id in range(splits_per_rank):
            src_buf = symm_mem_hdl.get_buffer(
                src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
            )
            chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
            # cuStreamWriteValue32 issues a system level fence before the write
            symm_mem_hdl.stream_write_value32(
                progress,
                offset=src_rank * splits_per_rank + split_id,
                val=1,
            )
    symm_mem_hdl.barrier()


def grpo_loss(
    log_probs_shard,
    log_probs_out,
    attention_mask,
    rewards,
    B,
    T,
    V,
    loss,
    BLOCK_SIZE_T,
    grpo_only = False,
    ref_log_probs_out=None,
    beta=0.1,
):
    if ref_log_probs_out is None:
        ref_log_probs_out = log_probs_out.detach()
    log_probs_detached = log_probs_out.detach()

    if grpo_only == True:
        rank = 0
        world_size = int(os.environ.get("WORLD_SIZE", "8"))
    else:
        symm_mem_hdl = symm_mem.rendezvous(log_probs_shard, group=dist.group.WORLD)
        assert symm_mem_hdl is not None, "a_shard must be allocated via SymmetricMemory"
        rank = symm_mem_hdl.rank
        world_size = symm_mem_hdl.world_size

    SPLITS_PER_RANK = 1
    COMM_BLOCK_SIZE_T = T // world_size // SPLITS_PER_RANK
    assert COMM_BLOCK_SIZE_T % BLOCK_SIZE_T == 0
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    if grpo_only == True:
        progress = torch.ones(world_size, dtype=torch.uint32, device="cuda")
    else:
        progress = torch.zeros(world_size, dtype=torch.uint32, device='cuda')
        symm_mem_hdl.barrier(0)
        backend_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(backend_stream):
            all_gather_with_progress(log_probs_out, log_probs_shard,
                                        progress, SPLITS_PER_RANK)

    grid = lambda META: (T // BLOCK_SIZE_T,)
    grpo_loss_kernel[grid](log_probs_shard, log_probs_out,
                           ref_log_probs_out, log_probs_detached,
                           attention_mask, rewards,
                           beta, loss, progress, B, T, V,
                           BLOCK_SIZE_T, COMM_BLOCK_SIZE_T,
                           rank, world_size)
    torch.cuda.current_stream().wait_stream(backend_stream)
    attn_sum = attention_mask.sum()

