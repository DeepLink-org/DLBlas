# Copyright (c) 2025, DeepLink.
import triton
import triton.language as tl
import torch


if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
else:
    from triton.language.math import fast_expf as tl_exp


KL = 0
UNBIAS = 1
MSE = 2

@triton.jit
def grpo_loss_fwd_kernel(
    log_probs,
    old_logprobs,
    ref_log_probs,
    advantages,
    kl_type,
    kl_coef,
    loss_factor,
    clip,
    out_loss,
    B: tl.constexpr,
    T: tl.constexpr,
    V: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    # current index of T block
    pid = tl.program_id(axis=0)

    # load data and calculate
    offs_probs_dim1 = pid * T + tl.arange(0, T)
    offs_probs_dim2 = tl.arange(0, V)
    d_log_probs = tl.load(log_probs +
                      offs_probs_dim1[:, None] * V +
                      offs_probs_dim2[None, :])

    d_log_probs_old = tl.load(old_logprobs +
                          offs_probs_dim1[:, None] * V +
                          offs_probs_dim2[None, :])
    log_probs_diff = d_log_probs - d_log_probs_old
    ratio = tl_exp(log_probs_diff)

    adv_off = tl.arange(0, T)
    d_advantages = tl.load(advantages + adv_off)
    adv_expand = tl.expand_dims(d_advantages, 1)
    pg_losses = -adv_expand * ratio
    ratio_clamp = tl.clamp(ratio, 1.0 - clip,
                      1.0 + clip)
    pg_losses2 = -adv_expand * ratio_clamp
    pg_loss_max = tl.maximum(pg_losses, pg_losses2)

    pg_loss = tl.sum(pg_loss_max)
    loss = pg_loss * loss_factor

    offs_ref_dim1 = pid * T + tl.arange(0, T)
    d_ref_log_probs = tl.load(ref_log_probs +
                          offs_ref_dim1[:, None] * V +
                          offs_probs_dim2[None, :])

    # init empty variable occupation
    kl_penalty_loss = tl.zeros((T,), dtype=tl.float32)
    if kl_type == 0:
        kl = d_ref_log_probs - d_log_probs
        kl_penalty_loss = kl_coef * kl
        kl_penalty_loss = tl.sum(kl_penalty_loss, axis=1) * loss_factor
    elif kl_type == 1:
        kl = d_ref_log_probs - d_log_probs
        nobias_kl = tl_exp(kl) - kl - 1
        kl_penalty_loss = kl_coef * nobias_kl
        kl_penalty_loss = tl.sum(kl_penalty_loss, axis=1) * loss_factor
    elif kl_type == 2:
        kl_square = (d_ref_log_probs - d_log_probs) * (d_ref_log_probs - d_log_probs)
        kl = kl_coef * kl_square / 2
        kl_penalty_loss = tl.sum(kl, axis=1) * loss_factor
    else:
        assert False
    final_loss = loss + kl_penalty_loss

    # output loss value
    off_loss = tl.arange(0, T)
    tl.store(out_loss + off_loss, final_loss)


@triton.jit
def grpo_loss_bwd_kernel(
    DLOSS,
    LOGP,
    OLD_LOGP,
    REF_LOGP,
    OUT_LOGP,
    ADVANTAGES,
    KL_TYPE: tl.constexpr,
    CLIP: tl.constexpr,
    BETA: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    V: tl.constexpr,
):
    # current index
    pid = tl.program_id(axis=0)
    offs_probs_dim1 = pid * T + tl.arange(0, T)
    offs_probs_dim2 = tl.arange(0, V)
    d_log_probs = tl.load(LOGP +
                      offs_probs_dim1[:, None] * V +
                      offs_probs_dim2[None, :])
    d_log_probs_old = tl.load(OLD_LOGP +
                          offs_probs_dim1[:, None] * V +
                          offs_probs_dim2[None, :])
    d_log_probs_ref = tl.load(REF_LOGP +
                          offs_probs_dim1[:, None] * V +
                          offs_probs_dim2[None, :])

    log_probs_diff_new = d_log_probs - d_log_probs_old
    log_probs_diff_ref = d_log_probs_ref - d_log_probs
    exp_new = tl_exp(log_probs_diff_new)
    exp_ref = tl_exp(log_probs_diff_ref)
    clamp = tl.clamp(exp_new, 1.0 - CLIP, 1.0 + CLIP)

    adv_off = tl.arange(0, T)
    d_advantages = tl.load(ADVANTAGES + adv_off)
    adv_expand = tl.expand_dims(d_advantages, 1)

    # load data and calculate
    loss_off = tl.arange(0, T)
    d_loss = tl.load(DLOSS + loss_off).to(tl.float32)

    # empty occupation for loss_calc
    loss_calc = tl.zeros((T, V), dtype=tl.float32)

    # 3 situation for kl_type
    if KL_TYPE == 0:
        d_loss = tl.expand_dims(d_loss, 1)
        loss_calc = tl.broadcast_to(d_loss, (T, V))
    elif KL_TYPE == 1:
        d_loss = tl.expand_dims(d_loss, 1)
        loss_expand = tl.broadcast_to(d_loss, (T, V))
        loss_calc = loss_expand * exp_ref - loss_expand
    elif KL_TYPE == 2:
        d_loss = tl.expand_dims(d_loss, 1)
        loss_expand = tl.broadcast_to(d_loss, (T, V))

        # deal with kl_type == 2
        # different calculate method
        loss_calc = loss_expand * log_probs_diff_ref
    else:
        assert False

    sum_loss = tl.sum(d_loss)
    sum_loss = tl.broadcast_to(sum_loss, (T, V))

    zeros = tl.zeros((T, V), tl.float32)
    ratio_left = -adv_expand * exp_new
    ratio_right = -adv_expand * clamp
    where = tl.where(ratio_left == ratio_right, sum_loss / 2, sum_loss)
    gt = tl.where(ratio_left > ratio_right, zeros, where)
    lt = tl.where(ratio_left < ratio_right, zeros, where)

    ratio_sum = tl.sum(gt * clamp, axis=1)
    where_exp = tl.where(exp_new >= 1 - CLIP and exp_new <= 1 + CLIP, -gt * adv_expand, zeros)
    ratio_expand = tl.expand_dims(-ratio_sum, 1)
    exp_sum = tl.sum(lt * exp_new, axis=1)

    exp_expand = tl.expand_dims(-exp_sum, 1)
    grad3 = ratio_expand + exp_expand
    exp_mul = (where_exp - lt * adv_expand) * exp_new
    grad1 = exp_mul - loss_calc

    # calculate loss gradients
    if KL_TYPE == 2:
        grad2 = -exp_mul
    elif KL_TYPE == 0 or KL_TYPE == 1:
        grad2 = loss_calc - exp_mul
    else:
        assert False

    # output loss value
    tl.store(OUT_LOGP +
             offs_probs_dim1[:, None] * V +
             offs_probs_dim2[None, :],
             grad1)


class GRPOLoss(torch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        log_probs,
        old_logprobs,
        ref_log_probs,
        advantages,
        kl_type,
        kl_coef,
        loss_factor,
        clip,
        loss,
        B,
        T,
        V,
        BLOCK_SIZE_T,
    ):
        if ref_log_probs is None:
            ref_log_probs = log_probs.detach()

        kl_type = {
            'kl': KL,
            'unbias': UNBIAS,
            'mse': MSE
        }.get(kl_type, None)

        grid = lambda META: (T // BLOCK_SIZE_T,)
        grpo_loss_fwd_kernel[grid](log_probs, old_logprobs,
                            ref_log_probs, advantages, kl_type, kl_coef,
                            loss_factor, clip, loss, B, T, V, BLOCK_SIZE_T)
        return loss

    def backward(
        self,
        loss,
        log_probs,
        old_logprobs,
        ref_logprobs,
        out_logprobs,
        advantages,
        kl_type,
        clip,
        B,
        T,
        V,
        BLOCK_SIZE_T,
    ):
        if ref_logprobs is None:
            ref_logprobs = log_probs.detach()
        beta = 1.0

        kl_type = {
            'kl': KL,
            'unbias': UNBIAS,
            'mse': MSE
        }.get(kl_type, None)

        grid = lambda META: (T // BLOCK_SIZE_T,)
        grpo_loss_bwd_kernel[grid](loss, log_probs, old_logprobs, ref_logprobs,
                                    out_logprobs, advantages, kl_type, clip, beta, B, T, V)
        return out_logprobs
