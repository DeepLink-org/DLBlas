# Copyright (c) 2025, DeepLink.
import triton
import triton.language as tl


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
    ratio = tl.exp(log_probs_diff)

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

    if kl_type == 0:
        kl = d_ref_log_probs - d_log_probs
        kl_penalty_loss = kl_coef * kl
        kl_penalty_loss = tl.sum(kl_penalty_loss, axis=1) * loss_factor
    elif kl_type == 1:
        kl = d_ref_log_probs - d_log_probs
        nobias_kl = tl.exp(kl) - kl - 1
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


def grpo_loss_forward(
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


@triton.jit
def grpo_loss_bwd_kernel(
    DLOSS,
    LOGP,
    OLD_LOGP,
    REF_LOGP,
    OUT_LOGP,
    ADVANTAGES,
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

    log_probs_diff = d_log_probs - d_log_probs_old
    log_probs_diff1 = d_log_probs_ref - d_log_probs
    exp = tl.exp(log_probs_diff)
    exp1 = tl.exp(log_probs_diff1)
    clamp = tl.clamp(exp, 0.8, 1.2)

    adv_off = tl.arange(0, T)
    d_advantages = tl.load(ADVANTAGES + adv_off)
    adv_expand = tl.expand_dims(d_advantages, 1)

    # load data and calculate
    loss_off = tl.arange(0, T)
    d_loss = tl.load(DLOSS + loss_off).to(tl.float32)
    d_loss = tl.expand_dims(d_loss, 1)
    loss_expand = tl.broadcast_to(d_loss, (T, V))
    neg2 = -loss_expand
    mul7 = loss_expand * exp1
    add1 = neg2 + mul7
    neg3 = -add1

    sum_loss = tl.sum(d_loss)
    sum_loss = tl.broadcast_to(sum_loss, (T, V))

    zeros = tl.zeros((T, V), tl.float32)
    neg = -adv_expand
    mul = neg * exp
    mul1 = neg * clamp
    where = tl.where(mul == mul1, sum_loss / 2, sum_loss)
    gt = tl.where(mul > mul1, zeros, where)
    lt = tl.where(mul < mul1, zeros, where)

    mul9 = gt * neg
    mul10 = gt * clamp
    sum4 = tl.sum(mul10, axis=1)
    where2 = tl.where(exp >= 0.8 and exp <= 1.2, mul9, zeros)

    neg4 = -sum4
    neg4_expand = tl.expand_dims(neg4, 1)
    mul11 = -lt * adv_expand
    mul12 = lt * exp
    sum5 = tl.sum(mul12, axis=1)

    add2 = where2 + mul11
    neg5 = -sum5
    neg5_expand = tl.expand_dims(neg5, 1)
    add3 = neg4_expand + neg5_expand
    mul13 = add2 * exp
    neg6 = -mul13
    add4 = neg3 + mul13
    add5 = add1 + neg6

    # output loss value
    tl.store(OUT_LOGP +
             offs_probs_dim1[:, None] * V +
             offs_probs_dim2[None, :],
             add4)


def grpo_loss_backward(
    loss,
    log_probs,
    old_logprobs,
    ref_logprobs,
    out_logprobs,
    advantages,
    clip,
    B,
    T,
    V,
    BLOCK_SIZE_T,
):
    if ref_logprobs is None:
        ref_logprobs = log_probs.detach()
    beta = 1.0

    grid = lambda META: (T // BLOCK_SIZE_T,)
    grpo_loss_bwd_kernel[grid](loss, log_probs, old_logprobs, ref_logprobs, 
                                out_logprobs, advantages, clip, beta, B, T, V)
    return out_logprobs
