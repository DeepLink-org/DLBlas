import torch
import triton
import triton.language as tl

from torch.cuda.amp import autocast


import triton
import triton.language as tl
#reference:  https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/rl_replacements.py
@triton.jit
def grpo_loss_kernel(
    # 输入指针
    new_logits_ptr, ref_logits_ptr, old_logits_ptr,
    input_ids_ptr, advantages_ptr, mask_ptr,
    # 预计算的 logsumexp 指针
    lse_new_ptr, lse_ref_ptr, lse_old_ptr,
    # 输出指针
    loss_ptr, kl_ptr,
    # 常量参数
    BL: tl.constexpr,
    V: tl.constexpr,  # [MODIFIED] V 必须作为 constexpr 显式传入
    BLOCK_M: tl.constexpr,
    beta: tl.constexpr, delta: tl.constexpr,
    epsilon_low: tl.constexpr, epsilon_high: tl.constexpr,
    TEMPERATURE: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * BLOCK_M
    offs_t = row0 + tl.arange(0, BLOCK_M)
    m_t = offs_t < BL

    ids = tl.load(input_ids_ptr + offs_t, mask=m_t, other=0)
    ids = tl.where(ids >= 0, ids, 0)

    # [REMOVED] 不再从指针推断 V，因为它不可靠
    # V = new_logits_ptr.shape[1]

    # 只加载目标 token 的 logit
    new_x = tl.load(new_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))
    ref_x = tl.load(ref_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))
    old_x = tl.load(old_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))

    if TEMPERATURE != 1.0:
        inv_temp = 1.0 / TEMPERATURE
        new_x, ref_x, old_x = new_x * inv_temp, ref_x * inv_temp, old_x * inv_temp

    # 直接加载预先计算好的 logsumexp 值
    lse_new = tl.load(lse_new_ptr + offs_t, mask=m_t, other=0.0)
    lse_ref = tl.load(lse_ref_ptr + offs_t, mask=m_t, other=0.0)
    lse_old = tl.load(lse_old_ptr + offs_t, mask=m_t, other=0.0)

    # 后续逻辑保持不变
    new_lp, ref_lp, old_lp = new_x - lse_new, ref_x - lse_ref, old_x - lse_old

    tmp_kl = tl.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
    kl_i = tl.where(beta != 0.0, tmp_kl, 0.0)

    coef1 = tl.exp(new_lp - old_lp)
    coef2 = tl.minimum(tl.maximum(coef1, 1.0 - epsilon_low), 1.0 + epsilon_high)
    
    adv = tl.load(advantages_ptr + offs_t, mask=m_t, other=0.0)

    if delta != -1.0:
        loss1 = tl.minimum(coef1, delta) * adv
    else:
        loss1 = coef1 * adv
    loss2 = coef2 * adv
    loss_i = -tl.minimum(loss1, loss2)

    if beta != 0.0:
        loss_i += beta * kl_i

    mask_val = tl.load(mask_ptr + offs_t, mask=m_t, other=0.0)
    loss_i_masked = loss_i * mask_val
    kl_i_masked = kl_i * mask_val

    tl.store(loss_ptr + offs_t, loss_i_masked, mask=m_t)
    tl.store(kl_ptr + offs_t, kl_i_masked, mask=m_t)


def grpo_compute_loss_torch(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
):
    # PyTorch 参考实现保持不变
    new_logits, ref_logits, old_logits = new_logits.to(torch.float32), ref_logits.to(torch.float32), old_logits.to(torch.float32)

    if temperature != 1.0:
        new_logits, ref_logits, old_logits = new_logits / temperature, ref_logits / temperature, old_logits / temperature

    input_ids_expanded = input_ids.unsqueeze(-1)
    new_x = torch.gather(new_logits, dim=-1, index=input_ids_expanded).squeeze(-1)
    ref_x = torch.gather(ref_logits, dim=-1, index=input_ids_expanded).squeeze(-1)
    old_x = torch.gather(old_logits, dim=-1, index=input_ids_expanded).squeeze(-1)

    lse_new, lse_ref, lse_old = torch.logsumexp(new_logits, dim=-1), torch.logsumexp(ref_logits, dim=-1), torch.logsumexp(old_logits, dim=-1)
    new_lp, ref_lp, old_lp = new_x - lse_new, ref_x - lse_ref, old_x - lse_old

    kl_i = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0 if beta != 0.0 else torch.zeros_like(ref_lp)

    coef1 = torch.exp(new_lp - old_lp)
    coef2 = torch.clamp(coef1, 1 - epsilon_low, 1 + epsilon_high)

    loss1 = torch.clamp(coef1, max=delta) * advantages if delta is not None else coef1 * advantages
    loss2 = coef2 * advantages
    loss_i = -torch.min(loss1, loss2)
    if beta != 0.0:
        loss_i += beta * kl_i

    mask_float = mask.to(torch.float32)
    masked_loss_i = loss_i * mask_float
    
    if loss_type == "grpo" or loss_type == "bnpo":
        loss = masked_loss_i.sum() / mask_float.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = masked_loss_i.sum() / (loss_i.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    with torch.inference_mode():
        completion_length = mask_float.sum()
        mean_kl = (kl_i * mask_float).sum() / mask_float.sum().clamp(min=1.0)

    return loss, completion_length, mean_kl, masked_loss_i, kl_i * mask_float


def grpo_loss_triton(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
):
    BL, V = new_logits.shape
    BLOCK_M = 64
    grid = ((BL + BLOCK_M - 1) // BLOCK_M, )
    
    loss_i = torch.empty(BL, device=new_logits.device, dtype=torch.float32)
    kl_i = torch.empty(BL, device=new_logits.device, dtype=torch.float32)

    delta_kernel = delta if delta is not None else -1.0

    # 计算 logsumexp
    new_logits_scaled = new_logits.to(torch.float32)
    ref_logits_scaled = ref_logits.to(torch.float32)
    old_logits_scaled = old_logits.to(torch.float32)
    
    if temperature != 1.0:
        new_logits_scaled, ref_logits_scaled, old_logits_scaled = new_logits_scaled / temperature, ref_logits_scaled / temperature, old_logits_scaled / temperature
    
    lse_new = torch.logsumexp(new_logits_scaled, dim=-1)
    lse_ref = torch.logsumexp(ref_logits_scaled, dim=-1)
    lse_old = torch.logsumexp(old_logits_scaled, dim=-1)

    grpo_loss_kernel[grid](
        new_logits, ref_logits, old_logits, input_ids, advantages, mask,
        lse_new, lse_ref, lse_old,
        loss_i, kl_i,
        BL=BL,
        V=V,  # [MODIFIED] 将 V 显式传入内核
        BLOCK_M=BLOCK_M,
        beta=beta, epsilon_low=epsilon_low, epsilon_high=epsilon_high, delta=delta_kernel,
        TEMPERATURE=temperature,
    )

    mask_float = mask.to(torch.float32)
    if loss_type == "grpo" or loss_type == "bnpo":
        loss_aggregated = loss_i.sum() / mask_float.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss_aggregated = loss_i.sum() / (loss_i.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    with torch.inference_mode():
        completion_length = mask_float.sum()
        mean_kl = kl_i.sum() / mask_float.sum().clamp(min=1.0)

    return loss_aggregated, completion_length, mean_kl, loss_i, kl_i
