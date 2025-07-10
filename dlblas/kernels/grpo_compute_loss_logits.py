import torch
import triton
import triton.language as tl

from torch.cuda.amp import autocast


import triton
import triton.language as tl
#reference:  https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/rl_replacements.py
@triton.jit
def grpo_loss_triton_kernel(
    # 输入指针
    new_logits_ptr, ref_logits_ptr, old_logits_ptr,
    input_ids_ptr, advantages_ptr, mask_ptr,
    # 输出指针
    loss_ptr, kl_ptr,
    # 常量参数
    BL: tl.constexpr, V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
    beta: tl.constexpr, delta: tl.constexpr,
    epsilon_low: tl.constexpr, epsilon_high: tl.constexpr,
    TEMPERATURE: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * BLOCK_M
    offs_t = row0 + tl.arange(0, BLOCK_M)  # [BLOCK_M], 负责的行索引
    m_t = offs_t < BL  # [BLOCK_M], 行掩码

 
    m_new, m_ref, m_old = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32), tl.full([BLOCK_M], -float('inf'), dtype=tl.float32), tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    s_new, s_ref, s_old = tl.zeros([BLOCK_M], dtype=tl.float32), tl.zeros([BLOCK_M], dtype=tl.float32), tl.zeros([BLOCK_M], dtype=tl.float32)

    # 内部循环，分块处理每一行
    for v_start in range(0, V, BLOCK_V):
        # 计算当前块的列偏移量
        v_offs = v_start + tl.arange(0, BLOCK_V)  # [BLOCK_V]
        v_mask = v_offs < V  # [BLOCK_V], 列掩码

        # --- 为 new_logits 更新 logsumexp ---
        # 加载一个 logits 块: [BLOCK_M, BLOCK_V]
        new_chunk = tl.load(new_logits_ptr + offs_t[:, None] * V + v_offs[None, :], mask=m_t[:, None] & v_mask[None, :], other=-float('inf'))
        if TEMPERATURE != 1.0: new_chunk /= TEMPERATURE
        
        # 在块内找到最大值
        chunk_m_new = tl.max(new_chunk, axis=1)
        # 更新全局最大值
        new_m_new = tl.maximum(m_new, chunk_m_new)
        # 更新 sum (这是数值稳定的关键步骤)
        s_new = s_new * tl.exp(m_new - new_m_new) + tl.sum(tl.exp(new_chunk - new_m_new[:, None]), axis=1)
        m_new = new_m_new

        # --- 为 ref_logits 更新 logsumexp ---
        ref_chunk = tl.load(ref_logits_ptr + offs_t[:, None] * V + v_offs[None, :], mask=m_t[:, None] & v_mask[None, :], other=-float('inf'))
        if TEMPERATURE != 1.0: ref_chunk /= TEMPERATURE
        chunk_m_ref = tl.max(ref_chunk, axis=1)
        new_m_ref = tl.maximum(m_ref, chunk_m_ref)
        s_ref = s_ref * tl.exp(m_ref - new_m_ref) + tl.sum(tl.exp(ref_chunk - new_m_ref[:, None]), axis=1)
        m_ref = new_m_ref

        # --- 为 old_logits 更新 logsumexp ---
        old_chunk = tl.load(old_logits_ptr + offs_t[:, None] * V + v_offs[None, :], mask=m_t[:, None] & v_mask[None, :], other=-float('inf'))
        if TEMPERATURE != 1.0: old_chunk /= TEMPERATURE
        chunk_m_old = tl.max(old_chunk, axis=1)
        new_m_old = tl.maximum(m_old, chunk_m_old)
        s_old = s_old * tl.exp(m_old - new_m_old) + tl.sum(tl.exp(old_chunk - new_m_old[:, None]), axis=1)
        m_old = new_m_old

    # 循环结束后，完成 logsumexp 的计算
    lse_new = m_new + tl.log(s_new)
    lse_ref = m_ref + tl.log(s_ref)
    lse_old = m_old + tl.log(s_old)


    ids = tl.load(input_ids_ptr + offs_t, mask=m_t, other=0)
    ids = tl.where(ids >= 0, ids, 0)

    new_x = tl.load(new_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))
    ref_x = tl.load(ref_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))
    old_x = tl.load(old_logits_ptr + offs_t * V + ids, mask=m_t, other=-float('inf'))

    if TEMPERATURE != 1.0:
        inv_temp = 1.0 / TEMPERATURE
        new_x, ref_x, old_x = new_x * inv_temp, ref_x * inv_temp, old_x * inv_temp

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


def grpo_loss_triton(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
):
    BL, V = new_logits.shape
    
    # 定义块大小
    BLOCK_M = 64  
    BLOCK_V = 2048 
    
    grid = ((BL + BLOCK_M - 1) // BLOCK_M, )
    
    loss_i = torch.empty(BL, device=new_logits.device, dtype=torch.float32)
    kl_i = torch.empty(BL, device=new_logits.device, dtype=torch.float32)

    delta_kernel = delta if delta is not None else -1.0

    grpo_loss_triton_kernel[grid](
        new_logits, ref_logits, old_logits, input_ids, advantages, mask,
        loss_i, kl_i,
        BL=BL, V=V,
        BLOCK_M=BLOCK_M, BLOCK_V=BLOCK_V,
        beta=beta, epsilon_low=epsilon_low, epsilon_high=epsilon_high, delta=delta_kernel,
        TEMPERATURE=temperature,
    )

    # 后续的聚合逻辑与之前相同
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



def grpo_compute_loss_torch(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
):
   
    new_logits_fp32, ref_logits_fp32, old_logits_fp32 = new_logits.to(torch.float32), ref_logits.to(torch.float32), old_logits.to(torch.float32)

    if temperature != 1.0:
        new_logits_fp32, ref_logits_fp32, old_logits_fp32 = new_logits_fp32 / temperature, ref_logits_fp32 / temperature, old_logits_fp32 / temperature

    input_ids_expanded = input_ids.unsqueeze(-1)
    new_x = torch.gather(new_logits_fp32, dim=-1, index=input_ids_expanded).squeeze(-1)
    ref_x = torch.gather(ref_logits_fp32, dim=-1, index=input_ids_expanded).squeeze(-1)
    old_x = torch.gather(old_logits_fp32, dim=-1, index=input_ids_expanded).squeeze(-1)

    lse_new, lse_ref, lse_old = torch.logsumexp(new_logits_fp32, dim=-1), torch.logsumexp(ref_logits_fp32, dim=-1), torch.logsumexp(old_logits_fp32, dim=-1)
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
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    with torch.inference_mode():
        completion_length = mask_float.sum()
        mean_kl = (kl_i * mask_float).sum() / mask_float.sum().clamp(min=1.0)

    return loss, completion_length, mean_kl, masked_loss_i, kl_i * mask_float
