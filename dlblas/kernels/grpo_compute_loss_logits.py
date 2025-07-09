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
    new_logits_ptr,    # [BL, V] 新策略 logits
    ref_logits_ptr,    # [BL, V] 参考策略 logits
    old_logits_ptr,    # [BL, V] 旧策略 logits
    input_ids_ptr,     # [BL]    token 索引
    advantages_ptr,    # [BL]    优势函数
    mask_ptr,          # [BL]    掩码（有效 token）
    # 输出指针
    loss_ptr,          # [BL]    每个 token 的损失
    kl_ptr,            # [BL]    每个 token 的 KL 散度
    completion_length_ptr,  # [1]   平均生成长度
    mean_kl_ptr,       # [1]    平均 KL 散度
    # 常量参数
    BL: tl.constexpr,          # 总样本数 (B*T)
    V: tl.constexpr,           # 词汇表大小
    BLOCK_M: tl.constexpr,     # 每个线程块处理的行数
    max_completion_length: tl.constexpr,
    beta: tl.constexpr,
    delta: tl.constexpr,       # PPO 梯度裁剪阈值
    epsilon_low: tl.constexpr,
    epsilon_high: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    logit_scale_multiply: tl.constexpr,
    logit_scale_divide: tl.constexpr,
    logit_softcapping: tl.constexpr,
    loss_type: tl.constexpr,    # 0: grpo, 1: bnpo, 2: dr_grpo
):
    # 线程块 ID
    pid = tl.program_id(0)
    row0 = pid * BLOCK_M
    offs_t = row0 + tl.arange(0, BLOCK_M)  # 当前块负责的行索引
    m_t = offs_t < BL  # 防越界掩码

    # 1. 加载 input_ids
    ids = tl.load(input_ids_ptr + offs_t, mask=m_t, other=0)  # [BLOCK_M]
    ids = tl.where(ids >= 0, ids, 0)  # 确保列索引合法

    # 2. 加载 logits 并应用 logit 缩放/softcapping
    col_idx = tl.arange(0, V)[None, :]  # [1, V]
    ptr_new = new_logits_ptr + offs_t * V + ids
    ptr_ref = ref_logits_ptr + offs_t * V + ids
    ptr_old = old_logits_ptr + offs_t * V + ids
    # 加载 logits 并应用 logit 缩放
    new_logits_ptr_x = tl.load(new_logits_ptr + offs_t[:, None] * V + col_idx, mask=m_t[:, None], other=-float('inf'))
    ref_logits_ptr_x = tl.load(ref_logits_ptr + offs_t[:, None] * V + col_idx, mask=m_t[:, None], other=-float('inf'))
    old_logits_ptr_X = tl.load(old_logits_ptr + offs_t[:, None] * V + col_idx, mask=m_t[:, None], other=-float('inf'))
    new_x = tl.load(ptr_new, mask=m_t, other=-float('inf'))  # 结果是 [BLOCK_M]
    ref_x = tl.load(ptr_ref, mask=m_t, other=-float('inf'))  # 结果是 [BLOCK_M]
    old_x = tl.load(ptr_old, mask=m_t, other=-float('inf'))  # 结果是 [BLOCK_M]
    # 应用温度缩放
    if TEMPERATURE != 1.0:
        inv_temp = 1.0 / TEMPERATURE
        new_logits_ptr_x = new_logits_ptr_x * inv_temp
        ref_logits_ptr_x = ref_logits_ptr_x * inv_temp
        old_logits_ptr_X = old_logits_ptr_X * inv_temp
        new_x = new_x * inv_temp
        ref_x = ref_x * inv_temp
        old_x = old_x * inv_temp

    # 4. 计算 logsumexp（分两阶段）
  
    max_val1 = tl.max(new_logits_ptr_x, axis=1)  # [BLOCK_M]
    exp_logits1 = tl.exp(new_logits_ptr_x - max_val1[:, None])  # [BLOCK_M, V]
    sum_exp = tl.sum(exp_logits1, axis=1)  # [BLOCK_M]
    lse_new= max_val1 + tl.log(sum_exp)  # [BLOCK_M]
    
    max_val2 = tl.max(ref_logits_ptr_x, axis=1)  # [BLOCK_M]
    exp_logits2 = tl.exp(ref_logits_ptr_x - max_val2[:, None])  # [BLOCK_M, V]
    sum_exp2 = tl.sum(exp_logits2, axis=1)  # [BLOCK_M]
    lse_ref= max_val2 + tl.log(sum_exp2)  # [BLOCK_M]
    
    
    max_val3 = tl.max(old_logits_ptr_X, axis=1)  # [BLOCK_M]
    exp_logits3 = tl.exp(old_logits_ptr_X - max_val3[:, None])  # [BLOCK_M, V]
    sum_exp3= tl.sum(exp_logits3, axis=1)  # [BLOCK_M]
    lse_old= max_val3 + tl.log(sum_exp3)  # [BLOCK_M]
   

    # 5. 计算 log probability
    new_lp = new_x - lse_new
    ref_lp = ref_x - lse_ref
    old_lp = old_x - lse_old

    # 6. 计算 KL 散度（Reverse KL 近似）
    tmp = tl.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
    kl_i = tl.where(beta != 0.0, tmp, 0.0)  # 仅当 beta ≠ 0 时启用 KL 惩罚

    # 7. PPO-style 梯度裁剪

    coef1 = tl.exp(new_lp - old_lp)  # [BLOCK_M]
    # coef2 = tl.clamp(coef1, 1.0 - epsilon_low, 1.0 + epsilon_high)  # [BLOCK_M]
    coef2= tl.minimum(tl.maximum(coef1, 1.0 - epsilon_low), 1.0 + epsilon_high)
    # 8. 加载优势函数
    adv = tl.load(advantages_ptr + offs_t, mask=m_t, other=0.0)  # [BLOCK_M]

    # 9. 计算损失项
    if delta is not None:
        loss1  = tl.minimum(coef1, delta)* adv
    else:
        loss1 = coef1 * adv
    loss2 = coef2 * adv
    loss_i = -tl.minimum(loss1, loss2)  # [BLOCK_M]
    # 添加 KL 惩罚
    if beta != 0.0:
        loss_i += beta * kl_i

    # 10. 应用 mask（忽略 padding token）
    mask_val = tl.load(mask_ptr + offs_t, mask=m_t, other=0.0)  # [BLOCK_M]
    loss_i_masked = loss_i * mask_val
    kl_i_masked = kl_i * mask_val

    # 11. 写入输出
    tl.store(loss_ptr + offs_t, loss_i_masked, mask=m_t)
    tl.store(kl_ptr + offs_t, kl_i_masked, mask=m_t)

    # 12. 计算指标 放在pytorch中计算
    # completion_length = mask_val.sum()
    # mean_kl = (kl_i_masked).sum() / mask_val.sum()



def grpo_loss_triton(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
    logit_scale_multiply=0.0, logit_scale_divide=0.0, logit_softcapping=0.0
):
    BL, V = new_logits.shape
    BLOCK_M = 64
    grid= ( (BL + BLOCK_M - 1) // BLOCK_M, )
    loss = torch.empty(BL, device=new_logits.device)
    kl = torch.empty(BL, device=new_logits.device)
    completion_length = torch.tensor(0.0, device=new_logits.device)
    mean_kl = torch.tensor(0.0, device=new_logits.device)

    loss_type_code = {"grpo": 0, "bnpo": 1, "dr_grpo": 2}[loss_type]

    grpo_loss_kernel[grid](
        new_logits, ref_logits, old_logits, input_ids, advantages, mask,
        loss, kl, completion_length, mean_kl,
        BL=BL, V=V, BLOCK_M=BLOCK_M,
        max_completion_length=max_completion_length,
        beta=beta, epsilon_low=epsilon_low, epsilon_high=epsilon_high,delta=1,
        TEMPERATURE=temperature, logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide, logit_softcapping=logit_softcapping,
        loss_type=loss_type_code
    )

    # 损失聚合（Triton 只写入 per-token loss_i）
    mask = mask.to(torch.float32)
    loss_i = loss
    if loss_type == "grpo":
        loss_aggregated = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss_aggregated = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss_aggregated = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)

    with torch.inference_mode():
        completion_length = mask.sum().float().mean()
        mean_kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        mean_kl = mean_kl.mean()

    return loss_aggregated, completion_length, mean_kl, loss_i, kl

