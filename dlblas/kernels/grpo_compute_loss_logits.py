import torch
import triton
import triton.language as tl

from torch.cuda.amp import autocast


import triton
import triton.language as tl
#reference:  https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/rl_replacements.py

@triton.jit
def _update_logsumexp(chunk, m, s):
    chunk_m = tl.max(chunk, 0)
    new_m = tl.maximum(m, chunk_m)
    s_rescaled = s * tl.exp(m - new_m)
    chunk_s = tl.sum(tl.exp(chunk - new_m), 0)
    s = s_rescaled + chunk_s
    m = new_m
    return m, s

@triton.jit
def grpo_loss_kernel(
    # Pointers
    new_logits_ptr, ref_logits_ptr, old_logits_ptr,
    input_ids_ptr, advantages_ptr, mask_ptr,
    loss_i_ptr, kl_i_ptr,
    # Dimensions
    BL, V,
    # Hyperparameters
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    delta: float,
    temperature: float,
    # Meta-parameters
    BLOCK_SIZE_V: tl.constexpr,
    HAS_DELTA: tl.constexpr, # 
):
   
    pid = tl.program_id(axis=0)
    mask = tl.load(mask_ptr + pid)
    if not mask:
        tl.store(loss_i_ptr + pid, 0.0)
        tl.store(kl_i_ptr + pid, 0.0)
        return

    new_ptr = new_logits_ptr + pid * V
    ref_ptr = ref_logits_ptr + pid * V
    old_ptr = old_logits_ptr + pid * V

    # --- 1. 串行化 LogSumExp 计算 (包含温度缩放) ---
    # new_logits
    m_val, s_val = -float('inf'), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(new_ptr + offsets, mask=v_mask, other=-float('inf'))
        chunk = chunk / temperature # 应用温度
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_new = m_val + tl.log(s_val)

    # ref_logits
    m_val, s_val = -float('inf'), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(ref_ptr + offsets, mask=v_mask, other=-float('inf'))
        chunk = chunk / temperature # 应用温度
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_ref = m_val + tl.log(s_val)

    # old_logits
    m_val, s_val = -float('inf'), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(old_ptr + offsets, mask=v_mask, other=-float('inf'))
        chunk = chunk / temperature # 应用温度
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_old = m_val + tl.log(s_val)

    # --- 2. Gather 并计算 Log-Probabilities  ---
    input_ids = tl.load(input_ids_ptr + pid)
    new_x = (tl.load(new_ptr + input_ids)) / temperature
    ref_x = (tl.load(ref_ptr + input_ids)) / temperature
    old_x = (tl.load(old_ptr + input_ids)) / temperature

    new_lp = new_x - lse_new
    ref_lp = ref_x - lse_ref
    old_lp = old_x - lse_old

    # --- 3. 计算 KL 散度 ---
    kl_i = 0.0
    if beta != 0.0:
        # kl(p_ref || p_new) = sum(p_ref * (log(p_ref) - log(p_new)))
        # 在单点采样下近似为 exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
        kl_i = tl.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0

   
    advantages = tl.load(advantages_ptr + pid)
    
   
    coef1 = tl.exp(new_lp - old_lp)
    
    # loss1 是带 delta 裁剪的损失项
    loss1_coef = coef1
    if HAS_DELTA:
        loss1_coef = tl.minimum(coef1, delta) 
    loss1 = loss1_coef * advantages
    
    
    coef2 = tl.clamp(coef1, 1.0 - epsilon_low, 1.0 + epsilon_high)
    loss2 = coef2 * advantages
    
    # 最终的 per-token loss
    loss_i = -tl.minimum(loss1, loss2)
    if beta != 0.0:
        loss_i += beta * kl_i

    # --- 4. 写回结果 ---
    tl.store(loss_i_ptr + pid, loss_i)
    tl.store(kl_i_ptr + pid, kl_i)



def grpo_compute_loss_torch(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=None, temperature=1.0,
):
    
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
    new_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    old_logits: torch.Tensor,
    input_ids: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "grpo",
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    max_completion_length: int = 8192,
    delta: float = None,
    temperature: float = 1.0,
):
 
    device = new_logits.device
    
    
    new_logits, ref_logits, old_logits, advantages = new_logits.to(torch.float32), ref_logits.to(torch.float32), old_logits.to(torch.float32), advantages.to(torch.float32)

    BL, V = new_logits.shape
    
    loss_i = torch.empty_like(advantages, device=device)
    kl_i = torch.empty_like(advantages, device=device)

    grid = (BL,)
    BLOCK_SIZE_V = 1024 # 可调超参数

   
    HAS_DELTA = (delta is not None)
   
    delta_val = delta if HAS_DELTA else 0.0 
    grpo_kernel[grid](
        new_logits, ref_logits, old_logits,
        input_ids, advantages, mask,
        loss_i, kl_i,
        BL, V,
        beta=beta,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        delta=delta_val,
        temperature=temperature,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        HAS_DELTA=HAS_DELTA,
    )

  
    mask_float = mask.to(torch.float32)
    masked_loss_i = loss_i * mask_float
    masked_kl_i = kl_i * mask_float # PyTorch版本返回的是掩码后的KL

   
    if loss_type == "grpo" or loss_type == "bnpo":
        # .clamp(min=1.0) 防止除以零
        loss = masked_loss_i.sum() / mask_float.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        # .size(0) 是批次大小 BL
        loss = masked_loss_i.sum() / (loss_i.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

   
    completion_length = mask_float.sum()
    mean_kl = masked_kl_i.sum() / completion_length.clamp(min=1.0)


    return loss, completion_length, mean_kl, masked_loss_i, masked_kl_i

