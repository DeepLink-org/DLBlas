import torch
from from dlblas.kernels.grpo_compute_loss_logits import grpo_loss_triton

def grpo_compute_loss_torch(
    new_logits, ref_logits, old_logits, input_ids, advantages, mask,
    beta=0.1, loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    max_completion_length=8192, delta=1, temperature=1.0,
    logit_scale_multiply=0.0, logit_scale_divide=0.0, logit_softcapping=0.0
):

    new_logits = new_logits.to(torch.float32)
    ref_logits = ref_logits.to(torch.float32)
    if old_logits is not None:
        old_logits = old_logits.to(torch.float32)

    # 温度缩放
    if temperature != 1.0:
        new_logits = new_logits / temperature
        ref_logits = ref_logits / temperature
        if old_logits is not None:
            old_logits = old_logits / temperature

    # 提取 log probability
    input_ids = input_ids.unsqueeze(-1)
    new_x = torch.gather(new_logits, dim=-1, index=input_ids).squeeze(-1)
    ref_x = torch.gather(ref_logits, dim=-1, index=input_ids).squeeze(-1)
    if old_logits is not None:
        old_x = torch.gather(old_logits, dim=-1, index=input_ids).squeeze(-1)
    else:
        old_x = None

    # 计算 logsumexp
    lse_new = torch.logsumexp(new_logits, dim=-1)
    lse_ref = torch.logsumexp(ref_logits, dim=-1)
    if old_logits is not None:
        lse_old = torch.logsumexp(old_logits, dim=-1)
    else:
        lse_old = None

    # 计算 log probability
    new_lp = new_x - lse_new
    ref_lp = ref_x - lse_ref
    if old_logits is not None:
        old_lp = old_x - lse_old
    else:
        old_lp = None

    # KL 散度计算
    if beta != 0.0:
        kl_i = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
    else:
        kl_i = torch.zeros_like(ref_lp)

    # 梯度裁剪
    if old_logits is not None:
        coef1 = torch.exp(new_lp - old_lp)
    else:
        coef1 = torch.exp(new_lp - new_lp.detach())

    coef2 = torch.clamp(coef1, 1 - epsilon_low, 1 + epsilon_high)

    if delta is not None:
        loss1 = torch.clamp(coef1, max=delta) * advantages
    else:
        loss1 = coef1 * advantages

    loss2 = coef2 * advantages
    loss_i = -torch.min(loss1, loss2)

    if beta != 0.0:
        loss_i += beta * kl_i

    # 损失聚合
    mask = mask.to(torch.float32)
    if loss_type == "grpo":
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    with torch.inference_mode():
        completion_length = mask.sum().float().mean()
        mean_kl = (kl_i * mask).sum() / mask.sum().clamp(min=1.0)
        mean_kl = mean_kl.mean()

    return loss, completion_length, mean_kl, loss_i, kl_i, new_lp, ref_lp, old_lp

def test_grpo_loss():
    # 设置随机种子
    torch.manual_seed(0)

    # 配置参数
    BL = 1024   # 总 token 数
    V = 512        # 词汇表大小
    temperature = 2.0
    beta = 0.1
    epsilon_low = 0.2
    epsilon_high = 0.2
    loss_type = "grpo"
    logit_scale_multiply = 0.5
    logit_scale_divide = 2.0
    logit_softcapping = 10.0

    # 构造输入数据
    new_logits = torch.randn((BL, V), device='cuda', requires_grad=True)
    ref_logits = torch.randn((BL, V), device='cuda')
    old_logits = torch.randn((BL, V), device='cuda')
    input_ids = torch.randint(0, V, (BL,), device='cuda')
    advantages = torch.randn((BL,), device='cuda')
    mask = torch.randint(0, 2, (BL,), device='cuda').bool()

    # PyTorch 计算
    loss_pt, completion_pt, mean_kl_pt, loss_i_pt, kl_i_pt, new_lp_pt, ref_lp_pt, old_lp_pt = grpo_compute_loss_torch(
        new_logits, ref_logits, old_logits, input_ids, advantages, mask,
        beta=beta, loss_type=loss_type,
        epsilon_low=epsilon_low, epsilon_high=epsilon_high,
        temperature=temperature,
        logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide,
        logit_softcapping=logit_softcapping
    )

    # Triton 计算
    loss_tr, completion_tr, mean_kl_tr, loss_i_tr, kl_i_tr = grpo_loss_triton(
        new_logits, ref_logits, old_logits, input_ids, advantages, mask,
        beta=beta, loss_type=loss_type,
        epsilon_low=epsilon_low, epsilon_high=epsilon_high,
        temperature=temperature,
        logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide,
        logit_softcapping=logit_softcapping
    )

    # ----------------------------
    # 比较输出
    # ----------------------------
    print("Loss (Torch):", loss_pt.item())
    print("Loss (Triton):", loss_tr.item())
    assert torch.allclose(loss_pt, loss_tr, atol=1e-3, rtol=1e-3), "Loss 不一致"

    print("Mean KL (Torch):", mean_kl_pt.item())
    print("Mean KL (Triton):", mean_kl_tr.item())
    assert torch.allclose(mean_kl_pt, mean_kl_tr, atol=1e-3, rtol=1e-3), "Mean KL 不一致"

    print("Completion Length (Torch):", completion_pt.item())
    print("Completion Length (Triton):", completion_tr.item())
    assert torch.allclose(completion_pt, completion_tr, atol=1e-3, rtol=1e-3), "Completion Length 不一致"
    
    float_mask = mask.to(torch.float32)
    assert torch.allclose(loss_i_pt * float_mask, loss_i_tr, atol=1e-3, rtol=1e-3), "Masked Loss_i 不一致"

    # 对 kl_i 做同样的操作
    assert torch.allclose(kl_i_pt * float_mask, kl_i_tr, atol=1e-3, rtol=1e-3), "Masked KL_i 不一致"

    print("✅ 数值一致性验证通过！")
