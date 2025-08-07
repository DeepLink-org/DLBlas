
import unittest
import torch
from  dlblas.kernels.grpo_compute_loss_logits import GRPO_Loss_Optimized, grpo_compute_loss_torch
from dlblas.utils.device_utils import infer_device

def run_full_verification():
    print("="*50)
    print("Running Full Forward & Backward Verification")
    print("="*50)

    # --- 设置测试参数 ---
    B, S, V = 2, 64, 2048
    BL = B * S
    DEVICE = infer_device()
    DTYPE = torch.float32
    SEED = 42
    
   
    kwargs = {
        "beta": 0.1,
        "loss_type": "grpo",
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "max_completion_length": 8192,
        "delta": 2.7,
        "temperature": 0.7,
    }

   
    def generate_inputs(requires_grad=False):
        torch.manual_seed(SEED)
        new_logits_3d = torch.randn((B, S, V), device=DEVICE, dtype=DTYPE)
        ref_logits_3d = torch.randn((B, S, V), device=DEVICE, dtype=DTYPE)
        old_logits_3d = torch.randn((B, S, V), device=DEVICE, dtype=DTYPE)
        new_logits_2d = new_logits_3d.view(BL, V).clone()
        ref_logits_2d = ref_logits_3d.view(BL, V).clone()
        old_logits_2d = old_logits_3d.view(BL, V).clone()
        if requires_grad:
            new_logits_3d.requires_grad_(True)
            new_logits_2d.requires_grad_(True)
        input_ids_3d = torch.randint(0, V, (B, S), device=DEVICE)
        mask_3d = torch.ones((B, S), device=DEVICE, dtype=torch.long)
        mask_3d[0, -5:] = 0
        input_ids_2d = input_ids_3d.view(BL)
        mask_2d = mask_3d.view(BL)
        advantages_3d = torch.randn((B,), device=DEVICE, dtype=DTYPE)
        advantages_2d = advantages_3d.unsqueeze(1).expand(-1, S).reshape(BL)
        return (new_logits_3d, ref_logits_3d, old_logits_3d, input_ids_3d, advantages_3d, mask_3d), \
               (new_logits_2d, ref_logits_2d, old_logits_2d, input_ids_2d, advantages_2d, mask_2d)

    # --- 运行 PyTorch 参考实现  ---
    print("\n--- Running PyTorch Reference Implementation ---")
    (new_logits_ref_3d, ref_logits_ref_3d, old_logits_ref_3d, ids_ref_3d, adv_ref_3d, mask_ref_3d) = \
        generate_inputs(requires_grad=True)[0]
    adv_ref_torch = adv_ref_3d.unsqueeze(1).expand(-1, S)
    loss_ref, _, _, _, _ = grpo_compute_loss_torch(
        new_logits=new_logits_ref_3d, ref_logits=ref_logits_ref_3d, old_logits=old_logits_ref_3d,
        input_ids=ids_ref_3d, advantages=adv_ref_torch, mask=mask_ref_3d, **kwargs
    )
    loss_ref.backward()
    grad_ref = new_logits_ref_3d.grad.clone().view(BL, V)
    print(f"PyTorch Loss: {loss_ref.item()}")

    # ---  运行 Triton 实现 (修正调用方式) ---
    print("\n--- Running Triton Custom Kernel Implementation ---")
    (new_logits_triton, ref_logits_triton, old_logits_triton, ids_triton, adv_triton, mask_triton) = \
        generate_inputs(requires_grad=True)[1]
    
    # 3: 按位置传递所有参数给 .apply
    loss_triton = GRPO_Loss_Optimized.apply(
        new_logits_triton, ref_logits_triton, old_logits_triton,
        ids_triton, adv_triton, mask_triton,
        kwargs["beta"],
        kwargs["loss_type"],
        kwargs["epsilon_low"],
        kwargs["epsilon_high"],
        kwargs["max_completion_length"],
        kwargs["delta"],
        kwargs["temperature"]
    )
    loss_triton.backward()
    grad_triton = new_logits_triton.grad.clone()
    print(f"Triton Loss:  {loss_triton.item()}")

    # ---  比较结果  ---
    print("\n--- Comparing Results ---")
    loss_check = torch.allclose(loss_ref, loss_triton, atol=1e-5, rtol=1e-4)
    print(f"Forward Pass (Loss) Correct: {loss_check}")
    if not loss_check:
        print(f"  Max absolute difference: {torch.max(torch.abs(loss_ref - loss_triton))}")
    grad_check = torch.allclose(grad_triton, grad_ref, atol=1e-5, rtol=1e-4)
    print(f"Backward Pass (Gradient) Correct: {grad_check}")
    if not grad_check:
        abs_diff = torch.abs(grad_triton - grad_ref)
        rel_diff = abs_diff / torch.abs(grad_ref).clamp(min=1e-8)
        print(f"  Max absolute difference: {torch.max(abs_diff)}")
        print(f"  Max relative difference: {torch.max(rel_diff)}")
        max_diff_idx = torch.argmax(abs_diff)
        print(f"  Location of max diff: {max_diff_idx}")
        print(f"  Triton grad at max diff: {grad_triton.flatten()[max_diff_idx]}")
        print(f"  PyTorch grad at max diff: {grad_ref.flatten()[max_diff_idx]}")

run_full_verification()
