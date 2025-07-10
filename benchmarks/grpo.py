# benchmark_grpo.py

import torch
import triton
from dlblas.kernel.grpo_compute_loss_logits import grpo_compute_loss_torch, grpo_loss_triton

def run_benchmark(BL, V, dtype):
    """
    运行给定 shape 和 dtype 的基准测试。
    
    参数:
    BL (int): 总 token 数 (Batch Size * Sequence Length)
    V (int): 词汇表大小
    dtype (torch.dtype): 数据类型 (通常是 torch.float16)
    """
    print("-" * 60)
    print(f"Benchmarking with BL={BL}, V={V}, dtype={dtype}")
    print("-" * 60)
    
    new_logits = torch.randn((BL, V), device='cuda', dtype=dtype).contiguous()
    ref_logits = torch.randn((BL, V), device='cuda', dtype=dtype).contiguous()
    old_logits = torch.randn((BL, V), device='cuda', dtype=dtype).contiguous()
    input_ids = torch.randint(0, V, (BL,), device='cuda').contiguous()
    advantages = torch.randn((BL,), device='cuda', dtype=dtype).contiguous()
    mask = torch.randint(0, 2, (BL,), device='cuda').bool().contiguous()

    
    kwargs = {
        'beta': 0.1,
        'loss_type': 'grpo',
        'epsilon_low': 0.2,
        'epsilon_high': 0.2,
        'delta': 5.0,
        'temperature': 1.0,
    }

  
    # PyTorch 基准测试
    torch_ms = triton.testing.do_bench(
        lambda: grpo_compute_loss_torch(
            new_logits, ref_logits, old_logits, input_ids, advantages, mask, **kwargs
        )
    )

    # Triton 基准测试
    triton_ms = triton.testing.do_bench(
        lambda: grpo_loss_triton(
            new_logits, ref_logits, old_logits, input_ids, advantages, mask, **kwargs
        )
    )

    # 3. 打印结果
    speedup = torch_ms / triton_ms
    print(f"PyTorch implementation: {torch_ms:.4f} ms")
    print(f"Triton implementation:  {triton_ms:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print("\n")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        exit()

  
    benchmark_configs = [
      # 场景 1: V = 32768 (2^15)
        (8 * 2048, 32768),  # 典型训练 (BL=16384)
        (2048, 32768),  # 极长上下文 (BL=32768)

        # 场景 2: V = 65536 (2^16)
        (1* 2048, 65536),   # 典型训练 (BL=16384)
       
    ]

    for bl, v in benchmark_configs:
        
        run_benchmark(BL=bl, V=v, dtype=torch.float16)