# benchmark_grpo.py

import torch
import triton
from dlblas.kernel.grpo_compute_loss_logits import GRPO_Loss_Optimized, grpo_compute_loss_torch

def run_benchmark():
  
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return

    # 再次提醒：你的文件名可能是 bechmark_grpo.py，建议修正为 benchmark_grpo.py
    
    benchmark_shapes = [
        ("Small-Batch", 1, 1024, 32000),   
        ("Medium-Batch", 4, 512, 50257),  
        ("Large-Batch", 8, 512, 50257),  
        ("XLarge-Batch", 16, 256, 128000), 
    ]
    dtypes = [torch.float16, torch.float32]

    print("=" * 80)
    print("GRPO Loss Forward Pass Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    for name, B, S, V in benchmark_shapes:
        for dtype in dtypes:
            BL = B * S
            if dtype == torch.float32 and BL * V > 8 * 512 * 50257:
                 print(f"\nSkipping {name} BL={BL}, V={V}, dtype=float32 to avoid potential OOM.")
                 continue

            print(f"\n--- Benchmarking: {name} (B={B}, S={S}, V={V}, dtype={str(dtype).split('.')[-1]}) ---")
            
            # 生成输入数据
            new_logits = torch.randn((BL, V), device='cuda', dtype=dtype)
            ref_logits = torch.randn((BL, V), device='cuda', dtype=dtype)
            old_logits = torch.randn((BL, V), device='cuda', dtype=dtype)
            input_ids = torch.randint(0, V, (BL,), device='cuda')
            advantages = torch.randn((BL,), device='cuda', dtype=dtype)
            mask = torch.randint(0, 2, (BL,), device='cuda').bool()
            kwargs = {
                "beta": 0.1, "loss_type": "grpo", "epsilon_low": 0.2, "epsilon_high": 0.2,
                "max_completion_length": 8192, "delta": 2.7, "temperature": 1.0,
            }

            # 定义要测试的函数
            torch_fn = lambda: grpo_compute_loss_torch(new_logits, ref_logits, old_logits, input_ids, advantages, mask, **kwargs)
            triton_fn = lambda: GRPO_Loss_Optimized.apply(
                new_logits, ref_logits, old_logits, input_ids, advantages, mask,
                kwargs["beta"], kwargs["loss_type"], kwargs["epsilon_low"], kwargs["epsilon_high"],
                kwargs["max_completion_length"], kwargs["delta"], kwargs["temperature"]
            )

          
            print("Benchmarking PyTorch implementation...")
            torch_latency_ms = triton.testing.do_bench(torch_fn, quantiles=[0.5, 0.2, 0.8], rep=100)[0]
            
            print("Benchmarking Triton implementation...")
            triton_latency_ms = triton.testing.do_bench(triton_fn, quantiles=[0.5, 0.2, 0.8], rep=100)[0]
            
            speedup = torch_latency_ms / triton_latency_ms
            
            # 吞吐量
            throughput_torch = BL / (torch_latency_ms / 1000)
            throughput_triton = BL / (triton_latency_ms / 1000)

            print(f"\nResults for {name}:")
            print(f"  PyTorch Latency : {torch_latency_ms:.4f} ms | Throughput: {throughput_torch:,.0f} tokens/sec")
            print(f"  Triton  Latency : {triton_latency_ms:.4f} ms | Throughput: {throughput_triton:,.0f} tokens/sec")
            print(f"  Speedup         : {speedup:.2f}x")

if __name__ == '__main__':
    run_benchmark()
