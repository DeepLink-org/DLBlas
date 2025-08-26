import torch
import triton
import dlblas
# op/dlBLAS/dlblas/kernels/moe.py
from dlblas.kernels.moe import silu_and_mul_masked_post_quant_fwd


# 新增PyTorch实现版本
def torch_silu_mul_quant(input, output_scale, quant_group):
    # 分解输入为gate和up两部分
    gate, up = torch.chunk(input, 2, dim=-1)
    
    # 计算SILU(gate) * up
    gate = torch.nn.functional.silu(gate)
    gate_up = gate * up
    
    # 分组量化
    B, T, H = gate_up.shape
    finfo = torch.finfo(torch.bfloat16)
    gate_up = gate_up.view(B, T, H // quant_group, quant_group)
    abs_max, _ = torch.max(torch.abs(gate_up), dim=-1, keepdim=True)
    scale = abs_max / torch.finfo(torch.bfloat16).max
    quantized = (gate_up / scale).clamp(finfo.min, finfo.min).to(torch.bfloat16)
    
    return quantized, scale

def benchmark_silu_mul_quant():
    # 配置参数
    expert_num = 8          # 专家数量
    token_per_expert = 1024 # 每个专家的token数
    hidden_size = 4096      # 隐藏层维度
    quant_group = 128       # 量化组大小
    
    # 生成测试数据
    input = torch.randn(
        expert_num, 
        token_per_expert, 
        hidden_size * 2,    # 输入最后一维是隐藏层的两倍
        device="npu", 
        dtype=torch.float16
    )
    output = torch.empty_like(input[:, :, :hidden_size], dtype=torch.bfloat16)
    output_scale = torch.empty(expert_num, token_per_expert, hidden_size//quant_group, dtype=torch.float32, device="npu")
    masked_m = torch.full((expert_num,), token_per_expert, device="npu", dtype=torch.int32)

    # 性能测试
    def bench_triton():
        silu_and_mul_masked_post_quant_fwd(input, output, output_scale, quant_group, masked_m)
        
    def bench_torch():
        torch_silu_mul_quant(input, output_scale, quant_group)
    
    # # 运行基准测试
    # triton_ms = triton.testing.do_bench(bench_triton, warmup=100, rep=500)
    # torch_ms = triton.testing.do_bench(bench_torch, warmup=100, rep=500)
    
    # # 计算加速比
    # speedup = torch_ms / triton_ms
    
    # print(f"\n=== 性能对比 ===")
    # print(f"Triton 耗时: {triton_ms:.4f} ms | Torch 耗时: {torch_ms:.4f} ms")
    # print(f"加速比: {speedup:.2f}x")

    
    
    import triton
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            ms = triton.testing.do_bench(lambda: silu_and_mul_masked_post_quant_fwd(input, output, output_scale, quant_group, masked_m), warmup=warmup, rep=rep)
        if 'pytorch' in provider:
            ms = triton.testing.do_bench(lambda: torch_silu_mul_quant(input, output_scale, quant_group), warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)



if __name__ == "__main__":
    benchmark_silu_mul_quant()

