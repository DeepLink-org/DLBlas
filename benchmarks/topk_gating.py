import torch
import triton
import dlblas
import sys
# sys.path.insert(0,"/cpfs01/user/xuhaoran/202411top_gate/Triton/python/dlBLAS/tests/for_megatron_topk")
sys.path.insert(0,"/home/aigc/PRJ/Triton/python/dlBLAS/benchmarks/test_gating")
import time
from torch.profiler import profile, ProfilerActivity
device_ = torch.device("cuda")
def test():
    k, SeqLen, NumberExperts = 8, 4096, 64
    shape = (SeqLen, NumberExperts)
    logits_torch = torch.randn(shape, device=device_, dtype=torch.float32, requires_grad=True)
    capacity_factor: float = 1.0
    min_capacity: int = 2
    drop_policy = "position"
    pad_to_capacity = False
    enable_token_rearrange_opt = True

    with torch.no_grad():
        logits_triton = logits_torch.clone()

    logits_triton.requires_grad = True


    # 版本4: megatron实现
    from megatron_gating import megatron_topgating
    model_megatron = megatron_topgating
    output1_megatron_aux_loss, output2_megatron_probs, output3_megatron_indices, output4_megatron_masked_gates, output5_megatron_capacity_mask = model_megatron(logits_torch, k, capacity_factor, drop_policy, pad_to_capacity)
    
    
    # 版本3 :triton实现
    model_triton = dlblas.topk_gating
    output1_triton_aux_loss, output2_triton_probs, output3_triton_indices, output4_triton_masked_gates, output5_triton_capacity_mask = model_triton(
        logits_triton, k, capacity_factor, (drop_policy == "probs"), min_capacity, False
    )

    # # 版本5 :triton_bwd实现
    # model_triton_bwd = dlblas._topk_gating_bwd
    

    assert torch.allclose(output1_megatron_aux_loss, output1_triton_aux_loss, rtol=1e-4, atol=1e-6)
    assert torch.allclose(output4_megatron_masked_gates, output4_triton_masked_gates, rtol=1e-4, atol=1e-6)
    assert torch.allclose(output5_megatron_capacity_mask.float(), output5_triton_capacity_mask, rtol=1e-4, atol=1e-6)
    assert torch.allclose(output2_megatron_probs, output2_triton_probs, rtol=1e-4, atol=1e-6)
    assert torch.allclose(output3_megatron_indices, output3_triton_indices, rtol=1e-4, atol=1e-6)
    
    
    # for backward
    dout_torch = torch.randn_like(output1_megatron_aux_loss)
    with torch.no_grad():
        dout_triton = dout_torch.clone()
    # print("dout_torch:", dout_torch)
    output1_megatron_aux_loss.backward(dout_torch, retain_graph=True)
    output1_triton_aux_loss.backward(dout_triton, retain_graph=True)
    # output1_triton_bwd_aux_loss = model_triton_bwd(tokens_per_expert, scores, dout_triton, k)

    # assert torch.allclose(logits_torch.grad, output1_triton_bwd_aux_loss, rtol=5e-3)
    assert torch.allclose(logits_torch.grad, logits_triton.grad, rtol=5e-3)

    # vary seq length for fixed head and batch=4
    configs = []

    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd", "bwd"],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"Experts{NumberExperts}-top{k}-gating-seqLen:{SeqLen}",
            args={"SeqLen": SeqLen},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_top2gating(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200
        shape = (SeqLen, NumberExperts)
        logits = torch.randn(shape, device=device, requires_grad=True)

        if "triton" in provider:
            if "fwd" == op:
                fn = lambda: model_triton(logits_triton, k, capacity_factor, (drop_policy == "probs"), min_capacity, False)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif "bwd" == op:
                loss, _, _, _, _ = model_triton(logits_triton, k, capacity_factor, (drop_policy == "probs"), min_capacity, False)
                bwd_fn = lambda: loss.backward(retain_graph=True)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        if "pytorch" in provider:
            if "fwd" == op:
                fn = lambda: model_megatron(logits_torch, k, capacity_factor, drop_policy, pad_to_capacity)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif "bwd" == op:
                loss, _, _, _, _ = model_megatron(logits_torch, k, capacity_factor, drop_policy, pad_to_capacity)
                bwd_fn = lambda: loss.backward(retain_graph=True)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        return ms

    bench_top2gating.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")

