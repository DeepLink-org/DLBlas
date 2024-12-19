import time
from typing import List
import torch
import torch_mlu
from torch_mlu.utils.model_transfer import transfer

from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import triton
import dlblas
from dlblas.utils.device_utils import get_idle_device

from functorch.compile import aot_module, make_boxed_func, aot_function
from torch._dynamo.backends.common import aot_autograd

device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)

def topk_norm(topk_weight):
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight_norm = topk_weight / denominator
    return topk_weight_norm

def test():
    seq_len = 4096
    q_head_dim = 64
    topk_weight = torch.abs(torch.randn(size=(seq_len, q_head_dim), dtype=torch.bfloat16, device=device_))

    with torch.no_grad():
        topk_weight_tri = (
            topk_weight.clone()
        )
    topk_weight.requires_grad = True
    topk_weight_tri.requires_grad = True

    topk_weight_norm = topk_norm(topk_weight)

    from dlblas.kernels.camb.topk_megatron_part2 import topk_part2
    topk_weight_norm_tri = topk_part2.apply(topk_weight_tri)

    print(f"out max diff: {(topk_weight_norm - topk_weight_norm_tri).abs().max().item()}")
    assert torch.allclose(topk_weight_norm, topk_weight_norm_tri,atol=1e-3,rtol=1e-3)
        
    loss_torch = torch.sum(torch.mean(topk_weight_norm))
    loss_torch.backward(retain_graph=True)
    loss_tri = torch.sum(torch.mean(topk_weight_norm_tri))
    loss_tri.backward(retain_graph=True)
    print(f"out max diff: {(topk_weight.grad - topk_weight_tri.grad).abs().max().item()}")



    assert torch.allclose(topk_weight.grad, topk_weight_tri.grad,atol=1e-3,rtol=1e-3)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd", "bwd"],
            line_arg="provider",
            line_vals=["pytorch", "triton"],
            line_names=[ "PyTorch","Triton"],
            ylabel="ms",
            plot_name=f"yarn_ROPE",
            args={"q_head_dim": q_head_dim},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(q_head_dim, op, provider, device=device_):
        warmup = 100
        rep = 200

        if "triton" in provider:
            if "fwd" == op:
                fn = lambda: topk_part2.apply(topk_weight_tri)
            elif "bwd" == op:
                topk_weight_norm_tri = topk_part2.apply(topk_weight_tri)
                loss_tri = torch.sum(topk_weight_norm_tri)
                fn = lambda: loss_tri.backward(retain_graph=True)
        if "pytorch" in provider:
            if "fwd" == op:
                fn = lambda: topk_norm(topk_weight)
            elif "bwd" == op:
                topk_weight_norm = topk_norm(topk_weight)
                loss_torch = torch.sum(topk_weight_norm)
                fn = lambda: loss_torch.backward(retain_graph=True)
            else:
                raise Exception()
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
