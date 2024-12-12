import os
import argparse
import torch
import torch_mlu
from torch_mlu.utils.model_transfer import transfer
import triton
import triton.language as tl

import dlblas
import time 
from test_utils import check_output, test_latency_and_output
# from dlblas.kernels.camb.grouped_gemm import group_gemm_batch
from dlblas.kernels.camb import grouped_gemm

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-batch_sizes', nargs='+', type=int, default=[4])
    parser.add_argument('-z', type=int, default=4)
    parser.add_argument('-m', type=int, default=32)
    parser.add_argument('-n', type=int, default=32)
    parser.add_argument('-k', type=int, default=16)
    parser.add_argument('--bench',
                        default=False,
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def is_cuda():
    return torch.cuda.is_available()

# a [z*m, k]   b [z,n,k]
# batch_sizes = torch.tensor([m] * z)
def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)

class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_size should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return grouped_gemm.group_gemm_batch(a, b, batch_sizes, trans_a=False, trans_b = trans_b)
    
    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            #print("agrad:")
            #print(grad.shape)
            #print(b.shape)
            agrad = grouped_gemm.group_gemm_batch(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            #print("bgrad:")
            #print(lhs.shape)
            #print(rhs.shape)
            bgrad = grouped_gemm.group_gemm_batch(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)

        return agrad, bgrad, None, None

def gmm_op(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

# def main():
#     args = parse_args()
#     dtype = torch.float16
#     device = 'mlu'
#     batch_group_A = []
#     batch_group_B = []
#     batch_group_A_test = []
#     batch_group_B_test = []
#     for i in range(len(args.batch_sizes)):
#         group_m = [1024, 512, 256, 128]
#         group_n = [1024, 512, 256, 128]
#         group_k = [1024, 512, 256, 128]
#         group_A = []
#         group_B = []
#         assert len(group_m) == len(group_n)
#         assert len(group_n) == len(group_k)
#         group_size = len(group_m)
#         for i in range(group_size):
#             M = group_m[i]
#             N = group_n[i]
#             K = group_k[i]
#             A = torch.rand((M, K), device = device, dtype=torch.float16)
#             B = torch.rand((K, N), device = device, dtype=torch.float16)
#             group_A.append(A)
#             group_B.append(B)
#         # batch_group_A += group_A
#         # batch_group_B += group_B
#         # batch_group_A_test += group_A
#         # batch_group_B_test += group_B
        
#         batch_group_A.append(torch.stack(group_A, dim=0))
#         batch_group_B.append(torch.stack(group_A, dim=0))
#         batch_group_A_test.append(torch.stack(group_A, dim=0))
#         batch_group_B_test.append(torch.stack(group_A, dim=0))
        
#     # with torch.no_grad():
#     #     batch_group_A_test, batch_group_B_test = (batch_group_A.clone(), batch_group_B.clone())
#     # batch_group_A = torch.tensor(batch_group_A)
#     # batch_group_B = torch.tensor(batch_group_B)
#     # batch_group_A_test = torch.tensor(batch_group_A_test)
#     # batch_group_B_test = torch.tensor(batch_group_B_test)
#     batch_group_A = torch.stack(batch_group_A, dim=0)
#     batch_group_B = torch.stack(batch_group_B, dim=0)
#     batch_group_A_test = torch.stack(batch_group_A_test, dim=0)
#     batch_group_B_test = torch.stack(batch_group_B_test, dim=0)
    
#     batch_group_A.requires_grad = True
#     batch_group_B.required_grad = True
#     batch_group_A_test.requires_grad = True
#     batch_group_B_test.required_grad = True
#     # test
    
#     tri_out = GroupedGemm.apply(batch_group_A, batch_group_B)
#     ref_out = GroupedGemm_ref.apply(batch_group_A_test, batch_group_B_test)
#     check_output(tri_out, ref_out)
#     loss_tri = torch.sum(torch.mean(tri_out))
#     loss_tri.backward(retain_graph=True)
#     loss_ref = torch.sum(torch.mean(ref_out))
#     loss_ref.backward(retain_graph=True)
#     check_output(batch_group_A.grad, batch_group_A_test.grad)
#     check_output(batch_group_B.grad, batch_group_B_test.grad)

def main():
    # grouped_gemm.test()
    # return 

    args = parse_args()
    # z = args.z
    z = 3
    m = 1024
    k = 1024
    n = 1024
    trans_b = False
    device = "mlu"
    torch.manual_seed(0)
    a = torch.randn(z, m, k, dtype=torch.float16, device=device).view(-1, k)
    b = torch.randn(z, n, k, dtype=torch.float16, device=device) if trans_b else torch.randn(z, k, n, dtype=torch.float16, device=device)
    batch_sizes = torch.tensor([m] * z)

    a.requires_grad_(True)
    b.requires_grad_(True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out = gmm_op(a, b, batch_sizes, trans_b)
    expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)

    # check_output(out, expected_out, reduce_dim=k)
    assert torch.allclose(out, expected_out, atol=1e-2, rtol=0.001)
    # assert torch.allclose(out[8192:, :], expected_out[8192:, :], atol=1e-2, rtol=0.001)
    # assert torch.allclose(out[4096:, :], expected_out[4096:, :], atol=1e-2, rtol=0.001)
    
    # Check gradients.
    out.sum().backward()
    expected_out.sum().backward()
    check_output(a.grad, a_ref.grad)
    check_output(b.grad, b_ref.grad)
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-2, rtol=0.001)
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-2, rtol=0.001)
    device_ = "mlu"


    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd", "bwd"],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            ylabel="ms",
            plot_name=f"grouped gemm(z={z})",
            args={"z": z},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(z, op, provider, device=device_):
        warmup = 100
        rep = 200

        if "triton" in provider:
            if "fwd" == op:
                fn = lambda: GroupedGemm.apply(
                    a, b, batch_sizes, trans_b
                )
            elif "bwd" == op:
                c_triton = GroupedGemm.apply(
                    a, b, batch_sizes, trans_b
                )
                loss_tri = c_triton.sum()
                fn = lambda: loss_tri.backward(retain_graph=True)
            else:
                raise Exception()

        if "pytorch" in provider:
            if "fwd" == op:
                fn = lambda: gmm(a_ref, b_ref, batch_sizes, trans_b)
            elif "bwd" == op:
                c = gmm(a_ref, b_ref, batch_sizes, trans_b)
                loss_torch = c.sum()
                fn = lambda: loss_torch.backward(retain_graph=True)
            else:
                raise Exception()
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    bench_fn.run(show_plots=True, print_data=True)
    
if __name__ == '__main__':
    main()
