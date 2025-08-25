# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import sys
import os
import torch
import triton
import torch.cuda.nvtx as nvtx
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 或者你想使用的GPU编号
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 或者你想使用的GPU编号
try:
    from triton_ops import permute_triton, unpermute_triton
    TRITON_AVAILABLE = True
except ImportError:
    print("Triton ops not found, skipping Triton tests.")
    TRITON_AVAILABLE = False
try:
  from grouped_gemm.ops import permute as permute_topK, unpermute as unpermute_topK
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
print(TRITON_AVAILABLE)
def permute(tokens, indices, expand_factor: int = 1, is_fp8=False):
    """Permute the tokens based on the indices.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token2expert indices tensor.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    expand_factor = indices.size(1)

    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, merge_factor: int = 1):
    """Unpermute the sorted tokens based on the indices.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        probs (torch.Tensor, optional): The probabilities tensor. Defaults to None.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    merge_factor = probs.size(1)

    if merge_factor > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // merge_factor
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // merge_factor
        assert (
            probs.size(1) == merge_factor
        ), f"probs size {probs.size()} merge_factor {merge_factor}"

    # unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, merge_factor, permuted_tokens.size(-1))

    if probs is not None:
        dtype = unpermuted_tokens.dtype
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.to(dtype)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_topK_test(
    dtype,
    num_token,
    num_expert,
    hidden_size,
    num_topK,
    PRINT,
    BENCHMARK):

    print(f"------------------{dtype} token:{num_token} hidden_size:{hidden_size} expert:{num_expert} topK:{num_topK}---------------------")

    is_fp8 = dtype in [torch.float8_e5m2, torch.float8_e4m3fn]

    permute_input = torch.rand((num_token, hidden_size), dtype=torch.float32).cuda()
    permute_input = permute_input.to(dtype)
    if is_fp8:
        permute_input = permute_input.half()
    permute_input.requires_grad_(True)

    if num_token > 0:
        indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])
    else:
        indices = torch.empty((num_token, num_topK))
    indices = indices.to(torch.int32).cuda()

    probs = torch.rand(num_token, num_topK).cuda()
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch (Ground Truth)
    #
    ###################################################################################################################################
    nvtx.range_push("PyTorch permute forward")
    permute_output, sorted_indices = permute(permute_input, indices, num_topK, is_fp8)
    nvtx.range_pop()

    permute_bwd_input = torch.rand_like(permute_output)

    nvtx.range_push("PyTorch permute backward")
    permute_output.backward(permute_bwd_input, retain_graph=True)
    nvtx.range_pop()
    pytorch_permute_grad = permute_input.grad.clone()

    unpermute_input = permute_output.detach().clone()
    unpermute_input.requires_grad_(True)
    unpermute_output = unpermute(unpermute_input, sorted_indices, probs=probs, merge_factor=num_topK)
    unpermute_bwd_input = torch.rand_like(unpermute_output)
    unpermute_output.backward(unpermute_bwd_input, retain_graph=True)

    pytorch_unpermute_act_grad = unpermute_input.grad.clone()
    pytorch_unpermute_probs_grad = probs.grad.clone()


    ###################################################################################################################################
    #
    # torch 
    #
    ###################################################################################################################################
    new_permute_input = permute_input.detach().to(dtype)
    new_permute_bwd_input = permute_bwd_input.detach().to(dtype)
    new_unpermute_bwd_input = unpermute_bwd_input.detach().to(dtype)
    new_permute_input.requires_grad_(True)

    new_permute_output, row_id_map = permute_topK(new_permute_input, indices)

    assert torch.allclose(permute_output.float(), new_permute_output.float())

    if PRINT:
        print("--------------row_id_map--------------")
        print(row_id_map)
        print("--------------new_permute_input--------------")
        print(new_permute_input)
        print("--------------new_permute_output--------------")
        print(new_permute_output)
        print("permute_input.grad", permute_input.grad)
        print("new_permute_input.grad", new_permute_input.grad)

    new_permute_output.backward(new_permute_bwd_input, retain_graph=True)

    if torch.allclose(permute_input.grad.float(), new_permute_input.grad.float()) == False:
        original_inputs = new_permute_input.grad.float().cpu().numpy().flatten()
        original_output = permute_input.grad.float().cpu().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"permute_topK bwd max error (mine vs pytorch): \t\t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print("permute_input.grad", permute_input.grad)
            print("new_permute_input.grad", new_permute_input.grad)
            # print("new_probs.grad =", new_probs.grad)

    new_probs = probs.detach()
    new_probs.requires_grad_(True)
    new_unpermute_input = new_permute_output.detach()
    new_unpermute_input.requires_grad_(True)

    # print("new_probs.grad =", new_probs.grad)
    new_unpermute_output = unpermute_topK(new_unpermute_input, row_id_map, new_probs)

    if torch.allclose(unpermute_output.float(), new_unpermute_output.float()) == False:
        original_inputs = unpermute_output.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_output.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print("unpermute_topK fwd cuda and torch max error(mine vs pytorch): ", abs(new_unpermute_output.float().cpu().detach().numpy().flatten() - unpermute_output.float().cpu().detach().numpy().flatten()).max())
        print(f"unpermute_topK fwd max error (mine vs pytorch): \t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print(unpermute_output)
            print(new_unpermute_output)

    new_unpermute_output.backward(new_unpermute_bwd_input, retain_graph=True)

    if torch.allclose(unpermute_input.grad.float(), new_unpermute_input.grad.float()) == False:
        original_inputs = unpermute_input.grad.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_input.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd act_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(new_unpermute_input.grad)
            print(unpermute_input.grad)
            print("new_probs.grad=", new_probs.grad)
            print("probs.grad=", probs.grad)
    if num_topK > 1 and torch.allclose(new_probs.grad, probs.grad) == False:
        original_inputs = new_probs.grad.float().cpu().detach().numpy().flatten()
        original_output = probs.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd prob_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(new_probs.grad)
            print(probs.grad)

    if not permute_input.numel():
      print("Empty permute_input activation test passed.")
      return

    ###################################################################################################################################
    #
    # Triton Implementation 
    #
    ###################################################################################################################################
    if TRITON_AVAILABLE:


        triton_permute_input = permute_input.detach().clone().to(dtype)
        triton_permute_bwd_input = permute_bwd_input.detach().to(dtype)
        triton_unpermute_bwd_input = unpermute_bwd_input.detach().to(dtype)
        triton_permute_input.requires_grad_(True)

        # 假设 Triton permute 也返回一个映射，如果不是，请相应调整
        triton_permute_output, triton_row_id_map = permute_triton(triton_permute_input, indices)
        if not torch.equal(row_id_map, triton_row_id_map):
            print("!!!!!! FATAL ERROR: row_id_map from CUDA and Triton are DIFFERENT! !!!!!!")
            print("CUDA map:", row_id_map)
            print("Triton map:", triton_row_id_map)
            print("abs diff:", (row_id_map - triton_row_id_map).abs().max())

        print(">>> VERIFYING PERMUTE OUTPUTS DIRECTLY <<<")
        permute_direct_error = abs(new_permute_output.float() - triton_permute_output.float()).max()
        print(f"Max error between CUDA and Triton PERMUTE outputs: {permute_direct_error:.3e}")
        if permute_direct_error > 1e-6:
            print("!!! DIAGNOSIS CONFIRMED: The permute outputs are NOT identical!")
        else:
            print("Diagnosis incorrect, outputs are identical.")
        if PRINT:
            print("fwd_new_permute_output", new_permute_output)
            print("fwd_triton_permute_output", triton_permute_output)
            print("fwd_permute_output", permute_output)
            print("permute_topK fwd cuda and torch max error(mine vs pytorch): ", abs(new_permute_output.float() - permute_output.float()).max())
            print("permute_topK fwd cuda and pytorch max error( triton vs pytorch): ", abs(triton_permute_output.float() - permute_output.float()).max())
        # 验证前向传播
        if not torch.allclose(new_permute_output.float(), triton_permute_output.float()):
             print(f"fwd_Triton permute_topK fwd cuda and triton max error: {abs(new_permute_output.float() - triton_permute_output.float()).max():.3e}")
        
        triton_permute_output.backward(triton_permute_bwd_input, retain_graph=True)
        # 验证反向传播
        if PRINT:
            print("bwd_pytorch_permute_grad", pytorch_permute_grad)
            print("bwd_triton_permute_input.grad", triton_permute_input.grad)
            print("bwd_new_permute_input.grad", new_permute_input.grad)
            print("bwd_permutate_input.grad(mine vs pytorch)", new_permute_input.grad.float()- pytorch_permute_grad.float())
            print("bwd_triton_permute_input.grad trion vs pytorch", triton_permute_input.grad.float()- pytorch_permute_grad.float())
        if not torch.allclose(new_permute_input.grad.float(), triton_permute_input.grad.float()):
             print(f"bwd_Triton permute_topK bwd max error cuda vs triton : {abs(new_permute_input.grad.float()- triton_permute_input.grad.float()).max():.3e}")

        triton_probs = probs.detach().clone()
        triton_probs.requires_grad_(True)
        triton_unpermute_input = triton_permute_output.detach().clone()
        triton_unpermute_input.requires_grad_(True)
        triton_unpermute_output = unpermute_triton(triton_unpermute_input, triton_row_id_map, triton_probs)
        if PRINT:
            print("fwd_new_unpermute_output", new_unpermute_output)
            print("fwd_triton_unpermute_output", triton_unpermute_output)
            print("fwd_unpermute_output", unpermute_output)
            print("unpermute_topK fwd cuda and triton max error(cuda vs triton): ", abs(new_unpermute_output.float()- triton_unpermute_output.float()).max())
            print(f"unpermute_topK fwd cuda and torch max error(mine vs pytorch): ,{abs(new_unpermute_output.float().cpu().detach().numpy().flatten() - unpermute_output.float().cpu().detach().numpy().flatten()).max():.3e}")
            print(f"unpermute_topK fwd cuda and triton max error(trtion vs pytorch): {abs(triton_unpermute_output.float().cpu().detach().numpy().flatten() - unpermute_output.float().cpu().detach().numpy().flatten()).max():.3e}")
        print("triton_unpermute_output shape", triton_unpermute_output.shape)
        print("new_unpermute_output shape", new_unpermute_output.shape)
        if not torch.allclose(new_unpermute_output.float(), triton_unpermute_output.float()):
            print(f"fwd_Triton unpermute_topK fwd max cuda and triton error: {abs(new_unpermute_output.float() - triton_unpermute_output.float()).max():.3e}")
            # print(f"fwd_Triton unpermute_topK fwd max triton and pytorch error: {abs(triton_permute_output.float()- unpermute_output.float()).max():.3e}")
        triton_unpermute_output.backward(triton_unpermute_bwd_input, retain_graph=True)
        if PRINT:
            print("bwd_new_unpermute_input.grad", new_unpermute_input.grad)
            print("bwd_triton_unpermute_input.grad", triton_unpermute_input.grad)
            print("bwd_pytorch unpermute_input.grad", unpermute_input.grad)
            print("bwd_triton_probs.grad", triton_probs.grad)
            print("bwd_new_unpermute_probs.grad", new_probs.grad)
            print("bwd_pytorch_unpermute_act_grad", probs.grad)
        # 验证反向传播
        print("unpermute_topK bwd act_grad max error (cuda vs pytorch)",abs(new_unpermute_input.grad.float()-triton_unpermute_input.grad.float()).max())
        print("unpermute_topK bwd prob_grad max error (cuda vs pytorch):",abs(new_probs.grad.float()-triton_probs.grad.float()).max())
        if not torch.allclose(new_unpermute_input.grad.float(), triton_unpermute_input.grad.float()):
            print(f"unpermute_topK bwd act_grad max error (cuda  vs pytorch)1: {abs(new_unpermute_input.grad.float().cpu().detach().numpy().flatten() - triton_unpermute_input.grad.float().cpu().detach().numpy().flatten()).max():.3e}")
        if num_topK > 1 and not torch.allclose(new_probs.grad.float(), triton_probs.grad.float()):
            print(f"unpermute_topK bwd prob_grad max error (cuda vs pytorch)1: {abs(new_probs.grad.float().cpu().detach().numpy().flatten() - triton_probs.grad.float().cpu().detach().numpy().flatten()).max():.3e}")


    if not permute_input.numel():
      print("Empty permute_input activation test passed.")
      return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    def backward_wrapper(act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False):
        if accumulate_grad == False:
            for i in forward_input:
                i.grad = None
        return act.backward(backward_input, retain_graph=retain_graph)

    if BENCHMARK:
        print(f"----permute topK---------------------------------------------------------")
        t = perf_test_cuda_kernel(lambda: permute(permute_input, indices, num_topK))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(lambda: permute_topK(new_permute_input, indices))
        print(f"cuda fwd: {t:.3f} ms") # Renamed for clarity
        if TRITON_AVAILABLE: # Add Triton benchmark
            t = perf_test_cuda_kernel(lambda: permute_triton(triton_permute_input, indices))
            print(f"triton   fwd: {t:.3f} ms")
        t= perf_test_cuda_kernel(
            lambda: backward_wrapper(permute_output, permute_bwd_input, forward_input=[permute_input], retain_graph=True, accumulate_grad=False))
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(new_permute_output, new_permute_bwd_input, forward_input=[new_permute_input], retain_graph=True, accumulate_grad=False))
        print(f"cuda bwd: {t:.3f} ms") # Renamed for clarity
        if TRITON_AVAILABLE: # Add Triton benchmark
            t = perf_test_cuda_kernel(
                lambda: backward_wrapper(triton_permute_output, triton_permute_bwd_input, forward_input=[triton_permute_input], retain_graph=True, accumulate_grad=False))
            print(f"triton   bwd: {t:.3f} ms")


        print(f"----unpermute topK------------------------------------------------------")
        t = perf_test_cuda_kernel(
            lambda: unpermute(unpermute_input, sorted_indices, probs=probs, merge_factor=num_topK))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: unpermute_topK(new_unpermute_input, row_id_map, new_probs))
        print(f"c++/cuda fwd: {t:.3f} ms") # Renamed for clarity
        if TRITON_AVAILABLE: # Add Triton benchmark
            t = perf_test_cuda_kernel(
                lambda: unpermute_triton(triton_unpermute_input, triton_row_id_map, triton_probs))
            print(f"triton   fwd: {t:.3f} ms")

        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(unpermute_output, unpermute_bwd_input, forward_input=[unpermute_input, probs], retain_graph=True, accumulate_grad=False))
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(new_unpermute_output, new_unpermute_bwd_input, forward_input=[new_unpermute_input, new_probs], retain_graph=True, accumulate_grad=False))
        print(f"c++/cuda bwd: {t:.3f} ms") # Renamed for clarity
        if TRITON_AVAILABLE: # Add Triton benchmark
            t = perf_test_cuda_kernel(
                lambda: backward_wrapper(triton_unpermute_output, triton_unpermute_bwd_input, forward_input=[triton_unpermute_input, triton_probs], retain_graph=True, accumulate_grad=False))
            print(f"triton   bwd: {t:.3f} ms")
            
            
def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(50):
            cuda_kernel_fn()

        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"Elapsed Time: {elapsed_time_ms / 100} ms")
        return elapsed_time_ms / 100
    else:
        print("CUDA is not available.")

def test_permute_topK():

    torch.manual_seed(1)

    num_token = 4096 * 2
    num_expert = 8
    hidden_size = 4096
    num_topK = 8

    PRINT=True
    Benchmark = True
    print("GPU:", torch.cuda.get_device_name(0))

    dtype = torch.float32
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, PRINT, Benchmark)
    dtype = torch.float16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, PRINT, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, PRINT, Benchmark)
    # dtype = torch.float8_e5m2
    # permute_topK_test(dtype, num_token, num_expert,
    #                   hidden_size, num_topK, False, Benchmark)
    # dtype = torch.float8_e4m3fn
    # permute_topK_test(dtype, num_token, num_expert,
    #                   hidden_size, num_topK, False, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, PRINT, Benchmark)
    num_token = 0
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, PRINT, Benchmark)
    dtype = torch.float16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4, PRINT, Benchmark)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, PRINT, Benchmark)
    num_token = 0
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, PRINT, Benchmark)
if __name__ == "__main__":
    test_permute_topK()