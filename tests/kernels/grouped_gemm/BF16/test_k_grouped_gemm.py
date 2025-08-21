from typing import Tuple
import random
import torch

from torch.profiler import ProfilerActivity, profile, record_function

from dlblas.kernels.grouped_gemm.BF16 import k_grouped_gemm
from dlblas.kernels.grouped_gemm.BF16.utils import generate_random_list, row_max_normalization
print(f"{k_grouped_gemm = }")
def gmm_dw(a, b, batch_sizes):
    K, M = a.shape
    K_, N = b.shape

    assert a.stride(-1) == 1, "Please make sure A is K-major"
    assert b.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    num_groups = batch_sizes.shape[0]

    out = a.new_empty(num_groups, M, N)

    group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
    group_start = batch_sizes.cumsum(0) - batch_sizes
    for g, (start, end) in enumerate(zip(group_start, group_end)):
        rhs = b[start:end, :]
        lhs = a[start:end, :]
        out[g] = lhs.T @ rhs
    return out.contiguous()

groups = 128; z = groups

batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).cuda().to(torch.int64).abs()
batch_sizes_cpu = batch_sizes.cpu()
K = batch_sizes.sum().item()
for (m, n) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
    torch.cuda.empty_cache()
    a = torch.randn(K, m, dtype = torch.bfloat16, device = "cuda").view(K, -1)
    b = torch.randn(K, n, dtype = torch.bfloat16, device = "cuda").view(K, -1)
    out_ref = gmm_dw(a, b, batch_sizes.cpu())
    out_cublas = torch.empty_like(out_ref)

    from pathlib import Path
    script_path = Path(__file__).resolve()
    parent_dir = script_path.parent.parent
    trace_file = f"{parent_dir}/trace/gmm_dw_triton_cublas_M{m}_N{n}"  + ".json"
    import os
    Path(os.path.join(parent_dir, "trace")) .mkdir(parents=True, exist_ok=True)
    active_ = 3
    def trace_handler(prof):
        prof.export_chrome_trace(trace_file)
    with profile(
        activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=3,
            active=active_,
            repeat=0),
        on_trace_ready=trace_handler,
        with_modules = True,
        record_shapes=True,) as prof:
        for i in range(4+active_):
            with record_function(f"Triton_record"):
                out_triton = k_grouped_gemm(a, b, batch_sizes)

            prof.step()

    # post-process, row normalization
    out_triton = row_max_normalization(out_triton)
    out_ref = row_max_normalization(out_ref)

    torch.testing.assert_close(out_triton, out_ref, rtol = 0.01, atol = 0.01)
    torch.cuda.empty_cache()

    print(f"{m = }, {n = }, {K = }")
    

    import json
    with open(trace_file, "r") as file:
        data = json.load(file)

    def process_events(data, record_function):
        func_dict = {}
        
        # Process each event to collect cublas records
        for event in data["traceEvents"]:
            if event["name"] == record_function and "gpu_user_annotation" in event["cat"]:
                start = event["ts"]
                end = start + event["dur"]
                # import pdb; pdb.set_trace()
                cpu_id = event['args']['External id']
                
                if cpu_id not in func_dict:
                    # Initialize if id doesn't exist
                    func_dict[cpu_id] = {"start": start, "end": end}
                else:
                    # Update start and end if id exists
                    func_dict[cpu_id]["start"] = min(start, func_dict[cpu_id]["start"])
                    func_dict[cpu_id]["end"] = max(end, func_dict[cpu_id]["end"])
        
        # Calculate duration for each cublas event in microseconds
        durations = []
        for cpu_id in func_dict:
            duration = (func_dict[cpu_id]["end"] - func_dict[cpu_id]["start"]) / 1000  # Convert to milliseconds
            func_dict[cpu_id]["dur"] = duration
            durations.append(duration)
        
        # Calculate average duration if there are any events
        func_time = sum(durations) / len(durations) if durations else 0
        
        return func_dict, func_time

    triton_dict, triton_time = process_events(data, "Triton_record")

    print(f"    Pure Triton kernel Elapsed time {round((triton_time), 2)} ms, {round((2*m*n*K )/(triton_time)/10**9, 0)} tflops")

"""
k_grouped_gemm = <function k_grouped_gemm_general at 0x7fe3b9288180>
    m = 1536, n = 2048, K = 655360
        Pure Triton kernel Elapsed time 6.49 ms, 635.0 tflops
    m = 2048, n = 768, K = 655360
        Pure Triton kernel Elapsed time 3.71 ms, 556.0 tflops
    m = 3072, n = 4096, K = 655360
        Pure Triton kernel Elapsed time 29.18 ms, 565.0 tflops
    m = 4096, n = 1536, K = 655360
        Pure Triton kernel Elapsed time 15.1 ms, 546.0 tflops

k_grouped_gemm = <function k_grouped_gemm_TMA at 0x7fa5213996c0>
    m = 1536, n = 2048, K = 655360
        Pure Triton kernel Elapsed time 6.03 ms, 684.0 tflops
    m = 2048, n = 768, K = 655360
        Pure Triton kernel Elapsed time 3.5 ms, 590.0 tflops
    m = 3072, n = 4096, K = 655360
        Pure Triton kernel Elapsed time 25.43 ms, 649.0 tflops
    m = 4096, n = 1536, K = 655360
        Pure Triton kernel Elapsed time 13.53 ms, 610.0 tflops

"""