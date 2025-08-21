from typing import Tuple
import random
import torch

from torch.profiler import ProfilerActivity, profile, record_function

from dlblas.kernels.grouped_gemm.BF16 import m_grouped_gemm
from dlblas.kernels.grouped_gemm.BF16.utils import generate_random_list, row_max_normalization
print(f"{m_grouped_gemm = }")


def gmm(a, b, batch_sizes, trans_b=False):
        batch_sizes = batch_sizes.numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start:start + size, :] @ rhs)
            start += size
        return torch.cat(out)


groups = 128; z = groups
trans_b = False; print(f"{trans_b = }")
device = f"cuda"
batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64)
# batch_sizes = torch.tensor([1] * 128, device=device, dtype=torch.int64)

batch_sizes_cpu = batch_sizes.cpu()
M = batch_sizes.sum().item()

for (n, k) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
    torch.cuda.empty_cache()
    a = torch.randn(M, k, dtype = torch.bfloat16, device = device).view(-1, k)
    b = torch.randn(z, n, k, dtype = torch.bfloat16, device = device) if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = device)
    out_ref = gmm(a, b, batch_sizes.cpu(), trans_b)

    for i in range(3):
        out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)

    from pathlib import Path
    script_path = Path(__file__).resolve()
    parent_dir = script_path.parent.parent
    from torch.profiler import ProfilerActivity, profile, record_function
    trace_file = f"{parent_dir}/trace/gmm_triton_cublas_cutlass_N{n}_K{k}"  + ".json"
    import os
    Path(os.path.join(parent_dir, "trace")) .mkdir(parents=True, exist_ok=True)
    activate_ = 10
    def trace_handler(prof):
        prof.export_chrome_trace(trace_file)
    with profile(
        activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=3,
            active=activate_,
            repeat=0),
        on_trace_ready=trace_handler,
        with_modules = True,
        record_shapes=True,) as prof:
        for i in range(4+activate_):
            out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)
            torch.cuda.synchronize(device = device)
            prof.step()
    # diff = out_triton - out_ref
    # breakpoint()
    # post-process, row normalization
    out_triton = row_max_normalization(out_triton)
    out_ref = row_max_normalization(out_ref)
    torch.cuda.empty_cache()

    torch.testing.assert_close(out_triton, out_ref, rtol = 1e-02, atol = 1e-02)

    print(f"{n = }, {k = }, {M = }, {trace_file = }")
    

    import json
    with open(trace_file, "r") as file:
        data = json.load(file)

    triton_time = 0
    for event in data["traceEvents"]:
        try:
            if "m_grouped_gemm_" in event["name"]:
                triton_time += event["dur"] / 1000
        except:
            pass
    triton_time /= activate_
    print(f"    Pure kernel Elapsed time {round((triton_time), 2)} ms, {round((2*M*n*k )/(triton_time)/10**9, 0)} tflops")