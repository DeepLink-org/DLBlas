# run_check.py
import torch
import torch_npu
import torch.nn as nn
from layernorm01 import Model, get_inputs

DEVICE = "npu:0"
WARMUP = 100
TEST_ITER = 1000

def run_verify():
    torch_npu.npu.set_device(0)
    model = Model().to(DEVICE).eval()
    x = get_inputs()[0]

    # 1. 精度校验 fp32/fp16/bf16
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_dt = x.to(dtype)
        # CPU基准
        cpu_mod = Model().cpu()
        cpu_x = x_dt.cpu()
        with torch.no_grad():
            gt = cpu_mod(cpu_x)
        
        # NPU推理
        with torch.no_grad():
            if dtype in (torch.float16, torch.bfloat16):
                with torch.npu.amp.autocast(dtype=dtype):
                    out = model(x_dt)
            else:
                out = model(x_dt)
        
        out_cpu = out.cpu().float()
        gt_f32 = gt.float()
        # 误差容忍
        if dtype == torch.float32:
            torch.testing.assert_close(out_cpu, gt_f32, rtol=1e-5, atol=1e-5)
        elif dtype == torch.float16:
            torch.testing.assert_close(out_cpu, gt_f32, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(out_cpu, gt_f32, rtol=5e-3, atol=5e-3)
        print(f"✅ {dtype} 精度通过")

    # 2. JIT校验
    trace_x = torch.randn(10,10, dtype=torch.float16, device=DEVICE)
    jit_m = torch.jit.trace(model, trace_x)
    with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.float16):
        o1 = model(trace_x)
        o2 = jit_m(trace_x)
    torch.testing.assert_close(o1, o2, rtol=1e-3, atol=1e-3)
    print("✅ JIT Trace 一致")

    # 3. 性能测速
    print("\n====性能测速====")
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_dt = x.to(dtype)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream), torch.no_grad():
            # 热身
            for _ in range(WARMUP):
                if dtype in (torch.float16, torch.bfloat16):
                    with torch.npu.amp.autocast(dtype=dtype):
                        _ = model(x_dt)
                else:
                    _ = model(x_dt)
            torch.npu.synchronize()

            start = torch_npu.npu.Event(enable_timing=True)
            end = torch_npu.npu.Event(enable_timing=True)
            start.record()
            for _ in range(TEST_ITER):
                if dtype in (torch.float16, torch.bfloat16):
                    with torch.npu.amp.autocast(dtype=dtype):
                        _ = model(x_dt)
                else:
                    _ = model(x_dt)
            end.record()
            torch.npu.synchronize()
        total_ms = start.elapsed_time(end)
        avg = total_ms / TEST_ITER
        print(f"{dtype} 总耗时{total_ms:.3f}ms 单次{avg:.6f}ms QPS={1000/avg:.1f}")

if __name__ == "__main__":
    run_verify()