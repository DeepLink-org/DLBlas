import pytest
import torch
import torch_npu
import torch.nn as nn

# 加载上面适配后的Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(10)

    def forward(self, x):
        return self.ln(x)

# 全局配置
NORM_DIM = 10
INPUT_BATCH = 10
INPUT_SHAPE = (INPUT_BATCH, NORM_DIM)
DEVICE = "npu:0"
WARMUP = 100
RUN_TIMES = 1000

class TestLayerNormNPU:
    @pytest.fixture(scope="class")
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def model(self):
        torch_npu.npu.set_device(0)
        net = Model().to(DEVICE)
        net.eval()
        return net

    @pytest.fixture(scope="class")
    def input_x(self, dtype):
        torch.manual_seed(42)
        x = torch.randn(*INPUT_SHAPE, dtype=dtype, device=DEVICE)
        return x

    @pytest.fixture(scope="class")
    def cpu_gt(self, model, input_x, dtype):
        cpu_net = Model().cpu()
        cpu_x = input_x.cpu().to(dtype)
        with torch.no_grad():
            ground = cpu_net(cpu_x)
        return ground

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], indirect=True)
    def test_forward_correctness(self, model, input_x, cpu_gt, dtype):
        model.eval()
        with torch.no_grad():
            if dtype in (torch.float16, torch.bfloat16):
                with torch.npu.amp.autocast(dtype=dtype):
                    out = model(input_x)
            else:
                out = model(input_x)

        assert out.shape == INPUT_SHAPE
        out_cpu = out.cpu().float()
        gt_float = cpu_gt.float()

        if dtype == torch.float32:
            torch.testing.assert_close(out_cpu, gt_float, rtol=1e-5, atol=1e-5, check_dtype=False)
        elif dtype == torch.float16:
            torch.testing.assert_close(out_cpu, gt_float, rtol=1e-3, atol=1e-3, check_dtype=False)
        else:
            torch.testing.assert_close(out_cpu, gt_float, rtol=5e-3, atol=5e-3, check_dtype=False)

    def test_jit_trace_consistent(self, model):
        model.eval()
        torch.manual_seed(42)
        trace_in = torch.randn(*INPUT_SHAPE, dtype=torch.float16, device=DEVICE)
        jit_net = torch.jit.trace(model, trace_in)

        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.float16):
            origin_out = model(trace_in)
            jit_out = jit_net(trace_in)

        torch.testing.assert_close(jit_out, origin_out, rtol=1e-3, atol=1e-3, check_dtype=False)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], indirect=True)
    def test_performance(self, model, input_x, dtype):
        model.eval()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream), torch.no_grad():
            # 热身
            for _ in range(WARMUP):
                if dtype in (torch.float16, torch.bfloat16):
                    with torch.npu.amp.autocast(dtype=dtype):
                        _ = model(input_x)
                else:
                    _ = model(input_x)
            torch.npu.synchronize()

            start = torch_npu.npu.Event(enable_timing=True)
            end = torch_npu.npu.Event(enable_timing=True)
            start.record()
            for _ in range(RUN_TIMES):
                if dtype in (torch.float16, torch.bfloat16):
                    with torch.npu.amp.autocast(dtype=dtype):
                        _ = model(input_x)
                else:
                    _ = model(input_x)
            end.record()
            torch.npu.synchronize()

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / RUN_TIMES
        print(f"\n[{dtype}] 总耗时={total_ms:.4f}ms, 单次平均={avg_ms:.6f}ms, QPS={1000/avg_ms:.2f}")