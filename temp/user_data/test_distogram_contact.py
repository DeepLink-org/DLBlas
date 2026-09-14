# Copyright (c) 2026
# pytest test_distogram_contact.py
import pytest
import torch
import torch_npu
import torch.nn as nn

# ---------------------- 待测模型 ----------------------
class Model(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.no_bins = int(no_bins)

        edges = torch.linspace(min_bin, max_bin, self.no_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        self.thres_idx = int((bin_centers < thres).sum().item())
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(distogram_logits, dim=-1)
        contact_prob = prob[..., :self.thres_idx].sum(dim=-1)
        return contact_prob

# 全局超参
N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0
DEVICE = "npu:0"

# ---------------------- pytest 测试类 ----------------------
class TestDistogramContact:
    @pytest.fixture(scope="class")
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def model(self):
        torch_npu.npu.set_device(0)
        m = Model(MIN_BIN, MAX_BIN, NO_BINS, THRES).to(DEVICE)
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def logits_input(self, dtype):
        torch.manual_seed(42)
        logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, dtype=dtype, device=DEVICE)
        return logits

    @pytest.fixture(scope="class")
    def gt_contact_prob(self, model, logits_input):
        with torch.no_grad():
            prob = torch.softmax(logits_input, dim=-1)
            gt = prob[..., :model.thres_idx].sum(dim=-1)
        return gt

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], indirect=True)
    def test_forward_correctness(self, model, logits_input, gt_contact_prob, dtype):
        model.eval()
        with torch.no_grad():
            if dtype in (torch.float16, torch.bfloat16):
                with torch.npu.amp.autocast(dtype=dtype):
                    out = model(logits_input)
            else:
                out = model(logits_input)

        assert out.shape == (N_TOKEN, N_TOKEN)
        # 分精度设置误差阈值，bf16放宽
        if dtype == torch.float32:
            torch.testing.assert_close(
                out, gt_contact_prob,
                rtol=1e-5, atol=1e-5,
                check_dtype=False
            )
        elif dtype == torch.float16:
            torch.testing.assert_close(
                out, gt_contact_prob,
                rtol=1e-3, atol=1e-3,
                check_dtype=False
            )
        else: # bfloat16 放大容错
            torch.testing.assert_close(
                out, gt_contact_prob,
                rtol=5e-3, atol=5e-3,
                check_dtype=False
            )

    # 改用jit.trace，不依赖源码读取，解决OSError
    def test_jit_trace_consistent(self, model):
        model.eval()
        torch.manual_seed(42)
        trace_input = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, dtype=torch.float16, device=DEVICE)
        # 追踪生成jit模型
        jit_model = torch.jit.trace(model, trace_input)

        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.float16):
            orig_out = model(trace_input)
            jit_out = jit_model(trace_input)

        # fp16误差标准
        torch.testing.assert_close(jit_out, orig_out, rtol=1e-3, atol=1e-3, check_dtype=False)