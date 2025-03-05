import pytest
import torch
from dlblas.kernels.quant_dequant import per_channel_quant_fp8, per_channel_dequant_bf16


class TestQuantDequant:

    @pytest.fixture(scope="class")
    def x(self):
        yield torch.rand(10, 10, dtype=torch.bfloat16)

    def test_fp8_gemm(self, x):
        x_quanted, x_scales = per_channel_quant_fp8(x)
        x_origin = per_channel_dequant_bf16(x_quanted, x_scales)
        assert torch.allclose(x, x_origin, atol=1, rtol=0.01)
