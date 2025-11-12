import torch
import torch.nn as nn
from dlblas.kernels.ascend.gemm_scale_bn import ModelNew as ModelTriton

class ModelTorch(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelTorch, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, device='npu')
        self.scale = nn.Parameter(torch.randn(scale_shape, device='npu'))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum, device='npu')

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x


def test_gemm_scale_bn():
    batch_size = 128
    in_features = 1024
    out_features = 512
    scale_shape = out_features

    x = torch.randn(batch_size, in_features, device='npu')

    torch.manual_seed(41)
    modeltorch = ModelTorch(in_features, out_features, scale_shape)
    torch.manual_seed(41)
    modeltriton = ModelTriton(in_features, out_features, scale_shape)
    torch_C = modeltorch.forward(x)
    triton_C = modeltriton.forward(x)
    assert torch.allclose(torch_C, triton_C, rtol=1e-2, atol=1e-2)
