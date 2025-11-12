import torch
from dlblas.kernels.ascend.three_matrix_multiply import ModelNew as ModelTriton
import torch.nn as nn


class ModelTorch(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelTorch, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return torch.matmul(A, B)


def test_three_matrix_multiply():
    N = 16
    M = 1024
    K = 2048
    L = 768

    A = torch.randn(N, M, K, device='npu')
    B = torch.randn(K, L, device='npu')

    modeltorch = ModelTorch()
    modeltriton = ModelTriton()
    torch_C = modeltorch.forward(A, B)
    triton_C = modeltriton.forward(A, B)
    assert torch.allclose(torch_C, triton_C, rtol=1e-2, atol=1e-2)
