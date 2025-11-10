import torch
from dlblas.kernels.ascend.fourd_matrix_multiply import ModelNew as ModelTriton
import torch.nn as nn


class ModelTorch(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return torch.einsum("bijl,lk->bijk", A, B)


# @pytest.mark.skipif(not importlib.util.find_spec("mindspeed"), reason="requires mindspeed")
# @pytest.mark.parametrize(['N', 'K'], [(4096, 4096), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
# @pytest.mark.parametrize('groups', [8])
def test_fourd_matrix_multiply():
    b = 16
    i = 256
    j = 512
    l = 256
    k = 768

    A = torch.randn(b, i, j, l, device='npu')
    B = torch.randn(l, k, device='npu')

    modeltorch = ModelTorch()
    modeltriton = ModelTriton()
    torch_C = modeltorch.forward(A, B)
    triton_C = modeltriton.forward(A, B)
    assert torch.allclose(torch_C, triton_C, rtol=1e-4, atol=1e-4)
