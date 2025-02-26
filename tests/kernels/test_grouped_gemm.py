import pytest
import torch
from dlblas.kernels.grouped_gemm import grouped_gemm

class TestGroupedGemm:

    def test_grouped_gemm(self):
        DEVICE = 'cuda'
        block_size = 32
        group_m = [1024, 512, 256, 128]
        group_n = [1024, 512, 256, 128]
        group_k = [1024, 512, 256, 128]
        group_A = []
        group_B = []
        group_As = []
        group_Bs = []
        assert len(group_m) == len(group_n)
        assert len(group_n) == len(group_k)
        group_size = len(group_m)
        for i in range(group_size):
            M = group_m[i]
            N = group_n[i]
            K = group_k[i]
            num = int(K / block_size)
            A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
            B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
            group_A.append(A)
            group_B.append(B)

        tri_out = grouped_gemm(group_A, group_B)
        ref_out = [torch.matmul(group_A[i], group_B[i]) for i in range(group_size)]
        for i in range(group_size):
            assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)
