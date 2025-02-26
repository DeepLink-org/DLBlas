# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
import pytest
import torch
from dlblas.kernels.fp8_gemm import fp8_gemm


M = 512
N = 512
K = 512
block_size = 128
num_block = int(K / block_size)


class TestRMSNorm:

    @pytest.fixture(scope="class")
    def A(self):
        yield torch.randn((M, K), device='cuda').to(torch.float8_e4m3fn)

    @pytest.fixture(scope="class")
    def B(self):
        yield torch.randn((K, N), device='cuda').to(torch.float8_e4m3fn)

    @pytest.fixture(scope="class")
    def As(self):
        yield torch.randn((M, num_block), device='cuda')

    @pytest.fixture(scope="class")
    def Bs(self):
        yield torch.randn((num_block, num_block), device='cuda')

    @pytest.fixture(scope="class")
    def gt(self, A, B, As, Bs, output_dtype=torch.float32):
        """This function performs matrix multiplication with block-wise quantization using native torch.

        It takes two input tensors `A` and `B` with scales `As` and `Bs`.
        The output is returned in the specified `output_dtype`.
        """
    
        n_tiles = (N + block_size - 1) // block_size
        k_tiles = (K + block_size - 1) // block_size
        assert n_tiles == Bs.shape[0]
        assert k_tiles == Bs.shape[1]

        C_shape = (M, N)
        C = torch.zeros(C_shape, dtype=output_dtype, device=A.device)

        A_tiles = [A[:, i * block_size : min((i + 1) * block_size, K)] for i in range(k_tiles)]
        B_tiles = [
            [
                B[
                    j * block_size : min((j + 1) * block_size, N),
                    i * block_size : min((i + 1) * block_size, K),
                ]
                for i in range(k_tiles)
            ]
            for j in range(n_tiles)
        ]
        C_tiles = [C[:, j * block_size : min((j + 1) * block_size, N)] for j in range(n_tiles)]
        As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

        for i in range(k_tiles):
            for j in range(n_tiles):
                a = A_tiles[i].to(output_dtype)     # [M, 128]
                b = B_tiles[j][i].to(output_dtype)  #[128, 128]
                c = C_tiles[j]     # [M, 128]
                s = As_tiles[i] * Bs[j][i]  #[M, 1]
                c[:, :] += torch.matmul(a, b.t()) * s

        C = C.reshape((M, N)).to(output_dtype)
        return C

    def test_fp8_gemm(self, A, B, As, Bs, gt):
        out = fp8_gemm(A, B, As, Bs)
        torch.testing.assert_close(out, gt, atol=1e-1, rtol=0)
