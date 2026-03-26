# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_fp8_gemm.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/fp8_gemm.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, M, N, K, block_size=128):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.block_size = block_size

    def forward(self, A, B, As, Bs, output_dtype=torch.float32):
        """This function performs matrix multiplication with block-wise quantization using native torch.

        It takes two input tensors `A` and `B` with scales `As` and `Bs`.
        The output is returned in the specified `output_dtype`.
        """

        M = self.M
        N = self.N
        K = self.K
        block_size = self.block_size
        num_block = int(K / block_size)

        n_tiles = (N + block_size - 1) // block_size
        k_tiles = (K + block_size - 1) // block_size
        assert n_tiles == Bs.shape[0]
        assert k_tiles == Bs.shape[1]

        C_shape = (M, N)
        C = torch.zeros(C_shape, dtype=output_dtype, device=A.device)

        A_tiles = [A[:, i * block_size:min((i + 1) * block_size, K)] for i in range(k_tiles)]
        B_tiles = [[
            B[
                j * block_size:min((j + 1) * block_size, N),
                i * block_size:min((i + 1) * block_size, K),
            ] for i in range(k_tiles)
        ] for j in range(n_tiles)]
        C_tiles = [C[:, j * block_size:min((j + 1) * block_size, N)] for j in range(n_tiles)]
        As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

        for i in range(k_tiles):
            for j in range(n_tiles):
                a = A_tiles[i].to(output_dtype)  # [M, 128]
                b = B_tiles[j][i].to(output_dtype)  # [128, 128]
                c = C_tiles[j]  # [M, 128]
                s = As_tiles[i] * Bs[j][i]  # [M, 1]
                c[:, :] += torch.matmul(a, b.t()) * s

        C = C.reshape((M, N)).to(output_dtype)
        return C

M = 512
N = 512
K = 512
block_size = 128
num_block = int(K / block_size)

def get_inputs():
    A = torch.randn((M, K), device='cpu').to(torch.float8_e4m3fn)
    B = torch.randn((K, N), device='cpu').to(torch.float8_e4m3fn)
    As = torch.randn((M, num_block), device='cpu')
    Bs = torch.randn((num_block, num_block), device='cpu')
    return [A, B, As, Bs]

def get_init_inputs():
    return [M, N, K]