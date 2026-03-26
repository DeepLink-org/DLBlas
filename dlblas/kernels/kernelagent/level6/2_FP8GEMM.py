# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_fp8_gemm.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/fp8_gemm.py
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _block_scaled_matmul_kernel(
    A_ptr, B_ptr, As_ptr, Bs_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_asm, stride_ask,
    stride_bsm, stride_bsn,
    stride_cm, stride_cn,
    n_tiles, k_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs: tile over M (rows of output) and N (columns of output)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_start = 0
    while k_start < K:
        tile_k = k_start // BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile as in reference: b = B[j-block rows, i-block cols], then use b.T in matmul
        # Here we load b_sub with shape [BLOCK_N, BLOCK_K] = B[offs_n, offs_k]
        bsub_ptrs = B_ptr + (offs_n[:, None] * stride_bk + offs_k[None, :] * stride_bn)
        b_sub = tl.load(bsub_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Compute partial = A @ b_sub.T  -> use tl.trans on b_sub
        partial = tl.dot(a, tl.trans(b_sub))  # [BLOCK_M, BLOCK_N]

        # Load scaling: s = As[:, tile_k] * Bs[pid_n, tile_k]
        as_ptrs = As_ptr + (offs_m * stride_asm + tile_k * stride_ask)
        as_vec = tl.load(as_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M]
        bs_scalar = tl.load(Bs_ptr + pid_n * stride_bsm + tile_k * stride_bsn)

        s_vec = as_vec * bs_scalar
        partial = partial * s_vec[:, None]

        acc += partial
        k_start += BLOCK_K

    # Store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, M, N, K, block_size=128):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.block_size = block_size

    def forward(self, A, B, As, Bs, output_dtype=torch.float32):
        """This function performs matrix multiplication with block-wise quantization.

        It takes two input tensors `A` and `B` with scales `As` and `Bs`.
        The output is returned in the specified `output_dtype`.
        """
        M = self.M
        N = self.N
        K = self.K
        block_size = self.block_size

        n_tiles = (N + block_size - 1) // block_size
        k_tiles = (K + block_size - 1) // block_size
        assert n_tiles == Bs.shape[0]
        assert k_tiles == Bs.shape[1]

        # Convert A and B to float32 for stable compute (inputs may be fp8)
        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)

        # Allocate output accumulator in float32; cast to output_dtype at the end
        C32 = torch.zeros((M, N), dtype=torch.float32, device=A.device)

        # Strides
        stride_am, stride_ak = A32.stride()
        stride_bk, stride_bn = B32.stride()
        stride_asm, stride_ask = As.stride()
        stride_bsm, stride_bsn = Bs.stride()
        stride_cm, stride_cn = C32.stride()

        # Tile sizes
        BLOCK_M = 128
        BLOCK_N = block_size
        BLOCK_K = block_size

        grid = (triton.cdiv(M, BLOCK_M), n_tiles)

        _block_scaled_matmul_kernel[grid](
            A32, B32, As, Bs, C32,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_asm, stride_ask,
            stride_bsm, stride_bsn,
            stride_cm, stride_cn,
            n_tiles, k_tiles,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3
        )

        return C32.to(output_dtype)


M = 512
N = 512
K = 512
block_size = 128
num_block = int(K / block_size)

def get_inputs():
    A = torch.randn((M, K), device='cuda').to(torch.float8_e4m3fn)
    B = torch.randn((K, N), device='cuda').to(torch.float8_e4m3fn)
    As = torch.randn((M, num_block), device='cuda')
    Bs = torch.randn((num_block, num_block), device='cuda')
    return [A, B, As, Bs]

def get_init_inputs():
    return [M, N, K]