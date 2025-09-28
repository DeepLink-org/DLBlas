# EVOLVE-BLOCK
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def grouped_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Group information
    group_size: tl.constexpr,
    group_stride_a: tl.constexpr,
    group_stride_b: tl.constexpr,
    group_stride_c: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing grouped matrix multiplication C = A x B.
    A has shape (G, M, K), B has shape (G, K, N), C has shape (G, M, N)
    """
    # Group ID
    g_id = tl.program_id(axis=0)
    # Position within group
    pid = tl.program_id(axis=1)
    
    # Create program ids for the matrix multiplication grid
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Calculate group offsets
    offs_g = g_id * group_size
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute pointers for A and B
    a_ptrs = a_ptr + offs_g * group_stride_a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offs_g * group_stride_b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize and iteratively update accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load data with boundary checks
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute partial matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers to next block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk * K  # Stride to next K block
    
    # Write back result with boundary checks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_g * group_stride_c + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def grouped_matmul(A, B):
    # Check dimensions
    assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3D tensors"
    assert A.shape[0] == B.shape[0], "Group dimension mismatch"
    assert A.shape[2] == B.shape[1], "Inner dimension mismatch"
    
    G, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty((G, M, N), device=A.device, dtype=A.dtype)
    
    # 1D launch kernel for groups and 2D grid for matrix multiplication
    grid = lambda META: (G, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']))
    
    grouped_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        C.stride(1), C.stride(2),
        group_size=M*K,
        group_stride_a=A.stride(0),
        group_stride_b=B.stride(0),
        group_stride_c=C.stride(0),
    )
    return C

class ModelNew:
    def __init__(self):
        pass

    def forward(self, A, B):
        return grouped_matmul(A, B)