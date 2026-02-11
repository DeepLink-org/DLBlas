import triton
import triton.language as tl
import torch
import torch_npu


@triton.jit
def mhc_pre_gemm_sqrsum_triton_kernel(
    # Pointers to matrices
    x_ptr, fn_ptr, out_ptr, sqrsum_ptr,
    # Matrix dimensions
    num_tokens, hidden_size, hc_mult3,
    # Stride variables
    stride_x_token, stride_x_hidden,
    stride_fn_mult3, stride_fn_hidden,
    stride_out_token,
    # Block sizes
    token_block: tl.constexpr = 32,
    hidden_block: tl.constexpr = 64,
    # Meta-parameter for BLOCK_SIZE_N
    BLOCK_SIZE_K: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 32,
):
    """
    Triton kernel for fused GEMM and sqrsum computation.
    
    Args:
        x_ptr: Input tensor of shape (num_tokens, hc_hidden_size)
        fn_ptr: Weight tensor of shape (hc_mult3, hc_hidden_size) 
        out_ptr: Output tensor of shape (num_tokens, hc_mult3)
        sqrsum_ptr: Square sum output of shape (num_tokens,)
        num_tokens: Number of input tokens
        hc_hidden_size: Hidden dimension size
        hc_mult3: Size of the third dimension
    """
    # Program IDs
    token_pid = tl.program_id(axis=0)  # Token block index
    
    # Token range for this program
    token_offset = token_pid * token_block + tl.arange(0, token_block)
    
    # Initialize accumulators for square sums
    sqrsum_acc = tl.zeros((1,), dtype=tl.float32)  # token_block
    thirty = tl.arange(0, 32)

    for hidden_idx in range(0, hidden_size, hidden_block):
        hidden_offset = hidden_idx + tl.arange(0, hidden_block)
        x_offset = token_offset[:, None] + hidden_offset[None, :]

        x = tl.load(
            x_ptr + token_offset[:, None] + hidden_offset[None, :]
        ).to(tl.bfloat16)
        x = x.to(tl.float32)
        sqrsum_prt = x * x
        sqrsum_acc += tl.sum(sqrsum_prt)

        fn_offset = tl.arange(0, 32)
        fn = tl.load(
            fn_ptr + fn_offset[:, None] * stride_fn_mult3 + hidden_offset[None, :] * stride_fn_hidden
            # mask=fn_mask[:, None] & (hidden_offsets[None, :] < hc_hidden_size),
            # other=0.0
        ).to(tl.float32)

        # !TODO transpose fn
        fn = tl.trans(fn, 1, 0)
        out_val = tl.dot(x, fn)

        out_token_offset = token_offset[:, None]
        out_mult3_offset = thirty[None, :]
        tl.store(
            out_ptr + out_token_offset * stride_out_token + out_mult3_offset * 1,
            out_val,
        )
    
    # Write square sum results
    sqrsum_val = tl.full((32, 32), 1, dtype=tl.float32) * sqrsum_acc
    thirty = tl.arange(0, 32)
    sqrsum_offset = thirty[:, None] + thirty[None, :]
    tl.store(
        sqrsum_ptr + sqrsum_offset,
        sqrsum_val
    )


def mhc_pre_gemm_sqrsum_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 64,
):
    """
    Triton implementation of fused gemm and sqrsum in mHC pre block.
    
    Args:
        x: Input tensor of shape (num_tokens, hc_hidden_size), dtype torch.bfloat16
        fn: Weight tensor of shape (hc_mult3, hc_hidden_size), dtype torch.float32
        out: Output tensor of shape (num_tokens, hc_mult3), dtype torch.float32
        sqrsum: Square sum tensor of shape (num_tokens,), dtype torch.float32
        hc_mult3: Size of the third dimension
        hc_hidden_size: Hidden size dimension
        token_block: Size of token block for tiling
        hidden_block: Size of hidden block for tiling
    """
    # Get tensor dimensions
    num_tokens = x.shape[0]
    
    # Define strides
    stride_x_token = x.stride(0)
    stride_x_hidden = x.stride(1)
    stride_fn_mult3 = fn.stride(0)
    stride_fn_hidden = fn.stride(1)
    stride_out_token = out.stride(0)
    
    # Launch grid
    grid = lambda META: (1,)
    
    # Set appropriate block sizes based on actual dimensions
    BLOCK_SIZE_N = min(32, hc_mult3)
    BLOCK_SIZE_K = hidden_block
    
    # Launch kernel
    import time
    start = time.time()
    mhc_pre_gemm_sqrsum_triton_kernel[grid](
        x, fn, out, sqrsum,
        num_tokens, hc_hidden_size, hc_mult3,
        stride_x_token, stride_x_hidden,
        stride_fn_mult3, stride_fn_hidden,
        stride_out_token,
        token_block=token_block,
        hidden_block=hidden_block,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    print(time.time() - start)
    
    return out, sqrsum


def generate_test_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
):
    """Generate test data for big fuse operator."""
    torch.random.manual_seed(0)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "npu"

    residual = (
        torch.ones((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.ones((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.ones((3,), dtype=torch.float, device=device) * 0.1

    hc_base = torch.ones((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def main():
    """
    Main function to test the fused moe kernel.
    """
    # Parameters
    dhidden = 1280  # 7168
    dexpert = 2048
    n_routed_experts = 8
    n_shared_experts = 1
    n_experts_per_token = 4
    batch_size = 1
    seq_len = 512  # 8192
    
    # Generate test data
    test_data = generate_test_data(
        n=batch_size * seq_len,
        hc_mult=n_experts_per_token,
        hidden_size=dhidden,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
    )

    out, sqrsum = mhc_pre_gemm_sqrsum_triton(
        test_data["residual"],
        test_data["fn"],
        torch.empty_like(test_data["residual"]),
        torch.empty_like(test_data["residual"][:, 0]),
        n_experts_per_token,
        dhidden,
        token_block=32,
        hidden_block=64,
    )

if __name__ == "__main__":
    main()
