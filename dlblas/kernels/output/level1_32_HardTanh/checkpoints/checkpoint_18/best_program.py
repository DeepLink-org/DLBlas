import triton
import triton.language as tl

@triton.jit
def hard_tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    VEC_SIZE = 4  # Process 4 elements per vector operation
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply HardTanh with vectorized operations
    x_vec = tl.reshape(x, (BLOCK_SIZE // VEC_SIZE, VEC_SIZE))
    # Vectorized min/max operations
    clamped_vec = tl.minimum(tl.maximum(x_vec, -1.0), 1.0)
    clamped = tl.reshape(clamped_vec, (BLOCK_SIZE,))
    
    tl.store(output_ptr + offsets, clamped, mask=mask)