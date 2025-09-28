import triton
import triton.language as tl

@triton.jit
def hard_tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Apply HardTanh: clamp between -1 and 1
    # We can use tl.minimum and tl.maximum
    # Alternatively, we can write: 
    #   x = tl.where(x < -1.0, -1.0, x)
    #   x = tl.where(x > 1.0, 1.0, x)
    # But note: using min and max might be faster.
    #   x = tl.maximum(tl.minimum(x, 1.0), -1.0)
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    tl.store(output_ptr + offsets, x, mask=mask)