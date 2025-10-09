import triton
import triton.language as tl

@triton.jit
def _leaky_relu_kernel(
    x_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute LeakyReLU: x >= 0 ? x : negative_slope * x
    # Note: negative_slope is passed as a scalar (constant for the kernel)
    # We use tl.where(condition, x, y) -> condition is x>=0? But note: condition must be element-wise.
    # Alternatively, we can do: 
    #   zero = 0.0
    #   result = tl.where(x >= zero, x, negative_slope * x)
    # But we can avoid the multiplication for positive values? It's pointwise so it's cheap.

    # Alternatively, we can use: 
    #   result = x * tl.where(x >= 0, 1.0, negative_slope)
    # But note: tl.where returns a tensor of the same shape.

    # We'll do:
    #   condition = (x >= 0)
    #   scale = tl.where(condition, 1.0, negative_slope)
    #   output = x * scale
    # But note: negative_slope is a scalar, so we can broadcast.

    # However, a simpler way is to use:
    output = tl.where(x >= 0, x, negative_slope * x)
    tl.store(output_ptr + offsets, output, mask=mask)