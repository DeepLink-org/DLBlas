import triton
import triton.language as tl

@triton.jit
def relu_kernel(
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
    zero = tl.zeros_like(x)
    output = tl.where(x > 0, x, zero)
    tl.store(output_ptr + offsets, output, mask=mask)