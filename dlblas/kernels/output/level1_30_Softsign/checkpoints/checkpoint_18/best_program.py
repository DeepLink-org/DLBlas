import triton
import triton.language as tl

@triton.jit
def softsign_kernel(
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
    abs_x = tl.abs(x)
    one = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    # If the input is float16, we do the computation in float32? Actually, we can use the same dtype as x.
    # But note: the kernel must be generic. We can use the type of x_ptr? We can let Triton handle it by using the same dtype.

    # We'll do: output = x / (1 + abs_x)
    denominator = one + abs_x
    output = x / denominator
    tl.store(output_ptr + offsets, output, mask=mask)