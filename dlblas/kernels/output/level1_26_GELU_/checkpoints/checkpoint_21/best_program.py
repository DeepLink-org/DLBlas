import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
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
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x*x*x)))
    # Constants
    a = 0.5
    b = 0.7978845608028654   # sqrt(2/pi)
    c = 0.044715
    x_cubed = x * x * x
    inner = b * (x + c * x_cubed)
    tanh_inner = tl.tanh(inner)
    output = a * x * (1 + tanh_inner)
    tl.store(output_ptr + offsets, output, mask=mask)