import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(x_ptr, y_ptr, n_elements, neg, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Stream through memory via L2 (.cg) as data is used once
    x = tl.load(x_ptr + offsets, mask=mask, other=0, cache_modifier=".cg")

    # Branchless LeakyReLU: y = max(x, 0) + neg * min(x, 0)
    zero = tl.zeros([1], dtype=x.dtype)
    slope = tl.full([1], neg, dtype=x.dtype)
    x_pos = tl.maximum(x, zero)
    x_neg = tl.minimum(x, zero)
    y = x_pos + x_neg * slope

    tl.store(y_ptr + offsets, y, mask=mask, cache_modifier=".cg")


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        use_triton = (
            x.is_cuda
            and x.is_floating_point()
            and (x.dtype in (torch.float16, torch.bfloat16, torch.float32))
            and x.is_contiguous()
            and (not x.requires_grad)  # ensure autograd correctness by falling back
        )
        if not use_triton:
            return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)

        y = torch.empty_like(x)
        n_elements = x.numel()

        # Tuneable block size; good default for H100/H200 memory BW
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _leaky_relu_kernel[grid](
            x, y, n_elements, float(self.negative_slope),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed