# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _max_pool_1d_kernel(
    x_ptr,
    output_ptr,
    indices_ptr,
    input_length,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    features,
    output_length,
    return_indices: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_output_elements = batch_size * features * output_length
    if pid >= total_output_elements:
        return

    idx = pid
    output_index = idx % output_length
    idx = idx // output_length
    feature_index = idx % features
    batch_index = idx // features

    window_start = output_index * stride - padding
    max_value = -float('inf')
    max_index = -1

    for i in range(0, kernel_size):
        pos = window_start + i * dilation
        if pos < 0 or pos >= input_length:
            value = -float('inf')
        else:
            offset = batch_index * features * input_length + feature_index * input_length + pos
            value = tl.load(x_ptr + offset)

        if value > max_value:
            max_value = value
            max_index = pos

    tl.store(output_ptr + pid, max_value)

    if return_indices:
        tl.store(indices_ptr + pid, max_index)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_contiguous = x.contiguous()
        batch_size, features, input_length = x_contiguous.shape
        stride = self.stride

        L_out = ((input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // stride) + 1

        output = torch.empty((batch_size, features, L_out), device=x.device, dtype=x.dtype)
        if self.return_indices:
            indices = torch.empty((batch_size, features, L_out), device=x.device, dtype=torch.long)
        else:
            indices = None

        total_elements = batch_size * features * L_out

        if total_elements == 0:
            if self.return_indices:
                return output, indices
            else:
                return output

        grid = lambda meta: (total_elements,)
        _max_pool_1d_kernel[grid](
            x_contiguous,
            output,
            indices if self.return_indices else 0,
            input_length,
            self.kernel_size,
            stride,
            self.padding,
            self.dilation,
            batch_size,
            features,
            L_out,
            self.return_indices,
        )

        if self.return_indices:
            return output, indices
        else:
            return output

batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
# =================== EVOLVE-BLOCK-END ===================