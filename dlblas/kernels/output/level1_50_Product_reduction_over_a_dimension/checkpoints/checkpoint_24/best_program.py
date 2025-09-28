# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _prod_reduction_kernel(
    x_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    stride_batch,
    stride_reduction,
    stride_col,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    total_output_elements = batch_size * dim2
    if pid >= total_output_elements:
        return
    
    pid_batch = pid // dim2
    pid_col = pid % dim2
    
    base = pid_batch * stride_batch + pid_col * stride_col
    r_offsets = tl.arange(0, BLOCK_R)
    offsets = base + r_offsets * stride_reduction
    mask = r_offsets < dim1
    data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for numerical stability
    data = data.to(tl.float32)
    
    # Compute log-sum-exp for product
    abs_data = tl.abs(data)
    log_abs = tl.log(abs_data + 1e-12)
    log_sum = tl.sum(log_abs, axis=0)
    abs_result = tl.exp(log_sum)
    
    # Compute sign
    sign = tl.where(tl.sum(tl.where(data < 0, 1, 0), axis=0) % 2 == 1, -1.0, 1.0)
    result = sign * abs_result
    
    # Convert back to original dtype
    result_orig = result.to(x_ptr.dtype.element_ty)
    tl.store(output_ptr + pid, result_orig)

class ModelNew(nn.Module):
    """
    Simple model that performs product reduction over a dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        if self.dim != 1 or x.dim() != 3:
            return torch.prod(x, dim=self.dim)
            
        batch_size, dim1, dim2 = x.shape
        output = torch.empty(batch_size, dim2, device=x.device, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
            
        stride_batch = x.stride(0)
        stride_reduction = x.stride(1)
        stride_col = x.stride(2)
        
        total_elements = batch_size * dim2
        BLOCK_R = triton.next_power_of_2(dim1)
        
        if BLOCK_R > 1024:
            return torch.prod(x, dim=self.dim)
            
        grid = (total_elements,)
        _prod_reduction_kernel[grid](
            x, output,
            batch_size, dim1, dim2,
            stride_batch, stride_reduction, stride_col,
            BLOCK_R=BLOCK_R,
        )
        
        return output

batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduction_dim]
# =================== EVOLVE-BLOCK-END ===================