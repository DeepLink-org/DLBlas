# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr,          # normalized_x output from instance norm
    y_ptr,          # residual input
    output_ptr,     # result tensor
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    stride_out0, stride_out1,
    out_features, 
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_features
    
    feature_id = offsets % out_features
    batch_id = offsets // out_features
    
    # Load normalized_x and residual y
    x_vals = tl.load(x_ptr + batch_id * stride_x0 + feature_id * stride_x1, mask=mask)
    y_vals = tl.load(y_ptr + batch_id * stride_y0 + feature_id * stride_y1, mask=mask)
    
    # Compute: (normalized_x + y) * y
    tmp = x_vals + y_vals
    out_vals = tmp * y_vals
    
    tl.store(output_ptr + batch_id * stride_out0 + feature_id * stride_out1, out_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
    
    def forward(self, x, y):
        # Compute normalized_x: bmm -> reshape -> instance norm -> reshape back
        normalized_x = self.bmm(x)
        normalized_x = self.instance_norm(
            normalized_x.unsqueeze(1).unsqueeze(1)
        ).squeeze(1).squeeze(1)
        
        batch_size, out_features = normalized_x.shape
        output = torch.empty_like(y)
        
        total_elements = batch_size * out_features
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        
        _forward_kernel[grid](
            normalized_x, y, output,
            normalized_x.stride(0), normalized_x.stride(1),
            y.stride(0), y.stride(1),
            output.stride(0), output.stride(1),
            out_features, batch_size,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================