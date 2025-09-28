# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr, y_ptr, output_ptr,
    w_ptr, linear_bias_ptr,
    running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr,
    sum_feature_ptr,
    in_features, out_features,
    stride_x_batch, stride_x_feature,
    stride_y_batch, stride_y_feature,
    stride_w_feature, stride_w_in,
    training: tl.constexpr, 
    eps: tl.constexpr,
    BLOCK_IN: tl.constexpr
):
    pid0 = tl.program_id(0)  # Batch index
    pid1 = tl.program_id(1)  # Feature index

    # Compute linear transformation
    dot = 0.0
    for k in range(0, in_features, BLOCK_IN):
        k_offs = k + tl.arange(0, BLOCK_IN)
        mask = k_offs < in_features
        
        x_val = tl.load(x_ptr + pid0 * stride_x_batch + k_offs * stride_x_feature, mask=mask)
        w_val = tl.load(w_ptr + pid1 * stride_w_feature + k_offs * stride_w_in, mask=mask)
        dot += tl.sum(x_val * w_val, axis=0)
    
    linear_out = dot + tl.load(linear_bias_ptr + pid1)

    # Training vs inference logic
    if training:
        tl.atomic_add(sum_feature_ptr + pid1, linear_out)
        normalized = tl.load(beta_ptr + pid1)
    else:
        running_mean = tl.load(running_mean_ptr + pid1)
        running_var = tl.load(running_var_ptr + pid1)
        gamma = tl.load(gamma_ptr + pid1)
        beta_val = tl.load(beta_ptr + pid1)
        normalized = gamma * ( (linear_out - running_mean) / tl.sqrt(running_var + eps) ) + beta_val

    # Pointwise operations
    y_val = tl.load(y_ptr + pid0 * stride_y_batch + pid1 * stride_y_feature)
    out = (normalized + y_val) * y_val
    tl.store(output_ptr + pid0 * out_features + pid1, out)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, y):
        batch_size, out_features = x.shape[0], self.bmm.out_features
        
        # Ensure contiguous tensors
        x = x.contiguous()
        y = y.contiguous()
        output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
        
        # Setup reduction buffer for training
        if self.training:
            sum_feature = torch.zeros(out_features, device=x.device, dtype=torch.float32)
        else:
            sum_feature = torch.empty(0, device=x.device)  # Dummy tensor
            
        # Launch kernel
        grid = (batch_size, out_features)
        _forward_kernel[grid](
            x, y, output,
            self.bmm.weight, self.bmm.bias,
            self.instance_norm.running_mean, self.instance_norm.running_var,
            self.instance_norm.weight, self.instance_norm.bias,
            sum_feature,
            in_features, out_features,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self.bmm.weight.stride(0), self.bmm.weight.stride(1),
            training=self.training,
            eps=self.eps,
            BLOCK_IN=64
        )
        
        # Update running stats if training
        if self.training:
            mean_feature = sum_feature / batch_size
            self.instance_norm.running_mean = (
                (1 - self.momentum) * self.instance_norm.running_mean + 
                self.momentum * mean_feature
            )
            self.instance_norm.running_var = (
                (1 - self.momentum) * self.instance_norm.running_var
            )  # Variance is zero in this case
            
        return output

batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================