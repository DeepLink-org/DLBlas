# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def matmul_clamp_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    scale_factor,
    clamp_min,
    clamp_max,
    input_size,
    hidden_size,
    stride_x,
    stride_weight,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_INPUT: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hid = tl.program_id(1)
    
    offs_hid = pid_hid * BLOCK_SIZE_HIDDEN + tl.arange(0, BLOCK_SIZE_HIDDEN)
    offs_input = tl.arange(0, BLOCK_SIZE_INPUT)
    
    x_ptr += pid_batch * stride_x
    weight_ptr += offs_hid[:, None] * stride_weight + offs_input[None, :]
    bias_ptr += offs_hid
    
    mask_hid = offs_hid < hidden_size
    acc = tl.zeros((BLOCK_SIZE_HIDDEN,), dtype=tl.float32)
    
    for k in range(0, input_size, BLOCK_SIZE_INPUT):
        mask_input = (k + offs_input) < input_size
        x_val = tl.load(x_ptr + k + offs_input, mask=mask_input, other=0.0)
        w_val = tl.load(weight_ptr, mask=mask_input[None, :] & mask_hid[:, None], other=0.0)
        acc += tl.sum(x_val[None, :] * w_val, axis=1)
        weight_ptr += BLOCK_SIZE_INPUT
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr, mask=mask_hid, other=0.0)
        acc += bias_val
    
    acc = acc * (scale_factor * 2.0)
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)
    
    offs_output = pid_batch * hidden_size + offs_hid
    tl.store(output_ptr + offs_output, acc, mask=mask_hid)

@triton.jit
def reduce_logsumexp_mish_kernel(
    input_ptr,
    output_ptr,
    hidden_size,
    stride_input,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    row_start = pid_batch * stride_input
    output_ptr += pid_batch * stride_output
    
    max_val = tl.full((1,), float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    
    offs = tl.arange(0, BLOCK_SIZE)
    for k in range(0, hidden_size, BLOCK_SIZE):
        mask = (k + offs) < hidden_size
        val = tl.load(input_ptr + row_start + k + offs, mask=mask, other=float('-inf'))
        cur_max = tl.max(val, axis=0)
        max_val = tl.maximum(max_val, cur_max)
    
    for k in range(0, hidden_size, BLOCK_SIZE):
        mask = (k + offs) < hidden_size
        val = tl.load(input_ptr + row_start + k + offs, mask=mask, other=0.0)
        exp_val = tl.exp(val - max_val)
        sum_exp += tl.sum(exp_val, axis=0)
    
    log_sum_exp = tl.log(sum_exp) + max_val
    sp = tl.log(1.0 + tl.exp(log_sum_exp))
    tanh_sp = (tl.exp(2 * sp) - 1.0) / (tl.exp(2 * sp) + 1.0)
    mish_val = log_sum_exp * tanh_sp
    result = log_sum_exp * mish_val
    
    tl.store(output_ptr, result)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        batch_size, _ = x.shape
        hidden_size = self.matmul.out_features
        
        intermediate = torch.empty((batch_size, hidden_size), device=x.device, dtype=torch.float32)
        output = torch.empty((batch_size, 1), device=x.device, dtype=torch.float32)
        
        grid1 = (batch_size, triton.cdiv(hidden_size, 32))
        matmul_clamp_kernel[grid1](
            x,
            self.matmul.weight,
            self.matmul.bias if self.matmul.bias is not None else None,
            intermediate,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
            self.matmul.in_features,
            hidden_size,
            x.stride(0),
            self.matmul.weight.stride(0),
            BLOCK_SIZE_HIDDEN=32,
            BLOCK_SIZE_INPUT=32
        )
        
        grid2 = (batch_size,)
        reduce_logsumexp_mish_kernel[grid2](
            intermediate,
            output,
            hidden_size,
            intermediate.stride(0),
            output.stride(0),
            BLOCK_SIZE=128
        )
        
        return output

batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
# =================== EVOLVE-BLOCK-END ===================