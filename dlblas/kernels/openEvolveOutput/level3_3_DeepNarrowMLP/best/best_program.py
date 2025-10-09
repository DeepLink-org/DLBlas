# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _linear_relu_forward(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    in_features,
    out_features,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pid_in = tl.program_id(1)
    
    in_offs = pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
    out_offs = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    
    in_mask = in_offs < in_features
    out_mask = out_offs < out_features
    
    acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    
    for i in range(0, tl.cdiv(in_features, BLOCK_SIZE_IN)):
        in_idx = i * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
        in_mask_cur = in_idx < in_features
        
        w_ptrs = w_ptr + (out_offs[:, None] * in_features + in_idx[None, :])
        x_ptrs = x_ptr + in_idx
        
        x = tl.load(x_ptrs, mask=in_mask_cur, other=0.0)
        w = tl.load(w_ptrs, mask=out_mask[:, None] & in_mask_cur[None, :], other=0.0)
        
        acc += tl.sum(x[None, :] * w, axis=1)
    
    if b_ptr is not None:
        b = tl.load(b_ptr + out_offs, mask=out_mask, other=0.0)
        acc += b
    
    acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)

@triton.jit
def _linear_forward(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    in_features,
    out_features,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pid_in = tl.program_id(1)
    
    in_offs = pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
    out_offs = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    
    in_mask = in_offs < in_features
    out_mask = out_offs < out_features
    
    acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    
    for i in range(0, tl.cdiv(in_features, BLOCK_SIZE_IN)):
        in_idx = i * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
        in_mask_cur = in_idx < in_features
        
        w_ptrs = w_ptr + (out_offs[:, None] * in_features + in_idx[None, :])
        x_ptrs = x_ptr + in_idx
        
        x = tl.load(x_ptrs, mask=in_mask_cur, other=0.0)
        w = tl.load(w_ptrs, mask=out_mask[:, None] & in_mask_cur[None, :], other=0.0)
        
        acc += tl.sum(x[None, :] * w, axis=1)
    
    if b_ptr is not None:
        b = tl.load(b_ptr + out_offs, mask=out_mask, other=0.0)
        acc += b
    
    tl.store(out_ptr + out_offs, acc, mask=out_mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        
        current_size = input_size
        for hidden_size in hidden_layer_sizes:
            linear = nn.Linear(current_size, hidden_size)
            self.layers.append(linear)
            self.activations.append('relu')
            current_size = hidden_size
        
        output_layer = nn.Linear(current_size, output_size)
        self.layers.append(output_layer)
        self.activations.append(None)
    
    def forward(self, x):
        BLOCK_SIZE_IN = 128
        BLOCK_SIZE_OUT = 32
        
        for layer, activation in zip(self.layers, self.activations):
            in_features = layer.weight.size(1)
            out_features = layer.weight.size(0)
            
            out = torch.empty(x.size(0), out_features, device=x.device, dtype=x.dtype)
            
            grid = lambda meta: (
                triton.cdiv(out_features, BLOCK_SIZE_OUT),
                triton.cdiv(in_features, BLOCK_SIZE_IN),
            )
            
            if activation == 'relu':
                _linear_relu_forward[grid](
                    x, 
                    layer.weight, 
                    layer.bias,
                    out,
                    in_features,
                    out_features,
                    BLOCK_SIZE_IN,
                    BLOCK_SIZE_OUT
                )
            else:
                _linear_forward[grid](
                    x, 
                    layer.weight, 
                    layer.bias,
                    out,
                    in_features,
                    out_features,
                    BLOCK_SIZE_IN,
                    BLOCK_SIZE_OUT
                )
            x = out
        
        return x

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
# =================== EVOLVE-BLOCK-END ===================