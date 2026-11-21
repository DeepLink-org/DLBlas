import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu

@triton.jit
def layer_norm_lastdim_kernel(X_ptr, Y_ptr, M, stride, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
  
    row_start = pid * stride
    ptrs = row_start + offs

    x_raw = tl.load(X_ptr + ptrs)          # 无越界，无需 other
    x = x_raw.to(tl.float32)

    invN = 1.0 / BLOCK_SIZE               # 通用写法，此处 = 0.1
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x * invN
    var = sum_x2 * invN - mean * mean

    rstd = tl.rsqrt(var + eps)
    y_f32 = (x - mean) * rstd
    y = y_f32.to(x_raw.dtype)
    tl.store(Y_ptr + ptrs, y)




class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.eps = 1e-5
        self.block_size = 10   # 精确等于归一化维度

    def forward(self, x):
        x_in = x.contiguous()
        assert x_in.dim() == 2 and x_in.size(1) == 10, "Expected input of shape (N, 10)"
        N, M = x_in.shape
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        stride = x_in.stride(0)
        grid = (N,)
        layer_norm_lastdim_kernel[grid](
            x_in, y, M, stride, self.eps,
            BLOCK_SIZE=self.block_size,
            num_warps=1
        )
        return y

def get_inputs():
    x = torch.rand(10, 10, device="npu")
    return [x]


def get_init_inputs():
    return []

if __name__ == "__main__":
      import torch_npu
      torch_npu.npu.set_device(0)
      device = torch.device("npu:0")

      model = ModelNew().to(device)
      inputs = [x.to(device) for x in get_inputs()]
      with torch.no_grad():
          res = model(*inputs)
      print(res.shape)