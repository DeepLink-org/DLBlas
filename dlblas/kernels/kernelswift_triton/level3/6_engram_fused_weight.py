import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _bf16_mul_to_f32_kernel(a_ptr, b_ptr, out_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)

    # Hints to help the compiler generate better memory accesses
    tl.multiple_of(start, BLOCK)
    tl.max_contiguous(offsets, BLOCK)

    # Fast path for full tiles: avoid mask overhead
    full = start + BLOCK <= n_elements
    if full:
        a_bf16 = tl.load(a_ptr + offsets)
        b_bf16 = tl.load(b_ptr + offsets)
        a = a_bf16.to(tl.float32)
        b = b_bf16.to(tl.float32)
        c = a * b
        tl.store(out_ptr + offsets, c)
    else:
        mask = offsets < n_elements
        a_bf16 = tl.load(a_ptr + offsets, mask=mask, other=0)
        b_bf16 = tl.load(b_ptr + offsets, mask=mask, other=0)
        a = a_bf16.to(tl.float32)
        b = b_bf16.to(tl.float32)
        c = a * b
        tl.store(out_ptr + offsets, c, mask=mask)


class fused_weight(nn.Module):
  def __init__(self, hc_mult: int, hidden_size: int):
      super().__init__()
      self.hc_mult = hc_mult
      self.hidden_size = hidden_size

  def forward(self, wh_data, we_data):
      # Fallback to PyTorch if tensors are not on CUDA
      if not (wh_data.is_cuda and we_data.is_cuda):
          return wh_data.float() * we_data.float()

      # Ensure contiguous layout for coalesced memory access
      a = wh_data.contiguous()
      b = we_data.contiguous()
      assert a.numel() == b.numel(), "Input tensors must have the same number of elements"

      n_elements = a.numel()
      # Very small tensors: PyTorch kernel launch overhead is lower; keep semantics identical
      if n_elements == 0:
          return torch.empty_like(a, dtype=torch.float32)

      out = torch.empty_like(a, dtype=torch.float32)

      # Tune block size and kernel launch params for better latency on small inputs and throughput on larger ones
      if n_elements <= 512:
          block_size = 512
          num_warps = 2
          num_stages = 1
      elif n_elements <= 16384:
          block_size = 1024
          num_warps = 4
          num_stages = 2
      else:
          block_size = 4096
          num_warps = 8
          num_stages = 2

      grid = (triton.cdiv(n_elements, block_size),)
      _bf16_mul_to_f32_kernel[grid](a, b, out, n_elements, BLOCK=block_size, num_warps=num_warps, num_stages=num_stages)
      return out


hc_mult = 4
hidden_size = 128


def generate_test_data(hc_mult, hidden_size):
  wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
  we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
  return [wh_data, we_data]


def get_inputs():
  wh_data, we_data = generate_test_data(hc_mult, hidden_size)
  return [wh_data, we_data]


def get_init_inputs():
  return [hc_mult, hidden_size]