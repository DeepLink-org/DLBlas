# -*- coding: utf-8 -*-
import torch_npu
import torch
import torch.nn as nn

# Triton仅CUDA平台可用，NPU环境自动降级PyTorch原生实现
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def _prefix_softmax_sum_kernel(
        x_ptr, out_ptr,
        S0, S1,
        stride_x0, stride_x1, stride_x2,
        stride_out0, stride_out1,
        B, K,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        i = pid // S1
        j = pid % S1

        row_x_ptr = x_ptr + i * stride_x0 + j * stride_x1
        row_out_ptr = out_ptr + i * stride_out0 + j * stride_out1

        m = tl.full((1,), -float('inf'), dtype=tl.float32)
        denom = tl.zeros((1,), dtype=tl.float32)
        numer = tl.zeros((1,), dtype=tl.float32)

        for start in range(0, B, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < B
            ptrs = row_x_ptr + offs * stride_x2
            x = tl.load(ptrs, mask=mask, other=-float('inf'))
            x = x.to(tl.float32)

            block_max = tl.max(x, axis=0)
            new_m = tl.maximum(m, block_max)
            scale = tl.exp(m - new_m)
            e = tl.exp(x - new_m)
            e = tl.where(mask, e, 0.0)

            denom = denom * scale + tl.sum(e, axis=0)
            prefix_mask = (offs < K) & mask
            numer = numer * scale + tl.sum(tl.where(prefix_mask, e, 0.0), axis=0)
            m = new_m

        out_val = numer / denom
        tl.store(row_out_ptr, out_val)


# 标准类名 Model
class Model(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        no_bins = int(no_bins)
        edges = torch.linspace(min_bin, max_bin, no_bins + 1, dtype=torch.float32)
        # in_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_centers = (torch.tensor(0.5, dtype=torch.float32) * (edges[:-1] + edges[1:])).to(torch.float32)
        thres_idx_val = int((bin_centers < thres).sum().item())

        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("thres_idx", torch.tensor(thres_idx_val, dtype=torch.long))

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        B = distogram_logits.shape[-1]
        K = self.thres_idx

        # 去掉python if分支，用张量掩码，消除tensor转bool警告
        row_max = torch.max(distogram_logits, dim=-1, keepdim=True)[0]
        x_shift = distogram_logits - row_max
        exp_all = torch.exp(x_shift)
        total_sum = exp_all.sum(dim=-1, keepdim=True)
        prefix_sum = exp_all.narrow(-1, 0, K).sum(dim=-1)
        total_sum = total_sum.squeeze(-1)
        contact_prob = prefix_sum / total_sum

        # 等效边界兜底：K<=0填0；K>=B填1
        mask_zero = (K <= 0)
        mask_one = (K >= B)
        contact_prob = torch.where(mask_zero, torch.zeros_like(contact_prob), contact_prob)
        contact_prob = torch.where(mask_one, torch.ones_like(contact_prob), contact_prob)
        return contact_prob


# 全局超参
N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0

# 标准初始化入参函数
def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]

# 标准前向输入样例函数
def get_inputs():
    device = "npu:0"
    torch.manual_seed(42)
    distogram_logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)
    return [distogram_logits]


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    dev = torch.device("npu:0")
    # 初始化模型
    model = Model(*get_init_inputs()).to(dev)
    inputs = get_inputs()
    # 使用trace规避script源码读取报错
    traced_model = torch.jit.trace(model, inputs)

    # NPU混合精度推理
    with torch.npu.amp.autocast(dtype=torch.float16):
        output = traced_model(*inputs)

    print("Output shape:", output.shape)
    print(output)