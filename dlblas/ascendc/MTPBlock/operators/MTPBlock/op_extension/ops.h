// MTPBlock PyTorch Extension - ops.h
#pragma once
#include <torch/torch.h>

namespace ascend_kernel {

at::Tensor mtpblock_hc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb);

} // namespace ascend_kernel
