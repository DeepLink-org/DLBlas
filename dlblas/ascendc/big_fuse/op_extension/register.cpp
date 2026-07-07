/**
 * big_fuse TORCH_LIBRARY registration
 * Stub file for PyTorch operator registration.
 */

#include <torch/extension.h>
#include "ops.h"

// Operator signature definition
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("big_fuse(Tensor residual, Tensor fn, Tensor mhc_scale, Tensor mhc_base) -> Tensor[]");
}

// Meta backend registration (for torch.compile / fx)
at::Tensor big_fuse_meta(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base)
{
    return at::empty_like(residual);  // Stub
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("big_fuse", &big_fuse_meta);
}
