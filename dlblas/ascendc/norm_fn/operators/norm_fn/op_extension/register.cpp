#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("norm_fn(Tensor residual, Tensor mhc_fn, Tensor? mhc_norm_weight, float mhc_norm_eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("norm_fn", TORCH_FN(ascend_kernel::norm_fn_torch));
}

at::Tensor norm_fn_meta(
    const at::Tensor& residual,
    const at::Tensor& mhc_fn,
    const c10::optional<at::Tensor>& mhc_norm_weight,
    double mhc_norm_eps)
{
    // Output shape: (1, 13, 24)
    int64_t mhc_mult = mhc_fn.size(0);
    return at::empty({residual.size(0), residual.size(1), mhc_mult},
                     residual.options().dtype(at::kFloat));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("norm_fn", &norm_fn_meta);
}

} // namespace
