#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("head_compute_mix_fwd(Tensor input_mix, Tensor mhc_scale, "
          "Tensor mhc_base, float mhc_pre_eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("head_compute_mix_fwd",
           TORCH_FN(ascend_kernel::head_compute_mix_fwd_torch));
}

// Meta backend for torch.compile / fx graph capture
at::Tensor head_compute_mix_fwd_meta(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    double mhc_pre_eps)
{
    return at::empty_like(input_mix);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("head_compute_mix_fwd", &head_compute_mix_fwd_meta);
}

} // namespace
