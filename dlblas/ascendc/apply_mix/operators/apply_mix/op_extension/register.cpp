#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// Register operator signature
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("apply_mix(Tensor x, Tensor mix) -> Tensor");
}

// NPU backend implementation
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("apply_mix", TORCH_FN(ascend_kernel::apply_mix_torch));
}

// Meta backend (for torch.compile / fx)
at::Tensor apply_mix_meta(const at::Tensor& x, const at::Tensor& mix)
{
    // x: [n0, n1, mhc, h], mix: [n0, n1, mhc, 1]
    // output: [n0, n1, h]
    auto n0 = x.size(0);
    auto n1 = x.size(1);
    auto h = x.size(3);
    return at::empty({n0, n1, h}, x.options());
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("apply_mix", &apply_mix_meta);
}

} // namespace
