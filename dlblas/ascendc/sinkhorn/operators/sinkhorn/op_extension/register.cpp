#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sinkhorn_normalize(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sinkhorn_normalize", TORCH_FN(ascend_kernel::sinkhorn_torch));
}

at::Tensor sinkhorn_meta(const at::Tensor& x)
{
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sinkhorn_normalize", &sinkhorn_meta);
}

} // namespace
