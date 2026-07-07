// MTPBlock PyTorch Extension - register.cpp
#include <torch/torch.h>
#include "ops.h"

// ============================================================================
// K4: hc_post 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(mtpblock, m) {
    m.def("hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
}

TORCH_LIBRARY_IMPL(mtpblock, PrivateUse1, m) {
    m.impl("hc_post", TORCH_FN(ascend_kernel::mtpblock_hc_post));
}

at::Tensor mtpblock_hc_post_meta(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    // output shape = residual shape [b, s, hc, d]
    return at::empty_like(residual);
}

TORCH_LIBRARY_IMPL(mtpblock, Meta, m) {
    m.impl("hc_post", &mtpblock_hc_post_meta);
}
