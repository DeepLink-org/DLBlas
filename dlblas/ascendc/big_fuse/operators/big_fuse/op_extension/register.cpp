/**
 * big_fuse TORCH_LIBRARY registration
 * Registers torch.ops.npu.big_fuse with PrivateUse1 (NPU) and Meta backends.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("big_fuse(Tensor residual, Tensor fn_weight, "
          "Tensor mhc_scale, Tensor mhc_base) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("big_fuse", TORCH_FN(ascend_kernel::big_fuse_torch));
}

// Meta backend for torch.compile / fx / aclgraph compatibility
std::vector<at::Tensor> big_fuse_meta(
    const at::Tensor& residual,
    const at::Tensor& fn_weight,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base)
{
    auto res_shape = residual.sizes();  // [B, S, M, H]
    int64_t B  = res_shape[0];
    int64_t S  = res_shape[1];
    int64_t M4 = res_shape[2];
    int64_t HS = res_shape[3];

    auto post_mix = at::empty({B, S, M4, 1},
        residual.options().dtype(at::kFloat));
    auto comb_mix = at::empty({B, S, M4, M4},
        residual.options().dtype(at::kFloat));
    auto layer_input = at::empty({B, S, HS},
        residual.options().dtype(at::kBFloat16));

    return {post_mix, comb_mix, layer_input};
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("big_fuse", &big_fuse_meta);
}

} // namespace
