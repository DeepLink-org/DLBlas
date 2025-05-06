#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "dlblas_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "n_share_experts_fusion, float routed_scaling_factor) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kCUDA, &moe_fused_gate);
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor! input, Tensor output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // temporarily adapted from
  // https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
  m.def(
      "sgl_moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                         int block_size, Tensor! sorted_token_ids,"
      "                         Tensor! experts_ids,"
      "                         Tensor! num_tokens_post_pad) -> ()");
  m.impl("sgl_moe_align_block_size", torch::kCUDA, &sgl_moe_align_block_size);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
