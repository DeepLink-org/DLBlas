#pragma once
#include <torch/torch.h>

namespace ascend_kernel {

/**
 * engram_hash — N-gram embedding index hash.
 *
 *   ngram_token_ids: [NT, N]      int32
 *   multipliers:     [L, N]       int64
 *   vocab_sizes:     [L, N-1, T]  int32
 *   offsets:         [L, W]       int32   (W = (N-1)*T)
 * → output:          [L, NT, W]   int32
 */
at::Tensor engram_hash(
    const at::Tensor& ngram_token_ids,
    const at::Tensor& multipliers,
    const at::Tensor& vocab_sizes,
    const at::Tensor& offsets);

}  // namespace ascend_kernel
