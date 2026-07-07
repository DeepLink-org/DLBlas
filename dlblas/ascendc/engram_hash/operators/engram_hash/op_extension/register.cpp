/**
 * TORCH_LIBRARY registration for the engram_hash custom NPU operator.
 */
#include <torch/library.h>
#include "ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("engram_hash(Tensor ngram_token_ids, Tensor multipliers, "
          "Tensor vocab_sizes, Tensor offsets) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("engram_hash", TORCH_FN(ascend_kernel::engram_hash));
}

// =========================================================================
// Meta backend (for torch.compile / fx tracing): output shape [L, NT, W].
// =========================================================================
static at::Tensor engram_hash_meta(
    const at::Tensor& ngram_token_ids,
    const at::Tensor& multipliers,
    const at::Tensor& vocab_sizes,
    const at::Tensor& offsets)
{
    const int64_t NT = ngram_token_ids.size(0);
    const int64_t L  = multipliers.size(0);
    const int64_t W  = offsets.size(-1);
    return at::empty({L, NT, W}, ngram_token_ids.options().dtype(at::kInt));
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("engram_hash", &engram_hash_meta);
}
