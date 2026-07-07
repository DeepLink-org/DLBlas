/**
 * engram_hash PyTorch integration — host-side tiling + kernel launch.
 * Compiled by the C++ compiler, linked with the ASC-compiled kernel.
 */
#include "acl/acl.h"
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "../op_kernel/engram_hash_tiling.h"
#include "../op_host/engram_hash_compute_tiling.h"
#include "ops.h"

// Kernel launch alias (compiled by the ASC compiler).
extern "C" void engram_hash_kernel(
    uint32_t blockDim, void* l2ctrl, aclrtStream stream,
    void* ngram, void* mult, void* vocab, void* offsets,
    void* output, void* tiling);

namespace ascend_kernel {

// DAV_2201 Vector core count. Kept as a constant here (matches platform);
// the kernel itself never hardcodes core count (uses tiling.blockNum).
static constexpr uint32_t EH_CORE_NUM = 48;

at::Tensor engram_hash(
    const at::Tensor& ngram_token_ids,
    const at::Tensor& multipliers,
    const at::Tensor& vocab_sizes,
    const at::Tensor& offsets)
{
    TORCH_CHECK(ngram_token_ids.is_privateuseone(), "ngram_token_ids must be on NPU");
    TORCH_CHECK(multipliers.is_privateuseone(),     "multipliers must be on NPU");
    TORCH_CHECK(vocab_sizes.is_privateuseone(),     "vocab_sizes must be on NPU");
    TORCH_CHECK(offsets.is_privateuseone(),         "offsets must be on NPU");

    TORCH_CHECK(ngram_token_ids.scalar_type() == at::kInt,  "ngram_token_ids must be int32");
    TORCH_CHECK(multipliers.scalar_type()     == at::kLong, "multipliers must be int64");
    TORCH_CHECK(vocab_sizes.scalar_type()     == at::kInt,  "vocab_sizes must be int32");
    TORCH_CHECK(offsets.scalar_type()         == at::kInt,  "offsets must be int32");

    TORCH_CHECK(ngram_token_ids.dim() == 2, "ngram_token_ids must be [NT, N]");
    TORCH_CHECK(multipliers.dim()     == 2, "multipliers must be [L, N]");

    auto ng = ngram_token_ids.contiguous();
    auto mu = multipliers.contiguous();
    auto vo = vocab_sizes.contiguous();
    auto of = offsets.contiguous();

    const int64_t NT = ng.size(0);
    const int64_t N  = ng.size(1);
    const int64_t L  = mu.size(0);
    TORCH_CHECK(mu.size(1) == N, "multipliers dim1 must equal ngram size N");

    const int64_t P = (N >= 1) ? (N - 1) : 0;
    // T inferred from offsets width W = P*T (robust even if vocab_sizes is empty).
    const int64_t W = of.size(-1);
    TORCH_CHECK(P == 0 || (W % P) == 0, "offsets width must be divisible by (N-1)");
    const int64_t T = (P > 0) ? (W / P) : 0;

    auto out = at::empty({L, NT, W}, ng.options().dtype(at::kInt));
    if (W == 0 || NT == 0 || L == 0) return out;

    EngramHashTilingData td;
    ComputeEngramHashTiling((uint32_t)NT, (uint32_t)N, (uint32_t)L, (uint32_t)T,
                            EH_CORE_NUM, td);

    EngramHashTilingData* tdDev = nullptr;
    aclrtMalloc((void**)&tdDev, sizeof(EngramHashTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tdDev, sizeof(EngramHashTilingData), &td,
                sizeof(EngramHashTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    engram_hash_kernel(td.blockNum, nullptr, aclStream,
        ng.data_ptr(), mu.data_ptr(), vo.data_ptr(), of.data_ptr(),
        out.data_ptr(), tdDev);

    aclrtFree(tdDev);
    return out;
}

}  // namespace ascend_kernel
