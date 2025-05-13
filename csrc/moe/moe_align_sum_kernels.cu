// modify from: https://github.com/vllm-project/vllm
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace dlblas {
namespace moe {
// taken from
// https://github.com/sgl-project/sglang/commit/cdae77b03dfc6fec3863630550b45bbfc789f957
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];

  const int warp_id = threadIdx.x / 32;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  // Initialize shared_counts for this warp's experts
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  __syncthreads();

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  // Single thread computes cumulative sum and total tokens
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] =
          cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  // Assign expert IDs to blocks
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

// taken from
// https://github.com/sgl-project/sglang/commit/cdae77b03dfc6fec3863630550b45bbfc789f957
template <typename scalar_t>
__global__ void sgl_moe_token_sort_kernel(scalar_t* __restrict__ topk_ids,
                                          int32_t* sorted_token_ids,
                                          int32_t* cumsum_buffer,
                                          size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += DLBLAS_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

}  // namespace moe
}  // namespace dlblas

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor experts_ids,
                              torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(num_experts == 256,
              "moe_align_block_size kernel only supports deepseek v3.");

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::zeros({num_experts + 1}, options_int);

        auto align_kernel =
            dlblas::moe::moe_align_block_size_kernel<scalar_t>;
        align_kernel<<<1, 1024, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
            topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());

        const int block_threads = 256;
        const int num_blocks =
            (topk_ids.numel() + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);
        auto sort_kernel = dlblas::moe::sgl_moe_token_sort_kernel<scalar_t>;
        sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
      });
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const int num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        dlblas::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        dlblas::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        dlblas::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
