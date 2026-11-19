// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <torch/extension.h>
#include <torch_sdaa/sdaa_extension.h>

#include <dirent.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "interface/include/tecoops.h"

namespace py = pybind11;

static tecoopsHandle_t g_handle = nullptr;

static int get_env_int(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') return -1;

    char *end = nullptr;
    errno = 0;
    int64_t parsed = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
        parsed > CPU_SETSIZE) {
        return -1;
    }
    return static_cast<int>(parsed);
}

static void add_cpu_mask(const std::string &mask, std::set<int> *cpus) {
    size_t end = mask.size();
    int group_index = 0;
    while (end > 0) {
        size_t begin = mask.rfind(',', end - 1);
        if (begin == std::string::npos) begin = 0;
        else ++begin;

        const std::string group = mask.substr(begin, end - begin);
        char *group_end = nullptr;
        errno = 0;
        uint64_t value = std::strtoull(group.c_str(), &group_end, 16);
        if (errno != 0 || group_end == group.c_str() || *group_end != '\0') {
            return;
        }
        for (int bit = 0; value != 0; ++bit, value >>= 1) {
            if ((value & 1UL) != 0) {
                const int cpu = group_index * 32 + bit;
                if (cpu < CPU_SETSIZE) cpus->insert(cpu);
            }
        }

        ++group_index;
        if (begin == 0) break;
        end = begin - 1;
    }
}

static std::vector<int> get_teco_local_cpus(const cpu_set_t &allowed) {
    std::set<int> local_cpus;
    DIR *devices = opendir("/sys/bus/pci/devices");
    if (devices == nullptr) return {};

    while (dirent *entry = readdir(devices)) {
        const std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        const std::string path = "/sys/bus/pci/devices/" + name;

        std::ifstream vendor_file(path + "/vendor");
        std::string vendor;
        if (!vendor_file || !(vendor_file >> vendor) || vendor != "0x2006") {
            continue;
        }

        std::ifstream cpus_file(path + "/local_cpus");
        std::string mask;
        if (cpus_file && (cpus_file >> mask)) add_cpu_mask(mask, &local_cpus);
    }
    closedir(devices);

    std::vector<int> result;
    result.reserve(local_cpus.size());
    for (const int cpu : local_cpus) {
        if (CPU_ISSET(cpu, &allowed)) result.push_back(cpu);
    }
    return result;
}

static bool get_rank_and_world(int *rank, int *world_size) {
    *rank = get_env_int("LOCAL_RANK");
    const int local_world_size = get_env_int("LOCAL_WORLD_SIZE");
    *world_size = local_world_size >= 0
        ? local_world_size
        : get_env_int("WORLD_SIZE");
    if (*rank >= 0 && *world_size >= 1) return true;

    py::module_ distributed = py::module_::import("torch.distributed");
    if (!distributed.attr("is_initialized")().cast<bool>()) return false;
    if (*rank < 0) *rank = distributed.attr("get_rank")().cast<int>();
    if (*world_size < 0) {
        *world_size = distributed.attr("get_world_size")().cast<int>();
    }
    return *rank >= 0 && *world_size >= 1;
}

static void bind_local_numa_once() {
    static std::atomic<int> state{0};
    if (state.load(std::memory_order_acquire) != 0) return;

    int expected = 0;
    if (!state.compare_exchange_strong(
            expected, 2, std::memory_order_acq_rel)) {
        return;
    }

    const char *mode = std::getenv("TECO_SDAA_TP_AFFINITY");
    if (mode != nullptr &&
        (std::strcmp(mode, "off") == 0 || std::strcmp(mode, "0") == 0)) {
        state.store(1, std::memory_order_release);
        return;
    }

    int local_rank = -1;
    int world_size = -1;
    if (!get_rank_and_world(&local_rank, &world_size)) {
        state.store(0, std::memory_order_release);
        return;
    }
    if (world_size < 2 || local_rank >= world_size) {
        state.store(1, std::memory_order_release);
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        state.store(1, std::memory_order_release);
        return;
    }

    const std::vector<int> local_cpus = get_teco_local_cpus(allowed);
    if (static_cast<int>(local_cpus.size()) < world_size) {
        state.store(1, std::memory_order_release);
        return;
    }

    const int base = static_cast<int>(local_cpus.size()) / world_size;
    const int remainder = static_cast<int>(local_cpus.size()) % world_size;
    const int start = local_rank * base + std::min(local_rank, remainder);
    const int count = base + (local_rank < remainder ? 1 : 0);
    if (count <= 0) {
        state.store(1, std::memory_order_release);
        return;
    }

    cpu_set_t worker_cpus;
    CPU_ZERO(&worker_cpus);
    for (int i = 0; i < count; ++i) CPU_SET(local_cpus[start + i], &worker_cpus);
    sched_setaffinity(0, sizeof(worker_cpus), &worker_cpus);
    state.store(1, std::memory_order_release);
}

static tecoopsHandle_t getGlobalHandle() {
    bind_local_numa_once();
    if (g_handle == nullptr) {
        tecoopsCreate(&g_handle);
    }
    return g_handle;
}

void flatten_rays_torch(torch::Tensor rays, uint32_t N, uint32_t M, torch::Tensor res) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsFlattenRays(handle,
                       rays.data_ptr<int>(),
                       N, M,
                       res.data_ptr<int>(),
                       TECOOPS_ALGO_0);
}

void morton3D_invert_torch(torch::Tensor indices, uint32_t N, torch::Tensor coords) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsMorton3DInvert(handle,
                          indices.data_ptr<int>(),
                          N,
                          coords.data_ptr<int>());
}

void reshape_and_cache_torch(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor slot_mapping,
    torch::Tensor key_cache, torch::Tensor value_cache) {
    tecoopsHandle_t handle = getGlobalHandle();
    int num_tokens = key.size(0);
    int num_kv_heads = key.size(1);
    int head_size = key.size(2);
    int num_blocks = key_cache.size(0);
    int block_size = key_cache.size(2);

    tecoopsReshapeAndCache(
        handle,
        key.data_ptr(), value.data_ptr(),
        slot_mapping.data_ptr<int64_t>(),
        key_cache.data_ptr(), value_cache.data_ptr(),
        num_tokens, num_kv_heads, head_size,
        num_blocks, block_size);
}

void rms_norm_torch(
    torch::Tensor input, torch::Tensor weight,
    c10::optional<torch::Tensor> residual,
    torch::Tensor output, c10::optional<torch::Tensor> residual_out,
    double eps) {
    tecoopsHandle_t handle = getGlobalHandle();
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    const void *residual_ptr = residual.has_value() ? residual.value().data_ptr() : nullptr;
    void *res_out_ptr = residual_out.has_value() ? residual_out.value().data_ptr() : nullptr;

    tecoopsRmsNorm(handle,
                   input.data_ptr(), weight.data_ptr(),
                   residual_ptr, output.data_ptr(), res_out_ptr,
                   num_tokens, hidden_size, static_cast<float>(eps));
}

void flash_attn_varlen_func_torch(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int max_seqlen_q,
    torch::Tensor cu_seqlens_q,
    int max_seqlen_k,
    torch::Tensor cu_seqlens_k,
    torch::Tensor seqused_k,
    double softmax_scale,
    bool causal,
    torch::Tensor window_size,
    torch::Tensor block_table,
    bool return_softmax_lse,
    torch::Tensor out) {

    int batch_size = seqused_k.size(0);
    int max_block_num = k.size(0);

    // A valid cumulative-length array always starts at zero.  Therefore, for
    // the latency-critical single-sequence path, cu_seqlens_q[1] is already
    // the one-element q_seq_lens array expected by tecoopsFlashAttention.
    // Reuse that device value directly and avoid a device-to-host sync,
    // temporary CPU allocation, and host-to-device copy in every layer.
    torch::Tensor q_lens;
    const int *q_lens_ptr = nullptr;
    const bool can_reuse_device_q_len =
        batch_size == 1 &&
        cu_seqlens_q.scalar_type() == torch::kInt32 &&
        cu_seqlens_q.is_contiguous() &&
        cu_seqlens_q.numel() >= 2 &&
        cu_seqlens_q.device() == q.device();
    if (can_reuse_device_q_len) {
        q_lens_ptr = cu_seqlens_q.data_ptr<int>() + 1;
    } else {
        auto cu_seqlens_q_cpu = cu_seqlens_q.cpu();
        auto cu_ptr = cu_seqlens_q_cpu.data_ptr<int>();
        auto q_lens_cpu = torch::empty({batch_size}, torch::kInt32);
        auto ql_cpu_ptr = q_lens_cpu.data_ptr<int>();
        for (int b = 0; b < batch_size; b++) {
            ql_cpu_ptr[b] = cu_ptr[b + 1] - cu_ptr[b];
        }
        q_lens = q_lens_cpu.to("sdaa");
        q_lens_ptr = q_lens.data_ptr<int>();
    }

    // In the valid OneGenomeRice single-sequence decode shape,
    // max_seqlen_k is seqused_k[0] and is positive.  All four specialized
    // d64/b32 dispatches overwrite every one of the 4 x 64 output elements
    // without reading the previous output, so the memset launch is redundant.
    const bool fully_overwritten_onegenome_decode =
        out.defined() &&
        batch_size == 1 &&
        max_seqlen_q == 1 &&
        max_seqlen_k > 0 &&
        q.scalar_type() == torch::kFloat16 &&
        out.scalar_type() == torch::kFloat16 &&
        k.scalar_type() == torch::kFloat16 &&
        v.scalar_type() == torch::kFloat16 &&
        q.is_contiguous() && out.is_contiguous() &&
        k.is_contiguous() && v.is_contiguous() &&
        q.dim() == 3 &&
        q.size(0) == 1 && q.size(1) == 4 && q.size(2) == 64 &&
        out.dim() == 3 &&
        out.size(0) == 1 && out.size(1) == 4 && out.size(2) == 64 &&
        k.dim() == 4 && v.dim() == 4 &&
        k.size(1) == 2 && k.size(2) == 32 && k.size(3) == 64 &&
        v.size(1) == 2 && v.size(2) == 32 && v.size(3) == 64 &&
        block_table.dim() == 2 && block_table.size(0) == 1;

    if (!out.defined()) {
        out = torch::zeros_like(q);
    } else if (!fully_overwritten_onegenome_decode) {
        out.zero_();
    }

    tecoopsHandle_t handle = getGlobalHandle();

    // Descriptors contain dtype and shape only; tensor data pointers are
    // supplied separately to tecoopsFlashAttention on every invocation.  Keep
    // one cache per model-runner thread and refresh it whenever any dimension
    // changes, avoiding five create/set/destroy cycles in every model layer.
    struct DescriptorCache {
        tecoopsTensorDescriptor_t desc{};
        tecoopsDataType_t dtype{};
        std::vector<int> dims;
        bool created = false;
    };
    static thread_local DescriptorCache descriptor_cache[5];
    auto cached_desc = [&](int slot, tecoopsDataType_t dtype,
                           torch::Tensor &t) {
        auto &cache = descriptor_cache[slot];
        int ndim = t.dim();
        bool needs_update =
            !cache.created || cache.dtype != dtype ||
            static_cast<int>(cache.dims.size()) != ndim;
        if (!cache.created) {
            tecoopsCreateTensorDescriptor(&cache.desc);
            cache.created = true;
        }
        if (!needs_update) {
            for (int i = 0; i < ndim; i++) {
                if (cache.dims[i] != t.size(i)) {
                    needs_update = true;
                    break;
                }
            }
        }
        if (needs_update) {
            cache.dims.resize(ndim);
            for (int i = 0; i < ndim; i++) cache.dims[i] = t.size(i);
            tecoopsSetTensorNdDescriptor(
                cache.desc, dtype, ndim, cache.dims.data(), nullptr);
            cache.dtype = dtype;
        }
        return cache.desc;
    };
    auto blockTableDesc = cached_desc(
        0, TECOOPS_DATA_INT32, block_table);
    auto qDataDesc = cached_desc(1, TECOOPS_DATA_HALF, q);
    auto kCacheDesc = cached_desc(2, TECOOPS_DATA_HALF, k);
    auto vCacheDesc = cached_desc(3, TECOOPS_DATA_HALF, v);
    auto oDataDesc = cached_desc(4, TECOOPS_DATA_HALF, out);

    tecoopsFlashAttention(handle,
                          max_seqlen_q, max_seqlen_k, max_block_num,
                          q_lens_ptr, (const int *)seqused_k.data_ptr(),
                          blockTableDesc, block_table.data_ptr(),
                          qDataDesc, q.data_ptr(),
                          kCacheDesc, k.data_ptr(),
                          vCacheDesc, v.data_ptr(),
                          oDataDesc, out.data_ptr(),
                          /*workspace=*/nullptr);
}

void causal_conv1d_fn_torch(
    torch::Tensor out,          // [dim, totalSeqLen]  (modified in-place)
    torch::Tensor conv_states,  // [num_entries, convStateLen, dim]  (modified in-place)
    torch::Tensor x,            // [dim, totalSeqLen]
    torch::Tensor weight,       // [width, dim]
    torch::Tensor query_start_loc,  // [batch+1] int32
    torch::Tensor cache_indices,    // [batch] int32
    torch::Tensor has_initial_state,  // [batch] int8
    int64_t pad_slot_id) {
    tecoopsHandle_t handle = getGlobalHandle();

    int dim = x.size(0);
    int totalSeqLen = x.size(1);
    int batch = query_start_loc.size(0) - 1;
    int width = weight.size(0);
    int convStateStride = conv_states.stride(0);
    int xSeqStride = dim;
    int outSeqStride = dim;

    // C API expects [totalSeqLen, dim]; torch passes [dim, totalSeqLen]
    auto x_t = x.t().contiguous();
    auto out_t = out.t().contiguous();

    tecoopsCausalConv1d(
        handle,
        batch, totalSeqLen, dim, width,
        convStateStride, xSeqStride, outSeqStride,
        pad_slot_id,
        /*api_mode=*/0, /*actMode=*/0,
        x_t.data_ptr(),
        conv_states.data_ptr(),
        weight.data_ptr(),
        out_t.data_ptr(),
        query_start_loc.data_ptr<int>(),
        cache_indices.data_ptr<int>(),
        has_initial_state.data_ptr<int8_t>());

    // Copy result from [totalSeqLen, dim] back to original out [dim, totalSeqLen]
    out.copy_(out_t.t());
}

PYBIND11_MODULE(_torch_ext, m) {
    m.def("flatten_rays", &flatten_rays_torch, "flatten_rays (SDAA)");
    m.def("morton3D_invert", &morton3D_invert_torch, "morton3D_invert (SDAA)");
    m.def("reshape_and_cache", &reshape_and_cache_torch, "reshape_and_cache (SDAA)");
    m.def("rms_norm", &rms_norm_torch, "rms_norm (SDAA)");
    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_torch, "flash_attn_varlen_func (SDAA)");
    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
}
