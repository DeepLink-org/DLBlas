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

#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <torch/extension.h>
#include <torch_sdaa/sdaa_extension.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "interface/include/tecoops.h"

static tecoopsHandle_t g_handle = nullptr;
static torch::Tensor g_flash_attention_workspace;

namespace {

bool pinSdaaHandlerThreadToIrqSiblingCpu(int device) {
    constexpr int kNuma3FirstIrqSiblingCpu = 96;
    constexpr int kHandlerCpuStride = 2;
    const int handler_cpu =
        kNuma3FirstIrqSiblingCpu + device * kHandlerCpuStride;

    DIR *task_dir = opendir("/proc/self/task");
    if (task_dir == nullptr)
        return false;

    bool pinned = false;
    dirent *entry = nullptr;
    while ((entry = readdir(task_dir)) != nullptr) {
        char *name_end = nullptr;
        int tid = static_cast<int>(std::strtol(entry->d_name, &name_end, 10));
        if (name_end == entry->d_name || *name_end != '\0' || tid <= 0)
            continue;

        char comm_path[64];
        int path_len = std::snprintf(
            comm_path, sizeof(comm_path),
            "/proc/self/task/%d/comm", tid);
        if (path_len <= 0 ||
            path_len >= static_cast<int>(sizeof(comm_path)))
            continue;

        int fd = open(comm_path, O_RDONLY);
        if (fd < 0)
            continue;
        char thread_name[32] = {};
        ssize_t name_len = read(fd, thread_name, sizeof(thread_name) - 1);
        close(fd);
        if (name_len <= 0 ||
            std::strncmp(thread_name, "SdaaHandlerThre", 15) != 0)
            continue;

        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(handler_cpu, &affinity);
        if (sched_setaffinity(
                static_cast<pid_t>(tid), sizeof(affinity), &affinity) == 0) {
            pinned = true;
        }
    }
    closedir(task_dir);
    return pinned;
}

void pinTargetOpThreadToLocalCpu() {
    static thread_local bool worker_pinned = false;
    static thread_local bool handler_pinned = false;
    if (worker_pinned && handler_pinned)
        return;

    int device = -1;
    int device_count = 0;
    if (sdaaGetDevice(&device) != sdaaSuccess || sdaaGetDeviceCount(&device_count) != sdaaSuccess ||
        device_count != 4 || device < 0 || device >= device_count ||
        sysconf(_SC_NPROCESSORS_CONF) != 128) {
        return;
    }

    // On the four-device inference host, NUMA node 2 is local to the SDAA
    // submission path. Spread TP workers across its physical cores.
    constexpr int kNuma2FirstWorkerCpu = 68;
    constexpr int kWorkerCpuStride = 8;
    const int cpu = kNuma2FirstWorkerCpu + device * kWorkerCpuStride;

    if (!worker_pinned) {
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0 ||
            !CPU_ISSET(cpu, &allowed)) {
            return;
        }

        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        worker_pinned =
            sched_setaffinity(0, sizeof(affinity), &affinity) == 0;
        if (worker_pinned) {
            // Reduce Host wake-up coalescing on the target submission thread.
            prctl(PR_SET_TIMERSLACK, 1UL);
        }
    }

    if (!handler_pinned) {
        handler_pinned = pinSdaaHandlerThreadToIrqSiblingCpu(device);
    }
}

}  // namespace

static tecoopsHandle_t getGlobalHandle() {
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
    pinTargetOpThreadToLocalCpu();
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

    pinTargetOpThreadToLocalCpu();

    int batch_size = seqused_k.size(0);
    int max_block_num = k.size(0);

    // A single sequence has length max_seqlen_q, while decode has one query
    // token per sequence. Avoid a synchronous device-host-device round trip
    // on every layer; a null q_seq_lens marks these implicit-length paths.
    // Multi-sequence prefill retains the generic length conversion.
    torch::Tensor q_lens;
    const int *q_lens_ptr = nullptr;
    if (batch_size != 1 && max_seqlen_q != 1) {
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

    if (!out.defined()) {
        out = torch::empty_like(q);
    }

    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsTensorDescriptor_t blockTableDesc, qDataDesc, kCacheDesc, vCacheDesc, oDataDesc;
    auto make_desc = [&](tecoopsTensorDescriptor_t &desc, tecoopsDataType_t dtype,
                         torch::Tensor &t) {
        tecoopsCreateTensorDescriptor(&desc);
        int ndim = t.dim();
        std::vector<int> dims(ndim);
        for (int i = 0; i < ndim; i++) dims[i] = t.size(i);
        tecoopsSetTensorNdDescriptor(desc, dtype, ndim, dims.data(), nullptr);
    };
    make_desc(blockTableDesc, TECOOPS_DATA_INT32, block_table);
    make_desc(qDataDesc, TECOOPS_DATA_HALF, q);
    make_desc(kCacheDesc, TECOOPS_DATA_HALF, k);
    make_desc(vCacheDesc, TECOOPS_DATA_HALF, v);
    make_desc(oDataDesc, TECOOPS_DATA_HALF, out);

    // Preallocate the official GQA2 prefill layout cache during warmup.
    // Generic kernels ignore this internal workspace and retain their
    // established path; the public operator signature is unchanged.
    void *workspace_ptr = nullptr;
    if (q.dim() == 3 && k.dim() == 4 && q.size(1) == 4 &&
        k.size(1) == 2 && q.size(2) == 64 && k.size(3) == 64) {
        const int workspace_tokens = max_seqlen_k > 16384
            ? max_seqlen_k : 16384;
        const int64_t workspace_bytes =
            2LL * k.size(1) * workspace_tokens * k.size(3) *
            static_cast<int64_t>(q.element_size());
        if (!g_flash_attention_workspace.defined() ||
            g_flash_attention_workspace.numel() < workspace_bytes) {
            g_flash_attention_workspace = torch::empty(
                {workspace_bytes}, q.options().dtype(torch::kUInt8));
        }
        workspace_ptr = g_flash_attention_workspace.data_ptr();
    }

    tecoopsFlashAttention(handle,
                          max_seqlen_q, max_seqlen_k, max_block_num,
                          q_lens_ptr, (const int *)seqused_k.data_ptr(),
                          blockTableDesc, block_table.data_ptr(),
                          qDataDesc, q.data_ptr(),
                          kCacheDesc, k.data_ptr(),
                          vCacheDesc, v.data_ptr(),
                          oDataDesc, out.data_ptr(),
                          workspace_ptr);

    tecoopsDestroyTensorDescriptor(blockTableDesc);
    tecoopsDestroyTensorDescriptor(qDataDesc);
    tecoopsDestroyTensorDescriptor(kCacheDesc);
    tecoopsDestroyTensorDescriptor(vCacheDesc);
    tecoopsDestroyTensorDescriptor(oDataDesc);
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
