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
#include <torch/csrc/autograd/python_variable.h>
#include <torch_sdaa/sdaa_extension.h>

#include <array>

#include "interface/include/tecoops.h"

static tecoopsHandle_t g_handle = nullptr;

struct CachedTensorDescriptor {
    tecoopsTensorDescriptor_t desc = nullptr;
    tecoopsDataType_t dtype = TECOOPS_DATA_FLOAT;
    int ndim = -1;
    std::array<int, TECOOPS_DIM_MAX> dims{};
};

struct FlashAttentionDescriptorCache {
    CachedTensorDescriptor block_table;
    CachedTensorDescriptor q;
    CachedTensorDescriptor k;
    CachedTensorDescriptor v;
    CachedTensorDescriptor out;
};

static thread_local FlashAttentionDescriptorCache g_flash_desc_cache;

static tecoopsTensorDescriptor_t getCachedTensorDescriptor(
    CachedTensorDescriptor &cached, tecoopsDataType_t dtype,
    const torch::Tensor &tensor) {
    int ndim = tensor.dim();
    bool shape_changed = cached.ndim != ndim || cached.dtype != dtype;
    for (int i = 0; i < ndim && !shape_changed; i++) {
        shape_changed = cached.dims[i] != tensor.size(i);
    }

    if (cached.desc == nullptr) {
        tecoopsCreateTensorDescriptor(&cached.desc);
        shape_changed = true;
    }
    if (shape_changed) {
        cached.dtype = dtype;
        cached.ndim = ndim;
        for (int i = 0; i < ndim; i++) {
            cached.dims[i] = tensor.size(i);
        }
        tecoopsSetTensorNdDescriptor(
            cached.desc, dtype, ndim, cached.dims.data(), nullptr);
    }
    return cached.desc;
}

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

static void reshape_and_cache_impl(
    const torch::Tensor &key, const torch::Tensor &value,
    const torch::Tensor &slot_mapping,
    const torch::Tensor &key_cache, const torch::Tensor &value_cache) {
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

static PyObject *reshape_and_cache_fastcall(
    PyObject *, PyObject *const *objects, Py_ssize_t nargs) {
    if (__builtin_expect(nargs != 5, 0)) {
        PyErr_Format(
            PyExc_TypeError,
            "reshape_and_cache() takes 5 positional arguments but %zd were given",
            nargs);
        return nullptr;
    }
    for (int index = 0; index < 5; index++) {
        if (__builtin_expect(!THPVariable_Check(objects[index]), 0)) {
            PyErr_Format(
                PyExc_TypeError,
                "reshape_and_cache() argument %d must be a torch.Tensor",
                index + 1);
            return nullptr;
        }
    }
    try {
        reshape_and_cache_impl(
            THPVariable_Unpack(objects[0]),
            THPVariable_Unpack(objects[1]),
            THPVariable_Unpack(objects[2]),
            THPVariable_Unpack(objects[3]),
            THPVariable_Unpack(objects[4]));
    } catch (pybind11::error_already_set &error) {
        error.restore();
        return nullptr;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyMethodDef kReshapeAndCacheFastCallMethod = {
    "reshape_and_cache",
    // CPython dispatches this pointer using the METH_FASTCALL signature.
    reinterpret_cast<PyCFunction>(
        reinterpret_cast<void (*)()>(&reshape_and_cache_fastcall)),
    METH_FASTCALL,
    "reshape_and_cache (SDAA)",
};

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
    torch::Tensor /*cu_seqlens_k*/,
    torch::Tensor seqused_k,
    double /*softmax_scale*/,
    bool /*causal*/,
    torch::Tensor block_table,
    bool /*return_softmax_lse*/,
    torch::Tensor out) {

    // q is flattened as [total_q, num_heads, head_size].  The batch
    // dimension is encoded by cu_seqlens_q, not by q.size(0).
    int batch_size = cu_seqlens_q.size(0) - 1;
    int max_block_num = k.size(0);

    // q_seq_lens is the adjacent difference of cu_seqlens_q. For the
    // latency-critical batch-1 path cu_seqlens_q is [0, q_len], so its second
    // element can be consumed directly on device. This avoids a device-to-host
    // synchronization, a CPU allocation, and a host-to-device copy on every
    // attention layer.
    torch::Tensor q_lens;
    const int *q_lens_ptr = nullptr;
    if (batch_size == 1) {
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

    if (!out.defined()) {
        out = torch::empty_like(q);
    }

    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsTensorDescriptor_t blockTableDesc =
        getCachedTensorDescriptor(
            g_flash_desc_cache.block_table,
            TECOOPS_DATA_INT32, block_table);
    tecoopsTensorDescriptor_t qDataDesc =
        getCachedTensorDescriptor(
            g_flash_desc_cache.q, TECOOPS_DATA_HALF, q);
    tecoopsTensorDescriptor_t kCacheDesc =
        getCachedTensorDescriptor(
            g_flash_desc_cache.k, TECOOPS_DATA_HALF, k);
    tecoopsTensorDescriptor_t vCacheDesc =
        getCachedTensorDescriptor(
            g_flash_desc_cache.v, TECOOPS_DATA_HALF, v);
    tecoopsTensorDescriptor_t oDataDesc =
        getCachedTensorDescriptor(
            g_flash_desc_cache.out, TECOOPS_DATA_HALF, out);

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

// Keep accepting the legacy positional window_size argument.
void flash_attn_varlen_func_compat_torch(
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
    torch::Tensor /*window_size*/,
    torch::Tensor block_table,
    bool return_softmax_lse,
    torch::Tensor out) {
    flash_attn_varlen_func_torch(
        q, k, v,
        max_seqlen_q, cu_seqlens_q,
        max_seqlen_k, cu_seqlens_k,
        seqused_k, softmax_scale, causal,
        block_table, return_softmax_lse, out);
}

void causal_conv1d_fn_torch(
    torch::Tensor out,          // [dim, totalSeqLen]  (modified in-place)
    torch::Tensor conv_states,  // [num_entries, convStateLen, dim]  (modified in-place)
    torch::Tensor x,            // [dim, totalSeqLen]
    torch::Tensor weight,       // [width, dim]
    torch::Tensor query_start_loc,  // [batch+1] int32
    torch::Tensor cache_indices,    // [batch] int32
    torch::Tensor has_initial_state, // [batch] int8
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
    m.add_object(
        "reshape_and_cache",
        pybind11::reinterpret_steal<pybind11::object>(
            PyCFunction_NewEx(
                &kReshapeAndCacheFastCallMethod, nullptr, nullptr)));
    m.def("rms_norm", &rms_norm_torch, "rms_norm (SDAA)");
    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_torch, "flash_attn_varlen_func (SDAA)");
    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_compat_torch, "flash_attn_varlen_func compatibility ABI (SDAA)");
    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
}
