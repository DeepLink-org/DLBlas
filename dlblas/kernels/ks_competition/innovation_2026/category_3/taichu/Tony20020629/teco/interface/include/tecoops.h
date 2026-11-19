// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#ifndef TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_
#define TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_

#include <sdaa_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TECOOPS_DIM_MAX 8

typedef enum {
    TECOOPS_STATUS_SUCCESS = 0,
    TECOOPS_STATUS_NOT_INITIALIZED = 1,
    TECOOPS_STATUS_ALLOC_FAILED = 2,
    TECOOPS_STATUS_BAD_PARAM = 3,
    TECOOPS_STATUS_INTERNAL_ERROR = 4,
    TECOOPS_STATUS_INVALID_VALUE = 5,
    TECOOPS_STATUS_NOT_SUPPORTED = 6,
} tecoopsStatus_t;

typedef enum {
    TECOOPS_TENSOR_NCHW = 0, /* row major (wStride = 1, hStride = w) */
    TECOOPS_TENSOR_NHWC = 1, /* feature maps interleaved (cStride = 1) */
    TECOOPS_TENSOR_CHWN =
        2, /* each image point is vector of element of C, vector length in data type */
    TECOOPS_TENSOR_NWHC = 3,
    TECOOPS_TENSOR_NCDHW = 4,
    TECOOPS_TENSOR_NDHWC = 5,
    TECOOPS_TENSOR_CDHWN = 6,
} tecoopsTensorFormat_t;

typedef enum {
    TECOOPS_ALGO_0 = 0,
    TECOOPS_ALGO_1,
    TECOOPS_ALGO_2,
    TECOOPS_ALGO_3,
    TECOOPS_ALGO_4,
    TECOOPS_ALGO_5,
    TECOOPS_ALGO_6,
    TECOOPS_ALGO_7,
    TECOOPS_ALGO_8,
    TECOOPS_ALGO_9,
} tecoopsAlgo_t;

typedef enum {
    TECOOPS_DATA_FLOAT = 0,
    TECOOPS_DATA_HALF = 1,
    TECOOPS_DATA_INT8 = 2,
    TECOOPS_DATA_INT16 = 3,
    TECOOPS_DATA_INT32 = 4,
    TECOOPS_DATA_INT64 = 5,
    TECOOPS_DATA_UINT8 = 6,
    TECOOPS_DATA_BOOL = 7,
    TECOOPS_DATA_DOUBLE = 8,
    TECOOPS_DATA_BFLOAT16 = 9,
    TECOOPS_DATA_COMPLEX_FLOAT = 10,
} tecoopsDataType_t;

typedef struct tecoopsContext *tecoopsHandle_t;
typedef struct tecoopsTensorStruct *tecoopsTensorDescriptor_t;

tecoopsStatus_t tecoopsCreate(tecoopsHandle_t *handle);
tecoopsStatus_t tecoopsDestroy(tecoopsHandle_t handle);
tecoopsStatus_t tecoopsSetStream(tecoopsHandle_t handle, sdaaStream_t stream);
tecoopsStatus_t tecoopsGetStream(tecoopsHandle_t handle, sdaaStream_t *stream);
const char *tecoopsGetErrorString(tecoopsStatus_t status);

tecoopsStatus_t tecoopsCreateTensorDescriptor(tecoopsTensorDescriptor_t *tensorDesc);

tecoopsStatus_t tecoopsSetTensor4dDescriptor(tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsTensorFormat_t format,
                                             tecoopsDataType_t dataType,  // image data type
                                             int n,   // number of inputs (batch size)
                                             int c,   // number of input feature maps
                                             int h,   // height of input section
                                             int w);  // width of input section

tecoopsStatus_t tecoopsGetTensor4dDescriptor(const tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsDataType_t* dataType,  // image data type
                                             int* n,  // number of inputs (batch size)
                                             int* c,  // number of input feature maps
                                             int* h,  // height of input section
                                             int* w,  // width of input section
                                             int* nStride, int* cStride, int* hStride,
                                             int* wStride);

tecoopsStatus_t tecoopsSetTensorNdDescriptor(tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsDataType_t dataType, int nbDims,
                                             const int dimA[], const int strideA[]);

tecoopsStatus_t tecoopsGetTensorNdDescriptor(const tecoopsTensorDescriptor_t tensorDesc,
                                             int nbDimsRequested, tecoopsDataType_t* dataType,
                                             int* nbDims, int dimA[], int strideA[]);

tecoopsStatus_t tecoopsDestroyTensorDescriptor(tecoopsTensorDescriptor_t tensorDesc);

tecoopsStatus_t tecoopsFlattenRays(tecoopsHandle_t handle, const int* rays, uint32_t N, uint32_t M,
                                   int* res, tecoopsAlgo_t algo);

tecoopsStatus_t tecoopsMemset(tecoopsHandle_t handle, void* x, const int value, size_t size);

tecoopsStatus_t tecoopsReshapeAndCache(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size);

tecoopsStatus_t tecoopsReduceVariance(const tecoopsHandle_t handle, int axis, int correction,
                                      const tecoopsTensorDescriptor_t xDesc, const void* x,
                                      const tecoopsTensorDescriptor_t yDesc, void* y);

tecoopsStatus_t tecoopsMorton3DInvert(tecoopsHandle_t handle, const int* indices, uint32_t N,
                                      int *coords);

tecoopsStatus_t tecoopsRmsNorm(
    tecoopsHandle_t handle,
    const void *input,
    const void *weight,
    const void *residual,
    void *output,
    void *residual_out,
    int num_tokens,
    int hidden_size,
    float eps);

tecoopsStatus_t tecoopsFlashAttention(tecoopsHandle_t handle,
                                      int max_seqlen_q, int max_seqlen_k,
                                      int max_block_num, const int *q_seq_lens,
                                      const int *kv_seq_lens,
                                      const tecoopsTensorDescriptor_t blockTableDesc,
                                      const void *blockTable,
                                      const tecoopsTensorDescriptor_t qDataDesc,
                                      const void *qData,
                                      const tecoopsTensorDescriptor_t kCacheDesc,
                                      const void *kCache,
                                      const tecoopsTensorDescriptor_t vCacheDesc,
                                      const void *vCache,
                                      const tecoopsTensorDescriptor_t oDataDesc,
                                      void *oData, void *workspace);

tecoopsStatus_t tecoopsCausalConv1d(tecoopsHandle_t handle,
                                    int batch,
                                    int totalSeqLen,
                                    int dim,
                                    int width,
                                    int convStateStride,
                                    int xSeqStride,
                                    int outSeqStride,
                                    int padSlotId,
                                    int api_mode,
                                    int actMode,
                                    const void *x,
                                    void *convState,
                                    const void *weight,
                                    void *out,
                                    const int *queryStartLoc,
                                    const int *convStateIndices,
                                    const int8_t *hasInitialState);

#ifdef __cplusplus
}
#endif

#endif  // TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_