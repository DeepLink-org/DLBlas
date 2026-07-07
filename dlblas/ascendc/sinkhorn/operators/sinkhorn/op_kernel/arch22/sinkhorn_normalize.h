/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize.h
 * @brief Sinkhorn Normalize kernel class (arch22 / DAV_2201 / Ascend 910B2)
 *
 * Algorithm per 4x4 matrix (all iterations in UB):
 *   Step A: softmax(dim=-1) + eps
 *   Step B: column normalize (sum over dim=-2)
 *   Repeat (repeat-1) times:
 *     Step C1: row normalize (sum over dim=-1)
 *     Step C2: column normalize (sum over dim=-2)
 *
 * UB Layout (padded row-major):
 *   Each 4x4 matrix: 4 rows x 8 padded elements = 32 floats = 128B
 *   Row r: [c0, c1, c2, c3, pad, pad, pad, pad]  (32B aligned)
 */

#ifndef SINKHORN_NORMALIZE_KERNEL_H
#define SINKHORN_NORMALIZE_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sinkhorn_normalize_tiling_data.h"
#include "sinkhorn_normalize_tiling_key.h"

namespace NsSinkhornNormalize {

using namespace AscendC;

// Constants
constexpr int64_t ELEMS_PER_MATRIX        = 16;
constexpr int64_t PADDED_ROW_LEN          = 8;
constexpr int64_t PADDED_ELEMS_PER_MATRIX = 32;
constexpr int64_t MATRIX_ROWS             = 4;
constexpr int64_t MATRIX_COLS             = 4;
constexpr int64_t TRANSPOSE_BUF_SIZE      = 32;
constexpr int64_t TRANSPOSE_STRIDE        = 8;
constexpr int64_t REDUCE_TMP_BUF_SIZE     = 2048;
constexpr int64_t SCALAR_BUF_SIZE         = 8;

template <typename T>
class SinkhornNormalize {
public:
    __aicore__ inline SinkhornNormalize() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SinkhornNormalizeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    __aicore__ inline void SoftmaxRow(LocalTensor<T>& workLocal, int64_t rowOff,
                                       LocalTensor<T>& scalarLocal,
                                       LocalTensor<T>& reduceTmpLocal);
    __aicore__ inline void AddEps(LocalTensor<T>& workLocal, int64_t matBase, T eps);
    __aicore__ inline void TransposeToCol(LocalTensor<T>& transpLocal,
                                           LocalTensor<T>& workLocal, int64_t matBase);
    __aicore__ inline void TransposeFromCol(LocalTensor<T>& workLocal,
                                             LocalTensor<T>& transpLocal, int64_t matBase);
    __aicore__ inline void RowNormalize(LocalTensor<T>& workLocal, int64_t matBase,
                                         T eps, LocalTensor<T>& scalarLocal,
                                         LocalTensor<T>& reduceTmpLocal);
    __aicore__ inline void ColNormalize(LocalTensor<T>& transpLocal,
                                         T eps, LocalTensor<T>& scalarLocal,
                                         LocalTensor<T>& reduceTmpLocal);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> flatBuf;
    TBuf<TPosition::VECCALC> transpBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    int64_t totalMatrices_ = 0;
    int64_t matricesPerCore_ = 0;
    int64_t matrixStartOffset_ = 0;
    int64_t repeat_ = 10;
    float eps_ = 1e-6f;
};

// =========================================================================
// Init
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::Init(
    GM_ADDR input, GM_ADDR output,
    const SinkhornNormalizeTilingData* tilingData)
{
    totalMatrices_ = tilingData->total_matrices;
    repeat_ = tilingData->repeat;
    eps_ = tilingData->eps;

    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t basePerCore = tilingData->matrices_per_core_base;
    int64_t rem = tilingData->remainder;
    matricesPerCore_ = basePerCore + (blockIdx < rem ? 1 : 0);
    matrixStartOffset_ = blockIdx * basePerCore + (blockIdx < rem ? blockIdx : rem);

    if (matricesPerCore_ == 0) return;

    int64_t workBufElems = matricesPerCore_ * PADDED_ELEMS_PER_MATRIX;
    int64_t flatBufElems = matricesPerCore_ * ELEMS_PER_MATRIX;

    pipe.InitBuffer(workBuf, workBufElems * sizeof(T));
    pipe.InitBuffer(flatBuf, flatBufElems * sizeof(T));
    pipe.InitBuffer(transpBuf, TRANSPOSE_BUF_SIZE * sizeof(T));
    pipe.InitBuffer(scalarBuf, SCALAR_BUF_SIZE * sizeof(T));
    pipe.InitBuffer(reduceTmpBuf, REDUCE_TMP_BUF_SIZE * sizeof(T));

    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
}

// =========================================================================
// Process
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::Process()
{
    if (matricesPerCore_ == 0) return;
    CopyIn();
    Compute();
    CopyOut();
}

// =========================================================================
// CopyIn
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::CopyIn()
{
    LocalTensor<T> flatLocal = flatBuf.Get<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();

    int64_t gmOffset = matrixStartOffset_ * ELEMS_PER_MATRIX;
    int64_t flatElems = matricesPerCore_ * ELEMS_PER_MATRIX;

    DataCopyParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint32_t>(flatElems / AscendCUtils::GetC0Count(sizeof(T)));
    DataCopy(flatLocal, inputGlobal[gmOffset], loadParams);

    for (int64_t m = 0; m < matricesPerCore_; m++) {
        int64_t ubBase = m * PADDED_ELEMS_PER_MATRIX;
        int64_t gmBase = m * ELEMS_PER_MATRIX;
        for (int64_t r = 0; r < MATRIX_ROWS; r++) {
            int64_t ubR = ubBase + r * PADDED_ROW_LEN;
            int64_t gmR = gmBase + r * MATRIX_COLS;
            workLocal.SetValue(ubR + 0, flatLocal.GetValue(gmR + 0));
            workLocal.SetValue(ubR + 1, flatLocal.GetValue(gmR + 1));
            workLocal.SetValue(ubR + 2, flatLocal.GetValue(gmR + 2));
            workLocal.SetValue(ubR + 3, flatLocal.GetValue(gmR + 3));
        }
    }
}

// =========================================================================
// Compute
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::Compute()
{
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> transpLocal = transpBuf.Get<T>();
    LocalTensor<T> scalarLocal = scalarBuf.Get<T>();
    LocalTensor<T> reduceTmpLocal = reduceTmpBuf.Get<T>();

    T eps = static_cast<T>(eps_);

    for (int64_t m = 0; m < matricesPerCore_; m++) {
        int64_t matBase = m * PADDED_ELEMS_PER_MATRIX;

        // Step A: Softmax per row
        for (int64_t r = 0; r < MATRIX_ROWS; r++) {
            int64_t rowOff = matBase + r * PADDED_ROW_LEN;
            SoftmaxRow(workLocal, rowOff, scalarLocal, reduceTmpLocal);
        }

        // Step B: Add epsilon
        AddEps(workLocal, matBase, eps);

        // Step C: Column normalize via transpose
        TransposeToCol(transpLocal, workLocal, matBase);
        ColNormalize(transpLocal, eps, scalarLocal, reduceTmpLocal);
        TransposeFromCol(workLocal, transpLocal, matBase);

        // Repeat loop - use actual repeat_ with safety cap
        int64_t nRepeat = repeat_;
        if (nRepeat <= 0 || nRepeat > 100) nRepeat = 10;
        for (int64_t iter = 1; iter < nRepeat; iter++) {
            RowNormalize(workLocal, matBase, eps, scalarLocal, reduceTmpLocal);
            TransposeToCol(transpLocal, workLocal, matBase);
            ColNormalize(transpLocal, eps, scalarLocal, reduceTmpLocal);
            TransposeFromCol(workLocal, transpLocal, matBase);
        }
    }
}

// =========================================================================
// CopyOut
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::CopyOut()
{
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> flatLocal = flatBuf.Get<T>();

    for (int64_t m = 0; m < matricesPerCore_; m++) {
        int64_t ubBase = m * PADDED_ELEMS_PER_MATRIX;
        int64_t gmBase = m * ELEMS_PER_MATRIX;
        for (int64_t r = 0; r < MATRIX_ROWS; r++) {
            int64_t ubR = ubBase + r * PADDED_ROW_LEN;
            int64_t gmR = gmBase + r * MATRIX_COLS;
            flatLocal.SetValue(gmR + 0, workLocal.GetValue(ubR + 0));
            flatLocal.SetValue(gmR + 1, workLocal.GetValue(ubR + 1));
            flatLocal.SetValue(gmR + 2, workLocal.GetValue(ubR + 2));
            flatLocal.SetValue(gmR + 3, workLocal.GetValue(ubR + 3));
        }
    }

    int64_t gmOffset = matrixStartOffset_ * ELEMS_PER_MATRIX;
    int64_t flatElems = matricesPerCore_ * ELEMS_PER_MATRIX;

    DataCopyParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = static_cast<uint32_t>(flatElems / AscendCUtils::GetC0Count(sizeof(T)));
    DataCopy(outputGlobal[gmOffset], flatLocal, storeParams);
}

// =========================================================================
// SoftmaxRow
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::SoftmaxRow(
    LocalTensor<T>& workLocal, int64_t rowOff,
    LocalTensor<T>& scalarLocal, LocalTensor<T>& reduceTmpLocal)
{
    ReduceMax<T>(scalarLocal, workLocal[rowOff], reduceTmpLocal, MATRIX_COLS, false);
    T maxVal = scalarLocal.GetValue(0);
    Adds<T>(workLocal[rowOff], workLocal[rowOff], static_cast<T>(-maxVal), MATRIX_COLS);
    Exp<T>(workLocal[rowOff], workLocal[rowOff], MATRIX_COLS);
    ReduceSum<T>(scalarLocal, workLocal[rowOff], reduceTmpLocal, MATRIX_COLS);
    T sumVal = scalarLocal.GetValue(0);
    // Use a tiny epsilon for softmax numerical stability (separate from sinkhorn eps)
    T invSum = static_cast<T>(1.0) / (sumVal + static_cast<T>(1e-10));
    Muls<T>(workLocal[rowOff], workLocal[rowOff], invSum, MATRIX_COLS);
}

// =========================================================================
// AddEps
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::AddEps(
    LocalTensor<T>& workLocal, int64_t matBase, T eps)
{
    for (int64_t r = 0; r < MATRIX_ROWS; r++) {
        int64_t rowOff = matBase + r * PADDED_ROW_LEN;
        Adds<T>(workLocal[rowOff], workLocal[rowOff], eps, MATRIX_COLS);
    }
}

// =========================================================================
// TransposeToCol
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::TransposeToCol(
    LocalTensor<T>& transpLocal, LocalTensor<T>& workLocal, int64_t matBase)
{
    for (int64_t r = 0; r < MATRIX_ROWS; r++) {
        int64_t pr = matBase + r * PADDED_ROW_LEN;
        transpLocal.SetValue(0 * TRANSPOSE_STRIDE + r, workLocal.GetValue(pr + 0));
        transpLocal.SetValue(1 * TRANSPOSE_STRIDE + r, workLocal.GetValue(pr + 1));
        transpLocal.SetValue(2 * TRANSPOSE_STRIDE + r, workLocal.GetValue(pr + 2));
        transpLocal.SetValue(3 * TRANSPOSE_STRIDE + r, workLocal.GetValue(pr + 3));
    }
}

// =========================================================================
// TransposeFromCol
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::TransposeFromCol(
    LocalTensor<T>& workLocal, LocalTensor<T>& transpLocal, int64_t matBase)
{
    for (int64_t r = 0; r < MATRIX_ROWS; r++) {
        int64_t pr = matBase + r * PADDED_ROW_LEN;
        workLocal.SetValue(pr + 0, transpLocal.GetValue(0 * TRANSPOSE_STRIDE + r));
        workLocal.SetValue(pr + 1, transpLocal.GetValue(1 * TRANSPOSE_STRIDE + r));
        workLocal.SetValue(pr + 2, transpLocal.GetValue(2 * TRANSPOSE_STRIDE + r));
        workLocal.SetValue(pr + 3, transpLocal.GetValue(3 * TRANSPOSE_STRIDE + r));
    }
}

// =========================================================================
// RowNormalize
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::RowNormalize(
    LocalTensor<T>& workLocal, int64_t matBase, T eps,
    LocalTensor<T>& scalarLocal, LocalTensor<T>& reduceTmpLocal)
{
    for (int64_t r = 0; r < MATRIX_ROWS; r++) {
        int64_t rowOff = matBase + r * PADDED_ROW_LEN;
        ReduceSum<T>(scalarLocal, workLocal[rowOff], reduceTmpLocal, MATRIX_COLS);
        T sumVal = scalarLocal.GetValue(0);
        T invSum = static_cast<T>(1.0) / (sumVal + eps);
        Muls<T>(workLocal[rowOff], workLocal[rowOff], invSum, MATRIX_COLS);
    }
}

// =========================================================================
// ColNormalize
// =========================================================================
template <typename T>
__aicore__ inline void SinkhornNormalize<T>::ColNormalize(
    LocalTensor<T>& transpLocal, T eps,
    LocalTensor<T>& scalarLocal, LocalTensor<T>& reduceTmpLocal)
{
    for (int64_t c = 0; c < MATRIX_COLS; c++) {
        int64_t colRowOff = c * TRANSPOSE_STRIDE;
        ReduceSum<T>(scalarLocal, transpLocal[colRowOff], reduceTmpLocal, MATRIX_ROWS);
        T colSum = scalarLocal.GetValue(0);
        T invColSum = static_cast<T>(1.0) / (colSum + eps);
        Muls<T>(transpLocal[colRowOff], transpLocal[colRowOff], invColSum, MATRIX_ROWS);
    }
}

} // namespace NsSinkhornNormalize

#endif // SINKHORN_NORMALIZE_KERNEL_H
