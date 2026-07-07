/**
 * @file sinkhorn_normalize_kernel.cpp
 * @brief Sinkhorn Normalize Ascend C Kernel Implementation
 *
 * Implements the sinkhorn_normalize algorithm on Ascend 910B2 using the Vector API.
 *
 * Algorithm per 4x4 matrix:
 *   Step A: softmax(dim=-1) + eps
 *   Step B: column normalize (sum over dim=-2)
 *   Repeat (repeat-1) times:
 *     Step C1: row normalize (sum over dim=-1)
 *     Step C2: column normalize (sum over dim=-2)
 *
 * UB Layout (padded, Layout B):
 *   Each 4x4 matrix occupies 32 float elements:
 *     Row 0: [r0c0, r0c1, r0c2, r0c3, pad, pad, pad, pad]  (8 elements, 32B aligned)
 *     Row 1: [r1c0, r1c1, r1c2, r1c3, pad, pad, pad, pad]
 *     Row 2: [r2c0, r2c1, r2c2, r2c3, pad, pad, pad, pad]
 *     Row 3: [r3c0, r3c1, r3c2, r3c3, pad, pad, pad, pad]
 *   Total: 32 elements = 128B per matrix
 *
 * Multi-core: each core processes matrices_per_core matrices independently.
 */

#include "kernel_operator.h"
#include "sinkhorn_normalize_kernel.h"

using namespace AscendC;

// Constants
constexpr uint32_t ELEMS_PER_MATRIX        = 16;   // 4x4 valid elements
constexpr uint32_t PADDED_ROW_LEN          = 8;    // 4 valid + 4 padding
constexpr uint32_t PADDED_ELEMS_PER_MATRIX = 32;   // 4 rows x 8 padded
constexpr uint32_t MATRIX_ROWS             = 4;
constexpr uint32_t MATRIX_COLS             = 4;
constexpr uint32_t TRANSPOSE_BUF_SIZE      = 16;   // 4x4 for transpose buffer
constexpr uint32_t REDUCE_TMP_BUF_SIZE     = 2048; // tmp buffer for Reduce (8KB = 2048 floats)
constexpr uint32_t SCALAR_BUF_SIZE         = 4;    // 4 floats for scalar results
constexpr float    MAX_FLOAT_VAL           = 85.0f; // clamp for exp stability

extern "C" __global__ __aicore__ void sinkhorn_normalize_kernel(
    __gm__ float* input,
    __gm__ float* output,
    SinkhornNormalizeTilingData tiling)
{
    uint32_t total_matrices       = tiling.total_matrices;
    uint32_t matrices_per_core    = tiling.matrices_per_core;
    uint32_t matrix_start_offset  = tiling.matrix_start_offset;
    uint32_t repeat               = tiling.repeat;
    float    eps                  = tiling.eps;

    // Calculate buffer sizes based on matrices assigned to this core
    uint32_t work_buf_elems = matrices_per_core * PADDED_ELEMS_PER_MATRIX;

    // Declare pipe and buffers
    TPipe pipe;

    // Work buffer: holds all matrices in padded layout (input + computation workspace)
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf, work_buf_elems * sizeof(float));

    // Transpose buffer: 4x4 float buffer for column operations
    TBuf<TPosition::VECCALC> transpBuf;
    pipe.InitBuffer(transpBuf, TRANSPOSE_BUF_SIZE * sizeof(float));

    // Scalar buffer: for ReduceMax/ReduceSum results
    TBuf<TPosition::VECCALC> scalarBuf;
    pipe.InitBuffer(scalarBuf, SCALAR_BUF_SIZE * sizeof(float));

    // Reduce tmp buffer: workspace for ReduceMax/ReduceSum
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    pipe.InitBuffer(reduceTmpBuf, REDUCE_TMP_BUF_SIZE * sizeof(float));

    // Get local tensors
    LocalTensor<float> workLocal       = workBuf.Get<float>();
    LocalTensor<float> transpLocal     = transpBuf.Get<float>();
    LocalTensor<float> scalarLocal     = scalarBuf.Get<float>();
    LocalTensor<float> reduceTmpLocal  = reduceTmpBuf.Get<float>();

    // Set up global tensor views for input and output
    GlobalTensor<float> inputGlobal;
    GlobalTensor<float> outputGlobal;
    inputGlobal.SetGlobalBuffer(input);
    outputGlobal.SetGlobalBuffer(output);

    // ================================================================
    // Step 1: Load data from GM to UB
    // ================================================================
    // Strategy: Load all matrices in flat layout (no padding) using one
    // DataCopy with correct DataBlock count (=total_elems / C0Count).
    // Then manually redistribute to padded work buffer.
    // This avoids the DataCopyPad blockLen unit issue (blockLen is in
    // 32B DataBlocks, so row-by-row loading of 4 floats is impossible).

    uint32_t gm_base_offset = matrix_start_offset * ELEMS_PER_MATRIX;
    uint32_t flat_elems = matrices_per_core * ELEMS_PER_MATRIX;

    // Flat buffer for GM<->UB transfers (no padding, just the valid 4x4 elements)
    TBuf<TPosition::VECCALC> flatBuf;
    pipe.InitBuffer(flatBuf, flat_elems * sizeof(float));
    LocalTensor<float> flatLocal = flatBuf.Get<float>();

    // Load from GM to flat buffer using DataCopy with correct DataBlock count
    {
        uint32_t c0_count = AscendCUtils::GetC0Count(sizeof(float)); // = 8 floats/DataBlock
        DataCopyParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen   = flat_elems / c0_count;  // flat_elems always divisible by 8 (16 floats/matrix)
        DataCopy(flatLocal, inputGlobal[gm_base_offset], loadParams);
    }

    // Redistribute: flat layout -> padded work buffer
    for (uint32_t m = 0; m < matrices_per_core; m++) {
        uint32_t ub_base = m * PADDED_ELEMS_PER_MATRIX;
        uint32_t gm_base = m * ELEMS_PER_MATRIX;
        for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
            uint32_t ub_r = ub_base + r * PADDED_ROW_LEN;
            uint32_t gm_r = gm_base + r * MATRIX_COLS;
            workLocal[ub_r + 0] = flatLocal[gm_r + 0];
            workLocal[ub_r + 1] = flatLocal[gm_r + 1];
            workLocal[ub_r + 2] = flatLocal[gm_r + 2];
            workLocal[ub_r + 3] = flatLocal[gm_r + 3];
            // Padding elements [4..7] remain 0 from TBuf initialization
        }
    }

    // ================================================================
    // Step 2: Process each matrix (all iterations in UB)
    // ================================================================
    for (uint32_t m = 0; m < matrices_per_core; m++) {
        uint32_t mat_base = m * PADDED_ELEMS_PER_MATRIX;

        // ---- Step 2a: Softmax over dim=-1 (per row) ----
        for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
            uint32_t row_off = mat_base + r * PADDED_ROW_LEN;

            // 1) ReduceMax for numerical stability
            ReduceMax<float>(scalarLocal, workLocal[row_off], reduceTmpLocal, MATRIX_COLS, false);
            float max_val = scalarLocal.GetValue(0);

            // 2) Subtract max: row = row - max
            Adds<float>(workLocal[row_off], workLocal[row_off], -max_val, MATRIX_COLS);

            // 3) Exp: row = exp(row)
            Exp<float>(workLocal[row_off], workLocal[row_off], MATRIX_COLS);

            // 4) ReduceSum
            ReduceSum<float>(scalarLocal, workLocal[row_off], reduceTmpLocal, MATRIX_COLS);
            float sum_val = scalarLocal.GetValue(0);

            // 5) Normalize: row = row / sum
            float inv_sum = 1.0f / (sum_val + eps);
            Muls<float>(workLocal[row_off], workLocal[row_off], inv_sum, MATRIX_COLS);
        }

        // ---- Step 2b: Add epsilon to all elements ----
        for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
            uint32_t row_off = mat_base + r * PADDED_ROW_LEN;
            Adds<float>(workLocal[row_off], workLocal[row_off], eps, MATRIX_COLS);
        }

        // ---- Step 2c: Column normalize (dim=-2) via transpose ----
        {
            // Transpose 4x4: source (padded layout) -> transpLocal (column-major layout)
            // Row r of padded source -> Column r of transpLocal
            // Direct UB-local element assignments (no DataCopy DMA overhead)
            for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
                uint32_t pr = mat_base + r * PADDED_ROW_LEN;
                transpLocal[0 + r]  = workLocal[pr + 0];
                transpLocal[4 + r]  = workLocal[pr + 1];
                transpLocal[8 + r]  = workLocal[pr + 2];
                transpLocal[12 + r] = workLocal[pr + 3];
            }

            // Row-wise reduce and normalize on transposed buffer
            for (uint32_t c = 0; c < MATRIX_COLS; c++) {
                uint32_t col_row_off = c * MATRIX_ROWS;
                ReduceSum<float>(scalarLocal, transpLocal[col_row_off], reduceTmpLocal, MATRIX_ROWS);
                float col_sum = scalarLocal.GetValue(0);
                float inv_col_sum = 1.0f / (col_sum + eps);
                Muls<float>(transpLocal[col_row_off], transpLocal[col_row_off], inv_col_sum, MATRIX_ROWS);
            }

            // Transpose back: transpLocal -> workLocal (padded layout)
            // Direct UB-local element assignments
            for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
                uint32_t pr = mat_base + r * PADDED_ROW_LEN;
                workLocal[pr + 0] = transpLocal[r];
                workLocal[pr + 1] = transpLocal[4 + r];
                workLocal[pr + 2] = transpLocal[8 + r];
                workLocal[pr + 3] = transpLocal[12 + r];
            }
        }

        // ---- Step 2d: Repeat loop (repeat-1) times ----
        for (uint32_t iter = 1; iter < repeat; iter++) {
            // Row normalize (dim=-1): sum each row -> scale
            for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
                uint32_t row_off = mat_base + r * PADDED_ROW_LEN;
                ReduceSum<float>(scalarLocal, workLocal[row_off], reduceTmpLocal, MATRIX_COLS);
                float row_sum = scalarLocal.GetValue(0);
                float inv_row_sum = 1.0f / (row_sum + eps);
                Muls<float>(workLocal[row_off], workLocal[row_off], inv_row_sum, MATRIX_COLS);
            }

            // Column normalize (transpose approach)
            {
                // Transpose 4x4: source -> transpLocal (column-major layout)
                // Direct UB-local element assignments
                for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
                    uint32_t pr = mat_base + r * PADDED_ROW_LEN;
                    transpLocal[0 + r]  = workLocal[pr + 0];
                    transpLocal[4 + r]  = workLocal[pr + 1];
                    transpLocal[8 + r]  = workLocal[pr + 2];
                    transpLocal[12 + r] = workLocal[pr + 3];
                }

                // Normalize transposed rows
                for (uint32_t c = 0; c < MATRIX_COLS; c++) {
                    uint32_t col_row_off = c * MATRIX_ROWS;
                    ReduceSum<float>(scalarLocal, transpLocal[col_row_off], reduceTmpLocal, MATRIX_ROWS);
                    float col_sum = scalarLocal.GetValue(0);
                    float inv_col_sum = 1.0f / (col_sum + eps);
                    Muls<float>(transpLocal[col_row_off], transpLocal[col_row_off], inv_col_sum, MATRIX_ROWS);
                }

                // Transpose back: transpLocal -> workLocal (padded layout)
                // Direct UB-local element assignments
                for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
                    uint32_t pr = mat_base + r * PADDED_ROW_LEN;
                    workLocal[pr + 0] = transpLocal[r];
                    workLocal[pr + 1] = transpLocal[4 + r];
                    workLocal[pr + 2] = transpLocal[8 + r];
                    workLocal[pr + 3] = transpLocal[12 + r];
                }
            }
        }
    }

    // ================================================================
    // Step 3: Store results from UB to GM
    // ================================================================
    // Flatten: padded work buffer -> flat layout (strip padding)
    for (uint32_t m = 0; m < matrices_per_core; m++) {
        uint32_t ub_base = m * PADDED_ELEMS_PER_MATRIX;
        uint32_t gm_base = m * ELEMS_PER_MATRIX;
        for (uint32_t r = 0; r < MATRIX_ROWS; r++) {
            uint32_t ub_r = ub_base + r * PADDED_ROW_LEN;
            uint32_t gm_r = gm_base + r * MATRIX_COLS;
            flatLocal[gm_r + 0] = workLocal[ub_r + 0];
            flatLocal[gm_r + 1] = workLocal[ub_r + 1];
            flatLocal[gm_r + 2] = workLocal[ub_r + 2];
            flatLocal[gm_r + 3] = workLocal[ub_r + 3];
        }
    }

    // Store from flat buffer to GM using DataCopy with correct DataBlock count
    {
        uint32_t c0_count = AscendCUtils::GetC0Count(sizeof(float));
        DataCopyParams storeParams;
        storeParams.blockCount = 1;
        storeParams.blockLen   = flat_elems / c0_count;
        DataCopy(outputGlobal[gm_base_offset], flatLocal, storeParams);
    }
}
