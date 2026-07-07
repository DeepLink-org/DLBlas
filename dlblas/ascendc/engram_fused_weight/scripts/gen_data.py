# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Test data generation for engram_fused_weight
#
# Inputs:  wh_data (bf16 uint16 binary), we_data (bf16 uint16 binary)
# Output:  golden   (fp32 binary)
#
# Per DESIGN.md §8.3: Golden = wh_data.float() * we_data.float()
# ============================================================================

import numpy as np
import os
import sys

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Default parameters matching source code (hc_mult=4, hidden_size=128)
hc_mult = 4
hidden_size = 128

# Allow overriding from command line
if len(sys.argv) >= 3:
    hc_mult = int(sys.argv[1])
    hidden_size = int(sys.argv[2])

dim0 = hc_mult * hidden_size

# Generate FP32 random data, then truncate to bfloat16 for input
wh_data = np.random.randn(hc_mult, hidden_size).astype(np.float32)
we_data = np.random.randn(hc_mult, hidden_size).astype(np.float32)


def fp32_to_bf16(arr):
    """Convert FP32 array to bfloat16 (truncate lower 16 bits of mantissa)."""
    ui32 = arr.view(np.uint32)
    ui16 = (ui32 >> 16).astype(np.uint16)
    return ui16


def bf16_to_fp32(ui16):
    """Convert bfloat16 uint16 back to FP32."""
    ui32 = ui16.astype(np.uint32) << 16
    return ui32.view(np.float32)


# Store BF16 inputs as raw uint16 binary for AscendC DataCopy
wh_bf16 = fp32_to_bf16(wh_data)
we_bf16 = fp32_to_bf16(we_data)
wh_bf16.tofile("input/input_wh.bin")
we_bf16.tofile("input/input_we.bin")

# Compute golden: wh.float() * we.float() in FP32 (per DESIGN.md §8.3)
wh_fp32 = bf16_to_fp32(wh_bf16.reshape(-1)).reshape(hc_mult, hidden_size)
we_fp32 = bf16_to_fp32(we_bf16.reshape(-1)).reshape(hc_mult, hidden_size)

# FP32 element-wise multiply
golden_fp32 = wh_fp32 * we_fp32

# Save golden as FP32 binary (output from kernel is also FP32)
golden_fp32.astype(np.float32).tofile("output/golden.bin")

print(f"Generated test data: hc_mult={hc_mult}, hidden_size={hidden_size}, dim0={dim0}")
print(f"  input/input_wh.bin: bf16 uint16 ({wh_bf16.size} elements, {wh_bf16.nbytes} bytes)")
print(f"  input/input_we.bin: bf16 uint16 ({we_bf16.size} elements, {we_bf16.nbytes} bytes)")
print(f"  output/golden.bin:  fp32 ({golden_fp32.size} elements, {golden_fp32.nbytes} bytes)")
