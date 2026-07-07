# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Sinkhorn Normalize kernel build configuration
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Kernel sources - Architecture-specific
# Only arch22 (ascend910b, ascend910_93) is supported for DAV_2201
# =============================================================================

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE SinkhornNormalize
    KERNEL_DIR .
    KERNEL_FILE sinkhorn_normalize_arch22.cpp
)

# =============================================================================
# Build kernel library
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
