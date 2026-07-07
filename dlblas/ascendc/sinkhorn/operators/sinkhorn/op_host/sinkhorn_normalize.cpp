/**
 * @file sinkhorn_normalize.cpp
 * @brief Sinkhorn Normalize Host Operator - ACL/runtime-based kernel launch
 *
 * Uses runtime (rt) APIs to:
 * 1. Detect number of AI Cores dynamically
 * 2. Compute per-core tiling data (matrix partition)
 * 3. Register kernel binary and get function handle
 * 4. Allocate device memory
 * 5. Copy input data to device
 * 6. Launch the Ascend C kernel per-core
 * 7. Copy results back to host
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <mutex>

#include "acl/acl.h"
#include "acl/acl_rt.h"

// Manual declarations from runtime/runtime/kernel.h and runtime/runtime/base.h
// (avoiding complex include chain from pkg_inc/runtime headers)
typedef int32_t rtError_t;
#define RT_ERROR_NONE 0

typedef struct tagRtDevBinary {
    uint32_t magic;
    uint32_t version;
    const void *data;
    uint64_t length;
} rtDevBinary_t;

typedef void *rtStream_t;

struct tagRtSmCtrl;
typedef struct tagRtSmCtrl rtSmDesc_t;

#define RT_NORMAL_KERNEL_MODE (0x01U)

extern "C" {
    rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);
    rtError_t rtDevBinaryUnRegister(void *hdl);
    rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName,
                                 const void *kernelInfoExt, uint32_t funcMode);
    rtError_t rtGetFunctionByName(const char *stubName, void **stubFunc);
    rtError_t rtKernelLaunch(const void *stubFunc, uint32_t numBlocks, void *args,
                             uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm);
}

#include "sinkhorn_normalize.h"
#include "../op_kernel/sinkhorn_normalize_kernel.h"

// Thread-safe ACL initialization using std::call_once
static std::once_flag acl_init_flag;
static bool acl_initialized = false;

static int init_acl() {
    std::call_once(acl_init_flag, []() {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclInit failed: %d\n", ret);
            return;
        }
        // Set the active device context
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSetDevice(0) failed: %d\n", ret);
            return;
        }
        acl_initialized = true;
        fprintf(stdout, "[INFO] ACL runtime initialized (device 0)\n");
    });
    if (!acl_initialized) {
        fprintf(stderr, "[ERROR] ACL runtime not initialized\n");
        return -1;
    }
    return 0;
}

static int finalize_acl() {
    // Note: aclFinalize is intentionally not called here to avoid
    // issues with multiple threads sharing the ACL runtime.
    return 0;
}

// Helper: read entire file into memory buffer
static char* read_file(const char* path, size_t* out_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open kernel binary: %s\n", path);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* data = (char*)malloc(size);
    if (!data) {
        fclose(fp);
        return nullptr;
    }
    if (fread(data, 1, size, fp) != size) {
        fprintf(stderr, "[ERROR] Failed to read kernel binary\n");
        free(data);
        fclose(fp);
        return nullptr;
    }
    fclose(fp);
    *out_size = size;
    return data;
}

int sinkhorn_normalize_launch(const float* input, float* output,
                              uint32_t B, uint32_t S,
                              uint32_t repeat, float eps,
                              const char* kernel_bin_path)
{
    // ---- Input Validation ----
    if (input == nullptr || output == nullptr) {
        fprintf(stderr, "[ERROR] input or output pointer is null\n");
        return -1;
    }
    if (B != 1) {
        fprintf(stderr, "[ERROR] Currently only B=1 is supported, got B=%u\n", B);
        return -1;
    }
    if (S == 0) {
        fprintf(stderr, "[ERROR] S (number of matrices) must be > 0, got %u\n", S);
        return -1;
    }
    if (repeat < 1) {
        fprintf(stderr, "[ERROR] repeat must be >= 1, got %u\n", repeat);
        return -1;
    }
    if (eps <= 0.0f) {
        fprintf(stderr, "[ERROR] eps must be > 0, got %e\n", eps);
        return -1;
    }
    if (kernel_bin_path == nullptr || kernel_bin_path[0] == '\0') {
        fprintf(stderr, "[ERROR] kernel_bin_path is null or empty\n");
        return -1;
    }

    // Initialize ACL
    if (init_acl() != 0) return -1;

    aclError ret;
    int result = 0;
    aclrtStream stream = nullptr;
    aclrtBinHandle bin_handle = nullptr;
    aclrtFuncHandle func_handle = nullptr;
    void* dev_input = nullptr;
    void* dev_output = nullptr;
    char* kernel_data = nullptr;
    size_t kernel_size = 0;

    uint32_t total_matrices = S;
    uint32_t elems_per_matrix = 16;
    size_t data_size = total_matrices * elems_per_matrix * sizeof(float);

    // ---- P0-3: Dynamically detect AI Core count ----
    uint32_t num_cores = 1;
    int64_t aicore_count = 0;
    ret = aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_count);
    if (ret == ACL_SUCCESS && aicore_count > 0) {
        num_cores = (uint32_t)aicore_count;
        fprintf(stdout, "[INFO] Dynamic query: %u AI cores detected\n", num_cores);
    } else {
        num_cores = 32;
        fprintf(stdout, "[INFO] Could not query AICORE count (err=%d), fallback to %u cores\n", ret, num_cores);
    }

    // Compute per-core tiling data
    uint32_t matrices_per_core_base = total_matrices / num_cores;
    uint32_t remainder = total_matrices % num_cores;

    std::vector<SinkhornNormalizeTilingData> tiling_data(num_cores);
    std::vector<uint32_t> matrices_per_core(num_cores);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        matrices_per_core[i] = matrices_per_core_base + (i < remainder ? 1 : 0);
        tiling_data[i].total_matrices      = total_matrices;
        tiling_data[i].matrices_per_core   = matrices_per_core[i];
        tiling_data[i].matrix_start_offset = offset;
        tiling_data[i].repeat              = repeat;
        tiling_data[i].eps                 = eps;
        offset += matrices_per_core[i];
    }

    // ---- P0-1: Load kernel binary using aclrtBinaryLoadFromData ----
    // Read kernel .o file into memory
    kernel_data = read_file(kernel_bin_path, &kernel_size);
    if (!kernel_data) {
        fprintf(stderr, "[ERROR] Failed to read kernel binary: %s\n", kernel_bin_path);
        result = -1;
        goto cleanup;
    }
    fprintf(stdout, "[INFO] Kernel binary read: %zu bytes from %s\n", kernel_size, kernel_bin_path);

    // Load binary from in-memory data
    {
        aclrtBinaryLoadOptions loadOpts;
        aclrtBinaryLoadOption loadOpt;
        loadOpt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        loadOpt.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
        loadOpts.options = &loadOpt;
        loadOpts.numOpt = 1;

        ret = aclrtBinaryLoadFromData(kernel_data, kernel_size, &loadOpts, &bin_handle);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtBinaryLoadFromData failed: %d\n", ret);
            result = -1;
            goto cleanup;
        }
    }
    fprintf(stdout, "[INFO] Kernel binary loaded via aclrtBinaryLoadFromData\n");

    // Get kernel function handle by name
    ret = aclrtBinaryGetFunction(bin_handle, "sinkhorn_normalize_kernel", &func_handle);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtBinaryGetFunction failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }
    fprintf(stdout, "[INFO] Kernel function handle obtained\n");

    // Allocate device memory
    ret = aclrtMalloc(&dev_input, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtMalloc for input failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }

    ret = aclrtMalloc(&dev_output, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtMalloc for output failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }

    // Copy input data to device
    ret = aclrtMemcpy(dev_input, data_size, input, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtMemcpy H2D failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }

    // Create stream
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtCreateStream failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }

    // ---- Kernel launch: per-core with individual tiling data ----
    // Raw args approach: the kernel __CCE_KernelArgSize is 24 bytes.
    // The runtime expects the kernel args buffer to match the kernel parameter
    // layout. For static-shape kernel, pass the tiling struct directly.
    struct __attribute__((packed)) KernelArgs {
        void* gm_input;
        void* gm_output;
        SinkhornNormalizeTilingData tiling;
    };

    fprintf(stdout, "[INFO] Launching kernel on %u cores, %u total matrices\n",
            num_cores, total_matrices);
    fprintf(stdout, "[INFO] Input ptr: %p, Output ptr: %p\n",
            dev_input, dev_output);

    for (uint32_t i = 0; i < num_cores; i++) {
        // Skip cores with no work
        if (matrices_per_core[i] == 0) continue;

        KernelArgs args;
        args.gm_input  = dev_input;
        args.gm_output = dev_output;
        args.tiling    = tiling_data[i];

        ret = aclrtLaunchKernel(func_handle, 1, &args, sizeof(args), stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtLaunchKernel for core %u failed: %d\n", i, ret);
            result = -1;
        }
    }

    // Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed: %d\n", ret);
        result = -1;
    }

    aclrtDestroyStream(stream);
    stream = nullptr;

    // Copy results back to host
    ret = aclrtMemcpy(output, data_size, dev_output, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ERROR] aclrtMemcpy D2H failed: %d\n", ret);
        result = -1;
        goto cleanup;
    }

    fprintf(stdout, "[INFO] Kernel execution completed successfully\n");

cleanup:
    if (stream)     aclrtDestroyStream(stream);
    if (bin_handle) aclrtBinaryUnLoad(bin_handle);
    if (dev_input)  aclrtFree(dev_input);
    if (dev_output) aclrtFree(dev_output);
    if (kernel_data) free(kernel_data);

    return result;
}
