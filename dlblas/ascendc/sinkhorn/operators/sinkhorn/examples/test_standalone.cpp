#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn_sinkhorn_normalize.h"

#define CHECK_RET(cond, msg) do { if (!(cond)) { std::cerr << msg << std::endl; return -1; } } while (0)

int main() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == 0, "aclInit failed: " << ret);
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == 0, "aclrtSetDevice failed: " << ret);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == 0, "aclrtCreateStream failed: " << ret);

    std::vector<int64_t> shape = {1, 1024, 4, 4};
    int64_t total = 1 * 1024 * 4 * 4;
    std::vector<float> hIn(total, 0.5f);
    std::vector<float> hOut(total, 0.0f);

    std::vector<int64_t> strides = {1024*16, 16, 4, 1};

    void *dIn = nullptr, *dOut = nullptr;
    ret = aclrtMalloc(&dIn, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == 0, "malloc in: " << ret);
    ret = aclrtMalloc(&dOut, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == 0, "malloc out: " << ret);

    ret = aclrtMemcpy(dIn, total * sizeof(float), hIn.data(), total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == 0, "memcpy in: " << ret);

    aclTensor *tIn = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                      strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), dIn);
    aclTensor *tOut = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                       strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), dOut);
    CHECK_RET(tIn != nullptr && tOut != nullptr, "create tensor failed");

    std::cout << "Tensors created successfully" << std::endl;

    // Try calling with null attrs first (default eps=1e-6, repeat=10)
    float eps = 1e-6f;
    int64_t repeat = 10;

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnSinkhornNormalizeGetWorkspaceSize(tIn, eps, repeat, tOut, &wsSize, &executor);
    std::cout << "GetWorkspaceSize result: " << ret << " (wsSize=" << wsSize << ")" << std::endl;

    if (ret == 0) {
        void *ws = nullptr;
        if (wsSize > 0) {
            aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        ret = aclnnSinkhornNormalize(ws, wsSize, executor, stream);
        std::cout << "Execute result: " << ret << std::endl;
        if (ret == 0) {
            aclrtSynchronizeStream(stream);
            std::cout << "Kernel completed!" << std::endl;
            aclrtMemcpy(hOut.data(), total * sizeof(float), dOut, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::cout << "First 4 values: " << hOut[0] << " " << hOut[1] << " " << hOut[2] << " " << hOut[3] << std::endl;
        }
        if (ws) aclrtFree(ws);
    }

    aclDestroyTensor(tIn);
    aclDestroyTensor(tOut);
    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
