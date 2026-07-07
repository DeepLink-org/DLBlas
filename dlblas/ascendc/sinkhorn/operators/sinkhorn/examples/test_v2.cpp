#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"

// Forward declare the L0 API
namespace l0op {
const aclTensor* SinkhornNormalize(const aclTensor* x, float eps, int64_t repeat,
                                    aclOpExecutor* executor);
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
const aclTensor* ViewCopy(const aclTensor* src, const aclTensor* dst, aclOpExecutor* executor);
}

#include "aclnn_sinkhorn_normalize.h"

int main() {
    auto ret = aclInit(nullptr);
    if (ret != 0) { std::cerr << "aclInit: " << ret << std::endl; return -1; }
    ret = aclrtSetDevice(0);
    if (ret != 0) { std::cerr << "aclrtSetDevice: " << ret << std::endl; return -1; }

    std::vector<int64_t> shape = {1, 1024, 4, 4};
    int64_t total = 16384;
    std::vector<float> hIn(total, 0.5f);
    std::vector<float> hOut(total, 0.0f);
    std::vector<int64_t> strides = {1024*16, 16, 4, 1};

    void *dIn = nullptr, *dOut = nullptr;
    aclrtMalloc(&dIn, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dIn, total * sizeof(float), hIn.data(), total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *tIn = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                      strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), dIn);
    aclTensor *tOut = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                       strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), dOut);
    std::cout << "Tensors: in=" << (tIn!=nullptr) << " out=" << (tOut!=nullptr) << std::endl;

    // Test the two-stage API
    float eps = 1e-6f;
    int64_t repeat = 10;
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnSinkhornNormalizeGetWorkspaceSize(tIn, eps, repeat, tOut, &wsSize, &executor);
    std::cout << "GetWorkspaceSize: ret=" << ret << " wsSize=" << wsSize << " exec=" << (executor!=nullptr) << std::endl;

    if (ret == 0) {
        void *ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclrtStream stream;
        aclrtCreateStream(&stream);
        ret = aclnnSinkhornNormalize(ws, wsSize, executor, stream);
        std::cout << "Execute: ret=" << ret << std::endl;
        
        if (ret == 0) {
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(hOut.data(), total*sizeof(float), dOut, total*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::cout << "First 4: " << hOut[0] << " " << hOut[1] << " " << hOut[2] << " " << hOut[3] << std::endl;
        }
        aclrtDestroyStream(stream);
        if (ws) aclrtFree(ws);
    }

    aclDestroyTensor(tIn);
    aclDestroyTensor(tOut);
    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
