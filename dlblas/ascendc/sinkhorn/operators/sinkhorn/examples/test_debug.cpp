#include <iostream>
#include <vector>
#include "acl/acl.h"

int main() {
    auto ret = aclInit(nullptr);
    std::cout << "aclInit: " << ret << std::endl;
    ret = aclrtSetDevice(0);
    std::cout << "aclrtSetDevice: " << ret << std::endl;
    
    std::vector<int64_t> shape = {1, 1024, 4, 4};
    int64_t total = 16384;
    std::vector<float> data(total, 1.0f);
    
    void* devPtr = nullptr;
    ret = aclrtMalloc(&devPtr, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    std::cout << "aclrtMalloc: " << ret << std::endl;
    
    ret = aclrtMemcpy(devPtr, total * sizeof(float), data.data(), total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    std::cout << "aclrtMemcpy: " << ret << std::endl;
    
    aclTensor* t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
    std::cout << "aclCreateTensor: " << (t != nullptr) << std::endl;
    
    // Check tensor properties
    auto viewShape = aclGetTensorViewShape(t);
    auto* dims = aclGetViewShapeDims(viewShape);
    int64_t dimNum = 0;
    aclGetViewShapeDimNum(viewShape, &dimNum);
    std::cout << "dimNum: " << dimNum << std::endl;
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t d = 0;
        aclGetViewShapeDim(viewShape, i, &d);
        std::cout << "  dim[" << i << "] = " << d << std::endl;
    }
    
    auto dtype = aclGetTensorDataType(t);
    std::cout << "dtype: " << static_cast<int>(dtype) << " (ACL_FLOAT=" << ACL_FLOAT << ")" << std::endl;
    
    auto format = aclGetTensorFormat(t);
    std::cout << "format: " << static_cast<int>(format) << " (ACL_FORMAT_ND=" << ACL_FORMAT_ND << ")" << std::endl;
    
    aclDestroyTensor(t);
    aclrtFree(devPtr);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
