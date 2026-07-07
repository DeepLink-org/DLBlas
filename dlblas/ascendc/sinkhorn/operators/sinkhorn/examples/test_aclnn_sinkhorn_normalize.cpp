#include <iostream>
#include <vector>
#include <cstring>
#include <numeric>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_sinkhorn_normalize.h"

int main() {
    auto ret = aclInit(nullptr); if (ret!=0) {std::cerr<<"aclInit:"<<ret<<std::endl; return -1;}
    ret = aclrtSetDevice(0); if (ret!=0) {std::cerr<<"setDev:"<<ret<<std::endl; return -1;}

    aclrtStream stream; aclrtCreateStream(&stream);

    std::vector<int64_t> shape = {1, 1024, 4, 4};
    int64_t total = 16384;
    std::vector<float> hIn(total, 0.5f);
    // Create contiguous strides
    std::vector<int64_t> strides = {1024*16, 16, 4, 1};

    void *dIn=nullptr, *dOut=nullptr;
    aclrtMalloc(&dIn, total*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, total*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dIn, total*sizeof(float), hIn.data(), total*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *tIn = aclCreateTensor(shape.data(), 4, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 4, dIn);
    aclTensor *tOut = aclCreateTensor(shape.data(), 4, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 4, dOut);

    std::cout << "Tensors created" << std::endl;

    uint64_t wsSize=0; aclOpExecutor *exec=nullptr;
    ret = aclnnSinkhornNormalizeGetWorkspaceSize(tIn, 1e-6f, 10, tOut, &wsSize, &exec);
    std::cout << "GetWS: ret=" << ret << " ws=" << wsSize << std::endl;

    if (ret==0) {
        void *ws=nullptr;
        if (wsSize>0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnSinkhornNormalize(ws, wsSize, exec, stream);
        std::cout << "Exec: ret=" << ret << std::endl;
        if (ret==0) {
            aclrtSynchronizeStream(stream);
            // Read back
            std::vector<float> hOut(total, -999);
            aclrtMemcpy(hOut.data(), total*sizeof(float), dOut, total*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            // Check for any non-zero values
            int nz=0; float s=0;
            for (int i=0;i<total;i++) {
                if (hOut[i]!=0.0f) nz++;
                s += hOut[i];
            }
            std::cout << "Non-zero elements: " << nz << "/" << total << std::endl;
            std::cout << "Sum: " << s << std::endl;
            std::cout << "First 16: ";
            for (int i=0;i<16;i++) std::cout << hOut[i] << " ";
            std::cout << std::endl;
            std::cout << "Last 16: ";
            for (int i=total-16;i<total;i++) std::cout << hOut[i] << " ";
            std::cout << std::endl;
        }
        if (ws) aclrtFree(ws);
    }

    aclDestroyTensor(tIn); aclDestroyTensor(tOut);
    aclrtFree(dIn); aclrtFree(dOut);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
