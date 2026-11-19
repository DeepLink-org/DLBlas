// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ZOO_TECO_FLASH_ATTENTION_FLASH_ATTENTION_H_  // NOLINT
#define ZOO_TECO_FLASH_ATTENTION_FLASH_ATTENTION_H_

#include <vector>
#include "interface/include/tecoops.h"
#include "zoo/teco/executor.h"

namespace optest {

class FlashAttentionExecutor : public TecoExecutor {
 public:
    FlashAttentionExecutor() {}
    ~FlashAttentionExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;

 private:
    int max_prefill_len_;
    int max_decode_len_;
    int batch_size_;
    int total_q_tokens_;
    int local_head_num_;
    int local_kv_head_num_;
    int size_per_head_;
    int block_size_;
    int block_table_dim_;
    int max_block_num_;

    tecoopsTensorDescriptor_t blockTableDesc_;
    tecoopsTensorDescriptor_t qDataDesc_;
    tecoopsTensorDescriptor_t kCacheDesc_;
    tecoopsTensorDescriptor_t vCacheDesc_;
    tecoopsTensorDescriptor_t oDataDesc_;

    const void *blockTable_;
    const void *qData_;
    const void *kCache_;
    const void *vCache_;
    void *oData_;

    std::vector<int> q_seq_lens_;
    std::vector<int> kv_seq_lens_;
};

}  // namespace optest

#endif  // ZOO_TECO_FLASH_ATTENTION_FLASH_ATTENTION_H_  // NOLINT
