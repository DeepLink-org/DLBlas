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

#include <stdio.h>
#include <iostream>
#include <string>
#include <numeric>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/flash_attention/flash_attention.h"
#include "interface/include/tecoops.h"


namespace optest {

void FlashAttentionExecutor::paramCheck() {
    if (parser_->inputs().size() != 6) {
        ALLOG(ERROR) << "flash_attention expects 6 inputs (blockTable, qData, kCache, vCache, q_seq_lens, kv_seq_lens).";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
    if (parser_->outputs().size() != 1) {
        ALLOG(ERROR) << "flash_attention expects 1 output (oData).";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
}

void FlashAttentionExecutor::paramParse() {
    // Read proto scalar params
    auto fa_param = parser_->getProtoNode()->tecokernel_param().flash_attention_param();

    // Read q/kv seq_lens from tensor inputs (prev_value)
    batch_size_ = parser_->input(4)->shape[0];
    q_seq_lens_.resize(batch_size_);
    kv_seq_lens_.resize(batch_size_);
    parser_->getInputData(4, q_seq_lens_.data());
    parser_->getInputData(5, kv_seq_lens_.data());

    total_q_tokens_ = std::accumulate(q_seq_lens_.begin(), q_seq_lens_.end(), 0);

    // Input 0: blockTable [batch_size, block_table_dim]
    block_table_dim_ = parser_->input(0)->shape[1];
    // Input 1: qData [total_q_tokens, local_head_num, size_per_head]
    local_head_num_ = parser_->input(1)->shape[1];
    size_per_head_ = parser_->input(1)->shape[2];
    // Input 2: kCache [max_block_num, local_kv_head_num, block_size, size_per_head]
    max_block_num_ = parser_->input(2)->shape[0];
    local_kv_head_num_ = parser_->input(2)->shape[1];
    block_size_ = parser_->input(2)->shape[2];
}

void FlashAttentionExecutor::paramGeneration() {
    blockTableDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(0);
    blockTable_ = dev_input[0];
    qDataDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(1);
    qData_ = dev_input[1];
    kCacheDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(2);
    kCache_ = dev_input[2];
    vCacheDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(3);
    vCache_ = dev_input[3];

    oDataDesc_ = getOutputDesc<tecoopsTensorDescriptor_t>(0);
    oData_ = dev_output[0];
}

void FlashAttentionExecutor::compute() {
#ifdef USE_TECO
    checkTECOOPS(tecoopsFlashAttention(handle_,
        max_prefill_len_, max_decode_len_, max_block_num_,
        static_cast<const int*>(dev_input[4]),
        static_cast<const int*>(dev_input[5]),
        blockTableDesc_, blockTable_,
        qDataDesc_, qData_,
        kCacheDesc_, kCache_,
        vCacheDesc_, vCache_,
        oDataDesc_, oData_,
        /*workspace=*/nullptr));
#endif
}

int64_t FlashAttentionExecutor::getTheoryOps() {
    int total_kv_tokens = std::accumulate(kv_seq_lens_.begin(), kv_seq_lens_.end(), 0);
    int64_t theory_ops = 4LL * total_q_tokens_ * total_kv_tokens * size_per_head_ * local_head_num_;
    return theory_ops;
}

int64_t FlashAttentionExecutor::getTheoryIoSize() {
    return getIoSize();
}

void FlashAttentionExecutor::cpuCompute() {
    pythonComputeCPU("cpu");
}

}  // namespace optest
