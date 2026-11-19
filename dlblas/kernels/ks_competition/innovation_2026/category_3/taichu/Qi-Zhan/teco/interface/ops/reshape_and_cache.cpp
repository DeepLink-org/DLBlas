// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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

#include <dirent.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sdaa_runtime.h>
#include "ual/ops/reshape_and_cache/reshape_and_cache.hpp"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/reshape_and_cache_args.h"

using tecoops::ual::args::ReshapeAndCacheArgs;
using tecoops::ual::ops::ReshapeAndCacheOp;

namespace {

bool parseCpuList(const char *list, cpu_set_t *mask) {
    CPU_ZERO(mask);
    const char *cursor = list;
    bool found = false;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0' || *cursor == '\n') {
            break;
        }

        char *end = nullptr;
        long first = std::strtol(cursor, &end, 10);
        if (end == cursor || first < 0) {
            return false;
        }
        long last = first;
        cursor = end;
        if (*cursor == '-') {
            last = std::strtol(cursor + 1, &end, 10);
            if (end == cursor + 1 || last < first) {
                return false;
            }
            cursor = end;
        }
        if (last >= CPU_SETSIZE) {
            return false;
        }
        for (long cpu = first; cpu <= last; ++cpu) {
            CPU_SET(static_cast<int>(cpu), mask);
        }
        found = true;
        if (*cursor != ',' && *cursor != '\0' && *cursor != '\n' &&
            *cursor != ' ' && *cursor != '\t') {
            return false;
        }
    }
    return found;
}

bool readDeviceLocalCpuMask(int pci_id, cpu_set_t *mask) {
    constexpr const char *kDriverPath =
        "/sys/bus/pci/drivers/tecorigin-aicard";
    int bus = (pci_id >> 16) & 0xff;
    int device = (pci_id >> 8) & 0xff;
    int function = pci_id & 0x7;
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), ":%02x:%02x.%x",
                  bus, device, function);

    DIR *directory = opendir(kDriverPath);
    if (directory == nullptr) {
        return false;
    }

    bool parsed = false;
    for (dirent *entry = readdir(directory); entry != nullptr;
         entry = readdir(directory)) {
        size_t name_len = std::strlen(entry->d_name);
        size_t suffix_len = std::strlen(suffix);
        if (name_len < suffix_len ||
            std::strcmp(entry->d_name + name_len - suffix_len, suffix) != 0) {
            continue;
        }

        char path[256];
        std::snprintf(path, sizeof(path), "%s/%s/local_cpulist",
                      kDriverPath, entry->d_name);
        FILE *file = std::fopen(path, "r");
        if (file == nullptr) {
            break;
        }
        char cpu_list[256];
        parsed = std::fgets(cpu_list, sizeof(cpu_list), file) != nullptr &&
                 parseCpuList(cpu_list, mask);
        std::fclose(file);
        break;
    }
    closedir(directory);
    return parsed;
}

void bindSubmitThreadToDeviceLocalCpus() {
    static thread_local bool attempted = false;
    if (attempted) {
        return;
    }
    attempted = true;

    int current_device = 0;
    int pci_id = 0;
    if (sdaaGetDevice(&current_device) != sdaaSuccess ||
        sdaaDeviceGetAttribute(&pci_id, sdaaDevAttrPciBusId,
                               current_device) != sdaaSuccess) {
        return;
    }

    cpu_set_t current_mask;
    cpu_set_t local_mask;
    cpu_set_t target_mask;
    if (sched_getaffinity(0, sizeof(current_mask), &current_mask) != 0 ||
        !readDeviceLocalCpuMask(pci_id, &local_mask)) {
        return;
    }

    CPU_ZERO(&target_mask);
    bool any = false;
    bool changed = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        bool current = CPU_ISSET(cpu, &current_mask);
        bool target = current && CPU_ISSET(cpu, &local_mask);
        if (target) {
            CPU_SET(cpu, &target_mask);
            any = true;
        }
        changed = changed || current != target;
    }
    if (any && changed) {
        (void)sched_setaffinity(0, sizeof(target_mask), &target_mask);
    }
}

}  // namespace

static tecoopsStatus_t checkReshapeAndCacheInput(tecoopsHandle_t handle,
                                                  const void *key, const void *value,
                                                  const int64_t *slot_mapping,
                                                  void *key_cache, void *value_cache,
                                                  int num_tokens, int num_kv_heads, int head_size,
                                                  int num_blocks, int block_size) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (key == nullptr || value == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (slot_mapping == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (key_cache == nullptr || value_cache == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (num_kv_heads <= 0 || head_size <= 0 || num_tokens <= 0 ||
        num_blocks <= 0 || block_size <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

static tecoopsStatus_t launchReshapeAndCache(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size,
    int key_stride, int key_head_stride,
    int value_stride, int value_head_stride) {
    constexpr int kHalfSize = 2;
    int entry_stride = head_size * kHalfSize;
    int head_stride = block_size * entry_stride;

    if (num_kv_heads == 2 && head_size == 64 && block_size == 32 &&
        num_blocks >= 8192) {
        // Keep the latency-sensitive submission thread on CPUs local to the
        // SDAA device. Existing narrower affinity constraints are preserved.
        bindSubmitThreadToDeviceLocalCpus();
    }

    int block_stride = num_kv_heads * head_stride;

    ReshapeAndCacheArgs arg;
    arg.num_kv_heads = num_kv_heads;
    arg.size_per_head = head_size;
    arg.num_tokens = num_tokens;
    arg.num_blocks = num_blocks;
    arg.block_size = block_size;
    arg.block_stride = block_stride;
    arg.head_stride = head_stride;
    arg.entry_stride = entry_stride;
    arg.key_stride = key_stride;
    arg.key_head_stride = key_head_stride;
    arg.value_stride = value_stride;
    arg.value_head_stride = value_head_stride;
    arg.key = key;
    arg.value = value;
    arg.slot_mapping = slot_mapping;
    arg.key_cache = key_cache;
    arg.value_cache = value_cache;
    arg.key_cache_transposed = nullptr;

    RUN_OP(ReshapeAndCacheOp, arg, arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsReshapeAndCache(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size) {
    tecoopsStatus_t input_error = checkReshapeAndCacheInput(
        handle, key, value, slot_mapping, key_cache, value_cache,
        num_tokens, num_kv_heads, head_size, num_blocks, block_size);
    if (input_error != TECOOPS_STATUS_SUCCESS) return input_error;

    constexpr int kHalfSize = 2;
    int head_stride = head_size * kHalfSize;
    int token_stride = num_kv_heads * head_stride;
    return launchReshapeAndCache(
        handle, key, value, slot_mapping, key_cache, value_cache,
        num_tokens, num_kv_heads, head_size, num_blocks, block_size,
        token_stride, head_stride, token_stride, head_stride);
}

tecoopsStatus_t tecoopsReshapeAndCacheStrided(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size,
    int key_token_stride, int key_head_stride,
    int value_token_stride, int value_head_stride) {
    tecoopsStatus_t input_error = checkReshapeAndCacheInput(
        handle, key, value, slot_mapping, key_cache, value_cache,
        num_tokens, num_kv_heads, head_size, num_blocks, block_size);
    if (input_error != TECOOPS_STATUS_SUCCESS) return input_error;
    if (key_token_stride <= 0 || key_head_stride <= 0 ||
        value_token_stride <= 0 || value_head_stride <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    constexpr int kHalfSize = 2;
    return launchReshapeAndCache(
        handle, key, value, slot_mapping, key_cache, value_cache,
        num_tokens, num_kv_heads, head_size, num_blocks, block_size,
        key_token_stride * kHalfSize, key_head_stride * kHalfSize,
        value_token_stride * kHalfSize, value_head_stride * kHalfSize);
}
