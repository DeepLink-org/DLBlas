# tecoopsReshapeAndCache 设计文档

## 计算原理

将输入的 key/value token 按 `slot_mapping` 写入分页 KV cache。

- **slot_mapping**：长度为 `num_tokens` 的一维数组，`slot_mapping[i]` 表示第 `i` 个 token 应写入 cache 的第几个 slot
- **slot 到 block 映射**：`block_id = slot / block_size`，`block_offset = slot % block_size`
- 对于 `slot_mapping[i] < 0` 的 token，跳过不写入

**计算公式：**

```
对于 t in [0, num_tokens):
    slot = slot_mapping[t]
    如果 slot < 0: continue
    block_id = slot / block_size
    offset = slot % block_size
    对于 h in [0, num_kv_heads):
        key_cache[block_id][h][offset][:] = key[t][h][:]
        value_cache[block_id][h][offset][:] = value[t][h][:]
```

**参数解释：**

- `key`：输入 key 张量，形状 `[num_tokens, num_kv_heads, head_size]`
- `value`：输入 value 张量，形状同 key
- `slot_mapping`：token 到 cache slot 的映射，形状 `[num_tokens]`
- `key_cache`：输出的分页 key cache，形状 `[num_blocks, num_kv_heads, block_size, head_size]`
- `value_cache`：输出的分页 value cache，形状同 key_cache

## 功能实现

### 接口设计

参考 vLLM 的 `reshape_and_cache` 操作和 PagedAttention 场景需求，设计 userAPI 接口：

```c++
tecoopsStatus_t tecoopsReshapeAndCache(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size);
```

### 参数信息

其中，各参数含义如下：

| 参数         | 输入/输出 | 主机端/设备端 | 说明                                                                             |
| ------------ | --------- | ------------- | -------------------------------------------------------------------------------- |
| handle       | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文                                                    |
| key          | 输入      | 设备端        | 指向 key 张量的指针，形状`[num_tokens, num_kv_heads, head_size]`               |
| value        | 输入      | 设备端        | 指向 value 张量的指针，形状同 key                                                |
| slot_mapping | 输入      | 设备端        | 指向 slot 映射数组的指针，形状`[num_tokens]`，值 `-1` 表示跳过               |
| key_cache    | 输出      | 设备端        | 指向 key cache 的指针，形状`[num_blocks, num_kv_heads, block_size, head_size]` |
| value_cache  | 输出      | 设备端        | 指向 value cache 的指针，形状同 key_cache                                        |
| num_tokens   | 输入      | 主机端        | token 数量                                                                       |
| num_kv_heads | 输入      | 主机端        | KV head 数量                                                                     |
| head_size    | 输入      | 主机端        | 每个 head 的维度                                                                 |
| num_blocks   | 输入      | 主机端        | cache 的 block 数量                                                              |
| block_size   | 输入      | 主机端        | 每个 block 的 slot 数量                                                          |

### 类型限制

当前计算分支，主要完成以下功能实现，其余情况暂不支持。

| 参数         | 数据类型 | 维度信息                                              | 存储格式 |
| ------------ | -------- | ----------------------------------------------------- | -------- |
| key          | float16  | `[num_tokens, num_kv_heads, head_size]`             | NCHW     |
| value        | float16  | 同 key                                                | NCHW     |
| slot_mapping | int64    | `[num_tokens]`                                      | Array    |
| key_cache    | float16  | `[num_blocks, num_kv_heads, block_size, head_size]` | NCHW     |
| value_cache  | float16  | 同 key_cache                                          | NCHW     |

> **注意**：当前 kernel 仅支持 fp16 数据类型，stride 计算基于 `sizeof(half) = 2`。

## 性能优化

### 数据分块

计算总量为 `num_tokens` 个 token，均分到 32 个 SPE 线程并行执行。每个 SPE 按 `tid` 跨步处理不同的 token。

```
对于 t = tid; t < num_tokens; t += SPE_NUM:
    处理 token[t]（跳过 slot < 0 的无效映射）
```

### 性能数据

| 测试环境       | 硬件时间                                                        |
| -------------- | --------------------------------------------------------------- |
| teco-ops/test  | case_0.prototxt  (512 tokens, 8 kv_heads, 128 dim)  [OK]  590ms |
| teco-ops/test  | case_1.prototxt  (512 tokens, 4 kv_heads, 128 dim)  [OK]  515ms |
| teco-ops/test  | case_2.prototxt  (512 tokens, 2 kv_heads, 128 dim)  [OK]  514ms |

## 分支派发

| 算法取值           | 计算分支                              | 含义说明                            |
| ------------------ | ------------------------------------- | ----------------------------------- |
| `TECOOPS_ALGO_0` | `teco_slave_reshape_and_cache_fp16` | 基础实现，阻塞式memcpy，32 SPE 并行 |
| 其他               | 未实现                                | 暂不支持                            |

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h                      # userAPI 声明
│   └── ops/reshape_and_cache.cpp              # 接口实现（参数校验 + stride 计算）
├── ual/
│   ├── args/reshape_and_cache_args.h          # 参数结构体
│   ├── ops/reshape_and_cache/
│   │   ├── reshape_and_cache.hpp              # Op 类定义
│   │   ├── find_reshape_and_cache.cpp         # 分支选择
│   │   └── find_reshape_and_cache.h
│   └── kernel/reshape_and_cache/
│       ├── reshape_and_cache.h                # kernel 声明
│       └── reshape_and_cache_fp16.scpp        # kernel 实现
├── api/
│   └── torch_ext.cpp                          # PyTorch 绑定
└── plugin/
    └── pluginReshapeAndCache/
        └── plugin_reshape_and_cache.cc        # Plugin 推理接口
```

## 使用示例

### PyTorch 扩展

```python
import torch
import tecoops

num_tokens, num_kv_heads, head_size = 4, 2, 8
num_blocks, block_size = 4, 2

key = torch.randn(num_tokens, num_kv_heads, head_size, device='sdaa', dtype=torch.float16)
value = torch.randn(num_tokens, num_kv_heads, head_size, device='sdaa', dtype=torch.float16)
key_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, device='sdaa', dtype=torch.float16)
value_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, device='sdaa', dtype=torch.float16)
slot_mapping = torch.tensor([0, 1, 3, 6], dtype=torch.int64, device='sdaa')

tecoops.reshape_and_cache(key, value, slot_mapping, key_cache, value_cache)
```

### C++ 接口

```c++
tecoopsHandle_t handle;
tecoopsCreate(&handle);

tecoopsReshapeAndCache(
    handle,
    key_dptr, value_dptr,
    slot_mapping_dptr,
    key_cache_dptr, value_cache_dptr,
    num_tokens, num_kv_heads, head_size,
    num_blocks, block_size);

tecoopsDestroy(handle);
```

## 性能优化版本说明

### 分支 0：基础实现（原始）
- 阻塞式 memcpy，32 SPE 并行
- 每个 head 单独一次 memcpy（K + V），共 2H 次 HBM 事务/Token

### 分支 1：合并 memcpy（`teco_slave_reshape_and_cache_merged_fp16`）
- num_tokens >= 64 时自动启用
- K 和 V 合并写入，减少 HBM 事务数

### 分支 2：SIMD 向量化（`teco_slave_reshape_and_cache_simd_fp16`）
- head_size % 16 == 0 时自动启用
- 用 16-wide `halfv16` SIMD 指令做向量化写，替代逐元素 memcpy
- 消除 head 循环的标量开销
- 特别适合 OneGenomeRice 的 decode 场景（num_tokens 小）
