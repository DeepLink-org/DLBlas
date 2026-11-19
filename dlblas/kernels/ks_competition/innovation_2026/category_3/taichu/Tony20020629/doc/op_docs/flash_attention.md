# tecoopsFlashAttention 设计文档

## 计算原理

Flash Attention 通过分块（tiling）和在线 softmax（online softmax）技术，在不牺牲精度的情况下减少 HBM 访问量。计算过程为：

```
O = Softmax(Q @ K^T * scale) @ V
```

**分块策略：**

- Q 分块大小 `BM = 128`，K/V 分块大小 `BN = 32`
- 对于每个 Q block，依次加载所有的 KV block 进行分块计算
- 每个 KV block 的中间结果通过 **online softmax** 累积到 float 累加器中

**Online Softmax 原理：**

```
对于每个 Q block:
  初始化 o_accum = 0, l = 0, m = -inf
  
  对于每个 KV block:
    1. S = Q_block @ K_block^T * scale              ← gemm 计算分数
    2. 对每行 i:
         m_new[i] = max(m[i], max(S[i]), 0)
         scale[i] = exp(m[i] - m_new[i])            ← 旧累积的重缩放因子
         o_accum[i] *= scale[i]                     ← 重缩放旧累积
         S[i] = exp(S[i] - m_new[i])                ← 未归一化概率
         l_new[i] = l[i] * scale[i] + sum(S[i])     ← 新累积和
         m[i] = m_new[i], l[i] = l_new[i]
    3. o_accum += S @ V_block                       ← 累加当前块贡献
  
  输出: obuf = o_accum / l                           ← 归一化
```

**GQA (Grouped Query Attention) 支持：**

- 每个 KV head 服务 `num_heads / num_kv_heads` 个 query head
- `kv_group = tid * num_kv_heads / num_heads`

## 功能实现

### 接口设计

```c++
tecoopsStatus_t tecoopsFlashAttention(
    tecoopsHandle_t handle,
    int max_seqlen_q,
    int max_seqlen_k,
    int max_block_num,
    const int *q_seq_lens,           // [batch_size]
    const int *kv_seq_lens,          // [batch_size]
    const tecoopsTensorDescriptor_t blockTableDesc,
    const void *blockTable,          // [batch_size, block_table_dim] int32
    const tecoopsTensorDescriptor_t qDataDesc,
    const void *qData,               // [total_q, num_heads, head_size] half
    const tecoopsTensorDescriptor_t kCacheDesc,
    const void *kCache,              // [max_block, num_kv_heads, block_size, head_size] half
    const tecoopsTensorDescriptor_t vCacheDesc,
    const void *vCache,              // [max_block, num_kv_heads, block_size, head_size] half
    const tecoopsTensorDescriptor_t oDataDesc,
    void *oData,                     // [total_q, num_heads, head_size] half
    void *workspace);
```

Python API：

```python
tecoops.flash_attn_varlen_func(
    q,              # [total_q, num_heads, head_size]
    k,              # [num_blocks, num_kv_heads, block_size, head_size]
    v,              # [num_blocks, num_kv_heads, block_size, head_size]
    max_seqlen_q,
    cu_seqlens_q,   # [batch_size + 1] 累积和
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None, # [batch_size] 每 batch 实际 KV 数
    softmax_scale=None,
    causal=False,
    window_size=None,
    block_table=None, # [batch_size, block_table_dim]
    return_softmax_lse=False,
    out=None,
)
```

### 参数信息

tecoopsFlashAttention参数信息

| 参数           | 输入/输出 | 主机端/设备端 | 说明                                                               |
| -------------- | --------- | ------------- | ------------------------------------------------------------------ |
| handle         | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文                                      |
| max_seqlen_q   | 输入      | 主机端        | 最大 query 序列长度                                                |
| max_seqlen_k   | 输入      | 主机端        | 最大 KV 序列长度                                                   |
| max_block_num  | 输入      | 主机端        | KV cache 中最大 block 数量                                         |
| q_seq_lens     | 输入      | 主机端        | 每 batch 的 query 长度，`[batch_size]`                           |
| kv_seq_lens    | 输入      | 主机端        | 每 batch 的 KV 长度，`[batch_size]`                              |
| blockTableDesc | 输入      | 主机端        | block table 描述符                                                 |
| blockTable     | 输入      | 设备端        | block id 映射表，`[batch_size, block_table_dim]` int32           |
| qDataDesc      | 输入      | 主机端        | Q 数据描述符                                                       |
| qData          | 输入      | 设备端        | Q 矩阵，`[total_q, num_heads, head_size]` half                   |
| kCacheDesc     | 输入      | 主机端        | K cache 描述符                                                     |
| kCache         | 输入      | 设备端        | K cache，`[max_block, num_kv_heads, block_size, head_size]` half |
| vCacheDesc     | 输入      | 主机端        | V cache 描述符                                                     |
| vCache         | 输入      | 设备端        | V cache，`[max_block, num_kv_heads, block_size, head_size]` half |
| oDataDesc      | 输入      | 主机端        | 输出描述符                                                         |
| oData          | 输出      | 设备端        | 输出矩阵，`[total_q, num_heads, head_size]` half                 |
| workspace      | 输入      | 设备端        | 工作空间（当前未使用）                                             |

### 类型限制

| 参数          | 数据类型 | 维度信息                                             | 存储格式 |
| ------------- | -------- | ---------------------------------------------------- | -------- |
| max_seqlen_q  | int32    | 标量                                                 | -        |
| max_seqlen_k  | int32    | 标量                                                 | -        |
| max_block_num | int32    | 标量                                                 | -        |
| softmax_scale | float    | 标量                                                 | -        |
| qData         | float16  | `[total_q, num_heads, head_size]`                  | Array    |
| kCache        | float16  | `[max_block, num_kv_heads, block_size, head_size]` | Array    |
| vCache        | float16  | `[max_block, num_kv_heads, block_size, head_size]` | Array    |
| blockTable    | int32    | `[batch_size, block_table_dim]`                    | Array    |
| q_seq_lens    | int32    | `[batch_size]`                                     | Array    |
| kv_seq_lens   | int32    | `[batch_size]`                                     | Array    |
| oData         | float16  | `[total_q, num_heads, head_size]`                  | Array    |
| workspace     | void*    | 标量                                                 | -        |

## 性能优化

### 当前实现的计算分支

支持以下场景：

| 场景            | q_seq_len    | kv_seq_len  | causal | 说明          |
| --------------- | ------------ | ----------- | ------ | ------------- |
| prefill         | = kv_seq_len | = q_seq_len | 是     | 首轮全量计算  |
| decode          | 1            | 任意        | 是     | 逐 token 推理 |
| chunked prefill | < kv_seq_len | > q_seq_len | 是     | 分块预填充    |

### 优化设计

**1. 数据分块 (Tiling)**

- Q 按 `BM=128` 分块，K/V 按 `BN=32` 分块
- 每块独立加载到 SPM，减少 HBM 访问
- 每个 Q block 遍历所有 KV block 后一次性写出结果

**2. Online Softmax 分块累积**

- 使用 float32 累加器（`o_accum`）跨 KV block 累积
- 每步记录 `m`（row max）和 `l`（row sum），更新时重缩放旧累积
- 所有 KV block 处理完后归一化：`obuf = o_accum / l` → half

**3. SIMD 向量化**

- 计算 `exp(S - m_new)`、`l_block`、float→half 转换均使用 `floatv16` / `halfv16` 向量化
- 每批次处理 16 个元素，64 字节对齐

**4. GQA (Grouped Query Attention)**

每个 query head 映射到对应的 KV head：`kv_group = tid * num_kv_heads / num_heads`

### 性能数据

| 测试环境    | 测例                 | 硬件时间 (us) |
| ----------- | -------------------- | ------------- |
| CI 测试环境 | test_case_0.prototxt | 待补充        |

## 分支派发

| 算法取值           | 计算分支                            | 含义说明                            |
| ------------------ | ----------------------------------- | ----------------------------------- |
| `TECOOPS_ALGO_0` | `teco_slave_flash_attention_half` | 基础实现，half 精度，单 SPE 单 head |

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h                    # userAPI 声明
│   └── ops/flash_attention.cpp              # 接口实现
├── ual/
│   ├── args/flash_attention_args.h         # 参数结构体
│   ├── ops/flash_attention/
│   │   ├── flash_attention.hpp             # Op 类定义
│   │   ├── find_flash_attention.cpp        # 分支选择
│   │   └── find_flash_attention.h
│   └── kernel/flash_attention/
│       ├── flash_attention.h              # kernel 声明
│       └── flash_attention.scpp            # kernel 实现
├── plugin/
│   └── pluginFlashAttention/               # teco-inference 插件
│       └── plugin_flash_attention.cc
test/
├── zoo/teco/flash_attention/
│   ├── flash_attention.cpp                 # 测试 executor
│   ├── flash_attention.h
│   ├── flash_attention.py                  # Python reference
│   └── test_case/
│       ├── 0.prototxt ~ 6.prototxt         # 测试用例
api/
├── torch_ext.cpp                           # PyTorch 扩展绑定
└── tecoops/
    └── __init__.py                         # Python API
```

## 使用示例

```python
import torch
import torch_sdaa
import tecoops

# 构造输入 (prefill: L=S=256)
batch_size, num_heads, num_kv_heads, head_size, block_size = 1, 32, 8, 128, 32
total_q, total_kv = 256, 256
block_table_dim = (total_kv + block_size - 1) // block_size

q = torch.randn(total_q, num_heads, head_size, dtype=torch.float16, device='sdaa')
k_cache = torch.randn(block_table_dim, num_kv_heads, block_size, head_size, dtype=torch.float16, device='sdaa')
v_cache = torch.randn(block_table_dim, num_kv_heads, block_size, head_size, dtype=torch.float16, device='sdaa')
block_table = torch.zeros(batch_size, block_table_dim, dtype=torch.int32, device='sdaa')
cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device='sdaa')
seqused_k = torch.tensor([total_kv], dtype=torch.int32, device='sdaa')
out = torch.empty_like(q)

tecoops.flash_attn_varlen_func(
    q, k_cache, v_cache, total_q, cu_seqlens_q, total_kv,
    seqused_k=seqused_k, causal=True, block_table=block_table, out=out,
)
```
