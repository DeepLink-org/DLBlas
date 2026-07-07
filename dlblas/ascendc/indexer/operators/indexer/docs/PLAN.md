# Indexer 算子 AscendC 实施计划

> 版本: v1.0
> 日期: 2026-07-03
> 关联设计: [DESIGN.md](./DESIGN.md)
> 目标设备: Ascend910B2 (DAV_2201), CANN 9.0.0

---

## 1. 实施总览

### 1.1 里程碑

```
Phase 0: 环境准备 + 基础设施        [1 day]
Phase 1: Kernel 1+2 (MatMul)        [1-2 days]
Phase 2: Kernel 4 (Postprocess)     [1-2 days]
Phase 3: Kernel 3 (RoPE + Score)    [2-3 days]  ← 核心难点
Phase 4: Host 调度 + 端到端集成      [1 day]
Phase 5: 测试 + 性能调优             [2-3 days]
──────────────────────────────────────────────
总计预估: 8-12 工作日
```

### 1.2 依赖关系

```
Phase 0 ──→ Phase 1 ──→ Phase 4 ──→ Phase 5
         \            /
          → Phase 2 -/
         \            /
          → Phase 3 -/
```

Phase 1、2、3 可并行开发；Phase 4 依赖前三者完成。

---

## 2. Phase 0: 环境准备与基础设施

### 2.1 目录结构创建

```bash
operators/indexer/
├── docs/
│   ├── DESIGN.md
│   └── PLAN.md                  ← 本文件
├── ascendc/
│   ├── CMakeLists.txt           # 构建配置
│   ├── common/
│   │   ├── common.h             # 共享宏、类型定义
│   │   └── utils.h              # reshape、bf16 工具函数
│   ├── kernel_q_proj.cpp        # Kernel 1
│   ├── kernel_w_proj.cpp        # Kernel 2
│   ├── kernel_rope_score.cpp    # Kernel 3
│   ├── kernel_post_topk.cpp     # Kernel 4
│   └── indexer_launcher.cpp     # Host 侧入口
├── test/
│   ├── torch_ref/
│   │   └── indexer_torch.py     # PyTorch 参考实现
│   ├── generate_inputs.py       # 输入生成脚本
│   ├── test_prefill.py          # Prefill 测试
│   └── test_decode.py           # Decode 测试
├── benchmark/
│   └── bench_indexer.py         # 性能基准测试
└── configs/
    └── default.json             # 默认参数配置
```

### 2.2 PyTorch 参考实现

从用户提供的 `indexer.py` 中提取 `Model` 和 `get_inputs/get_init_inputs`，构造一个独立的测试基准。

**要点**：
- world_size=1，移除 `ColumnParallelLinear` 的 TP 分片逻辑（等价于普通 `Linear`）
- 固定随机种子确保可复现
- 输出 `topk_idxs` 的参考结果用于后续精度比对

### 2.3 参数配置

定义标准的算子参数结构体：

```cpp
struct IndexerArgs {
    int batch_size;
    int seq_len;
    int dim;              // 1024
    int q_lora_rank;      // 256
    int n_heads;          // 16 (index_n_heads)
    int head_dim;         // 64  (index_head_dim)
    int rope_head_dim;    // 32
    int index_topk;       // 128
    int compress_ratio;   // 4
    int max_seq_len;      // 1024
};
```

---

## 3. Phase 1: Kernel 1 + Kernel 2 (标准 MatMul)

### 3.1 Kernel 1: q_projection

**输入/输出**：
```
Input:  qr        (B*S, q_lora_rank)    bf16  [GM]
Input:  wq_weight (H*D, q_lora_rank)    bf16  [GM, column-major]
Output: q_flat    (B*S, H*D)            bf16  [GM]
```

**实施步骤**：

1. 使用 AscendC 的 `MatMul` API 模板创建 Kernel
2. 配置 MatMul 参数：
   - `SetTensorA(qr_gm,    shape={M, K}, format=ND)`
   - `SetTensorB(wq_gm,    shape={K, N}, format=NZ)`  # column-major
   - `SetTensorC(q_flat_gm, shape={M, N}, format=ND)`
3. 设置 `SetBias(nullptr)` — 无 bias
4. `IterateAll()` 完成全量计算
5. 编译验证：`ascendc --target=ascend910b2 kernel_q_proj.cpp`

**验证点**：
- 输入 reshape B×S → B*S 正确
- 输出与 PyTorch `F.linear(qr, wq_b.weight)` 精度一致（bf16 容差 1e-2）

### 3.2 Kernel 2: weights_projection

**输入/输出**：
```
Input:  x            (B*S, dim)        bf16  [GM]
Input:  w_proj_weight (n_heads, dim)   bf16  [GM, column-major]
Output: weights      (B*S, n_heads)    bf16  [GM]
```

**实施步骤**：与 Kernel 1 完全相同的 MatMul pattern，替换 M/N/K 即可。

**验证点**：
- 输出与 PyTorch `F.linear(x, weights_proj.weight)` 精度一致

---

## 4. Phase 2: Kernel 4 (后处理 + TopK)

### 4.1 功能拆解

按顺序实现以下子步骤：

```
Step 1: 加载 scores(B,S,H,kv_len), weights(B,S,H)
Step 2: 逐 (b,s) 处理:
    2a. 遍历 h: ReLU(scores) * weights → 累加到 index_score(b,s,kv_len)
    2b. 如果 start_pos==0: causal_mask → -inf
    2c. TopK(index_score, k) → topk_indices, topk_values
    2d. 如果 start_pos==0: mask 无效index → -1; else: += offset
Step 3: 写出 topk_idxs(B,S,k)
```

### 4.2 AscendC 实现细节

**Tiling 策略**：
- 外层 tiling: 按 `(B, S)` 维分块。对于 decode (S=1)，按 B 分块；对于 prefill (S 大)，按 S 分块
- 每个 block 处理一个或多个 (b, s) 对

**ReLU + 加权求和**：
```cpp
// 伪代码
float accumulator[kv_len] = {0.0f};
for (int h = 0; h < n_heads; h++) {
    for (int j = 0; j < kv_len; j++) {
        float s = (float)scores[b][s][h][j];   // bf16 → fp32
        float w = (float)weights[b][s][h];      // bf16 → fp32
        if (s > 0.0f) {
            accumulator[j] += s * w;
        }
    }
}
// accumulator → bf16 → index_score
```

**Causal Mask**：
```cpp
if (start_pos == 0) {
    int threshold = (s + ratio) / ratio;  // ceil((s+1)/ratio)
    for (int j = threshold; j < kv_len; j++) {
        index_score[j] = bf16_neg_inf;    // 0xFF80
    }
}
```

**TopK 实现**：
```cpp
// 方案: k-pass max selection
// 对于每个 (b, s)，维护 topk_indices[k] 和 topk_values[k]

// Step 1: copy index_score to working buffer
// Step 2: for pass in range(k):
//           find argmax and value in working buffer
//           record in topk_indices[pass], topk_values[pass]
//           set working_buffer[argmax] = -inf
```

**复杂度分析**：O(k × kv_len) per (b, s)。对于 k=128, kv_len=256: ~32K 次比较/行。

**优化**：当 k 较大时（如 512），可考虑部分排序；当前默认 k=128，pass 选择法足够。

**后处理**：
```cpp
if (start_pos == 0) {
    for (int i = 0; i < k; i++) {
        if (topk_indices[i] >= threshold) {
            topk_indices[i] = -1;
        }
    }
} else {
    for (int i = 0; i < k; i++) {
        topk_indices[i] += offset;
    }
}
```

### 4.3 验证点

- ReLU 行为：负值清零，正值保留
- Causal Mask：每个 (b, s) 的 threshold 正确计算
- TopK：选择的是最大的 k 个值的索引，值相等时索引顺序与 PyTorch 一致
- 后处理：mask 和 offset 与 PyTorch 一致

---

## 5. Phase 3: Kernel 3 (RoPE + Batched MatMul)

这是**整个实现的核心难点**。

### 5.1 数据流拆解

```
输入:
  q_flat:       (B, S, H*D)        bf16  [GM]
  kv_cache:     (B, kv_len, D)     bf16  [GM]
  freqs_cis:    (max_seq_len, rd)  bf16  [GM, 存储 (cos, sin) 对]
  start_pos:    scalar
  end_pos:      scalar

中间:
  q_rot:        (B, H, S, D)       bf16  [L1/UB]
  kv_t:         (B, H, kv_len, D)  bf16  [L1/UB, broadcast]

输出:
  scores:       (B, H, S, kv_len)  bf16  [GM]   # 或 (B*H, S, kv_len)
```

### 5.2 子步骤详解

#### Step A: Q Reshape + RoPE

**实现方案**：

1. 将 q_flat 从 GM 加载到 L1/UB
2. Reshape: (B, S, H*D) → (B, S, H, D)，在内存中只是视角变换
3. 对每个 (b, s, h)，取 q[b,s,h, D-rd:D] 的 rd 个元素做 RoPE

**RoPE 向量化实现**：
```
// q 的 D 维度中，前 D-rd 个元素不变，后 rd 个元素做旋转变换
// rd = rope_head_dim = 32, D = head_dim = 64
// freqs_cis 预存为 (cos_0, sin_0, cos_1, sin_1, ... cos_{rd/2-1}, sin_{rd/2-1})

对于 (b,s,h) 和 i = 0..rd/2-1:
  offset = D - rd
  a = q[b,s,h, offset + 2i]       // 实部
  b = q[b,s,h, offset + 2i+1]     // 虚部
  c = freqs_cis[start_pos+s, 2i]  // cos
  d = freqs_cis[start_pos+s, 2i+1]// sin
  q_rot[b,s,h, offset + 2i]   = a*c - b*d
  q_rot[b,s,h, offset + 2i+1] = a*d + b*c
```

**Tiling**：按 (B, H, S) 分块，每块处理若干 (b, h) 组合的完整 S×D 子矩阵。

#### Step B: Q Permute + Reshape for BatchMatMul

```
q: (B, S, H, D) → 逻辑 permute → (B, H, S, D) → 逻辑 reshape → (B*H, S, D)
```

在 AscendC 中，可以通过 index 计算绕过显式 permute：
```
// 对于输出位置 (batch_idx, s, d)，其中 batch_idx = b*H + h
// 输入位置为 (b, s, h, d)
```

#### Step C: KV Cache Broadcast

```
kv_cache: (B, kv_len, D) → 每个 head 复制一份 → (B*H, kv_len, D)
```

**Zero-copy 方案**：
- 不实际复制数据，通过地址计算实现 broadcast
- 在 BatchMatMul 中设置 `tensorB` 时，让 B*H 个 batch 都指向同一个 kv_cache[b] 切片

```cpp
// 伪代码：对于 batch_idx = b*H + h
// tensorB[batch_idx] 指向 kv_cache_gm + b * kv_len * D * sizeof(bf16)
// 即同一个 b 的所有 h 共享同一份 kv_cache 数据
```

#### Step D: BatchMatMul

```
输入A: Q_perm    (B*H, S, D)       bf16  row-major
输入B: KV_bc     (B*H, D, kv_len)  bf16  column-major  (kv_cache 转置后 broadcast)
输出C: Scores    (B*H, S, kv_len)  bf16  row-major
```

**AscendC MatMul 配置**：
- Shape: [M=B*H*S, K=D, N=kv_len]
- 使用 BatchMatMul，batch = B*H
- `SetTensorA(q_gm, format=ND)` 从 q_perm 的位置读取
- `SetTensorB(kv_bc_gm, format=NZ)` column-major，利用 zero-copy broadcast
- `SetTensorC(scores_gm, format=ND)`

### 5.3 实施策略

**推荐分步实施**：

1. **Step 0**: 先用 PyTorch 在 Host 端完成所有中间步骤（q reshape、RoPE、permute、broadcast），仅用 AscendC 做最终的 BatchMatMul。验证 MatMul 结果正确。

2. **Step 1**: 将 RoPE 移入 AscendC Kernel，与 MatMul 合并。验证端到端正确性。

3. **Step 2**: 将 permute 和 broadcast 逻辑移入 Kernel，减少 Host 端操作。验证。

4. **Step 3**: 性能调优 — Tiling 参数调优、L1 Buffer 复用策略优化。

### 5.4 验证点

- RoPE 旋转角度正确：验证每个 (b, s, h) 的后 rd 个元素与 PyTorch `apply_rotary_emb` 一致
- BatchMatMul 结果：与 PyTorch `einsum("bshd,btd->bsht", q, kv_cache)` 一致
- 精度：bf16 下相对误差 < 1e-2（MatMul 累加误差）

---

## 6. Phase 4: Host 调度与端到端集成

### 6.1 IndexerLauncher 实现

```cpp
class IndexerLauncher {
public:
    IndexerLauncher(const IndexerArgs& args);
    
    // 加载权重到 Device
    void load_weights(void* wq_ptr, void* ww_ptr, void* kv_ptr, void* freqs_ptr);
    
    // 执行推理
    void run(
        void* x,           // (B, S, dim) bf16
        void* qr,          // (B, S, q_lora_rank) bf16
        int   start_pos,
        int   offset,
        void* topk_idxs    // (B, S, k) int64
    );
    
private:
    IndexerArgs args_;
    
    // Device 侧临时 buffer
    void* d_q_flat_;       // (B*S, H*D) bf16
    void* d_weights_;      // (B*S, n_heads) bf16
    void* d_scores_;       // (B*H, S, kv_len) bf16
    
    // Device 侧权重
    void* d_wq_weight_;    // (H*D, q_lora_rank) bf16 col-major
    void* d_ww_weight_;    // (n_heads, dim) bf16 col-major
    void* d_kv_cache_;     // (B, max_kv_len, D) bf16
    void* d_freqs_cis_;    // (max_seq_len, rd) bf16
    
    // AscendCL Stream
    aclrtStream stream_;
};
```

### 6.2 执行流程

```
1. 计算当前 end_pos = start_pos + seq_len
2. 计算 kv_len = end_pos / compress_ratio

3. Kernel 1: q_projection
   → d_q_flat_  (B*S, H*D)

4. Kernel 2: weights_projection  (可与 Kernel 1 并发)
   → d_weights_  (B*S, n_heads)

5. 同步 (如果并发)

6. Kernel 3: rope_score_compute
   → d_scores_  (B*H, S, kv_len)

7. Kernel 4: postprocess_topk
   → topk_idxs  (B, S, k)
```

### 6.3 内存管理

- 临时 buffer 在构造函数中一次性分配，复用整个推理过程
- buffer 大小根据 `IndexerArgs` 中的最大值预计算
- 支持内存池模式以便多 batch 交替推理

### 6.4 错误处理

- 每个 Kernel Launch 后检查 `aclrtSynchronizeStream` 返回值
- 输入 shape 合法性检查（nullptr、zero dim）
- kv_len > max_kv_len 时截断并告警

---

## 7. Phase 5: 测试与性能调优

### 7.1 测试矩阵

| 场景 | B | S | start_pos | 验证重点 |
|------|---|---|-----------|---------|
| Prefill-小 | 2 | 64 | 0 | 基础功能、causal mask |
| Prefill-中 | 2 | 512 | 0 | 中等计算量、tiling 正确 |
| Prefill-大 | 2 | 4096 | 0 | 最大计算量、OOM 检查 |
| Decode-早 | 2 | 1 | 64 | decode 路径、无 causal mask |
| Decode-中 | 2 | 1 | 512 | kv_len 增长 |
| Decode-晚 | 2 | 1 | 4096 | max_kv_len 边界 |
| 边界-kv_len | 2 | 8 | 0 | kv_len 小于 index_topk |

### 7.2 精度测试方法

```python
# test_prefill.py
import torch
from indexer_torch import Model, get_inputs, get_init_inputs

# 1. PyTorch 参考
model_torch = Model(*get_init_inputs()).npu()
output_ref = model_torch(*get_inputs())

# 2. AscendC 实现
output_ascendc = ascendc_indexer(*get_inputs())

# 3. 比对
assert output_ascendc.shape == output_ref.shape
# topk 索引比对（排序顺序可能不同，但集合应一致）
for b in range(B):
    for s in range(S):
        assert set(output_ascendc[b,s,:].tolist()) == set(output_ref[b,s,:].tolist())
```

### 7.3 性能测试方法

```python
# bench_indexer.py
import time
import numpy as np

# Warmup: 10 iterations
for _ in range(10):
    ascendc_indexer(*inputs)

# Benchmark: 100 iterations
torch.npu.synchronize()
start = time.perf_counter()
for _ in range(100):
    ascendc_indexer(*inputs)
torch.npu.synchronize()
elapsed = time.perf_counter() - start

print(f"Avg latency: {elapsed/100*1000:.3f} ms")
```

### 7.4 性能调优清单

| 优化项 | 方法 | 预期收益 |
|--------|------|---------|
| Tiling 参数调优 | 实验 BLOCK_M/BLOCK_N/BLOCK_K | 10-20% |
| L1 Buffer 复用 | 减少 GM↔L1 搬运次数 | 5-10% |
| Kernel 3+4 融合 | 避免 scores GM 写回再读入 | 15-25% |
| Decode 快速路径 | S=1 特殊处理，减化 causal mask 分支 | 20-30% (decode) |
| Stream 并发 | Kernel 1 和 Kernel 2 并发 Launch | 5-10% (prefill) |

### 7.5 目标性能

| 场景 | 目标延迟 | 备注 |
|------|---------|------|
| Prefill (B=2,S=4096) | < 5 ms | 全量计算 |
| Decode (B=2,S=1) | < 0.3 ms | 增量计算 |

---

## 8. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| AscendC BatchMatMul API 不支持 zero-copy broadcast | Kernel 3 需要显式复制 KV | 中 | 预留方案 B：Host 端预扩展 KV Cache |
| bf16 TopK 精度问题 | 相等值时选错索引 | 低 | 相等值时索引稳定性不影响正确性（只要选的是 top-k 值之一） |
| Kernel 3 的 L1 Buffer 不足 | Tiling 效率下降 | 低 | 减小 tiling 块大小，增加拆分步数 |
| int64 输出格式不匹配 | 下游处理错误 | 低 | 与下游约定输出格式（32-bit 索引足够） |
| Decode 阶段 launch overhead 过大 | 延迟超标 | 中 | 考虑 Kernel 融合（将 Kernel 1/2/3/4 融合为单一 decode 专用 Kernel） |

---

## 9. 交付物清单

| 文件 | 说明 | 完成标准 |
|------|------|---------|
| `ascendc/kernel_q_proj.cpp` | Kernel 1: Q 投影 | 精度与 torch 一致 |
| `ascendc/kernel_w_proj.cpp` | Kernel 2: 权重投影 | 精度与 torch 一致 |
| `ascendc/kernel_rope_score.cpp` | Kernel 3: RoPE + 打分 | 精度与 torch 一致 |
| `ascendc/kernel_post_topk.cpp` | Kernel 4: 后处理 + TopK | 精度与 torch 一致 |
| `ascendc/indexer_launcher.cpp` | Host 调度器 | 端到端可运行 |
| `ascendc/CMakeLists.txt` | 构建配置 | 一键编译 |
| `test/torch_ref/indexer_torch.py` | PyTorch 参考 | 基准可运行 |
| `test/test_prefill.py` | Prefill 测试 | 全部通过 |
| `test/test_decode.py` | Decode 测试 | 全部通过 |
| `benchmark/bench_indexer.py` | 性能测试 | 输出延迟报告 |
| `docs/DESIGN.md` | 架构设计文档 | 本文档 |
| `docs/PLAN.md` | 实施计划 | 本文件 |

---

## 10. 附录：Key Shape 速查表

### Prefill (start_pos=0)

| 参数 | 小 | 中 | 大 |
|------|----|----|-----|
| B | 2 | 2 | 2 |
| S | 64 | 512 | 4096 |
| kv_len | 16 | 128 | 1024 |
| q_flat | (128, 1024) | (1024, 1024) | (8192, 1024) |
| weights | (128, 16) | (1024, 16) | (8192, 16) |
| scores | (32, 64, 16) | (32, 512, 128) | (32, 4096, 1024) |
| topk | (2, 64, 16) | (2, 512, 128) | (2, 4096, 128) |

### Decode (start_pos>0, S=1)

| 参数 | 早期 | 中期 | 晚期 |
|------|------|------|------|
| B | 2 | 2 | 2 |
| S | 1 | 1 | 1 |
| start_pos | 64 | 512 | 4096 |
| kv_len | 16 | 128 | 1024 |
| q_flat | (2, 1024) | (2, 1024) | (2, 1024) |
| scores | (32, 1, 16) | (32, 1, 128) | (32, 1, 1024) |
| topk | (2, 1, 16) | (2, 1, 128) | (2, 1, 128) |
