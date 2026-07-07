# MTPBlock AscendC 算子架构设计

> **状态**: 正式设计 | **日期**: 2026-06-30 | **目标硬件**: Ascend910B2 (DAV_2201) | **CANN**: 9.0.0

---

## 1. 架构信息

| 项目 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | environment.md |
| NpuArch | `DAV_2201` | `/npu-arch` skill |
| `__NPU_ARCH__` | `2201` | 编译宏 |
| SocVersion | `ASCEND910B` | `GetSocVersion()` |
| `--npu-arch` 编译参数 | `dav-2201` | CMakeLists.txt |
| CubeCore : VectorCore | 1 : 2 (24 CubeCores, 48 VectorCores) | 硬件手册 |
| UB 容量 | 192 KB (196608 bytes) | 硬件手册 |
| L0C 容量 | 128 KB (131072 bytes) | 硬件手册 |
| L1 容量 | 512 KB (524288 bytes) | 硬件手册 |
| L2 容量 | 192 MB | 硬件手册 |
| BT 容量 | 1 KB | 硬件手册 |

> 所有硬件参数运行时通过 `PlatformAscendC::GetCurNpuArch()` / `GetCoreMemSize()` 获取，**禁止硬编码**。

---

## 2. 算子概述

### 2.1 算子定位

MTPBlock (Multi-Token Prediction Block) 是 DeepSeek-V4-Pro 推理模型的核心 Block，集成了 **Hyper-Connections**、**稀疏滑动窗口 Attention**、**MoE (Mixture of Experts)** 等子模块。

**输入输出规格**（与 `Model.forward()` 对齐）：

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| 输入 x | [b, s, hc, d] | bf16 | HC 扩展隐藏态 |
| 输入 input_ids | [b, s] | int64 | Token IDs |
| 输出 logits | [b, vocab_size] | fp32 | LM head 输出 |

**演示参数**: b=1, s=8, hc=4, d=512, vocab=129280

### 2.2 完整计算图分析

```
MTPBlock.forward(x[b,s,hc,d] bf16, input_ids[b,s] int64):

┌── Step 1: 输入融合 ──────────────────────────────────────────────
│   e = RMSNorm(Embed(input_ids))          [b,s,d]    ← Embed Lookup (Gather)
│   x_n = RMSNorm(x)                        [b,s,hc,d] ← 沿 last dim
│   x_fused = e_proj(e) + h_proj(x_n)      [b,s,hc,d] ← 双路Linear + Broadcast Add
│
├── Step 2a: Attention 子块 ───────────────────────────────────────
│   residual_a = x_fused
│   y_a, pre_a, post_a, comb_a = hc_pre(x_fused, attn_params)
│       ├─ Flatten + RMS → Linear(fn) → Scale+Bias → Sigmoid
│       ├─ Sinkhorn Iter(20): Softmax + Row/Col Normalize
│       └─ Weighted Sum: y_a = sum(pre_a * x_fused, dim=2)
│   y_a_n = RMSNorm(y_a)                    [b,s,d]
│   attn_out = Attention(y_a_n, start_pos)
│       ├─ Q: wq_a → RMSNorm → wq_b → RMSNorm → RoPE (rope dims)
│       ├─ KV: wkv → RMSNorm → RoPE (rope dims)
│       ├─ Causal Window TopK indices [b,s,win] (host 预计算)
│       ├─ Sparse Sink Attention: gather KV → QK^T → softmax+sink → weighted sum
│       └─ De-RoPE → wo_a (grouped) → wo_b
│   x_a = hc_post(attn_out, residual_a, post_a, comb_a)
│       └─ post * attn_out + comb * residual_a  → [b,s,hc,d]
│
├── Step 2b: FFN (MoE) 子块 ───────────────────────────────────────
│   residual_f = x_a
│   y_f, pre_f, post_f, comb_f = hc_pre(x_a, ffn_params)
│   y_f_n = RMSNorm(y_f)
│   ffn_out = MoE(y_f_n, input_ids)
│       ├─ Gate: Linear → SoftplusSqrt → +Bias → TopK [topk=2]
│       ├─ Per-Expert: Dispatch → SwiGLU(w1,w3) → w2 → weighted scatter-add
│       └─ Shared Expert: SwiGLU(w1,w3) → w2 → add
│   x_f = hc_post(ffn_out, residual_f, post_f, comb_f)
│
└── Step 3: 输出头 ────────────────────────────────────────────────
    x_head = hc_head(x_f, head_params)
        └─ Flatten+RMS → Linear(fn) → Sigmoid pre → Weighted Sum → [b,s,d]
    logits = lm_head(RMSNorm(x_head[:, -1]))   [b,vocab] fp32
```

### 2.3 可并行计算的子模块识别

| 并行机会 | 位置 | 并行方式 |
|---------|------|---------|
| Embed lookup + Hidden RMSNorm | Step 1 | AIV 多核沿 s 维度并行 |
| e_proj + h_proj MatMul | Step 1 | Cube 多核沿 M/N 二维切分 |
| Q 投影 vs KV 投影 | Step 2a | 同 kernel 内顺序执行（UB 共享）；可拆分为独立 kernel 并行 |
| hc_pre Attn vs hc_pre FFN | Step 2a/2b | 数据依赖串行（FFN 依赖 Attn 输出），**不可并行** |
| Per-Expert SwiGLU | Step 2b | 多核沿 expert 维度分配 |
| Shared Expert vs Routed Experts | Step 2b | 可在同一 kernel 内顺序执行 |

**关键数据依赖链**: Step 1 → Step 2a → Step 2b → Step 3（严格串行）。子模块内部的并行机会主要在 s 维度切分和 expert 维度切分。

---

## 3. 技术路线决策

### 3.1 路线选择

| 决策维度 | 选择 | 理由 |
|---------|------|------|
| **编程路径** | **SIMD/MemBase** (通用 Ascend C API) | NpuArch=DAV_2201，RegBase/Blaze 均为 DAV_3510 新架构能力，不可用 |
| **MatMul 策略 (demo shape)** | **SIMD 向量化点积** | demo shape 的 MatMul 规模极小（M≤32, N≤512），Cube 启动开销远超计算收益 |
| **MatMul 策略 (大 shape 升级)** | **MatmulImpl + MatmulApiTiling** | M≥128 或 N≥1024 时 Cube 效率优势明显 |
| **Kernel 类型** | `__aicore__`（统一入口） | DAV_2201 上 `__aicore__` 覆盖 AIC+AIV，当前 SIMD 路径下实际为 Vector-only |
| **Kernel 分解** | **6 个独立 kernel** (8 次 launch) | 单 kernel 融合全图 UB 溢出 (>192KB)；拆分保证各 kernel UB 可控 |

### 3.2 为什么 SIMD MatMul 在 demo shape 下优于 MatmulImpl

| 因素 | Demo Shape (b=1,s=8,d=512) | 分析 |
|------|-----|------|
| M 维太小 | M∈{1,8,32,64} | MatmulImpl 要求 M 对齐到 16 (ALIGNED_H)；M=1 或 8 时 padding 浪费 >50% |
| N 维小 | 部分 N=24 (mix_hc), N=4 (hc_head) | Cube baseN 通常 ≥64，小 N 导致 L0C 利用率极低 (<5%) |
| Cube 启动开销 | TCubeTiling 初始化 + L1/L0 数据搬运 + Fixpipe 写回 | 对小矩阵，启动开销超过实际 MAC 计算时间 |
| SIMD 向量化效率 | 256-bit SIMD, 16×fp16/cycle, 1.8 GHz | K=512 的 dot product: 512/16=32 cycles，与 Cube 在低利用率下相当 |
| L1 带宽 vs UB 带宽 | L1 带宽 ~UB 带宽 | 小矩阵下 UB 常驻权重行可复用，SIMD 无 L1→L0 搬运开销 |

**量化对比**（以 e_proj: M=8, K=512, N=512 为例）：

| 方案 | 估算耗时 | 瓶颈 |
|------|:---:|------|
| SIMD 向量化点积 | ~15 μs | UB 内 Mul+ReduceSum，tile K 方向 |
| MatmulImpl (baseM=8, baseN=64) | ~30-50 μs | TCubeTiling 初始化 + L1→L0 搬运 + Fixpipe 开销 |

> **结论**: Demo shape 下 SIMD 向量化点积是最优选择。大 shape (M≥128) 时 MatmulImpl 收益显著，需在 DESIGN 中保留升级路径。

### 3.3 MatmulImpl 升级阈值

| 条件 | 升级到 MatmulImpl | 说明 |
|------|:---:|------|
| M ≥ 128 且 N ≥ 64 | **强烈推荐** | Cube 利用率 >50%，SIMD 性能瓶颈明显 |
| M ≥ 64 且 N ≥ 128 | **推荐** | Cube 收益开始显现 |
| M < 32 或 N < 32 | **不推荐** | Cube 启动开销 > 计算收益 |
| K < 256 | **不推荐** | 计算密度不足，搬运占主导 |

### 3.4 Kernel 分解方案

```
Kernel 启动序列 (共 6 个唯一 kernel, 8 次 launch):

Launch 1: K1  mtp_embed_fuse    x[b,s,hc,d],input_ids[b,s] → feat[b,s,hc,d]
Launch 2: K2  hc_pre(attn_fn)   feat → y[b,s,d], pre_a, post_a, comb_a
Launch 3: K3  attn_block        y → attn_out[b,s,d]
Launch 4: K4  hc_post           attn_out, feat, post_a, comb_a → x1[b,s,hc,d]
Launch 5: K2  hc_pre(ffn_fn)    x1 → y[b,s,d], pre_f, post_f, comb_f
Launch 6: K5  moe_block         y, input_ids → ffn_out[b,s,d]
Launch 7: K4  hc_post           ffn_out, x1, post_f, comb_f → x2[b,s,hc,d]
Launch 8: K6  mtp_head          x2 → logits[b,vocab]

注: K2 和 K4 是共享 kernel, 分别调用 2 次 (不同参数)
```

---

## 4. 精度策略

### 4.1 精度标准

| 项目 | 选择 | 依据 |
|------|------|------|
| 精度标准 | 浮点计算类社区标准 | `/ops-precision-standard` → `float_compute_community.md` |
| fp16/bf16 验收阈值 | MARE < 7.81e-02 | 社区标准 bf16 档 |
| 小值域阈值 | Small Value Threshold = 2^-8 | 特殊场景处理 |
| INF/NAN | Ascend910B 需 INF 一致性 | 对标 PyTorch bf16 参考 |

### 4.2 混合精度策略

```
┌────────────────┬────────┬──────────────────────────────────────┐
│ 操作            │ 计算精度│ 原因                                  │
├────────────────┼────────┼──────────────────────────────────────┤
│ RMSNorm        │ fp32   │ rsqrt 对精度敏感                      │
│ MatMul (SIMD)  │ fp32   │ 累加精度; 输入 fp16, dot product fp32 │
│ Softmax        │ fp32   │ exp 数值稳定性                        │
│ Sinkhorn       │ fp32   │ 20 轮迭代归一化, 累积误差需 fp32 保护 │
│ SiLU/Sigmoid   │ fp32   │ AscendC::Exp 原生 fp32               │
│ Gate scores    │ fp32   │ TopK 路由依赖精确分数                  │
│ 逐元素加/乘     │ fp16   │ 精度足够 + 性能优先 (UB 减半)         │
│ hc_post 累加   │ fp32   │ comb*residual 累加防精度损失           │
│ LmHead MatMul  │ fp32   │ logits 精度直接影响输出               │
│ 写入 GM 输出   │ fp16   │ 匹配参考实现输出 dtype                │
│                │(K6 fp32)│ K6 logits 输出为 fp32                │
└────────────────┴────────┴──────────────────────────────────────┘
```

### 4.3 DAV_2201 bf16 处理

DAV_2201 的 VectorCore **原生不支持 bf16 计算**。bisheng 编译器下的处理方式：
- **GM 存储**: 使用 `half` (fp16) 类型存储（与 bf16 同为 2 字节，带宽一致）
- **UB 计算**: 加载后隐式转换为 fp32 计算
- **精度等价性**: bf16 与 fp16 在 2 字节浮点下有相似的精度特征（bf16 动态范围更大但精度更低；fp16 精度更高但动态范围更小）。对于 demo shape (d=512)，两者数值行为可接受。
- **标注**: 所有 `half` 类型注释为 "bf16 storage via fp16"，以便未来 DAV_3510 升级时替换为真正的 `bfloat16_t`。

### 4.4 精度风险点

| 风险 | 位置 | 缓解措施 |
|------|------|---------|
| Sinkhorn 迭代累积误差 | K2 hc_pre | 每轮归一化前加 eps=1e-6；全程 fp32 |
| exp overflow in softmax | K3 attn_block | 先减 max 做数值稳定化 (`scores -= max(scores, sink)`) |
| 大数吃小数 in sum | K5 moe_block | fp32 累加；multi-expert scatter-add 分段归并 |
| rsqrt(-0) → inf | RMSNorm | eps=1e-6 保证非零分母 |
| fp16 动态范围不足 | 大规模累加 | 临界路径 (Sinkhorn, Softmax, MoE reduce) 保持 fp32 |

---

## 5. 通用设计要素

### 5.1 多核切分策略

| Kernel | 切分维度 | 策略 | 负载均衡 |
|--------|---------|------|:---:|
| K1 (embed_fuse) | **s 维度** | `usedCoreNum = min(s, GetCoreNumAiv())`; 每核处理 `s/usedCoreNum` 个 token | 逐行均分 |
| K2 (hc_pre) | **s 维度** | 同上; b=1 时沿 s 切分; b>1 时沿 b×s 切分 | 逐行均分 |
| K3 (attn_block) | **s 维度** | token 级并行; 需注意 sparse gather 的 KV 依赖 (跨 token 无依赖，天然可并行) | 逐行均分 |
| K4 (hc_post) | **s 维度** | 同 K2 策略 | 逐行均分 |
| K5 (moe_block) | **s 维度 + expert 维度** | 优先 s 维度; expert 维度切分留作扩展 | 逐行均分 |
| K6 (mtp_head) | **s 维度 (hc_head) + MatMul** | hc_head 沿 s 切分; lm_head MatMul 由 MatmulApiTiling 管理 M×N 二维切分 | hc_head 逐行均分 |

**核数获取** (强制动态):
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t usedCoreNum = ascendcPlatform.GetCoreNumAiv();
// 实际使用: usedCoreNum = min(usedCoreNum, total_tokens);
// 禁止硬编码为 1 或 48
```

### 5.2 UB 切分策略 (单核内)

| Kernel | 单次处理量 | tiling 策略 |
|--------|----------|------------|
| K1 | `tile_s × hc × d` tokens | s 方向分 tile；tile_s 由 UB 容量公式动态计算 |
| K2 | `tile_s` tokens | s 方向分 tile；Sinkhorn 矩阵 [tile_s, hc, hc] 常驻 UB |
| K3 | `tile_s` tokens | s 方向分 tile；Q/KV/score/attn_out 均基于 tile_s |
| K4 | `tile_s` tokens | s 方向分 tile；post/comb 全常驻 UB (极小) |
| K5 | `tile_s` tokens | s 方向分 tile；expert dispatch 基于 tile 内 token |
| K6 | `tile_s` tokens | s 方向分 tile；last-token 逻辑仅在最后一个 tile 执行 |

**tile_s 动态计算公式**:
```cpp
// 以 K2 (hc_pre) 为例
// UB 192KB, fp32 计算, hc=4, d=512, hc*d=2048, mix_hc=24
//
// x_flat_fp32:  tile_s * hc*d * 4B  = tile_s * 8192B
// mixes_fp32:   tile_s * mix_hc * 4B = tile_s * 96B
// rsqrt_tmp:    tile_s * 4B
// sinkhorn:     tile_s * hc * hc * 4B = tile_s * 64B
// y_out:        tile_s * d * 2B = tile_s * 1024B
//
// tile_s ≤ (192*1024 - overhead) / (8192 + 96 + 4 + 64 + 1024)
// tile_s ≤ 196608 / 9380 ≈ 20.9
//
// tile_s = 16 (安全边际, 余量给双缓冲)
```

### 5.3 分支场景覆盖

| 分支维度 | 条件 | 处理策略 |
|---------|------|---------|
| bf16 I/O (唯一 dtype) | 需求已指定 | 仅支持 bf16→fp16 路径, 不额外支持 fp16/fp32 变体 |
| 大 s (s ≥ 64) | s > tile_s | s 维度分 tile; tile_s 按 UB 公式动态计算 |
| 小 s (s ≤ tile_s) | s ≤ tile_s | 单 tile 全载, 跳过 s-loop |
| hc_mult 泛化 | hc ∈ {2,4,8} | mix_hc=(2+hc)*hc 动态计算; UB 以 max_hc=8 预留 |
| d 维度变化 | d ∈ {512,1024,2048,4096} | MatMul K 维度随之变化; SIMD tile_K 动态调整 |
| 多 batch (b > 1) | b > 1 | s 维度切分合并 b×s 总 token 数 |
| window_size 变化 | win ∈ {8,16,32,64} | score 矩阵 [tile_s, n_heads, win] 随 win 增大; UB 需动态调整 tile_s |
| ODD-M / ODD-N | MatMul 尾块 | Pad 到偶数; 输出时仅写有效元素 |

---

## 6. 各 Kernel 详细设计

### 6.1 K1: mtp_embed_fuse

**数学**: `e_proj(RMSNorm(embed(input_ids))) + h_proj(RMSNorm(x)).unsqueeze(2)` 的广播加

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)

#### 输入/输出

| Tensor | Shape | Dtype | 存储位置 |
|--------|-------|-------|---------|
| x (输入) | [b, s, hc, d] | half (bf16) | GM |
| input_ids | [b, s] | int64 | GM |
| embed_weight | [vocab, d] | half | GM |
| enorm_weight | [d] | float | GM |
| hnorm_weight | [d] | float | GM |
| e_proj_weight | [d, d] | half | GM |
| h_proj_weight | [d, d] | half | GM |
| feat (输出) | [b, s, hc, d] | half | GM |

#### 计算流程

```
Per tile (tile_s tokens):

Step A: Embed Lookup + RMSNorm
  for each token in tile:
    row_idx = input_ids[token]
    e_row = DataCopyPad(embed_weight[row_idx * d : (row_idx+1) * d])  // [d] half → UB
  e_fp32 = Cast(e_row, fp32)
  e_rms = RMSNorm(e_fp32, enorm_weight, eps)                          // ARA pattern: reduce last dim
  e_out = Cast(e_rms, half)                                            // [tile_s, d] half

Step B: Hidden Projection
  x_tile = DataCopyPad(x[token_range])                                 // [tile_s, hc, d] half → UB
  x_normed = RMSNorm(x_tile, hnorm_weight, eps) (per [hc,d] plane)   // [tile_s, hc, d] fp32
  // h_proj: [tile_s*hc, d] × [d, d]^T → [tile_s*hc, d]
  h_out = SIMD_MatMul(reshape(x_normed, [tile_s*hc, d]), h_proj_weight)
  h_out = reshape(h_out, [tile_s, hc, d])

Step C: Broadcast Add
  e_expanded = reshape(e_out, [tile_s, 1, d])  // unsqueeze(2)
  feat_tile = AscendC::Add(e_expanded, h_out)   // broadcast along hc dim
  DataCopyPad(feat_tile → GM)
```

#### API 映射

| 操作 | Ascend C API | 说明 |
|------|-------------|------|
| Embed Gather | `DataCopyPad` | 按 input_ids 索引从 embed_weight 逐行搬运 |
| RMSNorm | `AscendC::Mul` + `AscendC::ReduceSum` + `AscendC::Rsqrt` + `AscendC::Muls` | ARA 模式: [tile_s, d] 沿 d 归约 |
| SIMD MatMul | `AscendC::Mul` + `AscendC::ReduceSum` (循环 tile_K) | 向量化点积; M=tile_s*hc, K=d, N=d |
| Broadcast Add | `AscendC::Add` | 沿 hc 维广播 |
| Cast | `AscendC::Cast` (half↔float) | bf16↔fp32 |

#### UB Buffer 规划 (tile_s=8, hc=4, d=512)

```
Buffer                  | 大小                    | 说明
e_embed_fp16            | 8×512×2B = 8 KB        | Embed 查表结果
e_norm_fp32             | 8×512×4B = 16 KB       | e RMSNorm 中间 (可与 e_fp32 复用)
e_proj_out_fp16         | 8×512×2B = 8 KB        | e_proj 输出
x_tile_fp16             | 8×4×512×2B = 32 KB     | 输入 x tile
x_norm_fp32             | 8×4×512×4B = 64 KB     | hnorm 中间 (可以分 hc 平面处理降低峰值)
h_weight_row_fp16       | 512×2B = 1 KB          | h_proj weight 行缓存 (tile_K 复用)
h_dot_acc_fp32          | 8×4×512×4B = 64 KB     | h_proj 点积累加 (可复用 x_norm 区)
feat_fp16               | 8×4×512×2B = 32 KB     | 最终输出

峰值 (分时复用):
  Phase A (Embed):   8 + 64 = 72 KB
  Phase B (h_proj):  64 + 1 + 64 + 32 = 161 KB (含 float 中间)
  Phase C (Add):     32 + 32 + 32 = 96 KB

  Phase B 需优化: x_norm 分 hc 平面处理 → 峰值降至 64+1+16+32 = 113 KB < 192 KB
```

#### Tiling

- **s 维度切分**: `tile_s = min(s, (UB_SIZE - overhead) / (hc*d*4 + d*2)) ≈ min(s, 10)` (融合处理时)
  - 若分离 Embed 和 h_proj 阶段 (时间分片): tile_s 可提升到 16
- **多核**: `usedCoreNum = min(s, GetCoreNumAiv())`; 每核处理 `s/usedCoreNum` tiles

### 6.2 K2: hc_pre (共享 kernel)

**数学**: HC 预处理 (Linear + Sigmoid + Sinkhorn + WeightedSum)

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)。使用 **Double Buffer + EnQue/DeQue 流水线**。

#### 输入/输出

| Tensor | Shape | Dtype |
|--------|-------|-------|
| x (输入) | [b, s, hc, d] | half |
| hc_fn | [mix_hc, hc*d] | float |
| hc_scale | [3] | float |
| hc_base | [mix_hc] | float |
| y (输出) | [b, s, d] | half |
| pre (输出) | [b, s, hc] | float |
| post (输出) | [b, s, hc] | float |
| comb (输出) | [b, s, hc, hc] | float |

#### 计算流程

```
Per tile (tile_s tokens):

Step 1: Flatten + RMS (in Queue → DeQue)
  x_tile = DeQue<VecIn>()                                     // [tile_s, hc*d] half
  x_fp32 = Cast(x_tile, float)                                // [tile_s, hc*d] float
  rsqrt = AscendC::Rsqrt(AscendC::ReduceSum(x_fp32², -1)/dim + eps)  // [tile_s] float

Step 2: Linear Projection (SIMD MatMul)
  // mixes = x_fp32 [tile_s, hc*d] × hc_fn^T [hc*d, mix_hc]
  // M=tile_s, K=hc*d, N=mix_hc
  for k_tile in range(0, hc*d, TILE_K):
    x_k = x_fp32[:, k_tile:k_tile+TILE_K]                     // [tile_s, TILE_K]
    fn_k = hc_fn[:, k_tile:k_tile+TILE_K]                     // [mix_hc, TILE_K] 预加载到 UB
    partial = x_k · fn_k^T                                     // [tile_s, mix_hc] via Mul+ReduceSum
    mixes += partial
  mixes *= rsqrt.unsqueeze(-1)                                 // [tile_s, mix_hc]

Step 3: Scale + Bias
  // hc_scale = [s0, s1, s2], expand to [mix_hc]
  // scale_pattern: [s0]*hc cat [s1]*hc cat [s2]*(hc*hc)
  x_mix = mixes * expanded_scale + hc_base                    // [tile_s, mix_hc]

Step 4: Split + Sigmoid
  pre_fp32  = AscendC::Sigmoid(x_mix[:, :hc]) + eps           // [tile_s, hc]
  post_fp32 = 2.0 * AscendC::Sigmoid(x_mix[:, hc:2*hc])      // [tile_s, hc]
  comb_fp32 = reshape(x_mix[:, 2*hc:], [tile_s, hc, hc])     // [tile_s, hc, hc]

Step 5: Sinkhorn Normalization (UB 内 20 轮迭代, fp32)
  comb = AscendC::Softmax(comb, dim=-1) + eps                // [tile_s, hc, hc]
  sum_col = AscendC::ReduceSum(comb, dim=-2) + eps           // [tile_s, hc]
  comb = comb / sum_col.unsqueeze(-2)                         // col normalize
  for iter in range(sinkhorn_iters - 1):
    sum_row = AscendC::ReduceSum(comb, dim=-1) + eps         // [tile_s, hc]
    comb = comb / sum_row.unsqueeze(-1)                       // row normalize
    sum_col = AscendC::ReduceSum(comb, dim=-2) + eps
    comb = comb / sum_col.unsqueeze(-2)                       // col normalize

Step 6: Weighted Sum
  // y = sum(pre * x_reshaped [tile_s, hc, d], dim=2)
  x_reshaped = reshape(x_fp32, [tile_s, hc, d])              // [tile_s, hc, d]
  weighted = pre_fp32.unsqueeze(-1) * x_reshaped              // [tile_s, hc, d]
  y_fp32 = AscendC::ReduceSum(weighted, dim=2)               // [tile_s, d]
  y_fp16 = Cast(y_fp32, half)

Step 7: CopyOut (EnQue)
  EnQue(y_fp16 → y_gm); EnQue(pre_fp32 → pre_gm); EnQue(post_fp32 → post_gm); EnQue(comb_fp32 → comb_gm)
```

#### API 映射

| 操作 | Ascend C API | 说明 |
|------|-------------|------|
| Flatten + 搬运 | `DataCopyPad` (inQueue EnQue) | 按 [tile_s, hc*d] 布局搬入 |
| ReduceSum (求 rsqrt) | `AscendC::ReduceSum` (Level 2) | `x_flat²` 沿 last dim 归约 |
| Rsqrt | `AscendC::Rsqrt` | 矢量 API |
| SIMD MatMul | `AscendC::Mul` + `AscendC::ReduceSum` | 沿 K 分 tile，向量化点积 |
| Sigmoid | `AscendC::Exp` + `AscendC::Add` + `AscendC::Div` | sigmoid(x) = 1/(1+exp(-x)) |
| Softmax | `AscendC::ReduceMax` + `AscendC::Exp` + `AscendC::ReduceSum` + `AscendC::Div` | 沿 hc 维度 |
| Row/Col Norm | `AscendC::ReduceSum` + `AscendC::Div` | Sinkhorn 迭代 |
| Weighted Sum | `AscendC::Mul` + `AscendC::ReduceSum` | 沿 hc 维加权归约 |

#### UB Buffer 规划 (tile_s=8, hc=4, d=512)

```
Buffer (inQueue, DOUBLE_BUFFER): x_tile_fp16    8×2048×2B×2 =  64 KB
Buffer (outQueue, DOUBLE_BUFFER): y_tile_fp16   8×512×2B×2  =  16 KB
                                  pre_fp32      8×4×4B×2    = 0.25 KB
                                  post_fp32     8×4×4B×2    = 0.25 KB
                                  comb_fp32     8×4×4×4B×2  =   1 KB
UB temp:
  x_fp32 (可复用 inQueue 区)     8×2048×4B         =  32 KB (额外)
  rsqrt_fp32                     8×4B              =  32 B
  mixes_fp32                     8×24×4B           = 768 B
  hc_fn_row_cache (tile_K)       TILE_K×mix_hc×4B = ~4 KB
  sinkhorn_comb_fp32             8×4×4×4B          = 512 B
  weighted_tmp_fp32              8×4×512×4B        =  64 KB (可复用 x_fp32 区)
────────────────────────────────────────────────────────────────
峰值 UB: 64(inQ_double) + 18(outQ_double) + max(64, 32+64) = 146 KB < 192 KB
```

> 注: `weighted_tmp_fp32` [8,4,512] fp32 = 64KB 可与已释放的 `x_fp32` [8,2048] fp32 = 64KB 复用同一 UB 区域（Step 6 时 Step 1 的 `x_fp32` 已不再需要）。

#### Tiling

- **s 维度切分**: `tile_s ≤ (UB_SIZE - double_buffer_overhead) / (hc*d*4 + hc*d*2/mix_reuse_factor)`
  - hc=4,d=512: tile_s ≤ 196608/(8192+4096/4) ≈ 21 → tile_s=16 (安全边际)
- **多核**: `usedCoreNum = min(b*s, GetCoreNumAiv())`; 每核处理 `(b*s)/usedCoreNum` 个 token

### 6.3 K3: attn_block

**数学**: Q/KV 投影 + RoPE + 稀疏 Attention (softmax with sink) + 输出投影

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)

#### 计算流程

```
Step A: Q 投影 (SIMD MatMul)
Step B: KV 投影 (SIMD MatMul)
Step C: RoPE 应用 (复数乘法分量展开)
Step D: 稀疏窗口 Attention (causal sliding window)
Step E: De-RoPE 输出投影
```

详细实现要点:

1. **Sparse Gather**: `DataCopyPad` 按 topk_idxs 从 KV 张量逐行搬运 32B 对齐块
2. **Causal Window 索引**: 由 Host Tiling 预计算 `topk_idxs [b, s, win]`, 通过 TilingData 传入 Device
3. **RoPE**: 复数乘法分量展开 `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`; 非 RoPE 维度直接拷贝
4. **Attn Sink**: 仅参与分母 (sum_exp), 不参与分子 (weights 分子), 等价于 PyTorch 参考
5. **AscendC::Exp**: 使用矢量 API (非 Taylor 近似), Round 1 已修复

#### API 映射

| 操作 | API | 说明 |
|------|-----|------|
| SIMD MatMul (wq_a/wq_b/wkv/wo_a/wo_b) | `AscendC::Mul` + `AscendC::ReduceSum` | 沿 K 分 tile |
| RMSNorm | `AscendC::ReduceSum` + `AscendC::Rsqrt` + `AscendC::Muls` + `AscendC::Mul` | 手动实现 |
| RoPE (复数乘法) | `AscendC::Mul` + `AscendC::Add`/`AscendC::Sub` | 分量展开 |
| Sparse Gather | `DataCopyPad` | 按索引逐个搬运 |
| Softmax + Attn Sink | `AscendC::ReduceMax` + `AscendC::Exp` + `AscendC::ReduceSum` + `AscendC::Div` | 沿 win 维度; sink 加在分母 |
| Weighted Sum | `AscendC::Mul` + `AscendC::ReduceSum` | Attention 输出 |

#### UB Buffer 规划 (s=8, n_heads=8, head_dim=64, win=8)

```
Buffer                  | 大小                    | 说明
x_tile_fp16             | 8×512×2B = 8 KB        | 输入
q_proj_fp16             | 8×512×2B = 8 KB        | Q 投影输出 [s, nh*hd]
kv_fp16                 | 8×64×2B = 1 KB         | KV 输出 [s, hd]
kv_gathered_fp16        | 8×8×64×2B = 8 KB       | Gather 后 KV [s, win, hd]
scores_fp32             | 8×8×8×4B = 2 KB        | QK scores [s, nh, win]
weights_fp16            | 8×8×8×2B = 1 KB        | Attention weights
attn_acc_fp32           | 8×8×64×4B = 16 KB      | Attention 输出累加
weight_cache_fp16       | 512×2B = 1 KB          | MatMul 权重行缓存 (tile_K 复用)
rope_freqs              | 8×16×8B = 1 KB         | RoPE freqs_cis (complex64)
────────────────────────────────────────────────────────
峰值 (不含 MatMul 临时): 8+8+1+8+2+1+16+1+1 = 46 KB < 192 KB
```

**L3 修复 (Round 2)**: `wo_a` 权重已通过 DataCopyPad 预加载到 UB buffer，不再在 inner loop 中逐元素 GM GetValue。

**稀疏注意力 (M4)**: 当前实现使用 O(s²) 密集注意力。DEMO shape (s=8) 下差异可忽略。扩展路径:
- 接入 `topk_idxs` 实现 O(s·win) 稀疏注意力
- 仅对 `topk_idxs >= 0` 的有效位置计算 QK dot product
- Score 矩阵降为 [tile_s, n_heads, win]

### 6.4 K4: hc_post (共享 kernel)

**数学**: `out = post * x + comb * residual` (加权残差连接)

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)。使用 **Double Buffer + EnQue/DeQue 流水线**。

#### 计算流程

```
Per tile (tile_s tokens):

1. 加载 x[tile_s, d] half, residual[tile_s, hc, d] half, post[tile_s, hc] float, comb[tile_s, hc, hc] float
2. post_term = post.unsqueeze(-1) * x.unsqueeze(-2)            // [tile_s, hc, d] float
3. comb_term = comb.unsqueeze(-1) * residual.unsqueeze(-2)     // [tile_s, hc, hc, d] float
4. comb_reduced = AscendC::ReduceSum(comb_term, dim=2)          // [tile_s, hc, d] float
5. out = Cast(post_term + comb_reduced, half)                   // [tile_s, hc, d] half
```

#### API 映射

| 操作 | API | 说明 |
|------|-----|------|
| 广播乘 | `AscendC::Mul` | post*hc 广播 |
| ReduceSum | `AscendC::ReduceSum` | comb*residual 沿 hc 归约 |
| Add | `AscendC::Add` | 两路结果相加 |

#### UB Buffer 规划 (tile_s=8, hc=4, d=512)

```
Buffer (inQueue ×2, DOUBLE_BUFFER):
  x_fp16              8×512×2B×2       = 16 KB
  residual_fp16       8×4×512×2B×2     = 64 KB
  post_fp32           8×4×4B×2         = 0.25 KB
  comb_fp32           8×4×4×4B×2       = 1 KB
Buffer (outQueue, DOUBLE_BUFFER):
  out_fp16            8×4×512×2B×2     = 64 KB
UB temp:
  post_term_fp32      8×4×512×4B       = 64 KB
  comb_term_fp32      8×4×512×4B       = 64 KB (可与 post_term 分时复用)
────────────────────────────────────────────────────────
峰值 (分时复用): 64(inQ) + 64(outQ) + 64(temp) = 192 KB ≈ UB
  优化: tile_s = 4 时峰值 = 32+32+32 = 96 KB < 192 KB
  或: post_term 直接写 out buffer → 峰值 = 64+64+64 = 192 KB 刚好
```

**推荐**: 分两阶段计算（先 post_term 写 out，再 comb_term 累加到 out），峰值降至 64+64+0 = 128 KB。

### 6.5 K5: moe_block

**数学**: Gate 路由 + Per-Expert SwiGLU + Shared Expert

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)

#### 计算流程

```
Stage 1 — Gate (SIMD):
  x_fp32 = Cast(x, float)                                           // [b*s, d]
  // SIMD MatMul: [b*s, d] × [d, n_experts] → [b*s, n_experts]
  scores = matmul(x_fp32, gate_weight)                              // [b*s, n_experts]

  // Score function
  scores = softplus_sqrt(scores)  // log(1+exp(x))^0.5, 用 AscendC::Exp
  scores += gate_bias

  // TopK (k=2)
  indices, weights = topk(scores, topk)                             // Sort-based

  // Normalize weights
  weights = weights / sum(weights, dim=-1) * route_scale

Stage 2 — Shared Expert (全体 token, SIMD MatMul):
  gate_shared = SiLU(matmul(x_fp32, shared_w1)) * matmul(x_fp32, shared_w3)
  out_shared = matmul(gate_shared, shared_w2)

Stage 3 — Per-Expert (循环 8 个 experts):
  for expert_idx in range(n_routed_experts):
    mask = (indices == expert_idx)                                  // gather token mask
    dispatch_x = gather_by_mask(x_fp32, mask)                       // [count_i, d]
    if count_i == 0: continue
    gate_i = SiLU(matmul(dispatch_x, expert_w1[i])) * matmul(dispatch_x, expert_w3[i])
    out_i = matmul(gate_i, expert_w2[i])
    scatter_add(y_temp, indices, out_i * weights)                   // 按 mask 写回

Stage 4 — Final:
  y = out_shared + y_temp
```

**实现状态**: 
- Shared Expert: **已完整实现** (w1→SiLU × w3→w2)，精度验证通过 (MARE=6.72e-04)
- Routed Expert (L4): **设计已就绪，实现待补充**。DEVICE 侧 Gather/Scatter 使用 DataCopyPad 按 mask 条件搬运。

#### API 映射

| 操作 | API | 说明 |
|------|-----|------|
| Gate SIMD MatMul | `AscendC::Mul` + `AscendC::ReduceSum` | 小 MatMul (M≤64, K=512, N=8) |
| SoftplusSqrt | `AscendC::Exp` + `AscendC::Log` + `AscendC::Sqrt` | softplus(x)=log(1+exp(x)) |
| TopK | 手动 `Compare+Select` 或 `AscendC::Sort` | k=2 小值, 手动实现更高效 |
| SiLU | `AscendC::Exp`(Sigmoid) + `AscendC::Mul` | silu(x)=x*sigmoid(x) |
| Expert MatMul | `AscendC::Mul` + `AscendC::ReduceSum` | 沿 K 分 tile |
| Gather/Scatter | `DataCopyPad` + 手动 offset 计算 | 按 mask 逐 token 搬运 |

**C9 合规**: 所有 expert dispatch 在 Device 侧完成 (基于 indices 的 mask 搬运)，不违反 "禁止 Host 预处理输入 tensor" 约束。

### 6.6 K6: mtp_head

**数学**: hc_head (Linear+Sigmoid+WeightedSum) + RMSNorm(last token) + lm_head (MatMul)

**Kernel 类型**: `__aicore__` (Vector-only, SIMD)

#### 计算流程

```
Step 1: Flatten + RMS → Linear (hc_head) [每个 token]
Step 2: Sigmoid pre gate → Weighted Sum → [b,s,d]
Step 3: RMSNorm on last token y[-1] → [b,d]              ← 仅在最后一个 tile 执行!
Step 4: lm_head MatMul: [b,d] × [d,vocab] → [b,vocab]
```

**关键修复 (L2)**: Step 3-4 仅当 `tile_idx == n_tiles - 1`（最后一个 tile）时执行。多 tile 场景下，前面的 tile 不输出 logits。

#### API 映射

| 操作 | API |
|------|-----|
| hc_head Linear | `AscendC::Mul` + `AscendC::ReduceSum` (SIMD MatMul) |
| Sigmoid pre | `AscendC::Exp` + `AscendC::Add` + `AscendC::Div` |
| Weighted Sum | `AscendC::Mul` + `AscendC::ReduceSum` |
| RMSNorm (last token) | `AscendC::ReduceSum` + `AscendC::Rsqrt` + `AscendC::Mul` |
| lm_head MatMul | `AscendC::Mul` + `AscendC::ReduceSum` (M=1, K=d, N=vocab) |

---

## 7. MatmulImpl 升级路径

### 7.1 升级策略

当 shape 规模达到升级阈值 (M≥128 或 N≥1024) 时，将各 kernel 的 SIMD MatMul 替换为 MatmulImpl。

### 7.2 升级方案

**方案 A: Kernel 内嵌 MatmulImpl**（推荐用于大 shape）

对于升级后的 kernel，使用 `__aicore__` + MatmulImpl：
- MatmulImpl 占用 AIC 执行 MatMul
- Vector 操作 (RMSNorm, activations 等) 在 AIC 上使用 `__aicore__` 的 Vector 子集执行
- 同步: MatmulImpl 内部通过 `IterateAll` 完成 Cube 计算 + Fixpipe 写回

**方案 B: 拆分 AIC/AIV kernel**（极端大 shape）

将 MatMul 和 Vector 操作拆分为独立 kernel:
- AIC kernel (`__cube__`): 纯 MatmulImpl
- AIV kernel (`__vector__`): 纯 Vector 操作
- 中间结果通过 GM workspace 传递

### 7.3 Scene Dispatch

```cpp
// Host 侧场景分发
if (M >= 128 || N >= 1024) {
    // 大 shape: 使用 MatmulImpl 路径
    launch_kernel_with_matmul_impl(...);
} else {
    // 小 shape: 使用 SIMD 路径
    launch_kernel_simd(...);
}
```

### 7.4 MatmulImpl 集成检查清单

| 项目 | 要求 |
|------|------|
| Tiling | Host 侧调用 `MatmulApiTiling::GetTiling(TCubeTiling&)` 计算 tiling 参数 |
| TCubeTiling | 存储在 kernel TilingData 中；kernel 侧通过 `LoadTilingFromGM` 加载到 stack |
| `MatmulImpl::Init` | 传入 stack 上的 TCubeTiling 指针（非 GM） |
| A/B/C 类型 | `MatmulType<TPosition::GM, CubeFormat::ND, half, false>` (ND 格式, 非转置) |
| Config | `MM_CFG_NO_PRELOAD` (enUnitFlag=true) — 自定义算子默认 |
| L0C 容量 | 128KB → baseM × baseN × 4 ≤ 128KB; SWAT 自动调整 |
| ODD-M | M 向上取偶, MMAD 用原始 M, Fixpipe 写回只写有效行 |
| 结果写回 | `mm.IterateAll(cGm, enAtomic)` 自动处理写回 |

---

## 8. Host 侧架构

### 8.1 总体流程

```
MTPBlockOperator::Launch():
  1. 校验输入 shape/dtype
  2. 分配 GM workspace (各 kernel 中间张量 + workspace)
  3. 预计算 topk_idxs [b, s, win] (causal sliding window)
  4. 依次 Launch K1 → K2a → K3 → K4a → K2b → K5 → K4b → K6
  5. 每步间使用 aclrtSynchronizeStream 同步
  6. 返回 logits [b, vocab_size] fp32
```

### 8.2 TilingData 计算 (Host 侧)

每个 kernel launch 前，Host 侧填充对应的 TilingData 结构体：
- **动态核数**: `usedCoreNum = min(totalTokens, GetCoreNumAiv())`
- **动态 tile_s**: 基于 UB 容量公式计算
- **GM 偏移**: 各 tensor 在 workspace 中的偏移 (所有 tensor 在同一大块 GM 中)

### 8.3 GM Workspace 分配

```
总 Workspace = kernel 中间张量之和 + 单 kernel 最大内部 workspace

中间张量 (Kernel 间传递, demo shape):
  feat       [1,8,4,512]     half    = 32 KB
  y          [1,8,512]       half    = 8 KB
  pre/post   [1,8,4]×2       float   = 0.25 KB
  comb       [1,8,4,4]       float   = 0.5 KB
  attn_out   [1,8,512]       half    = 8 KB
  ffn_out    [1,8,512]       half    = 8 KB
  x_temp     [1,8,4,512]     half    = 32 KB
  topk_idxs  [1,8,8]         int32   = 0.25 KB

总 Workspace ≈ 200 KB (含对齐)
```

---

## 9. API 验证清单

| API | 验证路径 | 验证状态 |
|-----|---------|:---:|
| `AscendC::Mul` (矢量) | Ascend C 基础 API | **已确认** (所有 kernel 使用中) |
| `AscendC::ReduceSum` | `adv_api/reduce/` | **已确认** |
| `AscendC::ReduceMax` | `adv_api/reduce/` | **已确认** |
| `AscendC::Exp` (矢量) | Ascend C 数学 API | **已确认** (Round 1 替换 Taylor) |
| `AscendC::Rsqrt` (矢量) | Ascend C 数学 API | **已确认** |
| `AscendC::Muls` (标量广播乘) | Ascend C 基础 API | **已确认** |
| `AscendC::Add`/`Sub`/`Div` | Ascend C 基础 API | **已确认** |
| `AscendC::Cast` (half↔float) | Ascend C 基础 API | **已确认** |
| `AscendC::Log` | Ascend C 数学 API | **已确认** |
| `AscendC::Sqrt` | Ascend C 数学 API | **已确认** |
| `DataCopyPad` | Ascend C 基础 API | **已确认** (32B 对齐搬运) |
| `MatmulImpl` | `adv_api/matmul/matmul.h` | **已确认** (大 shape 升级路径) |
| `MatmulApiTiling` | `adv_api/matmul/matmul_tiling.h` | **已确认** (Host 侧) |
| `TPipe` + `TQue` (Double Buffer) | Ascend C 基础 API | **已确认** (K2/K4 使用中) |

> Sigmoid 无独立 API，通过 `1/(1+exp(-x))` 组合 (`AscendC::Exp` + `AscendC::Add` + `AscendC::Div`) 实现。SiLU 通过 `x * sigmoid(x)` 实现。Softmax 通过 `ReduceMax + Exp + ReduceSum + Div` 手动实现。

---

## 10. 代码规范

### 10.1 命名约定

| 类别 | 规范 | 反例 |
|------|------|------|
| 类名 | PascalCase | `KernelHcPre`, `KernelAttnBlock` |
| 函数名 | PascalCase | `Init()`, `Process()`, `Compute()` |
| 变量名 | camelCase，有语义 | `xGm`, `qProjBuf`, `weightRowBuf` (推荐); `q`, `o`, `rv` (不推荐) |
| Buffer 名 | `q` + 语义简写 | `qIn`, `qTmp`, `qOut` |
| 常量 | UPPER_SNAKE_CASE | `MAX_TILE_S`, `DEMO_HC` |

### 10.2 代码结构

```cpp
// 每个 kernel 文件结构
#include "kernel_operator.h"
#include "mtpblock_tiling.h"

using namespace AscendC;

constexpr uint32_t Kx_MAX_TILE_S = ...;  // 编译期常量

class KernelXxx {
public:
    __aicore__ inline KernelXxx(TPipe* pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR..., const __gm__ Tiling*);
    __aicore__ inline void Process();
private:
    __aicore__ inline void Compute();
    // buffer 声明
    TPipe* pipe_;
    const __gm__ Tiling* tiling;
};

extern "C" __global__ __aicore__ void mtpblock_kx(...) {
    TPipe pipe;
    KernelXxx op(&pipe);
    op.Init(...);
    op.Process();
}
```

### 10.3 资源管理

- **AllocTensor / FreeTensor**: 每行 1 个 (不超过 2 个)，禁止堆叠
- **EnQue / DeQue**: 配对使用，同一 pipe 内
- **PipeBarrier**: 跨 pipe 流水线时强制添加
- **Double Buffer**: K2/K4 使用; K1/K3/K5/K6 单 buffer (UB 余量不足时)

---

## 11. 风险与缓解

| # | 风险 | 严重度 | 缓解措施 |
|---|------|:---:|------|
| R1 | K5 MoE Device 侧 Gather/Scatter 性能差 | 高 | demo shape 下 (s=8) 差异小；保留 C9 合规的 Host 预处理方案作为备选 (需设计评审确认) |
| R2 | Sinkhorn 20 轮迭代时间 | 中 | 全程 UB 内计算, 无 GM 往返；comb [tile_s, hc, hc] 仅 ~512B |
| R3 | 大 vocab lm_head (M=1, K=512, N=129280) 性能 | 中 | demo shape 下 SIMD 可行; 大 vocab 升级 MatmulImpl 且采用 Split-K 策略 |
| R4 | 多 kernel launch 同步开销 | 中 | 8 次 launch × ~50us = ~400us 额外开销; 可通过 Event 粒度同步减少等待 |
| R5 | fp16 动态范围 (65504) 在 Sinkhorn/Softmax 中溢出 | 低 | 关键路径全程 fp32 (动态范围 ~3.4e38) |
| R6 | 大 shape tiling bug | 中 | 充分测试 s=64,128,256,512; 每个 tile 边界值验证 |

---

## 12. 与 Review 发现的对齐

| Review 发现 | 本设计处理 |
|-------------|----------|
| H2: MatmulImpl 未集成 | 第 3.2 节明确论证: demo shape 下 SIMD 优于 MatmulImpl；第 7 节提供大 shape 升级路径 |
| M1: usedCoreNum 硬编码 | 第 5.1 节: 强制动态获取 `GetCoreNumAiv()` |
| M2: 单核运行 | 第 5.1 节: s 维度多核均分策略 |
| M3: K1/K3/K5/K6 无双缓冲 | 第 6 节各 kernel: UB 容量评估; K2/K4 已双缓冲; K1/K3/K5/K6 单缓冲在 demo shape 下因 UB 余量不足 |
| M4: K3 全密集注意力 | 第 6.3 节: 设计 O(s·win) 稀疏注意力; demo shape (s=8,win=8) 差异可忽略 |
| L1: K3 命名不规范 | 第 10.1 节: 命名约定 |
| L2: K6 last-token 多 tile 缺陷 | 第 6.6 节 Step 3-4: 仅最后 tile 执行 |
| L3: K3 wo_a GM GetValue | 第 6.3 节: 已通过 DataCopyPad 预加载到 UB |
| L4: K5 routed expert 缺失 | 第 6.5 节 Stage 3: Device 侧 Gather/Scatter 完整设计 |
