# MTPBlock AscendC 算子开发计划

> **状态**: 开发中 (Round 2 修复完成) | **日期**: 2026-06-30 | **设计参考**: [DESIGN.md](./DESIGN.md)

---

## 0. 实现进度总览

### 0.1 代码产物

| 文件 | 状态 | 说明 |
|------|:---:|------|
| `op_kernel/mtpblock_tiling.h` | **完成** | 共享 Tiling 结构体, 6 个 kernel 共用 |
| `op_kernel/k1_embed_fuse_kernel.asc` | **完成** | Embedding gather + RMSNorm x2 + SIMD MatMul (e_proj/h_proj) + broadcast add |
| `op_kernel/k2_hc_pre_kernel.asc` | **完成** | RMSNorm + SIMD MatMul (hc_fn) + Sigmoid + Sinkhorn 20 轮 + 加权和; **Double Buffer 流水线** |
| `op_kernel/k3_attn_block_kernel.asc` | **完成** | Q/KV 投影 (SIMD MatMul) + 注意力 (AscendC::Exp) + 输出投影; **L3 修复已完成** |
| `op_kernel/k4_hc_post_kernel.asc` | **完成** | HC 后处理; **Double Buffer 流水线** |
| `op_kernel/k5_moe_block_kernel.asc` | **完成** | Shared Expert (SwiGLU); routed expert 待实现 (L4) |
| `op_kernel/k6_mtp_head_kernel.asc` | **完成** | hc_head + RMSNorm + lm_head; **L2 last-token 多 tile 修复已完成** |
| `op_host/mtpblock_host.asc` | **完成** | 6 kernel 直调验证 (K1-K6) |
| `op_extension/*` | **完成** | PyTorch 接入; **kernel 调用签名已修复** |
| `CMakeLists.txt` | **完成** | 双 Target (可执行文件 + .so) |
| `scripts/gen_data.py` | **完成** | 6 kernel 测试数据生成 |
| `scripts/golden.py` | **完成** | 6 kernel 参考实现 |
| `scripts/verify_result.py` | **完成** | fp16/fp32 精度验证 |
| `run.sh` | **完成** | 一键编译+运行+验证 (支持全部 6 kernel + PyTorch) |

### 0.2 编译与测试状态 (Round 2 修复后)

| 项目 | 状态 |
|------|:---:|
| 所有 6 个 kernel 编译 | **全部通过** (零警告) |
| mtpblock_custom 可执行文件 | **通过** |
| libmtpblock_ops.so | **通过** |
| K1 embed_fuse 精度 (MARE) | 5.09e-03 < 7.81e-02 **PASS** |
| K2 hc_pre y 精度 (MARE) | 1.12e-03 < 7.81e-02 **PASS** |
| K2 hc_pre pre/post/comb (fp32) | MARE < 2.43e-04 **PASS** |
| K3 attn_block 精度 (MARE) | 1.62e-02 < 7.81e-02 **PASS** |
| K4 hc_post 精度 (MARE) | 6.30e-04 < 7.81e-02 **PASS** |
| K5 moe_block 精度 (MARE) | 8.62e-04 < 7.81e-02 **PASS** |
| K6 mtp_head 精度 (MARE, fp32) | 1.35e-03 < 7.81e-02 **PASS** |
| PyTorch 集成测试 (hc_post) | **PASS** (sigfault 已修复) |

### 0.3 性能采集结果 (Round 001)

| Kernel | 名称 | Duration (us) | Task Type | Scalar % | Vec % |
|--------|------|:---:|------|:---:|:---:|
| K1 | embed_fuse | 167,862.87 | MIX_AIC | 98.7 | 0.00 |
| K2 | hc_pre | 6,364.57 | AI_VECTOR_CORE | 99.2 | 0.02 |
| K3 | attn_block | 72,721.15 | MIX_AIC | 99.1 | 0.00 |
| K4 | hc_post | 2,342.01 | AI_VECTOR_CORE | 99.8 | 0.00 |
| K5 | moe_block | 113,944.48 | MIX_AIC | 99.6 | 0.00 |
| K6 | mtp_head | 9,174.64 | MIX_AIC | 96.9 | 0.01 |
| **Total** | | **372,409.72** | | | |

> 数据归档: `docs/perf/round_001/` | 详细分析: `docs/perf/round_001/summary.txt`

### 0.4 设计决策符合性

| DESIGN.md 决策 | 实现符合 | 备注 |
|------|:---:|------|
| SIMD/MemBase 路径 (DAV_2201) | **是** | Ascend C SIMD API |
| 6 独立 kernel 方案 | **是** | 6 个 .asc 文件 |
| Demo shape SIMD MatMul (非 MatmulImpl) | **是** | 第 3.2 节论证: M<64 时 SIMD 优于 Cube |
| fp32 中间精度 | **是** | 所有 kernel 中间计算用 fp32 |
| bf16→fp16 类型替代 | **是** | DAV_2201 bisheng 编译器约束 |
| 32B 对齐 DataCopyPad | **是** | 所有 GM↔UB 搬运 |
| Double Buffer (K2/K4) | **是** | EnQue/DeQue 流水线 |
| AscendC::Exp 矢量 API | **是** | 无 Taylor 近似 |
| AscendC::Rsqrt 矢量 API | **是** | -- |
| UB 192KB 容量约束 | **是** | 所有 kernel UB 峰值 < 192KB |
| 动态 tile_s (kernel 侧) | **是** | 从 tiling data 运行时读取 |
| 动态 usedCoreNum | **部分完成** | PyTorch 侧已动态化; ASC host 直调仍硬编码=1 (demo shape 下单核合理) |

### 0.5 Round 2 修复记录

| 来源 | 问题 | 修复状态 |
|------|------|:---:|
| L2 | K6 last-token 多 tile 逻辑缺陷 | **已修复**: `if (tile_idx == n_tiles - 1)` 保护 Step 6-7 |
| L3 | K3 wo_a weight GM GetValue | **已修复**: DataCopyPad 预加载到 UB buffer |
| -- | PyTorch 扩展 segfault | **已修复**: kernel 调用签名添加 (blockDim, l2Ctrl, stream) |
| M1 | usedCoreNum 动态化 (PyTorch) | **已修复**: torch extension 动态获取核数 |

---

## 1. 待完成工作 (按优先级)

### P0 — 阻塞项 (影响正确性或关键功能)

| # | 项目 | 来源 | 描述 | 预计工时 |
|---|------|------|------|:---:|
| **P0-2** | K5 routed expert 实现 | L4 | 补充 Stage 3: Gate indices→Per-Expert mask 搬运→SwiGLU→scatter-add。当前仅 Shared Expert | 3 天 |

### P1 — 强烈建议 (影响性能和可扩展性)

| # | 项目 | 来源 | 描述 | 预计工时 |
|---|------|------|------|:---:|
| **P1-1** | usedCoreNum 动态化 (ASC host) | M1 | ASC host 直调侧 `usedCoreNum` 仍硬编码; PyTorch 侧已动态化 | 0.5 天 |
| **P1-2** | Multi-core s 维度切分 | M2 | 每个 kernel: blockDim = usedCoreNum; kernel 内按 blockIdx 计算 token 范围 | 2 天 |
| **P1-3** | K1/K3/K5/K6 Double Buffer 评估 | M3 | 评估各 kernel UB 余量; K3/K6 有余量 (>50%), 可优先添加; K1/K5 UB 紧张, 暂保持单缓冲 | 2 天 |
| **P1-4** | K3 稀疏注意力 O(s·win) | M4 | 接入 topk_idxs; collect 操作仅对有效 KV 位置计算 (win=8 而非 s) | 2 天 |
| **P1-5** | 端到端集成 | -- | 串联 8 次 kernel launch + workspace 管理 + 端到端精度验证 | 2 天 |
| **P1-6** | SIMD MatMul 向量化 | -- | 将标量 GetValue/SetValue 替换为 AscendC::Mul + AscendC::ReduceSum 向量点积 | 3 天 |

### P2 — 建议改进 (提升代码质量)

| # | 项目 | 来源 | 描述 | 预计工时 |
|---|------|------|------|:---:|
| **P2-1** | K3 变量命名重构 | L1 | `q→qProjBuf`, `o→outProjBuf`, `wh→wHalfBuf`, `wf→wFullBuf`, `sc→scoreBuf`, `ao→attnOutBuf`, `rv→rsqrtVec` | 0.5 天 |
| **P2-2** | AllocTensor/FreeTensor 格式化 | Review §7 | 每行 ≤ 2 个 (当前 K3/K5 堆叠), 提升可审查性 | 0.5 天 |
| **P2-3** | PipeBarrier 补充 | Review §8 | K2/K4 双缓冲路径: 确保 EnQue/DeQue 间隐式同步正确; 添加注释说明同步策略 | 0.5 天 |

### P3 — MatmulImpl 升级路径 (大 shape 扩展)

| # | 项目 | 描述 | 预计工时 |
|---|------|------|:---:|
| **P3-1** | MatmulImpl 集成框架 | Host 侧 `MatmulApiTiling::GetTiling` 集成; `TCubeTiling` 结构体正确填充 | 2 天 |
| **P3-2** | Scene Dispatch 实现 | Host 侧按 M/N 阈值自动选择 SIMD vs MatmulImpl 路径 | 1 天 |
| **P3-3** | K1/K3/K5 MatmulImpl 替换 | M≥128 时替换 SIMD MatMul 为 MatmulImpl | 3 天 |
| **P3-4** | K6 lm_head MatmulImpl | 大 vocab (129280) 的 lm_head 使用 MatmulImpl + Split-K | 2 天 |

---

## 2. 测试用例设计

### 2.1 单元测试 (per-kernel Level 0, 当前已全部覆盖)

| Kernel | 测试参数 | 验证点 | 状态 |
|--------|---------|--------|:---:|
| K1 | b=1,s=8,hc=4,d=512 | feat vs PyTorch reference | PASS |
| K2 | b=1,s=8,hc=4,d=512 | y,pre,post,comb vs PyTorch; Sinkhorn 精度 | PASS |
| K3 | b=1,s=8,n_heads=8,head_dim=64 | attn_out vs PyTorch; softmax+sink 数值等价 | PASS |
| K4 | b=1,s=8,hc=4,d=512 | hc_post 输出 vs PyTorch | PASS |
| K5 | b=1,s=8,d=512,n_experts=8 | gate scores + ffn_out vs PyTorch | PASS |
| K6 | b=1,s=8,hc=4,d=512,vocab=1000 (demo) | logits vs PyTorch | PASS |

### 2.2 集成测试 (端到端, 待完成)

| 场景 | 参数 | 验证点 |
|------|------|--------|
| 演示场景 | b=1,s=8,hc=4,d=512,vocab=129280 | 端到端 logits 精度; MARE < 7.81e-02 |
| 中等 Seq | b=1,s=64,hc=4,d=512 | Tiling 正确性 (s > tile_s) |
| 大 Seq | b=1,s=256,hc=4,d=512 | s 维度多 tile 压力测试 |
| 大 Dim | b=1,s=8,hc=4,d=1024 | d 维度变化 |
| 多 Batch | b=4,s=8,hc=4,d=512 | 多核调度正确性 |
| HC 变化 | b=1,s=8,hc=8,d=512 | hc_mult=8, mix_hc=80 |
| Window 变化 | b=1,s=64,win=32 | 大 window 稀疏注意力 |

### 2.3 边界测试

| 类型 | 用例 | 预期 |
|------|------|------|
| 极小值 | x 全为 fp16 最小正规数 (~6e-8) | 无 NaN/Inf |
| 极大值 | x 含 fp16 最大值 (65504) | Softmax 不溢出 (max subtraction) |
| 零值 | 某 token 全零 | rsqrt 不产生 Inf (eps=1e-6 保护) |
| 单 token | s=1, start_pos=0 | RoPE 频率零相位; 所有 kernel tile_s=1 |
| b>1 多 token | b=4, s=2, hc=4, d=512 | totalTokens=8; 多核切分正确 |

---

## 3. 里程碑

### Phase 1: 关键缺陷修复 (当前阶段, 预计 5 天)

```
P0-1: K6 last-token 修复 (0.5天)
P0-2: K5 routed expert 实现 (3天)
P1-1: usedCoreNum 动态化 (1天)
P2-2: 代码格式化 (0.5天)
```

**验收**: 所有 REVIEW.md 中 HIGH/MEDIUM 项修复; 精度回归测试通过。

### Phase 2: 多核与性能 (预计 5 天)

```
P1-2: Multi-core s 维度切分 (2天)
P1-3: K1/K3/K5/K6 Double Buffer 评估 (2天)
P2-3: PipeBarrier 同步补充 (0.5天)
P1-5: 端到端集成 (0.5天, 取决于 P1-2)
```

**验收**: 多核运行正确; b=4 验证通过; 端到端精度 MARE < 7.81e-02。

### Phase 3: 扩展与优化 (预计 7 天)

```
P1-4: K3 稀疏注意力 (2天)
P2-1: K3 命名重构 (0.5天)
集成测试 (大 shape): s=64,128,256 (2天)
P3: MatmulImpl 升级路径框架 (2.5天)
```

**验收**: 大 shape (s=256) 精度通过; MatmulImpl Scene Dispatch 框架就绪。

### Phase 4: 生产就绪 (预计 5 天)

```
P3: MatmulImpl 全量替换 (3天)
性能 profiling (msprof) (1天)
边界测试全覆盖 (1天)
```

**验收**: 性能基线建立; 所有集成测试 + 边界测试通过。

---

## 4. 风险登记表

| # | 风险 | 影响 | 概率 | 缓解 | 状态 |
|---|------|------|:---:|------|:---:|
| R1 | K5 routed expert Device 侧 Gather/Scatter 性能不可接受 | 整体性能不达标 | 中 | demo shape 下可接受; 大 shape 评估 Host 预处理方案 | **开放** |
| R2 | Multi-core 引入竞争条件 (race condition) | 输出错乱 | 中 | 充分测试; 每个 kernel 的 tile 范围严格隔离 | **开放** |
| R3 | MatmulImpl 集成复杂度 (TilingHeader 栈加载) | 编译/运行失败 | 低 | DAV_2201 已有成熟示例 (asc-devkit MatMul 示例) | **开放** |
| R4 | 端到端精度累积误差 | MARE 超标 | 低 | 各 kernel 单独精度已通过; 端到端误差 = sum(各 kernel 误差) | **监控** |
| R5 | 大 shape (s=256) UB 溢出 | 运行时崩溃 | 低 | tile_s 动态计算; 充分测试 | **开放** |

---

## 5. 时间估算

| Phase | 内容 | 预计工时 | 依赖 |
|-------|------|:---:|------|
| Phase 1 | 关键缺陷修复 | 5 天 | -- |
| Phase 2 | 多核与性能 | 5 天 | Phase 1 |
| Phase 3 | 扩展与优化 | 7 天 | Phase 2 |
| Phase 4 | 生产就绪 | 5 天 | Phase 3 |
| **总计** | | **22 天** | |

> 注: 不含 P3 MatmulImpl 全量替换 (额外 5 天, Phase 4)。

---

## 6. 目录结构

```
operators/MTPBlock/
├── docs/
│   ├── DESIGN.md              # 架构设计文档
│   ├── PLAN.md                # 本开发计划
│   └── REVIEW.md              # Round 1/2 审查报告
├── op_kernel/
│   ├── mtpblock_tiling.h      # 共享 Tiling 头文件
│   ├── k1_embed_fuse_kernel.asc
│   ├── k2_hc_pre_kernel.asc
│   ├── k3_attn_block_kernel.asc
│   ├── k4_hc_post_kernel.asc
│   ├── k5_moe_block_kernel.asc
│   └── k6_mtp_head_kernel.asc
├── op_host/
│   └── mtpblock_host.asc      # Host 侧直调代码
├── op_extension/
│   └── mtpblock_extension.cpp # PyTorch 接入
├── scripts/
│   ├── gen_data.py            # 测试数据生成
│   ├── golden.py              # PyTorch 参考实现
│   └── verify_result.py       # 精度验证脚本
├── CMakeLists.txt
├── run.sh
└── README.md
```
