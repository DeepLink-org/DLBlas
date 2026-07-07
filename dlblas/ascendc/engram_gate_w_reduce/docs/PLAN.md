# engram_gate_w_reduce 开发计划

## 1. 需求概述

| 项目 | 说明 |
|------|------|
| 算子名称 | `engram_gate_w_reduce` |
| 算子类型 | Reduction (axis=0) + Broadcast Element-wise (Mul+Add) |
| 硬件平台 | Ascend 910B2 (DAV_2201) |
| CANN 版本 | 9.0.0 |
| 精度标准 | 浮点计算社区标准 (MERE < 2^-13, MARE < 10*2^-13) |
| 数据路径 | SIMD / MemBase |

### 1.1 算子语义 (PyTorch 等价)

```python
grad_w_sum = grad_w_partial.sum(dim=0)                    # [108, 4, 4096] -> [4, 4096]
grad_weight_hidden += grad_w_sum * weight_embed.float()    # [4, 4096], in-place
grad_weight_embed  += grad_w_sum * weight_hidden.float()   # [4, 4096], in-place
```

### 1.2 设计要点

- 手动迭代归约 (逐行累加), 不使用 `Pattern::Reduce::RA`
- tileElems = 4 * tileHidLen, 沿 hidden 维度切分
- 默认 tileHidLen=512, 8 cores
- bf16 权重通过 `Cast<float, bfloat16_t>` 转换为 fp32 后参与计算
- 输出保持 fp32

---

## 2. 开发阶段

### Phase 1: 算子框架搭建

**目标**: 搭建算子编译框架, 验证基本数据通路

**任务清单**:
- [ ] 创建算子工程目录结构 (`op_impl/ai_core/tbe/custom_impl/ascendc/engram_gate_w_reduce/`)
- [ ] 编写 CMakeLists.txt (kernel 编译 + host 编译)
- [ ] 编写 `engram_gate_w_reduce_tiling.h` (Tiling 数据结构)
- [ ] 编写 `engram_gate_w_reduce.h` (Host 侧入口 + Tiling 计算)
- [ ] 编写 `engram_gate_w_reduce.cpp` (Kernel 侧骨架, 空实现)
- [ ] 编译验证 (确保框架可编译通过)

**产出**:
```
engram_gate_w_reduce/
├── CMakeLists.txt
├── engram_gate_w_reduce_tiling.h
├── engram_gate_w_reduce.h
├── engram_gate_w_reduce.cpp
└── scripts/
    └── run.sh
```

---

### Phase 2: 归约核心实现

**目标**: 实现 Phase 1 归约 (grad_w_partial sum along axis=0)

**任务清单**:
- [ ] `InitBuffer`: 分配 accum, rowBuf0, rowBuf1 (TBuf)
- [ ] `Duplicate<float>`: 初始化 accum 为 0
- [ ] 循环 108 次:
  - [ ] `DataCopyPad` (GM→UB): 加载一行 tile (blockCount=4, 带 stride)
  - [ ] `Add<float>`: accum += rowBuf
- [ ] 验证: 仅输出 accum, 与 PyTorch sum(dim=0) 比对

**验证指标**:
- accum 值与 `torch.sum(grad_w_partial, dim=0)` 逐元素一致 (MERE < 2^-13)

---

### Phase 3: 逐元素后处理实现

**目标**: 实现 Phase 2/3 逐元素乘法+原位加法

**任务清单**:
- [ ] weight_embed bf16→fp32 转换:
  - [ ] `DataCopyPad` (GM→UB): 加载 weight_embed bf16 tile
  - [ ] `Cast<float, bfloat16_t>`: bf16 → fp32, CAST_NONE
- [ ] grad_weight_hidden 更新:
  - [ ] `DataCopyPad` (GM→UB): 加载 grad_weight_hidden tile
  - [ ] `Mul<float>`: weightFp32 = accum * weightFp32
  - [ ] `Add<float>`: gradBuf += weightFp32
  - [ ] `DataCopyPad` (UB→GM): 存储回 grad_weight_hidden
- [ ] weight_hidden 对称处理:
  - [ ] 同上步骤, 操作 weight_hidden 和 grad_weight_embed

**验证指标**:
- 全算子输出与 PyTorch reference 一致 (MERE < 2^-13, MARE < 10*2^-13)

---

### Phase 4: 多核并行

**目标**: 实现多核并行, 沿 hidden 维度均分 tile

**任务清单**:
- [ ] Host 侧: 计算 tileHidLen 和 numCores
  - [ ] 约束: tileHidLen * numChannels * elemSize 满足 UB 限制
  - [ ] 约束: tileHidLen 为 8 的倍数 (fp32 32B 对齐)
  - [ ] 约束: numCores ≤ 可用 core 数
- [ ] Host 侧: 为每个 core 计算 tileHidStart
- [ ] Kernel 侧: 根据 tileHidStart 计算各 tensor 的 GM 偏移

**验证指标**:
- 8 core 运行结果与单 core 一致
- 无数据竞争 (输出 tile 互不重叠)

---

### Phase 5: 边界与鲁棒性

**目标**: 处理非理想 tiling 场景

**任务清单**:
- [ ] 非整除 tileHidLen: `lastTileHidLen = hiddenSize - tileHidStart`
- [ ] 非对齐 tileHidLen: `alignedHidLen = ceil(tileHidLen*sizeof(float)/32)*32/sizeof(float)`
- [ ] R=0 校验 (Host 侧抛错)
- [ ] 单 core 场景 (totalTiles=1)

**验证指标**:
- 任意合法 hidden_size 输入均可正确执行
- 异常输入正确报错

---

### Phase 6: 性能测试

**目标**: 性能回归测试

**任务清单**:
- [ ] 编写性能测试用例
- [ ] 对比 PyTorch reference 耗时
- [ ] Profiling 分析 (Memory 带宽、计算利用率)

**预期性能**: 算子为 memory-bound (108 次迭代 × 8KB/iter), 主要受限于 DDR 带宽。

---

## 3. 测试用例

### 3.1 基础功能测试

| Case | hidden_size | R (rows) | channels | tileHidLen | numCores | 验证 |
|------|-------------|----------|----------|------------|----------|------|
| TC01 | 4096 | 108 | 4 | 512 | 8 | 精度全通过 |
| TC02 | 2048 | 108 | 4 | 512 | 4 | 精度全通过 |
| TC03 | 4096 | 1 | 4 | 512 | 8 | 精度全通过 (R=1 边界) |
| TC04 | 4096 | 108 | 4 | 256 | 16 | 精度全通过 (多核) |

### 3.2 精度边界测试

| Case | 描述 | 验证重点 |
|------|------|---------|
| TC05 | 权重全为 1.0 (bf16) | mul+add 结果正确性 |
| TC06 | 权重全为 0.0 (bf16) | 梯度值不变 |
| TC07 | grad_w_partial 含大值 (1e6) | 归约精度, 无溢出 |
| TC08 | grad_w_partial 含负值 | 有符号归约正确性 |
| TC09 | grad_w_partial 含小值 (1e-6) | 大数吃小数 (bf16 限制) |

### 3.3 异常输入测试

| Case | 描述 | 期望行为 |
|------|------|---------|
| TC10 | R=0 | Host 侧报错 |
| TC11 | hidden_size=0 | Host 侧报错 |
| TC12 | hidden_size=1 | 正确执行 (小 tensor) |
| TC13 | hidden_size=100 (非 2^n) | 正确执行 (非对齐 tile) |

### 3.4 回归测试

| Case | 描述 | 期望行为 |
|------|------|---------|
| TC14 | 与 PyTorch CPU 结果比对 | MERE < 2^-13, MARE < 10*2^-13 |
| TC15 | 与 PyTorch NPU 结果比对 (如有离线实现) | 精度一致 |

---

## 4. 阶段检查清单

### 4.1 代码规范

- [ ] 统一使用 `DataCopyExtParams` + `DataCopyPadExtParams` (避免 range 溢出)
- [ ] GM→UB / UB→GM 均使用 `DataCopyPad` (不混用 `DataCopy`)
- [ ] 不直接使用 `GlobalTensor::GetValue/SetValue` (仅调试允许)
- [ ] `Cast` RoundMode: bf16→fp32 用 `CAST_NONE`
- [ ] UB 地址偏移使用字节对齐值 (32B 对齐)
- [ ] buffer size 使用元素对齐后的值

### 4.2 架构兼容性

- [ ] `__NPU_ARCH__` 条件编译宏 = `2201`
- [ ] 不依赖 DAV_3510 专属能力 (RegBase / BufferID / NDDMA / CCU)
- [ ] 不依赖 `tensor_api` / Blaze 路径
- [ ] Host 侧使用 `PlatformAscendC` 获取架构参数, 不硬编码

### 4.3 设计原则

- [ ] Host 侧不做张量预处理 (如转置)
- [ ] 多核并行无锁 (各 core 输出区域独立)
- [ ] Tiling 参数从 Host 传递, Kernel 侧只读取不计算

---

## 5. 关键 API 确认列表

| API | 头文件/来源 | 验证方式 |
|-----|-----------|---------|
| `DataCopyPad` | `kernel_operator_data_copy_intf.h` | ARA-FullLoad 模式文档 + 内置算子参考 |
| `Duplicate<float>` | `kernel_operator_intf.h` | api-buffer.md 标准用法 |
| `Add<float>(dst, src0, src1, count)` | `kernel_operator_vec_binary_intf.h` | 已确认支持 in-place (api-arithmetic.md) |
| `Mul<float>(dst, src0, src1, count)` | `kernel_operator_vec_binary_intf.h` | 已确认支持 in-place (api-arithmetic.md) |
| `Cast<float, bfloat16_t>` | `kernel_operator.h` (template) | api-precision.md 规范: CAST_NONE |

---

## 6. 预期时间线

| 阶段 | 预估工作量 | 依赖 |
|------|-----------|------|
| Phase 1: 框架搭建 | 0.5 天 | 无 |
| Phase 2: 归约核心 | 1 天 | Phase 1 |
| Phase 3: 逐元素后处理 | 0.5 天 | Phase 2 |
| Phase 4: 多核并行 | 0.5 天 | Phase 3 |
| Phase 5: 边界鲁棒性 | 0.5 天 | Phase 4 |
| Phase 6: 性能测试 | 0.5 天 | Phase 5 |
| **总计** | **3.5 天** | |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| bf16→fp32 Cast API 签名不匹配 | Phase 3 阻塞 | 参考内置算子 `post_layer_norm` 等使用 `CastFrom32To16` / `Cast16AndCopyOut` 封装 |
| DataCopyPad stride 计算错误导致数据错位 | 精度不通过 | 先用最小 tile (tileHidLen=64) 验证 stride 公式, 逐步放大 |
| 108 次迭代 latency 过高 | 性能不达预期 | 考虑加载多行合并 (每 2-3 行做一次 DMA+多次 Add) |
| 多 core 负载不均 | 部分 core 闲置 | tileHidLen 调小以增加 tile 总数和 core 利用率 |
