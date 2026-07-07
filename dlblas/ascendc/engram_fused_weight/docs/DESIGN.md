# engram_fused_weight 算子技术设计文档 (DESIGN.md)

## 1. 算子概述

### 1.1 数学定义

```
output = wh_data.float() * we_data.float()
```

- 将两个 bfloat16 输入张量分别转为 float32 后，逐元素相乘
- 等价于: `output[i] = float32(wh_data[i]) * float32(we_data[i])`

### 1.2 输入输出规格

| 项目 | 张量名 | 数据类型 | 形状 | 元素数 | 字节数 |
|------|--------|----------|------|--------|--------|
| 输入1 | wh_data | bfloat16 | (hc_mult, hidden_size) = (4, 128) | 512 | 1,024 |
| 输入2 | we_data | bfloat16 | (hc_mult, hidden_size) = (4, 128) | 512 | 1,024 |
| 输出 | output | float32 | (hc_mult, hidden_size) = (4, 128) | 512 | 2,048 |

### 1.3 硬件环境

| 项目 | 值 |
|------|-----|
| 芯片型号 | Ascend910B2 |
| NpuArch | DAV_2201 |
| SocVersion | Ascend910B2 |
| __NPU_ARCH__ | 2201 |
| --npu-arch | dav_2201_vec |
| UB 容量 | 192 KB (196,608 bytes) |
| L0C 容量 | 128 KB |
| Cube 核数 | 24 |
| CANN 版本 | 9.0.0 |
| CPU 架构 | aarch64-linux |

> NpuArch 来源: `/npu-arch` skill。Ascend910B2 → SocVersion `Ascend910B2` → NpuArch `DAV_2201`。

---

## 2. 方案决策

### 2.1 算子类型判定

| 判定维度 | 结论 |
|----------|------|
| 输入 Shape 关系 | 两输入 Shape 完全相同 → Elementwise（非 Broadcast） |
| 跨元素依赖 | 无，逐元素独立计算 |
| 主要计算形态 | Elementwise 二元运算 + 类型转换 |

### 2.2 路线决策

按设计流程进行路线决策：

```
Step 0.5: 方案决策
├── 目标架构: DAV_2201（非 DAV_3510）
├── 算子类型: Elementwise（非 Matmul/Cube，非纯 vector 类）
└── 决策: 通用 SIMD/MemBase 路线
```

**决策理由**:
- DAV_2201 不支持 RegBase（RegBase 为 DAV_3510 专属能力）
- 算子为 Elementwise 类，非 Matmul/Cube 类（不涉及 Blaze/tensor_api）
- 走通用 SIMD/MemBase 路线，使用 `AscendC::Mul` + `AscendC::Cast` + `AscendC::DataCopy` API

### 2.3 设计方法论来源

| 来源 | 章节 | 用途 |
|------|------|------|
| `/ascendc-tiling-design` EleWise patterns | 场景路由、通用规则 | 确认 EleWise 判定、多核切分公式、UB 切分公式 |
| `/ascendc-tiling-design` EleWise tiling | 常量、公式、模板 | TilingData 结构、ubFormer 计算公式 |
| `/npu-arch` | 硬件参数 | UB 容量、核数确认 |
| `/ops-precision-standard` | 浮点社区标准 | 精度阈值确定 |
| API 头文件 (`kernel_operator_vec_vconv_intf_impl.h` 等) | Cast/Mul/DataCopy 签名 | API 参数验证 |

---

## 3. API 映射

### 3.1 API 列表

| 功能 | API | 参数签名 | 管道 | 验证状态 |
|------|-----|----------|------|----------|
| GM→UB 搬运 | `DataCopy(LocalTensor<T>&, GlobalTensor<T>&, DataCopyParams)` | dst=(VECIN), src=(GM) | MTE2 | ✅ 已验证 |
| UB→GM 搬运 | `DataCopy(GlobalTensor<T>&, LocalTensor<T>&, DataCopyParams)` | dst=(GM), src=(VECOUT) | MTE3 | ✅ 已验证 |
| BF16→FP32 转换 | `Cast<float, bfloat16_t>(LocalTensor<float>&, LocalTensor<bfloat16_t>&, RoundMode&, ...)` | dst=(FP32), src=(BF16) | Vector | ✅ 已验证 |
| FP32 逐元素乘 | `Mul<float>(LocalTensor<float>&, LocalTensor<float>&, LocalTensor<float>&, int32_t)` | dst/src0/src1 均 FP32 | Vector | ✅ 已验证 |

### 3.2 API 约束与设计影响

| API | 关键约束 | 设计影响 |
|-----|----------|----------|
| `Cast` | dst/src 为不同类型，需指定 `RoundMode` | BF16→FP32 无需舍入，使用 `RoundMode::CAST_NONE`（或等价模式） |
| `Mul` | dst/src0/src1 **必须同一类型 T** | 两输入必须先各自 Cast 为 FP32，才能调用 Mul |
| `DataCopy` | GM↔UB 要求 32 字节对齐 | 总数据量 1KB/2KB 均为 32B 倍数，天然对齐 |

### 3.3 未使用 API 说明

- `ElemwiseFrame`: 该框架仅支持单输入单输出同 dtype 算子，本算子需两输入且类型转换，不适用
- `DataCopyPad`: 数据天然 32 字节对齐，无需 Padding

---

## 4. Tiling 设计

### 4.1 多核切分

```
dim0 = hc_mult × hidden_size = 4 × 128 = 512
```

按 Elementwise tiling 公式：

```
// Step 1: 核数计算
minDtypeBits = 32 (按输出 FP32 计算，取最宽 dtype)
MIN_TILING_BITS_SIZE_PER_CORE = 32768 (4KB)
coreNum = (512 × 32 + 32768 - 1) / 32768 = (16384 + 32767) / 32768 = 1

// Step 2: blockFormer 计算
blockFormer = ((512 + 1 - 1) / 1 + 511) / 512 × 512 = 512

// Step 3: blockNum 计算
blockNum = (512 + 512 - 1) / 512 = 1
```

**结论**:
- 总输入数据仅 2KB (2 × 512 × 2 bytes)，低于 4KB 最小阈值
- 使用 **1 个 AI Core**
- blockFormer = 512（全量数据单 block 处理）
- blockNum = 1（无需跨 block 循环）

### 4.2 UB 切分

单次可处理全部 512 元素。ubFormer 按 256B 对齐计算：

```
// 按输入 BF16 数据类型 (elemBytes=2) 对齐
alignFactor = 256 / 2 = 128 元素

// bufferDivisor 计算
//   - whQue ping-pong: 2 × 2 bytes = 4
//   - weQue ping-pong: 2 × 2 bytes = 4
//   - 中间 FP32(wh): 1 × 4 bytes = 4
//   - 中间 FP32(we): 1 × 4 bytes = 4
//   - outQue ping-pong: 2 × 4 bytes = 8
//   bufferDivisor = 4 + 4 + 4 + 4 + 8 = 24

maxElemNum = (192 × 1024 × 8) / 24 = 1,572,864 / 24 = 65,536
ubFormer_256_aligned = (65536 / 128) × 128 = 65,536
```

实际上 dim0 = 512 < 65536，取 ubFormer = 512（全量处理）。

**Tiling 参数汇总**:

| 参数 | 值 | 说明 |
|------|-----|------|
| dim0 | 512 | 总元素数 |
| coreNum | 1 | 使用核数 |
| blockFormer | 512 | 每核处理元素数 |
| blockNum | 1 | 虚拟 block 数 |
| ubFormer | 512 | 单次 UB 处理元素数 |
| ubLoopOfFormerBlock | 1 | 首 block 循环次数 |
| ubTailOfFormerBlock | 512 | 首 block 尾部 (等同于全量) |
| ubLoopOfTailBlock | — | 无尾 block (blockNum=1) |
| ubTailOfTailBlock | — | 无尾 block |

### 4.3 TilingData 结构

```cpp
struct EngramFusedWeightTilingData {
    int64_t dim0;           // 总元素数 = 512
    int32_t coreNum;        // 使用核数 = 1
    int64_t blockFormer;    // 每核数据量 = 512
    int64_t blockNum;       // block 数 = 1
    int64_t ubFormer;       // UB 块大小 = 512
    int64_t ubLoop;         // UB 循环次数 = 1
    int64_t ubTail;         // UB 尾部大小 = 512
};
```

---

## 5. Buffer 规划

### 5.1 UB Buffer 分配

| Buffer | 类型 | 数量 | dtype | 单份大小 | 总大小 | 用途 |
|--------|------|------|-------|----------|--------|------|
| whQue | TQue\<VECIN\> | 2 (ping-pong) | bfloat16 | 512 × 2B = 1KB | 2KB | wh_data 输入队列 |
| weQue | TQue\<VECIN\> | 2 (ping-pong) | bfloat16 | 512 × 2B = 1KB | 2KB | we_data 输入队列 |
| tmpWH | TBuf | 1 | float32 | 512 × 4B = 2KB | 2KB | wh_data Cast 中间结果 |
| tmpWE | TBuf | 1 | float32 | 512 × 4B = 2KB | 2KB | we_data Cast 中间结果 |
| outQue | TQue\<VECOUT\> | 2 (ping-pong) | float32 | 512 × 4B = 2KB | 4KB | 输出队列 |
| **合计** | | | | | **12KB** | |

UB 总使用量 12KB，远低于 192KB 上限，有充足余量。

### 5.2 Double Buffer 说明

采用 Ping-Pong 双缓冲优化数据搬运与计算的并行性：

```
         MTE2                  Vector               MTE3
    [whQue[0] load] ──┐
    [weQue[0] load] ──┤
                       ├──> [Cast + Mul → outQue[0]] ──> [outQue[0] store]
    [whQue[1] load] ──┐
    [weQue[1] load] ──┤
                       └──> [Cast + Mul → outQue[1]] ──> [outQue[1] store]
```

由于数据量极小（仅 512 元素），实际只需 1 轮。双缓冲结构保留以遵循标准工程范式。

### 5.3 Buffer 生命周期

```
Init:
  pipe.InitBuffer(whQue, 2, 512 * sizeof(bfloat16_t))
  pipe.InitBuffer(weQue, 2, 512 * sizeof(bfloat16_t))
  pipe.InitBuffer(tmpWH, 512 * sizeof(float))
  pipe.InitBuffer(tmpWE, 512 * sizeof(float))
  pipe.InitBuffer(outQue, 2, 512 * sizeof(float))

CopyIn (progress 0):
  whBuf = whQue.AllocTensor<bfloat16_t>()
  DataCopy(whBuf, whGM[0..511], params)
  whQue.EnQue(whBuf)
  weBuf = weQue.AllocTensor<bfloat16_t>()
  DataCopy(weBuf, weGM[0..511], params)
  weQue.EnQue(weBuf)

Compute:
  whBF16 = whQue.DeQue<bfloat16_t>()
  weBF16 = weQue.DeQue<bfloat16_t>()
  Cast<float>(tmpWH, whBF16, RoundMode::CAST_NONE, 512)
  Cast<float>(tmpWE, weBF16, RoundMode::CAST_NONE, 512)
  Mul<float>(tmpOut, tmpWH, tmpWE, 512)
  outQue.EnQue<float>(tmpOut)
  whQue.FreeTensor(whBF16)
  weQue.FreeTensor(weBF16)

CopyOut:
  outFP32 = outQue.DeQue<float>()
  DataCopy(outGM[0..511], outFP32, params)
  outQue.FreeTensor(outFP32)
```

---

## 6. 数据流

### 6.1 三级流水线

```
┌──────────────────────────────────────────────────────────┐
│                      AI Core Pipeline                      │
│                                                            │
│  Stage 0 (CopyIn)   Stage 1 (Compute)   Stage 2 (CopyOut) │
│  ════════════════   ════════════════   ═════════════════  │
│  GM ──MTE2──> UB    UB ──Vector──> UB    UB ──MTE3──> GM  │
│                                                            │
│  wh_data[BF16]      Cast BF16→FP32      output[FP32]      │
│  we_data[BF16]      Cast BF16→FP32                        │
│                     Mul FP32×FP32                         │
└──────────────────────────────────────────────────────────┘
```

### 6.2 单轮数据流详细步骤

```
Step 1 [MTE2]:  DataCopy wh_bf16_ub ← wh_gm[0:512]
Step 2 [MTE2]:  DataCopy we_bf16_ub ← we_gm[0:512]
Step 3 [VEC]:   Cast wh_fp32_ub ← wh_bf16_ub    (BF16→FP32)
Step 4 [VEC]:   Cast we_fp32_ub ← we_bf16_ub    (BF16→FP32)
Step 5 [VEC]:   Mul  out_fp32_ub ← wh_fp32_ub × we_fp32_ub
Step 6 [MTE3]:  DataCopy out_gm[0:512] ← out_fp32_ub
```

### 6.3 同步机制

使用 AscendC Pipeline 同步原语（EnQue/DeQue/SetFlag/WaitFlag），由 TQue 和 TPipe 自动管理 MTE2/Vector/MTE3 之间的数据依赖。

---

## 7. 精度策略

### 7.1 精度标准

按 `/ops-precision-standard` 决策树：
- 输入 BF16（浮点），输出 FP32（浮点）→ 均为浮点
- 用户未声明商用标准 → 使用 **浮点计算类社区标准**

| 指标 | 阈值 | 数值 |
|------|------|------|
| MERE (平均相对误差) | < 2^-13 | < 0.000122 |
| MARE (最大相对误差) | < 10 × 2^-13 | < 0.00122 |

> 阈值按输出 dtype (FP32) 选取。输入为 BF16，有效精度受 BF16 尾数（7位）限制，实际误差上限由 BF16 量化误差主导。

### 7.2 数值稳定性分析

| 风险点 | 分析 | 结论 |
|--------|------|------|
| BF16→FP32 精度损失 | Cast 为无损扩展（BF16 值域是 FP32 子集） | 无精度损失 |
| FP32 乘法 | 标准 IEEE 754 FP32 乘法，单次舍入 | 精度良好 |
| 大数吃小数 | 两输入均为 BF16 来源，精度量级相近 | 风险低 |
| INF/NAN 传播 | BF16 支持 INF/NAN，FP32 乘法保持语义 | 行为一致 |

### 7.3 标杆构造

以 PyTorch CPU 实现为 Golden:
```python
golden = wh_data.float() * we_data.float()  # BF16→FP32→Mul in FP32
```

---

## 8. 边界与异常处理

### 8.1 边界情况

| 场景 | 处理策略 |
|------|----------|
| hc_mult=1, hidden_size=128 | dim0=128, 数据量更小，单核全量处理，逻辑不变 |
| hc_mult=4, hidden_size=1 | dim0=4, 极小数据，单核处理，对齐到 alignFactor 边界 |
| 任意组合导致 dim0=0 | Tiling 阶段检测 dim0==0，提前返回空输出 |
| 输入含 INF/NAN | 正常参与 FP32 乘法，结果按 IEEE 754 规则 |

### 8.2 对齐处理

- GM 数据天然 32 字节对齐（1KB / 32 = 32，2KB / 32 = 64，均为整数）
- UB 数据按 256B 对齐（Vector 指令要求）

---

## 9. Kernel 执行模型

### 9.1 算子入口

```cpp
extern "C" __global__ __aicore__ void engram_fused_weight_kernel(
    GM_ADDR whGM, GM_ADDR weGM, GM_ADDR outGM,
    EngramFusedWeightTilingData tiling);
```

### 9.2 Process 伪代码

```cpp
void Process() {
    // 初始化 Pipeline 和 Buffer
    pipe.InitBuffer(whQue, 2, tiling.ubFormer * sizeof(bfloat16_t));
    pipe.InitBuffer(weQue, 2, tiling.ubFormer * sizeof(bfloat16_t));
    pipe.InitBuffer(tmpWH, tiling.ubFormer * sizeof(float));
    pipe.InitBuffer(tmpWE, tiling.ubFormer * sizeof(float));
    pipe.InitBuffer(outQue, 2, tiling.ubFormer * sizeof(float));

    // 单轮处理（dim0 = 512，ubFormer = 512，仅1轮）
    for (int64_t i = 0; i < tiling.ubLoop; i++) {
        int64_t offset = i * tiling.ubFormer;  // i=0 → offset=0
        int32_t curLen = (i == tiling.ubLoop - 1) ? tiling.ubTail : tiling.ubFormer;

        // CopyIn: 双输入加载
        auto whBuf = whQue.AllocTensor<bfloat16_t>();
        DataCopy(whBuf, whGM[offset], {1, curLen, 0, 0});
        whQue.EnQue(whBuf);

        auto weBuf = weQue.AllocTensor<bfloat16_t>();
        DataCopy(weBuf, weGM[offset], {1, curLen, 0, 0});
        weQue.EnQue(weBuf);

        // Compute: Cast + Mul
        auto whBF16 = whQue.DeQue<bfloat16_t>();
        auto weBF16 = weQue.DeQue<bfloat16_t>();
        Cast<float>(tmpWH, whBF16, RoundMode::CAST_NONE, curLen);
        Cast<float>(tmpWE, weBF16, RoundMode::CAST_NONE, curLen);
        auto outBuf = outQue.AllocTensor<float>();
        Mul<float>(outBuf, tmpWH, tmpWE, curLen);
        outQue.EnQue<float>(outBuf);
        whQue.FreeTensor(whBF16);
        weQue.FreeTensor(weBF16);

        // CopyOut: 写回 GM
        auto outFP32 = outQue.DeQue<float>();
        DataCopy(outGM[offset], outFP32, {1, curLen, 0, 0});
        outQue.FreeTensor(outFP32);
    }
}
```

---

## 10. Tiling 代码模板

```cpp
struct EngramFusedWeightTilingData {
    int64_t dim0;
    int32_t coreNum;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoop;
    int64_t ubTail;
};

EngramFusedWeightTilingData ComputeTiling(int64_t hc_mult, int64_t hidden_size,
                                           int32_t availableCoreNum) {
    constexpr int64_t MIN_TILING_BITS = 32768;   // 4KB
    constexpr int64_t ELEM_ALIGN_FACTOR = 512;
    constexpr int64_t ALIGN_256 = 256;

    int64_t dim0 = hc_mult * hidden_size;
    int32_t minDtypeBits = 32;  // 按输出 FP32

    // 多核切分
    int32_t coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    coreNum = std::min(coreNum, availableCoreNum);

    int64_t blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1)
                          / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    int64_t blockNum = (dim0 + blockFormer - 1) / blockFormer;

    // UB 切分（按 BF16 输入 + FP32 中间/输出混合计算）
    int64_t bufferDivisor = 4 * 2 + 4 * 2 + 4 + 4 + 8;  // 24 bytes/elem
    int64_t ubSize = 192 * 1024;
    int64_t maxElemNum = ubSize * 8 / bufferDivisor;
    int64_t alignFactor = ALIGN_256 * 8 / 2;  // BF16: alignFactor=128

    int64_t ubFormer = (maxElemNum / alignFactor) * alignFactor;
    ubFormer = std::min(ubFormer, blockFormer);

    int64_t ubLoop = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubTail = blockFormer - (ubLoop - 1) * ubFormer;

    return {dim0, coreNum, blockFormer, blockNum, ubFormer, ubLoop, ubTail};
}
```

---

## 11. 精度验证方案

按 `/ops-precision-standard` 浮点计算类社区标准：

| 项目 | 配置 |
|------|------|
| 比对方法 | 单标杆比对 (PyTorch CPU FP32 Golden) |
| MERE 阈值 | 2^-13 ≈ 0.000122 |
| MARE 阈值 | 10 × 2^-13 ≈ 0.00122 |
| 验证脚本 | `scripts/mare_mere_threshold.py` |
| 测试用例 | 参见 PLAN.md §3 |
