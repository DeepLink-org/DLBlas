# DESIGN.md -- engram_fused_weight 算子技术设计文档

## 1. 算子概述

### 1.1 数学定义

```
output[i][j] = float32(wh_data[i][j]) * float32(we_data[i][j])
```

对两个 bfloat16 输入张量分别提升为 float32 后做逐元素乘法，输出 float32 张量。等价于 PyTorch 语义:

```python
return wh_data.float() * we_data.float()
```

### 1.2 输入输出规格

| 项目 | 张量名 | 数据类型 | 默认形状 | 元素数 | 单份字节数 |
|------|--------|----------|----------|--------|-----------|
| 输入1 | wh_data | bfloat16 | (hc_mult, hidden_size) = (4, 128) | 512 | 1,024 |
| 输入2 | we_data | bfloat16 | (hc_mult, hidden_size) = (4, 128) | 512 | 1,024 |
| 输出 | output | float32 | (hc_mult, hidden_size) = (4, 128) | 512 | 2,048 |

### 1.3 算子分类

| 维度 | 判定 | 依据 |
|------|------|------|
| 算子类型 | **Elementwise** | 两输入 Shape 完全相同，逐元素独立计算，无跨元素依赖 |
| 运算形态 | 二元乘法 + 类型提升 | BF16 输入 --> FP32 内部计算 --> FP32 输出 |
| 主计算单元 | Vector（SIMD） | 纯标量逐元素运算，不涉及 Cube |

---

## 2. 硬件架构信息

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | 用户指定 |
| NpuArch | DAV_2201 | `/npu-arch` skill 查表 |
| SocVersion | Ascend910B2 | `/npu-arch` skill 查表 |
| `__NPU_ARCH__` | 2201 | `/npu-arch` skill 查表 |
| `--npu-arch` 编译参数 | `dav-2201` | CANN 9.0.0 bisheng 编译器实际支持格式 |
| CANN 版本 | 9.0.0 | 用户指定 |
| CPU 架构 | aarch64-linux | CANN 9.0.0 安装路径确认 |
| UB 容量 | **192 KB** (196,608 bytes) | DAV_2201 硬件参数 |
| L1 容量 | 512 KB | DAV_2201 硬件参数 |
| L0C 容量 | 128 KB | DAV_2201 硬件参数 |
| BT 容量 | 1 KB | DAV_2201 硬件参数 |
| Vector 核数 | 48 (Cube:Vector = 1:2) | DAV_2201 硬件参数 |

> NpuArch 通过 `/npu-arch` skill 查得，禁止猜测。

---

## 3. 方案决策

### 3.1 路线判定

按设计流程 Step 0.5 逐级判定:

```
1. 算子类型判定: Elementwise 二元运算（两输入 shape 相同，逐元素独立）
2. 目标架构: DAV_2201 (Ascend910B2)，非 DAV_3510
3. 规则匹配:
   - "目标架构为 DAV_3510 且算子类型为 Matmul/Cube" → NO
   - "目标架构为 DAV_3510 且算子类型为 vector 类" → NO
   - "目标架构不是 DAV_3510" → YES → 默认 SIMD/MemBase 路线
```

**决策: 通用 SIMD/MemBase 路线**。

理由:
1. DAV_2201 不支持 RegBase（DAV_3510 专属能力），不支持 Blaze/tensor_api（DAV_3510 专属能力）
2. 算子为纯 Elementwise 乘法，充分利用 Vector 单元的 SIMD 并行能力即可
3. 通用路线有成熟的 Tiling 方法论（`ascendc-tiling-design` EleWise 族）和稳定的 Pipeline 编程模型
4. 采用 Direct Invoke 调用模式，Host 侧直接通过 `<<<>>>` 语法启动 kernel

### 3.2 调用模式选择

| 模式 | 适用性 | 选择 |
|------|--------|------|
| Direct Invoke (`<<<>>>`) | 纯 Vector 算子，无需 Cube 单元 | **选用** |
| Registry Invoke (`OP_HOST_REGISTER`) | 需要 Cube + Vector 混合调度 | 不适用 |

Direct Invoke 模式的优势:
- kernel 入口 `__global__ __vector__` 语义正确（纯 Vector，无 Cube 操作）
- 无 Cube 调度开销，launch 延迟更低
- 适用于小数据量场景（默认 dim0=512）

### 3.3 dtype 分支判定

按 `ascendc-tiling-design` Elementwise patterns.md Step 2:

```
运算为 Add/Sub / 以加减为主的累加链路 AND dtype ∈ {FP16, BF16}?
  --> NO (运算是 Mul)
  --> "乘法/除法等其他场景暂未覆盖，沿用原 dtype 直算分支"

但是: 需求源码显式声明 wh_data.float() * we_data.float()，
即 spec 强制要求内部 FP32 计算 + FP32 输出。此需求优先级高于默认路线。
```

**决策: 强制升精度 + FP32 输出**。输入以 BF16 搬运，Cast 到 FP32 后计算，输出直接以 FP32 写回。

### 3.4 设计方法论来源

| 来源 | 章节 | 用途 |
|------|------|------|
| `/ascendc-tiling-design` EleWise patterns | 场景路由、通用规则 | 确认 EleWise 判定、多核切分公式、UB 切分公式 |
| `/ascendc-tiling-design` EleWise tiling | 常量、公式、模板 | TilingData 结构、ubFormer 计算公式 |
| `/npu-arch` | 硬件参数 | UB 容量 (192KB)、核数确认 |
| `/ops-precision-standard` | 浮点社区标准 | 精度阈值: FP32 输出 → MERE < 2^-13 |

---

## 4. API 映射

### 4.1 API 列表（已通过头文件验证）

| 功能 | API | 级别/模式 | 签名 | 管道单元 |
|------|-----|----------|------|---------|
| GM-->UB 搬运 | `DataCopy` | Level 0 | `DataCopy(LocalTensor<T>&, GlobalTensor<T>&, DataCopyParams)` | MTE2 |
| UB-->GM 搬运 | `DataCopy` | Level 0 | `DataCopy(GlobalTensor<T>&, LocalTensor<T>&, DataCopyParams)` | MTE3 |
| BF16-->FP32 Cast | `Cast` | Level 2, count 模式 | `Cast<float, bfloat16_t>(LocalTensor<float>&, LocalTensor<bfloat16_t>&, RoundMode&, uint32_t count)` | Vector |
| FP32 逐元素乘法 | `Mul` | Level 2, count 模式 | `Mul<float>(LocalTensor<float>&, LocalTensor<float>&, LocalTensor<float>&, int32_t count)` | Vector |

验证方式: 所有 API 通过查阅 CANN 9.0.0 头文件 `$ASC_DEVKIT_DIR/include/ascendc/basic_api/interface/` 确认签名:

- `Cast`: `kernel_operator_vec_vconv_intf.h` 行 72-74
- `Mul`: `kernel_operator_vec_binary_intf.h` 行 154-156
- `DataCopy`: `kernel_operator_data_copy_intf.h` 行 51-52

### 4.2 API 约束与设计影响

| 约束 | 影响 | 设计应对 |
|------|------|----------|
| `Mul` dst/src0/src1 **必须同一类型** | 无法直接在 BF16 上做乘法 | 两输入先各自 Cast 到 FP32，再调用 Mul |
| `Cast` dst/src 可为不同类型 | 可做 BF16<-->FP32 转换 | BF16-->FP32 使用 `RoundMode::CAST_NONE`（无损扩展） |
| `DataCopy` GM<-->UB 要求 32 字节对齐 | 非对齐搬运不可用 DataCopy | 数据量 1KB/2KB 均为 32B 整数倍，天然对齐 |
| `DataCopyParams` 参数 | `{blockCount, blockLen, srcGap, dstGap}` | 连续搬运: `{1, nElem, 0, 0}` |
| `Cast` count 参数类型 | `uint32_t` | 需显式类型转换: `static_cast<uint32_t>(total)` |

### 4.3 类型映射

BF16 数据在 Device 侧以 `bfloat16_t` 类型运算（Ascend C 原生支持 `bfloat16_t`，区别于 `half`）。

| 概念 | C++ 类型 | 字节数 |
|------|----------|--------|
| bfloat16 输入 | `bfloat16_t` | 2 |
| float32 计算/输出 | `float` | 4 |

### 4.4 未使用 API 说明

| API | 原因 |
|-----|------|
| `ElemwiseFrame` | 该框架仅支持单输入单输出同 dtype 算子；本算子是双输入 + 类型转换 |
| `DataCopyPad` | 数据天然 32 字节对齐（512 * 2 = 1024, 512 * 4 = 2048），无需 Padding |

---

## 5. Tiling 设计

### 5.1 多核切分

按 Elementwise Tiling 公式（`ascendc-tiling-design` tiling.md §一）:

```
dim0 = hc_mult * hidden_size = 4 * 128 = 512

// Step 1: 核数计算
minDtypeBits = 32  (按输出 FP32，取最宽 dtype 计算)
MIN_TILING_BITS_SIZE_PER_CORE = 32768  (4KB，常量)
coreNum = (512 * 32 + 32768 - 1) / 32768 = 1
coreNum = min(1, availableCoreNum) = 1

// Step 2: blockFormer 对齐到 512 元素
blockFormer = ((512 + 1 - 1) / 1 + 511) / 512 * 512 = 512

// Step 3: blockNum
blockNum = (512 + 512 - 1) / 512 = 1
```

**结论**: 总输入数据仅 2KB (2 * 512 * 2 bytes)，低于 4KB 最小开核阈值。使用 **单 AI Core**，`blockFormer=512`（全量单 block 处理）。

对于更大 shape（如 hc_mult=32, hidden_size=256, dim0=8192），公式自动递增多核:
```
coreNum = (8192 * 32 + 32768 - 1) / 32768 = 9
blockFormer = ((8192 + 9 - 1) / 9 + 511) / 512 * 512 = 1024
blockNum = (8192 + 1024 - 1) / 1024 = 8
```

### 5.2 UB 切分

按 Elementwise Tiling 公式（`ascendc-tiling-design` tiling.md §二），混合精度场景需手动计算 bufferDivisor:

```
// Buffer 组成（详见 §6.1）:
//   whQue (单缓冲 BF16):    1 * 2 = 2  bytes/elem
//   weQue (单缓冲 BF16):    1 * 2 = 2  bytes/elem
//   tmpWH (FP32):           1 * 4 = 4  bytes/elem
//   tmpWE (FP32):           1 * 4 = 4  bytes/elem
//   outQue (单缓冲 FP32):   1 * 4 = 4  bytes/elem
bufferDivisor = 2 + 2 + 4 + 4 + 4 = 16  bytes/elem

ubSize = 192 * 1024 = 196608
maxElemNum = ubSize * 8 / bufferDivisor = 196608 * 8 / 16 = 98304

// 按 BF16 输入类型对齐 256B: alignFactor = 256 / 2 = 128 元素
alignFactor = 128
ubFormer = (98304 / 128) * 128 = 98304

// 实际受 DataCopy 单次搬运限制和 blockFormer 约束
ubFormer = min(98304, 2048, blockFormer) = min(2048, 512) = 512
```

**ubFormer 上限 2048 说明**: DAV_2201 上 DataCopy 单次连续搬运元素数存在实践上限（约 2048 个元素），超出后需分多次搬运。当前 dim0=512 不受此限制。

**结论**: ubFormer=512，单次处理全部 512 元素，无需分 chunk。

### 5.3 Tiling 参数汇总

| 参数 | 值 (dim0=512) | 值 (dim0=8192) | 说明 |
|------|---------------|-----------------|------|
| dim0 | 512 | 8192 | 总元素数 |
| coreNum | 1 | 9 | 使用核数 |
| blockFormer | 512 | 1024 | 每核处理元素数 |
| blockNum | 1 | 8 | 虚拟 block 数 |
| ubFormer | 512 | 2048 | UB 单次处理元素数（上限 2048） |
| ubLoop | 1 | 1 (blockFormer <= 2048) | UB 循环次数 |
| ubTail | 512 | 1024 | 尾部元素数 |

### 5.4 TilingData 结构

```cpp
struct EngramFusedWeightTilingData {
    int64_t dim0;           // 总元素数 (hc_mult * hidden_size)
    int32_t coreNum;        // 使用核数
    int64_t blockFormer;    // 每核数据量
    int64_t blockNum;       // block 数
    int64_t ubFormer;       // UB 块大小 (上限 2048)
    int64_t ubLoop;         // UB 循环次数
    int64_t ubTail;         // UB 尾部
};
```

### 5.5 Tiling 计算函数 (Host 侧)

```cpp
EngramFusedWeightTilingData ComputeTiling(
    int64_t hc_mult, int64_t hidden_size, int32_t availableCoreNum)
{
    constexpr int64_t MIN_TILING_BITS = 32768;     // 4KB
    constexpr int64_t ELEM_ALIGN_FACTOR = 512;     // 多核对齐
    constexpr int64_t ALIGN_256 = 256;             // UB 对齐字节数
    constexpr int64_t MAX_UB_FORMER = 2048;        // DataCopy 实践上限

    int64_t dim0 = hc_mult * hidden_size;
    if (dim0 == 0) {
        return {0, 0, 0, 0, 0, 0, 0};
    }

    // 多核切分
    int32_t coreNum = (dim0 * 32 + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    coreNum = std::min(coreNum, availableCoreNum);

    int64_t blockFormer = ((dim0 + coreNum - 1) / coreNum
        + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    int64_t blockNum = (dim0 + blockFormer - 1) / blockFormer;

    // UB 切分（混合精度 bufferDivisor = 16）
    int64_t bufferDivisor = 2 + 2 + 4 + 4 + 4;   // 16
    int64_t ubSize = 192 * 1024;
    int64_t maxElemNum = ubSize * 8 / bufferDivisor;
    int64_t alignFactor = ALIGN_256 * 8 / 2;       // BF16 -> 128

    int64_t ubFormer = (maxElemNum / alignFactor) * alignFactor;
    ubFormer = std::min(ubFormer, MAX_UB_FORMER);
    ubFormer = std::min(ubFormer, blockFormer);

    int64_t ubLoop = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubTail = blockFormer - (ubLoop - 1) * ubFormer;

    return {dim0, coreNum, blockFormer, blockNum, ubFormer, ubLoop, ubTail};
}
```

---

## 6. Buffer 规划

### 6.1 UB Buffer 分配

| Buffer | 类型 | 深度 | dtype | 单份大小 (dim0=512) | 总大小 | 用途 |
|--------|------|------|-------|----------------------|--------|------|
| whQue | TQue\<VECIN\> | 1 | bfloat16_t | 512 * 2B = 1KB | **1KB** | wh_data 输入 |
| weQue | TQue\<VECIN\> | 1 | bfloat16_t | 512 * 2B = 1KB | **1KB** | we_data 输入 |
| tmpWH | TBuf | 1 | float | 512 * 4B = 2KB | **2KB** | wh Cast 中间结果 |
| tmpWE | TBuf | 1 | float | 512 * 4B = 2KB | **2KB** | we Cast 中间结果 |
| outQue | TQue\<VECOUT\> | 1 | float | 512 * 4B = 2KB | **2KB** | 输出 FP32 |
| **合计** | | | | | **8KB** | |

UB 总使用量 8KB (8192 bytes)，占 UB 总容量 192KB 的 **4.17%**，余量充裕。

### 6.2 Queue 深度说明

采用 **QUE_DEPTH=1**（单缓冲）而非双缓冲，理由:
1. 小数据量下（dim0 <= 2048），单次处理即可完成，无流水线重叠需求
2. 单缓冲避免 EnQue/DeQue 同步在 multi-tile 场景下的间歇性问题
3. 代码更简洁，运行时开销更低
4. 对于该算子 I/O bound 的特性（MTE2 + MTE3 占比 > 35%），单缓冲的吞吐已满足需求

### 6.3 Pipeline 数据流

```
MTE2                     Vector                    MTE3
[whQue load] ──┐
[weQue load] ──┤
               ├──> [Cast + Mul --> outQue] ──> [outQue store]
               (顺序执行，无流水线重叠)
```

### 6.4 Buffer 生命周期伪代码

```
// Init: 注册 Buffer
pipe.InitBuffer(whQ, 1, total * sizeof(bfloat16_t));   // QUE_DEPTH=1
pipe.InitBuffer(weQ, 1, total * sizeof(bfloat16_t));
pipe.InitBuffer(tmpWH, total * sizeof(float));
pipe.InitBuffer(tmpWE, total * sizeof(float));
pipe.InitBuffer(outQ, 1, total * sizeof(float));

// CopyIn: MTE2 搬运
auto whBuf = whQ.AllocTensor<bfloat16_t>();
DataCopy(whBuf, whGM[0:total], {1, (uint16_t)total, 0, 0});
whQ.EnQue(whBuf);

auto weBuf = weQ.AllocTensor<bfloat16_t>();
DataCopy(weBuf, weGM[0:total], {1, (uint16_t)total, 0, 0});
weQ.EnQue(weBuf);

// Compute: Vector Cast + Mul
auto whBF16 = whQ.DeQue<bfloat16_t>();
auto weBF16 = weQ.DeQue<bfloat16_t>();
Cast<float>(tmpWH.Get<float>(), whBF16, RoundMode::CAST_NONE, (uint32_t)total);
Cast<float>(tmpWE.Get<float>(), weBF16, RoundMode::CAST_NONE, (uint32_t)total);
auto outBuf = outQ.AllocTensor<float>();
Mul<float>(outBuf, tmpWH.Get<float>(), tmpWE.Get<float>(), (int32_t)total);
outQ.EnQue<float>(outBuf);
whQ.FreeTensor(whBF16);
weQ.FreeTensor(weBF16);

// CopyOut: MTE3 写回
auto outFP32 = outQ.DeQue<float>();
DataCopy(outGM[0:total], outFP32, {1, (uint16_t)total, 0, 0});
outQ.FreeTensor(outFP32);
```

---

## 7. 数据流

### 7.1 三级处理阶段

```
+---------------------------------------------------------------+
|                       AI Core Pipeline                         |
|                                                                |
|   Stage 0 (CopyIn)     Stage 1 (Compute)   Stage 2 (CopyOut)  |
|   =================    =================   ==================  |
|   GM --MTE2--> UB      UB --Vector--> UB   UB --MTE3--> GM    |
|                                                                |
|   wh_data[BF16]        Cast BF16-->FP32    output[FP32]        |
|   we_data[BF16]        Cast BF16-->FP32                        |
|                        Mul FP32 * FP32                         |
+---------------------------------------------------------------+
```

### 7.2 单轮数据流详细步骤

```
Step 1 [MTE2]:  DataCopy wh_bf16_ub <-- wh_gm[0:total]
Step 2 [MTE2]:  DataCopy we_bf16_ub <-- we_gm[0:total]
Step 3 [VEC]:   Cast wh_fp32_ub <-- wh_bf16_ub    (BF16-->FP32, CAST_NONE)
Step 4 [VEC]:   Cast we_fp32_ub <-- we_bf16_ub    (BF16-->FP32, CAST_NONE)
Step 5 [VEC]:   Mul  out_fp32_ub <-- wh_fp32_ub * we_fp32_ub
Step 6 [MTE3]:  DataCopy out_gm[0:total] <-- out_fp32_ub
```

### 7.3 同步机制

使用 AscendC Pipeline 同步原语 (EnQue/DeQue/FreeTensor)，由 TQue 和 TPipe 自动管理 MTE2/Vector/MTE3 间的数据依赖。无需手动 SetFlag/WaitFlag。

### 7.4 典型运行时统计 (dim0=512)

| 指标 | 值 | 说明 |
|------|-----|------|
| Kernel 总耗时 | ~6.3 us | msprof PipeUtilization 采集 |
| AIV Vector 时间 | ~0.07 us (1.3%) | 纯计算占比极低 |
| MTE2 (copy-in) | ~1.1 us (18.6%) | BF16 输入搬运 |
| MTE3 (copy-out) | ~1.1 us (18.8%) | FP32 输出搬运 |
| AIV Scalar | ~2.1 us (36.6%) | Cast 及地址计算开销 |

> 算子为 I/O bound（小数据量下 DMA 和标量开销占主导），属于预期行为。数据量增大后 Vector 占比提升。

---

## 8. 精度策略

### 8.1 精度标准

按 `/ops-precision-standard` 决策树:

```
输入 dtype: bfloat16 (浮点)
输出 dtype: float32 (浮点)
用户未声明商用标准
--> 浮点计算类社区标准
```

| 指标 | 阈值（FP32 输出） | 数值 |
|------|-------------------|------|
| MERE (平均相对误差) | < 2^-13 | < 0.000122 |
| MARE (最大相对误差) | < 10 * 2^-13 | < 0.00122 |

> 阈值按输出 dtype (FP32) 选取。输入 BF16 有效精度受尾数 (7位) 限制。

### 8.2 数值稳定性分析

| 风险点 | 分析 | 结论 |
|--------|------|------|
| BF16-->FP32 Cast | 无损扩展（BF16 值域是 FP32 子集），二进制精确 | **0 误差** |
| FP32 乘法 | IEEE 754 标准乘法，单次舍入误差 <= 0.5 ULP | **误差可控** |
| 两输入精度量级 | 两输入均为 BF16 来源，尾数精度相近 (7-bit) | **无大数吃小数风险** |
| INF/NAN 传播 | BF16 支持 INF/NAN，FP32 乘法保持 IEEE 754 语义 | **行为一致** |

整体误差来源: BF16 输入的初始表示误差主导；Cast 和 Mul 引入的额外误差可忽略。实际验证中 MERE 和 MARE 通常接近于 0（二进制精确，因为 BF16 值恰好也是 FP32 值）。

### 8.3 标杆构造

以 PyTorch CPU FP32 计算为 Golden:

```python
# Golden = wh_data.float() * we_data.float()
# 即: BF16 --> FP32 --> 逐元素乘 --> FP32 输出
golden = wh_data.float() * we_data.float()
```

---

## 9. Kernel 执行模型

### 9.1 算子入口

```cpp
extern "C" __global__ __vector__ void engram_fused_weight_kernel(
    GM_ADDR whGM, GM_ADDR weGM, GM_ADDR outGM,
    EngramFusedWeightTilingData tiling);
```

`__global__ __vector__` 声明: 纯 Vector 算子，无 Cube 操作，避免编译器告警且语义正确。

### 9.2 Kernel 内部结构

```cpp
class KernelEngramFusedWeight {
public:
    __aicore__ inline KernelEngramFusedWeight(TPipe* p) : pipe_(p) {}

    __aicore__ inline void Init(
        GM_ADDR whGM, GM_ADDR weGM, GM_ADDR outGM,
        const EngramFusedWeightTilingData* tiling)
    {
        // 1. 获取当前 block 信息
        uint32_t blockIdx = GetBlockIdx();
        bool isLastBlock = (blockIdx == tiling->blockNum - 1);
        int64_t total = isLastBlock ? tiling->ubTail : tiling->blockFormer;
        int64_t offset = blockIdx * tiling->blockFormer;

        // 2. 设置 GlobalTensor 视图
        whG_.SetGlobalBuffer((__gm__ bfloat16_t*)whGM + offset, total);
        weG_.SetGlobalBuffer((__gm__ bfloat16_t*)weGM + offset, total);
        outG_.SetGlobalBuffer((__gm__ float*)outGM + offset, total);

        // 3. 初始化 Buffer (QUE_DEPTH=1)
        pipe_->InitBuffer(whQ_, 1, total * sizeof(bfloat16_t));
        pipe_->InitBuffer(weQ_, 1, total * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpWH_, total * sizeof(float));
        pipe_->InitBuffer(tmpWE_, total * sizeof(float));
        pipe_->InitBuffer(outQ_, 1, total * sizeof(float));

        // 4. CopyIn: 双输入
        LocalTensor<bfloat16_t> whBuf = whQ_.AllocTensor<bfloat16_t>();
        DataCopy(whBuf, whG_, {1, (uint16_t)total, 0, 0});
        whQ_.EnQue(whBuf);

        LocalTensor<bfloat16_t> weBuf = weQ_.AllocTensor<bfloat16_t>();
        DataCopy(weBuf, weG_, {1, (uint16_t)total, 0, 0});
        weQ_.EnQue(weBuf);

        // 5. Compute: Cast + Mul
        LocalTensor<bfloat16_t> whBF16 = whQ_.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> weBF16 = weQ_.DeQue<bfloat16_t>();
        Cast<float>(tmpWH_.Get<float>(), whBF16, RoundMode::CAST_NONE, (uint32_t)total);
        Cast<float>(tmpWE_.Get<float>(), weBF16, RoundMode::CAST_NONE, (uint32_t)total);
        LocalTensor<float> outBuf = outQ_.AllocTensor<float>();
        Mul<float>(outBuf, tmpWH_.Get<float>(), tmpWE_.Get<float>(), (int32_t)total);
        outQ_.EnQue<float>(outBuf);
        whQ_.FreeTensor(whBF16);
        weQ_.FreeTensor(weBF16);

        // 6. CopyOut: FP32 写回
        LocalTensor<float> outFP32 = outQ_.DeQue<float>();
        DataCopy(outG_, outFP32, {1, (uint16_t)total, 0, 0});
        outQ_.FreeTensor(outFP32);
    }

private:
    TPipe* pipe_;
    GlobalTensor<bfloat16_t> whG_, weG_;
    GlobalTensor<float> outG_;
    TQue<TPosition::VECIN, 1> whQ_, weQ_;       // QUE_DEPTH=1
    TQue<TPosition::VECOUT, 1> outQ_;
    TBuf<> tmpWH_, tmpWE_;
};
```

### 9.3 Pip-Block 结构

```
CopyIn (wh_data) ─┐
CopyIn (we_data) ─┤
                   ├─ SetFlag(V) ──► Cast(wh) ──► Cast(we) ──► Mul ──► SetFlag(MTE3) ──► CopyOut
```

---

## 10. 边界与异常处理

| 场景 | 处理策略 |
|------|----------|
| dim0 = 0 (hc_mult=0 或 hidden_size=0) | Tiling 阶段检测，返回零值 TilingData；Kernel 不启动 |
| 极小数据 (dim0 < 128) | alignFactor=128，ubFormer 取对齐后的值；单核处理 |
| 大 shape (如 hc_mult > 128) | Tiling 公式自动增加 coreNum；超出单核 UB 容量的自动分 chunk |
| ubFormer > 2048 | 受 DataCopy 实践上限约束，上限 2048 元素；更大需分多次搬运 |
| 输入含 INF/NAN | FP32 乘法按 IEEE 754 规则处理；输出保留相应特殊值 |
| 全零输入 | Cast + Mul 正常计算，输出全零 |
| 32 字节对齐 | 数据量均为 32B 整数倍，满足 DataCopy 对齐要求 |
| 单元素 (dim0=1) | 在 Tiling 和 UB 切分时 blockFormer 对齐后可能 > 实际 dim0，通过 ubTail 截断控制 |

---

## 11. 文件规划

```
operators/engram_fused_weight/
├── CMakeLists.txt                                  # 工程配置 (双 target: 可执行 + .so)
├── op_host/
│   └── engram_fused_weight.asc                     # Host 侧：Tiling 计算 + main 入口 (Direct Invoke)
├── op_kernel/
│   ├── engram_fused_weight_tiling.h                # TilingData 结构 + ComputeTiling() 函数
│   └── engram_fused_weight_kernel.asc              # Device 侧 Kernel 实现
├── op_extension/
│   ├── engram_fused_weight_torch.cpp               # PyTorch 接入层
│   ├── register.cpp                                # TORCH_LIBRARY 注册
│   └── ops.h                                       # 函数声明
├── scripts/
│   ├── gen_data.py                                 # 测试数据生成
│   ├── golden.py                                   # Golden 计算 (FP32)
│   └── verify_result.py                            # 精度验证 (MERE/MARE)
├── docs/
│   ├── DESIGN.md                                   # 本设计文档
│   └── PLAN.md                                     # 开发计划
└── run.sh                                          # 一键编译+运行脚本
```

---

## 12. 精度验证方案

| 项目 | 配置 |
|------|------|
| Golden 方法 | PyTorch CPU FP32: `wh_data.float() * we_data.float()` |
| 比对方法 | 单标杆比对 (Threshold 标准) |
| MERE 阈值 | 2^-13 = 0.000122 (FP32 输出标准) |
| MARE 阈值 | 10 * 2^-13 = 0.00122 |
| 测试数据 | 随机正态分布 (bfloat16)，覆盖正常值/全零/极值/边界值 |
| 验证工具 | `scripts/verify_result.py` 逐元素比对 |
