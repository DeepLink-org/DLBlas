
---

## Round 2 审查报告（Step 5 复审）

- **审查日期**：2026-07-02
- **审查人**：算子审查者代理 (Reviewer Agent)
- **判定**：**PASS WITH NOTES**
- **总分**：**78 / 100**

---

### 审查背景

本轮（Round 2）审查的对象是 Developer 在 Round 1 判定 FAIL 后重新提交的代码。当前 `mhc_post_kernel.asc` 已是完整实现版本，与 Round 1 的骨架代码**完全不同**。

主要变化：
- `mhc_post_kernel.asc` 实现了完整计算逻辑（Cast bf16→fp32、K=4 向量点积 Muls+Add、Broadcast MulAdd、Cast fp32→bf16）
- 使用 `TQue<VECIN/VECOUT, 2>` + `TBuf<VECCALC>` 实现双缓冲流水线
- 系数改用 `TBuf` + `PipeBarrier<PIPE_MTE2>` 替代前一版本的 TQue 系数方案
- PyTorch 扩展 (`op_extension/`) 已完整实现并接入 CMakeLists.txt

---

### 1. 独立编译验证（维度 1：10/10）

**独立构建目录**：`build_review_round2/`（新建，与 Developer 的 `build/` 完全隔离）

**编译命令**：
```bash
cd /mnt/data01/zmz/workspace/12agent/waic/build/mhc_post/operators/mhc_post
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p build_review_round2 && cd build_review_round2
cmake .. && make -j4
```

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 1.1 独立编译成功 | `cmake && make -j4` 成功，产出 `mhc_post` 可执行文件 + `libmhc_post_ops.so` | 7/7 |
| 1.2 无代码级警告 | bisheng 编译输出零警告，`make` 输出无任何 warning/error 行 | 3/3 |

**CMake 配置验证**：
- `find_package(ASC REQUIRED)` - 通过
- `LANGUAGES ASC CXX` - 通过
- `--npu-arch=dav-2201` - 通过（匹配 Ascend910B2 / DAV_2201）
- `target_link_libraries` 含 `tiling_api` - 通过
- 双 Target（`mhc_post` 可执行 + `mhc_post_ops` 共享库）- 通过

**结论**：编译全程成功，零警告。

---

### 2. 架构合规性（维度 2：15/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 2.1 TPipe/TQue 模式 | `TPipe pipe` 构造传递给 `KernelMhcPost`；`TQue<VECIN,2>` × 5（inQueRes_[4] + inQueX_）；`TQue<VECOUT,2>` × 4（outQue_[4]）；`TBuf<VECCALC>` × 10（cmbBuf_, pmBuf_, resFp32_[4], term2_[4], xFp32_, tmpFp32_）。模式正确 | 3/3 |
| 2.2 入口属性正确 | `extern "C" __global__ __vector__ void mhc_post_kernel(...)` 声明正确，与 `mhc_post_kernel_decl.h` 一致 | 3/3 |
| 2.3 定义顺序正确 | KernelMhcPost 构造函数 → Init() → Process() → private: TileSize/LoadCoefficients/CopyInData/ComputeTile/CopyOutTile/PipelineBatch → 成员变量。顺序清晰合规 | 3/3 |
| 2.4 内存管理配对 | AllocTensor/FreeTensor 全部配对：inQueRes_[m].AllocTensor↔FreeTensor × 4；inQueX_.AllocTensor↔FreeTensor × 1；outQue_[m].AllocTensor→EnQue 后 DeQue↔FreeTensor × 4。EnQue/DeQue 配对：所有 TQue 数量一一对应 | 3/3 |
| 2.5 数据流完整 | CopyIn（GM→UB，5个TQue）→ ComputeTile（DeQue→Cast→Muls→Add→Cast→EnQue）→ CopyOut（DeQue→GM）三阶段完整。双缓冲流水 PipelineBatch 正确实现（预加载→CopyIn(N+1)∥Compute(N)∥CopyOut(N-1)→CopyOut(last)） | 3/3 |

**硬件参数 Grep 检查（阻塞项全通过）**：
```bash
grep -n "blockDim\s*=\s*[0-9]" op_kernel/mhc_post_kernel.asc  → 无匹配 (通过)
grep -n "blockIdx\s*=\s*[0-9]" op_kernel/mhc_post_kernel.asc  → 无匹配 (通过)
grep -rn -E "(192000|196608|192\s*\*\s*1024)" op_kernel/ op_host/  → 无匹配 (通过)
```

---

### 3. 编码规范（维度 3：13/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 3.1 矢量 API | Cast（bf16↔fp32, CAST_NONE/CAST_ROUND）、Muls（float scalar×vector）、Add（in-place）全部正确使用。共 28 条向量指令/tile（16 Muls + 12 Add）+ 8 条（Broadcast MulAdd）+ 4 Cast | 4/4 |
| 3.2 API 约束满足 | `LocalTensor::GetValue()` 用于 TBuf 系数标量读取（正确，不在黑名单）；无 `GlobalTensor::SetValue/GetValue`；无裸 `DataCopy`（全部使用 `DataCopyPad`）。API 黑名单合规 | 4/4 |
| 3.3 数据对齐 | C_TILE=64，64×sizeof(bfloat16_t)=128B=4×32B，自然对齐。DataCopyPad 参数 `{1, bytes, 0, 0}` 格式正确，适用于非整除 h 尾块场景 | 3/3 |
| 3.4 命名规范 | 变量命名清晰（`n0_/n1_/cTile_/h_/cTiles_` 成员变量带下划线后缀，`inQueRes_/outQue_/cmbBuf_/pmBuf_` 语义明确）。但 Host 侧存在大量调试 printf（`KernelCall: start`、`alloc input N, size=M` 等），在正式提交中属于冗余日志 | 2/3 |

**3.4 说明**：Host 侧共 9 条 `printf` 调试日志，其中 `KernelCall: alloc input/output` 逐个打印每个 tensor 的分配信息，在直调验证场景可接受，但 production 代码需改为条件日志或移除。扣 1 分。

---

### 4. 性能分析（维度 4：12/20）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 4.1 动态硬件参数 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 运行时获取核数；`blockNum = min(availableCoreNum, MAX_CORE_NUM, n1)` 正确三重限制；cTile/h 均通过 tiling 结构传递，无硬编码 | 4/4 |
| 4.2 多核并行 | 沿 n1 维度均匀切分，`n1PerCore = ceil(n1/blockNum)`，尾核自动缩小范围（`n1End = min(n1Start + n1PerCore, n1)`），空闲核正确跳过（`n1Start >= n1_` 判断）。TC-06/TC-07 均验证通过 | 4/4 |
| 4.3 流水线/双缓冲 | `TQue<VECIN,2>` + `TQue<VECOUT,2>` 实现双缓冲。PipelineBatch 正确实现三阶段流水（预加载 tile 0 → 主循环中 CopyIn(N+1)∥Compute(N)∥CopyOut(N-1) → 收尾 CopyOut(last)）。设计正确 | 3/4 |
| 4.4 同步策略 | 逐项依赖分析见下方，1 个 PipeBarrier 全部必要且类型正确，无冗余 barrier | 2/4 |
| 4.5 计算效率与上板性能 | 向量计算无循环内逐行 API 调用问题（K=4 展开每 tile 调用 Muls/Add 批量操作）。但上板性能 **8,691 us 远差于理论 ~150 us（HBM bound）**，距理论 58 倍差距。主因：GetValue() 导致 scalar=99% 瓶颈（见下方分析） | 0/4 |

#### 4.4 同步策略逐项依赖分析

代码中仅有 1 个 PipeBarrier：`PipeBarrier<PIPE_MTE2>()` 在 `LoadCoefficients()` 末尾。

**依赖链分析**：

| PipeBarrier 位置 | 上游操作 | 下游依赖 | 分析结论 |
|-----------------|---------|---------|---------|
| `LoadCoefficients()` 末尾，`PipeBarrier<PIPE_MTE2>()` | 两次 `DataCopyPad(cmbBuf_, ...)` + `DataCopyPad(pmBuf_, ...)` 写 TBuf（MTE2 路径） | `ComputeTile()` 中 `cmbLocal.GetValue(...)` + `pmLocal.GetValue(...)` 直接读取 TBuf 标量 | **必要且正确**。TBuf 不经过 TQue，无 DeQue 同步机制，DataCopyPad 是异步 MTE2，若无 barrier 则 GetValue 可能读到未完成加载的数据。PIPE_MTE2 精确覆盖所需同步范围，优于 PIPE_ALL |

其余 TQue 操作（inQueRes_/inQueX_/outQue_）通过 DeQue/FreeTensor 内建同步，无需额外 barrier。

**结论**：1 个 PipeBarrier 全部必要，无冗余 PipeBarrier。但正是因为 TBuf 方案下每 batch 共享系数绕过了 TQue 机制，导致系数读取退化为 GetValue() 标量读，成为 scalar 瓶颈。

**4.4 得分 2/4 说明**：同步正确无冗余（2分），但在 scalar=99% 严重瓶颈背景下，未进行系数预读优化（TBuf 中的数据可在 batch 开始时用局部变量一次性缓存到寄存器，消除 tile 内重复 GetValue），属于设计层面的遗漏优化点，扣 2 分。

#### 4.5 上板性能详细分析

来自 `docs/perf/round_002/summary.txt`（v2 双缓冲，20 核，TC-01 shape）：

| 指标 | 值 |
|------|-----|
| Task Duration | 8,691 us |
| AIV vec | 46.70%（4,055 us） |
| AIV scalar | **99.00%（8,589 us）** |
| AIV MTE2 (Load) | 28.60%（2,478 us） |
| AIV MTE3 (Store) | 18.80%（1,631 us） |
| HBM read | 0.04 GB/s |
| HBM write | 0.03 GB/s |
| 理论上限（HBM bound） | ~150 us |
| 实测/理论 | ~58x |

**瓶颈分析**：scalar=99% 明确为 SCALAR BOUND。根因：每 tile 20 次 `GetValue()` 标量读（16 个 comb_res_mix + 4 个 post_layer_mix），每批次 20 tiles × 20 次 = 400 次/批次，每核 410 批次 × 400 = 164,000 次/核。这些标量操作占据了 scalar 流水线 99% 时间，远超预期。

双缓冲设计本身正确（流水重叠逻辑实现无误），但因 scalar 瓶颈掩盖了流水收益，v2 (8,691 us) 甚至略慢于 v1 (7,767 us)。

**优化路径**（已在 PLAN.md §8.2 中正确识别，但未在代码中实现）**：
```cpp
// 在 LoadCoefficients() 末尾 PipeBarrier 之后，读入寄存器变量：
float cmbCached[MHC_MULT][MHC_MULT];
float pmCached[MHC_MULT];
LocalTensor<float> cmbLocal = cmbBuf_.Get<float>();
LocalTensor<float> pmLocal = pmBuf_.Get<float>();
for (uint32_t m = 0; m < MHC_MULT; m++) {
    for (uint32_t k = 0; k < MHC_MULT; k++)
        cmbCached[m][k] = cmbLocal.GetValue(m * MHC_MULT + k);
    pmCached[m] = pmLocal.GetValue(m);
}
// ComputeTile 改用 cmbCached/pmCached，消除 164,000 次 GetValue/核
```
这一优化预期可将 scalar 时间从 8,589 us 降低到可忽略水平，使性能向 HBM-bound（150 us）靠近。

---

### 5. 测试覆盖（维度 5：14/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 5.1 测试数据生成 | `gen_data.py` 支持 TC-01~TC-12（12 个用例），`golden.py` 精确模拟 bf16 CAST_NONE/CAST_ROUND 路径，`write_bf16_bin`/`bf16_to_fp32` 工具函数完备 | 4/4 |
| 5.2 结果验证脚本 | `verify_result.py` 实现 MERE/MARE 双阈值标准（2^-7 和 10×2^-7），排除近零元素（`atol_gate=1e-5`），提供 Top-10 worst error 诊断输出。功能完整 | 4/4 |
| 5.3 Level 0-3 覆盖 | Level 0（基础功能）：TC-02 (1,1,64,4), TC-03 (1,1,1,4)；Level 1（典型）：TC-01, TC-11；Level 2（边界）：TC-04~TC-07, TC-09~TC-10；Level 3（大数据量）：TC-01 (2,4096,1280,4) 也可归 Level 3。覆盖完善 | 3/3 |
| 5.4 精度标准明确 | DESIGN.md §9.2、PLAN.md §2.4、verify_result.py 三处一致声明 bf16 MERE < 2^-7、MARE < 10×2^-7。但精度标准仅针对 bf16 单一 dtype，未扩展 fp16/fp32 | 3/4 |

**独立验证结果（12/12 全通过）**：
```
TC-01 (2,4096,1280,4): MERE=1.36e-7 < 7.81e-3  MARE=7.81e-3 < 7.81e-2  → PASSED
TC-02 (1,1,64,4):      MERE=0.0     MARE=0.0                             → PASSED
TC-03 (1,1,1,4):       MERE=0.0     MARE=0.0                             → PASSED
TC-04 (1,16,1280,4):   PASSED
TC-05 (2,4096,64,4):   PASSED
TC-06 (2,4097,1280,4): PASSED  ← n1 尾核逻辑验证
TC-07 (1,1,1280,4):    PASSED  ← 空闲核跳过验证
TC-08 全零输入:         PASSED
TC-09 极大值输入:       PASSED
TC-10 混合正负值:       PASSED
TC-11 (1,4096,1280,4): PASSED
TC-12 (2,4096,130,4):  PASSED  ← h 非整除尾块验证
```

---

### 6. 精度验证（维度 6：10/10）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 6.1 FP32 全用例 PASS | N/A（算子设计输出 bf16，无 fp32 输出路径） | - |
| 6.2 FP16 全用例 PASS | N/A（算子使用 bfloat16_t，非 half） | - |
| 6.3 BF16 全用例 PASS | **全部通过**：12/12 测试用例精度达标。TC-01: MERE=1.36e-7（远低于 7.81e-3 阈值），MARE=7.81e-3（低于 7.81e-2 阈值） | 10/10 |

**精度评定**：完全达标。算子通过 fp32 中间计算（Cast_NONE + Muls/Add + Cast_ROUND）有效控制了 bf16 精度误差。13 个接近 1-ULP 边界的元素（DESIGN.md §9.5 中记录）属正常 bf16 舍入差异。

---

### 7. 文档审查（维度 7：14/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 7.1 README.md 存在 | 存在，内容完整覆盖算子功能、规格、构建运行指南、测试结果、性能数据、文件结构、PyTorch 接入示例 | 3/3 |
| 7.2 数学公式 | README 含 `term2 = Σ_k cmb[m,k] × res[k,:]` + `out = x × pm[m] + term2[m]` 及 PyTorch einsum 等效表达，DESIGN.md §1.1/§1.2 含完整矩阵展开 | 3/3 |
| 7.3 编译运行指南 | README 提供 `bash run.sh`（标准）、`--test-all`（全量）、`--test=TC-XX`（单例）、`--skip-build`（跳过编译）四种模式；CANN 环境设置说明完整 | 3/3 |
| 7.4 API 映射/约束 | DESIGN.md §8.1/§8.3 列出完整 API 映射表（DataCopyPad/Cast/Muls/Add/TQue/TBuf）和黑名单合规说明；README §关键设计要点 6 条匹配代码实现 | 3/3 |
| 7.5 已知限制 | README 性能数据节已说明 scalar=99% 瓶颈及根因（GetValue 系数读取），PLAN.md §8.2 有详细分析和优化建议。但 README 未列出"h 必须至少为 1"、"MHC_MULT 固定为 4 的限制（无法运行时修改）"等已知约束 | 2/3 |

**7.5 说明**：扣 1 分原因：当前 `MHC_MULT=4` 为编译期常量（`constexpr`），在 tiling 结构中不可修改；但 README/DESIGN.md 未明确声明此为已知限制（即不支持 M≠4 的配置）。若用户尝试 M=8 会静默错误。

---

### 8. 问题列表

#### MEDIUM 级问题（建议修复）

| # | 类别 | 描述 | 修复建议 |
|---|------|------|---------|
| **M1** | **性能** | **SCALAR BOUND 未解决**：GetValue() 调用导致 scalar=99%，上板延迟 8,691 us 较理论 ~150 us 差 58 倍。开发者已正确识别根因（PLAN.md §8.2）但代码中未实施优化 | 在 `LoadCoefficients()` 中 `PipeBarrier<PIPE_MTE2>()` 后立即将 20 个系数读入 `float cmbCached[4][4]` 和 `float pmCached[4]` 寄存器数组（成员变量），`ComputeTile()` 改用缓存变量，消除 tile 内重复 GetValue。预期可将 scalar bound 降低至可忽略水平 |

#### LOW 级问题（可接受，建议修复）

| # | 类别 | 描述 | 修复建议 |
|---|------|------|---------|
| **L1** | **清洁** | **op_kernel/ 存在遗留文件**：`mhc_post_kernel.asc.bak`（旧骨架版本）、`mhc_post_kernel.asc.bak_v2`（另一备份）、`mhc_post_kernel_test.asc`（测试版本，含 TQue 系数方案）均不在 CMakeLists.txt 编译流程中，但留在代码库中会引起混乱 | 删除 `.bak`/`.bak_v2` 文件；若 `_test.asc` 有参考价值可保留在 `docs/` 下并重命名，或直接删除 |
| **L2** | **代码** | **Host 侧 aclrtMalloc/aclrtMallocHost/aclrtMemcpy 缺少返回值检查**：`KernelCall()` 中 6+ 处内存操作未检查返回值。若 OOM 或 device 异常会静默失败 | 参照现有 `aclrtSetDevice`/`aclrtGetDeviceInfo` 的 `if (ret != ACL_SUCCESS)` 模式，对每个 `aclrtMalloc*`/`aclrtMemcpy` 调用添加错误检查 |
| **L3** | **文档** | **README 未列出 MHC_MULT=4 编译期固定的约束**：用户可能误以为可通过参数修改 M 值 | 在"已知限制"或"规格"节添加："`M (MHC_MULT) = 4` 为编译期常量，不支持运行时修改" |
| **L4** | **代码** | **Host 侧调试 printf 过于详细**：`KernelCall: alloc input 0, size=20971520` 等 9 条 printf 在正式代码中属冗余日志 | 将调试 printf 改为条件编译（`#ifdef DEBUG_LOG`）或降级为仅保留关键信息（启动/完成/错误） |

---

### 9. 硬件参数检查

| 检查项 | 命令 | 结果 |
|--------|------|------|
| `blockDim = 数字` | `grep -n "blockDim\s*=\s*[0-9]" op_kernel/mhc_post_kernel.asc` | **通过** - 无硬编码 |
| `blockIdx = 数字` | `grep -n "blockIdx\s*=\s*[0-9]" op_kernel/mhc_post_kernel.asc` | **通过** - 无硬编码 |
| 硬编码 UB 大小 | `grep -rn -E "(192000|196608|192\s*\*\s*1024)" op_kernel/ op_host/` | **通过** - 无硬编码 |
| 核数动态获取 | `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)` | **通过** |
| API 黑名单 | `grep -n "GlobalTensor.*SetValue\|GlobalTensor.*GetValue"` | **通过** - 无违规 |
| DataCopy 使用 | `grep -n "DataCopy(" ... \| grep -v DataCopyPad` | **通过** - 全部使用 DataCopyPad |

---

### 10. 设计合规检查

对照 DESIGN.md 逐项验证：

| 设计项 | 状态 | 说明 |
|--------|------|------|
| §3 技术路线：SIMD/MemBase | **符合** | TPipe/TQue/TBuf + Vector API，无 RegBase/Blaze/Cube API |
| §4 多核切分：n1 维度，max 20 核 | **符合** | `min(availableCoreNum, MAX_CORE_NUM, n1)` 三重限制 |
| §5 UB 切分：C_TILE=64 | **符合** | C_TILE=64 通过 tiling 传递，64×2B=128B=4×32B 对齐 |
| §5 B_TILE=1 逐 batch | **符合** | 外循环 n0→n1→cTiles，系数每 batch 加载一次 |
| §6 Buffer 规划 | **基本符合** | TBuf 方案（cmbBuf_+pmBuf_）替代了 DESIGN.md §6.3 建议的单一 coeffBuf_，但功能等价 |
| §7 计算流程 | **符合** | Cast NONE→Muls→Add（K=4）→Muls→Add（Broadcast）→Cast ROUND。28+8=36 条向量指令/tile |
| §8 API 映射 | **符合** | DataCopyPad/Cast/Muls/Add/EnQue/DeQue/InitBuffer 全部按规格使用 |
| §9 精度策略 | **符合** | bf16→fp32 CAST_NONE、fp32→bf16 CAST_ROUND、全部 12 用例达标 |
| §10 分支场景 | **符合** | 空闲核/尾核/尾块 TileSize() 全部正确，12 用例验证覆盖 |
| §12 Tiling 数据结构 | **符合** | MhcPostTiling 24B，Host→Device 一次 aclrtMemcpy |
| §13 架构合规 | **符合** | 所有检查项通过 |

**NpuArch 验证**：
- 芯片型号：Ascend910B2 → NpuArch = DAV_2201（通过 `/npu-arch` skill 确认）
- CMakeLists.txt `--npu-arch=dav-2201` 与目标芯片一致

---

### 11. RegBase 路线检查

DESIGN.md §3.1 明确排除 RegBase（DAV_2201 不支持）。代码中无 `RegTensor`/`MaskReg`/`asc_vf_call`/`__simd_vf__` 等 RegBase 标记。检查通过。

---

### 12. 总结

**本轮审查关键结论**：

Round 1 的 3 个 HIGH 问题（H1 骨架代码、H2 类型不匹配、H3 SetValue 反模式）**全部修复**。当前实现质量显著提升。

**优点**：
1. 完整计算逻辑：Cast→K=4向量点积→Broadcast MulAdd→Cast 全部正确实现
2. 双缓冲流水线：TQue<VECIN/VECOUT,2> + PipelineBatch 三阶段重叠实现正确
3. 精度全部达标：12/12 测试用例 bf16 MERE/MARE 双阈值通过
4. API 黑名单合规：LocalTensor::GetValue()（TBuf 标量读）正确；无 GlobalTensor::SetValue/GetValue；全部 DataCopyPad
5. 架构合规：动态核数、n1 维度切分、空闲核跳过、尾块处理全部正确
6. 同步策略：唯一 PipeBarrier<PIPE_MTE2> 必要且精确，无冗余 barrier
7. PyTorch 扩展完整：mhc_post_torch.cpp + register.cpp + ops.h 全部实现，libmhc_post_ops.so 编译成功

**主要问题（不阻塞通过，建议修复）**：
1. **M1（性能）**：GetValue() 致 scalar=99% 瓶颈，8,691 us 距理论 150 us 差 58 倍。优化路径已知（PLAN.md §8.2），建议在下一轮实施寄存器变量缓存
2. **L1（清洁）**：`op_kernel/` 中存在 `.bak`/`_test.asc` 遗留文件，建议清理
3. **L2（代码）**：Host 侧 aclrtMalloc/Memcpy 缺少错误检查
4. **L4（代码）**：Host 侧 9 条调试 printf 冗余

**评分汇总**：

| 维度 | 得分 | 满分 | 关键扣分 |
|------|------|------|---------|
| 1. 编译验证 | 10 | 10 | - |
| 2. 架构合规 | 15 | 15 | - |
| 3. 编码规范 | 13 | 15 | 3.4 Host 调试 printf |
| 4. 性能分析 | 12 | 20 | 4.4 GetValue 未优化（-2），4.5 上板性能差 58x（-4） |
| 5. 测试覆盖 | 14 | 15 | 5.4 单 dtype 精度标准 |
| 6. 精度验证 | 10 | 10 | - |
| 7. 文档审查 | 14 | 15 | 7.5 MHC_MULT 限制未声明 |
| **总计** | **78** | **100** | |

**判定：PASS WITH NOTES**（总分 78 ≥ 70，无 HIGH/必须修复问题）

**主要修复建议（优先级排序）**：
1. M1：在 `LoadCoefficients()` 中实施系数寄存器变量缓存（预期性能提升 10-50x）
2. L1：清理 `op_kernel/` 中的 `.bak`/`_test.asc` 遗留文件
3. L2：为 Host 侧内存操作添加返回值检查

---

## Round 1 审查报告（Step 5 复审 - 独立审查）

- **审查日期**：2026-07-02
- **审查人**：算子审查者代理 (Reviewer Agent)
- **判定**：**FAIL**
- **总分**：**37 / 100**

---

### 审查概要

**核心发现：当前 `mhc_post_kernel.asc` 是一个非功能性骨架代码。** Kernel 仅使用 `LocalTensor::SetValue()` 逐元素写入零值到输出，未实现 DESIGN.md 中描述的任何计算逻辑（Cast/Muls/Add/K=4 向量点积/Broadcast MulAdd/双缓冲流水线）。

独立精度验证确认：输出 **100% 为零**，MERE=1.0, MARE=1.0（完全错误）。这是代码层面的严重缺陷，判定为 **FAIL**。

代码库中存在一个 `mhc_post_kernel_test.asc`（含实际计算逻辑），但：(a) 未接入 CMakeLists.txt 编译流程；(b) 使用 `half` (fp16) 而非 DESIGN.md 和 tiling 头文件声明的 `bfloat16_t`；(c) 仅有单缓冲（BUFFER_NUM=1）、无流水线重叠；(d) 独立替换后运行挂起（疑似类型不匹配或同步 bug）。

注：此前 2026-07-01 的 Round 0 REVIEW.md 所审查的代码版本（含完整 Muls/Add/Cast 计算逻辑）在当前代码库中**已不存在**，当前 `mhc_post_kernel.asc` 已被替换为骨架版本。

---

### 1. 独立编译验证（维度 1：10/10）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 1.1 独立编译成功 | `cmake .. && make -j4` 在 `build_reviewer/` 中成功，产出 `mhc_post` 可执行文件 | 7/7 |
| 1.2 无代码级警告 | bisheng 编译输出零警告 | 3/3 |

**CMake 配置验证**：
- `find_package(ASC REQUIRED)` - 通过
- `LANGUAGES ASC CXX` - 通过
- `--npu-arch=dav-2201` - 通过 (匹配 Ascend910B2 / DAV_2201)
- `target_link_libraries` 含 `tiling_api` - 通过

独立编译命令：
```bash
cd /mnt/data01/zmz/workspace/12agent/waic/build/mhc_post/operators/mhc_post/build_reviewer
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cmake .. && make -j4
```

**结论**：编译通过、配置正确。编译验证层面无阻塞问题。

---

### 2. 架构合规性（维度 2：9/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 2.1 TPipe/TQue 模式 | TPipe 构造 + TQue<VECOUT> 使用正确 | 3/3 |
| 2.2 入口属性正确 | `extern "C" __global__ __vector__ void mhc_post_kernel(...)` 声明正确 | 3/3 |
| 2.3 定义顺序正确 | 构造函数 → Init → Process → private 成员 | 3/3 |
| 2.4 内存管理配对 | **严重不足**：仅初始化 1 个输出队列 (`outQue_`)，缺 4 个输入队列、系数队列、6 个 scratch buffer (resFp32_[0..3], term2_[0..3], xFp32_, tmpFp32_)。AllocTensor/FreeTensor/EnQue/DeQue 配对正确但范围极小 | 0/3 |
| 2.5 数据流完整 | **完全缺失**：无 CopyIn (输入数据不读取)、无 Compute (计算逻辑不存在)、仅有骨架 CopyOut (写零)。DESIGN.md §7.1 描述的三阶段数据流一条未实现 | 0/3 |

**详细分析**：

当前 `Init()` 函数仅执行：
```cpp
outGm_.SetGlobalBuffer(...);
pipe_->InitBuffer(outQue_, 1, cTile_ * sizeof(bfloat16_t));
```

DESIGN.md §6 要求初始化：
- `inQueRes_[0..3]` (4 个 VECIN 队列)
- `inQueX_` (VECIN 队列)
- `coeffBuf_` (TBuf, 80B) 或 `inQueCmb_` + `inQuePm_`
- `outQue_[0..3]` (4 个 VECOUT 队列) — 当前仅有 1 个
- `resFp32_[0..3]`, `term2_[0..3]`, `xFp32_`, `tmpFp32_` (6 个 VECCALC TBuf)

**比对**：实现 vs 设计，缺失率 = 15/16 的 buffer 未初始化。

---

### 3. 编码规范（维度 3：3/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 3.1 矢量 API | **完全缺失**：Kernel 无 Cast/Muls/Add 等任何矢量计算 API 调用。DESIGN.md §7.3 描述的 28 条向量指令/tile (16 Muls + 12 Add) 未实现 | 0/4 |
| 3.2 API 约束满足 | **严重违规**：`LocalTensor::SetValue()` 在逐元素循环中使用（`for i in 0..count: SetValue(i, 0)`），这是知名性能反模式。虽不在 API 黑名单中 (黑名单仅禁 `GlobalTensor::SetValue/GetValue`)，但在生产代码中逐元素写零不可接受。注释自身标注 "(debugging only)" | 0/4 |
| 3.3 数据对齐 | C_TILE=64 (128B = 4×32B) 对齐设计正确 | 3/3 |
| 3.4 命名规范 | 变量命名清晰（`n0_`, `n1Start_`, `outGm_` 等），风格一致 | 0/3 |

**3.4 扣分**：代码注释 `// Just write zeros to output` 和 `// Use SetValue to write zeros (debugging only)` 直接暴露代码为调试/骨架状态，非生产就绪代码。

---

### 4. 性能分析（维度 4：8/20）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 4.1 动态硬件参数 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取核数；`min(availableCoreNum, MAX_CORE_NUM, n1)` 正确计算 blockNum | 4/4 |
| 4.2 多核并行 | 沿 n1 维度均匀切分，空闲核正确跳过 (`n1Start_ >= n1_` 判断) | 4/4 |
| 4.3 流水线/双缓冲 | **无流水线**：BUFFER_NUM=1，仅 1 个输出队列。CopyIn/CopyOut 不存在，无计算-搬运重叠 | 0/4 |
| 4.4 同步策略 | **不适用**：无输入搬运、无计算、无 EnQue/DeQue 配对需求。当前唯一同步路径 (AllocTensor→SetValue loop→EnQue→DeQue→DataCopyPad→FreeTensor) 功能正确但无实际意义 | 0/4 |
| 4.5 计算效率与上板性能 | **无有效计算**：整个 kernel 只做 memset-zero。不计分 | 0/4 |

**硬件参数 Grep 检查**：
```bash
grep -n "blockDim\s*=\s*[0-9]" operators/mhc_post/*.asc  → 无匹配 (通过)
grep -n "blockIdx\s*=\s*[0-9]" operators/mhc_post/*.asc  → 无匹配 (通过)
```
核数/UB 大小均动态获取，无硬编码。

---

### 5. 测试覆盖（维度 5：10/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 5.1 测试数据生成 | `gen_data.py` + `golden.py` 完善：支持 11 个测试用例 (TC-01 ~ TC-11)，多 shape 覆盖，bf16 精度模拟 (CAST_NONE/CAST_ROUND)，全零/极大值/混合正负等边界 case | 4/4 |
| 5.2 结果验证脚本 | `verify_result.py` 使用 MERE/MARE 双阈值标准，含 Top-10 worst error 输出。但仅支持 bf16，无 fp16 或 fp32 路径 | 3/4 |
| 5.3 Level 0-3 覆盖 | Level 0 (小规模基础验证) 通过 TC-02/TC-03 覆盖；Level 1 (典型场景) TC-01/TC-08/TC-11；Level 2 (边界) TC-04~TC-07/TC-09~TC-10。测试用例定义 **完善**，但 kernel 骨架导致无法跑通验证 | 3/4 |
| 5.4 精度标准明确 | DESIGN.md §9.2 和 verify_result.py 明确 bf16 MERE < 2^-7 ≈ 7.81e-3, MARE < 10×2^-7 | 0/3 |

**5.4 扣分**：PLAN.md §2.4 定义了精度标准，但 `verify_result.py` 标准与 DESIGN.md 一致。扣分原因为精度阈值仅针对 bf16，未扩展 fp16/fp32 路径。

---

### 6. 精度验证（维度 6：0/10）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 6.1 FP32 全用例 PASS | N/A (算子设计输出为 bf16) | - |
| 6.2 FP16 全用例 PASS | N/A (算子使用 bfloat16_t，非 half) | - |
| 6.3 BF16 全用例 PASS | **完全失败**：输出 100% 为零（41,943,040 元素全部为 0），MERE=1.0, MARE=1.0 | 0/3 |

**独立精度测试结果**：
```
Shape: (2, 4096, 4, 1280), Total elements: 41,943,040
Output: Non-zero elements: 0 / 41,943,040 (0.000000%)
Max diff: 2.325000e+01, Mean diff: 1.692735e+00
MERE (9.999999e-01) < threshold (7.812500e-03): FAIL
MARE (1.000000e+00) < threshold (7.812500e-02): FAIL
Verification FAILED!
```

**精度评定**：**严重不达标**。100% 元素错误，输出全零。这属于代码 bug（Kernel 未实现计算逻辑），非数值精度问题。

---

### 7. 文档审查（维度 7：7/15）

| 检查项 | 结果 | 得分 |
|--------|------|------|
| 7.1 README.md 存在 | 存在，结构完整 | 3/3 |
| 7.2 数学公式 | README + DESIGN.md 均含完整公式 | 3/3 |
| 7.3 编译运行指南 | README 含 cmake/make/run 命令，run.sh 全自动 | 3/3 |
| 7.4 API 映射/约束 | DESIGN.md §8 列出详细 API 映射表。但与当前代码严重脱节：文档描述的 Cast/Muls/Add/EnQue×7/DeQue×7 在代码中全不存在 | 0/3 |
| 7.5 已知限制 | README"设计偏差说明"列出 3 项偏差（fp16 替代 bf16、per-row buffer、单 Tiling 结构），但**未列出最严重限制**：Kernel 为非功能性骨架，仅写零。且宣称 "99.99998% pass rate" 在当前代码版本下为**虚假陈述** | 0/3 |

**7.5 额外发现**：
- README 称"fp16"但头文件/tiling 均声明 `bfloat16_t`，存在文档-代码类型矛盾
- README 引用 `docs/perf/round_001/` 但该目录不存在
- `test_torch.py` 仍为模板代码 (包含 `add_custom` / `libadd_custom_ops.so`)，未针对 mhc_post 定制

---

### 8. 问题列表

#### HIGH 级问题（必须修复，阻塞通过）

| # | 类别 | 描述 | 修复建议 |
|---|------|------|---------|
| **H1** | **代码** | **Kernel 为非功能性骨架，仅写零到输出。** `mhc_post_kernel.asc` 的 `Process()` 未实现任何计算逻辑。DESIGN.md 描述的全部算法 (Cast bf16→fp32、K=4 向量点积 16 Muls + 12 Add、Broadcast MulAdd 4 Muls + 4 Add、Cast fp32→bf16、双缓冲流水线) 均缺失。精度验证 100% 失败 | 实现完整计算逻辑。建议参考 `mhc_post_kernel_test.asc` 中的计算架构（含 CopyIn/Compute/CopyOut 三阶段分离），但需修复：<br>1. 将 `half` 替换为 `bfloat16_t` 以匹配 DESIGN.md<br>2. 将 `BUFFER_NUM` 从 1 提升到 2（双缓冲）<br>3. 实现 PipelineBatch 流水线（预加载→CopyIn(N+1)∥Compute(N)∥CopyOut(N-1)→CopyOut(last)）<br>4. 参考 `$ASC_DEVKIT_DIR/examples/00_introduction/01_add/basic_api_memory_allocator_add/` 的双缓冲模式 |
| **H2** | **代码** | **mhc_post_kernel_test.asc 未接入编译流程且存在类型不匹配。** 该文件含实际计算逻辑但 CMakeLists.txt 编译的是骨架版 `mhc_post_kernel.asc`。test kernel 使用 `half` 而 gen_data.py 生成 bf16 数据，类型不匹配导致 kernel 运行挂起 | 确定最终 kernel 文件后统一 CMakeLists.txt、数据类型、数据生成脚本三者的类型声明。路径 A：修改 test kernel 为 bf16 并接入 CMake 编译；路径 B：统一改为 fp16 并同步更新 tiling 头文件、gen_data.py、verify_result.py、DESIGN.md |
| **H3** | **代码** | **LocalTensor::SetValue() 逐元素循环写入**。Kernel skeleton 使用 `for i in 0..count: SetValue(i, 0)` 逐元素写零，此模式性能极差。即使骨架阶段也应避免 | 移除 SetValue 循环，用 `Muls(zeroOut, zeroOut, 0.0f, count)` 批量清零，或在计算实现后完全移除清零逻辑 |

#### MEDIUM 级问题

| # | 类别 | 描述 | 修复建议 |
|---|------|------|---------|
| **M1** | **性能** | **BUFFER_NUM=1 无流水线。** 测试 kernel 中所有 TQue 声明为 `TQue<..., 1>`，CopyIn/Compute/CopyOut 完全串行。DESIGN.md §6.5 设计的双缓冲流水线（预期 2-3x 加速）未实现 | 将所有 TQue 改为 `TQue<..., 2>`，实现 PipelineBatch 模式（见 DESIGN.md §7.2）。参考 `$ASC_DEVKIT_DIR/examples/00_introduction/01_add/basic_api_memory_allocator_add/` |
| **M2** | **代码** | **系数加载未使用 TBuf 优化。** 测试 kernel 使用 TQue 加载 comb_res_mix (64B) 和 post_layer_mix (16B)，每列 tile 都重复加载。DESIGN.md §5.4/§6.4 设计使用 TBuf 逐 batch 一次性加载 80B 系数并在所有列 tile 间共享 | 将系数从 per-tile TQue 改为 per-batch TBuf：`DataCopyPad → TBuf → LocalTensor::GetValue()`，减少 20x 的系数加载次数 |
| **M3** | **文档** | **README 精度声明与代码状态矛盾。** README 宣称 "99.99998% pass rate" 但在当前代码版本下（输出全零）完全不能成立。`test_torch.py` 仍为模板代码 | 更新 README 为当前代码状态的真实描述；修复 `test_torch.py` 以匹配 mhc_post 算子 |
| **M4** | **测试** | **测试 kernel 运行挂起。** 将 test kernel 替换为主 kernel 后运行，在 `KernelCall: waiting for sync` 处挂起。疑似类型不匹配 (half vs bf16) 或同步 bug | 在修复 H2 后重新编译和测试。先用小 shape (TC-02: 1,1,64,4) 验证功能正确性 |

#### LOW 级问题

| # | 类别 | 描述 | 修复建议 |
|---|------|------|---------|
| **L1** | **文档** | **README 数据类型声明矛盾。** 宣称 "fp16" 但 `mhc_post_tiling.h`、`mhc_post_kernel.asc` 均使用 `bfloat16_t`；DESIGN.md 也指定 bf16 | 统一所有文档和代码的类型声明。若最终选择 bf16：更新 README。若选择 fp16：更新 DESIGN.md 和 tiling 头文件 |
| **L2** | **文档** | **`docs/perf/` 目录缺失。** README 引用 `docs/perf/round_001/` 但该目录不存在 | 创建 perf 目录并归档性能数据，或在 README 中移除此引用直至性能测试完成 |
| **L3** | **代码** | **Host 侧错误处理不完整。** `KernelCall()` 中 `aclrtMallocHost`/`aclrtMalloc`/`aclrtMemcpy` 返回值未检查 | 对每个 aclrt* 调用添加返回值检查和错误日志（参考现有 `aclrtSetDevice`/`aclrtGetDeviceInfo` 的错误处理模式） |
| **L4** | **文档** | **`op_extension/` 目录为空。** PLAN.md Phase 6 标记为"可选，未开始"，文档与代码一致 | 无操作项（仅记录）。如需 PyTorch 接入，填充 `mhc_post_torch.cpp` + `register.cpp` |

---

### 9. 硬件参数检查

| 检查项 | 命令 | 结果 |
|--------|------|------|
| `blockDim = 数字` | `grep -rn "blockDim\s*=\s*[0-9]" operators/mhc_post/op_kernel/ operators/mhc_post/op_host/` | **通过** - 无硬编码 |
| `blockIdx = 数字` | `grep -rn "blockIdx\s*=\s*[0-9]" operators/mhc_post/op_kernel/ operators/mhc_post/op_host/` | **通过** - 无硬编码 |
| 硬编码 TILE/UB 大小 | `grep -rn -E "(192000\|196608\|192\s*\*\s*1024)" operators/mhc_post/op_kernel/ operators/mhc_post/op_host/` | **通过** - 无硬编码 |

核数获取方式：`aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)` → 动态获取。通过。

---

### 10. 设计合规检查

对照 DESIGN.md 逐项验证：

| 设计项 | 状态 | 说明 |
|--------|------|------|
| §3 技术路线：SIMD/MemBase | **符合** | 使用 TPipe/TQue，无 RegBase/Blaze API |
| §4 多核切分：n1 维度, max 20 核 | **符合** | `min(availableCoreNum, MAX_CORE_NUM, n1)` 动态切分 |
| §5 UB 切分：C_TILE=64 | **符合** | C_TILE=64 正确定义和使用 |
| §5 B_TILE=1 逐 batch | **符合** | 骨架循环结构正确 (n0→n1→cTiles 三层) |
| §6 Buffer 规划 | **严重不符** | 仅 1 个 outQue_；缺 4 inQueRes_ + inQueX_ + 系数队列 + 8 scratch buffer |
| §7 计算流程 | **完全缺失** | 无 Cast、Muls、Add、Broadcast MulAdd、双缓冲流水线 |
| §8 API 映射 | **完全不符** | 文档列出 10+ API，代码仅使用 SetValue + DataCopyPad |
| §9 精度策略 | **未验证** | 文档设计正确 (混合精度、数值稳定性)，但 kernel 未实现故无法验证 |
| §10 分支场景 | **部分符合** | 空闲核跳过、尾块 TileSize() 已实现；h 非整除逻辑正确 |
| §12 Tiling 数据 | **符合** | MhcPostTiling 结构体 24B，Host→Kernel 一次搬运 |
| §13 架构合规 | **不符合** | 文档声称"API 黑名单合规"和"TPipe/TQue 模式"，但核心功能全部缺失 |

**NpuArch 验证**：
- 芯片型号：Ascend910B2 → NpuArch = DAV_2201 (通过 `/npu-arch` skill 查询)
- CMakeLists.txt `--npu-arch=dav-2201` 匹配目标芯片

---

### 11. RegBase 路线检查

DESIGN.md §3.1 明确排除 RegBase 路线（DAV_2201 不支持），代码中未使用任何 RegBase API (`RegTensor`/`MaskReg`/`asc_vf_call`/`__simd_vf__`)。检查通过。

---

### 12. 代码清洁检查（最终轮附加项）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 无残留模板文件 (`add_custom_*`) | **通过** | 此前 REVIEW.md Round 0 报告的模板文件已清理 |
| 无调试 printf 残留 | **通过** | Kernel 侧无 printf；Host 侧 printf 为正常日志 |
| `op_extension/` 为可选项 | **通过** | 目录为空，PLAN.md 标记为"可选，未开始" |
| 编译零警告 | **通过** | bisheng 编译无任何警告 |

---

### 13. 同步策略分析

当前骨架代码中唯一的同步路径：

```
Process() 循环内:
  AllocTensor(zeroOut) → SetValue 循环 → EnQue(zeroOut) → DeQue(z) → DataCopyPad → FreeTensor(z)
```

分析：
- AllocTensor/FreeTensor 配对：正确（EnQue 后 zeroOut 所有权转移至队列，DeQue 返回 z 后 FreeTensor(z) 正确释放）
- EnQue/DeQue 配对：正确 (1:1)
- 数据依赖：DeQue 保证 DMA 完成后再 DataCopyPad，正确
- **问题**：无任何冗余 PipeBarrier，但也无有效计算。同步分析无实际意义

---

### 14. 总结

**优点**：
- 编译配置正确（CMakeLists.txt find_package/LANGUAGES/--npu-arch/tiling_api 链接均合规），编译零警告
- 多核切分逻辑正确（动态核数获取、n1 维度均匀切分、空闲核跳过、尾核自适应）
- Tiling 参数设计合理（C_TILE=64 对齐、B_TILE=1 简化、MhcPostTiling 24B 紧凑）
- 测试基础设施完善（11 个测试用例覆盖多 shape/边界/极值场景，gen_data.py/golden.py 支持 bf16 CAST_NONE/CAST_ROUND 精度模拟）
- DESIGN.md 设计文档全面（方案决策、UB 规划、数据流、API 映射、精度分析均详实）

**严重差距（必须修复）**：
1. **H1**: `mhc_post_kernel.asc` 为非功能性骨架，仅写零到输出。DESIGN.md 全部计算逻辑未实现。精度验证 100% 失败
2. **H2**: `mhc_post_kernel_test.asc` 存在但类型不匹配 (half vs bf16) 且运行挂起，未接入编译
3. **H3**: `LocalTensor::SetValue()` 逐元素循环为性能反模式

**评分汇总**：

| 维度 | 得分 | 满分 | 关键扣分项 |
|------|------|------|-----------|
| 1. 编译验证 | 10 | 10 | - |
| 2. 架构合规 | 9 | 15 | 2.4 缺 15/16 buffer, 2.5 数据流全缺失 |
| 3. 编码规范 | 3 | 15 | 3.1 无矢量 API, 3.2 SetValue 反模式 |
| 4. 性能分析 | 8 | 20 | 4.3 无流水线, 4.4/4.5 无有效计算 |
| 5. 测试覆盖 | 10 | 15 | 测试基础设施完善但 kernel 无法跑通验证 |
| 6. 精度验证 | 0 | 10 | 输出全零，100% 失败 |
| 7. 文档审查 | 7 | 15 | 7.4 与代码脱节, 7.5 精度声明为虚假陈述 |
| **总计** | **37** | **100** | **FAIL (< 70)** |

**必须修复项未通过清单**：3.1 (矢量 API), 3.2 (API 约束), 6.3 (BF16 精度验证) 全部未通过 → **必须判定为 FAIL**。

**下一轮修复优先级**：H1 (实现完整计算逻辑) → H2 (统一类型声明) → M1 (双缓冲流水线) → M2 (系数 TBuf 优化)
