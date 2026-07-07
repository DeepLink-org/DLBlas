# AscendC vs PyTorch (NPU) 性能测试报告（中文版）

**测试日期**：2026-07-03  
**硬件平台**：8× Ascend 910B2 NPU（实际使用 1 卡测试）  
**软件环境**：PyTorch 2.8.0+cpu，torch_npu 可用  
**测试方法**：每个算子的 AscendC 内核与等价 PyTorch 参考实现均在 NPU 上实测延迟，Warmup=10 次，Repeat=100 次，每次测量前后执行 `torch.npu.synchronize()`。  
**加速比定义**：加速比 = PyTorch_NPU_延迟 / AscendC_延迟（数值 > 1 表示 AscendC 更快）。

---

## 一、总体情况

| 统计项 | 数值 |
|--------|------|
| 测试算子总数 | 20 |
| 获得有效加速比数据 | 14 个（70%） |
| 无法完成测试 | 6 个（30%） |
| AscendC 快于 PyTorch | 11 个（占有效数据的 79%） |
| PyTorch 快于 AscendC | 2 个（占有效数据的 14%） |
| 基本持平 | 1 个（占有效数据的 7%） |
| **几何平均加速比** | **7.01×** |
| 算术平均加速比 | 32.13× |
| 中位数加速比 | 5.43× |
| 最小加速比 | 0.41× |
| 最大加速比 | 269.86× |

---

## 二、逐算子性能明细

### （一）加速比 > 10×（共 6 个）

#### 1. expand_kenel_bwd — 269.86×

| 项目 | 数值 |
|------|------|
| 功能说明 | expand_to_mhc 反向传播，沿广播维度做规约求和 |
| AscendC 延迟 | 100.0 us（1 次调用平均） |
| PyTorch NPU 延迟 | 26,983 us（1 次调用平均） |
| 提升倍数 | **269.86 倍** |
| 测试 Shape 数 | 1（通过 1） |
| 分析 | 反向传播涉及多步 gather/scatter/reduce 操作，AscendC 通过 kernel 融合将所有中间步骤合并为单次 kernel 调用，消除了大量中间张量的显存读写。 |

#### 2. engram_gate_w_reduce — 74.32×

| 项目 | 数值 |
|------|------|
| 功能说明 | Engram 门控权重梯度规约 |
| AscendC 延迟 | 199.5 us |
| PyTorch NPU 延迟 | 14,823 us |
| 提升倍数 | **74.32 倍** |
| 测试 Shape 数 | 1（通过 1） |
| 分析 | 权重梯度规约涉及多个权重矩阵的部分梯度累积，AscendC 在单次 kernel 内完成所有规约操作。 |

#### 3. sinkhorn — 26.49×

| 项目 | 数值 |
|------|------|
| 功能说明 | Sinkhorn 归一化：将方阵迭代归一化为双随机矩阵 |
| AscendC 延迟 | 7.6 us |
| PyTorch NPU 延迟 | 201 us |
| 提升倍数 | **26.49 倍** |
| 测试 Shape 数 | 1（通过 1） |
| 分析 | 迭代归一化在 PyTorch 中需要多次 kernel launch（每次归一化行+列），AscendC 在单次 kernel 内完成全部迭代。 |

#### 4. engram_fused_weight — 23.70×

| 项目 | 数值 |
|------|------|
| 功能说明 | 融合权重计算：`weight_hidden.T @ weight_embed` |
| AscendC 延迟 | 40.6 us |
| PyTorch NPU 延迟 | 961 us |
| 提升倍数 | **23.70 倍** |
| 测试 Shape 数 | 3（通过 3） |
| Shape 范围 | H=16~128, D=128~1280 |
| 分析 | 权重矩阵的转置+乘法融合为单次 kernel 调用。 |

#### 5. norm_fn — 23.61×

| 项目 | 数值 |
|------|------|
| 功能说明 | RMS 归一化（MHC 前置处理） |
| AscendC 延迟 | 9.5 us |
| PyTorch NPU 延迟 | 224 us |
| 提升倍数 | **23.61 倍** |
| 测试 Shape 数 | 3（通过 1） |
| ⚠️ 异常情况 | 3 个 Shape 中 2 个触发 aicore 异常（`rtStreamSynchronize` 执行失败），仅 1 个 Shape 通过。这说明 kernel 在部分输入尺寸下存在 bug，需要修复。 |

#### 6. hc_split_sinkhorn — 10.20×

| 项目 | 数值 |
|------|------|
| 功能说明 | 多头 Sinkhorn 分割：将 mix 参数分割为 pre/post/comb，含迭代 Sinkhorn 归一化 |
| AscendC 延迟 | 425.8 us |
| PyTorch NPU 延迟 | 4,343 us |
| 提升倍数 | **10.20 倍**（5 个 Shape 的几何平均） |
| 测试 Shape 数 | 5（全部通过） |
| Shape 范围 | b=1~64, s=1~16, hc=4~8, iters=20 |

---

### （二）加速比 2× ~ 10×（共 5 个）

#### 7. engram_hash — 5.46×

| 项目 | 数值 |
|------|------|
| 功能说明 | N-gram 哈希查找 |
| AscendC 延迟 | 553.0 us |
| PyTorch NPU 延迟 | 3,021 us |
| 提升倍数 | **5.46 倍**（4 个 Shape 的几何平均） |
| 测试 Shape 数 | 4（全部通过） |
| Shape 范围 | nt=32~4096, N=3~5, L=2~4, T=8~16 |

#### 8. expand_kenel_fwd — 5.41×

| 项目 | 数值 |
|------|------|
| 功能说明 | 张量扩展前向：将输入沿 MHC 维度广播扩展 |
| AscendC 延迟 | 83.2 us |
| PyTorch NPU 延迟 | 450 us |
| 提升倍数 | **5.41 倍**（5 个 Shape 的几何平均） |
| 测试 Shape 数 | 5（全部通过） |
| Shape 范围 | B=1~4, S=1~1024, H=128~1280, M=1~16 |

#### 9. pre_split_mixes — 3.02×

| 项目 | 数值 |
|------|------|
| 功能说明 | MoE 前置分割：将输入 mix 按通道分割为 pre/post/comb |
| AscendC 延迟 | 119.6 us |
| PyTorch NPU 延迟 | 362 us |
| 提升倍数 | **3.02 倍** |
| 测试 Shape 数 | 1（通过 1） |

#### 10. head_compute_mix_fwd — 2.95×

| 项目 | 数值 |
|------|------|
| 功能说明 | 头部计算混合前向：`sigmoid(x*s+b) + eps` |
| AscendC 延迟 | 4,912 us |
| PyTorch NPU 延迟 | 14,500 us |
| 提升倍数 | **2.95 倍**（4 个 Shape 的几何平均） |
| 测试 Shape 数 | 4（全部通过） |
| Shape 范围 | bs=1~32, n1=1~32768 |

#### 11. engram_gate_fwd — 2.31×

| 项目 | 数值 |
|------|------|
| 功能说明 | Engram 门控前向 |
| AscendC 延迟 | 140.9 us |
| PyTorch NPU 延迟 | 325 us |
| 提升倍数 | **2.31 倍** |
| 测试 Shape 数 | 1（通过 1） |

---

### （三）加速比 ≤ 2×（共 3 个）

#### 12. act_quant_kernel — 1.38×

| 项目 | 数值 |
|------|------|
| 功能说明 | 激活量化：`round(clip(x / amax * fp8_max, -448, 448))` |
| AscendC 延迟 | 245.5 us |
| PyTorch NPU 延迟 | 340 us |
| 提升倍数 | **1.38 倍**（5 个 Shape 的几何平均） |
| 测试 Shape 数 | 5（全部通过） |
| 分析 | 量化操作中 NPU 硬件已有部分原生支持，AscendC 优势有限。 |

#### 13. head_compute_mix_bwd — 0.66× ⚠️

| 项目 | 数值 |
|------|------|
| 功能说明 | 头部计算混合反向传播 |
| AscendC 延迟 | 495 us |
| PyTorch NPU 延迟 | 327 us |
| 提升倍数 | **0.66 倍（AscendC 慢 34%）** |
| 分析 | 反向传播计算量相对较小，AscendC kernel launch 开销和显存搬运时间超过了实际计算节省的时间。建议将该算子与上下游算子**融合**使用。 |

#### 14. apply_mix — 0.41× ⚠️

| 项目 | 数值 |
|------|------|
| 功能说明 | 逐元素混合：`(x * mix).sum(dim=-2).bfloat16()` |
| AscendC 延迟 | 280.9 us |
| PyTorch NPU 延迟 | 115 us |
| 提升倍数 | **0.41 倍（AscendC 慢 59%）** |
| 测试 Shape 数 | 4（全部通过） |
| 分析 | `乘法→求和→类型转换` 流水线已被 PyTorch AclNN 高度优化。独立 AscendC kernel 的 launch 开销远大于计算本身。**强烈建议与上下游算子融合**。 |

---

## 三、无法完成测试的算子（共 6 个）

| 序号 | 算子名称 | 失败现象 | 根本原因 | 分类 |
|------|----------|---------|---------|------|
| 15 | **sparse_attn** | .so 加载时进程崩溃（SIGABRT） | C++ 注册中 `Scalar` 与 `float` 类型不匹配：TORCH_LIBRARY 声明 `Scalar` 但 kernel 实现签名用 `float` | **工程质量 — .so 构建问题** |
| 16 | **engram_gate_bwd** | .so 加载时段错误（SIGSEGV, rc=-11） | 内核初始化时空指针访问，疑似 tiling 参数计算错误 | **工程质量 — 内核 bug** |
| 17 | **big_fuse** | `get_inputs()` 仅返回 1 个张量，`torch.ops.npu.big_fuse` 需要 4 个输入 | 测试数据依赖外部二进制文件 `input/*.bin`，仓库中未提供 | **测试基础设施** |
| 18 | **MTPBlock** | `hc_post` 调用 `aclnnMatmul` 报 k 轴不匹配：`[2,1024,4,1280]` vs `[4,4]` | 输入张量形状与 AscendC 内核预期不兼容，需确认实际调用接口 | **输入形状不匹配** |
| 19 | **mhc_post** | 测试数据解析失败 | `generate_mhc_post_test_data` 返回 dict 格式而非 tuple，需适配 | **测试基础设施** |
| 20 | **indexer** | 无 AscendC .so 文件；此外 origin 参考代码依赖 `torch.cuda`，与 NPU 不兼容 | 该算子尚未开发 AscendC 内核 | **无 AscendC 实现** |

---

## 四、分类分析

### 大幅提升（>20×）：6 个算子

这些算子涉及**反向传播、多步规约、迭代算法**等复杂计算模式。AscendC 通过 kernel 融合将多个中间步骤合并为单次 kernel 调用，完全消除了中间张量的显存分配和读写。

| 算子 | 加速比 | 核心优化点 |
|------|--------|-----------|
| expand_kenel_bwd | 269.86× | 反向传播 gather/scatter/reduce 全融合 |
| engram_gate_w_reduce | 74.32× | 多权重矩阵梯度规约单 kernel 完成 |
| sinkhorn | 26.49× | 迭代 Sinkhorn 归一化单 kernel 完成 |
| engram_fused_weight | 23.70× | 权重转置+矩阵乘法融合 |
| norm_fn | 23.61× | RMS 归一化全融合 |
| hc_split_sinkhorn | 10.20× | sigmoid + Sinkhorn 迭代融合 |

### 稳健提升（2×~10×）：5 个算子

这些算子从中等复杂度的 AscendC 实现中获益，加速比稳定在 2×~6×。

| 算子 | 加速比 |
|------|--------|
| engram_hash | 5.46× |
| expand_kenel_fwd | 5.41× |
| pre_split_mixes | 3.02× |
| head_compute_mix_fwd | 2.95× |
| engram_gate_fwd | 2.31× |

### 优势有限或倒挂（≤2×）：3 个算子

这些算子均为**简单逐元素操作**，NPU 的 AclNN 库已经对其高度优化。AscendC kernel launch 开销（约数十微秒）和显存搬运时间超过了计算节省的时间。

| 算子 | 加速比 | 建议 |
|------|--------|------|
| act_quant_kernel | 1.38× | 可接受，仍有微小优势 |
| head_compute_mix_bwd | 0.66× | 应与上游算子**融合** |
| apply_mix | 0.41× | **强烈建议融合**，独立部署无意义 |

---

## 五、建议

1. **优先推广** `expand_kenel_bwd`、`engram_gate_w_reduce`、`sinkhorn`、`engram_fused_weight` 四个算子（>20× 加速比），它们是最大的性能收益来源。

2. **修复 norm_fn 的 aicore 异常**：3 个 Shape 中 2 个失败，存在严重的 kernel 健壮性问题，需要排查 tiling 逻辑在特定尺寸下的边界条件。

3. **修复工程质量问题**：
   - `sparse_attn`：修正 C++ TORCH_LIBRARY 注册中 `Scalar` 与 `float` 的类型不一致
   - `engram_gate_bwd`：排查段错误根因，优先检查 tiling 参数计算
   - `big_fuse`、`MTPBlock`：补齐测试数据和接口文档

4. **对 `apply_mix` 和 `head_compute_mix_bwd` 实施算子融合**：这两个算子单独运行的 kernel launch 开销超过计算收益。将它们与上下游算子（如 `pre_split_mixes`、`hc_split_sinkhorn` 等）融合可大幅提升端到端性能。

5. **`indexer` 补充 AscendC 实现**：该算子目前仅有 PyTorch 参考，且依赖 CUDA，无法在 NPU 上运行。

---

## 六、测试配置详情

| 参数 | 取值 |
|------|------|
| 硬件 | 8× Ascend 910B2 NPU |
| NPU 驱动 | 25.2.3 |
| PyTorch 版本 | 2.8.0+cpu |
| torch_npu | 可用 |
| 测试使用 NPU 数量 | 1 卡 |
| Warmup 次数 | 10 |
| Repeat 次数 | 100 |
| 数据类型 | float16 / bfloat16（依算子而定） |
| 计时方式 | `time.perf_counter()` + `torch.npu.synchronize()` |
| 加速比计算方式 | 几何平均（仅对通过的 Shape 计算） |

---

*本报告所有数据均来自 NPU 硬件实测，非估算值。*
