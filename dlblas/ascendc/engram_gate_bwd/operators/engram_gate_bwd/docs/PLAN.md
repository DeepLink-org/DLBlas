# engram_gate_bwd 算子开发计划

> **版本**: v2.0 | **日期**: 2026-07-07

---

## v17 修复：挂起 Bug + 精度修复

### 根因分析

#### 1. 挂起 Bug（DAV_2201 TQue 队列死锁）

**问题**：`C1()` 在 T 循环内被反复调用，每次调用对单槽队列执行 `AllocTensor()` 且从不 `FreeTensor()`。

- goQ_, xQ_, kQ_, gxQ_, gkQ_, gvQ_, t0_, t1_, t2_, d0_, d1_, s_[0..13], rQ_, cQ_ 均为深度=1 的 TQue
- 第 1 次调用 C1: 所有 AllocTensor 成功（队列为空）
- 第 2 次调用 C1: AllocTensor 永久阻塞（队列槽位已被占用）

**修复**：将所有 per-T buffer 分配从 `C1()` 移到 `Process()` 中 T 循环之前（一次性分配），将 `LocalTensor` 引用传入 `C1()`。每次迭代在同一 buffer 上覆盖数据。

#### 2. 槽位复用冲突

原代码中 4 个 s_ 队列被一个 C1 调用内部双重分配（先分配 cm/sg/ng/ss，后分配 mk/gs/gg/gd）。虽单次 C1 内不立即挂起（编译器/运行时特殊行为），但 v17 中将所有分配提前到 Process() 时，这些槽位必须全部共存。

**修复**：s_[] 从 14 扩展到 18，新增 s_[14..17] 分别用于 gg, gs, mk, gd。

#### 3. grad_v 精度问题（Compares + Select 符号计算异常）

printf 调试发现：gate 值在某些 H 索引上与 PyTorch golden 存在显著差异（符号反转）。根因为 Compares + Select 在 DAV_2201 上处理少量元素（H=4）的 mask 时存在异常。

**修复**：将 `s_sqrt = sign(dot) * sqrt(max(|dot|, cv))` 中的 sign 计算从 Compares+Select 改为纯算术方法：
```
sign(dot) = dot / max(|dot|, eps)
s_sqrt = sign * sqrt(max(|dot|, cv))
```

此方法避免了 mask 格式对齐问题。

#### 4. XR() 中 t0_ 队列冲突

`t0_` 被 T 循环中的 `th0` 和 XR() 中的 `w` 同时使用。由于 v17 中 th0 在 Process() 中预分配且不释放，XR() 中的 `AllocTensor` 会阻塞。

**修复**：将 `th0` 通过引用传递给 XR()，在 XR() 内直接复用，无需重新分配。

| 修复项 | 文件 | 变更 |
|--------|------|------|
| 挂起 Bug | engram_gate_bwd_kernel.asc | AllocTensor 从 C1() 移到 Process()，通过引用传参 |
| 槽位复用 | engram_gate_bwd_kernel.asc | s_[14] → s_[18]，新增 4 个队列 |
| grad_v 精度 | engram_gate_bwd_kernel.asc | sign 计算从 Compares+Select 改为算术 Div |
| XR t0_ 冲突 | engram_gate_bwd_kernel.asc | th0 通过引用传递给 XR() |
| 编译兼容 | engram_gate_bwd_kernel.asc | 去除 FreeTensor 调用（TPipe 无此方法） |

### 测试结果

| 输出 | 状态 | max_diff | mean_diff | mismatches |
|------|------|----------|-----------|------------|
| grad_x (7168) | PASS | 9.65e-04 | 8.91e-05 | 0 |
| grad_k (7168) | PASS | 1.05e-04 | 1.27e-06 | 0 |
| grad_v (1792) | PASS | 9.73e-04 | 1.12e-04 | 0 |
| grad_wh (512) | PASS | 1.04e-04 | 8.06e-06 | 0 |
| grad_we (512) | PASS | 9.05e-05 | 7.27e-06 | 0 |

验证参数: rtol=0.01, atol=0.001

### 当前状态

| 项目 | 状态 |
|------|------|
| 代码实现 | 完成（v17） |
| 编译 | 通过（可执行文件 target） |
| 运行 | 通过（无挂起） |
| 精度验证 | 全部通过（5/5 输出） |
| Torch 扩展库 | 未编译（pre-existing CANN 9.0.0 环境兼容问题） |
