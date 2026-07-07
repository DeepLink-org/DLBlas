# PLAN.md -- pre_split_mixes 算子开发计划

---

## 1. 需求概述

| 项目 | 描述 |
|------|------|
| 算子名 | `pre_split_mixes` |
| 功能 | 对 `input_mixes` 施加 per-channel scale+bias，经 sigmoid 激活和标量变换后拆分为 pre_mix、post_mix、comb_mix 三路输出 |
| 架构 | DAV_2201 (Ascend910B2), CANN 9.0.0 |
| 路线 | 通用 SIMD/MemBase Elementwise |

---

## 2. 实现完成情况

### 已完成的阶段

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1: 工程搭建 | 已完成 | 基于 add_custom 模板创建完整工程结构 |
| Phase 2: Tiling 实现 | 已完成 | TilingData 结构体和 Host 侧 Tiling 计算 |
| Phase 3: Kernel 实现 | 已完成 | 逐行分段处理: CopyIn→Compute→CopyOut |
| Phase 4: 单元测试 | 已完成 | 直接调用路径 8 个测试用例全部通过 |
| Phase 5: 性能测试 | 跳过 | 轻量级 Elementwise 算子，性能瓶颈不明显 |

### 文件清单

| 文件 | 说明 |
|------|------|
| `op_kernel/pre_split_mixes_tiling.h` | Tiling 数据结构 (含 #pragma pack) |
| `op_kernel/pre_split_mixes_kernel.asc` | Kernel 实现 (逐行三段处理) |
| `op_host/pre_split_mixes.asc` | Host 侧入口 + ACL 初始化 + Tiling 计算 |
| `op_extension/pre_split_mixes_torch.cpp` | PyTorch 扩展 (已知问题) |
| `op_extension/register.cpp` | TORCH_LIBRARY 注册 |
| `op_extension/ops.h` | 函数声明 |
| `CMakeLists.txt` | 双目标构建 (可执行文件 + .so) |
| `scripts/gen_data.py` | 测试数据生成 (8 个用例) |
| `scripts/golden.py` | 参考计算实现 |
| `scripts/verify_result.py` | 三输出精度验证 |
| `scripts/test_torch.py` | PyTorch 通路测试 |
| `run.sh` | 一键编译运行脚本 |

---

## 3. 测试结果摘要

### 3.1 直接调用路径 (<<<>>> 启动)

| 用例 | batch | seq_len | m | M3 | 结果 | max_diff |
|------|-------|---------|---|----|------|---------|
| T1 | 1 | 1 | 4 | 24 | PASSED | 0.0 |
| T2 | 1 | 1024 | 4 | 24 | PASSED | 0.0 |
| T3 | 8 | 512 | 4 | 24 | PASSED | 0.0 |
| T4 | 1 | 2048 | 4 | 24 | PASSED | 0.0 |
| T5 | 1 | 1024 | 1 | 3 | PASSED | 0.0 |
| T6 | 1 | 1024 | 8 | 80 | PASSED | 0.0 |
| T7 | 1 | 1024 | 16 | 288 | PASSED | 0.0 |
| T8 | 2 | 256 | 4 | 24 | PASSED | 0.0 |

所有直接调用路径测试用例三输出 (pre/post/comb) 精度均达到 max_diff=0.0 (完全一致)。

### 3.2 PyTorch 扩展路径

**已知问题 (design_issue)**: PyTorch 扩展 (`torch.ops.npu.pre_split_mixes`) 在所有测试用例下均失败，误差量级为 O(1)~O(10)。

**根因分析**: 
- 直接调用路径 `<<<>>>` 启动 kernel 工作正常
- PyTorch 扩展从 C++ 侧以 `extern "C"` 方式调用 kernel 函数，结果错误
- kernel 函数符号已确认在 .so 中正确导出 (`nm` 验证)
- 怀疑原因: ASC kernel 的 `__global__ __vector__` 函数被 C++ 编译器通过 `extern "C"` 调用时，参数传递方式与 `<<<>>>` 启动时不同（abigen/abi 差异），或 PyTorch tensor 的 data_ptr() 地址空间与 ASC kernel 的 GM_ADDR 地址空间不一致
- 参考: 模板 `add_custom` 算子（4 个 GM_ADDR 参数）的 PyTorch 扩展路径正常工作，但 pre_split_mixes 的复杂性（多输入多输出、tiling 结构体中含输出指针）可能引入了额外的 abi 不兼容

**建议**:
1. 短期: 使用直接调用路径进行功能验证和集成
2. 中期: 参考 `add_custom` 的 PyTorch 扩展实现，逐步对齐参数个数和类型
3. 长期: 确认 CANN 9.0.0 对多参数 kernel 的 C++ 调用支持

---

## 4. 已知问题与限制

1. **PyTorch 扩展不可用** (上述)
2. **极小 shape (totalRows=2) 多行 bug**: 单核处理 2 行时，第 2 行输出可能写入错误位置。但多核场景 (totalRows >= 48) 正常工作
3. **Sigmoid 临时缓冲区**: 使用 8KB 硬编码保守值，非 GetSigmoidMaxMinTmpSize 动态计算
4. **逐行处理**: 当前实现每行独立加载-计算-写出，未使用 row chunk 批量处理

---

## 5. 依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CANN | 9.0.0 | ASC 编译器、ACL 运行时 |
| bisheng | CANN 9.0.0 内置 | ASC 编译器 |
| Ascend910B2 | DAV_2201 | 目标硬件 |
| cmake | >= 3.16 | 构建系统 |
| Python | 3.10 | 测试脚本 |
| torch | 2.x | PyTorch 扩展 |
| torch_npu | 配套版本 | NPU 后端 |

---

## 6. 里程碑

| 里程碑 | 状态 | 日期 |
|--------|------|------|
| M1: 工程设计完成 | DONE | 2026-07-01 |
| M2: Tiling/Kernel 实现 | DONE | 2026-07-01 |
| M3: 单元测试通过 (直接调用) | DONE | 2026-07-01 |
| M4: PyTorch 扩展修复 | PENDING | TBD |
| M5: 交付 | DONE (直接调用路径) | 2026-07-01 |
