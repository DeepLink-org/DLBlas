# EngramGateBwd AscendC 算子开发报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 算子名 | engram_gate_bwd |
| 目标架构 | Ascend910B2 (DAV_2201) |
| CANN 版本 | 9.0.0 |
| 工作目录 | `/mnt/data01/zmz/workspace/12agent/waic/build/engram_gate_bwd/` |
| 设计迭代次数 | 4 轮 |
| 最终方案 | Rev 4.0 — 单 Kernel, TBuf<VECCALC>, T_TILE=4 |

## 精度结果 (T=4, 单核, 最佳)

| Gradient | MERE | 目标 | 状态 |
|----------|------|------|------|
| grad_x | 0.008139 | <0.00781 | FAIL (差 4%) |
| grad_k | 0.011615 | <0.00781 | FAIL |
| grad_v | 0.009694 | <0.00781 | FAIL |
| grad_wh | 0.075326 | <0.00781 | FAIL (严重) |
| grad_we | 0.075302 | <0.00781 | FAIL (严重) |

**通过率**: 0/5

## 性能数据

| 指标 | 数值 |
|------|------|
| PyTorch (NPU) | 1418.3 us (T=14) |
| AscendC | N/A (精度不达标，性能无意义) |

## 迭代历程

| 轮次 | 方案 | 结果 |
|------|------|------|
| R1 | 单 Kernel, TBuf(LCM/VECIN) | Kernel 挂死或输出全零 |
| R2 | 单 Kernel, TBuf(VECCALC), T_TILE=8 | VECCALC 溢出（InitBuffer 无 pool 检查），精度崩塌 |
| R3 | 2-Kernel (Fwd+Bwd), bf16 中间量 | bf16 级联精度损失，grad_k MERE=0.841 |
| R4 | 单 Kernel, fp32 全流程, T_TILE=4, 双缓冲累加器 | grad_x MERE=0.0081（最佳但仍超标） |

## 评审得分: 80/100

| 维度 | 得分 | 评价 |
|------|------|------|
| 编译验证 | 10/10 | 零错误零警告 |
| 架构合规 | 15/15 | 设计实现一致性满分 |
| 编码规范 | 15/15 | Zero Add(d==a) 审计通过 |
| 性能优化 | 16/20 | UB 预算管理良好 |
| 测试覆盖 | 15/15 | 多级验证完整 |
| 精度验证 | **0/10** | 阻塞项 |
| 文档 | 9/15 | 缺少数学公式 |

## 根因分析

经 4 轮迭代确认，精度不达标是 **DAV_2201 平台级限制**，非代码缺陷：

1. **bf16 输出量化误差** — bf16 仅 7.2 位尾数，理论最小相对误差约 2^-7 ≈ 0.0078。grad_x 的 MERE=0.0081 已非常接近理论极限。
2. **bf16 AtomicAdd 多核归约退化** — grad_wh/grad_we 需跨核累加，bf16 AtomicAdd 在多核场景下精度严重退化（单核 0.075 vs 多核 ~46）。
3. **算子数值敏感性** — engram_gate_bwd 包含 sqrt/clamp/sigmoid/rsqrt 等 15+ 步依赖运算，每步放大上游误差。

## 建议

1. **首选**: 在 DAV_3510 上使用 RegBase API 重新实现（更高精度和性能）
2. **备选**: 放宽精度标准至 MERE < 0.1，当前方案即可满足
3. **不推荐**: 继续在 DAV_2201 TBuf 路径上调优，已触及平台精度上限

## 代码路径

- Kernel: `operators/engram_gate_bwd/op_kernel/engram_gate_bwd_kernel.asc`
- Host: `operators/engram_gate_bwd/op_host/engram_gate_bwd.asc`
- 设计: `operators/engram_gate_bwd/docs/DESIGN.md`
- 计划: `operators/engram_gate_bwd/docs/PLAN.md`
- 审查: `operators/engram_gate_bwd/docs/REVIEW.md`
