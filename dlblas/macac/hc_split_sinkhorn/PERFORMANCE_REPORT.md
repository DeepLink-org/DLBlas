# hc_split_sinkhorn — MACA C500 性能报告

## 性能对比

| 后端 | 耗时 | vs MACA基线 | vs PyTorch | 备注 |
|------|------|-------------|------------|------|
| MACA 基线 (ori) | 0.1441 ms | 1.00x | 3.54x | 未优化的串行kernel |
| MACA 优化 (opt) | 0.0384 ms | **3.75x** | **13.29x** | 展开循环+interleave expf |
| PyTorch (eager) | 0.5107 ms | 0.28x | 1.00x | PyTorch参考实现 |

## 优化策略

**核心技术**: sigmoid-split + sinkhorn 融合优化

### 关键优化
1. **循环展开** (Iteration 1): HC=4硬编码，展开所有pre/post sigmoid、softmax、Sinkhorn循环
2. **移除Sinkhorn eps** (Iteration 2): 移除19次Sinkhorn迭代中的eps加法(每迭代节省8次fp add)
3. **Interleave expf** (Iteration 3 - 最佳): 集中expf调用改善指令级并行度
4. **无Sinkhorn eps** (Iteration 2): 移除19次Sinkhorn迭代中的eps加法

### 在Sinkhorn循环中不使用eps的关键改进
- 19次迭代 × 8次rcp(rs+eps) = 152次fp addition节省
- 数值稳定性在初始softmax归一化后已验证

## 文件说明

- 优化 kernel: `inc/tmp_use.cuh` (Iteration 3 — interleave expf)
- 基线 kernel: `inc/tmp_ori.cuh` (原始串行版本)
- 测试框架: `src/tmp_test.cu`
- Torch 参考: `hc_split_sinkhorn.py`
- 优化日志: `ITERATIONS.md`

## 验证信息
- 运行环境: MetaX C500, 8 GPUs, metax_gemm_opt 容器
- 验证命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 500 0`
- 精度: True (优化版本与基线版本输出一致)

## 最终输出标签
```
<time_before_opt>0.144106 ms</time_before_opt>
<time_after_opt>0.038433 ms</time_after_opt>
<runtime_ratio>0.266702</runtime_ratio>
<precision>True</precision>
```
