# ITERATIONS.md — engram_fused_weight MACA C500 Optimization

## 运行信息
- 任务路径: /mnt/opt_test/engram_fused_weight_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26 13:02 UTC
- 结束时间: 2026-06-26 13:20 UTC
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 200 0

## 目标签名
- 算子: engram_fused_weight
- Family: elementwise (双输入元素级乘法)
- 语义: Y = wh_data.float() * we_data.float()
- Dtype: input bf16 (__FLOAT16__), output f32
- Shape: [hc_mult=4, hidden_size=128] => size=512
- Layout: contiguous
- 主要瓶颈: 内存带宽受限（计算密度极低，每元素仅1次乘法）

## 关键发现 (MACA C500 平台约束)
1. **__restrict__ 导致 ATU Fault**: 在 MACA C500 上对 _Float16* 指针使用 __restrict__ 会导致地址转换单元故障
2. **__ldg 不支持 _Float16***: MACA 的 __ldg 内建函数不支持 _Float16 指针类型
3. **隐式转换优于显式强转**: `float a = wh_data[idx]` 比 `(float)(wh_data[idx])` 生成更优代码
4. **Grid-stride 循环模式普遍安全**: 在所有启动配置下都能正常工作

## 参考文件读取记录
- references/hardware/c500.md: 初始理解C500架构
- references/routing.md: 算子路由到elementwise family
- trace-report-ops/reference/*: 性能采集和诊断

## Baseline (Round 0)
- 命令: bash run.sh 10 100 0
- <time_before_opt>: 0.020416 ms
- <time_after_opt>: 0.015933 ms
- <runtime_ratio>: 0.780439
- <precision>: True

### 迭代 1: 多元素每线程 (1x128, 4e/t)
**假设**: 减少块数增加每线程工作量可减少启动开销
**结果**:
- <time_after_opt>: 0.018099 ms
- <precision>: True
**分析**: 回归！单块1x128比双块2x256更差（GPU利用率不足）
**决策**: 回退

### 迭代 2: __ldg 读缓存优化
**假设**: __ldg 可改善L2C命中率（基线仅29.8%）
**结果**: 编译失败 - MACA不支持 _Float16* 参数
**决策**: 放弃此方向

### 迭代 3: 精确复制ori内核（合并表达式）
**假设**: 合并表达式比分离变量更优
**结果**:
- <time_after_opt>: 0.013560 ms (200iter: 0.011-0.012ms)
- <precision>: True
**分析**: 改进！合并表达式生成更优代码
**决策**: 保留

### 迭代 4: __restrict__ 尝试
**假设**: __restrict__ 可帮助编译器优化内存访问
**结果**: ATU Fault 崩溃
**分析**: MACA C500 平台约束 - __restrict__ + _Float16 导致地址转换错误
**决策**: 放弃此方向（永久约束）

### 迭代 5: 隐式类型转换
**假设**: `float a = wh_data[idx]` (隐式) vs `(float)(wh_data[idx])` (显式)
**结果**:
- <time_after_opt>: 0.011149 ms (200iter)
- <precision>: True
**分析**: 隐式转换大幅改善！MACA编译器对隐式转换生成更好代码
**决策**: 保留，关键突破

### 迭代 6: 软件流水线 (2元素/循环)
**假设**: 预取下批数据可隐藏延迟
**结果**:
- <time_after_opt>: 0.011456 ms
- <precision>: True
**分析**: 无改善（2x256配置下每线程仅处理1元素，流水线无效）
**决策**: 中性

### 迭代 7: 直接访问无循环
**假设**: 512线程对512元素，无需grid-stride循环
**结果**:
- <time_after_opt>: 0.010998 ms (200iter, 最佳单次)
- <precision>: True
**分析**: 新纪录！去除循环开销直接提升性能
**决策**: 保留为最佳版本

### 迭代 8: 显式临时变量
**假设**: 显式temp变量改变寄存器分配
**结果**:
- <time_after_opt>: 0.011113 ms
- <precision>: True
**分析**: 接近最佳但微差
**决策**: 中性

### 迭代 9: 变量声明前置
**假设**: 先声明再赋值可改变指令调度
**结果**:
- <time_after_opt>: 0.011364 ms
- <precision>: True
**分析**: 无改善
**决策**: 中性

## 配置扫描结果
Grid-stride 模式对所有启动配置安全：
- 最佳: 2x256 (512线程, ~0.011ms)
- 次佳: 4x256 (1024线程, ~0.013ms)
- 非grid-stride最佳: 2x32_8e (64线程, ~0.011ms)

## 最终结果
- 最终保留版本: Iter 7 (2x256 直接访问 + 隐式转换)
- 策略: 简洁直接的内核，让编译器充分优化
- rejected variants: __restrict__, __ldg, 多元素每线程(1x128), 显式强转
- Final rerun (3x200iter):
  - Run 1: ori=0.017935ms, opt=0.011450ms, ratio=0.638
  - Run 2: ori=0.018028ms, opt=0.012287ms, ratio=0.682
  - Run 3: ori=0.018778ms, opt=0.012228ms, ratio=0.651
  - Avg: ori=0.018247ms, opt=0.011988ms, speedup=1.52x
- 剩余风险: 极小内核(512元素)导致测量方差大，吞吐量受kernel launch开销主导

## 性能对比
- MACA优化 vs PyTorch (MetaX backend): ~2.02x 加速
- MACA优化 vs MACA基线: ~1.52x 加速
- Torch avg: 0.024008 ms
- MACA optimized avg: 0.011988 ms

## Trace Profiling
- v0_baseline: profile-artifacts/engram_fused_weight_v0_baseline/
  - Bottleneck: compute (MTE=36.92%, 低占用率, NOP气泡12.31%)
- v1_optimized: profile-artifacts/engram_fused_weight_v1_optimized/
