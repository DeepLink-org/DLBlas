# indexer MACA 算子优化迭代记录

## 运行信息
- 算子: indexer (sparse attention index computation)
- 工作区: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-29
- 验证命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0`

## 目标签名
- 算子: indexer (einsum + ReLU + scale + sum + mask + TopK)
- Family: matmul-like (batched dot product + elementwise post-processing)
- dtype: bfloat16 (storage), float32 (computation)
- Shape: B=2, S=64, H=16, D=64, T_total=256, T_used=16, TopK=16
- Layout: contiguous
- 主要瓶颈: 冗余全局内存加载 — 每个 kv_cache 行被 H=16 个头重复加载

## 参考文件读取记录
- `references/routing.md`: 执行前路由
- `references/hardware/c500.md`: C500 硬件特性
- `references/verification.md`: 验证流程
- `references/case_retrieval.md`: 案例检索
- `references/operator_families/matmul.md`: Matmul 优化
- `references/operator_families/elementwise.md`: Elementwise 优化

## Baseline
- Round 0 命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0`
- `<time_before_opt>`: 0.110899 ms
- `<time_after_opt>`: 0.096064 ms
- `<runtime_ratio>`: 0.866228
- `<precision>`: True

### Round 0 分析
- Block 64 threads, grid 128 blocks, effective occupancy 5%
- GVM share 11%, L2C hit 98.46%
- ISU wsm_stall 45.92% — shared memory atomicAdd contention
- STE 28.65% — significant scalar/control overhead

### 迭代 1: 移除 atomicAdd，使用 per-thread 私有累积 + 规约
**假设**: atomicAdd 是主要瓶颈，移除它会有显著提升
**目标**: 使用 per-thread 私有分数 + 共享内存规约替代 atomicAdd
**结果**:
- `<time_after_opt>`: 0.101983 ms
- `<runtime_ratio>`: 0.912
- `<precision>`: True
**分析**: 变差了！额外的共享内存初始化和规约开销超过了 atomicAdd 节省的时间
**决策**: 回退

### 迭代 2: 展开内积循环 + 增大 block size
**假设**: 展开循环和更大 block 能提升吞吐
**目标**: 4x 展开内积循环 (D=64)，block size 256
**结果**:
- `<time_after_opt>`: 0.095608 ms
- `<runtime_ratio>`: 0.824
- `<precision>`: True
**分析**: 相比 baseline 有小幅提升（~0.5%），展开和更大 block 有帮助
**决策**: 保留

### 迭代 3: 2D block 映射 (h,t) 对
**假设**: 2D block 天然映射到 (h,t) 对，消除 grid-stride 循环
**目标**: blockDim=(T_used=16, H=16)，warp 级规约
**结果**:
- `<time_after_opt>`: 0.095854 ms
- `<runtime_ratio>`: 0.935
- `<precision>`: True
**分析**: 2D block 的同步开销（4 次 `__syncthreads`）抵消了映射优势
**决策**: 回退

### 迭代 4: 预加载 kv_cache 到共享内存 [★ 最佳版本]
**假设**: kv_cache 每行被 H=16 个头重复加载，预加载到共享内存能消除冗余
**目标**: 协作加载 kv_cache[b,0:T_used,:] 到共享内存
**结果**:
- `<time_after_opt>`: 0.043579 ms
- `<runtime_ratio>`: 0.3915
- `<precision>`: True
**分析**: **巨大成功！2.23x 加速！** 预加载消除了 H-1=15 次冗余全局加载
**决策**: ★ 保留为最佳版本

### 迭代 5: 测试不同 block size
**假设**: 不同 block size 影响 occupancy
**目标**: 测试 block_size=128, 256, 512
**结果**: 128: 0.044 ms, 256: 0.044 ms, 512: 0.043 ms — 差异在噪声范围内
**决策**: 保留 block_size=256 作为最平衡的选择

### 迭代 6: 预加载 q 到共享内存
**假设**: q 数据也能从预加载中受益
**目标**: 同时预加载 q 和 kv_cache 到共享内存
**结果**: 0.048059 ms — 比仅预加载 kv 更慢
**分析**: q 数据每个头只访问一次，预加载无益，反而增加共享内存压力和同步开销
**决策**: 回退

### 迭代 7: 8x 展开 (替代 4x)
**假设**: 更大的展开因子能减少循环开销
**目标**: 8x 展开内积循环
**结果**: 0.044119 ms — 与 4x 基本相同
**决策**: 保留 4x 展开（代码更简洁）

### 迭代 8: per-t 线程分配
**假设**: 每个线程专属于一个 t 位置，消除 atomicAdd
**目标**: 线程按 t 位置分组，组内规约
**结果**: 0.059062 ms — 显著变差
**分析**: 分支发散和线程分组逻辑的开销很大
**决策**: 回退

### 迭代 9: 每个 block 处理 2 个 (b,s) 对
**假设**: 减少 grid 中的 block 数量能降低 launch 开销
**目标**: grid=64 blocks, 每 block 处理 2 个 (b,s) 对
**结果**: 0.060004 ms — 显著变差
**分析**: 翻倍的共享内存和循环开销抵消了 block 数量减少的收益
**决策**: 回退

## 性能分析 (trace-report)
- 见: `profile-artifacts/indexer_v0_baseline/REPORT_baseline.md`
- 瓶颈类别: compute
- AP MTE duty: 3.40%
- 有效 occupancy: 5.00%
- L2C hit rate: 98.46%
- ISU wsm_stall: 45.92%

## MACAC vs Torch 性能比较
- MACAC (优化版): 0.043486 ms
- MACAC (基线版): 0.097076 ms
- PyTorch: 0.213847 ms
- **MACAC 优化 vs Torch: 4.92x 加速**
- **MACAC 优化 vs 基线: 2.23x 加速**
- 运行时比值 (MACAC/Torch): 0.2034

## 最终结果
- 最终保留版本: Iteration 4 (kv_cache 预加载到共享内存)
- 优化策略: 共享内存 kv_cache 预加载 + 4x 内积展开 + block_size=256
- 总加速: 2.23x vs baseline, 4.92x vs Torch
- Final rerun 标签:
  - `<time_before_opt>`: 0.097076 ms
  - `<time_after_opt>`: 0.043486 ms
  - `<runtime_ratio>`: 0.447954
  - `<precision>`: True
- Rejected variants: 迭代 1,3,6,8,9 (全部回归)
- 剩余风险: 小问题规模下测量噪声可能影响精度；更大 T_used 和 H 需额外验证
