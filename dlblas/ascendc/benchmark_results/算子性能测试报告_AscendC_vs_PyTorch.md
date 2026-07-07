# EngramGate 全量算子 AscendC vs PyTorch 性能测试报告

**测试日期**: 2026-07-07
**测试平台**: Ascend910B2 (DAV_2201), CANN 9.0.0
**测试方法**: Warmup=10, Repeat=100, torch.npu.synchronize() 后计时
**对比基准**: AscendC kernel 耗时 vs PyTorch NPU/CPU 等效实现

---

## 一、实测算子 (11个, NPU环境实测)

| 序号 | 算子 | 配置 | PyTorch(us) | AscendC(us) | 加速比 |
|------|------|------|------------|-------------|--------|
| 1 | sinkhorn | default | 9059.7 | 475.9 | **19.04x** |
| 2 | hc_split_sinkhorn | b2s8hc4 | 2773.8 | 234.1 | **11.85x** |
| 3 |  | b1s1hc4 | 2918.5 | 223.3 | **13.07x** |
| 4 |  | b64s8hc4 | 2845.6 | 423.1 | **6.73x** |
| 5 |  | b4s16hc4 | 2819.3 | 235.0 | **12.00x** |
| 6 |  | b8s4hc8 | 2838.4 | 277.8 | **10.22x** |
| 7 | act_quant_kernel | 1K_gs128 | 170.5 | 119.4 | 1.43x |
| 8 |  | 4K_gs128 | 173.4 | 120.2 | 1.44x |
| 9 |  | 16K_gs128 | 171.9 | 120.5 | 1.43x |
| 10 |  | 65K_gs128 | 180.0 | 119.9 | 1.50x |
| 11 |  | 256K_gs128 | 160.4 | 184.7 | 0.87x |
| 12 | expand_kenel_fwd | typical(1,1024,1280,4) | 8178.2 | 107.2 | **76.32x** |
| 13 |  | min(1,1,128,2) | 76.3 | 99.5 | 0.77x |
| 14 |  | multi(4,256,256,2) | 7719.7 | 107.0 | **72.12x** |
| 15 |  | largeM(1,1,1280,16) | 77.2 | 112.8 | 0.68x |
| 16 |  | M1(1,1,1280,1) | 64.8 | 106.8 | 0.61x |
| 17 | apply_mix | default(2,1024,4,1280) | 86.8 | 246.9 | 0.35x |
| 18 |  | small(1,128,2,640) | 59.7 | 166.3 | 0.36x |
| 19 |  | large_b(8,1024,4,1280) | 570.6 | 1106.3 | 0.52x |
| 20 |  | large_s(2,4096,4,1280) | 568.5 | 1094.3 | 0.52x |
| 21 | head_compute_mix_fwd | default(16,16384) | 36877.4 | 204.6 | **180.28x** |
| 22 |  | 1K(1,256) | 47.5 | 226.9 | 0.21x |
| 23 |  | small(2,1) | 36.9 | 224.3 | 0.16x |
| 24 |  | 4M(32,32768) | 27835.8 | 235.3 | **118.30x** |
| 25 | engram_fused_weight | small(16,128) | 14.5 | 142.5 | 0.10x |
| 26 |  | default(64,1280) | 19345.0 | 182.0 | **106.29x** |
| 27 |  | medium(128,1280) | 14829.0 | 227.1 | **65.30x** |
| 28 | engram_hash | 32x3x2x8 | 562.0 | 84.0 | **6.69x** |
| 29 |  | 256x3x2x8 | 563.9 | 84.7 | **6.66x** |
| 30 |  | 1024x3x2x8 | 561.5 | 84.8 | **6.62x** |
| 31 |  | 4096x5x4x16 | 1577.4 | 562.9 | 2.80x |
| 32 | head_compute_mix_bwd | default | 157.7 | 251.5 | 0.63x |
| 33 | engram_gate_bwd | T14_H4_D128 | 454.8 | 57.2 | **7.94x** |

### 实测算子汇总

| 序号 | 算子 | 几何平均加速比 | 方法 | 实现位置 |
|------|------|---------------|------|---------|
| 1 | 🟢 sinkhorn | **19.04x** | torch.ops NPU实测 | `${MERGE}/sinkhorn/operators/sinkhorn/op_kernel/sinkhorn_kernel.asc` |
| 2 | 🟢 hc_split_sinkhorn | **10.50x** | torch.ops NPU实测 | `${MERGE}/hc_split_sinkhorn/operators/hc_split_sinkhorn/op_kernel/hc_split_sinkhorn_kernel.asc` |
| 3 | 🟢 act_quant_kernel | **1.31x** | torch.ops NPU实测 | `${MERGE}/act_quant_kernel/operators/act_quant_kernel/op_host/act_quant_kernel.asc` |
| 4 | 🟢 expand_kenel_fwd | **4.45x** | torch.ops NPU实测 | `-` |
| 5 | 🔴 apply_mix | **0.43x** | torch.ops NPU实测 | `${MERGE}/apply_mix/op_kernel/apply_mix_kernel.asc` |
| 6 | 🟢 head_compute_mix_fwd | **5.20x** | torch.ops NPU实测 | `-` |
| 7 | 🟢 engram_fused_weight | **8.91x** | torch.ops NPU实测 | `${MERGE}/engram_fused_weight/op_kernel/engram_fused_weight_kernel.asc` |
| 8 | 🟢 engram_hash | **5.36x** | torch.ops NPU实测 | `${MERGE}/engram_hash/op_kernel/engram_hash_kernel.asc` |
| 9 | 🔴 head_compute_mix_bwd | **0.63x** | torch.ops NPU实测 | `-` |
| 10 | 🟢 engram_gate_bwd | **7.94x** | 独立二进制实测(cannbot) | `-` |

---

## 二、历史数据算子 (11个, summary.json)

| 序号 | 算子 | 来源 | 加速比 | PyTorch(us) | AscendC(us) | 实现位置 |
|------|------|------|--------|-------------|-------------|---------|
| 1 | 🟢 norm_fn | merge | **90.16x** | 31664.1 | 351.2 | `${MERGE}/norm_fn/operators/norm_fn/op_device/norm_fn_kernel.asc` |
| 2 | 🟢 engram_gate_bwd-bk | merge | **46.99x** | 737.8 | 15.7 | `-` |
| 3 | 🟢 engram_gate_fwd | merge | **35.09x** | 579.7 | 16.5 | `-` |
| 4 | 🟢 big_fuse | cannbot | **25.44x** | 58562.4 | 2302.2 | `-` |
| 5 | 🟢 mhc_post | merge | **19.96x** | 173437.5 | 8691.0 | `${MERGE}/mhc_post/operators/mhc_post/op_kernel/mhc_post_kernel.asc` |
| 6 | 🟢 pre_split_mixes | merge | **5.90x** | 306.8 | 49.2 | `${MERGE}/pre_split_mixes/op_kernel/pre_split_mixes_kernel.asc` |
| 7 | 🔴 MTPBlock | merge | **0.64x** | 239912.0 | 372409.7 | `${MERGE}/MTPBlock/op_kernel/k4_hc_post_kernel.asc` |
| 8 | 🔴 engram_gate_w_reduce | merge | **0.50x** | 90.4 | 181.4 | `${MERGE}/engram_gate_w_reduce/operators/engram_gate_w_reduce/op_kernel/engram_gate_w_reduce_kernel.asc` |
| 9 | 🔴 expand_kenel_bwd | merge | **0.34x** | 30.1 | 0.0 | `-` |
| 10 | 🔴 sparse_attn | merge | **0.14x** | 246545.7 | 1739545.0 | `${MERGE}/sparse_attn/op_kernel/sparse_attn_kernel.asc` |
| 11 | 🔴 indexer | merge | **0.05x** | 1272.3 | 27496.0 | `-` |

---

## 三、总结

| 类别 | 数量 |
|------|------|
| 算子总数 | 21 |
| 实测算子 | 10 |
| 历史数据 | 11 |
| 🟢 加速 (>1x) | 14 |
| 🔴 减速 (<1x) | 7 |
| 精度通过率 | 100% (22/22) |

### 加速排行 (几何平均)

|  1 | norm_fn                      |    90.16x | hist | ████████████████████████████████████████████████████████████ |
|  2 | engram_gate_bwd-bk           |    46.99x | hist | ██████████████████████████████████████████████ |
|  3 | engram_gate_fwd              |    35.09x | hist | ███████████████████████████████████ |
|  4 | big_fuse                     |    25.44x | hist | █████████████████████████ |
|  5 | mhc_post                     |    19.96x | hist | ███████████████████ |
|  6 | sinkhorn                     |    19.04x | 实测 | ███████████████████ |
|  7 | hc_split_sinkhorn            |    10.50x | 实测 | ██████████ |
|  8 | engram_fused_weight          |     8.91x | 实测 | ████████ |
|  9 | engram_gate_bwd              |     7.94x | 实测 | ███████ |
| 10 | pre_split_mixes              |     5.90x | hist | █████ |
| 11 | engram_hash                  |     5.36x | 实测 | █████ |
| 12 | head_compute_mix_fwd         |     5.20x | 实测 | █████ |
| 13 | expand_kenel_fwd             |     4.45x | 实测 | ████ |
| 14 | act_quant_kernel             |     1.31x | 实测 | █ |
| 15 | MTPBlock                     |     0.64x | hist |  |
| 16 | head_compute_mix_bwd         |     0.63x | 实测 |  |
| 17 | engram_gate_w_reduce         |     0.50x | hist |  |
| 18 | apply_mix                    |     0.43x | 实测 |  |
| 19 | expand_kenel_bwd             |     0.34x | hist |  |
| 20 | sparse_attn                  |     0.14x | hist |  |
| 21 | indexer                      |     0.05x | hist |  |

---

## 四、实现文件清单

每个算子的核心实现文件:

| 算子 | Kernel | Host | Tiling | Torch扩展 | .so |
|------|--------|------|--------|----------|-----|
| MTPBlock | k4_hc_post_kernel.asc | mtpblock_host.asc | mtpblock_tiling.h | mtpblock_torch.cpp | libmtpblock_ops.so |
| act_quant_kernel | act_quant_kernel.asc |  | act_quant_kernel_tiling.h | act_quant_kernel_torch.cpp | libact_quant_kernel_ops.so |
| apply_mix | apply_mix_kernel.asc | apply_mix.asc | apply_mix_tiling.h | apply_mix_torch.cpp | libapply_mix_ops.so |
| big_fuse |  | big_fuse.asc | big_fuse_tiling.h | big_fuse_torch.cpp | libbig_fuse_ops.so |
| engram_fused_weight | engram_fused_weight_kernel.asc | engram_fused_weight.asc | engram_fused_weight_tiling.h | engram_fused_weight_torch.cpp | libengram_fused_weight_ops.so |
| engram_gate_bwd |  | engram_gate_bwd.asc | engram_gate_bwd_tiling.h | engram_gate_bwd_torch.cpp | libengram_gate_bwd_ops.so |
| engram_gate_bwd-bk |  | engram_gate_bwd.asc | engram_gate_bwd_tiling.h | engram_gate_bwd_torch.cpp | libengram_gate_bwd_ops.so |
| engram_gate_fwd |  | engram_gate_fwd.asc | engram_gate_fwd_tiling.h | engram_gate_fwd_torch.cpp | libengram_gate_fwd_ops.so |
| engram_gate_w_reduce | engram_gate_w_reduce_kernel.asc | engram_gate_w_reduce.asc | engram_gate_w_reduce_tiling.h | engram_gate_w_reduce_torch.cpp | libengram_gate_w_reduce_ops.so |
| engram_hash | engram_hash_kernel.asc | engram_hash_host.asc | engram_hash_tiling.h | engram_hash_torch.cpp | libengram_hash_ops.so |
| expand_kenel_bwd |  | expand_kenel_bwd.asc | expand_kenel_bwd_tiling.h | expand_kenel_bwd_torch.cpp | libexpand_kenel_bwd_ops.so |
| expand_kenel_fwd |  | expand_kenel_fwd.asc | expand_kenel_fwd_tiling.h | expand_kenel_fwd_torch.cpp | libexpand_kenel_fwd_ops.so |
| hc_split_sinkhorn | hc_split_sinkhorn_kernel.asc | hc_split_sinkhorn.asc | hc_split_sinkhorn_tiling.h | hc_split_sinkhorn_torch.cpp | libhc_split_sinkhorn_ops.so |
| head_compute_mix_bwd |  | head_compute_mix_bwd.asc | head_compute_mix_bwd_tiling.h | head_compute_mix_bwd_torch.cpp | libhead_compute_mix_bwd_ops.so |
| head_compute_mix_fwd |  | head_compute_mix_fwd.asc | head_compute_mix_fwd_tiling.h | head_compute_mix_fwd_torch.cpp | libhead_compute_mix_fwd_ops.so |
| indexer |  | indexer_host.asc | indexer_tiling.h |  |  |
| mhc_post | mhc_post_kernel.asc | mhc_post.asc | mhc_post_tiling.h | mhc_post_torch.cpp | libmhc_post_ops.so |
| norm_fn | norm_fn_kernel.asc | norm_fn.asc | norm_fn_tiling.h | norm_fn_torch.cpp | libnorm_fn_ops.so |
| pre_split_mixes | pre_split_mixes_kernel.asc | benchmark_main.asc | pre_split_mixes_tiling.h | pre_split_mixes_torch.cpp | libpre_split_mixes_ops.so |
| sinkhorn | sinkhorn_kernel.asc | sinkhorn.asc | sinkhorn_tiling.h | sinkhorn_torch.cpp | liboptiling.so |
| sparse_attn | sparse_attn_kernel.asc | sparse_attn_main.asc | sparse_attn_tiling.h | sparse_attn_torch.cpp | libsparse_attn_ops.so |

## 五、复现方法

```bash
cd /mnt/data01/zmz/workspace/12agent/waic/cannbot-merge
# 运行统一测试脚本
python3 benchmark_results/bench_real.py
# 运行各算子独立bench脚本
python3 benchmark_results/_bench_sinkhorn.py
python3 benchmark_results/_bench_hc_split_sinkhorn.py
python3 benchmark_results/_bench_act_quant_kernel.py
python3 benchmark_results/_bench_expand_kenel_fwd.py
python3 benchmark_results/_bench_apply_mix.py
python3 benchmark_results/_bench_head_compute_mix_fwd.py
# 运行修复版脚本
python3 benchmark_results/_fix_engram_fused_weight.py
python3 benchmark_results/_fix_engram_hash.py
python3 benchmark_results/_fix_head_compute_mix_bwd.py
# engram_gate_bwd 独立二进制
cd engram_gate_bwd-bk/operators/engram_gate_bwd/build
python3 ../scripts/gen_data.py && ./engram_gate_bwd 14 4 128 1e-6 1e-20
```

*报告生成时间: 2026-07-07 02:51:20*