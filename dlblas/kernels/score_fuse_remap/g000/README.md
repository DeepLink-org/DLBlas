# G000 after：fuse_opt_64k

同一套 TMA / WGMMA / scheduler，只改 epilogue：写成 pack `(score, physical_id)`。
topk 仍独立（KEEP_SEPARATE）。MLA remap SKIP。

## 文件

| 文件 | 作用 |
|---|---|
| `sm90_fp8_paged_mqa_logits_fused.cuh` | fused score kernel，搜 `pack_scores` |
| `launch_fused.cu` | score/topk launch、gather、TMA |
| `merge_cta_topk.cuh` | ≤8192 CUB merge；`gather_physical_kernel` |
| `cta_bitonic_select.cuh` | CUB merge 用的 `kNegInf` / bitonic helpers |
| `fuse_opt_64k_indexer.py` | serving hook：小路径 CUB，大路径 pack→topk→gather |
| `sparse_attn_indexer_hook.py` | vLLM `_try_fuse_opt_64k_topk` 摘录 |
| `run_fuse_opt_64k.py` | 独立 runner：精度 + 微基准 |
| `test_fuse_opt_64k.py` | pytest（对 DeepGEMM original） |
| `g000_fuse_opt_64k_results.json` | H200 精度 + 微基准 |

`launch_fused.cu` 入口：`fused_paged_mqa_score`、
`fused_paged_mqa_topk`、`gather_physical`、TMA。

## 两条路径

- `max_parts ≤ 32`（容量 ≤8192）：`fused_paged_mqa_topk` → 物理 id
- `max_parts > 32`（到 64k）：score pack → `cooperative_topk` → `gather_physical`

约束：SM90，`next_n=1`，H=32，D=128，page=64，FP8，
无 DCP/PCP/prefill mix。topk ∈ {512,1024,2048}，容量 ≤65536。
否则回退 `before/g000`。

## 精度与性能

H200。`fuse_opt_64k` 对 DeepGEMM score + vLLM topk + remap。
物理 id 集合召回；`opt/orig` 应接近 1.0（同 WGMMA logits）。
`orig/ref` 允许少量 FP8 vs FP32 边界差。physical → logical 经 `block_table` 可逆。

6/6 通过：CUB 与 pack→topk→gather 两条路径都覆盖。乱序 page table。

| case | path | orig/ref | opt/orig | invert | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| B4 N512 K512 | cub | 1.0000 | 1.0000 | 0 | pass |
| B4 N1024 K512 ragged | cub | 1.0000 | 1.0000 | 0 | pass |
| B2 N4096 K2048 | cub | 1.0000 | 1.0000 | 0 | pass |
| B4 N4096 K2048 ragged | cub | 0.9999 | 1.0000 | 0 | pass |
| B1 N8192 K2048 | cub | 1.0000 | 1.0000 | 0 | pass |
| B1 N16384 K2048 | pack_topk_gather | 1.0000 | 1.0000 | 0 | pass |

微基准对 DeepGEMM e2e（`run_fuse_opt_64k.py`），不是 serving tok/s。
`score` / `topk` / `remap` 是 original 拆分。

| case | path | orig e2e (ms) | fused (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| B1 N4096 K2048 | cub | 0.133 | 0.032 | 4.09× |
| B8 N4096 K2048 | cub | 0.120 | 0.033 | 3.65× |
| B8 N8192 K2048 | cub | 0.122 | 0.050 | 2.46× |
| B32 N8192 K2048 | cub | 0.123 | 0.054 | 2.27× |
| B1 N16384 K2048 | pack_topk_gather | 0.119 | 0.031 | 3.84× |

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_fuse_opt_64k.py
```
