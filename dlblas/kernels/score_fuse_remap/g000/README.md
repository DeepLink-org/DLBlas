# G000 after：fuse_opt_64k

同一套 TMA / WGMMA / scheduler，只改 epilogue：写成 pack `(score, physical_id)`。
topk 仍独立（KEEP_SEPARATE）。MLA remap SKIP。

## 文件

| 文件 | 作用 |
|---|---|
| `sm90_fp8_paged_mqa_logits_fused.cuh` | fused score kernel，搜 `pack_scores` |
| `launch_fused.cu` | score/topk launch、gather、TMA |
| `merge_cta_topk.cuh` | ≤8192 CUB merge；`gather_physical_kernel` |
| `fuse_opt_64k_indexer.py` | serving hook：小路径 CUB，大路径 pack→topk→gather |
| `sparse_attn_indexer_hook.py` | vLLM `_try_fuse_opt_64k_topk` 摘录 |

`launch_fused.cu` 入口：`fused_paged_mqa_score`、
`fused_paged_mqa_topk`、`gather_physical`、TMA。

## 两条路径

- `max_parts ≤ 32`（容量 ≤8192）：`fused_paged_mqa_topk` → 物理 id
- `max_parts > 32`（到 64k）：score pack → `cooperative_topk` → `gather_physical`

约束：SM90，`next_n=1`，H=32，D=128，page=64，FP8，
无 DCP/PCP/prefill mix。topk ∈ {512,1024,2048}，容量 ≤65536。
否则回退 `before/g000`。
