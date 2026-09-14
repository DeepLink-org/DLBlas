# after：fuse 之后的算子

本目录是 G000 / G001 融合后的算子。

## G000：`fuse_opt_64k`

```text
同一套 TMA + WGMMA
    │
    ▼
sm90_fp8_paged_mqa_logits_fused
    写 pack_score + pack_id（物理 slot）
    │
    ├─ table ≤ 8192  → merge_cta_topk（CUB）直接物理 id
    └─ table ≤ 65536 → cooperative_topk(pack) → gather_physical
    │
    ▼
set_indexer_topk_physical(True)   remap SKIP
```

开关：`export VLLM_INDEXER_BACKEND=fuse_opt_64k`。
topk 仍是独立 kernel（KEEP_SEPARATE），没有塞进 WGMMA。

文件在 `g000/`。

## G001：`fused_indexer_k_store`

```text
raw k（GEMM 输出）
    │
    ▼
_fused_norm_rope_kernel[(1, num_tokens)]   只 launch pid 0
    LayerNorm + RoPE + UE8M0 + cache write
    │
    ▼
indexer K cache
```

不再 launch `k_norm` / `rotary_embedding` / `indexer_k_quant_and_cache`。
挂在同一个 `sparse_attn_indexer` custom op 里，避免 decode 热路径再加一个 op。

文件在 `g001/`。

## 精度与性能

### G000

`g000/g000_fuse_opt_64k_results.json`。H200。
`fuse_opt_64k` 对 DeepGEMM score + vLLM topk + remap。
物理 id 集合召回；`opt/orig` = 1.0。6/6 通过。
CUB（≤8192）和 pack→topk→gather（16k）都覆盖。

| case | path | orig/ref | opt/orig | 结果 |
| --- | --- | ---: | ---: | --- |
| B4 N512 K512 | cub | 1.0000 | 1.0000 | pass |
| B4 N1024 K512 ragged | cub | 1.0000 | 1.0000 | pass |
| B2 N4096 K2048 | cub | 1.0000 | 1.0000 | pass |
| B4 N4096 K2048 ragged | cub | 0.9999 | 1.0000 | pass |
| B1 N8192 K2048 | cub | 1.0000 | 1.0000 | pass |
| B1 N16384 K2048 | pack_topk_gather | 1.0000 | 1.0000 | pass |

微基准对 DeepGEMM e2e（`g000/run_fuse_opt_64k.py`）。

| case | path | orig e2e (ms) | fused (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| B1 N4096 K2048 | cub | 0.133 | 0.032 | 4.09× |
| B8 N4096 K2048 | cub | 0.120 | 0.033 | 3.65× |
| B8 N8192 K2048 | cub | 0.122 | 0.050 | 2.46× |
| B32 N8192 K2048 | cub | 0.123 | 0.054 | 2.27× |
| B1 N16384 K2048 | pack_topk_gather | 0.119 | 0.031 | 3.84× |

### G001

`g001/g001_fused_indexer_k_store_results.json`。
H200。`fused_indexer_k_store` 对 LayerNorm + RoPE + UE8M0 参考。
48/48 通过：fp8 ULP = 0，scale bit-exact。
覆盖 dense / scatter slot / wrap position / tiny·large / padding。

| 组 | 覆盖 | 结果 |
| --- | --- | --- |
| dense | tokens 1…1024，GLM + NeoX | 26/26 pass |
| scatter | 32 / 128 / 256 | 6/6 pass |
| wrap pos | 32 / 128 | 4/4 pass |
| tiny / large | 16 / 64 | 8/8 pass |
| padding | trailing / mixed / all `slot=-1` | 4/4 pass |

微基准对 eager PyTorch LN+RoPE+quant（`g001/run_fused_indexer_k_store.py`），
GLM interleaved：

| tokens | fused (ms) | unfused ref (ms) | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.043 | 0.205 | 4.80× |
| 128 | 0.037 | 0.183 | 4.98× |
| 1024 | 0.040 | 0.222 | 5.60× |
| 4096 | 0.041 | 0.196 | 4.84× |
| 8192 | 0.046 | 0.181 | 3.91× |

NeoX 同档 4.3–5.2×，完整表见 `g001/README.md`。
