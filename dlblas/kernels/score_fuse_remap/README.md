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

本目录内有实测结果的只有 G001：
`g001/g001_fused_indexer_k_store_results.json`。
H200。`fused_indexer_k_store` 对 LayerNorm + RoPE + UE8M0 参考。
10/10 通过：fp8 ULP = 0，scale bit-exact；
`slot=-1` 的 padding 不写 cache。

| tokens | RoPE | fp8 ULP | scale | 结果 |
| ---: | --- | ---: | --- | --- |
| 1 / 4 / 17 / 128 / 512 | GLM interleaved | 0 | exact | pass |
| 1 / 4 / 17 / 128 / 512 | DeepSeek NeoX | 0 | exact | pass |

微基准对 eager PyTorch LN+RoPE+quant
（`g001/test_fused_indexer_k_store.py`），
不是 `indexer_k_quant_and_cache` CUDA。

| tokens | fused (ms) | unfused ref (ms) | speedup |
| ---: | ---: | ---: | ---: |
| 128 | 0.040 | 0.178 | 4.43× |
| 1024 | 0.039 | 0.186 | 4.79× |
| 4096 | 0.043 | 0.172 | 3.98× |
