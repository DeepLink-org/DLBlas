# G001 after：fused_indexer_k_store

把 `LayerNorm + RoPE + UE8M0 + cache write` 收进 `_fused_norm_rope_kernel` 的 pid 0。
Q-RMS / MLA / topk-fill 的 pid 不 launch。
挂在 `sparse_attn_indexer` 里，和 G000 同一个 custom op。

## 文件

| 文件 | 作用 |
|---|---|
| `fused_indexer_k_store.py` | **提交看这个**：helpers + kernel pid 0 + launcher |
| `kernels.py` | 完整文件（含 Q/MLA 其它 pid） |
| `g001_fused_indexer_k_store_results.json` | H200 精度 + 微基准 |
| `indexer_forward_fused.py` | Indexer.forward / sparse_attn_indexer 调用摘录 |
| `test_fused_indexer_k_store.py` | 精度 + microbench（对 unfused PyTorch 参考） |
| `run_fused_indexer_k_store.py` | 独立 runner |

## kernel 里对应 before 的三步

`_fused_norm_rope_kernel` `pid == 0`：

1. LayerNorm：token 维 mean/rstd，仿射 `w,b`
2. RoPE：`INDEX_ROPE_INTERLEAVE` 走 GLM 相邻对，否则 NeoX 半维对。
   partner 用同一组 (mean, rstd) 再算，不写 scratch
3. `_fp8_quant_and_cache_write`：UE8M0 scale + e4m3，写入 indexer cache

## Fallback（不半融合）

PCP / DCP、空 cache、非 CUDA、FP4 cache → 走 `before/g001` 三连。
`skip_k_cache_insert=True` 与 `fuse_indexer_k_store=True` 一起设，
避免 fused store 之后再跑一遍 `indexer_k_quant_and_cache`。
