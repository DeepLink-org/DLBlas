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

## 精度与性能

H200。`fused_indexer_k_store` 对 LayerNorm + RoPE + UE8M0 参考。
cache 按 serving 分页：每 page 64，先 packed fp8 再 scale。
48/48 通过：fp8 ULP = 0，scale bit-exact。

| 组 | 设定 | RoPE | ULP | scale | 结果 |
| --- | --- | --- | ---: | --- | --- |
| dense | 1–1024, 13 sizes | GLM/NeoX | 0 | exact | 26/26 |
| scatter | 32, 128, 256 | GLM/NeoX | 0 | exact | 6/6 |
| wrap pos | 32, 128 | GLM/NeoX | 0 | exact | 4/4 |
| tiny/large | 16, 64 | GLM/NeoX | 0 | exact | 8/8 |
| padding | trailing/mixed/all -1 | GLM/NeoX | 0 | exact | 4/4 |

微基准对 eager PyTorch LN+RoPE+quant（`run_fused_indexer_k_store.py`），
CUDA event 中位数。不是 `indexer_k_quant_and_cache` CUDA。

| tok | GLM ms | GLM ref | GLM | NeoX ms | NeoX ref | NeoX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.043 | 0.205 | 4.80× | 0.041 | 0.200 | 4.83× |
| 4 | 0.040 | 0.208 | 5.18× | 0.041 | 0.197 | 4.79× |
| 16 | 0.040 | 0.182 | 4.58× | 0.039 | 0.184 | 4.77× |
| 32 | 0.040 | 0.198 | 4.99× | 0.040 | 0.199 | 4.99× |
| 64 | 0.041 | 0.197 | 4.78× | 0.040 | 0.180 | 4.53× |
| 128 | 0.037 | 0.183 | 4.98× | 0.036 | 0.183 | 5.11× |
| 256 | 0.038 | 0.197 | 5.19× | 0.042 | 0.204 | 4.84× |
| 512 | 0.041 | 0.209 | 5.17× | 0.042 | 0.221 | 5.24× |
| 1024 | 0.040 | 0.222 | 5.60× | 0.044 | 0.208 | 4.73× |
| 2048 | 0.040 | 0.203 | 5.10× | 0.043 | 0.205 | 4.84× |
| 4096 | 0.041 | 0.196 | 4.84× | 0.041 | 0.196 | 4.80× |
| 8192 | 0.046 | 0.181 | 3.91× | 0.046 | 0.196 | 4.28× |

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_fused_indexer_k_store.py
```
