# G001 fused path — extracted from installed vLLM 0.27.1
# Sources:
#   vllm/model_executor/models/deepseek_v2.py          Indexer.forward
#   vllm/model_executor/layers/sparse_attn_indexer.py  fuse_indexer_k_store
#   vllm/models/deepseek_v32/common/kernels.py         fused_indexer_k_store
#
# Chain (one custom op, same as G000):
#   wk_weights_proj(hidden) → raw k  (GEMM stays KEEP_SEPARATE)
#   sparse_attn_indexer(..., store_positions, store_rope_cos_sin, store_k_norm_*)
#     fused_indexer_k_store: LayerNorm + RoPE + UE8M0 + cache write   # pid 0 only
#   skip standalone k_norm / k_rope / indexer_k_quant_and_cache

def indexer_k_fused(self, hidden_states, q_fp8, k_raw, weights, positions, rotary_emb):
    self.indexer_op.skip_k_cache_insert = True
    self.indexer_op.fuse_indexer_k_store = True
    return self.indexer_op(
        hidden_states,
        q_fp8,
        k_raw,  # raw GEMM output, BEFORE LayerNorm
        weights,
        store_positions=positions,
        store_rope_cos_sin=rotary_emb.cos_sin_cache,
        store_k_norm_w=self.k_norm.weight,
        store_k_norm_b=self.k_norm.bias,
    )


# inside sparse_attn_indexer when fuse_indexer_k_store:
from vllm.models.deepseek_v32.common.kernels import fused_indexer_k_store

fused_indexer_k_store(
    store_positions[:num_tokens],
    k,                    # raw K
    store_k_norm_w,
    store_k_norm_b,
    store_k_norm_eps,
    store_rope_cos_sin,
    slot_mapping[:num_tokens],
    kv_cache,
    index_rope_interleave=store_rope_interleave,
)
