# G000 fused hook — extracted from
#   vllm/model_executor/layers/sparse_attn_indexer.py
# Functions: _load_fuse_opt_64k, _try_fuse_opt_64k_topk
#
# Serving switch: VLLM_INDEXER_BACKEND=fuse_opt_64k
# Kernel tree: this directory (fuse_opt_64k_indexer.py → launch_fused.cu)

def _try_fuse_opt_64k_topk(...) -> bool:
    """Fused SM90 score + topk + physical gather. True if it wrote topk_indices."""
    if _indexer_backend() != "fuse_opt_64k":
        return False
    mod = _load_fuse_opt_64k()  # import fuse_opt_64k_indexer from dsa_opt/optimized
    if not mod.can_use_fuse_opt_64k(...):
        return False  # fall back to DeepGEMM (before/g000)
    pack_scores, pack_indices, logical, topk_workspace = workspace.get_simultaneous(...)
    mod.fuse_opt_64k_topk_indexer(
        q, kv_cache, weights, seq_lens, block_table, schedule_metadata,
        topk_indices, topk_tokens,
        pack_scores=pack_scores, pack_indices=pack_indices,
        logical=logical, topk_workspace=topk_workspace,
        max_seq_len=max_seq_len,
    )
    return True
    # caller then: set_indexer_topk_physical(True)  → remap is identity / skipped
