# AscendC Kernel Package for Indexer Operator
# 4-Kernel design targeting Ascend910B2 via Triton-Ascend backend

from .kernel_q_proj import q_projection
from .kernel_w_proj import weights_projection
from .kernel_rope_score import rope_score_compute
from .kernel_post_topk import postprocess_topk
from .indexer_launcher import IndexerLauncher

__all__ = [
    "q_projection",
    "weights_projection",
    "rope_score_compute",
    "postprocess_topk",
    "IndexerLauncher",
]
