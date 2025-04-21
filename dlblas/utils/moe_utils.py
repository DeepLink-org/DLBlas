
from collections import defaultdict

# 模块级别的单例
global_tokens_per_expert = defaultdict(list)
# print(f"global_tokens_per_expert id in utils: {id(global_tokens_per_expert)}")

def save_expert_stats_to_file(rank: int, filepath: str = None):
    # print(f"global_tokens_per_expert id in utils: {id(global_tokens_per_expert)}")
    import os, json
    if filepath is None:
        filepath = f"/tmp/expert_load_rank{rank}.json"
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    stats_summary = {}
    for layer, stats in global_tokens_per_expert.items():
        if stats:
            total = torch.stack(stats).sum(dim=0).tolist()
            stats_summary[f"layer_{layer}"] = total

    with open(filepath, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"[Rank {rank}] Expert stats saved to {filepath}")
