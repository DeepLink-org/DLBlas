import os
import torch
import torch.distributed as dist
from datetime import datetime
from dlblas.utils.logger import get_logger
logger = get_logger(__name__)

class ExpertsDistributionRecorder:
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.dispatch_count = {}
        self.accum_token_counts = {}
        self.last_dump_minute = {}

    def record(self, topk_ids, layer_index, num_experts):
        key = f"{layer_index}_{num_experts}"
        if key not in self.dispatch_count:
            self.dispatch_count[key] = 0
        self.dispatch_count[key] += 1
        if key not in self.accum_token_counts:
            self.accum_token_counts[key] = torch.zeros(num_experts,
                                                         dtype=torch.int64,
                                                        device='cuda')
        topk_ids_flat = topk_ids.view(-1)
        step_local_counts = torch.bincount(topk_ids_flat, minlength=num_experts)
        self.accum_token_counts[key] += step_local_counts
        global_token_counts = self.accum_token_counts[key].clone()
        if dist.is_initialized():
            dist.all_reduce(global_token_counts, op=dist.ReduceOp.SUM)
        rank = dist.get_rank() if dist.is_initialized() else 0
        now = datetime.now()
        if rank == 0 and now.minute % 5 == 0 and now.minute != self.last_dump_minute.get(key, -1):
            self.last_dump_minute[key] = now.minute
            global_list = global_token_counts.cpu().tolist()
            step = self.dispatch_count[key]
            step_dir = f"{self.output_dir}/step{step}/"        
            os.makedirs(step_dir, exist_ok=True)
            token_counts_file_name = f"rank{rank}_layer{layer_index}_experts_counts.json"
            filepath = os.path.join(step_dir, token_counts_file_name)
            with open(filepath, 'w') as f:
                import json
                json.dump(global_list, f, indent=2)
            logger.info(f"[EPLB]{token_counts_file_name} dumped to {step_dir}")

