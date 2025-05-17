import torch
import triton
import triton.language as tl


@triton.jit
def map_logic_to_physical_hash_kernel(topk_idx_ptr, physical_idx_ptr, log2phy_ptr, logcnt_ptr, seed, num_tokens,
                                      num_topk, num_logical_experts, max_replica, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start_t = pid * BLOCK
    end_t = tl.minimum(start_t + BLOCK, num_tokens)
    for t in range(start_t, end_t):
        row_off = t * num_topk
        for k in range(num_topk):
            logic_exp = tl.load(topk_idx_ptr + row_off + k)
            # 条件判断不能用 continue，只能嵌套写
            if logic_exp >= 0:
                cnt = tl.load(logcnt_ptr + logic_exp)
                if cnt > 0:
                    # 构造 hash 值
                    combined = ((t << 16) ^ (k << 8) ^ seed) & 0xFFFFFFFF
                    x = combined
                    x = ((x >> 16) ^ x) * 0x45d9f3b
                    x = ((x >> 16) ^ x) * 0x45d9f3b
                    x = (x >> 16) ^ x
                    rand_val = x & 0x7fffffff

                    replica_id = rand_val % cnt
                    phy_id_addr = log2phy_ptr + logic_exp * max_replica + replica_id
                    phy_id = tl.load(phy_id_addr)
                    tl.store(physical_idx_ptr + row_off + k, phy_id)
                else:
                    tl.store(physical_idx_ptr + row_off + k, -1)
            else:
                tl.store(physical_idx_ptr + row_off + k, -1)


def map_logic_to_physical_idx_hash_random(topk_idx: torch.Tensor,
                                          log2phy: torch.Tensor,
                                          logcnt: torch.Tensor,
                                          seed: int = 12345,
                                          block_size: int = 128) -> torch.Tensor:
    num_tokens, num_topk = topk_idx.shape
    physical_idx = torch.empty_like(topk_idx, dtype=torch.int32, device=topk_idx.device)
    grid = ((num_tokens + block_size - 1) // block_size, )
    map_logic_to_physical_hash_kernel[grid](topk_idx,
                                            physical_idx,
                                            log2phy,
                                            logcnt,
                                            seed,
                                            num_tokens,
                                            num_topk,
                                            log2phy.shape[0],
                                            log2phy.shape[1],
                                            BLOCK=block_size)
    return physical_idx
