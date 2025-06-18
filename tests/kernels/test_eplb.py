import torch
import triton
import triton.language as tl

from dlblas.utils.utils import infer_device


@triton.jit
def kernel_map_logic_to_physical_hash(topk_idx_ptr, physical_idx_ptr, log2phy_ptr, logcnt_ptr, seed, num_tokens,
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
                    replica_id = rand_val % cnt.to(tl.uint32)
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
    kernel_map_logic_to_physical_hash[grid](topk_idx,
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


def reference_map_logic_to_physical_idx_hash_random(topk_idx: torch.Tensor,
                                                    log2phy_data: torch.Tensor,
                                                    logcnt_data: torch.Tensor,
                                                    seed: int = 12345) -> torch.Tensor:

    def simple_hash32_local(x: int) -> int:
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x &= 0xFFFFFFFF
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x &= 0xFFFFFFFF
        x = (x >> 16) ^ x
        x &= 0xFFFFFFFF
        return x & 0x7FFFFFFF

    num_tokens, num_topk = topk_idx.shape
    physical_idx = torch.full_like(topk_idx, -1, dtype=torch.int32)
    for t in range(num_tokens):
        for k in range(num_topk):
            logic_exp = int(topk_idx[t, k])
            if logic_exp < 0:
                continue
            cnt = int(logcnt_data[logic_exp])
            if cnt <= 0:
                continue
            combined = (int(t) << 16) ^ (int(k) << 8) ^ int(seed)
            rand_val = simple_hash32_local(combined & 0xFFFFFFFF)
            replica_id = rand_val % cnt
            phy_id = int(log2phy_data[logic_exp, replica_id])
            if phy_id >= 0:
                physical_idx[t, k] = phy_id
    return physical_idx


def test_hash_random():
    topk_idx_cpu = torch.tensor([[1, 2], [2, 2], [0, 1], [1, 3]], dtype=torch.int32)
    log2phy_cpu = torch.tensor([
        [10, 11],
        [12, 13],
        [14, -1],
        [15, 16],
    ], dtype=torch.int32)
    logcnt_cpu = torch.tensor([2, 2, 1, 2], dtype=torch.int32)
    seed_val = 99999
    device = torch.device(infer_device())
    topk_idx_gpu = topk_idx_cpu.to(device)
    log2phy_gpu = log2phy_cpu.to(device)
    logcnt_gpu = logcnt_cpu.to(device)
    physical_idx_gpu = map_logic_to_physical_idx_hash_random(topk_idx_gpu, log2phy_gpu, logcnt_gpu, seed=seed_val)
    physical_idx_cpu_from_triton = physical_idx_gpu.cpu()
    golden_idx_cpu = reference_map_logic_to_physical_idx_hash_random(topk_idx_cpu,
                                                                     log2phy_cpu,
                                                                     logcnt_cpu,
                                                                     seed=seed_val)
    is_all_equal = bool(torch.all(golden_idx_cpu.eq(physical_idx_cpu_from_triton)))
    print('[HashRandom TEST] Golden Model =>\n', golden_idx_cpu)
    print('[HashRandom TEST] Triton Kernel =>\n', physical_idx_cpu_from_triton)
    print('[HashRandom TEST] match = ', is_all_equal)
    if not is_all_equal:
        raise RuntimeError('HashRandom test failed: Triton result != Python golden model.')


if __name__ == '__main__':
    test_hash_random()
