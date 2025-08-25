import torch
import triton
import triton.language as tl
# permute_fwd, permute_bwd, unpermute_fwd, unpermute_bwd

#1. permute_fwd (Scatter)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_ROWS': 512, 'BLOCK_SIZE_COLS': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_SIZE_ROWS': 128, 'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_SIZE_ROWS': 64, 'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=3),
    ],
    key=['num_cols', 'num_topK', 'N_ELEMENTS'],
)
@triton.jit
def permute_fused_gather_and_map_kernel(
    # Pointers
    input_ptr, sorted_row_id_ptr, output_ptr, row_id_map_ptr,
    # Dimensions
    num_cols, num_topK, num_tokens, num_out_tokens,
    num_negative_one_in_indices,
    # Meta
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    row_offset = pid_row * BLOCK_SIZE_ROWS
    row_indices = row_offset + tl.arange(0, BLOCK_SIZE_ROWS)
    row_mask = row_indices < N_ELEMENTS
    flat_idx = tl.load(sorted_row_id_ptr + row_indices, mask=row_mask, other=-1)
    valid_flat_mask = flat_idx >= 0
    token_ids = flat_idx // num_topK
    k_ids = flat_idx % num_topK

    map_idx = k_ids * num_tokens + token_ids  # [num_topK, num_tokens] layout

    target_idx = row_indices - num_negative_one_in_indices
    target_valid = (target_idx >= 0) & (target_idx < num_out_tokens)

    
    write_mask_map = valid_flat_mask & target_valid
    tl.store(row_id_map_ptr + map_idx, target_idx, mask=write_mask_map)

    src_ptrs_base = input_ptr + token_ids * num_cols
    dst_rows = target_idx  # Use shifted row index
    dst_ptrs_base = output_ptr + dst_rows * num_cols

    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        src_ptrs = src_ptrs_base[:, None] + col_offsets[None, :]
        load_mask = row_mask[:, None] & col_mask[None, :] & valid_flat_mask[:, None]
        data = tl.load(src_ptrs, mask=load_mask, other=0.0, cache_modifier=".ca")
        dst_ptrs = dst_ptrs_base[:, None] + col_offsets[None, :]
        store_mask = target_valid[:, None] & col_mask[None, :]
        tl.store(dst_ptrs, data, mask=store_mask, cache_modifier=".cs")
        
        
        
def moe_permute_topk_op_triton(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: list[torch.Tensor],
    max_expanded_token_num: int,
    num_negative_one_in_indices: int,
    permuted_output: torch.Tensor,
    row_id_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    
    num_tokens = input_tensor.size(0)
    num_cols = input_tensor.size(1)
    num_topK = indices.size(1)
    device = input_tensor.device
    dtype = input_tensor.dtype
    num_elements = num_tokens * num_topK

    int32_options = {'dtype': torch.int32, 'device': device}
    int64_options = {'dtype': torch.int64, 'device': device}

    if not workspace or workspace[0].numel() < max_expanded_token_num:
        workspace.clear()
        workspace.extend([
            torch.empty(max_expanded_token_num, **int32_options),  # sorted_values
            torch.empty(max_expanded_token_num, **int32_options),  # scratch
            torch.empty(max_expanded_token_num, **int64_options),  # sorted_indices
        ])

    # Reuse workspace
    sorted_values_output_view = workspace[0][:num_elements]
    sorted_indices_output_view = workspace[2][:num_elements]

    # Flatten and sort indices
    flat_indices = indices.flatten()
    torch.sort(flat_indices, out=(sorted_values_output_view, sorted_indices_output_view))
    sorted_row_id_result = sorted_indices_output_view

    # Auto-determine num_out_tokens if not provided
    if num_out_tokens <= 0:
        num_out_tokens = num_elements - num_negative_one_in_indices

    # Ensure output shapes match
    assert permuted_output.shape == (num_out_tokens, num_cols), "permuted_output shape mismatch"
    assert row_id_map.numel() == num_elements, "row_id_map size mismatch"
    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE_ROWS']),)


    permute_fused_gather_and_map_kernel[grid](
        input_tensor,
        sorted_row_id_result,
        permuted_output,
        row_id_map,
        num_cols,
        num_topK,
        num_tokens,
        num_out_tokens,
        num_negative_one_in_indices,
        N_ELEMENTS=num_elements,
    )

    return permuted_output, row_id_map, sorted_row_id_result, workspace
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=6),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=6),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_COLS': 1024}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_COLS': 2048}, num_warps=8, num_stages=6),
        triton.Config({'BLOCK_SIZE_COLS': 32}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_SIZE_COLS': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 8}, num_warps=4, num_stages=4),
    ],
    key=['num_cols', 'num_topK'],
)
@triton.jit
def permute_bwd_kernel(
    go_ptr, inv_idx_ptr, out_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    for col_block_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_offsets = col_block_start * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        acc = tl.zeros((BLOCK_SIZE_COLS,), dtype=out_ptr.dtype.element_ty)
        
        # 内部循环：累加来自 topK 个专家的梯度
        # tl.static_range 确保这个循环在编译时展开，没有运行时开销
        for k in tl.static_range(num_topK):
            # 加载需要 gather 的行索引
            perm_row = tl.load(inv_idx_ptr + pid_m * num_topK + k)
            is_valid = perm_row >= 0

            vals = tl.load(go_ptr + perm_row * num_cols + col_offsets, 
                           mask=col_mask & is_valid, 
                           other=0.0)
            acc += vals
        
        # 写回当前块的结果
        tl.store(out_ptr + pid_m * num_cols + col_offsets, acc, mask=col_mask)

@triton.jit
def build_inv_idx_kernel(
    sorted_row_id_ptr, # *i64, [num_permuted_rows]
    inv_idx_ptr,       # *i32, [num_rows, num_topK]
    num_permuted_rows,
    num_topK: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    if pid >= num_permuted_rows:
        return

    original_flat_index = tl.load(sorted_row_id_ptr + pid)
    target_token_id = original_flat_index // num_topK
    k_id = original_flat_index % num_topK
    value_to_write = pid.to(tl.int32)
    out_ptr = inv_idx_ptr + target_token_id * num_topK + k_id
    tl.store(out_ptr, value_to_write)


#2 permute_bwd (Gather)
def moe_permute_topk_bwd_op(permuted_act_grad, sorted_row_id, original_shape, num_topK):
    num_permuted_rows, num_cols = permuted_act_grad.shape
    num_rows = original_shape[0]
    device = permuted_act_grad.device
    inv_idx = torch.full((num_rows, num_topK), -1, dtype=torch.int32, device=device)
    grid = (num_permuted_rows,)
    build_inv_idx_kernel[grid](
        sorted_row_id.to(torch.int64), # 确保输入是 int64
        inv_idx,
        num_permuted_rows,
        num_topK=num_topK,
    )
    
    act_grad = torch.empty(original_shape, dtype=permuted_act_grad.dtype, device=device)

    grid = (num_rows,)
    permute_bwd_kernel[grid](
        permuted_act_grad, inv_idx, act_grad,
        num_rows, num_cols,
        num_topK=num_topK, 
    )
    return act_grad

@triton.autotune(
    configs=[
        # 为 BLOCK_SIZE_COLS 测试一系列的二次幂值
        triton.Config({'BLOCK_SIZE_COLS': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 1024}, num_warps=8, num_stages=3),
    ],
    key=['num_cols', 'num_topK'],
)


# unpermute_fwd
@triton.jit
def gather_kernel(
    input_ptr, output_ptr, row_id_map_ptr, prob_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr, HAS_PROB: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    # 目标精度，例如 tl.bfloat16 或 tl.float16
    TARGET_DTYPE = input_ptr.dtype.element_ty
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        accum = tl.zeros((BLOCK_SIZE_COLS,), dtype=TARGET_DTYPE)
        source_row_k0 = tl.load(row_id_map_ptr + source_token_id)
        is_valid_expert_k0 = source_row_k0 != -1
        if is_valid_expert_k0:
            load_mask_k0 = col_mask
            frag_k0 = tl.load(
                input_ptr + (source_row_k0 * num_cols + col_offsets),
                mask=load_mask_k0, other=0.0
            )
            if HAS_PROB:
                prob_k0 = tl.load(prob_ptr + source_token_id * num_topK + 0).to(TARGET_DTYPE)
                accum = (frag_k0 * prob_k0).to(TARGET_DTYPE)
            else:
                accum = frag_k0.to(TARGET_DTYPE)
        else:
            accum = tl.zeros((BLOCK_SIZE_COLS,), dtype=TARGET_DTYPE)
        for k in range(1, num_topK):
            source_row_k_gt_0 = tl.load(row_id_map_ptr + k * num_rows + source_token_id)
            is_valid_expert_k_gt_0 = source_row_k_gt_0 != -1
            if is_valid_expert_k_gt_0:
                load_mask_k_gt_0 = col_mask
                expert_frag = tl.load(
                    input_ptr + (source_row_k_gt_0 * num_cols + col_offsets),
                    mask=load_mask_k_gt_0, other=0.0
                )
                if HAS_PROB:
                    prob_k = tl.load(prob_ptr + source_token_id * num_topK + k).to(TARGET_DTYPE)
                    weighted = (expert_frag * prob_k).to(TARGET_DTYPE)
                    accum = (accum + weighted).to(TARGET_DTYPE)
                else:   
                    accum = (accum + expert_frag).to(TARGET_DTYPE)
        output_base_ptr = output_ptr + source_token_id * num_cols
        tl.store(output_base_ptr + col_offsets, accum, mask=col_mask)

def moe_recover_topk_op_triton(
    input_tensor: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor, # 可以是 None
    num_tokens: int,
    num_topK: int
) -> torch.Tensor:
    
    num_cols = input_tensor.size(1)
    dtype = input_tensor.dtype
    device = input_tensor.device

    unpermuted_output = torch.empty(
        (num_tokens, num_cols), dtype=dtype, device=device, requires_grad=False
    )
    moe_permute_topK_kernel_launcher_triton(
        FWD=False,  
        input_tensor=input_tensor,
        output_tensor=unpermuted_output,
        sorted_row_id=None, 
        row_id_map=row_id_map,
        prob=prob,
        num_rows=num_tokens,
        num_topK=num_topK,
        num_cols=num_cols,
        num_out_tokens=0, # recover/gather操作不需要此参数
        prob_grad=None,   # 非梯度计算
        input_fwd=None,   # 非梯度计算
    )
    return unpermuted_output
def moe_permute_topK_kernel_launcher_triton(
    FWD: bool,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    sorted_row_id: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor,
    num_rows: int,
    num_topK: int,
    num_cols: int,
    num_out_tokens: int,
    prob_grad: torch.Tensor = None,
    input_fwd: torch.Tensor = None,
):
    grid = (num_rows,)
    BLOCK_SIZE_COLS = 64
    if FWD:
        if prob_grad is None:
            # 路径 1: permute_fwd (Scatter)
            # print("正在执行 C++ 逻辑: permute_fwd (Scatter)")
            n_elements = sorted_row_id.numel()
            map_grid = (triton.cdiv(n_elements, 1024),)
            row_map_kernel[map_grid](
                sorted_row_id, row_id_map, num_rows, num_topK, num_out_tokens, n_elements, BLOCK_SIZE=1024
            )
            scatter_fwd_kernel[grid](
                input_tensor, output_tensor, row_id_map, num_rows, num_cols, num_topK, BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
            )
        else:
            # 执行 unpermute_bwd 
            padded_top_K = triton.next_power_of_2(num_topK)
            scatter_bwd_kernel_optimized[grid](
                input_tensor, input_fwd, output_tensor, prob, prob_grad, row_id_map,
                num_rows, num_cols, num_topK=num_topK, PADDED_TOP_K=padded_top_K,
            )
           
    else: # not FWD
        if prob is None:
            # 路径 4: unpermute_bwd (Gather, permute_fwd的梯度)
            # print("正在执行 C++ 逻辑: unpermute_bwd (Gather)")
            prob_placeholder = torch.empty(num_rows * num_topK, device=input_tensor.device, dtype=torch.float32)
            gather_kernel[grid](
                input_tensor, output_tensor, row_id_map, prob_ptr=prob_placeholder,
                num_rows=num_rows, num_cols=num_cols, num_topK=num_topK, HAS_PROB=False
            )
        else:
            # 路径 2: unpermute_fwd (Weighted Gather)
            # print("正在执行 C++ 逻辑: unpermute_fwd (Weighted Gather)")
            # print("正在执行unpermute_fwd")
            gather_kernel[grid](
                input_tensor, output_tensor, row_id_map, prob_ptr=prob,
                num_rows=num_rows, num_cols=num_cols, num_topK=num_topK, HAS_PROB=True)



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8, num_stages=3),
    ],
    key=['num_cols', 'num_topK'],
)
@triton.jit
def atomic_bwd_kernel(
    # 指针
    input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr,
    # 反向映射指针
    source_token_map_ptr, k_map_ptr,
    # 维度
    num_rows, num_cols, total_expert_tokens,
    # 常量
    num_topK: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    expert_token_id = tl.program_id(axis=0)
    if expert_token_id >= total_expert_tokens:
        return
    source_token_id = tl.load(source_token_map_ptr + expert_token_id)
    k_index = tl.load(k_map_ptr + expert_token_id)
    prob_offset = source_token_id * num_topK + k_index
    prob = tl.load(prob_ptr + prob_offset)

    dot_prod_accum = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        fwd_offset = expert_token_id * num_cols + col_offsets
        input_fwd_frag = tl.load(input_fwd_ptr + fwd_offset, mask=col_mask, other=0.0)
        bwd_offset = source_token_id * num_cols + col_offsets
        input_bwd_frag = tl.load(input_bwd_ptr + bwd_offset, mask=col_mask, other=0.0)

        act_grad_frag = input_bwd_frag * prob.to(input_bwd_frag.dtype)

        act_grad_offset = expert_token_id * num_cols + col_offsets
        tl.store(act_grad_ptr + act_grad_offset, act_grad_frag, mask=col_mask)

        dot_prod_frag = input_fwd_frag * input_bwd_frag
        partial_dot_prod = tl.sum(dot_prod_frag.to(tl.float32))
        dot_prod_accum += partial_dot_prod


    prob_grad_offset = source_token_id * num_topK + k_index
    tl.atomic_add(prob_grad_ptr + prob_grad_offset, dot_prod_accum)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8, num_stages=3), 
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=2, num_stages=4),
      
    ],
    key=['num_rows', 'num_topK'],
)

# umpermute_bwd 

@triton.jit
def scatter_bwd_kernel_optimized(
    # 指针
    input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr, row_id_map_ptr,
    # 维度
    num_rows, num_cols,
    # 常量
    num_topK: tl.constexpr,
    PADDED_TOP_K: tl.constexpr, 
    BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    k_offsets = tl.arange(0, PADDED_TOP_K)
    k_mask = k_offsets < num_topK

    map_offsets = k_offsets * num_rows + source_token_id
    dest_rows = tl.load(row_id_map_ptr + map_offsets, mask=k_mask, other=-1)
    
    prob_offsets = source_token_id * num_topK + k_offsets
    probs = tl.load(prob_ptr + prob_offsets, mask=k_mask, other=0.0)

    is_valid_expert_mask = dest_rows != -1

    dot_prod_accum = tl.zeros((PADDED_TOP_K,), dtype=tl.float32)

    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
       
        col_offsets = col_start + tl.max_contiguous(tl.arange(0, BLOCK_SIZE_COLS), BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        input_bwd_frag = tl.load(
            input_bwd_ptr + source_token_id * num_cols + col_offsets,
            mask=col_mask, other=0.0
        )
        # act_grad_frags 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        act_grad_frags = input_bwd_frag[None, :] * probs[:, None].to(input_bwd_frag.dtype)    
        # act_grad_base_ptrs 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        act_grad_base_ptrs = act_grad_ptr + dest_rows[:, None] * num_cols + col_offsets[None, :]
        act_grad_mask = is_valid_expert_mask[:, None] & col_mask[None, :]
        tl.store(act_grad_base_ptrs, act_grad_frags.to(act_grad_ptr.dtype.element_ty), mask=act_grad_mask)
        input_fwd_base_ptrs = input_fwd_ptr + dest_rows[:, None] * num_cols + col_offsets[None, :]
        input_fwd_mask = is_valid_expert_mask[:, None] & col_mask[None, :]
        input_fwd_frags = tl.load(
            input_fwd_base_ptrs,
            mask=input_fwd_mask, other=0.0,
            cache_modifier=".ca"
        )
        low_prec_products = input_bwd_frag[None, :] * input_fwd_frags    
        partial_dot_prods = tl.sum(low_prec_products.to(tl.float32), axis=1) 
        dot_prod_accum += partial_dot_prods
    final_prob_offsets = source_token_id * num_topK + tl.arange(0, PADDED_TOP_K)
    final_store_mask = tl.arange(0, PADDED_TOP_K) < num_topK
    tl.store(prob_grad_ptr + final_prob_offsets, dot_prod_accum, mask=final_store_mask,cache_modifier=".cs")

def moe_recover_topk_bwd_op_triton(
    input_bwd: torch.Tensor,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    num_tokens = prob.size(0)
    num_topK = prob.size(1)
    num_cols = input_bwd.size(1)


    act_grad = torch.zeros_like(input_fwd, dtype=torch.float32)
    prob_grad = torch.empty_like(prob, dtype=torch.float32)

    moe_permute_topK_kernel_launcher_triton(
        FWD=True,
        input_tensor=input_bwd,
        output_tensor=act_grad, 
        sorted_row_id=None,
        row_id_map=row_id_map,
        prob=prob,
        num_rows=num_tokens,
        num_topK=num_topK,
        num_cols=num_cols,
        num_out_tokens=0,
        prob_grad=prob_grad,
        input_fwd=input_fwd
    )

    return act_grad, prob_grad


@triton.autotune(
    configs=[
        # 统一使用 triton.Config
        triton.Config({'BLOCK_SIZE_COLS': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_COLS': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 2048}, num_warps=8),
    ],
    key=['num_cols'],
)     


@triton.jit
def moe_recover_topk_kernel_triton(
    # 指针
    permuted_grad_ptr,      # 输入：permute 后的梯度 (对应 CUDA 的 input)
    unpermuted_grad_ptr,    # 输出：原始梯度 (对应 CUDA 的 unpermuted_output)
    row_id_map_ptr,         # 输入：行映射 (对应 CUDA 的 row_id_map)
    # 维度参数 (必须是 constexpr 以便编译时优化)
    original_num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    num_topK: tl.constexpr,
    # 调优参数
    BLOCK_SIZE_COLS: tl.constexpr,
):

    source_token_id = tl.program_id(axis=0)
    
    TARGET_DTYPE = unpermuted_grad_ptr.dtype.element_ty
    
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        

        accum_block = tl.zeros((BLOCK_SIZE_COLS,), dtype=tl.float32)
        
  
        for k in range(num_topK):

            map_offset = k * original_num_rows + source_token_id
            source_row = tl.load(row_id_map_ptr + map_offset)
            is_valid_expert = source_row != -1
            load_mask = col_mask & is_valid_expert
            grad_fragment = tl.load(
                permuted_grad_ptr + source_row * num_cols + col_offsets,
                mask=load_mask,
                other=0.0,
            )  
            accum_block += grad_fragment.to(tl.float32)
        dest_ptr = unpermuted_grad_ptr + source_token_id * num_cols + col_offsets
        tl.store(
            dest_ptr,
            accum_block.to(TARGET_DTYPE), # 写回前转换回原始类型
            mask=col_mask
        ) 

@triton.jit
def gather_kernel_optimized(
    input_ptr, output_ptr, row_id_map_ptr, prob_ptr,
    num_rows: tl.constexpr, 
    num_cols: tl.constexpr,
    num_topK: tl.constexpr, 
    HAS_PROB: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    TARGET_DTYPE = output_ptr.dtype.element_ty
    k_offsets = tl.arange(0, num_topK)
    map_offsets = k_offsets * num_rows + source_token_id
    source_rows = tl.load(row_id_map_ptr + map_offsets)
    
    probs = tl.zeros((num_topK,), dtype=tl.float32)
    if HAS_PROB:
        prob_offsets = source_token_id * num_topK + k_offsets
        probs = tl.load(prob_ptr + prob_offsets)


    for col_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_idx = col_start * BLOCK_SIZE_COLS
        col_offsets = col_idx + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        
        source_pointers = input_ptr + \
                          source_rows[:, None] * num_cols + \
                          col_offsets[None, :]
        
        load_mask = (source_rows[:, None] != -1) & col_mask[None, :]
        
        expert_frags = tl.load(source_pointers, mask=load_mask, other=0.0)
        
       
        if HAS_PROB:
            weighted_frags = expert_frags.to(tl.float32) * probs[:, None].to(tl.float32)
            accum_block = tl.sum(weighted_frags, axis=0) # shape: [BLOCK_SIZE_COLS]
        else:
            accum_block = tl.sum(expert_frags.to(tl.float32), axis=0) # shape: [BLOCK_SIZE_COLS]
        

        output_pointers = output_ptr + source_token_id * num_cols + col_offsets
        
        tl.store(
            output_pointers, 
            accum_block.to(TARGET_DTYPE), 
            mask=col_mask
        )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=2),
    ],
    key=['num_cols', 'num_topK'],  # 按列数与 K 选择最优配置并缓存
)
@triton.jit
def scatter_bwd_kernel(
    input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr, row_id_map_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    
    dot_prod_accum = tl.zeros((PADDED_TOP_K,), dtype=tl.float32)

    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        input_bwd_frag = tl.load(
            input_bwd_ptr + source_token_id * num_cols + col_offsets,
            mask=col_mask, other=0.0
        )

        for k in range(num_topK):
            dest_row = tl.load(row_id_map_ptr + k * num_rows + source_token_id)
            is_valid_expert = dest_row != -1
            
            # --- act_grad 的计算部分 ---
            prob_k = tl.load(prob_ptr + source_token_id * num_topK + k)
            act_grad_frag = input_bwd_frag * prob_k.to(input_bwd_frag.dtype)
            
            act_grad_base_ptr = act_grad_ptr + dest_row * num_cols
            

            tl.store(
                act_grad_base_ptr + col_offsets,
                act_grad_frag.to(act_grad_ptr.dtype.element_ty), # 确保写入类型与缓冲区一致
                mask=col_mask & is_valid_expert
            )
   
            input_fwd_base_ptr = input_fwd_ptr + dest_row * num_cols
            input_fwd_frag = tl.load(
                input_fwd_base_ptr + col_offsets,
                mask=col_mask & is_valid_expert, other=0.0
            )
        
            low_prec_product = input_bwd_frag * input_fwd_frag
            partial_dot_prod = tl.sum(low_prec_product.to(tl.float32))

            if is_valid_expert:
                dot_prod_accum = dot_prod_accum + tl.where(tl.arange(0, PADDED_TOP_K) == k, partial_dot_prod, 0.0)

    prob_offsets = source_token_id * num_topK + tl.arange(0, PADDED_TOP_K)
    store_mask = tl.arange(0, PADDED_TOP_K) < num_topK
    tl.store(prob_grad_ptr + prob_offsets, dot_prod_accum, mask=store_mask)
