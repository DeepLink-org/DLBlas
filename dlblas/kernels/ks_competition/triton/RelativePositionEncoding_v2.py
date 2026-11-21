"""
RelativePositionEncoding (AF3 Algo 3-like)

From: protenix/model/modules/embedders.py:RelativePositionEncoding

This version fuses relp feature construction with the linear projection
and prefers a fast Torch path on NPU that avoids explicit one-hot tensors.
A Triton kernel is provided and can be enabled via environment flag for large inputs.
"""
import os
import torch_npu
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    
def generate_relp(
    *,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
    sym_id: torch.Tensor,
    r_max: int = 32,
    s_max: int = 2,
) -> torch.Tensor:
    """
    生成 relp 特征（和仓库实现一致的特征集合与 one-hot 形状）：
    relp = concat([a_rel_pos, a_rel_token, b_same_entity, a_rel_chain])

    Inputs: 都是 [N_token] int64
    Output: [N_token, N_token, (4*r_max + 2*s_max + 7)]
    """
    # same_* masks: [N,N]
    b_same_chain = (asym_id[:, None] == asym_id[None, :]).long()
    b_same_residue = (residue_index[:, None] == residue_index[None, :]).long()
    b_same_entity = (entity_id[:, None] == entity_id[None, :]).long()

    # residue relative index one-hot: size = 2*(r_max+1)
    d_residue = torch.clamp(
        residue_index[:, None] - residue_index[None, :] + r_max, min=0, max=2 * r_max
    )
    d_residue = d_residue * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
    a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))

    # token relative index one-hot: size = 2*(r_max+1)
    d_token = torch.clamp(
        token_index[:, None] - token_index[None, :] + r_max, min=0, max=2 * r_max
    )
    d_token = d_token * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
        2 * r_max + 1
    )
    a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))

    # sym_id relative one-hot: size = 2*(s_max+1)
    d_chain = torch.clamp(sym_id[:, None] - sym_id[None, :] + s_max, min=0, max=2 * s_max)
    d_chain = d_chain * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
    a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))

    relp = torch.cat(
        [
            a_rel_pos,
            a_rel_token,
            b_same_entity[..., None],
            a_rel_chain,
        ],
        dim=-1,
    ).float()
    return relp


# Triton kernel: Fused computation of z = Linear(generate_relp(...)) without materializing one-hots.
# This version loads from transposed weight [IN_DIM, C_Z] to make channel loads contiguous.
if TRITON_AVAILABLE:
    @triton.jit  # fast math enables more aggressive fusion/reassociation for simple arithmetics
    def fused_relp_linear_kernel_wt(
        asym_ptr, res_ptr, ent_ptr, tok_ptr, sym_ptr,
        wT_ptr, out_ptr,
        N, CZ, R_MAX, S_MAX,
        OFF_POS, OFF_TOK, OFF_SE, OFF_CHAIN,
        stride_wt_row, stride_wt_col,
        stride_z_m, stride_z_n, stride_z_c,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_c = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

        mask_m = rm < N
        mask_n = rn < N
        mask_c = rc < CZ

        # Load 1D feature vectors for rows and cols and cast to int32 for consistent integer math
        asym_i = tl.load(asym_ptr + rm, mask=mask_m, other=0).to(tl.int32)
        asym_j = tl.load(asym_ptr + rn, mask=mask_n, other=0).to(tl.int32)

        res_i = tl.load(res_ptr + rm, mask=mask_m, other=0).to(tl.int32)
        res_j = tl.load(res_ptr + rn, mask=mask_n, other=0).to(tl.int32)

        ent_i = tl.load(ent_ptr + rm, mask=mask_m, other=0).to(tl.int32)
        ent_j = tl.load(ent_ptr + rn, mask=mask_n, other=0).to(tl.int32)

        tok_i = tl.load(tok_ptr + rm, mask=mask_m, other=0).to(tl.int32)
        tok_j = tl.load(tok_ptr + rn, mask=mask_n, other=0).to(tl.int32)

        sym_i = tl.load(sym_ptr + rm, mask=mask_m, other=0).to(tl.int32)
        sym_j = tl.load(sym_ptr + rn, mask=mask_n, other=0).to(tl.int32)

        # Broadcast to [BM, BN]
        asym_eq = asym_i[:, None] == asym_j[None, :]
        res_eq = res_i[:, None] == res_j[None, :]
        ent_eq = ent_i[:, None] == ent_j[None, :]

        # Compute indices for one-hots (as in PyTorch implementation)
        # d_residue
        d_res = res_i[:, None] - res_j[None, :] + R_MAX
        d_res = tl.minimum(tl.maximum(d_res, 0), 2 * R_MAX)
        idx_pos = tl.where(asym_eq, d_res, 2 * R_MAX + 1)

        # d_token
        d_tok = tok_i[:, None] - tok_j[None, :] + R_MAX
        d_tok = tl.minimum(tl.maximum(d_tok, 0), 2 * R_MAX)
        cond_tok = asym_eq & res_eq
        idx_tok = tl.where(cond_tok, d_tok, 2 * R_MAX + 1)

        # d_chain
        d_chain = sym_i[:, None] - sym_j[None, :] + S_MAX
        d_chain = tl.minimum(tl.maximum(d_chain, 0), 2 * S_MAX)
        idx_chain = tl.where(ent_eq, d_chain, 2 * S_MAX + 1)

        # Offsets into transposed weight (shape [IN_DIM, CZ]); we sum four selected rows
        k_pos = OFF_POS + idx_pos
        k_tok = OFF_TOK + idx_tok
        k_chain = OFF_CHAIN + idx_chain
        k_se = OFF_SE  # scalar row index for same-entity contribution

        pair_mask = mask_m[:, None] & mask_n[None, :]
        mask3 = pair_mask[:, :, None] & mask_c[None, None, :]

        # Build channel offsets for contiguous loads across CZ
        off_c = rc[None, None, :] * stride_wt_col  # shape [1,1,BC]

        # Row base offsets
        base_pos = k_pos[:, :, None] * stride_wt_row
        base_tok = k_tok[:, :, None] * stride_wt_row
        base_chn = k_chain[:, :, None] * stride_wt_row

        # Load and sum selected weight rows (contiguous along channel dimension)
        w_pos = tl.load(wT_ptr + base_pos + off_c, mask=mask3, other=0)
        w_tok = tl.load(wT_ptr + base_tok + off_c, mask=mask3, other=0)
        w_chain = tl.load(wT_ptr + base_chn + off_c, mask=mask3, other=0)

        # Same-entity contribution: row is constant across pairs
        w_se_c = tl.load(wT_ptr + k_se * stride_wt_row + rc * stride_wt_col, mask=mask_c, other=0)
        # Match dtype of weights to avoid implicit upcasting
        ent_eq_f = ent_eq.to(w_se_c.dtype)
        w_se = w_se_c[None, None, :] * ent_eq_f[:, :, None]

        out = w_pos + w_tok + w_chain + w_se

        # Store to Z [N, N, CZ] with appropriate strides
        z_ptrs = (
            out_ptr
            + rm[:, None, None] * stride_z_m
            + rn[None, :, None] * stride_z_n
            + rc[None, None, :] * stride_z_c
        )
        tl.store(z_ptrs, out, mask=mask3)


class ModelNew(nn.Module):
    """
    相对位置编码：raw features -> relp -> pair embedding (线性投影到 c_z)
    使用等价的融合实现避免显式 one-hot 与大矩阵乘。
    默认使用高效的 Torch 路径；在开启环境变量 USE_TRITON_RELP=1 且 no-grad 时使用 Triton。
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        in_dim = 4 * r_max + 2 * s_max + 7
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.proj = nn.Linear(in_dim, c_z, bias=False)
        torch.set_rng_state(rng_state)

    def _fused_relp_linear_torch(self, asym_id, residue_index, entity_id, token_index, sym_id):
        # Compute z directly by summing selected weight rows from transposed weight
        # This avoids materializing one-hot and minimizes permutes for better memory locality on NPU.
        dtype = self.proj.weight.dtype
        N = asym_id.shape[0]
        r_max = self.r_max
        s_max = self.s_max

        # Masks
        b_same_chain = (asym_id[:, None] == asym_id[None, :])
        b_same_res = (residue_index[:, None] == residue_index[None, :])
        b_same_entity = (entity_id[:, None] == entity_id[None, :])

        # Indices (exactly as original)
        d_res = (residue_index[:, None] - residue_index[None, :] + r_max).clamp_(0, 2 * r_max)
        idx_pos = torch.where(b_same_chain, d_res, torch.full_like(d_res, 2 * r_max + 1))

        d_tok = (token_index[:, None] - token_index[None, :] + r_max).clamp_(0, 2 * r_max)
        idx_tok = torch.where(b_same_chain & b_same_res, d_tok, torch.full_like(d_tok, 2 * r_max + 1))

        d_chain = (sym_id[:, None] - sym_id[None, :] + s_max).clamp_(0, 2 * s_max)
        idx_chain = torch.where(b_same_entity, d_chain, torch.full_like(d_chain, 2 * s_max + 1))

        # Offsets in feature space
        L_pos = 2 * (r_max + 1)
        L_tok = 2 * (r_max + 1)
        OFF_POS = 0
        OFF_TOK = L_pos
        OFF_SE = L_pos + L_tok
        OFF_CHAIN = OFF_SE + 1

        # Use transposed weight for contiguous channel loads: [IN_DIM, C_Z]
        w_T = self.proj.weight.t().contiguous()

        # Gather rows per pair from w_T by flattening indices, then reshape to [N, N, C_Z]
        idx_pos_flat = (OFF_POS + idx_pos).reshape(-1)
        idx_tok_flat = (OFF_TOK + idx_tok).reshape(-1)
        idx_chain_flat = (OFF_CHAIN + idx_chain).reshape(-1)

        w_pos = w_T.index_select(0, idx_pos_flat).view(N, N, self.c_z)
        w_tok = w_T.index_select(0, idx_tok_flat).view(N, N, self.c_z)
        w_chain = w_T.index_select(0, idx_chain_flat).view(N, N, self.c_z)

        # Same-entity scalar one-hot contribution
        w_se_vec = w_T[OFF_SE]  # [C_Z]
        w_se = w_se_vec.view(1, 1, self.c_z) * b_same_entity.to(dtype).unsqueeze(-1)  # [N, N, C_Z]

        z = (w_pos + w_tok + w_chain + w_se).contiguous()  # [N, N, C_Z]
        return z

    def forward(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        entity_id: torch.Tensor,
        token_index: torch.Tensor,
        sym_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            asym_id: [N_token]
            residue_index: [N_token]
            entity_id: [N_token]
            token_index: [N_token]
            sym_id: [N_token]
        Returns:
            z_rel: [N_token, N_token, c_z]
        """
        N = asym_id.shape[0]

        # Prefer the fused Torch implementation for performance and autograd safety.
        # Enable Triton only when explicitly requested and in no-grad context (useful for very large N).
        if self.proj.weight.device != asym_id.device:
            self.to(asym_id.device)
        try:
            import triton
            import triton.language as tl
            TRITON_AVAILABLE = True
        except Exception:
            TRITON_AVAILABLE = False
        use_triton = (
            TRITON_AVAILABLE
            and (not torch.is_grad_enabled())
            and os.environ.get("USE_TRITON_RELP", "0") == "1"
            and N >= 256
        )

        if use_triton and N > 0:
            try:
                r_max = self.r_max
                s_max = self.s_max

                L_pos = 2 * (r_max + 1)
                L_tok = 2 * (r_max + 1)
                OFF_POS = 0
                OFF_TOK = L_pos
                OFF_SE = L_pos + L_tok
                OFF_CHAIN = OFF_SE + 1

                # Ensure int32 inputs for kernel
                asym32 = asym_id.to(torch.int32)
                res32 = residue_index.to(torch.int32)
                ent32 = entity_id.to(torch.int32)
                tok32 = token_index.to(torch.int32)
                sym32 = sym_id.to(torch.int32)

                # Prepare transposed weight [IN_DIM, C_Z] for contiguous channel loads
                w_T = self.proj.weight.t().contiguous()
                stride_wt_row = w_T.stride(0)
                stride_wt_col = w_T.stride(1)

                # Prepare output
                Z = torch.empty((N, N, self.c_z), device=asym_id.device, dtype=self.proj.weight.dtype)

                # Output strides
                stride_z_m, stride_z_n, stride_z_c = Z.stride()

                # Launch kernel
                BLOCK_M = 32
                BLOCK_N = 64
                BLOCK_C = 64
                grid = (
                    triton.cdiv(N, BLOCK_M),
                    triton.cdiv(N, BLOCK_N),
                    triton.cdiv(self.c_z, BLOCK_C),
                )
                fused_relp_linear_kernel_wt[grid](
                    asym32, res32, ent32, tok32, sym32,
                    w_T, Z,
                    N, self.c_z, r_max, s_max,
                    OFF_POS, OFF_TOK, OFF_SE, OFF_CHAIN,
                    stride_wt_row, stride_wt_col,
                    stride_z_m, stride_z_n, stride_z_c,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
                    num_warps=4, num_stages=2,
                )
                return Z
            except Exception:
                # Fallback to torch path if Triton path fails for any reason
                pass

        # Fused Torch path (exact and differentiable)
        return self._fused_relp_linear_torch(asym_id, residue_index, entity_id, token_index, sym_id)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
N_CHAIN = 2
R_MAX = 32
S_MAX = 2
C_Z = 128


def get_inputs():
    device = 'npu'
    torch.manual_seed(42)

    # Generate minimal but "semantically correct" toy input
    asym_id = torch.arange(N_TOKEN, device=device) % max(1, N_CHAIN)
    residue_index = torch.arange(N_TOKEN, device=device)
    entity_id = asym_id.clone()
    token_index = torch.arange(N_TOKEN, device=device)
    sym_id = torch.zeros(N_TOKEN, device=device, dtype=torch.long)

    # Return raw features as a list (not pre-computed relp)
    return [asym_id, residue_index, entity_id, token_index, sym_id]


def get_init_inputs():
    return [R_MAX, S_MAX, C_Z]


if __name__ == "__main__":
    import os
    from torch_npu.profiler import profile, ProfilerActivity

    # 实例化模型 + 生成固定输入
    model = ModelNew(*get_init_inputs())
    inputs = get_inputs()

    # 自动创建trace存放目录
    os.makedirs("./prof_relpos_trace", exist_ok=True)

    # ========== 1. 预热阶段（Triton JIT编译 + Autotune + 框架初始化，不计入正式测速） ==========
    print("===== Warmup Phase: Triton Compile & Autotune =====")
    warmup_rounds = 10
    for _ in range(warmup_rounds):
        _ = model(*inputs)
    torch.npu.synchronize()
    print(f"Warmup {warmup_rounds} iterations finished\n")

    # ========== 2. 异步批量测速（NPU Event 精准耗时，无频繁同步干扰性能） ==========
    print("===== Performance Benchmark (100 iterations Async NPU Event) =====")
    iter_num = 10
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    torch.npu.synchronize()
    start_event.record()
    for _ in range(iter_num):
        _ = model(*inputs)
    end_event.record()
    torch.npu.synchronize()

    total_cost_ms = start_event.elapsed_time(end_event)
    avg_ms_per_iter = total_cost_ms / iter_num
    avg_us_per_iter = avg_ms_per_iter * 1000
    print(f"Total {iter_num} runs total time: {total_cost_ms:.3f} ms")
    print(f"Average single forward: {avg_ms_per_iter:.4f} ms | {avg_us_per_iter:.2f} μs\n")

    # ========== 3. Profiler 捕获算子/显存/硬件流水线Trace，导出Chrome可打开json ==========
    print("===== Start NPU+CPU Profiler Trace Capture =====")
    trace_iter = 10
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        record_shapes=True,    # 记录张量shape，方便定位访存问题
        with_stack=False,
        with_flops=True,       # 可选：统计浮点算力
    )
    prof.start()
    for _ in range(trace_iter):
        out = model(*inputs)
    torch.npu.synchronize()
    prof.stop()

    # 导出trace文件
    trace_save_path = "./prof_relpos_trace/relpos_triton_trace.json"
    prof.export_chrome_trace(trace_save_path)
    print(f"Profiler Chrome Trace saved to: {trace_save_path}")
    print("Open this json file in Chrome chrome://tracing to analyze kernel latency/bandwidth/launch overhead")

    # ========== 4. 附带精度校验（可选，确认采样时结果无漂移） ==========
    print("\n===== Sanity Check: Max Absolute Error vs PyTorch Reference =====")
    out_triton = out
    relp_ref = generate_relp(
        asym_id=inputs[0],
        residue_index=inputs[1],
        entity_id=inputs[2],
        token_index=inputs[3],
        sym_id=inputs[4],
        r_max=R_MAX,
        s_max=S_MAX
    )
    out_ref = model.proj(relp_ref)
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"Max abs error: {max_diff:.10e}")