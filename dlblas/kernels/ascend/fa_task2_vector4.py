import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

DEVICE = "npu"


@triton.jit
def get_params(
    Q,
    K,
    V,
    Out,
    WS1,
    WS2,
    WS3,
    WS4,
    block_idx,
    NUM_BLOCKS_M,
    H,
    stride_qz,
    stride_qh,
    N_CTX,
    HEAD_DIM,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_on,
    w12_stride_nb,
    w12_stride_bm,
    w12_stride_bn,
    w34_stride_nb,
    w34_stride_bm,
    w34_stride_dm,
    BLOCK_M,
    SUB_BLOCK_M,
    BLOCK_N,
    TASK_ID,
):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    task_m_idx = block_idx % NUM_BLOCKS_M
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    # Create block pointers for Q, K, V, Output
    Q_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr_a = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr_b = tl.advance(O_ptr_a, (SUB_BLOCK_M, 0))
    O_ptr_c = tl.advance(O_ptr_a, (SUB_BLOCK_M * 2, 0))
    O_ptr_d = tl.advance(O_ptr_a, (SUB_BLOCK_M * 3, 0))

    WS1_block_ptr = tl.make_block_ptr(
        base=WS1 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS1_a = tl.make_block_ptr(
        base=WS1 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS1_b = tl.advance(WS1_a, (SUB_BLOCK_M, 0))
    WS1_c = tl.advance(WS1_a, (SUB_BLOCK_M * 2, 0))
    WS1_d = tl.advance(WS1_a, (SUB_BLOCK_M * 3, 0))

    WS2_block_ptr = tl.make_block_ptr(
        base=WS2 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS2_a = tl.make_block_ptr(
        base=WS2 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS2_b = tl.advance(WS2_a, (SUB_BLOCK_M, 0))
    WS2_c = tl.advance(WS2_a, (SUB_BLOCK_M * 2, 0))
    WS2_d = tl.advance(WS2_a, (SUB_BLOCK_M * 3, 0))

    WS3_block_ptr = tl.make_block_ptr(
        base=WS3 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS3_a = tl.make_block_ptr(
        base=WS3 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS3_b = tl.advance(WS3_a, (SUB_BLOCK_M, 0))
    WS3_c = tl.advance(WS3_a, (SUB_BLOCK_M * 2, 0))
    WS3_d = tl.advance(WS3_a, (SUB_BLOCK_M * 3, 0))

    WS4_a = tl.make_block_ptr(
        base=WS4 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS4_b = tl.advance(WS4_a, (SUB_BLOCK_M, 0))
    WS4_c = tl.advance(WS4_a, (SUB_BLOCK_M * 2, 0))
    WS4_d = tl.advance(WS4_a, (SUB_BLOCK_M * 3, 0))

    return (
        Q_ptr,
        K_ptr,
        V_ptr,
        (O_ptr_a, O_ptr_b, O_ptr_c, O_ptr_d),
        WS1_block_ptr,
        (WS1_a, WS1_b, WS1_c, WS1_d),
        WS2_block_ptr,
        (WS2_a, WS2_b, WS2_c, WS2_d),
        WS3_block_ptr,
        (WS3_a, WS3_b, WS3_c, WS3_d),
        (WS4_a, WS4_b, WS4_c, WS4_d),
    )


@triton.jit
def create_ub_tensor(SUB_BLOCK_M: tl.constexpr):
    m_i = tl.zeros([SUB_BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([SUB_BLOCK_M], dtype=tl.float32) + 1.0
    return m_i, l_i


@triton.jit
def cube_part_a(K_ptr, Q_ptr, WS1, V_ptr, set_event_id):
    q = tl.load(Q_ptr)
    k = tl.load(K_ptr)  # (BLOCK_N, HEAD_DIM)
    k_t = tl.trans(k)
    # dl.compile_hint(k_t, "dot_pad_only_k")
    qk = tl.dot(q, k_t)  # [BM, HEAD_DIM] * [HEAD_DIM, BN] -> [BM, BN]
    tl.store(WS1, qk)
    dl.set_cross_flag(dl.SyncFlag.C2V, set_event_id)
    # v = tl.load(V_ptr)  # shape: [BLOCK_N, HEAD_DIM]
    # return v


@triton.jit
def vec_a(Q, WS1, WS2, m_i, l_i, sm):
    qk = tl.load(WS1)
    qk = qk * sm
    m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize
    p = tl.math.exp(qk)
    l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
    p_cast = p.to(Q.type.element_ty)
    tl.store(WS2, p_cast)
    # return l_ij, m_ij
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    return m_ij, alpha, l_i


@triton.jit
def cube_part_b(WS2, WS3, V_ptr, wait_event_id, set_event_id):
    v = tl.load(V_ptr)
    dl.wait_cross_flag(dl.SyncFlag.V2C, wait_event_id)
    p_cast = tl.load(WS2)  # shape: [BLOCK_M, BLOCK_N]
    acc_l0c = tl.dot(p_cast, v)  # [BM, BN] * [BN, HEAD_DIM] -> [BM, HEAD_DIM]
    tl.store(WS3, acc_l0c)
    dl.set_cross_flag(dl.SyncFlag.C2V, set_event_id)


@triton.jit
def vector_part_b(WS3, WS4, alpha):
    acc = tl.load(WS4) * alpha[:, None]
    acc = tl.load(WS3) + acc
    tl.store(WS4, acc)


@triton.jit
def vector_last(Out, O_ptr, WS4, l_i):
    acc_ws = tl.load(WS4)
    accumulator = acc_ws / l_i[:, None]
    tl.store(O_ptr, accumulator.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_split_cv(
    Q,
    K,
    V,
    Out,
    sm,
    WS1,
    WS2,
    WS3,
    WS4,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    w12_stride_nb: tl.constexpr,
    w12_stride_bm: tl.constexpr,
    w12_stride_bn: tl.constexpr,
    w34_stride_nb: tl.constexpr,
    w34_stride_bm: tl.constexpr,
    w34_stride_dm: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    BATCH_NUM_BLOCKS: tl.constexpr = NUM_BLOCKS // 2  # 2 tasks
    SUB_BLOCK_M: tl.constexpr = BLOCK_M // 4
    # Current M-dimension block index
    pid = tl.program_id(0)
    for block_idx in range(pid, BATCH_NUM_BLOCKS, NUM_CORES):
        block_idx_0 = block_idx * 2
        block_idx_1 = block_idx * 2 + 1
        # --- task 0
        (
            Q_ptr_0,
            K_ptr_0,
            V_ptr_0,
            O_ptr_0,
            WS1_0,
            WS1_L,
            WS2_0,
            WS2_L,
            WS3_0,
            WS3_L,
            WS4_L,
        ) = get_params(
            Q,
            K,
            V,
            Out,
            WS1,
            WS2,
            WS3,
            WS4,
            block_idx_0,
            NUM_BLOCKS_M,
            H,
            stride_qz,
            stride_qh,
            N_CTX,
            HEAD_DIM,
            stride_qm,
            stride_qk,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            stride_om,
            stride_on,
            w12_stride_nb,
            w12_stride_bm,
            w12_stride_bn,
            w34_stride_nb,
            w34_stride_bm,
            w34_stride_dm,
            BLOCK_M,
            SUB_BLOCK_M,
            BLOCK_N,
            pid,
        )
        (O_0_a, O_0_b, O_0_c, O_0_d) = O_ptr_0
        (WS1_0_a, WS1_0_b, WS1_0_c, WS1_0_d) = WS1_L
        (WS2_0_a, WS2_0_b, WS2_0_c, WS2_0_d) = WS2_L
        (WS3_0_a, WS3_0_b, WS3_0_c, WS3_0_d) = WS3_L
        (WS4_0_a, WS4_0_b, WS4_0_c, WS4_0_d) = WS4_L

        # -- task 1
        (
            Q_ptr_1,
            K_ptr_1,
            V_ptr_1,
            O_ptr_X_1,
            WS1_1,
            WS1_X,
            WS2_1,
            WS2_X,
            WS3_1,
            WS3_X,
            WS4_X,
        ) = get_params(
            Q,
            K,
            V,
            Out,
            WS1,
            WS2,
            WS3,
            WS4,
            block_idx_1,
            NUM_BLOCKS_M,
            H,
            stride_qz,
            stride_qh,
            N_CTX,
            HEAD_DIM,
            stride_qm,
            stride_qk,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            stride_om,
            stride_on,
            w12_stride_nb,
            w12_stride_bm,
            w12_stride_bn,
            w34_stride_nb,
            w34_stride_bm,
            w34_stride_dm,
            BLOCK_M,
            SUB_BLOCK_M,
            BLOCK_N,
            pid + NUM_CORES,
        )
        (O_1_a, O_1_b, O_1_c, O_1_d) = O_ptr_X_1
        (WS1_1_a, WS1_1_b, WS1_1_c, WS1_1_d) = WS1_X
        (WS2_1_a, WS2_1_b, WS2_1_c, WS2_1_d) = WS2_X
        (WS3_1_a, WS3_1_b, WS3_1_c, WS3_1_d) = WS3_X
        (WS4_1_a, WS4_1_b, WS4_1_c, WS4_1_d) = WS4_X

        with dl.async_task(scope=dl.async_task.cube):
            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(
                    start_n, BLOCK_N
                )  # Align column start position
                # task 0
                cube_part_a(K_ptr_0, Q_ptr_0, WS1_0, V_ptr_0, 0)
                # task 1
                cube_part_a(K_ptr_1, Q_ptr_1, WS1_1, V_ptr_1, 1)

                # task 0
                cube_part_b(WS2_0, WS3_0, V_ptr_0, 2, 4)
                V_ptr_0 = tl.advance(V_ptr_0, (BLOCK_N, 0))
                K_ptr_0 = tl.advance(K_ptr_0, (BLOCK_N, 0))

                # task 1
                cube_part_b(WS2_1, WS3_1, V_ptr_1, 3, 5)
                V_ptr_1 = tl.advance(V_ptr_1, (BLOCK_N, 0))
                K_ptr_1 = tl.advance(K_ptr_1, (BLOCK_N, 0))

        with dl.async_task(scope=dl.async_task.vector):
            # task 0
            mi0_a, li0_a = create_ub_tensor(SUB_BLOCK_M)
            mi0_b, li0_b = create_ub_tensor(SUB_BLOCK_M)
            mi0_c, li0_c = create_ub_tensor(SUB_BLOCK_M)
            mi0_d, li0_d = create_ub_tensor(SUB_BLOCK_M)

            # task 1
            mi1_a, li1_a = create_ub_tensor(SUB_BLOCK_M)
            mi1_b, li1_b = create_ub_tensor(SUB_BLOCK_M)
            mi1_c, li1_c = create_ub_tensor(SUB_BLOCK_M)
            mi1_d, li1_d = create_ub_tensor(SUB_BLOCK_M)

            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                # task 0
                dl.wait_cross_flag(dl.SyncFlag.C2V, 0)
                mi0_a, alpha_0_a, li0_a = vec_a(Q, WS1_0_a, WS2_0_a, mi0_a, li0_a, sm)
                mi0_b, alpha_0_b, li0_b = vec_a(Q, WS1_0_b, WS2_0_b, mi0_b, li0_b, sm)
                mi0_c, alpha_0_c, li0_c = vec_a(Q, WS1_0_c, WS2_0_c, mi0_c, li0_c, sm)
                mi0_d, alpha_0_d, li0_d = vec_a(Q, WS1_0_d, WS2_0_d, mi0_d, li0_d, sm)

                dl.set_cross_flag(dl.SyncFlag.V2C, 2)

                # task 1
                dl.wait_cross_flag(dl.SyncFlag.C2V, 1)
                mi1_a, alpha_1_a, li1_a = vec_a(Q, WS1_1_a, WS2_1_a, mi1_a, li1_a, sm)
                mi1_b, alpha_1_b, li1_b = vec_a(Q, WS1_1_b, WS2_1_b, mi1_b, li1_b, sm)
                mi1_c, alpha_1_c, li1_c = vec_a(Q, WS1_1_c, WS2_1_c, mi1_c, li1_c, sm)
                mi1_d, alpha_1_d, li1_d = vec_a(Q, WS1_1_d, WS2_1_d, mi1_d, li1_d, sm)

                dl.set_cross_flag(dl.SyncFlag.V2C, 3)

                # task 0
                dl.wait_cross_flag(dl.SyncFlag.C2V, 4)
                vector_part_b(WS3_0_a, WS4_0_a, alpha_0_a)
                vector_part_b(WS3_0_b, WS4_0_b, alpha_0_b)
                vector_part_b(WS3_0_c, WS4_0_c, alpha_0_c)
                vector_part_b(WS3_0_d, WS4_0_d, alpha_0_d)

                # task 1
                dl.wait_cross_flag(dl.SyncFlag.C2V, 5)
                vector_part_b(WS3_1_a, WS4_1_a, alpha_1_a)
                vector_part_b(WS3_1_b, WS4_1_b, alpha_1_b)
                vector_part_b(WS3_1_c, WS4_1_c, alpha_1_c)
                vector_part_b(WS3_1_d, WS4_1_d, alpha_1_d)

            # task 0
            vector_last(Out, O_0_a, WS4_0_a, li0_a)
            vector_last(Out, O_0_b, WS4_0_b, li0_b)
            vector_last(Out, O_0_c, WS4_0_c, li0_c)
            vector_last(Out, O_0_d, WS4_0_d, li0_d)

            # task 1
            vector_last(Out, O_1_a, WS4_1_a, li1_a)
            vector_last(Out, O_1_b, WS4_1_b, li1_b)
            vector_last(Out, O_1_c, WS4_1_c, li1_c)
            vector_last(Out, O_1_d, WS4_1_d, li1_d)


def attention_split_cv(q, k, v, sm, BM, BN):
    """
    Forward computation interface:
    Args:
        ctx: Context object
        q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
        k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
        v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
        sm: Scaling factor for QK product
        BM: Q block size (BLOCK_M)
        BN: K/V block size (BLOCK_N)
    Returns:
        o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
    """
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    extra_kern_args = {}
    DIM = q.shape[-1]
    # Number of NPU cores (adjust based on hardware)
    NUM_CORES = 24
    acc_type = torch.float32
    workspace_1 = torch.empty((NUM_CORES * 2, BM, BN), device=q.device, dtype=acc_type)
    workspace_2 = torch.empty((NUM_CORES * 2, BM, BN), device=q.device, dtype=q.dtype)
    workspace_3 = torch.empty((NUM_CORES * 2, BM, DIM), device=q.device, dtype=acc_type)
    workspace_4 = torch.empty((NUM_CORES * 2, BM, DIM), device=q.device, dtype=acc_type)
    _attn_fwd_split_cv[(NUM_CORES,)](
        q,
        k,
        v,
        o,
        sm,
        workspace_1,
        workspace_2,
        workspace_3,
        workspace_4,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        workspace_1.stride(0),
        workspace_1.stride(1),
        workspace_1.stride(2),
        workspace_3.stride(0),
        workspace_3.stride(1),
        workspace_3.stride(2),
        q.shape[0],
        q.shape[1],
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BM,
        BLOCK_N=BN,
        NUM_CORES=NUM_CORES,
        disable_auto_inject_block_sync=True,
        disable_auto_cv_work_space_manage=True,
        multibuffer=True,  # 控制开double_buffer
        unit_flag=True,  # cube搬出的一个优化项
        **extra_kern_args,
    )
    return o


def fa_ascendc(q, k, v, sm, H):
    ref_out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        H,
        padding_mask=None,
        atten_mask=None,
        scale=sm,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]
    return ref_out


def benchmark(Z, H, N_CTX, HEAD_DIM, dtype, BM, BN):
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    sm = 0.5
    tri_split_cv_out = attention_split_cv(q, k, v, sm, BM, BN)
    # return
    ref_out = fa_ascendc(q, k, v, sm, H)
    torch.testing.assert_close(
        tri_split_cv_out, ref_out, atol=1e-2, rtol=1e-2, equal_nan=True
    )

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["cnt"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
            x_vals=[1],  # NOTE: the tunning framework specialized to one shape
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["fa_base", "fa_split_cv"],  # Label name for the lines
            line_names=["fa_base", "fa_split_cv"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="flash-attention-"
            + (
                f"{dtype}-[Batch={Z} H={H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}]"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(cnt, provider):
        warmup = 500
        rep = 500
        quantiles = [0.5, 0.2, 0.8]
        if provider == "fa_base":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fa_ascendc(q, k, v, sm, H),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        if provider == "fa_split_cv":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: attention_split_cv(q, k, v, sm, BM, BN),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )

        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    # benchmark(1, 1, 128, 128, dtype=torch.float16, BM=32, BN=128)
    # benchmark(1, 1, 512, 128, dtype=torch.float16, BM=32, BN=128)
    # benchmark(1, 1, 1024, 128, dtype=torch.float16, BM=32, BN=128)
    # benchmark(1, 1, 128, 128, dtype=torch.bfloat16, BM=64, BN=128)
    # benchmark(1, 2, 256, 256, dtype=torch.bfloat16, BM=32, BN=256)
    # benchmark(2, 2, 128, 256, dtype=torch.float16, BM=64, BN=128)
    # benchmark(4, 32, 64, 64, dtype=torch.float16, BM=32, BN=64)
    # benchmark(8, 32, 2048, 128, dtype=torch.bfloat16, BM=64, BN=128)
    benchmark(Z=16, H=4, N_CTX=1024, HEAD_DIM=256, dtype=torch.float16, BM=128, BN=256)
