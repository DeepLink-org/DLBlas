mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)