# Ragged TMA ptr_shift: Works for TMA Store, Fails for TMA Load

## Summary

The 1-extra-dim ragged tensor approach (`ptr_shift=True`) works for **TMA stores** but
fails for **TMA loads** when `big_int` is large enough that the shifted base pointer
falls outside physically mapped GPU memory.

## Background

To handle variable-length (ragged) sequences with TMA without updating the TMA
descriptor per batch, we create a higher-rank tensor with a fixed large dimension
(`big_int`). Two approaches exist:

| Approach | Extra dims | Max input rank | Pointer shifted? |
|----------|-----------|----------------|------------------|
| `ptr_shift=True` (1-extra-dim) | 1 | 4D (4+1=5, TMA max) | Yes, backward by `big_int * stride_r * elem_bytes` |
| `ptr_shift=False` (2-extra-dim) | 2 | 3D (3+2=5, TMA max) | No (uses 64-bit wraparound) |

Both approaches compute the correct final address via algebra — the extra terms cancel
out. The difference is that `ptr_shift=True` requires shifting the base pointer backward
by ~1 GB (when `big_int = 2^30`), creating a `globalAddress` in the TMA descriptor that
points to unmapped GPU memory.

## Finding

**TMA loads validate or access the `globalAddress` field of the TMA descriptor.**
If `globalAddress` points to unmapped GPU memory, TMA loads fail — even though the actual
data access (after applying coordinates) resolves to a valid address.

**TMA stores do not have this issue.** The store path only accesses the computed target
address, so the base pointer being in unmapped memory is harmless.

Concretely:
- `ptr_shift=True` with `big_int=2^30` **works** for `TMA_STORE` (epilogue D writes)
- `ptr_shift=True` with `big_int=2^30` **fails** for `TMA_LOAD` (mainloop A/B reads, epilogue C reads)
- `ptr_shift=True` with small `big_int` (shifted pointer still in allocated range) **works** for loads too
- `ptr_shift=False` (no pointer shift) **works** for all TMA paths

### Evidence

1. **Threshold test**: For a tensor A of shape (2048, 512) in bf16, the 1-extra-dim
   approach fails for `big_int >= ~262144`. At `big_int=262144`, the pointer shift is
   `262144 * 2048 * 2 = 1 GB` — just beyond the tensor's own allocation. Smaller values
   keep the shifted pointer within the allocation and work fine.

2. **Contiguous buffer test**: When A and B are allocated inside a single large buffer
   that covers the entire shifted address range, `ptr_shift=True` works with any `big_int`.
   This confirms the issue is purely about address validity, not TMA descriptor encoding.

3. **Store vs load**: varlen_m (D store with `ptr_shift=True`) works with `big_int=2^30`.
   varlen_k A/B loads and epilogue C loads with `ptr_shift=True` fail with the same `big_int`.

## Consequence for the codebase

- **TMA loads** (A, B, C): must use `ptr_shift=False` (2-extra-dim, max 3D input)
- **TMA stores** (D): can use `ptr_shift=True` (1-extra-dim, max 4D input)

This is what `cute_gemm/gemm_sm90.py` and `cute_gemm/gemm_sm100.py` currently implement.

## A/B/C checks

- varlen_m GEMM (D store uses ptr_shift=True): **PASS**
- varlen_k GEMM (A/B loads use ptr_shift=True): **FAIL** — the base pointer, shifted backward ~1 GB, lands in unmapped memory
- varlen_k GEMM (A/B loads use ptr_shift=False): **PASS**
