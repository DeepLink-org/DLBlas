# fa_hdim128 — Fully-Builtin Flash-Attention Forward (MetaX C500 / xcore1000)

A from-scratch reimplementation of dense flash-attention **forward** (D=128, fp16,
non-causal, even-MN/K) that depends **only on the official MACA system SDK**
(cute + mctlass via the `MACA_PATH` install root) plus the headers in this
directory. It does **not** reference any external source tree (no csrc copy /
`/home/...` paths).

- Algorithm: online-softmax + tiled MMA (`MACA_16x16x16_F32F16F16F32` atom) +
  swizzled shared memory + Q-in-regs + K reg-staged prefetch + direct-V gmem→smem.
- Config: B=1, H=32, D=128, blockM=128, blockN=64, 4 warps (256 threads, wave=64).
- Correctness: element-wise `allclose` (atol 1e-2) vs the torch flash_attn wheel.
- This is the kernel body from the project's Round-8 async-hybrid; only the
  traits / utils / softmax / params headers were reauthored on top of the
  system cute/mctlass (the fa_src copies were dropped).

## Directory layout

```
builtin/
├── build.sh                 # self-locating build (uses $MACA_PATH)
├── README.md
├── src/
│   ├── fa_my.cu             # harness: alloc / init / time / dump, calls the kernel
│   ├── my_compute.cuh       # kernel body: compute_attn_myimpl + my_flash_fwd_kernel
│   └── builtin/
│       ├── fa_params.h      # Flash_fwd_params struct (namespace mcFlashAttn)
│       ├── fa_traits.cuh     # concrete Flash_fwd_kernel_traits<128,128,64,4,true,true,half_t,128>
│       ├── fa_utils.cuh      # flash::{gemm,gemm_rs,copy_*,clear,barrier*,softmax helpers,...}
│       └── fa_softmax.cuh    # flash::Softmax (softmax_rescale_o + normalize_softmax_lse)
└── tests/
    ├── cmp_my.py             # correctness vs torch _flash_attn_forward
    ├── bench_torch.py        # torch flash_attn timing (single run)
    └── bench_torch_multi.py  # torch flash_attn timing (min/med/max, 5 runs)
```

## Prerequisites

- MetaX MACA SDK installed (provides cute, mctlass, cu-bridge `cucc`, runtime libs).
  The install root is exposed via the **`MACA_PATH`** env var
  (e.g. `export MACA_PATH=/opt/maca`). It must contain:
  - `$MACA_PATH/include` — cute + mctlass headers
  - `$MACA_PATH/tools/cu-bridge/bin/cucc` — the compiler
  - `$MACA_PATH/tools/cu-bridge/include` — CUDA shim headers
  - `$MACA_PATH/lib` — runtime libs (`libmcruntime`, `libmxc-runtime64`, …)
- A MetaX C500 (xcore1000) device.
- For testing only: torch + the `flash_attn` wheel (`_flash_attn_forward`).

> In the `metax_gemm_opt` container, `MACA_PATH=/opt/maca` is already set.

## How to compile

```bash
export MACA_PATH=/opt/maca      # your MACA install root
bash build.sh
# -> produces ./fa_my_builtin
```

`build.sh` is self-locating (finds `./src` relative to itself) and reads the MACA
SDK only from `$MACA_PATH`. No hardcoded absolute paths.

## How to test

The binary takes: `./fa_my_builtin <S> <warmup> <iters> <dump>`
(defaults: S=512, warmup=5, iters=30, dump=0). When `dump=1` it writes
`$FA_DUMP_DIR/fa_my_<S>.bin` (defaults to `./` if `FA_DUMP_DIR` unset) containing
B,H,S,D + Q,K,V,O in fp16.

### Correctness (vs torch flash_attn)

```bash
export FA_DUMP_DIR=/tmp        # optional; default is current dir
./fa_my_builtin 512 5 30 1     # dump
./fa_my_builtin 1024 5 30 1
python tests/cmp_my.py         # reads $FA_DUMP_DIR/fa_my_{512,1024}.bin, compares to torch
# expect: allclose(1e-2)=True, max_diff < 0.01
```

### Performance vs torch

```bash
# Our kernel (median over iters):
for S in 512 1024 10240 102400; do ./fa_my_builtin $S 5 30 0; done

# torch flash_attn API (median of 5 runs):
python tests/bench_torch.py            # S in {512,1024,10240,102400}
python tests/bench_torch_multi.py      # min/med/max for small/medium shapes
```

## Measured results (MetaX C500, 2026-07-22, container metax_gemm_opt)

| Shape (1x32xSx128) | fa_my_builtin (ms) | torch API median (ms) |
|---|---|---|
| 512    | 0.0826  | 0.0666 |
| 1024   | 0.2266  | 0.1739 |
| 10240  | 16.052  | 17.86  |
| 102400 | 1564.0  | 1195.2 |

Correctness: `allclose(1e-2)=True` at S=512 (max_diff≈0.0065) and S=1024 (≈0.0042).

## Notes

- All source/build/test files are path-free (portable). System SDK location comes
  from `$MACA_PATH`; dump location from `$FA_DUMP_DIR` (default `./`).
- The kernel reuses no external flash_attn source headers — traits / utils /
  softmax / params are reauthored here on top of the official cute/mctlass.
  The one fa_src-local cute extension (`get_swizzle_offset`) is avoided by using
  the standard `Swizzle<3,4,3>` + `composition` (the kernel never needed it).


====

  编译 / 测试（README 已详述）

  export MACA_PATH=/opt/maca     # MACA SDK 根 (容器已设)
  bash build.sh                  # → ./fa_my_builtin

  export FA_DUMP_DIR=/tmp        # 可选, 默认 ./
  ./fa_my_builtin 512 5 30 1     # dump
  ./fa_my_builtin 1024 5 30 1
  python tests/cmp_my.py         # 正确性 vs torch, 期望 allclose(1e-2)=True

  for S in 512 1024 10240 102400; do ./fa_my_builtin $S 5 30 0; done   # 计时
  python tests/bench_torch.py                                          # torch 对照

  验证结果（scrub 后临时副本实测，已清理）

  - 编译：BUILD_OK, Exit 0
  - 正确性：S=512 max_diff=0.0065、S=1024 max_diff=0.0042，allclose(1e-2)=True ✅
  - 计时：512→0.0825 / 1024→0.2267 / 10240→16.052 / 1024

====