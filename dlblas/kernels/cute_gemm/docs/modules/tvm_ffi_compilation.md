# CuTe DSL 的 TVM-FFI 编译

## 概述

CuTe DSL kernel 以 `--enable-tvm-ffi` 编译时采用两阶段：

1. 编译期：用 fake（符号）张量构建 MLIR，下译到 PTX。
2. 调用期：传入真实张量；TVM-FFI 提取指针并校验形状。

`@jit_cache` 缓存编译产物（内存 dict + 文件系统 `.o`）。因此只有新的 dtype / major / tile / activation 组合才触发重新编译。

## 符号形状与 `sym_int`

fake 张量用 `cute.sym_int()` 表示符号维。多个张量复用同一个 `sym_int`，即向 TVM-FFI 声明这些维在运行时必须相等。

```python
m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()

# 三者共享同一个 'm' — TVM-FFI 强制 D.shape[0] == A.shape[0] == ColVec.shape[0]
mA = fake_tensor(a_dtype, (m, k, l), leading_dim=..., divisibility=...)
mD = fake_tensor(d_dtype, (m, n, l), leading_dim=..., divisibility=...)
mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
```

`make_fake_tensor`（`cute_gemm/compile_utils.py:8`）创建的张量具有：

- 符号形状维（`sym_int`）；
- 符号步长（`sym_int64`，带整除提示）；leading dim 的步长静态为 1；
- 由整除提示与 dtype 位宽推出的对齐假设（字节数 = `divisibility * width // 8`，下取整且至少 1）。

### varlen 场景

`varlen_m=True` 时，A/D/C 没有 batch 维 `l`。M 维变为 `total_m`（所有序列长度之和）。代码重新赋值 `m = cute.sym_int()`，使调用方仍能从返回值拿到正确的 sym_int：

```python
if varlen_m:
    # m 是 total_m：D/C 展平后的 M 维
    m = cute.sym_int()
    a_m = cute.sym_int() if gather_A else m  # A 的行数可能不同
    mA = fake_tensor(a_dtype, (a_m, k), ...)
    mD = fake_tensor(d_dtype, (m, n), ...)
```

`mAIdx`（gather 索引）复用 `m`（= `total_m`）：每个输出行对应一项。

### 对称 GEMM

`m == n` 的对称 GEMM 对两个维用同一个 sym_int：

```python
m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
mA = fake_tensor(a_dtype, (m, k, l), ...)
mD = fake_tensor(d_dtype, (m, m, l), ...)  # 方阵输出
```

## NamedTuple 作为 kernel 参数

Epilogue、scheduler、varlen 参数都是 `@mlir_namedtuple` 装饰的 NamedTuple。装饰器添加 `__new_from_mlir_values__`。TVM-FFI 运行时据此从展平的 MLIR 值重建结构体。

```python
@mlir_namedtuple
class EpilogueArguments(NamedTuple):
    mPostAct: cute.Tensor
    act_fn: cutlass.Constexpr[Callable] = None   # 编译期烘焙
    alpha: Optional[Float32 | cute.Tensor] = None
    beta: Optional[Float32 | cute.Tensor] = None
```

### 重建过程

`_namedtuple_new_from_mlir_values(self, values)`（`cute_gemm/cute_dsl_utils.py:239`）：

1. 遍历编译期模板（`self`）的字段；
2. `None` 与 `StaticTypes`（int、bool、str、Constexpr 等）原样保留；
3. 复杂类型（cute.Tensor、指针）调用 `cutlass.new_from_mlir_values(field_val, values[:n_items])`，消费 N 个 MLIR 值；
4. 返回重建字段的 NamedTuple 实例。

## Constexpr 字段

标注 `cutlass.Constexpr[T]` 的字段是编译期常量。其值在编译期烘焙进 kernel。调用期对这些字段传 `None`。

对 TVM-FFI 参数转换器的 monkey-patch 启用该行为（`cute_gemm/cute_dsl_utils.py:39`）：

```python
def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)  # 非运行期参数
    ...
```

示例：`act_fn` 是 `Constexpr[Callable]`。编译期将激活函数烘焙进 kernel。不同 activation 产出不同的编译产物。

### NamedTuple 类型注解的变通

NamedTuple 以 `tuple` 注解传入时（如 `epilogue_args: tuple`），TVM-FFI 转换器看不到 NamedTuple 的字段类型。patch 改用 `type(arg)`（实际的 NamedTuple 类）：

```python
if (isinstance(arg, tuple) and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))):
    return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
```

## ParamsBase（dataclass 参数）

`ParamsBase` 服务经 `epi_to_underlying_arguments` 传递的 kernel 参数（JIT 层结构体，不是用户面参数）。它把字段划分为 constexpr（静态）与非 constexpr：

```python
@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values
```

`_values_pos` 记录每个字段消费的 MLIR 值个数。`_new_from_mlir_values` 据此按正确顺序重建字段。

## 编译与缓存流程

```
gemm_with_activation(A, B, ...)  # 用户面类入口（GemmBase）
  │
  ├─ 从真实张量提取 dtype、major
  ├─ _compile_gemm_epi(...)     # @jit_cache — 内存 + 磁盘缓存
  │    │                         #（cute_gemm/gemm_runtime/host.py:84）
  │    ├─ make_fake_gemm_tensors(...)  # 共享 sym_int 的符号张量
  │    ├─ 构建 fake epilogue / scheduler / varlen 参数
  │    └─ compile_gemm_kernel(...)
  │         └─ cute.compile(gemm_obj, *fake_args, options="--enable-tvm-ffi")
  │
  ├─ 构建运行期 epilogue 参数（真实张量；Constexpr 字段为 None）
  ├─ 构建运行期 scheduler / varlen 参数
  └─ compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)
```

`@jit_cache` 的 key 来自函数参数（dtype、major、tile 尺寸、cluster 维、activation 等）。参数相同，得到同一份编译产物。

## 关键文件

| 文件 | 职责 |
|------|------|
| `cute_gemm/compile_utils.py` | `make_fake_tensor` — 构建符号张量 |
| `cute_gemm/gemm_tvm_ffi_utils.py` | `make_fake_gemm_tensors`、`compile_gemm_kernel`、varlen / scheduler 辅助 |
| `cute_gemm/cute_dsl_utils.py` | `@mlir_namedtuple`、`ParamsBase`、Constexpr 转换器 patch |
| `cute_gemm/cache/jit.py` | `@jit_cache` 装饰器 — 内存 + 文件系统 `.o` 缓存 |
