# SM90 指针数组 grouped GEMM(gemm_sm90_ptr_array)

单次 grouped GEMM launch 消费来自多块独立显存的权重。设备端 `B_ptrs[G]` 指针数组给出每个 expert 的权重矩阵地址,kernel 在组切换时更新自己 CTA 的 TMA descriptor 基址。面向多权重池训练(如 MoonEP 的 W_local + W_remote);不做路由、permute/combine、autograd,归调用方。

## 入口

```python
cute_gemm.gemm_ptr_array(A, weights, cu_seqlens_m, b_ptrs, *,
                        out=None, alpha=1.0, out_dtype=None,
                        dynamic_scheduler=True, tuned=True) -> Tensor
```

- `A`:`(M_total, K)`,行布局同现有 varlen_m 契约。
- `weights`:`Sequence[Tensor]`,每池 `(Gi, N, K)`,同 dtype、同 `shape[1:]`、同 `stride()[1:]`(仅 `stride(0)` 可异)。
- `cu_seqlens_m`:int32 `(G+1)`,`G = ΣGi`;host 不读值。
- `b_ptrs`:int64 `(G)`,**缓冲归调用方**,内容由入口的 prep kernel 每次调用重写。
- 输出:未提供 `out` 时分配 `(M_total, N)`。

## 机制

| 部件 | 设计 |
|---|---|
| `PtrArrayB` 束 | `@mlir_namedtuple`,字段 `(template, b_ptrs, slots)`。以 B 的身份过既有 B 槽,TVM-FFI 烘焙 `TupleParam`(TransformAOperand 同配方);duck 的 `dtype/shape/stride` 委托模板,plan-build 读取零改动 |
| prep kernel | 池基址/stride/累积边界烘焙进闭包按元组缓存;运行期只收 `b_ptrs` 与 counter 两个设备张量。`b_ptrs[g] = bases[p] + slot * strides[p]`,strides 为**字节步长**(`w.stride(0) * element_size()`)。DYNAMIC 时线程 0 顺带清零 tile counter(取代 torch.zeros,保零 H2D) |
| tensormap 槽 | `(num_sms, 16)` int64,GMEM,每行 128B。**每次调用现取**(跨 stream 并发调用不共享可变 descriptor);CTA 启动时 `init_tensormap_from_atom` 全量覆写自己的槽 |
| 组切换 | `batch_idx != last_group` 时:读 `B_ptrs[g]` → 重建单专家张量 → `update_tensormap` + `fence_tensormap_update`。L 模钉 0(模板 batch 维为 1),地址选择数据、坐标只走 (n, k) |
| 窄 fork | `GemmSm90PtrArray(GemmDefaultSm90)` 只覆写 `__call__` 与 `kernel`,其余继承。上游 `gemm_sm90.py` 演进不自动同步,升级时人工比对这两个方法(文件头互注) |
| 编译与调优 | 自有 plan cache + `_compile_ptr_array_gemm`(gemm_symmetric 形制),复用参数化的 `compile_gemm_kernel(GemmCls, ...)`;共享文件零改动。tune 扫 SM90 协作式 `(tile, cluster)` 候选,B multicast 数量 = cluster_m |

## 不变量

- 每次调用 H2D/D2H 为 **0 字节**:池基址走 kernel 标量参数、槽初始化在 kernel 内完成、host 不读任何设备值。
- 权重内容原位更新(地址不变)后同参再调,输出反映新内容;权重**换地址**需重新 capture CUDA Graph。
- GEMM kernel 对池结构 P 无关(池分解只在 prep 闭包);plan 缓存键不含 P 是正确设计。

## 性能边界(ncu 实测)

单 launch 单 tile 服务混合负载:大形状上 fused 的 Compute SOL 约 90%,split 两池各自调优可达 97%+——差距来自小组 tile padding 的无效发射,不是 descriptor 更新(组切换全套含 fence 约 0.5µs,prep kernel 约 2µs)。完整调用 Gate:fused ≤ 1.10 × split。

## 测试

`tests/test_gemm_ptr_array.py`(结构负例、prep 地址、双/三池 e2e 对拍 fp32 oracle、STATIC/DYNAMIC 一致性、tuned 扫描、池边界 × B multicast、长尾 G 窗口、内容原位更新、拆分等价、值域负例、CUDA Graph、冷子进程);`tests/test_public_api.py` 锁签名与 4 位置/keyword-only 边界。
