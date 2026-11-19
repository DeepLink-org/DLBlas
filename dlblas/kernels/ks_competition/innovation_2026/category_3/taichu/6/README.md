# 太初赛道 · OneGenomeRice · 6

本目录提交参赛仓库 `teco-ops-6` 的算子实现快照，对应提交 `7fc2e090ebd032d2396768558e2dbf6a3ec7146f`。

## 实现内容

### `reshape_and_cache`

将 FP16 的 K/V 按 `slot_mapping` 写入分页 KV cache，支持 packed QKV 切分后的 key/value 步长、单 token append 以及 prefill 的块内连续写入。

- 按形状自动分支：原始逐头 memcpy（分支 0）、K/V 合并 memcpy（分支 1）、16-wide `halfv16` SIMD 写（分支 2）、整块 `memcpy_stride` 拷贝（分支 3）。
- 整块路径增加单次整块 memcpy 与 SPM 中转的异步整块拷贝，用于 prefill 的块连续 `slot_mapping`；非整块场景回退逐 token 路径，结果逐位一致。
- decode 的固定形状（`num_tokens=1`、`kv_heads=2`、`head_size=64`、`block_size=32`）绕过通用分支选择，直接启动已选中的 SIMD kernel。
- 相关实现：`teco/interface/ops/reshape_and_cache.cpp`、`teco/ual/args/reshape_and_cache_args.h`、`teco/ual/kernel/reshape_and_cache/`、`teco/ual/ops/reshape_and_cache/`。

### `flash_attn_varlen_func`

在 SDAA 上执行分页 KV cache 的变长注意力（GQA、causal、head size 64），包含四个阶段：prefill（phase 0）、split-KV decode（phase 1）、partial 归约（phase 2）、K 预转置（phase 3）。

- 分页 KV cache、GQA（含 pack-2/pack-4 多头打包）、causal mask 与前后缀共享的 varlen 布局。
- prefill：BM=256 大块、2/4 页 KV 批处理、K 预转置 scratch、异步 K 双缓冲、softmax 批量 `exp128v2`、rescale 16 行向量化跳读、QK/softmax 分组流水、V 转置散写与 QK 重叠。
- decode：split-KV 两 kernel 路径、连续 block 的 2 块合并扫描、2 次迭代 DMA 预取、partial 批量取回、split-16/32 的 q=1/D=64 专用打包路径，以及把 phase-2 合并折叠进 phase-1 kernel 尾部的单次提交路径。
- 相关实现：`teco/interface/ops/flash_attention.cpp`（含 q=1/D=64 raw 入口）、`teco/ual/args/flash_attention_args.h`、`teco/ual/kernel/flash_attention/`。

### Python / PyTorch 集成

`teco-ops-integration.patch` 修改 `api/torch_ext.cpp`：新增 q=1/D=64 的 raw 调用入口，decode 形状复用 tensor descriptor，环境开关只读取一次，并加入 host 侧调度与跨算子提交协调逻辑。

## 交付清单

```
6/
├── LICENSE
├── README.md
├── teco-ops-integration.patch        # 相对赛事模板的非算子目录改动（api/torch_ext.cpp）
├── teco/                             # 算子实现（保持 teco-ops 相对路径）
│   ├── interface/ops/{flash_attention,reshape_and_cache}.cpp
│   └── ual/{args,kernel,ops}/...
├── doc/op_docs/reshape_and_cache.md  # 更新后的算子设计文档
└── tools/validate_b2_d64.py          # batch1/q1/qh4/kvh2/d64/block32 精度校验脚本
```

其余文件（公共头文件、测试框架、已有测例）沿用赛事模板。

## 在 teco-ops 中构建

交付基于赛事模板提交 `38cbdbf2c092458cfd6961490f0cd882c10fec06`。在具备 SDAA SDK 与 PyTorch SDAA 扩展的环境中，从该模板开始，将 `submission` 设置为本目录的绝对路径：

```bash
submission=/path/to/6
git apply "$submission/teco-ops-integration.patch"
rsync -a "$submission/teco/" teco/
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF python setup.py build_ext --inplace
```

补丁集成共享的 C API 与 PyTorch 扩展；算子实现与文档保留模板中的相对路径。

## 验证

```bash
python python_api_test/test_flash_attention.py
python python_api_test/test_reshape_and_cache.py
python tools/validate_b2_d64.py
```

`tools/validate_b2_d64.py` 覆盖 seq_len 1024/1025/4095/8191/16383/32767 的 decode 精度（max_abs ≤ 3e-3、mean_abs ≤ 5e-4）。SDAA 编译与运行需在硬件环境中完成。

## 环境开关

实现在生产路径上默认开启，保留以下开关用于同构建 A/B（`0` 关闭）：

| 开关 | 作用 |
| --- | --- |
| `TECOOPS_RAC_BSMEM` / `TECOOPS_RAC_SPM` / `TECOOPS_RAC_BLOCKED` | `reshape_and_cache` 整块/SPM/块拷贝路径 |
| `TECOOPS_FA22_HEXP` | softmax 批量 half `exp128v2` |
| `TECOOPS_FA31` / `TECOOPS_FA33` / `TECOOPS_FA40` | decode 异步 K ping-pong、2 块合并扫描、DMA 预取 |
| `TECOOPS_FA35` / `TECOOPS_FA44` / `TECOOPS_FA47` | partial 批量取回、rescale 跳读、prefill 4 页批处理 |
| `TECOOPS_FA48X` / `TECOOPS_SPLIT_KV` | prefill fa48x 链、split-KV decode |
| `TECOOPS_NUMA_AFFINITY` | host 侧加速卡本地 CPU 绑定（`0` 关闭） |
| `TECOOPS_RAC_DIAG` | `reshape_and_cache` 计时探针（诊断用，结果不保证正确） |
