# KernelSwift 算子创新大赛 2026 · 赛道三 · 太初赛道

## 赛题

AI4S 与新型模型架构算子优化 —— 面向水稻基因组学模型 **OneGenomeRice**，在太初元碁 SDAA 平台上完成两个算子的实现与优化，并接入 Teco-vLLM 推理框架：

| 算子 | 说明 |
| --- | --- |
| `reshape_and_cache_flash` | 按 `slot_mapping` 将 key/value 写入 paged KV cache |
| `flash_attn_varlen_func` | 变长序列 FlashAttention，支持 paged KV cache、GQA、causal |

## 平台

太初元碁 SDAA，SDAA C 编程模型（SPE 众核 + SPM 片上内存 + SIMD 向量化）。

## 提交内容

本目录收录两个算子从 userAPI 到设备端 kernel 的完整实现链路，以及 PyTorch 扩展绑定、Plugin 推理接口、精度测试和设计文档。所有文件保留在原始工程中的相对路径，可整体复制回 teco-ops 工程根目录直接编译。

## 目录结构

```
Tony20020629/
├── teco/interface/include/tecoops.h                        # userAPI 声明
├── teco/interface/ops/reshape_and_cache.cpp                # reshape_and_cache 接口实现
├── teco/interface/ops/flash_attention.cpp                  # flash_attn_varlen_func 接口实现
├── teco/ual/args/reshape_and_cache_args.h                  # 参数结构体
├── teco/ual/args/flash_attention_args.h
├── teco/ual/ops/reshape_and_cache/                         # Op 类 + 分支派发
├── teco/ual/ops/flash_attention/
├── teco/ual/kernel/reshape_and_cache/                      # 设备端 kernel（SDAA C）
├── teco/ual/kernel/flash_attention/
├── teco/plugin/pluginReshapeAndCache/                      # Teco-Inference Plugin 接口
├── teco/plugin/pluginFlashAttention/
├── api/torch_ext.cpp                                       # PyTorch 扩展绑定
├── python_api_test/test_reshape_and_cache.py               # Python 精度自测
├── python_api_test/test_flash_attention.py
├── test/zoo/teco/reshape_and_cache/                        # C++ 测试 executor + prototxt 测例
├── test/zoo/teco/flash_attention/
└── doc/op_docs/                                            # 算子设计文档
```

## 实现要点

### reshape_and_cache（fp16）

- 32 个 SPE 按 `tid` 跨步分担 token，单 token 内按 head 分配给多个 SPE
- `slot_mapping[i] < 0` 的 token 直接跳过，不产生写操作
- slot → `(block_id, block_offset)` 在设备端由除余完成，避免 host 侧预计算
- 针对 OneGenomeRice decode 形状（T=1、H=2、D=64）做了 head 级并行切分

### flash_attn_varlen_func

- Q 按 `BM=128` 分块、K/V 按 `BN=32` 分块，逐块载入 SPM，降低 HBM 访问
- online softmax 跨 KV block 累积：float32 累加器 + 行最大值 `m`、行和 `l` 重缩放，全部 KV block 处理完后统一归一化并转 half
- 通过 `block_table` 直接寻址 paged KV cache，无需物化连续的 K/V 张量
- GQA 按 `kv_group = tid * num_kv_heads / num_heads` 映射，一个 KV head 服务多个 query head
- `exp`、行和、float→half 转换走 `floatv16` / `halfv16` 向量化路径
- 覆盖 prefill（`q_len == kv_len`）、decode（`q_len == 1`）、chunked prefill（`q_len < kv_len`）三种计算分支

## 精度验证

```bash
# 构建 PyTorch 扩展
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF python setup.py build_ext --inplace

# 算子级精度自测
python python_api_test/test_reshape_and_cache.py
python python_api_test/test_flash_attention.py
```

`test_flash_attention.py` 覆盖以下场景，逐项与 PyTorch reference 比对：

- `test_prefill` —— 首轮全量计算
- `test_decode` —— 逐 token 推理
- `test_chunked_prefill` —— 分块预填充
- `test_gqa` —— grouped query attention
- `test_multi_batch` —— 多 batch 变长序列

C++ 侧测例位于 `test/zoo/teco/*/test_case/`，通过 teco-ops 内置测试框架执行。

## 性能数据

见 `doc/op_docs/reshape_and_cache.md` 与 `doc/op_docs/flash_attention.md` 中的性能章节。

## 说明

本目录为参赛工程的算子相关子集。完整工程包含构建脚本、C++ 测试框架和其余算子，源码位于太初元碁提供给参赛队伍的 teco-ops 仓库。
