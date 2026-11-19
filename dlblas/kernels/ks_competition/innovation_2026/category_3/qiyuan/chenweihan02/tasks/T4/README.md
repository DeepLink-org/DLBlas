# T4：MLA RoPE 与压缩 KV Cache 写入

本题融合 RoPE、非位置编码与位置编码数据拼接，以及分页 KV Cache 写入，并按后端能力选择标量或成对实现。共用后端提供带掩码的写入优化、CUDA 直接启动和 CoreX 编译器发现。

- [算子补丁](ntops.patch)：标量和成对算子、Torch 封装、正确性测试、基准评测和评测脚本，源自 `ntops` 提交 `080910d`。
- [技术报告](技术报告.md) · [原始复现说明](原始复现说明.md) · [共享后端补丁](../../shared/ninetoothed-215.patch)

从[交付根目录](../../README.md)执行。第一条只还原本题和共享后端源码；第二条只在远程 Linux GPU 主机执行。两条命令各使用一个新目录。

```bash
bash scripts/prepare.sh T4 /path/to/workspace/t4-source
bash scripts/reproduce.sh --remote-gpu T4 /path/to/workspace/t4-eval
```

原四题合装版本记录了 26 项算子正确性测试和 122 项定向编译器回归通过，计时基线为未修改的 vLLM `RoPE.forward_cuda` 与 `concat_and_cache_mla` 调用链。单题布局本次仅做 Git 还原验证。
