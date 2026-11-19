# T1：MXFP4 W4A16 分组专家 GEMM

本题融合 MXFP4 解码、专家路由和分块累加，支持 BF16/FP16 输入及非均衡路由。共用后端提供 `bitcast`、`interleave`、转置 lowering 和 Triton 启动参数直接绑定。

- [算子补丁](ntops.patch)：算子、Torch 封装、正确性测试、基准评测和评测脚本，源自 `ntops` 提交 `ec8f4e1`。
- [技术报告](技术报告.md) · [原始复现说明](原始复现说明.md) · [共享后端补丁](../../shared/ninetoothed-215.patch)

从[交付根目录](../../README.md)执行。第一条只还原本题和共享后端源码；第二条只在远程 Linux GPU 主机执行。两条命令各使用一个新目录。

```bash
bash scripts/prepare.sh T1 /path/to/workspace/t1-source
bash scripts/reproduce.sh --remote-gpu T1 /path/to/workspace/t1-eval
```

原四题合装版本记录了 15 项算子正确性测试和 98 项定向编译器回归通过。计时基线是 `vllm_semantic_fallback_predecoded`，即报告披露的 vLLM/OCP 语义兼容实现，不是未修改的上游 vLLM 二进制。单题布局本次仅做 Git 还原验证。
