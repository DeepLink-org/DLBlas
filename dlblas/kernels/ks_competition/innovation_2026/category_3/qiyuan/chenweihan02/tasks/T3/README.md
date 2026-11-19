# T3：门控 RMSNorm

本题融合 RMSNorm、门控激活、仿射变换和残差更新，并针对静态语义变体专门化。共用后端提供静态布局、标量参数支持和预验证的启动路径。

- [算子补丁](ntops.patch)：算子、Torch 封装、正确性测试、正式基准和运行时消融评测，源自 `ntops` 提交 `7fc8074`。
- [技术报告](技术报告.md) · [原始复现说明](原始复现说明.md) · [共享后端补丁](../../shared/ninetoothed-215.patch)

从[交付根目录](../../README.md)执行。第一条只还原本题和共享后端源码；第二条只在远程 Linux GPU 主机执行。两条命令各使用一个新目录。

```bash
bash scripts/prepare.sh T3 /path/to/workspace/t3-source
bash scripts/reproduce.sh --remote-gpu T3 /path/to/workspace/t3-eval
```

原四题合装版本记录了 32 项算子正确性测试和 94 项定向编译器回归通过，计时基线为未修改的 vLLM `RMSNormGated` 调用链。单题布局本次仅做 Git 还原验证。
