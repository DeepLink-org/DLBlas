# tcdd · 太初 SDAA 算子实现

2026 KernelSwift 算子创新大赛，赛道三太初赛道。队伍为 `tcdd`，提交者为 `KivingY`。
针对 OneGenomeRice 的 SDAA 推理优化了两个算子，沿用 tecoops 的原始调用接口：

- `flash_attn_varlen_func`：变长 Flash Attention 前向，包含 GQA 解码路径、K/V 预取和融合 softmax/PV。
- `reshape_and_cache_flash`：KV cache 写入，包含单 token 解码写入路径。

## 源码与版本

本目录保留两个算子的设备内核、C++ 调用层、PyTorch 绑定和构建文件的原始路径。九个源码与构建文件来自比赛正式提交，保留了原有 BSD 3-Clause 许可和版权声明。

| 项目 | Git 标识 |
| --- | --- |
| 比赛基线（主办方同步 Flash Attention 后的版本） | `32e6dba9de895bb6844e5aeb9305b2bb87e908ae` |
| 比赛正式提交 | `a53860e315056365d13129dc3b0a419fe0cb709d` |
| 正式提交的源码树 | `fcb47d022651bf71a4c77d7a207bfcfe776ec233` |

来源为 [tcdd 的比赛提交](https://github.com/tecorigin-waic/teco-ops-tcdd/commit/a53860e315056365d13129dc3b0a419fe0cb709d)。

## 在太初环境复现

在 tecoops 比赛仓库中，以基线提交建立一个新的工作树，将 `submission_dir` 设为本目录的绝对路径。在赛事提供的 SDAA、PyTorch 和 torch_sdaa 环境中执行：

```bash
git worktree add --detach ../tcdd-repro 32e6dba9de895bb6844e5aeb9305b2bb87e908ae
cd ../tcdd-repro
cp -R "$submission_dir/api" "$submission_dir/teco" "$submission_dir/setup.py" .
git add -u
test "$(git write-tree)" = fcb47d022651bf71a4c77d7a207bfcfe776ec233
source /opt/tecoai/setvars.sh
source env.sh
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/tecoops-*.whl
python -m pytest -q python_api_test/test_flash_attention.py \
  python_api_test/test_reshape_and_cache.py
```

公共头文件、其余构建依赖和接口测试沿用 tecoops 比赛基线。本目录供源码展示和在 SDAA 环境中复现。
