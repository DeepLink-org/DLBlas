# 缓存与并行编译

cute_gemm 在运行期经 CuTe DSL 编译 CUDA kernel。每次编译生成 MLIR IR、下译 PTX、JIT 链接。reduction 类 kernel 约 0.5–1 s，大 GEMM 配置 10 s 以上。一次测试运行或 tune sweep 触发数十到数百次编译（不同 dtype、tile、activation）。朴素串行编译主导墙钟时间。本文说明缓存 + 异步编译的设计。

核心设计：`.o` 缓存未命中时，编译任务交给一组无 GPU 的 CPU worker，并抛出 `CompilePending`。调用方挂起该工作项，先跑别的。待 `.o` 落盘后重试。一套机制同时服务测试（`pytest --async-compile`）、tune sweep 与 CI。

## 第 1 层 `@jit_cache` 装饰器（`cute_gemm/cache/jit.py`）

每个 `_compile_*` 函数（如 `gemm.py` 的 `_compile_gemm`、`softmax.py` 的 `Softmax.compile`）都经 `@jit_cache` 装饰。装饰器提供内存缓存与磁盘缓存：

```python
@jit_cache
def _compile_rmsnorm_fwd(dtype, out_dtype, ..., N, ...):
    # ... 用 cute.sym_int() 的批维构建 fake CuTe 张量 ...
    return cute.compile(RMSNorm(...), ...)
```

这些函数有两条性质。下面的一切都依赖它们：

- 不含张量。参数是可 pickle 的标量（cutlass dtype、int、flag、config dataclass）。torch 张量不跨过这条边界。子进程因此能凭 pickle 的 key 重放一次编译。
- 从不启动 kernel。编译与启动在此边界分离。worker 调用 `_compile_*` 在构造上就碰不到 GPU。

### 内存缓存

每函数一个 `dict`，key 是函数的位置/关键字参数。同一进程内，同一 kernel 配置的第二次调用完全跳过编译与磁盘 I/O。

### 磁盘缓存

内存未命中时，`jit_cache` 检查磁盘上已缓存的 `.o`（目标文件）。磁盘 key 是 `(fn.__qualname__, *args, **sorted_kwargs)`，经 SHA-256 哈希（`cache/jit.py:289`）。缓存目录包含源码指纹：`cute_gemm/*.py` 全部源码 + Python/CUTLASS/TVM-FFI 版本 + 归一化的 codegen target / dispatch arch / torch ABI 的 SHA-256。任何源码变动使整个 `.o` 缓存失效。编译 key 本身与源码无关。因此编辑后重新预热既便宜又可并行。跨模块的同名 qualname 不做区分。

cutlass-dsl >= 4.4.2 时，`tvm_ffi` 直接加载 `.o`（约 1 ms）。没有 `.o` → `.so` 链接步骤，也不依赖 distutils。

### 未命中流程

```
@jit_cache wrapper(dtype, out_dtype, N)
  ├─ 1. 内存 dict 查找 → 命中 ✓（约 0 开销）
  ├─ 2. hash (fn.__qualname__, *args) → SHA-256 hex
  ├─ 3. 共享锁：.o 存在 → 经 tvm_ffi 加载（约 1 ms）✓
  ├─ 3b. 异步池已激活？
  │     ├─ 别的进程持有该 key 的 flock → 标记 external，抛 CompilePending
  │     ├─ key 尚未提交 → 提交给池，      抛 CompilePending
  │     ├─ 池报 "pending"              → 抛 CompilePending
  │     ├─ 池报 "done" → 共享锁，加载 .o ✓
  │     └─ 池报 "failed" → 落回（进程内编译，真实 traceback）
  └─ 4. 排他锁：fn(...) → cute.compile → export_to_c() → 写 .o
```

### 并发安全

多个进程可能同时编译同一 key（并行测试 worker、池 worker、tune sweep）。`FileLock`（`fcntl.flock`）串行化访问：

- 读用共享锁：多个读者并发加载 `.o`。
- 写用排他锁：一个写者编译 + 导出；其余等待后加载。
- 拿到排他锁后二次检查：别的进程可能已抢先写完。
- 锁文件兼作跨进程“编译进行中”信号。非阻塞探测（`_flock_held_exclusively`）让消费者对别的进程正在编译的 key 挂起。因此不浪费池位做重复编译。

### 线程 / 进程 / 机器的并发模型

| 范围 | 是否安全 | 机制 / 注意 |
|---|---|---|
| 同一进程内的线程 | 同 key 经 `flock` 去重；无跨线程发布保证 | `cute.compile()` 非线程安全，并行 CuTe 编译按契约不支持。锁超时回落在进程内编译且不写盘。临时文件名是 `.o.tmp.<pid>`：同进程两个线程同时超时会共用该名。 |
| 同一机器上的进程 | 冷 key 在排他锁持有期内只编译一次 | 每 key 排他 `flock` + 锁内复查，正常每冷 key 一次编译。超过 60 s 锁超时后，调用方落回进程内编译且**不写盘**——该 key 下次冷启动仍会重编。导出先写私有 `.o.tmp.<pid>`，再 `os.replace` 原子落位（导出中途被杀的进程不会留下截断的 `.o`）。 |
| 经网络文件系统共享 `DLKERNEL_CACHE_DIR` 的机器 | 未加固 | 默认每主机 `/tmp/$USER/dlkernel_cache`，机器不会意外共享。没有 no-clobber 发布或跨主机校验。 |
| 共享同一 winner 库的机器 | 尽力而为 | winner JSON 在 `flock` 锁下写入（autotuner `check_disk_cache`）。共享 `$HOME` 且网络文件系统锁不可靠时，残余风险是丢一个/坏一个 winner JSON（下次运行重测），不影响 kernel 正确性。 |

### 环境开关

| 变量 | 默认 | 说明 |
|---|---|---|
| `DLKERNEL_CACHE_ENABLED` | `1` | `=0` 关闭磁盘缓存（内存缓存仍生效） |
| `DLKERNEL_CACHE_DIR` | `.o`：`/tmp/$USER/dlkernel_cache`；winner：`~/.dlkernel/cache` | 同时覆盖 `.o` 根（`cache.get_cache_path()`）与 tune winner 根（`autotuner.FileCacheManager`）。没有独立的 artifact 旋钮。 |

## 第 2 层 异步编译池（`cute_gemm/cache/async_compile.py`）

### 约束

1. **`cute.compile()` 非线程安全**——MLIR 使用线程局部状态。并行必须基于进程。
2. **CUDA 初始化后 `fork()` 不安全**——消费进程持有 CUDA context 与线程。fork 它是未定义行为。
3. **编译产物不可 pickle**——worker 无法经管道把结果发回。

### `CompilePending` 与挂起重试机制

池激活时，`.o` 未命中不在进程内编译。`jit_cache` 把 `_compile_*` 的 `(module, qualname, args, kwargs)` pickle 后提交给池，并抛出 `CompilePending(sha)`。调用方捕获它，挂起该工作项，先跑其他工作。待 `pool.poll(sha)` 报 done 后重试。重试命中 `.o` 快路径（约 1 ms）。约束 3 借持久缓存本身解决：worker 的唯一输出就是 `.o` 文件。

`CompilePending` 继承自 `BaseException`（同 `KeyboardInterrupt`）。用户/测试代码里的 `except Exception` 因此无法吞掉挂起信号、把未运行的工作变成假通过。只有挂起循环捕获它。

两类调用方实现该循环：

- pytest（`cute_gemm/testing/pytest_plugin.py`，`--async-compile[=N]`）：测试是工作项。测试挂起后，插件丢弃报告，稍后重跑该测试。
- autotuner（`Autotuner.__call__` → `pool_scope()` 内的 `benchmark()`）：候选配置是工作项。冷配置轮转到 bench 队尾；哪个配置就绪就测哪个。

两个循环共用失败/终止规则：

- **池的结果不可信**：poll 报 failed 时回退到进程内编译。真实异常带着本地 traceback 浮出。
- 强制同步上限：挂起超过 `_MAX_ATTEMPTS`（20）次，或某个 sha 超过卡死期限，该条目在 `suppress_pool()` 下进程内编译。期限取值：autotuner `_POOL_WEDGE_TIMEOUT_S` = 300 s，pytest `_WEDGE_TIMEOUT_S` = 600 s。卡死的 worker 或挂死的外部 flock 持有者都无法拖死整个运行。

### worker 经 forkserver sidecar 启动

架构同 PyTorch Inductor 编译 worker 的 `SubprocPool`。一个 sidecar 进程付一次重导入；worker 从它 copy-on-write fork。

- `_make_executor()` 在 `forkserver` context 上构建 `ProcessPoolExecutor`，并 `set_forkserver_preload(["cute_gemm.cache._pool_preload"])`。preload 一次导入 torch + cutlass + tvm_ffi（约 13 s）；每个 worker fork 约 0.1 s。实测：130-key 冷运行的池 CPU 开销从约 11 CPU-min（spawn，32 worker × 各自导入）降到约 1 CPU-min。
- worker 无 GPU。preload 用 `nvidia-smi --query-gpu=compute_cap` 探测能力（不建 CUDA context）。据此固定 `DLKERNEL_ARCH` / `CUTE_DSL_ARCH`，并设 `CUDA_VISIBLE_DEVICES=""`。显式 `DLKERNEL_ARCH` 环境变量优先。CI 交叉编译（如 H100 上 `DLKERNEL_ARCH=120`）时，worker 必须面向所请求的 arch，而不是物理 arch。sidecar 从不初始化 CUDA。
- `_neutral_main()`：multiprocessing child 准备会重执行用户 `__main__` 脚本（`runpy.run_path`），以解析引用 `__main__` 的 pickle。本仓库的任务从不引用 `__main__`；顶层带 CUDA 工作的用户脚本会在 spawn 时杀死所有 worker。executor worker 在 `submit` 内同步启动。因此，提交期间用一个空 stub 覆盖 `sys.modules["__main__"]` 即可。（Inductor 用专门的 worker 入口解决同一问题。）
- `prewarm()`（Inductor `warm_pool()` 的对应物）：pytest 插件在 `pytest_configure` 提交一个 no-op。sidecar 的约 13 s 导入与用例收集重叠，而不是占第一个冷编译。实测：基准测试集的会话内冷时间 38 s → 29 s。

### 池 API

| | |
|---|---|
| `activate(jobs)` / `deactivate()` / `get_active_pool()` | 会话级池（pytest 插件） |
| `pool_scope()` | 作用域激活（autotuner）；复用活动池或包住共享 executor；`CompilePending` 不会逸出该块进入无关代码 |
| `suppress_pool()` | 块内让 `get_active_pool()` 返回 None——强制同步的逃生口 |
| `CompilePool.submit / poll / mark_external / prewarm / stats` | 按 sha 记账；`poll` 状态：`new / pending / done / failed` |
| `get_shared_executor()` | 进程级共享 executor（`DLKERNEL_COMPILE_WORKERS`，默认 8） |

| 变量 | 默认 | 说明 |
|---|---|---|
| `DLKERNEL_ASYNC_COMPILE_START` | `forkserver` | 设为 `spawn` 可关闭 fork sidecar |
| `DLKERNEL_COMPILE_WORKERS` | `8` | 共享 executor 大小（pytest 之外的 tune sweep 用） |

## 第 3 层 tune（`cute_gemm/autotuner.py`）

tune 子系统的完整设计见 `autotuning_subsystem.md`；本文只讲它与缓存/编译池的交点。

### 结果缓存

autotuner 测量多个配置（如 SM90 GEMM 的 44 个）取最快。结果经 `FileCacheManager`（Triton 的缓存基础设施）以 JSON 写盘。JSON key 含包版本、tuning key（张量元数据）与各配置字符串。

| 变量 | 默认 | 说明 |
|---|---|---|
| `DLKERNEL_CACHE_AUTOTUNING` | unset | `=1` 时让 `cache_results=False` 的 tuner 也写盘 |
| `DLKERNEL_FORCE_CACHE_UPDATE` | unset | `=1` 忽略磁盘结果重测；`"0"`/`"false"`/`"no"`/`"off"` 不触发 |

### 编译/测量重叠

没有单独的预编译阶段。bench 循环跑在 `pool_scope()` 内：

```
with pool_scope() as pool:
    queue = deque(pruned_configs)
    while queue:
        config = queue.popleft()
        if awaiting sha still pending (and not past wedge deadline): rotate; continue
        try:
            timings[config] = self._bench(*args, config=config, **kwargs)
        except CompilePending as e:
            remember e.sha; rotate config to the back
```

key 发现用真实张量、在进程内完成：冷配置的第一次 bench 经正常 wrapper 到达 `_compile_*`；key 交给池；循环转测已就绪的配置。总墙钟 = max(并行编译，串行 bench)。`CompilePending` 只能在 CUDA graph 捕获之外抛出：L2-cold bench 在捕获前做普通预热发射，冷 key 在第一次发射处抛出。

编译真失败的配置在 `_bench` 内回退到进程内编译。`_bench` 的 `(RuntimeError, MemoryError)` 处理器记 `float("inf")`。失败配置不会赢，也不会用编译耗时污染计时。全部候选都失败时，sweep 抛 `RuntimeError`，而不是缓存一个空 winner。

## 单遍测试工作流（`cute_gemm/testing/pytest_plugin.py`）

```bash
# 一条命令，冷热缓存皆可：
pytest tests/test_softmax.py --async-compile=16
pytest tests/ -n 8 --dist worksteal --async-compile=32     # 多 GPU xdist
```

冷运行成本 ≈ 热运行 + 编译关键路径（实测：398 测试 / 130 key 的集合，热 26 s，会话内冷 29–39 s）。热运行不提交任何任务，只付一个后台空闲 sidecar 导入。

### 单进程模式（`_SingleProcDeferLoop`）

用 deque 替换 `pytest_runtestloop`。挂起的测试轮转到队尾。报告丢弃，不记日志。其 sha 解决后重试。轮转是 poll，不是重跑。重试之前，测试不会再次执行。fixture teardown 的作用域由 `_run_protocol` 的误预测防护处理。`runtestprotocol` 把 teardown 作用于*预测的*下一个 item。挂起打断预测链，pytest 会对不匹配断言（"previous item was not torn down properly"）。防护在"实际运行的 item ≠ 预测"时强制完整 teardown。

### xdist 模式（`_XdistWorkerDefer`）

xdist 的 `WorkerInteractor` 拥有 runtestloop，但经 hook 调用 `pytest_runtest_protocol`。插件以 tryfirst + firstresult 接管该协议：

- 插件暂存挂起的 item；worker 仍发送 `runtest_protocol_complete`。master 把 item 记为完成（顺序无关的 `.remove()`），并继续流入新 item。跳先行因此免费获得。
- 暂存 item 在流入 item 的间隙机会性重试。xdist 内层循环退出后，插件在 `pytest_runtestloop` hookwrapper 里排空（channel 仍开，迟到报告正常流动）。迟到报告临时改指 `WorkerInteractor.item_index`（worker 断言报告与 index 一致）。
- `-x` / `--maxfail` 中止时跳过排空。
- 挂起时廉价抑制 call 阶段（`force_result`）。每次挂起省下约 100 ms 的失败 `longrepr` 格式化。但 setup 阶段的 `CompilePending` 仍按失败处理，pytest 因此跳过建立在半成品 fixture 上的测试体。挂起尝试中的 teardown 错误仍会上报。

### 会话末完整性检查

提前上报的技巧留下一个洞。带着暂存 item 崩溃的 worker 会*静默*丢掉它们（master 已计为完成）。`pytest_sessionfinish` 时，插件对 collected − reported − deselected 的 nodeid 集合求差。有缺口时打印丢失的测试，并把退出状态翻为失败。

### CI（`.github/actions/gpu-test/action.yml`）

单遍：`pytest tests/ -n $NUM_GPUS --dist worksteal --async-compile=24`。每 runner 的 `DLKERNEL_CACHE_DIR`（`.dlkernel_cache_${CUDA}_sm$ARCH`）跨运行携带 `.o` 与 tune winner。跨 arch 的 CI 腿设置 `DLKERNEL_ARCH`（池的 arch 固定遵从它）。

## `custom_op` 的 fake 是纯 no-op

所有 kernel 经 `cute_op` 装饰器使用 `@torch.library.custom_op`（`cute_gemm/dsl/torch_library_op.py`）。注册的 fake 是纯 no-op。本仓库的 op 只改写入参数，Dynamo / AOT autograd 不需要形状效应。在追踪期运行 op body 会在 trace 时付出编译延迟（或对 kernel 有意拒绝的 shape/dtype 组合直接崩溃）。kernel 编译完全由 `jit_cache` + 池在真实执行期负责。（少数*返回*张量的 op——如 `gemm_rms_out`——保留一个仅算形状的 minimal fake。）

## 完整数据流

### 正常执行（训练/推理）

```
用户调用 gemm(A, B, ...)
  └─ gemm_out（custom_op → CUDA 实现）
       └─ gemm_tuned（autotuner）
            ├─ JSON 缓存命中 → 用最优 config 调 fn
            └─ 未命中 → pool_scope() 内 benchmark()
                 ├─ 冷配置 → CompilePending → 轮转；池编译 .o
                 ├─ 热配置 → bench（编译/测量重叠）
                 └─ 缓存最优 config
            └─ fn(A, B, out, config=best)
                 └─ _compile_gemm(...) → @jit_cache（.o 命中，约 1 ms）
                      └─ compiled_fn（真实 A/B/D 指针, ...）
```

### `pytest --async-compile`（冷缓存）

```
pytest --async-compile=32
  └─ pytest_configure: activate(pool), pool.prewarm()   # sidecar 导入与收集重叠
       └─ 测试运行，kernel 分发 → jit_cache 未命中
            ├─ 把 pickle 的 (_compile_*, args) 提交给池 → 抛 CompilePending
            ├─ 挂起循环：测试轮转到队尾，跑其他测试
            ├─ 池 worker（fork 出、无 GPU）：_compile_*(...) → 导出 .o
            └─ 重试：.o 快路径（约 1 ms）→ kernel 发射 → 断言执行
  └─ pytest_sessionfinish: collected 对 reported 的完整性检查
```

## 关键文件

| 文件 | 职责 |
|------|------|
| `cute_gemm/cache/jit.py` | `@jit_cache`（内存 + `.o` 磁盘缓存）、`FileLock`、池交互（步骤 3b） |
| `cute_gemm/cache/async_compile.py` | `CompilePending`、`CompilePool`、forkserver sidecar、`pool_scope`、`suppress_pool`、`_neutral_main` |
| `cute_gemm/cache/_pool_preload.py` | forkserver preload：arch 固定（无 CUDA）、torch/cutlass 导入 |
| `cute_gemm/testing/pytest_plugin.py` | `--async-compile` 旗标、单进程 + xdist 挂起循环、完整性检查 |
| `cute_gemm/autotuner.py` | `pool_scope` 内的 bench 循环（编译/测量重叠）、`FileCacheManager` 结果缓存 |
| `cute_gemm/dsl/torch_library_op.py` | `cute_op`：带纯 no-op fake 的 custom_op |
| `tests/test_async_compile.py`、`tests/test_autotuner.py`、`tests/test_cache.py` | 编码上述失败模式的回归测试 |

## 性能特征与冷启动

测量对象：MetaMoE 工作负载的纯 CPU 剖面（每 rank 250,001 令牌，m-group/k-group），2026-09-22。证据保存在 `../../.ecc/benchmarks/metamoe_compile_20260922/`。纯编译运行不发射 GPU kernel，不做 tune bench，不做数值验证。环境：`CUDA_VISIBLE_DEVICES=''`、`DLKERNEL_ARCH=90`、`CUTE_DSL_ARCH=sm_90a`（目标 SM90a）；Python 3.12、PyTorch 2.9.1+cu128、CuTe/CUTLASS DSL 4.7.1。工作负载为 BF16，H=2048、I=768、EP=8（每 rank 768 专家）。六条 GEMM 路径：GU 与 Down，各以 NT / NN 的 m-group 形式与 TN 的 k-group 形式。每路径 8 个候选 shortlist，共 48 个编译请求。

- 冷编译墙钟（24 个不同编译 key，每轮全新缓存目录；不含主进程约 2.7–3.9 s 导入与约 6–8 ms 源码指纹，含首次提交时的 forkserver/worker 启动）：

  | 实验 | 24 个不同 key 的墙钟 | 备注 |
  |---|---:|---|
  | 串行 | 24.01 s；30.08 s | 后者 = 全部 48 个请求，后 24 个为内存命中 |
  | 4 worker，一次性提交 | 9.98 / 9.52 s | 两轮 |
  | 8 worker，一次性提交 | 7.29 / 6.84 / 7.03 s | 三轮 |
  | 16 worker，一次性提交 | 5.85 / 5.91 s | 两轮；不是每轮 8 候选的模式 |

  8-worker 池的首次提交阻塞约 2.0–2.4 s。按每算子 8 个 key 分批提交（autotune 的真实形态：先 NT，再 NN，再 TN）时，16 worker 无收益。8 与 16 worker 上限都只启动了 8 个 worker。第 2/3 批分别跑 1.256 s / 1.458 s 对 1.256 s / 1.508 s。`DLKERNEL_COMPILE_WORKERS` 超过默认 8 在真实候选数下无稳定收益。16 worker 一次性提交的胜出把任务累计 CPU 时间从约 35 s 提高到约 47 s，worker 峰值 RSS 约 654–666 MiB。
- 串行运行的分阶段拆解（基线：实际调用时间 24.006 s，低开销函数边界计时）：

  | 阶段 | 累计 | 约占比 |
  |---|---:|---:|
  | MLIR 编译流水线（含二进制生成） | 13.451 s | 56.0% |
  | Python/DSL 生成原始 IR | 5.061 s | 21.1% |
  | 导出 `.o` | 2.138 s | 8.9% |
  | 创建 TVM-FFI 函数 | 1.585 s | 6.6% |
  | Host JIT engine 各阶段 | 0.622 s | 2.6% |
  | 其余包装 / 元数据 | 约 1.15 s | 4.8% |

  `run_preprocessor`（0.222 s）已含在 IR 生成内，不要重复计入。m-group 的 NT/NN 两批各 7.615 / 7.654 s；k-group 的 TN 一批 8.737 s。对一个 k-group 192×128 pingpong 候选开 SDK MLIR pass 计时：`CudaToBinaryPass` 占 0.897 s 流水线中的 0.765 s（85.3%）。这是单候选占比，不能外推到全部配置。
- 该工作负载编译 24 个不同二进制（`.o`），不是 8 个。GU 与 Down 共享同一套 per-group key 集合，而 NT、NN、TN 三套 key 互不相交。同一个 192×128 tile / 1×2 cluster / 无 swap 配置按家族带着不同的 a_major / b_major / d_major 与 varlen_m / varlen_k 各编一次（每家族 8 个特化，3 个家族）。跨形状去重已生效：48 个候选请求在完全冷的缓存上编 24 个对象；全热磁盘上编 0 个。
- 路径查找：SDK `cutlass.runtime.load_module` 每次加载都调 `find_runtime_libraries()`，每次又重跑 `tvm_ffi.libinfo.find_libtvm_ffi()`。24 次加载里，重复的库路径发现共 7.33 ms。缓存依赖库列表（仅诊断进程内的 `lru_cache`）把每 48 请求一轮的路径发现从 7.44 / 8.90 ms 降到 0.52 / 0.55 ms。"48 探测 ≈ 40 ms"是 `meta_disk48` 诊断循环的初始总值（39.98 ms）。总值含每请求 JSON 写与打印（函数内 35.15 ms，循环开销约 4.83 ms）。它不是单次探测成本；单次探测未测。纯 host 开销，每个新进程在跑任何 kernel 前都要付。
- host 侧 `.o` 加载 + cache key 查找（专用只加载剖面脚本测量）：新进程加载 24 个既有对象共 29.36 / 34.94 ms（两轮；24 次磁盘加载 + 24 次内存命中，零编译）。分项：

  - 执行引擎创建 12.03 ms（最大份额）
  - TVM-FFI 库路径查找 7.33 ms
  - 符号查找/函数绑定 4.66 ms
  - pickle + SHA-256 key 计算 0.48 ms
  - 文件锁 open/acquire/release/close 0.49 ms

  24 个对象共 1,420,672 字节，热页缓存下仅读文件约 0.43 ms。成本在引擎创建与绑定，不在 I/O。加载完成后，纯内存命中每次请求约 0.67–0.91 µs（48 请求一轮的中位数约 32–44 µs，100 轮）。

已有机制缓解的部分：

- 第 1 层 `.o` 磁盘缓存消除热启动的重编译（全热磁盘 0 次编译）。
- 异步编译池把剩余冷编译与测试执行重叠（24 key：串行 24.01 s → 8 worker 一次性提交 6.84–7.29 s）。
- autotuner 的 shortlist 把每路径控制在 8 个候选。

固有开销：

- 全冷进程仍要付主进程导入约 2.7–3.9 s。
- 编译工作本身：阶段表每一行（MLIR 流水线 56.0%、IR 生成 21.1%、`.o` 导出 8.9%）都是每冷 key 的 CPU，池只是拿它与其他工作重叠。
- 关键路径上首次提交的池启动约 2.0–2.4 s。
- 每个新进程的加载：24 个对象约 29–35 ms，其中 7–9 ms 是重复的 TVM-FFI 库路径发现，现有机制不缓存它。
