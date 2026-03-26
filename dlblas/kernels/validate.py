import json

import torch
# import torch_mlu
import torch.nn as nn
import numpy as np
import random
import tempfile
import importlib
import inspect
import os

from typing import Dict, Iterator, List, Optional
from pathlib import Path

def sort_key_1(filepath):
    return int(filepath.name.split('_')[0])

def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point:str
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}", flush=True)
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}", flush=True)
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get(entry_point)
    return (Model, get_init_inputs_fn, get_inputs_fn)

def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current mlu device
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    # torch.backends.cudnn.benchmark = False    # Close optimization
    # torch.backends.cudnn.deterministic = True # Close optimization
    # torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def _parse_init_inputs(raw):
    """
    统一把各种 get_init_inputs 返回格式解析成
        (init_args: list, init_kwargs: dict)

    兼容情况
    1. get_init_inputs 返回
       • [ [], {'k': v} ]            -> ([], {'k': v})
       • [a, b, c]                   -> ([a, b, c], {})
       • {'k': v}                    -> ([], {'k': v})
       • 其它对象 obj                -> ([obj], {})

    2. 调用点不小心把 `get_init_inputs` 函数本身传进来，
       本函数会检测到 “可调用 + 无参数” 的情况并自行调用。
    """
    # --------- 如果 raw 还是函数，就尝试调用一次 ---------
    if callable(raw):
        sig = inspect.signature(raw)
        if len(sig.parameters) == 0:   # 只有零参时才安全调用
            raw = raw()

    # --------- 正常解析 ---------
    args, kwargs = [], {}
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and isinstance(raw[0], (list, tuple))
        and isinstance(raw[1], dict)
    ):
        # kernelbook: [positional_list, keyword_dict]
        args, kwargs = list(raw[0]), dict(raw[1])
    elif isinstance(raw, (list, tuple)):
        # kernelbench: 只给一个 list/tuple
        args = list(raw)
    elif isinstance(raw, dict):
        kwargs = dict(raw)
    elif raw is not None:
        args = [raw]

    return args, kwargs


def _move_to_device(obj, device):
    """
    递归地把对象搬到 device，并将 “单元素 Tensor” 转为 Python 标量。

    规则：
      • torch.Tensor 且 numel()==1  -> obj.item()
      • torch.Tensor (其它)        -> obj.to(device, non_blocking=True)
      • list / tuple              -> 保持类型递归处理
      • dict                      -> value 递归处理
      • 其它                      -> 原样返回
    """
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.to(device, non_blocking=True)

    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}

    return obj


class KernelBenchDataset:
    """
    条目示例
    ----------
    {
        "uid":        "level3_layernorm",
        "file_path":  "/abs/path/to/kernelbench/level3/layernorm.py",
        "reference_code": <原始 PyTorch 源码>,
        "metadata":   {"level": "level3"}
    }
    """

    # ------------------------------------------------------------------ #
    # 构造                                                               #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        root: str = "kernelbench",
        _items: Optional[List[Dict]] = None,  # 私有，用于 shard
    ):
        if _items is not None:  # 通过 shard() 创建
            self._items = _items
            return

        self._items: List[Dict] = []
        root_path = Path(root).resolve()

        for level_dir in sorted(root_path.glob("level*")):
            for py in sorted(level_dir.glob("*.py"), key=sort_key_1):
                self._items.append(
                    {
                        "uid": f"{level_dir.name}_{py.stem}",
                        "file_path": str(py),
                        "reference_code": py.read_text(encoding="utf-8"),
                        "metadata": {"level": level_dir.name, "entry_point": "Model"},
                        "file_name": py.stem,
                    }
                )

    # ------------------------------------------------------------------ #
    # 容器协议                                                            #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]

    def __iter__(self) -> Iterator[Dict]:
        return iter(self._items)
    
    def filter_out_completed(self, output_dir: str) -> "KernelBenchDataset":
        """
        过滤掉 output_dir 下已经存在 best 结果的任务条目。
        """
        output_path = Path(output_dir).resolve()
        filtered_items = [
            item
            for item in self._items
            if not (output_path / item["uid"] / "best").exists()
        ]
        return KernelBenchDataset(_items=filtered_items)

    # ------------------------------------------------------------------ #
    # 分片                                                                #
    # ------------------------------------------------------------------ #
    def shard(self, num_shards: int, shard_id: int) -> "KernelBenchDataset":
        """
        把数据集均分成 num_shards 份，返回第 shard_id 份（0-based）。

        采用 round-robin：self._items[shard_id::num_shards]
        """
        if num_shards <= 0:
            raise ValueError("num_shards 必须 > 0")
        if not 0 <= shard_id < num_shards:
            raise ValueError(f"shard_id 必须介于 0~{num_shards-1}")

        sub_items = self._items[shard_id :: num_shards]
        return KernelBenchDataset(_items=sub_items)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # 方便调试
        return f"<KernelBenchDataset len={len(self)}>"



def main():

    # defined here
    device = 'cuda'
    root_path = f"/datapool/zmz/04kernelagent/caizheng/DLBlas-add-kernelbench-triton-gpt5high/dlblas/kernels"
    output_file = f"/datapool/zmz/04kernelagent/caizheng/DLBlas-add-kernelbench-triton-gpt5high/dlblas/kernels/output_{device}.json"
    
    
    # init
    result_list = []
    total_cnt = 0
    correct_cnt = 0
    dataset = KernelBenchDataset(os.path.join(root_path, "kernelagent_original")).shard(1, 0)

    for idx, item in enumerate(dataset, 1):
        total_cnt = total_cnt + 1
        results = {}
        tol = 1e-2
        seed_num=42
        entry_point = "Model"
        
        context = {}
        original_model_src=item['reference_code']
        uid = item['uid'].split('_', 1)
        custom_src_path=os.path.join(root_path, 'kernelagent', uid[0], uid[1]+'.py')

        correctness = True
        
        # extract info
        try:
            with open(custom_src_path, 'r', encoding="utf-8") as f:
                custom_model_src = f.read()
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                original_model_src, context, entry_point
            ) 
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point=entry_point+"New"
            )
            set_seed(seed_num)  # set seed for reproducible input
            # ---------- 解析 get_init_inputs ----------
            raw_init_inputs = get_init_inputs() if get_init_inputs else []
            init_args, init_kwargs = _parse_init_inputs(raw_init_inputs)
            # 把 tensor 放到指定 device
            init_args  = _move_to_device(init_args,  device)
            init_kwargs = _move_to_device(init_kwargs, device)
        except Exception as e:
            print(f"{item['uid']} init with exception: {e}", flush=True)
            correctness = False
        
        # check
        try:
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                original_model = Model(*init_args, **init_kwargs)
                assert hasattr(original_model, "forward")
                original_model=original_model.to(device)
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                custom_model = ModelNew(*init_args, **init_kwargs)
                assert hasattr(custom_model, "forward")
                custom_model=custom_model.to(device)
            inputs = get_inputs()
            inputs = _move_to_device(inputs, device)
            output = original_model(*inputs)
            output_new = custom_model(*inputs)
            outputs = (output,) if not isinstance(output, tuple) else output
            outputs_new = (output_new,) if not isinstance(output_new, tuple) else output_new
            if len(outputs) != len(outputs_new):
                correctness=False
            # 遍历每个输出张量
            for i, (out, out_new) in enumerate(zip(outputs, outputs_new)):
                # 检查形状是否一致
                if out.shape != out_new.shape:
                    correctness=False

                # 检查数值是否一致
                if not torch.allclose(out, out_new, atol=tol, rtol=tol):
                    correctness=False
        except Exception as e:
            print(f"{item['uid']} run with exception: {e}", flush=True)
            correctness=False
        results[item['uid']] = correctness
        result_list.append(results)
        print(f"results:{results}", flush=True)
        if correctness:
            correct_cnt = correct_cnt+1

    pass_rate = correct_cnt/total_cnt
    print(f"{device} pass rate: {pass_rate}", flush=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)
        print(f"\n处理完成！共保存 {len(result_list)} 条数据到 '{output_file}'", flush=True)
    except Exception as e:
        print(f"保存 JSON 失败: {e}", flush=True)

if __name__ == "__main__":
    main()