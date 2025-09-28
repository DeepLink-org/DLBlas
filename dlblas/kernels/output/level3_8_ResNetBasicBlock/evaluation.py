"""
evaluation_template.py  · 2025-07-28

OpenEvolve 调用顺序
├─ 若 cascade_evaluation = true
│     stage-1 → stage-2
└─ 否则
      evaluate
"""

from __future__ import annotations

import os
import logging
import re
from pathlib import Path
from typing import Sequence

from remote_speedup_evaluator import RemoteSpeedupEvaluator

logger = logging.getLogger("openevolve.Evaluator")

# ────────────────── 常量 ──────────────────
_REF_CODE  = Path('/cpfs01/shared/llm_kernel/hujiakai/KernelLLM/openevolve/triton_evolve/kernelbench/level3/8_ResNetBasicBlock.py').read_text(encoding="utf-8")
_UID       = 'level3_8_ResNetBasicBlock'
_TEMPLATE  = Path("kernel_template.py").read_text(encoding="utf-8")
_EVAL      = RemoteSpeedupEvaluator()

EVOLVE_START = "# ================== EVOLVE-BLOCK-START =================="
EVOLVE_END   = "# =================== EVOLVE-BLOCK-END ==================="

# ────────────────── 工具函数 ──────────────────
def extract_last_code(text: str, langs: Sequence[str]) -> str | None:
    """
    抽取最后一个 ```fenced-code``` block。
    若无，返回 None。
    """
    matches = list(re.finditer(r"```(.*?)```", text.strip(), re.DOTALL))
    if not matches:
        return None
    code = matches[-1].group(1).strip()
    for lang in langs:
        if code.lower().startswith(lang.lower()):
            code = code[len(lang):].strip()
    return code or None


def inject_kernel(template: str, kernel: str) -> str:
    if EVOLVE_START not in template or EVOLVE_END not in template:
        raise ValueError("template missing EVOLVE markers")
    head, _ = template.split(EVOLVE_START, 1)
    _, tail = template.split(EVOLVE_END,   1)
    return f"{head}{EVOLVE_START}\n{kernel.rstrip()}\n{EVOLVE_END}{tail}"


def _normalize(program_text: str) -> str:
    """
    1. 若 program_text 为文件路径 → 读文件内容
    2. 若貌似完整 .py 源码           → 原样返回
    3. 否则视作 LLM 原始回复，取最后一个 fenced-code block
         a. 若 block 是完整 .py     → 用它
         b. 若仅 Triton kernel      → 注入模板
    """
    pt = program_text.strip()

    # ① 文件路径
    if os.path.isfile(pt):
        return Path(pt).read_text(encoding="utf-8")

    # ② 看起来已是完整源码
    if pt.startswith(("import", "from", "def", "class", "#")):
        return pt

    # ③ 解析 LLM 回复
    code = extract_last_code(pt, ["python", "py", ""])
    if code is None:
        logger.info("[%s] No code block found in LLM output", _UID)
        return pt

    if "import triton" in code or code.count("def ") > 1:
        logger.info("[%s] Use extracted full program", _UID)
        return code
    else:
        logger.info("[%s] Inject extracted kernel into template", _UID)
        return inject_kernel(_TEMPLATE, code)


# ────────────────── 单阶段接口 ──────────────────
def evaluate(program_text: str, check_triton=True):
    """
    若未启用 cascade_evaluation，OpenEvolve 调此函数。
    """
    prog = _normalize(program_text)
    return _EVAL.evaluate(_UID, _REF_CODE, prog, check_triton=check_triton)


# ────────────────── cascade_evaluation 接口 ──────────────────
def evaluate_stage1(program_text: str):
    """
    Stage-1：编译 + 正确性检查，不计分。
    """
    prog = _normalize(program_text)
    res  = _EVAL.evaluate(_UID, _REF_CODE, prog)
    logger.info("[%s] Stage-1 program_text:\n%s", _UID, prog)
    logger.info("[%s] Stage-1 evaluation result: %s", _UID, res)

    return {
        "compiled":    res["compiled"],
        "correctness": res["correctness"],
        # Stage-1 不给分
        "score":       0.0,
        "passed":      res["compiled"] and res["correctness"],
    }


def evaluate_stage2(program_text: str, stage1_payload: dict | None = None):
    """
    Stage-2：性能测速与计分。
    OpenEvolve 仅会传入 program_text；stage1_payload 设为可选参数
    以兼容未来版本或手工调用。
    """
    if stage1_payload:
        logger.info("[%s] Stage-2 evaluation with payload: %s", _UID, stage1_payload)

    prog = _normalize(program_text)
    logger.info("[%s] Stage-2 program_text:\n%s", _UID, prog)
    return _EVAL.evaluate(_UID, _REF_CODE, prog)