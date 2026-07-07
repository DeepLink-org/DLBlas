"""Target-aware validation for trace-report environment config."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .env_config import TOOL_KEYS, parse_exec_cmd, target_mode, value
from .target import (
    run_checked,
    run_in_target_workdir,
    shell_join,
    ssh_command_prefix,
    ssh_password_enabled,
    target_process_env,
    target_test_command,
)

def validate_config(config: dict[str, str], *, check_target: bool) -> list[str]:
    errors: list[str] = []

    if bool(value(config, "SERVER_USER")) != bool(value(config, "SERVER_HOST")):
        errors.append("SERVER_USER and SERVER_HOST must be both set or both empty")

    if value(config, "BOUND_MODE") not in {"coarse", "detailed"}:
        errors.append("BOUND_MODE must be coarse or detailed")

    if not value(config, "LOCAL_WORKDIR"):
        errors.append("LOCAL_WORKDIR must not be empty")
    elif not Path(value(config, "LOCAL_WORKDIR")).is_dir():
        errors.append(f"LOCAL_WORKDIR does not exist on local machine: {value(config, 'LOCAL_WORKDIR')}")

    if not value(config, "REMOTE_WORKDIR"):
        errors.append("REMOTE_WORKDIR must not be empty")

    if not value(config, "OP_EXEC_CMD"):
        errors.append("OP_EXEC_CMD must not be empty")
    elif parse_exec_cmd(config) is None:
        errors.append("OP_EXEC_CMD must be valid shell-style argv text")

    if not value(config, "MACA_VISIBLE_DEVICES"):
        errors.append("MACA_VISIBLE_DEVICES must not be empty")

    target_peu = value(config, "CYCLE_TRACE_TARGET_PEU")
    if not target_peu:
        errors.append("CYCLE_TRACE_TARGET_PEU must not be empty")
    elif not target_peu.isdigit():
        errors.append("CYCLE_TRACE_TARGET_PEU must be an integer from 1 to 15")
    elif not 1 <= int(target_peu) <= 15:
        errors.append("CYCLE_TRACE_TARGET_PEU must be in [1, 15]; 1 (0x1) captures PEU 0")

    dpg_page_num = value(config, "CYCLE_TRACE_DPG_PAGE_NUM")
    if not dpg_page_num:
        errors.append("CYCLE_TRACE_DPG_PAGE_NUM must not be empty")
    elif not dpg_page_num.isdigit():
        errors.append("CYCLE_TRACE_DPG_PAGE_NUM must be a positive integer")
    elif int(dpg_page_num) <= 0:
        errors.append("CYCLE_TRACE_DPG_PAGE_NUM must be greater than 0")

    target_ap = value(config, "CYCLE_TRACE_TARGET_AP")
    if not target_ap:
        errors.append("CYCLE_TRACE_TARGET_AP must not be empty")
    elif not target_ap.isdigit():
        errors.append("CYCLE_TRACE_TARGET_AP must be an integer from 0 to 15")
    elif not 0 <= int(target_ap) <= 15:
        errors.append("CYCLE_TRACE_TARGET_AP must be in [0, 15]")

    dpc_id = value(config, "CYCLE_TRACE_DPC_ID")
    if not dpc_id:
        errors.append("CYCLE_TRACE_DPC_ID must not be empty")
    elif not re.fullmatch(r"[0-7](,[0-7])*", dpc_id):
        errors.append("CYCLE_TRACE_DPC_ID must be a comma-separated list of integers from 0 to 7 with no spaces")

    for key in TOOL_KEYS:
        if not value(config, key):
            errors.append(f"{key} must not be empty")

    mode = target_mode(config)
    if mode in {"docker", "ssh+docker"} and not value(config, "CONTAINER_NAME"):
        errors.append("CONTAINER_NAME must not be empty in docker mode")

    if mode in {"ssh", "ssh+docker"} and ssh_password_enabled() and shutil.which("sshpass") is None:
        errors.append("sshpass is required when TRACE_REPORT_SSH_PASSWORD is set")

    if errors or not check_target:
        return errors

    if mode in {"docker", "ssh+docker"}:
        cmd = target_test_command(config, ["test", "-d", value(config, "REMOTE_WORKDIR")])
    else:
        cmd = target_test_command(config, ["test", "-d", value(config, "REMOTE_WORKDIR")])
    if run_checked(cmd, capture=True, env=target_process_env(config)).returncode != 0:
        errors.append(f"REMOTE_WORKDIR does not exist in {mode} target: {value(config, 'REMOTE_WORKDIR')}")

    if mode in {"docker", "ssh+docker"}:
        cmd = (
            ["docker", "inspect", value(config, "CONTAINER_NAME")]
            if mode == "docker"
            else [
                *ssh_command_prefix(),
                f"{value(config, 'SERVER_USER')}@{value(config, 'SERVER_HOST')}",
                shell_join(["docker", "inspect", value(config, "CONTAINER_NAME")]),
            ]
        )
        if run_checked(cmd, capture=True, env=target_process_env(config)).returncode != 0:
            errors.append(f"container does not exist or is not inspectable: {value(config, 'CONTAINER_NAME')}")

    for key in TOOL_KEYS:
        cmd = target_test_command(config, ["test", "-x", value(config, key)])
        if run_checked(cmd, capture=True, env=target_process_env(config)).returncode != 0:
            errors.append(f"{key} is not executable in {mode} target: {value(config, key)}")

    exec_argv = parse_exec_cmd(config)
    if exec_argv is not None:
        try:
            op = run_in_target_workdir(config, exec_argv, capture=True)
        except OSError as exc:
            errors.append(f"OP_EXEC_CMD failed in REMOTE_WORKDIR ({value(config, 'REMOTE_WORKDIR')}): {exc}")
        else:
            if op.returncode != 0:
                detail = (op.stderr or op.stdout or "").strip().splitlines()
                suffix = f": {detail[-1]}" if detail else ""
                errors.append(f"OP_EXEC_CMD failed in REMOTE_WORKDIR ({value(config, 'REMOTE_WORKDIR')}){suffix}")

    mxsmi = run_checked(target_test_command(config, ["mx-smi"]), capture=True, env=target_process_env(config))
    if mxsmi.returncode != 0:
        errors.append("mx-smi is not available in target environment; cannot validate MACA_VISIBLE_DEVICES")
    else:
        device = value(config, "MACA_VISIBLE_DEVICES")
        visible = re.findall(r"(?m)^\s*(\d+)\s", mxsmi.stdout or "")
        if device not in visible and not re.search(rf"(?<!\d){re.escape(device)}(?!\d)", mxsmi.stdout or ""):
            errors.append(f"MACA_VISIBLE_DEVICES={device} was not found in mx-smi output")

    return errors
