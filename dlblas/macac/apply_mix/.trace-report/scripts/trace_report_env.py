#!/usr/bin/env python3
"""Validate trace-report env YAML and run commands in the selected target."""

from __future__ import annotations

import argparse
import errno
import shutil
import shlex
import sys
from pathlib import Path

from trace_report.env_config import (
    ACTIVE_CONFIG_REL_PATH,
    CONFIG_PATH,
    ConfigError,
    KNOWN_KEYS,
    derive_run_env,
    format_config_text,
    local_active_config_path,
    load_config,
    target_mode,
)
from trace_report.env_validate import validate_config
from trace_report.target import (
    command_for_run,
    rsync_command_prefix,
    run_checked,
    run_in_target_workdir,
    shell_join,
    ssh_command_prefix,
    target_process_env,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEMPLATE_CONFIG_PATH = REPO_ROOT / CONFIG_PATH


def emit_exports(config: dict[str, str]) -> None:
    from trace_report.env_config import KNOWN_KEYS

    for key in KNOWN_KEYS:
        if key in config:
            print(f"export {key}={shlex.quote(config[key])}")


def emit_derived_run_exports(run_dir: str) -> None:
    for key, value in derive_run_env(run_dir).items():
        print(f"export {key}={shlex.quote(value)}")

def load_bootstrap_config() -> dict[str, str]:
    return load_config(str(TEMPLATE_CONFIG_PATH))

def find_active_config(explicit: str | None = None) -> Path:
    if not explicit:
        raise ConfigError(
            "active config must be passed explicitly with "
            "`--config \"$ACTIVE_CONFIG\"`; set "
            f"`ACTIVE_CONFIG=\"$LOCAL_WORKDIR/{ACTIVE_CONFIG_REL_PATH}\"` after init-config"
        )

    candidate = Path(explicit)
    if candidate.is_file():
        return candidate
    raise ConfigError(f"active config not found: {explicit}")

def load_active_config(explicit: str | None = None) -> dict[str, str]:
    config_path = find_active_config(explicit)
    return load_config(str(config_path))

def parse_set_arg(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ConfigError(f"--set requires KEY=VALUE: {item}")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"--set requires a non-empty KEY: {item}")
    if key not in KNOWN_KEYS:
        raise ConfigError(f"unknown config key: {key}")
    return key, value

def build_init_config(set_items: list[str]) -> dict[str, str]:
    config = load_bootstrap_config()
    for item in set_items:
        key, value = parse_set_arg(item)
        config[key] = value
    return config

def emit_errors(errors: list[str]) -> None:
    for err in errors:
        print(f"[trace_report_env] error: {err}", file=sys.stderr)

def init_local_config(config: dict[str, str], *, force: bool) -> str:
    active_config = local_active_config_path(config)
    if active_config.exists() and not force:
        raise ConfigError(
            f"active config already exists: {active_config}; "
            "use --force to overwrite it"
        )
    try:
        active_config.parent.mkdir(parents=True, exist_ok=True)
        active_config.write_text(format_config_text(config), encoding="utf-8")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EROFS, errno.EPERM}:
            raise ConfigError(
                "cannot write active config at "
                f"{active_config}; grant write access to LOCAL_WORKDIR or rerun with "
                "the required filesystem permission. Do not edit the repository "
                "template config as a fallback."
            ) from exc
        raise
    return str(active_config)

def _run_or_raise(argv: list[str], config: dict[str, str], *, capture: bool = False):
    result = run_checked(argv, capture=capture, env=target_process_env(config))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ConfigError(f"command failed ({result.returncode}): {shell_join(argv)}{suffix}")
    return result

def _copy_local_scripts(dst: Path) -> None:
    src = SCRIPT_DIR
    if src.resolve() == dst.resolve():
        raise ConfigError(f"refuse to overwrite current skill scripts: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(dst.iterdir()):
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in src.iterdir():
        target = dst / child.name
        shutil.copytree(child, target) if child.is_dir() else shutil.copy2(child, target)

def sync_scripts(config: dict[str, str]) -> None:
    mode = target_mode(config)
    remote_workdir = config["REMOTE_WORKDIR"]
    container = config["CONTAINER_NAME"]
    server = f"{config['SERVER_USER']}@{config['SERVER_HOST']}"

    if mode == "local":
        _copy_local_scripts(Path(remote_workdir) / ".trace-report" / "scripts")
        return
    if mode == "docker":
        _run_or_raise(command_for_run(config, ["sh", "-lc", "rm -rf .trace-report/scripts && mkdir -p .trace-report/scripts"]), config)
        _run_or_raise(["docker", "cp", f"{SCRIPT_DIR}/.", f"{container}:{remote_workdir}/.trace-report/scripts/"], config)
        return
    if mode == "ssh":
        _run_or_raise([*ssh_command_prefix(), server, f"rm -rf {shlex.quote(remote_workdir + '/.trace-report/scripts')} && mkdir -p {shlex.quote(remote_workdir + '/.trace-report/scripts')}"], config)
        _run_or_raise([*rsync_command_prefix(), "-a", "--delete", f"{SCRIPT_DIR}/", f"{server}:{remote_workdir}/.trace-report/scripts/"], config)
        return

    remote_stage = ""
    try:
        result = _run_or_raise([*ssh_command_prefix(), server, "mktemp -d /tmp/trace-report-scripts.XXXXXX"], config, capture=True)
        remote_stage = (result.stdout or "").strip()
        _run_or_raise([*rsync_command_prefix(), "-a", "--delete", f"{SCRIPT_DIR}/", f"{server}:{remote_stage}/"], config)
        _run_or_raise(command_for_run(config, ["sh", "-lc", "rm -rf .trace-report/scripts && mkdir -p .trace-report/scripts"]), config)
        _run_or_raise([*ssh_command_prefix(), server, shell_join(["docker", "cp", f"{remote_stage}/.", f"{container}:{remote_workdir}/.trace-report/scripts/"])], config)
    finally:
        if remote_stage:
            run_checked([*ssh_command_prefix(), server, f"rm -rf {shlex.quote(remote_stage)}"], env=target_process_env(config))

def sync_artifacts(config: dict[str, str], run_dir: str) -> None:
    mode = target_mode(config)
    run_rel = run_dir.rstrip("/")
    run_path = Path(run_rel)
    if run_path.is_absolute() or ".." in run_path.parts or not run_rel.startswith("profile-artifacts/"):
        raise ConfigError("sync-artifacts --run-dir must be under profile-artifacts/")
    run_name = Path(run_rel).name
    local_parent = Path(config["LOCAL_WORKDIR"]) / "profile-artifacts"
    local_target = local_parent / run_name
    server = f"{config['SERVER_USER']}@{config['SERVER_HOST']}"
    container = config["CONTAINER_NAME"]
    remote_workdir = config["REMOTE_WORKDIR"]

    local_parent.mkdir(parents=True, exist_ok=True)
    if mode == "local":
        source = Path(remote_workdir) / run_rel
        if source.resolve() == local_target.resolve():
            return
        if local_target.exists():
            shutil.rmtree(local_target)
        shutil.copytree(source, local_target)
        return
    if mode == "docker":
        result = run_checked(["docker", "cp", f"{container}:{remote_workdir}/{run_rel}", str(local_parent)])
        if result.returncode != 0:
            raise ConfigError("docker artifact sync failed")
        return
    if mode == "ssh":
        _run_or_raise([*rsync_command_prefix(), "-a", f"{server}:{remote_workdir}/{run_rel}", f"{local_parent}/"], config)
        return

    remote_stage = ""
    try:
        result = _run_or_raise([*ssh_command_prefix(), server, "mktemp -d /tmp/trace-report-artifacts.XXXXXX"], config, capture=True)
        remote_stage = (result.stdout or "").strip()
        _run_or_raise([*ssh_command_prefix(), server, f"mkdir -p {shlex.quote(remote_stage + '/profile-artifacts')} && {shell_join(['docker', 'cp', f'{container}:{remote_workdir}/{run_rel}', f'{remote_stage}/profile-artifacts/'])}"], config)
        _run_or_raise([*rsync_command_prefix(), "-a", f"{server}:{remote_stage}/profile-artifacts/{run_name}", f"{local_parent}/"], config)
    finally:
        if remote_stage:
            run_checked([*ssh_command_prefix(), server, f"rm -rf {shlex.quote(remote_stage)}"], env=target_process_env(config))

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help=f"Required path to local active config, usually $LOCAL_WORKDIR/{ACTIVE_CONFIG_REL_PATH}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_local_p = sub.add_parser(
        "init-config",
        help="Generate LOCAL_WORKDIR/.trace-report/config/trace_report_env.yaml from --set values",
    )
    init_local_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override one config key; can be repeated")
    init_local_p.add_argument("--force", action="store_true", help="Overwrite an existing local active config")

    for name in ("export", "mode", "validate"):
        sub.add_parser(name)
    validate_p = sub.choices["validate"]
    validate_p.add_argument("--skip-target-checks", action="store_true")

    sub.add_parser("sync-scripts", help="Sync this skill's scripts/ directory to target .trace-report/scripts/")
    sync_artifacts_p = sub.add_parser("sync-artifacts", help="Sync one target profile-artifacts run directory back to LOCAL_WORKDIR")
    sync_artifacts_p.add_argument("--run-dir", required=True, help="Run directory under profile-artifacts/")

    run_p = sub.add_parser("run")
    run_p.add_argument("target_command", nargs=argparse.REMAINDER)
    derive_p = sub.add_parser("derive-run", help="Derive PROFILE_* exports from profile-artifacts/<kernel>_v<N>_<tag>")
    derive_p.add_argument("run_dir")

    args = parser.parse_args(argv)

    try:
        if args.command == "derive-run":
            emit_derived_run_exports(args.run_dir)
            return 0
        if args.command == "init-config":
            config = build_init_config(args.set)
            errs = validate_config(config, check_target=False)
            if errs:
                emit_errors(errs)
                return 1
            active_path = init_local_config(config, force=args.force)
            print(f"[trace_report_env] active config: {active_path}")
            return 0

        config = load_active_config(args.config)
        if args.command == "export":
            emit_exports(config)
            return 0
        if args.command == "mode":
            errs = validate_config(config, check_target=False)
            if errs:
                emit_errors(errs)
                return 1
            print(target_mode(config))
            return 0
        if args.command == "validate":
            errs = validate_config(config, check_target=not args.skip_target_checks)
            if errs:
                emit_errors(errs)
                return 1
            print(f"[trace_report_env] ok: mode={target_mode(config)}")
            return 0
        if args.command == "sync-scripts":
            errs = validate_config(config, check_target=True)
            if errs:
                emit_errors(errs)
                return 1
            sync_scripts(config)
            print("[trace_report_env] synced scripts")
            return 0
        if args.command == "sync-artifacts":
            errs = validate_config(config, check_target=True)
            if errs:
                emit_errors(errs)
                return 1
            sync_artifacts(config, args.run_dir)
            print(f"[trace_report_env] synced artifacts: {Path(args.run_dir).name}")
            return 0
        if args.command == "run":
            command = list(args.target_command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                raise ConfigError("run requires a command after --")
            errs = validate_config(config, check_target=True)
            if errs:
                emit_errors(errs)
                return 1
            return run_in_target_workdir(config, command, stdin=None).returncode
    except ConfigError as exc:
        print(f"[trace_report_env] error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
