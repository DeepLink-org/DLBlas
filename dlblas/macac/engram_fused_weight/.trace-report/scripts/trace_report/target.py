"""Build and execute commands for local, docker, ssh, and ssh+docker targets."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Iterable

from .env_config import KNOWN_KEYS, target_mode, value

TARGET_ENV_MARKER = "TRACE_REPORT_TARGET_ENV"
TARGET_ENV_MARKER_VALUE = "1"

def shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)

def shell_exports(config: dict[str, str]) -> str:
    exports = [
        f"{key}={shlex.quote(config[key])}"
        for key in KNOWN_KEYS
        if key in config
    ]
    exports.append(f"{TARGET_ENV_MARKER}={TARGET_ENV_MARKER_VALUE}")
    return " ".join(exports)

def docker_exec_prefix(config: dict[str, str]) -> list[str]:
    cmd = ["docker", "exec", "-i", "-w", value(config, "REMOTE_WORKDIR")]
    for key in KNOWN_KEYS:
        if key in config:
            cmd.extend(["-e", f"{key}={config[key]}"])
    cmd.extend(["-e", f"{TARGET_ENV_MARKER}={TARGET_ENV_MARKER_VALUE}"])
    cmd.append(value(config, "CONTAINER_NAME"))
    return cmd

def ssh_password_enabled() -> bool:
    return bool(os.environ.get("TRACE_REPORT_SSH_PASSWORD"))

def ssh_command_prefix() -> list[str]:
    if ssh_password_enabled():
        return ["sshpass", "-e", "ssh"]
    return ["ssh"]

def rsync_command_prefix() -> list[str]:
    if ssh_password_enabled():
        return ["sshpass", "-e", "rsync"]
    return ["rsync"]

def ssh_process_env() -> dict[str, str] | None:
    password = os.environ.get("TRACE_REPORT_SSH_PASSWORD")
    if not password:
        return None
    env = os.environ.copy()
    env["SSHPASS"] = password
    return env

def target_process_env(config: dict[str, str]) -> dict[str, str] | None:
    if target_mode(config) in {"ssh", "ssh+docker"}:
        return ssh_process_env()
    return None

def run_checked(
    argv: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
    stdin=subprocess.DEVNULL,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        text=True,
        env=env,
        stdin=stdin,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )

def target_test_command(config: dict[str, str], test_argv: list[str]) -> list[str]:
    mode = target_mode(config)
    server = f"{value(config, 'SERVER_USER')}@{value(config, 'SERVER_HOST')}"
    container = value(config, "CONTAINER_NAME")
    remote_workdir = value(config, "REMOTE_WORKDIR")

    if mode == "local":
        return test_argv
    if mode == "docker":
        return ["docker", "exec", "-w", remote_workdir, container, *test_argv]
    if mode == "ssh":
        return [*ssh_command_prefix(), server, shell_join(test_argv)]
    return [*ssh_command_prefix(), server, shell_join(["docker", "exec", "-w", remote_workdir, container, *test_argv])]

def run_in_target_workdir(
    config: dict[str, str],
    command: list[str],
    *,
    capture: bool = False,
    stdin=subprocess.DEVNULL,
) -> subprocess.CompletedProcess[str]:
    mode = target_mode(config)
    env = None
    if mode == "local":
        local_env = None
        if KNOWN_KEYS:
            import os

            local_env = os.environ.copy()
            for key in KNOWN_KEYS:
                if key in config:
                    local_env[key] = config[key]
            local_env[TARGET_ENV_MARKER] = TARGET_ENV_MARKER_VALUE
        env = local_env
        return subprocess.run(
            command,
            cwd=value(config, "REMOTE_WORKDIR"),
            env=env,
            check=False,
            text=True,
            stdin=stdin,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    return run_checked(command_for_run(config, command), capture=capture, env=target_process_env(config), stdin=stdin)

def command_for_run(config: dict[str, str], command: list[str]) -> list[str]:
    mode = target_mode(config)
    server = f"{value(config, 'SERVER_USER')}@{value(config, 'SERVER_HOST')}"
    container = value(config, "CONTAINER_NAME")
    remote_workdir = value(config, "REMOTE_WORKDIR")

    if mode == "local":
        return command
    if mode == "docker":
        return [*docker_exec_prefix(config), *command]
    if mode == "ssh":
        remote = f"cd {shlex.quote(remote_workdir)} && {shell_exports(config)} {shell_join(command)}"
        return [*ssh_command_prefix(), server, remote]
    remote = shell_join([*docker_exec_prefix(config), *command])
    return [*ssh_command_prefix(), server, remote]
