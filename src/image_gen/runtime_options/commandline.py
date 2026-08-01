from __future__ import annotations

import subprocess
from typing import Any, Mapping


RUNTIME_COMMAND_FORMAT = "image-gen-runtime-command-v1"
RUNTIME_COMMAND_SCHEMA_VERSION = 1


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _windows_join(arguments: list[str]) -> str:
    return subprocess.list2cmdline([str(value) for value in arguments])


def _add_value(arguments: list[str], option: str, value: Any) -> None:
    arguments.extend((option, "" if value is None else str(value)))


def _add_bool(arguments: list[str], enabled: Any, positive: str, negative: str) -> None:
    arguments.append(positive if bool(enabled) else negative)


def build_runtime_command_from_status(status: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a paste-ready Windows ``COMMANDLINE_ARGS`` command.

    The copied command uses saved process-start values when a restart is pending;
    otherwise it uses the active process values. Per-job values always come from
    the resolved next-job contract. This formatter does not create another
    settings model: it serializes the existing Phase 14K startup-status contract.
    """

    payload = _mapping(status)
    runtime = _mapping(payload.get("runtime"))
    profile = _mapping(runtime.get("runtime_profile"))
    attention = _mapping(runtime.get("attention"))
    allocator = _mapping(runtime.get("cuda_allocator"))
    next_job = _mapping(runtime.get("next_job_settings"))

    use_pending = bool(
        payload.get("restart_required")
        or payload.get("pending_change_blocked")
        or runtime.get("restart_required")
        or runtime.get("pending_change_blocked")
    )
    mode = "pending" if use_pending else "active"

    arguments: list[str] = []
    selector = str(profile.get("selector") or profile.get("profile_id") or "").strip()
    if selector:
        _add_value(arguments, "--runtime-profile", selector)

    attention_backend = (
        attention.get("saved_next_restart")
        if use_pending
        else attention.get("requested_backend")
    )
    _add_value(arguments, "--attention-backend", attention_backend or "auto")

    _add_value(arguments, "--memory-policy", next_job.get("memory_policy", "auto"))
    _add_value(
        arguments,
        "--vram-safety-margin-mb",
        int(next_job.get("memory_vram_safety_margin_mb", 1024) or 0),
    )
    _add_bool(
        arguments,
        next_job.get("memory_retain_checkpoint_between_jobs", True),
        "--retain-unet-between-jobs",
        "--no-retain-unet-between-jobs",
    )
    _add_bool(
        arguments,
        next_job.get("memory_retain_vae_between_jobs", True),
        "--retain-vae-between-jobs",
        "--no-retain-vae-between-jobs",
    )
    _add_bool(
        arguments,
        next_job.get("model_runtime_retain_text_encoder_between_jobs", True),
        "--retain-text-encoder-between-jobs",
        "--no-retain-text-encoder-between-jobs",
    )
    _add_value(arguments, "--attention-slicing", next_job.get("attention_slicing", "off"))
    _add_bool(arguments, next_job.get("vae_tiling", False), "--vae-tiling", "--no-vae-tiling")
    _add_bool(arguments, next_job.get("vae_slicing", False), "--vae-slicing", "--no-vae-slicing")
    _add_value(arguments, "--vae-device", next_job.get("vae_device", "auto"))
    _add_value(arguments, "--preview-policy", next_job.get("preview_policy", "normal"))
    _add_value(
        arguments,
        "--hires-memory-profile",
        next_job.get("hires_memory_profile", "inherit"),
    )
    _add_bool(
        arguments,
        next_job.get("pre_hires_cleanup", False),
        "--pre-hires-cleanup",
        "--no-pre-hires-cleanup",
    )

    retry_profile = str(next_job.get("oom_retry_profile") or "disabled")
    retry_limit = int(next_job.get("oom_retry_limit", 0) or 0)
    if retry_profile == "disabled" or retry_limit <= 0:
        arguments.append("--no-retry-on-oom")
        _add_value(arguments, "--oom-retry-limit", 0)
    else:
        arguments.append("--retry-on-oom")
        _add_value(arguments, "--oom-retry-profile", retry_profile)
        _add_value(arguments, "--oom-retry-limit", retry_limit)

    mslk_values = _mapping(
        payload.get("saved_next_restart") if use_pending else payload.get("active")
    )
    for field, option in (
        ("policy", "--mslk-fmha-policy"),
        ("debug", "--mslk-fmha-debug"),
        ("block_n", "--mslk-fmha-block-n"),
        ("block_m", "--mslk-fmha-block-m"),
        ("num_warps", "--mslk-fmha-num-warps"),
        ("num_stages", "--mslk-fmha-num-stages"),
        ("experimental_head_dims", "--mslk-fmha-experimental-head-dims"),
    ):
        _add_value(arguments, option, mslk_values.get(field, ""))

    allocator_value = (
        allocator.get("saved_next_restart_config")
        if use_pending
        else allocator.get("active_config")
    )
    _add_value(arguments, "--cuda-alloc-conf", allocator_value or "")

    commandline_args = _windows_join(arguments)
    return {
        "format": RUNTIME_COMMAND_FORMAT,
        "schema_version": RUNTIME_COMMAND_SCHEMA_VERSION,
        "mode": mode,
        "uses_pending_process_settings": use_pending,
        "arguments": arguments,
        "commandline_args": commandline_args,
        "set_command": f'set "COMMANDLINE_ARGS={commandline_args}"',
        "launcher_examples": ["run.bat", "run_webui.bat"],
        "restart_required": bool(payload.get("restart_required")),
        "pending_change_blocked": bool(payload.get("pending_change_blocked")),
    }


__all__ = [
    "RUNTIME_COMMAND_FORMAT",
    "RUNTIME_COMMAND_SCHEMA_VERSION",
    "build_runtime_command_from_status",
]
