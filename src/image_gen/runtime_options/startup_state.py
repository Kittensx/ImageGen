from __future__ import annotations

from image_gen.program_metadata import PRODUCT_NAME

import hashlib
import json
from typing import Any, Mapping

from .contracts import MSLKFMHAOptions, RuntimeStartupOptions
from .cuda_allocator import (
    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
    build_cuda_allocator_diagnostics,
    canonicalize_cuda_allocator_conf,
)
from .normalization import resolve_runtime_startup_options, runtime_request_settings

MSLK_RESTART_MESSAGE = (
    f"MSLK/Triton startup settings have changed. Restart {PRODUCT_NAME} to compile "
    "and use the new attention configuration."
)
MSLK_OVERRIDE_MESSAGE = (
    "Saved MSLK/Triton settings differ, but one or more values are controlled "
    "by environment or command-line startup options. Remove those overrides "
    f"before restarting {PRODUCT_NAME} to use the saved settings."
)
MSLK_FIELDS = (
    "policy",
    "debug",
    "block_n",
    "block_m",
    "num_warps",
    "num_stages",
    "experimental_head_dims",
)
_HIGH_PRECEDENCE_SOURCES = {
    "cli",
    "explicit_cli",
    "commandline_args",
    "environment",
}

_PER_JOB_RUNTIME_FIELDS = (
    "memory_policy",
    "memory_vram_safety_margin_mb",
    "memory_retain_checkpoint_between_jobs",
    "memory_retain_vae_between_jobs",
    "model_runtime_retain_text_encoder_between_jobs",
    "attention_slicing",
    "vae_tiling",
    "vae_slicing",
    "vae_device",
    "preview_policy",
    "hires_memory_profile",
    "pre_hires_cleanup",
    "oom_retry_profile",
    "oom_retry_limit",
)


def _canonical_mslk(values: Mapping[str, Any] | None) -> dict[str, str]:
    defaults = MSLKFMHAOptions().to_dict()
    source = dict(values or {})
    return {field: str(source.get(field, defaults[field]) or "") for field in MSLK_FIELDS}


def _fingerprint(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_mslk(values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _runtime_options(
    options: RuntimeStartupOptions | Mapping[str, Any] | None,
) -> RuntimeStartupOptions:
    if isinstance(options, RuntimeStartupOptions):
        return options
    return RuntimeStartupOptions.from_mapping(options)


def _worker_attention_report(
    worker_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = dict(worker_status or {})
    provenance = status.get("model_provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    report = provenance.get("attention_backend")
    if not isinstance(report, Mapping):
        report = status.get("attention_backend")
    return dict(report) if isinstance(report, Mapping) else {}


def _verified_kernel_provider(report: Mapping[str, Any]) -> str | None:
    provider = str(report.get("kernel_provider") or "").strip()
    if not provider:
        return None
    if bool(
        report.get("operator_executed")
        or report.get("custom_provider_executed")
        or report.get("verified")
    ):
        return provider
    return None


def _settings_runtime_options(
    settings: Mapping[str, Any] | None,
) -> RuntimeStartupOptions:
    return resolve_runtime_startup_options(environment={}, settings=settings)


def _next_job_runtime_settings(
    settings: Mapping[str, Any] | None,
    active_options: RuntimeStartupOptions,
) -> dict[str, Any]:
    values = runtime_request_settings(active_options)
    settings_payload = dict(settings or {})
    raw_overrides = settings_payload.get("runtime_job_overrides")
    overrides = dict(raw_overrides) if isinstance(raw_overrides, Mapping) else {}
    if overrides:
        normalized = runtime_request_settings(
            resolve_runtime_startup_options(environment={}, settings=overrides)
        )
        for key in _PER_JOB_RUNTIME_FIELDS:
            if key in overrides and key in normalized:
                values[key] = normalized[key]
    return {key: values[key] for key in _PER_JOB_RUNTIME_FIELDS if key in values}


def runtime_options_mslk_values(
    options: RuntimeStartupOptions | Mapping[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if isinstance(options, RuntimeStartupOptions):
        payload = options.to_dict()
    else:
        payload = dict(options or {})
    return (
        _canonical_mslk(payload.get("mslk_fmha") if isinstance(payload, dict) else {}),
        {
            str(key): str(value)
            for key, value in dict(payload.get("source_map") or {}).items()
        },
    )


def saved_settings_mslk_values(settings: Mapping[str, Any] | None) -> dict[str, str]:
    source = dict(settings or {})
    nested = source.get("mslk_fmha")
    return _canonical_mslk(nested if isinstance(nested, Mapping) else {})


def runtime_options_allocator_value(
    options: RuntimeStartupOptions | Mapping[str, Any] | None,
) -> str:
    if isinstance(options, RuntimeStartupOptions):
        payload = options.to_dict()
    else:
        payload = dict(options or {})
    allocator = dict(payload.get("allocator_options") or {})
    return canonicalize_cuda_allocator_conf(
        allocator.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, "")
    )


def runtime_replay_warnings(
    recorded_options: RuntimeStartupOptions | Mapping[str, Any] | None,
    active_options: RuntimeStartupOptions | Mapping[str, Any] | None,
) -> list[str]:
    """Report output-sensitive startup differences that prevent exact replay."""

    if not recorded_options:
        return []
    recorded_mslk, _ = runtime_options_mslk_values(recorded_options)
    active_mslk, _ = runtime_options_mslk_values(active_options)
    differences = [
        field
        for field in MSLK_FIELDS
        if recorded_mslk.get(field, "") != active_mslk.get(field, "")
    ]
    warnings: list[str] = []
    if differences:
        details = ", ".join(
            f"{field}: recorded={recorded_mslk.get(field, '')!r}, "
            f"active={active_mslk.get(field, '')!r}"
            for field in differences
        )
        warnings.append(
            "Recorded MSLK/Triton process-start settings differ from the active "
            f"runtime ({details}). Exact replay is not guaranteed; restart with "
            "the recorded values to reproduce the original FMHA configuration."
        )

    recorded_allocator = runtime_options_allocator_value(recorded_options)
    active_allocator = runtime_options_allocator_value(active_options)
    if recorded_allocator != active_allocator:
        warnings.append(
            "Recorded CUDA allocator process-start settings differ from the active "
            "runtime ("
            f"recorded={recorded_allocator!r}, active={active_allocator!r}). "
            "Allocator changes require a fresh process before CUDA initialization; "
            "exact replay is not guaranteed."
        )
    return warnings


def build_runtime_startup_status(
    active_options: RuntimeStartupOptions | Mapping[str, Any] | None,
    pending_settings: Mapping[str, Any] | None,
    *,
    worker_ready: Mapping[str, Any] | None = None,
    worker_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare immutable process-start MSLK values with saved next-start values."""

    active_options_contract = _runtime_options(active_options)
    active_options_payload = active_options_contract.to_dict()
    active, source_map = runtime_options_mslk_values(active_options_contract)
    saved = saved_settings_mslk_values(pending_settings)
    pending_effective = dict(saved)
    blocked_fields: list[dict[str, str]] = []

    for field in MSLK_FIELDS:
        source = source_map.get(f"mslk_fmha.{field}", "default")
        if source in _HIGH_PRECEDENCE_SOURCES:
            pending_effective[field] = active[field]
            if saved[field] != active[field]:
                blocked_fields.append(
                    {
                        "field": field,
                        "source": source,
                        "active": active[field],
                        "saved": saved[field],
                    }
                )

    changed_fields = [
        field for field in MSLK_FIELDS if active[field] != pending_effective[field]
    ]
    saved_changed_fields = [
        field for field in MSLK_FIELDS if active[field] != saved[field]
    ]
    restart_required = bool(changed_fields)
    blocked = bool(blocked_fields)

    worker_payload = dict(worker_ready or {})
    worker_options = worker_payload.get("runtime_startup_options")
    worker_values = None
    worker_fingerprint = None
    worker_matches_active = None
    if isinstance(worker_options, Mapping):
        worker_values, _worker_sources = runtime_options_mslk_values(worker_options)
        worker_fingerprint = _fingerprint(worker_values)
        worker_matches_active = worker_values == active

    message = None
    if restart_required:
        message = MSLK_RESTART_MESSAGE
    elif blocked and saved_changed_fields:
        message = MSLK_OVERRIDE_MESSAGE

    settings_options = _settings_runtime_options(pending_settings)
    settings_payload = settings_options.to_dict()

    active_attention = str(active_options_contract.attention_backend)
    saved_attention = str(settings_options.attention_backend)
    attention_source = str(
        active_options_payload.get("source_map", {}).get("attention_backend", "default")
    )
    attention_blocked = bool(
        saved_attention != active_attention
        and attention_source in _HIGH_PRECEDENCE_SOURCES
    )
    pending_attention = active_attention if attention_blocked else saved_attention
    attention_restart_required = pending_attention != active_attention

    active_allocator = runtime_options_allocator_value(active_options_contract)
    saved_allocator = runtime_options_allocator_value(settings_options)
    allocator_source = str(
        active_options_payload.get("source_map", {}).get("allocator_options", "default")
    )
    allocator_blocked = bool(
        saved_allocator != active_allocator
        and allocator_source in _HIGH_PRECEDENCE_SOURCES
    )
    pending_allocator = active_allocator if allocator_blocked else saved_allocator
    allocator_restart_required = pending_allocator != active_allocator

    runtime_worker_status = dict(worker_status or {})
    attention_report = _worker_attention_report(runtime_worker_status)
    effective_attention = str(
        attention_report.get("effective_backend") or "unverified"
    )
    verified_provider = _verified_kernel_provider(attention_report)
    next_job = _next_job_runtime_settings(pending_settings, active_options_contract)
    runtime_restart_required = bool(
        restart_required or attention_restart_required or allocator_restart_required
    )
    runtime_pending_blocked = bool(blocked or attention_blocked or allocator_blocked)
    if runtime_restart_required and not message:
        changed = []
        if attention_restart_required:
            changed.append("attention backend")
        if allocator_restart_required:
            changed.append("CUDA allocator")
        message = (
            "Saved process-start runtime settings differ from the active process "
            f"({', '.join(changed)}). Restart {PRODUCT_NAME} to activate them."
        )
    elif runtime_pending_blocked and not message:
        message = (
            "Saved process-start runtime settings are overridden by environment or "
            "command-line options. Remove the higher-precedence override, then restart "
            f"{PRODUCT_NAME} to activate the saved values."
        )

    return {
        "schema_version": 1,
        "restart_required": runtime_restart_required,
        "saved_settings_differ": bool(saved_changed_fields),
        "pending_change_blocked": runtime_pending_blocked,
        "message": message,
        "active": active,
        "active_fingerprint": _fingerprint(active),
        "active_source_map": {
            field: source_map.get(f"mslk_fmha.{field}", "default")
            for field in MSLK_FIELDS
        },
        "saved_next_restart": saved,
        "pending_effective_next_restart": pending_effective,
        "pending_fingerprint": _fingerprint(pending_effective),
        "changed_fields": changed_fields,
        "saved_changed_fields": saved_changed_fields,
        "blocked_fields": blocked_fields,
        "cuda_allocator": {
            **build_cuda_allocator_diagnostics(active_options),
            "immutable_after_cuda_initialization": True,
            "active_config": active_allocator,
            "saved_next_restart_config": saved_allocator,
            "pending_effective_next_restart_config": pending_allocator,
            "restart_required": allocator_restart_required,
            "pending_change_blocked": allocator_blocked,
            "active_source": allocator_source,
        },
        "worker": {
            "online": bool(worker_payload),
            "active": worker_values,
            "active_fingerprint": worker_fingerprint,
            "matches_parent_process": worker_matches_active,
        },
        "runtime": {
            "schema_version": 1,
            "runtime_profile": active_options_contract.runtime_profile.to_dict(),
            "attention": {
                "requested_backend": active_attention,
                "effective_backend": effective_attention,
                "verified_kernel_provider": verified_provider,
                "provider_verified": verified_provider is not None,
                "backend_verified": bool(attention_report.get("verified")),
                "active_source": attention_source,
                "saved_next_restart": saved_attention,
                "pending_effective_next_restart": pending_attention,
                "restart_required": attention_restart_required,
                "pending_change_blocked": attention_blocked,
                "immutable_after_model_initialization": True,
            },
            "memory": {
                "policy": next_job.get("memory_policy", settings_payload.get("memory_policy")),
                "vram_safety_margin_mb": next_job.get(
                    "memory_vram_safety_margin_mb",
                    settings_payload.get("vram_safety_margin_mb"),
                ),
                "per_job_mutable": True,
            },
            "hires": {
                "memory_profile": next_job.get("hires_memory_profile"),
                "pre_hires_cleanup": next_job.get("pre_hires_cleanup"),
                "per_job_mutable": True,
            },
            "preview": {
                "policy": next_job.get("preview_policy"),
                "per_job_mutable": True,
            },
            "vae": {
                "tiling": next_job.get("vae_tiling"),
                "slicing": next_job.get("vae_slicing"),
                "device": next_job.get("vae_device"),
                "per_job_mutable": True,
            },
            "oom_retry": {
                "profile": next_job.get("oom_retry_profile"),
                "limit": next_job.get("oom_retry_limit"),
                "enabled": bool(
                    next_job.get("oom_retry_profile") != "disabled"
                    and int(next_job.get("oom_retry_limit") or 0) > 0
                ),
                "per_job_mutable": True,
            },
            "attention_slicing": {
                "mode": next_job.get("attention_slicing"),
                "per_job_mutable": True,
            },
            "cuda_allocator": {
                "active_config": active_allocator,
                "saved_next_restart_config": saved_allocator,
                "pending_effective_next_restart_config": pending_allocator,
                "restart_required": allocator_restart_required,
                "pending_change_blocked": allocator_blocked,
                "active_source": allocator_source,
                "immutable_after_cuda_initialization": True,
            },
            "next_job_settings": next_job,
            "restart_required": runtime_restart_required,
            "pending_change_blocked": runtime_pending_blocked,
            "blocked_process_settings": [
                name
                for name, is_blocked in (
                    ("attention_backend", attention_blocked),
                    ("mslk_fmha", blocked),
                    ("cuda_allocator", allocator_blocked),
                )
                if is_blocked
            ],
            "restart_required_settings": [
                name
                for name, required in (
                    ("attention_backend", attention_restart_required),
                    ("mslk_fmha", restart_required),
                    ("cuda_allocator", allocator_restart_required),
                )
                if required
            ],
            "process_scoped_settings": [
                "attention_backend",
                "mslk_fmha",
                "cuda_allocator",
            ],
            "per_job_settings": list(_PER_JOB_RUNTIME_FIELDS),
        },
    }


__all__ = [
    "MSLK_FIELDS",
    "MSLK_OVERRIDE_MESSAGE",
    "MSLK_RESTART_MESSAGE",
    "build_runtime_startup_status",
    "runtime_options_allocator_value",
    "runtime_options_mslk_values",
    "runtime_replay_warnings",
    "saved_settings_mslk_values",
]
