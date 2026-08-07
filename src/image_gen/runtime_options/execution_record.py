from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from typing import Any, Mapping

from .contracts import RuntimeStartupOptions
from .cuda_allocator import build_cuda_allocator_diagnostics

RUNTIME_EXECUTION_SCHEMA_VERSION = 1
RUNTIME_EXECUTION_FORMAT = "image-gen-runtime-execution-v1"

MSLK_ENVIRONMENT_FIELDS = {
    "policy": "MSLK_FMHA_POLICY",
    "debug": "MSLK_FMHA_DEBUG",
    "block_n": "MSLK_FMHA_BLOCK_N",
    "block_m": "MSLK_FMHA_BLOCK_M",
    "num_warps": "MSLK_FMHA_NUM_WARPS",
    "num_stages": "MSLK_FMHA_NUM_STAGES",
    "experimental_head_dims": "MSLK_FMHA_EXPERIMENTAL_HEAD_DIMS",
}


RUNTIME_CONFORMANCE_SCHEMA_VERSION = 1
RUNTIME_CONFORMANCE_FORMAT = "image-gen-runtime-conformance-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def runtime_execution_conformance_snapshot(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the stable, output-sensitive runtime subset used for replay comparison."""

    source = _mapping(record)
    attention = _mapping(source.get("attention"))
    memory = _mapping(source.get("memory"))
    vae = _mapping(source.get("vae"))
    preview = _mapping(source.get("preview"))
    hires = _mapping(source.get("hires"))
    allocator = _mapping(source.get("cuda_allocator"))
    oom = _mapping(source.get("oom_recovery"))
    replay = _mapping(source.get("replay"))
    mslk = _mapping(attention.get("mslk_fmha"))
    runtime_profile = _mapping(source.get("runtime_profile"))
    oom_history = _mapping(oom.get("history"))

    return {
        "runtime_profile": {
            "profile_id": runtime_profile.get("profile_id"),
            "schema_version": runtime_profile.get("schema_version"),
            "source": runtime_profile.get("source"),
        },
        "attention": {
            "requested_backend": attention.get("requested_backend"),
            "effective_backend": attention.get("effective_backend"),
            "verified_kernel_provider": attention.get("verified_kernel_provider"),
            "effective_operator": attention.get("effective_operator"),
            "attention_slicing": attention.get("attention_slicing"),
            "mslk_fmha_requested": _mapping(mslk.get("requested")),
            "mslk_fmha_effective": _mapping(mslk.get("effective")),
        },
        "memory": {
            "requested_policy": memory.get("requested_policy"),
            "effective_policy_by_stage": _mapping(memory.get("effective_policy_by_stage")),
            "vram_safety_margin_mb": memory.get("vram_safety_margin_mb"),
            "component_retention": _mapping(memory.get("component_retention")),
            "execution_device": memory.get("execution_device"),
        },
        "vae": {
            "requested": _mapping(vae.get("requested")),
            "effective": _mapping(vae.get("effective")),
        },
        "preview": {
            "requested_policy": preview.get("requested_policy"),
            "image_decode_suspended": bool(preview.get("image_decode_suspended")),
            "one_way_for_job": bool(preview.get("one_way_for_job", True)),
        },
        "hires": {
            "requested_memory_profile": hires.get("requested_memory_profile"),
            "pre_hires_cleanup_requested": hires.get("pre_hires_cleanup_requested"),
            "memory_behavior": _mapping(hires.get("memory_behavior")),
        },
        "cuda_allocator": {
            "effective_config": allocator.get("effective_config"),
            "requested_config": allocator.get("requested_config"),
        },
        "oom_recovery": {
            "configured_profile": oom.get("configured_profile"),
            "retry_limit": oom.get("retry_limit"),
            "runtime_path_changed": bool(oom_history.get("runtime_path_changed")),
            "fallback_profiles_applied": list(oom_history.get("fallback_profiles_applied") or []),
            "final_result_by_stage": _mapping(oom_history.get("final_result_by_stage")),
        },
        "restorable_job_settings": _mapping(replay.get("restorable_job_settings")),
    }


def _fingerprint_conformance_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _mapping(snapshot)
    digest = hashlib.sha256(_canonical_json(canonical).encode("ascii")).hexdigest()
    return {
        "schema_version": RUNTIME_CONFORMANCE_SCHEMA_VERSION,
        "format": RUNTIME_CONFORMANCE_FORMAT,
        "sha256": digest,
        "snapshot": canonical,
    }


def runtime_execution_fingerprint(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = _mapping(record)
    # Compact replay manifests retain the exact conformance SHA but omit the
    # bulky snapshot. Honor that authoritative fingerprint instead of hashing
    # the intentionally reduced runtime record.
    existing = _mapping(source.get("conformance_fingerprint"))
    if (
        existing.get("format") == RUNTIME_CONFORMANCE_FORMAT
        and str(existing.get("sha256") or "")
    ):
        output = {
            "schema_version": int(
                existing.get("schema_version") or RUNTIME_CONFORMANCE_SCHEMA_VERSION
            ),
            "format": RUNTIME_CONFORMANCE_FORMAT,
            "sha256": str(existing.get("sha256") or ""),
        }
        if isinstance(existing.get("snapshot"), Mapping):
            output["snapshot"] = _mapping(existing.get("snapshot"))
        return output
    if (
        source.get("format") == RUNTIME_CONFORMANCE_FORMAT
        and str(source.get("sha256") or "")
    ):
        output = {
            "schema_version": int(
                source.get("schema_version") or RUNTIME_CONFORMANCE_SCHEMA_VERSION
            ),
            "format": RUNTIME_CONFORMANCE_FORMAT,
            "sha256": str(source.get("sha256") or ""),
        }
        if isinstance(source.get("snapshot"), Mapping):
            output["snapshot"] = _mapping(source.get("snapshot"))
        return output
    return _fingerprint_conformance_snapshot(
        runtime_execution_conformance_snapshot(source)
    )


def _conformance_differences(
    original: Any, replayed: Any, *, path: str = ""
) -> list[dict[str, Any]]:
    if isinstance(original, Mapping) and isinstance(replayed, Mapping):
        output: list[dict[str, Any]] = []
        for key in sorted(set(original) | set(replayed)):
            child = f"{path}.{key}" if path else str(key)
            output.extend(
                _conformance_differences(original.get(key), replayed.get(key), path=child)
            )
        return output
    if isinstance(original, list) and isinstance(replayed, list):
        if original == replayed:
            return []
        return [{"path": path, "original": original, "replayed": replayed}]
    if original == replayed:
        return []
    return [{"path": path, "original": original, "replayed": replayed}]


def compare_runtime_execution_records(
    original: Mapping[str, Any] | None,
    replayed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare original and replayed runtime paths using stable fingerprints."""

    original_fp = runtime_execution_fingerprint(original)
    replay_fp = runtime_execution_fingerprint(replayed)
    original_sha = str(original_fp.get("sha256") or "")
    replay_sha = str(replay_fp.get("sha256") or "")
    if original_sha and original_sha == replay_sha:
        differences: list[dict[str, Any]] = []
    elif isinstance(original_fp.get("snapshot"), Mapping) and isinstance(
        replay_fp.get("snapshot"), Mapping
    ):
        differences = _conformance_differences(
            _mapping(original_fp.get("snapshot")),
            _mapping(replay_fp.get("snapshot")),
        )
    else:
        differences = [{
            "path": "runtime_conformance_fingerprint",
            "original": original_sha,
            "replayed": replay_sha,
        }]
    categories = sorted({item["path"].split(".", 1)[0] for item in differences if item.get("path")})
    return {
        "schema_version": RUNTIME_CONFORMANCE_SCHEMA_VERSION,
        "format": RUNTIME_CONFORMANCE_FORMAT,
        "matches": not differences,
        "original_fingerprint": str(original_fp.get("sha256") or ""),
        "replay_fingerprint": str(replay_fp.get("sha256") or ""),
        "difference_count": len(differences),
        "difference_categories": categories,
        "differences": differences,
    }

RUNTIME_REPLAY_JOB_FIELDS = (
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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if str(item or "").strip()))


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _first_present(
    sources: tuple[tuple[Mapping[str, Any], str], ...],
    default: Any,
) -> Any:
    for source, key in sources:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _effective_memory_policy_by_stage(memory_summary: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for plan in list(memory_summary.get("plans") or []):
        if not isinstance(plan, Mapping):
            continue
        stage = str(plan.get("stage") or "").strip()
        profile = str(plan.get("effective_profile") or "").strip()
        if stage and profile:
            output[stage] = profile
    active = str(memory_summary.get("effective_policy") or "").strip()
    if active:
        output.setdefault("generation_complete", active)
    return output


def _preview_suspension_events(memory_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for action in list(memory_summary.get("automatic_actions") or []):
        if not isinstance(action, Mapping):
            continue
        token = str(action.get("action") or "").lower()
        if "preview" in token and ("suspend" in token or "release" in token or "drain" in token):
            events.append(dict(action))
    for action in list(memory_summary.get("oom_recovery_actions") or []):
        if not isinstance(action, Mapping):
            continue
        token = str(action.get("action") or "").lower()
        if "preview" in token:
            events.append(dict(action))
    if memory_summary.get("preview_image_decode_suspended"):
        events.append(
            {
                "action": "preview_image_decode_suspended_final_state",
                "reason": str(memory_summary.get("preview_image_decode_suspension_reason") or ""),
                "source": str(memory_summary.get("preview_image_decode_suspension_source") or ""),
                "one_way_for_job": bool(memory_summary.get("preview_suspension_one_way_for_job", True)),
            }
        )
    return events


def _external_fallback_reasons(
    *,
    attention_report: Mapping[str, Any],
    memory_summary: Mapping[str, Any],
    preview_policy: Mapping[str, Any],
    external_fallback_reasons: list[str] | tuple[str, ...] | None,
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []

    attention_reason = str(attention_report.get("fallback_reason") or "").strip()
    if attention_reason:
        reasons.append({"source": "attention_backend", "reason": attention_reason})

    for plan in list(memory_summary.get("plans") or []):
        if not isinstance(plan, Mapping):
            continue
        requested = str(plan.get("requested_profile") or "").strip()
        effective = str(plan.get("effective_profile") or "").strip()
        if requested and effective and requested != effective:
            reasons.append(
                {
                    "source": f"memory_policy:{plan.get('stage') or 'unknown'}",
                    "reason": f"Requested {requested}; effective policy was {effective}.",
                }
            )
        for reason in list(plan.get("reasons") or []):
            text = str(reason or "").strip()
            if text and any(token in text.lower() for token in ("fallback", "unavailable", "violate", "suspend")):
                reasons.append(
                    {
                        "source": f"memory_policy:{plan.get('stage') or 'unknown'}",
                        "reason": text,
                    }
                )

    for stage, report in preview_policy.items():
        if not isinstance(report, Mapping):
            continue
        reason = str(
            report.get("reason") or report.get("suspension_reason") or ""
        ).strip()
        if reason:
            reasons.append(
                {
                    "source": f"preview_policy:{stage}",
                    "reason": reason,
                }
            )

    for value in list(external_fallback_reasons or []):
        text = str(value or "").strip()
        if text:
            reasons.append({"source": "runtime", "reason": text})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in reasons:
        identity = (item["source"], item["reason"])
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


def runtime_replay_request_values(record: Mapping[str, Any] | None) -> dict[str, Any]:
    replay = _mapping(_mapping(record).get("replay"))
    values = _mapping(replay.get("restorable_job_settings"))
    return {
        key: values[key]
        for key in RUNTIME_REPLAY_JOB_FIELDS
        if key in values
    }


def build_runtime_execution_record(
    *,
    startup_options: RuntimeStartupOptions | Mapping[str, Any] | None,
    attention_report: Mapping[str, Any] | None,
    memory_summary: Mapping[str, Any] | None,
    preview_policy: Mapping[str, Any] | None,
    vae_report: Mapping[str, Any] | None,
    hires_metadata: Mapping[str, Any] | None,
    runtime_job_settings: Mapping[str, Any] | None = None,
    execution_device: str | None = None,
    external_fallback_reasons: list[str] | tuple[str, ...] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    options = (
        startup_options
        if isinstance(startup_options, RuntimeStartupOptions)
        else RuntimeStartupOptions.from_mapping(startup_options)
    )
    startup = options.to_dict()
    attention = _mapping(attention_report)
    memory = _mapping(memory_summary)
    preview = _mapping(preview_policy)
    vae = _mapping(vae_report)
    hires = _mapping(hires_metadata)
    job = _mapping(runtime_job_settings)
    manager_settings = _mapping(memory.get("settings"))
    target_environment = environment if environment is not None else os.environ

    memory_policy = str(
        _first_present(
            ((job, "memory_policy"), (manager_settings, "policy")),
            options.memory_policy,
        )
    )
    safety_margin_mb = int(
        _first_present(
            (
                (job, "memory_vram_safety_margin_mb"),
                (manager_settings, "safety_margin_mb"),
            ),
            options.vram_safety_margin_mb,
        )
    )
    retain_unet = bool(
        _first_present(
            (
                (job, "memory_retain_checkpoint_between_jobs"),
                (manager_settings, "retain_checkpoint_between_jobs"),
            ),
            options.retain_unet_between_jobs,
        )
    )
    retain_vae = bool(
        _first_present(
            (
                (job, "memory_retain_vae_between_jobs"),
                (manager_settings, "retain_vae_between_jobs"),
            ),
            options.retain_vae_between_jobs,
        )
    )
    retain_text_encoder = bool(
        _first_present(
            ((job, "model_runtime_retain_text_encoder_between_jobs"),),
            options.retain_text_encoder_between_jobs,
        )
    )
    attention_slicing = str(
        _first_present(
            ((job, "attention_slicing"), (manager_settings, "attention_slicing")),
            options.attention_slicing,
        )
    )
    vae_tiling = bool(
        _first_present(
            ((job, "vae_tiling"), (manager_settings, "vae_tiling")),
            options.vae_tiling,
        )
    )
    vae_slicing = bool(
        _first_present(
            ((job, "vae_slicing"), (manager_settings, "vae_slicing")),
            options.vae_slicing,
        )
    )
    vae_device = str(
        _first_present(
            ((job, "vae_device"), (manager_settings, "vae_device")),
            options.vae_device,
        )
    )
    preview_policy_value = str(
        _first_present(
            ((job, "preview_policy"), (manager_settings, "preview_policy")),
            options.preview_policy,
        )
    )
    hires_memory_profile = str(
        _first_present(
            (
                (job, "hires_memory_profile"),
                (manager_settings, "hires_memory_profile"),
            ),
            options.hires_memory_profile,
        )
    )
    pre_hires_cleanup = bool(
        _first_present(
            ((job, "pre_hires_cleanup"), (manager_settings, "pre_hires_cleanup")),
            options.pre_hires_cleanup,
        )
    )
    oom_retry_profile = str(
        _first_present(
            ((job, "oom_retry_profile"), (manager_settings, "oom_retry_profile")),
            options.oom_retry_profile,
        )
    )
    oom_retry_limit = int(
        _first_present(
            ((job, "oom_retry_limit"), (manager_settings, "oom_retry_limit")),
            options.oom_retry_limit,
        )
    )

    versions = _mapping(attention.get("versions"))
    processor_classes = _string_list(
        attention.get("processor_types_after")
        or attention.get("effective_processor")
        or []
    )
    processor_modules = _string_list(attention.get("processor_modules_after") or [])
    verified_provider = attention.get("kernel_provider")
    provider_verified = bool(
        verified_provider
        and (
            attention.get("operator_executed")
            or attention.get("custom_provider_executed")
            or str(verified_provider).startswith("torch_")
        )
    )
    if not provider_verified:
        verified_provider = None

    requested_mslk = options.mslk_fmha.to_dict()
    effective_mslk = {
        field: str(target_environment.get(env_name, requested_mslk.get(field, "")) or "")
        for field, env_name in MSLK_ENVIRONMENT_FIELDS.items()
    }
    mslk_environment = {
        env_name: effective_mslk[field]
        for field, env_name in MSLK_ENVIRONMENT_FIELDS.items()
    }

    allocator = build_cuda_allocator_diagnostics(options, environment=target_environment)
    oom = _mapping(memory.get("oom_recovery"))
    peak_by_stage = _mapping(memory.get("peak_vram_by_stage"))
    hires_cleanup = list(memory.get("hires_cleanup_reports") or [])
    if not hires_cleanup:
        report = hires.get("pre_hires_cleanup")
        if isinstance(report, Mapping):
            hires_cleanup = [dict(report)]

    restorable = {
        "memory_policy": memory_policy,
        "memory_vram_safety_margin_mb": safety_margin_mb,
        "memory_retain_checkpoint_between_jobs": retain_unet,
        "memory_retain_vae_between_jobs": retain_vae,
        "model_runtime_retain_text_encoder_between_jobs": retain_text_encoder,
        "attention_slicing": attention_slicing,
        "vae_tiling": vae_tiling,
        "vae_slicing": vae_slicing,
        "vae_device": vae_device,
        "preview_policy": preview_policy_value,
        "hires_memory_profile": hires_memory_profile,
        "pre_hires_cleanup": pre_hires_cleanup,
        "oom_retry_profile": oom_retry_profile,
        "oom_retry_limit": oom_retry_limit,
    }

    return {
        "schema_version": RUNTIME_EXECUTION_SCHEMA_VERSION,
        "format": RUNTIME_EXECUTION_FORMAT,
        "runtime_profile": {
            "profile_id": str(options.runtime_profile.profile_id),
            "schema_version": int(options.runtime_profile.schema_version),
            "label": str(options.runtime_profile.label),
            "source": str(options.runtime_profile.source),
            "selector": str(options.runtime_profile.selector),
            "selected_from": str(options.runtime_profile.selected_from),
        },
        "startup_options_schema_version": int(options.schema_version),
        "attention": {
            "requested_backend": str(options.attention_backend),
            "effective_backend": str(attention.get("effective_backend") or "unverified"),
            "processor_classes": [str(item) for item in processor_classes],
            "processor_modules": [str(item) for item in processor_modules],
            "verified_kernel_provider": verified_provider,
            "provider_verified": provider_verified,
            "effective_operator": attention.get("effective_operator"),
            "backend_verified": bool(attention.get("verified")),
            "fallback_reason": attention.get("fallback_reason"),
            "xformers_version": versions.get("xformers"),
            "mslk_version": versions.get("mslk"),
            "triton_version": versions.get("triton"),
            "torch_version": versions.get("torch"),
            "mslk_fmha": {
                "requested": requested_mslk,
                "effective": effective_mslk,
                "effective_environment": mslk_environment,
                "source_map": {
                    field: str(options.source_map.get(f"mslk_fmha.{field}", "default"))
                    for field in MSLK_ENVIRONMENT_FIELDS
                },
                "startup_configuration": attention.get("mslk_startup_configuration"),
                "kernel_first_use": attention.get("mslk_kernel_first_use"),
            },
            "attention_slicing": attention_slicing,
        },
        "memory": {
            "requested_policy": memory_policy,
            "effective_policy_by_stage": _effective_memory_policy_by_stage(memory),
            "vram_safety_margin_mb": safety_margin_mb,
            "component_retention": {
                "unet_between_jobs": retain_unet,
                "vae_between_jobs": retain_vae,
                "text_encoder_between_jobs": retain_text_encoder,
            },
            "peak_vram_by_stage": peak_by_stage,
            "execution_device": str(execution_device or ""),
        },
        "vae": {
            "requested": {
                "tiling": vae_tiling,
                "slicing": vae_slicing,
                "device": vae_device,
            },
            "effective": vae,
        },
        "preview": {
            "requested_policy": preview_policy_value,
            "effective_by_stage": preview,
            "image_decode_suspended": bool(memory.get("preview_image_decode_suspended")),
            "suspension_reason": str(memory.get("preview_image_decode_suspension_reason") or ""),
            "suspension_source": str(memory.get("preview_image_decode_suspension_source") or ""),
            "one_way_for_job": bool(memory.get("preview_suspension_one_way_for_job", True)),
            "suspension_events": _preview_suspension_events(memory),
        },
        "hires": {
            "requested_memory_profile": hires_memory_profile,
            "memory_behavior": _mapping(hires.get("memory_behavior")),
            "pre_hires_cleanup_requested": pre_hires_cleanup,
            "pre_hires_cleanup_reports": hires_cleanup,
        },
        "cuda_allocator": allocator,
        "oom_recovery": {
            "configured_profile": oom_retry_profile,
            "retry_limit": oom_retry_limit,
            "history": oom,
        },
        "external_fallback_reasons": _external_fallback_reasons(
            attention_report=attention,
            memory_summary=memory,
            preview_policy=preview,
            external_fallback_reasons=external_fallback_reasons,
        ),
        "replay": {
            "restorable_job_settings": restorable,
            "process_start_settings": {
                "attention_backend": str(options.attention_backend),
                "runtime_profile": options.runtime_profile.to_dict(),
                "mslk_fmha": requested_mslk,
                "allocator_options": dict(options.allocator_options),
            },
            "process_restart_required_for": [
                "attention_backend",
                "mslk_fmha",
                "cuda_allocator",
            ],
            "runtime_path_changed_by_oom_recovery": bool(oom.get("runtime_path_changed")),
        },
        "runtime_startup_options": startup,
    }


def extract_runtime_execution_record(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(manifest)
    extra = _mapping(source.get("extra"))
    direct = _mapping(extra.get("runtime_execution"))
    if direct:
        return direct
    pipeline = _mapping(extra.get("pipeline_metadata"))
    return _mapping(pipeline.get("runtime_execution"))


def _package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def runtime_replay_assessment(
    recorded_record: Mapping[str, Any] | None,
    active_options: RuntimeStartupOptions | Mapping[str, Any] | None,
    *,
    outgoing_request: Mapping[str, Any] | None = None,
    active_attention_report: Mapping[str, Any] | None = None,
    package_availability: Mapping[str, bool] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    recorded = _mapping(recorded_record)
    if not recorded:
        return {
            "schema_version": 1,
            "recorded_runtime_available": False,
            "can_continue": True,
            "exact_runtime_match": False,
            "exact_replay_supported": False,
            "warnings": [
                "This output does not contain Phase 14K-11 runtime execution metadata. "
                "Replay may continue, but the original runtime backend and memory path cannot be verified."
            ],
            "substitutions": [],
            "restored_job_settings": {},
            "material_differences": ["runtime_execution_metadata_missing"],
        }

    active = (
        active_options
        if isinstance(active_options, RuntimeStartupOptions)
        else RuntimeStartupOptions.from_mapping(active_options)
    )
    active_attention = _mapping(active_attention_report)
    active_environment = environment if environment is not None else os.environ
    packages = {
        "xformers": _package_available("xformers"),
        "mslk": _package_available("mslk"),
        "triton": _package_available("triton"),
    }
    packages.update({str(key): bool(value) for key, value in _mapping(package_availability).items()})

    warnings: list[str] = []
    substitutions: list[dict[str, Any]] = []
    material: list[str] = []

    recorded_attention = _mapping(recorded.get("attention"))
    recorded_backend = str(recorded_attention.get("effective_backend") or "unverified")
    recorded_provider = str(recorded_attention.get("verified_kernel_provider") or "")
    active_requested = str(active.attention_backend)
    active_effective = str(active_attention.get("effective_backend") or "")
    active_provider = str(active_attention.get("kernel_provider") or "")

    recorded_profile = _mapping(recorded.get("runtime_profile"))
    active_profile = active.runtime_profile.to_dict()
    if (
        str(recorded_profile.get("profile_id") or "")
        and (
            str(recorded_profile.get("profile_id") or "")
            != str(active_profile.get("profile_id") or "")
            or int(recorded_profile.get("schema_version") or 0)
            != int(active_profile.get("schema_version") or 0)
        )
    ):
        warnings.append(
            "The recorded runtime profile identity differs from the active process. "
            "Per-job memory values can be restored, but startup-only profile behavior may differ."
        )
        substitutions.append(
            {
                "setting": "runtime_profile",
                "recorded": recorded_profile,
                "active": active_profile,
                "reason": "active_runtime_profile_differs",
            }
        )
        material.append("runtime_profile_differs")

    if recorded_backend == "xformers" and not packages.get("xformers", False):
        warnings.append(
            "The recorded effective attention backend was xformers, but xformers is unavailable in the active runtime. "
            "Replay may continue with the active backend as a documented substitution."
        )
        substitutions.append(
            {
                "setting": "attention_backend",
                "recorded": recorded_backend,
                "active": active_effective or active_requested,
                "reason": "recorded_backend_unavailable",
            }
        )
        material.append("attention_backend_unavailable")
    elif active_effective and active_effective not in {"unverified", recorded_backend}:
        warnings.append(
            f"The recorded effective attention backend was {recorded_backend!r}, but the active model reports {active_effective!r}. "
            "Replay will continue with the active backend and is not an exact runtime replay."
        )
        substitutions.append(
            {
                "setting": "attention_backend",
                "recorded": recorded_backend,
                "active": active_effective,
                "reason": "effective_backend_differs",
            }
        )
        material.append("effective_attention_backend_differs")
    elif not active_effective and active_requested not in {"auto", recorded_backend}:
        warnings.append(
            f"The recorded effective attention backend was {recorded_backend!r}, while the active process requests {active_requested!r}. "
            "The active backend will be used as a documented substitution."
        )
        substitutions.append(
            {
                "setting": "attention_backend",
                "recorded": recorded_backend,
                "active": active_requested,
                "reason": "requested_backend_differs",
            }
        )
        material.append("requested_attention_backend_differs")
    elif not active_effective and active_requested == "auto":
        warnings.append(
            "The active process uses automatic attention selection and has not reported a verified effective backend yet. "
            "Replay may continue, but the recorded backend cannot be guaranteed before generation."
        )
        material.append("active_attention_backend_unverified")
    elif not active_effective:
        warnings.append(
            "The active model has not reported its effective attention backend yet. "
            "The requested backend matches the recorded backend; the replayed generation will verify the executed provider in its own metadata."
        )

    if recorded_provider:
        provider_available = True
        if recorded_provider.startswith("mslk_") or "triton_split" in recorded_provider:
            provider_available = bool(
                packages.get("mslk")
                and packages.get("xformers")
                and packages.get("triton")
            )
        if not provider_available:
            warnings.append(
                f"The recorded kernel provider {recorded_provider!r} is unavailable in the active environment. "
                "Replay may continue with a substituted provider."
            )
            substitutions.append(
                {
                    "setting": "kernel_provider",
                    "recorded": recorded_provider,
                    "active": active_provider or None,
                    "reason": "recorded_provider_unavailable",
                }
            )
            material.append("kernel_provider_unavailable")
        elif active_provider and active_provider != recorded_provider:
            warnings.append(
                f"The recorded kernel provider was {recorded_provider!r}, but the active model reports {active_provider!r}."
            )
            substitutions.append(
                {
                    "setting": "kernel_provider",
                    "recorded": recorded_provider,
                    "active": active_provider,
                    "reason": "kernel_provider_differs",
                }
            )
            material.append("kernel_provider_differs")

    recorded_mslk = _mapping(_mapping(recorded_attention.get("mslk_fmha")).get("effective"))
    active_mslk = {
        field: str(
            active_environment.get(env_name, active.mslk_fmha.to_dict().get(field, ""))
            or ""
        )
        for field, env_name in MSLK_ENVIRONMENT_FIELDS.items()
    }
    mslk_differences = {
        key: {"recorded": recorded_mslk.get(key, ""), "active": active_mslk.get(key, "")}
        for key in MSLK_ENVIRONMENT_FIELDS
        if str(recorded_mslk.get(key, "")) != str(active_mslk.get(key, ""))
    }
    if mslk_differences:
        warnings.append(
            "Recorded MSLK FMHA values differ from the active process. These settings require restart before attention initialization; "
            "replay will continue with the active values and is not exact."
        )
        substitutions.append(
            {
                "setting": "mslk_fmha",
                "recorded": recorded_mslk,
                "active": active_mslk,
                "differences": mslk_differences,
                "reason": "restart_required",
            }
        )
        material.append("mslk_fmha_differs")

    recorded_allocator = _mapping(recorded.get("cuda_allocator"))
    active_allocator = build_cuda_allocator_diagnostics(
        active,
        environment=active_environment,
    )
    if str(recorded_allocator.get("effective_config") or "") != str(
        active_allocator.get("effective_config") or ""
    ):
        warnings.append(
            "Recorded CUDA allocator settings differ from the active process. Allocator settings cannot be changed after CUDA initialization; "
            "restart with the recorded environment for exact runtime replay."
        )
        substitutions.append(
            {
                "setting": "cuda_allocator",
                "recorded": recorded_allocator.get("effective_config"),
                "active": active_allocator.get("effective_config"),
                "reason": "restart_required",
            }
        )
        material.append("cuda_allocator_differs")

    restorable = runtime_replay_request_values(recorded)
    outgoing = _mapping(outgoing_request)
    restored: dict[str, Any] = {}
    for key, recorded_value in restorable.items():
        active_value = outgoing.get(key, recorded_value)
        restored[key] = active_value
        if active_value != recorded_value:
            substitutions.append(
                {
                    "setting": key,
                    "recorded": recorded_value,
                    "active": active_value,
                    "reason": "job_setting_not_restored",
                }
            )
            material.append(f"job_setting_differs:{key}")

    replay_info = _mapping(recorded.get("replay"))
    if replay_info.get("runtime_path_changed_by_oom_recovery"):
        warnings.append(
            "The recorded generation used an OOM recovery fallback path. Creative request values are preserved, but exact runtime-path replay "
            "requires reproducing the recorded OOM and fallback sequence."
        )
        material.append("recorded_oom_fallback_path")

    return {
        "schema_version": 1,
        "recorded_runtime_available": True,
        "can_continue": True,
        "exact_runtime_match": not material,
        "exact_replay_supported": not material,
        "warnings": _dedupe_strings(warnings),
        "substitutions": substitutions,
        "restored_job_settings": restored,
        "material_differences": _dedupe_strings(material),
        "recorded": {
            "attention_backend": recorded_backend,
            "kernel_provider": recorded_provider or None,
            "runtime_profile": _mapping(recorded.get("runtime_profile")),
        },
        "active": {
            "requested_attention_backend": active_requested,
            "effective_attention_backend": active_effective or None,
            "kernel_provider": active_provider or None,
            "runtime_profile": active.runtime_profile.to_dict(),
            "package_availability": packages,
        },
    }


__all__ = [
    "MSLK_ENVIRONMENT_FIELDS",
    "RUNTIME_CONFORMANCE_FORMAT",
    "RUNTIME_CONFORMANCE_SCHEMA_VERSION",
    "RUNTIME_EXECUTION_FORMAT",
    "RUNTIME_EXECUTION_SCHEMA_VERSION",
    "RUNTIME_REPLAY_JOB_FIELDS",
    "build_runtime_execution_record",
    "compare_runtime_execution_records",
    "extract_runtime_execution_record",
    "runtime_execution_conformance_snapshot",
    "runtime_execution_fingerprint",
    "runtime_replay_assessment",
    "runtime_replay_request_values",
]
