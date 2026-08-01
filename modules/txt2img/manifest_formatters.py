from __future__ import annotations

import json
from typing import Any

from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import GenerationManifest, AssetReference


def _json_compact(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def _asset_label(asset: AssetReference) -> str:
    return (
        asset.requested_label
        or asset.resolved_label
        or asset.asset_type
        or "unknown"
    )


def manifest_to_infotext(
    manifest: GenerationManifest,
    include_optional: bool = True,
    include_runtime: bool = True,
    include_assets: bool = True,
) -> str:
    req = manifest.required_for_rerun
    opt = manifest.optional_for_rerun
    run = manifest.runtime_info

    lines: list[str] = []
    lines.append(f"Prompt: {req.prompt}")
    lines.append(f"Negative prompt: {req.negative_prompt}")

    lines.append(
        f"Steps: {req.steps}, "
        f"CFG scale: {req.cfg_scale}, "
        f"Seed: {req.seed}, "
        f"Size: {req.width}x{req.height}, "
        f"Batch size: {req.batch_size}, "
        f"Batch count: {req.batch_count}"
    )
    lines.append(
        f"Sampler: {req.sampler_name}, "
        f"Scheduler: {req.scheduler_name}, "
        f"Model: {req.model_path}"
    )
    model_provenance = dict(manifest.extra.get("model_provenance") or {})
    if model_provenance:
        lines.append(
            f"Requested model path: {model_provenance.get('requested_path', '')}"
        )
        lines.append(
            f"Resolved model path: {model_provenance.get('resolved_path', '')}"
        )
        lines.append(
            f"Loaded model path: {model_provenance.get('loaded_path', '')}"
        )
        if model_provenance.get("sha256"):
            lines.append(f"Loaded model SHA-256: {model_provenance['sha256']}")
        if model_provenance.get("architecture"):
            lines.append(
                f"Loaded model architecture: {model_provenance['architecture']}"
            )
        lines.append(
            f"Model cache reused: {bool(model_provenance.get('cache_reused', False))}"
        )

    if include_optional:
        if opt.compatibility_mode:
            lines.append(f"Compatibility mode: {opt.compatibility_mode}")
        if opt.clip_skip is not None:
            lines.append(f"Clip skip: {opt.clip_skip}")
        if opt.guidance_rescale is not None:
            lines.append(f"Guidance rescale: {opt.guidance_rescale}")
        if opt.tiling is not None:
            lines.append(f"Tiling: {opt.tiling}")
        if opt.scheduler_kwargs:
            lines.append(f"Scheduler kwargs: {_json_compact(opt.scheduler_kwargs)}")
        scheduler_audit = dict(manifest.extra.get("scheduler_settings") or {})
        if scheduler_audit:
            preset = dict(scheduler_audit.get("preset_reference") or {})
            if preset.get("name"):
                lines.append(f"Scheduler preset reference: {preset.get('name')}")
            warnings = list(scheduler_audit.get("validation_warnings") or [])
            lines.append(f"Scheduler validation warnings: {_json_compact(warnings)}")
            lines.append(
                f"Scheduler compatibility policy: {_json_compact(scheduler_audit.get('compatibility_policy') or {})}"
            )
            if scheduler_audit.get("effective_hash"):
                lines.append(f"Scheduler effective SHA-256: {scheduler_audit['effective_hash']}")
        if opt.sampler_kwargs:
            lines.append(f"Sampler kwargs: {_json_compact(opt.sampler_kwargs)}")
        guidance_metadata = {
            key: manifest.extra.get(key)
            for key in (
                "guidance_owner", "guidance_mode", "guidance_math_version",
                "cfg_rescale", "cfg_rescale_applied", "legacy_clamp_guidance",
                "cfg_effective_range", "cfg_effective_guidance_summary",
            )
            if key in manifest.extra
        }
        if guidance_metadata:
            lines.append(f"Guidance metadata: {_json_compact(guidance_metadata)}")
        cfg_step_series = manifest.extra.get("cfg_step_series")
        if cfg_step_series:
            lines.append(f"CFG step series: {_json_compact(cfg_step_series)}")
        if opt.extra:
            lines.append(f"Optional extras: {_json_compact(opt.extra)}")

    if include_runtime:
        if run.effective_steps is not None:
            lines.append(f"Effective steps: {run.effective_steps}")
        if run.scheduler_step_override_applied is not None:
            lines.append(
                f"Scheduler override applied: {run.scheduler_step_override_applied}"
            )
        if run.active_blend_methods:
            lines.append(f"Active blend methods: {_json_compact(run.active_blend_methods)}")
        if run.active_blend_weights:
            lines.append(f"Active blend weights: {_json_compact(run.active_blend_weights)}")
        if run.tail_features_used:
            lines.append(f"Tail features used: {_json_compact(run.tail_features_used)}")
        if run.predicted_stop_step is not None:
            lines.append(f"Predicted stop step: {run.predicted_stop_step}")
        if run.device:
            lines.append(f"Device: {run.device}")
        if run.generation_time_sec is not None:
            lines.append(f"Generation time sec: {run.generation_time_sec}")
        if run.timestamp:
            lines.append(f"Timestamp: {run.timestamp}")

    if include_assets:
        if manifest.base_model.requested_label or manifest.base_model.resolved_label:
            lines.append(
                f"Base model asset: {_asset_label(manifest.base_model)} "
                f"[status={manifest.base_model.resolution_status}]"
            )
        if manifest.vae.requested_label or manifest.vae.resolved_label:
            lines.append(
                f"VAE asset: {_asset_label(manifest.vae)} "
                f"[status={manifest.vae.resolution_status}]"
            )

        if manifest.loras:
            lora_summary = [
                f"{_asset_label(asset)} ({asset.resolution_status})"
                for asset in manifest.loras
            ]
            lines.append(f"LoRAs: {', '.join(lora_summary)}")

        if manifest.embeddings:
            emb_summary = [
                f"{_asset_label(asset)} ({asset.resolution_status})"
                for asset in manifest.embeddings
            ]
            lines.append(f"Embeddings: {', '.join(emb_summary)}")

    return "\n".join(lines)


def manifest_to_warning_lines(manifest: GenerationManifest) -> list[str]:
    warnings: list[str] = []

    all_assets = [manifest.base_model, manifest.vae] + manifest.loras + manifest.embeddings + manifest.hypernetworks + manifest.extras
    for asset in all_assets:
        for msg in asset.warning_messages:
            warnings.append(f"{asset.asset_type}: {msg}")

        if asset.resolution_status in {"missing", "missing_used_ui_default", "unresolved"}:
            warnings.append(
                f"{asset.asset_type}: requested '{asset.requested_label}' -> status '{asset.resolution_status}'"
            )

    return warnings


def manifest_to_flat_row(manifest: GenerationManifest) -> dict[str, Any]:
    req = manifest.required_for_rerun
    opt = manifest.optional_for_rerun
    run = manifest.runtime_info

    return {
        "manifest_version": manifest.manifest_version,
        "manifest_type": manifest.manifest_type,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": req.seed,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "batch_size": req.batch_size,
        "batch_count": req.batch_count,
        "sampler_name": req.sampler_name,
        "scheduler_name": req.scheduler_name,
        "model_path": req.model_path,
        "compatibility_mode": opt.compatibility_mode,
        "clip_skip": opt.clip_skip,
        "guidance_rescale": opt.guidance_rescale,
        "tiling": opt.tiling,
        "scheduler_kwargs": _json_compact(opt.scheduler_kwargs) if opt.scheduler_kwargs else "",
        "sampler_kwargs": _json_compact(opt.sampler_kwargs) if opt.sampler_kwargs else "",
        "effective_steps": run.effective_steps,
        "scheduler_step_override_applied": run.scheduler_step_override_applied,
        "active_blend_methods": _json_compact(run.active_blend_methods) if run.active_blend_methods else "",
        "active_blend_weights": _json_compact(run.active_blend_weights) if run.active_blend_weights else "",
        "tail_features_used": _json_compact(run.tail_features_used) if run.tail_features_used else "",
        "predicted_stop_step": run.predicted_stop_step,
        "timestamp": run.timestamp,
        "device": run.device,
        "generation_time_sec": run.generation_time_sec,
        "output_image_path": run.output_image_path,
        "base_model_requested": manifest.base_model.requested_label,
        "base_model_resolved": manifest.base_model.resolved_label,
        "base_model_status": manifest.base_model.resolution_status,
        "vae_requested": manifest.vae.requested_label,
        "vae_resolved": manifest.vae.resolved_label,
        "vae_status": manifest.vae.resolution_status,
        "loras_count": len(manifest.loras),
        "embeddings_count": len(manifest.embeddings),
        "warnings": " | ".join(manifest_to_warning_lines(manifest)),
    }