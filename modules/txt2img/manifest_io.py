from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from image_gen.runtime_options import runtime_execution_fingerprint
from image_gen.systems.diagnostics.serialization import json_safe
from image_gen.systems.regional_prompting import compact_region_record_for_replay
from modules.txt2img.generation_manifest import AssetReference, GenerationManifest
from modules.txt2img.manifest_formatters import manifest_to_infotext


SERIALIZATION_PROFILE_COMPACT = "compact_replay_v1"
SERIALIZATION_PROFILE_DIAGNOSTICS = "diagnostics_v1"


_DIAGNOSTIC_REDUNDANT_TOP_LEVEL_KEYS = {
    "live_sampler_map",
    "live_scheduler_map",
    "resolved_sampler_entry",
    "resolved_scheduler_entry",
    "resolved_sampler_descriptor",
    "resolved_scheduler_descriptor",
    "resolved_hires_sampler_descriptor",
    "resolved_hires_scheduler_descriptor",
}

_NON_GENERATION_SCHEDULER_KEYS = {
    "sigma_save_subfolder",
    "graph_save_enable",
    "graph_save_directory",
    "log_save_directory",
    "debug",
    "verbose",
    "save_prepass_sigmas",
    "save_sigma_cache",
}

_HIRES_REPLAY_FIELDS = (
    "hires_prompt_parser_mode",
    "hires_prompt_parser_name",
    "hires_prompt_parser_kwargs",
    "hires_shortcut_profile_mode",
    "hires_shortcut_profile_name",
    "hires_positive_prompt",
    "hires_negative_prompt",
    "hires_size_mode",
    "hires_scale",
    "hires_width",
    "hires_height",
    "hires_dimension_plan_version",
    "hires_dimension_plan",
    "hires_enabled",
    "hires_steps",
    "hires_denoising_strength",
    "hires_step_policy",
    "hires_sampler_name",
    "hires_scheduler_name",
    "hires_cfg_scale",
    "hires_cfg_rescale",
    "hires_prompt_route_plan",
    "hires_strategy",
    "hires_upscaler",
    "hires_upscaler_id",
    "hires_expected_upscaler_sha256",
    "hires_expected_native_scale",
    "hires_expected_vae_sha256",
    "hires_expected_vae_source_kind",
    "hires_tile_size",
    "hires_tile_overlap",
    "hires_tile_batch_size",
    "hires_exact_resize_filter",
    "hires_final_size_correction_filter",
    "hires_aspect_policy",
    "hires_padding_mode",
    "hires_blurred_edge_method",
    "hires_blurred_edge_compare_diagnostics",
    "hires_recorded_target_correction",
    "hires_correction_fingerprint_enabled",
    "hires_recorded_correction_fingerprint",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _drop_empty(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            cleaned = _drop_empty(child)
            if _is_empty(cleaned):
                continue
            output[str(key)] = cleaned
        return output
    if isinstance(value, list):
        output = [_drop_empty(child) for child in value]
        return [child for child in output if not _is_empty(child)]
    return value


def _compact_asset(asset: AssetReference, *, include_replay_metadata: bool = False) -> dict[str, Any]:
    source = json_safe(asset.to_dict())
    if not isinstance(source, dict):
        return {}
    keys = (
        "asset_type",
        "provider",
        "requested_display_name",
        "requested_filename",
        "requested_path",
        "requested_identifier",
        "requested_version",
        "requested_hash",
        "requested_hash_type",
        "resolved_display_name",
        "resolved_filename",
        "resolved_path",
        "resolved_identifier",
        "resolved_version",
        "resolved_hash",
        "resolved_hash_type",
        "resolution_status",
        "was_found",
        "was_used_for_generation",
        "is_required_for_rerun",
        "source_url",
    )
    output = {key: source.get(key) for key in keys if key in source}
    if include_replay_metadata and isinstance(source.get("extra"), Mapping):
        # Prompt-asset weight, activation, compatibility hashes, and source
        # metadata can affect exact replay. Keep that small asset-specific
        # record instead of duplicating the full prompt-asset contract.
        output["extra"] = source.get("extra")
    return _drop_empty(output)


def _compact_scheduler_kwargs(value: Any) -> dict[str, Any]:
    source = _mapping(json_safe(value or {}))
    for key in _NON_GENERATION_SCHEDULER_KEYS:
        source.pop(key, None)

    # Randomization ranges/types do not influence a run only when all three
    # activation paths are off: the per-setting min/max switch, the per-setting
    # type switch, and global randomization. Keep both explicit false switches
    # so a future default change cannot silently enable randomization.
    global_randomize = bool(source.get("global_randomize", False))
    for key, range_enabled in list(source.items()):
        if not key.endswith("_rand"):
            continue
        prefix = key[:-5]
        type_key = prefix + "_enable_randomization_type"
        type_enabled = bool(source.get(type_key, False))
        if global_randomize or bool(range_enabled) or type_enabled:
            continue
        for suffix in (
            "_rand_min",
            "_rand_max",
            "_randomization_type",
            "_randomization_percent",
        ):
            source.pop(prefix + suffix, None)
    return _drop_empty(source)


def _builtin_shortcut_snapshot(value: Any) -> bool:
    snapshot = _mapping(value)
    return bool(snapshot.get("builtin")) and bool(snapshot.get("profile_id"))


def _compact_runtime_execution(value: Any) -> dict[str, Any]:
    source = _mapping(json_safe(value or {}))
    if not source:
        return {}
    attention = _mapping(source.get("attention"))
    mslk = _mapping(attention.get("mslk_fmha"))
    replay = _mapping(source.get("replay"))
    allocator = _mapping(source.get("cuda_allocator"))
    fingerprint = runtime_execution_fingerprint(source)
    output = {
        "schema_version": source.get("schema_version"),
        "format": source.get("format"),
        "runtime_profile": {
            key: _mapping(source.get("runtime_profile")).get(key)
            for key in ("profile_id", "schema_version", "source")
        },
        "attention": {
            "requested_backend": attention.get("requested_backend"),
            "effective_backend": attention.get("effective_backend"),
            "verified_kernel_provider": attention.get("verified_kernel_provider"),
            "effective_operator": attention.get("effective_operator"),
            "attention_slicing": attention.get("attention_slicing"),
            "mslk_fmha": {
                "effective": _mapping(mslk.get("effective")),
            },
        },
        "cuda_allocator": {
            "requested_config": allocator.get("requested_config"),
            "effective_config": allocator.get("effective_config"),
        },
        "replay": {
            "restorable_job_settings": _mapping(replay.get("restorable_job_settings")),
            "process_start_settings": _mapping(replay.get("process_start_settings")),
            "runtime_path_changed_by_oom_recovery": bool(
                replay.get("runtime_path_changed_by_oom_recovery", False)
            ),
        },
        # The SHA is sufficient for exact runtime-path conformance checks. The
        # full conformance snapshot remains available in diagnostics JSON.
        "conformance_fingerprint": {
            "schema_version": fingerprint.get("schema_version"),
            "format": fingerprint.get("format"),
            "sha256": fingerprint.get("sha256"),
        },
    }
    return _drop_empty(output)


def _compact_schedule_replay_record(value: Any) -> dict[str, Any]:
    record = _mapping(json_safe(value or {}))
    if not record:
        return {}
    # Scheduler configuration already lives in optional_for_rerun.scheduler_kwargs.
    # Recorded schedule tensors keep the exact raw bytes; their expanded float
    # arrays are redundant and make hires sidecars much larger.
    record.pop("scheduler_configuration", None)
    for schedule_name in ("full_schedule", "active_schedule"):
        schedule = _mapping(record.get(schedule_name))
        if not schedule:
            continue
        for tensor_name in ("sigmas", "timesteps"):
            tensor = _mapping(schedule.get(tensor_name))
            if tensor.get("bytes_base64"):
                tensor.pop("values", None)
                schedule[tensor_name] = tensor
        record[schedule_name] = schedule
    return _drop_empty(record)


def _compact_region_pass_records(value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pass_name, record in _mapping(value).items():
        if not isinstance(record, Mapping):
            continue
        compact = compact_region_record_for_replay(record)
        if compact:
            output[str(pass_name)] = compact
    return output


def _minimal_model_provenance(manifest: GenerationManifest) -> dict[str, Any]:
    source = _mapping(json_safe(manifest.extra.get("model_provenance") or {}))
    base = manifest.base_model
    output = {
        "requested_path": source.get("requested_path") or base.requested_path or manifest.required_for_rerun.model_path,
        "resolved_path": source.get("resolved_path") or base.resolved_path,
        "loaded_path": source.get("loaded_path") or base.resolved_path,
        "file_name": source.get("file_name") or base.resolved_filename or base.requested_filename,
        "model_name": source.get("model_name"),
        "model_name_source": source.get("model_name_source"),
        "sha256": source.get("sha256") or base.resolved_hash or base.requested_hash,
        "architecture": source.get("architecture"),
        "prediction_type": source.get("prediction_type"),
        "conditioning_dimension": source.get("conditioning_dimension"),
        "checkpoint_kind": source.get("checkpoint_kind"),
        "dtype": source.get("dtype"),
    }
    return _drop_empty(output)


def _recorded_hires_replay_fields(manifest: GenerationManifest) -> dict[str, Any]:
    pipeline = _mapping(manifest.extra.get("pipeline_metadata"))
    hires = _mapping(pipeline.get("hires_fix"))
    optional_extra = _mapping(manifest.optional_for_rerun.extra)
    output: dict[str, Any] = {}
    replay = hires.get("schedule_replay") or optional_extra.get(
        "hires_recorded_schedule_replay"
    )
    fingerprint = hires.get("schedule_fingerprint") or optional_extra.get(
        "hires_recorded_schedule_fingerprint"
    )
    if isinstance(replay, Mapping) and replay:
        output["hires_recorded_schedule_replay"] = _compact_schedule_replay_record(replay)
    if isinstance(fingerprint, Mapping) and fingerprint:
        output["hires_recorded_schedule_fingerprint"] = json_safe(dict(fingerprint))

    source = _mapping(hires.get("pixel_source_preparation"))
    upscale = _mapping(source.get("upscale_metadata"))
    vae_encode = _mapping(source.get("vae_encode"))
    vae = _mapping(vae_encode.get("vae"))
    plan = _mapping(hires.get("upscale_plan"))
    descriptor = _mapping(plan.get("descriptor"))
    diagnostics = _mapping(hires.get("phase14n7_diagnostics"))
    diagnostic_upscaler = _mapping(diagnostics.get("upscaler"))
    diagnostic_vae = _mapping(diagnostics.get("vae"))

    strategy = diagnostics.get("strategy") or plan.get("strategy")
    upscaler_hash = (
        upscale.get("upscaler_sha256")
        or descriptor.get("sha256")
        or diagnostic_upscaler.get("sha256")
    )
    vae_hash = vae.get("sha256") or diagnostic_vae.get("sha256")
    vae_source = vae.get("source_kind") or diagnostic_vae.get("source_kind")
    if strategy:
        output.setdefault("hires_strategy", strategy)
    if upscaler_hash:
        output["hires_expected_upscaler_sha256"] = str(upscaler_hash).casefold()
    native_scale = int(
        upscale.get("upscaler_native_scale")
        or descriptor.get("native_scale")
        or diagnostic_upscaler.get("native_scale")
        or 0
    )
    if native_scale:
        output["hires_expected_native_scale"] = native_scale
    correction = upscale.get("target_correction")
    if isinstance(correction, Mapping) and correction:
        output["hires_recorded_target_correction"] = json_safe(dict(correction))
        output["hires_aspect_policy"] = str(correction.get("aspect_policy") or "stretch")
        output["hires_padding_mode"] = str(correction.get("padding_mode") or "reflect")
        output["hires_blurred_edge_method"] = str(correction.get("blurred_edge_method") or "box")
        output["hires_blurred_edge_compare_diagnostics"] = bool(
            correction.get("blurred_edge_compare_diagnostics", False)
        )
        output["hires_final_size_correction_filter"] = str(
            correction.get("final_size_correction_filter_requested") or "auto"
        )
    correction_fingerprint = upscale.get("correction_fingerprint")
    if isinstance(correction_fingerprint, Mapping) and correction_fingerprint:
        output["hires_correction_fingerprint_enabled"] = True
        output["hires_recorded_correction_fingerprint"] = json_safe(dict(correction_fingerprint))
    if vae_hash:
        output["hires_expected_vae_sha256"] = str(vae_hash).casefold()
    if vae_source:
        output["hires_expected_vae_source_kind"] = str(vae_source)
    return output


def manifest_to_replay_dict(manifest: GenerationManifest) -> dict[str, Any]:
    """Return the compact, authoritative replay sidecar payload.

    The payload intentionally contains generation inputs and reproducibility
    identity, not execution telemetry or registry/schema snapshots. Optional
    feature records are included only when they can change replay behavior.
    """

    required = json_safe(manifest.required_for_rerun.to_dict())
    optional = manifest.optional_for_rerun
    optional_extra = _mapping(json_safe(optional.extra or {}))

    compact_extra: dict[str, Any] = {
        "vae_path": optional_extra.get("vae_path"),
        "prompt_parser_name": optional_extra.get("prompt_parser_name") or "legacy",
        "prompt_shortcut_profile_name": optional_extra.get("prompt_shortcut_profile_name") or "legacy_default",
    }

    for key in (
        "prompt_parser_kwargs",
        "prompt_parser_preset_name",
        "base_prompt_parser_name",
        "base_shortcut_profile_name",
        "prompt_cfg_pass_schedules",
        "prompt_expansion_pass_records",
        "prompt_semantic_pass_records",
        "prompt_route_plan",
        "parser_kwargs",
    ):
        value = optional_extra.get(key)
        if not _is_empty(value):
            compact_extra[key] = value

    region_pass_records = _compact_region_pass_records(
        optional_extra.get("region_pass_records")
    )
    if region_pass_records:
        compact_extra["region_pass_records"] = region_pass_records
    batch_region_pass_records = _compact_region_pass_records(
        optional_extra.get("batch_region_pass_records")
    )
    if batch_region_pass_records:
        compact_extra["batch_region_pass_records"] = batch_region_pass_records
    if optional_extra.get("region_projected_image_slot") is not None:
        compact_extra["region_projected_image_slot"] = optional_extra.get(
            "region_projected_image_slot"
        )

    if not compact_extra.get("prompt_cfg_pass_schedules"):
        prompt_cfg_schedule = optional_extra.get("prompt_cfg_schedule")
        if not _is_empty(prompt_cfg_schedule):
            compact_extra["prompt_cfg_schedule"] = prompt_cfg_schedule
    if not compact_extra.get("prompt_expansion_pass_records"):
        prompt_expansion_record = optional_extra.get("prompt_expansion_record")
        if not _is_empty(prompt_expansion_record):
            compact_extra["prompt_expansion_record"] = prompt_expansion_record

    shortcut_snapshot = optional_extra.get("prompt_shortcut_profile_snapshot")
    if shortcut_snapshot and not _builtin_shortcut_snapshot(shortcut_snapshot):
        compact_extra["prompt_shortcut_profile_snapshot"] = shortcut_snapshot

    prompt_assets = _mapping(optional_extra.get("prompt_assets"))
    if (
        manifest.loras
        or manifest.embeddings
        or list(prompt_assets.get("loras") or [])
        or list(prompt_assets.get("textual_inversions") or [])
    ):
        compact_extra["prompt_asset_contract_version"] = (
            optional_extra.get("prompt_asset_contract_version")
            or prompt_assets.get("contract_version")
        )

    hires_enabled = bool(optional_extra.get("hires_enabled", False))
    # Keep the feature switch even when disabled so replay never depends on a
    # future default changing from false to true. The rest of the hires block
    # remains conditional on the feature actually being active.
    if "hires_enabled" in optional_extra:
        compact_extra["hires_enabled"] = hires_enabled
    if hires_enabled:
        for key in _HIRES_REPLAY_FIELDS:
            if key not in optional_extra:
                continue
            value = optional_extra.get(key)
            if not _is_empty(value) or value in (False, 0, 0.0):
                compact_extra[key] = value
        hires_snapshot = optional_extra.get("hires_shortcut_profile_snapshot")
        if hires_snapshot and not _builtin_shortcut_snapshot(hires_snapshot):
            compact_extra["hires_shortcut_profile_snapshot"] = hires_snapshot
        compact_extra.update(_recorded_hires_replay_fields(manifest))

    optional_payload: dict[str, Any] = {
        "scheduler_kwargs": _compact_scheduler_kwargs(optional.scheduler_kwargs),
        "sampler_kwargs": _drop_empty(json_safe(optional.sampler_kwargs or {})),
    }
    if optional.compatibility_mode is not None:
        optional_payload["compatibility_mode"] = optional.compatibility_mode
    if optional.clip_skip is not None:
        optional_payload["clip_skip"] = optional.clip_skip
    if optional.guidance_rescale is not None:
        optional_payload["guidance_rescale"] = optional.guidance_rescale
    if optional.tiling is not None:
        optional_payload["tiling"] = optional.tiling
    compact_extra = _drop_empty(compact_extra)
    if compact_extra:
        optional_payload["extra"] = compact_extra

    if hires_enabled:
        hires_plan = _mapping(optional_extra.get("hires_dimension_plan"))
        base_dimensions = _mapping(manifest.extra.get("base_dimensions"))
        base_width = int(base_dimensions.get("width") or hires_plan.get("base_width") or 0)
        base_height = int(base_dimensions.get("height") or hires_plan.get("base_height") or 0)
        if base_width > 0 and base_height > 0:
            required["width"] = base_width
            required["height"] = base_height

    payload: dict[str, Any] = {
        "manifest_version": str(manifest.manifest_version),
        "manifest_type": str(manifest.manifest_type),
        "serialization_profile": SERIALIZATION_PROFILE_COMPACT,
        "required_for_rerun": required,
        "optional_for_rerun": _drop_empty(optional_payload),
    }

    base_model = _compact_asset(manifest.base_model)
    if base_model:
        payload["base_model"] = base_model
    vae = _compact_asset(manifest.vae)
    if vae and any(vae.get(key) for key in ("requested_path", "resolved_path", "requested_hash", "resolved_hash")):
        payload["vae"] = vae

    # Prompt assets are replay inputs, so keep one compact top-level copy.
    # The legacy manifest duplicated the same records in top-level assets,
    # optional_for_rerun.extra.loras/textual_inversions, prompt_assets, and
    # pipeline metadata. One authoritative copy is sufficient.
    loras = [
        _compact_asset(asset, include_replay_metadata=True)
        for asset in manifest.loras
    ]
    loras = [asset for asset in loras if asset]
    embeddings = [
        _compact_asset(asset, include_replay_metadata=True)
        for asset in manifest.embeddings
    ]
    embeddings = [asset for asset in embeddings if asset]
    if loras:
        payload["loras"] = loras
    if embeddings:
        payload["embeddings"] = embeddings

    # Older manifests can contain prompt assets only in the optional replay
    # contract. Preserve that contract only as a fallback when no top-level
    # asset references are available.
    if not loras and not embeddings and (
        list(prompt_assets.get("loras") or [])
        or list(prompt_assets.get("textual_inversions") or [])
    ):
        compact_extra["prompt_assets"] = prompt_assets
        optional_payload["extra"] = _drop_empty(compact_extra)
        payload["optional_for_rerun"] = _drop_empty(optional_payload)

    top_extra: dict[str, Any] = {}
    application = _mapping(manifest.extra.get("application"))
    if application:
        top_extra["application"] = json_safe(application)

    model_provenance = _minimal_model_provenance(manifest)
    if model_provenance:
        top_extra["model_provenance"] = model_provenance

    runtime_execution = _compact_runtime_execution(
        manifest.extra.get("runtime_execution")
    )
    if runtime_execution:
        top_extra["runtime_execution"] = runtime_execution

    if hires_enabled:
        hires_plan = _mapping(optional_extra.get("hires_dimension_plan"))
        base_dimensions = _mapping(manifest.extra.get("base_dimensions"))
        internal_dimensions = _mapping(manifest.extra.get("internal_dimensions"))
        output_dimensions = _mapping(manifest.extra.get("output_dimensions"))
        if not base_dimensions and hires_plan:
            base_dimensions = {
                "width": hires_plan.get("base_width"),
                "height": hires_plan.get("base_height"),
            }
        if not internal_dimensions and hires_plan:
            internal_dimensions = {
                "width": hires_plan.get("internal_width") or hires_plan.get("effective_width"),
                "height": hires_plan.get("internal_height") or hires_plan.get("effective_height"),
            }
        if hires_plan:
            # Prefer the planned final/requested target over legacy manifests
            # whose output_dimensions accidentally captured the aligned canvas.
            output_dimensions = {
                "width": hires_plan.get("final_width") or hires_plan.get("requested_width") or output_dimensions.get("width"),
                "height": hires_plan.get("final_height") or hires_plan.get("requested_height") or output_dimensions.get("height"),
            }
        if base_dimensions:
            top_extra["base_dimensions"] = json_safe(base_dimensions)
        if internal_dimensions:
            top_extra["internal_dimensions"] = json_safe(internal_dimensions)
        if output_dimensions:
            top_extra["output_dimensions"] = json_safe(output_dimensions)

    if top_extra:
        payload["extra"] = _drop_empty(top_extra)
    return _drop_empty(payload)


def _prune_diagnostic_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    output = _mapping(json_safe(extra))
    for key in _DIAGNOSTIC_REDUNDANT_TOP_LEVEL_KEYS:
        output.pop(key, None)

    if output.get("scheduler_settings_resolution") == output.get("scheduler_settings"):
        output.pop("scheduler_settings_resolution", None)

    model_provenance = _mapping(output.get("model_provenance"))
    attention_backend = _mapping(output.get("attention_backend"))
    if model_provenance and attention_backend and model_provenance.get("attention_backend") == attention_backend:
        model_provenance.pop("attention_backend", None)
        output["model_provenance"] = model_provenance

    if attention_backend and attention_backend.get("xformers_compatibility") == attention_backend.get("xformers"):
        attention_backend.pop("xformers_compatibility", None)

    # Keep one release identity record. Some backends historically nested the
    # exact same object under xformers.production_dispatch as well.
    if attention_backend:
        xformers = _mapping(attention_backend.get("xformers"))
        production_dispatch = _mapping(xformers.get("production_dispatch"))
        if (
            production_dispatch.get("release_identity")
            == attention_backend.get("release_reproducibility")
        ):
            production_dispatch.pop("release_identity", None)
            if production_dispatch:
                xformers["production_dispatch"] = production_dispatch
            else:
                xformers.pop("production_dispatch", None)
            if xformers:
                attention_backend["xformers"] = xformers
        output["attention_backend"] = attention_backend

    memory_management = _mapping(output.get("memory_management"))
    if memory_management:
        snapshots = list(memory_management.get("snapshots") or [])
        if snapshots and memory_management.get("latest_snapshot") == snapshots[-1]:
            memory_management.pop("latest_snapshot", None)
        oom_recovery = _mapping(memory_management.get("oom_recovery"))
        if (
            memory_management.get("oom_recovery_actions")
            == oom_recovery.get("actions")
        ):
            memory_management.pop("oom_recovery_actions", None)
        overall_oom_actions = list(oom_recovery.get("actions") or [])
        attempts = []
        for attempt in list(oom_recovery.get("attempts") or []):
            item = _mapping(attempt)
            if item.get("actions") == overall_oom_actions:
                item.pop("actions", None)
            attempts.append(item)
        if attempts:
            oom_recovery["attempts"] = attempts
            memory_management["oom_recovery"] = oom_recovery
        output["memory_management"] = memory_management

    # Runtime execution is primarily a cross-record conformance summary in the
    # diagnostic sidecar; detailed attention/memory/VAE telemetry already has
    # dedicated top-level records. Preserve the full conformance snapshot here.
    runtime_execution = _mapping(output.get("runtime_execution"))
    if runtime_execution:
        diagnostic_runtime = _compact_runtime_execution(runtime_execution)
        diagnostic_runtime["conformance_fingerprint"] = runtime_execution_fingerprint(
            runtime_execution
        )
        output["runtime_execution"] = _drop_empty(diagnostic_runtime)

    prompt_contract = _mapping(output.get("prompt_contract"))
    prompt_preflight = _mapping(output.get("prompt_preflight"))
    if prompt_contract:
        hires_contract = _mapping(prompt_contract.get("hires"))
        if hires_contract.get("preflight") == prompt_preflight.get("hires"):
            hires_contract.pop("preflight", None)
            prompt_contract["hires"] = hires_contract
        if prompt_contract.get("shortcut_profile") == output.get("prompt_shortcut_profile"):
            prompt_contract.pop("shortcut_profile", None)
        output["prompt_contract"] = prompt_contract

    schedule = _mapping(output.get("schedule"))
    if schedule and schedule.get("validated_settings") == schedule.get("scheduler_settings"):
        schedule.pop("validated_settings", None)
        output["schedule"] = schedule

    scheduler_settings = _mapping(output.get("scheduler_settings"))
    if scheduler_settings and scheduler_settings.get("runtime_settings") == scheduler_settings.get("requested_settings"):
        scheduler_settings.pop("runtime_settings", None)
        output["scheduler_settings"] = scheduler_settings

    pipeline = _mapping(output.get("pipeline_metadata"))
    if pipeline:
        duplicate_map = {
            "attention_backend": "attention_backend",
            "model_provenance": "model_provenance",
            "memory_management": "memory_management",
            "runtime_execution": "runtime_execution",
            "prompt_parser": "prompt_parser",
            "prompt_shortcut_profile": "prompt_shortcut_profile",
            "prompt_translation": "prompt_translation",
            "prompt_contract": "prompt_contract",
            "denoising_contract": "denoising_contract",
            "vae_provenance": "vae_provenance",
        }
        for pipeline_key, top_key in duplicate_map.items():
            # Pipeline metadata historically carried another copy of these
            # diagnostics. Keep the dedicated top-level record as the single
            # authoritative diagnostic copy even when the pipeline snapshot
            # contains a few extra formatting/projection fields.
            if top_key in output:
                pipeline.pop(pipeline_key, None)

        memory_management = _mapping(output.get("memory_management"))
        if memory_management and pipeline.get("oom_recovery") == memory_management.get("oom_recovery"):
            pipeline.pop("oom_recovery", None)

        hires_fix = _mapping(pipeline.get("hires_fix"))
        if hires_fix:
            schedule = _mapping(output.get("schedule"))
            if hires_fix.get("schedule_replay") == schedule.get("hires_schedule_replay"):
                hires_fix.pop("schedule_replay", None)
                schedule.pop("hires_schedule_replay", None)
            if hires_fix.get("schedule_fingerprint") == schedule.get("hires_schedule_fingerprint"):
                hires_fix.pop("schedule_fingerprint", None)
                schedule.pop("hires_schedule_fingerprint", None)
            if memory_management:
                # These exact residency/cleanup records are already represented
                # by memory_management external telemetry and cleanup reports.
                hires_fix.pop("upscaler_residency_after_stage", None)
                hires_fix.pop("pre_hires_cleanup", None)
            # The compact replay contract carries the behavior-defining REGION
            # plan; pipeline metadata retains only runtime REGION telemetry.
            hires_fix.pop("regional_prompting", None)
            pipeline["hires_fix"] = hires_fix
            if schedule:
                output["schedule"] = schedule
        output["pipeline_metadata"] = pipeline

    # Registry and dispatch tables describe the installed implementation, not
    # what this particular generation executed. The selected backend, package
    # versions, model attention signature, kernel evidence, and reproducibility
    # identity remain in the diagnostic record.
    attention_backend = _mapping(output.get("attention_backend"))
    if attention_backend:
        attention_backend.pop("provider_registry", None)
        attention_backend.pop("production_dispatch", None)
        xformers = _mapping(attention_backend.get("xformers"))
        if xformers.get("compatibility_matrix") == xformers.get("smoke_test"):
            xformers.pop("smoke_test", None)
        if xformers:
            attention_backend["xformers"] = xformers
        output["attention_backend"] = attention_backend

    pipeline = _mapping(output.get("pipeline_metadata"))
    attention_by_pass = _mapping(pipeline.get("attention_execution_by_pass"))
    if (
        attention_by_pass.get("before_generation")
        == _mapping(output.get("attention_backend")).get("custom_provider_execution")
    ):
        attention_by_pass.pop("before_generation", None)
        if attention_by_pass:
            pipeline["attention_execution_by_pass"] = attention_by_pass
        else:
            pipeline.pop("attention_execution_by_pass", None)
        output["pipeline_metadata"] = pipeline

    sampler = _mapping(output.get("sampler"))
    if sampler:
        duplicate_sampler_fields = {
            "schedule_extra": "schedule",
            "cfg_step_series": "cfg_step_series",
            "cfg_effective_guidance_summary": "cfg_effective_guidance_summary",
            "cfg_effective_range": "cfg_effective_range",
        }
        for sampler_key, top_key in duplicate_sampler_fields.items():
            if top_key in output:
                sampler.pop(sampler_key, None)
        output["sampler"] = sampler

    return _drop_empty(output)


def manifest_to_diagnostics_dict(manifest: GenerationManifest) -> dict[str, Any]:
    """Return a pruned diagnostic manifest without replay/schema duplication."""

    payload = manifest_to_replay_dict(manifest)
    payload["serialization_profile"] = SERIALIZATION_PROFILE_DIAGNOSTICS
    runtime = _drop_empty(json_safe(manifest.runtime_info.to_dict()))
    if runtime:
        payload["runtime_info"] = runtime

    # Diagnostics retain asset resolution detail, but empty/default fields are
    # removed to avoid repeating dozens of blank values for every output.
    for key, value in (
        ("base_model", manifest.base_model.to_dict()),
        ("vae", manifest.vae.to_dict()),
        ("loras", [asset.to_dict() for asset in manifest.loras]),
        ("embeddings", [asset.to_dict() for asset in manifest.embeddings]),
        ("hypernetworks", [asset.to_dict() for asset in manifest.hypernetworks]),
        ("extras", [asset.to_dict() for asset in manifest.extras]),
    ):
        cleaned = _drop_empty(json_safe(value))
        if not _is_empty(cleaned):
            payload[key] = cleaned

    diagnostic_extra = _prune_diagnostic_extra(_mapping(manifest.extra))
    if diagnostic_extra:
        payload["extra"] = diagnostic_extra
    return _drop_empty(payload)


def save_manifest_json(
    manifest: GenerationManifest,
    json_path: str | Path,
    indent: int = 2,
) -> Path:
    """Write the compact replay manifest used by default for exact reruns."""

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_to_replay_dict(manifest)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)

    manifest.update_runtime_paths(json_path=str(path))
    return path


def save_manifest_diagnostics_json(
    manifest: GenerationManifest,
    json_path: str | Path,
    indent: int = 2,
) -> Path:
    """Write the optional pruned full-diagnostics sidecar."""

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_to_diagnostics_dict(manifest)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)
    return path


def load_manifest_json(json_path: str | Path) -> GenerationManifest:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return GenerationManifest.from_dict(payload)


def save_manifest_txt(
    manifest: GenerationManifest,
    txt_path: str | Path,
    include_optional: bool = True,
    include_runtime: bool = True,
    include_assets: bool = True,
) -> Path:
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    infotext = manifest_to_infotext(
        manifest=manifest,
        include_optional=include_optional,
        include_runtime=include_runtime,
        include_assets=include_assets,
    )

    with path.open("w", encoding="utf-8") as file:
        file.write(infotext)

    manifest.update_runtime_paths(txt_path=str(path))
    return path
