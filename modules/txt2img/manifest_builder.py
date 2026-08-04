from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import (
    GenerationManifest,
    RequiredForRerun,
)


def build_generation_manifest(
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    scheduler_name: str,
    model_path: str,
    request=None,
    compatibility_mode: str | None = None,
    effective_steps: int | None = None,
    scheduler_step_override_applied: bool | None = None,
    active_blend_methods: list[str] | None = None,
    active_blend_weights: list[float] | None = None,
    tail_features_used: dict | None = None,
    predicted_stop_step: int | None = None,
    device_name: str | None = None,
    generation_time_sec: float | None = None,
    batch_size: int = 1,
    batch_count: int = 1,
) -> GenerationManifest:
    
    
    
    manifest = GenerationManifest(
        required_for_rerun=RequiredForRerun(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=int(batch_size),
            batch_count=int(batch_count),
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            model_path=model_path,
        )
    )

    scheduler_kwargs = getattr(request, "scheduler_kwargs", {}) if request is not None else {}
    sampler_kwargs = getattr(request, "sampler_kwargs", {}) if request is not None else {}

    manifest.optional_for_rerun.scheduler_kwargs = dict(
        json_safe(scheduler_kwargs or {})
    )
    manifest.optional_for_rerun.sampler_kwargs = dict(
        json_safe(sampler_kwargs or {})
    )
    manifest.optional_for_rerun.compatibility_mode = compatibility_mode
    if request is not None:
        manifest.optional_for_rerun.guidance_rescale = float(
            getattr(request, "cfg_rescale", 0.0) or 0.0
        )
        manifest.optional_for_rerun.extra["vae_path"] = getattr(
            request, "vae_path", None
        )
        manifest.optional_for_rerun.extra["prompt_parser_name"] = str(
            getattr(request, "prompt_parser_name", "legacy") or "legacy"
        )
        manifest.optional_for_rerun.extra["prompt_parser_kwargs"] = dict(
            json_safe(getattr(request, "prompt_parser_kwargs", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_cfg_pass_schedules"] = dict(
            json_safe(getattr(request, "prompt_cfg_pass_schedules", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_cfg_schedule"] = dict(
            json_safe(getattr(request, "prompt_cfg_schedule", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_cfg_contract_version"] = str(
            (getattr(request, "prompt_cfg_schedule", {}) or {}).get("contract_version") or ""
        )
        manifest.optional_for_rerun.extra["prompt_expansion_pass_records"] = dict(
            json_safe(getattr(request, "prompt_expansion_pass_records", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_expansion_record"] = dict(
            json_safe(getattr(request, "prompt_expansion_record", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_expansion_contract_version"] = str(
            (getattr(request, "prompt_expansion_record", {}) or {}).get("contract_version") or ""
        )
        manifest.optional_for_rerun.extra["prompt_semantic_pass_records"] = dict(
            json_safe(getattr(request, "prompt_semantic_pass_records", {}) or {})
        )
        manifest.optional_for_rerun.extra["region_pass_records"] = dict(
            json_safe(getattr(request, "region_pass_records", {}) or {})
        )
        request_diagnostics = dict(getattr(request, "diagnostics", {}) or {})
        regional_runtime = dict(json_safe(request_diagnostics.get("regional_runtime") or {}))
        regional_runtime_passes = dict(
            json_safe(request_diagnostics.get("regional_runtime_passes") or {})
        )
        if regional_runtime:
            manifest.optional_for_rerun.extra["regional_runtime"] = regional_runtime
        if regional_runtime_passes:
            manifest.optional_for_rerun.extra["regional_runtime_passes"] = regional_runtime_passes
        manifest.optional_for_rerun.extra["prompt_shortcut_profile_name"] = str(
            getattr(request, "prompt_shortcut_profile_name", "legacy_default") or "legacy_default"
        )
        manifest.optional_for_rerun.extra["prompt_shortcut_profile_snapshot"] = dict(
            json_safe(getattr(request, "prompt_shortcut_profile_snapshot", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_parser_preset_name"] = str(
            getattr(request, "prompt_parser_preset_name", "") or ""
        )
        for field_name, default in (
            ("base_prompt_parser_name", getattr(request, "prompt_parser_name", "legacy")),
            ("base_shortcut_profile_name", getattr(request, "prompt_shortcut_profile_name", "legacy_default")),
            ("hires_prompt_parser_mode", "same_as_base"),
            ("hires_prompt_parser_name", getattr(request, "prompt_parser_name", "legacy")),
            ("hires_shortcut_profile_mode", "same_as_base"),
            ("hires_shortcut_profile_name", getattr(request, "prompt_shortcut_profile_name", "legacy_default")),
            ("hires_positive_prompt", getattr(request, "positive_prompt", "")),
            ("hires_negative_prompt", getattr(request, "negative_prompt", "")),
            ("hires_size_mode", "same_as_base"),
            ("hires_scale", 2.0),
            ("hires_width", 0),
            ("hires_height", 0),
            ("hires_dimension_plan", {}),
            ("hires_enabled", False),
            ("hires_steps", 20),
            ("hires_denoising_strength", 0.45),
            ("hires_step_policy", "a1111_fixed_steps_v1"),
            ("hires_sampler_name", ""),
            ("hires_scheduler_name", ""),
            ("hires_cfg_scale", None),
            ("hires_cfg_rescale", None),
            ("hires_upscaler", "latent_bilinear"),
            ("hires_save_lowres", False),
            ("prompt_shadow_compare", False),
            ("prompt_route_plan", {}),
            ("hires_prompt_route_plan", {}),
        ):
            manifest.optional_for_rerun.extra[field_name] = json_safe(
                getattr(request, field_name, default)
            )
        manifest.optional_for_rerun.extra["hires_prompt_parser_kwargs"] = dict(
            json_safe(getattr(request, "hires_prompt_parser_kwargs", {}) or {})
        )
        manifest.optional_for_rerun.extra["hires_shortcut_profile_snapshot"] = dict(
            json_safe(getattr(request, "hires_shortcut_profile_snapshot", {}) or {})
        )
        manifest.optional_for_rerun.extra["prompt_preflight"] = dict(
            json_safe(getattr(request, "prompt_preflight", {}) or {})
        )
        manifest.optional_for_rerun.extra["parser_kwargs"] = dict(
            json_safe(getattr(request, "parser_kwargs", {}) or {})
        )
        for optional_name in ("clip_skip", "tiling"):
            if hasattr(request, optional_name):
                setattr(
                    manifest.optional_for_rerun,
                    optional_name,
                    json_safe(getattr(request, optional_name)),
                )

    manifest.runtime_info.effective_steps = effective_steps
    manifest.runtime_info.scheduler_step_override_applied = scheduler_step_override_applied
    manifest.runtime_info.active_blend_methods = list(active_blend_methods or [])
    manifest.runtime_info.active_blend_weights = list(active_blend_weights or [])
    manifest.runtime_info.tail_features_used = dict(tail_features_used or {})
    manifest.runtime_info.predicted_stop_step = predicted_stop_step
    manifest.runtime_info.device = device_name
    manifest.runtime_info.generation_time_sec = generation_time_sec

    resolved_seeds = (
        list(getattr(request, "resolved_seeds", []) or [])
        if request is not None
        else []
    )
    if not resolved_seeds:
        resolved_seeds = [int(seed)]
    manifest.extra["resolved_seeds"] = [int(value) for value in resolved_seeds]

    return manifest
