from modules.txt2img.generation_manifest import GenerationManifest

manifest = GenerationManifest(
    required_for_rerun=RequiredForRerun(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        model_path=model_path,
    )
)

manifest.optional_for_rerun.scheduler_kwargs = dict(request.scheduler_kwargs or {})
manifest.optional_for_rerun.sampler_kwargs = dict(request.sampler_kwargs or {})
manifest.optional_for_rerun.compatibility_mode = compatibility_mode

manifest.runtime_info.effective_steps = effective_steps
manifest.runtime_info.scheduler_step_override_applied = scheduler_step_override_applied
manifest.runtime_info.active_blend_methods = list(active_blend_methods or [])
manifest.runtime_info.active_blend_weights = list(active_blend_weights or [])
manifest.runtime_info.tail_features_used = dict(tail_features_used or {})
manifest.runtime_info.predicted_stop_step = predicted_stop_step
manifest.runtime_info.device = device_name
manifest.runtime_info.generation_time_sec = generation_time_sec


def manifest_to_infotext(manifest: GenerationManifest) -> str:
    req = manifest.required_for_rerun
    opt = manifest.optional_for_rerun
    run = manifest.runtime_info

    lines = []

    # --- Core prompt ---
    lines.append(f"Prompt: {req.prompt}")
    lines.append(f"Negative prompt: {req.negative_prompt}")

    # --- Core settings ---
    lines.append(
        f"Steps: {req.steps}, "
        f"CFG scale: {req.cfg_scale}, "
        f"Seed: {req.seed}, "
        f"Size: {req.width}x{req.height}"
    )

    lines.append(
        f"Sampler: {req.sampler_name}, "
        f"Scheduler: {req.scheduler_name}"
    )

    # --- Optional (only if present) ---
    if opt.compatibility_mode:
        lines.append(f"Compatibility mode: {opt.compatibility_mode}")

    if opt.scheduler_kwargs:
        lines.append(f"Scheduler kwargs: {opt.scheduler_kwargs}")

    if opt.sampler_kwargs:
        lines.append(f"Sampler kwargs: {opt.sampler_kwargs}")

    # --- Runtime (debug info) ---
    if run.effective_steps is not None:
        lines.append(f"Effective steps: {run.effective_steps}")

    if run.scheduler_step_override_applied is not None:
        lines.append(
            f"Scheduler override applied: {run.scheduler_step_override_applied}"
        )

    if run.active_blend_methods:
        lines.append(f"Blend methods: {run.active_blend_methods}")

    if run.tail_features_used:
        lines.append(f"Tail features: {run.tail_features_used}")

    return "\n".join(lines)
    
FIELD_ALIASES = {
  "prompt": "prompt",
  "negative prompt": "negative_prompt",
  "steps": "steps",
  "cfg scale": "cfg_scale",
  "seed": "seed",
  "sampler": "sampler_name",
  "scheduler": "scheduler_name",
  "size": "size",
  "model": "model_path",
  "model hash": "model_hash",
}

SPECIAL_FIELD_HANDLERS = {
  "size": parse_size_field,
}