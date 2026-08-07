from __future__ import annotations

from typing import Any


def prompt_with_default(label: str, default: Any) -> Any:
    raw = input(f"{label} [{default}]: ").strip()
    return default if raw == "" else raw


def prompt_yes_no(label: str, default: bool) -> bool:
    default_token = "Y/n" if default else "y/N"
    raw = input(f"{label} [{default_token}]: ").strip().casefold()
    if raw == "":
        return bool(default)
    if raw in {"y", "yes", "1", "true", "on"}:
        return True
    if raw in {"n", "no", "0", "false", "off"}:
        return False
    print("Invalid selection. Please answer y or n.")
    return prompt_yes_no(label, default)


def choose_from_registry(title: str, registry_map: dict[str, Any]) -> dict[str, Any]:
    entries = [(key, value) for key, value in registry_map.items() if isinstance(value, dict)]
    if not entries:
        raise RuntimeError(f"No entries available for {title}.")

    print(f"\n=== {title} ===")
    for idx, (_, entry) in enumerate(entries, start=1):
        label = entry.get("label") or entry.get("name") or str(idx)
        name = entry.get("name") or ""
        if name and name != label:
            print(f"{idx}. {label} ({name})")
        else:
            print(f"{idx}. {label}")

    while True:
        raw = input(f"Choose {title} [1-{len(entries)}]: ").strip()
        try:
            index = int(raw)
            if 1 <= index <= len(entries):
                return entries[index - 1][1]
        except ValueError:
            pass
        print("Invalid selection.")


def build_interactive_overrides() -> dict[str, Any]:
    print("\n=== Generation Settings ===")

    prompt = input("Positive prompt: ").strip()
    negative = input("Negative prompt []: ").strip()

    steps = int(prompt_with_default("Steps", 20))
    cfg_scale = float(prompt_with_default("CFG Scale", 7.0))
    seed = int(prompt_with_default("Seed", -1))
    width = int(prompt_with_default("Width", 640))
    height = int(prompt_with_default("Height", 960))
    batch_size = int(prompt_with_default("Batch size", 1))
    batch_count = int(prompt_with_default("Batch count", 1))
    unlimited_raw = str(prompt_with_default("Unlimited generation (y/n)", "n")).strip().lower()
    unlimited = unlimited_raw in {"y", "yes", "1", "true", "on"}
    print(
        "Filename fields: {index:05d}, {seed}, {datetime}, {model}, "
        "{vae}, {lora}, {sampler}, {scheduler}, {width}, {height}"
    )
    filename_pattern = prompt_with_default(
        "Filename pattern",
        "{index:05d}-{seed}",
    )
    model_path = prompt_with_default("Model path", "")

    return {
        "positive_prompt": prompt,
        "negative_prompt": negative,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "unlimited": unlimited,
        "output_prefix": filename_pattern,
        "model_path": model_path,
    }


def _supported_neural_upscalers(project_context: Any) -> tuple[Any, ...]:
    from image_gen.systems.upscaling.discovery import discover_upscalers

    discovery = discover_upscalers(project_context, mode="unidentified")
    return tuple(discovery.supported_neural)


def _choose_supported_neural_upscaler(project_context: Any) -> Any:
    supported = _supported_neural_upscalers(project_context)
    if not supported:
        from image_gen.systems.upscaling.discovery import configured_upscaler_roots

        roots = ", ".join(str(item) for item in configured_upscaler_roots(project_context)) or "<no configured roots>"
        raise RuntimeError(
            "No supported neural .pth upscalers were discovered. "
            f"Place a supported .pth file under one of these roots and retry: {roots}"
        )

    print("\n=== Supported neural .pth hires upscalers ===")
    for index, descriptor in enumerate(supported, start=1):
        relative = str(getattr(descriptor, "relative_path", "") or getattr(descriptor, "file_name", "") or "")
        architecture = str(getattr(descriptor, "architecture", "") or "unknown")
        native_scale = int(getattr(descriptor, "native_scale", 0) or 0)
        tile_label = "tiled" if bool(getattr(descriptor, "tile_supported", False)) else "untiled"
        path_label = f" [{relative}]" if relative else ""
        print(
            f"{index}. {descriptor.display_name} · {architecture} · x{native_scale} · {tile_label}{path_label}"
        )

    while True:
        raw = input(f"Choose neural upscaler [1-{len(supported)}] [1]: ").strip()
        if raw == "":
            return supported[0]
        try:
            selected = int(raw)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(supported):
            return supported[selected - 1]
        print("Invalid selection.")


def build_hires_interactive_overrides(project_context: Any | None = None) -> dict[str, Any]:
    """Prompt for the standard run.bat settings plus Phase 14N PTH hires controls.

    Blank hires prompt overrides explicitly inherit the selected base prompts.
    New interactive hires runs focus on supported neural `.pth` upscalers and
    source-fidelity artifacts instead of the retired Python-only latent methods.
    """

    if project_context is None:
        raise ValueError("project_context is required for interactive neural hires selection.")

    overrides = build_interactive_overrides()

    print("\n=== Hires / Second Pass Settings ===")
    print("Press Enter at either hires prompt to keep the matching base prompt.")
    hires_positive_raw = input(
        "Hires positive prompt [Enter = base positive prompt]: "
    ).strip()
    hires_negative_raw = input(
        "Hires negative prompt [Enter = base negative prompt]: "
    ).strip()

    hires_scale = float(prompt_with_default("Hires scale", 1.5))
    hires_denoising_strength = float(
        prompt_with_default("Hires denoising strength", 0.4)
    )
    hires_steps = int(prompt_with_default("Hires steps", 20))

    if hires_scale < 1.0:
        raise ValueError("Hires scale must be at least 1.0.")
    if not 0.0 <= hires_denoising_strength <= 1.0:
        raise ValueError("Hires denoising strength must be between 0.0 and 1.0.")
    if hires_steps < 1:
        raise ValueError("Hires steps must be at least 1.")

    descriptor = _choose_supported_neural_upscaler(project_context)
    print(f"Selected hires upscaler: {descriptor.display_name} ({descriptor.upscaler_id})")

    save_lowres = prompt_yes_no(
        "Save the exact low-resolution base artifact",
        True,
    )
    save_pre_denoise = prompt_yes_no(
        "Save the pixel-upscaled pre-denoise artifact",
        True,
    )
    save_vae_roundtrip = prompt_yes_no(
        "Save the deterministic VAE round-trip artifact",
        False,
    )

    overrides.update(
        {
            "hires_enabled": True,
            "hires_prompt_parser_mode": "same_as_base",
            "hires_shortcut_profile_mode": "same_as_base",
            "hires_positive_prompt": (
                hires_positive_raw
                if hires_positive_raw
                else str(overrides.get("positive_prompt") or "")
            ),
            "hires_negative_prompt": (
                hires_negative_raw
                if hires_negative_raw
                else str(overrides.get("negative_prompt") or "")
            ),
            "hires_size_mode": "scale_from_base",
            "hires_scale": hires_scale,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_strategy": "pixel_neural",
            "hires_upscaler": descriptor.upscaler_id,
            "hires_upscaler_id": descriptor.upscaler_id,
            "hires_save_lowres": save_lowres,
            "hires_save_upscaled_pre_denoise": save_pre_denoise,
            "hires_save_vae_roundtrip": save_vae_roundtrip,
        }
    )

    return overrides
