from __future__ import annotations
from typing import Any


def prompt_with_default(label: str, default: Any) -> Any:
    raw = input(f"{label} [{default}]: ").strip()
    return default if raw == "" else raw


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

def build_hires_interactive_overrides() -> dict[str, Any]:
    """Prompt for the standard run.bat settings plus latent hires controls.

    Blank hires prompt overrides explicitly inherit the selected base prompts.
    This launcher intentionally keeps the already validated latent-bilinear path;
    external ESRGAN model selection is a later extension after quality tuning.
    """

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
            "hires_upscaler": "latent_bilinear",
        }
    )

    print("Hires upscaler: latent_bilinear (validated current path)")
    return overrides

