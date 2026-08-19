from __future__ import annotations

from typing import Any, Iterable

from .contracts import HiresProfile
from .schema import HiresProfileSchemaRegistry

HIRES_AUTO_PROFILE_SERIES = "ha5"
HIRES_AUTO_PROFILE_VERSION = "ha5-v1"
REQUIRED_HIRES_AUTO_FAMILIES: tuple[str, ...] = ("sd1.x", "sd2.x", "sdxl", "sd3.x")


def builtin_auto_profile_id(family: str) -> str:
    token = str(family or "generic").strip().lower() or "generic"
    return f"builtin.auto.{token}"


def builtin_auto_profile_name(family: str) -> str:
    label = str(family or "Generic").strip() or "Generic"
    return f"IMAGE_GEN Auto - {label}"


def builtin_auto_profile_values(family: str) -> dict[str, Any]:
    token = str(family or "").strip().lower()
    values = {
        "hires_enabled": True,
        "hires_prompt_parser_mode": "same_as_base",
        "hires_shortcut_profile_mode": "same_as_base",
        "hires_size_mode": "scale_from_base",
        "hires_scale": 2.0,
        "hires_width": 0,
        "hires_height": 0,
        "hires_steps": 20,
        "hires_denoising_strength": 0.45,
        "hires_step_policy": "a1111_fixed_steps_v1",
        "hires_sampler_name": "auto",
        "hires_scheduler_name": "auto",
        "hires_cfg_scale": None,
        "hires_cfg_rescale": None,
        "hires_strategy": "pixel_neural",
        "hires_upscaler_id": "auto",
        "hires_tile_size": 0,
        "hires_tile_overlap": 16,
        "hires_tile_batch_size": 1,
        "hires_exact_resize_filter": "bicubic",
        "hires_final_size_correction_filter": "auto",
        "hires_aspect_policy": "stretch",
        "hires_padding_mode": "reflect",
        "hires_blurred_edge_method": "box",
    }
    if token == "sd3.x":
        values.update(
            hires_steps=20,
            hires_denoising_strength=0.30,
            hires_step_policy="a1111_fixed_steps_v1",
            hires_sampler_name="flow_euler",
            hires_scheduler_name="flow_match_euler",
        )
    elif token == "sdxl":
        values.update(
            hires_steps=18,
            hires_denoising_strength=0.35,
        )
    elif token == "sd2.x":
        values.update(
            hires_steps=18,
            hires_denoising_strength=0.38,
        )
    elif token == "sd1.x":
        values.update(
            hires_steps=16,
            hires_denoising_strength=0.40,
        )
    return values


def builtin_auto_profile_description(family: str) -> str:
    token = str(family or "").strip().lower()
    if token == "sd3.x":
        return (
            "Built-in HA5 SD3.x Auto recommendation. Targets preservation-oriented hires refinement "
            "with explicit active-step semantics and quality-aware native-scale correction policy."
        )
    if token == "sdxl":
        return "Built-in HA5 SDXL Auto recommendation tuned as a quality-oriented family baseline."
    if token == "sd2.x":
        return "Built-in HA5 SD2.x Auto recommendation tuned as a quality-oriented family baseline."
    if token == "sd1.x":
        return "Built-in HA5 SD1.x Auto recommendation tuned as a quality-oriented family baseline."
    return "Built-in IMAGE_GEN Auto recommendation."


def build_builtin_auto_profiles(
    schema_registry: HiresProfileSchemaRegistry | None = None,
    *,
    families: Iterable[str] = REQUIRED_HIRES_AUTO_FAMILIES,
) -> list[HiresProfile]:
    schema = schema_registry or HiresProfileSchemaRegistry()
    default_values = schema.default_values()
    profiles: list[HiresProfile] = []
    for family in families:
        values = {**default_values, **builtin_auto_profile_values(family)}
        profiles.append(
            HiresProfile(
                profile_id=builtin_auto_profile_id(family),
                name=builtin_auto_profile_name(family),
                description=builtin_auto_profile_description(family),
                source="builtin",
                read_only=True,
                included_fields=tuple(sorted(values)),
                values=values,
                compatibility={
                    "model_families": [family],
                    "builtin_series": HIRES_AUTO_PROFILE_SERIES,
                    "builtin_version": HIRES_AUTO_PROFILE_VERSION,
                    "auto_policy": True,
                },
                baseline_profile_id="",
            )
        )
    return profiles
