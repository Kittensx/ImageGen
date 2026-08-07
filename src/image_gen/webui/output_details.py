from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import quote

from PIL import Image

from image_gen.runtime_options import (
    extract_runtime_execution_record,
    runtime_replay_request_values,
)
from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.contracts import PROMPT_ASSET_CONTRACT_VERSION
from image_gen.webui.prompt_assets import extract_inline_loras_from_prompts, merge_replay_loras
from image_gen.webui.schema_utils import normalize_config_schema
from modules.checkpoint_inspector import build_architecture_contract
from modules.project_context import ProjectContext


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_FORM_REPLAY_FIELDS = {
    "positive_prompt",
    "negative_prompt",
    "model_path",
    "vae_path",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "seed",
    "batch_size",
    "batch_count",
    "sampler_name",
    "scheduler_name",
    "sampler_kwargs",
    "scheduler_kwargs",
    "prompt_parser_name",
    "prompt_parser_kwargs",
    "prompt_cfg_pass_schedules",
    "prompt_expansion_pass_records",
    "prompt_semantic_pass_records",
    "batch_prompt_semantic_pass_records",
    "region_pass_records",
    "batch_region_pass_records",
    "regional_runtime",
    "regional_runtime_passes",
    "batch_regional_runtime_passes",
    "prompt_shortcut_profile_name",
    "prompt_shortcut_profile_snapshot",
    "prompt_parser_preset_name",
    "base_prompt_parser_name",
    "base_shortcut_profile_name",
    "hires_prompt_parser_mode",
    "hires_prompt_parser_name",
    "hires_prompt_parser_kwargs",
    "hires_shortcut_profile_mode",
    "hires_shortcut_profile_name",
    "hires_shortcut_profile_snapshot",
    "hires_positive_prompt",
    "hires_negative_prompt",
    "hires_size_mode",
    "hires_scale",
    "hires_width",
    "hires_height",
    "hires_dimension_plan_version",
    "hires_dimension_plan",
    "hires_axis_scale_width",
    "hires_axis_scale_height",
    "hires_uniform_scale",
    "hires_aspect_ratio_changed",
    "hires_enabled",
    "hires_steps",
    "hires_denoising_strength",
    "hires_step_policy",
    "hires_sampler_name",
    "hires_scheduler_name",
    "hires_cfg_scale",
    "hires_cfg_rescale",
    "hires_upscaler",
    "hires_save_lowres",
    "outpaint_prototype_enabled",
    "outpaint_source_image",
    "outpaint_anchor",
    "outpaint_source_x",
    "outpaint_source_y",
    "outpaint_feather_px",
    "outpaint_context_seed_mode",
    "outpaint_denoising_strength",
    "outpaint_latent_strategy",
    "outpaint_prompt_mode",
    "outpaint_overlay_positive_prompt",
    "outpaint_overlay_negative_prompt",
    "outpaint_diagnostic_artifacts",
    "outpaint_prototype_record",
    "outpaint_shape_expansion_enabled",
    "outpaint_shape_target_mode",
    "outpaint_shape_target_width",
    "outpaint_shape_target_height",
    "outpaint_shape_base_width",
    "outpaint_shape_base_height",
    "outpaint_shape_anchor",
    "outpaint_shape_context_seed_mode",
    "outpaint_shape_source_handoff",
    "outpaint_shape_prompt_mode",
    "outpaint_shape_overlay_positive_prompt",
    "outpaint_shape_overlay_negative_prompt",
    "outpaint_shape_denoising_strength",
    "outpaint_shape_save_base",
    "outpaint_shape_runtime_record",
    "prompt_preflight",
    "prompt_shadow_compare",
    "prompt_route_plan",
    "hires_prompt_route_plan",
}
_BACKEND_REPLAY_OPTIONAL_MAP = {
    "compatibility_mode": "compatibility_mode",
    "clip_skip": "clip_skip",
    "guidance_rescale": "cfg_rescale",
    "tiling": "tiling",
}
_BACKEND_REPLAY_EXTRA_FIELDS = {
    "prompt_parser_name",
    "prompt_parser_kwargs",
    "prompt_cfg_pass_schedules",
    "prompt_expansion_pass_records",
    "prompt_semantic_pass_records",
    "batch_prompt_semantic_pass_records",
    "region_pass_records",
    "batch_region_pass_records",
    "regional_runtime",
    "regional_runtime_passes",
    "batch_regional_runtime_passes",
    "prompt_shortcut_profile_name",
    "prompt_shortcut_profile_snapshot",
    "prompt_parser_preset_name",
    "base_prompt_parser_name",
    "base_shortcut_profile_name",
    "hires_prompt_parser_mode",
    "hires_prompt_parser_name",
    "hires_prompt_parser_kwargs",
    "hires_shortcut_profile_mode",
    "hires_shortcut_profile_name",
    "hires_shortcut_profile_snapshot",
    "hires_positive_prompt",
    "hires_negative_prompt",
    "hires_size_mode",
    "hires_scale",
    "hires_width",
    "hires_height",
    "hires_dimension_plan_version",
    "hires_dimension_plan",
    "hires_axis_scale_width",
    "hires_axis_scale_height",
    "hires_uniform_scale",
    "hires_aspect_ratio_changed",
    "hires_enabled",
    "hires_steps",
    "hires_denoising_strength",
    "hires_step_policy",
    "hires_sampler_name",
    "hires_scheduler_name",
    "hires_cfg_scale",
    "hires_cfg_rescale",
    "hires_strategy",
    "hires_upscaler",
    "hires_upscaler_id",
    "hires_expected_upscaler_sha256",
    "hires_expected_native_scale",
    "hires_final_size_correction_filter",
    "hires_aspect_policy",
    "hires_padding_mode",
    "hires_recorded_target_correction",
    "hires_correction_fingerprint_enabled",
    "hires_recorded_correction_fingerprint",
    "outpaint_prototype_enabled",
    "outpaint_source_image",
    "outpaint_anchor",
    "outpaint_source_x",
    "outpaint_source_y",
    "outpaint_feather_px",
    "outpaint_context_seed_mode",
    "outpaint_denoising_strength",
    "outpaint_latent_strategy",
    "outpaint_prompt_mode",
    "outpaint_overlay_positive_prompt",
    "outpaint_overlay_negative_prompt",
    "outpaint_diagnostic_artifacts",
    "outpaint_prototype_record",
    "outpaint_shape_expansion_enabled",
    "outpaint_shape_target_mode",
    "outpaint_shape_target_width",
    "outpaint_shape_target_height",
    "outpaint_shape_base_width",
    "outpaint_shape_base_height",
    "outpaint_shape_anchor",
    "outpaint_shape_context_seed_mode",
    "outpaint_shape_source_handoff",
    "outpaint_shape_prompt_mode",
    "outpaint_shape_overlay_positive_prompt",
    "outpaint_shape_overlay_negative_prompt",
    "outpaint_shape_denoising_strength",
    "outpaint_shape_save_base",
    "outpaint_shape_runtime_record",
    "hires_expected_vae_sha256",
    "hires_expected_vae_source_kind",
    "hires_recorded_schedule_replay",
    "hires_recorded_schedule_fingerprint",
    "hires_save_lowres",
    "prompt_preflight",
    "prompt_shadow_compare",
    "prompt_route_plan",
    "hires_prompt_route_plan",
    "parser_kwargs",
    "canonical_prompt_contract",
    "lora_paths",
    "prompt_asset_contract_version",
    "loras",
    "textual_inversions",
    "_webui_active_prompt_assets",
    "vae_name",
    "vae_hash",
}
_REQUIRED_FIELD_MAP = {
    "prompt": "positive_prompt",
    "positive_prompt": "positive_prompt",
    "negative_prompt": "negative_prompt",
    "seed": "seed",
    "width": "width",
    "height": "height",
    "steps": "steps",
    "cfg_scale": "cfg_scale",
    "sampler_name": "sampler_name",
    "scheduler_name": "scheduler_name",
    "model_path": "model_path",
}
_ADVANCED_LINKED_FIELDS = {
    "scheduler_kwargs.steps",
    "scheduler_kwargs.device",
}
_A1111_SETTING_PATTERN = re.compile(
    r"(?:^|,\s*)([A-Za-z][A-Za-z0-9 _/+().-]*):\s*",
)


@dataclass
class OutputMetadataDetails:
    output_id: str
    image: dict[str, Any]
    metadata_source: str
    manifest: dict[str, Any]
    replay: dict[str, Any]
    unsupported: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "metadata_source": self.metadata_source,
            "image": self.image,
            "manifest": self.manifest,
            "replay": self.replay,
            "unsupported": self.unsupported,
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }


def _empty_manifest() -> dict[str, Any]:
    return {
        "required_for_rerun": {},
        "optional_for_rerun": {},
        "runtime_info": {},
        "base_model": {},
        "vae": {},
        "loras": [],
        "embeddings": [],
        "hypernetworks": [],
        "extras": [],
        "extra": {},
    }


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _asset_label(asset: Mapping[str, Any] | None) -> str:
    source = _dict(asset)
    for key in (
        "resolved_display_name",
        "requested_display_name",
        "resolved_filename",
        "requested_filename",
        "resolved_identifier",
        "requested_identifier",
        "resolved_path",
        "requested_path",
    ):
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _asset_path(asset: Mapping[str, Any] | None) -> str:
    source = _dict(asset)
    for key in ("resolved_path", "requested_path"):
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _asset_hash(asset: Mapping[str, Any] | None) -> str:
    source = _dict(asset)
    for key in ("resolved_hash", "requested_hash"):
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _safe_relative_sidecar(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def resolve_output_path(context: ProjectContext, output_id: str) -> tuple[Path, str]:
    raw = str(output_id or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError("A valid output identifier is required.")
    if "\\" in raw:
        raise ValueError("Output identifiers must use forward slashes.")

    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError("Absolute output paths are not allowed.")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("Output path traversal is not allowed.")

    root = context.txt2img_output_root.resolve()
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("The requested output is outside the configured output root.") from exc

    if candidate.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise ValueError("The requested output is not a supported image file.")
    if not candidate.is_file():
        raise FileNotFoundError("The requested output image was not found.")
    return candidate, relative.as_posix()


def _coerce_scalar(value: str) -> Any:
    text = str(value).strip()
    lowered = text.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _parse_settings_blob(text: str) -> dict[str, str]:
    matches = list(_A1111_SETTING_PATTERN.finditer(text.strip()))
    output: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip().strip(",").strip()
        output[match.group(1).strip()] = value
    return output


def parse_a1111_parameters(text: str) -> dict[str, Any]:
    raw = str(text or "").replace("\r\n", "\n").strip()
    manifest = _empty_manifest()
    if not raw:
        return manifest

    lines = raw.split("\n")
    settings_index = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if "Steps:" in lines[index] and ("Seed:" in lines[index] or "Size:" in lines[index])
        ),
        None,
    )
    if settings_index is None:
        settings_index = len(lines)

    negative_index = next(
        (index for index, line in enumerate(lines[:settings_index]) if line.startswith("Negative prompt:")),
        None,
    )
    if negative_index is None:
        positive_lines = lines[:settings_index]
        negative_lines: list[str] = []
    else:
        positive_lines = lines[:negative_index]
        negative_lines = [lines[negative_index][len("Negative prompt:"):].lstrip()]
        negative_lines.extend(lines[negative_index + 1:settings_index])

    required = manifest["required_for_rerun"]
    if positive_lines:
        required["prompt"] = "\n".join(positive_lines)
    if negative_index is not None:
        required["negative_prompt"] = "\n".join(negative_lines)

    settings_text = ", ".join(line.strip() for line in lines[settings_index:] if line.strip())
    settings = _parse_settings_blob(settings_text)
    known_settings = {
        "Steps",
        "Sampler",
        "Schedule type",
        "Scheduler",
        "CFG scale",
        "Seed",
        "Size",
        "Model",
        "Model hash",
        "VAE",
        "VAE hash",
        "Clip skip",
        "Guidance rescale",
        "Tiling",
        "Batch size",
        "Batch count",
        "Lora hashes",
    }

    mapping = {
        "Steps": "steps",
        "Sampler": "sampler_name",
        "Schedule type": "scheduler_name",
        "Scheduler": "scheduler_name",
        "CFG scale": "cfg_scale",
        "Seed": "seed",
    }
    for source, target in mapping.items():
        if source in settings and settings[source] != "":
            required[target] = _coerce_scalar(settings[source])

    size = settings.get("Size", "")
    size_match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", size)
    if size_match:
        required["width"] = int(size_match.group(1))
        required["height"] = int(size_match.group(2))

    model = settings.get("Model", "")
    model_hash = settings.get("Model hash", "")
    if model:
        manifest["base_model"]["requested_display_name"] = model
    if model_hash:
        manifest["base_model"]["requested_hash"] = model_hash

    vae = settings.get("VAE", "")
    vae_hash = settings.get("VAE hash", "")
    if vae:
        manifest["vae"]["requested_display_name"] = vae
    if vae_hash:
        manifest["vae"]["requested_hash"] = vae_hash

    optional = manifest["optional_for_rerun"]
    if "Clip skip" in settings:
        optional["clip_skip"] = _coerce_scalar(settings["Clip skip"])
    if "Guidance rescale" in settings:
        optional["guidance_rescale"] = _coerce_scalar(settings["Guidance rescale"])
    if "Tiling" in settings:
        optional["tiling"] = _coerce_scalar(settings["Tiling"])

    extra = manifest["extra"]
    if "Batch size" in settings:
        extra["batch_size"] = _coerce_scalar(settings["Batch size"])
    if "Batch count" in settings:
        extra["batch_count"] = _coerce_scalar(settings["Batch count"])
    if settings.get("Lora hashes"):
        manifest["loras"] = [
            {"requested_display_name": item.strip(), "asset_type": "lora"}
            for item in settings["Lora hashes"].split(",")
            if item.strip()
        ]

    unknown = {key: value for key, value in settings.items() if key not in known_settings}
    if unknown:
        optional["extra"] = {"a1111_parameters": unknown}
    return manifest


def parse_txt_infotext(text: str) -> dict[str, Any]:
    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return _empty_manifest()
    if not raw.startswith("Prompt:"):
        return parse_a1111_parameters(raw)

    manifest = _empty_manifest()
    required = manifest["required_for_rerun"]
    optional = manifest["optional_for_rerun"]
    runtime = manifest["runtime_info"]
    extra = manifest["extra"]
    provenance: dict[str, Any] = {}

    lines = raw.splitlines()
    settings_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Steps:")),
        len(lines),
    )
    negative_index = next(
        (index for index, line in enumerate(lines[:settings_index]) if line.startswith("Negative prompt:")),
        None,
    )
    if negative_index is None:
        prompt_lines = [lines[0].split(":", 1)[1].lstrip(), *lines[1:settings_index]]
        required["prompt"] = "\n".join(prompt_lines)
    else:
        prompt_lines = [lines[0].split(":", 1)[1].lstrip(), *lines[1:negative_index]]
        negative_lines = [
            lines[negative_index].split(":", 1)[1].lstrip(),
            *lines[negative_index + 1:settings_index],
        ]
        required["prompt"] = "\n".join(prompt_lines)
        required["negative_prompt"] = "\n".join(negative_lines)

    for line in lines[settings_index:]:
        if line.startswith("Steps:"):
            settings = _parse_settings_blob(line)
            for source, target in (
                ("Steps", "steps"),
                ("CFG scale", "cfg_scale"),
                ("Seed", "seed"),
            ):
                if source in settings:
                    required[target] = _coerce_scalar(settings[source])
            size = settings.get("Size", "")
            match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", size)
            if match:
                required["width"] = int(match.group(1))
                required["height"] = int(match.group(2))
        elif line.startswith("Sampler:"):
            settings = _parse_settings_blob(line)
            if "Sampler" in settings:
                required["sampler_name"] = settings["Sampler"]
            if "Scheduler" in settings:
                required["scheduler_name"] = settings["Scheduler"]
            if "Model" in settings:
                required["model_path"] = settings["Model"]
        elif line.startswith("Requested model path:"):
            provenance["requested_path"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Resolved model path:"):
            provenance["resolved_path"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Loaded model path:"):
            provenance["loaded_path"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Loaded model SHA-256:"):
            provenance["sha256"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Scheduler kwargs:"):
            try:
                value = json.loads(line.split(":", 1)[1].lstrip())
                if isinstance(value, dict):
                    optional["scheduler_kwargs"] = value
            except json.JSONDecodeError:
                optional.setdefault("extra", {})["scheduler_kwargs_text"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Sampler kwargs:"):
            try:
                value = json.loads(line.split(":", 1)[1].lstrip())
                if isinstance(value, dict):
                    optional["sampler_kwargs"] = value
            except json.JSONDecodeError:
                optional.setdefault("extra", {})["sampler_kwargs_text"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Guidance metadata:"):
            try:
                value = json.loads(line.split(":", 1)[1].lstrip())
                if isinstance(value, dict):
                    extra.update(value)
            except json.JSONDecodeError:
                extra["guidance_metadata_text"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("CFG step series:"):
            try:
                value = json.loads(line.split(":", 1)[1].lstrip())
                if isinstance(value, dict):
                    extra["cfg_step_series"] = value
            except json.JSONDecodeError:
                extra["cfg_step_series_text"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Optional extras:"):
            try:
                value = json.loads(line.split(":", 1)[1].lstrip())
                if isinstance(value, dict):
                    optional["extra"] = {**_dict(optional.get("extra")), **value}
            except json.JSONDecodeError:
                optional.setdefault("extra", {})["optional_extras_text"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Timestamp:"):
            runtime["timestamp"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Device:"):
            runtime["device"] = line.split(":", 1)[1].lstrip()
        elif line.startswith("Generation time sec:"):
            runtime["generation_time_sec"] = _coerce_scalar(line.split(":", 1)[1].lstrip())

    if provenance:
        extra["model_provenance"] = provenance
    return manifest


def _unknown_field(
    unsupported: dict[str, Any],
    path: str,
    value: Any,
    reason: str,
    *,
    status: str = "unsupported_by_current_ui",
) -> None:
    unsupported[path] = {
        "value": value,
        "status": status,
        "reason": reason,
    }


def manifest_to_replay_payload(
    manifest: Mapping[str, Any] | None,
    *,
    context: ProjectContext | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _dict(manifest)
    required = _dict(source.get("required_for_rerun"))
    optional = _dict(source.get("optional_for_rerun"))
    optional_extra = _dict(optional.get("extra"))
    extra = _dict(source.get("extra"))
    replay: dict[str, Any] = {}
    unsupported: dict[str, Any] = {}
    # Legacy manifests predate explicit hires policy metadata. Preserve their
    # historical proportional-tail behavior instead of applying the new form
    # default during replay normalization.
    if "hires_step_policy" not in optional_extra and "hires_step_policy" not in extra:
        replay["hires_step_policy"] = "proportional_tail_v1"
    runtime_execution = extract_runtime_execution_record(source)
    replay.update(runtime_replay_request_values(runtime_execution))

    if not required:
        required = {
            key: value
            for key, value in source.items()
            if key in _REQUIRED_FIELD_MAP
        }

    for source_name, target_name in _REQUIRED_FIELD_MAP.items():
        if source_name not in required or required[source_name] is None:
            continue
        value = required[source_name]
        if target_name not in {"positive_prompt", "negative_prompt"} and value == "":
            continue
        replay[target_name] = value

    canonical_contract = _dict(optional_extra.get("canonical_prompt_contract") or extra.get("prompt_contract"))
    positive_structure = _dict(canonical_contract.get("canonical_positive_structure"))
    negative_structure = _dict(canonical_contract.get("canonical_negative_structure"))
    if positive_structure.get("lossless_source") is not None:
        replay["positive_prompt"] = str(positive_structure.get("lossless_source") or "")
    if negative_structure.get("lossless_source") is not None:
        replay["negative_prompt"] = str(negative_structure.get("lossless_source") or "")

    provenance = _dict(extra.get("model_provenance"))
    loaded_model_path = (
        provenance.get("loaded_path")
        or provenance.get("resolved_path")
        or provenance.get("requested_path")
    )
    if loaded_model_path:
        replay["model_path"] = str(loaded_model_path)

    vae = _dict(source.get("vae"))
    vae_path = _asset_path(vae) or extra.get("vae_path") or extra.get("vae_name")
    if vae_path:
        replay["vae_path"] = str(vae_path)

    for key in ("batch_size", "batch_count"):
        if key in required:
            replay[key] = required[key]
        elif key in optional_extra:
            replay[key] = optional_extra[key]
        elif key in extra:
            replay[key] = extra[key]

    for key in ("sampler_kwargs", "scheduler_kwargs"):
        if key in optional:
            value = optional.get(key)
            if isinstance(value, Mapping):
                replay[key] = dict(value)
            elif value not in (None, ""):
                _unknown_field(
                    unsupported,
                    key,
                    value,
                    "Advanced settings were recorded in a non-object format.",
                    status="invalid_metadata",
                )

    for key, value in required.items():
        if key not in _REQUIRED_FIELD_MAP and key not in {"batch_size", "batch_count"}:
            _unknown_field(
                unsupported,
                f"required_for_rerun.{key}",
                value,
                "The current generation form has no mapping for this required field.",
            )

    for key, value in optional.items():
        if key in {"sampler_kwargs", "scheduler_kwargs", "extra"}:
            continue
        if value is None:
            continue
        mapped = _BACKEND_REPLAY_OPTIONAL_MAP.get(key)
        if mapped:
            replay[mapped] = value
            continue
        _unknown_field(
            unsupported,
            f"optional_for_rerun.{key}",
            value,
            "The canonical replay request does not have a safe mapping for this optional field.",
        )

    for container_name, values in (("optional_for_rerun.extra", optional_extra), ("extra", extra)):
        for key, value in values.items():
            if key in {
                "batch_size",
                "batch_count",
                "model_provenance",
                "resolved_seeds",
                "prompt_parser",
                "prompt_shortcut_profile",
                "prompt_translation",
                "prompt_contract",
                "runtime_execution",
                "runtime_execution_schema_version",
                "runtime_startup_options",
                "memory_management",
                "pipeline_metadata",
                "prompt_assets",
            }:
                continue
            if key in _FORM_REPLAY_FIELDS and key not in replay:
                replay[key] = value
                continue
            if key == "vae_path":
                if value not in (None, "") and "vae_path" not in replay:
                    replay["vae_path"] = value
                continue
            mapped_optional = _BACKEND_REPLAY_OPTIONAL_MAP.get(key)
            if mapped_optional:
                replay[mapped_optional] = value
                continue
            if key in _BACKEND_REPLAY_EXTRA_FIELDS:
                replay[key] = value
                continue
            _unknown_field(
                unsupported,
                f"{container_name}.{key}",
                value,
                "The metadata value is preserved for inspection but has no current form control.",
            )

    inline_loras = extract_inline_loras_from_prompts(
        replay.get("positive_prompt"),
        replay.get("hires_positive_prompt"),
    )
    merged_loras = merge_replay_loras(replay.get("loras") or [], inline_loras)
    if merged_loras:
        replay["prompt_asset_contract_version"] = str(
            replay.get("prompt_asset_contract_version") or PROMPT_ASSET_CONTRACT_VERSION
        )
        replay["loras"] = merged_loras
        replay["lora_paths"] = [
            item.get("resolved_path") or item.get("path") or item.get("requested_path") or ""
            for item in merged_loras
            if item.get("resolved_path") or item.get("path") or item.get("requested_path")
        ]

    if context is not None:
        _classify_plugin_support(context, replay, unsupported)
    return replay, unsupported


def _classify_plugin_support(
    context: ProjectContext,
    replay: dict[str, Any],
    unsupported: dict[str, Any],
) -> None:
    from modules.prompt_parsers import default_prompt_parser_registry

    parser_registry = default_prompt_parser_registry()
    requested_parser = replay.get("prompt_parser_name") or "legacy"
    if not parser_registry.has(requested_parser, require_available=True):
        _unknown_field(
            unsupported,
            "prompt_parser_name",
            requested_parser,
            "The recorded prompt parser is not available in the current prompt parser registry.",
            status="unavailable_plugin",
        )

    from modules.prompt_shortcuts import PromptShortcutProfileDescriptor, default_prompt_shortcut_registry, validate_prompt_shortcut_profile

    profile_name = replay.get("prompt_shortcut_profile_name") or ("legacy_default" if requested_parser == "legacy" else ("superhybrid_native" if requested_parser == "superhybrid" else "parser21_native"))
    snapshot = replay.get("prompt_shortcut_profile_snapshot")
    if isinstance(snapshot, Mapping) and snapshot:
        profile = PromptShortcutProfileDescriptor.from_dict(dict(snapshot), builtin=bool(snapshot.get("builtin", False)))
        validation = validate_prompt_shortcut_profile(profile)
        if not validation.valid:
            _unknown_field(
                unsupported,
                "prompt_shortcut_profile_snapshot",
                snapshot,
                "The embedded prompt shortcut profile snapshot is invalid.",
                status="invalid_metadata",
            )
    elif not default_prompt_shortcut_registry().has(profile_name):
        _unknown_field(
            unsupported,
            "prompt_shortcut_profile_name",
            profile_name,
            "The recorded prompt shortcut profile is not installed and no effective mapping snapshot was embedded.",
            status="unavailable_plugin",
        )

    registry = RuntimeRegistrySystem(project_context=context)
    for kind, name_key, kwargs_key in (
        ("sampler", "sampler_name", "sampler_kwargs"),
        ("scheduler", "scheduler_name", "scheduler_kwargs"),
    ):
        requested = replay.get(name_key)
        descriptor = registry.resolve_descriptor(requested, kind=kind) if requested else None
        if requested and descriptor is None:
            _unknown_field(
                unsupported,
                name_key,
                requested,
                f"The recorded {kind} is not available in the current plugin registry.",
                status="unavailable_plugin",
            )

        kwargs = replay.get(kwargs_key)
        if not isinstance(kwargs, Mapping) or not kwargs:
            continue
        if descriptor is None:
            for key, value in kwargs.items():
                _unknown_field(
                    unsupported,
                    f"{kwargs_key}.{key}",
                    value,
                    f"The {kind} plugin is unavailable, so this setting cannot be edited.",
                    status="unavailable_plugin",
                )
            continue

        schema = normalize_config_schema(descriptor.config_schema, kind=kind)
        properties = _dict(schema.get("properties"))
        allow_additional = bool(schema.get("additionalProperties", False))
        for key, value in kwargs.items():
            path = f"{kwargs_key}.{key}"
            if path in _ADVANCED_LINKED_FIELDS:
                _unknown_field(
                    unsupported,
                    path,
                    value,
                    "This value is linked to another form or runtime control and is not independently editable.",
                )
            elif key not in properties and not allow_additional:
                _unknown_field(
                    unsupported,
                    path,
                    value,
                    f"The current {kind} editor does not expose this recorded setting.",
                )


def _metadata_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = _dict(manifest)
    required = _dict(source.get("required_for_rerun"))
    runtime = _dict(source.get("runtime_info"))
    extra = _dict(source.get("extra"))
    provenance = _dict(extra.get("model_provenance"))
    vae_provenance = _dict(extra.get("vae_provenance"))
    base_model = _dict(source.get("base_model"))
    vae = _dict(source.get("vae"))

    model_path = (
        provenance.get("loaded_path")
        or provenance.get("resolved_path")
        or required.get("model_path")
        or _asset_path(base_model)
        or ""
    )
    model_name = (
        provenance.get("file_name")
        or _asset_label(base_model)
        or (Path(str(model_path)).name if model_path else "")
    )
    model_hash = provenance.get("sha256") or _asset_hash(base_model)
    vae_path = _asset_path(vae) or extra.get("vae_path") or ""
    vae_name = _asset_label(vae) or (Path(str(vae_path)).name if vae_path else "")

    architecture_contract = build_architecture_contract(
        provenance.get("architecture"),
        provenance.get("prediction_type"),
        provenance.get("conditioning_dimension"),
        summary=provenance.get("architecture_summary"),
        source=provenance.get("architecture_source"),
    ).to_dict()

    return {
        "timestamp": runtime.get("timestamp"),
        "model": {
            "display_name": str(model_name or ""),
            "path": str(model_path or ""),
            "hash": str(model_hash or ""),
            "architecture": provenance.get("architecture"),
            "prediction_type": provenance.get("prediction_type"),
            "conditioning_dimension": provenance.get("conditioning_dimension"),
            "architecture_summary": provenance.get("architecture_summary") or architecture_contract.get("summary") or "",
            "architecture_contract": architecture_contract,
        },
        "vae": {
            "display_name": str(vae_name or ""),
            "path": str(vae_path or ""),
            "hash": _asset_hash(vae),
            "mode": str(vae_provenance.get("mode") or ("manual_external_selection" if vae_path else "checkpoint_embedded_auto")),
            "effective_source": str(vae_provenance.get("effective_source") or "checkpoint_embedded"),
            "requested_path": str(vae_provenance.get("requested_path") or vae_path or ""),
            "component_device": str(vae_provenance.get("component_device") or ""),
            "component_dtype": str(vae_provenance.get("component_dtype") or ""),
        },
        "loras": _list(source.get("loras")),
        "embeddings": _list(source.get("embeddings")),
        "hypernetworks": _list(source.get("hypernetworks")),
        "other_assets": _list(source.get("extras")),
        "prompt_parser": _dict(extra.get("prompt_parser")),
        "prompt_contract": _dict(extra.get("prompt_contract")),
    }


def _load_details_for_image_path(context: ProjectContext, image_path: Path, normalized_id: str) -> OutputMetadataDetails:
    root = context.txt2img_output_root.resolve()
    json_path = image_path.with_suffix(".json")
    diagnostics_json_path = image_path.with_name(f"{image_path.stem}.diagnostics.json")
    txt_path = image_path.with_suffix(".txt")
    warnings: list[str] = []
    manifest: dict[str, Any] | None = None
    replay_manifest: dict[str, Any] | None = None
    metadata_source = "partial_summary"

    image_info: dict[str, Any] = {}
    with Image.open(image_path) as image:
        width, height = image.size
        image_info = dict(image.info or {})

    if json_path.is_file():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                replay_manifest = loaded
                metadata_source = "json_sidecar"
            else:
                warnings.append("The JSON sidecar did not contain an object and was skipped.")
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"The JSON sidecar could not be read: {exc}")

    if diagnostics_json_path.is_file():
        try:
            loaded = json.loads(diagnostics_json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                metadata_source = (
                    "json_sidecar+diagnostics"
                    if replay_manifest is not None
                    else "diagnostics_json_sidecar"
                )
            else:
                warnings.append("The diagnostics JSON sidecar did not contain an object and was skipped.")
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"The diagnostics JSON sidecar could not be read: {exc}")

    if manifest is None and image_info.get("image_gen_manifest"):
        try:
            loaded = json.loads(str(image_info["image_gen_manifest"]))
            if isinstance(loaded, dict):
                manifest = loaded
                metadata_source = "png_manifest"
            else:
                warnings.append("The embedded IMAGE_GEN manifest was not an object and was skipped.")
        except json.JSONDecodeError as exc:
            warnings.append(f"The embedded IMAGE_GEN manifest was invalid JSON: {exc}")

    if manifest is None and image_info.get("parameters"):
        manifest = parse_a1111_parameters(str(image_info["parameters"]))
        metadata_source = "png_parameters"

    if manifest is None and txt_path.is_file():
        try:
            manifest = parse_txt_infotext(txt_path.read_text(encoding="utf-8"))
            metadata_source = "txt_sidecar"
        except OSError as exc:
            warnings.append(f"The TXT sidecar could not be read: {exc}")

    if manifest is None:
        manifest = _empty_manifest()
        warnings.append("No JSON, embedded PNG, or TXT generation metadata was found.")

    replay, unsupported = manifest_to_replay_payload(
        replay_manifest or manifest,
        context=context,
    )
    summary = _metadata_summary(manifest)
    timestamp = summary.get("timestamp")
    if not timestamp:
        timestamp = datetime.fromtimestamp(image_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")

    missing_fields = [
        field_name
        for field_name in (
            "positive_prompt",
            "negative_prompt",
            "seed",
            "width",
            "height",
            "steps",
            "cfg_scale",
            "sampler_name",
            "scheduler_name",
            "model_path",
        )
        if field_name not in replay
    ]
    if missing_fields:
        warnings.append("Missing replay metadata: " + ", ".join(missing_fields) + ".")

    provenance = {
        "json_sidecar": {
            "available": json_path.is_file(),
            "path": _safe_relative_sidecar(root, json_path) if json_path.is_file() else "",
        },
        "diagnostics_json_sidecar": {
            "available": diagnostics_json_path.is_file(),
            "path": (
                _safe_relative_sidecar(root, diagnostics_json_path)
                if diagnostics_json_path.is_file()
                else ""
            ),
        },
        "txt_sidecar": {
            "available": txt_path.is_file(),
            "path": _safe_relative_sidecar(root, txt_path) if txt_path.is_file() else "",
        },
        "png_manifest_available": bool(image_info.get("image_gen_manifest")),
        "png_parameters_available": bool(image_info.get("parameters")),
        "metadata_source": metadata_source,
        "runtime_diagnostics": _dict(_dict(manifest.get("runtime_info")).get("extra")).get("diagnostics"),
    }

    return OutputMetadataDetails(
        output_id=normalized_id,
        image={
            "name": image_path.name,
            "relative_path": normalized_id,
            "url": f"/outputs/{quote(normalized_id, safe='/')}",
            "width": width,
            "height": height,
            "timestamp": timestamp,
            "format": image_path.suffix.lower().lstrip("."),
            "model": summary["model"],
            "vae": summary["vae"],
            "loras": summary["loras"],
            "embeddings": summary["embeddings"],
            "hypernetworks": summary["hypernetworks"],
            "other_assets": summary["other_assets"],
            "prompt_parser": summary["prompt_parser"],
            "prompt_contract": summary["prompt_contract"],
        },
        metadata_source=metadata_source,
        manifest=manifest,
        replay=replay,
        unsupported=unsupported,
        warnings=warnings,
        provenance=provenance,
    )


def load_output_details(context: ProjectContext, output_id: str) -> OutputMetadataDetails:
    image_path, normalized_id = resolve_output_path(context, output_id)
    return _load_details_for_image_path(context, image_path, normalized_id)


def load_image_file_details(context: ProjectContext, image_path: Path, *, display_name: str | None = None) -> OutputMetadataDetails:
    resolved = Path(image_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    normalized_id = str(display_name or resolved.name)
    return _load_details_for_image_path(context, resolved, normalized_id)
