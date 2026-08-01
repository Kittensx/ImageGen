from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import GenerationManifest


def _trim_number(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.8f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _stem_from_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem


def manifest_to_civitai_parameters(manifest: GenerationManifest) -> str:
    """Return an A1111/Civitai-style PNG infotext string.

    Civitai and many other tools inspect the PNG text chunk named
    ``parameters`` and expect the Automatic1111-style layout:

    * first line: positive prompt only
    * second line: ``Negative prompt: ...``
    * third line onward: comma-separated key/value settings

    We preserve the project's richer TXT/JSON sidecars separately, but embed a
    compatibility-focused summary here so uploads can auto-detect prompt,
    settings, and model provenance.
    """

    req = manifest.required_for_rerun
    opt = manifest.optional_for_rerun

    positive = str(req.prompt or "")
    negative = str(req.negative_prompt or "")

    provenance = dict(manifest.extra.get("model_provenance") or {})
    model_name = (
        _stem_from_path(provenance.get("file_name"))
        or _stem_from_path(provenance.get("loaded_path"))
        or _stem_from_path(provenance.get("resolved_path"))
        or _stem_from_path(req.model_path)
    )
    model_hash = str(provenance.get("sha256") or "").strip()

    vae_name = (
        _stem_from_path(getattr(manifest.vae, "resolved_filename", ""))
        or _stem_from_path(getattr(manifest.vae, "resolved_path", ""))
        or _stem_from_path(getattr(manifest.vae, "requested_filename", ""))
        or _stem_from_path(getattr(manifest.vae, "requested_path", ""))
        or _stem_from_path(manifest.extra.get("vae_path") or manifest.extra.get("vae_name") or "")
    )
    vae_hash = str(getattr(manifest.vae, "resolved_hash", "") or "").strip()

    fields: list[tuple[str, str]] = [
        ("Steps", _trim_number(req.steps)),
        ("Sampler", str(req.sampler_name or "")),
        ("Schedule type", str(req.scheduler_name or "")),
        ("CFG scale", _trim_number(req.cfg_scale)),
        ("Seed", _trim_number(req.seed)),
        ("Size", f"{int(req.width)}x{int(req.height)}"),
    ]

    if model_hash:
        fields.append(("Model hash", model_hash))
    if model_name:
        fields.append(("Model", model_name))
    if vae_hash:
        fields.append(("VAE hash", vae_hash))
    if vae_name:
        fields.append(("VAE", vae_name))
    if opt.clip_skip is not None:
        fields.append(("Clip skip", _trim_number(opt.clip_skip)))
    if opt.guidance_rescale is not None:
        fields.append(("Guidance rescale", _trim_number(opt.guidance_rescale)))
    if opt.tiling is not None:
        fields.append(("Tiling", _trim_number(opt.tiling)))
    if manifest.loras:
        lora_names = []
        for asset in manifest.loras:
            label = (
                getattr(asset, "resolved_display_name", "")
                or getattr(asset, "requested_display_name", "")
                or getattr(asset, "resolved_filename", "")
                or getattr(asset, "requested_filename", "")
                or getattr(asset, "resolved_path", "")
                or getattr(asset, "requested_path", "")
            )
            stem = _stem_from_path(label)
            if stem:
                lora_names.append(stem)
        if lora_names:
            fields.append(("Lora hashes", ", ".join(lora_names)))

    settings_line = ", ".join(f"{key}: {value}" for key, value in fields if str(value).strip())
    return "\n".join([
        positive,
        f"Negative prompt: {negative}",
        settings_line,
    ])


def build_png_text_chunks(manifest: GenerationManifest) -> dict[str, str]:
    payload = json_safe(manifest.to_dict())
    parameters = manifest_to_civitai_parameters(manifest)
    manifest_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "parameters": parameters,
        "Comment": parameters,
        "Software": "IMAGE_GEN",
        "image_gen_manifest": manifest_json,
        "image_gen_manifest_version": str(getattr(manifest, "manifest_version", "1.0")),
        "image_gen_manifest_type": str(getattr(manifest, "manifest_type", "txt2img")),
    }


def build_pnginfo(manifest: GenerationManifest, *, existing: PngInfo | None = None) -> PngInfo:
    info = existing if existing is not None else PngInfo()
    for key, value in build_png_text_chunks(manifest).items():
        info.add_text(str(key), str(value))
    return info
