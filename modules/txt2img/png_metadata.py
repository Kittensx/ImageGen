from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from PIL.PngImagePlugin import PngInfo

from image_gen.program_metadata import (
    APPLICATION_VERSION,
    METADATA_SCHEMA_VERSION,
    PRODUCT_NAME,
    build_program_metadata,
)
from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import GenerationManifest
from modules.txt2img.manifest_io import manifest_to_replay_dict


_LORA_INFOTEXT_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sanitize_lora_infotext_name(value: Any, *, fallback_hash: str = "") -> str:
    """Return a Civitai/A1111 prompt-tag-safe LoRA name."""

    stem = _stem_from_path(value)
    cleaned = _LORA_INFOTEXT_NAME_RE.sub("_", stem).strip("_.-")
    if cleaned:
        return cleaned
    suffix = re.sub(r"[^0-9a-fA-F]", "", str(fallback_hash or ""))[:12]
    return f"lora_{suffix or 'asset'}"


def _lora_compatibility_hash(asset: Any) -> str:
    extra = _mapping(getattr(asset, "extra", {}))
    metadata = _mapping(extra.get("metadata"))
    scan_cache = _mapping(metadata.get("_lora_scan_cache"))
    candidates = (
        extra.get("a1111_short_hash"),
        metadata.get("a1111_short_hash"),
        scan_cache.get("a1111_short_hash"),
        extra.get("a1111_hash"),
        metadata.get("a1111_hash"),
        scan_cache.get("a1111_hash"),
    )
    for value in candidates:
        token = str(value or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{12,128}", token):
            return token[:12]

    # For non-Safetensors adapters the A1111 compatibility hash is the normal
    # file SHA-256. Do not use this fallback for Safetensors because its
    # compatible identity excludes the mutable header.
    filename = str(
        getattr(asset, "resolved_filename", "")
        or getattr(asset, "requested_filename", "")
        or getattr(asset, "resolved_path", "")
        or getattr(asset, "requested_path", "")
        or ""
    ).lower()
    if not filename.endswith(".safetensors"):
        fallback = str(
            getattr(asset, "resolved_hash", "")
            or getattr(asset, "requested_hash", "")
            or ""
        ).strip().lower()
        if re.fullmatch(r"[0-9a-f]{12,128}", fallback):
            return fallback[:12]
    return ""


def _active_lora_metadata(manifest: GenerationManifest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for asset in manifest.loras:
        extra = _mapping(getattr(asset, "extra", {}))
        enabled = bool(extra.get("enabled", True))
        used = bool(getattr(asset, "was_used_for_generation", False))
        applied = str(getattr(asset, "action_taken", "") or "").strip().lower() == "applied"
        if not enabled or not (used or applied):
            continue

        short_hash = _lora_compatibility_hash(asset)
        label = (
            getattr(asset, "resolved_display_name", "")
            or getattr(asset, "requested_display_name", "")
            or getattr(asset, "resolved_filename", "")
            or getattr(asset, "requested_filename", "")
            or getattr(asset, "resolved_path", "")
            or getattr(asset, "requested_path", "")
        )
        base_name = _sanitize_lora_infotext_name(label, fallback_hash=short_hash)
        name = base_name
        if name.casefold() in used_names:
            suffix = short_hash[:8] if short_hash else str(len(records) + 1)
            name = f"{base_name}_{suffix}"
        used_names.add(name.casefold())
        try:
            weight = float(extra.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        records.append(
            {
                "name": name,
                "hash": short_hash,
                "weight": weight,
            }
        )
    return records


def _append_lora_infotext_tags(prompt: str, records: list[dict[str, Any]]) -> str:
    tags = [
        f"<lora:{item['name']}:{_trim_number(item['weight'])}>"
        for item in records
        if item.get("name")
    ]
    if not tags:
        return prompt
    base = str(prompt or "").rstrip()
    suffix = " ".join(tags)
    return f"{base} {suffix}".strip()


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

    lora_records = _active_lora_metadata(manifest)
    positive = _append_lora_infotext_tags(str(req.prompt or ""), lora_records)
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
    lora_hash_entries = [
        f"{item['name']}: {item['hash']}"
        for item in lora_records
        if item.get("name") and item.get("hash")
    ]
    if lora_hash_entries:
        # The nested name/hash mapping contains commas and colons, so it must
        # be emitted as one JSON-quoted infotext value. Civitai/A1111 then
        # parse that quoted value as the ``Lora hashes`` mapping.
        fields.append(
            (
                "Lora hashes",
                json.dumps(", ".join(lora_hash_entries), ensure_ascii=False),
            )
        )

    settings_line = ", ".join(f"{key}: {value}" for key, value in fields if str(value).strip())
    return "\n".join([
        positive,
        f"Negative prompt: {negative}",
        settings_line,
    ])


def build_png_text_chunks(manifest: GenerationManifest) -> dict[str, str]:
    # Embed the same compact replay payload used by the default JSON sidecar.
    # Full runtime diagnostics remain available in the optional
    # ``*.diagnostics.json`` sidecar instead of inflating every PNG.
    application = dict(manifest.extra.get("application") or build_program_metadata())
    build = dict(application.get("build") or {})

    payload = json_safe(manifest_to_replay_dict(manifest))
    payload_extra = dict(payload.get("extra") or {})
    payload_extra["application"] = json_safe(application)
    payload["extra"] = payload_extra

    parameters = manifest_to_civitai_parameters(manifest)
    manifest_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "parameters": parameters,
        "Comment": parameters,
        "Software": PRODUCT_NAME,
        "ImageGen Version": str(application.get("version") or APPLICATION_VERSION),
        "ImageGen Build": str(build.get("display") or build.get("commit_short") or ""),
        "ImageGen Commit": str(build.get("commit_full") or ""),
        "ImageGen Metadata Version": str(
            application.get("metadata_schema_version") or METADATA_SCHEMA_VERSION
        ),
        "imagegen_version": str(application.get("version") or APPLICATION_VERSION),
        "imagegen_build": str(build.get("commit_short") or ""),
        "imagegen_commit": str(build.get("commit_full") or ""),
        "imagegen_build_exact": "true" if bool(build.get("exact_source_snapshot")) else "false",
        "imagegen_metadata_version": str(
            application.get("metadata_schema_version") or METADATA_SCHEMA_VERSION
        ),
        "image_gen_manifest": manifest_json,
        "image_gen_manifest_version": str(getattr(manifest, "manifest_version", "1.0")),
        "image_gen_manifest_type": str(getattr(manifest, "manifest_type", "txt2img")),
    }


def build_pnginfo(manifest: GenerationManifest, *, existing: PngInfo | None = None) -> PngInfo:
    info = existing if existing is not None else PngInfo()
    for key, value in build_png_text_chunks(manifest).items():
        info.add_text(str(key), str(value))
    return info
