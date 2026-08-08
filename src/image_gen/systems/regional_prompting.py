# -----------------------------------------------------------------------------
# REGION ATTRIBUTION
#
# Original REGION syntax and core regional-conditioning design are based on
# work by GitHub user Konpr:
#   https://github.com/Konpr/whats-/tree/main/new_version3
#
# The original author granted permission to use/adapt the code with credit.
# IMAGE_GEN retains that attribution for the REGION language and semantics,
# including REGION{...}, spatial coordinates/tiling, *base=, overlay/common
# modes, latent backend semantics, branch weights/curves, start/stop windows,
# blur, canvas masks, and base_ratio.
#
# This file's IMAGE_GEN implementation is a substantial native adaptation:
# parser-independent conditioning, native model-output execution, canonical CFG
# integration, batching, hires/resolution transforms, replay/fingerprinting,
# validation, telemetry, caching, and low-VRAM execution are IMAGE_GEN work.
# -----------------------------------------------------------------------------

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

REGION_CONTRACT_VERSION = "image-gen-superhybrid-region-v1"
MAX_REGIONS_PER_SLOT = 16
MAX_CANVAS_BYTES = 8 * 1024 * 1024
MAX_CANVAS_PIXELS = 16_777_216
REGION_MASK_CACHE_MAX_ENTRIES = 128
REGION_MASK_CACHE_MAX_BYTES = 64 * 1024 * 1024
_REGION_MASK_CACHE: "OrderedDict[tuple[Any, ...], torch.Tensor]" = OrderedDict()
_REGION_MASK_CACHE_BYTES = 0
_REGION_MASK_CACHE_HITS = 0
_REGION_MASK_CACHE_MISSES = 0
_REGION_MASK_CACHE_LOCK = threading.RLock()


class RegionalPromptError(ValueError):
    """Raised when a SuperHybrid REGION plan is invalid or cannot be replayed."""


@dataclass(frozen=True)
class RuntimeRegionSpec:
    slot_index: int
    region_index: int
    prompt: str
    x1: float
    x2: float
    y1: float
    y2: float
    coords_pixels: bool
    weight: float
    start: float
    stop: float
    blur: float
    curve: str
    mode: str
    base_ratio: float
    canvas: str = ""


@dataclass(frozen=True)
class RuntimeRegionCondition:
    slot_index: int
    region_index: int
    conditioning: torch.Tensor
    mask: torch.Tensor
    strength: float
    mode: str
    metadata: Mapping[str, Any]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _region_slots_behavior_source(slots: Any) -> list[dict[str, Any]]:
    """Return REGION slot data that can materially change conditioning output."""
    output: list[dict[str, Any]] = []
    for slot_value in list(slots or []):
        slot = dict(slot_value or {})
        # Coordinate-resolution provenance explains how resolved coordinates
        # were obtained; the resolved coordinates themselves define behavior.
        slot.pop("coordinate_resolution", None)
        regions: list[dict[str, Any]] = []
        for region_value in list(slot.get("regions") or []):
            region = dict(region_value or {})
            region.pop("coordinate_resolution", None)
            regions.append(region)
        slot["regions"] = regions
        output.append(slot)
    return output


def _fingerprint_source(record: Mapping[str, Any]) -> dict[str, Any]:
    # Runtime estimates and coordinate-resolution provenance are derived
    # diagnostics, not replay inputs. Excluding them keeps the REGION
    # fingerprint stable while preserving prompts, resolved geometry, weights,
    # temporal windows, masks/canvas identity, and semantic digests.
    source = {
        str(key): value
        for key, value in dict(record or {}).items()
        if key not in {"fingerprint", "replay_locked", "replay_source", "runtime_estimate"}
    }
    if "slots" in source:
        source["slots"] = _region_slots_behavior_source(source.get("slots"))
    return source


def _legacy_fingerprint_source(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the pre-hotfix fingerprint source for legacy manifest validation."""
    return {
        str(key): value
        for key, value in dict(record or {}).items()
        if key not in {"fingerprint", "replay_locked", "replay_source"}
    }


def compact_region_record_for_replay(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the behavior-defining REGION record without derived telemetry."""
    value = dict(record or {})
    if not value:
        return {}
    value.pop("runtime_estimate", None)
    value.pop("replay_locked", None)
    value.pop("replay_source", None)
    if "slots" in value:
        value["slots"] = _region_slots_behavior_source(value.get("slots"))
    value["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(value)),
    }
    return value


def _join_prompt_parts(*parts: str) -> str:
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _resolve_region_pixel_coordinates(
    region_blocks: Sequence[Any],
    *,
    width: int,
    height: int,
    coordinate_reference_slot: Mapping[str, Any] | None = None,
    coordinate_reference_width: int | None = None,
    coordinate_reference_height: int | None = None,
) -> tuple[dict[int, tuple[float, float, float, float]], dict[str, Any]]:
    """Resolve REGION pixel coordinates against the active generation canvas.

    REGION prompt text can contain absolute pixel ranges that were authored for an
    older canvas size. The runtime is authoritative, so when those coordinates no
    longer fit the current request we rescale the entire pixel-coordinate region
    plan to the active width/height instead of failing preflight/runtime.

    The source canvas is inferred conservatively from the rightmost/bottommost
    pixel extents present in the REGION block. Normalized-coordinate regions are
    not modified.
    """
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    epsilon = 1e-6
    pixel_entries: list[tuple[int, float, float, float, float]] = []
    for index, item in enumerate(region_blocks):
        if bool(getattr(item, "coords_pixels", False)):
            pixel_entries.append((
                index,
                float(getattr(item, "x1")),
                float(getattr(item, "x2")),
                float(getattr(item, "y1")),
                float(getattr(item, "y2")),
            ))
    if not pixel_entries:
        return {}, {
            "unit": "normalized",
            "source_width": None,
            "source_height": None,
            "target_width": target_width,
            "target_height": target_height,
            "auto_rescaled": False,
        }

    # When a hires pass inherits the same REGION prompt, scale from the already
    # UI-resolved base plan rather than re-interpreting the literal parser-box
    # coordinates.  This preserves the WebUI dimensions as the source of truth
    # while maintaining identical relative REGION geometry at the hires canvas.
    reference = dict(coordinate_reference_slot or {})
    reference_regions = [
        dict(item or {}) for item in list(reference.get("regions") or [])
    ]
    reference_width = int(coordinate_reference_width or 0)
    reference_height = int(coordinate_reference_height or 0)
    if (
        reference_width > 0
        and reference_height > 0
        and len(reference_regions) == len(region_blocks)
    ):
        reference_pixels: dict[int, tuple[float, float, float, float]] = {}
        valid_reference = True
        for index, item in enumerate(region_blocks):
            if not bool(getattr(item, "coords_pixels", False)):
                continue
            coordinates = dict(reference_regions[index].get("coordinates") or {})
            if str(coordinates.get("unit") or "").strip().lower() != "pixels":
                valid_reference = False
                break
            try:
                values = (
                    float(coordinates["x1"]),
                    float(coordinates["x2"]),
                    float(coordinates["y1"]),
                    float(coordinates["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                valid_reference = False
                break
            if not (
                0.0 <= values[0] < values[1] <= float(reference_width) + epsilon
                and 0.0 <= values[2] < values[3] <= float(reference_height) + epsilon
            ):
                valid_reference = False
                break
            reference_pixels[index] = values
        if valid_reference and len(reference_pixels) == len(pixel_entries):
            scale_x = float(target_width) / float(reference_width)
            scale_y = float(target_height) / float(reference_height)
            resolved = {
                index: (
                    max(0.0, min(float(target_width), x1 * scale_x)),
                    max(0.0, min(float(target_width), x2 * scale_x)),
                    max(0.0, min(float(target_height), y1 * scale_y)),
                    max(0.0, min(float(target_height), y2 * scale_y)),
                )
                for index, (x1, x2, y1, y2) in reference_pixels.items()
            }
            return resolved, {
                "unit": "pixels",
                "source": "base_ui_resolved_region_plan",
                "source_width": float(reference_width),
                "source_height": float(reference_height),
                "target_width": target_width,
                "target_height": target_height,
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "auto_rescaled": bool(
                    reference_width != target_width or reference_height != target_height
                ),
            }

    source_width = max(value[2] for value in pixel_entries)
    source_height = max(value[4] for value in pixel_entries)
    if source_width <= 0.0 or source_height <= 0.0:
        raise RegionalPromptError("REGION pixel coordinates require positive source dimensions.")

    overflow = any(
        x1 < -epsilon or y1 < -epsilon or x2 > float(target_width) + epsilon or y2 > float(target_height) + epsilon
        for _index, x1, x2, y1, y2 in pixel_entries
    )
    if not overflow:
        return {index: (x1, x2, y1, y2) for index, x1, x2, y1, y2 in pixel_entries}, {
            "unit": "pixels",
            "source_width": float(source_width),
            "source_height": float(source_height),
            "target_width": target_width,
            "target_height": target_height,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "auto_rescaled": False,
        }

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    resolved: dict[int, tuple[float, float, float, float]] = {}
    for index, x1, x2, y1, y2 in pixel_entries:
        nx1 = max(0.0, min(float(target_width), x1 * scale_x))
        nx2 = max(0.0, min(float(target_width), x2 * scale_x))
        ny1 = max(0.0, min(float(target_height), y1 * scale_y))
        ny2 = max(0.0, min(float(target_height), y2 * scale_y))
        if not (0.0 <= nx1 < nx2 <= float(target_width) + epsilon):
            raise RegionalPromptError("REGION pixel x coordinates could not be reconciled with the active generation width.")
        if not (0.0 <= ny1 < ny2 <= float(target_height) + epsilon):
            raise RegionalPromptError("REGION pixel y coordinates could not be reconciled with the active generation height.")
        resolved[index] = (nx1, nx2, ny1, ny2)

    return resolved, {
        "unit": "pixels",
        "source_width": float(source_width),
        "source_height": float(source_height),
        "target_width": target_width,
        "target_height": target_height,
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "auto_rescaled": True,
    }


def _canvas_metadata(value: str) -> dict[str, Any]:
    raw = str(value or "")
    if not raw:
        return {}
    try:
        payload = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise RegionalPromptError("REGION canvas must be valid base64 data.") from exc
    if len(payload) > MAX_CANVAS_BYTES:
        raise RegionalPromptError("REGION canvas exceeds the 8 MiB decoded-data limit.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            mode = image.mode
            fmt = image.format or ""
    except Exception as exc:
        raise RegionalPromptError("REGION canvas is not a valid image.") from exc
    if width < 1 or height < 1 or width * height > MAX_CANVAS_PIXELS:
        raise RegionalPromptError("REGION canvas dimensions are outside the supported limit.")
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "width": int(width),
        "height": int(height),
        "mode": str(mode),
        "format": str(fmt),
    }


def extract_superhybrid_region_slot(
    prompt: str,
    *,
    slot_index: int,
    steps: int,
    seed: int,
    width: int,
    height: int,
    coordinate_reference_slot: Mapping[str, Any] | None = None,
    coordinate_reference_width: int | None = None,
    coordinate_reference_height: int | None = None,
) -> tuple[str, list[RuntimeRegionSpec], dict[str, Any]]:
    """Extract a SuperHybrid REGION block without executing any A1111 runtime code."""
    from modules.prompt_parsers.vendor import prompt_parser_superhybrid as backend

    text = str(prompt or "")
    try:
        clean_text, region_blocks = backend.get_prompt_regions(
            text,
            steps=int(steps),
            use_scheduling=True,
            seed=int(seed),
        )
    except Exception as exc:
        raise RegionalPromptError(f"SuperHybrid REGION parsing failed: {exc}") from exc
    if not region_blocks:
        return text, [], {
            "slot_index": int(slot_index),
            "source_prompt": text,
            "base_prompt": text,
            "region_count": 0,
            "regions": [],
        }
    if len(region_blocks) > MAX_REGIONS_PER_SLOT:
        raise RegionalPromptError(
            f"SuperHybrid REGION supports at most {MAX_REGIONS_PER_SLOT} regions per image slot."
        )

    extracted_base_prompt = backend.extract_non_region_text(
        clean_text,
        region_blocks,
        original_text=text,
    ).strip()
    explicit_base_prompt = next(
        (str(item.base_text or "").strip() for item in region_blocks if str(item.base_text or "").strip()),
        "",
    )
    base_prompt = explicit_base_prompt or extracted_base_prompt
    runtime_specs: list[RuntimeRegionSpec] = []
    record_regions: list[dict[str, Any]] = []
    pixel_coordinate_map, pixel_coordinate_metadata = _resolve_region_pixel_coordinates(
        region_blocks,
        width=int(width),
        height=int(height),
        coordinate_reference_slot=coordinate_reference_slot,
        coordinate_reference_width=coordinate_reference_width,
        coordinate_reference_height=coordinate_reference_height,
    )
    for region_index, item in enumerate(region_blocks):
        backend_name = str(item.backend or "").strip().lower()
        if backend_name and backend_name not in {"latent"}:
            raise RegionalPromptError(
                f"REGION backend={backend_name!r} is incompatible with IMAGE_GEN. "
                "Use the native model-output backend or omit backend=."
            )
        coords_pixels = bool(item.coords_pixels)
        if coords_pixels:
            values = pixel_coordinate_map.get(
                region_index,
                (float(item.x1), float(item.x2), float(item.y1), float(item.y2)),
            )
            if not (0.0 <= values[0] < values[1] <= float(width)):
                raise RegionalPromptError("REGION pixel x coordinates must fit the generation width.")
            if not (0.0 <= values[2] < values[3] <= float(height)):
                raise RegionalPromptError("REGION pixel y coordinates must fit the generation height.")
        else:
            values = (float(item.x1), float(item.x2), float(item.y1), float(item.y2))
            if not (0.0 <= values[0] < values[1] <= 1.0):
                raise RegionalPromptError("REGION normalized x coordinates must be within 0..1.")
            if not (0.0 <= values[2] < values[3] <= 1.0):
                raise RegionalPromptError("REGION normalized y coordinates must be within 0..1.")
        weight = float(item.weight)
        if not math.isfinite(weight) or weight < 0.0 or weight > 4.0:
            raise RegionalPromptError("REGION weight must be finite and between 0 and 4.")
        start = float(item.start)
        stop = float(item.stop)
        if not (0.0 <= start <= stop <= 1.0):
            raise RegionalPromptError("REGION activation window must satisfy 0 <= start <= stop <= 1.")
        blur = float(item.blur)
        if not (0.0 <= blur <= 1.0):
            raise RegionalPromptError("REGION blur must be between 0 and 1.")
        base_ratio = float(item.base_ratio)
        if not (0.0 <= base_ratio <= 1.0):
            raise RegionalPromptError("REGION base_ratio must be between 0 and 1.")
        mode = str(item.mode or "overlay").strip().lower()
        if mode not in {"overlay", "common"}:
            raise RegionalPromptError(f"Unsupported REGION mode: {mode!r}.")
        curve = str(item.curve or "linear").strip().lower()
        branch_prompt = str(item.text or "").strip()
        region_prompt = (
            f"{base_prompt}, {branch_prompt}".strip(", ")
            if mode == "common"
            else branch_prompt
        )
        if not region_prompt:
            raise RegionalPromptError("REGION branch resolved to an empty prompt.")
        canvas = str(item.canvas or "")
        canvas_meta = _canvas_metadata(canvas)
        spec = RuntimeRegionSpec(
            slot_index=int(slot_index),
            region_index=int(region_index),
            prompt=region_prompt,
            x1=values[0],
            x2=values[1],
            y1=values[2],
            y2=values[3],
            coords_pixels=coords_pixels,
            weight=weight,
            start=start,
            stop=stop,
            blur=blur,
            curve=curve,
            mode=mode,
            base_ratio=base_ratio,
            canvas=canvas,
        )
        runtime_specs.append(spec)
        record_regions.append({
            "region_index": int(region_index),
            "prompt": region_prompt,
            "coordinates": {
                "x1": values[0], "x2": values[1], "y1": values[2], "y2": values[3],
                "unit": "pixels" if coords_pixels else "normalized",
            },
            "coordinate_resolution": dict(pixel_coordinate_metadata) if coords_pixels else {
                "unit": "normalized",
                "target_width": int(width),
                "target_height": int(height),
                "auto_rescaled": False,
            },
            "weight": weight,
            "start": start,
            "stop": stop,
            "blur": blur,
            "curve": curve,
            "mode": mode,
            "base_ratio": base_ratio,
            "canvas": canvas_meta,
            "backend": "image_gen_model_output",
        })
    return base_prompt, runtime_specs, {
        "slot_index": int(slot_index),
        "source_prompt": text,
        "base_prompt": base_prompt,
        "region_count": len(record_regions),
        "coordinate_resolution": dict(pixel_coordinate_metadata),
        "regions": record_regions,
    }


def _active_region_steps(start: float, stop: float, steps: int) -> int:
    count = max(1, int(steps))
    return sum(
        1
        for index in range(count)
        if float(start) <= ((index / count) if index > 0 else 0.0) <= float(stop)
    )


def estimate_region_runtime(
    *,
    width: int,
    height: int,
    steps: int,
    slots: Sequence[Mapping[str, Any]],
    latent_channels: int = 4,
) -> dict[str, Any]:
    """Estimate REGION overhead without claiming exact device memory usage.

    The native backend evaluates one regional branch at a time.  The estimate
    therefore reports extra UNet calls and bounded temporary buffers instead of
    multiplying full model residency by the region count.
    """
    latent_width = max(1, math.ceil(int(width) / 8))
    latent_height = max(1, math.ceil(int(height) / 8))
    slot_values = [dict(item or {}) for item in slots]
    region_values = [
        (int(slot.get("slot_index", slot_index)), dict(region or {}))
        for slot_index, slot in enumerate(slot_values)
        for region in list(slot.get("regions") or [])
    ]
    extra_calls = sum(
        _active_region_steps(
            float(region.get("start", 0.0) or 0.0),
            float(region.get("stop", 1.0) or 1.0),
            int(steps),
        )
        for _slot_index, region in region_values
    )
    max_active = 0
    active_by_step: list[int] = []
    for step_index in range(max(1, int(steps))):
        progress = (step_index / max(1, int(steps))) if step_index > 0 else 0.0
        active = sum(
            1
            for _slot_index, region in region_values
            if float(region.get("start", 0.0) or 0.0)
            <= progress
            <= float(region.get("stop", 1.0) or 1.0)
        )
        active_by_step.append(active)
        max_active = max(max_active, active)
    pixels = latent_width * latent_height
    mask_bytes_fp16 = len(region_values) * pixels * 2
    mask_bytes_fp32 = len(region_values) * pixels * 4
    branch_bytes_fp16 = max(1, int(latent_channels)) * pixels * 2
    branch_bytes_fp32 = max(1, int(latent_channels)) * pixels * 4
    return {
        "estimate_version": "image-gen-region-estimate-v1",
        "assumptions": {
            "vae_scale_factor": 8,
            "latent_channels": int(latent_channels),
            "evaluation": "sequential_one_region_at_a_time",
            "excludes_model_residency": True,
        },
        "latent_width": latent_width,
        "latent_height": latent_height,
        "region_count": len(region_values),
        "extra_unet_calls": int(extra_calls),
        "max_active_regions_per_step": int(max_active),
        "active_region_counts_by_step": active_by_step,
        "estimated_mask_cache_bytes": {"fp16": mask_bytes_fp16, "fp32": mask_bytes_fp32},
        "estimated_branch_output_bytes": {"fp16": branch_bytes_fp16, "fp32": branch_bytes_fp32},
        "estimated_incremental_peak_bytes": {
            "fp16": mask_bytes_fp16 + branch_bytes_fp16,
            "fp32": mask_bytes_fp32 + branch_bytes_fp32,
        },
    }


def build_region_record(
    *,
    parser_id: str,
    parser_version: str,
    pass_name: str,
    width: int,
    height: int,
    steps: int,
    overlap_policy: str,
    slots: Sequence[Mapping[str, Any]],
    semantic_digests_by_slot: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    normalized_policy = str(overlap_policy or "additive").strip().lower()
    if normalized_policy not in {"normalize", "additive", "priority"}:
        raise RegionalPromptError("REGION overlap policy must be normalize, additive, or priority.")
    normalized_slots = [dict(item or {}) for item in slots]
    semantics = [list(map(str, values)) for values in (semantic_digests_by_slot or [[] for _ in normalized_slots])]
    if len(semantics) != len(normalized_slots):
        raise RegionalPromptError("REGION semantic slot count does not match the region plan.")
    for slot, slot_semantics in zip(normalized_slots, semantics):
        if len(slot_semantics) != int(slot.get("region_count", 0) or 0):
            raise RegionalPromptError("REGION semantic count does not match the region count.")
        for region, digest in zip(list(slot.get("regions") or []), slot_semantics):
            region["semantic_digest"] = digest
    record = {
        "contract_version": REGION_CONTRACT_VERSION,
        "parser_id": str(parser_id or "region_addon"),
        "parser_version": str(parser_version or ""),
        "pass": str(pass_name or "base"),
        "backend": "image_gen_model_output",
        "evaluation_mode": "sequential_low_vram",
        "overlap_policy": normalized_policy,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "slot_count": len(normalized_slots),
        "slots": normalized_slots,
        "region_count": sum(int(item.get("region_count", 0) or 0) for item in normalized_slots),
        "runtime_estimate": estimate_region_runtime(
            width=int(width),
            height=int(height),
            steps=int(steps),
            slots=normalized_slots,
        ),
        "replay_locked": False,
        "replay_source": "reconstruct",
    }
    record["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(record)),
    }
    return record


def validate_recorded_region_record(recorded: Mapping[str, Any], *, current: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(recorded or {})
    if value.get("contract_version") != REGION_CONTRACT_VERSION:
        raise RegionalPromptError("Recorded REGION plan uses an unsupported contract version.")
    fingerprint = dict(value.get("fingerprint") or {})
    digest = str(fingerprint.get("digest") or "")
    current_digest = _stable_hash(_fingerprint_source(value))
    legacy_digest = _stable_hash(_legacy_fingerprint_source(value))
    if fingerprint.get("algorithm") != "sha256" or digest not in {current_digest, legacy_digest}:
        raise RegionalPromptError("Recorded REGION plan fingerprint validation failed.")
    for key in ("parser_id", "parser_version", "pass", "backend", "overlap_policy", "width", "height", "steps", "slot_count", "region_count"):
        if value.get(key) != dict(current or {}).get(key):
            raise RegionalPromptError(f"Recorded REGION {key.replace('_', ' ')} does not match the current request.")
    if _region_slots_behavior_source(value.get("slots")) != _region_slots_behavior_source(
        dict(current or {}).get("slots")
    ):
        raise RegionalPromptError("Recorded REGION prompts, geometry, or semantics changed.")
    value["replay_locked"] = True
    value["replay_source"] = "recorded_exact"
    return value


def select_region_record_slot(recorded: Mapping[str, Any], slot_index: int) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != REGION_CONTRACT_VERSION:
        return record
    slots = [dict(item or {}) for item in list(record.get("slots") or [])]
    index = int(slot_index)
    if index < 0 or index >= len(slots):
        raise RegionalPromptError("REGION slot index is outside the recorded batch.")
    selected = dict(record)
    selected["slots"] = [{**slots[index], "slot_index": 0}]
    selected["source_batch_slot_index"] = index
    selected["slot_count"] = 1
    selected["region_count"] = int(slots[index].get("region_count", 0) or 0)
    selected["runtime_estimate"] = estimate_region_runtime(
        width=int(selected.get("width", 1) or 1),
        height=int(selected.get("height", 1) or 1),
        steps=int(selected.get("steps", 1) or 1),
        slots=selected["slots"],
    )
    selected["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(selected)),
    }
    return selected


def _curve_value(name: str, progress: float) -> float:
    p = max(0.0, min(1.0, float(progress)))
    normalized = str(name or "linear").strip().lower().replace("_", "-")
    if normalized in {"linear", "none"}:
        return 1.0
    aliases = {
        "smooth": "ease-in-out",
        "smoothstep": "ease-in-out",
        "cos": "sine-in-out",
        "cosine": "sine-in-out",
        "quadratic": "ease-in",
        "exp": "expo-in",
        "exponential": "expo-in",
        "exp-decay": "expo-in",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        from modules.prompt_parsers.vendor import prompt_parser_superhybrid as backend

        value = float(backend._apply_easing(p, normalized))
    except Exception:
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return max(0.0, value)


def region_strength_for_step(spec: RuntimeRegionSpec, *, step_index: int, total_steps: int) -> float:
    """Return the temporal REGION factor; branch weight/base blending are applied later.

    SuperHybrid's source implementation treats ``linear`` as constant strength
    inside the active window. Named easing curves ramp within that window.
    """
    count = max(1, int(total_steps))
    index = max(0, int(step_index))
    progress = (index / count) if index > 0 else 0.0
    if progress < spec.start or progress > spec.stop:
        return 0.0
    if spec.stop <= spec.start:
        local = 1.0
    else:
        local = (progress - spec.start) / (spec.stop - spec.start)
    return float(_curve_value(spec.curve, local))


def _decode_canvas_mask(canvas: str, *, region_index: int, region_count: int) -> torch.Tensor:
    payload = base64.b64decode(canvas, validate=True)
    with Image.open(io.BytesIO(payload)) as image:
        rgba = image.convert("RGBA")
        pixel_values = (
            rgba.get_flattened_data()
            if hasattr(rgba, "get_flattened_data")
            else rgba.getdata()
        )
        pixels = torch.tensor(list(pixel_values), dtype=torch.float32).reshape(rgba.height, rgba.width, 4)
    alpha = pixels[..., 3] / 255.0
    rgb = pixels[..., :3]
    if region_count == 1:
        luminance = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) / 255.0
        mask = torch.maximum(luminance, alpha * (luminance <= 1e-6).to(luminance.dtype))
        return mask.clamp(0.0, 1.0)
    opaque = alpha > 0.0
    flat_rgb = rgb.to(torch.uint8).reshape(-1, 3)
    flat_opaque = opaque.reshape(-1)
    palette: list[tuple[int, int, int]] = []
    for color, is_opaque in zip(flat_rgb.tolist(), flat_opaque.tolist()):
        tup = tuple(int(v) for v in color)
        if not is_opaque or tup == (0, 0, 0):
            continue
        if tup not in palette:
            palette.append(tup)
        if len(palette) >= region_count:
            break
    if len(palette) < region_count:
        raise RegionalPromptError(
            "Multi-region canvas must contain at least one distinct non-black opaque color per region."
        )
    target = torch.tensor(palette[region_index], dtype=torch.float32)
    # Exact color matching is deliberate; antialiased edges are supplied by blur.
    mask = ((rgb == target).all(dim=-1) & opaque).to(torch.float32)
    return mask


def _region_mask_cache_key(
    spec: RuntimeRegionSpec,
    *,
    latent_height: int,
    latent_width: int,
    generation_width: int,
    generation_height: int,
    region_count: int,
) -> tuple[Any, ...]:
    canvas_digest = hashlib.sha256(str(spec.canvas or "").encode("utf-8")).hexdigest() if spec.canvas else ""
    return (
        int(spec.region_index),
        round(float(spec.x1), 8), round(float(spec.x2), 8),
        round(float(spec.y1), 8), round(float(spec.y2), 8),
        bool(spec.coords_pixels), round(float(spec.blur), 8), canvas_digest,
        int(latent_height), int(latent_width), int(generation_width), int(generation_height),
        int(region_count),
    )


def _build_region_mask_cpu(
    spec: RuntimeRegionSpec,
    *,
    latent_height: int,
    latent_width: int,
    generation_width: int,
    generation_height: int,
    region_count: int,
) -> torch.Tensor:
    h = int(latent_height)
    w = int(latent_width)
    if spec.coords_pixels:
        x1 = int(math.floor(spec.x1 / max(1.0, generation_width) * w))
        x2 = int(math.ceil(spec.x2 / max(1.0, generation_width) * w))
        y1 = int(math.floor(spec.y1 / max(1.0, generation_height) * h))
        y2 = int(math.ceil(spec.y2 / max(1.0, generation_height) * h))
    else:
        x1 = int(math.floor(spec.x1 * w))
        x2 = int(math.ceil(spec.x2 * w))
        y1 = int(math.floor(spec.y1 * h))
        y2 = int(math.ceil(spec.y2 * h))
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    mask = torch.zeros((1, 1, h, w), dtype=torch.float32, device="cpu")
    mask[..., y1:y2, x1:x2] = 1.0
    if spec.canvas:
        canvas_mask = _decode_canvas_mask(
            spec.canvas,
            region_index=spec.region_index,
            region_count=region_count,
        ).unsqueeze(0).unsqueeze(0)
        canvas_mask = F.interpolate(canvas_mask, size=(h, w), mode="nearest")
        mask = mask * canvas_mask
    if spec.blur > 0.0:
        sigma = max(0.5, float(spec.blur) * min(h, w) * 0.08)
        radius = min(15, max(1, int(math.ceil(3.0 * sigma))))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        mask = F.conv2d(mask, kernel.view(1, 1, 1, -1), padding=(0, radius))
        mask = F.conv2d(mask, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return mask.clamp(0.0, 1.0).contiguous()


def build_region_mask_cached(
    spec: RuntimeRegionSpec,
    *,
    latent_height: int,
    latent_width: int,
    generation_width: int,
    generation_height: int,
    region_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Return a region mask and whether it came from the shared CPU LRU cache."""
    global _REGION_MASK_CACHE_BYTES, _REGION_MASK_CACHE_HITS, _REGION_MASK_CACHE_MISSES
    key = _region_mask_cache_key(
        spec,
        latent_height=latent_height,
        latent_width=latent_width,
        generation_width=generation_width,
        generation_height=generation_height,
        region_count=region_count,
    )
    with _REGION_MASK_CACHE_LOCK:
        cached = _REGION_MASK_CACHE.get(key)
        if cached is not None:
            _REGION_MASK_CACHE.move_to_end(key)
            _REGION_MASK_CACHE_HITS += 1
            return cached.to(device=device, dtype=dtype), True
        _REGION_MASK_CACHE_MISSES += 1

    built = _build_region_mask_cpu(
        spec,
        latent_height=latent_height,
        latent_width=latent_width,
        generation_width=generation_width,
        generation_height=generation_height,
        region_count=region_count,
    )
    bytes_used = int(built.numel() * built.element_size())
    with _REGION_MASK_CACHE_LOCK:
        previous = _REGION_MASK_CACHE.pop(key, None)
        if previous is not None:
            _REGION_MASK_CACHE_BYTES -= int(previous.numel() * previous.element_size())
        _REGION_MASK_CACHE[key] = built
        _REGION_MASK_CACHE_BYTES += bytes_used
        while (
            len(_REGION_MASK_CACHE) > REGION_MASK_CACHE_MAX_ENTRIES
            or _REGION_MASK_CACHE_BYTES > REGION_MASK_CACHE_MAX_BYTES
        ):
            _old_key, old_value = _REGION_MASK_CACHE.popitem(last=False)
            _REGION_MASK_CACHE_BYTES -= int(old_value.numel() * old_value.element_size())
    return built.to(device=device, dtype=dtype), False


def build_region_mask(
    spec: RuntimeRegionSpec,
    *,
    latent_height: int,
    latent_width: int,
    generation_width: int,
    generation_height: int,
    region_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask, _cache_hit = build_region_mask_cached(
        spec,
        latent_height=latent_height,
        latent_width=latent_width,
        generation_width=generation_width,
        generation_height=generation_height,
        region_count=region_count,
        device=device,
        dtype=dtype,
    )
    return mask


def region_mask_cache_stats() -> dict[str, Any]:
    with _REGION_MASK_CACHE_LOCK:
        return {
            "entries": len(_REGION_MASK_CACHE),
            "bytes": int(_REGION_MASK_CACHE_BYTES),
            "hits": int(_REGION_MASK_CACHE_HITS),
            "misses": int(_REGION_MASK_CACHE_MISSES),
            "max_entries": REGION_MASK_CACHE_MAX_ENTRIES,
            "max_bytes": REGION_MASK_CACHE_MAX_BYTES,
        }


def clear_region_mask_cache() -> None:
    global _REGION_MASK_CACHE_BYTES, _REGION_MASK_CACHE_HITS, _REGION_MASK_CACHE_MISSES
    with _REGION_MASK_CACHE_LOCK:
        _REGION_MASK_CACHE.clear()
        _REGION_MASK_CACHE_BYTES = 0
        _REGION_MASK_CACHE_HITS = 0
        _REGION_MASK_CACHE_MISSES = 0
