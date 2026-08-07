from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

HIRES_FAILURE_STAGE_CODES = (
    "pth_native_inference",
    "native_dimension_verification",
    "target_aspect_correction",
    "vae_encode",
    "second_pass_diffusion",
    "final_exact_size_correction",
)
HIRES_FAILURE_STAGE_LABELS = {
    "pth_native_inference": "PTH native inference",
    "native_dimension_verification": "Native dimension verification",
    "target_aspect_correction": "Target aspect correction",
    "vae_encode": "VAE encode",
    "second_pass_diffusion": "Second-pass diffusion",
    "final_exact_size_correction": "Final exact-size correction",
}
HIRES_CORRECTION_FINGERPRINT_VERSION = "phase14n12c-correction-fingerprint-v1"
_STAGE_PATTERN = re.compile(r"\[HIRES_STAGE:(?P<code>[a-z0-9_]+)\]")


def format_hires_failure(stage_code: str, message: str, **context: Any) -> str:
    code = str(stage_code or "").strip().casefold()
    if code not in HIRES_FAILURE_STAGE_CODES:
        raise ValueError(f"Unsupported hires failure stage: {stage_code!r}.")
    details = []
    for key, value in context.items():
        if value is None or value == "":
            continue
        details.append(f"{key}={value}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"[HIRES_STAGE:{code}] {message}{suffix}"


def extract_hires_failure_stage(message: Any) -> str:
    match = _STAGE_PATTERN.search(str(message or ""))
    if not match:
        return ""
    code = str(match.group("code") or "").casefold()
    return code if code in HIRES_FAILURE_STAGE_CODES else ""


def hires_failure_stage_label(stage_code: str) -> str:
    code = str(stage_code or "").strip().casefold()
    return HIRES_FAILURE_STAGE_LABELS.get(code, code.replace("_", " ").title())


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_hires_correction_fingerprint(
    *,
    upscaler_id: str,
    upscaler_sha256: str,
    native_scale: int,
    actual_native_width: int,
    actual_native_height: int,
    target_width: int,
    target_height: int,
    aspect_policy: str,
    padding_mode: str,
    resolved_filter: str,
    target_correction: Mapping[str, Any],
    dimension_plan_version: str,
) -> dict[str, Any]:
    correction = dict(target_correction or {})
    geometry_keys = (
        "contract_version",
        "aspect_policy",
        "padding_mode",
        "source_width",
        "source_height",
        "target_width",
        "target_height",
        "resize_scale",
        "pre_crop_width",
        "pre_crop_height",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "fitted_width",
        "fitted_height",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "non_uniform_geometry_applied",
        "final_size_correction_filter_resolved",
    )
    contract = {
        "dimension_plan_version": str(dimension_plan_version or ""),
        "upscaler_id": str(upscaler_id or ""),
        "upscaler_sha256": str(upscaler_sha256 or "").casefold(),
        "native_scale": int(native_scale or 0),
        "actual_native_width": int(actual_native_width or 0),
        "actual_native_height": int(actual_native_height or 0),
        "target_width": int(target_width or 0),
        "target_height": int(target_height or 0),
        "aspect_policy": str(aspect_policy or "stretch"),
        "padding_mode": str(padding_mode or "reflect"),
        "resolved_filter": str(resolved_filter or "none"),
        "geometry": {key: correction.get(key) for key in geometry_keys if key in correction},
    }
    canonical = _canonical_json(contract)
    return {
        "schema_version": HIRES_CORRECTION_FINGERPRINT_VERSION,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "contract": contract,
    }


def build_hires_correction_audit(upscale_metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(upscale_metadata or {})
    correction = dict(source.get("target_correction") or {})
    native_width = int(source.get("actual_native_width") or 0)
    native_height = int(source.get("actual_native_height") or 0)
    target_width = int(correction.get("target_width") or source.get("target_width") or 0)
    target_height = int(correction.get("target_height") or source.get("target_height") or 0)
    policy = str(correction.get("aspect_policy") or source.get("aspect_policy") or "stretch")
    resolved_filter = str(
        correction.get("final_size_correction_filter_resolved")
        or source.get("final_size_correction_filter")
        or "none"
    )
    severity = float(correction.get("correction_severity") or 0.0)
    if policy == "crop_to_fill":
        geometry = (
            f"intermediate {int(correction.get('pre_crop_width') or 0)}x{int(correction.get('pre_crop_height') or 0)}; "
            f"crop L{int(correction.get('crop_left') or 0)} T{int(correction.get('crop_top') or 0)} "
            f"R{max(0, int(correction.get('pre_crop_width') or 0) - int(correction.get('crop_right') or 0))} "
            f"B{max(0, int(correction.get('pre_crop_height') or 0) - int(correction.get('crop_bottom') or 0))}"
        )
        label = "Centered crop to fill"
    elif policy == "pad_to_fit":
        geometry = (
            f"fitted {int(correction.get('fitted_width') or 0)}x{int(correction.get('fitted_height') or 0)}; "
            f"pad L{int(correction.get('pad_left') or 0)} T{int(correction.get('pad_top') or 0)} "
            f"R{int(correction.get('pad_right') or 0)} B{int(correction.get('pad_bottom') or 0)}"
        )
        label = f"Pad to fit ({str(correction.get('padding_mode') or source.get('padding_mode') or 'reflect')})"
    else:
        geometry = "direct independent-axis resize" if correction.get("non_uniform_geometry_applied") else "direct resize"
        label = "Stretch"
    summary = (
        f"PTH native {native_width}x{native_height} | {label} | {geometry} | "
        f"canvas {target_width}x{target_height} | filter {resolved_filter} | severity {severity * 100.0:.1f}%"
    )
    return {
        "schema_version": "phase14n12c-correction-audit-v1",
        "summary": summary,
        "native_width": native_width,
        "native_height": native_height,
        "target_width": target_width,
        "target_height": target_height,
        "aspect_policy": policy,
        "resolved_filter": resolved_filter,
        "severity": severity,
        "geometry_summary": geometry,
    }
