from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from image_gen.webui.residency_telemetry import latest_runtime_job_report


IMAGE_EXECUTION_SCHEMA_VERSION = 1
_IMAGE_EXECUTION_KEY = "webui_image_execution"
_TIMING_KEYS = (
    "last_job_total_ms",
    "request_setup_time_ms",
    "next_job_preparation_time_ms",
    "generation_execution_time_ms",
    "post_generation_residency_time_ms",
    "output_save_wait_time_ms",
    "post_generation_finalize_time_ms",
    "checkpoint_hydration_time_ms",
    "cpu_to_gpu_promotion_time_ms",
    "first_step_latency_ms",
)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_ms(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 3)


def build_image_execution_record(
    runtime_status: Mapping[str, Any] | None,
    *,
    job_id: str | None = None,
    output_path: str | Path | None = None,
    image_number: int | None = None,
) -> dict[str, Any]:
    report = latest_runtime_job_report(runtime_status, job_id=job_id)
    if not report:
        return {}
    raw_timings = _dict(report.get("timings"))
    timings = {
        key: value
        for key in _TIMING_KEYS
        if (value := _finite_ms(raw_timings.get(key))) is not None
    }
    total_ms = timings.get("last_job_total_ms")
    if total_ms is None:
        return {}
    classification = str(report.get("generation_residency_classification") or "").strip().lower()
    resident_reuse = bool(raw_timings.get("resident_reuse_benefited_last_job")) or classification in {
        "hot_reuse",
        "hot_staged_reuse",
        "managed_reuse",
        "resident_managed_reuse",
        "managed_resident_reuse",
    }
    hydration_ms = timings.get("checkpoint_hydration_time_ms")
    checkpoint_hydration_occurred = bool(not resident_reuse and hydration_ms is not None and hydration_ms > 0)
    path_value = str(output_path or "").strip()
    return {
        "schema_version": IMAGE_EXECUTION_SCHEMA_VERSION,
        "job_id": str(report.get("job_id") or job_id or "").strip(),
        "image_number": int(image_number) if image_number is not None else None,
        "output_name": Path(path_value).name if path_value else None,
        "execution_time_ms": total_ms,
        "timing_scope": "resident_command_execution_excludes_queue_wait_and_pause",
        "timings": timings,
        "generation_residency_classification": classification or None,
        "residency_state_effective": str(report.get("residency_state_effective") or "").strip().lower() or None,
        "residency_mode_requested": str(report.get("residency_mode_requested") or "").strip().lower() or None,
        "resident_change_classification": str(report.get("resident_change_classification") or "").strip().lower() or None,
        "post_job_residency_action": str(report.get("post_job_residency_action") or "").strip().lower() or None,
        "resident_reuse_benefited": resident_reuse,
        "checkpoint_hydration_occurred": checkpoint_hydration_occurred,
        "checkpoint_hydration_time_ms": hydration_ms,
    }


def image_execution_from_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _dict(manifest)
    runtime_info = _dict(source.get("runtime_info"))
    extra = _dict(runtime_info.get("extra"))
    payload = _dict(extra.get(_IMAGE_EXECUTION_KEY))
    if int(payload.get("schema_version") or 0) != IMAGE_EXECUTION_SCHEMA_VERSION:
        return {}
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.hmr06.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def persist_image_execution_record(image_path: str | Path, record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Best-effort augmentation of existing output JSON sidecars.

    HMR-06 deliberately does not create a new timing sidecar/authority. When the
    normal replay or diagnostics JSON already exists, the timing record is added
    under runtime_info.extra. If sidecars were disabled by the user, session UI
    timing still works from GenerationJob.image_execution_reports.
    """

    payload = _dict(record)
    if not payload:
        return {"persisted": False, "reason": "empty_record", "paths": []}
    image = Path(image_path)
    candidates = [
        image.with_suffix(".json"),
        image.with_name(f"{image.stem}.diagnostics.json"),
    ]
    written: list[str] = []
    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("sidecar root is not a JSON object")
            runtime_info = _dict(loaded.get("runtime_info"))
            extra = _dict(runtime_info.get("extra"))
            extra[_IMAGE_EXECUTION_KEY] = payload
            runtime_info["extra"] = extra
            loaded["runtime_info"] = runtime_info
            _write_json_atomic(path, loaded)
            written.append(str(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "persisted": bool(written),
        "reason": "written" if written else ("no_existing_json_sidecar" if not errors else "write_failed"),
        "paths": written,
        "errors": errors,
    }
