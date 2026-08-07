from __future__ import annotations

import copy
import csv
import io
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.model_selection import WebUIModelSelectionState
from modules.project_context import ProjectContext

NATIVE_FORMAT = "image_gen_queue"
NATIVE_VERSION = 1
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PARSED_JOBS = 1000
MAX_IMMEDIATE_SUBMISSION = 250
_TOKEN_TTL_SECONDS = 15 * 60

_INT_FIELDS = {"seed", "width", "height", "steps", "batch_size", "batch_count", "clip_skip", "hires_width", "hires_height", "hires_steps", "hires_expected_native_scale", "outpaint_shape_target_width", "outpaint_shape_target_height", "outpaint_shape_base_width", "outpaint_shape_base_height"}
_FLOAT_FIELDS = {"cfg_scale", "cfg_rescale", "guidance_rescale", "hires_scale", "hires_axis_scale_width", "hires_axis_scale_height", "hires_uniform_scale", "hires_denoising_strength", "outpaint_shape_denoising_strength"}
_BOOL_FIELDS = {"save_images", "save_txt", "save_json", "save_diagnostics_json", "tiling", "prompt_shadow_compare", "hires_enabled", "hires_aspect_ratio_changed", "hires_save_lowres", "outpaint_shape_expansion_enabled", "outpaint_shape_save_base"}
_DICT_FIELDS = {"sampler_kwargs", "scheduler_kwargs", "prompt_parser_kwargs", "prompt_shortcut_profile_snapshot", "hires_prompt_parser_kwargs", "hires_shortcut_profile_snapshot", "prompt_preflight", "prompt_route_plan", "hires_prompt_route_plan", "hires_recorded_target_correction", "hires_recorded_correction_fingerprint", "hires_dimension_plan", "outpaint_shape_runtime_record", "parser_kwargs", "extras", "variation_matrix"}
_REMAP_FIELDS = {"model_path", "vae_path", "sampler_name", "scheduler_name"}
_ALIASES = {
    "prompt": "positive_prompt",
    "sampler": "sampler_name",
    "scheduler": "scheduler_name",
    "model": "model_path",
    "vae": "vae_path",
    "guidance_rescale": "cfg_rescale",
}
_REQUEST_FIELDS = {
    "positive_prompt", "negative_prompt", "seed", "width", "height", "steps",
    "cfg_scale", "batch_size", "batch_count", "sampler_name", "scheduler_name",
    "model_path", "vae_path", "sampler_kwargs", "scheduler_kwargs", "cfg_rescale",
    "compatibility_mode", "clip_skip", "tiling", "prompt_parser_name", "prompt_parser_kwargs",
    "prompt_shortcut_profile_name", "prompt_shortcut_profile_snapshot", "prompt_parser_preset_name",
    "base_prompt_parser_name", "base_shortcut_profile_name", "hires_prompt_parser_mode",
    "hires_prompt_parser_name", "hires_prompt_parser_kwargs", "hires_shortcut_profile_mode",
    "hires_shortcut_profile_name", "hires_shortcut_profile_snapshot", "hires_positive_prompt",
    "hires_negative_prompt", "hires_size_mode", "hires_scale", "hires_width", "hires_height", "hires_axis_scale_width", "hires_axis_scale_height", "hires_uniform_scale", "hires_aspect_ratio_changed", "hires_dimension_plan_version", "hires_dimension_plan", "hires_enabled", "hires_steps", "hires_denoising_strength", "hires_step_policy", "hires_sampler_name", "hires_scheduler_name", "hires_cfg_scale", "hires_cfg_rescale", "hires_upscaler", "hires_upscaler_id", "hires_expected_upscaler_sha256", "hires_expected_native_scale", "hires_final_size_correction_filter", "hires_aspect_policy", "hires_padding_mode", "hires_recorded_target_correction", "hires_correction_fingerprint_enabled", "hires_recorded_correction_fingerprint", "hires_save_lowres", "prompt_preflight", "prompt_shadow_compare", "prompt_route_plan", "hires_prompt_route_plan", "parser_kwargs", "lora_paths",
    "prompt_asset_contract_version", "loras", "textual_inversions", "vae_name", "vae_hash", "output_dir", "output_prefix", "save_images", "save_txt", "save_json", "save_diagnostics_json",
    "outpaint_shape_expansion_enabled", "outpaint_shape_target_mode", "outpaint_shape_target_width", "outpaint_shape_target_height",
    "outpaint_shape_base_width", "outpaint_shape_base_height", "outpaint_shape_anchor", "outpaint_shape_context_seed_mode",
    "outpaint_shape_source_handoff", "outpaint_shape_prompt_mode", "outpaint_shape_overlay_positive_prompt", "outpaint_shape_overlay_negative_prompt",
    "outpaint_shape_denoising_strength", "outpaint_shape_save_base", "outpaint_shape_runtime_record",
    "extras", "variation_matrix",
}
_SECRET_KEYS = {
    "preflight_token", "auth_token", "access_token", "refresh_token", "api_key",
    "password", "secret", "cookie", "authorization",
}
_TEMPORARY_KEYS = {
    "selection_id", "selected_output_ids", "gallery_selection", "temporary_selection_ids",
    "preflight", "parse_token",
}
_PROVENANCE_KEYS = {
    "replay_quality", "replay_label", "metadata_source", "manifest_version",
    "applied_remaps",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _safe_key(key: Any) -> str:
    return str(key or "").strip()


def _sanitize_export(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _safe_key(raw_key)
            lowered = key.lower()
            if (
                not key
                or key.startswith("_webui_")
                or lowered in _SECRET_KEYS
                or lowered in _TEMPORARY_KEYS
                or lowered == "token"
                or lowered.endswith("_token")
            ):
                continue
            output[key] = _sanitize_export(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_sanitize_export(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class ImportedJob:
    source_index: int
    source_label: str
    job_id: str
    raw: dict[str, Any]
    normalized: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    unknown_fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportBatchResult:
    format: str
    version: int | None
    jobs: list[ImportedJob]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    top_level_unknown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "jobs": [item.to_dict() for item in self.jobs],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "top_level_unknown": copy.deepcopy(self.top_level_unknown),
            "job_count": len(self.jobs),
            "valid_parse_count": sum(1 for item in self.jobs if not item.errors),
            "invalid_parse_count": sum(1 for item in self.jobs if item.errors),
        }


@dataclass
class ImportPreflight:
    valid: bool
    jobs: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    summary: dict[str, Any]
    preflight_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportResult:
    content: bytes
    media_type: str
    filename: str
    warnings: list[str]
    job_count: int


@dataclass
class _StoredImportPreflight:
    token: str
    specification: dict[str, Any]
    created_monotonic: float = field(default_factory=time.monotonic)


class BatchIOService:
    """Phase 10D server-side import, validation, export, and FIFO submission."""

    def __init__(
        self,
        context: ProjectContext,
        jobs: GenerationJobManager,
        model_selection: WebUIModelSelectionState,
    ) -> None:
        self.context = context
        self.jobs = jobs
        self.model_selection = model_selection
        self._tokens: dict[str, _StoredImportPreflight] = {}
        self._lock = threading.RLock()

    @staticmethod
    def detect_format(filename: str, explicit: str | None = None) -> str:
        requested = str(explicit or "").strip().lower().replace(".", "")
        aliases = {
            "native": "native", "igqueuejson": "native", "json": "native",
            "jsonl": "jsonl", "igqueuejsonl": "jsonl", "csv": "csv",
        }
        if requested:
            if requested not in aliases:
                raise ValueError(f"Unsupported import format: {explicit!r}.")
            return aliases[requested]
        name = str(filename or "").strip().lower()
        if name.endswith(".igqueue.json") or name.endswith(".json"):
            return "native"
        if name.endswith(".igqueue.jsonl") or name.endswith(".jsonl"):
            return "jsonl"
        if name.endswith(".csv"):
            return "csv"
        raise ValueError("Unable to detect queue format. Choose native JSON, JSONL, or CSV explicitly.")

    @staticmethod
    def _decode(content: bytes) -> str:
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Import file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Queue files must use UTF-8 encoding.") from exc

    @staticmethod
    def _normalize_keys(source: Mapping[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for raw_key, value in source.items():
            key = _safe_key(raw_key)
            if not key:
                continue
            normalized_key = _ALIASES.get(key, key)
            output[normalized_key] = copy.deepcopy(value)
        return output

    @staticmethod
    def _safe_provenance(value: Any) -> dict[str, Any]:
        if isinstance(value, str) and value.strip():
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        source = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            key: _sanitize_export(item)
            for key, item in source.items()
            if key in _PROVENANCE_KEYS
        }

    @staticmethod
    def _safe_remap(value: Any) -> dict[str, Any]:
        source = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            key: copy.deepcopy(item)
            for key, item in source.items()
            if key in _REMAP_FIELDS and item not in (None, "")
        }

    @staticmethod
    def _coerce_int(field_name: str, value: Any) -> int | None:
        if value in (None, "") and field_name == "seed":
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer.")
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer; received {value!r}.") from exc
        if not number.is_integer():
            raise ValueError(f"{field_name} must be an integer; received {value!r}.")
        return int(number)

    @staticmethod
    def _coerce_float(field_name: str, value: Any) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be numeric.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric; received {value!r}.") from exc

    @staticmethod
    def _coerce_bool(field_name: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"{field_name} must be true or false; received {value!r}.")

    @staticmethod
    def _coerce_json_mapping(field_name: str, value: Any) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} contains malformed JSON: {exc.msg}.") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{field_name} must contain a JSON object.")
        return copy.deepcopy(dict(parsed))

    def _coerce_request(self, source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
        normalized_source = self._normalize_keys(source)
        errors: list[str] = []
        warnings: list[str] = []
        unknown = {
            key: copy.deepcopy(value)
            for key, value in normalized_source.items()
            if key not in _REQUEST_FIELDS and key not in {"id", "provenance"}
        }
        if unknown:
            warnings.append(
                "Unknown field(s) were preserved as import metadata and excluded from generation: "
                + ", ".join(sorted(unknown))
            )
        request = {
            key: copy.deepcopy(value)
            for key, value in normalized_source.items()
            if key in _REQUEST_FIELDS
        }
        for field_name in sorted(_INT_FIELDS):
            if field_name not in request:
                continue
            if request[field_name] in (None, ""):
                request.pop(field_name, None)
                continue
            try:
                request[field_name] = self._coerce_int(field_name, request[field_name])
            except ValueError as exc:
                errors.append(str(exc))
        for field_name in sorted(_FLOAT_FIELDS):
            if field_name not in request:
                continue
            if request[field_name] in (None, ""):
                request.pop(field_name, None)
                continue
            try:
                request[field_name] = self._coerce_float(field_name, request[field_name])
            except ValueError as exc:
                errors.append(str(exc))
        for field_name in sorted(_BOOL_FIELDS):
            if field_name not in request:
                continue
            try:
                request[field_name] = self._coerce_bool(field_name, request[field_name])
            except ValueError as exc:
                errors.append(str(exc))
        for field_name in sorted(_DICT_FIELDS):
            if field_name not in request:
                continue
            try:
                request[field_name] = self._coerce_json_mapping(field_name, request[field_name])
            except ValueError as exc:
                errors.append(str(exc))
        if "positive_prompt" not in request or request.get("positive_prompt") in (None, ""):
            errors.append("positive_prompt is required.")
        request.setdefault("negative_prompt", "")
        return request, unknown, warnings, errors

    @staticmethod
    def _merge_missing(request: Mapping[str, Any], defaults: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        output = copy.deepcopy(dict(request))
        applied: list[str] = []
        for key, value in dict(defaults or {}).items():
            if key.startswith("_webui_") or key not in _REQUEST_FIELDS:
                continue
            if key not in output or _is_missing(output.get(key)):
                output[key] = copy.deepcopy(value)
                applied.append(key)
        return output, applied

    def _parse_job(
        self,
        raw: Mapping[str, Any],
        *,
        source_index: int,
        source_label: str,
        defaults_policy: str,
        current_values: Mapping[str, Any] | None,
    ) -> ImportedJob:
        normalized_raw = self._normalize_keys(raw)
        provenance = self._safe_provenance(normalized_raw.pop("provenance", {}))
        job_id = str(normalized_raw.get("id") or f"job-{source_index:04d}")
        normalized_raw.pop("id", None)
        policy_errors: list[str] = []
        defaults: Mapping[str, Any] = {}
        if defaults_policy == "current_form":
            defaults = current_values or {}
        elif defaults_policy == "saved_defaults":
            defaults = self.context.generation_defaults()
        elif defaults_policy != "file_only":
            policy_errors.append("Defaults policy must be file_only, current_form, or saved_defaults.")
        applied: list[str] = []
        if defaults:
            normalized_raw, applied = self._merge_missing(normalized_raw, defaults)
        request, unknown, warnings, errors = self._coerce_request(normalized_raw)
        errors = policy_errors + errors
        if applied:
            warnings.append("Filled missing field(s) from explicit defaults: " + ", ".join(sorted(applied)))
        return ImportedJob(
            source_index=source_index,
            source_label=source_label,
            job_id=job_id,
            raw=copy.deepcopy(dict(raw)),
            normalized=request,
            provenance=provenance,
            unknown_fields=unknown,
            warnings=warnings,
            errors=errors,
        )

    @staticmethod
    def _ensure_unique_job_ids(jobs: list[ImportedJob]) -> None:
        seen: dict[str, int] = {}
        for job in jobs:
            base = str(job.job_id or f"job-{job.source_index:04d}")
            count = seen.get(base, 0) + 1
            seen[base] = count
            if count == 1:
                job.job_id = base
                continue
            replacement = f"{base}-{count}"
            job.warnings.append(
                f"Duplicate job id {base!r} was renamed to {replacement!r} for this import."
            )
            job.job_id = replacement

    def parse_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        format_hint: str | None = None,
        defaults_policy: str = "file_only",
        current_values: Mapping[str, Any] | None = None,
    ) -> ImportBatchResult:
        text = self._decode(content)
        format_name = self.detect_format(filename, format_hint)
        if format_name == "native":
            result = self._parse_native(text, defaults_policy=defaults_policy, current_values=current_values)
        elif format_name == "jsonl":
            result = self._parse_jsonl(text, defaults_policy=defaults_policy, current_values=current_values)
        else:
            result = self._parse_csv(text, defaults_policy=defaults_policy, current_values=current_values)
        if len(result.jobs) > MAX_PARSED_JOBS:
            result.jobs = result.jobs[:MAX_PARSED_JOBS]
            result.errors.append(f"Import contains more than {MAX_PARSED_JOBS} jobs; only the first {MAX_PARSED_JOBS} were inspected.")
        self._ensure_unique_job_ids(result.jobs)
        return result

    def _parse_native(
        self,
        text: str,
        *,
        defaults_policy: str,
        current_values: Mapping[str, Any] | None,
    ) -> ImportBatchResult:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return ImportBatchResult("native", None, [], errors=[f"Malformed JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}."])
        if not isinstance(payload, Mapping):
            return ImportBatchResult("native", None, [], errors=["Native queue JSON must contain one top-level object."])
        format_value = payload.get("format")
        version_value = payload.get("version")
        errors: list[str] = []
        if format_value != NATIVE_FORMAT:
            errors.append(f"Native queue format must be {NATIVE_FORMAT!r}.")
        if version_value is None:
            errors.append("Native queue version is required.")
            version = None
        else:
            try:
                version = int(version_value)
            except (TypeError, ValueError):
                version = None
                errors.append("Native queue version must be an integer.")
        if version is not None and version > NATIVE_VERSION:
            errors.append(
                f"Queue version {version} is newer than supported version {NATIVE_VERSION}; the jobs were not reinterpreted."
            )
        jobs_value = payload.get("jobs")
        if not isinstance(jobs_value, list):
            errors.append("Native queue jobs must be an array.")
            jobs_value = []
        known_top = {"format", "version", "created_at", "source", "defaults", "jobs", "provenance"}
        unknown_top = {key: copy.deepcopy(value) for key, value in payload.items() if key not in known_top}
        warnings: list[str] = []
        if unknown_top:
            warnings.append("Unknown top-level field(s) were preserved: " + ", ".join(sorted(unknown_top)))
        if errors:
            return ImportBatchResult("native", version, [], warnings=warnings, errors=errors, top_level_unknown=unknown_top)
        file_defaults = payload.get("defaults") if isinstance(payload.get("defaults"), Mapping) else {}
        jobs: list[ImportedJob] = []
        for index, raw in enumerate(jobs_value, start=1):
            if not isinstance(raw, Mapping):
                jobs.append(ImportedJob(index, f"job {index}", f"job-{index:04d}", {}, {}, errors=["Job must be a JSON object."]))
                continue
            effective_raw = {**copy.deepcopy(dict(file_defaults)), **copy.deepcopy(dict(raw))}
            parsed = self._parse_job(
                effective_raw,
                source_index=index,
                source_label=f"job {index}",
                defaults_policy=defaults_policy,
                current_values=current_values,
            )
            if file_defaults:
                inherited = sorted(
                    key for key in file_defaults
                    if key not in raw and key in parsed.normalized
                )
                if inherited:
                    parsed.warnings.insert(0, "Applied native file default(s): " + ", ".join(inherited))
            jobs.append(parsed)
        return ImportBatchResult("native", version, jobs, warnings=warnings, errors=errors, top_level_unknown=unknown_top)

    def _parse_jsonl(
        self,
        text: str,
        *,
        defaults_policy: str,
        current_values: Mapping[str, Any] | None,
    ) -> ImportBatchResult:
        jobs: list[ImportedJob] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                jobs.append(ImportedJob(
                    source_index=line_number,
                    source_label=f"line {line_number}",
                    job_id=f"line-{line_number}",
                    raw={},
                    normalized={},
                    errors=[f"Malformed JSON on line {line_number}, column {exc.colno}: {exc.msg}."],
                ))
                continue
            if not isinstance(raw, Mapping):
                jobs.append(ImportedJob(line_number, f"line {line_number}", f"line-{line_number}", {}, {}, errors=[f"Line {line_number} must contain one JSON object."]))
                continue
            jobs.append(self._parse_job(
                raw,
                source_index=line_number,
                source_label=f"line {line_number}",
                defaults_policy=defaults_policy,
                current_values=current_values,
            ))
        return ImportBatchResult("jsonl", None, jobs)

    def _parse_csv(
        self,
        text: str,
        *,
        defaults_policy: str,
        current_values: Mapping[str, Any] | None,
    ) -> ImportBatchResult:
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            fieldnames = [str(item or "").strip() for item in (reader.fieldnames or [])]
        except csv.Error as exc:
            return ImportBatchResult("csv", None, [], errors=[f"Unable to parse CSV header: {exc}."])
        if not fieldnames:
            return ImportBatchResult("csv", None, [], errors=["CSV must contain a header row."])
        if "positive_prompt" not in fieldnames and "prompt" not in fieldnames:
            return ImportBatchResult("csv", None, [], errors=["CSV requires a positive_prompt column."])
        jobs: list[ImportedJob] = []
        try:
            for row_number, row in enumerate(reader, start=2):
                raw: dict[str, Any] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    normalized_key = _safe_key(key)
                    if normalized_key == "sampler_kwargs_json":
                        normalized_key = "sampler_kwargs"
                    elif normalized_key == "scheduler_kwargs_json":
                        normalized_key = "scheduler_kwargs"
                    elif normalized_key == "prompt_parser_kwargs_json":
                        normalized_key = "prompt_parser_kwargs"
                    elif normalized_key == "prompt_shortcut_profile_snapshot_json":
                        normalized_key = "prompt_shortcut_profile_snapshot"
                    elif normalized_key == "hires_prompt_parser_kwargs_json":
                        normalized_key = "hires_prompt_parser_kwargs"
                    elif normalized_key == "hires_shortcut_profile_snapshot_json":
                        normalized_key = "hires_shortcut_profile_snapshot"
                    elif normalized_key == "hires_dimension_plan_json":
                        normalized_key = "hires_dimension_plan"
                    elif normalized_key == "prompt_preflight_json":
                        normalized_key = "prompt_preflight"
                    elif normalized_key == "parser_kwargs_json":
                        normalized_key = "parser_kwargs"
                    elif normalized_key == "provenance_json":
                        normalized_key = "provenance"
                    elif normalized_key == "extras_json":
                        normalized_key = "extras"
                    raw[normalized_key] = value
                jobs.append(self._parse_job(
                    raw,
                    source_index=row_number,
                    source_label=f"row {row_number}",
                    defaults_policy=defaults_policy,
                    current_values=current_values,
                ))
        except csv.Error as exc:
            return ImportBatchResult("csv", None, jobs, errors=[f"CSV parsing failed near row {reader.line_num}: {exc}."])
        return ImportBatchResult("csv", None, jobs)

    def _resolve_vae(self, value: Any) -> Path:
        text = str(value or "").strip()
        if not text:
            raise ValueError("VAE path is empty.")
        candidates = [self.context.resolve_project_path(text).expanduser().resolve()]
        vae_dir = getattr(self.context, "vae_dir", None)
        tail = Path(text.lstrip("/\\"))
        if vae_dir is not None and tail.name:
            candidates.extend([(vae_dir / tail.name).resolve(), (vae_dir / tail).resolve()])
        for path in candidates:
            if path.is_file():
                return path
        raise ValueError(f"Selected VAE could not be resolved: {text}.")

    def _validate_request(
        self,
        request: Mapping[str, Any],
        *,
        remap: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        raw = copy.deepcopy(dict(request or {}))
        applied_remap = self._safe_remap(remap)
        raw.update(applied_remap)
        warnings: list[str] = []
        errors: list[str] = []
        missing_assets: list[dict[str, Any]] = []

        coerced, unknown, coercion_warnings, coercion_errors = self._coerce_request(raw)
        warnings.extend(coercion_warnings)
        errors.extend(coercion_errors)
        if unknown:
            warnings.append("Unknown fields remain informational and will not be queued.")

        model_path = coerced.get("model_path")
        if not model_path:
            errors.append("model_path is required before queueing.")
        else:
            try:
                selection = self.model_selection.authorize(model_path, source="batch_import_preflight")
                coerced["model_path"] = selection.resolved_path
            except (OSError, ValueError) as exc:
                missing_assets.append({"kind": "model", "field": "model_path", "requested": model_path, "reason": str(exc)})
                errors.append(f"Checkpoint unavailable: {exc}")

        vae_path = coerced.get("vae_path")
        if vae_path not in (None, ""):
            try:
                coerced["vae_path"] = str(self._resolve_vae(vae_path))
            except (OSError, ValueError) as exc:
                missing_assets.append({"kind": "vae", "field": "vae_path", "requested": vae_path, "reason": str(exc)})
                errors.append(f"VAE unavailable: {exc}")

        for kind, field_name in (("sampler", "sampler_name"), ("scheduler", "scheduler_name")):
            requested = coerced.get(field_name)
            if not requested or self.jobs.registry.resolve_descriptor(requested, kind=kind) is None:
                reason = f"Imported {kind} plugin is not installed: {requested!r}."
                missing_assets.append({"kind": kind, "field": field_name, "requested": requested, "reason": reason})
                errors.append(reason)

        normalized: dict[str, Any] = {}
        if not errors:
            try:
                strict_source = dict(coerced)
                strict_source["_webui_scheduler_user_selected"] = True
                strict_source["_webui_selection_version"] = 2
                strict = self.jobs.selections.normalize(
                    strict_source,
                    fallback_payload={},
                    migrate_legacy_auto_fallback=False,
                    reject_unknown=True,
                )
                normalized = self.jobs.normalize_generation_request(strict.payload)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))

        for field_name in ("width", "height", "steps", "batch_size", "batch_count"):
            value = normalized.get(field_name, coerced.get(field_name))
            if value is not None and int(value) < 1:
                errors.append(f"{field_name} must be at least 1.")
        if normalized.get("cfg_scale", coerced.get("cfg_scale")) is not None and float(normalized.get("cfg_scale", coerced.get("cfg_scale"))) < 0:
            errors.append("cfg_scale must be zero or greater.")
        return normalized, list(dict.fromkeys(warnings)), list(dict.fromkeys(errors)), missing_assets

    @staticmethod
    def _job_input(item: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("edited", "request", "normalized", "raw"):
            value = item.get(key)
            if isinstance(value, Mapping):
                return copy.deepcopy(dict(value))
        return copy.deepcopy(dict(item))

    def _safe_preflight_specification(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        source = dict(payload or {})
        jobs_value = source.get("jobs")
        if not isinstance(jobs_value, Sequence) or isinstance(jobs_value, (str, bytes)):
            raise ValueError("Import preflight requires a jobs array.")
        if len(jobs_value) > MAX_PARSED_JOBS:
            raise ValueError(f"Import preflight may inspect at most {MAX_PARSED_JOBS} jobs.")
        jobs: list[dict[str, Any]] = []
        for index, item in enumerate(jobs_value, start=1):
            if not isinstance(item, Mapping):
                jobs.append({"job_id": f"job-{index:04d}", "source_index": index, "source_label": f"job {index}", "request": {}, "parse_errors": ["Job must be an object."], "provenance": {}})
                continue
            job_id = str(item.get("job_id") or item.get("id") or f"job-{index:04d}")
            jobs.append({
                "job_id": job_id,
                "source_index": int(item.get("source_index") or index),
                "source_label": str(item.get("source_label") or f"job {index}"),
                "request": self._job_input(item),
                "parse_errors": [str(value) for value in item.get("errors") or item.get("parse_errors") or []],
                "parse_warnings": [str(value) for value in item.get("warnings") or item.get("parse_warnings") or []],
                "provenance": self._safe_provenance(item.get("provenance")),
                "unknown_fields": _sanitize_export(item.get("unknown_fields") or {}),
            })
        job_ids = [item["job_id"] for item in jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Import preflight job IDs must be unique.")
        order = [str(item) for item in source.get("order") or job_ids]
        known_ids = set(job_ids)
        if any(item not in known_ids for item in order):
            raise ValueError("Import order references an unknown job ID.")
        selected_ids = [str(item) for item in source.get("selected_job_ids") or order]
        if any(item not in known_ids for item in selected_ids):
            raise ValueError("Selected import jobs include an unknown job ID.")
        common_remap = self._safe_remap(source.get("common_remap"))
        raw_item_remaps = source.get("item_remaps") or {}
        item_remaps = {
            str(job_id): self._safe_remap(remap)
            for job_id, remap in dict(raw_item_remaps).items()
            if str(job_id) in known_ids
        } if isinstance(raw_item_remaps, Mapping) else {}
        return {
            "jobs": jobs,
            "order": order,
            "selected_job_ids": selected_ids,
            "common_remap": common_remap,
            "item_remaps": item_remaps,
        }

    def _cleanup_tokens(self) -> None:
        cutoff = time.monotonic() - _TOKEN_TTL_SECONDS
        with self._lock:
            for token in [key for key, stored in self._tokens.items() if stored.created_monotonic < cutoff]:
                self._tokens.pop(token, None)

    def _issue_token(self, specification: dict[str, Any]) -> str:
        self._cleanup_tokens()
        token = uuid.uuid4().hex
        with self._lock:
            self._tokens[token] = _StoredImportPreflight(token, copy.deepcopy(specification))
        return token

    def _consume_specification(self, token: str) -> dict[str, Any]:
        self._cleanup_tokens()
        with self._lock:
            stored = self._tokens.get(str(token or ""))
        if stored is None:
            raise ValueError("Import preflight expired or was not found. Run validation again.")
        return copy.deepcopy(stored.specification)

    def _evaluate_specification(self, specification: dict[str, Any], *, issue_token: bool) -> ImportPreflight:
        by_id = {item["job_id"]: item for item in specification["jobs"]}
        selected = set(specification["selected_job_ids"])
        results: list[dict[str, Any]] = []
        aggregate_errors: list[str] = []
        aggregate_warnings: list[str] = []
        for order_index, job_id in enumerate(specification["order"], start=1):
            item = by_id[job_id]
            remap = dict(specification["common_remap"])
            remap.update(specification["item_remaps"].get(job_id, {}))
            normalized, warnings, errors, missing_assets = self._validate_request(item["request"], remap=remap)
            errors = list(item["parse_errors"]) + errors
            warnings = list(item["parse_warnings"]) + warnings
            provenance = copy.deepcopy(item["provenance"])
            if remap:
                provenance["applied_remaps"] = _sanitize_export(remap)
            valid = not errors
            row = {
                "order": order_index,
                "job_id": job_id,
                "source_index": item["source_index"],
                "source_label": item["source_label"],
                "selected": job_id in selected,
                "valid": valid,
                "request": normalized,
                "editable_request": copy.deepcopy(item["request"]),
                "provenance": provenance,
                "unknown_fields": copy.deepcopy(item["unknown_fields"]),
                "warnings": list(dict.fromkeys(warnings)),
                "errors": list(dict.fromkeys(errors)),
                "missing_assets": missing_assets,
                "summary": {
                    "prompt": str((normalized or item["request"]).get("positive_prompt") or ""),
                    "seed": (normalized or item["request"]).get("seed"),
                    "model_path": (normalized or item["request"]).get("model_path"),
                    "sampler_name": (normalized or item["request"]).get("sampler_name"),
                    "scheduler_name": (normalized or item["request"]).get("scheduler_name"),
                },
            }
            results.append(row)
            aggregate_errors.extend(f"{item['source_label']}: {message}" for message in row["errors"])
            aggregate_warnings.extend(f"{item['source_label']}: {message}" for message in row["warnings"])
        selected_rows = [item for item in results if item["selected"]]
        valid_selected = [item for item in selected_rows if item["valid"]]
        invalid_selected = [item for item in selected_rows if not item["valid"]]
        summary = {
            "job_count": len(results),
            "selected_count": len(selected_rows),
            "valid_count": sum(1 for item in results if item["valid"]),
            "invalid_count": sum(1 for item in results if not item["valid"]),
            "valid_selected_count": len(valid_selected),
            "invalid_selected_count": len(invalid_selected),
            "immediate_submission_limit": MAX_IMMEDIATE_SUBMISSION,
            "common_remap_fields": sorted(specification["common_remap"]),
        }
        result = ImportPreflight(
            valid=bool(selected_rows) and not invalid_selected and len(selected_rows) <= MAX_IMMEDIATE_SUBMISSION,
            jobs=results,
            warnings=list(dict.fromkeys(aggregate_warnings)),
            errors=list(dict.fromkeys(aggregate_errors)),
            summary=summary,
        )
        if len(selected_rows) > MAX_IMMEDIATE_SUBMISSION:
            result.errors.append(f"At most {MAX_IMMEDIATE_SUBMISSION} jobs may be queued in one submission.")
        if issue_token:
            result.preflight_token = self._issue_token(specification)
        return result

    def validate_request(
        self,
        request: Mapping[str, Any],
        *,
        remap: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        """Validate one request through the Phase 10D canonical preflight path.

        Phase 10E deliberately reuses this boundary so variation bases and every
        expanded request receive the same coercion, asset authorization, plugin
        schema validation, and selection normalization as imported queues.
        """

        return self._validate_request(request, remap=remap)

    def validated_requests_from_preflight(self, preflight_token: str) -> list[dict[str, Any]]:
        """Return only selected, validated Phase 10D requests for downstream use.

        The raw parsed rows are never exposed as trusted variation bases. The
        stored import specification is re-evaluated and all selected rows must
        still be valid at the time this method is called.
        """

        specification = self._consume_specification(preflight_token)
        result = self._evaluate_specification(specification, issue_token=False)
        selected = [item for item in result.jobs if item["selected"]]
        invalid = [item for item in selected if not item["valid"]]
        if not selected:
            raise ValueError("The import preflight has no selected jobs.")
        if invalid:
            labels = ", ".join(str(item.get("source_label") or item.get("job_id")) for item in invalid)
            raise ValueError(
                "Imported variation bases must come from valid Phase 10D preflight rows. "
                f"Correct or deselect: {labels}."
            )
        return [
            {
                "job_id": item["job_id"],
                "source_label": item["source_label"],
                "request": copy.deepcopy(item["request"]),
                "provenance": copy.deepcopy(item.get("provenance") or {}),
                "warnings": list(item.get("warnings") or []),
            }
            for item in selected
        ]

    def preflight(self, payload: Mapping[str, Any] | None) -> ImportPreflight:
        return self._evaluate_specification(self._safe_preflight_specification(payload), issue_token=True)

    async def submit(
        self,
        preflight_token: str,
        *,
        queue_valid_only: bool = False,
    ) -> tuple[ImportPreflight, list[Any], list[dict[str, Any]]]:
        specification = self._consume_specification(preflight_token)
        result = self._evaluate_specification(specification, issue_token=False)
        if not result.valid and not queue_valid_only:
            raise ValueError("Imported jobs contain blocking errors. Choose Queue Valid Jobs Only or correct the invalid rows.")
        selected = [item for item in result.jobs if item["selected"]]
        valid_selected = [item for item in selected if item["valid"]]
        if len(valid_selected) > MAX_IMMEDIATE_SUBMISSION:
            raise ValueError(f"At most {MAX_IMMEDIATE_SUBMISSION} jobs may be queued in one submission.")
        submitted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for item in selected:
            if not item["valid"]:
                rejected.append({"job_id": item["job_id"], "source_label": item["source_label"], "errors": list(item["errors"])})
                continue
            request = copy.deepcopy(item["request"])
            try:
                selection = self.model_selection.authorize(request.get("model_path"), source="batch_import_submission")
                request["model_path"] = selection.resolved_path
                job = await self.jobs.submit(request, model_selection=selection.to_dict())
                submitted.append(job)
            except (OSError, TypeError, ValueError) as exc:
                rejected.append({"job_id": item["job_id"], "source_label": item["source_label"], "errors": [str(exc)]})
        with self._lock:
            self._tokens.pop(preflight_token, None)
        return result, submitted, rejected

    @staticmethod
    def _extract_export_job(item: Mapping[str, Any], index: int) -> dict[str, Any]:
        wrapper = dict(item or {})
        request_value = wrapper.get("request")
        if isinstance(request_value, Mapping):
            request = dict(request_value)
        else:
            request = {
                key: value for key, value in wrapper.items()
                if key not in {"provenance", "job_id", "id", "source_index", "source_label", "valid", "warnings", "errors", "missing_assets", "summary"}
            }
        clean = _sanitize_export(request)
        clean["id"] = str(wrapper.get("job_id") or wrapper.get("id") or f"job-{index:04d}")
        provenance = BatchIOService._safe_provenance(wrapper.get("provenance"))
        if provenance:
            clean["provenance"] = provenance
        return clean

    def export(self, payload: Mapping[str, Any] | None) -> ExportResult:
        source = dict(payload or {})
        format_name = str(source.get("format") or "native").strip().lower()
        if format_name in {"igqueue", "json", "igqueue_json"}:
            format_name = "native"
        if format_name not in {"native", "jsonl", "csv"}:
            raise ValueError("Export format must be native, jsonl, or csv.")
        jobs_value = source.get("jobs")
        if not isinstance(jobs_value, Sequence) or isinstance(jobs_value, (str, bytes)) or not jobs_value:
            raise ValueError("Export requires at least one job.")
        if len(jobs_value) > MAX_PARSED_JOBS:
            raise ValueError(f"Export may contain at most {MAX_PARSED_JOBS} jobs.")
        jobs = [self._extract_export_job(item, index) for index, item in enumerate(jobs_value, start=1) if isinstance(item, Mapping)]
        warnings: list[str] = []
        stem = str(source.get("filename_stem") or "image_gen_queue").strip() or "image_gen_queue"
        stem = "".join(character for character in stem if character.isalnum() or character in {"-", "_"})[:80] or "image_gen_queue"
        if format_name == "native":
            document = {
                "format": NATIVE_FORMAT,
                "version": NATIVE_VERSION,
                "created_at": _utc_now(),
                "source": str(source.get("source") or "IMAGE_GEN WebUI"),
                "defaults": _sanitize_export(source.get("defaults") or {}),
                "jobs": jobs,
            }
            content = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            return ExportResult(content, "application/json", f"{stem}.igqueue.json", warnings, len(jobs))
        if format_name == "jsonl":
            content = ("\n".join(json.dumps(job, ensure_ascii=False, separators=(",", ":")) for job in jobs) + "\n").encode("utf-8")
            return ExportResult(content, "application/x-ndjson", f"{stem}.jsonl", warnings, len(jobs))

        columns = [
            "id", "positive_prompt", "negative_prompt", "seed", "width", "height", "steps",
            "cfg_scale", "batch_size", "batch_count", "sampler_name", "scheduler_name",
            "model_path", "vae_path", "sampler_kwargs_json", "scheduler_kwargs_json",
            "prompt_parser_name", "prompt_parser_kwargs_json", "prompt_shortcut_profile_name",
            "prompt_shortcut_profile_snapshot_json", "prompt_parser_preset_name",
            "base_prompt_parser_name", "base_shortcut_profile_name",
            "hires_prompt_parser_mode", "hires_prompt_parser_name", "hires_prompt_parser_kwargs_json",
            "hires_shortcut_profile_mode", "hires_shortcut_profile_name", "hires_shortcut_profile_snapshot_json",
            "hires_positive_prompt", "hires_negative_prompt", "hires_size_mode", "hires_scale", "hires_width", "hires_height", "hires_axis_scale_width", "hires_axis_scale_height", "hires_uniform_scale", "hires_aspect_ratio_changed", "hires_dimension_plan_version", "hires_dimension_plan_json", "hires_upscaler_id", "hires_expected_upscaler_sha256", "hires_expected_native_scale", "hires_final_size_correction_filter", "hires_aspect_policy", "hires_padding_mode", "hires_recorded_target_correction_json", "hires_correction_fingerprint_enabled", "hires_recorded_correction_fingerprint_json", "prompt_preflight_json",
            "parser_kwargs_json", "extras_json", "provenance_json",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        complex_present = False
        for job in jobs:
            row = {key: job.get(key, "") for key in columns}
            for source_key, column in (
                ("sampler_kwargs", "sampler_kwargs_json"),
                ("scheduler_kwargs", "scheduler_kwargs_json"),
                ("prompt_parser_kwargs", "prompt_parser_kwargs_json"),
                ("prompt_shortcut_profile_snapshot", "prompt_shortcut_profile_snapshot_json"),
                ("hires_prompt_parser_kwargs", "hires_prompt_parser_kwargs_json"),
                ("hires_shortcut_profile_snapshot", "hires_shortcut_profile_snapshot_json"),
                ("hires_dimension_plan", "hires_dimension_plan_json"),
                ("hires_recorded_target_correction", "hires_recorded_target_correction_json"),
                ("hires_recorded_correction_fingerprint", "hires_recorded_correction_fingerprint_json"),
                ("prompt_preflight", "prompt_preflight_json"),
                ("parser_kwargs", "parser_kwargs_json"),
                ("extras", "extras_json"),
                ("provenance", "provenance_json"),
            ):
                value = job.get(source_key)
                if value not in (None, {}, [], ""):
                    complex_present = True
                    row[column] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)
        if complex_present:
            warnings.append("CSV flattens advanced settings, extras, and provenance into JSON text columns; native .igqueue.json remains the highest-fidelity format.")
        return ExportResult(stream.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{stem}.csv", warnings, len(jobs))


__all__ = [
    "BatchIOService", "ImportedJob", "ImportBatchResult", "ImportPreflight", "ExportResult",
    "NATIVE_FORMAT", "NATIVE_VERSION", "MAX_UPLOAD_BYTES", "MAX_PARSED_JOBS",
    "MAX_IMMEDIATE_SUBMISSION",
]
