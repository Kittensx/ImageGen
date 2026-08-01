from __future__ import annotations

import copy
import hashlib
import itertools
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from image_gen.webui.batch_io import BatchIOService, MAX_IMMEDIATE_SUBMISSION
from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.model_selection import WebUIModelSelectionState
from modules.project_context import ProjectContext

_TOKEN_TTL_SECONDS = 15 * 60
_DEFAULT_LIMIT = MAX_IMMEDIATE_SUBMISSION
_HARD_LIMIT = 1000
_WARNING_THRESHOLD = 50

_CORE_FIELDS = {
    "seed",
    "model_path",
    "vae_path",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "sampler_name",
    "scheduler_name",
    "positive_prompt",
    "negative_prompt",
    "batch_size",
    "batch_count",
    "cfg_rescale",
    "clip_skip",
}
_INTERNAL_BLOCKLIST = {
    "output_dir",
    "output_prefix",
    "save_images",
    "extras.variation_matrix",
}
_ADVANCED_ROOTS = {
    "sampler_kwargs": "sampler",
    "scheduler_kwargs": "scheduler",
}
_MODES = {"cartesian", "paired", "one_at_a_time"}
_ZIP_POLICIES = {"reject", "repeat_last", "cycle"}
_BASE_MODES = {"apply_to_each", "base_dimension"}


@dataclass
class VariationDimension:
    field: str
    values: list[Any]
    source: str = "manual"
    label: str = ""


@dataclass
class VariationPlan:
    mode: str
    base_requests: list[dict[str, Any]]
    dimensions: list[VariationDimension]
    base_lineage: list[dict[str, Any]] = field(default_factory=list)
    zip_length_policy: str = "reject"
    base_mode: str = "apply_to_each"
    deduplicate: bool = True
    recipe_name: str = "Variation Matrix"
    job_limit: int = _DEFAULT_LIMIT
    confirm_large_plan: bool = False


@dataclass
class VariationPreflight:
    valid: bool
    base_count: int
    combination_count: int
    total_job_count: int
    jobs: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    removed_duplicate_count: int = 0
    preflight_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _StoredVariationPreflight:
    specification: dict[str, Any]
    created_monotonic: float = field(default_factory=time.monotonic)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_fingerprint(request: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(request))
    payload.pop("_webui_selection_version", None)
    payload.pop("_webui_scheduler_user_selected", None)
    payload.pop("variation_matrix", None)
    # Prompt preflight is a derived validation artifact and can contain timing diagnostics.
    # Parser/profile fields and snapshots remain in the generation fingerprint.
    payload.pop("prompt_preflight", None)
    payload.pop("prompt_route_plan", None)
    payload.pop("hires_prompt_route_plan", None)
    extras = dict(payload.get("extras") or {})
    extras.pop("variation_matrix", None)
    if extras:
        payload["extras"] = extras
    else:
        payload.pop("extras", None)
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class VariationMatrixService:
    """Phase 10E deterministic expansion, validation, export, and FIFO submission."""

    def __init__(
        self,
        context: ProjectContext,
        jobs: GenerationJobManager,
        model_selection: WebUIModelSelectionState,
        batch_io: BatchIOService,
    ) -> None:
        self.context = context
        self.jobs = jobs
        self.model_selection = model_selection
        self.batch_io = batch_io
        self._tokens: dict[str, _StoredVariationPreflight] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _range_values(source: Mapping[str, Any]) -> list[Any]:
        try:
            start = Decimal(str(source.get("start")))
            stop = Decimal(str(source.get("stop")))
            step = Decimal(str(source.get("step")))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("Numeric ranges require valid start, stop, and step values.") from exc
        if step == 0:
            raise ValueError("Numeric range step cannot be zero.")
        if (stop - start) * step < 0:
            raise ValueError("Numeric range step moves away from the stop value.")
        values: list[Any] = []
        current = start
        compare = (lambda value: value <= stop) if step > 0 else (lambda value: value >= stop)
        while compare(current):
            if len(values) >= _HARD_LIMIT:
                raise ValueError(f"A single numeric range may contain at most {_HARD_LIMIT} values.")
            normalized = current.normalize()
            values.append(int(normalized) if normalized == normalized.to_integral() else float(normalized))
            current += step
        if not values:
            raise ValueError("Numeric range produced no values.")
        return values

    @classmethod
    def _dimension(cls, value: Mapping[str, Any], index: int) -> VariationDimension:
        field_name = str(value.get("field") or "").strip()
        if not field_name:
            raise ValueError(f"Variation dimension {index} requires a field.")
        if field_name.startswith("_") or field_name.startswith("_webui_"):
            raise ValueError(f"Internal field {field_name!r} cannot be varied.")
        if field_name in _INTERNAL_BLOCKLIST:
            raise ValueError(f"Unsafe field {field_name!r} cannot be varied.")
        if field_name not in _CORE_FIELDS and not any(
            field_name.startswith(root + ".") for root in _ADVANCED_ROOTS
        ):
            raise ValueError(f"Unsupported variation field: {field_name!r}.")
        if field_name.count(".") > 1:
            raise ValueError("Advanced variation paths must identify one schema property only.")
        source_kind = str(value.get("source") or "manual").strip().lower()
        if source_kind == "range" or isinstance(value.get("range"), Mapping):
            range_source = value.get("range") if isinstance(value.get("range"), Mapping) else value
            values = cls._range_values(dict(range_source))
            source_kind = "range"
        else:
            raw_values = value.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                raise ValueError(f"Variation dimension {field_name!r} requires a values array.")
            values = [copy.deepcopy(item) for item in raw_values]
        if not values:
            raise ValueError(f"Variation dimension {field_name!r} has no values.")
        return VariationDimension(
            field=field_name,
            values=values,
            source=source_kind,
            label=str(value.get("label") or field_name),
        )

    def _validated_direct_bases(
        self, values: Sequence[Any], lineage_values: Sequence[Any] | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bases: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        raw_lineage = list(lineage_values or [])
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise ValueError(f"Base request {index + 1} must be an object.")
            normalized, warnings, errors, _ = self.batch_io.validate_request(item)
            if errors:
                raise ValueError(f"Base request {index + 1} is invalid: " + "; ".join(errors))
            bases.append(normalized)
            supplied = raw_lineage[index] if index < len(raw_lineage) and isinstance(raw_lineage[index], Mapping) else {}
            lineage.append({
                "source": str(supplied.get("source") or "direct"),
                "source_id": supplied.get("source_id"),
                "source_label": str(supplied.get("source_label") or f"Base request {index + 1}"),
                "warnings": list(warnings),
            })
        return bases, lineage

    def _safe_plan(self, payload: Mapping[str, Any] | None) -> VariationPlan:
        source = dict(payload or {})
        import_token = str(source.get("import_preflight_token") or "").strip()
        if import_token:
            validated = self.batch_io.validated_requests_from_preflight(import_token)
            bases = [copy.deepcopy(item["request"]) for item in validated]
            lineage = [
                {
                    "source": "phase10d_import_preflight",
                    "source_id": item.get("job_id"),
                    "source_label": item.get("source_label"),
                    "provenance": copy.deepcopy(item.get("provenance") or {}),
                }
                for item in validated
            ]
        else:
            raw_bases = source.get("base_requests")
            if not isinstance(raw_bases, Sequence) or isinstance(raw_bases, (str, bytes)):
                raise ValueError("Variation preflight requires base_requests or an import_preflight_token.")
            bases, lineage = self._validated_direct_bases(raw_bases, source.get("base_lineage"))
        if not bases:
            raise ValueError("Variation preflight requires at least one validated base request.")

        raw_dimensions = source.get("dimensions")
        if not isinstance(raw_dimensions, Sequence) or isinstance(raw_dimensions, (str, bytes)):
            raise ValueError("Variation preflight requires a dimensions array.")
        dimensions = [self._dimension(item, index) for index, item in enumerate(raw_dimensions, start=1) if isinstance(item, Mapping)]
        if len(dimensions) != len(raw_dimensions):
            raise ValueError("Every variation dimension must be an object.")
        fields = [item.field for item in dimensions]
        if len(set(fields)) != len(fields):
            raise ValueError("Each field may appear only once in a variation plan.")

        mode = str(source.get("mode") or "cartesian").strip().lower()
        if mode not in _MODES:
            raise ValueError("Variation mode must be cartesian, paired, or one_at_a_time.")
        zip_policy = str(source.get("zip_length_policy") or "reject").strip().lower()
        if zip_policy not in _ZIP_POLICIES:
            raise ValueError("zip_length_policy must be reject, repeat_last, or cycle.")
        base_mode = str(source.get("base_mode") or "apply_to_each").strip().lower()
        if base_mode not in _BASE_MODES:
            raise ValueError("base_mode must be apply_to_each or base_dimension.")
        try:
            job_limit = int(source.get("job_limit") or _DEFAULT_LIMIT)
        except (TypeError, ValueError) as exc:
            raise ValueError("job_limit must be an integer.") from exc
        if job_limit < 1 or job_limit > _HARD_LIMIT:
            raise ValueError(f"job_limit must be between 1 and {_HARD_LIMIT}.")
        return VariationPlan(
            mode=mode,
            base_requests=bases,
            base_lineage=lineage,
            dimensions=dimensions,
            zip_length_policy=zip_policy,
            base_mode=base_mode,
            deduplicate=bool(source.get("deduplicate", True)),
            recipe_name=str(source.get("recipe_name") or "Variation Matrix").strip()[:120] or "Variation Matrix",
            job_limit=job_limit,
            confirm_large_plan=bool(source.get("confirm_large_plan", False)),
        )

    @staticmethod
    def _paired_value(values: list[Any], index: int, length: int, policy: str) -> Any:
        if len(values) == length:
            return copy.deepcopy(values[index])
        if policy == "repeat_last":
            return copy.deepcopy(values[min(index, len(values) - 1)])
        if policy == "cycle":
            return copy.deepcopy(values[index % len(values)])
        raise ValueError("Paired variation lists must have equal lengths unless repeat_last or cycle is selected.")

    @classmethod
    def _combinations(cls, plan: VariationPlan) -> list[dict[str, Any]]:
        if not plan.dimensions:
            return [{}]
        if plan.mode == "cartesian":
            return [
                {dimension.field: copy.deepcopy(value) for dimension, value in zip(plan.dimensions, items)}
                for items in itertools.product(*(dimension.values for dimension in plan.dimensions))
            ]
        if plan.mode == "paired":
            lengths = [len(item.values) for item in plan.dimensions]
            if plan.zip_length_policy == "reject" and len(set(lengths)) > 1:
                raise ValueError("Paired variation lists have unequal lengths. Choose repeat_last or cycle explicitly.")
            length = max(lengths)
            return [
                {
                    dimension.field: cls._paired_value(dimension.values, index, length, plan.zip_length_policy)
                    for dimension in plan.dimensions
                }
                for index in range(length)
            ]
        output: list[dict[str, Any]] = [{}]
        for dimension in plan.dimensions:
            output.extend({dimension.field: copy.deepcopy(value)} for value in dimension.values)
        return output

    @staticmethod
    def _schema_value(value: Any, schema: Mapping[str, Any], field_name: str) -> Any:
        enum = list(schema.get("enum") or [])
        expected = str(schema.get("type") or "").lower()
        if expected == "integer":
            if isinstance(value, bool):
                raise ValueError(f"{field_name} must be an integer.")
            number = Decimal(str(value))
            if number != number.to_integral_value():
                raise ValueError(f"{field_name} must be an integer.")
            value = int(number)
        elif expected == "number":
            if isinstance(value, bool):
                raise ValueError(f"{field_name} must be numeric.")
            value = float(value)
        elif expected == "boolean":
            if not isinstance(value, bool):
                text = str(value).strip().lower()
                if text in {"true", "1", "yes", "on"}:
                    value = True
                elif text in {"false", "0", "no", "off"}:
                    value = False
                else:
                    raise ValueError(f"{field_name} must be true or false.")
        elif expected == "string":
            value = str(value)
        if enum and value not in enum:
            raise ValueError(f"{field_name} must be one of: " + ", ".join(map(str, enum)))
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and isinstance(value, (int, float)) and value < minimum:
            raise ValueError(f"{field_name} must be at least {minimum}.")
        if maximum is not None and isinstance(value, (int, float)) and value > maximum:
            raise ValueError(f"{field_name} must be no greater than {maximum}.")
        return value

    def _apply_override(self, request: dict[str, Any], field_name: str, value: Any) -> None:
        if field_name in _CORE_FIELDS:
            request[field_name] = copy.deepcopy(value)
            return
        root, child = field_name.split(".", 1)
        kind = _ADVANCED_ROOTS[root]
        plugin_field = "sampler_name" if kind == "sampler" else "scheduler_name"
        descriptor = self.jobs.registry.resolve_descriptor(request.get(plugin_field), kind=kind)
        if descriptor is None:
            raise ValueError(f"Cannot vary {field_name}: selected {kind} plugin is unavailable.")
        properties = dict(descriptor.config_schema.get("properties") or {})
        if child not in properties:
            raise ValueError(
                f"Cannot vary {field_name}: {child!r} is not declared by {descriptor.name!r}."
            )
        nested = dict(request.get(root) or {})
        nested[child] = self._schema_value(value, dict(properties[child] or {}), field_name)
        request[root] = nested

    def _expanded_candidates(self, plan: VariationPlan) -> tuple[list[dict[str, Any]], int]:
        combinations = self._combinations(plan)
        candidates: list[dict[str, Any]] = []
        pairs = (
            ((base_index, base, lineage, combo_index, combo)
             for base_index, (base, lineage) in enumerate(zip(plan.base_requests, plan.base_lineage))
             for combo_index, combo in enumerate(combinations))
            if plan.base_mode == "apply_to_each"
            else
            ((base_index, base, lineage, combo_index, combo)
             for combo_index, combo in enumerate(combinations)
             for base_index, (base, lineage) in enumerate(zip(plan.base_requests, plan.base_lineage)))
        )
        variation_index = 0
        for base_index, base, lineage, combo_index, combo in pairs:
            request = copy.deepcopy(base)
            errors: list[str] = []
            for field_name, value in combo.items():
                try:
                    self._apply_override(request, field_name, value)
                except (InvalidOperation, TypeError, ValueError) as exc:
                    errors.append(str(exc))
            variation_index += 1
            candidates.append({
                "base_index": base_index,
                "base_lineage": copy.deepcopy(lineage),
                "combination_index": combo_index,
                "variation_index": variation_index,
                "values": copy.deepcopy(combo),
                "request": request,
                "application_errors": errors,
            })
        return candidates, len(combinations)

    def _evaluate(self, plan: VariationPlan, *, issue_token: bool) -> VariationPreflight:
        candidates, combination_count = self._expanded_candidates(plan)
        errors: list[str] = []
        warnings: list[str] = []
        if len(candidates) > plan.job_limit:
            errors.append(
                f"Variation plan creates {len(candidates)} jobs, above the configured limit of {plan.job_limit}."
            )
        if len(candidates) > _WARNING_THRESHOLD and not plan.confirm_large_plan:
            errors.append(
                f"Variation plan creates {len(candidates)} jobs. Confirm the large plan before continuing."
            )
        elif len(candidates) > _WARNING_THRESHOLD:
            warnings.append(f"Large variation plan confirmed: {len(candidates)} jobs.")

        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        removed_duplicates = 0
        for item in candidates:
            request = copy.deepcopy(item["request"])
            lineage = copy.deepcopy(item["base_lineage"])
            variation_metadata = {
                "recipe_name": plan.recipe_name,
                "base_index": item["base_index"],
                "base_source": lineage.get("source"),
                "base_source_id": lineage.get("source_id"),
                "base_source_label": lineage.get("source_label"),
                "combination_index": item["combination_index"],
                "variation_index": item["variation_index"],
                "varied_fields": list(item["values"]),
                "values": _json_safe(item["values"]),
            }
            row_errors = list(item["application_errors"])
            row_warnings: list[str] = []
            normalized: dict[str, Any] = {}
            missing_assets: list[dict[str, Any]] = []
            if not row_errors:
                normalized, row_warnings, validation_errors, missing_assets = self.batch_io.validate_request(request)
                row_errors.extend(validation_errors)
            if normalized:
                normalized["variation_matrix"] = variation_metadata
            fingerprint = _canonical_fingerprint(normalized or request)
            duplicate = fingerprint in seen
            if duplicate and plan.deduplicate:
                removed_duplicates += 1
                continue
            seen.add(fingerprint)
            source_label = str(lineage.get("source_label") or f"Base {item['base_index'] + 1}")
            jobs.append({
                "job_index": len(jobs) + 1,
                "base_index": item["base_index"],
                "base_source": lineage.get("source"),
                "base_source_id": lineage.get("source_id"),
                "base_source_label": source_label,
                "combination_index": item["combination_index"],
                "variation_index": item["variation_index"],
                "varied_fields": list(item["values"]),
                "variation_values": _json_safe(item["values"]),
                "request": normalized,
                "valid": not row_errors,
                "errors": list(dict.fromkeys(row_errors)),
                "warnings": list(dict.fromkeys(row_warnings)),
                "missing_assets": missing_assets,
                "duplicate": duplicate,
                "summary": {
                    "prompt": str((normalized or request).get("positive_prompt") or ""),
                    "seed": (normalized or request).get("seed"),
                    "model_path": (normalized or request).get("model_path"),
                    "sampler_name": (normalized or request).get("sampler_name"),
                    "scheduler_name": (normalized or request).get("scheduler_name"),
                    "width": (normalized or request).get("width"),
                    "height": (normalized or request).get("height"),
                    "steps": (normalized or request).get("steps"),
                    "cfg_scale": (normalized or request).get("cfg_scale"),
                },
            })

        invalid = [item for item in jobs if not item["valid"]]
        errors.extend(
            f"Variation job {item['job_index']}: {message}"
            for item in invalid
            for message in item["errors"]
        )
        if removed_duplicates:
            warnings.append(f"Removed {removed_duplicates} exact duplicate request(s).")
        result = VariationPreflight(
            valid=bool(jobs) and not errors and not invalid,
            base_count=len(plan.base_requests),
            combination_count=combination_count,
            total_job_count=len(jobs),
            jobs=jobs,
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            removed_duplicate_count=removed_duplicates,
        )
        if issue_token:
            result.preflight_token = self._issue_token(asdict(plan))
        return result

    def _cleanup_tokens(self) -> None:
        cutoff = time.monotonic() - _TOKEN_TTL_SECONDS
        with self._lock:
            for token in [key for key, value in self._tokens.items() if value.created_monotonic < cutoff]:
                self._tokens.pop(token, None)

    def _issue_token(self, specification: dict[str, Any]) -> str:
        self._cleanup_tokens()
        token = uuid.uuid4().hex
        with self._lock:
            self._tokens[token] = _StoredVariationPreflight(copy.deepcopy(specification))
        return token

    def _stored_plan(self, token: str) -> VariationPlan:
        self._cleanup_tokens()
        with self._lock:
            stored = self._tokens.get(str(token or ""))
        if stored is None:
            raise ValueError("Variation preflight expired or was not found. Run preflight again.")
        source = copy.deepcopy(stored.specification)
        source["dimensions"] = [VariationDimension(**item) for item in source.get("dimensions") or []]
        return VariationPlan(**source)

    def preflight(self, payload: Mapping[str, Any] | None) -> VariationPreflight:
        return self._evaluate(self._safe_plan(payload), issue_token=True)

    def preflight_from_token(self, preflight_token: str) -> VariationPreflight:
        return self._evaluate(self._stored_plan(preflight_token), issue_token=False)

    async def submit(self, preflight_token: str) -> tuple[VariationPreflight, list[Any], list[dict[str, Any]]]:
        plan = self._stored_plan(preflight_token)
        result = self._evaluate(plan, issue_token=False)
        if not result.valid:
            raise ValueError("Variation plan no longer passes preflight. Review the validation results.")
        submitted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for item in result.jobs:
            request = copy.deepcopy(item["request"])
            try:
                selection = self.model_selection.authorize(
                    request.get("model_path"), source="variation_matrix_submission"
                )
                request["model_path"] = selection.resolved_path
                job = await self.jobs.submit(request, model_selection=selection.to_dict())
                submitted.append(job)
            except (OSError, TypeError, ValueError) as exc:
                rejected.append({
                    "job_index": item["job_index"],
                    "base_source_label": item["base_source_label"],
                    "errors": [str(exc)],
                })
        with self._lock:
            self._tokens.pop(preflight_token, None)
        return result, submitted, rejected
