from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import (
    KES_ALLOWED_DECAY_PATTERNS,
    KES_ALLOWED_STABILIZATION_METHODS,
    KES_ALLOWEDS,
    KES_COMPATIBILITY_PRESETS,
    KES_RUNTIME_DEFAULTS,
    validate_simple_kes_settings,
)

_INTERNAL_KEYS = {"_policy", "_validation_warnings", "device"}
_LINKED_KEYS = {"steps", "device"}
_LEGACY_CONTROL_KEYS = {"pipeline_mode", "config_path", "preset_name", "blend_weights"}
_INTEGER_FIELDS = {
    key
    for key, value in KES_RUNTIME_DEFAULTS.items()
    if isinstance(value, int) and not isinstance(value, bool)
}
_BOOLEAN_FIELDS = {
    key for key, value in KES_RUNTIME_DEFAULTS.items() if isinstance(value, bool)
}
_FLOAT_FIELDS = {
    key
    for key, value in KES_RUNTIME_DEFAULTS.items()
    if isinstance(value, (int, float))
    and not isinstance(value, bool)
    and key not in _INTEGER_FIELDS
}
_ENUM_FIELDS: dict[str, set[str]] = {
    **{key: {str(item) for item in values} for key, values in KES_ALLOWEDS.items()},
    "decay_pattern": {str(item) for item in KES_ALLOWED_DECAY_PATTERNS},
    "rho_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "sigma_min_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "sigma_max_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "start_blend_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "end_blend_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "sharpness_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "smooth_blend_factor_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "initial_step_size_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "final_step_size_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "step_size_factor_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "initial_noise_scale_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "final_noise_scale_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "noise_scale_factor_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
    "early_stopping_threshold_randomization_type": {"symmetric", "asymmetric", "logarithmic", "exponential", "log", "exp"},
}
_COMPATIBILITY_KEYS = {
    "pipeline_mode",
    "truncate_to_requested_steps",
    "warn_on_feature_downgrade",
    "requested_by_sampler",
    "scheduler_family",
    "negotiated_pipeline_mode",
    "step_expansion_clamped",
    "tail_metadata_clamped",
    "warnings",
}
_BLEND_METHOD_KEYS = {"weight", "decay_pattern", "decay_mode", "tail_steps"}
_BLEND_METHOD_SHARED_KEYS = ("decay_pattern", "decay_mode", "tail_steps")


class SchedulerSettingsValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        summary = "; ".join(
            f"{item.get('path', 'scheduler_kwargs')}: {item.get('message', 'invalid value')}"
            for item in errors
        )
        super().__init__(summary or "Scheduler settings validation failed.")


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SchedulerSettingsValidationError([
            {
                "path": "scheduler_kwargs",
                "message": f"settings cannot be serialized safely: {exc}",
            }
        ]) from exc


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_scheduler_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_integer(path: str, value: Any, errors: list[dict[str, Any]]) -> int | None:
    if isinstance(value, bool):
        errors.append({"path": path, "message": "must be an integer, not a boolean."})
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        errors.append({"path": path, "message": "must be a whole number."})
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        errors.append({"path": path, "message": "must be a whole number."})
        return None
    if not math.isfinite(number) or not number.is_integer():
        errors.append({"path": path, "message": "must be a whole number."})
        return None
    return int(number)


def _coerce_float(path: str, value: Any, errors: list[dict[str, Any]]) -> float | None:
    if isinstance(value, bool):
        errors.append({"path": path, "message": "must be numeric, not a boolean."})
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append({"path": path, "message": "must be numeric."})
        return None
    if not math.isfinite(number):
        errors.append({"path": path, "message": "must be a finite number."})
        return None
    return number


def _coerce_bool(path: str, value: Any, errors: list[dict[str, Any]]) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    errors.append({"path": path, "message": "must be true or false."})
    return None


def _select_legacy_shared_setting(values: list[tuple[str, Any]], *, preferred_methods: list[str]) -> Any:
    if not values:
        return None
    by_method = {method: value for method, value in values}
    for method_name in preferred_methods:
        if method_name in by_method:
            return by_method[method_name]
    counts: dict[str, tuple[int, Any]] = {}
    for _method, value in values:
        token = json.dumps(value, ensure_ascii=False, sort_keys=True)
        current_count, _current_value = counts.get(token, (0, value))
        counts[token] = (current_count + 1, value)
    return max(counts.values(), key=lambda item: item[0])[1]


def _validate_blend_methods(
    value: Any,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(value, Mapping):
        errors.append({"path": "scheduler_kwargs.blend_methods", "message": "must be an object."})
        return None, {}
    output: dict[str, Any] = {}
    legacy_shared_candidates: dict[str, list[tuple[str, Any]]] = {
        key: [] for key in _BLEND_METHOD_SHARED_KEYS
    }
    preferred_methods: list[str] = []
    for method_name, raw_config in value.items():
        method_path = f"scheduler_kwargs.blend_methods.{method_name}"
        if not isinstance(method_name, str) or not method_name.strip():
            errors.append({"path": method_path, "message": "method names must be nonempty strings."})
            continue
        if not isinstance(raw_config, Mapping):
            errors.append({"path": method_path, "message": "must be an object."})
            continue
        unknown = sorted(set(raw_config) - _BLEND_METHOD_KEYS)
        for key in unknown:
            errors.append({"path": f"{method_path}.{key}", "message": "is not a recognized blend-method setting."})
        normalized_name = method_name.strip()
        clean: dict[str, Any] = {}
        weight = _coerce_float(
            f"{method_path}.weight",
            raw_config.get("weight", 1.0),
            errors,
        )
        if weight is not None and weight > 0:
            clean["weight"] = weight
            preferred_methods.append(normalized_name)
        if "tail_steps" in raw_config:
            tail_steps = _coerce_integer(f"{method_path}.tail_steps", raw_config["tail_steps"], errors)
            if tail_steps is not None:
                legacy_shared_candidates["tail_steps"].append((normalized_name, tail_steps))
        if "decay_pattern" in raw_config:
            token = str(raw_config["decay_pattern"]).strip()
            if token not in KES_ALLOWED_DECAY_PATTERNS:
                errors.append({"path": f"{method_path}.decay_pattern", "message": f"unknown option {token!r}."})
            else:
                legacy_shared_candidates["decay_pattern"].append((normalized_name, token))
        if "decay_mode" in raw_config:
            token = str(raw_config["decay_mode"]).strip()
            if token not in KES_ALLOWEDS["decay_mode"]:
                errors.append({"path": f"{method_path}.decay_mode", "message": f"unknown option {token!r}."})
            else:
                legacy_shared_candidates["decay_mode"].append((normalized_name, token))
        if clean.get("weight", 0.0) > 0:
            output[normalized_name] = {key: item for key, item in clean.items() if item is not None}

    if value and not output:
        errors.append({
            "path": "scheduler_kwargs.blend_methods",
            "message": "must contain at least one method with weight greater than zero.",
        })

    promoted = {
        key: _select_legacy_shared_setting(values, preferred_methods=preferred_methods)
        for key, values in legacy_shared_candidates.items()
    }
    return output, {key: value for key, value in promoted.items() if value is not None}


def _validate_compatibility(value: Any, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append({"path": "scheduler_kwargs.compatibility", "message": "must be an object."})
        return None
    output = dict(value)
    for key in sorted(set(output) - _COMPATIBILITY_KEYS):
        errors.append({"path": f"scheduler_kwargs.compatibility.{key}", "message": "is not a recognized compatibility setting."})
    if "pipeline_mode" in output:
        mode = str(output["pipeline_mode"]).strip().lower()
        if mode not in KES_COMPATIBILITY_PRESETS:
            errors.append({"path": "scheduler_kwargs.compatibility.pipeline_mode", "message": f"unknown option {mode!r}."})
        output["pipeline_mode"] = mode
    for key in ("truncate_to_requested_steps", "warn_on_feature_downgrade"):
        if key in output:
            output[key] = _coerce_bool(f"scheduler_kwargs.compatibility.{key}", output[key], errors)
    return {key: item for key, item in output.items() if item is not None}


def _validate_requested_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    # Preserve extension/backend-only values for replay compatibility. The
    # canonical validator retains them; the resolver reports them as warnings
    # instead of silently dropping them. Controlled enum fields remain strict.
    output: dict[str, Any] = {}
    legacy_shared_blend_settings: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"scheduler_kwargs.{key}"
        if key in _INTERNAL_KEYS:
            continue
        if key in _INTEGER_FIELDS:
            output[key] = _coerce_integer(path, value, errors)
        elif key in _FLOAT_FIELDS:
            output[key] = _coerce_float(path, value, errors)
        elif key in _BOOLEAN_FIELDS:
            output[key] = _coerce_bool(path, value, errors)
        elif key in _ENUM_FIELDS:
            token = str(value).strip()
            if token not in _ENUM_FIELDS[key]:
                errors.append({"path": path, "message": f"unknown option {token!r}."})
            output[key] = token
        elif key == "auto_stabilization_sequence":
            if not isinstance(value, list):
                errors.append({"path": path, "message": "must be an ordered list."})
                continue
            sequence = [str(item).strip() for item in value]
            unknown = [item for item in sequence if item not in KES_ALLOWED_STABILIZATION_METHODS]
            if unknown:
                errors.append({"path": path, "message": f"contains unknown methods: {', '.join(unknown)}."})
            if len(sequence) != len(set(sequence)):
                errors.append({"path": path, "message": "cannot contain duplicate methods."})
            if not sequence:
                errors.append({"path": path, "message": "must contain at least one method."})
            output[key] = sequence
        elif key == "blend_methods":
            normalized, promoted = _validate_blend_methods(value, errors)
            if normalized is not None:
                output[key] = normalized
            for shared_key, shared_value in promoted.items():
                legacy_shared_blend_settings.setdefault(shared_key, shared_value)
        elif key == "compatibility":
            normalized = _validate_compatibility(value, errors)
            if normalized is not None:
                output[key] = normalized
        elif key == "blend_weights":
            if not isinstance(value, list):
                errors.append({"path": path, "message": "must be a list of numeric weights."})
                continue
            weights = [_coerce_float(f"{path}[{index}]", item, errors) for index, item in enumerate(value)]
            output[key] = [item for item in weights if item is not None]
        elif key in {"pipeline_mode", "config_path", "preset_name"}:
            output[key] = str(value).strip()
        else:
            # Remaining runtime-default values are strings used by controlled UI choices/paths.
            output[key] = str(value) if isinstance(KES_RUNTIME_DEFAULTS.get(key), str) else deepcopy(value)

    for shared_key, shared_value in legacy_shared_blend_settings.items():
        output.setdefault(shared_key, shared_value)

    active_blend_methods = output.get("blend_methods")
    if isinstance(active_blend_methods, Mapping) and active_blend_methods:
        requested_mode = str(output.get("blending_mode", "default")).strip().lower()
        if requested_mode in {"", "default", "auto"}:
            # Nonzero weights are the source of truth. When one or more methods
            # are explicitly active, use weighted mode so valid methods are not
            # pruned by the legacy Karras/Exponential default pairing.
            output["blending_mode"] = "weights"

    if errors:
        raise SchedulerSettingsValidationError(errors)
    return output


def _diff_values(requested: Any, effective: Any, path: str = "scheduler_kwargs") -> list[str]:
    warnings: list[str] = []
    if isinstance(requested, Mapping) and isinstance(effective, Mapping):
        for key, value in requested.items():
            if key in _INTERNAL_KEYS or key == "pipeline_mode":
                continue
            child = f"{path}.{key}"
            if key not in effective:
                warnings.append(f"{child} was not retained in the effective scheduler settings.")
            else:
                warnings.extend(_diff_values(value, effective[key], child))
        return warnings
    if isinstance(requested, list) and isinstance(effective, list):
        if requested != effective:
            warnings.append(f"{path} was normalized from {requested!r} to {effective!r}.")
        return warnings
    if requested != effective:
        warnings.append(f"{path} was normalized from {requested!r} to {effective!r}.")
    return warnings


@dataclass(frozen=True)
class SchedulerSettingsResolution:
    scheduler_name: str
    requested_settings: dict[str, Any]
    effective_settings: dict[str, Any]
    runtime_settings: dict[str, Any]
    compatibility_policy: dict[str, Any] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    preset_reference: dict[str, Any] = field(default_factory=dict)
    step_count_source: str = "generation.steps"
    explicit_settings: bool = False
    fallback_applied: bool = False
    requested_hash: str = ""
    effective_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_name": self.scheduler_name,
            "requested_settings": deepcopy(self.requested_settings),
            "effective_settings": deepcopy(self.effective_settings),
            "runtime_settings": deepcopy(self.runtime_settings),
            "compatibility_policy": deepcopy(self.compatibility_policy),
            "validation_warnings": list(self.validation_warnings),
            "validation_warning_count": len(self.validation_warnings),
            "preset_reference": deepcopy(self.preset_reference),
            "step_count_source": self.step_count_source,
            "explicit_settings": self.explicit_settings,
            "fallback_applied": self.fallback_applied,
            "requested_hash": self.requested_hash,
            "effective_hash": self.effective_hash,
        }


def _preset_reference(payload: Mapping[str, Any], prior: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prior_reference = dict((prior or {}).get("preset_reference") or {})
    result = {
        "id": payload.get("_webui_scheduler_preset_id") or prior_reference.get("id") or None,
        "name": payload.get("_webui_scheduler_preset_name") or prior_reference.get("name") or None,
        "plugin_id": payload.get("_webui_scheduler_preset_plugin_id") or prior_reference.get("plugin_id") or None,
        "source": payload.get("_webui_scheduler_preset_source") or prior_reference.get("source") or None,
    }
    return {key: value for key, value in result.items() if value not in (None, "")}


def resolve_simple_kes_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], SchedulerSettingsResolution]:
    normalized = deepcopy(dict(payload or {}))
    scheduler_name = _normalized_scheduler_name(normalized.get("scheduler_name"))
    if scheduler_name not in {"simple_kes", "scheduler.simple_kes"}:
        raise SchedulerSettingsValidationError([
            {"path": "scheduler_name", "message": "Simple KES resolution requires scheduler_name=simple_kes."}
        ])

    browser_scheduler = _normalized_scheduler_name(normalized.get("_webui_scheduler_browser_name"))
    if browser_scheduler and browser_scheduler not in {"simple_kes", "scheduler.simple_kes"}:
        raise SchedulerSettingsValidationError([
            {
                "path": "scheduler_name",
                "message": f"browser selected {browser_scheduler!r}, but backend request selected 'simple_kes'.",
            }
        ])

    diagnostics = dict(normalized.get("diagnostics") or {})
    prior = diagnostics.get("scheduler_settings")
    prior = dict(prior) if isinstance(prior, Mapping) else {}
    preset_reference = _preset_reference(normalized, prior)
    preset_plugin = _normalized_scheduler_name(preset_reference.get("plugin_id"))
    if preset_plugin and preset_plugin not in {"simple_kes", "scheduler.simple_kes"}:
        raise SchedulerSettingsValidationError([
            {
                "path": "scheduler_preset_reference.plugin_id",
                "message": f"preset belongs to {preset_reference.get('plugin_id')!r}, not Simple KES.",
            }
        ])

    raw_settings = normalized.get("scheduler_kwargs") or {}
    if not isinstance(raw_settings, Mapping):
        raise SchedulerSettingsValidationError([
            {"path": "scheduler_kwargs", "message": "must be an object."}
        ])
    requested_runtime = _validate_requested_settings(raw_settings)

    step_errors: list[dict[str, Any]] = []
    resolved_steps = _coerce_integer("steps", normalized.get("steps", KES_RUNTIME_DEFAULTS["steps"]), step_errors)
    if step_errors or resolved_steps is None or resolved_steps < 1:
        if resolved_steps is not None and resolved_steps < 1:
            step_errors.append({"path": "steps", "message": "must be at least 1."})
        raise SchedulerSettingsValidationError(step_errors)

    warnings: list[str] = []
    known_keys = set(KES_RUNTIME_DEFAULTS) | _LEGACY_CONTROL_KEYS | _INTERNAL_KEYS
    for unknown_key in sorted(set(raw_settings) - known_keys):
        warnings.append(
            f"scheduler_kwargs.{unknown_key} is not in the current Simple KES UI schema and was preserved for replay/runtime compatibility."
        )
    requested_step = requested_runtime.get("steps")
    if requested_step is not None and int(requested_step) != int(resolved_steps):
        warnings.append(
            "scheduler_kwargs.steps differed from generation.steps and was synchronized to the canonical generation step count."
        )
    validator_input = deepcopy(requested_runtime)
    validator_input["steps"] = int(resolved_steps)
    pipeline_mode = validator_input.pop("pipeline_mode", None)
    validator_input.pop("config_path", None)
    validator_input.pop("preset_name", None)

    validated = validate_simple_kes_settings(validator_input, pipeline_mode=pipeline_mode)
    policy = dict(validated.pop("_policy", {}) or {})
    warnings.extend(str(item) for item in validated.pop("_validation_warnings", []) or [])
    validated.pop("device", None)
    validated["steps"] = int(resolved_steps)
    for integer_key in _INTEGER_FIELDS:
        if integer_key in validated:
            validated[integer_key] = int(validated[integer_key])

    warnings.extend(_diff_values(validator_input, validated))
    warnings = list(dict.fromkeys(warnings))

    effective_settings = _json_copy(validated)
    runtime_settings = {
        key: deepcopy(value)
        for key, value in effective_settings.items()
        if key not in _LINKED_KEYS and key not in _INTERNAL_KEYS
    }

    provenance_requested = prior.get("requested_settings")
    if isinstance(provenance_requested, Mapping) and prior.get("runtime_settings") == dict(raw_settings):
        requested_for_audit = _json_copy(provenance_requested)
    else:
        requested_for_audit = _json_copy(dict(raw_settings))

    resolution = SchedulerSettingsResolution(
        scheduler_name="simple_kes",
        requested_settings=requested_for_audit,
        effective_settings=effective_settings,
        runtime_settings=_json_copy(runtime_settings),
        compatibility_policy=_json_copy(policy),
        validation_warnings=warnings,
        preset_reference=_json_copy(preset_reference),
        step_count_source="generation.steps",
        explicit_settings=bool(requested_for_audit),
        fallback_applied=not bool(requested_for_audit),
        requested_hash=_stable_hash(requested_for_audit),
        effective_hash=_stable_hash(effective_settings),
    )

    normalized["scheduler_name"] = "simple_kes"
    normalized["steps"] = int(resolved_steps)
    normalized["scheduler_kwargs"] = deepcopy(resolution.runtime_settings)
    diagnostics["scheduler_settings"] = resolution.to_dict()
    normalized["diagnostics"] = diagnostics
    return normalized, resolution


def normalize_scheduler_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], SchedulerSettingsResolution | None]:
    normalized = deepcopy(dict(payload or {}))
    scheduler_name = _normalized_scheduler_name(normalized.get("scheduler_name"))
    if scheduler_name in {"simple_kes", "scheduler.simple_kes"}:
        return resolve_simple_kes_payload(normalized)
    return normalized, None


def scheduler_resolution_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    diagnostics = dict((payload or {}).get("diagnostics") or {})
    resolution = diagnostics.get("scheduler_settings")
    return dict(resolution) if isinstance(resolution, Mapping) else {}
