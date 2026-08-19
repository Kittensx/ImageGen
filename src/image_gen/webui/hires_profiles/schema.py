from __future__ import annotations

import math
import re
from dataclasses import MISSING, dataclass, fields
from types import UnionType
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

from image_gen.contracts import GenerationRequest
from image_gen.systems.image_conditioning import SUPPORTED_HIRES_STEP_POLICIES
from image_gen.systems.upscaling import (
    SUPPORTED_ASPECT_POLICIES,
    SUPPORTED_BLURRED_EDGE_METHODS,
    SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS,
    SUPPORTED_PADDING_MODES,
)

from .contracts import HiresProfileSaveManifest, HiresSettingDescriptor


_ACRONYMS = {
    "cfg": "CFG",
    "vae": "VAE",
    "sd": "SD",
    "sd1": "SD1",
    "sd2": "SD2",
    "sd3": "SD3",
    "sdxl": "SDXL",
    "id": "ID",
    "rgb": "RGB",
}


@dataclass(frozen=True)
class HiresFieldPolicy:
    group: str
    description: str = ""
    editor_kind: str = ""
    choice_source: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    asset_kind: str = ""


# The policy registry is the authoritative backend whitelist. The inspector still
# discovers every GenerationRequest field with a hires_ prefix; hires fields that
# are absent here are classified as unrecognized and are never persisted. This
# makes newly introduced drift visible without silently expanding the save boundary.
_FIELD_POLICIES: dict[str, HiresFieldPolicy] = {
    "hires_enabled": HiresFieldPolicy("General", "Whether the hires pass is enabled.", "boolean"),
    "hires_prompt_parser_mode": HiresFieldPolicy("Prompt Processing", "How hires prompt parsing relates to the base pass.", "select", "prompt_parser_modes"),
    "hires_prompt_parser_name": HiresFieldPolicy("Prompt Processing", "Prompt parser used by the hires pass.", "select", "prompt_parsers", asset_kind="prompt_parser"),
    "hires_prompt_parser_kwargs": HiresFieldPolicy("Prompt Processing", "Structured settings owned by the selected hires prompt parser.", "dynamic_object", "prompt_parser_options"),
    "hires_shortcut_profile_mode": HiresFieldPolicy("Prompt Processing", "How the hires shortcut profile relates to the base pass.", "select", "shortcut_profile_modes"),
    "hires_shortcut_profile_name": HiresFieldPolicy("Prompt Processing", "Shortcut profile used by the hires pass.", "select", "shortcut_profiles", asset_kind="prompt_shortcut_profile"),
    "hires_size_mode": HiresFieldPolicy("Dimensions", "How the hires target size is selected.", "select", "size_modes"),
    "hires_scale": HiresFieldPolicy("Dimensions", "Scale applied to the base dimensions.", "number", minimum=1.01, maximum=8.0, step=0.05),
    "hires_width": HiresFieldPolicy("Dimensions", "Explicit hires target width.", "integer", minimum=64, maximum=16384, step=1),
    "hires_height": HiresFieldPolicy("Dimensions", "Explicit hires target height.", "integer", minimum=64, maximum=16384, step=1),
    "hires_steps": HiresFieldPolicy("Refinement", "Requested hires refinement steps.", "integer", minimum=1, maximum=200, step=1),
    "hires_denoising_strength": HiresFieldPolicy("Refinement", "Hires denoising strength.", "number", minimum=0.01, maximum=1.0, step=0.01),
    "hires_step_policy": HiresFieldPolicy("Refinement", "Policy used to convert requested hires steps into an active schedule.", "select", "step_policies"),
    "hires_sampler_name": HiresFieldPolicy("Refinement", "Sampler used for hires refinement; empty means inherit the base sampler.", "select", "samplers", asset_kind="sampler"),
    "hires_scheduler_name": HiresFieldPolicy("Refinement", "Scheduler used for hires refinement; empty means inherit the base scheduler.", "select", "schedulers", asset_kind="scheduler"),
    "hires_cfg_scale": HiresFieldPolicy("Refinement", "CFG scale used by the hires pass; null means inherit base CFG.", "number", minimum=0.0, maximum=50.0, step=0.1),
    "hires_cfg_rescale": HiresFieldPolicy("Refinement", "CFG rescale used by the hires pass; null means inherit base rescale.", "number", minimum=0.0, maximum=1.0, step=0.01),
    "hires_strategy": HiresFieldPolicy("Upscaling", "Hires source preparation strategy.", "select", "strategies"),
    "hires_upscaler_id": HiresFieldPolicy("Upscaling", "Canonical neural upscaler selection.", "asset_select", "upscalers", asset_kind="upscaler"),
    "hires_tile_size": HiresFieldPolicy("Upscaling", "Neural upscaler tile size; zero means automatic.", "integer", minimum=0, maximum=4096, step=8),
    "hires_tile_overlap": HiresFieldPolicy("Upscaling", "Neural upscaler tile overlap.", "integer", minimum=0, maximum=512, step=1),
    "hires_tile_batch_size": HiresFieldPolicy("Upscaling", "Number of neural tiles processed together.", "integer", minimum=1, maximum=64, step=1),
    "hires_exact_resize_filter": HiresFieldPolicy("Upscaling", "Legacy exact resize filter retained for compatibility.", "select", "exact_resize_filters"),
    "hires_final_size_correction_filter": HiresFieldPolicy("Upscaling", "Filter used when the native neural output must be corrected to the requested target.", "select", "final_resize_filters"),
    "hires_aspect_policy": HiresFieldPolicy("Upscaling", "How aspect-ratio differences are handled.", "select", "aspect_policies"),
    "hires_padding_mode": HiresFieldPolicy("Upscaling", "Padding mode used when the target requires padding.", "select", "padding_modes"),
    "hires_blurred_edge_method": HiresFieldPolicy("Upscaling", "Blur method used by blurred-edge padding.", "select", "blurred_edge_methods"),
    "hires_blurred_edge_compare_diagnostics": HiresFieldPolicy("Diagnostics", "Run the alternate blurred-edge method for comparison diagnostics.", "boolean"),
    "hires_correction_fingerprint_enabled": HiresFieldPolicy("Diagnostics", "Record the deterministic correction-contract fingerprint.", "boolean"),
    "hires_save_upscaled_pre_denoise": HiresFieldPolicy("Diagnostics", "Save the pixel-upscaled image before second-pass denoising.", "boolean"),
    "hires_save_vae_roundtrip": HiresFieldPolicy("Diagnostics", "Save the deterministic hires VAE round-trip image.", "boolean"),
    "hires_diagnostic_vae_execution_fingerprint": HiresFieldPolicy("Diagnostics", "Record hires VAE execution fingerprint diagnostics.", "boolean"),
    "hires_save_lowres": HiresFieldPolicy("Diagnostics", "Save the exact low-resolution base artifact.", "boolean"),
    "hires_memory_preflight": HiresFieldPolicy("Memory", "Run hires memory preflight before the second pass.", "boolean"),
    "hires_host_staging_policy": HiresFieldPolicy("Memory", "CPU host staging policy for large hires tensors.", "select", "host_staging_policies"),
    "hires_host_staging_cap_mb": HiresFieldPolicy("Memory", "Maximum pinned host staging allocation in MiB; zero disables the cap.", "integer", minimum=0, maximum=262144, step=64),
    "hires_artifact_disk_budget_mb": HiresFieldPolicy("Memory", "Optional disk budget for hires diagnostic artifacts in MiB; zero means no explicit budget.", "integer", minimum=0, maximum=1048576, step=64),
}


_EXCLUDED_FIELD_REASONS: dict[str, str] = {
    "hires_shortcut_profile_snapshot": "runtime_snapshot",
    "hires_configuration_mode": "runtime_resolution_provenance",
    "hires_auto_resolution_record": "runtime_resolution_provenance",
    "hires_lifecycle_state": "runtime_resolution_provenance",
    "hires_positive_prompt": "prompt_content",
    "hires_negative_prompt": "prompt_content",
    "hires_dimension_plan_version": "runtime_derived",
    "hires_dimension_plan": "runtime_derived",
    "hires_axis_scale_width": "runtime_derived",
    "hires_axis_scale_height": "runtime_derived",
    "hires_uniform_scale": "runtime_derived",
    "hires_aspect_ratio_changed": "runtime_derived",
    "hires_recorded_schedule_replay": "replay_state",
    "hires_recorded_schedule_fingerprint": "replay_state",
    "hires_schedule_conformance_source_replay": "replay_state",
    "hires_schedule_conformance_source_fingerprint": "replay_state",
    "hires_schedule_replay_mode": "replay_state",
    "hires_upscaler": "legacy_alias",
    "hires_expected_upscaler_sha256": "replay_identity",
    "hires_expected_native_scale": "runtime_derived",
    "hires_expected_vae_sha256": "replay_identity",
    "hires_expected_vae_source_kind": "replay_identity",
    "hires_recorded_target_correction": "replay_state",
    "hires_recorded_correction_fingerprint": "replay_state",
    "hires_prompt_route_plan": "runtime_derived",
}

_STATIC_CHOICE_SOURCES: dict[str, tuple[Any, ...]] = {
    "prompt_parser_modes": ("same_as_base", "explicit"),
    "shortcut_profile_modes": ("same_as_base", "explicit", "canonical_only"),
    "size_modes": ("same_as_base", "scale_from_base", "explicit_dimensions"),
    "step_policies": tuple(sorted(SUPPORTED_HIRES_STEP_POLICIES)),
    "strategies": ("pixel_neural", "pixel_resize"),
    "exact_resize_filters": ("nearest", "bilinear", "bicubic", "area"),
    "final_resize_filters": tuple(sorted(SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS)),
    "aspect_policies": tuple(sorted(SUPPORTED_ASPECT_POLICIES)),
    "padding_modes": tuple(sorted(SUPPORTED_PADDING_MODES)),
    "blurred_edge_methods": tuple(sorted(SUPPORTED_BLURRED_EDGE_METHODS)),
    "host_staging_policies": ("pageable", "pinned", "auto"),
}


def humanize_identifier(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("hires_"):
        text = text[6:]
    tokens = [token for token in re.split(r"[_\s-]+", text) if token]
    rendered: list[str] = []
    for token in tokens:
        lowered = token.casefold()
        rendered.append(_ACRONYMS.get(lowered, token[:1].upper() + token[1:]))
    return " ".join(rendered) or "Setting"


def _display_label(key: str, policy: HiresFieldPolicy) -> str:
    source = str(key)
    if policy.asset_kind:
        if source.endswith("_id"):
            source = source[:-3]
        elif source.endswith("_name"):
            source = source[:-5]
    return humanize_identifier(source)


def _field_default(field_def) -> Any:
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return field_def.default_factory()
    return None


def _value_type(annotation: Any, default: Any) -> tuple[str, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    nullable = False
    if origin in {Union, UnionType}:
        nullable = type(None) in args
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1:
            annotation = non_null[0]
            origin = get_origin(annotation)
    if annotation is bool or isinstance(default, bool):
        return "boolean", nullable
    if annotation is int or (isinstance(default, int) and not isinstance(default, bool)):
        return "integer", nullable
    if annotation is float or isinstance(default, float):
        return "number", nullable
    if origin in {dict, Mapping} or annotation is dict or isinstance(default, dict):
        return "object", nullable
    if origin in {list, tuple, Sequence} or annotation in {list, tuple} or isinstance(default, (list, tuple)):
        return "array", nullable
    return "string", nullable


def _choice_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        token = value.get("value", value.get("id", value.get("key", "")))
        label = value.get("label", value.get("name", humanize_identifier(str(token))))
        payload = {"value": token, "label": str(label)}
        for key in ("available", "description", "sha256", "asset_id", "source"):
            if key in value:
                payload[key] = value[key]
        return payload
    return {"value": value, "label": humanize_identifier(str(value))}


class HiresProfileSchemaRegistry:
    """Authoritative, dynamically inspected backend schema for hires profiles."""

    def __init__(self) -> None:
        self._fields = {item.name: item for item in fields(GenerationRequest) if item.name.startswith("hires_")}
        self._type_hints = get_type_hints(GenerationRequest)

    @property
    def eligible_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._fields if key in _FIELD_POLICIES)

    @property
    def schema_excluded_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._fields if key in _EXCLUDED_FIELD_REASONS)

    @property
    def discovered_unclassified_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self._fields
            if key not in _FIELD_POLICIES and key not in _EXCLUDED_FIELD_REASONS
        )

    def rejection_reason(self, key: str) -> str:
        if key in _EXCLUDED_FIELD_REASONS:
            return _EXCLUDED_FIELD_REASONS[key]
        if str(key).startswith("hires_"):
            return "unclassified_hires_setting"
        return "outside_hires_namespace"

    def default_values(self) -> dict[str, Any]:
        return {
            key: _field_default(self._fields[key])
            for key in self.eligible_keys
        }

    def _choices(
        self,
        policy: HiresFieldPolicy,
        choice_overrides: Mapping[str, Sequence[Any]] | None,
    ) -> tuple[dict[str, Any], ...]:
        source = policy.choice_source
        if not source:
            return ()
        if choice_overrides and source in choice_overrides:
            raw = tuple(choice_overrides[source])
        else:
            raw = _STATIC_CHOICE_SOURCES.get(source, ())
        choices = [_choice_payload(item) for item in raw]
        if source in {"samplers", "schedulers"}:
            choices.insert(0, {"value": "", "label": "Inherit Base"})
            choices.insert(1, {"value": "auto", "label": "Auto / Model Recommendation"})
        if source == "upscalers":
            choices.insert(0, {"value": "auto", "label": "Auto Select", "available": True})
        return tuple(choices)

    def _coerce_and_validate(
        self,
        key: str,
        value: Any,
        *,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> Any:
        policy = _FIELD_POLICIES[key]
        field_def = self._fields[key]
        value_type, nullable = _value_type(self._type_hints.get(key), _field_default(field_def))
        if value is None:
            if nullable:
                return None
            raise ValueError(f"{humanize_identifier(key)} cannot be null.")
        if value_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{humanize_identifier(key)} must be selected as on or off.")
            normalized: Any = value
        elif value_type == "integer":
            if isinstance(value, bool):
                raise ValueError(f"{humanize_identifier(key)} must be an integer.")
            try:
                normalized = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{humanize_identifier(key)} must be an integer.") from exc
        elif value_type == "number":
            if isinstance(value, bool):
                raise ValueError(f"{humanize_identifier(key)} must be numeric.")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{humanize_identifier(key)} must be numeric.") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{humanize_identifier(key)} must be finite.")
        elif value_type == "object":
            if not isinstance(value, Mapping):
                raise ValueError(f"{humanize_identifier(key)} must be a structured object.")
            normalized = dict(value)
        elif value_type == "array":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{humanize_identifier(key)} must be selected from a list of values.")
            normalized = list(value)
        else:
            if not isinstance(value, str):
                raise ValueError(f"{humanize_identifier(key)} must be a selected string value.")
            normalized = value.strip()

        if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
            if policy.minimum is not None and normalized < policy.minimum:
                raise ValueError(f"{humanize_identifier(key)} must be at least {policy.minimum}.")
            if policy.maximum is not None and normalized > policy.maximum:
                raise ValueError(f"{humanize_identifier(key)} must be at most {policy.maximum}.")

        choices = self._choices(policy, choice_overrides)
        if choices:
            allowed = {item.get("value") for item in choices if item.get("available", True) is not False}
            # Dynamic registries may be absent during backend-only persistence. In that
            # case asset/plugin identifiers remain representable as historical refs;
            # static enums are always validated strictly.
            dynamic_source = policy.choice_source in {"upscalers", "samplers", "schedulers", "prompt_parsers", "shortcut_profiles"}
            override_present = bool(choice_overrides and policy.choice_source in choice_overrides)
            if normalized not in allowed and (not dynamic_source or override_present):
                raise ValueError(
                    f"{humanize_identifier(key)} must be selected from the available choices."
                )
        return normalized

    def normalize_values(
        self,
        values: Mapping[str, Any] | None,
        *,
        included_fields: Sequence[str] | None = None,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        incoming = dict(values or {})
        requested = set(str(key) for key in (included_fields if included_fields is not None else incoming.keys()))
        rejected = sorted(
            key
            for key in set(incoming) | requested
            if key not in _FIELD_POLICIES
        )
        normalized: dict[str, Any] = {}
        for key in self.eligible_keys:
            if key not in requested:
                continue
            if key not in incoming:
                raise ValueError(f"Included hires setting {key!r} has no value.")
            normalized[key] = self._coerce_and_validate(
                key,
                incoming[key],
                choice_overrides=choice_overrides,
            )
        return normalized, tuple(rejected)

    def build_descriptors(
        self,
        *,
        values: Mapping[str, Any] | None = None,
        included_fields: Sequence[str] | None = None,
        baseline_values: Mapping[str, Any] | None = None,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
        unexpected_values: Mapping[str, Any] | None = None,
    ) -> tuple[HiresSettingDescriptor, ...]:
        current = dict(values or {})
        included = set(str(key) for key in (included_fields or ()))
        baseline = {**self.default_values(), **dict(baseline_values or {})}
        output: list[HiresSettingDescriptor] = []
        for key in self.eligible_keys:
            policy = _FIELD_POLICIES[key]
            field_def = self._fields[key]
            default = _field_default(field_def)
            value = current[key] if key in current else baseline.get(key, default)
            base = baseline.get(key, default)
            value_type, _nullable = _value_type(self._type_hints.get(key), default)
            choices = self._choices(policy, choice_overrides)
            dynamic_choice = policy.choice_source in {"upscalers", "samplers", "schedulers", "prompt_parsers", "shortcut_profiles"}
            editable = policy.editor_kind not in {"read_only", "object", "dynamic_object"}
            if dynamic_choice and not choices and value:
                editable = False
            output.append(
                HiresSettingDescriptor(
                    key=key,
                    label=_display_label(key, policy),
                    group=policy.group,
                    description=policy.description,
                    value_type=value_type,
                    current_value=value,
                    baseline_value=base,
                    allowed_values=choices,
                    minimum=policy.minimum,
                    maximum=policy.maximum,
                    step=policy.step,
                    asset_kind=policy.asset_kind,
                    editor_kind=policy.editor_kind or value_type,
                    available=True,
                    included=key in included,
                    modified=value != base,
                    editable=editable,
                    source="generation_request+hires_profile_schema",
                    persistence_eligibility="eligible",
                )
            )

        for key, value in sorted(dict(unexpected_values or {}).items()):
            if key in _FIELD_POLICIES:
                continue
            reason = self.rejection_reason(str(key))
            known_exclusion = str(key) in _EXCLUDED_FIELD_REASONS
            output.append(
                HiresSettingDescriptor(
                    key=str(key),
                    label=humanize_identifier(str(key)),
                    group="Excluded by Hires Profile Policy" if known_exclusion else "Unrecognized / Rejected",
                    description=(
                        f"This hires field is intentionally excluded from profile persistence ({reason})."
                        if known_exclusion
                        else "This field was presented to the hires profile serializer but is not owned by the hires profile schema."
                    ),
                    value_type=type(value).__name__,
                    current_value=value,
                    baseline_value=None,
                    editor_kind="read_only",
                    available=False,
                    included=False,
                    modified=True,
                    editable=False,
                    source="save_request",
                    persistence_eligibility=f"rejected_{reason}",
                )
            )
        return tuple(output)

    def build_save_manifest(
        self,
        *,
        profile_id: str,
        profile_name: str,
        values: Mapping[str, Any],
        included_fields: Sequence[str],
        baseline_profile_id: str = "",
        baseline_values: Mapping[str, Any] | None = None,
        rejected_fields: Sequence[str] = (),
        incoming_values: Mapping[str, Any] | None = None,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> HiresProfileSaveManifest:
        included = tuple(sorted(str(key) for key in included_fields))
        descriptors = self.build_descriptors(
            values=values,
            included_fields=included,
            baseline_values=baseline_values,
            choice_overrides=choice_overrides,
            unexpected_values={
                key: value
                for key, value in dict(incoming_values or {}).items()
                if key in set(rejected_fields)
            },
        )
        modified = tuple(sorted(item.key for item in descriptors if item.modified and item.available))
        excluded = tuple(key for key in self.eligible_keys if key not in set(included))
        rejected_unclassified = {
            key
            for key in rejected_fields
            if key not in _EXCLUDED_FIELD_REASONS
        }
        unclassified = tuple(sorted(set(self.discovered_unclassified_keys) | rejected_unclassified))
        warnings: list[str] = []
        if rejected_fields:
            warnings.append(
                "Rejected fields were presented to the hires profile serializer and were not persisted: "
                + ", ".join(sorted(rejected_fields))
            )
        return HiresProfileSaveManifest(
            profile_id=profile_id,
            profile_name=profile_name,
            baseline_profile_id=baseline_profile_id,
            descriptors=descriptors,
            included_fields=included,
            excluded_fields=excluded,
            modified_fields=modified,
            schema_excluded_fields=tuple(sorted(self.schema_excluded_keys)),
            unclassified_fields=unclassified,
            rejected_fields=tuple(sorted(set(rejected_fields))),
            warnings=tuple(warnings),
        )


__all__ = [
    "HiresFieldPolicy",
    "HiresProfileSchemaRegistry",
    "humanize_identifier",
]
