from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import RuntimeProfileSelection
from .cuda_allocator import (
    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
    canonicalize_cuda_allocator_conf,
)


RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION = 1
RUNTIME_MEMORY_PROFILE_JSON_SCHEMA_ID = "urn:image-gen:runtime-memory-profile:v1"

_MSLK_ENVIRONMENT_TO_FIELD = {
    "MSLK_FMHA_POLICY": "policy",
    "MSLK_FMHA_DEBUG": "debug",
    "MSLK_FMHA_BLOCK_N": "block_n",
    "MSLK_FMHA_BLOCK_M": "block_m",
    "MSLK_FMHA_NUM_WARPS": "num_warps",
    "MSLK_FMHA_NUM_STAGES": "num_stages",
    "MSLK_FMHA_EXPERIMENTAL_HEAD_DIMS": "experimental_head_dims",
}
_MSLK_FIELDS = frozenset(_MSLK_ENVIRONMENT_TO_FIELD.values())
_MSLK_POSITIVE_INTEGER_FIELDS = {"block_n", "block_m", "num_warps", "num_stages"}
_MSLK_POLICIES = {
    "",
    "default",
    "auto",
    "blackwell_safe",
    "env",
    "off",
    "benchmark",
}
_MSLK_DEBUG_VALUES = {"", "0", "1", "false", "true", "off", "on"}
_PROFILE_FIELDS = {
    "profile_id",
    "label",
    "schema_version",
    "source",
    "attention_backend",
    "attention_slicing",
    "memory_policy",
    "vram_safety_margin_mb",
    "retain_unet_between_jobs",
    "retain_vae_between_jobs",
    "retain_text_encoder_between_jobs",
    "preview_policy",
    "hires_memory_profile",
    "pre_hires_cleanup",
    "vae_tiling",
    "vae_slicing",
    "vae_device",
    "oom_retry_profile",
    "oom_retry_limit",
    "mslk_environment",
    "cuda_allocator_environment",
    "notes",
}
_ATTENTION_BACKENDS = {"auto", "default", "eager", "sdpa", "xformers"}
_ATTENTION_SLICING_MODES = {"off", "auto", "max"}
_MEMORY_POLICIES = {"auto", "high_vram", "balanced", "low_vram", "cpu_fallback"}
_PREVIEW_POLICIES = {
    "normal",
    "suspend_on_pressure",
    "disable_during_hires",
    "disabled",
}
_HIRES_MEMORY_PROFILES = {"inherit", "balanced", "low_vram", "maximum"}
_VAE_DEVICES = {"auto", "cuda", "cpu"}
_OOM_RETRY_PROFILES = {"disabled", "cleanup", "low_vram", "maximum"}


def runtime_profile_json_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema for a complete runtime profile file.

    The loader intentionally requires every canonical field so Phase 14L can
    consume profiles without inheriting hidden defaults from a different build.
    The optional ``runtime_memory_profile`` wrapper accepted by the loader is a
    transport convenience and is documented separately; this schema describes
    the portable profile object itself.
    """

    mslk_keys = sorted(set(_MSLK_FIELDS) | set(_MSLK_ENVIRONMENT_TO_FIELD))
    mslk_properties: dict[str, dict[str, Any]] = {}
    for key in mslk_keys:
        field_name = _MSLK_ENVIRONMENT_TO_FIELD.get(key, key)
        property_schema: dict[str, Any] = {"type": "string"}
        if field_name in _MSLK_POSITIVE_INTEGER_FIELDS:
            property_schema["pattern"] = r"^$|^[1-9][0-9]*$"
        elif field_name == "policy":
            property_schema["enum"] = sorted(_MSLK_POLICIES)
        elif field_name == "debug":
            property_schema["enum"] = sorted(_MSLK_DEBUG_VALUES)
        mslk_properties[key] = property_schema

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": RUNTIME_MEMORY_PROFILE_JSON_SCHEMA_ID,
        "title": "IMAGE_GEN Runtime Memory Profile",
        "description": (
            "Complete Phase 14K runtime profile consumed by CLI, WebUI, resident "
            "workers, and the Phase 14L autotuner."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_PROFILE_FIELDS),
        "properties": {
            "profile_id": {"type": "string", "minLength": 1},
            "label": {"type": "string", "minLength": 1},
            "schema_version": {
                "type": "integer",
                "const": RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION,
            },
            "source": {"type": "string", "minLength": 1},
            "attention_backend": {
                "type": "string",
                "enum": sorted(_ATTENTION_BACKENDS),
            },
            "attention_slicing": {
                "type": "string",
                "enum": sorted(_ATTENTION_SLICING_MODES),
            },
            "memory_policy": {
                "type": "string",
                "enum": sorted(_MEMORY_POLICIES),
            },
            "vram_safety_margin_mb": {"type": "integer", "minimum": 0},
            "retain_unet_between_jobs": {"type": "boolean"},
            "retain_vae_between_jobs": {"type": "boolean"},
            "retain_text_encoder_between_jobs": {"type": "boolean"},
            "preview_policy": {
                "type": "string",
                "enum": sorted(_PREVIEW_POLICIES),
            },
            "hires_memory_profile": {
                "type": "string",
                "enum": sorted(_HIRES_MEMORY_PROFILES),
            },
            "pre_hires_cleanup": {"type": "boolean"},
            "vae_tiling": {"type": "boolean"},
            "vae_slicing": {"type": "boolean"},
            "vae_device": {"type": "string", "enum": sorted(_VAE_DEVICES)},
            "oom_retry_profile": {
                "type": "string",
                "enum": sorted(_OOM_RETRY_PROFILES),
            },
            "oom_retry_limit": {"type": "integer", "minimum": 0},
            "mslk_environment": {
                "type": "object",
                "additionalProperties": False,
                "properties": mslk_properties,
            },
            "cuda_allocator_environment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE: {"type": "string"}
                },
            },
            "notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _require_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Runtime profile field {field_name!r} must be a string.")
    if not allow_empty and not value.strip():
        raise ValueError(f"Runtime profile field {field_name!r} must not be empty.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Runtime profile field {field_name!r} must be boolean.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Runtime profile field {field_name!r} must be an integer.")
    if value < 0:
        raise ValueError(
            f"Runtime profile field {field_name!r} must be non-negative."
        )
    return value


def _string_mapping(
    value: Any,
    field_name: str,
    *,
    allowed_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Runtime profile field {field_name!r} must be an object.")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError(
                f"Runtime profile field {field_name!r} must map strings to strings."
            )
        if allowed_keys is not None and raw_key not in allowed_keys:
            raise ValueError(
                f"Unknown {field_name} key {raw_key!r}."
            )
        normalized[raw_key] = raw_value
    return normalized


def _validate_choice(field_name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        joined = ", ".join(sorted(choices))
        raise ValueError(
            f"Runtime profile field {field_name!r} must be one of: {joined}."
        )


def _validate_mslk_environment(values: Mapping[str, str]) -> None:
    for raw_key, raw_value in values.items():
        field_name = _MSLK_ENVIRONMENT_TO_FIELD.get(raw_key, raw_key)
        if field_name in _MSLK_POSITIVE_INTEGER_FIELDS:
            if raw_value == "":
                continue
            try:
                parsed = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Runtime profile MSLK field {raw_key!r} must be blank or a positive integer."
                ) from exc
            if parsed <= 0:
                raise ValueError(
                    f"Runtime profile MSLK field {raw_key!r} must be greater than zero when supplied."
                )
        elif field_name == "policy" and raw_value.strip().lower() not in _MSLK_POLICIES:
            raise ValueError(
                "Runtime profile MSLK policy must be blank or one of: "
                "auto, benchmark, blackwell_safe, default, env, off."
            )
        elif field_name == "debug" and raw_value.strip().lower() not in _MSLK_DEBUG_VALUES:
            raise ValueError(
                "Runtime profile MSLK debug must be blank or one of: "
                "0, 1, false, true, off, on."
            )


@dataclass(frozen=True)
class RuntimeMemoryProfile:
    """Portable Phase 14K runtime profile consumed by CLI, WebUI, and Phase 14L.

    A profile is a template only. It is converted into the existing
    :class:`RuntimeStartupOptions` contract, after which environment and
    individual command-line values retain their documented higher precedence.
    """

    profile_id: str
    label: str
    schema_version: int = RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION
    source: str = "user"
    attention_backend: str = "auto"
    attention_slicing: str = "off"
    memory_policy: str = "auto"
    vram_safety_margin_mb: int = 1024
    retain_unet_between_jobs: bool = True
    retain_vae_between_jobs: bool = True
    retain_text_encoder_between_jobs: bool = True
    preview_policy: str = "normal"
    hires_memory_profile: str = "inherit"
    pre_hires_cleanup: bool = False
    vae_tiling: bool = False
    vae_slicing: bool = False
    vae_device: str = "auto"
    oom_retry_profile: str = "cleanup"
    oom_retry_limit: int = 1
    mslk_environment: dict[str, str] = field(default_factory=dict)
    cuda_allocator_environment: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.profile_id, "profile_id")
        _require_string(self.label, "label")
        _require_string(self.source, "source")
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise ValueError("Runtime profile schema_version must be an integer.")
        if self.schema_version != RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported runtime profile schema_version "
                f"{self.schema_version!r}; expected {RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION}."
            )

        _validate_choice(
            "attention_backend", self.attention_backend, _ATTENTION_BACKENDS
        )
        _validate_choice(
            "attention_slicing", self.attention_slicing, _ATTENTION_SLICING_MODES
        )
        _validate_choice("memory_policy", self.memory_policy, _MEMORY_POLICIES)
        _validate_choice("preview_policy", self.preview_policy, _PREVIEW_POLICIES)
        _validate_choice(
            "hires_memory_profile",
            self.hires_memory_profile,
            _HIRES_MEMORY_PROFILES,
        )
        _validate_choice("vae_device", self.vae_device, _VAE_DEVICES)
        _validate_choice(
            "oom_retry_profile", self.oom_retry_profile, _OOM_RETRY_PROFILES
        )

        _require_non_negative_int(
            self.vram_safety_margin_mb, "vram_safety_margin_mb"
        )
        _require_non_negative_int(self.oom_retry_limit, "oom_retry_limit")
        for field_name in (
            "retain_unet_between_jobs",
            "retain_vae_between_jobs",
            "retain_text_encoder_between_jobs",
            "pre_hires_cleanup",
            "vae_tiling",
            "vae_slicing",
        ):
            _require_bool(getattr(self, field_name), field_name)

        mslk_values = _string_mapping(
            self.mslk_environment,
            "mslk_environment",
            allowed_keys=set(_MSLK_FIELDS) | set(_MSLK_ENVIRONMENT_TO_FIELD),
        )
        _validate_mslk_environment(mslk_values)
        allocator_values = _string_mapping(
            self.cuda_allocator_environment,
            "cuda_allocator_environment",
            allowed_keys={CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE},
        )
        if CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in allocator_values:
            canonicalize_cuda_allocator_conf(
                allocator_values[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE]
            )

        if not isinstance(self.notes, (list, tuple)) or not all(
            isinstance(item, str) for item in self.notes
        ):
            raise ValueError("Runtime profile notes must be a list of strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "schema_version": self.schema_version,
            "source": self.source,
            "attention_backend": self.attention_backend,
            "attention_slicing": self.attention_slicing,
            "memory_policy": self.memory_policy,
            "vram_safety_margin_mb": self.vram_safety_margin_mb,
            "retain_unet_between_jobs": self.retain_unet_between_jobs,
            "retain_vae_between_jobs": self.retain_vae_between_jobs,
            "retain_text_encoder_between_jobs": self.retain_text_encoder_between_jobs,
            "preview_policy": self.preview_policy,
            "hires_memory_profile": self.hires_memory_profile,
            "pre_hires_cleanup": self.pre_hires_cleanup,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "vae_device": self.vae_device,
            "oom_retry_profile": self.oom_retry_profile,
            "oom_retry_limit": self.oom_retry_limit,
            "mslk_environment": dict(self.mslk_environment),
            "cuda_allocator_environment": dict(self.cuda_allocator_environment),
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        require_complete: bool = True,
    ) -> "RuntimeMemoryProfile":
        if not isinstance(values, Mapping):
            raise ValueError("Runtime profile must be an object.")
        payload = dict(values)
        unknown = set(payload) - _PROFILE_FIELDS
        if unknown:
            joined = ", ".join(sorted(str(item) for item in unknown))
            raise ValueError(f"Unknown runtime profile field(s): {joined}.")
        if require_complete:
            missing = _PROFILE_FIELDS - set(payload)
            if missing:
                joined = ", ".join(sorted(missing))
                raise ValueError(
                    f"Runtime profile is missing required field(s): {joined}."
                )

        defaults = {
            "schema_version": RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION,
            "source": "user",
            "attention_backend": "auto",
            "attention_slicing": "off",
            "memory_policy": "auto",
            "vram_safety_margin_mb": 1024,
            "retain_unet_between_jobs": True,
            "retain_vae_between_jobs": True,
            "retain_text_encoder_between_jobs": True,
            "preview_policy": "normal",
            "hires_memory_profile": "inherit",
            "pre_hires_cleanup": False,
            "vae_tiling": False,
            "vae_slicing": False,
            "vae_device": "auto",
            "oom_retry_profile": "cleanup",
            "oom_retry_limit": 1,
            "mslk_environment": {},
            "cuda_allocator_environment": {},
            "notes": [],
        }
        merged = {**defaults, **payload}
        return cls(
            profile_id=_require_string(merged.get("profile_id"), "profile_id"),
            label=_require_string(merged.get("label"), "label"),
            schema_version=_require_non_negative_int(
                merged["schema_version"], "schema_version"
            ),
            source=_require_string(merged["source"], "source"),
            attention_backend=_require_string(
                merged["attention_backend"], "attention_backend"
            ),
            attention_slicing=_require_string(
                merged["attention_slicing"], "attention_slicing"
            ),
            memory_policy=_require_string(
                merged["memory_policy"], "memory_policy"
            ),
            vram_safety_margin_mb=_require_non_negative_int(
                merged["vram_safety_margin_mb"], "vram_safety_margin_mb"
            ),
            retain_unet_between_jobs=_require_bool(
                merged["retain_unet_between_jobs"], "retain_unet_between_jobs"
            ),
            retain_vae_between_jobs=_require_bool(
                merged["retain_vae_between_jobs"], "retain_vae_between_jobs"
            ),
            retain_text_encoder_between_jobs=_require_bool(
                merged["retain_text_encoder_between_jobs"],
                "retain_text_encoder_between_jobs",
            ),
            preview_policy=_require_string(
                merged["preview_policy"], "preview_policy"
            ),
            hires_memory_profile=_require_string(
                merged["hires_memory_profile"], "hires_memory_profile"
            ),
            pre_hires_cleanup=_require_bool(
                merged["pre_hires_cleanup"], "pre_hires_cleanup"
            ),
            vae_tiling=_require_bool(merged["vae_tiling"], "vae_tiling"),
            vae_slicing=_require_bool(merged["vae_slicing"], "vae_slicing"),
            vae_device=_require_string(merged["vae_device"], "vae_device"),
            oom_retry_profile=_require_string(
                merged["oom_retry_profile"], "oom_retry_profile"
            ),
            oom_retry_limit=_require_non_negative_int(
                merged["oom_retry_limit"], "oom_retry_limit"
            ),
            mslk_environment=_string_mapping(
                merged["mslk_environment"],
                "mslk_environment",
                allowed_keys=set(_MSLK_FIELDS) | set(_MSLK_ENVIRONMENT_TO_FIELD),
            ),
            cuda_allocator_environment=_string_mapping(
                merged["cuda_allocator_environment"],
                "cuda_allocator_environment",
                allowed_keys={CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE},
            ),
            notes=tuple(
                _require_string(item, "notes[]", allow_empty=True)
                for item in _notes_list(merged["notes"])
            ),
        )

    def runtime_values(self) -> dict[str, Any]:
        mslk_values: dict[str, str] = {}
        for raw_key, value in self.mslk_environment.items():
            key = _MSLK_ENVIRONMENT_TO_FIELD.get(raw_key, raw_key)
            mslk_values[key] = value
        return {
            "attention_backend": self.attention_backend,
            "attention_slicing": self.attention_slicing,
            "memory_policy": self.memory_policy,
            "vram_safety_margin_mb": self.vram_safety_margin_mb,
            "retain_unet_between_jobs": self.retain_unet_between_jobs,
            "retain_vae_between_jobs": self.retain_vae_between_jobs,
            "retain_text_encoder_between_jobs": self.retain_text_encoder_between_jobs,
            "preview_policy": self.preview_policy,
            "hires_memory_profile": self.hires_memory_profile,
            "pre_hires_cleanup": self.pre_hires_cleanup,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "vae_device": self.vae_device,
            "oom_retry_profile": self.oom_retry_profile,
            "oom_retry_limit": self.oom_retry_limit,
            "mslk_fmha": mslk_values,
            "allocator_options": dict(self.cuda_allocator_environment),
        }

    def selection(
        self,
        *,
        selector: str,
        selected_from: str,
    ) -> RuntimeProfileSelection:
        return RuntimeProfileSelection(
            profile_id=self.profile_id,
            label=self.label,
            schema_version=self.schema_version,
            source=self.source,
            selector=str(selector),
            selected_from=str(selected_from),
            notes=tuple(self.notes),
        )


def _notes_list(value: Any) -> list[Any] | tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("Runtime profile notes must be a list of strings.")
    return value


_BUILTIN_PROFILES = {
    "auto": RuntimeMemoryProfile(
        profile_id="auto",
        label="Automatic",
        source="builtin",
        notes=(
            "Preserves the default automatic backend and memory-policy behavior.",
        ),
    ),
    "balanced": RuntimeMemoryProfile(
        profile_id="balanced",
        label="Balanced",
        source="builtin",
        memory_policy="balanced",
        retain_unet_between_jobs=True,
        retain_vae_between_jobs=False,
        retain_text_encoder_between_jobs=False,
        preview_policy="suspend_on_pressure",
        hires_memory_profile="balanced",
        oom_retry_profile="cleanup",
        notes=(
            "Keeps the UNet reusable while offloading the VAE and text encoder between jobs.",
            "Preview decoding may be suspended when the configured safety margin is threatened.",
        ),
    ),
    "low-memory": RuntimeMemoryProfile(
        profile_id="low-memory",
        label="Low Memory",
        source="builtin",
        memory_policy="low_vram",
        vram_safety_margin_mb=1536,
        retain_unet_between_jobs=False,
        retain_vae_between_jobs=False,
        retain_text_encoder_between_jobs=False,
        preview_policy="suspend_on_pressure",
        hires_memory_profile="low_vram",
        pre_hires_cleanup=True,
        vae_slicing=True,
        oom_retry_profile="low_vram",
        notes=(
            "Uses sequential whole-component residency and a stronger pre-hires cleanup boundary.",
            "Automatic OOM recovery suspends image preview for the remainder of the job.",
        ),
    ),
    "maximum-memory-savings": RuntimeMemoryProfile(
        profile_id="maximum-memory-savings",
        label="Maximum Memory Savings",
        source="builtin",
        attention_slicing="max",
        memory_policy="low_vram",
        vram_safety_margin_mb=2048,
        retain_unet_between_jobs=False,
        retain_vae_between_jobs=False,
        retain_text_encoder_between_jobs=False,
        preview_policy="disabled",
        hires_memory_profile="maximum",
        pre_hires_cleanup=True,
        vae_tiling=True,
        vae_slicing=True,
        vae_device="cpu",
        oom_retry_profile="maximum",
        notes=(
            "Prioritizes fit over speed and disables image preview decoding.",
            "Uses maximum attention and VAE memory controls with CPU VAE execution.",
        ),
    ),
}
_PROFILE_ALIASES = {
    "default": "auto",
    "automatic": "auto",
    "low_memory": "low-memory",
    "lowmemory": "low-memory",
    "low_vram": "low-memory",
    "lowvram": "low-memory",
    "maximum": "maximum-memory-savings",
    "max": "maximum-memory-savings",
    "maximum_memory_savings": "maximum-memory-savings",
}


def _clone_profile(profile: RuntimeMemoryProfile) -> RuntimeMemoryProfile:
    return RuntimeMemoryProfile.from_mapping(profile.to_dict(), require_complete=True)


def builtin_runtime_profiles() -> tuple[RuntimeMemoryProfile, ...]:
    return tuple(_clone_profile(_BUILTIN_PROFILES[key]) for key in _BUILTIN_PROFILES)


def runtime_profile_descriptors() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in builtin_runtime_profiles()]


def _builtin_key(selector: str) -> str:
    normalized = str(selector).strip().lower().replace(" ", "_")
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    if normalized in _BUILTIN_PROFILES:
        return normalized
    hyphenated = normalized.replace("_", "-")
    return _PROFILE_ALIASES.get(hyphenated, hyphenated)


def load_runtime_memory_profile(
    selector: str,
    *,
    base_dir: str | Path | None = None,
) -> RuntimeMemoryProfile:
    requested = str(selector or "").strip()
    if not requested:
        raise ValueError("--runtime-profile requires a built-in profile ID or JSON path.")
    builtin_key = _builtin_key(requested)
    if builtin_key in _BUILTIN_PROFILES:
        return _clone_profile(_BUILTIN_PROFILES[builtin_key])

    path = Path(requested).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        choices = ", ".join(_BUILTIN_PROFILES)
        raise ValueError(
            f"Unknown runtime profile {requested!r}. Use one of {choices}, "
            "or provide an existing JSON file path."
        )
    if path.suffix.lower() != ".json":
        raise ValueError("Runtime profile files must use the .json extension.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError(f"Runtime profile file must be valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Runtime profile JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime profile JSON must contain an object: {path}")
    if "runtime_memory_profile" in payload:
        extra_wrapper_fields = set(payload) - {"runtime_memory_profile"}
        if extra_wrapper_fields:
            joined = ", ".join(sorted(str(item) for item in extra_wrapper_fields))
            raise ValueError(
                f"Unknown runtime profile wrapper field(s): {joined}."
            )
        nested = payload["runtime_memory_profile"]
        if not isinstance(nested, dict):
            raise ValueError("runtime_memory_profile must contain an object.")
        payload = nested
    return RuntimeMemoryProfile.from_mapping(payload, require_complete=True)
