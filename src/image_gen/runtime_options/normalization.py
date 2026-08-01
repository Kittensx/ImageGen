from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Any, Mapping, MutableMapping

from .contracts import MSLKFMHAOptions, RuntimeProfileSelection, RuntimeStartupOptions
from .cuda_allocator import (
    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
    apply_cuda_allocator_environment,
    build_cuda_allocator_diagnostics,
    canonicalize_cuda_allocator_conf,
    set_cuda_allocator_option,
)


_RUNTIME_SCALAR_FIELDS = {
    item.name
    for item in fields(RuntimeStartupOptions)
    if item.name
    not in {"schema_version", "runtime_profile", "mslk_fmha", "allocator_options", "source_map"}
}

_SETTING_ALIASES = {
    "memory_retain_checkpoint_between_jobs": "retain_unet_between_jobs",
    "memory_retain_vae_between_jobs": "retain_vae_between_jobs",
    "model_runtime_retain_text_encoder_between_jobs": (
        "retain_text_encoder_between_jobs"
    ),
    "memory_vram_safety_margin_mb": "vram_safety_margin_mb",
}

_ENVIRONMENT_FIELDS = {
    "IMAGE_GEN_ATTENTION_BACKEND": "attention_backend",
    "IMAGE_GEN_MEMORY_POLICY": "memory_policy",
    "IMAGE_GEN_VRAM_SAFETY_MARGIN_MB": "vram_safety_margin_mb",
    "IMAGE_GEN_ATTENTION_SLICING": "attention_slicing",
    "IMAGE_GEN_VAE_TILING": "vae_tiling",
    "IMAGE_GEN_VAE_SLICING": "vae_slicing",
    "IMAGE_GEN_VAE_DEVICE": "vae_device",
    "IMAGE_GEN_RETAIN_UNET_BETWEEN_JOBS": "retain_unet_between_jobs",
    "IMAGE_GEN_RETAIN_VAE_BETWEEN_JOBS": "retain_vae_between_jobs",
    "IMAGE_GEN_RETAIN_TEXT_ENCODER_BETWEEN_JOBS": (
        "retain_text_encoder_between_jobs"
    ),
    "IMAGE_GEN_PREVIEW_POLICY": "preview_policy",
    "IMAGE_GEN_HIRES_MEMORY_PROFILE": "hires_memory_profile",
    "IMAGE_GEN_PRE_HIRES_CLEANUP": "pre_hires_cleanup",
    "IMAGE_GEN_OOM_RETRY_PROFILE": "oom_retry_profile",
    "IMAGE_GEN_OOM_RETRY_LIMIT": "oom_retry_limit",
}

_MSLK_ENVIRONMENT_FIELDS = {
    "MSLK_FMHA_POLICY": "policy",
    "MSLK_FMHA_DEBUG": "debug",
    "MSLK_FMHA_BLOCK_N": "block_n",
    "MSLK_FMHA_BLOCK_M": "block_m",
    "MSLK_FMHA_NUM_WARPS": "num_warps",
    "MSLK_FMHA_NUM_STAGES": "num_stages",
    "MSLK_FMHA_EXPERIMENTAL_HEAD_DIMS": "experimental_head_dims",
}

_BOOLEAN_FIELDS = {
    "vae_tiling",
    "vae_slicing",
    "retain_unet_between_jobs",
    "retain_vae_between_jobs",
    "retain_text_encoder_between_jobs",
    "pre_hires_cleanup",
}
_INTEGER_FIELDS = {"vram_safety_margin_mb", "oom_retry_limit"}

_ATTENTION_BACKENDS = {"auto", "default", "eager", "sdpa", "xformers"}
_ATTENTION_BACKEND_ALIASES = {
    "vanilla": "eager",
    "classic": "eager",
    "math": "eager",
    "torch": "sdpa",
    "torch_sdpa": "sdpa",
    "memory_efficient": "xformers",
    "unchanged": "default",
}

_MEMORY_POLICIES = {"auto", "high_vram", "balanced", "low_vram", "cpu_fallback"}
_HIRES_MEMORY_PROFILES = {"inherit", "balanced", "low_vram", "maximum"}
_HIRES_MEMORY_PROFILE_ALIASES = {
    "low": "low_vram",
    "lowvram": "low_vram",
    "memory_saver": "low_vram",
    "maximum_memory_savings": "maximum",
    "max": "maximum",
}
_PREVIEW_POLICIES = {"normal", "suspend_on_pressure", "disable_during_hires", "disabled"}
_ATTENTION_SLICING_MODES = {"off", "auto", "max"}
_VAE_DEVICES = {"auto", "cuda", "cpu"}
_OOM_RETRY_PROFILES = {"disabled", "cleanup", "low_vram", "maximum"}
_OOM_RETRY_PROFILE_ALIASES = {
    "off": "disabled",
    "none": "disabled",
    "no_retry": "disabled",
    "retry": "cleanup",
    "on": "cleanup",
    "low": "low_vram",
    "lowvram": "low_vram",
    "max": "maximum",
    "maximum_memory_savings": "maximum",
}
_MEMORY_POLICY_ALIASES = {
    "high": "high_vram",
    "highvram": "high_vram",
    "med": "balanced",
    "medvram": "balanced",
    "low": "low_vram",
    "lowvram": "low_vram",
    "cpu": "cpu_fallback",
}
_MEMORY_POLICY_RETENTION_DEFAULTS = {
    "high_vram": {
        "retain_unet_between_jobs": True,
        "retain_vae_between_jobs": True,
        "retain_text_encoder_between_jobs": True,
    },
    "balanced": {
        "retain_unet_between_jobs": True,
        "retain_vae_between_jobs": False,
        "retain_text_encoder_between_jobs": False,
    },
    "low_vram": {
        "retain_unet_between_jobs": False,
        "retain_vae_between_jobs": False,
        "retain_text_encoder_between_jobs": False,
    },
    "cpu_fallback": {
        "retain_unet_between_jobs": False,
        "retain_vae_between_jobs": False,
        "retain_text_encoder_between_jobs": False,
    },
}
_SOURCE_PRIORITY = {
    "default": 0,
    "settings": 1,
    "saved_profile": 2,
    "runtime_profile": 3,
    "environment": 4,
    "commandline_args": 5,
    "cli": 6,
}
_MSLK_POSITIVE_INTEGER_FIELDS = {"block_n", "block_m", "num_warps", "num_stages"}
_MSLK_POLICIES = {"", "default", "auto", "blackwell_safe", "env", "off", "benchmark"}
_MSLK_DEBUG_VALUES = {"", "0", "1", "false", "true", "off", "on"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", ""}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}.")


def _coerce_scalar(name: str, value: Any) -> Any:
    if name in _BOOLEAN_FIELDS:
        return _coerce_bool(value)
    if name in _INTEGER_FIELDS:
        parsed = int(value)
        if name == "vram_safety_margin_mb" and parsed < 0:
            raise ValueError("vram_safety_margin_mb must be non-negative.")
        if name == "oom_retry_limit" and parsed < 0:
            raise ValueError("oom_retry_limit must be non-negative.")
        return parsed
    if name == "attention_backend":
        selected = str(value).strip().lower()
        selected = _ATTENTION_BACKEND_ALIASES.get(selected, selected)
        if selected not in _ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be one of: auto, default, eager, sdpa, xformers."
            )
        return selected
    if name == "attention_slicing":
        selected = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if selected not in _ATTENTION_SLICING_MODES:
            raise ValueError("attention_slicing must be one of: off, auto, max.")
        return selected
    if name == "vae_device":
        selected = str(value).strip().lower()
        if selected not in _VAE_DEVICES:
            raise ValueError("vae_device must be one of: auto, cuda, cpu.")
        return selected
    if name == "memory_policy":
        selected = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        selected = _MEMORY_POLICY_ALIASES.get(selected, selected)
        if selected not in _MEMORY_POLICIES:
            raise ValueError(
                "memory_policy must be one of: auto, high_vram, balanced, "
                "low_vram, cpu_fallback."
            )
        return selected
    if name == "hires_memory_profile":
        selected = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        selected = _HIRES_MEMORY_PROFILE_ALIASES.get(selected, selected)
        if selected not in _HIRES_MEMORY_PROFILES:
            raise ValueError(
                "hires_memory_profile must be one of: inherit, balanced, "
                "low_vram, maximum."
            )
        return selected
    if name == "preview_policy":
        selected = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if selected not in _PREVIEW_POLICIES:
            raise ValueError(
                "preview_policy must be one of: normal, suspend_on_pressure, "
                "disable_during_hires, disabled."
            )
        return selected
    if name == "oom_retry_profile":
        selected = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        selected = _OOM_RETRY_PROFILE_ALIASES.get(selected, selected)
        if selected not in _OOM_RETRY_PROFILES:
            raise ValueError(
                "oom_retry_profile must be one of: disabled, cleanup, low_vram, maximum."
            )
        return selected
    return str(value)


def _coerce_mslk_value(name: str, value: Any) -> str:
    text = str(value)
    if name in _MSLK_POSITIVE_INTEGER_FIELDS:
        if text == "":
            return ""
        try:
            parsed = int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mslk_fmha.{name} must be blank or a positive integer."
            ) from exc
        if parsed <= 0:
            raise ValueError(
                f"mslk_fmha.{name} must be greater than zero when supplied."
            )
        return str(parsed)
    normalized = text.strip().lower() if text else ""
    if name == "policy" and normalized not in _MSLK_POLICIES:
        raise ValueError(
            "mslk_fmha.policy must be blank or one of: auto, benchmark, "
            "blackwell_safe, default, env, off."
        )
    if name == "debug" and normalized not in _MSLK_DEBUG_VALUES:
        raise ValueError(
            "mslk_fmha.debug must be blank or one of: 0, 1, false, true, off, on."
        )
    return normalized if name in {"policy", "debug"} else text


def _normalize_source_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in dict(values or {}).items():
        key = _SETTING_ALIASES.get(str(raw_key), str(raw_key))
        if key in _RUNTIME_SCALAR_FIELDS:
            normalized[key] = value
        elif key == "mslk_fmha" and isinstance(value, Mapping):
            normalized[key] = dict(value)
        elif key == "allocator_options" and isinstance(value, Mapping):
            normalized[key] = dict(value)
        elif key == "cuda_expandable_segments":
            normalized[key] = value
    return normalized


def _environment_mapping(environment: Mapping[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for env_name, field_name in _ENVIRONMENT_FIELDS.items():
        if env_name in environment:
            normalized[field_name] = environment[env_name]

    mslk_values: dict[str, str] = {}
    for env_name, field_name in _MSLK_ENVIRONMENT_FIELDS.items():
        if env_name in environment:
            # Blank MSLK values are meaningful and must survive normalization.
            mslk_values[field_name] = str(environment[env_name])
    if mslk_values:
        normalized["mslk_fmha"] = mslk_values

    allocator_value = environment.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE)
    if allocator_value is not None:
        normalized["allocator_options"] = {
            CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE: canonicalize_cuda_allocator_conf(
                allocator_value
            )
        }

    inherited_json = str(
        environment.get("IMAGE_GEN_RUNTIME_STARTUP_OPTIONS", "") or ""
    ).strip()
    if inherited_json:
        try:
            inherited = json.loads(inherited_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "IMAGE_GEN_RUNTIME_STARTUP_OPTIONS must contain a JSON object."
            ) from exc
        if not isinstance(inherited, dict):
            raise ValueError(
                "IMAGE_GEN_RUNTIME_STARTUP_OPTIONS must contain a JSON object."
            )
        # Explicit dedicated environment variables above override this transport
        # object, so merge the transport first.
        transported = _normalize_source_mapping(inherited)
        transported.update(normalized)
        normalized = transported
    return normalized


def resolve_runtime_startup_options(
    *,
    explicit_cli: Mapping[str, Any] | None = None,
    commandline_args: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    saved_profile: Mapping[str, Any] | None = None,
    runtime_profile: Mapping[str, Any] | None = None,
    runtime_profile_selection: RuntimeProfileSelection | Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> RuntimeStartupOptions:
    """Resolve the Phase 14K startup schema using the documented precedence.

    Sources are applied from lowest to highest priority:

    defaults -> settings -> saved profile -> runtime profile -> environment ->
    COMMANDLINE_ARGS -> explicit CLI.

    The command-line mappings are intentionally accepted as already parsed
    values.  BAT files control token order, while later Phase 14K subphases add
    the concrete argument definitions to the shared parsers.
    """

    base = RuntimeStartupOptions().to_dict()
    source_map = {
        key: "default"
        for key in _RUNTIME_SCALAR_FIELDS
    }
    for key in MSLKFMHAOptions().to_dict():
        source_map[f"mslk_fmha.{key}"] = "default"
    source_map["allocator_options"] = "default"
    source_map[
        f"allocator_options.{CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE}"
    ] = "default"
    source_map["allocator_options.expandable_segments"] = "default"

    scalar_values = {
        key: base[key]
        for key in _RUNTIME_SCALAR_FIELDS
    }
    mslk_values = dict(base["mslk_fmha"])
    allocator_options = dict(base["allocator_options"])

    sources: list[tuple[str, Mapping[str, Any] | None]] = [
        ("default", defaults),
        ("settings", settings),
        ("saved_profile", saved_profile),
        ("runtime_profile", runtime_profile),
        (
            "environment",
            _environment_mapping(
                environment if environment is not None else os.environ
            ),
        ),
        ("commandline_args", commandline_args),
        ("cli", explicit_cli),
    ]

    for source_name, raw_values in sources:
        values = _normalize_source_mapping(raw_values)
        for key in _RUNTIME_SCALAR_FIELDS:
            if key not in values or values[key] is None:
                continue
            scalar_values[key] = _coerce_scalar(key, values[key])
            source_map[key] = source_name

        nested_mslk = values.get("mslk_fmha")
        if isinstance(nested_mslk, Mapping):
            for key, value in nested_mslk.items():
                if key not in mslk_values or value is None:
                    continue
                mslk_values[key] = _coerce_mslk_value(str(key), value)
                source_map[f"mslk_fmha.{key}"] = source_name

        nested_allocator = values.get("allocator_options")
        if isinstance(nested_allocator, Mapping):
            for raw_key, raw_value in nested_allocator.items():
                key = str(raw_key)
                if key == CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE:
                    allocator_options[key] = canonicalize_cuda_allocator_conf(raw_value)
                    source_map[
                        f"allocator_options.{CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE}"
                    ] = source_name
                else:
                    allocator_options[key] = str(raw_value)
                source_map["allocator_options"] = source_name

        if "cuda_expandable_segments" in values:
            enabled = _coerce_bool(values.get("cuda_expandable_segments"))
            current_conf = allocator_options.get(
                CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, ""
            )
            allocator_options[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = (
                set_cuda_allocator_option(
                    current_conf,
                    "expandable_segments",
                    "True" if enabled else "False",
                )
            )
            source_map["allocator_options"] = source_name
            source_map[
                f"allocator_options.{CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE}"
            ] = source_name
            source_map["allocator_options.expandable_segments"] = source_name

    memory_policy = str(scalar_values["memory_policy"])
    retention_defaults = _MEMORY_POLICY_RETENTION_DEFAULTS.get(memory_policy)
    if retention_defaults:
        policy_source = source_map.get("memory_policy", "default")
        policy_priority = _SOURCE_PRIORITY.get(policy_source, 0)
        for field_name, default_value in retention_defaults.items():
            field_source = source_map.get(field_name, "default")
            if _SOURCE_PRIORITY.get(field_source, 0) < policy_priority:
                scalar_values[field_name] = default_value
                source_map[field_name] = f"memory_policy:{policy_source}"

    selected_profile = (
        runtime_profile_selection
        if isinstance(runtime_profile_selection, RuntimeProfileSelection)
        else RuntimeProfileSelection.from_mapping(runtime_profile_selection)
    )

    return RuntimeStartupOptions(
        runtime_profile=selected_profile,
        attention_backend=str(scalar_values["attention_backend"]),
        memory_policy=memory_policy,
        vram_safety_margin_mb=int(scalar_values["vram_safety_margin_mb"]),
        attention_slicing=str(scalar_values["attention_slicing"]),
        vae_tiling=bool(scalar_values["vae_tiling"]),
        vae_slicing=bool(scalar_values["vae_slicing"]),
        vae_device=str(scalar_values["vae_device"]),
        retain_unet_between_jobs=bool(
            scalar_values["retain_unet_between_jobs"]
        ),
        retain_vae_between_jobs=bool(
            scalar_values["retain_vae_between_jobs"]
        ),
        retain_text_encoder_between_jobs=bool(
            scalar_values["retain_text_encoder_between_jobs"]
        ),
        preview_policy=str(scalar_values["preview_policy"]),
        hires_memory_profile=str(scalar_values["hires_memory_profile"]),
        pre_hires_cleanup=bool(scalar_values["pre_hires_cleanup"]),
        oom_retry_profile=str(scalar_values["oom_retry_profile"]),
        oom_retry_limit=int(scalar_values["oom_retry_limit"]),
        mslk_fmha=MSLKFMHAOptions.from_mapping(mslk_values),
        allocator_options=allocator_options,
        source_map=source_map,
    )


def merge_runtime_startup_settings(
    options: RuntimeStartupOptions,
    settings: Mapping[str, Any] | None,
) -> RuntimeStartupOptions:
    """Fill default-sourced fields from application/project settings.

    WebUI startup must normalize process-level arguments before importing the
    application, but application settings are not available until the store is
    opened. This merge preserves higher-precedence source labels and only fills
    fields that still came from built-in defaults.
    """

    settings_options = resolve_runtime_startup_options(
        environment={},
        settings=settings,
    )
    payload = options.to_dict()
    source_map = dict(payload.get("source_map") or {})
    settings_payload = settings_options.to_dict()
    settings_source_map = dict(settings_payload.get("source_map") or {})

    for key in _RUNTIME_SCALAR_FIELDS:
        if source_map.get(key, "default") != "default":
            continue
        settings_source = settings_source_map.get(key)
        if settings_source not in {"settings", "memory_policy:settings"}:
            continue
        payload[key] = settings_payload[key]
        source_map[key] = str(settings_source)

    mslk_payload = dict(payload.get("mslk_fmha") or {})
    settings_mslk = dict(settings_payload.get("mslk_fmha") or {})
    for key in mslk_payload:
        source_key = f"mslk_fmha.{key}"
        if source_map.get(source_key, "default") != "default":
            continue
        if settings_source_map.get(source_key) != "settings":
            continue
        mslk_payload[key] = settings_mslk[key]
        source_map[source_key] = "settings"
    payload["mslk_fmha"] = mslk_payload

    if (
        source_map.get("allocator_options", "default") == "default"
        and settings_source_map.get("allocator_options") == "settings"
    ):
        payload["allocator_options"] = dict(
            settings_payload.get("allocator_options") or {}
        )
        source_map["allocator_options"] = "settings"
        allocator_source_key = (
            f"allocator_options.{CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE}"
        )
        source_map[allocator_source_key] = settings_source_map.get(
            allocator_source_key, "settings"
        )
        source_map["allocator_options.expandable_segments"] = (
            settings_source_map.get(
                "allocator_options.expandable_segments",
                "embedded_in_config",
            )
        )

    payload["source_map"] = source_map
    return RuntimeStartupOptions.from_mapping(payload)


def runtime_request_settings(
    options: RuntimeStartupOptions | Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the startup contract into canonical generation/runtime extras."""

    resolved = (
        options
        if isinstance(options, RuntimeStartupOptions)
        else RuntimeStartupOptions.from_mapping(options)
    )
    policy = str(resolved.memory_policy)
    execution_device = "cpu" if policy == "cpu_fallback" else "cuda_preferred"
    retention_device = "cpu" if policy == "cpu_fallback" else "cuda"
    return {
        "runtime_profile": resolved.runtime_profile.to_dict(),
        "memory_policy": policy,
        "memory_vram_safety_margin_mb": int(resolved.vram_safety_margin_mb),
        "memory_retain_checkpoint_between_jobs": bool(
            resolved.retain_unet_between_jobs
        ),
        "memory_retain_vae_between_jobs": bool(resolved.retain_vae_between_jobs),
        "model_runtime_retain_text_encoder_between_jobs": bool(
            resolved.retain_text_encoder_between_jobs
        ),
        "model_runtime_execution_device": execution_device,
        "model_runtime_retention_device": retention_device,
        "attention_slicing": str(resolved.attention_slicing),
        "vae_tiling": bool(resolved.vae_tiling),
        "vae_slicing": bool(resolved.vae_slicing),
        "vae_device": str(resolved.vae_device),
        "preview_policy": str(resolved.preview_policy),
        "hires_memory_profile": str(resolved.hires_memory_profile),
        "pre_hires_cleanup": bool(resolved.pre_hires_cleanup),
        "oom_retry_profile": str(resolved.oom_retry_profile),
        "oom_retry_limit": int(resolved.oom_retry_limit),
        "cuda_allocator_environment": dict(resolved.allocator_options),
        "cuda_allocator_diagnostics": build_cuda_allocator_diagnostics(resolved),
        "runtime_startup_options": resolved.to_dict(),
    }


def apply_runtime_startup_environment(
    options: RuntimeStartupOptions,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply import-time runtime values before CUDA/attention modules import."""

    target = environment if environment is not None else os.environ
    # The allocator environment is applied first and this module does not import
    # Torch. Entry points call this function before importing CUDA/attention code.
    apply_cuda_allocator_environment(
        options.allocator_options,
        environment=target,
    )
    applied = {
        "IMAGE_GEN_RUNTIME_PROFILE_ID": str(options.runtime_profile.profile_id),
        "IMAGE_GEN_RUNTIME_PROFILE_SCHEMA_VERSION": str(options.runtime_profile.schema_version),
        "IMAGE_GEN_RUNTIME_PROFILE_SOURCE": str(options.runtime_profile.source),
        "IMAGE_GEN_RUNTIME_PROFILE_SELECTION": json.dumps(
            options.runtime_profile.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "IMAGE_GEN_RUNTIME_STARTUP_OPTIONS": json.dumps(
            options.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "IMAGE_GEN_ATTENTION_BACKEND": str(options.attention_backend),
        "IMAGE_GEN_MEMORY_POLICY": str(options.memory_policy),
        "IMAGE_GEN_VRAM_SAFETY_MARGIN_MB": str(options.vram_safety_margin_mb),
        "IMAGE_GEN_ATTENTION_SLICING": str(options.attention_slicing),
        "IMAGE_GEN_VAE_TILING": "1" if options.vae_tiling else "0",
        "IMAGE_GEN_VAE_SLICING": "1" if options.vae_slicing else "0",
        "IMAGE_GEN_VAE_DEVICE": str(options.vae_device),
        "IMAGE_GEN_RETAIN_UNET_BETWEEN_JOBS": (
            "1" if options.retain_unet_between_jobs else "0"
        ),
        "IMAGE_GEN_RETAIN_VAE_BETWEEN_JOBS": (
            "1" if options.retain_vae_between_jobs else "0"
        ),
        "IMAGE_GEN_RETAIN_TEXT_ENCODER_BETWEEN_JOBS": (
            "1" if options.retain_text_encoder_between_jobs else "0"
        ),
        "IMAGE_GEN_PREVIEW_POLICY": str(options.preview_policy),
        "IMAGE_GEN_HIRES_MEMORY_PROFILE": str(options.hires_memory_profile),
        "IMAGE_GEN_PRE_HIRES_CLEANUP": (
            "1" if options.pre_hires_cleanup else "0"
        ),
        "IMAGE_GEN_OOM_RETRY_PROFILE": str(options.oom_retry_profile),
        "IMAGE_GEN_OOM_RETRY_LIMIT": str(options.oom_retry_limit),
        "MSLK_FMHA_POLICY": str(options.mslk_fmha.policy),
        "MSLK_FMHA_DEBUG": str(options.mslk_fmha.debug),
        "MSLK_FMHA_BLOCK_N": str(options.mslk_fmha.block_n),
        "MSLK_FMHA_BLOCK_M": str(options.mslk_fmha.block_m),
        "MSLK_FMHA_NUM_WARPS": str(options.mslk_fmha.num_warps),
        "MSLK_FMHA_NUM_STAGES": str(options.mslk_fmha.num_stages),
        "MSLK_FMHA_EXPERIMENTAL_HEAD_DIMS": str(
            options.mslk_fmha.experimental_head_dims
        ),
    }
    for key, value in applied.items():
        target[key] = value
    if CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in target:
        applied[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = str(
            target[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE]
        )
    return applied
