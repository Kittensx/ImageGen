from __future__ import annotations

import hashlib
import os
import sys
from collections import OrderedDict
from typing import Any, Mapping, MutableMapping

CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_ALLOCATOR_LIMITATION = (
    "CUDA allocator tuning can reduce fragmentation and improve reuse, but it "
    "cannot satisfy a single allocation that is larger than the available VRAM."
)

_LAST_BOOTSTRAP_STATUS: dict[str, Any] | None = None


def _entry_identity(key: str) -> str:
    return str(key).strip().lower()


def _split_top_level_entries(text: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    closing_for = {"[": "]", "(": ")", "{": "}"}
    opening_for = {value: key for key, value in closing_for.items()}

    for character in text:
        if character in closing_for:
            stack.append(character)
        elif character in opening_for:
            if not stack or stack[-1] != opening_for[character]:
                raise ValueError(
                    "CUDA allocator configuration contains mismatched brackets."
                )
            stack.pop()
        if character == "," and not stack:
            entries.append("".join(current))
            current = []
        else:
            current.append(character)

    if stack:
        raise ValueError(
            "CUDA allocator configuration contains an unclosed bracketed value."
        )
    entries.append("".join(current))
    return entries


def parse_cuda_allocator_conf(value: Any) -> list[tuple[str, str]]:
    """Parse a PyTorch CUDA allocator configuration without importing Torch.

    PyTorch accepts a comma-separated ``name:value`` string. IMAGE_GEN keeps
    unknown option names so newer PyTorch allocator switches remain usable, but
    rejects malformed entries early enough to avoid a confusing CUDA startup
    failure later in the process.
    """

    text = "" if value is None else str(value).strip()
    if not text:
        return []

    parsed: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for raw_segment in _split_top_level_entries(text):
        segment = raw_segment.strip()
        if not segment:
            raise ValueError(
                "CUDA allocator configuration contains an empty comma-separated entry."
            )
        if ":" not in segment:
            raise ValueError(
                "CUDA allocator entries must use name:value syntax; "
                f"received {segment!r}."
            )
        raw_key, raw_value = segment.split(":", 1)
        key = raw_key.strip()
        item_value = raw_value.strip()
        if not key or not item_value:
            raise ValueError(
                "CUDA allocator entries require both a non-empty name and value; "
                f"received {segment!r}."
            )
        if any(character in key for character in (" ", "\t", "\r", "\n", ",", ":")):
            raise ValueError(
                f"CUDA allocator option name {key!r} contains unsupported "
                "whitespace or punctuation."
            )
        identity = _entry_identity(key)
        normalized_key = "expandable_segments" if identity == "expandable_segments" else key
        if identity in parsed:
            # Last value wins, matching ordinary environment option precedence.
            del parsed[identity]
        parsed[identity] = (normalized_key, item_value)
    return list(parsed.values())


def canonicalize_cuda_allocator_conf(value: Any) -> str:
    return ",".join(f"{key}:{item_value}" for key, item_value in parse_cuda_allocator_conf(value))


def set_cuda_allocator_option(value: Any, key: str, option_value: Any) -> str:
    entries = OrderedDict(
        (_entry_identity(entry_key), (entry_key, entry_value))
        for entry_key, entry_value in parse_cuda_allocator_conf(value)
    )
    identity = _entry_identity(key)
    if identity in entries:
        del entries[identity]
    normalized_key = (
        "expandable_segments"
        if identity == "expandable_segments"
        else str(key).strip()
    )
    normalized_value = str(option_value).strip()
    if not normalized_key or not normalized_value:
        raise ValueError("CUDA allocator option names and values must be non-empty.")
    entries[identity] = (normalized_key, normalized_value)
    return ",".join(
        f"{entry_key}:{entry_value}"
        for entry_key, entry_value in entries.values()
    )


def cuda_allocator_entries(value: Any) -> dict[str, str]:
    return {
        str(key): str(item_value)
        for key, item_value in parse_cuda_allocator_conf(value)
    }


def cuda_allocator_fingerprint(value: Any) -> str:
    canonical = canonicalize_cuda_allocator_conf(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _torch_initialization_state() -> tuple[bool, bool | None]:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False, False
    cuda_module = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda_module, "is_initialized", None)
    if not callable(is_initialized):
        return True, None
    try:
        return True, bool(is_initialized())
    except Exception:
        return True, None


def apply_cuda_allocator_environment(
    allocator_options: Mapping[str, Any] | None,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the allocator environment and record whether startup timing was safe."""

    global _LAST_BOOTSTRAP_STATUS

    target = environment if environment is not None else os.environ
    options = dict(allocator_options or {})
    configured = CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in options
    requested = (
        canonicalize_cuda_allocator_conf(options.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, ""))
        if configured
        else None
    )

    torch_imported_before_apply, cuda_initialized_before_apply = _torch_initialization_state()
    if configured:
        target[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = str(requested or "")

    effective_raw = str(target.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, "") or "")
    effective = canonicalize_cuda_allocator_conf(effective_raw)
    if effective_raw != effective and CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in target:
        target[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = effective

    warnings: list[str] = []
    if configured and cuda_initialized_before_apply is True:
        warnings.append(
            "The CUDA runtime was already initialized before the allocator "
            "environment was applied; restart the process for allocator changes "
            "to take effect."
        )
    elif configured and torch_imported_before_apply:
        warnings.append(
            "Torch was already imported before the allocator environment was applied. CUDA was not "
            "reported as initialized, but process-start application is recommended."
        )

    _LAST_BOOTSTRAP_STATUS = {
        "schema_version": 1,
        "environment_variable": CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
        "configured_by_runtime_options": bool(configured),
        "requested_config": requested,
        "effective_config": effective,
        "effective_entries": cuda_allocator_entries(effective),
        "fingerprint": cuda_allocator_fingerprint(effective),
        "torch_imported_before_apply": bool(torch_imported_before_apply),
        "cuda_initialized_before_apply": cuda_initialized_before_apply,
        "applied_before_torch_import": not torch_imported_before_apply,
        "applied_before_cuda_initialization": cuda_initialized_before_apply is not True,
        "restart_required_for_late_change": bool(
            configured and cuda_initialized_before_apply
        ),
        "warnings": warnings,
        "limitation": CUDA_ALLOCATOR_LIMITATION,
    }
    return dict(_LAST_BOOTSTRAP_STATUS)


def last_cuda_allocator_bootstrap_status() -> dict[str, Any]:
    return dict(_LAST_BOOTSTRAP_STATUS or {})


def build_cuda_allocator_diagnostics(
    options: Any | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    target = environment if environment is not None else os.environ
    options_payload = options.to_dict() if hasattr(options, "to_dict") else dict(options or {})
    allocator_options = dict(options_payload.get("allocator_options") or {})
    source_map = dict(options_payload.get("source_map") or {})

    requested_present = CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in allocator_options
    requested = (
        canonicalize_cuda_allocator_conf(
            allocator_options.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, "")
        )
        if requested_present
        else None
    )
    effective = canonicalize_cuda_allocator_conf(
        target.get(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, requested or "")
    )
    entries = cuda_allocator_entries(effective)
    expandable_raw = entries.get("expandable_segments")
    expandable_enabled = (
        str(expandable_raw).strip().lower() in {"1", "true", "yes", "on"}
        if expandable_raw is not None
        else False
    )

    bootstrap = last_cuda_allocator_bootstrap_status()
    if bootstrap and bootstrap.get("effective_config") != effective:
        bootstrap = {
            **bootstrap,
            "matches_current_environment": False,
            "current_environment_config": effective,
        }
    elif bootstrap:
        bootstrap["matches_current_environment"] = True

    request_matches_effective = bool(
        requested is None or requested == effective
    )
    warnings = list(bootstrap.get("warnings") or []) if bootstrap else []
    if requested_present and not request_matches_effective:
        warnings.append(
            "The requested CUDA allocator configuration does not match the active "
            "process environment. Restart before CUDA initialization to apply it."
        )
    restart_required = bool(
        (requested_present and not request_matches_effective)
        or (bootstrap and bootstrap.get("restart_required_for_late_change"))
    )

    return {
        "schema_version": 1,
        "environment_variable": CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
        "requested_config": requested,
        "effective_config": effective,
        "effective_environment": (
            {CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE: effective}
            if CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE in target or requested_present
            else {}
        ),
        "effective_entries": entries,
        "expandable_segments_enabled": bool(expandable_enabled),
        "fingerprint": cuda_allocator_fingerprint(effective),
        "request_matches_effective": request_matches_effective,
        "restart_required": restart_required,
        "immutable_after_cuda_initialization": True,
        "warnings": warnings,
        "source": source_map.get(
            f"allocator_options.{CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE}",
            source_map.get("allocator_options", "default"),
        ),
        "expandable_segments_source": source_map.get(
            "allocator_options.expandable_segments",
            "embedded_in_config" if expandable_raw is not None else "default",
        ),
        "bootstrap": bootstrap,
        "limitation": CUDA_ALLOCATOR_LIMITATION,
    }


__all__ = [
    "CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE",
    "CUDA_ALLOCATOR_LIMITATION",
    "apply_cuda_allocator_environment",
    "build_cuda_allocator_diagnostics",
    "canonicalize_cuda_allocator_conf",
    "cuda_allocator_entries",
    "cuda_allocator_fingerprint",
    "last_cuda_allocator_bootstrap_status",
    "parse_cuda_allocator_conf",
    "set_cuda_allocator_option",
]
