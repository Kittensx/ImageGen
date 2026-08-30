from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import os

from modules.registry.component_selection import canonical_model_family
from modules.registry.contracts import (
    AVAILABILITY_AVAILABLE,
    SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
    SOURCE_FORM_PHYSICAL_COMPONENT,
    SOURCE_FORM_RECONSTRUCTED_EXPORT,
    SOURCE_FORM_STANDALONE_SHARED,
)


RUNTIME_COMPONENT_SOURCE_SCHEMA_VERSION = 1


def _digest(value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _resolved_path(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        return str(Path(token).expanduser().resolve())
    except OSError:
        return token


def _path_key(value: Any) -> str:
    token = _resolved_path(value)
    return os.path.normcase(token) if token else ""


def _entry_value(entry: Any, name: str, default: Any = "") -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def registry_source_payloads(registry: Any, component_sha256: str) -> list[dict[str, Any]]:
    """Return normalized registered occurrences for one exact component fingerprint.

    This is a read-only projection over registry evidence.  It performs no scan and no
    hashing.  Missing asset rows are ignored because they cannot safely hydrate a role.
    """

    digest = _digest(component_sha256)
    if not digest or registry is None:
        return []
    list_sources = getattr(registry, "list_component_sources", None)
    get_asset = getattr(registry, "get_asset_by_id", None)
    if not callable(list_sources) or not callable(get_asset):
        return []
    payloads: list[dict[str, Any]] = []
    for source in list_sources(component_sha256=digest, limit=1_000_000):
        asset = get_asset(int(getattr(source, "asset_id", 0) or 0))
        if asset is None:
            continue
        try:
            locator = json.loads(getattr(source, "locator_json", "") or "{}")
        except Exception:
            locator = {}
        payloads.append({
            "component_sha256": digest,
            "asset_id": int(getattr(source, "asset_id", 0) or 0),
            "path": str(getattr(asset, "path", "") or ""),
            "filename": str(getattr(asset, "filename", "") or ""),
            "asset_type": str(getattr(asset, "asset_type", "") or ""),
            "architecture": str(getattr(asset, "architecture", "") or ""),
            "component_role": str(getattr(source, "component_role", "") or ""),
            "source_form": str(getattr(source, "source_form", "") or "unknown"),
            "embedded_state": str(getattr(source, "embedded_state", "") or "unknown"),
            "provider_family": str(getattr(source, "provider_family", "") or ""),
            "provider_version": str(getattr(source, "provider_version", "") or ""),
            "availability_state": str(getattr(source, "availability_state", "") or "unknown"),
            "exists_on_disk": bool(getattr(asset, "exists_on_disk", False)),
            "locator": dict(locator) if isinstance(locator, dict) else {},
            "snapshot_version": str(getattr(source, "snapshot_version", "") or ""),
        })
    return payloads


def _registry_candidate_allowed(
    source: Mapping[str, Any],
    *,
    role: str,
    family: str,
    provider_version: str,
    allow_digital_components: bool,
) -> bool:
    if str(source.get("availability_state") or "").strip().lower() != AVAILABILITY_AVAILABLE:
        return False
    if not bool(source.get("exists_on_disk", True)):
        return False
    source_role = str(source.get("component_role") or role)
    if source_role and source_role != role:
        return False
    source_form = str(source.get("source_form") or "unknown")
    if source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT and not allow_digital_components:
        return False
    canonical = canonical_model_family(family)
    source_family = canonical_model_family(source.get("provider_family") or source.get("architecture"))
    if source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT or str(source.get("asset_type") or "").lower() == "checkpoint":
        if canonical and source_family != canonical:
            return False
    elif source_family and canonical and source_family != canonical:
        return False
    source_provider_version = str(source.get("provider_version") or "")
    if source_provider_version and provider_version and source_provider_version != provider_version:
        return False
    return True


def _registry_cost(source: Mapping[str, Any], active_transaction_paths: set[str]) -> tuple[int, str, str]:
    path = _path_key(source.get("path"))
    source_form = str(source.get("source_form") or "unknown")
    if path and path in active_transaction_paths:
        return 20, "active_load_transaction", "reuse_state_from_active_load_transaction"
    if source_form == SOURCE_FORM_PHYSICAL_COMPONENT:
        return 30, "physical_component", "load_direct_physical_component"
    if source_form == SOURCE_FORM_STANDALONE_SHARED:
        return 35, "standalone_component", "load_direct_standalone_component"
    if source_form == SOURCE_FORM_RECONSTRUCTED_EXPORT:
        return 40, "reconstructed_export", "load_reconstructed_component_export"
    if source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT:
        return 50, "digital_checkpoint_component", "selective_checkpoint_component_hydration"
    return 90, "unknown_registry_source", "registry_source_fallback"


def resolve_runtime_component_source(
    *,
    component_sha256: str,
    role: str,
    family: str,
    provider_version: str,
    resident_entries: Mapping[str, Any] | None = None,
    lease_entries_by_sha: Mapping[str, Any] | None = None,
    registry_sources: Iterable[Mapping[str, Any]] = (),
    required_device: str = "cpu",
    allow_digital_components: bool = True,
    allow_resident_components: bool = True,
    active_transaction_paths: Iterable[str] = (),
    source_override_path: str = "",
) -> dict[str, Any]:
    """Select the cheapest safe representation for an exact component identity.

    Ranking is intentionally heuristic and side-effect free.  It never benchmarks,
    scans, hashes, or changes component identity.  A resident exact live handle always
    outranks disk when its family/provider contract and mutation state remain reusable.
    """

    digest = _digest(component_sha256)
    canonical_family = canonical_model_family(family)
    required = str(required_device or "cpu").strip().lower()
    active_paths = {_path_key(item) for item in active_transaction_paths if _path_key(item)}
    override = _path_key(source_override_path)
    candidates: list[dict[str, Any]] = []

    entry = dict(resident_entries or {}).get(role)
    if allow_resident_components and entry is not None:
        entry_digest = _digest(_entry_value(entry, "component_sha256"))
        entry_family = canonical_model_family(_entry_value(entry, "family"))
        entry_provider_version = str(_entry_value(entry, "provider_version") or "")
        if (
            entry_digest == digest
            and bool(_entry_value(entry, "reuse_eligible", False))
            and (not canonical_family or entry_family == canonical_family)
            and (not provider_version or entry_provider_version == provider_version)
        ):
            device = str(_entry_value(entry, "device", "unknown") or "unknown").lower()
            if required.startswith("cuda") and device.startswith("cuda"):
                cost = 0
                kind = "resident_required_device"
                strategy = "reuse_live_resident_component"
            elif device.startswith("cpu"):
                cost = 5
                kind = "resident_cpu"
                strategy = "reuse_live_resident_component_then_promote_if_needed"
            else:
                cost = 8
                kind = "resident_restageable"
                strategy = "reuse_live_resident_component_then_restage"
            resident_occurrence = dict(_entry_value(entry, "source", {}) or {})
            if not resident_occurrence.get("path") and resident_occurrence.get("source_path"):
                resident_occurrence["path"] = resident_occurrence.get("source_path")
            resident_occurrence.setdefault("component_role", role)
            resident_occurrence.setdefault("component_sha256", digest)
            candidates.append({
                "cost_rank": cost,
                "source_kind": kind,
                "load_strategy": strategy,
                "reason": "exact_component_already_resident_and_reusable",
                "component_sha256": digest,
                "role": role,
                "device": device,
                "runtime_object_id": id(_entry_value(entry, "module", None)) if _entry_value(entry, "module", None) is not None else None,
                "occurrence": resident_occurrence,
            })

    lease_map = dict(lease_entries_by_sha or {})
    lease_entry = lease_map.get(f"{role}:{digest}") or lease_map.get(digest)
    if allow_resident_components and lease_entry is not None:
        entry_digest = _digest(_entry_value(lease_entry, "component_sha256"))
        entry_family = canonical_model_family(_entry_value(lease_entry, "family"))
        entry_provider_version = str(_entry_value(lease_entry, "provider_version") or "")
        if (
            entry_digest == digest
            and bool(_entry_value(lease_entry, "reuse_eligible", False))
            and (not canonical_family or entry_family == canonical_family)
            and (not provider_version or entry_provider_version == provider_version)
        ):
            module = _entry_value(lease_entry, "module", None)
            device = str(_entry_value(lease_entry, "device", "unknown") or "unknown").lower()
            if module is not None:
                if required.startswith("cuda") and device.startswith("cuda"):
                    cost = 1
                    kind = "lease_warm_required_device"
                    strategy = "reuse_leased_component_on_required_device"
                elif device.startswith("cpu"):
                    cost = 6
                    kind = "lease_warm_cpu"
                    strategy = "reuse_leased_cpu_component_then_promote_if_needed"
                else:
                    cost = 9
                    kind = "lease_warm_restageable"
                    strategy = "reuse_leased_component_then_restage"
                occurrence = dict(_entry_value(lease_entry, "source", {}) or {})
                occurrence.setdefault("component_role", role)
                occurrence.setdefault("component_sha256", digest)
                occurrence.setdefault("path", str(_entry_value(lease_entry, "source_model_path", "") or ""))
                candidates.append({
                    "cost_rank": cost,
                    "source_kind": kind,
                    "load_strategy": strategy,
                    "reason": "exact_component_already_warm_under_execution_lease",
                    "component_sha256": digest,
                    "role": role,
                    "device": device,
                    "runtime_object_id": id(module),
                    "occurrence": occurrence,
                })

    for raw in registry_sources:
        source = dict(raw or {})
        if _digest(source.get("component_sha256") or digest) not in {"", digest}:
            continue
        if not _registry_candidate_allowed(
            source,
            role=role,
            family=canonical_family,
            provider_version=provider_version,
            allow_digital_components=allow_digital_components,
        ):
            continue
        path = _path_key(source.get("path"))
        if override and path != override:
            continue
        cost, kind, strategy = _registry_cost(source, active_paths)
        candidates.append({
            "cost_rank": cost,
            "source_kind": kind,
            "load_strategy": strategy,
            "reason": (
                "expert_source_override"
                if override
                else "active_transaction_contains_exact_component"
                if kind == "active_load_transaction"
                else "lowest_known_safe_source_cost"
            ),
            "component_sha256": digest,
            "role": role,
            "device": "disk",
            "runtime_object_id": None,
            "occurrence": source,
        })

    candidates.sort(key=lambda item: (
        int(item.get("cost_rank", 999)),
        str(dict(item.get("occurrence") or {}).get("filename") or "").casefold(),
        str(dict(item.get("occurrence") or {}).get("path") or "").casefold(),
    ))
    selected = dict(candidates[0]) if candidates else {}
    fallbacks = [dict(item) for item in candidates[1:]]
    return {
        "schema_version": RUNTIME_COMPONENT_SOURCE_SCHEMA_VERSION,
        "component_sha256": digest,
        "role": role,
        "family": canonical_family,
        "provider_version": str(provider_version or ""),
        "required_device": required,
        "selected": selected,
        "selected_source_kind": str(selected.get("source_kind") or "unavailable"),
        "selected_occurrence": dict(selected.get("occurrence") or {}),
        "cost_class": int(selected.get("cost_rank", 999)) if selected else None,
        "load_strategy": str(selected.get("load_strategy") or "unavailable"),
        "reason": str(selected.get("reason") or "no_safe_available_source"),
        "fallback_candidates": fallbacks,
        "candidate_count": len(candidates),
        "identity_preserved": bool(digest),
    }


def public_runtime_source_plan(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(plan or {})
    roles: dict[str, Any] = {}
    for role, item in dict(payload.get("roles") or {}).items():
        public_item = dict(item or {})
        selected = dict(public_item.get("selected") or {})
        selected.pop("module", None)
        public_item["selected"] = selected
        roles[str(role)] = public_item
    payload["roles"] = roles
    return payload


__all__ = [
    "RUNTIME_COMPONENT_SOURCE_SCHEMA_VERSION",
    "registry_source_payloads",
    "resolve_runtime_component_source",
    "public_runtime_source_plan",
]
