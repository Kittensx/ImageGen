from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from image_gen.runtime.component_residency import build_resident_component_inventory
from image_gen.runtime.component_source_selection import resolve_runtime_component_source


COMPOSITION_TRANSITION_PLAN_SCHEMA_VERSION = 1


def _digest(value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _module_device(module: Any) -> str:
    if module is None:
        return "missing"
    try:
        return str(next(module.parameters()).device)
    except (StopIteration, AttributeError, TypeError):
        return str(getattr(module, "device", "unknown"))


@dataclass(frozen=True)
class CompositionTransitionPlan:
    lease_generation: int
    from_index: int
    to_index: int
    from_model_path: str
    to_model_path: str
    from_composition_sha256: str
    to_composition_sha256: str
    family: str
    provider_version: str
    role_diff: Mapping[str, Mapping[str, Any]]
    source_plan: Mapping[str, Mapping[str, Any]]
    placement_plan: Mapping[str, Mapping[str, Any]]
    compatibility_state: str
    transition_class: str
    ready: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPOSITION_TRANSITION_PLAN_SCHEMA_VERSION,
            "lease_generation": int(self.lease_generation),
            "from_index": int(self.from_index),
            "to_index": int(self.to_index),
            "from_model_path": self.from_model_path,
            "to_model_path": self.to_model_path,
            "from_composition_sha256": self.from_composition_sha256,
            "to_composition_sha256": self.to_composition_sha256,
            "family": self.family,
            "provider_version": self.provider_version,
            "role_diff": {str(role): dict(item) for role, item in dict(self.role_diff).items()},
            "source_plan": {str(role): dict(item) for role, item in dict(self.source_plan).items()},
            "placement_plan": {str(role): dict(item) for role, item in dict(self.placement_plan).items()},
            "compatibility_state": self.compatibility_state,
            "transition_class": self.transition_class,
            "ready": bool(self.ready),
            "reasons": list(self.reasons),
        }


def plan_execution_lease_transition(
    lease: Any,
    loaded: Any,
    *,
    target_index: int,
    required_device: str,
) -> CompositionTransitionPlan:
    """Create a stable, side-effect-free transition plan from an active execution lease.

    The planner consumes only identities and already-live handles. It never hashes,
    opens a checkpoint, moves a module, or mutates runtime state. Runtime object IDs
    are deliberately excluded from the serialized plan.
    """

    reasons: list[str] = []
    if lease is None or str(getattr(lease, "state", "")) != "active":
        return CompositionTransitionPlan(
            lease_generation=int(getattr(lease, "generation", 0) or 0),
            from_index=int(getattr(lease, "active_index", 0) or 0),
            to_index=int(target_index),
            from_model_path="",
            to_model_path="",
            from_composition_sha256="",
            to_composition_sha256="",
            family="",
            provider_version="",
            role_diff={},
            source_plan={},
            placement_plan={},
            compatibility_state="unsupported",
            transition_class="unavailable",
            ready=False,
            reasons=("no_active_execution_lease",),
        )

    schedule = list(getattr(lease, "schedule", ()) or ())
    from_index = int(getattr(lease, "active_index", 0) or 0)
    target_index = int(target_index)
    if target_index < 0 or target_index >= len(schedule):
        reasons.append("target_index_out_of_range")
        target = {}
    else:
        target = dict(schedule[target_index] or {})
    current = dict(schedule[from_index] or {}) if 0 <= from_index < len(schedule) else {}

    family = str(getattr(lease, "family", "") or "")
    provider_version = str(getattr(lease, "provider_version", "") or "")
    if str(target.get("family") or "") != family:
        reasons.append("family_mismatch")
    if str(target.get("provider_version") or "") != provider_version:
        reasons.append("provider_contract_version_mismatch")

    resident_inventory = build_resident_component_inventory(loaded) if loaded is not None else {}
    target_components = {
        str(role): _digest(digest)
        for role, digest in dict(target.get("components") or {}).items()
        if _digest(digest)
    }
    current_components = {
        str(role): _digest(digest)
        for role, digest in dict(current.get("components") or {}).items()
        if _digest(digest)
    }
    pool = dict(getattr(lease, "component_pool", {}) or {})
    role_diff: dict[str, dict[str, Any]] = {}
    source_plan: dict[str, dict[str, Any]] = {}
    placement_plan: dict[str, dict[str, Any]] = {}

    for role in sorted(set(current_components) | set(target_components)):
        old_sha = current_components.get(role, "")
        new_sha = target_components.get(role, "")
        action = "retain" if old_sha and old_sha == new_sha else "replace"
        if old_sha and not new_sha:
            action = "remove"
        elif new_sha and not old_sha:
            action = "add"

        selected_source_kind = ""
        source_reason = ""
        current_device = "missing"
        target_device = str(required_device or "cpu")
        if new_sha:
            source = resolve_runtime_component_source(
                component_sha256=new_sha,
                role=role,
                family=family,
                provider_version=provider_version,
                resident_entries=resident_inventory,
                lease_entries_by_sha=pool,
                registry_sources=(),
                required_device=target_device,
                allow_digital_components=False,
            )
            selected_source_kind = str(source.get("selected_source_kind") or "")
            source_reason = str(source.get("reason") or "")
            selected = dict(source.get("selected") or {})
            source_plan[role] = {
                "component_sha256": new_sha,
                "selected_source_kind": selected_source_kind,
                "load_strategy": str(source.get("load_strategy") or ""),
                "reason": source_reason,
                "selected_device": str(selected.get("device") or ""),
                "cost_class": source.get("cost_class"),
            }
            pool_entry = pool.get(f"{role}:{new_sha}") or pool.get(new_sha)
            if pool_entry is not None:
                current_device = _module_device(getattr(pool_entry, "module", None))
            else:
                resident_entry = resident_inventory.get(role)
                if resident_entry is not None and _digest(getattr(resident_entry, "component_sha256", "")) == new_sha:
                    current_device = _module_device(getattr(resident_entry, "module", None))
            if selected_source_kind in {"", "unavailable"}:
                reasons.append(f"target_component_not_prepared:{role}")

        role_diff[role] = {
            "role": role,
            "action": action,
            "from_component_sha256": old_sha,
            "to_component_sha256": new_sha,
            "source_kind": selected_source_kind,
        }
        placement_plan[role] = {
            "role": role,
            "action": "preserve" if action == "retain" else "promote_or_restage" if new_sha else "release_or_lease",
            "current_device": current_device,
            "required_device": target_device if new_sha else "",
        }

    prepared = bool(getattr(lease, "prepared_composition", lambda _i: None)(target_index)) if target else False
    if not prepared:
        reasons.append("target_prepared_composition_unavailable")
    if target_index == from_index:
        transition_class = "noop"
    elif not reasons and all(item.get("action") == "retain" for item in role_diff.values()):
        transition_class = "identity_commit"
    elif not reasons:
        transition_class = "prepared_atomic_component_swap"
    else:
        transition_class = "fallback_required"
    ready = not reasons and prepared
    compatibility = "same_family_prepared" if ready else "unsupported_or_unprepared"
    return CompositionTransitionPlan(
        lease_generation=int(getattr(lease, "generation", 0) or 0),
        from_index=from_index,
        to_index=target_index,
        from_model_path=str(current.get("model_path") or ""),
        to_model_path=str(target.get("model_path") or ""),
        from_composition_sha256=str(current.get("composition_sha256") or ""),
        to_composition_sha256=str(target.get("composition_sha256") or ""),
        family=family,
        provider_version=provider_version,
        role_diff=role_diff,
        source_plan=source_plan,
        placement_plan=placement_plan,
        compatibility_state=compatibility,
        transition_class=transition_class,
        ready=ready,
        reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "COMPOSITION_TRANSITION_PLAN_SCHEMA_VERSION",
    "CompositionTransitionPlan",
    "plan_execution_lease_transition",
]
