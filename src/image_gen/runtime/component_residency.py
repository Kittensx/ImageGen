from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from modules.registry.family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY


COMPONENT_TRANSITION_SCHEMA_VERSION = 1


_ROLE_ATTRS = {
    "unet": "unet",
    "transformer": "denoiser",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
    "text_encoder_3": "text_encoder_3",
}


def _digest(value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _module_device_dtype(module: Any) -> tuple[str, str]:
    if module is None:
        return "missing", "unknown"
    try:
        parameter = next(module.parameters())
        return str(parameter.device), str(parameter.dtype)
    except (StopIteration, AttributeError, TypeError):
        return str(getattr(module, "device", "unknown")), str(getattr(module, "dtype", "unknown"))


def _module_for_role(components: Any, role: str) -> Any:
    attr = _ROLE_ATTRS.get(str(role), str(role))
    module = getattr(components, attr, None)
    if role == "transformer" and module is None:
        module = getattr(components, "transformer", None)
    return module


def _runtime_class_id(module: Any) -> str:
    if module is None:
        return ""
    cls = type(module)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', getattr(cls, '__name__', ''))}".strip(".")


@dataclass(frozen=True)
class ResidentComponentEntry:
    role: str
    component_sha256: str
    module: Any
    family: str
    provider_version: str
    composition_sha256: str
    source: Mapping[str, Any]
    device: str
    dtype: str
    runtime_class: str
    mutation_state: str
    reuse_eligible: bool
    reuse_block_reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "component_sha256": self.component_sha256,
            "family": self.family,
            "provider_version": self.provider_version,
            "composition_sha256": self.composition_sha256,
            "source": dict(self.source),
            "runtime_object_id": id(self.module) if self.module is not None else None,
            "device": self.device,
            "dtype": self.dtype,
            "runtime_class": self.runtime_class,
            "mutation_state": self.mutation_state,
            "reuse_eligible": bool(self.reuse_eligible),
            "reuse_block_reason": self.reuse_block_reason,
        }


def build_resident_component_inventory(
    loaded: Any,
    *,
    adapter_state_dirty: bool = False,
) -> dict[str, ResidentComponentEntry]:
    """Build the internal role -> live handle inventory for a resident composition.

    The exact role digests come from CNRR-03's canonical composition contract.  Live
    Python handles never leave this internal mapping; diagnostics use ``public_dict``.
    Adapter-loaded modules fail closed for this phase because current LoRA loaders can
    attach runtime state to the base module even after deactivation.
    """

    components = getattr(loaded, "components", None)
    contract = dict(getattr(components, "composition_contract", {}) or {}) if components is not None else {}
    family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(contract.get("family") or getattr(components, "architecture", ""))
    provider_version = str(contract.get("provider_version") or "")
    composition_sha256 = _digest(contract.get("composition_sha256") or getattr(components, "composition_sha256", ""))
    component_hashes = dict(contract.get("components") or {})
    sources = dict(getattr(components, "component_sources", {}) or {}) if components is not None else {}
    inventory: dict[str, ResidentComponentEntry] = {}
    for role, raw_digest in sorted(component_hashes.items()):
        digest = _digest(raw_digest)
        module = _module_for_role(components, str(role))
        device, dtype = _module_device_dtype(module)
        block_reason = ""
        mutation_state = "pristine_base"
        eligible = bool(digest and module is not None and family and provider_version)
        if module is None:
            block_reason = "runtime_handle_missing"
        elif not digest:
            block_reason = "component_identity_missing"
        elif adapter_state_dirty:
            mutation_state = "adapter_runtime_state_present"
            eligible = False
            block_reason = "adapter_runtime_state_present"
        elif bool(getattr(module, "_image_gen_component_reuse_blocked", False)):
            mutation_state = str(getattr(module, "_image_gen_component_mutation_state", "runtime_mutated") or "runtime_mutated")
            eligible = False
            block_reason = str(getattr(module, "_image_gen_component_reuse_block_reason", "runtime_mutated") or "runtime_mutated")
        inventory[str(role)] = ResidentComponentEntry(
            role=str(role),
            component_sha256=digest,
            module=module,
            family=family,
            provider_version=provider_version,
            composition_sha256=composition_sha256,
            source=dict(sources.get(str(role)) or {}),
            device=device,
            dtype=dtype,
            runtime_class=_runtime_class_id(module),
            mutation_state=mutation_state,
            reuse_eligible=eligible,
            reuse_block_reason=block_reason,
        )
    return inventory


def resident_reuse_bundle(
    loaded: Any,
    *,
    adapter_state_dirty: bool = False,
) -> dict[str, Any]:
    inventory = build_resident_component_inventory(loaded, adapter_state_dirty=adapter_state_dirty)
    family = ""
    provider_version = ""
    composition_sha256 = ""
    if inventory:
        first = next(iter(inventory.values()))
        family = first.family
        provider_version = first.provider_version
        composition_sha256 = first.composition_sha256
    return {
        "schema_version": COMPONENT_TRANSITION_SCHEMA_VERSION,
        "family": family,
        "provider_version": provider_version,
        "composition_sha256": composition_sha256,
        "entries": inventory,
        "public_inventory": {role: entry.public_dict() for role, entry in inventory.items()},
    }


def target_component_hashes(
    plan: Any,
    *,
    advanced_composition: Mapping[str, Any] | None = None,
    sd3_text_encoder_sources: Mapping[str, Any] | None = None,
    external_vae_override: bool = False,
) -> tuple[str, str, dict[str, str], list[str]]:
    """Resolve target role identities from evidence already present in the load path.

    No scan or hash is performed.  Missing/ambiguous identities are returned as reasons
    and those roles simply remain non-reusable for this transition.
    """

    report = getattr(plan, "report", None)
    family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(getattr(report, "architecture", ""))
    provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(family) if family else None
    provider_version = str(getattr(provider, "version", "") or "")
    reasons: list[str] = []
    components: dict[str, str] = {}

    advanced = dict(advanced_composition or {})
    if advanced:
        for role, raw in sorted(dict(advanced.get("components") or {}).items()):
            digest = _digest(dict(raw or {}).get("component_sha256"))
            if digest:
                components[str(role)] = digest
            else:
                reasons.append(f"target_identity_missing:{role}")
    else:
        by_role: dict[str, set[str]] = {}
        for snapshot in tuple(getattr(plan, "component_snapshots", ()) or ()):
            role = str(getattr(snapshot, "component_role", "") or "")
            digest = _digest(getattr(snapshot, "component_sha256", ""))
            if role and digest:
                by_role.setdefault(role, set()).add(digest)
        for role, digests in sorted(by_role.items()):
            if len(digests) == 1:
                components[role] = next(iter(digests))
            elif len(digests) > 1:
                reasons.append(f"target_identity_ambiguous:{role}")

    sd3_sources = dict(sd3_text_encoder_sources or {})
    role_aliases = {"clip_l": "text_encoder", "clip_g": "text_encoder_2", "t5xxl": "text_encoder_3"}
    for source_role, raw in dict(sd3_sources.get("roles") or {}).items():
        role = role_aliases.get(str(source_role), str(source_role))
        item = dict(raw or {})
        kind = str(item.get("source_kind") or "").strip().lower()
        if kind == "disabled":
            components.pop(role, None)
            continue
        if kind == "external":
            digest = _digest(item.get("matched_component_sha256"))
            if digest:
                components[role] = digest
            else:
                components.pop(role, None)
                reasons.append(f"target_identity_missing:{role}")

    if external_vae_override:
        # CNRR-04 intentionally does not infer/rehash an external VAE here.  The
        # override loader remains authoritative and this role fails closed.
        components.pop("vae", None)
        reasons.append("external_vae_override_not_preprojected")

    if provider is not None:
        definitions = tuple(provider.role_definitions())
        allowed = {definition.canonical_role_id for definition in definitions}
        components = {role: digest for role, digest in components.items() if role in allowed}
        for definition in definitions:
            role = str(definition.canonical_role_id)
            if bool(getattr(definition, "required", False)) and role not in components:
                reasons.append(f"target_identity_missing:{role}")
    return family, provider_version, components, list(dict.fromkeys(reasons))


def plan_component_transition(
    resident_bundle: Mapping[str, Any] | None,
    *,
    target_family: str,
    target_provider_version: str,
    target_components: Mapping[str, str],
    target_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic per-role retain/replace/add/remove/cannot_reuse actions."""

    bundle = dict(resident_bundle or {})
    entries = dict(bundle.get("entries") or {})
    lease_entries_by_sha = dict(bundle.get("lease_entries_by_sha") or {})
    resident_family = str(bundle.get("family") or "")
    resident_provider_version = str(bundle.get("provider_version") or "")
    target = {str(role): _digest(digest) for role, digest in dict(target_components or {}).items() if _digest(digest)}
    reasons = list(target_reasons or [])
    unknown_target_roles = {
        item.split(":", 1)[1]
        for item in reasons
        if item.startswith(("target_identity_missing:", "target_identity_ambiguous:")) and ":" in item
    }
    if "external_vae_override_not_preprojected" in reasons:
        unknown_target_roles.add("vae")
    roles = sorted(set(entries) | set(target) | unknown_target_roles)
    role_diff: dict[str, dict[str, Any]] = {}
    reusable: dict[str, Any] = {}

    family_compatible = bool(target_family and resident_family == target_family)
    provider_compatible = bool(target_provider_version and resident_provider_version == target_provider_version)
    for role in roles:
        entry = entries.get(role)
        resident_digest = _digest(getattr(entry, "component_sha256", "") if entry is not None else "")
        target_digest = target.get(role, "")
        warm_entry = (
            lease_entries_by_sha.get(f"{role}:{target_digest}")
            or lease_entries_by_sha.get(target_digest)
        ) if target_digest else None
        warm_family = str(getattr(warm_entry, "family", "") or "") if warm_entry is not None else ""
        warm_provider = str(getattr(warm_entry, "provider_version", "") or "") if warm_entry is not None else ""
        warm_eligible = bool(getattr(warm_entry, "reuse_eligible", False)) if warm_entry is not None else False
        warm_digest = _digest(getattr(warm_entry, "component_sha256", "") if warm_entry is not None else "")
        warm_module = getattr(warm_entry, "module", None) if warm_entry is not None else None
        warm_matches = bool(
            warm_entry is not None
            and warm_digest == target_digest
            and warm_module is not None
            and warm_eligible
            and family_compatible
            and provider_compatible
            and (not warm_family or warm_family == target_family)
            and (not warm_provider or warm_provider == target_provider_version)
        )
        action = "same"
        reason = "exact_component_identity"
        reuse_source_module = None
        if role in unknown_target_roles:
            action, reason = "cannot_reuse", "target_component_identity_unavailable"
        elif entry is None and target_digest:
            if warm_matches:
                action, reason = "reuse_warm", "exact_component_identity_already_warm_under_lease"
                reusable[role] = warm_module
                reuse_source_module = warm_module
            else:
                action, reason = "add", "role_not_resident"
        elif entry is not None and not target_digest:
            action, reason = "remove", "role_not_requested"
        elif resident_digest != target_digest:
            if warm_matches:
                action, reason = "reuse_warm", "exact_component_identity_already_warm_under_lease"
                reusable[role] = warm_module
                reuse_source_module = warm_module
            else:
                action, reason = "replace", "component_sha256_changed"
        elif not family_compatible:
            action, reason = "cannot_reuse", "family_mismatch"
        elif not provider_compatible:
            action, reason = "cannot_reuse", "provider_contract_version_mismatch"
        elif not bool(getattr(entry, "reuse_eligible", False)):
            action = "cannot_reuse"
            reason = str(getattr(entry, "reuse_block_reason", "runtime_state_not_reusable") or "runtime_state_not_reusable")
        else:
            action = "retain"
            reusable[role] = entry.module
            reuse_source_module = entry.module

        role_diff[role] = {
            "role": role,
            "action": action,
            "reason": reason,
            "resident_component_sha256": resident_digest,
            "requested_component_sha256": target_digest,
            "runtime_object_id_before": id(entry.module) if entry is not None and getattr(entry, "module", None) is not None else None,
            "runtime_object_id_reuse_source": id(reuse_source_module) if reuse_source_module is not None else None,
            "reuse_source": "lease_warm" if action == "reuse_warm" else "active_resident" if action == "retain" else "",
            "resident_device": str(getattr(entry, "device", "") or "") if entry is not None else "",
            "resident_dtype": str(getattr(entry, "dtype", "") or "") if entry is not None else "",
            "mutation_state": str(getattr(entry, "mutation_state", "") or "") if entry is not None else "",
        }

    counts = {name: sum(1 for item in role_diff.values() if item["action"] == name) for name in ("same", "retain", "reuse_warm", "replace", "add", "remove", "cannot_reuse")}
    if reusable:
        fallback_reason = ""
    elif not entries:
        fallback_reason = "no_resident_component_inventory"
    elif not family_compatible:
        fallback_reason = "family_mismatch"
    elif not provider_compatible:
        fallback_reason = "provider_contract_version_mismatch"
    elif unknown_target_roles:
        fallback_reason = "target_component_identity_unavailable"
    else:
        fallback_reason = "no_reusable_exact_components"
    return {
        "schema_version": COMPONENT_TRANSITION_SCHEMA_VERSION,
        "previous_composition_sha256": str(bundle.get("composition_sha256") or ""),
        "requested_composition_sha256": "",
        "resident_family": resident_family,
        "requested_family": target_family,
        "resident_provider_version": resident_provider_version,
        "requested_provider_version": target_provider_version,
        "family_compatible": family_compatible,
        "provider_compatible": provider_compatible,
        "role_diff": role_diff,
        "counts": counts,
        "target_projection_reasons": reasons,
        "fallback_reason": fallback_reason,
        "reusable_components": reusable,
        "retained_component_count": counts["retain"] + counts["reuse_warm"],
        "warm_reused_component_count": counts["reuse_warm"],
        "replaced_component_count": counts["replace"] + counts["cannot_reuse"],
        "added_component_count": counts["add"],
        "removed_component_count": counts["remove"],
        "transition_classification": (
            "partial_reuse" if reusable and any(item["action"] in {"replace", "add", "remove", "cannot_reuse"} for item in role_diff.values())
            else "full_component_reuse" if reusable and len(reusable) == len(target)
            else "full_rebuild"
        ),
    }


def public_transition_report(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(plan or {})
    payload.pop("reusable_components", None)
    return payload


__all__ = [
    "COMPONENT_TRANSITION_SCHEMA_VERSION",
    "ResidentComponentEntry",
    "build_resident_component_inventory",
    "resident_reuse_bundle",
    "target_component_hashes",
    "plan_component_transition",
    "public_transition_report",
]
