from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import CompositionIdentity
from .family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY


PROJECTION_COMPLETE = "complete"
PROJECTION_INCOMPLETE = "incomplete"
PROJECTION_UNAVAILABLE = "unavailable"


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
        return str(Path(token).expanduser().resolve(strict=False))
    except OSError:
        return token


@dataclass(frozen=True)
class RuntimeCompositionProjection:
    status: str
    family: str
    provider_version: str
    whole_checkpoint_sha256: str
    composition: CompositionIdentity | None = None
    components: Mapping[str, str] = field(default_factory=dict)
    component_sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    missing_roles: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    source_mode: str = "normal_checkpoint"

    def to_dict(self) -> dict[str, Any]:
        contract = self.composition.to_dict() if self.composition is not None else {}
        return {
            "status": self.status,
            "source_mode": self.source_mode,
            "family": self.family,
            "provider_version": self.provider_version,
            "identity_version": str(contract.get("identity_version") or ""),
            "composition_sha256": str(contract.get("composition_sha256") or ""),
            "composition_short_hash": str(contract.get("composition_short_hash") or ""),
            "composition_contract": contract,
            "components": dict(self.components),
            "component_sources": {
                str(role): dict(source)
                for role, source in sorted(self.component_sources.items())
            },
            "missing_roles": list(self.missing_roles),
            "reasons": list(self.reasons),
            "whole_checkpoint_sha256": self.whole_checkpoint_sha256,
        }


def _source_record(
    *,
    role: str,
    component_sha256: str,
    source_kind: str,
    source_path: Any,
    source_form: str = "",
    source_role: str = "",
) -> dict[str, Any]:
    return {
        "role": str(role),
        "component_sha256": _digest(component_sha256),
        "source_kind": str(source_kind or "unknown"),
        "source_path": _resolved_path(source_path),
        "source_form": str(source_form or ""),
        "source_role": str(source_role or role),
    }


def _advanced_projection(
    *,
    family: str,
    provider: Any,
    advanced_composition: Mapping[str, Any],
    whole_checkpoint_sha256: str,
) -> RuntimeCompositionProjection:
    applied = dict(advanced_composition.get("components") or {})
    components: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for role, raw in sorted(applied.items()):
        item = dict(raw or {})
        digest = _digest(item.get("component_sha256"))
        if not digest:
            reasons.append(f"advanced_component_missing_sha256:{role}")
            continue
        components[str(role)] = digest
        sources[str(role)] = _source_record(
            role=str(role),
            component_sha256=digest,
            source_kind=str(item.get("source_mode") or "advanced_component"),
            source_path=item.get("source_path"),
            source_form=str(item.get("source_asset_type") or ""),
            source_role=str(item.get("source_role") or role),
        )

    missing = tuple(
        definition.canonical_role_id
        for definition in provider.required_roles
        if definition.canonical_role_id not in components
    )
    if missing:
        reasons.extend(f"missing_required_role:{role}" for role in missing)

    behavior_choices: dict[str, Any] = {}
    if "text_encoder_3" in components:
        behavior_choices["text_encoder_3_placement"] = str(
            advanced_composition.get("t5_device") or "cpu"
        ).strip().lower()

    composition = None
    if not reasons:
        composition = CompositionIdentity.derive(
            family=family,
            provider_version=provider.version,
            components=components,
            behavior_choices=behavior_choices,
        )
        recorded = _digest(advanced_composition.get("composition_sha256"))
        if recorded and recorded != composition.composition_sha256:
            reasons.append("advanced_composition_contract_mismatch")
            composition = None

    return RuntimeCompositionProjection(
        status=PROJECTION_COMPLETE if composition is not None else PROJECTION_INCOMPLETE,
        family=family,
        provider_version=provider.version,
        whole_checkpoint_sha256=_digest(whole_checkpoint_sha256),
        composition=composition,
        components=components,
        component_sources=sources,
        missing_roles=missing,
        reasons=tuple(reasons),
        source_mode="advanced_models",
    )


def _registry_component_for_path(
    registry: Any,
    *,
    path: Any,
    role: str,
) -> tuple[str, dict[str, Any] | None, str]:
    if registry is None:
        return "", None, "registry_unavailable_for_detached_component"
    resolved = _resolved_path(path)
    if not resolved:
        return "", None, "detached_component_path_missing"
    get_asset = getattr(registry, "get_asset_by_path", None)
    get_snapshots = getattr(registry, "get_component_snapshots", None)
    if not callable(get_asset) or not callable(get_snapshots):
        return "", None, "registry_missing_projection_api"
    asset = get_asset(resolved)
    if asset is None:
        return "", None, "detached_component_not_registered"
    snapshots = [
        item for item in get_snapshots(int(asset.id))
        if str(getattr(item, "component_role", "") or "") == role
    ]
    digests = sorted({_digest(getattr(item, "component_sha256", "")) for item in snapshots} - {""})
    if len(digests) != 1:
        return "", None, (
            "detached_component_snapshot_missing"
            if not digests
            else "detached_component_snapshot_ambiguous"
        )
    digest = digests[0]
    return digest, _source_record(
        role=role,
        component_sha256=digest,
        source_kind="detached_component",
        source_path=resolved,
        source_form=str(getattr(asset, "asset_type", "") or ""),
        source_role=role,
    ), ""


def project_runtime_composition(
    plan: Any,
    *,
    registry: Any = None,
    advanced_composition: Mapping[str, Any] | None = None,
    sd3_text_encoder_sources: Mapping[str, Any] | None = None,
    vae_provenance: Mapping[str, Any] | None = None,
    text_encoder_3_device: Any = "auto",
) -> RuntimeCompositionProjection:
    """Project the effective runtime components onto the shared CompositionIdentity.

    This consumes evidence already produced by the model-loading path. It never scans
    the library and never re-hashes checkpoint components. Missing evidence is exposed
    as an incomplete projection so CNRR-02 whole-checkpoint reuse remains authoritative.
    """

    report = getattr(plan, "report", None)
    raw_family = str(getattr(report, "architecture", "") or "")
    family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(raw_family)
    checkpoint_sha = _digest(getattr(report, "sha256", ""))
    if not family:
        return RuntimeCompositionProjection(
            status=PROJECTION_UNAVAILABLE,
            family=raw_family,
            provider_version="",
            whole_checkpoint_sha256=checkpoint_sha,
            reasons=("unsupported_or_unclassified_provider_family",),
        )
    provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.require(family)

    advanced = dict(advanced_composition or {})
    if advanced:
        return _advanced_projection(
            family=family,
            provider=provider,
            advanced_composition=advanced,
            whole_checkpoint_sha256=checkpoint_sha,
        )

    snapshots = tuple(getattr(plan, "component_snapshots", ()) or ())
    by_role: dict[str, set[str]] = {}
    for snapshot in snapshots:
        role = str(getattr(snapshot, "component_role", "") or "")
        digest = _digest(getattr(snapshot, "component_sha256", ""))
        if role and digest and provider.role_definition(role) is not None:
            by_role.setdefault(role, set()).add(digest)

    reasons: list[str] = []
    components: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    checkpoint_path = str(getattr(report, "model_path", "") or "")
    for role, digests in sorted(by_role.items()):
        if len(digests) != 1:
            reasons.append(f"ambiguous_checkpoint_component:{role}")
            continue
        digest = next(iter(digests))
        components[role] = digest
        sources[role] = _source_record(
            role=role,
            component_sha256=digest,
            source_kind="embedded_checkpoint",
            source_path=checkpoint_path,
            source_form="checkpoint",
            source_role=role,
        )

    sd3_sources = dict(sd3_text_encoder_sources or {})
    role_aliases = {
        "clip_l": "text_encoder",
        "clip_g": "text_encoder_2",
        "t5xxl": "text_encoder_3",
    }
    for source_role, raw in dict(sd3_sources.get("roles") or {}).items():
        role = role_aliases.get(str(source_role), str(source_role))
        if provider.role_definition(role) is None:
            continue
        item = dict(raw or {})
        kind = str(item.get("source_kind") or "")
        if kind == "disabled":
            components.pop(role, None)
            sources.pop(role, None)
            continue
        if kind == "external":
            digest = _digest(item.get("matched_component_sha256"))
            if not digest:
                reasons.append(f"external_component_identity_missing:{role}")
                components.pop(role, None)
                sources.pop(role, None)
                continue
            components[role] = digest
            sources[role] = _source_record(
                role=role,
                component_sha256=digest,
                source_kind="external_text_encoder",
                source_path=item.get("source_path"),
                source_form="standalone_shared",
                source_role=role,
            )

    vae_info = dict(vae_provenance or {})
    if str(vae_info.get("source_kind") or "") == "external_vae_override":
        digest, source, error = _registry_component_for_path(
            registry,
            path=vae_info.get("source_path"),
            role="vae",
        )
        if error:
            reasons.append(error + ":vae")
            components.pop("vae", None)
            sources.pop("vae", None)
        else:
            components["vae"] = digest
            if source is not None:
                sources["vae"] = source

    missing = tuple(
        definition.canonical_role_id
        for definition in provider.required_roles
        if definition.canonical_role_id not in components
    )
    if missing:
        reasons.extend(f"missing_required_role:{role}" for role in missing)

    behavior_choices: dict[str, Any] = {}
    if "text_encoder_3" in components:
        t5_device = str(text_encoder_3_device or "auto").strip().lower()
        if t5_device not in {"auto", "cpu", "cuda"}:
            t5_device = "auto"
        behavior_choices["text_encoder_3_placement"] = t5_device

    composition = None
    if not reasons:
        composition = CompositionIdentity.derive(
            family=family,
            provider_version=provider.version,
            components=components,
            behavior_choices=behavior_choices,
        )

    return RuntimeCompositionProjection(
        status=PROJECTION_COMPLETE if composition is not None else PROJECTION_INCOMPLETE,
        family=family,
        provider_version=provider.version,
        whole_checkpoint_sha256=checkpoint_sha,
        composition=composition,
        components=components,
        component_sources=sources,
        missing_roles=missing,
        reasons=tuple(dict.fromkeys(reasons)),
        source_mode="normal_checkpoint",
    )


__all__ = [
    "PROJECTION_COMPLETE",
    "PROJECTION_INCOMPLETE",
    "PROJECTION_UNAVAILABLE",
    "RuntimeCompositionProjection",
    "project_runtime_composition",
]
