from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from modules.project_context import ProjectContext
from modules.registry.asset_registry import AssetRegistry
from modules.sd3_shared_text_encoders import (
    register_shared_text_encoder_asset,
    resolve_shared_text_encoder,
    text_encoder_component_role,
)


@dataclass(frozen=True)
class SD3TextEncoderSourceSelection:
    role: str
    source_kind: str
    source_path: str
    source_layout: str
    expected_component_sha256: str = ""
    matched_component_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_layout": self.source_layout,
            "expected_component_sha256": self.expected_component_sha256,
            "matched_component_sha256": self.matched_component_sha256,
        }


def _snapshot_hash(plan: Any, component_role: str) -> str:
    for item in tuple(getattr(plan, "component_snapshots", ()) or ()):
        if str(getattr(item, "component_role", "")) == component_role:
            return str(getattr(item, "component_sha256", "") or "").strip().lower()
    return ""


def _normalize_source(value: Any, default: str = "auto") -> str:
    token = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "shared": "external",
        "standalone": "external",
        "separate": "external",
        "checkpoint": "embedded",
    }
    token = aliases.get(token, token)
    if token not in {"auto", "embedded", "external"}:
        raise ValueError(
            f"Unsupported SD3 text-encoder source policy {value!r}; expected auto, embedded, or external."
        )
    return token


def _ensure_registered(context: ProjectContext, path: Path, role: str) -> None:
    registry_path = getattr(context, "registry_db_path", None)
    if registry_path is None:
        return
    try:
        registry_file = Path(registry_path).resolve()
        if registry_file.is_file():
            registry = AssetRegistry(str(registry_file))
            asset = registry.get_asset_by_path(str(path.resolve()))
            if asset is not None:
                component_role = text_encoder_component_role(role)
                snapshots = registry.get_component_snapshots(asset.id)
                for item in snapshots:
                    if item.component_role != component_role:
                        continue
                    try:
                        metadata = json.loads(item.metadata_json or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        metadata = {}
                    if str(metadata.get("source_quick_fingerprint") or "") == str(asset.quick_fingerprint or ""):
                        return
        register_shared_text_encoder_asset(context, path, role=role)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to update the asset registry for SD3 shared text encoder {role} at {path}: {exc}"
        ) from exc


def _load_external_state(
    *,
    context: ProjectContext,
    role: str,
    expected_component_sha256: str,
) -> tuple[dict[str, Any], SD3TextEncoderSourceSelection]:
    resolution = resolve_shared_text_encoder(
        context,
        role,
        expected_component_sha256=expected_component_sha256 or None,
    )
    selected = resolution.selected
    if selected is None:
        checked = ", ".join(str(path) for path in resolution.checked) or "no candidates"
        raise FileNotFoundError(
            f"SD3 requires a standalone {role} text encoder for this checkpoint, but no unambiguous local asset was found. "
            f"Checked: {checked}"
        )
    selected = Path(selected).resolve()
    if expected_component_sha256:
        matched = str(resolution.matched_component_sha256 or "").strip().lower()
        if matched != expected_component_sha256:
            raise RuntimeError(
                f"SD3 external {role} selection was requested for a checkpoint that already embeds that component, "
                "but the registry could not prove exact component identity. Refresh the asset registry before using "
                "the external override."
            )
    _ensure_registered(context, selected, role)
    state = dict(load_file(str(selected), device="cpu"))
    return state, SD3TextEncoderSourceSelection(
        role=role,
        source_kind="external",
        source_path=str(selected),
        source_layout=str(resolution.source_layout),
        expected_component_sha256=expected_component_sha256,
        matched_component_sha256=str(resolution.matched_component_sha256 or "").strip().lower(),
    )


def prepare_sd3_text_encoder_states(
    plan: Any,
    *,
    context: ProjectContext,
    request_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve CLIP-L/G state sources for normal SD3 generation.

    ``auto`` preserves embedded encoders when present and falls back to the
    shared TextEncoders library for plain checkpoints. Explicit external source
    selection is fingerprint-checked against embedded components when an exact
    embedded identity is available.
    """
    if getattr(plan, "sd3_contract", None) is None:
        return {}
    extras = dict(request_extras or {})
    global_policy = _normalize_source(extras.get("sd3_text_encoder_source"), "auto")
    role_specs = (
        ("clip_l", "text_encoder"),
        ("clip_g", "text_encoder_2"),
    )
    evidence: dict[str, Any] = {
        "policy": global_policy,
        "roles": {},
    }
    mapped = plan.mapped_state
    for role, component_role in role_specs:
        role_policy = _normalize_source(
            extras.get(f"sd3_{role}_source"),
            global_policy,
        )
        existing = getattr(mapped, component_role)
        has_embedded = bool(existing)
        use_embedded = role_policy == "embedded" or (role_policy == "auto" and has_embedded)
        if use_embedded:
            if not has_embedded:
                raise ValueError(
                    f"SD3 {role} source is set to embedded, but this checkpoint does not contain {component_role}."
                )
            evidence["roles"][role] = SD3TextEncoderSourceSelection(
                role=role,
                source_kind="embedded",
                source_path=str(getattr(getattr(plan, "report", None), "model_path", "") or ""),
                source_layout="checkpoint",
                expected_component_sha256=_snapshot_hash(plan, component_role),
                matched_component_sha256=_snapshot_hash(plan, component_role),
            ).to_dict()
            continue

        expected_hash = _snapshot_hash(plan, component_role) if has_embedded else ""
        state, selection = _load_external_state(
            context=context,
            role=role,
            expected_component_sha256=expected_hash,
        )
        setattr(mapped, component_role, state)
        evidence["roles"][role] = selection.to_dict()

    evidence["mode"] = "+".join(
        f"{role}:{payload['source_kind']}" for role, payload in evidence["roles"].items()
    )
    return evidence
