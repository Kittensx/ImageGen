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


def _normalize_enabled(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if token in {"0", "false", "no", "off", "disabled", "disable", ""}:
        return False
    return bool(default)


def _normalize_source(value: Any, default: str = "auto") -> str:
    token = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "shared": "external",
        "standalone": "external",
        "separate": "external",
        "checkpoint": "embedded",
    }
    token = aliases.get(token, token)
    if token.startswith("component:"):
        digest = token.split(":", 1)[1].strip().lower()
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            return f"component:{digest}"
        raise ValueError(f"Invalid SD3 text-encoder component selector {value!r}.")
    if token not in {"auto", "embedded", "external"}:
        raise ValueError(
            f"Unsupported SD3 text-encoder source policy {value!r}; expected auto, embedded, external, or component:<sha256>."
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
    selected_component_sha256: str = "",
) -> tuple[dict[str, Any], SD3TextEncoderSourceSelection]:
    selected_digest = str(selected_component_sha256 or "").strip().lower()
    resolution = resolve_shared_text_encoder(
        context,
        role,
        expected_component_sha256=selected_digest or expected_component_sha256 or None,
    )
    selected = resolution.selected
    if selected is None:
        checked = ", ".join(str(path) for path in resolution.checked) or "no candidates"
        raise FileNotFoundError(
            f"SD3 requires a standalone {role} text encoder for this checkpoint, but no unambiguous local asset was found. "
            f"Checked: {checked}"
        )
    selected = Path(selected).resolve()
    matched = str(resolution.matched_component_sha256 or "").strip().lower()
    if selected_digest:
        if matched != selected_digest:
            raise RuntimeError(
                f"SD3 external {role} component selection could not be resolved to the requested component identity. "
                "Refresh the asset registry before using this T5 selection."
            )
    elif expected_component_sha256:
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
    """Resolve text-encoder state sources for normal SD3 generation.

    CLIP-L/G preserve the established ``auto`` policy: use an embedded encoder
    when present and otherwise resolve the matching shared TextEncoders asset.

    T5/T5XXL is optional.  When ``sd3_t5_enabled`` is omitted we preserve the
    pre-GFP behavior (embedded T5 remains enabled; an external T5 is not pulled
    in automatically).  An explicit true value enables either the embedded T5
    or a resolvable shared T5 asset.  Explicit false removes T5 from the mapped
    state before component construction, so the runtime uses the validated
    CLIP-L + CLIP-G zero-sequence path instead.
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

    t5_component_role = "text_encoder_3"
    embedded_t5 = bool(getattr(mapped, t5_component_role))
    t5_enabled = _normalize_enabled(extras.get("sd3_t5_enabled"), default=embedded_t5)
    t5_policy = _normalize_source(extras.get("sd3_t5_source"), "auto")
    if not t5_enabled:
        setattr(mapped, t5_component_role, {})
        evidence["roles"]["t5xxl"] = {
            "role": "t5xxl",
            "source_kind": "disabled",
            "source_path": "",
            "source_layout": "disabled_by_request",
            "expected_component_sha256": _snapshot_hash(plan, t5_component_role),
            "matched_component_sha256": "",
        }
    else:
        use_embedded_t5 = t5_policy == "embedded" or (t5_policy == "auto" and embedded_t5)
        if use_embedded_t5:
            if not embedded_t5:
                raise ValueError(
                    "SD3 T5 source is set to embedded, but this checkpoint does not contain text_encoder_3."
                )
            evidence["roles"]["t5xxl"] = SD3TextEncoderSourceSelection(
                role="t5xxl",
                source_kind="embedded",
                source_path=str(getattr(getattr(plan, "report", None), "model_path", "") or ""),
                source_layout="checkpoint",
                expected_component_sha256=_snapshot_hash(plan, t5_component_role),
                matched_component_sha256=_snapshot_hash(plan, t5_component_role),
            ).to_dict()
        else:
            expected_hash = _snapshot_hash(plan, t5_component_role) if embedded_t5 else ""
            selected_component_sha256 = (
                t5_policy.split(":", 1)[1]
                if t5_policy.startswith("component:")
                else ""
            )
            state, selection = _load_external_state(
                context=context,
                role="t5xxl",
                expected_component_sha256=expected_hash,
                selected_component_sha256=selected_component_sha256,
            )
            setattr(mapped, t5_component_role, state)
            evidence["roles"]["t5xxl"] = selection.to_dict()

    evidence["t5_enabled"] = bool(t5_enabled)
    evidence["t5_policy"] = t5_policy
    evidence["mode"] = "+".join(
        f"{role}:{payload['source_kind']}" for role, payload in evidence["roles"].items()
    )
    return evidence

