from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from modules.checkpoint_inspector import CheckpointInspector
from modules.state_dict_mapper import StateDictMapper
from modules.registry.component_selection import canonical_model_family


def apply_advanced_component_composition(
    plan: Any,
    resolved: Mapping[str, Any] | None,
    *,
    inspector: CheckpointInspector | None = None,
    mapper: StateDictMapper | None = None,
) -> dict[str, Any]:
    """Replace a prepared load plan's component states from registry-selected donors.

    ``prepare_load_plan`` still supplies architecture/runtime configs from the selected
    family and the denoiser donor checkpoint. The user's Advanced Models selection is
    then authoritative for every learned component role. Donor checkpoints are merely
    tensor containers; their unselected components are ignored.
    """
    payload = dict(resolved or {})
    if not payload.get("enabled"):
        return {}

    family = canonical_model_family(payload.get("family"))
    plan_family = canonical_model_family(getattr(getattr(plan, "report", None), "architecture", ""))
    if not family or family != plan_family:
        raise ValueError(
            f"Advanced component family {family or '<unknown>'!r} does not match the prepared runtime family {plan_family or '<unknown>'!r}."
        )

    components = dict(payload.get("components") or {})
    if not components:
        raise ValueError("Advanced Models did not resolve any components.")

    inspector = inspector or CheckpointInspector()
    mapper = mapper or StateDictMapper()
    base_path = Path(str(payload.get("base_source_path") or "")).expanduser().resolve()
    loaded_checkpoint_states: dict[tuple[str, str], Any] = {}
    applied: dict[str, Any] = {}

    for role, selection in sorted(components.items()):
        source = dict((selection or {}).get("source") or {})
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Advanced Models component source is missing: {source_path}")
        source_role = str(source.get("component_role") or role)
        asset_type = str(source.get("asset_type") or "unknown").strip().lower()
        force_digital_extract = bool(source.get("force_digital_extract", False))

        if asset_type == "checkpoint" and force_digital_extract:
            token = (str(source_path), source_role)
            mapped = loaded_checkpoint_states.get(token)
            if mapped is None:
                mapped = mapper.load_selected_checkpoint_components(
                    str(source_path),
                    architecture=family,
                    roles={source_role},
                )
                loaded_checkpoint_states[token] = mapped
            state = getattr(mapped, source_role, None)
            if not state:
                raise ValueError(
                    f"Forced digital component donor {source_path.name!r} does not contain registry role {source_role!r}."
                )
            source_mode = "checkpoint_component_donor_forced"
        elif source_path == base_path and source_role == role:
            state = getattr(plan.mapped_state, role, None)
            if not state:
                raise ValueError(f"Base denoiser checkpoint does not provide required component role {role!r}.")
            source_mode = "base_donor_checkpoint"
        elif asset_type == "checkpoint":
            token = (str(source_path), source_role)
            mapped = loaded_checkpoint_states.get(token)
            if mapped is None:
                mapped = mapper.load_selected_checkpoint_components(
                    str(source_path),
                    architecture=family,
                    roles={source_role},
                )
                loaded_checkpoint_states[token] = mapped
            state = getattr(mapped, source_role, None)
            if not state:
                raise ValueError(
                    f"Component donor checkpoint {source_path.name!r} does not contain registry role {source_role!r}."
                )
            source_mode = "checkpoint_component_donor"
        else:
            if source_path.suffix.lower() != ".safetensors":
                raise ValueError(
                    f"Standalone Advanced Models components must currently be safetensors files; got {source_path.name!r}."
                )
            state = dict(load_file(str(source_path), device="cpu"))
            if not state:
                raise ValueError(f"Standalone component {source_path.name!r} contained no tensors.")
            source_mode = "standalone_component"

        setattr(plan.mapped_state, role, state)
        applied[role] = {
            "component_sha256": str((selection or {}).get("component_sha256") or ""),
            "source_path": str(source_path),
            "source_role": source_role,
            "source_asset_type": asset_type,
            "source_mode": source_mode,
            "force_digital_extract": force_digital_extract,
            "tensor_count": len(state),
        }

    # Optional roles are authoritative too. If T5 is not selected, make sure an
    # embedded T5 from the denoiser donor cannot silently become active later.
    if family == "sd3.x" and "text_encoder_3" not in components:
        plan.mapped_state.text_encoder_3 = {}

    return {
        "family": family,
        "composition_sha256": str(payload.get("composition_sha256") or ""),
        "components": applied,
        "t5_device": str(payload.get("t5_device") or "off"),
        "checkpoint_selection_ignored": True,
    }


__all__ = ["apply_advanced_component_composition"]
