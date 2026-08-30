from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from modules.checkpoint_inspector import CheckpointInspector
from modules.state_dict_mapper import StateDictMapper
from modules.registry.component_selection import canonical_model_family


def _resolved_path(value: Any) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def apply_advanced_component_composition(
    plan: Any,
    resolved: Mapping[str, Any] | None,
    *,
    inspector: CheckpointInspector | None = None,
    mapper: StateDictMapper | None = None,
    runtime_source_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace a prepared load plan's component states from registry-selected donors.

    CNRR-05 keeps component identity separate from source choice.  When the runtime
    source plan proves an exact component is already resident, no donor tensor state is
    hydrated for that role.  When several selected roles come from one checkpoint,
    their tensors are materialized in one selective donor transaction instead of
    reopening the same checkpoint once per role.
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
    base_path = _resolved_path(payload.get("base_source_path"))
    source_roles = dict(dict(runtime_source_plan or {}).get("roles") or {})
    applied: dict[str, Any] = {}
    choices: dict[str, dict[str, Any]] = {}
    checkpoint_groups: dict[str, set[str]] = {}

    # Resolve the runtime-selected occurrence for each exact identity first.  If no
    # runtime-aware plan exists, retain the historical static source choice verbatim.
    for role, selection in sorted(components.items()):
        selection = dict(selection or {})
        static_source = dict(selection.get("source") or {})
        source_plan = dict(source_roles.get(role) or {})
        selected_kind = str(source_plan.get("selected_source_kind") or "")
        runtime_occurrence = dict(source_plan.get("selected_occurrence") or {})
        static_force_digital_extract = bool(static_source.get("force_digital_extract", False))
        # An explicit forced digital extraction is an authority contract, not a cost
        # hint.  It must not be replaced by resident reuse or an alternate occurrence.
        source = static_source if static_force_digital_extract else (runtime_occurrence or static_source)
        source_path = _resolved_path(source.get("path"))
        source_role = str(source.get("component_role") or role)
        asset_type = str(source.get("asset_type") or static_source.get("asset_type") or "unknown").strip().lower()
        force_digital_extract = bool(
            static_force_digital_extract
            or source.get("force_digital_extract", False)
        )
        if force_digital_extract:
            if not source_path.is_file():
                raise FileNotFoundError(f"Advanced Models component source is missing: {source_path}")
            mode = "checkpoint_component_donor_forced"
            checkpoint_groups.setdefault(str(source_path), set()).add(source_role)
        elif selected_kind.startswith("resident_"):
            choices[role] = {
                "mode": "resident_component",
                "source": source,
                "source_path": source_path,
                "source_role": source_role,
                "asset_type": asset_type,
                "force_digital_extract": force_digital_extract,
                "source_plan": source_plan,
            }
            continue
        elif not source_path.is_file():
            raise FileNotFoundError(f"Advanced Models component source is missing: {source_path}")
        elif source_path == base_path and source_role == role and bool(getattr(plan.mapped_state, role, None)):
            mode = "base_donor_checkpoint"
        elif asset_type == "checkpoint":
            mode = "checkpoint_component_donor"
            checkpoint_groups.setdefault(str(source_path), set()).add(source_role)
        else:
            mode = "standalone_component"
        choices[role] = {
            "mode": mode,
            "source": source,
            "source_path": source_path,
            "source_role": source_role,
            "asset_type": asset_type,
            "force_digital_extract": force_digital_extract,
            "source_plan": source_plan,
        }

    # One selective checkpoint read per donor, regardless of how many roles it serves.
    loaded_checkpoint_states: dict[str, Any] = {}
    for path_text, roles in sorted(checkpoint_groups.items()):
        loaded_checkpoint_states[path_text] = mapper.load_selected_checkpoint_components(
            path_text,
            architecture=family,
            roles=roles,
        )

    for role, selection in sorted(components.items()):
        selection = dict(selection or {})
        choice = choices[role]
        source_path: Path = choice["source_path"]
        source_role = str(choice["source_role"])
        asset_type = str(choice["asset_type"])
        force_digital_extract = bool(choice["force_digital_extract"])
        source_mode = str(choice["mode"])

        if source_mode == "resident_component":
            # The builder receives the exact live handle through the CNRR-04 reuse
            # bundle.  Leaving mapped state untouched/empty prevents pointless disk IO.
            state = None
            tensor_count = 0
        elif source_mode == "base_donor_checkpoint":
            state = getattr(plan.mapped_state, role, None)
            if not state:
                raise ValueError(f"Base denoiser checkpoint does not provide required component role {role!r}.")
            tensor_count = len(state)
        elif source_mode.startswith("checkpoint_component_donor"):
            mapped = loaded_checkpoint_states[str(source_path)]
            state = getattr(mapped, source_role, None)
            if not state:
                raise ValueError(
                    f"Component donor checkpoint {source_path.name!r} does not contain registry role {source_role!r}."
                )
            setattr(plan.mapped_state, role, state)
            tensor_count = len(state)
        else:
            if source_path.suffix.lower() != ".safetensors":
                raise ValueError(
                    f"Standalone Advanced Models components must currently be safetensors files; got {source_path.name!r}."
                )
            state = dict(load_file(str(source_path), device="cpu"))
            if not state:
                raise ValueError(f"Standalone component {source_path.name!r} contained no tensors.")
            setattr(plan.mapped_state, role, state)
            tensor_count = len(state)

        if state is not None and source_mode == "base_donor_checkpoint":
            setattr(plan.mapped_state, role, state)
        source_plan = dict(choice.get("source_plan") or {})
        applied[role] = {
            "component_sha256": str(selection.get("component_sha256") or ""),
            "source_path": str(source_path),
            "source_role": source_role,
            "source_asset_type": asset_type,
            "source_mode": source_mode,
            "runtime_source_kind": str(source_plan.get("selected_source_kind") or source_mode),
            "runtime_source_reason": str(source_plan.get("reason") or "static_selection_fallback"),
            "runtime_source_cost_class": source_plan.get("cost_class"),
            "force_digital_extract": force_digital_extract,
            "tensor_count": tensor_count,
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
        "runtime_source_selection": dict(runtime_source_plan or {}),
        "checkpoint_donor_transaction_count": len(loaded_checkpoint_states),
    }


__all__ = ["apply_advanced_component_composition"]
