from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.registry.component_selection import canonical_model_family

from .artifact_io import _load_yaml_mapping, _slug, _utc_now
from .contracts import (
    BlueprintSnapshot,
    QUALIFICATION_RUNNER_SCHEMA_VERSION,
    QualificationCase,
    QualificationPattern,
)


class QualificationPlanningMixin:
    """Cohesive qualification-runner responsibility mixin used by the public facade."""

    @staticmethod
    def _case_with_forced_component_source(
        case: QualificationCase,
        *,
        role: str,
        source_payload: Mapping[str, Any],
        case_id: str,
        label: str,
        source_kind: str,
        source_asset_id: int,
    ) -> QualificationCase:
        resolved = copy.deepcopy(dict(case.resolved_composition))
        components = copy.deepcopy(dict(resolved.get("components") or {}))
        if role not in components:
            raise ValueError(f"Resolved composition does not contain component role {role!r}.")
        components[role] = dict(components[role])
        components[role]["source"] = copy.deepcopy(dict(source_payload))
        resolved["components"] = components
        request = copy.deepcopy(dict(case.request_payload))
        request["_advanced_model_resolved"] = copy.deepcopy(resolved)
        request["qualification_parity_role"] = role
        request["qualification_parity_source_kind"] = source_kind
        request["qualification_parity_source_asset_id"] = int(source_asset_id)
        return QualificationCase(
            case_id=case_id,
            label=label,
            mutation_kind="component_source_parity",
            mutation={
                "component_role": role,
                "component_sha256": str(components[role].get("component_sha256") or ""),
                "source_kind": source_kind,
                "source_asset_id": int(source_asset_id),
                "source_path": str(source_payload.get("path") or ""),
                "reference_case_id": "control",
            },
            request_payload=request,
            resolved_composition=resolved,
            parent_case_id="control",
        )

    def build_component_source_parity_plan(
        self,
        *,
        model_path: str | Path,
        components: Iterable[Mapping[str, Any]],
        profile_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build a control plus forced digital-extraction parity cases for one checkpoint.

        Each component spec requires ``role`` and ``component_sha256`` and may provide a
        ``peer_asset_id``. The self case deliberately forces the mapper/extraction path even
        when the donor checkpoint is also the base checkpoint; otherwise a same-file source
        would reuse the already mapped blueprint state and would not validate digital extraction.
        """
        blueprint = self.blueprint_for_model(model_path)
        profile = self.load_generation_profile(blueprint, profile_path)
        base_request = copy.deepcopy(dict(profile["request"]))
        control = self._build_case(
            blueprint=blueprint,
            base_request=base_request,
            case_id="control",
            label="Untouched blueprint control",
            mutation_kind="control",
            mutation={"blueprint_components": True, "parity_reference": True},
        )
        cases: list[QualificationCase] = [control]
        normalized_specs: list[dict[str, Any]] = []
        for raw in components:
            spec = dict(raw)
            role = str(spec.get("role") or "").strip()
            digest = str(spec.get("component_sha256") or "").strip().lower()
            if not role or not digest:
                raise ValueError("Parity component specs require role and component_sha256.")
            expected = str(blueprint.components.get(role) or "").strip().lower()
            if expected != digest:
                raise ValueError(
                    f"Checkpoint blueprint role {role!r} has hash {expected or '<missing>'}, not requested parity hash {digest}."
                )
            self_source = self._component_source_payload(
                asset_id=blueprint.asset_id,
                role=role,
                component_sha256=digest,
                force_digital_extract=True,
            )
            self_case = self._case_with_forced_component_source(
                control,
                role=role,
                source_payload=self_source,
                case_id=f"{_slug(role)}-digital-self",
                label=f"{role} digital extraction from selected checkpoint",
                source_kind="self",
                source_asset_id=blueprint.asset_id,
            )
            cases.append(self_case)
            peer_asset_id = spec.get("peer_asset_id")
            peer_source: dict[str, Any] | None = None
            if peer_asset_id is not None:
                peer_source = self._component_source_payload(
                    asset_id=int(peer_asset_id),
                    role=role,
                    component_sha256=digest,
                    force_digital_extract=True,
                )
                peer_asset = self.registry.get_asset_by_id(int(peer_asset_id))
                peer_case = self._case_with_forced_component_source(
                    control,
                    role=role,
                    source_payload=peer_source,
                    case_id=f"{_slug(role)}-digital-peer",
                    label=f"{role} same-hash digital extraction from {peer_asset.filename if peer_asset else peer_asset_id}",
                    source_kind="peer",
                    source_asset_id=int(peer_asset_id),
                )
                cases.append(peer_case)
            normalized_specs.append(
                {
                    "role": role,
                    "component_sha256": digest,
                    "self_asset_id": blueprint.asset_id,
                    "self_source": self_source,
                    "peer_asset_id": int(peer_asset_id) if peer_asset_id is not None else None,
                    "peer_source": peer_source,
                }
            )
        return {
            "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "blueprint": blueprint.to_dict(),
            "profile": profile,
            "pattern": {
                "pattern_id": "model-component-digital-parity",
                "label": "Model component digital parity",
                "mutation_kind": "component_source_parity",
                "description": "Untouched blueprint control versus forced digital extraction from self and a same-hash peer donor.",
            },
            "parity_components": normalized_specs,
            "runtime_choices": self.runtime_choices(),
            "component_choices": self.component_choices(blueprint),
            "cases": [item.to_dict() for item in cases],
        }

    @staticmethod
    def load_patterns(path: str | Path) -> dict[str, QualificationPattern]:
        payload = _load_yaml_mapping(Path(path))
        raw_patterns = payload.get("patterns") or {}
        if not isinstance(raw_patterns, dict):
            raise ValueError("Pattern file must contain a 'patterns' mapping.")
        return {
            str(pattern_id): QualificationPattern.from_dict(str(pattern_id), spec or {})
            for pattern_id, spec in raw_patterns.items()
        }

    def _resolve_case_composition(
        self,
        blueprint: BlueprintSnapshot,
        components: Mapping[str, Any],
        *,
        t5_device: str = "cpu",
    ) -> dict[str, Any]:
        resolved = self.selection.resolve_selection(
            blueprint.family,
            components,
            t5_device=t5_device,
            allow_digital_components=True,
        )
        wrong_family_sources: list[str] = []
        for role, item in dict(resolved.get("components") or {}).items():
            source = dict(item.get("source") or {})
            source_family = canonical_model_family(source.get("architecture"))
            source_form = str(source.get("source_form") or "")
            if source_family and source_family != blueprint.family:
                wrong_family_sources.append(
                    f"{role}:{Path(str(source.get('path') or '')).name}:{source_family}:{source_form}"
                )
        if wrong_family_sources:
            raise ValueError(
                "Resolved composition crossed architecture families: " + ", ".join(wrong_family_sources)
            )
        return resolved

    @staticmethod
    def _apply_pattern_overrides(request: dict[str, Any], pattern: QualificationPattern) -> None:
        for key, value in dict(pattern.request_overrides).items():
            request[str(key)] = copy.deepcopy(value)

    def _build_case(
        self,
        *,
        blueprint: BlueprintSnapshot,
        base_request: Mapping[str, Any],
        case_id: str,
        label: str,
        mutation_kind: str,
        mutation: Mapping[str, Any],
        component_overrides: Mapping[str, str] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
        parent_case_id: str = "",
    ) -> QualificationCase:
        request = copy.deepcopy(dict(base_request))
        components = dict(blueprint.components)
        components.update({str(key): str(value) for key, value in dict(component_overrides or {}).items()})
        request.update(copy.deepcopy(dict(request_overrides or {})))
        request["advanced_models_enabled"] = True
        request["advanced_model_family"] = blueprint.family
        request["advanced_model_components"] = components
        request["advanced_model_allow_digital_components"] = True
        request["vae_path"] = None
        t5_device = str(request.get("advanced_model_t5_device") or "cpu")
        resolved = self._resolve_case_composition(blueprint, components, t5_device=t5_device)
        request["model_path"] = str(resolved["base_source_path"])
        request["_advanced_model_resolved"] = resolved
        request["advanced_model_composition_sha256"] = str(resolved["composition_sha256"])
        request["text_encoder_3_device"] = str(resolved.get("t5_device") or "off")
        return QualificationCase(
            case_id=case_id,
            label=label,
            mutation_kind=mutation_kind,
            mutation=dict(mutation),
            request_payload=request,
            resolved_composition=resolved,
            parent_case_id=parent_case_id,
        )

    def build_plan(
        self,
        *,
        model_path: str | Path,
        pattern: QualificationPattern,
        profile_path: str | Path | None = None,
    ) -> dict[str, Any]:
        blueprint = self.blueprint_for_model(model_path)
        profile = self.load_generation_profile(blueprint, profile_path)
        base_request = copy.deepcopy(dict(profile["request"]))
        self._apply_pattern_overrides(base_request, pattern)
        component_choices = self.component_choices(blueprint)
        runtime_choices = self.runtime_choices()
        cases: list[QualificationCase] = []

        if pattern.include_control or pattern.mutation_kind == "control":
            cases.append(
                self._build_case(
                    blueprint=blueprint,
                    base_request=base_request,
                    case_id="control",
                    label="Blueprint control",
                    mutation_kind="control",
                    mutation={"blueprint_components": True},
                )
            )

        kind = pattern.mutation_kind
        if kind == "component_role":
            role = pattern.component_role
            if not role:
                raise ValueError(f"Pattern {pattern.pattern_id!r} requires component_role.")
            blueprint_digest = blueprint.components.get(role, "")
            for index, candidate in enumerate(component_choices.get(role, []), start=1):
                digest = str(candidate.get("value") or "")
                if not digest or digest == blueprint_digest:
                    continue
                cases.append(
                    self._build_case(
                        blueprint=blueprint,
                        base_request=base_request,
                        case_id=f"{_slug(role)}-{index:03d}-{digest[:8]}",
                        label=f"{role} -> {candidate.get('label') or digest[:8]}",
                        mutation_kind=kind,
                        mutation={
                            "component_role": role,
                            "from": blueprint_digest,
                            "to": digest,
                            "candidate_label": candidate.get("label"),
                        },
                        component_overrides={role: digest},
                    )
                )
        elif kind == "scheduler":
            sampler = str(base_request.get("sampler_name") or "")
            for scheduler in runtime_choices["schedulers"]:
                if scheduler == str(base_request.get("scheduler_name") or ""):
                    continue
                try:
                    compatibility = self.runtime_registry.validate_pair(sampler, scheduler)
                    compatibility.raise_if_incompatible()
                except Exception:
                    continue
                cases.append(
                    self._build_case(
                        blueprint=blueprint,
                        base_request=base_request,
                        case_id=f"scheduler-{_slug(scheduler)}",
                        label=f"Scheduler -> {scheduler}",
                        mutation_kind=kind,
                        mutation={"field": "scheduler_name", "value": scheduler},
                        request_overrides={"scheduler_name": scheduler},
                    )
                )
        elif kind == "sampler":
            scheduler = str(base_request.get("scheduler_name") or "")
            for sampler in runtime_choices["samplers"]:
                if sampler == str(base_request.get("sampler_name") or ""):
                    continue
                try:
                    compatibility = self.runtime_registry.validate_pair(sampler, scheduler)
                    compatibility.raise_if_incompatible()
                except Exception:
                    continue
                cases.append(
                    self._build_case(
                        blueprint=blueprint,
                        base_request=base_request,
                        case_id=f"sampler-{_slug(sampler)}",
                        label=f"Sampler -> {sampler}",
                        mutation_kind=kind,
                        mutation={"field": "sampler_name", "value": sampler},
                        request_overrides={"sampler_name": sampler},
                    )
                )
        elif kind == "steps":
            values = pattern.values or (4, 8, 12, 20, 28, 40)
            current = int(base_request.get("steps") or 0)
            for value in values:
                steps = int(value)
                if steps <= 0 or steps == current:
                    continue
                cases.append(
                    self._build_case(
                        blueprint=blueprint,
                        base_request=base_request,
                        case_id=f"steps-{steps}",
                        label=f"Steps -> {steps}",
                        mutation_kind=kind,
                        mutation={"field": "steps", "value": steps},
                        request_overrides={
                            "steps": steps,
                            # A steps sweep must not silently restore the architecture
                            # recommendation we are intentionally testing against.
                            "model_enforce_recommended_steps": False,
                        },
                    )
                )
        elif kind not in {"control", ""}:
            raise ValueError(f"Unsupported qualification mutation_kind: {kind!r}")

        if not cases:
            raise ValueError(f"Pattern {pattern.pattern_id!r} produced no qualification cases.")

        return {
            "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "pattern": pattern.to_dict(),
            "blueprint": blueprint.to_dict(),
            "generation_profile": profile,
            "runtime_choices": runtime_choices,
            "component_choices": component_choices,
            "cases": [case.to_dict() for case in cases],
        }
