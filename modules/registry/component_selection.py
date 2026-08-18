from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.project_context import ProjectContext

from .asset_registry import AssetRegistry
from .contracts import (
    AVAILABILITY_AVAILABLE,
    ComponentOccurrence,
    ComponentSelection,
    CompositionIdentity,
    ResolvedComponent,
    SELECTION_AUTO,
    SELECTION_EXPLICIT,
    SELECTION_OFF,
    SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
    SOURCE_FORM_PHYSICAL_COMPONENT,
    SOURCE_FORM_RECONSTRUCTED_EXPORT,
    SOURCE_FORM_STANDALONE_SHARED,
    load_strategy_for_source_form,
    source_form_for_asset_type,
)
from .family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from .models import AssetRecord, ComponentSnapshotRecord, LOCATION_STATE_AVAILABLE
from .evidence_contracts import (
    POLICY_ACTION_DISABLE,
    POLICY_SCOPE_BASE,
    POLICY_SCOPE_GLOBAL,
    VALIDATION_BLOCKING,
    VALIDATION_RESULT_ERROR,
    VALIDATION_RESULT_FAIL,
    VALIDATION_RESULT_PASS,
    VALIDATION_STAGES,
)


ADVANCED_MODEL_SELECTION_VERSION = "advanced-model-components-v1"


@dataclass(frozen=True)
class ComponentRoleSpec:
    """Backward-compatible Step 0 view backed by a provider role definition."""

    role: str
    label: str
    required: bool
    off_allowed: bool = False
    auto_allowed: bool = True
    base_weight_role: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": self.label,
            "required": self.required,
            "off_allowed": self.off_allowed,
            "auto_allowed": self.auto_allowed,
            "base_weight_role": self.base_weight_role,
        }


# Compatibility export only. The authoritative role definitions live in registered
# ArchitectureFamilyProvider instances; UI/service code must not maintain a second
# family-role table.
FAMILY_ROLE_SPECS: dict[str, tuple[ComponentRoleSpec, ...]] = {
    provider.family_id: tuple(
        ComponentRoleSpec(
            role=definition.canonical_role_id,
            label=definition.display_label,
            required=definition.required,
            off_allowed=definition.off_allowed,
            auto_allowed=definition.auto_allowed,
            base_weight_role=definition.base_weight_role,
        )
        for definition in provider.role_definitions()
    )
    for provider in DEFAULT_FAMILY_PROVIDER_REGISTRY.providers()
}


def canonical_model_family(value: Any) -> str:
    return DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(value)


def _snapshot_metadata(record: ComponentSnapshotRecord) -> dict[str, Any]:
    try:
        payload = json.loads(record.metadata_json or "{}")
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


class ComponentSelectionService:
    """Registry-backed free-form model component selection.

    Family/role behavior comes from ArchitectureFamilyProvider contracts. Component
    identity is always the exact component fingerprint. A required role may use Auto
    only when exactly one unique eligible fingerprint exists. Optional roles never
    auto-enable.
    """

    def __init__(self, context: ProjectContext, registry: AssetRegistry | None = None) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.providers = DEFAULT_FAMILY_PROVIDER_REGISTRY

    @staticmethod
    def role_specs(family: str) -> tuple[ComponentRoleSpec, ...]:
        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.require(family)
        return tuple(
            ComponentRoleSpec(
                role=item.canonical_role_id,
                label=item.display_label,
                required=item.required,
                off_allowed=item.off_allowed,
                auto_allowed=item.auto_allowed,
                base_weight_role=item.base_weight_role,
            )
            for item in provider.role_definitions()
        )

    def _asset_map(self) -> dict[int, AssetRecord]:
        return {item.id: item for item in self.registry.list_assets(limit=1_000_000)}

    @staticmethod
    def _is_physical_like(source_form: str) -> bool:
        return str(source_form or "") in {
            SOURCE_FORM_PHYSICAL_COMPONENT,
            SOURCE_FORM_STANDALONE_SHARED,
            SOURCE_FORM_RECONSTRUCTED_EXPORT,
        }

    @staticmethod
    def _is_digital_like(source_form: str) -> bool:
        return str(source_form or "") == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT

    @classmethod
    def _source_status_payload(cls, sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        source_list = [dict(item) for item in sources]
        available = [item for item in source_list if cls._source_is_registered_available(item)]
        physical_available = sum(1 for item in available if cls._is_physical_like(str(item.get("source_form") or "")))
        digital_available = sum(1 for item in available if cls._is_digital_like(str(item.get("source_form") or "")))
        physical_total = sum(1 for item in source_list if cls._is_physical_like(str(item.get("source_form") or "")))
        digital_total = sum(1 for item in source_list if cls._is_digital_like(str(item.get("source_form") or "")))
        explicit_physical_total = sum(
            1 for item in source_list if str(item.get("source_form") or "") == SOURCE_FORM_PHYSICAL_COMPONENT
        )
        standalone_total = sum(
            1 for item in source_list if str(item.get("source_form") or "") == SOURCE_FORM_STANDALONE_SHARED
        )
        if physical_available and digital_available:
            code = "physical_and_digital"
            label = "Physical + Digital"
        elif physical_available:
            code = "physical"
            label = "Physical"
        elif digital_available:
            code = "digital"
            label = "Digital"
        else:
            code = "unavailable"
            label = "Unavailable"
        return {
            "source_status": code,
            "source_status_label": label,
            "available_source_count": len(available),
            "embedded_source_count": digital_total,
            "standalone_source_count": standalone_total,
            "physical_source_count": explicit_physical_total,
            "physical_like_source_count": physical_total,
            "digital_source_count": digital_total,
            "physical_available_source_count": physical_available,
            "digital_available_source_count": digital_available,
            "eligible_with_digital": bool(physical_available or digital_available),
            "eligible_without_digital": bool(physical_available),
        }

    @staticmethod
    def _source_payload(asset: AssetRecord, snapshot: ComponentSnapshotRecord) -> dict[str, Any]:
        metadata = _snapshot_metadata(snapshot)
        source_form = source_form_for_asset_type(asset.asset_type, metadata=metadata)
        registered_available = bool(
            asset.exists_on_disk and str(asset.location_state or "") == LOCATION_STATE_AVAILABLE
        )
        return {
            "asset_id": int(asset.id),
            "path": str(asset.path),
            "filename": str(asset.filename),
            "asset_type": str(asset.asset_type or "unclassified_asset"),
            "architecture": str(asset.architecture or "") or None,
            "architecture_state": str(asset.architecture_state or "observed_unclassified"),
            "component_role": str(snapshot.component_role),
            "exists_on_disk": registered_available,
            "availability_state": AVAILABILITY_AVAILABLE if registered_available else "missing",
            "location_state": str(asset.location_state or "missing"),
            "source_form": source_form,
            "embedded_state": "embedded" if source_form == "digital_checkpoint_component" else "standalone",
            "component_bytes": int(snapshot.total_bytes),
            "tensor_count": int(snapshot.tensor_count),
            "source_prefixes": json.loads(snapshot.source_prefixes_json or "[]") if snapshot.source_prefixes_json else [],
            "snapshot_version": str(snapshot.snapshot_version or ""),
        }

    @staticmethod
    def _records_by_component(records: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            digest = str(record.get("component_sha256") or "").strip().lower()
            if digest:
                grouped.setdefault(digest, []).append(dict(record))
        return grouped

    @staticmethod
    def _phase05_status(
        *,
        role: str,
        base_component_sha256: str = "",
        policy_records: Iterable[Mapping[str, Any]] = (),
        validation_records: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        base = str(base_component_sha256 or "").strip().lower()
        normalized_role = str(role or "").strip()
        policies = [dict(item) for item in policy_records]
        global_policies = [
            item for item in policies
            if str(item.get("policy_scope") or "").strip().lower() == POLICY_SCOPE_GLOBAL
            and str(item.get("policy_action") or "").strip().lower() == POLICY_ACTION_DISABLE
            and (
                not str(item.get("component_role") or "").strip()
                or str(item.get("component_role") or "").strip() == normalized_role
            )
        ]
        base_policies = [
            item for item in policies
            if base
            and str(item.get("policy_scope") or "").strip().lower() == POLICY_SCOPE_BASE
            and str(item.get("policy_action") or "").strip().lower() == POLICY_ACTION_DISABLE
            and str(item.get("base_component_sha256") or "").strip().lower() == base
            and (
                not str(item.get("component_role") or "").strip()
                or str(item.get("component_role") or "").strip() == normalized_role
            )
        ]

        latest_by_stage: dict[tuple[str, str], dict[str, Any]] = {}
        sorted_validations = sorted(
            (dict(item) for item in validation_records),
            key=lambda item: (
                str(item.get("validated_at") or item.get("updated_at") or ""),
                int(item.get("id") or 0),
            ),
            reverse=True,
        )
        latest_all_by_base_stage: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in sorted_validations:
            record_base = str(item.get("base_component_sha256") or "").strip().lower()
            record_role = str(item.get("component_role") or "").strip()
            stage = str(
                item.get("validation_stage")
                or item.get("evidence_type")
                or "structural"
            ).strip().lower()
            latest_all_by_base_stage.setdefault((record_base, record_role, stage), item)
        for item in sorted_validations:
            record_base = str(item.get("base_component_sha256") or "").strip().lower()
            if record_base and record_base != base:
                continue
            record_role = str(item.get("component_role") or "").strip()
            if record_role and record_role != normalized_role:
                continue
            stage = str(
                item.get("validation_stage")
                or item.get("evidence_type")
                or "structural"
            ).strip().lower()
            latest_by_stage.setdefault((record_base, stage), item)

        blocking_failures: list[dict[str, Any]] = []
        passed_stages: list[str] = []
        for (_record_base, stage), item in latest_by_stage.items():
            result = str(
                item.get("validation_result") or item.get("validation_state") or ""
            ).strip().lower()
            blocking = str(item.get("blocking_state") or "advisory").strip().lower()
            if result in {VALIDATION_RESULT_FAIL, VALIDATION_RESULT_ERROR} and blocking == VALIDATION_BLOCKING:
                blocking_failures.append(item)
            elif result == VALIDATION_RESULT_PASS:
                passed_stages.append(stage)

        if blocking_failures:
            validation_state = "validation_failed"
            validation_label = "Validation failed"
        elif passed_stages:
            validation_state = "validated"
            validation_label = "Validated"
        else:
            validation_state = "untested"
            validation_label = "Untested"

        if global_policies:
            exclusion_state = "globally_disabled"
            exclusion_label = "Globally disabled"
            reason = str(global_policies[0].get("reason") or "Globally disabled by component policy.")
        elif base_policies:
            exclusion_state = "disabled_for_base"
            exclusion_label = "Disabled for this base"
            reason = str(base_policies[0].get("reason") or "Disabled for the selected base component.")
        elif blocking_failures:
            exclusion_state = "validation_failed"
            exclusion_label = "Validation failed"
            reason = str(
                blocking_failures[0].get("error_message")
                or "A blocking compatibility validation failed for this exact fingerprint combination."
            )
        else:
            exclusion_state = "eligible"
            exclusion_label = validation_label
            reason = ""

        return {
            "eligible": exclusion_state == "eligible",
            "exclusion_state": exclusion_state,
            "exclusion_label": exclusion_label,
            "reason": reason,
            "global_disabled": bool(global_policies),
            "disabled_for_base": bool(base_policies),
            "validation_state": validation_state,
            "validation_label": validation_label,
            "passed_stages": sorted(
                set(passed_stages),
                key=lambda value: VALIDATION_STAGES.index(value) if value in VALIDATION_STAGES else -1,
            ),
            "per_base_exclusions": [
                {
                    "base_component_sha256": str(item.get("base_component_sha256") or ""),
                    "component_role": str(item.get("component_role") or ""),
                    "reason": str(item.get("reason") or ""),
                    "policy_source": str(item.get("policy_source") or ""),
                }
                for item in policies
                if str(item.get("policy_scope") or "").strip().lower() == POLICY_SCOPE_BASE
                and str(item.get("policy_action") or "").strip().lower() == POLICY_ACTION_DISABLE
            ],
            "blocking_failures_by_base": [
                {
                    "base_component_sha256": str(item.get("base_component_sha256") or ""),
                    "validation_stage": str(item.get("validation_stage") or ""),
                    "error_category": str(item.get("error_category") or ""),
                    "error_message": str(item.get("error_message") or ""),
                }
                for item in latest_all_by_base_stage.values()
                if str(item.get("blocking_state") or "").strip().lower() == VALIDATION_BLOCKING
                and str(item.get("validation_result") or item.get("validation_state") or "").strip().lower()
                in {VALIDATION_RESULT_FAIL, VALIDATION_RESULT_ERROR}
            ],
            "validation_passes_by_base": [
                {
                    "base_component_sha256": str(item.get("base_component_sha256") or ""),
                    "validation_stage": str(item.get("validation_stage") or ""),
                    "validated_at": str(item.get("validated_at") or ""),
                }
                for item in latest_all_by_base_stage.values()
                if str(item.get("validation_result") or item.get("validation_state") or "").strip().lower()
                == VALIDATION_RESULT_PASS
            ],
        }

    def catalog(self, *, base_component_sha256: str | None = None) -> dict[str, Any]:
        assets = self._asset_map()
        snapshots = self.registry.list_component_snapshots(limit=1_000_000)
        list_policies = getattr(self.registry, "list_component_policies", None)
        list_validations = getattr(self.registry, "list_component_validations", None)
        policies_by_hash = self._records_by_component(
            list_policies(limit=1_000_000) if callable(list_policies) else ()
        )
        validations_by_hash = self._records_by_component(
            list_validations(limit=1_000_000) if callable(list_validations) else ()
        )
        base_digest = str(base_component_sha256 or "").strip().lower()
        by_hash: dict[str, list[tuple[AssetRecord, ComponentSnapshotRecord]]] = {}
        for snapshot in snapshots:
            digest = str(snapshot.component_sha256 or "").strip().lower()
            asset = assets.get(int(snapshot.asset_id))
            if not digest or asset is None:
                continue
            by_hash.setdefault(digest, []).append((asset, snapshot))

        # Structural family inheritance is only valid for standalone component
        # candidates. A digital component already has a checkpoint-family source,
        # and matching another family's tensor layout is not sufficient evidence to
        # relabel that different fingerprint as cross-family compatible.
        standalone_families_by_structure_role: dict[tuple[str, str], set[str]] = {}
        for snapshot in snapshots:
            asset = assets.get(int(snapshot.asset_id))
            if asset is None or str(asset.asset_type or "").strip().lower() != "checkpoint":
                continue
            family = canonical_model_family(asset.architecture)
            structure = str(snapshot.structure_sha256 or "").strip().lower()
            role = str(snapshot.component_role or "").strip()
            if family and structure and role:
                standalone_families_by_structure_role.setdefault((structure, role), set()).add(family)

        families_by_hash: dict[str, set[str]] = {}
        family_evidence_by_hash: dict[str, dict[str, set[str]]] = {}
        for digest, occurrences in by_hash.items():
            families: set[str] = set()
            evidence: dict[str, set[str]] = {}

            def record_family(family_id: str, basis: str) -> None:
                canonical = canonical_model_family(family_id)
                if not canonical:
                    return
                families.add(canonical)
                evidence.setdefault(canonical, set()).add(basis)

            for asset, snapshot in occurrences:
                family = canonical_model_family(asset.architecture)
                if family:
                    # A family observed on a source containing this exact component
                    # fingerprint is direct evidence for that fingerprint only. Do
                    # not transfer it to a different fingerprint merely because the
                    # tensor layout/role happens to match.
                    record_family(family, "exact_fingerprint_source_architecture")

                metadata = _snapshot_metadata(snapshot)
                standalone_evidence = metadata.get("standalone_component_evidence")
                is_standalone_component = str(asset.asset_type or "").strip().lower() in {"vae", "text_encoder"}
                if is_standalone_component and isinstance(standalone_evidence, Mapping):
                    for candidate in standalone_evidence.get("provider_family_evidence") or ():
                        record_family(str(candidate), "standalone_provider_structural_contract")

                if is_standalone_component:
                    structure = str(snapshot.structure_sha256 or "").strip().lower()
                    role = str(snapshot.component_role or "").strip()
                    if structure and role:
                        for candidate in standalone_families_by_structure_role.get((structure, role), set()):
                            record_family(candidate, "standalone_structure_matches_family_checkpoint")

            # Exact component fingerprints are deduplicated across all occurrences,
            # so if the same bytes genuinely occur in multiple family checkpoints,
            # the source-architecture evidence above legitimately admits all of those
            # families. A unique provider role remains safe structural evidence (for
            # example SD3 text_encoder_3), but shared role names alone never imply a
            # family.
            observed_roles = {snapshot.component_role for _asset, snapshot in occurrences}
            for role in observed_roles:
                role_providers = [
                    provider.family_id
                    for provider in self.providers.providers()
                    if provider.role_definition(role) is not None
                ]
                if len(role_providers) == 1:
                    record_family(role_providers[0], "unique_provider_role_contract")

            families_by_hash[digest] = families
            family_evidence_by_hash[digest] = evidence

        family_payloads: list[dict[str, Any]] = []
        for provider in self.providers.providers():
            family = provider.family_id
            specs = self.role_specs(family)
            role_payloads: list[dict[str, Any]] = []
            for spec in specs:
                components: list[dict[str, Any]] = []
                for digest, occurrences in by_hash.items():
                    if family not in families_by_hash.get(digest, set()):
                        continue
                    matching = [(asset, snapshot) for asset, snapshot in occurrences if snapshot.component_role == spec.role]
                    if not matching:
                        continue
                    sources = [self._source_payload(asset, snapshot) for asset, snapshot in matching]
                    # A component fingerprint may legitimately occur in more than one
                    # architecture family. Selection is family-scoped, so keep only source
                    # occurrences that can hydrate this provider family. This prevents an
                    # SD1.x selection from resolving the same bytes through an SDXL/SD2/SD3
                    # donor checkpoint merely because that donor sorts first globally.
                    sources = [
                        source for source in sources
                        if self._source_matches_family(source, family)
                    ]
                    if not sources:
                        continue
                    sources.sort(key=self._source_sort_key)
                    preferred = sources[0]
                    availability = self._source_status_payload(sources)
                    selectable_with_digital = self._preferred_source(
                        sources,
                        require_checkpoint=spec.base_weight_role,
                        allow_digital_components=True,
                        family=family,
                    ) is not None
                    selectable_without_digital = self._preferred_source(
                        sources,
                        require_checkpoint=spec.base_weight_role,
                        allow_digital_components=False,
                        family=family,
                    ) is not None
                    phase05 = self._phase05_status(
                        role=spec.role,
                        base_component_sha256=("" if spec.base_weight_role else base_digest),
                        policy_records=policies_by_hash.get(digest, ()),
                        validation_records=validations_by_hash.get(digest, ()),
                    )
                    selectable_with_digital = bool(selectable_with_digital and phase05["eligible"])
                    selectable_without_digital = bool(selectable_without_digital and phase05["eligible"])
                    components.append({
                        "component_sha256": digest,
                        "short_hash": digest[:8],
                        "role": spec.role,
                        "display_name": self._display_name(digest, preferred, len(sources)),
                        "component_bytes": max(int(item["component_bytes"]) for item in sources),
                        "tensor_count": max(int(item["tensor_count"]) for item in sources),
                        "source_count": len(sources),
                        "selection_label": self._selection_label(digest, preferred, availability["source_status_label"], len(sources)),
                        "family_evidence": {
                            family_id: sorted(bases)
                            for family_id, bases in sorted(family_evidence_by_hash.get(digest, {}).items())
                        },
                        **availability,
                        "selectable_with_digital": selectable_with_digital,
                        "selectable_without_digital": selectable_without_digital,
                        "phase05": phase05,
                        "sources": sources,
                    })
                components.sort(key=lambda item: (item["display_name"].casefold(), item["component_sha256"]))
                eligible_with_digital_count = sum(
                    1 for component in components if bool(component.get("selectable_with_digital"))
                )
                eligible_without_digital_count = sum(
                    1 for component in components if bool(component.get("selectable_without_digital"))
                )
                role_payloads.append({
                    **spec.to_dict(),
                    "unique_component_count": len(components),
                    "eligible_component_count_with_digital": eligible_with_digital_count,
                    "eligible_component_count_without_digital": eligible_without_digital_count,
                    "auto_resolvable": bool(spec.required and spec.auto_allowed and eligible_with_digital_count == 1),
                    "components": components,
                })
            if any(role["components"] for role in role_payloads):
                required_roles = [role for role in role_payloads if role["required"]]
                required_role_coverage_complete = all(
                    any(bool(component.get("selectable_with_digital")) for component in role["components"])
                    for role in required_roles
                )
                family_payloads.append({
                    "family": family,
                    "label": provider.display_label,
                    "provider_version": provider.version,
                    "base_weight_role": provider.base_weight_role,
                    "required_role_coverage_complete": required_role_coverage_complete,
                    "constructible": bool(required_role_coverage_complete and provider.supports_runtime_composition()),
                    "roles": role_payloads,
                })

        return {
            "version": ADVANCED_MODEL_SELECTION_VERSION,
            "provider_contract_version": self.providers.to_dict()["contract_version"],
            "compatibility_policy_version": "component-phase05-v1",
            "base_component_sha256": base_digest or None,
            "families": family_payloads,
            "rules": {
                "required_auto": "select_when_exactly_one_unique_component_else_require_user_choice",
                "optional_auto": "off",
                "optional_default": "off",
                "identity": "component_sha256",
                "family_roles": "architecture_family_provider",
            },
        }

    @staticmethod
    def _family_label(family: str) -> str:
        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(family)
        return provider.display_label if provider is not None else family

    @staticmethod
    def _display_name(digest: str, preferred: Mapping[str, Any], source_count: int) -> str:
        name = Path(str(preferred.get("filename") or "component")).stem
        source_form = str(preferred.get("source_form") or "unknown")
        location = {
            "physical_component": "physical",
            "standalone_shared": "standalone",
            "digital_checkpoint_component": "digital",
            "reconstructed_export": "reconstructed",
        }.get(source_form, "source")
        shared = f" · {source_count} sources" if source_count > 1 else ""
        return f"{name} · {digest[:8]} · {location}{shared}"

    @staticmethod
    def _selection_label(digest: str, preferred: Mapping[str, Any], source_status_label: str, source_count: int) -> str:
        shared = f" · {source_count} sources" if source_count > 1 else ""
        name = Path(str(preferred.get("filename") or "component")).stem
        return f"{name} · {digest[:8]} · {source_status_label}{shared}"

    def resolve_selection(
        self,
        family: str,
        selections: Mapping[str, Any] | None,
        *,
        t5_device: Any = "cpu",
        allow_digital_components: Any = True,
    ) -> dict[str, Any]:
        canonical = canonical_model_family(family)
        provider = self.providers.require(canonical)
        specs = self.role_specs(canonical)
        requested_all = {str(key): str(value or "").strip().lower() for key, value in dict(selections or {}).items()}
        provider_roles = {spec.role for spec in specs}
        ignored_unsupported_selections = {
            key: value
            for key, value in requested_all.items()
            if key not in provider_roles
        }
        requested = {
            key: value
            for key, value in requested_all.items()
            if key in provider_roles
        }
        catalog = self.catalog()
        family_entry = next((item for item in catalog["families"] if item["family"] == canonical), None)
        if family_entry is None:
            raise ValueError(
                f"No scanned component-registry entries are available for {provider.display_label}. "
                "Update the component registry before using Advanced Models."
            )
        roles = {item["role"]: item for item in family_entry["roles"]}
        resolved: dict[str, dict[str, Any]] = {}
        unresolved: list[str] = []
        list_policies = getattr(self.registry, "list_component_policies", None)
        list_validations = getattr(self.registry, "list_component_validations", None)
        policies_by_hash = self._records_by_component(
            list_policies(limit=1_000_000) if callable(list_policies) else ()
        )
        validations_by_hash = self._records_by_component(
            list_validations(limit=1_000_000) if callable(list_validations) else ()
        )
        normalized_t5_device = str(t5_device or "cpu").strip().lower()
        digital_allowed = bool(allow_digital_components)
        if normalized_t5_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("T5 execution device must be one of: cpu, cuda, auto.")

        ordered_specs = sorted(specs, key=lambda item: (0 if item.role == provider.base_weight_role else 1, item.label.casefold(), item.role))
        for spec in ordered_specs:
            role_entry = roles.get(spec.role, {"components": []})
            components = list(role_entry.get("components") or [])
            selected_base_sha = ""
            if not spec.base_weight_role:
                selected_base_sha = str(
                    (resolved.get(provider.base_weight_role) or {}).get("component_sha256") or ""
                ).strip().lower()

            def phase05_status(item: Mapping[str, Any]) -> dict[str, Any]:
                digest = str(item.get("component_sha256") or "").strip().lower()
                return self._phase05_status(
                    role=spec.role,
                    base_component_sha256=selected_base_sha,
                    policy_records=policies_by_hash.get(digest, ()),
                    validation_records=validations_by_hash.get(digest, ()),
                )

            eligible_components = [
                item
                for item in components
                if self._preferred_source(
                    item.get("sources") or [],
                    require_checkpoint=spec.base_weight_role,
                    allow_digital_components=digital_allowed,
                    family=canonical,
                )
                is not None
                and bool(phase05_status(item).get("eligible"))
            ]
            by_digest = {str(item["component_sha256"]): item for item in components}
            value = requested.get(spec.role, "auto" if spec.required else "off")
            if value in {"", "default"}:
                value = "auto" if spec.required else "off"

            if not spec.required and value in {"auto", "off", "none", "disabled"}:
                request_contract = ComponentSelection(canonical, spec.role, SELECTION_OFF)
                continue
            if spec.required and value == "auto":
                request_contract = ComponentSelection(canonical, spec.role, SELECTION_AUTO)
                if len(eligible_components) == 1:
                    selected = eligible_components[0]
                elif not components:
                    unresolved.append(f"{spec.label}: no compatible scanned components")
                    continue
                elif not eligible_components:
                    source_hint = "physical-or-standalone" if not digital_allowed else "current"
                    unresolved.append(
                        f"{spec.label}: no compatible scanned components have a usable {source_hint} source"
                    )
                    continue
                else:
                    unresolved.append(
                        f"{spec.label}: Auto found {len(eligible_components)} unique compatible components; choose one explicitly"
                    )
                    continue
            else:
                request_contract = ComponentSelection(
                    canonical,
                    spec.role,
                    SELECTION_EXPLICIT,
                    explicit_fingerprint=value,
                    placement_policy=(normalized_t5_device if spec.role == "text_encoder_3" else ""),
                )
                selected = by_digest.get(value)
                if selected is None:
                    unresolved.append(f"{spec.label}: selected component {value or '<empty>'!r} is not available for {canonical}")
                    continue

            selected_phase05 = phase05_status(selected)
            if not bool(selected_phase05.get("eligible")):
                label = str(selected_phase05.get("exclusion_label") or "Unavailable")
                reason = str(selected_phase05.get("reason") or "The selected component is disabled by compatibility policy.")
                unresolved.append(
                    f"{spec.label}: {label} for component {str(selected['component_sha256'])[:8]} ({reason})"
                )
                continue

            source = self._preferred_source(
                selected.get("sources") or [],
                require_checkpoint=spec.base_weight_role,
                allow_digital_components=digital_allowed,
                family=canonical,
            )
            if source is None:
                guidance = "enable digital sources or rescan the library" if not digital_allowed else "rescan the library"
                unresolved.append(
                    f"{spec.label}: no usable source file remains for component {selected['component_sha256'][:8]} ({guidance})"
                )
                continue

            occurrence = ComponentOccurrence(
                component_sha256=str(selected["component_sha256"]),
                asset_id=(int(source["asset_id"]) if source.get("asset_id") is not None else None),
                asset_path=str(source.get("path") or ""),
                source_form=str(source.get("source_form") or "unknown"),
                embedded_state=str(source.get("embedded_state") or "unknown"),
                role=spec.role,
                source_prefixes=tuple(str(item) for item in (source.get("source_prefixes") or ())),
                availability_state=str(source.get("availability_state") or "unknown"),
                locator={"source_prefixes": list(source.get("source_prefixes") or ())},
                scan_timestamp="",
                scanner_version=str(source.get("snapshot_version") or ""),
                provider_family=provider.family_id,
                provider_version=provider.version,
            )
            resolved_contract = ResolvedComponent(
                component_sha256=str(selected["component_sha256"]),
                role=spec.role,
                source=occurrence,
                provider_family=provider.family_id,
                provider_version=provider.version,
                availability_evidence={"exists_on_disk": bool(source.get("exists_on_disk"))},
                exclusion_state=str(selected_phase05.get("exclusion_state") or "eligible"),
                validation_state=str(selected_phase05.get("validation_state") or "untested"),
                load_strategy=load_strategy_for_source_form(occurrence.source_form),
            )
            resolved[spec.role] = {
                "role": spec.role,
                "label": spec.label,
                "required": spec.required,
                "component_sha256": str(selected["component_sha256"]),
                "short_hash": str(selected["component_sha256"])[:8],
                "source": source,
                "source_count": int(selected.get("source_count") or len(selected.get("sources") or [])),
                "source_status": str(selected.get("source_status") or "unavailable"),
                "source_status_label": str(selected.get("source_status_label") or "Unavailable"),
                "provider_family": provider.family_id,
                "provider_version": provider.version,
                "source_form": occurrence.source_form,
                "load_strategy": resolved_contract.load_strategy,
                "phase05": selected_phase05,
                "selection_contract": request_contract.to_dict(),
                "resolved_contract": resolved_contract.to_dict(),
            }

        if unresolved:
            raise ValueError("Advanced Models component selection is incomplete: " + " | ".join(unresolved))

        denoiser_role = provider.base_weight_role
        denoiser = resolved.get(denoiser_role)
        if denoiser is None:
            raise ValueError(f"Advanced Models requires a {denoiser_role} component for {canonical}.")
        base_path = str((denoiser.get("source") or {}).get("path") or "")
        if not base_path:
            raise ValueError("Advanced Models could not resolve the denoiser source path.")

        if "text_encoder_3" not in resolved:
            normalized_t5_device = "off"

        behavior_choices: dict[str, Any] = {}
        if "text_encoder_3" in resolved:
            behavior_choices["text_encoder_3_placement"] = normalized_t5_device
        composition = CompositionIdentity.derive(
            family=canonical,
            provider_version=provider.version,
            components={role: item["component_sha256"] for role, item in resolved.items()},
            behavior_choices=behavior_choices,
        )
        return {
            "version": ADVANCED_MODEL_SELECTION_VERSION,
            "enabled": True,
            "family": canonical,
            "family_label": provider.display_label,
            "provider_version": provider.version,
            "base_denoiser_role": denoiser_role,
            "base_source_path": base_path,
            "components": resolved,
            "digital_components_allowed": digital_allowed,
            "t5_device": normalized_t5_device,
            "composition_identity_version": composition.identity_version,
            "composition_sha256": composition.composition_sha256,
            "composition_short_hash": composition.composition_sha256[:12],
            "composition_contract": composition.to_dict(),
            "ignored_unsupported_selections": ignored_unsupported_selections,
            "checkpoint_selection_ignored": True,
        }

    @staticmethod
    def _source_is_registered_available(item: Mapping[str, Any]) -> bool:
        return bool(
            item.get("exists_on_disk")
            and str(item.get("location_state") or LOCATION_STATE_AVAILABLE) == LOCATION_STATE_AVAILABLE
        )

    @classmethod
    def _source_sort_key(cls, item: Mapping[str, Any]) -> tuple[int, int, str, str]:
        source_priority = {
            "physical_component": 0,
            "standalone_shared": 1,
            "digital_checkpoint_component": 2,
            "reconstructed_export": 3,
            "unknown": 9,
        }
        return (
            0 if cls._source_is_registered_available(item) else 1,
            source_priority.get(str(item.get("source_form") or "unknown"), 9),
            str(item.get("filename") or "").casefold(),
            str(item.get("path") or "").casefold(),
        )

    @classmethod
    def _source_matches_family(cls, item: Mapping[str, Any], family: str) -> bool:
        target = canonical_model_family(family)
        if not target:
            return True
        source_family = canonical_model_family(item.get("architecture"))
        source_form = str(item.get("source_form") or "")
        asset_type = str(item.get("asset_type") or "").strip().lower()
        # Digital/checkpoint occurrences must carry direct same-family donor
        # provenance. A blank-family standalone/physical occurrence may remain
        # eligible when the fingerprint itself has already been qualified for
        # the family by the catalog evidence rules.
        if source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT or asset_type == "checkpoint":
            return bool(source_family and source_family == target)
        if source_family:
            return source_family == target
        return cls._is_physical_like(source_form)

    @classmethod
    def _preferred_source(
        cls,
        sources: Iterable[Mapping[str, Any]],
        *,
        require_checkpoint: bool,
        allow_digital_components: bool,
        family: str = "",
    ) -> dict[str, Any] | None:
        usable = [dict(item) for item in sources if cls._source_is_registered_available(item)]
        if family:
            usable = [item for item in usable if cls._source_matches_family(item, family)]
        if require_checkpoint:
            usable = [item for item in usable if item.get("asset_type") == "checkpoint"]
        if not allow_digital_components:
            usable = [item for item in usable if cls._is_physical_like(str(item.get("source_form") or ""))]
        if not usable:
            return None
        usable.sort(key=cls._source_sort_key)
        return usable[0]


__all__ = [
    "ADVANCED_MODEL_SELECTION_VERSION",
    "FAMILY_ROLE_SPECS",
    "ComponentRoleSpec",
    "ComponentSelectionService",
    "canonical_model_family",
]
