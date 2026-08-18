from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import (
    ArchitectureFamilyProvider,
    ComponentRoleDefinition,
    SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
    SOURCE_FORM_PHYSICAL_COMPONENT,
    SOURCE_FORM_STANDALONE_SHARED,
)
from .analysis_contracts import ComponentAnalysisLayout, ComponentAnalysisNode


PROVIDER_CONTRACT_VERSION = "architecture-family-provider-v1"
_COMMON_SOURCES = (
    SOURCE_FORM_PHYSICAL_COMPONENT,
    SOURCE_FORM_STANDALONE_SHARED,
    SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
)


def _required_role(role: str, label: str, *, base: bool = False, constraints: Mapping[str, Any] | None = None) -> ComponentRoleDefinition:
    return ComponentRoleDefinition(
        canonical_role_id=role,
        display_label=label,
        required=True,
        off_allowed=False,
        auto_allowed=True,
        expected_source_kinds=_COMMON_SOURCES,
        base_weight_role=base,
        structural_constraints=dict(constraints or {}),
    )


def _optional_role(role: str, label: str, *, constraints: Mapping[str, Any] | None = None) -> ComponentRoleDefinition:
    return ComponentRoleDefinition(
        canonical_role_id=role,
        display_label=label,
        required=False,
        off_allowed=True,
        auto_allowed=False,
        expected_source_kinds=_COMMON_SOURCES,
        base_weight_role=False,
        structural_constraints=dict(constraints or {}),
    )


@dataclass(frozen=True)
class StaticArchitectureFamilyProvider:
    family_id: str
    display_label: str
    architecture_aliases: tuple[str, ...]
    required_roles: tuple[ComponentRoleDefinition, ...]
    optional_roles: tuple[ComponentRoleDefinition, ...] = ()
    base_weight_role: str = ""
    version: str = PROVIDER_CONTRACT_VERSION
    extraction_rules: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    digital_hydration_roles: tuple[str, ...] = ()
    runtime_composition: bool = True
    runtime_composition_validation_state: str = "implementation_present_unvalidated"
    digital_hydration_validation_by_role: Mapping[str, str] = field(default_factory=dict)
    placement_by_role: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    blueprint_support: Mapping[str, Any] = field(default_factory=dict)
    analysis_layouts: Mapping[str, ComponentAnalysisLayout] = field(default_factory=dict)

    def role_definitions(self) -> tuple[ComponentRoleDefinition, ...]:
        return self.required_roles + self.optional_roles

    def role_definition(self, role: str) -> ComponentRoleDefinition | None:
        token = str(role or "").strip()
        return next((item for item in self.role_definitions() if item.canonical_role_id == token), None)

    def structurally_compatible(self, *, family: str, role: str) -> bool:
        return canonicalize_family(family, providers=(self,)) == self.family_id and self.role_definition(role) is not None

    def evaluate_structural_compatibility(self, *, role: str, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        definition = self.role_definition(role)
        if definition is None:
            return {"status": "incompatible", "reasons": ["role_not_defined_by_provider"]}
        reasons: list[str] = []
        family_evidence = str(evidence.get("family") or evidence.get("architecture") or "").strip()
        if family_evidence:
            canonical = canonicalize_family(family_evidence)
            if canonical and canonical != self.family_id:
                return {"status": "incompatible", "reasons": ["family_mismatch"]}
            if not canonical:
                reasons.append("family_evidence_unrecognized")
        constraints = dict(definition.structural_constraints)
        for key in ("denoiser_type", "latent_channels", "denoising_domain"):
            expected = constraints.get(key)
            actual = evidence.get(key)
            if expected is not None and actual is not None and str(actual) != str(expected):
                return {"status": "incompatible", "reasons": [f"constraint_mismatch:{key}"]}
            if expected is not None and actual is None:
                reasons.append(f"constraint_unobserved:{key}")
        return {
            "status": "structurally_eligible" if not reasons else "unknown",
            "reasons": reasons,
            "constraints": constraints,
        }

    def component_extraction_rules(self, role: str) -> Mapping[str, Any]:
        return dict(self.extraction_rules.get(str(role), {}))

    def supports_digital_hydration(self, role: str) -> bool:
        return str(role) in set(self.digital_hydration_roles)

    def supports_runtime_composition(self) -> bool:
        return bool(self.runtime_composition)

    def placement_capabilities(self, role: str) -> tuple[str, ...]:
        return tuple(self.placement_by_role.get(str(role), ("runtime_managed",)))

    def blueprint_capabilities(self) -> Mapping[str, Any]:
        return dict(self.blueprint_support)

    def supports_analysis_layout(self, role: str) -> bool:
        return str(role) in self.analysis_layouts

    def analysis_layout_version(self, role: str) -> int | None:
        layout = self.analysis_layouts.get(str(role))
        return int(layout.layout_version) if layout is not None else None

    def describe_analysis_layout(self, role: str) -> ComponentAnalysisLayout | None:
        return self.analysis_layouts.get(str(role))

    def resolve_analysis_nodes(self, role: str, tensor_names: Iterable[str]) -> Mapping[str, tuple[str, ...]]:
        layout = self.describe_analysis_layout(role)
        if layout is None:
            return {}
        return layout.resolve_tensor_names(tuple(str(item) for item in tensor_names))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_CONTRACT_VERSION,
            "family_id": self.family_id,
            "display_label": self.display_label,
            "architecture_aliases": list(self.architecture_aliases),
            "required_roles": [item.to_dict() for item in self.required_roles],
            "optional_roles": [item.to_dict() for item in self.optional_roles],
            "base_weight_role": self.base_weight_role,
            "version": self.version,
            "digital_hydration_support": {role: self.supports_digital_hydration(role) for role in [item.canonical_role_id for item in self.role_definitions()]},
            "runtime_composition_support": self.runtime_composition,
            "runtime_composition_validation_state": self.runtime_composition_validation_state,
            "digital_hydration_validation": {
                role: self.digital_hydration_validation_by_role.get(role, "implementation_present_unvalidated")
                for role in [item.canonical_role_id for item in self.role_definitions()]
            },
            "placement_capabilities": {role: list(self.placement_capabilities(role)) for role in [item.canonical_role_id for item in self.role_definitions()]},
            "analysis_layouts": {
                role: layout.to_dict()
                for role, layout in sorted(self.analysis_layouts.items())
            },
            "blueprint_capabilities": dict(self.blueprint_support),
        }


class ArchitectureFamilyProviderRegistry:
    def __init__(self, providers: Iterable[ArchitectureFamilyProvider]) -> None:
        ordered = tuple(providers)
        by_id = {provider.family_id: provider for provider in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("Architecture family provider IDs must be unique.")
        self._providers = ordered
        self._by_id = by_id

    def providers(self) -> tuple[ArchitectureFamilyProvider, ...]:
        return self._providers

    def get(self, family: str) -> ArchitectureFamilyProvider | None:
        canonical = self.canonicalize(family)
        return self._by_id.get(canonical)

    def require(self, family: str) -> ArchitectureFamilyProvider:
        provider = self.get(family)
        if provider is None:
            raise ValueError(f"No architecture family provider is registered for {family!r}.")
        return provider

    def canonicalize(self, value: Any) -> str:
        return canonicalize_family(value, providers=self._providers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_CONTRACT_VERSION,
            "providers": [provider.to_dict() for provider in self._providers],
        }


def _aliases(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip().lower().replace("_", "-") for value in values if str(value).strip()))


def _base_blueprint_capabilities() -> dict[str, Any]:
    return {
        "reference_serialization": True,
        "physical_reconstruction": "unvalidated",
        "exact_byte_reconstruction": "unvalidated",
    }


def _unet_analysis_layout(family_id: str, *, include_label_emb: bool = False) -> ComponentAnalysisLayout:
    nodes = [
        ComponentAnalysisNode("time_embed", "tensor_group", 10, tensor_prefixes=("time_embed.",)),
    ]
    if include_label_emb:
        # Promoted from ML-F01 real-library evidence: the same four label_emb tensors
        # were present in every validated SDXL base/refiner checkpoint.
        nodes.append(ComponentAnalysisNode(
            "label_emb",
            "tensor_group",
            15,
            tensor_prefixes=("label_emb.",),
            metadata={
                "promotion_basis": "ml_f01_real_model_structural_evidence",
                "semantic_interpretation": "not_asserted_by_model_lab",
            },
        ))
    nodes.extend((
        ComponentAnalysisNode("input_blocks", "block", 20, tensor_prefixes=("input_blocks.",)),
        ComponentAnalysisNode("middle_block", "block", 30, tensor_prefixes=("middle_block.",)),
        ComponentAnalysisNode("output_blocks", "block", 40, tensor_prefixes=("output_blocks.",)),
        ComponentAnalysisNode("out", "tensor_group", 50, tensor_prefixes=("out.",)),
    ))
    return ComponentAnalysisLayout(
        provider_id=family_id,
        family_id=family_id,
        component_role="unet",
        layout_version=2 if include_label_emb else 1,
        nodes=tuple(nodes),
        grouping_rules={
            "tensor_namespace": "component_relative",
            "unmatched_tensor_policy": "evidence_resolver",
            "auto_resolution": {
                "enabled": True,
                "strategy": "top_level_namespace_structural_variant",
                "allow_singleton_variant": True,
            },
        },
    )


def _sd3_transformer_analysis_layout() -> ComponentAnalysisLayout:
    return ComponentAnalysisLayout(
        provider_id="sd3.x",
        family_id="sd3.x",
        component_role="transformer",
        layout_version=2,
        nodes=(
            ComponentAnalysisNode("x_embedder", "tensor_group", 10, tensor_prefixes=("x_embedder.",)),
            # Promoted from ML-F01 real-library evidence: one pos_embed tensor was
            # consistently observed in every validated SD3/SD3.5 checkpoint.
            ComponentAnalysisNode(
                "pos_embed",
                "tensor_group",
                15,
                tensor_names=("pos_embed",),
                metadata={
                    "promotion_basis": "ml_f01_real_model_structural_evidence",
                    "semantic_interpretation": "not_asserted_by_model_lab",
                },
            ),
            ComponentAnalysisNode("context_embedder", "tensor_group", 20, tensor_prefixes=("context_embedder.",)),
            ComponentAnalysisNode("time_embedder", "tensor_group", 30, tensor_prefixes=("t_embedder.",)),
            ComponentAnalysisNode("pooled_text_embedder", "tensor_group", 40, tensor_prefixes=("y_embedder.",)),
            ComponentAnalysisNode("joint_blocks", "block", 50, tensor_prefixes=("joint_blocks.",)),
            ComponentAnalysisNode("final_layer", "tensor_group", 60, tensor_prefixes=("final_layer.",)),
        ),
        grouping_rules={
            "tensor_namespace": "component_relative",
            "unmatched_tensor_policy": "evidence_resolver",
            "auto_resolution": {
                "enabled": True,
                "strategy": "top_level_namespace_structural_variant",
                "allow_singleton_variant": True,
            },
        },
    )


SD1_PROVIDER = StaticArchitectureFamilyProvider(
    family_id="sd1.x",
    display_label="Stable Diffusion 1.x",
    architecture_aliases=_aliases("sd1", "sd1.x", "sd1-x", "stable-diffusion-1", "stable-diffusion-1.x", "stable diffusion 1"),
    required_roles=(
        _required_role("unet", "Model Weights / UNet", base=True, constraints={"denoiser_type": "unet"}),
        _required_role("vae", "VAE"),
        _required_role("text_encoder", "Text Encoder", constraints={"conditioning_family": "clip"}),
    ),
    base_weight_role="unet",
    extraction_rules={role: {"state_dict_mapper_role": role, "include_extras": False} for role in ("unet", "vae", "text_encoder")},
    digital_hydration_roles=("unet", "vae", "text_encoder"),
    runtime_composition_validation_state="implementation_present_needs_real_model_validation",
    blueprint_support=_base_blueprint_capabilities(),
    analysis_layouts={"unet": _unet_analysis_layout("sd1.x")},
)

SD2_PROVIDER = StaticArchitectureFamilyProvider(
    family_id="sd2.x",
    display_label="Stable Diffusion 2.x",
    architecture_aliases=_aliases("sd2", "sd2.0", "sd2.1", "sd2.x", "sd2-x", "stable-diffusion-2", "stable-diffusion-2.x", "stable diffusion 2"),
    required_roles=(
        _required_role("unet", "Model Weights / UNet", base=True, constraints={"denoiser_type": "unet"}),
        _required_role("vae", "VAE"),
        _required_role("text_encoder", "Text Encoder", constraints={"conditioning_family": "openclip_or_clip_by_profile"}),
    ),
    base_weight_role="unet",
    extraction_rules={role: {"state_dict_mapper_role": role, "include_extras": False} for role in ("unet", "vae", "text_encoder")},
    digital_hydration_roles=("unet", "vae", "text_encoder"),
    runtime_composition_validation_state="implementation_present_needs_real_model_validation",
    blueprint_support=_base_blueprint_capabilities(),
    analysis_layouts={"unet": _unet_analysis_layout("sd2.x")},
)

SDXL_PROVIDER = StaticArchitectureFamilyProvider(
    family_id="sdxl",
    display_label="Stable Diffusion XL",
    architecture_aliases=_aliases("sdxl", "sdxl-base", "stable-diffusion-xl", "stable-diffusion-xl-base", "stable diffusion xl"),
    required_roles=(
        _required_role("unet", "Model Weights / UNet", base=True, constraints={"denoiser_type": "unet"}),
        _required_role("vae", "VAE"),
        _required_role("text_encoder", "CLIP-L / Text Encoder 1"),
        _required_role("text_encoder_2", "OpenCLIP / Text Encoder 2"),
    ),
    base_weight_role="unet",
    extraction_rules={role: {"state_dict_mapper_role": role, "include_extras": False} for role in ("unet", "vae", "text_encoder", "text_encoder_2")},
    digital_hydration_roles=("unet", "vae", "text_encoder", "text_encoder_2"),
    runtime_composition_validation_state="implementation_present_needs_real_model_validation",
    blueprint_support=_base_blueprint_capabilities(),
    analysis_layouts={"unet": _unet_analysis_layout("sdxl", include_label_emb=True)},
)

SD3_PROVIDER = StaticArchitectureFamilyProvider(
    family_id="sd3.x",
    display_label="Stable Diffusion 3 / 3.5",
    architecture_aliases=_aliases("sd3", "sd3.0", "sd3.5", "sd3.x", "sd3-x", "stable-diffusion-3", "stable-diffusion-3.x", "stable diffusion 3"),
    required_roles=(
        _required_role("transformer", "Model Weights / Transformer", base=True, constraints={"denoiser_type": "transformer", "denoising_domain": "flow_match"}),
        _required_role("vae", "VAE", constraints={"latent_channels": 16}),
        _required_role("text_encoder", "CLIP-L"),
        _required_role("text_encoder_2", "CLIP-G"),
    ),
    optional_roles=(
        _optional_role("text_encoder_3", "T5 / T5XXL", constraints={"activation": "explicit_only"}),
    ),
    base_weight_role="transformer",
    extraction_rules={role: {"state_dict_mapper_role": role, "include_extras": False} for role in ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3")},
    digital_hydration_roles=("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"),
    runtime_composition_validation_state="sd3_production_path_validated_t5_cuda_requires_hardware_evidence",
    digital_hydration_validation_by_role={
        "transformer": "validated_by_sd3_phases",
        "vae": "validated_by_sd3_phases",
        "text_encoder": "validated_by_sd3_phases",
        "text_encoder_2": "validated_by_sd3_phases",
        "text_encoder_3": "cpu_path_validated_cuda_option_requires_hardware_evidence",
    },
    placement_by_role={"text_encoder_3": ("cpu", "cuda", "auto")},
    blueprint_support=_base_blueprint_capabilities(),
    analysis_layouts={"transformer": _sd3_transformer_analysis_layout()},
)

DEFAULT_FAMILY_PROVIDER_REGISTRY = ArchitectureFamilyProviderRegistry((
    SD1_PROVIDER,
    SD2_PROVIDER,
    SDXL_PROVIDER,
    SD3_PROVIDER,
))


def canonicalize_family(value: Any, *, providers: Iterable[ArchitectureFamilyProvider] | None = None) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    if not token:
        return ""
    candidates = tuple(providers) if providers is not None else DEFAULT_FAMILY_PROVIDER_REGISTRY.providers()
    for provider in candidates:
        aliases = set(provider.architecture_aliases) | {provider.family_id.lower().replace("_", "-")}
        if token in aliases:
            return provider.family_id
    # Preserve the broad aliases historically accepted by Step 0 without making
    # filenames or unknown architectures authoritative.
    if "stable diffusion 1" in token or token.startswith("sd1"):
        return "sd1.x" if any(p.family_id == "sd1.x" for p in candidates) else ""
    if "stable diffusion 2" in token or token.startswith("sd2"):
        return "sd2.x" if any(p.family_id == "sd2.x" for p in candidates) else ""
    if "sdxl" in token or "stable diffusion xl" in token:
        return "sdxl" if any(p.family_id == "sdxl" for p in candidates) else ""
    if "sd3" in token or "stable diffusion 3" in token:
        return "sd3.x" if any(p.family_id == "sd3.x" for p in candidates) else ""
    return ""


__all__ = [
    "ArchitectureFamilyProviderRegistry",
    "DEFAULT_FAMILY_PROVIDER_REGISTRY",
    "PROVIDER_CONTRACT_VERSION",
    "SD1_PROVIDER",
    "SD2_PROVIDER",
    "SDXL_PROVIDER",
    "SD3_PROVIDER",
    "StaticArchitectureFamilyProvider",
    "canonicalize_family",
]
