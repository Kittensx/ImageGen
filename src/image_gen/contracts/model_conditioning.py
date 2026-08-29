from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SEMANTIC_CONDITIONING_CAPABILITY_CONTRACT_VERSION = (
    "image-gen-semantic-conditioning-capabilities-v1"
)


@dataclass(frozen=True)
class SemanticConditioningCapabilities:
    """Runtime declaration for safely composing prompt-conditioning outputs.

    The parser/compiler remains model-neutral.  A conditioning runtime declares
    which semantic channels can be combined with the same hierarchical weights.
    PPSR-07 uses this declaration to keep grouping/sequence behavior aligned
    across SD1/2 tensor conditioning, SDXL token+pooled conditioning, and SD3
    CLIP/T5 conditioning.
    """

    architecture: str
    runtime_name: str
    output_kind: str = "tensor"
    composable_fields: tuple[str, ...] = ("cross_attention",)
    required_fields: tuple[str, ...] = ("cross_attention",)
    supports_group_conditioning: bool = True
    supports_sequence_conditioning: bool = True
    supports_temporal_conditioning: bool = True
    supports_pooled_conditioning: bool = False
    unsupported_structured_fields: tuple[str, ...] = ()
    t5_policy: str = "not_applicable"
    safe_flatten_supported: bool = True
    contract: str = SEMANTIC_CONDITIONING_CAPABILITY_CONTRACT_VERSION

    def supports_semantic_composition(self) -> bool:
        return bool(
            self.supports_group_conditioning
            and self.supports_sequence_conditioning
            and self.supports_temporal_conditioning
        )

    def can_compose_field(self, field_name: str) -> bool:
        return str(field_name) in set(self.composable_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "architecture": self.architecture,
            "runtime_name": self.runtime_name,
            "output_kind": self.output_kind,
            "composable_fields": list(self.composable_fields),
            "required_fields": list(self.required_fields),
            "supports_group_conditioning": bool(self.supports_group_conditioning),
            "supports_sequence_conditioning": bool(self.supports_sequence_conditioning),
            "supports_temporal_conditioning": bool(self.supports_temporal_conditioning),
            "supports_pooled_conditioning": bool(self.supports_pooled_conditioning),
            "unsupported_structured_fields": list(self.unsupported_structured_fields),
            "t5_policy": self.t5_policy,
            "safe_flatten_supported": bool(self.safe_flatten_supported),
        }


def semantic_conditioning_capabilities_for_runtime(
    runtime: Any,
) -> SemanticConditioningCapabilities | None:
    """Return a declared capability contract without guessing model family.

    Unknown/custom conditioning objects retain historical behavior and return
    ``None``.  IMAGE_GEN-owned runtimes declare this explicitly.
    """

    provider = getattr(runtime, "semantic_conditioning_capabilities", None)
    if not callable(provider):
        return None
    value = provider()
    if isinstance(value, SemanticConditioningCapabilities):
        return value
    if isinstance(value, Mapping):
        return SemanticConditioningCapabilities(**dict(value))
    raise TypeError(
        "semantic_conditioning_capabilities() must return "
        "SemanticConditioningCapabilities or a mapping."
    )


@dataclass(frozen=True)
class BranchModelConditioningKwargs:
    """Branch-aware kwargs supplied to the denoising model boundary.

    ``conditional`` and ``unconditional`` contain branch-specific keyword
    arguments. ``shared`` is merged into either branch before the UNet call.
    The denoising system owns branch selection for pipeline-guided CFG, while
    raw-model samplers may explicitly select a branch before invoking the model.
    """

    conditional: Mapping[str, Any] = field(default_factory=dict)
    unconditional: Mapping[str, Any] = field(default_factory=dict)
    shared: Mapping[str, Any] = field(default_factory=dict)

    def for_branch(self, branch: str) -> dict[str, Any]:
        normalized = str(branch or "").strip().lower().replace("-", "_")
        aliases = {
            "cond": "conditional",
            "positive": "conditional",
            "uncond": "unconditional",
            "negative": "unconditional",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"conditional", "unconditional"}:
            raise ValueError(
                "Model-conditioning branch must be 'conditional' or 'unconditional'."
            )
        branch_values = self.conditional if normalized == "conditional" else self.unconditional
        merged = dict(self.shared or {})
        merged.update(dict(branch_values or {}))
        return merged


ModelConditioningKwargs = dict[str, Any] | BranchModelConditioningKwargs | None


def select_model_conditioning_branch(
    value: ModelConditioningKwargs,
    branch: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BranchModelConditioningKwargs):
        return value.for_branch(branch)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        "Model conditioning kwargs must be a dict, BranchModelConditioningKwargs, or None."
    )
