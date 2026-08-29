from __future__ import annotations

"""PPSR-07 model-family semantic conditioning adaptation.

PromptIR/ConditioningPlan stay model neutral.  This layer consumes an explicit
runtime capability contract and either preserves semantic composition or
performs a deterministic punctuation-safe flatten with diagnostics.
"""

from dataclasses import dataclass, replace
from typing import Any

from image_gen.contracts.model_conditioning import (
    SemanticConditioningCapabilities,
    semantic_conditioning_capabilities_for_runtime,
)
from modules.prompt_parsers.compiler import (
    ConditioningBranch,
    ConditioningPlan,
    render_ir_node,
)
from modules.prompt_parsers.ir import PromptIR
from modules.prompt_parsers.temporal_semantics import compile_temporal_text, contains_temporal_syntax

MODEL_FAMILY_SEMANTIC_CONTRACT_VERSION = "image-gen-model-family-semantic-conditioning-v1"


@dataclass(frozen=True)
class ModelFamilySemanticCompilation:
    plan: ConditioningPlan
    capabilities: SemanticConditioningCapabilities | None
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
    fallback_text: str = ""
    contract: str = MODEL_FAMILY_SEMANTIC_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "degraded": bool(self.degraded),
            "degradation_reasons": list(self.degradation_reasons),
            "fallback_text": self.fallback_text,
            "conditioning_plan_contract": self.plan.contract,
            "conditioning_plan_fallbacks": list(self.plan.fallbacks),
        }


def _plan_requirements(plan: ConditioningPlan) -> dict[str, bool]:
    return {
        "group": any(branch.group_operation_id for branch in plan.branches),
        "sequence": any(branch.sequence_operation_id for branch in plan.branches),
        "temporal": any(branch.temporal_compiled for branch in plan.branches),
    }


def _safe_flat_text(prompt_ir: PromptIR, *, total_steps: int) -> str:
    text = render_ir_node(prompt_ir.root).strip()
    if contains_temporal_syntax(text):
        temporal = compile_temporal_text(text, max(1, int(total_steps)))
        # A runtime that cannot execute temporal semantics receives one stable
        # punctuation-free representation containing every distinct state.
        unique: list[str] = []
        for value in temporal.per_step_text:
            cleaned = str(value).strip(" ,")
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        text = ", ".join(unique)
    return text.strip(" ,")


def compile_plan_for_runtime(
    prompt_ir: PromptIR,
    plan: ConditioningPlan,
    runtime: Any,
    *,
    total_steps: int,
) -> ModelFamilySemanticCompilation:
    capabilities = semantic_conditioning_capabilities_for_runtime(runtime)
    if capabilities is None:
        # Backward-compatible custom/runtime path.  We do not infer capability
        # from filenames, tensor widths, or class names.
        return ModelFamilySemanticCompilation(plan=plan, capabilities=None)

    requirements = _plan_requirements(plan)
    reasons: list[str] = []
    if requirements["group"] and not capabilities.supports_group_conditioning:
        reasons.append("group_conditioning_unsupported")
    if requirements["sequence"] and not capabilities.supports_sequence_conditioning:
        reasons.append("sequence_conditioning_unsupported")
    if requirements["temporal"] and not capabilities.supports_temporal_conditioning:
        reasons.append("temporal_conditioning_unsupported")

    if not reasons:
        return ModelFamilySemanticCompilation(plan=plan, capabilities=capabilities)

    if not capabilities.safe_flatten_supported:
        raise RuntimeError(
            f"Conditioning runtime {capabilities.runtime_name!r} cannot preserve "
            f"semantic composition and does not permit safe flatten fallback: {reasons}"
        )

    fallback_text = _safe_flat_text(prompt_ir, total_steps=total_steps)
    warning = (
        f"Semantic conditioning degraded for {capabilities.architecture}: "
        + ", ".join(reasons)
        + "; structured prompt safely flattened before encoder invocation."
    )
    adapted = replace(
        plan,
        branches=(
            ConditioningBranch(
                text=fallback_text,
                semantic_role="model_family_safe_flatten",
                source_node_type="runtime_fallback",
            ),
        ),
        lowering_required=True,
        fallbacks=tuple(dict.fromkeys((*plan.fallbacks, "model_family_safe_flatten"))),
        warnings=tuple(dict.fromkeys((*plan.warnings, warning))),
    )
    return ModelFamilySemanticCompilation(
        plan=adapted,
        capabilities=capabilities,
        degraded=True,
        degradation_reasons=tuple(reasons),
        fallback_text=fallback_text,
    )
