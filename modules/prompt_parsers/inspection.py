from __future__ import annotations

from typing import Any, Mapping

from modules.prompt_parsers.canonical import canonical_ir_from_structure
from modules.prompt_parsers.compiler import ConditioningPlan, compile_conditioning_plan
from modules.prompt_parsers.semantic_replay import semantic_digest, semantic_structure_digest

PROMPT_SEMANTIC_INSPECTION_CONTRACT_VERSION = "image-gen-prompt-semantic-inspection-v1"


def _warning(category: str, message: str, *, severity: str = "warning") -> dict[str, str]:
    return {"category": category, "severity": severity, "message": str(message)}


def _warning_category(message: str) -> str:
    text = str(message or "").lower()
    if "compatibility alias" in text or "!!!" in text:
        return "compatibility_alias"
    if "legacy" in text and ("inference" in text or "inferred" in text):
        return "legacy_numeric_inference"
    if "combination" in text and ("limit" in text or "exceed" in text):
        return "combination_limit_fallback"
    if "unsupported" in text or "model_family_safe_flatten" in text:
        return "unsupported_runtime_feature"
    if "fallback" in text or "flatten" in text:
        return "semantic_fallback"
    return "parser_warning"


def _group_payload(plan: ConditioningPlan) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in plan.group_diagnostics:
        data = item.to_dict()
        members = []
        for index, source in enumerate(data.get("source_members") or []):
            raw = list(data.get("raw_member_weights") or [])
            normalized = list(data.get("normalized_local_weights") or [])
            explicit = list(data.get("explicit_weight_flags") or [])
            members.append({
                "index": index,
                "source": str(source),
                "raw_weight": float(raw[index]) if index < len(raw) else 1.0,
                "normalized_weight": float(normalized[index]) if index < len(normalized) else None,
                "explicit_weight": bool(explicit[index]) if index < len(explicit) else False,
            })
        output.append({
            "operation_id": data.get("operation_id"),
            "group_id": data.get("group_id"),
            "source": data.get("source_text"),
            "member_count": data.get("member_count"),
            "members": members,
            "combination_count": data.get("combination_count", 0),
            "fallback_used": bool(data.get("fallback_used", False)),
            "fallback_reason": str(data.get("fallback_reason") or ""),
        })
    return output


def _relationship_payload(plan: ConditioningPlan) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in plan.relationship_diagnostics:
        data = item.to_dict()
        output.append({
            "operation_id": data.get("operation_id"),
            "syntax_origin": data.get("syntax_origin"),
            "owner": data.get("owner"),
            "parent_scope": data.get("parent_scope"),
            "owner_composition": data.get("owner_composition"),
            "relations": list(data.get("source_items") or []),
            "raw_weights": list(data.get("raw_item_weights") or []),
            "normalized_weights": list(data.get("normalized_local_weights") or []),
            "activity_windows": list(data.get("activity_windows") or []),
            "terminators_consumed": list(data.get("terminators_consumed") or []),
            "top_terminator_consumed": data.get("top_terminator_consumed"),
        })
    return output


def _schedule_payload(plan: ConditioningPlan) -> list[dict[str, Any]]:
    schedules: list[dict[str, Any]] = []
    for index, branch in enumerate(plan.branches):
        if not branch.temporal_source and branch.active_until_step is None:
            continue
        schedules.append({
            "branch_index": index,
            "source": branch.temporal_source or branch.text,
            "compiled": bool(branch.temporal_compiled),
            "active_until_step": branch.active_until_step,
            "hold_after_step": bool(branch.hold_after_step),
            "encoder_text": branch.text,
        })
    return schedules



def _effective_static_branch_weights(plan: ConditioningPlan) -> list[float | None]:
    """Return final static contributions using runtime group -> sequence -> outer scope order.

    Temporal/activity-aware plans intentionally return ``None`` because their final
    contribution varies by denoising step and should not be represented as one number.
    """
    if any(
        branch.temporal_source or branch.temporal_compiled or branch.active_until_step is not None
        for branch in plan.branches
    ):
        return [None for _ in plan.branches]

    entries: list[dict[str, Any]] = []
    for index, branch in enumerate(plan.branches):
        entries.append({
            "contrib": {index: 1.0},
            "outer": float(branch.weight),
            "group_id": branch.group_operation_id,
            "group_weight": float(branch.group_local_weight),
            "sequence_id": branch.sequence_operation_id,
            "sequence_weight": float(branch.sequence_local_weight),
        })

    def combine(members: list[dict[str, Any]], weight_key: str) -> dict[int, float]:
        weights = [float(item[weight_key]) for item in members]
        total = sum(weights)
        if abs(total) <= 1e-12:
            return {}
        result: dict[int, float] = {}
        for item, weight in zip(members, weights):
            normalized = weight / total
            for index, contribution in item["contrib"].items():
                result[index] = result.get(index, 0.0) + float(contribution) * normalized
        return result

    group_collapsed: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for entry in entries:
        group_id = entry["group_id"]
        if not group_id:
            group_collapsed.append(entry)
            continue
        if group_id in consumed:
            continue
        consumed.add(group_id)
        members = [item for item in entries if item["group_id"] == group_id]
        first = members[0]
        group_collapsed.append({
            "contrib": combine(members, "group_weight"),
            "outer": first["outer"],
            "group_id": None,
            "group_weight": 1.0,
            "sequence_id": first["sequence_id"],
            "sequence_weight": first["sequence_weight"],
        })

    sequence_collapsed: list[dict[str, Any]] = []
    consumed = set()
    for entry in group_collapsed:
        sequence_id = entry["sequence_id"]
        if not sequence_id:
            sequence_collapsed.append(entry)
            continue
        if sequence_id in consumed:
            continue
        consumed.add(sequence_id)
        members = [item for item in group_collapsed if item["sequence_id"] == sequence_id]
        first = members[0]
        sequence_collapsed.append({
            "contrib": combine(members, "sequence_weight"),
            "outer": first["outer"],
            "group_id": None,
            "group_weight": 1.0,
            "sequence_id": None,
            "sequence_weight": 1.0,
        })

    outer_total = sum(float(item["outer"]) for item in sequence_collapsed)
    final: dict[int, float] = {}
    if abs(outer_total) > 1e-12:
        for entry in sequence_collapsed:
            share = float(entry["outer"]) / outer_total
            for index, contribution in entry["contrib"].items():
                final[index] = final.get(index, 0.0) + float(contribution) * share
    return [final.get(index, 0.0) for index in range(len(plan.branches))]

def build_semantic_inspection(
    canonical_structure: Mapping[str, Any],
    *,
    conditioning_plan: ConditioningPlan | None = None,
    parser_warnings: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    ir = canonical_ir_from_structure(canonical_structure)
    plan = conditioning_plan or compile_conditioning_plan(ir)
    warnings: list[dict[str, str]] = []
    for message in [*list(parser_warnings), *list(plan.warnings)]:
        warnings.append(_warning(_warning_category(message), message))
    for fallback in plan.fallbacks:
        warnings.append(_warning(_warning_category(fallback), fallback))
    for numeric in plan.numeric_semantics:
        if getattr(numeric, "inferred", False):
            warnings.append(_warning(
                "legacy_numeric_inference",
                getattr(numeric, "message", "") or f"Legacy numeric inference: {numeric.value}",
            ))
        if getattr(numeric, "valid", True) is False:
            warnings.append(_warning(
                "syntax_error",
                getattr(numeric, "message", "") or "Invalid numeric prompt syntax.",
                severity="error",
            ))

    effective_weights = _effective_static_branch_weights(plan)
    branches = []
    for index, item in enumerate(plan.branches):
        branches.append({
            "index": index,
            "encoder_text": item.text,
            "outer_weight": float(item.weight),
            "group_local_weight": float(item.group_local_weight),
            "sequence_local_weight": float(item.sequence_local_weight),
            "effective_final_weight": effective_weights[index],
            "effective_weight_dynamic": effective_weights[index] is None,
            "active_until_step": item.active_until_step,
            "semantic_role": item.semantic_role,
            "source_node_type": item.source_node_type,
            "parent_scope": item.parent_scope,
            "owner_composition": item.owner_composition,
        })

    payload = {
        "contract_version": PROMPT_SEMANTIC_INSPECTION_CONTRACT_VERSION,
        "canonical_contract_version": str(canonical_structure.get("contract") or ""),
        "parser_namespace": str(canonical_structure.get("parser_namespace") or "legacy"),
        "semantic_ir_contract": ir.contract,
        "conditioning_plan_contract": plan.contract,
        "semantic_digest": semantic_digest(ir, plan, degradation=list(plan.fallbacks)),
        "structure_digest": semantic_structure_digest(ir),
        "root_type": str((ir.to_dict().get("root") or {}).get("type") or "text"),
        "groups": _group_payload(plan),
        "relationships": _relationship_payload(plan),
        "schedules": _schedule_payload(plan),
        "fallbacks": list(plan.fallbacks),
        "warnings": warnings,
        "encoder_text_preview": branches,
        "lowering_required": bool(plan.lowering_required),
    }
    return payload
