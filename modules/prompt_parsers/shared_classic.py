from __future__ import annotations

"""Shared runtime for the Classic Prompt Architect semantic subset.

Parser21 and SuperHybrid still own their extension grammars. When a prompt uses
only IMAGE_GEN's locked Classic structural syntax (groups / closed relations /
owner sequences), all three parser selections route through the same PromptIR
and ConditioningPlan compiler so those shared operators cannot drift apart.
"""

from dataclasses import dataclass
import re
from types import SimpleNamespace
from typing import Any

from modules.parser.prompt_parser_class import PromptParserClass, SdConditioning
from modules.prompt_parsers.canonical import canonicalize_prompt
from modules.prompt_parsers.compiler import ConditioningPlan, compile_conditioning_plan
from modules.prompt_parsers.temporal_semantics import contains_temporal_syntax
from modules.prompt_parsers.contracts import PROMPT_PARSER_CONTRACT_VERSION
from modules.prompt_parsers.semantic_replay import (
    PromptSemanticReplayError,
    build_ppsr_semantic_record,
    resolve_request_prompt_ir,
    validate_replayed_ppsr_result,
)
from modules.prompt_parsers.model_family_compiler import compile_plan_for_runtime
from modules.prompt_parsers.ir import (
    Group,
    IRNode,
    OwnerSequence,
    Prompt,
    PromptIR,
    Relation,
    Sequence,
    Weighted,
    parse_prompt_ir,
)

_EXTENSION_RE = re.compile(
    r"(?<!\\)\b(?:BIND(?:2|3)?|CHUNK|BLEND|MORPH|ASSEMBLE|POOL|COMPOUND|DIFF|REGION|TONEG)\b|"
    r"(?<!\\)<param\s*\[|(?<!\\)__[^\s]+__|(?<!\\)\$\{|(?<!\\)\{\{",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SharedClassicExecution:
    semantic_ir: PromptIR
    conditioning_plan: ConditioningPlan
    parsed: Any
    canonical_prompt: str
    canonical_structure: dict[str, Any]
    warnings: tuple[str, ...]
    model_family_semantics: dict[str, Any]
    ppsr_semantic_record: dict[str, Any]
    replay_diagnostics: dict[str, Any]


def _contains_shared_node(node: IRNode) -> bool:
    if isinstance(node, (Group, Relation, OwnerSequence)):
        return True
    if isinstance(node, Sequence):
        return str(node.syntax_origin or "") in {"classic_closed_sequence", "classic_owner_sequence"}
    if isinstance(node, Prompt):
        return any(_contains_shared_node(item) for item in node.parts)
    if isinstance(node, Weighted):
        return _contains_shared_node(node.node)
    # Conjunction is intentionally duck-walked to avoid a circular import list.
    branches = getattr(node, "branches", None)
    if branches is not None:
        return any(_contains_shared_node(item.node) for item in branches)
    return False


def shared_classic_ir(source: str, *, parser_namespace: str) -> PromptIR | None:
    text = str(source or "")
    if _EXTENSION_RE.search(text):
        return None
    ir = parse_prompt_ir(text, parser_namespace=parser_namespace)
    return ir if (_contains_shared_node(ir.root) or contains_temporal_syntax(text)) else None


def uses_shared_classic_semantics(source: str, *, parser_namespace: str = "legacy") -> bool:
    return shared_classic_ir(source, parser_namespace=parser_namespace) is not None


def validate_shared_classic(
    source: str,
    *,
    parser_namespace: str,
    prompt_role: str,
    steps: int,
    hires_steps: int | None = None,
) -> dict[str, Any] | None:
    ir = shared_classic_ir(source, parser_namespace=parser_namespace)
    if ir is None:
        return None
    plan = compile_conditioning_plan(ir)

    # No encoder is called here. PromptParserClass only materializes branch text
    # and typed metadata for the parser-only BAT validation route.
    from modules.shared_state import SharedState

    state = SharedState()
    state.p.steps = int(steps)
    state.p.width = 64
    state.p.height = 64
    state.conditioning.use_old_scheduling = False
    parser = PromptParserClass(
        shared_state=state,
        prompts=SdConditioning(
            [source],
            is_negative_prompt=prompt_role in {"negative", "hires_negative"},
            width=64,
            height=64,
        ),
        steps=int(steps),
        model=SimpleNamespace(),
        hires_steps=hires_steps,
        conditioning_plans=[plan],
    )
    branches, flat, _ = parser.get_multicond_prompt_list()
    return {
        "valid": True,
        "schedule_count": 0,
        "branch_count": sum(len(item) for item in branches or []),
        "flat_prompt_count": len(flat or []),
        "warnings": list(ir.warnings) + list(plan.warnings),
        "semantic_ir": ir.to_dict(),
        "conditioning_plan": plan.to_dict(),
        "shared_classic_semantics": True,
        "shared_classic_contract": plan.contract,
    }


def execute_shared_classic(
    request: Any,
    *,
    parser_namespace: str,
    parser_version: str = "",
    source: str | None = None,
) -> SharedClassicExecution | None:
    prompt_source = str(request.raw_prompt if source is None else source)
    recorded = dict(getattr(request, "recorded_semantic_replay", {}) or {})
    replay_mode = str(getattr(request, "semantic_replay_mode", "reconstruct") or "reconstruct").strip().lower()
    if replay_mode == "recorded_exact" and recorded.get("shared_classic_semantics") is True:
        try:
            ir, replay_diagnostics = resolve_request_prompt_ir(
                request,
                source=prompt_source,
                parser_namespace=parser_namespace,
                parser_contract_version=PROMPT_PARSER_CONTRACT_VERSION,
            )
        except PromptSemanticReplayError:
            raise
        if not (_contains_shared_node(ir.root) or contains_temporal_syntax(prompt_source)):
            return None
    else:
        ir = shared_classic_ir(prompt_source, parser_namespace=parser_namespace)
        replay_diagnostics = {"mode": replay_mode, "source": "reconstruct", "migration_path": "none"}
        if ir is None:
            return None
    plan = compile_conditioning_plan(ir)
    model_family = compile_plan_for_runtime(
        ir, plan, request.model_context, total_steps=int(request.hires_steps or request.steps)
    )
    plan = model_family.plan
    replay_record = None
    if replay_mode == "recorded_exact":
        replay_record = validate_replayed_ppsr_result(
            recorded,
            semantic_ir=ir,
            conditioning_plan=plan,
        )
    prompts = SdConditioning(
        [prompt_source],
        is_negative_prompt=request.prompt_role in {"negative", "hires_negative"},
        width=request.width,
        height=request.height,
    )
    parsed = PromptParserClass(
        shared_state=request.shared_state,
        prompts=prompts,
        steps=request.steps,
        model=request.model_context,
        hires_steps=request.hires_steps,
        conditioning_plans=[plan],
    )()
    canonical, structure, warnings = canonicalize_prompt(
        prompt_source,
        parser_id=parser_namespace,
        semantic_ir=ir,
    )
    ppsr_record = build_ppsr_semantic_record(
        parser_id=parser_namespace,
        parser_version=str(parser_version or ""),
        parser_contract_version=PROMPT_PARSER_CONTRACT_VERSION,
        prompt_role=str(request.prompt_role or "positive"),
        raw_prompt=prompt_source,
        canonical_structure=structure,
        semantic_ir=ir,
        conditioning_plan=plan,
        parser_seed=getattr(request, "seed", None),
        replay_source="recorded_exact" if replay_record else "reconstruct",
        migration_path=str(replay_diagnostics.get("migration_path") or "none"),
        shared_classic_semantics=True,
        model_family_semantics=model_family.to_dict(),
    )
    return SharedClassicExecution(
        semantic_ir=ir,
        conditioning_plan=plan,
        parsed=parsed,
        canonical_prompt=canonical,
        canonical_structure=structure,
        warnings=tuple(warnings),
        model_family_semantics=model_family.to_dict(),
        ppsr_semantic_record=ppsr_record,
        replay_diagnostics={**dict(replay_diagnostics), "locked": bool(replay_record)},
    )
