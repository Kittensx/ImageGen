from __future__ import annotations

import time

from modules.parser.prompt_parser_class import PromptParserClass
from modules.prompt_parsers.canonical import canonicalize_prompt
from modules.prompt_parsers.compiler import compile_conditioning_plan
from modules.prompt_parsers.ir import parse_prompt_ir
from modules.prompt_parsers.semantic_replay import (
    PromptSemanticReplayError,
    build_ppsr_semantic_record,
    resolve_request_prompt_ir,
    validate_replayed_ppsr_result,
)
from modules.prompt_parsers.model_family_compiler import compile_plan_for_runtime
from modules.prompt_parsers.contracts import (
    PromptParseRequest,
    PromptParseResult,
    PromptParserDescriptor,
    PromptParserError,
)


class LegacyPromptParserAdapter:
    descriptor = PromptParserDescriptor(
        parser_id="legacy",
        label="IMAGE_GEN Legacy Prompt Parser",
        version="3",
        aliases=("default", "current", "image_gen", "scheduled"),
        capabilities={
            "positive_prompt": True,
            "negative_prompt": True,
            "hires": True,
            "plain_text": True,
            "attention_weights": True,
            "scheduled_prompts": True,
            "alternates": True,
            "and_composition": True,
            "sequence": True,
            "deep_sequence": True,
            "group": True,
            "scheduling": True,
            "alternation": True,
            "composable_prompts": True,
            "bind": False,
            "chunk": False,
            "blend": False,
            "pool": False,
            "morph": False,
            "assemble": False,
            "compound": False,
            "attention_interpolation": False,
            "canonical_serialization": True,
            "exact_replay": True,
            "group_syntax": True,
            "group_semantics": True,
            "sequence_syntax": True,
            "sequence_semantics": True,
            "temporal_semantics": True,
            "semantic_replay": True,
            "semantic_digest": True,
            "semantic_inspection": True,
        },
        experimental=False,
        credit="IMAGE_GEN built-in prompt parser",
        settings_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    )

    @staticmethod
    def availability() -> tuple[bool, str]:
        return True, ""

    def validate_syntax(
        self,
        raw_prompt: str,
        *,
        prompt_role: str,
        steps: int,
        hires_steps: int | None = None,
        parser_options: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        """Validate schedules and branch weights without loading a text encoder."""
        from types import SimpleNamespace

        from modules.parser.learned_conditioning import LearnedConditioning
        from modules.shared_state import SharedState

        state = SharedState()
        state.p.steps = int(steps)
        state.p.width = 64
        state.p.height = 64
        state.conditioning.use_old_scheduling = False
        learned = LearnedConditioning(
            shared_state=state,
            steps=int(steps),
            hires_steps=hires_steps,
            prompts=[raw_prompt],
        )
        try:
            schedules = learned.get_learned_cond([raw_prompt], int(steps), hires_steps)
            semantic_ir = parse_prompt_ir(
                raw_prompt,
                parser_namespace=self.descriptor.parser_id,
                semantic_modes=dict((parser_options or {}).get("_prompt_style_semantic_modes") or {}),
            )
            conditioning_plan = compile_conditioning_plan(semantic_ir)
            parser = PromptParserClass(
                shared_state=state,
                prompts=[raw_prompt],
                steps=int(steps),
                model=SimpleNamespace(),
                hires_steps=hires_steps,
                conditioning_plans=[conditioning_plan],
                semantic_modes=dict((parser_options or {}).get("_prompt_style_semantic_modes") or {}),
            )
            branches, flat, _ = parser.get_multicond_prompt_list()
        except Exception as exc:
            raise PromptParserError(
                f"Legacy prompt parser validation failed for {prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind=type(exc).__name__,
                diagnostics={"raw_prompt_length": len(raw_prompt)},
            ) from exc
        return {
            "valid": True,
            "schedule_count": sum(len(item) for item in schedules or []),
            "branch_count": sum(len(item) for item in branches or []),
            "flat_prompt_count": len(flat or []),
            "warnings": list(getattr(semantic_ir, "warnings", ()) or ()),
            "semantic_ir": semantic_ir.to_dict(),
            "conditioning_plan": conditioning_plan.to_dict(),
        }

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        started = time.perf_counter()
        try:
            semantic_ir, replay_diagnostics = resolve_request_prompt_ir(
                request,
                source=request.raw_prompt,
                parser_namespace=self.descriptor.parser_id,
                parser_contract_version=self.descriptor.contract_version,
            )
        except PromptSemanticReplayError as exc:
            raise PromptParserError(
                f"PPSR semantic replay failed for {request.prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=request.prompt_role,
                error_kind="ppsr_semantic_replay_failed",
            ) from exc
        conditioning_plan = compile_conditioning_plan(semantic_ir)
        model_family = compile_plan_for_runtime(
            semantic_ir, conditioning_plan, request.model_context,
            total_steps=int(request.hires_steps or request.steps),
        )
        conditioning_plan = model_family.plan
        replay_record = None
        if str(replay_diagnostics.get("mode") or "") == "recorded_exact":
            try:
                replay_record = validate_replayed_ppsr_result(
                    dict(request.recorded_semantic_replay or {}),
                    semantic_ir=semantic_ir,
                    conditioning_plan=conditioning_plan,
                )
            except PromptSemanticReplayError as exc:
                raise PromptParserError(
                    f"PPSR semantic replay validation failed for {request.prompt_role}: {exc}",
                    parser_id=self.descriptor.parser_id,
                    prompt_role=request.prompt_role,
                    error_kind="ppsr_semantic_replay_failed",
                ) from exc
        try:
            parsed = PromptParserClass(
                shared_state=request.shared_state,
                prompts=[request.raw_prompt],
                steps=request.steps,
                model=request.model_context,
                hires_steps=request.hires_steps,
                conditioning_plans=[conditioning_plan],
                semantic_modes=dict((request.parser_options or {}).get("_prompt_style_semantic_modes") or {}),
            )()
        except Exception as exc:
            raise PromptParserError(
                f"Legacy prompt parser failed for {request.prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=request.prompt_role,
                error_kind=type(exc).__name__,
            ) from exc
        canonical, structure, warnings = canonicalize_prompt(
            request.raw_prompt,
            parser_id=self.descriptor.parser_id,
            semantic_ir=semantic_ir,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        ppsr_record = build_ppsr_semantic_record(
            parser_id=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            parser_contract_version=self.descriptor.contract_version,
            prompt_role=request.prompt_role,
            raw_prompt=request.raw_prompt,
            canonical_structure=structure,
            semantic_ir=semantic_ir,
            conditioning_plan=conditioning_plan,
            parser_seed=request.seed,
            replay_source="recorded_exact" if replay_record else "reconstruct",
            migration_path=str(replay_diagnostics.get("migration_path") or "none"),
            shared_classic_semantics=True,
            model_family_semantics=model_family.to_dict(),
        )
        return PromptParseResult(
            parser_id=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            parser_contract_version=self.descriptor.contract_version,
            raw_prompt=request.raw_prompt,
            canonical_prompt=canonical,
            canonical_structure=structure,
            schedules=parsed.schedules,
            conditioning_source=parsed,
            semantic_ir=semantic_ir,
            conditioning_plan=conditioning_plan,
            warnings=warnings,
            diagnostics={
                "parse_duration_ms": elapsed_ms,
                "raw_prompt_length": len(request.raw_prompt),
                "canonical_prompt_length": len(canonical),
                "schedule_count": len(parsed.schedules or []),
                "branch_count": len(getattr(parsed.multicond, "batch", []) or []),
                "fallback_behavior": (
                    "safe_flatten" if conditioning_plan.fallbacks else "none"
                ),
                "conditioning_plan_contract": conditioning_plan.contract,
                "conditioning_plan_fallbacks": list(conditioning_plan.fallbacks),
                "group_operation_count": len({
                    item.group_operation_id
                    for item in conditioning_plan.branches
                    if getattr(item, "group_operation_id", None)
                }),
                "group_diagnostics": [
                    item.to_dict() for item in conditioning_plan.group_diagnostics
                ],
                "relationship_operation_count": len(conditioning_plan.relationship_diagnostics),
                "relationship_diagnostics": [
                    item.to_dict() for item in conditioning_plan.relationship_diagnostics
                ],
                "shared_classic_semantics": bool(
                    conditioning_plan.relationship_diagnostics or conditioning_plan.group_diagnostics
                ),
                "semantic_ir_contract": semantic_ir.contract,
                "semantic_root_type": type(semantic_ir.root).__name__,
                "options_used": dict(request.parser_options or {}),
                "model_family_semantics": model_family.to_dict(),
                "ppsr_semantic_record": ppsr_record,
                "ppsr_replay": {**dict(replay_diagnostics), "locked": bool(replay_record)},
                "semantic_digest": dict(ppsr_record.get("semantic_digest") or {}),
            },
        )
