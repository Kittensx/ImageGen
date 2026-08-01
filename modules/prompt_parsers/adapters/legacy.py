from __future__ import annotations

import time

from modules.parser.prompt_parser_class import PromptParserClass
from modules.prompt_parsers.canonical import canonicalize_prompt
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
        version="1",
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
            parser = PromptParserClass(
                shared_state=state,
                prompts=[raw_prompt],
                steps=int(steps),
                model=SimpleNamespace(),
                hires_steps=hires_steps,
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
            "warnings": [],
        }

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        started = time.perf_counter()
        try:
            parsed = PromptParserClass(
                shared_state=request.shared_state,
                prompts=[request.raw_prompt],
                steps=request.steps,
                model=request.model_context,
                hires_steps=request.hires_steps,
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
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return PromptParseResult(
            parser_id=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            parser_contract_version=self.descriptor.contract_version,
            raw_prompt=request.raw_prompt,
            canonical_prompt=canonical,
            canonical_structure=structure,
            schedules=parsed.schedules,
            conditioning_source=parsed,
            warnings=warnings,
            diagnostics={
                "parse_duration_ms": elapsed_ms,
                "raw_prompt_length": len(request.raw_prompt),
                "canonical_prompt_length": len(canonical),
                "schedule_count": len(parsed.schedules or []),
                "branch_count": len(getattr(parsed.multicond, "batch", []) or []),
                "fallback_behavior": "none",
                "options_used": dict(request.parser_options or {}),
            },
        )
