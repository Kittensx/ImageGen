from __future__ import annotations

import importlib.util
import time
from typing import Any

from modules.prompt_parsers.canonical import canonicalize_prompt
from modules.prompt_parsers.contracts import (
    PromptParseRequest,
    PromptParseResult,
    PromptParserDescriptor,
    PromptParserError,
)
from modules.prompt_parsers.shared_classic import (
    execute_shared_classic,
    validate_shared_classic,
)


class Parser21PromptParserAdapter:
    descriptor = PromptParserDescriptor(
        parser_id="parser21",
        label="Prompt Parser 21 — Experimental",
        version="21",
        aliases=("prompt_parser_21", "pp21", "konpr"),
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
            "bind": True,
            "chunk": True,
            "blend": True,
            "morph": True,
            "assemble": True,
            "pool": True,
            "compound": True,
            "attention_interpolation": True,
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
        experimental=True,
        credit="Contributed by GitHub user Konpr",
        source_url="https://github.com/Konpr/whats-new",
        settings_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "use_visitor": {
                    "type": "boolean",
                    "default": True,
                    "title": "Use visitor parser",
                    "description": "Use Parser 21's visitor-based schedule implementation.",
                },
                "use_old_scheduling": {
                    "type": "boolean",
                    "default": False,
                    "title": "Use old scheduling",
                    "description": "Use the base-pass schedule coordinate for hires-aware prompt schedules.",
                },
                "seed": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 2147483647,
                    "title": "Parser seed override",
                    "description": "Optional parser-only seed. Leave blank to inherit the generation seed.",
                    "x_nullable": True,
                },
            },
        },
        process_scoped_settings=(
            "ALLOW_EMPTY_ALTERNATE",
            "EXPAND_ALTERNATE_PER_STEP",
            "GROUP_COMBO_LIMIT",
            "DEDUP_SCHEDULE_STEPS",
            "GROUP_COMBO_FALLBACK",
            "SUPPRESS_STANDALONE_COLON",
            "PROMPT_PARSER_CACHE_SIZE",
            "PROMPT_PARSER_RECURSION_LIMIT",
            "BIND2_USE_PATH2",
            "BIND2_NORMALIZE_WEIGHTS",
        ),
    )

    @staticmethod
    def _backend():
        from modules.prompt_parsers.vendor import prompt_parser_fixed_v21

        return prompt_parser_fixed_v21

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        module_name = "modules.prompt_parsers.vendor.prompt_parser_fixed_v21"
        try:
            available = importlib.util.find_spec(module_name) is not None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return (True, "") if available else (False, f"Module not found: {module_name}")

    @staticmethod
    def _option(options: dict[str, Any], key: str, default: Any) -> Any:
        return options.get(key, default)

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
        """Run Parser 21's schedule grammar without loading conditioning tensors."""
        options = dict(parser_options or {})
        shared = validate_shared_classic(
            raw_prompt,
            parser_namespace=self.descriptor.parser_id,
            prompt_role=prompt_role,
            steps=int(steps),
            hires_steps=hires_steps,
        )
        if shared is not None:
            shared["options_used"] = {
                "seed": options.get("seed", seed if seed is not None else 42),
                "use_visitor": bool(options.get("use_visitor", True)),
                "use_old_scheduling": bool(options.get("use_old_scheduling", False)),
            }
            return shared

        backend = self._backend()
        parser_seed = options.get("seed", seed if seed is not None else 42)
        use_visitor = bool(options.get("use_visitor", True))
        use_old_scheduling = bool(options.get("use_old_scheduling", False))
        try:
            schedules = backend.get_learned_conditioning_prompt_schedules(
                [raw_prompt],
                int(steps),
                hires_steps,
                use_old_scheduling,
                parser_seed,
                use_visitor=use_visitor,
            )
            prompts = backend.SdConditioning(
                [raw_prompt],
                is_negative_prompt=prompt_role in {"negative", "hires_negative"},
            )
            branches, flat, _ = backend.get_multicond_prompt_list(prompts)
        except Exception as exc:
            error_kind = str(getattr(exc, "kind", None) or type(exc).__name__)
            token = getattr(exc, "token", None)
            diagnostics = {
                "error_kind": error_kind,
                "token": str(token) if token is not None else None,
                "raw_prompt_length": len(raw_prompt),
            }
            raise PromptParserError(
                f"Prompt Parser 21 validation failed for {prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind=error_kind,
                diagnostics=diagnostics,
            ) from exc
        return {
            "valid": True,
            "schedule_count": sum(len(item) for item in schedules or []),
            "branch_count": sum(len(item) for item in branches or []),
            "flat_prompt_count": len(flat or []),
            "warnings": [],
            "options_used": {
                "seed": parser_seed,
                "use_visitor": use_visitor,
                "use_old_scheduling": use_old_scheduling,
            },
        }

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        started = time.perf_counter()
        backend = self._backend()
        options = dict(request.parser_options or {})
        warnings: list[str] = []
        supported_options = {
            "seed",
            "use_visitor",
            "use_old_scheduling",
        }
        ignored = sorted(set(options) - supported_options)
        if ignored:
            warnings.append(
                "Parser 21 option(s) are recorded but not runtime-mutable in Phase 13C: "
                + ", ".join(ignored)
            )

        seed = self._option(options, "seed", request.seed if request.seed is not None else 42)
        use_visitor = bool(self._option(options, "use_visitor", True))
        use_old_scheduling = bool(self._option(options, "use_old_scheduling", False))

        shared = execute_shared_classic(
            request,
            parser_namespace=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
        )
        if shared is not None:
            warnings.extend(shared.warnings)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            plan = shared.conditioning_plan
            parsed = shared.parsed
            return PromptParseResult(
                parser_id=self.descriptor.parser_id,
                parser_version=self.descriptor.version,
                parser_contract_version=self.descriptor.contract_version,
                raw_prompt=request.raw_prompt,
                canonical_prompt=shared.canonical_prompt,
                canonical_structure=shared.canonical_structure,
                schedules=parsed.schedules,
                conditioning_source=parsed,
                semantic_ir=shared.semantic_ir,
                conditioning_plan=plan,
                warnings=warnings,
                diagnostics={
                    "parse_duration_ms": elapsed_ms,
                    "raw_prompt_length": len(request.raw_prompt),
                    "canonical_prompt_length": len(shared.canonical_prompt),
                    "schedule_count": len(parsed.schedules or []),
                    "branch_count": sum(len(row) for row in getattr(parsed.multicond, "batch", []) or []),
                    "warning_count": len(warnings),
                    "fallback_behavior": "safe_flatten" if plan.fallbacks else "none",
                    "shared_classic_semantics": True,
                    "conditioning_plan_contract": plan.contract,
                    "model_family_semantics": dict(shared.model_family_semantics),
                    "relationship_diagnostics": [item.to_dict() for item in plan.relationship_diagnostics],
                    "ppsr_semantic_record": dict(shared.ppsr_semantic_record),
                    "ppsr_replay": dict(shared.replay_diagnostics),
                    "semantic_digest": dict(shared.ppsr_semantic_record.get("semantic_digest") or {}),
                    "options_used": {
                        "seed": seed,
                        "use_visitor": use_visitor,
                        "use_old_scheduling": use_old_scheduling,
                    },
                },
            )

        prompts = backend.SdConditioning(
            [request.raw_prompt],
            is_negative_prompt=request.prompt_role in {"negative", "hires_negative"},
            width=request.width,
            height=request.height,
        )
        try:
            schedules = backend.get_learned_conditioning_prompt_schedules(
                [request.raw_prompt],
                request.steps,
                request.hires_steps,
                use_old_scheduling,
                seed,
                use_visitor=use_visitor,
            )
            multicond = backend.get_multicond_learned_conditioning(
                request.model_context,
                prompts,
                request.steps,
                request.hires_steps,
                use_old_scheduling,
                seed,
                use_visitor,
            )
            _indexes, flat_list, _prompt_indexes = backend.get_multicond_prompt_list(prompts)
        except Exception as exc:
            error_kind = str(getattr(exc, "kind", None) or type(exc).__name__)
            token = getattr(exc, "token", None)
            diagnostics = {
                "error_kind": error_kind,
                "token": str(token) if token is not None else None,
                "raw_prompt_length": len(request.raw_prompt),
            }
            raise PromptParserError(
                f"Prompt Parser 21 failed for {request.prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=request.prompt_role,
                error_kind=error_kind,
                diagnostics=diagnostics,
            ) from exc

        parsed = type(
            "Parser21ParsedPromptResult",
            (),
            {
                "multicond": multicond,
                "schedules": schedules,
                "flat_list": list(flat_list),
            },
        )()
        canonical, structure, canonical_warnings = canonicalize_prompt(
            request.raw_prompt,
            parser_id=self.descriptor.parser_id,
        )
        warnings.extend(canonical_warnings)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return PromptParseResult(
            parser_id=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            parser_contract_version=self.descriptor.contract_version,
            raw_prompt=request.raw_prompt,
            canonical_prompt=canonical,
            canonical_structure=structure,
            schedules=schedules,
            conditioning_source=parsed,
            warnings=warnings,
            diagnostics={
                "parse_duration_ms": elapsed_ms,
                "raw_prompt_length": len(request.raw_prompt),
                "canonical_prompt_length": len(canonical),
                "schedule_count": len(schedules or []),
                "branch_count": len(getattr(multicond, "batch", []) or []),
                "warning_count": len(warnings),
                "fallback_behavior": "none",
                "options_used": {
                    "seed": seed,
                    "use_visitor": use_visitor,
                    "use_old_scheduling": use_old_scheduling,
                },
            },
        )
