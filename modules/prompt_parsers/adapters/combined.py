from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping

from modules.prompt_parsers.adapters.legacy import LegacyPromptParserAdapter
from modules.prompt_parsers.adapters.parser21 import Parser21PromptParserAdapter
from modules.prompt_parsers.canonical import canonicalize_prompt
from modules.prompt_parsers.contracts import (
    PromptParseRequest,
    PromptParseResult,
    PromptParserDescriptor,
    PromptParserError,
)
from modules.prompt_parsers.routing import (
    PromptRoutePlanner,
    assert_recorded_route_matches,
    normalize_fallback_policy,
)


class CombinedPromptParserAdapter:
    descriptor = PromptParserDescriptor(
        parser_id="combined",
        label="Combined / Auto Dispatch — Experimental",
        version="1",
        aliases=("auto", "hybrid", "combined_dispatcher", "auto_dispatch"),
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
            "bind": True,
            "chunk": True,
            "blend": True,
            "pool": True,
            "morph": True,
            "assemble": True,
            "compound": True,
            "attention_interpolation": True,
            "canonical_serialization": True,
            "exact_replay": True,
            "combined_dispatch": True,
            "auto_split": False,
        },
        experimental=True,
        credit="IMAGE_GEN combined-dispatch layer over the installed prompt parsers",
        settings_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": [
                        "prefer_legacy",
                        "prefer_parser21",
                        "strict_by_capability",
                        "single_parser_only",
                        "auto_split",
                    ],
                    "default": "prefer_legacy",
                    "title": "Dispatcher strategy",
                    "description": "Select how the combined dispatcher chooses between installed parser engines.",
                },
                "preferred_parser": {
                    "type": "string",
                    "enum": ["legacy", "parser21"],
                    "default": "legacy",
                    "title": "Overlap preference",
                    "description": "Deterministic tie-break when both parsers support the complete canonical prompt.",
                },
                "fail_on_ambiguous_route": {
                    "type": "boolean",
                    "default": False,
                    "title": "Fail on ambiguous route",
                    "description": "Block generation instead of applying the configured tie-break when multiple routes are valid.",
                },
                "fallback_policy": {
                    "type": "string",
                    "enum": [
                        "fail",
                        "warn_and_literalize",
                        "warn_and_use_legacy",
                        "warn_and_use_parser21",
                    ],
                    "default": "fail",
                    "title": "Parser failure policy",
                    "description": "Exact replay should use fail. Other policies are explicit, diagnostic fallbacks.",
                },
                "parser21_use_visitor": {
                    "type": "boolean",
                    "default": True,
                    "title": "Parser 21 visitor parser",
                    "description": "Used only when the route selects Parser 21.",
                },
                "parser21_use_old_scheduling": {
                    "type": "boolean",
                    "default": False,
                    "title": "Parser 21 old scheduling",
                    "description": "Used only when the route selects Parser 21.",
                },
                "parser21_seed": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 2147483647,
                    "title": "Parser 21 seed override",
                    "description": "Leave blank to inherit the generation seed.",
                    "x_nullable": True,
                },
            },
        },
    )

    def __init__(self) -> None:
        self.legacy = LegacyPromptParserAdapter()
        self.parser21 = Parser21PromptParserAdapter()
        self._adapters = {"legacy": self.legacy, "parser21": self.parser21}

    def availability(self) -> tuple[bool, str]:
        available, reason = self.parser21.availability()
        if not available:
            return False, f"Parser 21 is required for combined dispatch: {reason}"
        return True, ""

    def _planner(self) -> PromptRoutePlanner:
        return PromptRoutePlanner({
            "legacy": self.legacy.descriptor.to_dict(),
            "parser21": self.parser21.descriptor.to_dict(),
        })

    @staticmethod
    def _route_options(options: Mapping[str, Any]) -> dict[str, Any]:
        source = dict(options or {})
        return {
            "strategy": source.get("strategy", "prefer_legacy"),
            "preferred_parser": source.get("preferred_parser", "legacy"),
            "fail_on_ambiguous_route": bool(source.get("fail_on_ambiguous_route", False)),
            "fallback_policy": normalize_fallback_policy(source.get("fallback_policy", "fail")),
        }

    @staticmethod
    def _underlying_options(parser_id: str, options: Mapping[str, Any], seed: int | None) -> dict[str, Any]:
        if parser_id != "parser21":
            return {}
        source = dict(options or {})
        parser_seed = source.get("parser21_seed", seed)
        output = {
            "use_visitor": bool(source.get("parser21_use_visitor", True)),
            "use_old_scheduling": bool(source.get("parser21_use_old_scheduling", False)),
        }
        if parser_seed not in (None, ""):
            output["seed"] = int(parser_seed)
        return output

    def _plan(self, raw_prompt: str, options: Mapping[str, Any]) -> dict[str, Any]:
        _canonical, structure, _warnings = canonicalize_prompt(raw_prompt, parser_id="combined")
        route_options = self._route_options(options)
        plan = self._planner().plan(structure, **route_options)
        payload = plan.to_dict()
        if not payload.get("selected_parser"):
            kind = "ambiguous_prompt_route" if payload.get("ambiguities") else "unsupported_prompt_route"
            raise PromptParserError(
                "Combined prompt dispatch could not select a safe parser route.",
                parser_id=self.descriptor.parser_id,
                prompt_role="unknown",
                error_kind=kind,
                diagnostics={"route_plan": payload},
            )
        return payload

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
        started = time.perf_counter()
        options = dict(parser_options or {})
        try:
            route_plan = self._plan(raw_prompt, options)
        except PromptParserError as exc:
            exc.prompt_role = prompt_role
            raise
        selected = str(route_plan["selected_parser"])
        adapter = self._adapters[selected]
        validation = adapter.validate_syntax(
            raw_prompt,
            prompt_role=prompt_role,
            steps=int(steps),
            hires_steps=hires_steps,
            parser_options=self._underlying_options(selected, options, seed),
            seed=seed,
        )
        return {
            **dict(validation),
            "valid": True,
            "selected_parser": selected,
            "route_plan": route_plan,
            "dispatcher_duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "warnings": [*list(route_plan.get("warnings") or []), *list(validation.get("warnings") or [])],
        }

    def _delegate_request(self, request: PromptParseRequest, parser_id: str) -> PromptParseRequest:
        return replace(
            request,
            parser_options=self._underlying_options(parser_id, request.parser_options, request.seed),
            recorded_route_plan=None,
        )

    def _parse_with(self, parser_id: str, request: PromptParseRequest) -> PromptParseResult:
        return self._adapters[parser_id].parse(self._delegate_request(request, parser_id))

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        started = time.perf_counter()
        options = dict(request.parser_options or {})
        try:
            route_plan = self._plan(request.raw_prompt, options)
        except PromptParserError as exc:
            exc.prompt_role = request.prompt_role
            raise
        assert_recorded_route_matches(request.recorded_route_plan, route_plan)
        selected = str(route_plan["selected_parser"])
        fallback_policy = str(route_plan.get("fallback_policy") or "fail")
        fallback_record: dict[str, Any] | None = None
        try:
            delegated = self._parse_with(selected, request)
        except PromptParserError as exc:
            if fallback_policy == "fail":
                exc.diagnostics.setdefault("route_plan", route_plan)
                exc.diagnostics.setdefault("selected_parser", selected)
                raise
            fallback_parser = {
                "warn_and_literalize": "legacy",
                "warn_and_use_legacy": "legacy",
                "warn_and_use_parser21": "parser21",
            }[fallback_policy]
            if fallback_parser == selected and fallback_policy != "warn_and_literalize":
                exc.diagnostics.setdefault("route_plan", route_plan)
                raise
            delegated = self._parse_with(fallback_parser, request)
            fallback_record = {
                "policy": fallback_policy,
                "failed_parser": selected,
                "fallback_parser": fallback_parser,
                "reason": exc.to_dict(),
                "literalized": fallback_policy == "warn_and_literalize",
            }
            selected = fallback_parser

        canonical, structure, canonical_warnings = canonicalize_prompt(
            request.raw_prompt,
            parser_id=self.descriptor.parser_id,
        )
        warnings = [
            *list(route_plan.get("warnings") or []),
            *list(delegated.warnings or []),
            *canonical_warnings,
        ]
        if fallback_record:
            warnings.append(
                f"Combined dispatch fallback used {fallback_record['fallback_parser']} after "
                f"{fallback_record['failed_parser']} failed under {fallback_record['policy']}."
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return PromptParseResult(
            parser_id=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            parser_contract_version=self.descriptor.contract_version,
            raw_prompt=request.raw_prompt,
            canonical_prompt=canonical,
            canonical_structure={
                **structure,
                "route_plan": route_plan,
                "selected_parser_canonical_structure": delegated.canonical_structure,
            },
            schedules=delegated.schedules,
            conditioning_source=delegated.conditioning_source,
            warnings=warnings,
            diagnostics={
                "parse_duration_ms": elapsed_ms,
                "raw_prompt_length": len(request.raw_prompt),
                "canonical_prompt_length": len(canonical),
                "schedule_count": delegated.diagnostics.get("schedule_count", len(delegated.schedules or [])),
                "branch_count": delegated.diagnostics.get("branch_count", 0),
                "warning_count": len(warnings),
                "selected_parser": selected,
                "selected_parser_version": delegated.parser_version,
                "selected_parser_diagnostics": dict(delegated.diagnostics),
                "route_plan": route_plan,
                "fallback_behavior": fallback_record or "none",
                "options_used": options,
            },
        )
