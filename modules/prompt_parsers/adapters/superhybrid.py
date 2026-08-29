# -----------------------------------------------------------------------------
# REGION ATTRIBUTION
#
# REGION syntax and the original regional-conditioning design originate from
# work by GitHub user Konpr:
#   https://github.com/Konpr/whats-/tree/main/new_version3
#
# The original author granted permission to use/adapt the code with credit.
# This adapter/bridge is IMAGE_GEN-specific: it separates REGION extraction from
# the selected prompt parser and routes the resulting regional prompts through
# IMAGE_GEN's parser registry and native conditioning/runtime contracts.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import threading
import time
from contextlib import contextmanager
from typing import Any

from image_gen.systems.guidance.prompt_cfg import (
    PromptCFGScheduleError,
    build_prompt_cfg_payload,
)
from image_gen.systems.prompt_expansion import prompt_has_superhybrid_expansion_syntax
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

_CFG_DIRECTIVE_RE = re.compile(
    r"(?<!\\)<param\s*\[\s*cfg\s*\]\s*:\s*([^<]+?)\s*(?<!-)>",
    re.IGNORECASE,
)
_ANY_PARAM_RE = re.compile(
    r"(?<!\\)<param\s*\[\s*([^\]]+)\s*\]\s*:\s*([^<]+?)(?<!-)>",
    re.IGNORECASE,
)
_TONEG_RE = re.compile(r"(?<!\\)\bTONEG\s*\{", re.IGNORECASE)
_REGION_RE = re.compile(r"(?<!\\)\bREGION(?:X|Y)?\s*\{", re.IGNORECASE)
_WILDCARD_RE = re.compile(r"(?<!\\)__[^_\n]+__")

_BACKEND_OPTION_LOCK = threading.RLock()
_REQUEST_SCOPED_BACKEND_OPTIONS = {
    "allow_empty_alternate": ("ALLOW_EMPTY_ALTERNATE", bool),
    "expand_alternate_per_step": ("EXPAND_ALTERNATE_PER_STEP", bool),
    "group_combo_limit": ("GROUP_COMBO_LIMIT", int),
    "group_combo_fallback": ("GROUP_COMBO_FALLBACK", str),
    "dedup_schedule_steps": ("DEDUP_SCHEDULE_STEPS", bool),
    "suppress_standalone_colon": ("SUPPRESS_STANDALONE_COLON", bool),
    "bind2_use_path2": ("BIND2_USE_PATH2", bool),
    "bind2_normalize_weights": ("BIND2_NORMALIZE_WEIGHTS", bool),
    "bind3_cumulative_context": ("BIND3_CUMULATIVE_CONTEXT", bool),
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "end_at_step") and hasattr(value, "cond"):
        return {"end_at_step": int(value.end_at_step), "cond_type": type(value.cond).__name__}
    return str(value)


def _semantic_fingerprint(*, schedules: Any, flat_list: list[str], canonical: str, options: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "contract_version": "image-gen-superhybrid-semantics-v1",
        "schedules": _json_safe(schedules),
        "flat_list": [str(item) for item in flat_list],
        "canonical_prompt": str(canonical),
        "options": {key: _json_safe(options.get(key)) for key in sorted(options)},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "contract_version": payload["contract_version"],
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def _remove_cfg_directive(text: str) -> str:
    return _CFG_DIRECTIVE_RE.sub("", str(text or "")).strip()


class SuperHybridPromptParserAdapter:
    descriptor = PromptParserDescriptor(
        parser_id="superhybrid",
        label="SuperHybrid PP21 — Experimental",
        version="phase5-20260803+region-native-v1",
        aliases=("super_hybrid", "pp21_superhybrid", "parser21_superhybrid"),
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
            "variables": True,
            "macros": True,
            "seeded_random": True,
            "wildcards": True,
            "wildcard_replay": True,
            "prompt_expansion": True,
            "prompt_expansion_replay_lock": True,
            "per_image_prompt_expansion": True,
            "batch_aligned_conditioning": True,
            "semantic_fingerprint": True,
            "semantic_replay_lock": True,
            "request_scoped_semantic_options": True,
            "prompt_cfg_schedule": True,
            "cfg_lab_bridge": True,
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
            "prompt_cfg_replay_lock": True,
            "prompt_cfg_curves": True,
            "prompt_cfg_segment_positions": True,
            "region_runtime": True,
            "region_backend": "image_gen_model_output",
            "region_replay_lock": True,
            "region_batch_alignment": True,
            "region_hires": True,
            "toneg_runtime": True,
        },
        experimental=True,
        credit="SuperHybrid update based on Prompt Parser 21",
        source_url="https://github.com/Konpr/whats-new",
        settings_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "use_visitor": {
                    "type": "boolean",
                    "default": True,
                    "title": "Use visitor parser",
                    "description": "Use SuperHybrid's visitor-based schedule implementation.",
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
                "prompt_cfg_behavior": {
                    "type": "string",
                    "default": "replace_ui",
                    "enum": ["replace_ui", "shape_ui", "disabled"],
                    "title": "Prompt CFG behavior",
                    "description": "Replace UI CFG, preserve the prompt curve shape while anchoring it to UI CFG, or disable prompt CFG execution.",
                },
                "wildcard_directory": {
                    "type": "string",
                    "default": "wildcards",
                    "title": "Wildcard directory",
                    "description": "Project-relative wildcard root used by deterministic SuperHybrid expansion.",
                },
                "prompt_expansion_scope": {
                    "type": "string",
                    "default": "per_batch",
                    "enum": ["per_batch", "per_image"],
                    "title": "Prompt expansion scope",
                    "description": "Share one deterministic expansion across the batch or resolve one expansion per image seed.",
                },
                "region_overlap_policy": {
                    "type": "string",
                    "default": "additive",
                    "enum": ["additive", "normalize", "priority"],
                    "title": "REGION overlap policy",
                    "description": "Add regional deltas for SuperHybrid source parity, normalize overlaps, or apply later regions with priority.",
                },
                "allow_empty_alternate": {"type": "boolean", "default": True, "title": "Allow empty alternate"},
                "expand_alternate_per_step": {"type": "boolean", "default": True, "title": "Expand alternates per step"},
                "group_combo_limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 10000, "title": "Group combination limit"},
                "group_combo_fallback": {"type": "string", "default": "truncate", "enum": ["truncate", "literal", "sample"], "title": "Group combination fallback"},
                "dedup_schedule_steps": {"type": "boolean", "default": False, "title": "Deduplicate schedule steps"},
                "suppress_standalone_colon": {"type": "boolean", "default": True, "title": "Suppress standalone colon"},
                "bind2_use_path2": {"type": "boolean", "default": False, "title": "BIND2 path 2"},
                "bind2_normalize_weights": {"type": "boolean", "default": False, "title": "Normalize BIND2 weights"},
                "bind3_cumulative_context": {"type": "boolean", "default": False, "title": "BIND3 cumulative context"},
            },
        },
        process_scoped_settings=(
            "WEIGHT_INTERPRETATION",
            "APPLY_ADVANCED_WEIGHTS_ENABLED",
            "ATTENTION_DELTA_ADDITIVE",
            "PROMPT_PARSER_CACHE_SIZE",
            "PROMPT_PARSER_RECURSION_LIMIT",
        ),
    )

    @staticmethod
    def _backend():
        from modules.prompt_parsers.vendor import prompt_parser_superhybrid

        return prompt_parser_superhybrid

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        module_name = "modules.prompt_parsers.vendor.prompt_parser_superhybrid"
        try:
            available = importlib.util.find_spec(module_name) is not None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return (True, "") if available else (False, f"Module not found: {module_name}")

    @staticmethod
    def _option(options: dict[str, Any], key: str, default: Any) -> Any:
        return options.get(key, default)

    @staticmethod
    @contextmanager
    def _backend_option_context(backend: Any, options: dict[str, Any]):
        updates: dict[str, Any] = {}
        for option_name, (attribute, converter) in _REQUEST_SCOPED_BACKEND_OPTIONS.items():
            if option_name not in options or not hasattr(backend, attribute):
                continue
            value = options[option_name]
            if converter is str:
                value = str(value).strip().lower()
            else:
                value = converter(value)
            updates[attribute] = value
        with _BACKEND_OPTION_LOCK:
            previous = {attribute: getattr(backend, attribute) for attribute in updates}
            try:
                for attribute, value in updates.items():
                    setattr(backend, attribute, value)
                yield updates
            finally:
                for attribute, value in previous.items():
                    setattr(backend, attribute, value)

    def _directives(
        self,
        raw_prompt: str,
        *,
        prompt_role: str,
        steps: int,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        if _REGION_RE.search(raw_prompt):
            raise PromptParserError(
                "SuperHybrid REGION syntax is recognized, but REGION syntax must be extracted by the IMAGE_GEN Phase 5 regional bridge before parser execution.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="unsupported_superhybrid_region_runtime",
            )
        if _TONEG_RE.search(raw_prompt):
            raise PromptParserError(
                "SuperHybrid TONEG must be resolved by the IMAGE_GEN Phase 4 prompt-expansion bridge before parser execution.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="unsupported_superhybrid_toneg_runtime",
            )
        if _WILDCARD_RE.search(raw_prompt):
            raise PromptParserError(
                "SuperHybrid wildcard syntax must be resolved by the IMAGE_GEN Phase 4 prompt-expansion bridge before parser execution.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="unsupported_superhybrid_wildcard_runtime",
            )
        if prompt_has_superhybrid_expansion_syntax(raw_prompt):
            raise PromptParserError(
                "SuperHybrid variable, macro, random, or escaped expansion syntax must be resolved by the IMAGE_GEN Phase 4 prompt-expansion bridge before parser execution.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="unresolved_superhybrid_prompt_expansion",
            )

        all_params = list(_ANY_PARAM_RE.finditer(raw_prompt))
        unsupported_params = sorted(
            {
                match.group(1).strip().lower()
                for match in all_params
                if match.group(1).strip().lower() != "cfg"
            }
        )
        if unsupported_params:
            raise PromptParserError(
                "SuperHybrid runtime parameter(s) are not bridged in Phase 5: "
                + ", ".join(unsupported_params),
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="unsupported_superhybrid_runtime_parameter",
                diagnostics={"unsupported_parameters": unsupported_params},
            )

        cfg_matches = list(_CFG_DIRECTIVE_RE.finditer(raw_prompt))
        if re.search(r"(?<!\\)<param\s*\[\s*cfg\s*\]", raw_prompt, re.IGNORECASE) and not cfg_matches:
            raise PromptParserError(
                "Invalid SuperHybrid CFG directive syntax. Expected forms such as <param[cfg]:7>, <param[cfg]:8->3:smoothstep>, or <param[cfg]:8->6@0.4->3>.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="invalid_superhybrid_cfg_directive",
            )
        if len(cfg_matches) > 1:
            raise PromptParserError(
                "Only one SuperHybrid <param[cfg]:...> directive is supported per prompt in Phase 5.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="multiple_superhybrid_cfg_directives",
                diagnostics={"directive_count": len(cfg_matches)},
            )
        if not cfg_matches:
            return {}, warnings
        if prompt_role not in {"positive", "hires_positive"}:
            raise PromptParserError(
                "SuperHybrid CFG directives are only valid in the positive prompt.",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="superhybrid_cfg_in_negative_prompt",
            )

        match = cfg_matches[0]
        behavior = str(options.get("prompt_cfg_behavior", "replace_ui") or "replace_ui")
        try:
            cfg_payload = build_prompt_cfg_payload(
                match.group(1),
                total_steps=steps,
                parser_id=self.descriptor.parser_id,
                parser_version=self.descriptor.version,
                behavior=behavior,
                raw_directive=match.group(0),
            )
        except PromptCFGScheduleError as exc:
            raise PromptParserError(
                f"Invalid SuperHybrid CFG directive: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind="invalid_superhybrid_cfg_directive",
                diagnostics={"raw_directive": match.group(0)},
            ) from exc
        if not cfg_payload["enabled"]:
            warnings.append(
                "SuperHybrid CFG directive was parsed and removed from conditioning, but prompt_cfg_behavior is disabled."
            )
        return {"cfg": cfg_payload}, warnings

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
        options = dict(parser_options or {})
        parser_seed = options.get("seed", seed if seed is not None else 42)
        use_visitor = bool(options.get("use_visitor", True))
        use_old_scheduling = bool(options.get("use_old_scheduling", False))
        directives, warnings = self._directives(
            raw_prompt,
            prompt_role=prompt_role,
            steps=int(steps),
            options=options,
        )
        conditioning_prompt = _remove_cfg_directive(raw_prompt)
        shared = validate_shared_classic(
            conditioning_prompt,
            parser_namespace=self.descriptor.parser_id,
            prompt_role=prompt_role,
            steps=int(steps),
            hires_steps=hires_steps,
        )
        if shared is not None:
            shared["warnings"] = [*warnings, *shared.get("warnings", [])]
            shared["directives"] = directives
            shared["options_used"] = {
                "seed": parser_seed,
                "use_visitor": use_visitor,
                "use_old_scheduling": use_old_scheduling,
                "prompt_cfg_behavior": str(options.get("prompt_cfg_behavior", "replace_ui")),
                "wildcard_directory": str(options.get("wildcard_directory", "wildcards")),
                "prompt_expansion_scope": str(options.get("prompt_expansion_scope", "per_batch")),
                "request_scoped_backend_updates": {},
            }
            return shared

        backend = self._backend()
        try:
            with self._backend_option_context(backend, options) as backend_updates:
                schedules = backend.get_learned_conditioning_prompt_schedules(
                    [conditioning_prompt],
                    int(steps),
                    hires_steps,
                    use_old_scheduling,
                    parser_seed,
                    use_visitor=use_visitor,
                )
                prompts = backend.SdConditioning(
                    [conditioning_prompt],
                    is_negative_prompt=prompt_role in {"negative", "hires_negative"},
                )
                branches, flat, _ = backend.get_multicond_prompt_list(prompts)
        except PromptParserError:
            raise
        except Exception as exc:
            error_kind = str(getattr(exc, "kind", None) or type(exc).__name__)
            token = getattr(exc, "token", None)
            raise PromptParserError(
                f"SuperHybrid PP21 validation failed for {prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=prompt_role,
                error_kind=error_kind,
                diagnostics={
                    "error_kind": error_kind,
                    "token": str(token) if token is not None else None,
                    "raw_prompt_length": len(raw_prompt),
                },
            ) from exc
        return {
            "valid": True,
            "schedule_count": sum(len(item) for item in schedules or []),
            "branch_count": sum(len(item) for item in branches or []),
            "flat_prompt_count": len(flat or []),
            "warnings": warnings,
            "directives": directives,
            "options_used": {
                "seed": parser_seed,
                "use_visitor": use_visitor,
                "use_old_scheduling": use_old_scheduling,
                "prompt_cfg_behavior": str(options.get("prompt_cfg_behavior", "replace_ui")),
                "wildcard_directory": str(options.get("wildcard_directory", "wildcards")),
                "prompt_expansion_scope": str(options.get("prompt_expansion_scope", "per_batch")),
                "request_scoped_backend_updates": dict(backend_updates),
            },
            "semantic_fingerprint": _semantic_fingerprint(
                schedules=schedules,
                flat_list=list(flat or []),
                canonical=conditioning_prompt,
                options=options,
            ),
        }

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        started = time.perf_counter()
        options = dict(request.parser_options or {})
        warnings: list[str] = []
        supported_options = {
            "seed",
            "use_visitor",
            "use_old_scheduling",
            "prompt_cfg_behavior",
            "wildcard_directory",
            "prompt_expansion_scope",
            *_REQUEST_SCOPED_BACKEND_OPTIONS.keys(),
        }
        ignored = sorted(set(options) - supported_options)
        if ignored:
            warnings.append(
                "SuperHybrid option(s) are recorded but not runtime-mutable in Phase 4: "
                + ", ".join(ignored)
            )

        directives, directive_warnings = self._directives(
            request.raw_prompt,
            prompt_role=request.prompt_role,
            steps=request.steps,
            options=options,
        )
        warnings.extend(directive_warnings)
        seed = self._option(options, "seed", request.seed if request.seed is not None else 42)
        use_visitor = bool(self._option(options, "use_visitor", True))
        use_old_scheduling = bool(self._option(options, "use_old_scheduling", False))
        conditioning_prompt = _remove_cfg_directive(request.raw_prompt)
        shared = execute_shared_classic(
            request,
            parser_namespace=self.descriptor.parser_id,
            parser_version=self.descriptor.version,
            source=conditioning_prompt,
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
                    "conditioning_prompt": conditioning_prompt,
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
                        "prompt_cfg_behavior": str(options.get("prompt_cfg_behavior", "replace_ui")),
                        "wildcard_directory": str(options.get("wildcard_directory", "wildcards")),
                        "prompt_expansion_scope": str(options.get("prompt_expansion_scope", "per_batch")),
                        "request_scoped_backend_updates": {},
                    },
                },
                directives=directives,
            )

        backend = self._backend()
        prompts = backend.SdConditioning(
            [conditioning_prompt],
            is_negative_prompt=request.prompt_role in {"negative", "hires_negative"},
            width=request.width,
            height=request.height,
        )
        try:
            with self._backend_option_context(backend, options) as backend_updates:
                schedules = backend.get_learned_conditioning_prompt_schedules(
                    [conditioning_prompt],
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
        except PromptParserError:
            raise
        except Exception as exc:
            error_kind = str(getattr(exc, "kind", None) or type(exc).__name__)
            token = getattr(exc, "token", None)
            raise PromptParserError(
                f"SuperHybrid PP21 failed for {request.prompt_role}: {exc}",
                parser_id=self.descriptor.parser_id,
                prompt_role=request.prompt_role,
                error_kind=error_kind,
                diagnostics={
                    "error_kind": error_kind,
                    "token": str(token) if token is not None else None,
                    "raw_prompt_length": len(request.raw_prompt),
                },
            ) from exc

        parsed = type(
            "SuperHybridParsedPromptResult",
            (),
            {
                "multicond": multicond,
                "schedules": schedules,
                "flat_list": list(flat_list),
            },
        )()
        canonical, structure, canonical_warnings = canonicalize_prompt(
            conditioning_prompt,
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
                "conditioning_prompt": conditioning_prompt,
                "canonical_prompt_length": len(canonical),
                "schedule_count": len(schedules or []),
                "branch_count": len(getattr(multicond, "batch", []) or []),
                "warning_count": len(warnings),
                "fallback_behavior": "none",
                "source_sha256": "4959872ed5a3ac22aed88565d2556f4fa4941c85a02710bfb9e201193dddee3c",
                "options_used": {
                    "seed": seed,
                    "use_visitor": use_visitor,
                    "use_old_scheduling": use_old_scheduling,
                    "prompt_cfg_behavior": str(options.get("prompt_cfg_behavior", "replace_ui")),
                    "wildcard_directory": str(options.get("wildcard_directory", "wildcards")),
                    "prompt_expansion_scope": str(options.get("prompt_expansion_scope", "per_batch")),
                    "request_scoped_backend_updates": dict(backend_updates),
                },
                "semantic_fingerprint": _semantic_fingerprint(
                    schedules=schedules,
                    flat_list=list(flat_list),
                    canonical=canonical,
                    options=options,
                ),
            },
            directives=directives,
        )
