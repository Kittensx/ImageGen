from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Mapping

from modules.prompt_parsers.canonical import canonicalize_prompt
from modules.prompt_parsers.contracts import PromptParserError
from modules.prompt_parsers.adapters.legacy import LegacyPromptParserAdapter
from modules.prompt_parsers.adapters.parser21 import Parser21PromptParserAdapter
from modules.prompt_parsers.routing import shadow_compare_parsers
from modules.prompt_parsers.registry import PromptParserRegistry, default_prompt_parser_registry
from modules.prompt_shortcuts import (
    PromptShortcutError,
    PromptShortcutProfileDescriptor,
    PromptShortcutProfileRegistry,
    PromptShortcutTranslator,
    default_prompt_shortcut_registry,
    validate_prompt_shortcut_profile,
)

PROMPT_PREFLIGHT_CONTRACT_VERSION = "image-gen-prompt-preflight-v1"
_HIRES_MODES = {"same_as_base", "explicit", "canonical_only"}
_SYNTAX_CACHE_LIMIT = 512
_SYNTAX_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_SYNTAX_CACHE_LOCK = threading.Lock()


def _cached_syntax_validation(
    adapter: Any,
    raw_prompt: str,
    *,
    prompt_role: str,
    steps: int,
    hires_steps: int | None,
    parser_options: Mapping[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    parser_id = str(adapter.descriptor.parser_id)
    effective_seed = seed if parser_id in {"parser21", "combined"} else None
    key = (
        parser_id,
        str(adapter.descriptor.version),
        prompt_role,
        raw_prompt,
        int(steps),
        hires_steps,
        json.dumps(dict(parser_options), sort_keys=True, default=str),
        effective_seed,
    )
    with _SYNTAX_CACHE_LOCK:
        cached = _SYNTAX_CACHE.get(key)
        if cached is not None:
            _SYNTAX_CACHE.move_to_end(key)
            result = copy.deepcopy(cached)
            result["validation_cache"] = "hit"
            return result
    validation_started = time.perf_counter()
    result = adapter.validate_syntax(
        raw_prompt,
        prompt_role=prompt_role,
        steps=int(steps),
        hires_steps=hires_steps,
        parser_options=dict(parser_options),
        seed=seed,
    )
    result = dict(result)
    result["validation_cache"] = "miss"
    result["validation_duration_ms"] = round((time.perf_counter() - validation_started) * 1000.0, 3)
    with _SYNTAX_CACHE_LOCK:
        _SYNTAX_CACHE[key] = copy.deepcopy(result)
        _SYNTAX_CACHE.move_to_end(key)
        while len(_SYNTAX_CACHE) > _SYNTAX_CACHE_LIMIT:
            _SYNTAX_CACHE.popitem(last=False)
    return result


def _message(severity: str, code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        **{key: value for key, value in details.items() if value is not None},
    }


def _normalize_mode(value: Any, *, field_name: str) -> str:
    mode = str(value or "same_as_base").strip().lower().replace("-", "_")
    if mode not in _HIRES_MODES:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(sorted(_HIRES_MODES))}."
        )
    return mode


def _coerce_parser_options(
    descriptor: Mapping[str, Any],
    raw_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = dict(descriptor.get("settings_schema") or {})
    properties = dict(schema.get("properties") or {})
    additional = bool(schema.get("additionalProperties", False))
    options = dict(raw_options or {})
    normalized: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []

    for key, raw in options.items():
        prop = dict(properties.get(key) or {})
        if not prop:
            if additional:
                normalized[key] = raw
            else:
                messages.append(_message(
                    "blocking_error",
                    "unknown_parser_option",
                    f"Parser option {key!r} is not supported by {descriptor.get('label') or descriptor.get('parser_id')}.",
                    field=f"prompt_parser_kwargs.{key}",
                ))
            continue
        if raw is None and prop.get("x_nullable"):
            continue
        kind = str(prop.get("type") or "string")
        try:
            if kind == "boolean":
                if isinstance(raw, bool):
                    value = raw
                elif str(raw).strip().lower() in {"1", "true", "yes", "on"}:
                    value = True
                elif str(raw).strip().lower() in {"0", "false", "no", "off"}:
                    value = False
                else:
                    raise ValueError("expected true or false")
            elif kind == "integer":
                value = int(raw)
            elif kind == "number":
                value = float(raw)
            else:
                value = str(raw)
        except (TypeError, ValueError) as exc:
            messages.append(_message(
                "blocking_error",
                "invalid_parser_option",
                f"Parser option {key!r} is invalid: {exc}.",
                field=f"prompt_parser_kwargs.{key}",
            ))
            continue
        if "enum" in prop and value not in prop.get("enum", []):
            messages.append(_message(
                "blocking_error",
                "parser_option_not_in_enum",
                f"Parser option {key!r} must be one of {prop.get('enum')}.",
                field=f"prompt_parser_kwargs.{key}",
            ))
            continue
        minimum = prop.get("minimum")
        maximum = prop.get("maximum")
        if minimum is not None and value < minimum:
            messages.append(_message(
                "blocking_error", "parser_option_below_minimum",
                f"Parser option {key!r} must be at least {minimum}.",
                field=f"prompt_parser_kwargs.{key}",
            ))
            continue
        if maximum is not None and value > maximum:
            messages.append(_message(
                "blocking_error", "parser_option_above_maximum",
                f"Parser option {key!r} must be no greater than {maximum}.",
                field=f"prompt_parser_kwargs.{key}",
            ))
            continue
        normalized[key] = value

    for key, prop_value in properties.items():
        prop = dict(prop_value or {})
        if key not in normalized and "default" in prop:
            normalized[key] = copy.deepcopy(prop["default"])
    return normalized, messages


class PromptProcessingPreflight:
    """Model-free parser/profile validation shared by WebUI and queued workflows."""

    def __init__(
        self,
        *,
        parser_registry: PromptParserRegistry | None = None,
        profile_registry: PromptShortcutProfileRegistry | None = None,
    ) -> None:
        self.parser_registry = parser_registry or default_prompt_parser_registry()
        self.profile_registry = profile_registry or default_prompt_shortcut_registry()
        self.translator = PromptShortcutTranslator()

    def _resolve_profile(
        self,
        profile_name: Any,
        *,
        parser_id: str,
        snapshot: Mapping[str, Any] | None = None,
    ) -> PromptShortcutProfileDescriptor:
        if snapshot:
            profile = PromptShortcutProfileDescriptor.from_dict(
                dict(snapshot), builtin=bool(snapshot.get("builtin", False))
            )
            validation = validate_prompt_shortcut_profile(profile)
            if not validation.valid:
                raise ValueError(
                    "Embedded prompt shortcut profile is invalid: "
                    + " | ".join(issue.message for issue in validation.errors)
                )
        else:
            if parser_id == "legacy":
                fallback = "legacy_default"
            elif parser_id == "parser21":
                fallback = "parser21_native"
            else:
                fallback = "canonical"
            profile = self.profile_registry.get(profile_name or fallback)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21"))
        )
        if not compatible:
            raise ValueError(
                f"Prompt shortcut profile {profile.profile_id!r} is not compatible with parser {parser_id!r}."
            )
        return profile

    def _validate_pass(
        self,
        *,
        pass_name: str,
        parser_name: Any,
        parser_options: Mapping[str, Any] | None,
        profile_name: Any,
        profile_snapshot: Mapping[str, Any] | None,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        hires_steps: int | None,
        seed: int | None,
        shadow_compare: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = []
        parser_id = self.parser_registry.resolve_id(parser_name or "legacy")
        available, reason = self.parser_registry.availability(parser_id)
        if not available:
            raise ValueError(f"Prompt parser {parser_id!r} is unavailable: {reason}")
        adapter = self.parser_registry.get(parser_id)
        descriptor = adapter.descriptor.to_dict()
        options, option_messages = _coerce_parser_options(descriptor, parser_options)
        messages.extend(option_messages)
        profile = self._resolve_profile(
            profile_name,
            parser_id=parser_id,
            snapshot=profile_snapshot,
        )

        if descriptor.get("experimental"):
            messages.append(_message(
                "behavior_warning",
                "experimental_prompt_parser",
                f"{descriptor.get('label') or parser_id} is experimental and may interpret prompts differently from the legacy parser.",
                pass_name=pass_name,
            ))
        process_settings = list(descriptor.get("process_scoped_settings") or [])
        if process_settings:
            messages.append(_message(
                "informational_notice",
                "process_scoped_parser_settings",
                "Some parser settings are process-scoped and cannot safely vary per queued request.",
                pass_name=pass_name,
                settings=process_settings,
            ))

        roles: dict[str, Any] = {}
        for role, raw_prompt in (("positive", positive_prompt), ("negative", negative_prompt)):
            try:
                translated = self.translator.translate(
                    str(raw_prompt or ""),
                    profile=profile,
                    parser_id=parser_id,
                    prompt_role=f"{pass_name}_{role}" if pass_name == "hires" else role,
                )
            except PromptShortcutError as exc:
                messages.append(_message(
                    "blocking_error",
                    exc.error_kind,
                    str(exc),
                    pass_name=pass_name,
                    prompt_role=role,
                    diagnostics=exc.to_dict(),
                ))
                fallback_prompt = str(raw_prompt or "")
                canonical_prompt, canonical_structure, canonical_warnings = canonicalize_prompt(
                    fallback_prompt,
                    parser_id=parser_id,
                )
                roles[role] = {
                    "raw_prompt": fallback_prompt,
                    "parser_input": fallback_prompt,
                    "canonical_prompt": fallback_prompt,
                    "canonical_structure": canonical_structure,
                    "substitutions": [],
                    "warnings": list(canonical_warnings),
                    "diagnostics": {"translation_error": exc.to_dict()},
                    "parser_canonical_prompt": canonical_prompt,
                    "parser_canonical_structure": canonical_structure,
                    "parser_validation": {"valid": False, "error": exc.to_dict()},
                }
                continue
            for warning in translated.warnings:
                messages.append(_message(
                    "behavior_warning",
                    "prompt_translation_warning",
                    str(warning),
                    pass_name=pass_name,
                    prompt_role=role,
                ))
            validation: dict[str, Any] = {"valid": True, "warnings": []}
            validator = getattr(adapter, "validate_syntax", None)
            if callable(validator) and not any(item["severity"] == "blocking_error" for item in option_messages):
                try:
                    validation = _cached_syntax_validation(
                        adapter,
                        translated.parser_input,
                        prompt_role=f"hires_{role}" if pass_name == "hires" else role,
                        steps=int(steps),
                        hires_steps=hires_steps,
                        parser_options=options,
                        seed=seed,
                    )
                except PromptParserError as exc:
                    messages.append(_message(
                        "blocking_error",
                        exc.error_kind,
                        str(exc),
                        pass_name=pass_name,
                        prompt_role=role,
                        diagnostics=exc.to_dict(),
                    ))
                    validation = {"valid": False, "error": exc.to_dict()}
            for warning in validation.get("warnings") or []:
                messages.append(_message(
                    "behavior_warning",
                    "prompt_parser_validation_warning",
                    str(warning),
                    pass_name=pass_name,
                    prompt_role=role,
                ))
            route_plan = dict(validation.get("route_plan") or {})
            for ambiguity in route_plan.get("ambiguities") or []:
                messages.append(_message(
                    "behavior_warning",
                    "ambiguous_prompt_route",
                    str(ambiguity.get("message") or "The combined dispatcher found multiple valid routes."),
                    pass_name=pass_name,
                    prompt_role=role,
                    route_plan=route_plan,
                ))
            shadow = None
            if shadow_compare:
                shadow = shadow_compare_parsers(
                    raw_prompt=translated.parser_input,
                    prompt_role=f"hires_{role}" if pass_name == "hires" else role,
                    steps=int(steps),
                    hires_steps=hires_steps,
                    seed=seed,
                    legacy_adapter=LegacyPromptParserAdapter(),
                    parser21_adapter=Parser21PromptParserAdapter(),
                    parser21_options=(options if parser_id == "parser21" else {
                        "use_visitor": bool(options.get("parser21_use_visitor", True)),
                        "use_old_scheduling": bool(options.get("parser21_use_old_scheduling", False)),
                        **({"seed": options.get("parser21_seed")} if options.get("parser21_seed") not in (None, "") else {}),
                    }),
                )
                classification = str(shadow.get("classification") or "")
                if classification in {"Compatible but different", "Parser-specific", "Unsupported", "Bug candidate"}:
                    messages.append(_message(
                        "informational_notice",
                        "prompt_parser_shadow_difference",
                        f"Parser shadow comparison classified this prompt as {classification}.",
                        pass_name=pass_name,
                        prompt_role=role,
                        shadow_comparison=shadow,
                    ))
            canonical_started = time.perf_counter()
            canonical_prompt, canonical_structure, canonical_warnings = canonicalize_prompt(
                translated.parser_input,
                parser_id=parser_id,
            )
            canonicalization_duration_ms = round((time.perf_counter() - canonical_started) * 1000.0, 3)
            for warning in canonical_warnings:
                messages.append(_message(
                    "behavior_warning",
                    "canonicalization_warning",
                    str(warning),
                    pass_name=pass_name,
                    prompt_role=role,
                ))
            roles[role] = {
                **translated.metadata(),
                "parser_canonical_prompt": canonical_prompt,
                "parser_canonical_structure": canonical_structure,
                "parser_validation": validation,
                "route_plan": route_plan,
                "shadow_comparison": shadow,
                "performance": {
                    "shortcut_translation_duration_ms": translated.diagnostics.get("translation_duration_ms"),
                    "canonicalization_duration_ms": canonicalization_duration_ms,
                    "route_planning_duration_ms": route_plan.get("planner_duration_ms"),
                    "parser_validation_duration_ms": validation.get("validation_duration_ms"),
                    "validation_cache": validation.get("validation_cache", "not_available"),
                    "canonical_structure_size": len(canonical_prompt),
                    "branch_count": validation.get("branch_count", 0),
                },
            }

        return {
            "pass": pass_name,
            "parser": descriptor,
            "parser_options": options,
            "shortcut_profile": profile.to_dict(parser_id=parser_id),
            "shortcut_profile_snapshot": profile.snapshot(),
            "positive": roles["positive"],
            "negative": roles["negative"],
        }, messages

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = dict(payload or {})
        steps = int(source.get("steps") or 20)
        hires_steps_raw = source.get("hires_steps")
        hires_steps = int(hires_steps_raw) if hires_steps_raw not in (None, "") else None
        seed_raw = source.get("seed")
        seed = int(seed_raw) if seed_raw not in (None, "") else None
        shadow_compare = bool(source.get("prompt_shadow_compare", False))

        base_parser = source.get("prompt_parser_name") or source.get("base_prompt_parser_name") or "legacy"
        base_profile = source.get("prompt_shortcut_profile_name") or source.get("base_shortcut_profile_name")
        base_options = dict(source.get("prompt_parser_kwargs") or {}) if isinstance(source.get("prompt_parser_kwargs"), Mapping) else {}
        legacy_options = dict(source.get("parser_kwargs") or {}) if isinstance(source.get("parser_kwargs"), Mapping) else {}
        for compatibility_key in ("prompt_parser", "prompt_parser_name", "hires_steps"):
            legacy_options.pop(compatibility_key, None)
        for key, value in legacy_options.items():
            base_options.setdefault(key, value)
        base, messages = self._validate_pass(
            pass_name="base",
            parser_name=base_parser,
            parser_options=base_options,
            profile_name=base_profile,
            profile_snapshot=source.get("prompt_shortcut_profile_snapshot") if isinstance(source.get("prompt_shortcut_profile_snapshot"), Mapping) else None,
            positive_prompt=str(source.get("positive_prompt") or ""),
            negative_prompt=str(source.get("negative_prompt") or ""),
            steps=steps,
            hires_steps=hires_steps,
            seed=seed,
            shadow_compare=shadow_compare,
        )

        parser_mode = _normalize_mode(source.get("hires_prompt_parser_mode"), field_name="hires_prompt_parser_mode")
        profile_mode = _normalize_mode(source.get("hires_shortcut_profile_mode"), field_name="hires_shortcut_profile_mode")
        if parser_mode == "same_as_base":
            hires_parser = base["parser"]["parser_id"]
            hires_options = dict(base["parser_options"])
        elif parser_mode == "canonical_only":
            hires_parser = base["parser"]["parser_id"]
            hires_options = dict(source.get("hires_prompt_parser_kwargs") or base["parser_options"])
        else:
            hires_parser = source.get("hires_prompt_parser_name")
            if not hires_parser:
                raise ValueError("hires_prompt_parser_name is required when hires_prompt_parser_mode is explicit.")
            hires_options = dict(source.get("hires_prompt_parser_kwargs") or {})

        if profile_mode == "same_as_base":
            hires_profile = base["shortcut_profile"]["profile_id"]
            hires_snapshot = base["shortcut_profile_snapshot"]
        elif profile_mode == "canonical_only" or parser_mode == "canonical_only":
            hires_profile = "canonical"
            hires_snapshot = None
        else:
            hires_profile = source.get("hires_shortcut_profile_name")
            if not hires_profile:
                raise ValueError("hires_shortcut_profile_name is required when hires_shortcut_profile_mode is explicit.")
            hires_snapshot = source.get("hires_shortcut_profile_snapshot") if isinstance(source.get("hires_shortcut_profile_snapshot"), Mapping) else None

        hires_positive = str(source.get("hires_positive_prompt") or source.get("positive_prompt") or "")
        hires_negative = str(source.get("hires_negative_prompt") or source.get("negative_prompt") or "")
        hires, hires_messages = self._validate_pass(
            pass_name="hires",
            parser_name=hires_parser,
            parser_options=hires_options,
            profile_name=hires_profile,
            profile_snapshot=hires_snapshot,
            positive_prompt=hires_positive,
            negative_prompt=hires_negative,
            steps=steps,
            hires_steps=hires_steps or steps,
            seed=seed,
            shadow_compare=shadow_compare,
        )
        messages.extend(hires_messages)

        interpretation_diff = {
            role: {
                "different": any((
                    base["parser"]["parser_id"] != hires["parser"]["parser_id"],
                    base["shortcut_profile"]["profile_id"] != hires["shortcut_profile"]["profile_id"],
                    base[role]["parser_input"] != hires[role]["parser_input"],
                    base[role]["parser_canonical_prompt"] != hires[role]["parser_canonical_prompt"],
                )),
                "base_parser_input": base[role]["parser_input"],
                "hires_parser_input": hires[role]["parser_input"],
                "base_canonical_prompt": base[role]["parser_canonical_prompt"],
                "hires_canonical_prompt": hires[role]["parser_canonical_prompt"],
            }
            for role in ("positive", "negative")
        }
        if any(item["different"] for item in interpretation_diff.values()):
            messages.append(_message(
                "behavior_warning",
                "hires_prompt_interpretation_differs",
                "The hires pass will interpret at least one prompt differently from the base pass.",
            ))
        elif parser_mode == "same_as_base" and profile_mode == "same_as_base":
            messages.append(_message(
                "informational_notice",
                "hires_inherits_base_prompt_processing",
                "The hires pass inherits the base parser, shortcut profile, and parser settings.",
            ))

        grouped = {
            "blocking_errors": [item for item in messages if item["severity"] == "blocking_error"],
            "behavior_warnings": [item for item in messages if item["severity"] == "behavior_warning"],
            "informational_notices": [item for item in messages if item["severity"] == "informational_notice"],
        }
        normalized_fields = {
            "prompt_parser_name": base["parser"]["parser_id"],
            "base_prompt_parser_name": base["parser"]["parser_id"],
            "prompt_parser_kwargs": base["parser_options"],
            "prompt_shortcut_profile_name": base["shortcut_profile"]["profile_id"],
            "base_shortcut_profile_name": base["shortcut_profile"]["profile_id"],
            "prompt_shortcut_profile_snapshot": base["shortcut_profile_snapshot"],
            "hires_prompt_parser_mode": parser_mode,
            "hires_prompt_parser_name": hires["parser"]["parser_id"],
            "hires_prompt_parser_kwargs": hires["parser_options"],
            "hires_shortcut_profile_mode": profile_mode,
            "hires_shortcut_profile_name": hires["shortcut_profile"]["profile_id"],
            "hires_shortcut_profile_snapshot": hires["shortcut_profile_snapshot"],
            "hires_positive_prompt": hires_positive,
            "hires_negative_prompt": hires_negative,
            "prompt_shadow_compare": shadow_compare,
            "prompt_route_plan": {
                "positive": dict(base["positive"].get("route_plan") or {}),
                "negative": dict(base["negative"].get("route_plan") or {}),
            },
            "hires_prompt_route_plan": {
                "positive": dict(hires["positive"].get("route_plan") or {}),
                "negative": dict(hires["negative"].get("route_plan") or {}),
            },
        }
        return {
            "contract_version": PROMPT_PREFLIGHT_CONTRACT_VERSION,
            "valid": not grouped["blocking_errors"],
            "messages": messages,
            **grouped,
            "base": base,
            "hires": {
                **hires,
                "parser_mode": parser_mode,
                "shortcut_profile_mode": profile_mode,
                "interpretation_diff": interpretation_diff,
            },
            "normalized_fields": normalized_fields,
        }


def apply_prompt_preflight(payload: Mapping[str, Any], *, preflight: PromptProcessingPreflight | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    validator = preflight or PromptProcessingPreflight()
    report = validator.validate(payload)
    if not report["valid"]:
        messages = " | ".join(item["message"] for item in report["blocking_errors"])
        raise ValueError(f"Prompt preflight failed: {messages}")
    normalized = dict(payload or {})
    normalized.update(copy.deepcopy(report["normalized_fields"]))
    normalized["prompt_preflight"] = copy.deepcopy(report)
    return normalized, report
