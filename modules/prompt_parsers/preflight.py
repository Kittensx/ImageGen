from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Mapping

from image_gen.systems.prompt_expansion import PromptExpansionError, expand_superhybrid_prompt_batch
from image_gen.systems.regional_prompting import (
    RegionalPromptError,
    estimate_region_runtime,
    extract_superhybrid_region_slot,
)
from modules.txt2img.seed_utils import parse_seed_range_expression, resolve_seed_sequence
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
    effective_seed = seed if parser_id in {"parser21", "combined", "superhybrid"} else None
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
            elif parser_id == "superhybrid":
                fallback = "superhybrid_native"
            else:
                fallback = "canonical"
            profile = self.profile_registry.get(profile_name or fallback)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
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
        batch_size: int,
        width: int,
        height: int,
        coordinate_reference_slots: list[Mapping[str, Any]] | None = None,
        coordinate_reference_width: int | None = None,
        coordinate_reference_height: int | None = None,
        prompt_expansion_recorded: Mapping[str, Any] | None = None,
        prompt_expansion_replay_mode: str = "reconstruct",
        defer_prompt_expansion: bool = False,
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

        expansion_record: dict[str, Any] = {}
        expansion_error: dict[str, Any] | None = None
        expanded_positive_prompt = str(positive_prompt or "")
        expanded_negative_prompt = str(negative_prompt or "")
        expanded_positive_slots = [expanded_positive_prompt for _ in range(max(1, int(batch_size)))]
        expanded_negative_slots = [expanded_negative_prompt for _ in range(max(1, int(batch_size)))]
        image_seeds = resolve_seed_sequence(seed, max(1, int(batch_size)))
        parser_slot_seeds = list(image_seeds)
        if parser_id == "superhybrid":
            if defer_prompt_expansion:
                messages.append(_message(
                    "informational_notice",
                    "disabled_hires_prompt_expansion_deferred",
                    "The disabled hires pass has no recorded prompt expansion, so its SuperHybrid expansion preview was deferred.",
                    pass_name=pass_name,
                ))
            else:
                parser_seed = options.get("seed")
                if parser_seed in (None, ""):
                    parser_seed = seed
                try:
                    recorded_expansions = dict(prompt_expansion_recorded or {})
                    if parser_seed in (None, "") or int(parser_seed) < 0:
                        selection_seeds = list(image_seeds)
                    else:
                        selection_seeds = resolve_seed_sequence(int(parser_seed), max(1, int(batch_size)))
                    expansion_scope = str(options.get("prompt_expansion_scope", "per_batch") or "per_batch")
                    parser_slot_seeds = (
                        list(selection_seeds)
                        if expansion_scope == "per_image"
                        else [int(selection_seeds[0])] * len(image_seeds)
                    )
                    expansion_record = expand_superhybrid_prompt_batch(
                        expanded_positive_prompt,
                        expanded_negative_prompt,
                        resolved_seeds=image_seeds,
                        selection_seeds=selection_seeds,
                        pass_name=pass_name,
                        parser_version=str(descriptor.get("version") or ""),
                        scope=expansion_scope,
                        wildcard_directory=str(options.get("wildcard_directory", "wildcards") or "wildcards"),
                        recorded=dict(recorded_expansions.get(pass_name) or {}),
                        replay_mode=prompt_expansion_replay_mode,
                    )
                    expanded_positive_slots = [str(value or "") for value in list(expansion_record.get("expanded_positive_by_slot") or [])]
                    expanded_negative_slots = [str(value or "") for value in list(expansion_record.get("expanded_negative_by_slot") or [])]
                    expanded_positive_prompt = expanded_positive_slots[0]
                    expanded_negative_prompt = expanded_negative_slots[0]
                except PromptExpansionError as exc:
                    expansion_error = {
                        "error_kind": "superhybrid_prompt_expansion_failed",
                        "message": str(exc),
                        "pass": pass_name,
                    }
                    messages.append(_message(
                        "blocking_error",
                        "superhybrid_prompt_expansion_failed",
                        f"SuperHybrid prompt expansion failed: {exc}",
                        pass_name=pass_name,
                        diagnostics=expansion_error,
                    ))
        elif str(prompt_expansion_replay_mode or "reconstruct").strip().lower() == "recorded_exact":
            recorded_for_pass = (
                dict(prompt_expansion_recorded.get(pass_name) or {})
                if isinstance(prompt_expansion_recorded, Mapping)
                else {}
            )
            if recorded_for_pass:
                expansion_error = {
                    "error_kind": "prompt_expansion_parser_mismatch",
                    "message": "Recorded SuperHybrid prompt expansion cannot be applied to a different prompt parser.",
                    "pass": pass_name,
                }
                messages.append(_message(
                    "blocking_error",
                    "prompt_expansion_parser_mismatch",
                    expansion_error["message"],
                    pass_name=pass_name,
                    diagnostics=expansion_error,
                ))

        region_slots: list[dict[str, Any]] = []
        region_error: RegionalPromptError | None = None
        if not defer_prompt_expansion and expansion_error is None:
            try:
                base_prompts: list[str] = []
                reference_slots = [
                    dict(item or {}) for item in list(coordinate_reference_slots or [])
                ]
                for slot_index, prompt in enumerate(expanded_positive_slots):
                    coordinate_reference_slot = None
                    if slot_index < len(reference_slots):
                        candidate = reference_slots[slot_index]
                        if str(candidate.get("source_prompt") or "") == str(prompt or ""):
                            coordinate_reference_slot = candidate
                    base_prompt, _runtime_specs, region_slot = extract_superhybrid_region_slot(
                        prompt,
                        slot_index=slot_index,
                        steps=int(steps),
                        seed=int(parser_slot_seeds[slot_index]),
                        width=int(width),
                        height=int(height),
                        coordinate_reference_slot=coordinate_reference_slot,
                        coordinate_reference_width=coordinate_reference_width if coordinate_reference_slot else None,
                        coordinate_reference_height=coordinate_reference_height if coordinate_reference_slot else None,
                    )
                    base_prompts.append(base_prompt)
                    region_slots.append(region_slot)
                expanded_positive_slots = base_prompts
                expanded_positive_prompt = expanded_positive_slots[0]
            except RegionalPromptError as exc:
                region_error = exc
                messages.append(_message(
                    "blocking_error",
                    "region_plan_failed",
                    f"REGION planning failed: {exc}",
                    pass_name=pass_name,
                ))

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
        prompt_pairs = (("positive", expanded_positive_prompt), ("negative", expanded_negative_prompt))
        for role, raw_prompt in prompt_pairs:
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
            if defer_prompt_expansion and parser_id == "superhybrid":
                validation = {
                    "valid": True,
                    "warnings": [],
                    "skipped": "disabled_hires_prompt_expansion_deferred",
                }
            elif role == "positive" and region_error is not None:
                validation = {
                    "valid": False,
                    "warnings": [],
                    "skipped": "region_plan_failed",
                    "error": {
                        "error_kind": "region_plan_failed",
                        "message": str(region_error),
                    },
                }
            validator = getattr(adapter, "validate_syntax", None)
            if (
                callable(validator)
                and not defer_prompt_expansion
                and expansion_error is None
                and not (role == "positive" and region_error is not None)
                and not any(item["severity"] == "blocking_error" for item in option_messages)
            ):
                try:
                    validation = _cached_syntax_validation(
                        adapter,
                        translated.parser_input,
                        prompt_role=f"hires_{role}" if pass_name == "hires" else role,
                        steps=int(steps),
                        hires_steps=hires_steps,
                        parser_options=(
                            {**options, "seed": parser_slot_seeds[0]}
                            if parser_id == "superhybrid" else options
                        ),
                        seed=(parser_slot_seeds[0] if parser_id == "superhybrid" else seed),
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
            if shadow_compare and parser_id != "superhybrid":
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
            slot_prompts = expanded_positive_slots if role == "positive" else expanded_negative_slots
            slot_seeds = list(expansion_record.get("resolved_seeds") or resolve_seed_sequence(seed, len(slot_prompts)))
            slot_previews = [{
                **translated.metadata(),
                "slot_index": 0,
                "parser_canonical_prompt": canonical_prompt,
                "parser_canonical_structure": canonical_structure,
                "parser_validation": validation,
            }]
            for slot_index, slot_prompt in enumerate(slot_prompts[1:], start=1):
                try:
                    slot_translation = self.translator.translate(
                        str(slot_prompt or ""),
                        profile=profile,
                        parser_id=parser_id,
                        prompt_role=f"{pass_name}_{role}" if pass_name == "hires" else role,
                    )
                    slot_validation = validator(
                        slot_translation.parser_input,
                        prompt_role=f"{pass_name}_{role}" if pass_name == "hires" else role,
                        steps=steps,
                        hires_steps=hires_steps,
                        parser_options=(
                            {**options, "seed": parser_slot_seeds[slot_index]}
                            if parser_id == "superhybrid" else options
                        ),
                        seed=(parser_slot_seeds[slot_index] if parser_id == "superhybrid" else slot_seeds[slot_index]),
                    ) if callable(validator) and expansion_error is None else {"valid": True}
                    slot_canonical, slot_structure, _ = canonicalize_prompt(
                        slot_translation.parser_input, parser_id=parser_id
                    )
                    slot_previews.append({
                        **slot_translation.metadata(),
                        "slot_index": slot_index,
                        "parser_canonical_prompt": slot_canonical,
                        "parser_canonical_structure": slot_structure,
                        "parser_validation": slot_validation,
                    })
                except Exception as exc:
                    messages.append(_message(
                        "blocking_error",
                        "per_image_prompt_slot_validation_failed",
                        f"Prompt slot {slot_index + 1} failed validation: {exc}",
                        pass_name=pass_name,
                        prompt_role=role,
                        slot_index=slot_index,
                    ))
                    slot_previews.append({"slot_index": slot_index, "raw_prompt": slot_prompt, "error": str(exc)})
            roles[role] = {
                **translated.metadata(),
                "parser_canonical_prompt": canonical_prompt,
                "parser_canonical_structure": canonical_structure,
                "parser_validation": validation,
                "route_plan": route_plan,
                "shadow_comparison": shadow,
                "slots": slot_previews,
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
            "prompt_expansion": dict(expansion_record),
            "prompt_expansion_replay_mode": str(prompt_expansion_replay_mode or "reconstruct"),
            "prompt_expansion_error": dict(expansion_error or {}),
            "prompt_expansion_deferred": bool(defer_prompt_expansion),
            "raw_prompts": {"positive": str(positive_prompt or ""), "negative": str(negative_prompt or "")},
            "expanded_prompts": {"positive": expanded_positive_prompt, "negative": expanded_negative_prompt},
            "expanded_prompts_by_slot": {
                "positive": list(expanded_positive_slots),
                "negative": list(expanded_negative_slots),
            },
            "prompt_expansion_scope": str(expansion_record.get("scope") or options.get("prompt_expansion_scope") or "per_batch"),
            "regional_prompting": {
                "backend": "image_gen_model_output",
                "overlap_policy": str(options.get("region_overlap_policy", "additive") or "additive"),
                "slots": region_slots,
                "region_count": sum(int(item.get("region_count", 0) or 0) for item in region_slots),
                "runtime_estimate": estimate_region_runtime(
                    width=int(width),
                    height=int(height),
                    steps=int(steps),
                    slots=region_slots,
                ),
            },
            "semantic_fingerprints_by_slot": {
                "positive": [dict(item.get("parser_validation", {}).get("diagnostics", {}).get("semantic_fingerprint") or item.get("parser_validation", {}).get("semantic_fingerprint") or {}) for item in roles["positive"].get("slots", [])],
                "negative": [dict(item.get("parser_validation", {}).get("diagnostics", {}).get("semantic_fingerprint") or item.get("parser_validation", {}).get("semantic_fingerprint") or {}) for item in roles["negative"].get("slots", [])],
            },
            "positive": roles["positive"],
            "negative": roles["negative"],
        }, messages

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = dict(payload or {})
        steps = int(source.get("steps") or 20)
        hires_steps_raw = source.get("hires_steps")
        hires_steps = int(hires_steps_raw) if hires_steps_raw not in (None, "") else None
        seed_raw = source.get("seed")
        if seed_raw in (None, ""):
            seed = None
        elif parse_seed_range_expression(seed_raw) is not None:
            # Prompt preflight needs a scalar seed only for parser syntax / preview
            # behavior. The actual finite-range selection is owned by the WebUI
            # generation seed plan, so preserve the random sentinel here instead
            # of coercing the range expression through int(...).
            seed = -1
        else:
            seed = int(seed_raw)
        shadow_compare = bool(source.get("prompt_shadow_compare", False))
        batch_size = max(1, int(source.get("batch_size") or 1))
        base_width = int(source.get("generation_width") or source.get("width") or 512)
        base_height = int(source.get("generation_height") or source.get("height") or 512)

        base_parser = source.get("prompt_parser_name") or source.get("base_prompt_parser_name") or "legacy"
        base_profile = source.get("prompt_shortcut_profile_name") or source.get("base_shortcut_profile_name")
        base_options = dict(source.get("prompt_parser_kwargs") or {}) if isinstance(source.get("prompt_parser_kwargs"), Mapping) else {}
        legacy_options = dict(source.get("parser_kwargs") or {}) if isinstance(source.get("parser_kwargs"), Mapping) else {}
        for compatibility_key in ("prompt_parser", "prompt_parser_name", "hires_steps"):
            legacy_options.pop(compatibility_key, None)
        for key, value in legacy_options.items():
            base_options.setdefault(key, value)
        prompt_expansion_recorded = (
            source.get("prompt_expansion_recorded")
            if isinstance(source.get("prompt_expansion_recorded"), Mapping)
            else None
        )
        prompt_expansion_replay_mode = str(
            source.get("prompt_expansion_replay_mode") or "reconstruct"
        ).strip().lower()

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
            batch_size=batch_size,
            width=base_width,
            height=base_height,
            prompt_expansion_recorded=prompt_expansion_recorded,
            prompt_expansion_replay_mode=prompt_expansion_replay_mode,
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
        recorded_hires_expansion = (
            dict(prompt_expansion_recorded.get("hires") or {})
            if isinstance(prompt_expansion_recorded, Mapping)
            else {}
        )
        defer_hires_prompt_expansion = bool(
            prompt_expansion_replay_mode == "recorded_exact"
            and not bool(source.get("hires_enabled", False))
            and not recorded_hires_expansion
        )
        hires_width = int(source.get("hires_width") or 0)
        hires_height = int(source.get("hires_height") or 0)
        if hires_width <= 0:
            hires_width = max(1, int(round(base_width * float(source.get("hires_scale") or 2.0))))
        if hires_height <= 0:
            hires_height = max(1, int(round(base_height * float(source.get("hires_scale") or 2.0))))
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
            batch_size=batch_size,
            width=hires_width,
            height=hires_height,
            coordinate_reference_slots=list(
                (base.get("regional_prompting") or {}).get("slots") or []
            ),
            coordinate_reference_width=base_width,
            coordinate_reference_height=base_height,
            prompt_expansion_recorded=prompt_expansion_recorded,
            prompt_expansion_replay_mode=prompt_expansion_replay_mode,
            defer_prompt_expansion=defer_hires_prompt_expansion,
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
