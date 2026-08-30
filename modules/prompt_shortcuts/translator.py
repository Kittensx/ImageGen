from __future__ import annotations

import re
import time
from typing import Any

from modules.prompt_shortcuts.contracts import (
    CANONICAL_OPERATOR_TOKENS,
    PROMPT_SHORTCUT_CONTRACT_VERSION,
    PromptShortcutError,
    ordered_profile_alias_entries,
    PromptShortcutProfileDescriptor,
    PromptTranslationResult,
)
from modules.prompt_shortcuts.validation import validate_prompt_shortcut_profile


def _is_word_alias(value: str) -> bool:
    return bool(value) and (value[0].isalnum() or value[0] == "_") and (value[-1].isalnum() or value[-1] == "_")


def _word_boundary_ok(text: str, start: int, end: int, alias: str) -> bool:
    if not _is_word_alias(alias):
        return True
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


class PromptShortcutTranslator:
    def translate(
        self,
        raw_prompt: str,
        *,
        profile: PromptShortcutProfileDescriptor,
        parser_id: str,
        prompt_role: str = "positive",
    ) -> PromptTranslationResult:
        started = time.perf_counter()
        parser = str(parser_id or "legacy").strip().lower()
        validation = validate_prompt_shortcut_profile(profile)
        if not validation.valid:
            raise PromptShortcutError(
                f"Shortcut profile {profile.profile_id!r} is invalid.",
                error_kind="invalid_shortcut_profile",
                diagnostics=validation.to_dict(),
            )
        compatible = parser in profile.compatible_parsers or (
            parser == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21"))
        )
        if not compatible:
            raise PromptShortcutError(
                f"Shortcut profile {profile.profile_id!r} is not compatible with prompt parser {parser!r}.",
                error_kind="shortcut_profile_parser_incompatible",
                diagnostics={"profile_id": profile.profile_id, "parser_id": parser},
            )
        emitters = (
            dict(CANONICAL_OPERATOR_TOKENS)
            if parser == "combined"
            else dict(profile.parser_emitters.get(parser) or {})
        )
        alias_entries = ordered_profile_alias_entries(profile.aliases)

        raw = str(raw_prompt or "")
        parser_parts: list[str] = []
        canonical_parts: list[str] = []
        substitutions: list[dict[str, Any]] = []
        index = 0
        escape = profile.escape_character
        while index < len(raw):
            if escape and raw.startswith(escape, index) and index + len(escape) < len(raw):
                escaped_start = index + len(escape)
                if (
                    profile.semantic_mode("attention_algorithm") == "a1111_attention_v1"
                    and raw[escaped_start] in "()[]\\"
                ):
                    # A1111 attention owns these escapes. Preserve the backslash
                    # through shortcut translation so the attention/schedule
                    # compilers can distinguish literal punctuation from syntax.
                    parser_parts.append(raw[index : escaped_start + 1])
                    canonical_parts.append(raw[index : escaped_start + 1])
                    index = escaped_start + 1
                    continue
                matched_escaped = None
                for alias, operator in alias_entries:
                    if raw.startswith(alias, escaped_start) and _word_boundary_ok(raw, escaped_start, escaped_start + len(alias), alias):
                        matched_escaped = (alias, operator)
                        break
                if matched_escaped:
                    alias, _operator = matched_escaped
                    # Preserve the escape in parser input so backend grammars and
                    # combined capability analysis cannot reinterpret the literal
                    # alias as an operator after translation.
                    parser_parts.append(f"{escape}{alias}")
                    canonical_parts.append(alias)
                    index = escaped_start + len(alias)
                    continue
                parser_parts.append(raw[escaped_start])
                canonical_parts.append(raw[escaped_start])
                index = escaped_start + 1
                continue

            match: tuple[str, str] | None = None
            for alias, operator in alias_entries:
                end = index + len(alias)
                if raw.startswith(alias, index) and _word_boundary_ok(raw, index, end, alias):
                    match = (alias, operator)
                    break
            if match is None:
                parser_parts.append(raw[index])
                canonical_parts.append(raw[index])
                index += 1
                continue

            alias, operator = match
            emitted = emitters.get(operator)
            if emitted is None:
                raise PromptShortcutError(
                    f"Shortcut {alias!r} maps to {operator}, which is not supported by parser {parser!r}.",
                    error_kind="shortcut_operator_unsupported",
                    diagnostics={
                        "profile_id": profile.profile_id,
                        "parser_id": parser,
                        "operator": operator,
                        "alias": alias,
                        "position": index,
                        "prompt_role": prompt_role,
                    },
                )
            canonical_token = CANONICAL_OPERATOR_TOKENS.get(operator, operator)
            parser_parts.append(emitted)
            canonical_parts.append(canonical_token)
            substitutions.append({
                "source": alias,
                "canonical_operator": operator,
                "canonical_token": canonical_token,
                "semantic_operator_id": profile.semantic_operator_id(operator),
                "semantic_algorithm": profile.semantic_algorithm(operator),
                "parser_emission": emitted,
                "start": index,
                "end": index + len(alias),
                "profile_id": profile.profile_id,
            })
            index += len(alias)

        parser_input = "".join(parser_parts)
        canonical_source = "".join(canonical_parts)
        structure = {
            "contract": PROMPT_SHORTCUT_CONTRACT_VERSION,
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "profile_schema_version": profile.profile_schema_version,
            "mapping_hash": profile.mapping_hash,
            "semantic_modes": dict(profile.semantic_modes),
            "preprocessing": dict(profile.preprocessing),
            "precedence": list(profile.precedence),
            "reserved_syntax": list(profile.reserved_syntax),
            "parser_id": parser,
            "prompt_role": prompt_role,
            "lossless_raw_source": raw,
            "parser_input": parser_input,
            "canonical_source": canonical_source,
            "substitutions": substitutions,
        }
        warning_messages = [issue.message for issue in validation.warnings]
        return PromptTranslationResult(
            raw_prompt=raw,
            parser_input=parser_input,
            canonical_prompt=canonical_source,
            canonical_structure=structure,
            substitutions=substitutions,
            warnings=warning_messages,
            diagnostics={
                "translation_duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "profile_id": profile.profile_id,
                "profile_version": profile.version,
                "profile_schema_version": profile.profile_schema_version,
                "mapping_hash": profile.mapping_hash,
                "semantic_modes": dict(profile.semantic_modes),
                "preprocessing": dict(profile.preprocessing),
                "precedence": list(profile.precedence),
                "reserved_syntax": list(profile.reserved_syntax),
                "semantic_operators": [
                    {
                        "source": item.get("source", ""),
                        "canonical_operator": item.get("canonical_operator", ""),
                        "semantic_operator_id": item.get("semantic_operator_id", ""),
                        "semantic_algorithm": item.get("semantic_algorithm", ""),
                    }
                    for item in substitutions
                ],
                "parser_id": parser,
                "prompt_role": prompt_role,
                "raw_length": len(raw),
                "parser_input_length": len(parser_input),
                "canonical_length": len(canonical_source),
                "substitution_count": len(substitutions),
                "warning_count": len(warning_messages),
            },
        )
