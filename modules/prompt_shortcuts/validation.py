from __future__ import annotations

import re
from typing import Any

from modules.prompt_shortcuts.contracts import (
    CANONICAL_OPERATOR_TOKENS,
    DEFAULT_PROFILE_PRECEDENCE,
    KNOWN_PRECEDENCE_STAGES,
    KNOWN_SEMANTIC_MODE_KEYS,
    PromptShortcutProfileDescriptor,
    PromptShortcutValidationIssue,
    PromptShortcutValidationResult,
)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def validate_prompt_shortcut_profile(profile_or_payload: PromptShortcutProfileDescriptor | dict[str, Any]) -> PromptShortcutValidationResult:
    try:
        profile = profile_or_payload if isinstance(profile_or_payload, PromptShortcutProfileDescriptor) else PromptShortcutProfileDescriptor.from_dict(profile_or_payload)
    except Exception as exc:
        issue = PromptShortcutValidationIssue("error", "invalid_profile", f"Unable to read shortcut profile: {exc}")
        return PromptShortcutValidationResult(valid=False, issues=[issue])

    issues: list[PromptShortcutValidationIssue] = []
    if not profile.profile_id:
        issues.append(PromptShortcutValidationIssue("error", "missing_profile_id", "Shortcut profile ID is required."))
    if not profile.label:
        issues.append(PromptShortcutValidationIssue("error", "missing_label", "Shortcut profile label is required."))
    if len(profile.escape_character) != 1:
        issues.append(PromptShortcutValidationIssue("error", "invalid_escape_character", "Escape character must be exactly one character."))

    for key, value in profile.semantic_modes.items():
        if not str(key).strip():
            issues.append(PromptShortcutValidationIssue("error", "empty_semantic_mode_key", "Semantic mode keys cannot be empty."))
        elif key not in KNOWN_SEMANTIC_MODE_KEYS:
            issues.append(PromptShortcutValidationIssue(
                "warning",
                "unknown_semantic_mode",
                f"Semantic mode {key!r} is not part of the current prompt-style contract and will be preserved as an extension.",
            ))
        if not str(value).strip():
            issues.append(PromptShortcutValidationIssue("error", "empty_semantic_mode", f"Semantic mode {key!r} must define a non-empty algorithm ID."))

    seen_precedence: set[str] = set()
    for stage in profile.precedence:
        token = str(stage or "").strip().upper()
        if token in seen_precedence:
            issues.append(PromptShortcutValidationIssue(
                "error",
                "duplicate_precedence_stage",
                f"Precedence stage {token!r} is listed more than once.",
            ))
        seen_precedence.add(token)
        if token not in KNOWN_PRECEDENCE_STAGES:
            issues.append(PromptShortcutValidationIssue(
                "warning",
                "unknown_precedence_stage",
                f"Precedence stage {token!r} is not part of the current canonical precedence contract.",
            ))
    if not profile.precedence:
        issues.append(PromptShortcutValidationIssue("error", "missing_precedence", "Prompt style profiles must define a precedence contract."))
    elif not {"ESCAPES", "TEXT"}.issubset(seen_precedence):
        issues.append(PromptShortcutValidationIssue(
            "error",
            "incomplete_precedence",
            "Prompt style precedence must contain at least ESCAPES and TEXT.",
        ))

    reserved_seen: set[str] = set()
    for token in profile.reserved_syntax:
        if token in reserved_seen:
            issues.append(PromptShortcutValidationIssue(
                "warning",
                "duplicate_reserved_syntax",
                f"Reserved syntax {token!r} is listed more than once.",
                alias=token,
                collision_kind="safe_alias_overlap",
            ))
        reserved_seen.add(token)
        if _CONTROL_RE.search(token):
            issues.append(PromptShortcutValidationIssue(
                "error",
                "reserved_control_character",
                f"Reserved syntax {token!r} contains a control character.",
                alias=token,
                collision_kind="hard_collision",
            ))

    alias_owners: dict[str, str] = {}
    for operator, aliases in profile.aliases.items():
        if operator not in CANONICAL_OPERATOR_TOKENS:
            issues.append(PromptShortcutValidationIssue(
                "warning",
                "unknown_operator",
                f"Operator {operator!r} is not in the current parser-emission vocabulary.",
                operator=operator,
            ))
        if not aliases:
            issues.append(PromptShortcutValidationIssue("error", "empty_operator_aliases", f"Operator {operator!r} has no aliases.", operator=operator))
        for alias in aliases:
            if alias == "":
                issues.append(PromptShortcutValidationIssue("error", "empty_alias", "Aliases cannot be empty.", operator=operator))
                continue
            if _CONTROL_RE.search(alias):
                issues.append(PromptShortcutValidationIssue("error", "control_character", f"Alias {alias!r} contains a control character.", operator=operator, alias=alias, collision_kind="hard_collision"))
            if any(0xE000 <= ord(char) <= 0xF8FF for char in alias):
                issues.append(PromptShortcutValidationIssue("error", "private_use_character", f"Alias {alias!r} uses a private-use Unicode character reserved for parser internals.", operator=operator, alias=alias, collision_kind="hard_collision"))
            if alias == profile.escape_character:
                issues.append(PromptShortcutValidationIssue("error", "escape_collision", f"Alias {alias!r} conflicts with the escape character.", operator=operator, alias=alias, collision_kind="hard_collision"))
            if "parser21" in profile.compatible_parsers and alias == "=>":
                issues.append(PromptShortcutValidationIssue(
                    "error",
                    "parser_reserved_syntax_collision",
                    "Alias '=>' is reserved inside Parser 21 BIND/MORPH structures and cannot be globally remapped.",
                    operator=operator,
                    alias=alias,
                    collision_kind="reserved_foreign_syntax_collision",
                ))
            if alias in reserved_seen:
                severity = "warning" if profile.builtin else "error"
                issues.append(PromptShortcutValidationIssue(
                    severity,
                    "reserved_foreign_syntax_collision",
                    f"Alias {alias!r} claims syntax reserved by this prompt-style profile. Built-in profiles may claim their own native syntax; user profiles must choose a non-reserved alias.",
                    operator=operator,
                    alias=alias,
                    collision_kind="reserved_foreign_syntax_collision",
                ))
            owner = alias_owners.get(alias)
            if owner and owner != operator:
                issues.append(PromptShortcutValidationIssue("error", "duplicate_alias", f"Alias {alias!r} is mapped to both {owner} and {operator}.", operator=operator, alias=alias, collision_kind="hard_collision"))
            alias_owners[alias] = operator

    unique_aliases = sorted(set(alias_owners), key=lambda value: (len(value), value))
    for index, shorter in enumerate(unique_aliases):
        for longer in unique_aliases[index + 1:]:
            if longer.startswith(shorter) and alias_owners[longer] != alias_owners[shorter]:
                issues.append(PromptShortcutValidationIssue(
                    "warning",
                    "ambiguous_prefix",
                    f"Alias {shorter!r} prefixes {longer!r}; translation uses deterministic longest-match ordering.",
                    operator=alias_owners[shorter],
                    alias=shorter,
                    collision_kind="prefix_ambiguity",
                ))

    for parser in profile.compatible_parsers:
        emitters = profile.parser_emitters.get(parser)
        if emitters is None:
            issues.append(PromptShortcutValidationIssue("error", "missing_parser_emitters", f"Profile declares compatibility with {parser!r} but has no emitter map for it."))
            continue
        for operator in profile.aliases:
            if operator not in emitters:
                issues.append(PromptShortcutValidationIssue(
                    "warning",
                    "operator_not_supported_by_parser",
                    f"{parser!r} has no emitter for {operator}; prompts using that shortcut will be rejected before generation.",
                    operator=operator,
                ))

    # A custom profile may intentionally replace the default precedence, but
    # warn if it omits stages that are part of the canonical architecture.
    missing_stages = [stage for stage in DEFAULT_PROFILE_PRECEDENCE if stage not in seen_precedence]
    if missing_stages and profile.precedence:
        issues.append(PromptShortcutValidationIssue(
            "warning",
            "noncanonical_precedence",
            "Prompt style precedence omits canonical stages: " + ", ".join(missing_stages),
        ))

    valid = not any(issue.severity == "error" for issue in issues)
    return PromptShortcutValidationResult(valid=valid, issues=issues, mapping_hash=profile.mapping_hash if valid else "")
