from __future__ import annotations

import re
from typing import Any

from modules.prompt_shortcuts.contracts import (
    CANONICAL_OPERATOR_TOKENS,
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

    alias_owners: dict[str, str] = {}
    all_aliases: list[tuple[str, str]] = []
    for operator, aliases in profile.aliases.items():
        if operator not in CANONICAL_OPERATOR_TOKENS:
            issues.append(PromptShortcutValidationIssue("warning", "unknown_operator", f"Operator {operator!r} is not in the Phase 13D canonical vocabulary.", operator=operator))
        if not aliases:
            issues.append(PromptShortcutValidationIssue("error", "empty_operator_aliases", f"Operator {operator!r} has no aliases.", operator=operator))
        for alias in aliases:
            if alias == "":
                issues.append(PromptShortcutValidationIssue("error", "empty_alias", "Aliases cannot be empty.", operator=operator))
                continue
            if _CONTROL_RE.search(alias):
                issues.append(PromptShortcutValidationIssue("error", "control_character", f"Alias {alias!r} contains a control character.", operator=operator, alias=alias))
            if any(0xE000 <= ord(char) <= 0xF8FF for char in alias):
                issues.append(PromptShortcutValidationIssue("error", "private_use_character", f"Alias {alias!r} uses a private-use Unicode character reserved for parser internals.", operator=operator, alias=alias))
            if alias == profile.escape_character:
                issues.append(PromptShortcutValidationIssue("error", "escape_collision", f"Alias {alias!r} conflicts with the escape character.", operator=operator, alias=alias))
            if "parser21" in profile.compatible_parsers and alias == "=>":
                issues.append(PromptShortcutValidationIssue(
                    "error",
                    "parser_reserved_syntax_collision",
                    "Alias '=>' is reserved inside Parser 21 BIND/MORPH structures and cannot be globally remapped.",
                    operator=operator,
                    alias=alias,
                ))
            owner = alias_owners.get(alias)
            if owner and owner != operator:
                issues.append(PromptShortcutValidationIssue("error", "duplicate_alias", f"Alias {alias!r} is mapped to both {owner} and {operator}.", operator=operator, alias=alias))
            alias_owners[alias] = operator
            all_aliases.append((alias, operator))

    unique_aliases = sorted(set(alias_owners), key=lambda value: (len(value), value))
    for index, shorter in enumerate(unique_aliases):
        for longer in unique_aliases[index + 1:]:
            if longer.startswith(shorter) and alias_owners[longer] != alias_owners[shorter]:
                issues.append(PromptShortcutValidationIssue(
                    "warning",
                    "ambiguous_prefix",
                    f"Alias {shorter!r} prefixes {longer!r}; translation uses longest-match ordering.",
                    operator=alias_owners[shorter],
                    alias=shorter,
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

    valid = not any(issue.severity == "error" for issue in issues)
    return PromptShortcutValidationResult(valid=valid, issues=issues, mapping_hash=profile.mapping_hash if valid else "")
