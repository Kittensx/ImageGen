from __future__ import annotations

"""PPSR Classic prompt semantic contract introspection.

This module is intentionally side-effect free and model-free.  It describes the
syntax that IMAGE_GEN treats as structural before the later PPSR phases wire the
structure into runtime conditioning.
"""

from dataclasses import dataclass
import re
from typing import Any

from modules.prompt_parsers.numeric_semantics import collect_numeric_semantics, numeric_label

from modules.parser.legacy_structured_prompt import (
    node_to_dict,
    normalize_legacy_structured_source,
    parse_legacy_node,
    parse_legacy_structured_prompt,
    split_top_level_and,
)

CLASSIC_SEMANTIC_CONTRACT_VERSION = "image-gen-classic-semantic-v1"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_SCHEDULE_RE = re.compile(
    rf"\[(?P<before>[^\[\]]*?):(?P<after>[^\[\]]*?):(?P<value>{_NUMBER})\]"
)
_ATTENTION_WEIGHT_RE = re.compile(rf"\((?P<body>[^()]*)\:(?P<value>{_NUMBER})\)")
_END_WEIGHT_RE = re.compile(rf"^(?P<body>.*?)(?<!\\):(?P<value>{_NUMBER})\s*$")


@dataclass(frozen=True)
class ClassicNumericToken:
    kind: str
    token: str
    value: float | int
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "token": self.token,
            "value": self.value,
            "context": self.context,
        }


def _is_escaped(source: str, index: int) -> bool:
    slash_count = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        slash_count += 1
        index -= 1
    return bool(slash_count % 2)


def _escaped_literals(source: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(source) - 1:
        if source[index] == "\\" and not _is_escaped(source, index):
            literal = source[index + 1]
            if literal in "{}:!\\":
                output.append(literal)
            index += 2
            continue
        index += 1
    return output


def _node_types(node: dict[str, Any]) -> list[str]:
    output: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            node_type = value.get("type")
            if isinstance(node_type, str):
                output.append(node_type)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return output


def _conjunction_node(source: str) -> dict[str, Any] | None:
    branches = [item.strip() for item in split_top_level_and(source) if item.strip()]
    if len(branches) <= 1:
        return None
    return {
        "type": "conjunction",
        "branches": [node_to_dict(parse_legacy_node(item)) for item in branches],
    }


def classic_semantic_nodes(source: str) -> list[dict[str, Any]]:
    """Return semantic nodes for the shared Classic subset.

    Existing canonical-v1 A1111 node shapes are preserved where possible. PPSR
    structural nodes replace plain-text fallback only when the structural parser
    actually recognizes a semantic construct.
    """

    normalized, _warnings = normalize_legacy_structured_source(source)
    conjunction = _conjunction_node(normalized)
    parsed = parse_legacy_structured_prompt(normalized)
    base_node = node_to_dict(parsed.node)

    if conjunction is not None:
        nodes: list[dict[str, Any]] = [conjunction]
    elif parsed.structured:
        nodes = [base_node]
    else:
        nodes = []

    if _SCHEDULE_RE.search(normalized):
        nodes.append({"type": "scheduled_text", "source": normalized})
    if re.search(r"\[[^\]|]+(?:\|[^\]]+)+\]", normalized):
        nodes.append({"type": "alternate_text", "source": normalized})
    if _ATTENTION_WEIGHT_RE.search(normalized):
        nodes.append({"type": "weighted_text", "source": normalized})
    elif ("(" in normalized or "[" in normalized) and not _SCHEDULE_RE.search(normalized):
        nodes.append({"type": "attention_group", "source": normalized})

    return nodes or [base_node]


def classify_classic_numeric_contexts(source: str) -> list[dict[str, Any]]:
    """Classify numeric tokens by the grammar construct that owns them.

    PPSR-01 records the meaning; it does not change runtime lowering.  The
    ambiguous legacy single-colon suffix remains explicitly named as a legacy
    compatibility context so PPSR-05 can replace the integer/decimal heuristic.
    """

    normalized, _warnings = normalize_legacy_structured_source(source)
    output: list[ClassicNumericToken] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in _SCHEDULE_RE.finditer(normalized):
        token = match.group("value")
        value = float(token)
        if "." in token:
            kind = "schedule_fraction"
            typed_value: float | int = value
        else:
            kind = "schedule_step"
            typed_value = int(token)
        output.append(ClassicNumericToken(kind, token, typed_value, "a1111_schedule"))
        consumed_spans.append(match.span("value"))

    for match in _ATTENTION_WEIGHT_RE.finditer(normalized):
        token = match.group("value")
        output.append(ClassicNumericToken("attention_weight", token, float(token), "attention"))
        consumed_spans.append(match.span("value"))

    and_branches = [item.strip() for item in split_top_level_and(normalized) if item.strip()]
    if len(and_branches) > 1:
        for branch in and_branches:
            match = _END_WEIGHT_RE.match(branch)
            if match:
                token = match.group("value")
                output.append(ClassicNumericToken("branch_weight", token, float(token), "AND"))
        return [item.to_dict() for item in output]

    if not consumed_spans and not _SCHEDULE_RE.search(normalized) and not _ATTENTION_WEIGHT_RE.search(normalized):
        parts = _split_unescaped_single_colons(normalized)
        if len(parts) >= 3:
            terminal = parts[-1].strip()
            if re.fullmatch(_NUMBER, terminal):
                if re.fullmatch(r"[-+]?\d+", terminal) and int(terminal) > 0:
                    output.append(
                        ClassicNumericToken(
                            "legacy_sequence_steps",
                            terminal,
                            int(terminal),
                            "legacy_single_colon_suffix",
                        )
                    )
                else:
                    output.append(
                        ClassicNumericToken(
                            "legacy_sequence_weight",
                            terminal,
                            float(terminal),
                            "legacy_single_colon_suffix",
                        )
                    )

    return [item.to_dict() for item in output]


def _split_unescaped_single_colons(source: str) -> list[str]:
    parts: list[str] = []
    start = 0
    brace = bracket = paren = 0
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\\" and not _is_escaped(source, index):
            index += 2
            continue
        if char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == ":" and brace == bracket == paren == 0:
            previous_is_colon = index > 0 and source[index - 1] == ":"
            next_is_colon = index + 1 < len(source) and source[index + 1] == ":"
            if not previous_is_colon and not next_is_colon:
                parts.append(source[start:index])
                start = index + 1
        index += 1
    if not parts:
        return [source]
    parts.append(source[start:])
    return parts


def _consumed_terminators(raw_source: str, normalized_source: str, node_types: list[str]) -> list[str]:
    consumed: list[str] = []
    if "deep_sequence" in node_types:
        raw_trimmed = raw_source.rstrip()
        if raw_trimmed.endswith("!!!") and not _is_escaped(raw_trimmed, len(raw_trimmed) - 1):
            consumed.append("!!!")
        else:
            consumed.append("!!")
    elif "sequence" in node_types:
        if any(char == "!" and not _is_escaped(raw_source, index) for index, char in enumerate(raw_source)):
            consumed.append("!")
    return consumed


def _fallback_encoder_texts(source: str) -> list[str]:
    normalized, _warnings = normalize_legacy_structured_source(source)
    and_branches = [item.strip() for item in split_top_level_and(normalized) if item.strip()]
    if len(and_branches) > 1:
        texts: list[str] = []
        for branch in and_branches:
            parsed = parse_legacy_structured_prompt(branch)
            texts.extend(item.text for item in parsed.branches)
        return texts
    parsed = parse_legacy_structured_prompt(normalized)
    return [item.text for item in parsed.branches]


def inspect_classic_prompt(source: str) -> dict[str, Any]:
    raw_source = str(source or "")
    normalized_source, compatibility_warnings = normalize_legacy_structured_source(raw_source)
    nodes = classic_semantic_nodes(normalized_source)
    node_types: list[str] = []
    for node in nodes:
        node_types.extend(_node_types(node))
    escaped_literals = _escaped_literals(raw_source)

    warnings = list(compatibility_warnings)

    return {
        "contract": CLASSIC_SEMANTIC_CONTRACT_VERSION,
        "raw_source": raw_source,
        "normalized_source": normalized_source,
        "nodes": nodes,
        "node_types": node_types,
        "terminators_consumed": _consumed_terminators(raw_source, normalized_source, node_types),
        "escaped_literals": escaped_literals,
        "numeric_contexts": classify_classic_numeric_contexts(normalized_source),
        "typed_numeric_semantics": [item.to_dict() for item in collect_numeric_semantics(normalized_source)],
        "numeric_labels": [numeric_label(item) for item in collect_numeric_semantics(normalized_source)],
        "encoder_policy": {
            "unescaped_structural_punctuation_must_not_leak": True,
            "escaped_structural_literals_are_text": True,
        },
        "fallback_encoder_texts": _fallback_encoder_texts(normalized_source),
        "warnings": warnings,
    }
