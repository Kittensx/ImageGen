from __future__ import annotations

"""Parser-neutral Prompt IR for IMAGE_GEN prompt semantics.

PPSR-02 introduces this module as the stable boundary between source syntax and
conditioning intent.  The active Legacy parser converts its existing structured
Classic nodes into this IR instead of flattening semantic punctuation directly
into strings.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from modules.prompt_parsers.numeric_semantics import (
    NumericSemantic,
    collect_numeric_semantics,
    numeric_semantic_from_dict,
)

from modules.parser.legacy_structured_prompt import (
    DeepSequenceNode,
    GroupNode,
    LegacyNode,
    ParentChildNode,
    SequenceNode,
    TextNode,
    WeightedNode,
    normalize_legacy_structured_source,
    parse_legacy_node,
    split_top_level_and,
    unescape_classic_literals,
)

LEGACY_PROMPT_IR_CONTRACT_VERSION = "image-gen-prompt-ir-v1"
PROMPT_IR_CONTRACT_VERSION = "image-gen-prompt-ir-v2"


@dataclass(frozen=True)
class IRNode:
    source_text: str = ""


@dataclass(frozen=True)
class Text(IRNode):
    value: str = ""
    escaped_literal: bool = False


@dataclass(frozen=True)
class Literal(IRNode):
    value: str = ""
    escaped_literal: bool = True


@dataclass(frozen=True)
class Group(IRNode):
    items: tuple[IRNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Prompt(IRNode):
    parts: tuple[IRNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Relation(IRNode):
    parent: IRNode = field(default_factory=Text)
    child: IRNode = field(default_factory=Text)


@dataclass(frozen=True)
class SequenceItemIR:
    node: IRNode
    weight: float = 1.0
    active_until_step: int | None = None
    source_text: str = ""
    source_start: int | None = None
    source_end: int | None = None
    terminator: str = ""


@dataclass(frozen=True)
class Sequence(IRNode):
    items: tuple[SequenceItemIR, ...] = field(default_factory=tuple)
    weight: float = 1.0
    active_until_step: int | None = None
    syntax_origin: str = "legacy_single_colon_sequence"


@dataclass(frozen=True)
class OwnerSequence(IRNode):
    owner: IRNode = field(default_factory=Text)
    items: tuple[SequenceItemIR, ...] = field(default_factory=tuple)
    syntax_origin: str = "classic_owner_sequence"
    top_terminator: str = "!!"


@dataclass(frozen=True)
class Weighted(IRNode):
    node: IRNode = field(default_factory=Text)
    weight: float = 1.0


@dataclass(frozen=True)
class Quantity(IRNode):
    node: IRNode = field(default_factory=Text)
    quantity: float = 1.0


@dataclass(frozen=True)
class ConjunctionBranch:
    node: IRNode
    weight: float = 1.0
    source_text: str = ""


@dataclass(frozen=True)
class Conjunction(IRNode):
    branches: tuple[ConjunctionBranch, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Scheduled(IRNode):
    value: str = ""


@dataclass(frozen=True)
class Alternate(IRNode):
    value: str = ""


@dataclass(frozen=True)
class PromptIR:
    parser_namespace: str
    raw_source: str
    normalized_source: str
    root: IRNode
    annotations: tuple[IRNode, ...] = field(default_factory=tuple)
    numeric_semantics: tuple[NumericSemantic, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    contract: str = PROMPT_IR_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return prompt_ir_to_dict(self)


_STRUCTURAL_ESCAPES = (r"\{", r"\}", r"\:", r"\!", r"\|", r"\\")


def _legacy_to_ir(node: LegacyNode, *, source_text: str = "") -> IRNode:
    if isinstance(node, TextNode):
        return _text_node_to_ir(str(node.value or ""), source_text=source_text)
    if isinstance(node, GroupNode):
        return Group(
            source_text=source_text,
            items=tuple(_legacy_to_ir(item) for item in node.items),
        )
    if isinstance(node, ParentChildNode):
        return Relation(
            source_text=source_text,
            parent=_legacy_to_ir(node.parent),
            child=_legacy_to_ir(node.child),
        )
    if isinstance(node, DeepSequenceNode):
        return OwnerSequence(
            source_text=source_text,
            owner=_legacy_to_ir(node.owner),
            items=tuple(
                SequenceItemIR(
                    node=_legacy_to_ir(item.node, source_text=item.source_text),
                    weight=float(item.weight),
                    active_until_step=item.active_until_step,
                    source_text=item.source_text,
                    source_start=item.source_start,
                    source_end=item.source_end,
                    terminator=item.terminator,
                )
                for item in node.items
            ),
            syntax_origin=node.syntax_origin,
            top_terminator=node.top_terminator,
        )
    if isinstance(node, SequenceNode):
        return Sequence(
            source_text=source_text,
            items=tuple(
                SequenceItemIR(
                    node=_legacy_to_ir(item.node, source_text=item.source_text),
                    weight=float(item.weight),
                    active_until_step=item.active_until_step,
                    source_text=item.source_text,
                    source_start=item.source_start,
                    source_end=item.source_end,
                    terminator=item.terminator,
                )
                for item in node.items
            ),
            weight=float(node.weight),
            active_until_step=node.active_until_step,
            syntax_origin=node.syntax_origin,
        )
    if isinstance(node, WeightedNode):
        return Weighted(
            source_text=source_text,
            node=_legacy_to_ir(node.node),
            weight=float(node.weight),
        )
    return Text(source_text=source_text, value=str(node))


def _annotations_for_source(source: str) -> tuple[IRNode, ...]:
    # These annotations deliberately preserve A1111 constructs for the existing
    # LearnedConditioning scheduler.  PPSR-02 records them in the common IR but
    # does not replace that mature scheduling implementation.
    import re

    output: list[IRNode] = []
    if re.search(r"\[[^\]]*:[^\]]*:[^\]]*\]", source):
        output.append(Scheduled(source_text=source, value=source))
    if re.search(r"\[[^\]|]+(?:\|[^\]]+)+\]", source):
        output.append(Alternate(source_text=source, value=source))
    return tuple(output)


def _is_escaped(source: str, index: int) -> bool:
    """Return True when ``source[index]`` is preceded by an odd slash count."""
    slash_count = 0
    cursor = int(index) - 1
    while cursor >= 0 and source[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return bool(slash_count % 2)




def _has_quantity_prefix(source: str, brace_index: int) -> bool:
    """Return True for count syntax like ``2{cat|dog}``.

    PPSR-05 treats the leading integer as Quantity metadata and leaves the
    execution syntax untouched for the parser that owns dynamic choices.
    """
    cursor = int(brace_index) - 1
    while cursor >= 0 and source[cursor].isspace():
        cursor -= 1
    end = cursor + 1
    while cursor >= 0 and source[cursor].isdigit():
        cursor -= 1
    if end == cursor + 1:
        return False
    start = cursor + 1
    if cursor >= 0 and (source[cursor].isalnum() or source[cursor] == "_"):
        return False
    body_end = source.find("}", brace_index + 1)
    if body_end < 0:
        return False
    return "|" in source[brace_index + 1 : body_end]



def _text_leaf(value: str, *, source_text: str = "") -> IRNode:
    raw = str(value or "")
    source = str(source_text or raw)
    escaped = any(token in source for token in _STRUCTURAL_ESCAPES) or any(
        token in raw for token in _STRUCTURAL_ESCAPES
    )
    cls = Literal if escaped else Text
    rendered = unescape_classic_literals(raw) if escaped else raw
    return cls(source_text=source, value=rendered, escaped_literal=escaped)


def _text_node_to_ir(value: str, *, source_text: str = "") -> IRNode:
    """Recursively preserve embedded Classic groups inside any text position.

    PPSR-02 originally scanned embedded ``{...}`` groups only when the entire
    prompt root was otherwise plain text.  PPSR-06A promotes that rule to every
    text-bearing position (relation parent/child, owner item, sequence item,
    and nested prompt fragment).  Quantity syntax such as ``2{cat|dog}`` and
    escaped braces remain literal.
    """
    text = str(value or "")
    if not text:
        return _text_leaf(text, source_text=source_text)

    parts: list[IRNode] = []
    start = 0
    index = 0
    found_group = False
    while index < len(text):
        if text[index] == "\\" and not _is_escaped(text, index):
            index += 2
            continue
        if text[index] != "{" or _is_escaped(text, index):
            index += 1
            continue
        if _has_quantity_prefix(text, index):
            index += 1
            continue

        depth = 0
        escaped = False
        close = -1
        cursor = index
        while cursor < len(text):
            char = text[cursor]
            if escaped:
                escaped = False
                cursor += 1
                continue
            if char == "\\":
                escaped = True
                cursor += 1
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    close = cursor
                    break
            cursor += 1
        if close < 0:
            break

        group_source = text[index : close + 1]
        group_node = parse_legacy_node(group_source)
        if not isinstance(group_node, GroupNode):
            index += 1
            continue

        if index > start:
            segment = text[start:index]
            parts.append(_text_leaf(segment, source_text=segment))
        parts.append(_legacy_to_ir(group_node, source_text=group_source))
        found_group = True
        start = close + 1
        index = close + 1

    if not found_group:
        return _text_leaf(text, source_text=source_text)
    if start < len(text):
        segment = text[start:]
        parts.append(_text_leaf(segment, source_text=segment))
    return Prompt(source_text=str(source_text or text), parts=tuple(parts))


def _mixed_prompt_root(source: str) -> IRNode:
    """Parse structural Classic syntax, then recursively preserve embedded groups."""
    legacy = parse_legacy_node(source)
    if not isinstance(legacy, TextNode):
        return _legacy_to_ir(legacy, source_text=source)
    return _text_node_to_ir(str(legacy.value or ""), source_text=source)


def parse_prompt_ir(source: str, *, parser_namespace: str = "legacy") -> PromptIR:
    raw = str(source or "")
    normalized, warnings = normalize_legacy_structured_source(raw)
    branches = split_top_level_and(normalized)
    clean_branches = [item for item in branches if item.strip()]

    if len(clean_branches) > 1:
        # Preserve AND as a first-class composition relationship.  Branch
        # weights remain a downstream Legacy compatibility concern unless the
        # branch itself owns structured Classic syntax.
        import re
        branch_weight_re = re.compile(
            r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$"
        )
        built: list[ConjunctionBranch] = []
        for branch in clean_branches:
            legacy_node = parse_legacy_node(branch)
            branch_weight = 1.0
            branch_source = branch
            # Preserve historical AND branch weights only when the branch does
            # not already own a structured numeric meaning.
            if isinstance(legacy_node, TextNode):
                match = branch_weight_re.search(branch)
                if match is not None and match.group(2) is not None:
                    branch_source = match.group(1)
                    branch_weight = float(match.group(2))
                    legacy_node = parse_legacy_node(branch_source)
            built.append(
                ConjunctionBranch(
                    node=_mixed_prompt_root(branch_source),
                    weight=branch_weight,
                    source_text=branch,
                )
            )
        ir_branches = tuple(built)
        root: IRNode = Conjunction(source_text=normalized, branches=ir_branches)
    else:
        root = _mixed_prompt_root(normalized)

    numeric_semantics = collect_numeric_semantics(normalized)
    numeric_warnings = [item.message for item in numeric_semantics if item.message and (not item.valid or item.inferred)]
    return PromptIR(
        parser_namespace=str(parser_namespace or "legacy").strip().lower(),
        raw_source=raw,
        normalized_source=normalized,
        root=root,
        annotations=_annotations_for_source(normalized),
        numeric_semantics=numeric_semantics,
        warnings=tuple([*warnings, *numeric_warnings]),
    )


def node_to_dict(node: IRNode) -> dict[str, Any]:
    base: dict[str, Any] = {"source_text": node.source_text}
    if isinstance(node, Literal):
        return {**base, "type": "literal", "value": node.value, "escaped_literal": True}
    if isinstance(node, Text):
        return {**base, "type": "text", "value": node.value, "escaped_literal": bool(node.escaped_literal)}
    if isinstance(node, Group):
        return {**base, "type": "group", "items": [node_to_dict(item) for item in node.items]}
    if isinstance(node, Prompt):
        return {**base, "type": "prompt", "parts": [node_to_dict(item) for item in node.parts]}
    if isinstance(node, Relation):
        return {
            **base,
            "type": "relation",
            "parent": node_to_dict(node.parent),
            "child": node_to_dict(node.child),
        }
    if isinstance(node, Sequence):
        return {
            **base,
            "type": "sequence",
            "weight": float(node.weight),
            "active_until_step": node.active_until_step,
            "syntax_origin": node.syntax_origin,
            "items": [
                {
                    "node": node_to_dict(item.node),
                    "weight": float(item.weight),
                    "active_until_step": item.active_until_step,
                    "source_text": item.source_text,
                    "source_start": item.source_start,
                    "source_end": item.source_end,
                    "terminator": item.terminator,
                }
                for item in node.items
            ],
        }
    if isinstance(node, OwnerSequence):
        return {
            **base,
            "type": "owner_sequence",
            "owner": node_to_dict(node.owner),
            "syntax_origin": node.syntax_origin,
            "top_terminator": node.top_terminator,
            "items": [
                {
                    "node": node_to_dict(item.node),
                    "weight": float(item.weight),
                    "active_until_step": item.active_until_step,
                    "source_text": item.source_text,
                    "source_start": item.source_start,
                    "source_end": item.source_end,
                    "terminator": item.terminator,
                }
                for item in node.items
            ],
        }
    if isinstance(node, Weighted):
        return {**base, "type": "weighted", "weight": float(node.weight), "node": node_to_dict(node.node)}
    if isinstance(node, Quantity):
        return {**base, "type": "quantity", "quantity": float(node.quantity), "node": node_to_dict(node.node)}
    if isinstance(node, Conjunction):
        return {
            **base,
            "type": "conjunction",
            "branches": [
                {
                    "node": node_to_dict(item.node),
                    "weight": float(item.weight),
                    "source_text": item.source_text,
                }
                for item in node.branches
            ],
        }
    if isinstance(node, Scheduled):
        return {**base, "type": "scheduled", "value": node.value}
    if isinstance(node, Alternate):
        return {**base, "type": "alternate", "value": node.value}
    raise TypeError(f"Unsupported Prompt IR node: {type(node)!r}")


def prompt_ir_to_dict(prompt_ir: PromptIR) -> dict[str, Any]:
    return {
        "contract": prompt_ir.contract,
        "parser_namespace": prompt_ir.parser_namespace,
        "raw_source": prompt_ir.raw_source,
        "normalized_source": prompt_ir.normalized_source,
        "root": node_to_dict(prompt_ir.root),
        "annotations": [node_to_dict(item) for item in prompt_ir.annotations],
        "numeric_semantics": [item.to_dict() for item in prompt_ir.numeric_semantics],
        "warnings": list(prompt_ir.warnings),
    }


def _node_from_dict(payload: Mapping[str, Any]) -> IRNode:
    data = dict(payload or {})
    node_type = str(data.get("type") or "text")
    source_text = str(data.get("source_text") or "")
    if node_type == "literal":
        return Literal(source_text=source_text, value=str(data.get("value") or ""), escaped_literal=True)
    if node_type == "text":
        return Text(
            source_text=source_text,
            value=str(data.get("value") or ""),
            escaped_literal=bool(data.get("escaped_literal", False)),
        )
    if node_type == "group":
        return Group(source_text=source_text, items=tuple(_node_from_dict(item) for item in data.get("items") or []))
    if node_type == "prompt":
        return Prompt(source_text=source_text, parts=tuple(_node_from_dict(item) for item in data.get("parts") or []))
    if node_type == "relation":
        return Relation(
            source_text=source_text,
            parent=_node_from_dict(data.get("parent") or {}),
            child=_node_from_dict(data.get("child") or {}),
        )
    if node_type in {"sequence", "owner_sequence"}:
        items = tuple(
            SequenceItemIR(
                node=_node_from_dict(item.get("node") or {}),
                weight=float(item.get("weight", 1.0)),
                active_until_step=item.get("active_until_step"),
                source_text=str(item.get("source_text") or ""),
                source_start=item.get("source_start"),
                source_end=item.get("source_end"),
                terminator=str(item.get("terminator") or ""),
            )
            for item in data.get("items") or []
        )
        if node_type == "owner_sequence":
            return OwnerSequence(
                source_text=source_text,
                owner=_node_from_dict(data.get("owner") or {}),
                items=items,
                syntax_origin=str(data.get("syntax_origin") or "classic_owner_sequence"),
                top_terminator=str(data.get("top_terminator") or "!!"),
            )
        return Sequence(
            source_text=source_text,
            items=items,
            weight=float(data.get("weight", 1.0)),
            active_until_step=data.get("active_until_step"),
            syntax_origin=str(data.get("syntax_origin") or "legacy_single_colon_sequence"),
        )
    if node_type == "weighted":
        return Weighted(
            source_text=source_text,
            node=_node_from_dict(data.get("node") or {}),
            weight=float(data.get("weight", 1.0)),
        )
    if node_type == "quantity":
        return Quantity(
            source_text=source_text,
            node=_node_from_dict(data.get("node") or {}),
            quantity=float(data.get("quantity", 1.0)),
        )
    if node_type == "conjunction":
        return Conjunction(
            source_text=source_text,
            branches=tuple(
                ConjunctionBranch(
                    node=_node_from_dict(item.get("node") or {}),
                    weight=float(item.get("weight", 1.0)),
                    source_text=str(item.get("source_text") or ""),
                )
                for item in data.get("branches") or []
            ),
        )
    if node_type == "scheduled":
        return Scheduled(source_text=source_text, value=str(data.get("value") or ""))
    if node_type == "alternate":
        return Alternate(source_text=source_text, value=str(data.get("value") or ""))
    raise ValueError(f"Unsupported serialized Prompt IR node type: {node_type!r}")


def prompt_ir_from_dict(payload: Mapping[str, Any]) -> PromptIR:
    data = dict(payload or {})
    contract = str(data.get("contract") or "")
    if contract not in {PROMPT_IR_CONTRACT_VERSION, LEGACY_PROMPT_IR_CONTRACT_VERSION}:
        raise ValueError(f"Unsupported Prompt IR contract: {contract!r}")
    normalized = str(data.get("normalized_source") or "")
    if contract == PROMPT_IR_CONTRACT_VERSION:
        numeric_semantics = tuple(
            numeric_semantic_from_dict(item) for item in data.get("numeric_semantics") or []
        )
    else:
        # PPSR-05 migration path for canonical-v2 payloads written while PromptIR
        # was v1. Old metadata stays readable; the typed meanings are rebuilt once
        # during in-memory migration and are serialized explicitly on the next v2
        # canonicalization.
        numeric_semantics = collect_numeric_semantics(normalized)
    return PromptIR(
        parser_namespace=str(data.get("parser_namespace") or "legacy"),
        raw_source=str(data.get("raw_source") or ""),
        normalized_source=normalized,
        root=_node_from_dict(data.get("root") or {}),
        annotations=tuple(_node_from_dict(item) for item in data.get("annotations") or []),
        numeric_semantics=numeric_semantics,
        warnings=tuple(str(item) for item in data.get("warnings") or []),
        contract=PROMPT_IR_CONTRACT_VERSION,
    )


def ir_equivalent(left: PromptIR, right: PromptIR) -> bool:
    """Semantic-equivalence helper used by canonical replay tests."""
    left_dict = prompt_ir_to_dict(left)
    right_dict = prompt_ir_to_dict(right)
    for payload in (left_dict, right_dict):
        payload.pop("raw_source", None)
        payload.pop("warnings", None)
    return left_dict == right_dict
