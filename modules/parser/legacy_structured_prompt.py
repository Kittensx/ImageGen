from __future__ import annotations

"""Structured Classic/Legacy prompt syntax helpers.

The IMAGE_GEN legacy parser historically supported A1111-style weighting,
scheduling, alternates, and composable ``AND`` branches. This module adds a
small escape-aware structural pre-pass so the IMAGE_GEN/A1111-derived sequence
syntax is interpreted before the downstream schedule parser sees the prompt.

Supported structured forms:

* ``{a, b}`` groups terms into one cohesive conditioning phrase.
* ``property::value!`` creates a closed sequence item.
* ``owner:::property::value!, other::value!!`` creates a top-level sequence.
* ``a:b:c`` remains supported as a backward-compatible equal-weight legacy
  branch sequence.
* ``[a:b:c]:modifier`` still applies a wrapper weight/step suffix to the whole
  backward-compatible legacy ``:`` sequence.

The legacy single-colon sequence behavior is preserved for compatibility, but
IMAGE_GEN's modeled A1111 sequence syntax is the closed ``::`` / ``:::`` form,
terminated by ``!`` and ``!!`` respectively.
"""

from dataclasses import dataclass, field
import re

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_AND_RE = re.compile(r"\bAND\b")


@dataclass(frozen=True)
class LegacyNode:
    pass


@dataclass(frozen=True)
class TextNode(LegacyNode):
    value: str


@dataclass(frozen=True)
class GroupNode(LegacyNode):
    items: tuple[LegacyNode, ...]


@dataclass(frozen=True)
class ParentChildNode(LegacyNode):
    parent: LegacyNode
    child: LegacyNode


@dataclass(frozen=True)
class SequenceItem:
    node: LegacyNode
    weight: float = 1.0
    active_until_step: int | None = None


@dataclass(frozen=True)
class SequenceNode(LegacyNode):
    items: tuple[SequenceItem, ...]
    weight: float = 1.0
    active_until_step: int | None = None


@dataclass(frozen=True)
class DeepSequenceNode(LegacyNode):
    owner: LegacyNode
    items: tuple[SequenceItem, ...]


@dataclass(frozen=True)
class WeightedNode(LegacyNode):
    node: LegacyNode
    weight: float


@dataclass(frozen=True)
class LegacyConditioningBranch:
    text: str
    weight: float = 1.0
    active_until_step: int | None = None
    hold_after_step: bool = False
    semantic_type: str = "text"


@dataclass(frozen=True)
class LegacyStructuredParse:
    node: LegacyNode
    structured: bool
    branches: tuple[LegacyConditioningBranch, ...] = field(default_factory=tuple)


def _is_escaped(source: str, index: int) -> bool:
    slash_count = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        slash_count += 1
        index -= 1
    return bool(slash_count % 2)


def _balanced_outer(source: str, opener: str, closer: str) -> bool:
    text = source.strip()
    if len(text) < 2 or text[0] != opener or text[-1] != closer:
        return False
    depth = 0
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def _find_top_level_operator(source: str, operator: str, start: int = 0) -> int:
    if not operator:
        return -1
    brace = bracket = paren = 0
    index = int(start)
    length = len(source)
    op_len = len(operator)
    while index <= length - op_len:
        char = source[index]
        if char == "\\" and not _is_escaped(source, index):
            index += 2
            continue
        if char == "{" and not _is_escaped(source, index):
            brace += 1
            index += 1
            continue
        if char == "}" and not _is_escaped(source, index):
            brace = max(0, brace - 1)
            index += 1
            continue
        if char == "[" and not _is_escaped(source, index):
            bracket += 1
            index += 1
            continue
        if char == "]" and not _is_escaped(source, index):
            bracket = max(0, bracket - 1)
            index += 1
            continue
        if char == "(" and not _is_escaped(source, index):
            paren += 1
            index += 1
            continue
        if char == ")" and not _is_escaped(source, index):
            paren = max(0, paren - 1)
            index += 1
            continue
        if brace == bracket == paren == 0 and source.startswith(operator, index):
            if operator == ":":
                if (index > 0 and source[index - 1] == ":") or (index + 1 < length and source[index + 1] == ":"):
                    index += 1
                    continue
            if operator == "::" and index + 2 < length and source[index + 2] == ":":
                index += 1
                continue
            return index
        index += 1
    return -1


def _split_top_level(source: str, operator: str) -> list[str]:
    """Split on an unescaped operator outside {}, [], and ()."""
    if not operator:
        return [source]
    parts: list[str] = []
    start = 0
    index = 0
    while True:
        found = _find_top_level_operator(source, operator, start=index)
        if found < 0:
            break
        parts.append(source[start:found])
        start = found + len(operator)
        index = start
    if not parts:
        return [source]
    parts.append(source[start:])
    return parts


def _split_group_items(source: str) -> list[str]:
    comma_parts = _split_top_level(source, ",")
    output: list[str] = []
    for part in comma_parts:
        output.extend(_split_top_level(part, "|"))
    return [item.strip() for item in output if item.strip()]


def _unescape_literal(text: str) -> str:
    return re.sub(r"\\([:{}|,\\])", r"\1", text)


def _strip_close_marker(text: str) -> str:
    value = str(text or "").rstrip()
    removed = 0
    while value.endswith("!") and removed < 2:
        index = len(value) - 1
        if _is_escaped(value, index):
            break
        value = value[:-1].rstrip()
        removed += 1
    return value


def _unescape_close_literal(text: str) -> str:
    return str(text or "").replace(r"\!", "!")


def _numeric_kind(value: str) -> tuple[str, float | int] | None:
    token = value.strip()
    if not _NUMBER_RE.fullmatch(token):
        return None
    if _INTEGER_RE.fullmatch(token):
        integer = int(token)
        if integer > 0:
            return "steps", integer
    return "weight", float(token)


def _parse_sequence_wrapper(source: str) -> SequenceNode | None:
    text = source.strip()
    if not text.startswith("["):
        return None
    depth = 0
    escaped = False
    close_index = -1
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                close_index = index
                break
    if close_index <= 0:
        return None
    tail = text[close_index + 1 :].strip()
    if not tail.startswith(":"):
        return None
    modifier = _numeric_kind(tail[1:])
    if modifier is None:
        return None
    inner = text[1:close_index]
    inner_node = parse_legacy_node(inner)
    if not isinstance(inner_node, SequenceNode):
        return None
    kind, value = modifier
    if kind == "steps":
        return SequenceNode(
            items=inner_node.items,
            weight=inner_node.weight,
            active_until_step=int(value),
        )
    return SequenceNode(
        items=inner_node.items,
        weight=float(value),
        active_until_step=inner_node.active_until_step,
    )


def _find_sequence_closer(source: str, start: int = 0) -> tuple[int, str] | None:
    brace = bracket = paren = 0
    index = int(start)
    while index < len(source):
        char = source[index]
        if char == "\\" and not _is_escaped(source, index):
            index += 2
            continue
        if char == "{" and not _is_escaped(source, index):
            brace += 1
            index += 1
            continue
        if char == "}" and not _is_escaped(source, index):
            brace = max(0, brace - 1)
            index += 1
            continue
        if char == "[" and not _is_escaped(source, index):
            bracket += 1
            index += 1
            continue
        if char == "]" and not _is_escaped(source, index):
            bracket = max(0, bracket - 1)
            index += 1
            continue
        if char == "(" and not _is_escaped(source, index):
            paren += 1
            index += 1
            continue
        if char == ")" and not _is_escaped(source, index):
            paren = max(0, paren - 1)
            index += 1
            continue
        if brace == bracket == paren == 0:
            if source.startswith("!", index) and not _is_escaped(source, index):
                return index, "!"
            if source.startswith("~", index) and not _is_escaped(source, index):
                return index, "~"
        index += 1
    return None


def _parse_closed_sequence_items(source: str) -> tuple[SequenceItem, ...] | None:
    text = str(source or "").strip()
    if not text or "::" not in text or not any(marker in text for marker in ("!", "~")):
        return None

    items: list[SequenceItem] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= length:
            break
        separator = _find_top_level_operator(text, "::", start=index)
        if separator < 0:
            return None
        label = text[index:separator].strip()
        if not label:
            return None
        value_start = separator + 2
        closer = _find_sequence_closer(text, value_start)
        if closer is None:
            return None
        close_index, _close_symbol = closer
        value = text[value_start:close_index].strip()
        if not value:
            return None
        items.append(
            SequenceItem(
                node=ParentChildNode(
                    _parse_nonsequence_atom(label),
                    _parse_nonsequence_atom(value),
                )
            )
        )
        index = close_index + 1
        while index < length and text[index].isspace():
            index += 1
        if index < length and text[index] == ",":
            index += 1
    return tuple(items) if items else None


def _parse_closed_sequence(source: str) -> SequenceNode | None:
    items = _parse_closed_sequence_items(source)
    if not items:
        return None
    return SequenceNode(items=items)


def _parse_deep_sequence(source: str) -> DeepSequenceNode | None:
    text = str(source or "").strip()
    if not text.endswith("!!") or _is_escaped(text, len(text) - 1):
        return None
    separator = _find_top_level_operator(text, ":::")
    if separator < 0:
        return None
    owner_text = text[:separator].strip()
    if not owner_text:
        return None
    body = text[separator + 3 : -2].strip()
    if body:
        body = f"{body}!"
    items = _parse_closed_sequence_items(body)
    if not items:
        return None
    return DeepSequenceNode(owner=_parse_nonsequence_atom(owner_text), items=items)


def parse_legacy_node(source: str) -> LegacyNode:
    text = str(source or "").strip()
    if not text:
        return TextNode("")

    wrapped = _parse_sequence_wrapper(text)
    if wrapped is not None:
        return wrapped

    deep_sequence = _parse_deep_sequence(text)
    if deep_sequence is not None:
        return deep_sequence

    sequence_parts = [part.strip() for part in _split_top_level(text, ":")]
    if len(sequence_parts) > 1:
        terminal_modifier = _numeric_kind(sequence_parts[-1])
        if terminal_modifier is not None and len(sequence_parts) == 2:
            base_node = _parse_nonsequence_node(sequence_parts[0])
            if not isinstance(base_node, TextNode):
                _kind, value = terminal_modifier
                return WeightedNode(base_node, float(value))
            return _parse_nonsequence_node(text)
        textual_parts = sequence_parts
        item_weight = 1.0
        item_steps: int | None = None
        if terminal_modifier is not None and len(sequence_parts) >= 3:
            textual_parts = sequence_parts[:-1]
            kind, value = terminal_modifier
            if kind == "steps":
                item_steps = int(value)
            else:
                item_weight = float(value)

        if len(textual_parts) >= 2 and all(part != "" for part in textual_parts):
            items: list[SequenceItem] = []
            for idx, part in enumerate(textual_parts):
                weight = item_weight if idx == len(textual_parts) - 1 else 1.0
                active_until = item_steps if idx == len(textual_parts) - 1 else None
                items.append(
                    SequenceItem(
                        node=_parse_nonsequence_node(part),
                        weight=weight,
                        active_until_step=active_until,
                    )
                )
            return SequenceNode(tuple(items))

    closed_sequence = _parse_closed_sequence(text)
    if closed_sequence is not None:
        return closed_sequence

    return _parse_nonsequence_node(text)


def _parse_nonsequence_node(text: str) -> LegacyNode:
    text = text.strip()
    parent_parts = [part.strip() for part in _split_top_level(text, "::")]
    if len(parent_parts) > 1 and all(parent_parts):
        node: LegacyNode = _parse_nonsequence_atom(parent_parts[0])
        for child_text in parent_parts[1:]:
            node = ParentChildNode(node, _parse_nonsequence_atom(child_text))
        return node
    return _parse_nonsequence_atom(text)


def _parse_nonsequence_atom(text: str) -> LegacyNode:
    text = text.strip()
    if _balanced_outer(text, "{", "}"):
        inner = text[1:-1]
        items = tuple(parse_legacy_node(item) for item in _split_group_items(inner))
        return GroupNode(items)
    return TextNode(_unescape_literal(text))


def _render_node(node: LegacyNode) -> str:
    if isinstance(node, TextNode):
        return node.value.strip()
    if isinstance(node, GroupNode):
        return ", ".join(filter(None, (_render_node(item) for item in node.items))).strip()
    if isinstance(node, ParentChildNode):
        parent = _render_node(node.parent).strip(" ,")
        child = _unescape_close_literal(_strip_close_marker(_render_node(node.child))).strip(" ,")
        return ", ".join(item for item in (parent, child) if item)
    if isinstance(node, WeightedNode):
        return _render_node(node.node)
    if isinstance(node, DeepSequenceNode):
        owner = _render_node(node.owner).strip(" ,")
        parts: list[str] = []
        for item in node.items:
            rendered_item = _render_node(item.node).strip(" ,")
            parts.append(", ".join(part for part in (owner, rendered_item) if part))
        return "; ".join(part for part in parts if part).strip()
    if isinstance(node, SequenceNode):
        return ", ".join(filter(None, (_render_node(item.node) for item in node.items))).strip()
    return str(node)


def _is_structured(node: LegacyNode) -> bool:
    return not isinstance(node, TextNode)


def _branches_for_node(node: LegacyNode) -> tuple[LegacyConditioningBranch, ...]:
    if isinstance(node, DeepSequenceNode):
        owner_text = _render_node(node.owner).strip(" ,")
        output: list[LegacyConditioningBranch] = []
        for item in node.items:
            rendered_item = _unescape_close_literal(_render_node(item.node)).strip(" ,")
            text = ", ".join(part for part in (owner_text, rendered_item) if part).strip(" ,")
            if not text:
                continue
            output.append(
                LegacyConditioningBranch(
                    text=text,
                    weight=float(item.weight),
                    active_until_step=item.active_until_step,
                    semantic_type="deep_sequence_item",
                )
            )
        return tuple(output)
    if isinstance(node, SequenceNode):
        output: list[LegacyConditioningBranch] = []
        last_index = len(node.items) - 1
        for index, item in enumerate(node.items):
            text = _render_node(item.node).strip()
            if index == last_index and not isinstance(item.node, ParentChildNode):
                text = _unescape_close_literal(_strip_close_marker(text))
            else:
                text = _unescape_close_literal(text)
            if not text:
                continue
            active_until = item.active_until_step
            hold_after = False
            if node.active_until_step is not None:
                active_until = node.active_until_step
                hold_after = index == last_index
            output.append(
                LegacyConditioningBranch(
                    text=text,
                    weight=float(item.weight) * float(node.weight),
                    active_until_step=active_until,
                    hold_after_step=hold_after,
                    semantic_type="sequence_item",
                )
            )
        return tuple(output)
    if isinstance(node, WeightedNode):
        return (
            LegacyConditioningBranch(
                text=_render_node(node.node),
                weight=float(node.weight),
                semantic_type="weighted_structured",
            ),
        )
    semantic_type = (
        "group" if isinstance(node, GroupNode)
        else "parent_child" if isinstance(node, ParentChildNode)
        else "text"
    )
    return (
        LegacyConditioningBranch(
            text=_render_node(node),
            semantic_type=semantic_type,
        ),
    )


def parse_legacy_structured_prompt(source: str) -> LegacyStructuredParse:
    node = parse_legacy_node(source)
    return LegacyStructuredParse(
        node=node,
        structured=_is_structured(node),
        branches=_branches_for_node(node),
    )


def node_to_dict(node: LegacyNode) -> dict:
    if isinstance(node, TextNode):
        return {"type": "text", "value": node.value}
    if isinstance(node, GroupNode):
        return {
            "type": "group",
            "items": [node_to_dict(item) for item in node.items],
            "rendered": _render_node(node),
        }
    if isinstance(node, ParentChildNode):
        return {
            "type": "parent_child",
            "parent": node_to_dict(node.parent),
            "child": node_to_dict(node.child),
            "rendered": _render_node(node),
        }
    if isinstance(node, DeepSequenceNode):
        return {
            "type": "deep_sequence",
            "equal_weight_default": True,
            "owner": node_to_dict(node.owner),
            "items": [
                {
                    "node": node_to_dict(item.node),
                    "weight": float(item.weight),
                    "active_until_step": item.active_until_step,
                }
                for item in node.items
            ],
            "rendered": _render_node(node),
        }
    if isinstance(node, SequenceNode):
        return {
            "type": "sequence",
            "equal_weight_default": True,
            "weight": float(node.weight),
            "active_until_step": node.active_until_step,
            "items": [
                {
                    "node": node_to_dict(item.node),
                    "weight": float(item.weight),
                    "active_until_step": item.active_until_step,
                }
                for item in node.items
            ],
        }
    if isinstance(node, WeightedNode):
        return {
            "type": "weighted_structured",
            "weight": float(node.weight),
            "node": node_to_dict(node.node),
        }
    return {"type": "text", "value": str(node)}


def split_top_level_and(source: str) -> list[str]:
    """Split ``AND`` only outside Classic/A1111 nested structures."""
    text = str(source or "")
    parts: list[str] = []
    start = 0
    brace = bracket = paren = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and not _is_escaped(text, index):
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
        elif brace == bracket == paren == 0:
            match = _AND_RE.match(text, index)
            if match:
                before_ok = index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")
                end = match.end()
                after_ok = end == len(text) or not (text[end].isalnum() or text[end] == "_")
                if before_ok and after_ok:
                    parts.append(text[start:index])
                    start = end
                    index = end
                    continue
        index += 1
    if not parts:
        return [text]
    parts.append(text[start:])
    return parts


def semantic_nodes_for_prompt(source: str) -> list[dict]:
    branches = [item.strip() for item in split_top_level_and(source) if item.strip()]
    if len(branches) > 1:
        return [{
            "type": "conjunction",
            "branches": [node_to_dict(parse_legacy_node(item)) for item in branches],
        }]
    parsed = parse_legacy_structured_prompt(source)
    return [node_to_dict(parsed.node)] if parsed.structured else []
