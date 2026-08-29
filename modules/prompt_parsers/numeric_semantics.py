from __future__ import annotations

"""Typed numeric semantics for the shared Classic prompt grammar.

PPSR-05 makes the grammar position authoritative.  Numeric spelling alone never
selects between weight, step, fraction, percent, or quantity outside the one
explicitly retained legacy single-colon compatibility rule.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from modules.parser.legacy_structured_prompt import normalize_legacy_structured_source, split_top_level_and

NUMERIC_SEMANTIC_CONTRACT_VERSION = "image-gen-numeric-semantics-v1"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_INTEGER = r"[-+]?\d+"
_SCHEDULE_RE = re.compile(
    rf"\[(?P<before>[^\[\]]*?):(?P<after>[^\[\]]*?):(?P<token>{_NUMBER})(?P<percent>%)?\]"
)
_ATTENTION_RE = re.compile(rf"\((?P<body>[^()]*)\:(?P<token>{_NUMBER})\)")
_END_WEIGHT_RE = re.compile(rf"^(?P<body>.*?)(?<!\\):(?P<token>{_NUMBER})\s*$")
_QUANTITY_RE = re.compile(r"(?<![A-Za-z0-9_])(?P<token>[-+]?\d+)\s*(?=\{[^{}]*\|[^{}]*\})")
_LEGACY_WRAPPER_RE = re.compile(rf"^\[(?P<body>.+)\]\s*:(?P<token>{_NUMBER})\s*$")
_STRUCTURED_OUTER_WEIGHT_RE = re.compile(rf"^(?P<body>\{{.*\}})\s*:(?P<token>{_NUMBER})\s*$")
_INVALID_NUMERIC_WORD = re.compile(r"(?i)^[+-]?(?:nan|inf|infinity)$")
_INVALID_SCHEDULE_RE = re.compile(r"\[(?P<before>[^\[\]]*?):(?P<after>[^\[\]]*?):(?P<token>[+-]?(?:nan|inf|infinity))\]", re.IGNORECASE)
_INVALID_ATTENTION_RE = re.compile(r"\((?P<body>[^()]*)\:(?P<token>[+-]?(?:nan|inf|infinity))\)", re.IGNORECASE)


@dataclass(frozen=True)
class NumericSemantic:
    token: str
    context: str
    source_start: int | None = None
    source_end: int | None = None
    inferred: bool = False
    valid: bool = True
    message: str = ""

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def value(self) -> float | int | str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "token": self.token,
            "value": self.value,
            "context": self.context,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "inferred": bool(self.inferred),
            "valid": bool(self.valid),
            "message": self.message,
        }




@dataclass(frozen=True)
class InvalidNumeric(NumericSemantic):
    raw_value: str = ""
    expected_kind: str = "number"

    @property
    def kind(self) -> str:
        return "invalid_numeric"

    @property
    def value(self) -> str:
        return self.raw_value

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "expected_kind": self.expected_kind}


@dataclass(frozen=True)
class WeightValue(NumericSemantic):
    weight: float = 1.0
    scope: str = "generic"

    @property
    def kind(self) -> str:
        return "weight"

    @property
    def value(self) -> float:
        return float(self.weight)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "scope": self.scope}


@dataclass(frozen=True)
class AbsoluteStep(NumericSemantic):
    step: int = 1

    @property
    def kind(self) -> str:
        return "absolute_step"

    @property
    def value(self) -> int:
        return int(self.step)


@dataclass(frozen=True)
class FractionBoundary(NumericSemantic):
    fraction: float = 0.5

    @property
    def kind(self) -> str:
        return "fraction_boundary"

    @property
    def value(self) -> float:
        return float(self.fraction)


@dataclass(frozen=True)
class PercentBoundary(NumericSemantic):
    percent: float = 50.0

    @property
    def kind(self) -> str:
        return "percent_boundary"

    @property
    def value(self) -> float:
        return float(self.percent)


@dataclass(frozen=True)
class QuantityValue(NumericSemantic):
    quantity: int = 1

    @property
    def kind(self) -> str:
        return "quantity"

    @property
    def value(self) -> int:
        return int(self.quantity)


def _is_escaped(source: str, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return bool(slashes % 2)


def _split_top_level_with_spans(source: str, delimiters: tuple[str, ...] = (",", "|")) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []
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
        elif char in delimiters and brace == bracket == paren == 0:
            output.append((source[start:index], start, index))
            start = index + 1
        index += 1
    output.append((source[start:], start, len(source)))
    return output


def _group_spans(source: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\\" and not _is_escaped(source, index):
            index += 2
            continue
        if char == "{" and not _is_escaped(source, index):
            stack.append(index)
        elif char == "}" and not _is_escaped(source, index) and stack:
            start = stack.pop()
            spans.append((start, index + 1))
        index += 1
    return sorted(spans)


def _single_colon_parts(source: str) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []
    start = 0
    brace = bracket = paren = 0
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\\" and not _is_escaped(source, index):
            index += 2
            continue
        if char == "{": brace += 1
        elif char == "}": brace = max(0, brace - 1)
        elif char == "[": bracket += 1
        elif char == "]": bracket = max(0, bracket - 1)
        elif char == "(": paren += 1
        elif char == ")": paren = max(0, paren - 1)
        elif char == ":" and brace == bracket == paren == 0:
            if (index > 0 and source[index - 1] == ":") or (index + 1 < len(source) and source[index + 1] == ":"):
                index += 1
                continue
            output.append((source[start:index], start, index))
            start = index + 1
        index += 1
    if output:
        output.append((source[start:], start, len(source)))
    return output


def _make_weight(token: str, context: str, start: int, end: int, *, scope: str, inferred: bool = False) -> WeightValue:
    value = float(token)
    valid = math.isfinite(value)
    message = "" if valid else "Weight must be finite."
    if scope in {"group_member", "sequence_local"} and value < 0:
        valid = False
        message = f"Negative {scope.replace('_', ' ')} weight is not supported."
    elif inferred and context.startswith("legacy_"):
        target = "sequence-local" if scope == "sequence_local" else "sequence outer"
        message = f"Legacy compatibility inference: terminal non-integer numeric interpreted as {target} weight."
    return WeightValue(token=token, context=context, source_start=start, source_end=end, inferred=inferred, valid=valid, message=message, weight=value, scope=scope)


def collect_numeric_semantics(source: str) -> tuple[NumericSemantic, ...]:
    normalized, _warnings = normalize_legacy_structured_source(str(source or ""))
    found: list[NumericSemantic] = []
    occupied: list[tuple[int, int]] = []

    for match in _INVALID_SCHEDULE_RE.finditer(normalized):
        start, end = match.span("token")
        token = match.group("token")
        found.append(InvalidNumeric(token=token, context="a1111_schedule", source_start=start, source_end=end, valid=False, message="Schedule boundary must be a finite numeric value.", raw_value=token, expected_kind="schedule_boundary"))
        occupied.append((start, end))
    for match in _INVALID_ATTENTION_RE.finditer(normalized):
        start, end = match.span("token")
        token = match.group("token")
        found.append(InvalidNumeric(token=token, context="attention", source_start=start, source_end=end, valid=False, message="Attention weight must be a finite numeric value.", raw_value=token, expected_kind="weight"))
        occupied.append((start, end))

    # A1111 schedule grammar owns its boundary, regardless of number spelling.
    for match in _SCHEDULE_RE.finditer(normalized):
        token = match.group("token")
        start, end = match.span("token")
        if match.group("percent"):
            percent = float(token)
            valid = 0.0 < percent <= 100.0 and math.isfinite(percent)
            found.append(PercentBoundary(token=f"{token}%", context="a1111_schedule", source_start=start, source_end=end + 1, valid=valid, message="" if valid else "Schedule percent must be > 0 and <= 100.", percent=percent))
        elif re.fullmatch(_INTEGER, token):
            step = int(token)
            valid = step > 0
            found.append(AbsoluteStep(token=token, context="a1111_schedule", source_start=start, source_end=end, valid=valid, message="" if valid else "Absolute schedule step must be > 0.", step=step))
        else:
            fraction = float(token)
            valid = 0.0 < fraction <= 1.0 and math.isfinite(fraction)
            found.append(FractionBoundary(token=token, context="a1111_schedule", source_start=start, source_end=end, valid=valid, message="" if valid else "Schedule fraction must be > 0 and <= 1.", fraction=fraction))
        occupied.append((start, end))

    # Explicit attention grammar always owns a weight.
    for match in _ATTENTION_RE.finditer(normalized):
        start, end = match.span("token")
        if any(a <= start < b for a, b in occupied):
            continue
        found.append(_make_weight(match.group("token"), "attention", start, end, scope="attention"))
        occupied.append((start, end))

    # Count syntax is quantity. PPSR-05 records it even when execution remains
    # parser-specific/dynamic-prompt behavior.
    for match in _QUANTITY_RE.finditer(normalized):
        token = match.group("token")
        start, end = match.span("token")
        quantity = int(token)
        valid = quantity > 0
        found.append(QuantityValue(token=token, context="choice_quantity", source_start=start, source_end=end, valid=valid, message="" if valid else "Quantity must be > 0.", quantity=quantity))
        occupied.append((start, end))

    # Group-member suffixes own relative group weights. Integer spelling does
    # not turn them into steps.
    for group_start, group_end in _group_spans(normalized):
        inner_start = group_start + 1
        inner = normalized[inner_start:group_end - 1]
        for item, local_start, _local_end in _split_top_level_with_spans(inner):
            match = _END_WEIGHT_RE.match(item.strip())
            if not match:
                continue
            token = match.group("token")
            stripped_offset = len(item) - len(item.lstrip())
            token_start = inner_start + local_start + stripped_offset + match.start("token")
            token_end = inner_start + local_start + stripped_offset + match.end("token")
            if any(a <= token_start < b for a, b in occupied):
                continue
            found.append(_make_weight(token, "group_member", token_start, token_end, scope="group_member"))
            occupied.append((token_start, token_end))

    # AND branch suffixes own top-level composable weights.
    branches = [item for item in split_top_level_and(normalized) if item.strip()]
    if len(branches) > 1:
        search_from = 0
        for branch in branches:
            branch_index = normalized.find(branch, search_from)
            search_from = max(search_from, branch_index + len(branch))
            match = _END_WEIGHT_RE.match(branch.strip())
            if not match:
                continue
            token = match.group("token")
            stripped_offset = len(branch) - len(branch.lstrip())
            token_start = branch_index + stripped_offset + match.start("token")
            token_end = branch_index + stripped_offset + match.end("token")
            if any(a <= token_start < b for a, b in occupied):
                continue
            found.append(_make_weight(token, "AND", token_start, token_end, scope="and_branch"))
            occupied.append((token_start, token_end))

    # Explicit structured outer weights are weights even when integer-spelled.
    structured_outer = _STRUCTURED_OUTER_WEIGHT_RE.match(normalized.strip())
    if structured_outer:
        token = structured_outer.group("token")
        base_offset = normalized.find(token, len(structured_outer.group("body")))
        if base_offset >= 0 and not any(a <= base_offset < b for a, b in occupied):
            found.append(_make_weight(token, "structured_outer_weight", base_offset, base_offset + len(token), scope="structured_outer"))
            occupied.append((base_offset, base_offset + len(token)))

    # The legacy [sequence]:modifier wrapper retains its historical heuristic,
    # but the inference is explicitly typed and quarantined to this grammar.
    wrapper = _LEGACY_WRAPPER_RE.match(normalized.strip())
    if wrapper and ":" in wrapper.group("body"):
        token = wrapper.group("token")
        start = normalized.rfind(token)
        end = start + len(token)
        if not any(a <= start < b for a, b in occupied):
            if re.fullmatch(_INTEGER, token) and int(token) > 0:
                found.append(AbsoluteStep(token=token, context="legacy_sequence_wrapper_suffix", source_start=start, source_end=end, inferred=True, step=int(token), message="Legacy compatibility inference: wrapper terminal positive integer interpreted as active-until step."))
            else:
                found.append(_make_weight(token, "legacy_sequence_wrapper_suffix", start, end, scope="sequence_outer", inferred=True))
            occupied.append((start, end))

    # Historical compatibility is intentionally quarantined here. Only a
    # terminal numeric suffix in the legacy single-colon sequence grammar may
    # use integer=>steps / otherwise=>weight inference.
    parts = _single_colon_parts(normalized)
    if len(parts) >= 3:
        terminal, start, end = parts[-1]
        token = terminal.strip()
        token_start = start + (len(terminal) - len(terminal.lstrip()))
        token_end = token_start + len(token)
        if re.fullmatch(_NUMBER, token) and not any(a <= token_start < b for a, b in occupied):
            if re.fullmatch(_INTEGER, token) and int(token) > 0:
                found.append(AbsoluteStep(token=token, context="legacy_single_colon_suffix", source_start=token_start, source_end=token_end, inferred=True, step=int(token), message="Legacy compatibility inference: terminal positive integer interpreted as active-until step."))
            else:
                found.append(_make_weight(token, "legacy_single_colon_suffix", token_start, token_end, scope="sequence_local", inferred=True))

    return tuple(sorted(found, key=lambda item: ((item.source_start if item.source_start is not None else 10**9), item.context, item.kind)))


def numeric_semantic_from_dict(payload: Mapping[str, Any]) -> NumericSemantic:
    data = dict(payload or {})
    kind = str(data.get("type") or "")
    common = dict(
        token=str(data.get("token") or ""),
        context=str(data.get("context") or ""),
        source_start=data.get("source_start"),
        source_end=data.get("source_end"),
        inferred=bool(data.get("inferred", False)),
        valid=bool(data.get("valid", True)),
        message=str(data.get("message") or ""),
    )
    value = data.get("value")
    if kind == "weight":
        return WeightValue(**common, weight=float(value), scope=str(data.get("scope") or "generic"))
    if kind == "absolute_step":
        return AbsoluteStep(**common, step=int(value))
    if kind == "fraction_boundary":
        return FractionBoundary(**common, fraction=float(value))
    if kind == "percent_boundary":
        return PercentBoundary(**common, percent=float(value))
    if kind == "quantity":
        return QuantityValue(**common, quantity=int(value))
    if kind == "invalid_numeric":
        return InvalidNumeric(**common, raw_value=str(value), expected_kind=str(data.get("expected_kind") or "number"))
    raise ValueError(f"Unsupported numeric semantic type: {kind!r}")


def numeric_label(item: NumericSemantic) -> str:
    if isinstance(item, WeightValue):
        if item.scope == "group_member": return f"Group member weight: {item.value:g}"
        if item.scope == "and_branch": return f"AND branch weight: {item.value:g}"
        if item.scope == "sequence_local": return f"Legacy sequence weight: {item.value:g}"
        if item.scope == "sequence_outer": return f"Legacy sequence outer weight: {item.value:g}"
        if item.scope == "structured_outer": return f"Structured outer weight: {item.value:g}"
        return f"Weight: {item.value:g}"
    if isinstance(item, AbsoluteStep):
        prefix = "Legacy inferred step" if item.inferred else "Absolute step"
        return f"{prefix}: {item.value}"
    if isinstance(item, FractionBoundary): return f"Fraction boundary: {item.value:g}"
    if isinstance(item, PercentBoundary): return f"Percent boundary: {item.value:g}%"
    if isinstance(item, QuantityValue): return f"Quantity: {item.value}"
    if isinstance(item, InvalidNumeric): return f"INVALID {item.expected_kind}: {item.value} ({item.message})"
    return f"{item.kind}: {item.value}"
