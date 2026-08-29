from __future__ import annotations

"""Deterministic Classic schedule/alternate compilation for PPSR-06.

This module intentionally operates on encoder-visible text after structural
PromptIR lowering. It resolves only unescaped A1111-style bracket temporal
syntax and never consumes parser-specific extension constructs.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Iterable

TEMPORAL_CONTRACT_VERSION = "image-gen-temporal-semantics-v1"
TEMPORAL_COMPILE_LIMIT = 256


@dataclass(frozen=True)
class TemporalSegment:
    start_step: int
    end_step: int
    text: str

    def to_dict(self):
        return {"start_step": self.start_step, "end_step": self.end_step, "text": self.text}


@dataclass(frozen=True)
class TemporalCompileResult:
    source: str
    total_steps: int
    per_step_text: tuple[str, ...]
    segments: tuple[TemporalSegment, ...]
    has_temporal: bool
    fallback_used: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    contract: str = TEMPORAL_CONTRACT_VERSION

    def to_dict(self):
        return {
            "contract": self.contract,
            "source": self.source,
            "total_steps": self.total_steps,
            "has_temporal": self.has_temporal,
            "fallback_used": self.fallback_used,
            "warnings": list(self.warnings),
            "segments": [item.to_dict() for item in self.segments],
        }


def _is_escaped(text: str, index: int) -> bool:
    count = 0
    i = index - 1
    while i >= 0 and text[i] == "\\":
        count += 1
        i -= 1
    return bool(count % 2)


def _matching_bracket(text: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch in "[]" and _is_escaped(text, i):
            i += 1
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_level(text: str, separator: str) -> list[str]:
    out: list[str] = []
    start = 0
    square = curly = paren = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "[]{}()|:" and _is_escaped(text, i):
            i += 1
            continue
        if ch == "[": square += 1
        elif ch == "]": square = max(0, square - 1)
        elif ch == "{": curly += 1
        elif ch == "}": curly = max(0, curly - 1)
        elif ch == "(": paren += 1
        elif ch == ")": paren = max(0, paren - 1)
        elif ch == separator and square == 0 and curly == 0 and paren == 0:
            out.append(text[start:i])
            start = i + 1
        i += 1
    out.append(text[start:])
    return out


def _parse_boundary(token: str, total_steps: int) -> int | None:
    raw = token.strip()
    if not raw:
        return None
    try:
        if raw.endswith("%"):
            pct = float(raw[:-1])
            if not math.isfinite(pct) or pct <= 0:
                return None
            return max(1, min(total_steps, int(round(total_steps * pct / 100.0))))
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    if value <= 1.0 and ("." in raw or "e" in raw.lower()):
        return max(1, min(total_steps, int(round(total_steps * value))))
    return max(1, min(total_steps, int(round(value))))


def _classify(content: str, total_steps: int):
    pipes = _split_top_level(content, "|")
    if len(pipes) > 1:
        return ("alternate", pipes)
    parts = _split_top_level(content, ":")
    if len(parts) >= 3:
        boundary = _parse_boundary(parts[-1], total_steps)
        if boundary is not None:
            before = ":".join(parts[:-2])
            after = parts[-2]
            return ("schedule", before, after, boundary)
    if len(parts) == 2:
        boundary = _parse_boundary(parts[-1], total_steps)
        if boundary is not None:
            # A1111 insertion shorthand: [text:when] == [:text:when].
            return ("schedule", "", parts[0], boundary)
    return None


def contains_temporal_syntax(text: str) -> bool:
    i = 0
    while i < len(text):
        if text[i] == "[" and not _is_escaped(text, i):
            end = _matching_bracket(text, i)
            if end > i:
                content = text[i + 1:end]
                if len(_split_top_level(content, "|")) > 1:
                    return True
                parts = _split_top_level(content, ":")
                if len(parts) >= 2 and _parse_boundary(parts[-1], 100) is not None:
                    return True
                if contains_temporal_syntax(content):
                    return True
                i = end + 1
                continue
        i += 1
    return False


def _resolve_text(text: str, step: int, total_steps: int, budget: list[int]) -> str:
    if budget[0] <= 0:
        raise OverflowError("Temporal semantic compile limit exceeded.")
    budget[0] -= 1
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "[" and not _is_escaped(text, i):
            end = _matching_bracket(text, i)
            if end > i:
                content = text[i + 1:end]
                kind = _classify(content, total_steps)
                if kind and kind[0] == "alternate":
                    options = kind[1]
                    selected = options[(step - 1) % len(options)]
                    out.append(_resolve_text(selected, step, total_steps, budget))
                    i = end + 1
                    continue
                if kind and kind[0] == "schedule":
                    _, before, after, boundary = kind
                    selected = before if step <= boundary else after
                    out.append(_resolve_text(selected, step, total_steps, budget))
                    i = end + 1
                    continue
                # Non-temporal brackets remain literal, but temporal constructs
                # nested inside them are still resolved recursively.
                inner = _resolve_text(content, step, total_steps, budget)
                out.append("[" + inner + "]")
                i = end + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out).replace(r"\[", "[").replace(r"\]", "]")


def _compress(per_step: Iterable[str]) -> tuple[TemporalSegment, ...]:
    values = list(per_step)
    if not values:
        return ()
    segments: list[TemporalSegment] = []
    start = 1
    current = values[0]
    for idx, value in enumerate(values[1:], start=2):
        if value == current:
            continue
        segments.append(TemporalSegment(start, idx - 1, current))
        start = idx
        current = value
    segments.append(TemporalSegment(start, len(values), current))
    return tuple(segments)


def compile_temporal_text(text: str, total_steps: int, *, compile_limit: int = TEMPORAL_COMPILE_LIMIT) -> TemporalCompileResult:
    source = str(text or "")
    total_steps = max(1, int(total_steps))
    if not contains_temporal_syntax(source):
        values = tuple(source for _ in range(total_steps))
        return TemporalCompileResult(source, total_steps, values, _compress(values), False)

    try:
        budget = [max(int(compile_limit), total_steps * 4)]
        values = tuple(_resolve_text(source, step, total_steps, budget) for step in range(1, total_steps + 1))
        return TemporalCompileResult(source, total_steps, values, _compress(values), True)
    except Exception as exc:
        # Deterministic safe fallback: remove only unescaped temporal brackets.
        flattened = re.sub(r"(?<!\\)[\[\]]", "", source).replace(r"\[", "[").replace(r"\]", "]")
        values = tuple(flattened for _ in range(total_steps))
        return TemporalCompileResult(
            source,
            total_steps,
            values,
            _compress(values),
            True,
            fallback_used=True,
            warnings=(f"Temporal compile fallback: {type(exc).__name__}: {exc}",),
        )
