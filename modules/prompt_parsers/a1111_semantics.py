from __future__ import annotations

"""A1111-compatible prompt semantics selected by the PPSR-10B profile.

This module is intentionally independent from the historical ImageGen Classic
schedule compiler.  Selecting A1111 Compatible changes only the algorithms
recorded by that prompt-style profile; Legacy/ImageGen profiles keep their
existing behavior.
"""

from dataclasses import dataclass
import math
import re

from modules.prompt_parsers.temporal_semantics import TemporalCompileResult, TemporalSegment

A1111_ATTENTION_ALGORITHM = "a1111_attention_v1"
A1111_SCHEDULE_ALGORITHM = "a1111_schedule_v1"
A1111_ALTERNATE_ALGORITHM = "a1111_alternate_v1"
A1111_CLIP_CHUNK_ALGORITHM = "a1111_clip_chunk_v1"
A1111_TEMPORAL_CONTRACT = "a1111-temporal-semantics-v1"
A1111_ROUND_MULTIPLIER = 1.1
A1111_SQUARE_MULTIPLIER = 1.0 / 1.1
A1111_CLIP_CONTENT_TOKENS = 75
A1111_CLIP_ENCODED_TOKENS = 77

_BREAK_RE = re.compile(r"\s*\bBREAK\b\s*", re.S)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass(frozen=True)
class A1111AttentionSpan:
    text: str
    weight: float


def _merge_attention_spans(spans: list[list[object]]) -> tuple[A1111AttentionSpan, ...]:
    if not spans:
        return (A1111AttentionSpan("", 1.0),)
    merged: list[list[object]] = []
    for text, weight in spans:
        text = str(text)
        weight = float(weight)
        if merged and float(merged[-1][1]) == weight:
            merged[-1][0] = str(merged[-1][0]) + text
        else:
            merged.append([text, weight])
    return tuple(A1111AttentionSpan(str(text), float(weight)) for text, weight in merged)


def parse_a1111_attention(text: str) -> tuple[A1111AttentionSpan, ...]:
    """Parse A1111 attention punctuation and BREAK sentinels.

    Escaped ``()[]\\`` remain literal.  Unclosed round/square groups retain the
    same multiplier behavior as the A1111 compatibility reference corpus.
    """

    source = str(text or "")
    spans: list[list[object]] = []
    round_stack: list[int] = []
    square_stack: list[int] = []

    def append_plain(value: str, weight: float = 1.0) -> None:
        if value:
            spans.append([value, float(weight)])

    def multiply_from(start: int, multiplier: float) -> None:
        for index in range(start, len(spans)):
            spans[index][1] = float(spans[index][1]) * float(multiplier)

    index = 0
    plain_start = 0

    def flush_plain(end: int) -> None:
        nonlocal plain_start
        if end <= plain_start:
            return
        value = source[plain_start:end]
        parts = _BREAK_RE.split(value)
        if len(parts) == 1:
            append_plain(value)
            return
        for part_index, part in enumerate(parts):
            if part_index:
                spans.append(["BREAK", -1.0])
            append_plain(part)

    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source) and source[index + 1] in "()[]\\":
            flush_plain(index)
            append_plain(source[index + 1])
            index += 2
            plain_start = index
            continue
        if char == "(":
            flush_plain(index)
            round_stack.append(len(spans))
            index += 1
            plain_start = index
            continue
        if char == "[":
            flush_plain(index)
            square_stack.append(len(spans))
            index += 1
            plain_start = index
            continue
        if char == ":" and round_stack:
            close = source.find(")", index + 1)
            if close != -1:
                candidate = source[index + 1 : close].strip()
                if _NUMBER_RE.fullmatch(candidate):
                    flush_plain(index)
                    multiply_from(round_stack.pop(), float(candidate))
                    index = close + 1
                    plain_start = index
                    continue
        if char == ")" and round_stack:
            flush_plain(index)
            multiply_from(round_stack.pop(), A1111_ROUND_MULTIPLIER)
            index += 1
            plain_start = index
            continue
        if char == "]" and square_stack:
            flush_plain(index)
            multiply_from(square_stack.pop(), A1111_SQUARE_MULTIPLIER)
            index += 1
            plain_start = index
            continue
        index += 1

    flush_plain(len(source))
    for start in round_stack:
        multiply_from(start, A1111_ROUND_MULTIPLIER)
    for start in square_stack:
        multiply_from(start, A1111_SQUARE_MULTIPLIER)
    return _merge_attention_spans(spans)


def a1111_plain_text(text: str) -> str:
    """Return encoder-visible text with attention syntax consumed."""

    return "".join(span.text for span in parse_a1111_attention(text) if span.weight >= 0.0)


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return bool(slash_count % 2)


def _matching_bracket(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in "[]" and _is_escaped(text, index):
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _split_top_level(text: str, delimiter: str) -> list[str]:
    out: list[str] = []
    start = 0
    square = paren = 0
    for index, char in enumerate(text):
        if _is_escaped(text, index):
            continue
        if char == "[":
            square += 1
        elif char == "]":
            square = max(0, square - 1)
        elif char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == delimiter and square == 0 and paren == 0:
            out.append(text[start:index])
            start = index + 1
    out.append(text[start:])
    return out


def a1111_schedule_boundary(
    raw_number: str,
    *,
    base_steps: int,
    hires_steps: int | None,
    use_old_scheduling: bool,
) -> tuple[int, int]:
    raw = str(raw_number).strip()
    value = float(raw)

    if hires_steps is None or use_old_scheduling:
        steps = int(base_steps)
        int_offset = 0
        float_offset = 0.0
    else:
        steps = int(hires_steps)
        int_offset = int(base_steps)
        float_offset = 1.0

    if use_old_scheduling:
        if value < 1.0:
            value *= steps
    elif "." in raw:
        value = (value - float_offset) * steps
    else:
        value = value - int_offset

    return min(steps, int(value)), steps


def _classify_temporal(
    content: str,
    *,
    base_steps: int,
    hires_steps: int | None,
    use_old_scheduling: bool,
):
    alternatives = _split_top_level(content, "|")
    if len(alternatives) > 1:
        return ("alternate", alternatives)
    pieces = _split_top_level(content, ":")
    if len(pieces) == 2 and _NUMBER_RE.fullmatch(pieces[-1].strip()):
        boundary, steps = a1111_schedule_boundary(
            pieces[-1],
            base_steps=base_steps,
            hires_steps=hires_steps,
            use_old_scheduling=use_old_scheduling,
        )
        return ("schedule", "", pieces[0], boundary, steps)
    if len(pieces) >= 3 and _NUMBER_RE.fullmatch(pieces[-1].strip()):
        boundary, steps = a1111_schedule_boundary(
            pieces[-1],
            base_steps=base_steps,
            hires_steps=hires_steps,
            use_old_scheduling=use_old_scheduling,
        )
        return ("schedule", ":".join(pieces[:-2]), pieces[-2], boundary, steps)
    return None


def _resolve_temporal_text(
    text: str,
    *,
    step: int,
    base_steps: int,
    hires_steps: int | None,
    use_old_scheduling: bool,
) -> str:
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "[" and not _is_escaped(text, cursor):
            end = _matching_bracket(text, cursor)
            if end != -1:
                content = text[cursor + 1 : end]
                classified = _classify_temporal(
                    content,
                    base_steps=base_steps,
                    hires_steps=hires_steps,
                    use_old_scheduling=use_old_scheduling,
                )
                if classified and classified[0] == "alternate":
                    choices = classified[1]
                    selected = choices[(step - 1) % len(choices)]
                    out.append(
                        _resolve_temporal_text(
                            selected,
                            step=step,
                            base_steps=base_steps,
                            hires_steps=hires_steps,
                            use_old_scheduling=use_old_scheduling,
                        )
                    )
                    cursor = end + 1
                    continue
                if classified and classified[0] == "schedule":
                    _, before, after, boundary, _steps = classified
                    selected = before if step <= boundary else after
                    out.append(
                        _resolve_temporal_text(
                            selected,
                            step=step,
                            base_steps=base_steps,
                            hires_steps=hires_steps,
                            use_old_scheduling=use_old_scheduling,
                        )
                    )
                    cursor = end + 1
                    continue
        out.append(text[cursor])
        cursor += 1
    return "".join(out).replace(r"\[", "[").replace(r"\]", "]")


def compile_a1111_temporal_text(
    text: str,
    *,
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
) -> TemporalCompileResult:
    """Compile current A1111 schedule/alternate semantics for one active pass."""

    source = str(text or "")
    total = int(base_steps) if hires_steps is None or use_old_scheduling else int(hires_steps)
    total = max(1, total)
    values = tuple(
        _resolve_temporal_text(
            source,
            step=step,
            base_steps=int(base_steps),
            hires_steps=hires_steps,
            use_old_scheduling=bool(use_old_scheduling),
        )
        for step in range(1, total + 1)
    )
    segments: list[TemporalSegment] = []
    current = values[0]
    for step, value in enumerate(values[1:], start=2):
        if value != current:
            segments.append(TemporalSegment(start_step=(segments[-1].end_step + 1 if segments else 1), end_step=step - 1, text=current))
            current = value
    start_step = segments[-1].end_step + 1 if segments else 1
    segments.append(TemporalSegment(start_step=start_step, end_step=total, text=current))
    has_temporal = len(segments) > 1 or (values and values[0] != source)
    return TemporalCompileResult(
        source=source,
        total_steps=total,
        per_step_text=values,
        segments=tuple(segments),
        has_temporal=bool(has_temporal),
        fallback_used=False,
        warnings=(),
        contract=A1111_TEMPORAL_CONTRACT,
    )
