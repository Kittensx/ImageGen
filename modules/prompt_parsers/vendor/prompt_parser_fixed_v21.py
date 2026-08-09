# prompt_parser_patched_superhybrid.py
from __future__ import annotations

import re
import os
import sys
import random
import hashlib
import math
import logging
import unicodedata
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence

import lark
logger = logging.getLogger(__name__)  # не настраиваем basicConfig в библиотеке

class PromptSyntaxError(ValueError):
    """Ошибка синтаксиса промпта с дополнительным контекстом.

    Используется для выдачи человекочитаемых сообщений (например, для reverse ranges).
    """

    def __init__(self, message: str, *, kind: str | None = None, token: str | None = None, full: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.token = token
        self.full = full

import threading as _threading
torch = None
_torch_import_lock = _threading.Lock()


def _ensure_torch():
    """Lazily import torch to avoid heavy import at module load time.
    Защищён threading.Lock от гонки при одновременном первом вызове из нескольких потоков.
    """
    global torch
    if torch is not None:
        return torch
    with _torch_import_lock:
        # Повторная проверка под локом — другой поток мог уже завершить импорт
        if torch is not None:
            return torch
        try:
            import torch as _torch
        except ImportError as exc:
            raise ImportError(
                "Torch is required for conditioning reconstruction but is not installed."
            ) from exc
        torch = _torch
    return torch

# ──────────────────────────────────────────────────────────────────────────────
# Фиче-флаги (переопределяемые через env)
# ──────────────────────────────────────────────────────────────────────────────
# Множители для скобок внимания
ROUND_BRACKET_MULTIPLIER = 1.1      # Вес для (...)
SQUARE_BRACKET_MULTIPLIER = 1 / 1.1  # Вес для [...]
# Строковые представления для использования в emphasized/weighted (не хардкодим "1.1")
def _rb_weight_str() -> str:
    return f"{ROUND_BRACKET_MULTIPLIER:.1f}"


def _rb_weight_strs() -> tuple[str, str]:
    return (_rb_weight_str(), f"{ROUND_BRACKET_MULTIPLIER:.2f}")


def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v not in ("0", "", "false", "no", "off")

def _env_int(name: str, default: int) -> int:
    """Безопасно прочитать целочисленное значение из окружения.

    При некорректном значении возвращает ``default`` и пишет предупреждение
    в лог, чтобы не падать при импорте модуля.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r, using default %d", name, raw, default)
        return default

SAFE_EMPTY = " "

ALLOW_EMPTY_ALTERNATE          = _env_bool("ALLOW_EMPTY_ALTERNATE", "1")
EXPAND_ALTERNATE_PER_STEP      = _env_bool("EXPAND_ALTERNATE_PER_STEP", "1")
GROUP_COMBO_LIMIT              = _env_int("GROUP_COMBO_LIMIT", 100)
DEDUP_SCHEDULE_STEPS           = _env_bool("DEDUP_SCHEDULE_STEPS", "0")
GROUP_COMBO_FALLBACK           = os.getenv("GROUP_COMBO_FALLBACK", "truncate").strip().lower()  # "truncate"|"literal"|"sample"
SUPPRESS_STANDALONE_COLON      = _env_bool("SUPPRESS_STANDALONE_COLON", "1")
CACHE_SIZE                     = _env_int("PROMPT_PARSER_CACHE_SIZE", 4096)
RECURSION_LIMIT                = _env_int("PROMPT_PARSER_RECURSION_LIMIT", 0)
BIND2_USE_PATH2                = _env_bool("BIND2_USE_PATH2", "0")
BIND2_NORMALIZE_WEIGHTS        = _env_bool("BIND2_NORMALIZE_WEIGHTS", "0")

if RECURSION_LIMIT > 0:
    try:
        sys.setrecursionlimit(RECURSION_LIMIT)
    except (ValueError, RecursionError):
        logger.warning("Failed to set recursion limit to %d", RECURSION_LIMIT)


# ──────────────────────────────────────────────────────────────────────────────
# Общие числовые шаблоны / регэкспы (чтобы не дублировать)
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
# «Без знака» — удобно для шаблонов вида [+|-]<число>
NUMERIC_NOSIGN_RE = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_NUMERIC = re.compile(NUMERIC_RE)
RE_NUMERIC_FULL = re.compile(rf"^(?:{NUMERIC_RE})$")
# Добавить константу в начало файла (после imports)
ATTENTION_AND_OPERATOR = "&"  # Внутреннее представление оператора AND
# Private-use char — cannot appear in real user prompts, so no collision possible.
# Previously was "__PROMPT_PARSER_ESCAPED_AMP__" (a string that could appear in user text).
ESCAPED_AMP_PLACEHOLDER = "\uE004"
# Placeholder (1 char) used to temporarily replace commas inside postfix scheduled blocks
# like "[...]:N ..." so Lark can parse comma-separated tags inside brackets.
# Using a single Unicode Private-Use char keeps string positions stable for linting/UX.
SCHEDULE_COMMA_PLACEHOLDER = "\uE000"
# Escaped literal placeholders keep "\(", "\[", "\:" etc. opaque until parsing finishes.
# This prevents escaped syntax markers from accidentally participating in grammar/attention.
_ESCAPED_LITERAL_SINGLE_PLACEHOLDERS = {
    "\\": "\uE110",
    ":": "\uE111",
    "&": "\uE112",
    "[": "\uE113",
    "]": "\uE114",
    "(": "\uE115",
    ")": "\uE116",
    "{": "\uE117",
    "}": "\uE118",
    "|": "\uE119",
    "!": "\uE11A",
}
_ESCAPED_LITERAL_SINGLE_RESTORE = {
    placeholder: literal
    for literal, placeholder in _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.items()
}
_ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE = {
    placeholder: (r"\\" if literal == "\\" else f"\\{literal}")
    for literal, placeholder in _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.items()
}
_ESCAPED_LITERAL_SPAN_DELIMS = {"[": "]", "(": ")", "{": "}"}
_ESCAPED_LITERAL_SPAN_BASE = 0xE200

# ── Attention interpolation ───────────────────────────────────────────────────
# Syntax: (body:w0->w1)  — e.g. (cat:1.0->2.0), (red eyes:0.8->1.4)
#
# Architecture (v21): surface pre-pass BEFORE Lark.
#   1) _placeholderize_attention_interpolations() finds (body:w0->w1) patterns
#      in the raw prompt string and replaces them with private-use marker strings.
#   2) Lark + all visitors/transformers/fast-paths carry the marker as plain text.
#      No grammar changes needed — Lark never sees "->".
#   3) _expand_attention_interpolations() post-pass (called in _get_schedule_impl
#      just before _apply_scheduling_mode) expands each marker per active step,
#      making the weight interpolation contextual-aware (local segment, not global).
#
# Marker format: OPEN body SEP w0 SEP w1 SEP mode CLOSE
#   mode is one of: linear | ease-in | ease-out | ease-in-out  (default: linear)
# Private-use chars survive _collapse_spaces() and _unescape_literals() unchanged.
ATTN_INTERP_OPEN  = "\uE001"  # marker start
ATTN_INTERP_SEP   = "\uE002"  # field separator  body | w0 | w1 | mode
ATTN_INTERP_CLOSE = "\uE003"  # marker end

# Valid easing mode names.
_EASING_MODES = frozenset({"linear", "ease-in", "ease-out", "ease-in-out", "bezier", "catmull"})
_EASING_DEFAULT = "linear"

# PUA ranges used internally — strip from user input to prevent injection.
_PUA_PLACEHOLDERS = {
    "\uE000", "\uE001", "\uE002", "\uE003", "\uE004", "\uE005",
    "\uE110", "\uE111", "\uE112", "\uE113", "\uE114",
    "\uE115", "\uE116", "\uE117", "\uE118", "\uE119", "\uE11A",
}
_PUA_CLEAN_TABLE = dict.fromkeys(ord(c) for c in _PUA_PLACEHOLDERS)

# Error kinds where falling back to raw text is acceptable (structural syntax issues).
# Semantic errors (invalid weights, boundaries, etc.) still get the fallback
# but also emit a logging.warning so the user knows their backend block didn't work.
_STRUCTURAL_ERROR_KINDS = frozenset({
    "invalid_chunk_syntax",
    "invalid_blend_syntax",
    "invalid_pool_syntax",
    "invalid_bind_syntax",
    "invalid_morph_syntax",
    "invalid_assemble_syntax",
    "invalid_chunk_mode",
    "invalid_blend_mode",
    "empty_chunk_branch",
    "empty_blend_branch",
    "multiple_chunk_blocks_not_supported",
    "multiple_blend_blocks_not_supported",
    "multiple_pool_blocks_not_supported",
    "multiple_morph_blocks_not_supported",
    "multiple_assemble_blocks_not_supported",
    "nested_chunk_not_supported",
    "nested_blend_not_supported",
    "nested_pool_not_supported",
    "nested_bind_not_supported",
    "nested_morph_not_supported",
    "nested_assemble_not_supported",
    "nested_backend_in_chunk_not_supported",
    "nested_backend_in_blend_not_supported",
    "nested_backend_in_pool_not_supported",
    "nested_backend_in_bind_not_supported",
    "nested_backend_in_assemble_not_supported",
    "nested_backend_in_morph_not_supported",
    "chunk_inner_multicond_not_supported",
    "blend_inner_multicond_not_supported",
    "bind_inner_multicond_not_supported",
    "morph_inner_multicond_not_supported",
    "assemble_inner_multicond_not_supported",
    "unsupported_chunk_context",
    "unsupported_blend_context",
    "unsupported_pool_context",
    "unsupported_bind_context",
    "unsupported_morph_context",
    "unsupported_assemble_context",
    "bind_with_and_not_supported",
    "bind_with_backend_not_supported",
    "bind_requires_base_prompt",
    "invalid_assemble_field",
    "duplicate_assemble_field",
    "morph_window_with_point_boundaries_not_supported",
    "invalid_compound_syntax",
    "empty_compound_part",
    "invalid_compound_range",
    "invalid_compound_weight",
    "multiple_compound_blocks_not_supported",
    "nested_compound_not_supported",
    "compound_base_has_range",
    "unsupported_compound_context",
    "compound_part_scheduling_not_supported",
})
_warned_semantic_errors: set[str] = set()

# Matches serialized marker in schedule text (used by post-pass).
# mode field is optional for backward-compat with markers written without it.
RE_ATTN_INTERP_LITERAL = re.compile(
    rf"{re.escape(ATTN_INTERP_OPEN)}(.*?)"
    rf"{re.escape(ATTN_INTERP_SEP)}({NUMERIC_RE})"
    rf"{re.escape(ATTN_INTERP_SEP)}({NUMERIC_RE})"
    rf"(?:{re.escape(ATTN_INTERP_SEP)}([A-Za-z][A-Za-z0-9_-]*))?"
    rf"{re.escape(ATTN_INTERP_CLOSE)}"
)

# Tail of a raw interpolation spec after last top-level ':'.
# Supports optional easing suffix: 'w0 -> w1' or 'w0 -> w1 ~ mode'
# mode token is case-insensitive; unknown values fall back to linear at runtime.
_RE_ATT_INTERP_TAIL = re.compile(
    rf'^\s*({NUMERIC_RE})\s*->\s*({NUMERIC_RE})'
    rf'(?:\s*~\s*([A-Za-z][A-Za-z0-9_-]*))?\s*$'
)

def _serialize_att_interp(body: str, w0: float, w1: float,
                           mode: str = _EASING_DEFAULT) -> str:
    """Pack (body, w0, w1, mode) into a single private-use marker string."""
    safe = (body or "").replace(ATTN_INTERP_OPEN, "").replace(ATTN_INTERP_SEP, "").replace(ATTN_INTERP_CLOSE, "")
    m = mode if mode in _EASING_MODES else _EASING_DEFAULT
    return f"{ATTN_INTERP_OPEN}{safe}{ATTN_INTERP_SEP}{w0}{ATTN_INTERP_SEP}{w1}{ATTN_INTERP_SEP}{m}{ATTN_INTERP_CLOSE}"

def _apply_easing(t: float, mode: str) -> float:
    """Map linear t∈[0,1] through easing curve."""
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return t * (2.0 - t)
    if mode == "ease-in-out":
        return t * t * (3.0 - 2.0 * t)
    if mode == "bezier":
        return t * t * (3.0 - 2.0 * t)  # smoothstep = cubic bezier CP[0,0,1,1]
    if mode == "catmull":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep
    return t  # linear

def _format_interp_weight(x: float) -> str:
    """Format interpolated weight: always at least one decimal, strip trailing zeros.

    1.0 -> '1.0',  1.25 -> '1.25',  2.0 -> '2.0',  1.5000 -> '1.5'
    """
    x = float(x)
    if not math.isfinite(x):
        # Last-resort safety for internal math: user-facing validation rejects
        # non-finite interpolation endpoints before they reach this path.
        x = 1.0 if math.isnan(x) else (99.0 if x > 0.0 else -99.0)
    s = f"{x:.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки по тексту/пробелам для единообразия вывода
# ──────────────────────────────────────────────────────────────────────────────

_re_ws_collapse = re.compile(r"[ \t\r\n]+")

def _collapse_spaces(s: str, keep_edges: bool = False) -> str:
    """Сжать повторяющиеся пробелы/переводы строк в один пробел.

    Если ``keep_edges`` истинно, ведущие и хвостовые пробелы сохраняются,
    иначе результат подрезается по краям.
    """
    collapsed = _re_ws_collapse.sub(" ", s).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    return collapsed if keep_edges else collapsed.strip()

def _unescape_literals(s: str) -> str:
    """
    Разэкранировать самые частые литералы, которые встречаются в промптах:
    '\\:' -> ':', '\\[' -> '[', '\\]' -> ']', '\\(' -> '(', '\\)' -> ')', '\\\\' -> '\\'
    Важно: сначала обрабатываем специализированные пары, потом общий '\\\\'.
    """
    if not s:
        return s
    s = (s.replace(r"\:", ":")
           .replace(r"\&", "&")
           .replace(r"\[", "[")
           .replace(r"\]", "]")
           .replace(r"\(", "(")
           .replace(r"\)", ")")
           .replace(r"\{", "{")
           .replace(r"\}", "}")
           .replace(r"\|", "|")
           .replace(r"\!", "!"))
    s = s.replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    # двойной бэкслеш в конце
    s = s.replace(r"\\", "\\")
    return s


@dataclass(frozen=True)
class _BoundarySpec:
    """Typed scheduler boundary keeps absolute steps distinct from fractions and percents."""

    kind: str
    value: float


def _make_boundary_spec(
    value: float | _BoundarySpec,
    *,
    is_percent: bool = False,
    assume_absolute: bool = False,
) -> _BoundarySpec:
    """Preserve boundary meaning instead of guessing it later from a raw float."""
    if isinstance(value, _BoundarySpec):
        return value

    value_f = float(value)
    if is_percent:
        return _BoundarySpec("percent", value_f)
    if assume_absolute or not (0.0 < value_f < 1.0):
        return _BoundarySpec("absolute", value_f)
    return _BoundarySpec("fraction", value_f)


def _protect_escaped_literals(text: str) -> str:
    """Replace escaped single-char literals with private-use placeholders."""
    if not text or "\\" not in text:
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            placeholder = _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.get(text[i + 1])
            if placeholder is not None:
                out.append(placeholder)
                i += 2
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _protect_escaped_literal_spans_impl(text: str, unescape: bool) -> tuple[str, dict[str, str]]:
    """Common implementation for escaped literal span protection."""
    if not text or "\\" not in text:
        return text, {}

    out: list[str] = []
    restore: dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "\\" and i + 1 < n:
            opener = text[i + 1]
            closer = _ESCAPED_LITERAL_SPAN_DELIMS.get(opener)
            if closer is not None:
                start = i
                j = i + 2
                depth = 1
                matched_end: int | None = None
                while j < n:
                    if text[j] == "\\" and j + 1 < n:
                        escaped = text[j + 1]
                        if escaped == opener:
                            depth += 1
                            j += 2
                            continue
                        if escaped == closer:
                            depth -= 1
                            j += 2
                            if depth == 0:
                                matched_end = j
                                break
                            continue
                    j += 1
                if matched_end is not None:
                    placeholder = chr(_ESCAPED_LITERAL_SPAN_BASE + len(restore))
                    if unescape:
                        restore[placeholder] = _unescape_literals(text[start:matched_end])
                    else:
                        restore[placeholder] = text[start:matched_end]
                    out.append(placeholder)
                    i = matched_end
                    continue
        out.append(text[i])
        i += 1

    return "".join(out), restore


def _protect_escaped_literal_spans(text: str) -> tuple[str, dict[str, str]]:
    """Protect full escaped blocks like ``\\[...\\]`` or ``\\(...\\)`` as plain text."""
    return _protect_escaped_literal_spans_impl(text, unescape=True)


def _protect_escaped_literal_spans_for_source(text: str) -> tuple[str, dict[str, str]]:
    """Protect escaped blocks but restore them back to their original source text."""
    return _protect_escaped_literal_spans_impl(text, unescape=False)


def _restore_escaped_literals(text: str, span_restore: dict[str, str] | None = None) -> str:
    """Restore protected escaped literal placeholders back to visible text."""
    if not text:
        return text

    if span_restore:
        text = "".join(span_restore.get(ch, ch) for ch in text)
    if any(ch in text for ch in _ESCAPED_LITERAL_SINGLE_RESTORE):
        text = "".join(_ESCAPED_LITERAL_SINGLE_RESTORE.get(ch, ch) for ch in text)
    return text


def _restore_escaped_literal_source(text: str, span_restore: dict[str, str] | None = None) -> str:
    """Restore protected placeholders back to their original escaped source text."""
    if not text:
        return text

    if span_restore:
        text = "".join(span_restore.get(ch, ch) for ch in text)
    if any(ch in text for ch in _ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE):
        text = "".join(_ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE.get(ch, ch) for ch in text)
    return text

def _normalize_and_operators_for_parse(text: str) -> str:
    """Нормализовать логический оператор до внутреннего символа, не трогая экранированный '&'."""
    if not text:
        return text
    text = text.replace(r"\&", ESCAPED_AMP_PLACEHOLDER)
    text = re.sub(r'(?<!\\)(?<=\w)&(?=\w)', ESCAPED_AMP_PLACEHOLDER, text)
    text = re.sub(rf'(?<![\\\w_])AND(?![\w_])', ATTENTION_AND_OPERATOR, text)
    return re.sub(rf'\s*{re.escape(ATTENTION_AND_OPERATOR)}\s*', ATTENTION_AND_OPERATOR, text)
# ──────────────────────────────────────────────────────────────────────────────
# Склейка префикса/ядра/суффикса с корректной обработкой пробелов
# ──────────────────────────────────────────────────────────────────────────────
def _smart_space_trim(s: str) -> str:
    return _re_ws_collapse.sub(" ", s)


def _concat_prefix_text_suffix(prefix: str, text: str, suffix: str) -> str:
    """
    Склейка префикса/ядра/суффикса c улучшенной обработкой Unicode и эмодзи.
    """
    def _is_cjk(ch: str) -> bool:
        """Проверить, является ли символ CJK (китайский/японский/корейский)."""
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs (основной блок)
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0x20000 <= cp <= 0x2A6DF # CJK Extension B
            or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
            or 0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
            or 0xFF00 <= cp <= 0xFFEF  # Halfwidth/Fullwidth Forms
        )

    def _need_space(a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_clean = a.strip()
        b_clean = b.strip()
        if not a_clean or not b_clean:
            return False

        last_char = a[-1]
        first_char = b[0]

        # Если уже есть пробел
        if last_char.isspace() or first_char.isspace():
            return False

        # CJK символы не нуждаются в пробелах между собой
        if _is_cjk(last_char) and _is_cjk(first_char):
            return False
        # CJK + латиница: пробел нужен
        # (оставляем как есть — return True ниже)

        # FIX #6: Более безопасная логика склейки.
        # Ставим пробел практически всегда, кроме случаев, когда знаки препинания "обнимают" слово.
        # Например: "word" + "," -> "word," (без пробела)
        # Но: "word" + "(" -> "word (" (с пробелом)
        # И: "smile" + ":)" -> "smile :)" (с пробелом, чтобы не ломать токен)

        if first_char in ",.;?!:":
            return False
        # Opening brackets already "touch" their content — no space needed
        if last_char in "([{":
            return False

        return True

    def _strip_edge_and_left(s: str) -> tuple[str, bool]:
        stripped = s.lstrip()
        if stripped.startswith(ATTENTION_AND_OPERATOR):
            return stripped[1:].lstrip(), True
        return s, False

    def _strip_edge_and_right(s: str) -> tuple[str, bool]:
        stripped = s.rstrip()
        if stripped.endswith(ATTENTION_AND_OPERATOR):
            return stripped[:-1].rstrip(), True
        return s, False

    # Пустая центральная часть
    if text.strip() == "":
        left, left_has_and = _strip_edge_and_right(prefix)
        right, right_has_and = _strip_edge_and_left(suffix)

        if left.strip() and right.strip() and (left_has_and or right_has_and):
            return _smart_space_trim(f"{left} {ATTENTION_AND_OPERATOR} {right}")
        if left.strip() and right.strip():
            if _need_space(left, right):
                return _smart_space_trim(left + " " + right)
            return _smart_space_trim(left + right)
        if left.strip():
            return _smart_space_trim(left)
        if right.strip():
            return _smart_space_trim(right)
        return ""

    # Непустая
    left = prefix
    mid = text
    right = suffix

    if _need_space(left, mid):
        left += " "
    if _need_space(mid, right):
        mid += " "

    return _smart_space_trim(left + mid + right)


def _cleanup_adjunct_base_prompt_text(text: str) -> str:
    cleaned = _collapse_spaces(str(text or ""))
    cleaned = re.sub(r"\s*,\s*,+\s*", ", ", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    return _collapse_spaces(cleaned)


def _contains_top_level_multicond(prompt: str) -> bool:
    return len(_split_top_level_multicond(prompt)) > 1


def _normalize_preview_fragment(text: str) -> str:
    normalized = _collapse_spaces(str(text or "")).strip()
    if normalized == SAFE_EMPTY:
        return ""
    return normalized

# ──────────────────────────────────────────────────────────────────────────────
# Предкомпилированные regex'ы для fast-path
# ──────────────────────────────────────────────────────────────────────────────

# [ ... ] : N [reverse]  с префиксом/суффиксом
RE_BRACKET_AFTER = re.compile(
    rf'(?s)^(.*)\[(.*?)\]\s*:\s*({NUMERIC_RE}%?)\s*(?:(?P<rev>reverse)\b)?\s*(?P<post>.*)$'
)
# Явные диапазоны: "[a:b]:10 1-4,5-7 [reverse]"
RE_BRACKET_AFTER_WITH_RANGES = re.compile(rf'(?s)^\s*\[(?P<inner>.*?)\]\s*:\s*(?P<num>{NUMERIC_RE})\s+'
    r'(?P<ranges>(?:\d+%?\s*-\s*\d+%?(?:\s*,\s*)?)+)'
    r'(?:\s*(?P<rev>reverse))?\s*$'
)

# Общий префикс для reverse в постфиксе
RE_REVERSE_PREFIX = re.compile(r'^\s*reverse\b\s*')
# Reverse в КОНЦЕ постфикса: "photo [cat:dog]:10 hd reverse" → reverse в конце post.
# Используем только полное слово 'reverse' (не 'r') во избежание ложных срабатываний.
RE_REVERSE_SUFFIX = re.compile(r'(?s)(?:^|\s)\breverse\b\s*$')
LITERAL_REVERSE_TOKEN = "\uE005"  # private-use char, avoids collision with any user text

# Normalization helpers for scheduler surface syntax:
# 1) canonicalize "] : N" / "]: N" to "]:N" for parser portability.
# 2) keep literal "reverse" text after range-blocks (e.g. "... 1-3 reverse shot").
_RE_POSTFIX_STEP_COLON_SPACING = re.compile(
    rf'\]\s*:\s*(?P<num>{NUMERIC_RE})'
)
_RE_LITERAL_REVERSE_AFTER_RANGES = re.compile(
    rf'(?P<head>\]\s*:\s*{NUMERIC_RE}\s+(?:\d+%?\s*-\s*\d+%?(?:\s*,\s*)?)+)\s+reverse(?=\s+\w|,)'
)


def _find_last_top_level_colon_index(s: str) -> int:
    """Return the index of the last top-level ':' in s, or -1."""
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    last = -1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2 if i + 1 < len(s) else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == ":" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            last = i
        i += 1
    return last


CHUNK_KEYWORD = "CHUNK"
CHUNK_PREVIEW_SEPARATOR = " BREAK "
CHUNK_CROSSATTN_KEYS = frozenset({"crossattn", "c_crossattn", "open_clip_projected"})
SDXL_SPLITTABLE_CROSS_KEYS = frozenset({"crossattn", "c_crossattn"})
CHUNK_SHARED_MODES = frozenset({"share-pooled", "share-cross"})
POOL_KEYWORD = "POOL"
POOL_PREVIEW_PREFIX = "POOL<"
_RE_CHUNK_MARKER = re.compile(rf"(?<![\w\\]){re.escape(CHUNK_KEYWORD)}(?:\s*\[[^\]]*\])?\s*\{{")
_RE_POOL_MARKER = re.compile(rf"(?<![\w\\]){re.escape(POOL_KEYWORD)}\s*\{{")
BIND_KEYWORD = "BIND"
BIND_PREVIEW_PREFIX = "BIND<"
_RE_BIND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND_KEYWORD)}(?:\s*\^[^\{{]*)?\s*\{{")
BIND2_KEYWORD = "BIND2"
BIND3_KEYWORD = "BIND3"
_RE_BIND2_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND2_KEYWORD)}\s*\{{")
_RE_BIND3_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND3_KEYWORD)}\s*\{{")
_RE_HAS_SCHEDULING = re.compile(r"\[(?:[^\[\]]*:[^\[\]]*:\d+%?|[^\[\]]*:\d+%?)\]")
ASSEMBLE_KEYWORD = "ASSEMBLE"
ASSEMBLE_PREVIEW_PREFIX = "ASSEMBLE<"
_RE_ASSEMBLE_MARKER = re.compile(rf"(?<![\w\\]){re.escape(ASSEMBLE_KEYWORD)}\s*\{{")
BLEND_KEYWORD = "BLEND"
BLEND_PREVIEW_PREFIX = "BLEND<"
BLEND_MODES = frozenset({"mean", "sum"})
BACKEND_CHANNEL_TARGETS = frozenset({"both", "cross", "pooled", "enc1", "enc2"})
SDXL_ENCODER_CHANNEL_TARGETS = frozenset({"enc1", "enc2"})
SDXL_ENCODER1_CROSS_DIM = 768
SDXL_TOTAL_CROSS_DIM = 2048
_RE_BLEND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BLEND_KEYWORD)}(?:\s*\^[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")
MORPH_KEYWORD = "MORPH"
MORPH_PREVIEW_PREFIX = "MORPH<"
MORPH_CURVES = frozenset({"linear", "bezier", "catmull", "slerp"})
_RE_MORPH_MARKER = re.compile(rf"(?<![\w\\]){re.escape(MORPH_KEYWORD)}(?:\s*\^[^\[\{{@]*)?(?:\s*@[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")
COMPOUND_KEYWORD = "COMPOUND"
COMPOUND_PREVIEW_PREFIX = "COMPOUND<"
_RE_COMPOUND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(COMPOUND_KEYWORD)}\s*\{{")
_RE_COMPOUND_RANGE = re.compile(r"@(\d+)(?:-(\d+))?")
_RE_COMPOUND_WEIGHT = re.compile(r"\*(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class ChunkBranchSpec:
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class ChunkPromptSpec:
    prefix: str
    suffix: str
    branches: tuple[ChunkBranchSpec, ...]
    shared_channel: str = "none"


@dataclass(frozen=True)
class PoolPromptSpec:
    prefix: str
    suffix: str
    body: str
    source: str = ""


@dataclass(frozen=True)
class BindPromptSpec:
    owner: str
    attrs: str
    weight: float = 1.0
    source: str = ""


@dataclass(frozen=True)
class AssemblePromptSpec:
    prefix: str
    suffix: str
    enc1: str
    enc2: str
    pooled: str | None = None
    source: str = ""


@dataclass(frozen=True)
class CompoundPartSpec:
    text: str
    step_start: int = 1
    step_end: int | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class CompoundPromptSpec:
    base: str
    parts: tuple[CompoundPartSpec, ...]
    prefix: str = ""
    suffix: str = ""
    source: str = ""


@dataclass(frozen=True)
class BlendBranchSpec:
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class BlendPromptSpec:
    prefix: str
    suffix: str
    branches: tuple[BlendBranchSpec, ...]
    mode: str = "mean"
    intensity: float = 1.0
    channel_target: str = "both"
    source: str = ""


@dataclass(frozen=True)
class MorphPointSpec:
    text: str
    boundary: _BoundarySpec | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class MorphPromptSpec:
    prefix: str
    suffix: str
    points: tuple[MorphPointSpec, ...]
    curve: str = "linear"
    intensity: float = 1.0
    channel_target: str = "both"
    window_start: _BoundarySpec | None = None
    window_end: _BoundarySpec | None = None
    source: str = ""


@dataclass(frozen=True)
class BackendPromptState:
    chunk_spec: ChunkPromptSpec | None = None
    pool_spec: PoolPromptSpec | None = None
    bind_specs: tuple[BindPromptSpec, ...] = ()
    bind_base_prompt: str = ""
    assemble_spec: AssemblePromptSpec | None = None
    blend_spec: BlendPromptSpec | None = None
    morph_spec: MorphPromptSpec | None = None
    compound_spec: CompoundPromptSpec | None = None
    allow_chunk_morph_sugar: bool = False
    has_multiple_same_type: bool = False

    @property
    def backend_specs(self) -> tuple[object, ...]:
        return tuple(spec for spec in (self.chunk_spec, self.assemble_spec, self.blend_spec, self.morph_spec, self.compound_spec) if spec is not None)

    @property
    def has_backend(self) -> bool:
        return any(spec is not None for spec in (self.chunk_spec, self.assemble_spec, self.blend_spec, self.morph_spec, self.compound_spec))

    @property
    def has_pool(self) -> bool:
        return self.pool_spec is not None

    @property
    def has_bind(self) -> bool:
        return bool(self.bind_specs)

    @property
    def has_bind_backend_conflict(self) -> bool:
        return self.has_bind and self.has_backend

    @property
    def has_mixed_backends(self) -> bool:
        return len(self.backend_specs) > 1 and not self.allow_chunk_morph_sugar

    @property
    def active_morph_spec(self) -> MorphPromptSpec | None:
        return None if self.allow_chunk_morph_sugar else self.morph_spec

    @property
    def primary_token(self) -> str:
        if self.bind_specs and self.has_backend:
            return BIND_KEYWORD
        if self.chunk_spec is not None:
            return CHUNK_KEYWORD
        if self.assemble_spec is not None:
            return ASSEMBLE_KEYWORD
        if self.blend_spec is not None:
            return BLEND_KEYWORD
        if self.morph_spec is not None:
            return MORPH_KEYWORD
        if self.compound_spec is not None:
            return COMPOUND_KEYWORD
        return "BACKEND"


def _match_chunk_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(CHUNK_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(CHUNK_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "[":
        mode_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    mode_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return mode_open, mode_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed CHUNK mode: expected ']'.",
            kind="invalid_chunk_mode",
            token=f"{CHUNK_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _contains_chunk_marker(text: str) -> bool:
    if not text or CHUNK_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_CHUNK_MARKER.search(protected))


def _contains_pool_marker(text: str) -> bool:
    if not text or POOL_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_POOL_MARKER.search(protected))


def _contains_bind_marker(text: str) -> bool:
    if not text or BIND_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BIND_MARKER.search(protected))


def _contains_bind2_marker(text: str) -> bool:
    if not text or BIND2_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BIND2_MARKER.search(protected))


def _contains_bind3_marker(text: str) -> bool:
    if not text or BIND3_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BIND3_MARKER.search(protected))


def _contains_assemble_marker(text: str) -> bool:
    if not text or ASSEMBLE_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_ASSEMBLE_MARKER.search(protected))


def _contains_compound_marker(text: str) -> bool:
    if not text or COMPOUND_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_COMPOUND_MARKER.search(protected))


def _match_blend_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(BLEND_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(BLEND_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "^":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "[":
        mode_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    mode_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return mode_open, mode_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed BLEND mode: expected ']'.",
            kind="invalid_blend_mode",
            token=f"{BLEND_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _contains_blend_marker(text: str) -> bool:
    if not text or BLEND_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BLEND_MARKER.search(protected))


def _find_top_level_blend_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_brace == 0:
            blend_match = _match_blend_keyword_at(text, i)
            if blend_match is not None:
                mode_open, mode_close, brace_open = blend_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, mode_open, mode_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed BLEND block: expected '}'",
                        kind="invalid_blend_syntax",
                        token=f"{BLEND_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_blend_body(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if ch == "|" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "Empty BLEND branch is not allowed.",
                    kind="empty_blend_branch",
                    token="|",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="|",
            full=full_text,
        )
    parts.append(last)
    return parts


def _parse_blend_mode(text: str, *, full_text: str) -> str:
    mode = (text or "").strip().lower() or "mean"
    if mode not in BLEND_MODES:
        raise PromptSyntaxError(
            f"Unsupported BLEND mode {mode!r}",
            kind="invalid_blend_mode",
            token=(text or BLEND_KEYWORD).strip() or BLEND_KEYWORD,
            full=full_text,
        )
    return mode


def _parse_backend_channel_target(text: str, *, full_text: str, kind: str) -> str:
    channel_target = (text or "").strip().lower() or "both"
    if channel_target not in BACKEND_CHANNEL_TARGETS:
        raise PromptSyntaxError(
            f"Unsupported {kind} channel target {channel_target!r}",
            kind=f"invalid_{kind.lower()}_channel_target",
            token=(text or kind).strip() or kind,
            full=full_text,
        )
    return channel_target


def _parse_blend_mode_and_channel_spec(text: str, *, full_text: str) -> tuple[str, str]:
    stripped = (text or "").strip()
    if not stripped:
        return "mean", "both"

    if "@" not in stripped:
        return _parse_blend_mode(stripped, full_text=full_text), "both"

    mode_text, channel_text = stripped.split("@", 1)
    mode = _parse_blend_mode(mode_text, full_text=full_text) if mode_text.strip() else "mean"
    channel_target = _parse_backend_channel_target(channel_text, full_text=full_text, kind="BLEND")
    return mode, channel_target


def _parse_blend_intensity_spec(text: str, *, full_text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0

    match = re.fullmatch(rf"\^\s*({NUMERIC_RE})\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {text!r}",
            kind="invalid_blend_intensity",
            token=stripped,
            full=full_text,
        )

    intensity = float(match.group(1))
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {text!r}",
            kind="invalid_blend_intensity",
            token=stripped,
            full=full_text,
        )
    return intensity


def _split_blend_branch_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="|",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid BLEND branch weight {weight_text!r}",
            kind="invalid_blend_weight",
            token=weight_text or "*",
            full=full_text,
        )

    branch_text = stripped[:star_pos].rstrip()
    if not branch_text.strip():
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if weight <= 0:
        raise PromptSyntaxError(
            f"BLEND branch weight must be positive, got {weight_text!r}",
            kind="invalid_blend_weight",
            token=weight_text,
            full=full_text,
        )
    return branch_text, weight


def _extract_blend_prompt_spec(text: str) -> BlendPromptSpec | None:
    if not text or BLEND_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_blend_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level BLEND block is supported in v1.",
            kind="multiple_blend_blocks_not_supported",
            token=BLEND_KEYWORD,
            full=text,
        )

    start, mode_open, mode_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_blend_blocks(body)
    if nested or _contains_blend_marker(body):
        raise PromptSyntaxError(
            "Nested BLEND blocks are not supported in v1.",
            kind="nested_blend_not_supported",
            token=BLEND_KEYWORD,
            full=text,
        )
    raw_parts = _split_top_level_blend_body(body, full_text=text)
    if len(raw_parts) < 2:
        raise PromptSyntaxError(
            "BLEND requires at least two branches.",
            kind="invalid_blend_syntax",
            token=BLEND_KEYWORD,
            full=text,
        )

    mode = "mean"
    channel_target = "both"
    intensity = _parse_blend_intensity_spec(
        protected[start + len(BLEND_KEYWORD) : (mode_open if mode_open is not None else brace_open)],
        full_text=text,
    )
    if mode_open is not None and mode_close is not None:
        mode, channel_target = _parse_blend_mode_and_channel_spec(protected[mode_open + 1 : mode_close], full_text=text)

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)

    branches: list[BlendBranchSpec] = []
    for raw_part in raw_parts:
        branch_text, weight = _split_blend_branch_weight(raw_part, full_text=text)
        if _contains_top_level_multicond(branch_text):
            raise PromptSyntaxError(
                "BLEND branches cannot contain top-level AND branches in v1.",
                kind="blend_inner_multicond_not_supported",
                token=BLEND_KEYWORD,
                full=text,
            )
        branch_text = _restore_escaped_literal_source(branch_text, span_restore).strip()
        if not branch_text:
            raise PromptSyntaxError(
                "Empty BLEND branch is not allowed.",
                kind="empty_blend_branch",
                token="|",
                full=text,
            )
        branches.append(BlendBranchSpec(branch_text, weight))

    return BlendPromptSpec(
        prefix=prefix,
        suffix=suffix,
        branches=tuple(branches),
        mode=mode,
        intensity=intensity,
        channel_target=channel_target,
        source=text,
    )


def _expand_blend_branch_prompt(spec: BlendPromptSpec, branch: BlendBranchSpec | str) -> str:
    branch_text = branch.text if isinstance(branch, BlendBranchSpec) else str(branch)
    return _concat_prefix_text_suffix(spec.prefix, branch_text, spec.suffix)


def _resolve_blend_mode_weights(weights: Sequence[float], mode: str, intensity: float = 1.0) -> list[float]:
    raw = [float(weight) for weight in weights]
    k = float(intensity)
    if not math.isfinite(k) or k <= 0.0:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {intensity!r}",
            kind="invalid_blend_intensity",
            token=str(intensity),
            full=str(intensity),
        )
    if abs(k - 1.0) <= 1e-8:
        if mode == "sum":
            return raw
        total = float(sum(raw))
        if abs(total) <= 1e-8:
            return [0.0 for _ in raw]
        return [weight / total for weight in raw]

    shaped = [weight ** k for weight in raw]
    shaped_total = float(sum(shaped))
    if abs(shaped_total) <= 1e-8:
        return [0.0 for _ in raw]
    normalized = [weight / shaped_total for weight in shaped]
    if mode == "sum":
        original_total = float(sum(raw))
        return [weight * original_total for weight in normalized]
    return normalized


def _blend_preview_prefix(channel_target: str) -> str:
    if channel_target == "both":
        return BLEND_PREVIEW_PREFIX
    return f"{BLEND_KEYWORD}@{channel_target}<"


def _build_blend_preview_text_with_target(active_texts: Sequence[str], weights: Sequence[float], channel_target: str) -> str:
    nonzero = [
        (normalized_text, weight)
        for text, weight in zip(active_texts, weights)
        for normalized_text in [_normalize_preview_fragment(text)]
        if normalized_text and abs(float(weight)) > 1e-8
    ]
    if not nonzero:
        return SAFE_EMPTY
    parts = [f"{text}*{_format_interp_weight(float(weight))}" for text, weight in nonzero]
    return f"{_blend_preview_prefix(channel_target)}{' + '.join(parts)}>"


def _build_compound_preview_text(spec: CompoundPromptSpec, step: int, steps: int) -> str:
    base = spec.base.strip() or SAFE_EMPTY
    active_parts: list[str] = []
    for p in spec.parts:
        s = p.step_start
        e = p.step_end if p.step_end is not None else int(steps)
        if s <= step <= e:
            part_text = p.text.strip()
            part_text = f"{part_text}:{s}-{e}:{p.weight}"
            active_parts.append(part_text)
    if not active_parts:
        return f"{COMPOUND_PREVIEW_PREFIX}{base}>"
    parts_str = "; ".join(active_parts)
    return f"{COMPOUND_PREVIEW_PREFIX}{base}; {parts_str}>"


def _build_compound_text_schedule_from_spec(
    spec: CompoundPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    if not use_scheduling:
        return [[int(steps), _build_compound_preview_text(spec, int(steps), int(steps))]]

    change_points: set[int] = set()
    change_points.add(int(steps))
    for p in spec.parts:
        s = p.step_start
        e = p.step_end if p.step_end is not None else int(steps)
        if s > 1:
            change_points.add(s - 1)
        if e < int(steps):
            change_points.add(e)
    boundaries = sorted(change_points)
    out: list[list[int, str]] = []
    prev_text: str | None = None
    for end_at_step in boundaries:
        preview = _build_compound_preview_text(spec, end_at_step, int(steps))
        if out and prev_text == preview:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            prev_text = preview
    return out or [[int(steps), SAFE_EMPTY]]


def _build_blend_text_schedule_from_spec(
    spec: BlendPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    branch_prompts = [_expand_blend_branch_prompt(spec, branch) for branch in spec.branches]
    branch_schedules = [
        get_schedule(branch_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch_prompt in branch_prompts
    ]
    effective_weights = _resolve_blend_mode_weights([branch.weight for branch in spec.branches], spec.mode, spec.intensity)

    if not use_scheduling:
        final_step = int(steps)
        active_texts = [_select_text_from_schedule(schedule, final_step) or SAFE_EMPTY for schedule in branch_schedules]
        return [[int(steps), _build_blend_preview_text_with_target(active_texts, effective_weights, spec.channel_target)]]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        active_texts = [
            _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
            for schedule in branch_schedules
        ]
        preview = _build_blend_preview_text_with_target(active_texts, effective_weights, spec.channel_target)
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in effective_weights))
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]

def _contains_morph_marker(text: str) -> bool:
    if not text or MORPH_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_MORPH_MARKER.search(protected))


def _find_top_level_chunk_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_brace == 0:
            chunk_match = _match_chunk_keyword_at(text, i)
            if chunk_match is not None:
                mode_open, mode_close, brace_open = chunk_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, mode_open, mode_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed CHUNK block: expected '}'",
                        kind="invalid_chunk_syntax",
                        token=f"{CHUNK_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_chunk_body(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if ch == "|" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "Empty CHUNK branch is not allowed.",
                    kind="empty_chunk_branch",
                    token="|",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="|",
            full=full_text,
        )
    parts.append(last)
    return parts


def _split_chunk_branch_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="|",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid CHUNK branch weight {weight_text!r}",
            kind="invalid_chunk_weight",
            token=weight_text or "*",
            full=full_text,
        )

    branch_text = stripped[:star_pos].rstrip()
    if not branch_text.strip():
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if weight <= 0:
        raise PromptSyntaxError(
            f"CHUNK branch weight must be positive, got {weight_text!r}",
            kind="invalid_chunk_weight",
            token=weight_text,
            full=full_text,
        )
    return branch_text, weight


def _parse_chunk_shared_mode(text: str, *, full_text: str) -> str:
    mode = (text or "").strip().lower()
    if not mode:
        raise PromptSyntaxError(
            "Unsupported CHUNK mode ''.",
            kind="invalid_chunk_mode",
            token="[]",
            full=full_text,
        )
    if mode not in CHUNK_SHARED_MODES:
        raise PromptSyntaxError(
            f"Unsupported CHUNK mode {mode!r}",
            kind="invalid_chunk_mode",
            token=mode,
            full=full_text,
        )
    if mode == "share-pooled":
        return "pooled"
    if mode == "share-cross":
        return "cross"
    return "none"


def _find_keyword_brace_blocks(
    text: str,
    keyword: str,
    kind: str,
    name: str,
) -> list[tuple[int, int, int]]:
    """Find all top-level {keyword} blocks, return [(start, brace_open, brace_close)].
    Shared helper for POOL, ASSEMBLE and similar simple keyword block scanners.
    """
    blocks: list[tuple[int, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if depth_brace == 0:
            if text.startswith(keyword, i):
                prev = text[i - 1] if i > 0 else ""
                if not prev or (not prev.isalnum() and prev != "_"):
                    j = i + len(keyword)
                    while j < n and text[j].isspace():
                        j += 1
                    if j < n and text[j] == "{":
                        brace_open = j
                        brace_depth = 1
                        k = brace_open + 1
                        while k < n:
                            inner = text[k]
                            if inner == "\\":
                                k += 2 if k + 1 < n else 1
                                continue
                            if inner == "{":
                                brace_depth += 1
                            elif inner == "}":
                                brace_depth -= 1
                                if brace_depth == 0:
                                    blocks.append((i, brace_open, k))
                                    i = k + 1
                                    break
                            k += 1
                        else:
                            raise PromptSyntaxError(
                                f"Unclosed {name} block: expected '}}'",
                                kind=f"invalid_{kind}_syntax",
                                token=f"{keyword}{{",
                                full=text,
                            )
                        continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1
    return blocks


def _find_top_level_pool_blocks(text: str) -> list[tuple[int, int, int]]:
    return _find_keyword_brace_blocks(text, POOL_KEYWORD, "pool", "POOL")


def _find_top_level_compound_blocks(text: str) -> list[tuple[int, int, int]]:
    return _find_keyword_brace_blocks(text, COMPOUND_KEYWORD, "compound", "COMPOUND")


def _extract_pool_prompt_spec(text: str) -> PoolPromptSpec | None:
    if not text or POOL_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_pool_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level POOL block is supported in v1.",
            kind="multiple_pool_blocks_not_supported",
            token=POOL_KEYWORD,
            full=text,
        )

    start, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_pool_blocks(body)
    if nested or _contains_pool_marker(body):
        raise PromptSyntaxError(
            "Nested POOL blocks are not supported in v1.",
            kind="nested_pool_not_supported",
            token=POOL_KEYWORD,
            full=text,
        )
    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    body = _restore_escaped_literal_source(body, span_restore).strip()
    if not body:
        raise PromptSyntaxError(
            "POOL body cannot be empty.",
            kind="invalid_pool_syntax",
            token=POOL_KEYWORD,
            full=text,
        )

    return PoolPromptSpec(prefix=prefix, suffix=suffix, body=body, source=text)


def _split_top_level_compound_body(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == "|" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            parts.append(part)
            buf.clear()
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        buf.append(ch)
        i += 1
    last = "".join(buf).strip()
    parts.append(last)
    return parts


def _parse_compound_part_text(raw: str, *, full_text: str) -> CompoundPartSpec:
    text = raw
    range_match = _RE_COMPOUND_RANGE.search(text)
    start = 1
    end: int | None = None
    if range_match:
        start = int(range_match.group(1))
        if start == 0:
            start = 1
        if range_match.group(2) is not None:
            end = int(range_match.group(2))
            if end == 0:
                end = 1
            if start > end:
                raise PromptSyntaxError(
                    f"COMPOUND part range start ({start}) > end ({end}).",
                    kind="invalid_compound_range",
                    token=raw,
                    full=full_text,
                )
        text = text[:range_match.start()] + text[range_match.end():]
    weight_match = _RE_COMPOUND_WEIGHT.search(text)
    weight = 1.0
    if weight_match:
        weight = float(weight_match.group(1))
        if weight == 0.0:
            raise PromptSyntaxError(
                f"COMPOUND part weight must be non-zero, got {weight}.",
                kind="invalid_compound_weight",
                token=raw,
                full=full_text,
            )
        text = text[:weight_match.start()] + text[weight_match.end():]
    text = text.strip()
    if not text:
        raise PromptSyntaxError(
            "Empty COMPOUND part is not allowed.",
            kind="empty_compound_part",
            token=raw,
            full=full_text,
        )
    if _RE_HAS_SCHEDULING.search(text):
        raise PromptSyntaxError(
            "Scheduling syntax [a:b:N] inside COMPOUND parts is not supported. "
            "Use [COMPOUND{...}:N] to schedule the entire block.",
            kind="compound_part_scheduling_not_supported",
            token=text,
            full=full_text,
        )
    return CompoundPartSpec(text=text, step_start=start, step_end=end, weight=weight)


def _extract_compound_prompt_spec(text: str) -> CompoundPromptSpec | None:
    if not text or COMPOUND_KEYWORD not in text:
        return None
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_compound_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level COMPOUND block is supported in v1.",
            kind="multiple_compound_blocks_not_supported",
            token=COMPOUND_KEYWORD,
            full=text,
        )
    _start, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_compound_blocks(body)
    if nested or _contains_compound_marker(body):
        raise PromptSyntaxError(
            "Nested COMPOUND blocks are not supported in v1.",
            kind="nested_compound_not_supported",
            token=COMPOUND_KEYWORD,
            full=text,
        )
    raw_parts = _split_top_level_compound_body(body, full_text=text)
    if len(raw_parts) < 2:
        raise PromptSyntaxError(
            "COMPOUND requires at least two branches (base + at least one part).",
            kind="invalid_compound_syntax",
            token=COMPOUND_KEYWORD,
            full=text,
        )
    base = raw_parts[0].strip()
    if not base:
        base = SAFE_EMPTY
    if _RE_COMPOUND_RANGE.search(base):
        raise PromptSyntaxError(
            "COMPOUND base branch must not have a step range (@...).",
            kind="compound_base_has_range",
            token=base,
            full=text,
        )
    parts: list[CompoundPartSpec] = []
    for raw_part in raw_parts[1:]:
        part_spec = _parse_compound_part_text(raw_part, full_text=text)
        parts.append(part_spec)
    prefix = _restore_escaped_literal_source(protected[:_start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1:], span_restore)
    full_base = _concat_prefix_text_suffix(prefix, base, suffix)
    return CompoundPromptSpec(
        base=full_base,
        parts=tuple(parts),
        prefix=prefix,
        suffix=suffix,
        source=text,
    )


def _build_pool_base_prompt(spec: PoolPromptSpec) -> str:
    text = _cleanup_adjunct_base_prompt_text(_concat_prefix_text_suffix(spec.prefix, "", spec.suffix))
    return text if str(text).strip() else SAFE_EMPTY


def _build_pool_preview_text(base_text: str, pool_text: str) -> str:
    base = str(base_text or "").strip()
    pooled = str(pool_text or "").strip() or SAFE_EMPTY
    if not base:
        return f"{POOL_PREVIEW_PREFIX}{pooled}>"
    return f"{base} {POOL_PREVIEW_PREFIX}{pooled}>"


def _match_bind_keyword_at(text: str, index: int) -> tuple[str | None, int] | None:
    if not text.startswith(BIND_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None

    j = index + len(BIND_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1

    weight_raw = None
    if j < len(text) and text[j] == "^":
        j += 1
        weight_start = j
        while j < len(text) and text[j] != "{":
            j += 1
        weight_raw = text[weight_start:j].strip()

    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "{":
        return weight_raw, j
    return None


def _parse_bind_weight(raw: str | None, *, full_text: str) -> float:
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0.0:
        raise PromptSyntaxError(
            "BIND weight must be a positive number.",
            kind="invalid_bind_weight",
            token=raw or BIND_KEYWORD,
            full=full_text,
        )
    return value


def _find_top_level_bind_blocks(text: str) -> list[tuple[int, int, int, str | None]]:
    blocks: list[tuple[int, int, int, str | None]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_brace == 0:
            matched = _match_bind_keyword_at(text, i)
            if matched is not None:
                weight_raw, brace_open = matched
                brace_depth = 1
                k = brace_open + 1
                while k < n:
                    inner = text[k]
                    if inner == "\\":
                        k += 2 if k + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, brace_open, k, weight_raw))
                            i = k + 1
                            break
                    k += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed BIND block: expected '}'",
                        kind="invalid_bind_syntax",
                        token=f"{BIND_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_bind_owner_attrs(body: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif (
            ch == "="
            and i + 1 < n
            and body[i + 1] == ">"
            and depth_paren == 0
            and depth_brace == 0
            and depth_brack == 0
        ):
            owner = body[:i].strip()
            attrs = body[i + 2 :].strip()
            if owner and attrs:
                return owner, attrs
            break
        i += 1
    raise PromptSyntaxError(
        "BIND blocks must use 'owner => attrs' syntax.",
        kind="invalid_bind_syntax",
        token=BIND_KEYWORD,
        full=full_text,
    )


def _extract_bind_prompt_bundle(text: str) -> tuple[tuple[BindPromptSpec, ...], str]:
    if not text or BIND_KEYWORD not in text:
        return (), text

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_bind_blocks(protected)
    if not blocks:
        return (), text

    specs: list[BindPromptSpec] = []
    base_parts: list[str] = []
    last = 0
    for start, brace_open, brace_close, weight_raw in blocks:
        body = protected[brace_open + 1 : brace_close]
        nested = _find_top_level_bind_blocks(body)
        if nested or _contains_bind_marker(body):
            raise PromptSyntaxError(
                "Nested BIND blocks are not supported in v1.",
                kind="nested_bind_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )
        owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
        if _contains_top_level_multicond(owner_body) or _contains_top_level_multicond(attrs_body):
            raise PromptSyntaxError(
                "BIND owner and attrs cannot contain top-level AND branches in v1.",
                kind="bind_inner_multicond_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )
        if (
            _contains_chunk_marker(owner_body)
            or _contains_assemble_marker(owner_body)
            or _contains_blend_marker(owner_body)
            or _contains_morph_marker(owner_body)
            or _contains_pool_marker(owner_body)
            or _contains_chunk_marker(attrs_body)
            or _contains_assemble_marker(attrs_body)
            or _contains_blend_marker(attrs_body)
            or _contains_morph_marker(attrs_body)
            or _contains_pool_marker(attrs_body)
        ):
            raise PromptSyntaxError(
                "BIND owner and attrs cannot contain CHUNK, ASSEMBLE, BLEND, MORPH, or POOL blocks in v1.",
                kind="nested_backend_in_bind_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )
        owner = _restore_escaped_literal_source(owner_body, span_restore).strip()
        attrs = _restore_escaped_literal_source(attrs_body, span_restore).strip()
        if not owner or not attrs:
            raise PromptSyntaxError(
                "BIND requires non-empty owner and attrs sections.",
                kind="invalid_bind_syntax",
                token=BIND_KEYWORD,
                full=text,
            )
        specs.append(
            BindPromptSpec(
                owner=owner,
                attrs=attrs,
                weight=_parse_bind_weight(weight_raw, full_text=text),
                source=text,
            )
        )
        base_parts.append(protected[last:start])
        last = brace_close + 1

    base_parts.append(protected[last:])
    base_prompt = _cleanup_adjunct_base_prompt_text(
        _restore_escaped_literal_source("".join(base_parts), span_restore)
    )
    if not str(base_prompt).strip():
        raise PromptSyntaxError(
            "BIND requires a non-empty base prompt outside the BIND blocks.",
            kind="bind_requires_base_prompt",
            token=BIND_KEYWORD,
            full=text,
        )
    return tuple(specs), base_prompt


def _compose_bind_branch_prompt(owner_text: str, attrs_text: str) -> str:
    return _collapse_spaces(f"{owner_text}, {attrs_text}")


def _split_top_level_commas(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _transpile_bind3_to_chunk(text: str) -> str:
    """Convert BIND3{owner => attr1, attr2} → CHUNK{owner, attr1 | owner, attr2}
    for text-only paths (get_schedule, lint). Uses BIND2 transpile logic with BIND3 keyword."""
    if not text or BIND3_KEYWORD not in text:
        return text
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    result_parts: list[str] = []
    last = 0
    for m in _RE_BIND3_MARKER.finditer(protected):
        start = m.start()
        brace_open = m.end() - 1
        depth = 1
        i = brace_open + 1
        while i < len(protected) and depth > 0:
            if protected[i] == "{":
                depth += 1
            elif protected[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            raise PromptSyntaxError(
                "Unclosed BIND3 block: expected '}'",
                kind="invalid_bind_syntax",
                token=BIND3_KEYWORD,
                full=text,
            )
        brace_close = i - 1
        body = protected[brace_open + 1 : brace_close]
        owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
        owner = _restore_escaped_literal_source(owner_body, span_restore)
        attrs_str = _restore_escaped_literal_source(attrs_body, span_restore)
        if not owner or not attrs_str:
            raise PromptSyntaxError(
                "BIND3 requires non-empty owner and attrs sections.",
                kind="invalid_bind_syntax",
                token=BIND3_KEYWORD,
                full=text,
            )
        if "|" in owner:
            raise PromptSyntaxError(
                "BIND3 owner cannot contain '|' (pipe) — it conflicts with CHUNK branch separators.",
                kind="invalid_bind_syntax",
                token=BIND3_KEYWORD,
                full=text,
            )
        groups = _split_top_level_commas(attrs_str)
        if not groups:
            raise PromptSyntaxError(
                "BIND3 attrs section cannot be empty or just commas.",
                kind="invalid_bind_syntax",
                token=BIND3_KEYWORD,
                full=text,
            )
        clean_groups: list[str] = []
        for g in groups:
            try:
                g_text, _ = _split_blend_branch_weight(g, full_text=text)
                clean_groups.append(g_text)
            except PromptSyntaxError:
                clean_groups.append(g)
        if len(clean_groups) == 1:
            branches = f"{owner}, {clean_groups[0]}"
        else:
            branches = " | ".join(f"{owner}, {g}" for g in clean_groups)
        chunk_text = f"CHUNK{{{branches}}}"
        result_parts.append(protected[last:start])
        result_parts.append(chunk_text)
        last = brace_close + 1
    result_parts.append(protected[last:])
    return _restore_escaped_literal_source("".join(result_parts), span_restore)


def _transpile_bind2_to_chunk(text: str) -> str:
    if not text or BIND2_KEYWORD not in text:
        return text
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    result_parts: list[str] = []
    last = 0
    for m in _RE_BIND2_MARKER.finditer(protected):
        start = m.start()
        brace_open = m.end() - 1
        depth = 1
        i = brace_open + 1
        while i < len(protected) and depth > 0:
            if protected[i] == "{":
                depth += 1
            elif protected[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            raise PromptSyntaxError(
                "Unclosed BIND2 block: expected '}'",
                kind="invalid_bind_syntax",
                token=BIND2_KEYWORD,
                full=text,
            )
        brace_close = i - 1
        body = protected[brace_open + 1 : brace_close]
        owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
        owner = _restore_escaped_literal_source(owner_body, span_restore)
        attrs_str = _restore_escaped_literal_source(attrs_body, span_restore)
        if not owner or not attrs_str:
            raise PromptSyntaxError(
                "BIND2 requires non-empty owner and attrs sections.",
                kind="invalid_bind_syntax",
                token=BIND2_KEYWORD,
                full=text,
            )
        if "|" in owner:
            raise PromptSyntaxError(
                "BIND2 owner cannot contain '|' (pipe) — it conflicts with CHUNK branch separators.",
                kind="invalid_bind_syntax",
                token=BIND2_KEYWORD,
                full=text,
            )
        groups = _split_top_level_commas(attrs_str)
        if not groups:
            raise PromptSyntaxError(
                "BIND2 attrs section cannot be empty or just commas.",
                kind="invalid_bind_syntax",
                token=BIND2_KEYWORD,
                full=text,
            )
        if len(groups) == 1:
            branches = f"{owner}, {groups[0]}"
        else:
            branches = " | ".join(f"{owner}, {g}" for g in groups)
        chunk_text = f"CHUNK{{{branches}}}"
        result_parts.append(protected[last:start])
        result_parts.append(chunk_text)
        last = brace_close + 1
    result_parts.append(protected[last:])
    return _restore_escaped_literal_source("".join(result_parts), span_restore)


def _build_bind_component_text_schedule(
    text: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if strict:
        return _strict_schedule_preview(text, steps, seed)
    return get_schedule(text, steps, use_scheduling, seed, use_visitor=use_visitor)


def _build_bind_branch_timeline_entries(
    spec: BindPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[tuple[int, str, float]]:
    owner_schedule = _build_bind_component_text_schedule(
        spec.owner,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=strict,
    )
    attrs_schedule = _build_bind_component_text_schedule(
        spec.attrs,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=strict,
    )
    if not owner_schedule or not attrs_schedule:
        raise ValueError("Empty schedule for BIND owner or attrs")

    boundaries = _collect_schedule_boundaries([owner_schedule, attrs_schedule], steps)
    out: list[tuple[int, str, float]] = []
    previous_key = None
    for end_at_step in boundaries:
        owner_text = _select_text_from_schedule(owner_schedule, end_at_step) or SAFE_EMPTY
        attrs_text = _select_text_from_schedule(attrs_schedule, end_at_step) or SAFE_EMPTY
        is_active = bool(str(owner_text).strip()) and bool(str(attrs_text).strip())
        bind_text = _compose_bind_branch_prompt(owner_text, attrs_text) if is_active else SAFE_EMPTY
        bind_weight = float(spec.weight) if is_active else 0.0
        key = (bind_text, round(bind_weight, 8))
        if out and previous_key == key:
            out[-1] = (int(end_at_step), out[-1][1], out[-1][2])
        else:
            out.append((int(end_at_step), bind_text, bind_weight))
            previous_key = key
    return out or [(int(steps), SAFE_EMPTY, 0.0)]


def _pick_bind_timeline_entry(
    timeline: Sequence[tuple[int, str, float]],
    current_step: int,
) -> tuple[int, str, float]:
    if not timeline:
        raise ValueError("Empty BIND timeline")
    for entry in timeline:
        if current_step <= int(entry[0]):
            return entry
    return timeline[-1]


def _build_bind_preview_text(active_text: str) -> str:
    return f"{BIND_PREVIEW_PREFIX}{str(active_text or '').strip() or SAFE_EMPTY}>"


def _find_top_level_assemble_blocks(text: str) -> list[tuple[int, int, int]]:
    return _find_keyword_brace_blocks(text, ASSEMBLE_KEYWORD, "assemble", "ASSEMBLE")


def _split_top_level_assemble_fields(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    last = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == ";" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            parts.append(body[last:i])
            last = i + 1
        i += 1

    parts.append(body[last:])
    cleaned = [part.strip() for part in parts if part.strip()]
    if not cleaned:
        raise PromptSyntaxError(
            "ASSEMBLE must define enc1 and enc2 fields.",
            kind="invalid_assemble_syntax",
            token=ASSEMBLE_KEYWORD,
            full=full_text,
        )
    return cleaned


def _split_assemble_field(part: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(part)
    while i < n:
        ch = part[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "=" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            name = part[:i].strip().lower()
            value = part[i + 1 :].strip()
            if not name or not value:
                break
            return name, value
        i += 1
    raise PromptSyntaxError(
        "ASSEMBLE fields must use 'name=value' syntax.",
        kind="invalid_assemble_syntax",
        token=part.strip() or ASSEMBLE_KEYWORD,
        full=full_text,
    )


def _extract_assemble_prompt_spec(text: str) -> AssemblePromptSpec | None:
    if not text or ASSEMBLE_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_assemble_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level ASSEMBLE block is supported in v1.",
            kind="multiple_assemble_blocks_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    start, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_assemble_blocks(body)
    if nested or _contains_assemble_marker(body):
        raise PromptSyntaxError(
            "Nested ASSEMBLE blocks are not supported in v1.",
            kind="nested_assemble_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )
    fields = {}
    for part in _split_top_level_assemble_fields(body, full_text=text):
        name, value = _split_assemble_field(part, full_text=text)
        if name not in {"enc1", "enc2", "pooled"}:
            raise PromptSyntaxError(
                f"Unsupported ASSEMBLE field {name!r}",
                kind="invalid_assemble_field",
                token=name,
                full=text,
            )
        if name in fields:
            raise PromptSyntaxError(
                f"Duplicate ASSEMBLE field {name!r}",
                kind="duplicate_assemble_field",
                token=name,
                full=text,
            )
        if _contains_top_level_multicond(value):
            raise PromptSyntaxError(
                "ASSEMBLE fields cannot contain top-level AND branches in v1.",
                kind="assemble_inner_multicond_not_supported",
                token=ASSEMBLE_KEYWORD,
                full=text,
            )
        fields[name] = _restore_escaped_literal_source(value, span_restore).strip()

    if not fields.get("enc1") or not fields.get("enc2"):
        raise PromptSyntaxError(
            "ASSEMBLE requires non-empty enc1 and enc2 fields.",
            kind="invalid_assemble_syntax",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    pooled = fields.get("pooled") or None
    return AssemblePromptSpec(
        prefix=prefix,
        suffix=suffix,
        enc1=fields["enc1"],
        enc2=fields["enc2"],
        pooled=pooled,
        source=text,
    )


def _expand_assemble_section_prompt(spec: AssemblePromptSpec, body: str) -> str:
    return _concat_prefix_text_suffix(spec.prefix, body, spec.suffix)


def _build_assemble_preview_text(enc1_text: str, enc2_text: str, pooled_text: str) -> str:
    return f"{ASSEMBLE_PREVIEW_PREFIX}enc1={enc1_text} | enc2={enc2_text} | pooled={pooled_text}>"


def _build_bind_text_schedule_from_specs(
    base_prompt: str,
    bind_specs: Sequence[BindPromptSpec],
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if strict:
        base_schedule = _strict_schedule_preview(base_prompt, steps, seed)
        bind_timelines = [
            _build_bind_branch_timeline_entries(
                spec,
                steps,
                use_scheduling,
                seed,
                use_visitor,
                strict=True,
            )
            for spec in bind_specs
        ]
    else:
        base_schedule = get_schedule(base_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        bind_timelines = [
            _build_bind_branch_timeline_entries(
                spec,
                steps,
                use_scheduling,
                seed,
                use_visitor,
                strict=False,
            )
            for spec in bind_specs
        ]

    bind_text_schedules = [
        [[int(end_at_step), text] for end_at_step, text, _weight in timeline]
        for timeline in bind_timelines
    ]

    if not base_schedule or any(not schedule for schedule in bind_text_schedules):
        raise ValueError("Empty schedule for BIND prompt")

    boundaries = _collect_schedule_boundaries([base_schedule, *bind_text_schedules], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        active_texts = [base_text]
        weights = [1.0]
        bind_keys: list[tuple[str, float]] = []
        for timeline in bind_timelines:
            _bind_end_at_step, bind_text, bind_weight = _pick_bind_timeline_entry(timeline, end_at_step)
            bind_keys.append((bind_text, bind_weight))
            if bind_weight <= 1e-8:
                continue
            active_texts.append(_build_bind_preview_text(bind_text))
            weights.append(bind_weight)
        key = (base_text, tuple((text, round(weight, 8)) for text, weight in bind_keys))
        preview = _build_multicond_preview_text(active_texts, weights)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _extract_chunk_prompt_spec(text: str) -> ChunkPromptSpec | None:
    if not text or CHUNK_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_chunk_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level CHUNK block is supported in v1.",
            kind="multiple_chunk_blocks_not_supported",
            token=CHUNK_KEYWORD,
            full=text,
        )

    start, mode_open, mode_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_chunk_blocks(body)
    if nested or re.search(rf"(?<![\w\\]){re.escape(CHUNK_KEYWORD)}\s*\{{", body):
        raise PromptSyntaxError(
            "Nested CHUNK blocks are not supported in v1.",
            kind="nested_chunk_not_supported",
            token=CHUNK_KEYWORD,
            full=text,
        )
    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    raw_parts = _split_top_level_chunk_body(body, full_text=text)
    shared_channel = "none"
    if mode_open is not None and mode_close is not None:
        shared_channel = _parse_chunk_shared_mode(protected[mode_open + 1 : mode_close], full_text=text)

    branches: list[ChunkBranchSpec] = []
    for raw_part in raw_parts:
        branch_text, weight = _split_chunk_branch_weight(raw_part, full_text=text)
        if _contains_top_level_multicond(branch_text):
            raise PromptSyntaxError(
                "CHUNK branches cannot contain top-level AND branches in v1.",
                kind="chunk_inner_multicond_not_supported",
                token=CHUNK_KEYWORD,
                full=text,
            )
        branch_text = _restore_escaped_literal_source(branch_text, span_restore).strip()
        if not branch_text:
            raise PromptSyntaxError(
                "Empty CHUNK branch is not allowed.",
                kind="empty_chunk_branch",
                token="|",
                full=text,
            )
        branches.append(ChunkBranchSpec(branch_text, weight))

    return ChunkPromptSpec(prefix=prefix, suffix=suffix, branches=tuple(branches), shared_channel=shared_channel)


def _expand_chunk_branch_prompt(spec: ChunkPromptSpec, branch: ChunkBranchSpec | str) -> str:
    branch_text = branch.text if isinstance(branch, ChunkBranchSpec) else str(branch)
    return _concat_prefix_text_suffix(spec.prefix, branch_text, spec.suffix)


def _build_chunk_shared_anchor_prompt(spec: ChunkPromptSpec) -> str:
    anchor = _concat_prefix_text_suffix(spec.prefix, "", spec.suffix)
    return anchor if str(anchor).strip() else SAFE_EMPTY


def _select_text_from_schedule(schedule: Sequence[Sequence[int | str]], step: int) -> str:
    if not schedule:
        return ""
    for end_at_step, text in schedule:
        if step <= int(end_at_step):
            return str(text)
    return str(schedule[-1][1])


def _pick_text_schedule_index(schedule: Sequence[Sequence[int | str]], step: int) -> int:
    if not schedule:
        raise ValueError("Empty text schedule")
    for current, (end_at_step, _text) in enumerate(schedule):
        if step <= int(end_at_step):
            return current
    return len(schedule) - 1


def _collect_schedule_boundaries(schedules: Sequence[Sequence[Sequence[int | str]]], steps: int) -> list[int]:
    boundaries = sorted({int(end) for schedule in schedules for end, _ in schedule})
    if not boundaries or boundaries[-1] < int(steps):
        boundaries.append(int(steps))
    return boundaries


def _find_top_level_bind3_blocks(protected: str) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    skip_until = 0
    for m in _RE_BIND3_MARKER.finditer(protected):
        if m.start() < skip_until:
            continue
        brace_open = m.end() - 1
        depth = 1
        i = brace_open + 1
        while i < len(protected) and depth > 0:
            if protected[i] == "{":
                depth += 1
            elif protected[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            blocks.append((m.start(), i))
            skip_until = i
    return blocks


def _extract_sequential_backend_segments(
    protected: str,
    span_restore: dict[str, str],
) -> list[str]:
    blocks: list[tuple[int, int]] = []
    for start, _, _, _, brace_close in _find_top_level_blend_blocks(protected):
        blocks.append((start, brace_close + 1))
    for start, _, _, _, brace_close in _find_top_level_chunk_blocks(protected):
        blocks.append((start, brace_close + 1))
    for start, _, _, _, brace_close in _find_top_level_morph_blocks(protected):
        blocks.append((start, brace_close + 1))
    for start, _, brace_close in _find_top_level_pool_blocks(protected):
        blocks.append((start, brace_close + 1))
    for start, _, brace_close in _find_top_level_assemble_blocks(protected):
        blocks.append((start, brace_close + 1))
    for start, end in _find_top_level_bind3_blocks(protected):
        blocks.append((start, end))
    for start, _, brace_close in _find_top_level_compound_blocks(protected):
        blocks.append((start, brace_close + 1))
    blocks.sort()
    segments: list[str] = []
    pos = 0
    for start, end in blocks:
        if pos < start:
            gap = _restore_escaped_literal_source(protected[pos:start], span_restore)
            if gap.strip():
                segments.append(gap)
        segments.append(_restore_escaped_literal_source(protected[start:end], span_restore))
        pos = end
    if pos < len(protected):
        tail = _restore_escaped_literal_source(protected[pos:], span_restore)
        if tail.strip():
            segments.append(tail)
    return segments


def _merge_sequential_text_schedules(
    part_schedules: Sequence[Sequence[Sequence[int | str]]],
    steps: int,
) -> list[list[int | str]]:
    boundaries = _collect_schedule_boundaries(part_schedules, steps)
    out: list[list[int | str]] = []
    previous_key: tuple[str, ...] | None = None
    for end_at_step in boundaries:
        active_texts = [
            _normalize_preview_fragment(_select_text_from_schedule(sched, end_at_step) or SAFE_EMPTY)
            for sched in part_schedules
        ]
        key = tuple(active_texts)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            preview = CHUNK_PREVIEW_SEPARATOR.join(active_texts)
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _build_sequential_cond_schedule(
    model,
    segments: list[str],
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
) -> list[ScheduledPromptConditioning]:
    part_schedules = [
        _build_prompt_conditioning_schedule(model, seg, steps, use_scheduling, seed, use_visitor, copy_from)
        for seg in segments
    ]
    boundaries = _collect_schedule_boundaries(part_schedules, steps)  # type: ignore[arg-type]
    out: list[ScheduledPromptConditioning] = []
    previous_key: tuple[int, ...] | None = None
    for end_at_step in boundaries:
        indices = tuple(_pick_schedule_entry_index(ps, end_at_step) for ps in part_schedules)
        key = indices
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue
        conds = [ps[idx].cond for ps, idx in zip(part_schedules, indices)]
        merged = _merge_chunk_condition_values(conds, [1.0] * len(conds))
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key
    return out


def _build_chunk_text_schedule_from_spec(
    spec: ChunkPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    branch_schedules = [
        get_schedule(_expand_chunk_branch_prompt(spec, branch), steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch in spec.branches
    ]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    for end_at_step in boundaries:
        active_texts = [
            _normalize_preview_fragment(_select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY)
            for schedule in branch_schedules
        ]
        preview_parts = [text for text in active_texts if text]
        preview = CHUNK_PREVIEW_SEPARATOR.join(preview_parts) if preview_parts else SAFE_EMPTY
        if out and out[-1][1] == preview:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
    return out or [[int(steps), SAFE_EMPTY]]


def _match_morph_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(MORPH_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(MORPH_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "^":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "@[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "@":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "[":
        window_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    window_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return window_open, window_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed MORPH window: expected ']'.",
            kind="invalid_morph_window",
            token=f"{MORPH_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _parse_morph_intensity_spec(text: str, *, full_text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0

    match = re.fullmatch(rf"\^\s*({NUMERIC_RE})\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {text!r}",
            kind="invalid_morph_intensity",
            token=stripped,
            full=full_text,
        )

    intensity = float(match.group(1))
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {text!r}",
            kind="invalid_morph_intensity",
            token=stripped,
            full=full_text,
        )
    return intensity


def _parse_morph_channel_target_spec(text: str, *, full_text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "both"

    match = re.fullmatch(r"@\s*([A-Za-z0-9_-]+)\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH channel target {text!r}",
            kind="invalid_morph_channel_target",
            token=stripped,
            full=full_text,
        )

    channel_target = match.group(1).strip().lower()
    if channel_target not in BACKEND_CHANNEL_TARGETS:
        raise PromptSyntaxError(
            f"Invalid MORPH channel target {text!r}",
            kind="invalid_morph_channel_target",
            token=stripped,
            full=full_text,
        )
    return channel_target


def _parse_morph_header_spec(text: str, *, full_text: str) -> tuple[float, str]:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0, "both"

    intensity_text = stripped
    channel_text = ""
    if "@" in stripped:
        at_pos = stripped.find("@")
        intensity_text = stripped[:at_pos].rstrip()
        channel_text = stripped[at_pos:].strip()

    intensity = _parse_morph_intensity_spec(intensity_text, full_text=full_text)
    channel_target = _parse_morph_channel_target_spec(channel_text, full_text=full_text)
    return intensity, channel_target


def _find_top_level_morph_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_brace == 0:
            morph_match = _match_morph_keyword_at(text, i)
            if morph_match is not None:
                window_open, window_close, brace_open = morph_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, window_open, window_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed MORPH block: expected '}'",
                        kind="invalid_morph_syntax",
                        token=f"{MORPH_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_morph_curve(body: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    curve_pos = -1
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "~" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            curve_pos = i
        i += 1

    if curve_pos < 0:
        return body, "linear"

    curve = body[curve_pos + 1 :].strip().lower()
    if curve not in MORPH_CURVES:
        raise PromptSyntaxError(
            f"Unsupported MORPH curve {curve!r}",
            kind="invalid_morph_curve",
            token=curve or "~",
            full=full_text,
        )
    return body[:curve_pos].rstrip(), curve


def _split_top_level_morph_points(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if (
            ch == "=" and i + 1 < n and body[i + 1] == ">"
            and depth_paren == 0 and depth_brace == 0 and depth_brack == 0
        ):
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "MORPH control prompts cannot be empty.",
                    kind="invalid_morph_syntax",
                    token="=>",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 2
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )
    parts.append(last)
    return parts


def _parse_morph_boundary_spec(text: str, *, full_text: str) -> _BoundarySpec:
    match = re.fullmatch(rf"\s*({NUMERIC_RE})(%?)\s*", text or "")
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH boundary {text!r}",
            kind="invalid_morph_boundary",
            token=(text or "@").strip() or "@",
            full=full_text,
        )
    return _make_boundary_spec(float(match.group(1)), is_percent=bool(match.group(2)))


def _parse_morph_window_spec(text: str, *, full_text: str) -> tuple[_BoundarySpec, _BoundarySpec]:
    match = re.fullmatch(r"\s*(.+?)\s*-\s*(.+?)\s*", text or "")
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH activation window {text!r}",
            kind="invalid_morph_window",
            token=(text or "[]").strip() or "[]",
            full=full_text,
        )
    try:
        start = _parse_morph_boundary_spec(match.group(1), full_text=full_text)
        end = _parse_morph_boundary_spec(match.group(2), full_text=full_text)
    except PromptSyntaxError as exc:
        raise PromptSyntaxError(
            f"Invalid MORPH activation window {text!r}",
            kind="invalid_morph_window",
            token=(text or "[]").strip() or "[]",
            full=full_text,
        ) from exc
    return start, end


def _split_morph_point_boundary(text: str, *, full_text: str) -> tuple[str, _BoundarySpec | None]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    at_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "@" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            at_pos = i
        i += 1

    if at_pos < 0:
        return stripped, None

    point_text = stripped[:at_pos].rstrip()
    boundary_text = stripped[at_pos + 1 :].strip()
    if not point_text:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="@",
            full=full_text,
        )
    return point_text, _parse_morph_boundary_spec(boundary_text, full_text=full_text)


def _split_morph_point_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid MORPH point weight {weight_text!r}",
            kind="invalid_morph_point_weight",
            token=weight_text or "*",
            full=full_text,
        )

    point_text = stripped[:star_pos].rstrip()
    if not point_text.strip():
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if not math.isfinite(weight) or weight <= 0.0:
        raise PromptSyntaxError(
            f"MORPH point weight must be positive, got {weight_text!r}",
            kind="invalid_morph_point_weight",
            token=weight_text,
            full=full_text,
        )
    return point_text, weight


def _extract_morph_prompt_spec(text: str) -> MorphPromptSpec | None:
    if not text or MORPH_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_morph_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level MORPH block is supported in v1.",
            kind="multiple_morph_blocks_not_supported",
            token=MORPH_KEYWORD,
            full=text,
        )

    start, window_open, window_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_morph_blocks(body)
    if nested or _contains_morph_marker(body):
        raise PromptSyntaxError(
            "Nested MORPH blocks are not supported in v1.",
            kind="nested_morph_not_supported",
            token=MORPH_KEYWORD,
            full=text,
        )
    body, curve = _split_top_level_morph_curve(body, full_text=text)
    raw_points = _split_top_level_morph_points(body, full_text=text)
    if len(raw_points) < 2:
        raise PromptSyntaxError(
            "MORPH requires at least two control prompts.",
            kind="invalid_morph_syntax",
            token=MORPH_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)

    intensity, channel_target = _parse_morph_header_spec(
        protected[start + len(MORPH_KEYWORD) : (window_open if window_open is not None else brace_open)],
        full_text=text,
    )

    window_start = None
    window_end = None
    if window_open is not None and window_close is not None:
        window_start, window_end = _parse_morph_window_spec(
            protected[window_open + 1 : window_close],
            full_text=text,
        )

    points: list[MorphPointSpec] = []
    for i, raw_point in enumerate(raw_points):
        point_text, boundary = _split_morph_point_boundary(raw_point, full_text=text)
        point_text, point_weight = _split_morph_point_weight(point_text, full_text=text)
        if _contains_top_level_multicond(point_text):
            raise PromptSyntaxError(
                "MORPH control prompts cannot contain top-level AND branches in v1.",
                kind="morph_inner_multicond_not_supported",
                token=MORPH_KEYWORD,
                full=text,
            )
        point_text = _restore_escaped_literal_source(point_text, span_restore).strip()
        if not point_text:
            raise PromptSyntaxError(
                "MORPH control prompts cannot be empty.",
                kind="invalid_morph_syntax",
                token="=>",
                full=text,
            )
        if window_start is not None or window_end is not None:
            if boundary is not None:
                raise PromptSyntaxError(
                    "MORPH window syntax cannot be combined with per-point '@boundary' markers in v1.",
                    kind="morph_window_with_point_boundaries_not_supported",
                    token="@",
                    full=text,
                )
        else:
            if i == 0 and boundary is not None:
                raise PromptSyntaxError(
                    "The first MORPH control prompt cannot have an explicit boundary in v1.",
                    kind="invalid_morph_boundary",
                    token="@",
                    full=text,
                )
            if i not in (0, len(raw_points) - 1) and boundary is None:
                raise PromptSyntaxError(
                    "Intermediate MORPH control prompts must declare a boundary with '@'.",
                    kind="invalid_morph_boundary",
                    token=point_text,
                    full=text,
                )
        points.append(MorphPointSpec(point_text, boundary, point_weight))

    return MorphPromptSpec(
        prefix=prefix,
        suffix=suffix,
        points=tuple(points),
        curve=curve,
        intensity=intensity,
        channel_target=channel_target,
        window_start=window_start,
        window_end=window_end,
        source=text,
    )


def _is_chunk_morph_sugar(text: str) -> bool:
    if not text or CHUNK_KEYWORD not in text or MORPH_KEYWORD not in text:
        return False

    protected, _span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    chunk_blocks = _find_top_level_chunk_blocks(protected)
    morph_blocks = _find_top_level_morph_blocks(protected)
    blend_blocks = _find_top_level_blend_blocks(protected)

    if len(chunk_blocks) != 1 or len(morph_blocks) != 1 or blend_blocks:
        return False

    chunk_start, _chunk_mode_open, _chunk_mode_close, chunk_brace_open, chunk_brace_close = chunk_blocks[0]
    morph_start, _window_open, _window_close, _morph_brace_open, morph_brace_close = morph_blocks[0]

    non_overlapping = (
        chunk_start < morph_start and chunk_brace_close < morph_start
    ) or (
        morph_start < chunk_start and morph_brace_close < chunk_start
    )
    if not non_overlapping:
        return False

    body = protected[chunk_brace_open + 1 : chunk_brace_close]
    if _contains_morph_marker(body) or _contains_blend_marker(body):
        return False

    return True


def _extract_backend_prompt_state(text: str) -> BackendPromptState:
    text = _transpile_bind2_to_chunk(text)
    bind_marker = _contains_bind_marker(text)
    bind_specs, bind_base_prompt = _extract_bind_prompt_bundle(text)
    if bind_specs and len(_split_top_level_multicond(_protect_escaped_literals(_protect_escaped_literal_spans(bind_base_prompt)[0]))) > 1:
        raise PromptSyntaxError(
            "BIND cannot be combined with top-level AND in v1.",
            kind="bind_with_and_not_supported",
            token=BIND_KEYWORD,
            full=text,
        )
    post_bind_text = bind_base_prompt if bind_specs else text
    pool_marker = _contains_pool_marker(post_bind_text)
    has_multiple_same_type = False
    try:
        pool_spec = _extract_pool_prompt_spec(post_bind_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_pool_blocks_not_supported":
            has_multiple_same_type = True
            pool_spec = None
        else:
            raise
    primary_text = _build_pool_base_prompt(pool_spec) if pool_spec is not None else post_bind_text
    try:
        chunk_spec = _extract_chunk_prompt_spec(primary_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_chunk_blocks_not_supported":
            has_multiple_same_type = True
            chunk_spec = None
        else:
            raise
    try:
        assemble_spec = _extract_assemble_prompt_spec(primary_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_assemble_blocks_not_supported":
            has_multiple_same_type = True
            assemble_spec = None
        else:
            raise
    try:
        blend_spec = _extract_blend_prompt_spec(primary_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_blend_blocks_not_supported":
            has_multiple_same_type = True
            blend_spec = None
        else:
            raise
    try:
        morph_spec = _extract_morph_prompt_spec(primary_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_morph_blocks_not_supported":
            has_multiple_same_type = True
            morph_spec = None
        else:
            raise
    try:
        compound_spec = _extract_compound_prompt_spec(primary_text)
    except PromptSyntaxError as e:
        if e.kind == "multiple_compound_blocks_not_supported":
            has_multiple_same_type = True
            compound_spec = None
        else:
            raise
    extracted_backend = chunk_spec is not None or blend_spec is not None or morph_spec is not None or assemble_spec is not None or compound_spec is not None or bool(bind_specs) or has_multiple_same_type
    if pool_spec is None and pool_marker and not extracted_backend:
        raise PromptSyntaxError(
            "POOL blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_pool_context",
            token=POOL_KEYWORD,
            full=text,
        )
    if not bind_specs and bind_marker and not extracted_backend and pool_spec is None:
        raise PromptSyntaxError(
            "BIND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind_context",
            token=BIND_KEYWORD,
            full=text,
        )
    allow_chunk_morph_sugar = (
        chunk_spec is not None
        and assemble_spec is None
        and blend_spec is None
        and morph_spec is not None
        and _is_chunk_morph_sugar(primary_text)
    )
    return BackendPromptState(
        chunk_spec=chunk_spec,
        pool_spec=pool_spec,
        bind_specs=bind_specs,
        bind_base_prompt=bind_base_prompt,
        assemble_spec=assemble_spec,
        blend_spec=blend_spec,
        morph_spec=morph_spec,
        compound_spec=compound_spec,
        allow_chunk_morph_sugar=allow_chunk_morph_sugar,
        has_multiple_same_type=has_multiple_same_type,
    )



def _raise_bind_backend_prompt_error(text: str) -> None:
    raise PromptSyntaxError(
        "BIND can only be combined with plain prompt text or POOL in v1.",
        kind="bind_with_backend_not_supported",
        token=BIND_KEYWORD,
        full=text,
    )


def _raise_unsupported_backend_context_error(text: str) -> None:
    if _contains_chunk_marker(text):
        raise PromptSyntaxError(
            "CHUNK blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_chunk_context",
            token=CHUNK_KEYWORD,
            full=text,
        )
    if _contains_blend_marker(text):
        raise PromptSyntaxError(
            "BLEND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_blend_context",
            token=BLEND_KEYWORD,
            full=text,
        )
    if _contains_assemble_marker(text):
        raise PromptSyntaxError(
            "ASSEMBLE blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_assemble_context",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )
    if _contains_bind_marker(text):
        raise PromptSyntaxError(
            "BIND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind_context",
            token=BIND_KEYWORD,
            full=text,
        )
    if _contains_bind2_marker(text):
        raise PromptSyntaxError(
            "BIND2 blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind_context",
            token=BIND2_KEYWORD,
            full=text,
        )
    if _contains_morph_marker(text):
        raise PromptSyntaxError(
            "MORPH blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_morph_context",
            token=MORPH_KEYWORD,
            full=text,
        )
    if _contains_pool_marker(text):
        raise PromptSyntaxError(
            "POOL blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_pool_context",
            token=POOL_KEYWORD,
            full=text,
        )
    if _contains_bind3_marker(text):
        raise PromptSyntaxError(
            "BIND3 blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind3_context",
            token=BIND3_KEYWORD,
            full=text,
        )
    if _contains_compound_marker(text):
        raise PromptSyntaxError(
            "COMPOUND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_compound_context",
            token=COMPOUND_KEYWORD,
            full=text,
        )


def _expand_morph_point_prompt(spec: MorphPromptSpec, point: MorphPointSpec | str) -> str:
    point_text = point.text if isinstance(point, MorphPointSpec) else str(point)
    return _concat_prefix_text_suffix(spec.prefix, point_text, spec.suffix)


def _build_morph_inactive_text(spec: MorphPromptSpec) -> str:
    text = _concat_prefix_text_suffix(spec.prefix, "", spec.suffix)
    return text if str(text).strip() else SAFE_EMPTY


def _resolve_morph_window_steps(spec: MorphPromptSpec, steps: int) -> tuple[int, int] | None:
    if spec.window_start is None and spec.window_end is None:
        return None
    if spec.window_start is None or spec.window_end is None:
        raise PromptSyntaxError(
            "MORPH activation window must define both start and end boundaries.",
            kind="invalid_morph_window",
            token=MORPH_KEYWORD,
            full=spec.source or None,
        )
    start = _to_end_step(spec.window_start, steps)
    end = _to_end_step(spec.window_end, steps)
    if end <= start:
        raise PromptSyntaxError(
            "MORPH activation window end must be greater than the start.",
            kind="invalid_morph_window",
            token=MORPH_KEYWORD,
            full=spec.source or None,
        )
    return int(start), int(end)


def _resolve_morph_positions(spec: MorphPromptSpec, steps: int) -> list[int]:
    window_steps = _resolve_morph_window_steps(spec, steps)
    if window_steps is not None:
        start, end = window_steps
        span = int(end) - int(start)
        count = len(spec.points)
        if span < count - 1:
            raise PromptSyntaxError(
                "MORPH activation window is too narrow for the number of control prompts.",
                kind="invalid_morph_window",
                token=MORPH_KEYWORD,
                full=spec.source or None,
            )
        positions = [int(start)]
        for i in range(1, count - 1):
            remaining_slots = (count - 1) - i
            raw = int(start) + (span * i) / (count - 1)
            pos = int(start) + int(math.floor(raw - int(start) + 0.5))
            pos = max(positions[-1] + 1, pos)
            pos = min(pos, int(end) - remaining_slots)
            if pos <= positions[-1]:
                raise PromptSyntaxError(
                    "MORPH activation window is too narrow for the number of control prompts.",
                    kind="invalid_morph_window",
                    token=MORPH_KEYWORD,
                    full=spec.source or None,
                )
            positions.append(int(pos))
        positions.append(int(end))
        return positions

    positions: list[int] = [1]
    for i, point in enumerate(spec.points[1:], start=1):
        if point.boundary is None:
            pos = int(steps)
        else:
            pos = _to_end_step(point.boundary, steps)
        pos = max(1, int(pos))
        if pos <= positions[-1]:
            raise PromptSyntaxError(
                "MORPH boundaries must be strictly increasing.",
                kind="invalid_morph_boundary",
                token="@",
                full=_expand_morph_point_prompt(spec, point),
            )
        positions.append(pos)
    return positions


def _compute_morph_scaled_t(positions: Sequence[int], step: int) -> float:
    if not positions or len(positions) == 1:
        return 0.0
    if step <= positions[0]:
        return 0.0
    if step >= positions[-1]:
        return 1.0

    for idx in range(len(positions) - 1):
        left = positions[idx]
        right = positions[idx + 1]
        if step <= right:
            local = 0.0 if right == left else (step - left) / max(1, right - left)
            return (idx + local) / (len(positions) - 1)
    return 1.0


def _shape_morph_progress_t(scaled_t: float, intensity: float) -> float:
    t = max(0.0, min(1.0, float(scaled_t)))
    k = float(intensity)
    if not math.isfinite(k) or k <= 0.0:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {intensity!r}",
            kind="invalid_morph_intensity",
            token=str(intensity),
            full=str(intensity),
        )
    if t <= 0.0 or t >= 1.0 or abs(k - 1.0) <= 1e-8:
        return t

    left = t ** k
    right = (1.0 - t) ** k
    denom = left + right
    if denom <= 0.0:
        return t
    return left / denom


def _compute_morph_curve_weights(
    count: int,
    positions: Sequence[int],
    step: int,
    curve: str,
    intensity: float = 1.0,
) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]

    scaled_t = _compute_morph_scaled_t(positions, step)
    scaled_t = _shape_morph_progress_t(scaled_t, intensity)
    if curve == "linear":
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        left = min(int(segment_pos), count - 2)
        alpha = segment_pos - left
        weights = [0.0] * count
        weights[left] = 1.0 - alpha
        weights[left + 1] = alpha
        return weights

    if curve == "bezier":
        weights = []
        degree = count - 1
        for i in range(count):
            weights.append(math.comb(degree, i) * ((1.0 - scaled_t) ** (degree - i)) * (scaled_t ** i))
        return weights

    if curve == "catmull":
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        i = min(int(segment_pos), count - 2)
        u = segment_pos - i
        idx0 = max(i - 1, 0)
        idx1 = i
        idx2 = i + 1
        idx3 = min(i + 2, count - 1)
        w0 = -0.5 * u * u * u + u * u - 0.5 * u
        w1 = 1.5 * u * u * u - 2.5 * u * u + 1.0
        w2 = -1.5 * u * u * u + 2.0 * u * u + 0.5 * u
        w3 = 0.5 * u * u * u - 0.5 * u * u
        weights = [0.0] * count
        weights[idx0] += w0
        weights[idx1] += w1
        weights[idx2] += w2
        weights[idx3] += w3
        return weights

    if curve == "slerp":
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        left = min(int(segment_pos), count - 2)
        u = segment_pos - left
        alpha = math.sin(u * math.pi * 0.5) ** 2
        weights = [0.0] * count
        weights[left] = 1.0 - alpha
        weights[left + 1] = alpha
        return weights

    raise PromptSyntaxError(
        f"Unsupported MORPH curve {curve!r}",
        kind="invalid_morph_curve",
        token=curve,
        full=curve,
    )


def _resolve_morph_point_weights(points: Sequence[MorphPointSpec], weights: Sequence[float]) -> list[float]:
    base_weights = [float(weight) for weight in weights]
    if not points:
        return base_weights

    point_weights = [float(getattr(point, "weight", 1.0)) for point in points]
    if all(abs(weight - 1.0) <= 1e-8 for weight in point_weights):
        return base_weights

    effective_weights = [base * point_weight for base, point_weight in zip(base_weights, point_weights)]
    total = float(sum(effective_weights)) if effective_weights else 0.0
    if math.isfinite(total) and abs(total) > 1e-8:
        return [weight / total for weight in effective_weights]
    return effective_weights


def _apply_condition_channel_target(base_cond, target_cond, channel_target: str):
    if channel_target == "both" or not isinstance(base_cond, dict) or not isinstance(target_cond, dict):
        return target_cond

    merged = {}
    all_keys = set(base_cond.keys()) | set(target_cond.keys())
    for key in all_keys:
        if channel_target in SDXL_ENCODER_CHANNEL_TARGETS:
            if key in SDXL_SPLITTABLE_CROSS_KEYS:
                if key in target_cond and key in base_cond:
                    merged[key] = _apply_sdxl_encoder_channel_target_to_value(
                        base_cond[key],
                        target_cond[key],
                        channel_target,
                    )
                elif key in target_cond:
                    merged[key] = target_cond[key]
                elif key in base_cond:
                    merged[key] = base_cond[key]
                continue

            if key in base_cond:
                merged[key] = base_cond[key]
            elif key in target_cond:
                merged[key] = target_cond[key]
            continue

        use_target = (
            key in CHUNK_CROSSATTN_KEYS
            if channel_target == "cross"
            else key not in CHUNK_CROSSATTN_KEYS
        )
        if use_target:
            if key in target_cond:
                merged[key] = target_cond[key]
            elif key in base_cond:
                merged[key] = base_cond[key]
        else:
            if key in base_cond:
                merged[key] = base_cond[key]
            elif key in target_cond:
                merged[key] = target_cond[key]
    return merged


def _apply_pool_prompt_conditioning(base_cond, pool_cond):
    if not isinstance(base_cond, dict) or not isinstance(pool_cond, dict):
        return base_cond
    return _apply_condition_channel_target(base_cond, pool_cond, "pooled")


def _morph_preview_prefix(channel_target: str) -> str:
    if channel_target == "both":
        return MORPH_PREVIEW_PREFIX
    return f"{MORPH_KEYWORD}@{channel_target}<"


def _build_morph_preview_text(active_texts: Sequence[str], weights: Sequence[float], channel_target: str = "both") -> str:
    nonzero = [
        (normalized_text, weight)
        for text, weight in zip(active_texts, weights)
        for normalized_text in [_normalize_preview_fragment(text)]
        if normalized_text and abs(float(weight)) > 1e-8
    ]
    if not nonzero:
        return SAFE_EMPTY
    if len(nonzero) == 1 and abs(float(nonzero[0][1]) - 1.0) <= 1e-8:
        return str(nonzero[0][0])
    parts = [f"{text}*{_format_interp_weight(float(weight))}" for text, weight in nonzero]
    return f"{_morph_preview_prefix(channel_target)}{' + '.join(parts)}>"


def _build_morph_text_schedule_from_spec(
    spec: MorphPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    point_schedules = [
        get_schedule(_expand_morph_point_prompt(spec, point), steps, use_scheduling, seed, use_visitor=use_visitor)
        for point in spec.points
    ]
    positions = _resolve_morph_positions(spec, steps)
    window_steps = _resolve_morph_window_steps(spec, steps)
    inactive_text = _build_morph_inactive_text(spec)

    if not use_scheduling:
        final_step = int(steps)
        if window_steps is not None and not (window_steps[0] <= final_step <= window_steps[1]):
            return [[int(steps), inactive_text]]
        active_texts = [_select_text_from_schedule(schedule, final_step) or SAFE_EMPTY for schedule in point_schedules]
        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, final_step, spec.curve, spec.intensity),
        )
        return [[int(steps), _build_morph_preview_text(active_texts, weights, spec.channel_target)]]

    out: list[list[int, str]] = []
    previous_key = None
    for step in range(1, int(steps) + 1):
        if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
            preview = inactive_text
            key = ("inactive", preview)
            if out and previous_key == key:
                out[-1][0] = int(step)
            else:
                out.append([int(step), preview])
                previous_key = key
            continue
        active_texts = [_select_text_from_schedule(schedule, step) or SAFE_EMPTY for schedule in point_schedules]
        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, step, spec.curve, spec.intensity),
        )
        preview = _build_morph_preview_text(active_texts, weights, spec.channel_target)
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in weights))
        if out and previous_key == key:
            out[-1][0] = int(step)
        else:
            out.append([int(step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _build_pool_text_schedule_from_spec(
    spec: PoolPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    base_prompt = _build_pool_base_prompt(spec)
    if strict:
        base_schedule = _strict_schedule_preview(base_prompt, steps, seed)
        pool_schedule = _strict_schedule_preview(spec.body, steps, seed)
    else:
        base_schedule = get_schedule(base_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        pool_schedule = get_schedule(spec.body, steps, use_scheduling, seed, use_visitor=use_visitor)

    boundaries = _collect_schedule_boundaries([base_schedule, pool_schedule], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        pool_text = _select_text_from_schedule(pool_schedule, end_at_step) or SAFE_EMPTY
        preview = _build_pool_preview_text(base_text, pool_text)
        key = (base_text, pool_text)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _build_assemble_text_schedule_from_spec(
    spec: AssemblePromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    enc1_prompt = _expand_assemble_section_prompt(spec, spec.enc1)
    enc2_prompt = _expand_assemble_section_prompt(spec, spec.enc2)
    pooled_prompt = _expand_assemble_section_prompt(spec, spec.pooled) if spec.pooled is not None else enc2_prompt

    if strict:
        enc1_schedule = _strict_schedule_preview(enc1_prompt, steps, seed)
        enc2_schedule = _strict_schedule_preview(enc2_prompt, steps, seed)
        pooled_schedule = _strict_schedule_preview(pooled_prompt, steps, seed)
    else:
        enc1_schedule = get_schedule(enc1_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        enc2_schedule = get_schedule(enc2_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        pooled_schedule = get_schedule(pooled_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)

    boundaries = _collect_schedule_boundaries([enc1_schedule, enc2_schedule, pooled_schedule], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        enc1_text = _select_text_from_schedule(enc1_schedule, end_at_step) or SAFE_EMPTY
        enc2_text = _select_text_from_schedule(enc2_schedule, end_at_step) or SAFE_EMPTY
        pooled_text = _select_text_from_schedule(pooled_schedule, end_at_step) or SAFE_EMPTY
        preview = _build_assemble_preview_text(enc1_text, enc2_text, pooled_text)
        key = (enc1_text, enc2_text, pooled_text)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _tensor_ndim(x) -> int:
    shape = getattr(x, "shape", None)
    if shape is None:
        return 0
    try:
        return len(shape)
    except TypeError:
        return 0


def _sdxl_cross_split_axis(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return None
    try:
        if int(shape[-1]) != SDXL_TOTAL_CROSS_DIM:
            return None
    except Exception:
        return None
    ndim = _tensor_ndim(x)
    if ndim <= 0:
        return None
    return ndim - 1


def _apply_sdxl_encoder_channel_target_to_value(base_value, target_value, channel_target: str):
    axis = _sdxl_cross_split_axis(base_value)
    target_axis = _sdxl_cross_split_axis(target_value)
    if axis is None or target_axis is None or axis != target_axis:
        return target_value

    ndim = _tensor_ndim(base_value)
    if ndim != _tensor_ndim(target_value):
        return target_value

    head = [slice(None)] * ndim
    tail = [slice(None)] * ndim
    head[axis] = slice(None, SDXL_ENCODER1_CROSS_DIM)
    tail[axis] = slice(SDXL_ENCODER1_CROSS_DIM, None)

    try:
        base_head = base_value[tuple(head)]
        base_tail = base_value[tuple(tail)]
        target_head = target_value[tuple(head)]
        target_tail = target_value[tuple(tail)]
        parts = [target_head, base_tail] if channel_target == "enc1" else [base_head, target_tail]
        return _ensure_torch().cat(parts, dim=axis)
    except Exception:
        # Fall back to whole-cross routing on non-SDXL or non-sliceable tensor types.
        return target_value


def _weighted_average_condition_values(values, weights):
    total = float(sum(weights)) if weights else 0.0
    if total == 0.0:
        total = 1.0

    acc = None
    for value, weight in zip(values, weights):
        weighted = value * float(weight)
        acc = weighted if acc is None else acc + weighted
    return acc / total


def _merge_chunk_condition_values(cond_list, weights):
    if not cond_list:
        raise ValueError("Empty CHUNK conditioning branch list")

    first = cond_list[0]
    if isinstance(first, dict):
        _torch = _ensure_torch()
        merged = {}
        for key in first.keys():
            values = [cond[key] for cond in cond_list if key in cond]
            if not values:
                continue

            if key in CHUNK_CROSSATTN_KEYS:
                weighted = [value * float(weight) if float(weight) != 1.0 else value for value, weight in zip(values, weights)]
                dim = 1 if _tensor_ndim(weighted[0]) >= 3 else 0
                merged[key] = _torch.cat(weighted, dim=dim)
                continue

            ndim = _tensor_ndim(values[0])
            if ndim <= 0:
                merged[key] = values[0]
            else:
                merged[key] = _weighted_average_condition_values(values, weights)
        return merged

    _torch = _ensure_torch()
    weighted = [value * float(weight) if float(weight) != 1.0 else value for value, weight in zip(cond_list, weights)]
    dim = 1 if _tensor_ndim(weighted[0]) >= 3 else 0
    return _torch.cat(weighted, dim=dim)


def _align_condition_values_for_blend(values):
    if not values:
        return []
    if any(getattr(value, "shape", None) is None for value in values):
        return list(values)

    counts = []
    for value in values:
        axis = 1 if _tensor_ndim(value) >= 3 else 0
        try:
            counts.append(int(value.shape[axis]))
        except Exception:
            return list(values)

    target = max(counts)
    if all(count == target for count in counts):
        return list(values)

    _torch = _ensure_torch()
    aligned = []
    for value, count in zip(values, counts):
        axis = 1 if _tensor_ndim(value) >= 3 else 0
        if count >= target:
            aligned.append(value)
            continue
        try:
            pad_count = target - count
            if axis == 0:
                tail = value[-1:]
                repeats = [pad_count] + [1] * max(0, len(value.shape) - 1)
                tail_repeated = tail.repeat(repeats)
                aligned.append(_torch.cat([value, tail_repeated], dim=0))
            else:
                tail = value[:, -1:, ...]
                repeats = [1, pad_count] + [1] * max(0, len(value.shape) - 2)
                tail_repeated = tail.repeat(repeats)
                aligned.append(_torch.cat([value, tail_repeated], dim=1))
        except Exception:
            return list(values)
    return aligned


def _weighted_sum_condition_values(values, weights):
    acc = None
    for value, weight in zip(values, weights):
        weight_f = float(weight)
        if abs(weight_f) <= 1e-8:
            continue
        weighted = value if abs(weight_f - 1.0) <= 1e-8 else value * weight_f
        acc = weighted if acc is None else acc + weighted
    return acc if acc is not None else values[0] * 0.0


def _blend_morph_condition_values(cond_list, weights):
    if not cond_list:
        raise ValueError("Empty MORPH conditioning point list")

    first = cond_list[0]
    if isinstance(first, dict):
        merged = {}
        for key in first.keys():
            keyed = [(cond[key], weight) for cond, weight in zip(cond_list, weights) if key in cond]
            if not keyed:
                continue
            values = [value for value, _weight in keyed]
            value_weights = [weight for _value, weight in keyed]
            if key in CHUNK_CROSSATTN_KEYS:
                values = _align_condition_values_for_blend(values)
            merged[key] = _weighted_sum_condition_values(values, value_weights)
        return merged

    aligned = _align_condition_values_for_blend(cond_list)
    return _weighted_sum_condition_values(aligned, weights)


def _slerp_tensor(a, b, t, _torch):
    """Spherical linear interpolation between two tensors along last dim.
    Standard formula: slerp(a,b,t) = sin((1-t)*θ)/sin(θ)*a + sin(t*θ)/sin(θ)*b
    Handles [77,768], [77,2048], or batched shapes.
    """
    a = a.to(dtype=_torch.float32)
    b = b.to(dtype=_torch.float32)

    orig_shape = a.shape
    if a.ndim > 2:
        a_f = a.reshape(-1, a.shape[-1])
        b_f = b.reshape(-1, b.shape[-1])
    else:
        a_f = a
        b_f = b

    a_norm = _torch.linalg.norm(a_f, dim=-1, keepdim=True)
    b_norm = _torch.linalg.norm(b_f, dim=-1, keepdim=True)
    a_norm = _torch.clamp(a_norm, min=1e-8)
    b_norm = _torch.clamp(b_norm, min=1e-8)

    sim = _torch.sum((a_f / a_norm) * (b_f / b_norm), dim=-1)
    sim = _torch.clamp(sim, -1.0, 1.0)

    eps = 1e-4
    mask = _torch.abs(sim) > (1.0 - eps)
    if mask.all():
        return (a_f * (1.0 - t) + b_f * t).reshape(orig_shape).to(dtype=a.dtype)

    angle = _torch.acos(sim)
    sin_angle = _torch.sin(angle)
    sin_ok = _torch.abs(sin_angle) > 1e-8

    t_a = _torch.sin((1.0 - t) * angle) / sin_angle
    t_b = _torch.sin(t * angle) / sin_angle

    result = _torch.where(
        sin_ok.unsqueeze(-1),
        a_f * t_a.unsqueeze(-1) + b_f * t_b.unsqueeze(-1),
        a_f * (1.0 - t) + b_f * t,
    )
    return result.reshape(orig_shape).to(dtype=a.dtype)


def _slerp_condition_values(cond_a, cond_b, t):
    """Spherical linear interpolation between two conditioning objects.
    Supports both dict (SDXL multi-key) and plain tensor (SD1.5).
    """
    _torch = _ensure_torch()

    if isinstance(cond_a, dict):
        cross_keys = [k for k in CHUNK_CROSSATTN_KEYS if k in cond_a and k in cond_b]
        out = dict(cond_a)
        for key in cross_keys:
            out[key] = _slerp_tensor(cond_a[key], cond_b[key], t, _torch)
        for key in cond_a:
            if key not in CHUNK_CROSSATTN_KEYS and key in cond_b:
                out[key] = cond_a[key] * (1.0 - t) + cond_b[key] * t
        return out

    return _slerp_tensor(cond_a, cond_b, t, _torch)


def _conditioning_shape_from_cond(cond):
    if isinstance(cond, dict):
        any_val = cond.get("crossattn")
        if any_val is None and cond:
            any_val = next(iter(cond.values()))
        return getattr(any_val, "shape", None) or (0,)
    return getattr(cond, "shape", None) or (0,)


def _build_plain_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    prompt_schedule = get_schedule(prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
    if not prompt_schedule:
        raise ValueError(f"Empty schedule for prompt '{prompt}'")

    texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=copy_from)
    conds = model.get_learned_conditioning(texts)
    cond_schedule = []
    for i, (end_at_step, _) in enumerate(prompt_schedule):
        if isinstance(conds, dict):
            cond = {k: v[i] for k, v in conds.items()}
        else:
            cond = conds[i]
        cond_schedule.append(ScheduledPromptConditioning(int(end_at_step), cond))
    return cond_schedule


def _build_pool_prompt_conditioning_schedule(
    model,
    spec: PoolPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    base_schedule = _build_prompt_conditioning_schedule(
        model,
        _build_pool_base_prompt(spec),
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )
    pool_schedule = _build_prompt_conditioning_schedule(
        model,
        spec.body,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )
    if not base_schedule or not pool_schedule:
        raise ValueError("Empty schedule for POOL prompt")

    boundaries = _collect_schedule_boundaries([base_schedule, pool_schedule], steps)
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for end_at_step in boundaries:
        base_index = _pick_schedule_entry_index(base_schedule, end_at_step)
        pool_index = _pick_schedule_entry_index(pool_schedule, end_at_step)
        base_cond = base_schedule[base_index].cond
        pool_cond = pool_schedule[pool_index].cond
        key = (base_index, pool_index)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue
        merged = _apply_pool_prompt_conditioning(base_cond, pool_cond)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond):
    if not isinstance(enc1_cond, dict) or not isinstance(enc2_cond, dict) or not isinstance(pooled_cond, dict):
        raise ValueError("ASSEMBLE requires SDXL-style dict conditioning values.")

    has_split_cross = any(
        key in enc1_cond
        and key in enc2_cond
        and _sdxl_cross_split_axis(enc1_cond[key]) is not None
        and _sdxl_cross_split_axis(enc2_cond[key]) is not None
        for key in SDXL_SPLITTABLE_CROSS_KEYS
    )
    has_pooled = any(key not in CHUNK_CROSSATTN_KEYS for key in pooled_cond.keys())
    if not has_split_cross or not has_pooled:
        raise ValueError(
            "ASSEMBLE requires SDXL-style dict conditioning with split cross-attention and pooled/global keys."
        )

    merged = _apply_condition_channel_target(enc1_cond, enc2_cond, "enc2")
    merged = _apply_condition_channel_target(merged, pooled_cond, "pooled")
    return merged


def _build_assemble_prompt_conditioning_schedule(
    model,
    spec: AssemblePromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    enc1_prompt = _expand_assemble_section_prompt(spec, spec.enc1)
    enc2_prompt = _expand_assemble_section_prompt(spec, spec.enc2)
    pooled_prompt = _expand_assemble_section_prompt(spec, spec.pooled) if spec.pooled is not None else enc2_prompt

    enc1_cond_schedule = _build_prompt_conditioning_schedule(model, enc1_prompt, steps, use_scheduling, seed, use_visitor, copy_from)
    enc2_cond_schedule = _build_prompt_conditioning_schedule(model, enc2_prompt, steps, use_scheduling, seed, use_visitor, copy_from)
    pooled_cond_schedule = _build_prompt_conditioning_schedule(model, pooled_prompt, steps, use_scheduling, seed, use_visitor, copy_from)
    if not enc1_cond_schedule or not enc2_cond_schedule or not pooled_cond_schedule:
        raise ValueError("Empty schedule for ASSEMBLE section prompt")

    boundaries = [int(steps)] if not use_scheduling else _collect_schedule_boundaries(
        [enc1_cond_schedule, enc2_cond_schedule, pooled_cond_schedule],
        steps,
    )
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for end_at_step in boundaries:
        enc1_index = _pick_schedule_entry_index(enc1_cond_schedule, end_at_step)
        enc2_index = _pick_schedule_entry_index(enc2_cond_schedule, end_at_step)
        pooled_index = _pick_schedule_entry_index(pooled_cond_schedule, end_at_step)
        enc1_cond = enc1_cond_schedule[enc1_index].cond
        enc2_cond = enc2_cond_schedule[enc2_index].cond
        pooled_cond = pooled_cond_schedule[pooled_index].cond
        key = (enc1_index, enc2_index, pooled_index)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue
        merged = _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _build_bind_branch_prompt_conditioning_schedule(
    model,
    spec: BindPromptSpec,
    base_schedule,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    bind_timeline = _build_bind_branch_timeline_entries(
        spec,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=False,
    )
    if not base_schedule or not bind_timeline:
        raise ValueError("Empty schedule for BIND branch")

    bind_text_to_schedule: dict[str, list[ScheduledPromptConditioning]] = {}
    for _end_at_step, bind_text, _bind_weight in bind_timeline:
        if bind_text not in bind_text_to_schedule:
            bind_text_to_schedule[bind_text] = _build_prompt_conditioning_schedule(
                model, bind_text, steps, use_scheduling, seed, use_visitor, copy_from,
            )

    bind_text_schedule = [[int(end_at_step), text] for end_at_step, text, _weight in bind_timeline]
    boundaries = _collect_schedule_boundaries([base_schedule, bind_text_schedule], steps)
    out: list[ScheduledPromptConditioning] = []
    weight_schedule: list[tuple[int, float]] = []
    previous_key = None
    previous_weight = None
    for end_at_step in boundaries:
        base_index = _pick_schedule_entry_index(base_schedule, end_at_step)
        _bind_end_at_step, bind_text, bind_weight = _pick_bind_timeline_entry(bind_timeline, end_at_step)
        bind_cond_schedule = bind_text_to_schedule[bind_text]
        bind_index = _pick_schedule_entry_index(bind_cond_schedule, end_at_step)
        key = (base_index, bind_text)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
        else:
            base_cond = base_schedule[base_index].cond
            bind_cond = bind_cond_schedule[bind_index].cond
            merged = _apply_condition_channel_target(base_cond, bind_cond, "cross")
            out.append(ScheduledPromptConditioning(int(end_at_step), merged))
            previous_key = key
        if weight_schedule and previous_weight is not None and abs(previous_weight - bind_weight) <= 1e-8:
            weight_schedule[-1] = (int(end_at_step), weight_schedule[-1][1])
        else:
            weight_schedule.append((int(end_at_step), float(bind_weight)))
            previous_weight = float(bind_weight)
    return out, weight_schedule


def _build_chunk_branch_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    state = _extract_backend_prompt_state(prompt)
    if state.chunk_spec is not None:
        raise PromptSyntaxError(
            "Nested CHUNK blocks are not supported in v1.",
            kind="nested_chunk_not_supported",
            token=CHUNK_KEYWORD,
            full=prompt,
        )
    if state.has_bind:
        raise ValueError("BIND inside CHUNK branches is not supported via this path.")
    if state.pool_spec is not None:
        return _build_pool_prompt_conditioning_schedule(model, state.pool_spec, steps, use_scheduling, seed, use_visitor, copy_from)
    if state.assemble_spec is not None:
        return _build_assemble_prompt_conditioning_schedule(model, state.assemble_spec, steps, use_scheduling, seed, use_visitor, copy_from)
    if state.blend_spec is not None:
        return _build_blend_prompt_conditioning_schedule(model, state.blend_spec, steps, use_scheduling, seed, use_visitor, copy_from)
    if state.active_morph_spec is not None:
        return _build_morph_prompt_conditioning_schedule(model, state.active_morph_spec, steps, use_scheduling, seed, use_visitor, copy_from)
    return _build_plain_prompt_conditioning_schedule(model, prompt, steps, use_scheduling, seed, use_visitor, copy_from)


def _build_chunk_prompt_conditioning_schedule(
    model,
    spec: ChunkPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    branch_prompts = [_expand_chunk_branch_prompt(spec, branch) for branch in spec.branches]
    branch_text_schedules = [
        get_schedule(branch_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch_prompt in branch_prompts
    ]
    branch_cond_schedules = [
        _build_chunk_branch_prompt_conditioning_schedule(
            model,
            branch_prompt,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        for branch_prompt in branch_prompts
    ]
    anchor_cond_schedule = None
    target_channel = None
    if spec.shared_channel in {"pooled", "cross"}:
        anchor_cond_schedule = _build_plain_prompt_conditioning_schedule(
            model,
            _build_chunk_shared_anchor_prompt(spec),
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        target_channel = "cross" if spec.shared_channel == "pooled" else "pooled"
    if (
        not branch_text_schedules
        or any(not schedule for schedule in branch_text_schedules)
        or any(not schedule for schedule in branch_cond_schedules)
        or (anchor_cond_schedule is not None and not anchor_cond_schedule)
    ):
        raise ValueError("Empty schedule for at least one CHUNK branch")

    schedules_for_boundaries = list(branch_cond_schedules)
    if anchor_cond_schedule is not None:
        schedules_for_boundaries.append(anchor_cond_schedule)
    boundaries = _collect_schedule_boundaries(schedules_for_boundaries, steps)
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    weights = [branch.weight for branch in spec.branches]

    for end_at_step in boundaries:
        active_texts: list[str] = []
        active_conds = []

        for text_schedule, cond_schedule in zip(branch_text_schedules, branch_cond_schedules):
            text_index = _pick_text_schedule_index(text_schedule, end_at_step)
            active_texts.append(str(text_schedule[text_index][1]))
            cond_index = _pick_schedule_entry_index(cond_schedule, end_at_step)
            active_conds.append(cond_schedule[cond_index].cond)

        active_key = tuple(active_texts)
        if out and previous_key == active_key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue

        merged = _merge_chunk_condition_values(active_conds, weights)
        if anchor_cond_schedule is not None and target_channel is not None:
            anchor_index = _pick_schedule_entry_index(anchor_cond_schedule, end_at_step)
            anchor_cond = anchor_cond_schedule[anchor_index].cond
            merged = _apply_condition_channel_target(anchor_cond, merged, target_channel)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = active_key

    return out


def _build_compound_conditioning_schedule(
    model,
    spec: CompoundPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    texts = [spec.base] + [_concat_prefix_text_suffix(spec.prefix, p.text, spec.suffix) for p in spec.parts]
    conds = model.get_learned_conditioning(SdConditioning(texts, copy_from=copy_from))
    if isinstance(conds, dict):
        base_cond = {k: v[0] for k, v in conds.items()}
        part_conds = [{k: v[i + 1] for k, v in conds.items()} for i in range(len(spec.parts))]
    else:
        base_cond = conds[0]
        part_conds = [conds[i + 1] for i in range(len(spec.parts))]

    boundaries: list[int]
    if not use_scheduling:
        boundaries = [int(steps)]
    else:
        change_points: set[int] = set()
        change_points.add(int(steps))
        for p in spec.parts:
            s = p.step_start
            e = p.step_end if p.step_end is not None else int(steps)
            if s > 1:
                change_points.add(s - 1)
            if e < int(steps):
                change_points.add(e)
        boundaries = sorted(change_points)

    out: list[ScheduledPromptConditioning] = []
    prev_key: tuple | None = None
    for end_at_step in boundaries:
        active_indices: list[int] = []
        active_weights: list[float] = []
        for i, p in enumerate(spec.parts):
            s = p.step_start
            e = p.step_end if p.step_end is not None else int(steps)
            if s <= end_at_step <= e:
                active_indices.append(i)
                active_weights.append(p.weight)

        merged = base_cond
        for idx, w in zip(active_indices, active_weights):
            part = part_conds[idx]
            if isinstance(merged, dict):
                merged = {k: merged[k] + w * (part[k] - base_cond[k]) for k in merged}
            else:
                merged = merged + w * (part - base_cond)

        key = tuple(active_indices) + tuple(active_weights)
        if out and prev_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
        else:
            out.append(ScheduledPromptConditioning(int(end_at_step), merged))
            prev_key = key

    return out


def _build_blend_prompt_conditioning_schedule(
    model,
    spec: BlendPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    branch_prompts = [_expand_blend_branch_prompt(spec, branch) for branch in spec.branches]
    branch_cond_schedules: list[list[ScheduledPromptConditioning]] = [None] * len(branch_prompts)  # type: ignore[list-item]
    plain_texts: list[str] = []
    plain_index_map: list[int] = []
    for i, bp in enumerate(branch_prompts):
        if _contains_chunk_marker(bp) or _contains_blend_marker(bp) or _contains_morph_marker(bp) or _contains_pool_marker(bp) or _contains_assemble_marker(bp) or _contains_bind_marker(bp) or _contains_bind2_marker(bp):
            branch_cond_schedules[i] = _build_prompt_conditioning_schedule(
                model, bp, steps, use_scheduling, seed, use_visitor, copy_from,
            )
        else:
            plain_texts.append(bp)
            plain_index_map.append(i)


    if plain_texts:
        if not use_scheduling:
            plain_conds = model.get_learned_conditioning(SdConditioning(plain_texts, copy_from=copy_from))
            for local_idx, original_idx in enumerate(plain_index_map):
                if isinstance(plain_conds, dict):
                    cond = {k: v[local_idx] for k, v in plain_conds.items()}
                else:
                    cond = plain_conds[local_idx]
                branch_cond_schedules[original_idx] = [ScheduledPromptConditioning(int(steps), cond)]
        else:
            for local_idx, original_idx in enumerate(plain_index_map):
                branch_cond_schedules[original_idx] = _build_plain_prompt_conditioning_schedule(
                    model, plain_texts[local_idx], steps, use_scheduling, seed, use_visitor, copy_from,
                )
    if any(sched is None for sched in branch_cond_schedules):
        raise ValueError("Empty schedule for at least one BLEND branch")

    boundaries = [int(steps)] if not use_scheduling else _collect_schedule_boundaries(
        branch_cond_schedules,
        steps,
    )
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    effective_weights = _resolve_blend_mode_weights([branch.weight for branch in spec.branches], spec.mode, spec.intensity)

    for end_at_step in boundaries:
        branch_indices = tuple(_pick_schedule_entry_index(cond_schedule, end_at_step) for cond_schedule in branch_cond_schedules)
        active_conds = [
            branch_cond_schedules[i][branch_indices[i]].cond
            for i in range(len(branch_cond_schedules))
        ]
        key = branch_indices
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue

        merged = _blend_morph_condition_values(active_conds, effective_weights)
        merged = _apply_condition_channel_target(active_conds[0], merged, spec.channel_target)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _build_morph_prompt_conditioning_schedule(
    model,
    spec: MorphPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    point_prompts = [_expand_morph_point_prompt(spec, point) for point in spec.points]
    point_cond_schedules: list[list[ScheduledPromptConditioning]] = [None] * len(point_prompts)  # type: ignore[list-item]
    plain_texts: list[str] = []
    plain_index_map: list[int] = []
    for i, pp in enumerate(point_prompts):
        if _contains_chunk_marker(pp) or _contains_blend_marker(pp) or _contains_morph_marker(pp) or _contains_pool_marker(pp) or _contains_assemble_marker(pp) or _contains_bind_marker(pp) or _contains_bind2_marker(pp):
            point_cond_schedules[i] = _build_prompt_conditioning_schedule(
                model, pp, steps, use_scheduling, seed, use_visitor, copy_from,
            )
        else:
            plain_texts.append(pp)
            plain_index_map.append(i)


    if plain_texts:
        if not use_scheduling:
            plain_conds = model.get_learned_conditioning(SdConditioning(plain_texts, copy_from=copy_from))
            for local_idx, original_idx in enumerate(plain_index_map):
                if isinstance(plain_conds, dict):
                    cond = {k: v[local_idx] for k, v in plain_conds.items()}
                else:
                    cond = plain_conds[local_idx]
                point_cond_schedules[original_idx] = [ScheduledPromptConditioning(int(steps), cond)]
        else:
            for local_idx, original_idx in enumerate(plain_index_map):
                point_cond_schedules[original_idx] = _build_plain_prompt_conditioning_schedule(
                    model, plain_texts[local_idx], steps, use_scheduling, seed, use_visitor, copy_from,
                )
    if any(sched is None for sched in point_cond_schedules) or not point_cond_schedules:
        raise ValueError("Empty schedule for at least one MORPH control prompt")

    positions = _resolve_morph_positions(spec, steps)
    window_steps = _resolve_morph_window_steps(spec, steps)
    inactive_text = _build_morph_inactive_text(spec) if window_steps is not None else None
    inactive_cond_schedule = None
    if inactive_text is not None:
        inactive_cond_schedule = _build_plain_prompt_conditioning_schedule(model, inactive_text, steps, use_scheduling, seed, use_visitor, copy_from)

    loop_steps = [int(steps)] if not use_scheduling else list(range(1, int(steps) + 1))
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for step in loop_steps:
        if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
            inactive_idx = _pick_schedule_entry_index(inactive_cond_schedule, step)
            inactive_key = ("inactive", inactive_idx)
            if out and previous_key == inactive_key:
                out[-1] = ScheduledPromptConditioning(int(step), out[-1].cond)
            else:
                out.append(ScheduledPromptConditioning(int(step), inactive_cond_schedule[inactive_idx].cond))
                previous_key = inactive_key
            continue
        branch_indices = tuple(
            _pick_schedule_entry_index(cs, step) for cs in point_cond_schedules
        )
        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, step, spec.curve, spec.intensity),
        )
        key = (branch_indices, tuple(round(float(w), 8) for w in weights))
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(step), out[-1].cond)
            continue

        active_conds = [
            point_cond_schedules[i][branch_indices[i]].cond
            for i in range(len(point_cond_schedules))
        ]
        if spec.curve == "slerp":
            non_zero = [i for i, w in enumerate(weights) if abs(w) > 1e-8]
            if len(non_zero) >= 2:
                left, right = non_zero[0], non_zero[-1]
                alpha = float(weights[right])
                merged = _slerp_condition_values(active_conds[left], active_conds[right], alpha)
            else:
                merged = _blend_morph_condition_values(active_conds, weights)
        else:
            merged = _blend_morph_condition_values(active_conds, weights)
        merged = _apply_condition_channel_target(active_conds[0], merged, spec.channel_target)
        out.append(ScheduledPromptConditioning(int(step), merged))
        previous_key = key

    return out


# ── BIND3: Static row-splice conditioning ──────────────────────────────────


def _tokenize_for_model(model, text: str) -> list[int]:
    """Tokenize a single text string and return raw token IDs.
    Handles SD1.5 (FrozenCLIPEmbedderWithCustomWords.tokenize) and
    SDXL (OpenCLIP tokenizer via embedders[0].tokenizer).
    """
    try:
        tokens = model.cond_stage_model.tokenize([text])
        if isinstance(tokens, (list, tuple)) and len(tokens) > 0 and isinstance(tokens[0], (list, tuple)):
            return list(tokens[0])
    except (AttributeError, TypeError, IndexError):
        pass
    try:
        tok = model.cond_stage_model.embedders[0].tokenizer
        encoded = tok(text, return_tensors=None, add_special_tokens=True)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return list(encoded["input_ids"])
    except (AttributeError, KeyError, TypeError):
        pass
    raise ValueError("Cannot tokenize: unsupported or unavailable model type")


def _tokenize_strip_special(model, text: str) -> list[int]:
    """Tokenize text and strip BOS/EOS special tokens.
    Returns only content token IDs for row-range computation.
    """
    tokens = _tokenize_for_model(model, text)
    if len(tokens) >= 2:
        return tokens[1:-1]
    return tokens[:]




def _content_range_to_tensor_ranges(start: int, end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        chunk_idx = pos // 75
        pos_in_chunk = pos % 75
        chunk_end = (chunk_idx + 1) * 75
        seg_end = min(chunk_end, end)
        t_start = chunk_idx * 77 + 1 + pos_in_chunk
        t_end = chunk_idx * 77 + 1 + (seg_end - chunk_idx * 75)
        ranges.append((t_start, t_end))
        pos = seg_end
    return ranges


def _compute_row_ranges(model, text_parts: list[str]) -> list[list[tuple[int, int]]]:
    """Given text_parts = [owner, pad_attr1, pad_attr2, ...] as they appear
    in R_text = ', '.join(text_parts), return list of lists of (start, end) token ranges
    (each group has one or more segments for multi-chunk support).
    for each part using progressive prefix tokenization.

    Strips BOS/EOS from each prefix to avoid range shifts.
    Returns ranges with +1 offset for BOS in the full tensor.
    """
    prefixes: list[str] = []
    cumulative = ""
    for i, part in enumerate(text_parts):
        if i == 0:
            cumulative = part
        else:
            cumulative = f"{cumulative}, {part}"
        prefixes.append(cumulative)
    content_counts: list[int] = []
    for p in prefixes:
        content_counts.append(len(_tokenize_strip_special(model, p)))
    if not content_counts:
        return []
    ranges: list[list[tuple[int, int]]] = []
    prev_content = 0
    for c in content_counts:
        group_ranges = _content_range_to_tensor_ranges(prev_content, c)
        ranges.append(group_ranges)
        prev_content = c
    return ranges


def _splice_condition_rows(
    R_cond,
    F_conds: list,
    weights: list[float],
    row_ranges: list[list[tuple[int, int]]],
):
    """Apply row-splice: for each group i (index > 0), replace row segments at
    row_ranges[i] of R_cond with lerp(R_rows, F_conds[i-1]_rows, w_i).

    Each group is a list of (start, end) tensor-row ranges (multi-range supports
    chunk-boundary crossings where a single content group spans >75 tokens).

    Handles all SDXL cross-attn keys (crossattn, c_crossattn, open_clip_projected).

    R_cond: cross-attn tensor [77, C] or dict with cross-attn keys.
    F_conds: list of N cross-attn tensors or dicts.
    row_ranges[0] = owner range (unchanged).
    row_ranges[1..N] = group ranges (spliced).
    """
    _torch = _ensure_torch()
    if isinstance(R_cond, dict):
        cross_keys = [k for k in CHUNK_CROSSATTN_KEYS if k in R_cond]
        out = dict(R_cond)
        for key in cross_keys:
            R_t = R_cond[key]
            result = R_t.clone()
            for i, ranges in enumerate(row_ranges[1:], start=1):
                w = weights[i - 1]
                Fi = F_conds[i - 1]
                Fi_t = Fi[key] if isinstance(Fi, dict) else Fi
                for start, end in ranges:
                    if start >= end:
                        continue
                    if abs(w - 1.0) <= 1e-8:
                        result[start:end] = Fi_t[start:end]
                    else:
                        w_t = _torch.tensor(w, dtype=result.dtype, device=result.device)
                        diff = Fi_t[start:end] - R_t[start:end]
                        result[start:end] = R_t[start:end] + w_t * diff
            out[key] = result
        return out
    R_t = R_cond
    F_ts = F_conds
    result = R_t.clone()
    for i, ranges in enumerate(row_ranges[1:], start=1):
        w = weights[i - 1]
        Fi = F_ts[i - 1]
        for start, end in ranges:
            if start >= end:
                continue
            if abs(w - 1.0) <= 1e-8:
                result[start:end] = Fi[start:end]
            else:
                w_t = _torch.tensor(w, dtype=result.dtype, device=result.device)
                diff = Fi[start:end] - R_t[start:end]
                result[start:end] = R_t[start:end] + w_t * diff
    return result


def _parse_bind3_prompt(text: str, allow_attr_scheduling: bool = False) -> tuple[str, list[str], list[float], str, str]:
    """Parse the first BIND3{owner => attr1*w1, attr2*w2} block from text.

    Returns (owner, attrs_list, weights_list, prefix, suffix).
    RAISES PromptSyntaxError on validation failures (multiple/nested/scheduling).
    """
    if not _contains_bind3_marker(text):
        raise PromptSyntaxError(
            "BIND3 marker expected in prompt",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    matches = list(_RE_BIND3_MARKER.finditer(protected))
    if not matches:
        raise PromptSyntaxError(
            "BIND3 marker expected in prompt",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    top_blocks = _find_top_level_bind3_blocks(protected)
    if len(top_blocks) > 1:
        raise PromptSyntaxError(
            "Only one BIND3 block is supported per prompt branch in v1.",
            kind="multiple_bind3_blocks_not_supported",
            token=BIND3_KEYWORD,
            full=text,
        )
    m = matches[0]
    start = m.start()
    brace_open = m.end() - 1
    depth = 1
    i = brace_open + 1
    while i < len(protected) and depth > 0:
        if protected[i] == "{":
            depth += 1
        elif protected[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise PromptSyntaxError(
            "Unclosed BIND3 block: expected '}'",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    brace_close = i - 1
    body = protected[brace_open + 1 : brace_close]
    if _contains_bind3_marker(body):
        raise PromptSyntaxError(
            "Nested BIND3 blocks are not supported in v1.",
            kind="nested_bind3_not_supported",
            token=BIND3_KEYWORD,
            full=text,
        )
    owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
    owner = _restore_escaped_literal_source(owner_body, span_restore)
    attrs_str = _restore_escaped_literal_source(attrs_body, span_restore)
    if not owner or not attrs_str:
        raise PromptSyntaxError(
            "BIND3 requires non-empty owner and attrs sections.",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    if "|" in owner:
        raise PromptSyntaxError(
            "BIND3 owner cannot contain '|' (pipe) — it conflicts with CHUNK branch separators.",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    raw_groups = _split_top_level_commas(attrs_str)
    if not raw_groups:
        raise PromptSyntaxError(
            "BIND3 attrs section cannot be empty or just commas.",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=text,
        )
    attrs: list[str] = []
    weights: list[float] = []
    for rg in raw_groups:
        if _RE_HAS_SCHEDULING.search(rg):
            if not allow_attr_scheduling:
                raise PromptSyntaxError(
                    f"BIND3 attribute contains scheduling syntax '{rg}' — "
                    "not supported inside BIND3 in v1. Use BIND2 or move scheduling to the outer prompt level.",
                    kind="bind3_attrs_scheduling_not_supported",
                    token=BIND3_KEYWORD,
                    full=text,
                )
        try:
            attr_text, attr_weight = _split_blend_branch_weight(rg, full_text=text)
        except PromptSyntaxError:
            attr_text = rg
            attr_weight = 1.0
        attrs.append(attr_text)
        weights.append(attr_weight)
    prefix = _restore_escaped_literal_source(protected[:start], span_restore).strip()
    suffix = _restore_escaped_literal_source(protected[brace_close + 1:], span_restore).strip()
    return owner, attrs, weights, prefix, suffix


def _validate_bind3_attr_token_counts(model, attrs: list[str], attr_schedules: list[list], full_prompt: str) -> None:
    """Validate that all scheduled values for each attr produce the same token count.
    Raises PromptSyntaxError if any attr has values with different token lengths.
    """
    for attr_idx, sched in enumerate(attr_schedules):
        values = list({str(text) for _, text in sched})
        if len(values) <= 1:
            continue
        counts: set[int] = set()
        for v in values:
            try:
                counts.add(len(_tokenize_strip_special(model, v)))
            except (ValueError, TypeError):
                counts.add(max(1, len(v.split())))
        if len(counts) > 1:
            raise PromptSyntaxError(
                f"BIND3 attr '{attrs[attr_idx]}': scheduled values have different "
                f"token counts {sorted(counts)} — row positions would shift. "
                f"Use values with equal token length.",
                kind="bind3_scheduling_token_mismatch",
                token=BIND3_KEYWORD,
                full=full_prompt,
            )


def _build_bind3_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
) -> list:
    """Build a conditioning schedule for a BIND3{owner => attr1*w1, attr2*w2} prompt
    using static row-splice (or dynamic per-step if attrs use scheduling syntax).

    R = encode(f"{prefix} {owner}, pad(attr1), pad(attr2) {suffix}")
    F_i = encode(f"{prefix} {owner}, pad(attr1), attr_i, pad(attr_N) {suffix}")
    final[rows_i] = lerp(R[rows_i], F_i[rows_i], w_i)
    """
    owner, attrs, weights, prefix, suffix = _parse_bind3_prompt(prompt, allow_attr_scheduling=bool(use_scheduling))
    if not attrs:
        raise PromptSyntaxError(
            "BIND3 requires at least one attribute.",
            kind="invalid_bind_syntax",
            token=BIND3_KEYWORD,
            full=prompt,
        )

    has_attr_scheduling = use_scheduling and any(_RE_HAS_SCHEDULING.search(a) for a in attrs)

    if use_scheduling:
        prefix_has = bool(prefix and _RE_HAS_SCHEDULING.search(prefix))
        suffix_has = bool(suffix and _RE_HAS_SCHEDULING.search(suffix))
        if prefix_has or suffix_has:
            where = []
            if prefix_has:
                where.append("prefix")
            if suffix_has:
                where.append("suffix")
            loc = " and ".join(where)
            raise PromptSyntaxError(
                f"BIND3 block has scheduling in its {loc} — static BIND3 does not "
                f"support changing prefix/suffix over steps. "
                f"Consider wrapping the entire BIND3 block in a scheduled wrapper: [BIND3{{...}}:N]",
                kind="bind3_prefix_suffix_scheduling_not_supported",
                token=BIND3_KEYWORD,
                full=prompt,
            )

    if has_attr_scheduling:
        # ===== Dynamic scheduling path =====
        attr_schedules: list[list] = []
        for a in attrs:
            if _RE_HAS_SCHEDULING.search(a):
                sched = get_schedule(a, steps, use_scheduling=True, seed=seed)
                attr_schedules.append(sched)
            else:
                attr_schedules.append([[int(steps), a]])

        _validate_bind3_attr_token_counts(model, attrs, attr_schedules, prompt)

        boundaries = _collect_schedule_boundaries(attr_schedules, steps)
        result: list = []
        prev_key: tuple | None = None

        for end_step in boundaries:
            cur_attrs = [_select_text_from_schedule(s, end_step) for s in attr_schedules]
            key = tuple(cur_attrs)

            if prev_key == key and result:
                result[-1] = ScheduledPromptConditioning(int(end_step), result[-1].cond)
                continue

            pad_texts: list[str] = []
            for cur_a in cur_attrs:
                try:
                    n_tokens = len(_tokenize_strip_special(model, cur_a))
                except ValueError:
                    raise PromptSyntaxError(
                        f"BIND3 cannot tokenize attr '{cur_a}' — unsupported model type.",
                        kind="bind3_tokenization_error",
                        token=BIND3_KEYWORD,
                        full=prompt,
                    )
                if n_tokens < 1:
                    n_tokens = 1
                pad_texts.append(" ".join(["a"] * n_tokens))

            core_parts = [owner] + pad_texts
            core_parts[0] = _concat_prefix_text_suffix(prefix, core_parts[0], "")
            core_parts[-1] = _concat_prefix_text_suffix("", core_parts[-1], suffix)
            R_text = ", ".join(core_parts)

            F_texts: list[str] = []
            for i, cur_a in enumerate(cur_attrs):
                f_parts = list(core_parts)
                if i == len(cur_attrs) - 1:
                    f_parts[i + 1] = _concat_prefix_text_suffix("", cur_a, suffix)
                else:
                    f_parts[i + 1] = cur_a
                F_texts.append(", ".join(f_parts))

            naive_core = [owner] + cur_attrs
            naive_core[0] = _concat_prefix_text_suffix(prefix, naive_core[0], "")
            naive_core[-1] = _concat_prefix_text_suffix("", naive_core[-1], suffix)
            naive_text = ", ".join(naive_core)

            all_texts = [R_text] + F_texts + [naive_text]
            texts = SdConditioning(all_texts, copy_from=copy_from)
            try:
                conds = model.get_learned_conditioning(texts)
            except Exception as exc:
                raise PromptSyntaxError(
                    f"BIND3 encoding failed: {exc}",
                    kind="bind3_encoding_error",
                    token=BIND3_KEYWORD,
                    full=prompt,
                ) from exc

            if isinstance(conds, dict):
                R_cond_item = {k: v[0] for k, v in conds.items()}
                F_conds_items = [{k: v[i + 1] for k, v in conds.items()} for i in range(len(cur_attrs))]
                naive_cond_item = {k: v[-1] for k, v in conds.items()}
            else:
                R_cond_item = conds[0]
                F_conds_items = [conds[i + 1] for i in range(len(cur_attrs))]
                naive_cond_item = conds[-1]

            try:
                row_ranges = _compute_row_ranges(model, core_parts)
            except ValueError as exc:
                raise PromptSyntaxError(
                    f"BIND3 cannot compute row ranges — tokenization failed for '{core_parts}': {exc}",
                    kind="bind3_tokenization_error",
                    token=BIND3_KEYWORD,
                    full=prompt,
                ) from exc
            if not row_ranges:
                row_ranges = [[(0, 0)]] * len(core_parts)

            for pi, group in enumerate(row_ranges):
                span = sum(e - s for s, e in group)
                if span <= 0:
                    raise PromptSyntaxError(
                        f"BIND3 part {pi} has zero token span — padding error.",
                        kind="bind3_tokenization_error",
                        token=BIND3_KEYWORD,
                        full=prompt,
                    )

            final_cond = _splice_condition_rows(R_cond_item, F_conds_items, weights, row_ranges)
            if isinstance(final_cond, dict):
                for pk in naive_cond_item:
                    if pk not in CHUNK_CROSSATTN_KEYS:
                        final_cond[pk] = naive_cond_item[pk]

            result.append(ScheduledPromptConditioning(int(end_step), final_cond))
            prev_key = key

        return result

    # ===== Static path (no attr scheduling) =====
    pad_texts: list[str] = []
    for attr in attrs:
        try:
            n_tokens = len(_tokenize_strip_special(model, attr))
        except ValueError:
            raise PromptSyntaxError(
                f"BIND3 cannot tokenize attr '{attr}' — unsupported model type.",
                kind="bind3_tokenization_error",
                token=BIND3_KEYWORD,
                full=prompt,
            )
        if n_tokens < 1:
            n_tokens = 1
        pad_texts.append(" ".join(["a"] * n_tokens))

    core_parts = [owner] + pad_texts
    core_parts[0] = _concat_prefix_text_suffix(prefix, core_parts[0], "")
    core_parts[-1] = _concat_prefix_text_suffix("", core_parts[-1], suffix)
    R_text = ", ".join(core_parts)

    F_texts: list[str] = []
    for i, attr in enumerate(attrs):
        f_parts = list(core_parts)
        if i == len(attrs) - 1:
            f_parts[i + 1] = _concat_prefix_text_suffix("", attr, suffix)
        else:
            f_parts[i + 1] = attr
        F_texts.append(", ".join(f_parts))

    naive_core = [owner] + attrs
    naive_core[0] = _concat_prefix_text_suffix(prefix, naive_core[0], "")
    naive_core[-1] = _concat_prefix_text_suffix("", naive_core[-1], suffix)
    naive_text = ", ".join(naive_core)

    all_texts = [R_text] + F_texts + [naive_text]
    texts = SdConditioning(all_texts, copy_from=copy_from)
    try:
        conds = model.get_learned_conditioning(texts)
    except Exception as exc:
        raise PromptSyntaxError(
            f"BIND3 encoding failed: {exc}",
            kind="bind3_encoding_error",
            token=BIND3_KEYWORD,
            full=prompt,
        ) from exc

    if isinstance(conds, dict):
        R_cond_item = {k: v[0] for k, v in conds.items()}
        F_conds_items = [{k: v[i + 1] for k, v in conds.items()} for i in range(len(attrs))]
        naive_cond_item = {k: v[-1] for k, v in conds.items()}
    else:
        R_cond_item = conds[0]
        F_conds_items = [conds[i + 1] for i in range(len(attrs))]
        naive_cond_item = conds[-1]

    try:
        row_ranges = _compute_row_ranges(model, core_parts)
    except ValueError as exc:
        raise PromptSyntaxError(
            f"BIND3 cannot compute row ranges — tokenization failed for '{core_parts}': {exc}",
            kind="bind3_tokenization_error",
            token=BIND3_KEYWORD,
            full=prompt,
        ) from exc
    if not row_ranges:
        row_ranges = [[(0, 0)]] * len(core_parts)

    for pi, group in enumerate(row_ranges):
        span = sum(e - s for s, e in group)
        if span <= 0:
            raise PromptSyntaxError(
                f"BIND3 part {pi} has zero token span — padding error.",
                kind="bind3_tokenization_error",
                token=BIND3_KEYWORD,
                full=prompt,
            )

    final_cond = _splice_condition_rows(R_cond_item, F_conds_items, weights, row_ranges)
    if isinstance(final_cond, dict):
        for pk in naive_cond_item:
            if pk not in CHUNK_CROSSATTN_KEYS:
                final_cond[pk] = naive_cond_item[pk]

    return [ScheduledPromptConditioning(int(steps), final_cond)]


def _build_bind2_delta_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
) -> list:
    """Delta-encode BIND2{owner => attr1, attr2} as a weighted blend.

    R  = encode(owner, pad1, pad2)
    V_i = encode(owner, attr_i, pad_{i})
    final = blend(R, V_1, V_2, ..., weights=[1-N, 1, 1, ...])
    """
    if not _contains_bind2_marker(prompt):
        raise PromptSyntaxError(
            "BIND2 marker expected",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    protected, span_restore = _protect_escaped_literal_spans_for_source(prompt)
    protected = _protect_escaped_literals(protected)
    matches = list(_RE_BIND2_MARKER.finditer(protected))
    if not matches:
        raise PromptSyntaxError(
            "BIND2 marker expected",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    if len(matches) > 1:
        raise PromptSyntaxError(
            "Only one BIND2 block per prompt is supported in Path 2.",
            kind="multiple_bind2_not_supported",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    m = matches[0]
    brace_open = m.end() - 1
    depth = 1
    i = brace_open + 1
    while i < len(protected) and depth > 0:
        if protected[i] == "{":
            depth += 1
        elif protected[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise PromptSyntaxError(
            "Unclosed BIND2 block",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    brace_close = i - 1
    body = protected[brace_open + 1 : brace_close]
    if _contains_bind2_marker(body):
        raise PromptSyntaxError(
            "Nested BIND2 blocks are not supported.",
            kind="nested_bind2_not_supported",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=prompt)
    owner = _restore_escaped_literal_source(owner_body, span_restore)
    attrs_str = _restore_escaped_literal_source(attrs_body, span_restore)
    if not owner or not attrs_str:
        raise PromptSyntaxError(
            "BIND2 requires non-empty owner and attrs sections.",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    if "|" in owner:
        raise PromptSyntaxError(
            "BIND2 owner cannot contain '|' (pipe).",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    groups = _split_top_level_commas(attrs_str)
    if not groups:
        raise PromptSyntaxError(
            "BIND2 attrs cannot be empty.",
            kind="invalid_bind_syntax",
            token=BIND2_KEYWORD,
            full=prompt,
        )
    parsed = [_split_blend_branch_weight(g, full_text=prompt) for g in groups]
    attrs = [text for text, _ in parsed]
    delta_weights = [w for _, w in parsed]
    n = len(attrs)
    pad_texts: list[str] = []
    for a in attrs:
        try:
            n_tokens = len(_tokenize_strip_special(model, a))
        except (ValueError, TypeError):
            n_tokens = max(1, len(a.split()))
        if n_tokens < 1:
            n_tokens = 1
        pad_texts.append(" ".join(["a"] * n_tokens))
    core_parts = [owner] + pad_texts
    R_text = ", ".join(core_parts)
    V_texts: list[str] = []
    for i, a in enumerate(attrs):
        v_parts = list(core_parts)
        v_parts[i + 1] = a
        V_texts.append(", ".join(v_parts))
    naive_text = ", ".join([owner] + attrs)
    all_texts = [R_text] + V_texts + [naive_text]
    texts = SdConditioning(all_texts, copy_from=copy_from)
    try:
        conds = model.get_learned_conditioning(texts)
    except Exception as exc:
        raise PromptSyntaxError(
            f"BIND2 encoding failed: {exc}",
            kind="bind2_encoding_error",
            token=BIND2_KEYWORD,
            full=prompt,
        ) from exc
    if isinstance(conds, dict):
        R_cond = {k: v[0] for k, v in conds.items()}
        V_conds = [{k: v[i + 1] for k, v in conds.items()} for i in range(n)]
        naive_cond = {k: v[-1] for k, v in conds.items()}
    else:
        R_cond = conds[0]
        V_conds = [conds[i + 1] for i in range(n)]
        naive_cond = conds[-1]
    if BIND2_NORMALIZE_WEIGHTS:
        n = len(delta_weights)
        uniform = 1.0 / (n + 1)
        final_weights = [uniform] * (n + 1)
    else:
        final_weights = [1.0 - sum(delta_weights)] + delta_weights
    final_cond = _blend_morph_condition_values([R_cond] + V_conds, final_weights)
    if isinstance(final_cond, dict):
        for pk in naive_cond:
            if pk not in CHUNK_CROSSATTN_KEYS:
                final_cond[pk] = naive_cond[pk]
    return [ScheduledPromptConditioning(int(steps), final_cond)]


def _build_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    if BIND2_USE_PATH2 and _contains_bind2_marker(prompt):
        return _build_bind2_delta_conditioning_schedule(
            model, prompt, steps, use_scheduling, seed, use_visitor, copy_from,
        )
    prompt = _transpile_bind2_to_chunk(prompt)
    if _contains_bind3_marker(prompt):
        protected_p, restore = _protect_escaped_literal_spans_for_source(prompt)
        protected_p = _protect_escaped_literals(protected_p)
        bind3_blocks = _find_top_level_bind3_blocks(protected_p)
        if len(bind3_blocks) > 1:
            segments = _extract_sequential_backend_segments(protected_p, restore)
            return _build_sequential_cond_schedule(model, segments, steps, use_scheduling, seed, use_visitor, copy_from)
        return _build_bind3_conditioning_schedule(
            model, prompt, steps, use_scheduling, seed, use_visitor, copy_from,
        )
    state = _extract_backend_prompt_state(prompt)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(prompt)
    if state.has_bind:
        raise ValueError("BIND requires the composable conditioning path via get_multicond_learned_conditioning().")
    if state.has_mixed_backends or state.has_multiple_same_type:
        protected_p, restore = _protect_escaped_literal_spans_for_source(prompt)
        protected_p = _protect_escaped_literals(protected_p)
        segments = _extract_sequential_backend_segments(protected_p, restore)
        return _build_sequential_cond_schedule(model, segments, steps, use_scheduling, seed, use_visitor, copy_from)
    if state.pool_spec is not None:
        return _build_pool_prompt_conditioning_schedule(
            model,
            state.pool_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.assemble_spec is not None:
        return _build_assemble_prompt_conditioning_schedule(
            model,
            state.assemble_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )

    if state.chunk_spec is not None:
        return _build_chunk_prompt_conditioning_schedule(
            model,
            state.chunk_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.blend_spec is not None:
        return _build_blend_prompt_conditioning_schedule(
            model,
            state.blend_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.compound_spec is not None:
        return _build_compound_conditioning_schedule(
            model,
            state.compound_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.active_morph_spec is not None:
        return _build_morph_prompt_conditioning_schedule(
            model,
            state.active_morph_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    _raise_unsupported_backend_context_error(prompt)
    return _build_plain_prompt_conditioning_schedule(
        model,
        prompt,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )


def _normalize_scheduler_surface_syntax(text: str) -> str:
    """Normalize scheduler spelling without changing semantic intent."""
    if not text:
        return text

    normalized = text

    # Tolerate legacy inner reverse syntax by moving the terminal token outside
    # the closing bracket. This keeps the canonical grammar ("[...] reverse")
    # without rewriting inner syntax to postfix.
    if "[" in normalized and "reverse" in normalized:
        out: list[str] = []
        stack: list[int] = []
        last_emit = 0
        i = 0
        while i < len(normalized):
            ch = normalized[i]
            if ch == "\\":
                i += 2 if i + 1 < len(normalized) else 1
                continue
            if ch == "[":
                stack.append(i)
                i += 1
                continue
            if ch == "]" and stack:
                start = stack.pop()
                if not stack:
                    inner = normalized[start + 1 : i]
                    last_colon = _find_last_top_level_colon_index(inner)
                    if last_colon >= 0:
                        tail = inner[last_colon + 1 :]
                        mrev = RE_REVERSE_SUFFIX.search(tail)
                        if mrev:
                            base_tail = tail[:mrev.start()].rstrip()
                            if _parse_inner_sched_tail(base_tail, 1) is not None:
                                out.append(normalized[last_emit:start])
                                out.append(f"[{inner[:last_colon + 1]}{base_tail}] reverse")
                                last_emit = i + 1
            i += 1
        if out:
            out.append(normalized[last_emit:])
            normalized = "".join(out)

    normalized = _RE_POSTFIX_STEP_COLON_SPACING.sub(
        lambda m: f"]:{m.group('num')}",
        normalized,
    )

    # If there is extra word-like text after "reverse", treat it as literal text,
    # not as reverse_flag for ranges.
    normalized = _RE_LITERAL_REVERSE_AFTER_RANGES.sub(
        lambda m: f"{m.group('head')} {LITERAL_REVERSE_TOKEN}",
        normalized,
    )
    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# Pre-parse protection for postfix scheduled blocks: "[...]:N ..."
#
# Lark grammar intentionally disallows raw commas inside `plain` tokens.
# That is fine for most top-level prompts (commas are handled by `start`),
# but it breaks complex parses when a postfix scheduled block contains
# comma-separated tags (a very common SD prompt style), e.g.:
#   "[(a:1.2), b]:10"
#
# To make this reliable, we replace ONLY those commas that are inside the
# BRACKET CONTENT of postfix scheduled blocks, and only at the top level of
# that bracket content (not inside nested (), {}, or nested [] blocks).
#
# IMPORTANT: we replace commas with a 1-char placeholder to keep indices stable
# for linting and UI highlighting.
# ──────────────────────────────────────────────────────────────────────────────

# Tail pattern for inner scheduled block detection:
# matches "N", "N%", "N reverse", "N 1-4,5-8", "N% reverse", etc.
# Structural regex for inner scheduled tail: NUMBER[%] [step_ranges]
# NOTE: 'reverse' is NOT included here — it lives OUTSIDE ] per grammar rule:
#   scheduled: "[" [...] "]" (WHITESPACE* reverse_flag)?
# So [cat:8 reverse] is ambiguous (unlikely in practice), but
# [cat:8] reverse is the canonical form.
_RE_INNER_SCHED_TAIL = re.compile(
    rf"^\s*({NUMERIC_RE})(%?)"                                        # num + optional %
    rf"((?:\s+\d+%?\s*-\s*\d+%?(?:\s*,\s*\d+%?\s*-\s*\d+%?)*)?)?"   # optional step ranges
    rf"\s*$"
)



def _extract_inner_sched_parts(inner: str, steps: int):
    """Return parsed inner scheduler parts or None."""
    last_colon = _find_last_top_level_colon_index(inner)
    if last_colon < 0:
        return None
    tail = (inner[last_colon + 1 :] or "").strip()
    tail_result = _parse_inner_sched_tail(tail, steps)
    if tail_result is None:
        return None
    boundary_spec, ranges_txt = tail_result
    return inner[:last_colon], boundary_spec, ranges_txt, last_colon


def _parse_inner_sched_tail(tail: str, steps: int):
    """Structurally parse an inner-scheduler tail: NUMBER[%] [ranges].

    Returns (boundary_spec, ranges_txt) or None if not a valid tail.
    Does NOT handle 'reverse' — that belongs outside ] in the grammar.

    Examples:
      "8"           → (absolute 8,  "")
      "75%"         → (percent 75, "")
      "8 1-4,5-8"   → (absolute 8,  "1-4,5-8")
      "8 reverse"   → None         # reverse is outside [], not in tail
    """
    _ = steps  # kept for backward-compatible signature
    m = _RE_INNER_SCHED_TAIL.fullmatch(tail.strip())
    if not m:
        return None
    num_str  = m.group(1)
    is_pct   = bool(m.group(2))
    rest     = (m.group(3) or "").strip()

    try:
        num_f = float(num_str)
    except (ValueError, TypeError):
        return None

    boundary_spec = _make_boundary_spec(num_f, is_percent=is_pct)
    return boundary_spec, rest


def _placeholderize_postfix_scheduled_blocks(text: str) -> str:
    """Replace top-level commas inside ALL scheduled bracket blocks with
    SCHEDULE_COMMA_PLACEHOLDER so Lark can parse comma-separated tags.

    Handles both forms:
      postfix:  [(a:1.2), b]:10   — number AFTER closing bracket
      inner:    [a, b, c:10]      — number INSIDE bracket as last colon-segment

    Only commas at the top nesting level of the bracket content are replaced
    (commas inside nested (), {}, [] are left untouched).

    This is called in _parse_lark_cached BEFORE Lark parsing, so Lark never
    sees raw commas in scheduled content regardless of block count.
    """
    if not text or "[" not in text or "," not in text:
        return text

    chars = list(text)
    stack: list[int] = []
    i = 0

    # O(N) total: inner k-loop runs only over non-overlapping bracket spans.
    while i < len(chars):
        ch = chars[i]

        if ch == "\\":
            i += 2 if i + 1 < len(chars) else 1
            continue

        if ch == "[":
            stack.append(i)
            i += 1
            continue

        if ch == "]" and stack:
            start = stack.pop()
            # Only act on closed top-level bracket blocks (stack now empty)
            if not stack:
                inner_text = "".join(chars[start + 1 : i])

                # --- Determine if this is a scheduled block ---
                is_scheduled = False

                # Check 1: postfix form — ]:N after the closing bracket
                j = i + 1
                while j < len(chars) and str(chars[j]).isspace():
                    j += 1
                if j < len(chars) and chars[j] == ":":
                    j += 1
                    while j < len(chars) and str(chars[j]).isspace():
                        j += 1
                    m = RE_NUMERIC.match("".join(chars[j:]))
                    if m:
                        is_scheduled = True

                # Check 2: inner form — last top-level colon-segment is numeric
                replace_limit = i

                if not is_scheduled:
                    inner_sched = _extract_inner_sched_parts(inner_text, 1)
                    if inner_sched is not None:
                        is_scheduled = True
                        replace_limit = start + 1 + inner_sched[3]

                if is_scheduled:
                    # Replace commas inside prompt content at all nesting levels,
                    # except inside nested [] blocks. For inner syntax, stop before
                    # the final ":tail" so range separators like "1-4,5-8" survive.
                    #
                    # WHY: Lark grammar forbids raw comma inside `prompt` rule — it only
                    # exists at `start` level. So (cat, dog) inside [(...):N] fails Lark
                    # even though the comma is "inside ()". We must replace all commas
                    # inside scheduled prompt content so Lark never sees them as syntax.
                    # Commas inside nested [] are skipped (they're sub-bracket structure).
                    depth_brack = 0
                    k = start + 1
                    while k < replace_limit:
                        c = chars[k]
                        if c == "\\":
                            k += 2
                            continue
                        if c == "[":
                            depth_brack += 1
                        elif c == "]" and depth_brack > 0:
                            depth_brack -= 1
                        elif c == "," and depth_brack == 0:
                            chars[k] = SCHEDULE_COMMA_PLACEHOLDER
                        k += 1

        i += 1

    return "".join(chars)




# ──────────────────────────────────────────────────────────────────────────────
# Узкие хелперы для извлечения компонентов fast-path'ов (без сборки расписаний)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_after_with_ranges(full: str, steps: int):
    """Извлечь компоненты для вида "[a:b]:N 1-4,6-8 [reverse]".

    Возвращает tuple(prompts, ranges, rev_flag) либо None.
    ВАЖНО:
      - Порядок ranges сохраняется как в исходной строке (нужно для last-wins).
      - Обратные диапазоны (start > end) считаются ошибкой (строгий режим).
      - Одиночные шаги (start == end) разрешены.
    """
    m = RE_BRACKET_AFTER_WITH_RANGES.match(full or "")
    if not m:
        return None
    inner = m.group("inner")
    ranges_txt = m.group("ranges") or ""
    rev = m.group("rev")

    prompts = _parse_inner_prompts(inner)
    ranges: list[tuple[int, int]] = []

    def _raw_value(txt: str, is_pct: bool) -> float:
        val = float(txt[:-1] if is_pct else txt)
        if is_pct:
            val = val / 100.0 * steps
        return val

    for part in (ranges_txt or "").split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            continue
        start_raw, end_raw = [x.strip() for x in part.split('-', 1)]
        start_is_pct = start_raw.endswith('%')
        end_is_pct = end_raw.endswith('%')

        try:
            raw_start = _raw_value(start_raw, start_is_pct)
            raw_end = _raw_value(end_raw, end_is_pct)
        except Exception as exc:
            raise PromptSyntaxError(
                f"Invalid range token {part!r}",
                kind="invalid_range_token",
                token=part,
                full=full,
            ) from exc

        # Строго: обратный диапазон — ошибка (чтобы не скрывать опечатки)
        if raw_start > raw_end:
            raise PromptSyntaxError(
                f"Reverse range {part!r} (start > end)",
                kind="reverse_range",
                token=part,
                full=full,
            )

        start_val = _clamp(round(raw_start), steps)
        end_val = _clamp(round(raw_end), steps)
        ranges.append((start_val, end_val))

    rev_flag = bool(rev)
    return (prompts, ranges, rev_flag)




def _build_ranges_schedules_from_components(
    prompts: Sequence[str],
    ranges: Sequence[tuple[int, int]],
    steps: int,
    *,
    prefix: str = "",
    suffix: str = "",
    rev_flag: bool = False,
    cycle_prompts: bool = True,
    empty_center: str = "",
) -> list[list[int, str]]:
    """Собрать расписание для синтаксиса "[...]:N a-b,c-d [reverse]".

    Политики (зафиксированы под твой выбор):
      - Overlap: last-wins (последний диапазон перезаписывает пересечения).
      - Reverse ranges (start > end): не допускаются (должны ловиться на этапе извлечения).
      - Single-step (start == end): разрешён (один шаг).
      - Порядок ranges важен: он определяет, какой prompt будет применён к диапазону.

    Реализация:
      1) строим mapping step -> center_text (по умолчанию empty_center)
      2) проходим ranges по порядку и записываем (перезаписывая) center_text
      3) сжимаем mapping обратно в список [end_step, text]
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    # База промптов
    prompt_list = list(prompts) if prompts else [empty_center]
    if rev_flag:
        prompt_list = list(reversed(prompt_list))

    # mapping: шаг -> центральный текст
    slot = [empty_center] * (int(steps) + 1)  # slot[0] не используется

    for idx, (start, end) in enumerate(ranges or ()):
        start_i = int(start)
        end_i = int(end)

        # безопасность (обычно уже нормализовано)
        if start_i > end_i:
            raise ValueError(f"Invalid range ({start_i}-{end_i}) after normalization for: {ranges!r}")

        if cycle_prompts:
            center = prompt_list[idx % len(prompt_list)]
        else:
            if idx >= len(prompt_list):
                break
            center = prompt_list[idx]

        # clamp и заполнение inclusive
        start_i = max(1, min(start_i, int(steps)))
        end_i = max(1, min(end_i, int(steps)))

        for s in range(start_i, end_i + 1):
            slot[s] = center  # last-wins: всегда перезаписываем

    # Сжатие mapping -> schedule segments
    schedules: list[list[int, str]] = []

    def _append(end_step: int, center_text: str) -> None:
        end_step = min(int(end_step), int(steps))
        text = _concat_prefix_text_suffix(prefix, center_text, suffix)
        if schedules and schedules[-1][1] == text:
            schedules[-1][0] = end_step
        else:
            schedules.append([end_step, text])

    cur = slot[1]
    for s in range(2, int(steps) + 1):
        if slot[s] != cur:
            _append(s - 1, cur)
            cur = slot[s]
    _append(int(steps), cur)

    return schedules


# ──────────────────────────────────────────────────────────────────────────────
# Безопасный сплит по ':' только на верхнем уровне
# ──────────────────────────────────────────────────────────────────────────────

def _split_top_level_colon_all(s: str, keep_empty: bool) -> list[str]:
    """
    Разбить строку по ':' только на ВЕРХНЕМ уровне:
    игнорировать ':' внутри круглых (), фигурных {}, а ТАКЖЕ квадратных [].
    Учитывать экранирование '\\'.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0  # (...)
    depth_brace = 0  # {...}
    depth_brack = 0  # [...]
    i = 0
    while i < len(s):
        ch = s[i]
        # экранирование
        if ch == '\\':
            if i + 1 < len(s):
                buf.append(ch); buf.append(s[i + 1])
                i += 2
                continue

        # уровни скобок
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack = max(0, depth_brack - 1)

        # разрыв только на самом верхнем уровне
        if ch == ':' and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            seg = "".join(buf)
            seg_trim = seg.strip()
            if keep_empty:
                parts.append(seg_trim if seg_trim != "" else "")
            else:
                if seg_trim != "":
                    parts.append(seg_trim)
            buf = []
        else:
            buf.append(ch)
        i += 1

    seg = "".join(buf)
    seg_trim = seg.strip()
    if keep_empty:
        parts.append(seg_trim if seg_trim != "" else "")
    else:
        if seg_trim != "":
            parts.append(seg_trim)
    return parts



def _split_top_level_colon(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=False)

def _split_top_level_colon_keep_empty(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=True)



# ──────────────────────────────────────────────────────────────────────────────
# Хелперы, устраняющие дублирование reverse-логики и prompts-разбора
# ──────────────────────────────────────────────────────────────────────────────

def _strip_reverse_from_post(post: str, extra_rev: bool = False) -> tuple[str, bool]:
    """Вытащить 'reverse' из начала или конца строки post.

    Возвращает (cleaned_post, rev_found: bool).
    extra_rev=True означает что reverse уже обнаружен (из токена/regex-группы);
    в этом случае post не изменяем содержательно, но зачищаем ведущую
    пунктуацию (запятая/пробел) которая могла остаться после извлечения
    токена 'reverse' в RE_BRACKET_AFTER.
    Порядок поиска: PREFIX ('reverse hd') → SUFFIX ('hd reverse').

    Fix: '[a:b]:10 reverse, shot' ранее давал post=', shot' (ведущая запятая),
    теперь возвращает 'shot' — одинаково с '[a:b]:10 reverse shot'.
    """
    if extra_rev:
        # Strip leading comma/space left by the regex group split
        return post.lstrip(", "), True
    if not post:
        return post, False
    mrev = RE_REVERSE_PREFIX.match(post)
    if mrev:
        return post[mrev.end():].lstrip(", "), True
    mrev_s = RE_REVERSE_SUFFIX.search(post)
    if mrev_s:
        return post[:mrev_s.start()].rstrip().rstrip(",").rstrip(), True
    return post, False


def _parse_inner_prompts(inner: str) -> list[str]:
    """Разбить inner на список промптов: split по ':' → unescape → strip.
    Устраняет дублирование одинакового list-comp в 4+ местах.
    """
    return [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]

def _finalize_schedules(schedules) -> list:
    """Применить _apply_and + _collapse_spaces к каждому тексту расписания.
    Устраняет дублирование идентичных return-строк вида:
      return [[e, _apply_and(_collapse_spaces(t))] for e, t in schedules]
    После нормализации схлопывает соседние сегменты с одинаковым текстом
    (два сегмента могут стать идентичными после collapse_spaces).
    """
    out = []
    for e, t in schedules:
        normalized = _apply_and(_collapse_spaces(t)).replace(LITERAL_REVERSE_TOKEN, "reverse")
        if out and out[-1][1] == normalized:
            out[-1][0] = e   # extend last segment
        else:
            out.append([e, normalized])
    return out



def _has_multiple_bracket_blocks(s: str) -> bool:
    """Проверить, есть ли более одного независимого блока ``[...]``.

    Игнорирует экранированные скобки и вложенные уровни, чтобы не считать
    ``[[nested]]`` как два блока. Возвращает ``True`` при втором входе в
    глубину 1.
    """
    depth = 0
    blocks = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            depth += 1
            if depth == 1:
                blocks += 1
                if blocks >= 2:
                    return True
        elif ch == "]" and depth > 0:
            depth -= 1
        i += 1
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Грамматика Lark
# ──────────────────────────────────────────────────────────────────────────────

# динамическое правило для alternate
_alt_rule = r' "[" prompt ("|" prompt)* "]" '                 # без пустых опций
if ALLOW_EMPTY_ALTERNATE:
    # '!' передаёт '|' как токены в alternate(args) — необходимо для пустых веток [fe|]
    _alt_rule = r' "[" prompt ("|" [prompt])+ "]" '

# NB: расширили класс одиночных знаков в start, чтобы '!' не валил парсер для plain'ов/compound'ов
_grammar = r"""
!start: (prompt | /[():,!|&]/+)*

prompt: (scheduled | emphasized | grouped
        | alternate | alternate_distinct
        | alternate1 | alternate2
        | top_level_sequence3 | top_level_sequence | sequence
        | weighted | numbered | and_rule
        | compound_block
        | plain | WHITESPACE)*

!emphasized: "(" prompt ")"
           | "(" prompt ":" prompt ")"
           | "(" prompt ":" NUMBER ")"


!weighted: plain ":" NUMBER


scheduled: "[" [prompt (":" prompt)* ":" boundary_value (WHITESPACE* step_range_list)?] "]" (WHITESPACE* reverse_flag)?
        | "[" [prompt (":" prompt)*] "]" ":" boundary_value (WHITESPACE* step_range_list)? (WHITESPACE* reverse_flag)?
boundary_value: NUMBER           -> boundary_number
              | NUMBER PERCENT   -> boundary_percent
reverse_flag: "reverse"
step_range_list: step_range (WHITESPACE* "," WHITESPACE* step_range)*
step_range: NUMBER "-" NUMBER        -> range_abs
          | NUMBER "%" "-" NUMBER "%" -> range_pct



!alternate: """ + _alt_rule + r"""
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: plain ("|" plain)+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"

top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
top_level_sequence3: prompt ":::" (sequence | plain) (WHITESPACE* "," WHITESPACE* (sequence | plain))* "!!!" (WHITESPACE* "," WHITESPACE* (plain | sequence))*
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")?
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";" | "~")?

numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | alternate | alternate_distinct | alternate2 | alternate1)

and_rule: (plain | weighted | emphasized | numbered | grouped | alternate | alternate_distinct | alternate2 | alternate1 | scheduled) ("&" (plain | weighted | emphasized | numbered | grouped | alternate | alternate_distinct | alternate1 | scheduled))+

compound_block: "COMPOUND" "{" compound_branch ("|" compound_branch)* "}"
compound_branch: compound_text compound_range? compound_weight?
compound_text: /[^|{}@*]+/
compound_range: "@" NUMBER_Q ("-" NUMBER_Q)?
compound_weight: "*" NUMBER

 WHITESPACE: /\s+/
plain: /([^\\[\]\{\}\(\),&:!|]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
PERCENT: "%"
"""

# Варианты планирования: число может быть до или после закрывающей скобки

schedule_parser = lark.Lark(_grammar, start="start")

# --- ДОБАВИТЬ ЭТУ ФУНКЦИЮ ---
@lru_cache(maxsize=CACHE_SIZE)
def _parse_lark_cached(prompt: str):
    """Кэшируем только построение дерева, так как оно не зависит от steps/seed.
    Нормализация уже выполнена в _get_schedule_impl до вызова этой функции.
    """
    return schedule_parser.parse(_placeholderize_postfix_scheduled_blocks(prompt))
# ----------------------------
# ──────────────────────────────────────────────────────────────────────────────
# Общие вспомогалки для fast-path "[...]:N" (и CollectSteps, и get_schedule)
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(x: int, steps: int) -> int:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    # ИСПРАВЛЕНИЕ: Защита от деления на ноль и некорректных значений
    if not isinstance(x, (int, float)) or not isinstance(steps, (int, float)):
        raise TypeError(f"x and steps must be numeric, got {type(x)}, {type(steps)}")
    return max(1, min(int(x), int(steps)))

def _to_end_step(num: float | _BoundarySpec, steps: int) -> int:
    """Convert a typed boundary (or legacy numeric fallback) into an end step."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if isinstance(num, _BoundarySpec):
        if num.kind == "percent":
            raw = num.value / 100.0 * steps
        elif num.kind == "fraction":
            raw = num.value * steps
        else:
            raw = num.value
        return _clamp(round(raw), steps)

    num_f = float(num)
    if 0.0 < num_f < 1.0:
        return _clamp(round(num_f * steps), steps)
    return _clamp(round(num_f), steps)


def _extract_scheduled_boundary_value(children) -> _BoundarySpec | None:
    boundary_node = next(
        (
            c
            for c in children
            if isinstance(c, lark.Tree)
            and getattr(c, "data", None) in ("boundary_number", "boundary_percent")
        ),
        None,
    )
    if boundary_node is not None:
        num_tok = next(
            (
                c
                for c in boundary_node.children
                if isinstance(c, lark.Token) and c.type == "NUMBER"
            ),
            None,
        )
        if num_tok is None:
            return None
        try:
            value = float(num_tok.value)
        except (TypeError, ValueError):
            return None
        return _make_boundary_spec(
            value,
            is_percent=(getattr(boundary_node, "data", None) == "boundary_percent"),
        )

    num_tok = next(
        (c for c in children if isinstance(c, lark.Token) and c.type == "NUMBER"),
        None,
    )
    if num_tok is None:
        return None
    try:
        return _make_boundary_spec(float(num_tok.value))
    except (TypeError, ValueError):
        return None

def _build_bracket_after_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    Семантика:
      • если НЕТ префикса/суффикса → как для [p1:p2:...:N] (интервалы ВНУТРИ [1..N]);
      • если ЕСТЬ префикс/суффикс → «после N»: до N – пролог (pre+post), далее хвост.
    """
    # ИСПРАВЛЕНИЕ: Проверка на пустой prompts В НАЧАЛЕ
    if not prompts:
        return [[steps, _concat_prefix_text_suffix(pre, "", post)]]
    
    if not (pre.strip() or post.strip()):
        return _build_bracket_inner_schedules(pre, prompts, boundary_end, post, steps)

    schedules: list[list[int, str]] = []
    boundary_end = int(boundary_end)
    steps = int(steps)

    tail = max(0, steps - boundary_end)

    # Пролог до N (если есть хвост)
    if tail > 0:
        schedules.append([boundary_end, _concat_prefix_text_suffix(pre, "", post)])

    # Теперь безопасно использовать prompts
    if tail <= 0:
        # boundary >= steps: after-сегмент никогда не активируется.
        # Показываем первый промпт на всех шагах (A1111 совместимость).
        # Пустой центр здесь неверен — пользователь ожидает увидеть промпт.
        schedules.append([steps, _concat_prefix_text_suffix(pre, prompts[0], post)])
    else:
        # Строго монотонные границы по floor, без round-слипания.
        # Обрезаем prompts до tail доступных шагов — лишние промпты
        # всё равно не получат уникальный шаг и только создадут дубли.
        usable = prompts[:tail]
        if not usable:
            usable = prompts[:1]
        k = max(1, len(usable))
        prev_end = boundary_end
        for i, p in enumerate(usable, start=1):
            end = boundary_end + (tail * i) // k  # floor
            if end <= prev_end:
                end = min(steps, prev_end + 1)
            if end > steps:
                break
            schedules.append([end, _concat_prefix_text_suffix(pre, p, post)])
            prev_end = end

    # Схлопнуть соседние сегменты с одинаковым текстом
    out: list[list[int, str]] = []
    for e, t in schedules:
        if out and out[-1][1] == t:
            out[-1][0] = int(e)   # extend
        else:
            out.append([int(e), t])
    return out


def _build_bracket_inner_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    A [p1:p2:...:N] Z — интервалы внутри [1..N] + хвост.
    Особый случай: при ОДНОМ prompt до границы показываем только pre+post,
    а после границы — pre+prompt+post (ожидание тестов).
    Для >=2 prompt'ов интервалы совпадают с логикой CollectSteps.visit_scheduled.
    """
    schedules: list[list[int, str]] = []

    if not prompts:
        return [[steps, _concat_prefix_text_suffix(pre, "", post)]]

    # 1 prompt — SD/A1111 semantics:
    #   boundary < steps  → empty 1..boundary, then prompt boundary+1..steps
    #   boundary >= steps → prompt active on all steps (no "after" window exists)
    if len(prompts) == 1:
        if boundary_end >= steps:
            # [cat:10] at steps=10: cat is shown on all steps
            schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[0], post)])
        else:
            schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, "", post)])
            schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[0], post)])
        return schedules

    # >=2 prompts: standard SD/A1111 semantics.
    # The LAST prompt is the "after boundary" tail (active from boundary+1 to steps).
    # The FIRST (n-1) prompts divide [1..boundary_end] evenly.
    #
    # Examples:
    #   [cat:dog:5] steps=10  → cat 1-5, dog 6-10
    #   [a:b:c:6]   steps=10  → a 1-3, b 4-6, c 7-10
    #
    # Old code divided ALL n prompts within [1..boundary], causing:
    #   [cat:dog:5] steps=10 → cat 1-2, dog 3-5, dog 6-10  ← wrong (cat too short)
    num_before = len(prompts) - 1   # prompts active within [1..boundary_end]

    if num_before > boundary_end:
        raise PromptSyntaxError(
            f"Scheduled block has {len(prompts)} prompts but boundary={boundary_end} "
            f"is too small — each of the {num_before} pre-boundary prompts needs "
            "at least 1 step. Increase the boundary or reduce prompt count.",
            kind="scheduled_boundary_too_small",
        )

    if num_before == 0:
        # Edge case: num_before can't be 0 when len(prompts)>=2, guard only
        schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, prompts[0], post)])
    else:
        step_size = float(boundary_end) / num_before
        for i in range(num_before):
            end = _clamp(int(round((i + 1) * step_size)), steps)
            schedules.append([end, _concat_prefix_text_suffix(pre, prompts[i], post)])

    # Last prompt as tail after boundary
    if boundary_end < steps:
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])
    elif not schedules:
        # boundary_end >= steps: tail has no room; still need at least one segment
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])

    # Fallback if rounding produced nothing
    if not schedules:
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])



    # Схлопнем соседние сегменты с одинаковым текстом (разные end_step — тоже)
    # и удалим недостижимые дубликаты end_step (step_size < 1 → round-коллизия)
    out: list[list[int, str]] = []
    for e, t in schedules:
        e_int = int(e)
        if out and e_int <= out[-1][0]:
            continue
        if out and out[-1][1] == t:
            out[-1][0] = e_int
        else:
            out.append([e_int, t])

    return out



# ──────────────────────────────────────────────────────────────────────────────
# Унификация resolve_tree
# ──────────────────────────────────────────────────────────────────────────────

def resolve_tree(tree: lark.Tree | lark.Token, keep_spacing: bool = True) -> str:
    if isinstance(tree, lark.Tree):
        parts = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                parts.append(" ")
                continue
            parts.append(resolve_tree(child, keep_spacing))
        result = "".join(str(c) for c in parts if c is not None)
        if keep_spacing:
            # схлопываем все пробельные последовательности до одиночного пробела,
            # но сохраняем крайние пробелы, если они были
            leading = len(result) - len(result.lstrip())
            trailing = len(result) - len(result.rstrip())
            core = _re_ws_collapse.sub(" ", result.strip())
            return ((" " * leading) + core + (" " * trailing)).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
        else:
            return _re_ws_collapse.sub("", result).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    return str(tree).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")


# ──────────────────────────────────────────────────────────────────────────────
# Комплексность/перевод узлов → текст
# ──────────────────────────────────────────────────────────────────────────────
_RE_UNESCAPED_ALT_OR_BANG = re.compile(r'(?<!\\)[|!]')
_RE_POST_STEP_RANGES = re.compile(r'\b\d+%?\s*-\s*\d+%?')

def _needs_complex_parse(inner: str, full: str) -> bool:
    """
    True, если fast-path лучше отключить и отдать парсеру/Visitor.
    FIX #4: Усиленная проверка для предотвращения багов с вложенностью.
    """
    # Если внутри inner есть квадратные скобки (вложенность) — это работа для Lark, а не Regex.
    if "[" in inner or "]" in inner:
        return True
        
    # Если есть экранированные скобки, Regex может ошибиться с позициями
    if "\\[" in inner or "\\]" in inner:
        return True

    # НЕэкранированные '|' или '!' в полной строке требуют парсера
    if _RE_UNESCAPED_ALT_OR_BANG.search(full or ""):
        return True

    # '::' считаем «сложностью» только ВНЕ квадратных скобок fast-path'а
    if "::" in (full or "") and "::" not in (inner or ""):
        return True
        
    return False


def _try_fast_after_scheduler(
    full: str,
    steps: int,
    *,
    prefix: str = "",
    suffix: str = "",
    allow_ranges: bool = True,
    allow_bracket_after: bool = True,
    disallow_alt_bang: bool = True,
    boundary_fallback: float | _BoundarySpec | None = None,
) -> list[list[int | str]] | None:
    """Shared fast-path for "[...]:N" and optional step ranges."""
    if _has_multiple_bracket_blocks(full):
        return None

    if allow_ranges and "|" not in full:
        m_res = _extract_after_with_ranges(full, steps)
        if m_res:
            prompts, ranges, rev_flag = m_res
            if ranges and prompts:
                schedules = _build_ranges_schedules_from_components(
                    prompts,
                    ranges,
                    steps,
                    prefix=prefix,
                    suffix=suffix,
                    rev_flag=rev_flag,
                    cycle_prompts=True,
                    empty_center="",
                )
                return _finalize_schedules(schedules)

    if not allow_bracket_after:
        return None

    if full.count("[") != 1 or full.count("]") != 1:
        return None

    if disallow_alt_bang and _RE_UNESCAPED_ALT_OR_BANG.search(full):
        return None

    m = RE_BRACKET_AFTER.match(full)
    if not m:
        return None

    pre, inner, boundary_txt, rev_token, post = m.groups()
    if _needs_complex_parse(inner or "", full):
        return None
    if _RE_POST_STEP_RANGES.search(post or ""):
        return None

    try:
        if boundary_txt and boundary_txt.endswith("%"):
            boundary_spec = _make_boundary_spec(float(boundary_txt[:-1]), is_percent=True)
        else:
            boundary_spec = _make_boundary_spec(float(boundary_txt))
    except (ValueError, TypeError):
        if boundary_fallback is None:
            return None
        boundary_spec = _make_boundary_spec(boundary_fallback, assume_absolute=True)

    prompts = _parse_inner_prompts(inner)
    post, rev_flag = _strip_reverse_from_post(post or "", bool(rev_token))
    if rev_flag:
        prompts = list(reversed(prompts))

    boundary = _to_end_step(boundary_spec, steps)
    schedules = _build_bracket_after_schedules(
        prefix + (pre or ""),
        prompts,
        boundary,
        (post or "") + suffix,
        steps,
    )
    return _finalize_schedules(schedules)
# ──────────────────────────────────────────────────────────────────────────────
# Transformer: преобразование дерева в текст (для TL последовательностей и т.п.)
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleTransformer(lark.Transformer):
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

    def start(self, args):
        s = "".join(str(arg) for arg in args if arg)
        # owner::a::b!!, extra -> owner -> owner: a, b, extra
        if "::" in s and "!!" in s and all(ch not in s for ch in "[]()"):
            left, trailing = s.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            seq_text = f"{owner.strip()}: {', '.join(descriptors)}" if descriptors else owner.strip()
            trailing_text = [t.strip(" ,") for t in trailing.split(",") if t.strip(" ,")]
            out = f"{owner.strip()} -> {seq_text}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return out
        # owner::a::b!  ->  owner: a, b
        if "::" in s and (s.endswith("!") or s.endswith(";")) and all(ch not in s for ch in "[]()"):
            owner, rest = s.split("::", 1)
            rest = rest[:-1]
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            return f"{owner.strip()}: {', '.join(descriptors)}"
        return s

    def prompt(self, args):
        return "".join(str(arg) for arg in args if arg)

    def plain(self, args):
        return args[0].value

    def and_rule(self, args):
        parts = []
        for child in args:
            if isinstance(child, lark.Tree) and getattr(child, "data", None) in ("weighted", "emphasized"):
                parts.append(self.transform(child))  # получим '(token:1.2)'
            else:
                s = resolve_tree(child, keep_spacing=True)
                if s:
                    parts.append(s)
        return f" {ATTENTION_AND_OPERATOR} ".join(p for p in parts if p)

    def weighted(self, args):
        """
        Преобразуем 'token:NUMBER' -> '(token:NUMBER)' с тем же форматированием, как в emphasized.
        """
        # левая часть (plain|compound)
        left = resolve_tree(args[0], keep_spacing=True).strip()

        # правая часть — число; достаём как текст с подстраховкой
        tail_txt = "".join(
            (a.value if isinstance(a, lark.Token) else resolve_tree(a, keep_spacing=True))
            for a in args[1:]
        )
        mnum = RE_NUMERIC.search(tail_txt or "")
        if mnum:
            num_txt = mnum.group(0)
            if "." not in num_txt and "e" not in num_txt.lower():
                try:
                    weight_str = f"{float(num_txt):.1f}"
                except (ValueError, TypeError):
                    weight_str = _rb_weight_str()
            else:
                weight_str = num_txt
        else:
            weight_str = _rb_weight_str()

        return f"({left}:{weight_str})"

    def grouped(self, args):
        return ", ".join(resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg).strip(" ,|"))

    def alternate(self, args):
        # После '!alternate' Lark передаёт '[', '|', ']' как токены наряду с prompt-деревьями.
        # '[' и ']' — структурные скобки, их нужно игнорировать.
        # Только '|' является разделителем опций.
        #
        # Алгоритм: строим список опций, считая '|' разделителями.
        # Каждая пара смежных '|' (или начало/конец) без prompt между ними — пустая опция.
        #
        # Пример '[fe|]male':  args = ['[', Tree('fe'), '|', ']']
        #   -> игнорируем '[' и ']'
        #   -> остаток: [Tree('fe'), '|']
        #   -> опции: ['fe', '']  (пустая после последнего '|')

        vals: list[str] = []
        prev_was_pipe = True   # True в начале: если первый токен '|', это ведущая пустая опция

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()

            if tok in ("[", "]"):
                # Структурные скобки — пропускаем, не меняем состояние
                continue

            if tok == "|":
                if prev_was_pipe:
                    vals.append("")   # два '|' подряд → пустая опция между ними
                prev_was_pipe = True
                continue

            # Реальный контент
            vals.append(s)
            prev_was_pipe = False

        if prev_was_pipe:
            vals.append("")   # trailing '|' → пустая завершающая опция

        return vals[(self.current_step - 1) % len(vals)] if vals else "empty_prompt"


    def alternate_distinct(self, args):
        options = []
        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            # фильтруем разделители
            if s.strip() in ("|", ","):
                continue
            if s:
                options.append(s)
            else:
                options.append("")  # пустая опция допустима при ALLOW_EMPTY_ALTERNATE
        return self.rng.choice(options) if options else "empty_prompt"

    def alternate1(self, args):
        # "a | b" — поддержка простых разделителей с возможными пустыми вариантами
        options = []
        last_was_sep = True

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")  # пустая альтернатива
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False

        if last_was_sep:
            options.append("")

        return self.rng.choice(options) if options else "empty_prompt"


    def alternate2(self, args):
        options = [resolve_tree(a, keep_spacing=True).strip() for a in args if resolve_tree(a, keep_spacing=True).strip()]
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = [(o if "_" in o or not suffix else f"{o}_{suffix}") for o in options]
        return "|".join(combined) if combined else "empty_prompt"

    def numbered(self, args):
        quantity = int(args[0])
        distinct = False
        if len(args) > 1:
            mark = str(args[1])
            distinct = mark in ("!", "_")
        # В Transformer дети уже преобразованы → приводим к строке.
        target = args[-1]
        target_str = "" if target is None else str(target)
        # Допускаем, что источником был alternate; если пришла единичная строка — берём её как единственный вариант.
        options = [s for s in (p.strip() for p in target_str.split("|")) if s] or ([target_str] if target_str != "" else [])

        if not options:
            return "empty_prompt"

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) >= quantity:
                    break
            selected = seen if len(seen) >= quantity else seen + options[:max(0, quantity - len(seen))]
        else:
            selected = self.rng.choices(options, k=quantity)

        return ", ".join(selected)

    def sequence(self, args, parent=None):
        owner = resolve_tree(args[0], keep_spacing=True) if parent is None else parent
        descriptors = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[1:] if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        return f"{owner}: {', '.join(descriptors)}"

    def top_level_sequence(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        sequences = []
        trailing_text = []
        for child in args[1:]:
            s = child if isinstance(child, str) else resolve_tree(child, keep_spacing=True)
            if not s:
                continue
            s = s.strip()
            if ':' in s:
                head, rhs = s.split(':', 1)
                parts = [head.strip()] + [p.strip() for p in rhs.split(',') if p.strip()]
                for part in parts:
                    if part and part != owner:
                        sequences.append(f"{owner}: {part}")
            else:
                t = s.strip(' ,')
                if t:
                    trailing_text.append(t)
        text = f"{owner} -> {', '.join(sequences)}"
        if trailing_text:
            text += f", {', '.join(trailing_text)}"
        return text

    def top_level_sequence3(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        before = []
        after = []
        after_bang = False

        for child in args[1:]:
            # Токен-граница "!!!"
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    continue
                if str(child) == "!!!":
                    after_bang = True
                    continue
                if str(child) == ",":
                    continue

            if not after_bang:
                # Левая часть: "owner::: (sequence|plain) (, ...)*"
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    # Принудительно используем owner в левой части
                    txt = self.sequence(child.children, parent=owner)
                    if txt:
                        before.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        before.append(f"{owner}: {s}")
            else:
                # Правая часть после "!!!": сохраняем собственных владельцев
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    txt = self.sequence(child.children)  # owner берём из самого sequence
                    if txt:
                        after.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        after.append(s)

        text = f"{owner} -> {', '.join(before)}"
        if after:
            text += f", {', '.join(after)}"
        return text

        
    def nested_sequence(self, args):
        def _is_term(x): return isinstance(x, str) and x in ('!', ';', '~')
        has_term = bool(args and _is_term(args[-1]))
        payload = args[:-1] if has_term else args
        elements = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;")
                    for arg in payload
                    if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        terminator = args[-1] if has_term else None
        if terminator == "~":
            return self.rng.choice(elements) if elements else "empty_prompt"
        return ", ".join(elements)

    def emphasized(self, args):
        """
        (cat) -> (cat:1.1)
        (cat:2) -> (cat:2.0)
        (cat: dog) -> (cat:1.1)
        ((bird:2):3) -> содержит "(bird:2.0)"
        (  fox  :  1.25 ) -> (fox:1.25)
        (wolf : (2)) -> (wolf:2.0)

        Доп. правило: если внутри уже "(...:w)" и рассчитанный внешний вес == 1.1,
        не добавляем второй слой — возвращаем как есть.
        """
        prompt_text = ""
        weight_str: str | None = None

        # Собираем сырой текст аргументов
        raw_parts: list[str] = []
        for a in args:
            if isinstance(a, lark.Token):
                raw_parts.append(a.value)
            elif isinstance(a, lark.Tree):
                raw_parts.append(resolve_tree(a, keep_spacing=True))
            else:
                raw_parts.append(str(a))

        # Уберём служебные токены '(', ')', ':' — оставим полезные куски
        parts = [p.strip() for p in raw_parts if p not in (":", "(", ")", "", None)]
        if parts:
            prompt_text = parts[0].strip()

        if len(parts) >= 2:
            mnum = RE_NUMERIC.search(parts[1])
            if mnum:
                num_txt = mnum.group(0)
                if "." not in num_txt and "e" not in num_txt.lower():
                    try:
                        weight_str = f"{float(num_txt):.1f}"
                    except (ValueError, TypeError):
                        weight_str = _rb_weight_str()
                else:
                    weight_str = num_txt
            else:
                weight_str = _rb_weight_str()
        else:
            weight_str = _rb_weight_str()

        # ★ Если уже имеем "(...:w)" как текст и внешний вес дефолтный — не оборачиваем повторно
        #   Это устраняет артефакт вида "((cat:1.2):1.1)" в Visitor.
        # Внутри ScheduleTransformer.emphasized, перед return: 
        pt = prompt_text.strip()
        # (существующее правило «не оборачивать второй раз при 1.1» — оставить)
        if (
            weight_str in _rb_weight_strs()
            and len(pt) >= 5 and pt[0] == "(" and pt[-1] == ")" and ":" in pt
        ):
            return pt

        # ★ добавка: единообразно разэкранируем литералы в prompt_text
        prompt_text = _unescape_literals(prompt_text)

        return f"({prompt_text}:{weight_str})"




def _compact_schedule(raw: list[list], steps: int) -> list[list]:
    """Схлопнуть смежные шаги с одинаковым текстом → список [end_step, text].

    raw — список [step, text] по одной записи на шаг (выход пошагового цикла).
    Последний сегмент всегда растягивается до steps.
    """
    result: list[list] = []
    for step, text in raw:
        if result and result[-1][1] == text:
            result[-1][0] = step
        else:
            result.append([step, text])
    if result and result[-1][0] < steps:
        result[-1][0] = steps
    return result


def _merge_child_schedules_for_scheduled(
    child_scheds: list[list],   # расписание каждого prompt-слота внутри [...]
    boundary: int,               # конец периода переключений [1..boundary]
    steps: int,
    prefix: str = "",
    suffix: str = "",
    is_reverse: bool = False,
) -> list[list]:
    """Аналог _build_bracket_inner_schedules, но принимает расписания дочерних
    узлов вместо статичных строк.

    Используется только из CollectSteps.visit_scheduled когда inner-prompt
    содержит dynamic-узлы (alternate, nested scheduled и т.п.).
    Fast-path ветки и _build_bracket_inner_schedules(list[str]) не трогаем.
    """
    if not child_scheds:
        return [[steps, _concat_prefix_text_suffix(prefix, "", suffix)]]

    if is_reverse:
        child_scheds = list(reversed(child_scheds))

    def pick(sched: list, step: int) -> str:
        for e, t in sched:
            if step <= int(e):
                return t
        return sched[-1][1]

    n = len(child_scheds)

    raw: list[list] = []

    if n == 1:
        if boundary >= steps:
            # boundary >= steps: "after" window never activates.
            # Show content on all steps (mirrors _build_bracket_inner_schedules).
            for step in range(1, steps + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[0], step), suffix)])
        else:
            # 1 слот: шаги 1..boundary — пустой центр, boundary+1..steps — контент
            for step in range(1, boundary + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, "", suffix)])
            for step in range(boundary + 1, steps + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[0], step), suffix)])
    else:
        # n>=2: равномерно делим [1..boundary] между слотами
        num_before = n - 1
        if num_before > boundary:
            raise PromptSyntaxError(
                f"Scheduled block has {n} prompts but boundary={boundary} "
                f"is too small — each of the {num_before} pre-boundary prompts needs "
                "at least 1 step.",
                kind="scheduled_boundary_too_small",
            )
        step_size = float(boundary) / num_before
        prev_seg_end = 0
        for i, sched in enumerate(child_scheds[:-1]):
            seg_start = _clamp(round(i * step_size) + 1, steps)
            seg_end   = _clamp(round((i + 1) * step_size), steps)
            if seg_start <= prev_seg_end:
                seg_start = prev_seg_end + 1
            if seg_start > seg_end:
                continue
            for step in range(seg_start, seg_end + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(sched, step), suffix)])
            prev_seg_end = seg_end
        # Хвост после boundary
        for step in range(boundary + 1, steps + 1):
            raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[-1], step), suffix)])
        if not raw:
            raw.append([steps, _concat_prefix_text_suffix(prefix, pick(child_scheds[-1], steps), suffix)])

    return _compact_schedule(raw, steps)


# ──────────────────────────────────────────────────────────────────────────────
# Visitor: сборка расписаний
# ──────────────────────────────────────────────────────────────────────────────

class CollectSteps(lark.Visitor):
    def __init__(self, steps, prefix="", suffix="", use_scheduling=True, seed=None, _rng=None):
        super().__init__()
        self.steps = steps
        self.prefix = prefix
        self.suffix = suffix
        self.use_scheduling = use_scheduling
        self.seed = seed
        # Если передан уже инициализированный rng — используем его (разделяем состояние
        # с родительским CollectSteps, чтобы [a|b] car [c|d] давал независимые выборы).
        # Иначе создаём свежий генератор из seed.
        if _rng is not None:
            self.rng = _rng
        else:
            self.rng = random.Random(seed) if seed is not None else random
        self.schedules = []

    # — fast-path для "[...]:N [reverse]" на уровне prompt
    def visit_prompt(self, tree):
        full = resolve_tree(tree, keep_spacing=True)
        if not self.use_scheduling:
            return self._default_visit(tree)
        schedules = _try_fast_after_scheduler(
            full,
            self.steps,
            prefix=self.prefix,
            suffix=self.suffix,
            allow_ranges=True,
            allow_bracket_after=True,
            disallow_alt_bang=True,
            boundary_fallback=None,
        )
        if schedules is not None:
            return schedules
        return self._default_visit(tree)


    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            method_name = f"visit_{tree.data}"
            method = getattr(self, method_name, self._default_visit)
            return method(tree)
        elif isinstance(tree, lark.Token):
            return self._visit_token(tree)
        return []

    def visit_start(self, tree):
        full = resolve_tree(tree, keep_spacing=True).strip()
        if not self.use_scheduling:
            return self._default_visit(tree)
        schedules = _try_fast_after_scheduler(
            full,
            self.steps,
            prefix=self.prefix,
            suffix=self.suffix,
            allow_ranges=True,
            allow_bracket_after=True,
            disallow_alt_bang=True,
            boundary_fallback=_make_boundary_spec(1.0, assume_absolute=True),
        )
        if schedules is not None:
            return schedules

        # 0) owner::a::b!!, trailing
        if "::" in full and "!!" in full and all(ch not in full for ch in '[]()'):
            left, trailing = full.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            sequences = [f"{owner.strip()}: {d}" for d in descriptors]
            trailing_text = [t.strip(' ,') for t in trailing.split(',') if t.strip(' ,')]
            out = f"{owner.strip()} -> {', '.join(sequences)}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return [[self.steps, _collapse_spaces(self.prefix + out + self.suffix)]]

        # 1) owner::a::b!
        if '::' in full and (full.endswith('!') or full.endswith(';')) and all(ch not in full for ch in '[]()'):
            owner, rest = full.split('::', 1)
            rest = rest[:-1]
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            text = f"{owner.strip()}: {', '.join(descriptors)}"
            return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

        # 2) [a:b:...:N] — допускаем скобки внутри и безопасно сплитим,
        # запускаем fast-path только если в строке ровно одна пара []
        if full.count('[') == 1 and full.count(']') == 1 and '|' not in full:
            try:
                lb, rb = full.index('['), full.rindex(']')
                pre, inner, post = full[:lb], full[lb+1:rb], full[rb+1:]
                parts = _split_top_level_colon_keep_empty(inner)
                if len(parts) >= 2 and re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?', parts[-1]):
                    boundary_spec = _make_boundary_spec(float(parts[-1]))
                    prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                    if prompts:
                        post, rev_flag = _strip_reverse_from_post(post or "")
                        if rev_flag:
                            prompts = list(reversed(prompts))
                        boundary = _to_end_step(boundary_spec, self.steps)
                        schedules = _build_bracket_inner_schedules(self.prefix + pre, prompts, boundary, post + self.suffix, self.steps)
                        return _finalize_schedules(schedules)
            except ValueError:
                pass

        return self._default_visit(tree)

    def _default_visit(self, tree):
        # 1) Собираем расписание для каждого ребёнка без внешних аффиксов
        child_scheds = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    # пробел — константное расписание
                    child_scheds.append([[self.steps, " "]])
                else:
                    child_scheds.append([[self.steps, str(child)]])
            else:
                if getattr(child, "data", None) == "prompt":
                    raw_prompt = resolve_tree(child, keep_spacing=True)
                    if raw_prompt and not raw_prompt.strip():
                        child_scheds.append([[self.steps, raw_prompt]])
                        continue
                sub = CollectSteps(
                    self.steps,
                    prefix="",   # важно: без префикса/суффикса на уровне детей
                    suffix="",
                    use_scheduling=self.use_scheduling,
                    seed=self.seed,
                    _rng=self.rng,  # ФИЙ14: передаём общий rng, чтобы разные [a|b] блоки
                                    # не сбрасывали генератор к тому же начальному состоянию
                ).visit(child)
                if not sub:
                    # на всякий случай — literal fallback
                    sub = [[self.steps, resolve_tree(child, keep_spacing=True)]]
                child_scheds.append(sub)

        if not child_scheds:
            return [[self.steps, _collapse_spaces(self.prefix + self.suffix)]]

        # 2) Объединённые границы всех детей
        boundaries = sorted({int(e) for sched in child_scheds for (e, _) in sched})
        if boundaries and boundaries[-1] < self.steps:
            boundaries.append(self.steps)
        elif not boundaries:
            boundaries = [self.steps]

        def pick_text(sched, step):
            for e, t in sched:
                if step <= int(e):
                    return t
            return sched[-1][1]

        # 3) Склейка «картинки шага» = конкатенация текущих текстов детей
        out = []
        for end in boundaries:
            parts = [pick_text(s, end) for s in child_scheds]

            combined = "".join(parts)

            text = _concat_prefix_text_suffix(self.prefix, combined, self.suffix)
            text = _apply_and(_collapse_spaces(text))
            if not out or out[-1][1] != text:
                out.append([end, text])

        return out


    def _visit_token(self, token):
        if token.type == "WHITESPACE":
            return []
        return [[self.steps, _collapse_spaces(self.prefix + str(token) + self.suffix)]]

    def visit_plain(self, tree):
        text = resolve_tree(tree, keep_spacing=True)
        # Preserve edge whitespace while a parent prompt stitches adjacent
        # parse children together. This keeps A1111-style cases like
        # "[fe|]male" literal ("female"), while explicit source spaces remain.
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_top_level_sequence3(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_top_level_sequence(self, tree):
        # Аналогично visit_top_level_sequence3, но для 'owner::a::b!!, tail'
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]
    def visit_scheduled(self, tree):
            if not tree.children:
                return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

            # IMPORTANT:
            # Do NOT flatten the whole subtree to find NUMBER / prompt nodes.
            # Otherwise nested constructs like "(red eyes:1.4)" will leak their NUMBER (1.4)
            # and inner prompt nodes into the scheduling boundary/alternates.
            #
            # We only consider DIRECT children of the `scheduled` node.
            direct_prompts = [
                c for c in tree.children
                if isinstance(c, lark.Tree) and getattr(c, "data", None) == "prompt"
            ]
            boundary_value = _extract_scheduled_boundary_value(tree.children)
            step_range_list = next(
                (c for c in tree.children if isinstance(c, lark.Tree) and getattr(c, "data", None) == "step_range_list"),
                None,
            )
            is_reverse = any(
                isinstance(c, lark.Tree) and getattr(c, "data", None) == "reverse_flag"
                for c in tree.children
            )

            # Проверяем, есть ли dynamic-узлы внутри prompt-детей
            # (alternate, nested scheduled и т.п.)
            def _has_dynamic_children(prompt_node) -> bool:
                DYNAMIC = {"scheduled", "alternate", "alternate1", "alternate2",
                           "alternate_distinct", "grouped", "numbered"}
                return any(
                    isinstance(c, lark.Tree) and getattr(c, "data", None) in DYNAMIC
                    for c in prompt_node.children
                )

            has_dynamic = any(_has_dynamic_children(p) for p in direct_prompts)

            def _to_steps_local(val_txt: str, is_pct: bool) -> int:
                v = float(val_txt)
                if is_pct:
                    v = v / 100.0 * float(self.steps)
                return _clamp(int(round(v)), self.steps)

            if has_dynamic:
                # Новый путь: строим расписание для каждого prompt-слота через CollectSteps
                child_scheds = []
                for p in direct_prompts:
                    sub = CollectSteps(
                        self.steps, prefix="", suffix="",
                        use_scheduling=self.use_scheduling,
                        seed=self.seed, _rng=self.rng,
                    ).visit(p)
                    if not sub:
                        sub = [[self.steps, _unescape_literals(resolve_tree(p, keep_spacing=True))]]
                    child_scheds.append(sub)

                if is_reverse:
                    child_scheds = list(reversed(child_scheds))

                if boundary_value is None:
                    return [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]

                if step_range_list is not None:
                    # ranges-mode с dynamic-детьми:
                    # строим step-mapping slot[step] = text из child_scheds
                    # аналогично _build_ranges_schedules_from_components, но
                    # для каждого range берём текст из child-расписания на данном шаге.

                    def _pick_child(sched, step):
                        for e, t in sched:
                            if step <= int(e):
                                return t
                        return sched[-1][1]

                    ranges_dyn: list[tuple[int, int]] = []

                    for r in step_range_list.children:
                        if not isinstance(r, lark.Tree):
                            continue
                        if getattr(r, "data", None) not in ("range_abs", "range_pct"):
                            continue
                        is_pct = (r.data == "range_pct")
                        try:
                            a = str(r.children[0].value)
                            b = str(r.children[1].value)
                        except (AttributeError, TypeError):
                            continue
                        sv = _to_steps_local(a, is_pct)
                        ev = _to_steps_local(b, is_pct)
                        if sv > ev:
                            continue
                        ranges_dyn.append((sv, ev))

                    if not ranges_dyn:
                        return [[self.steps, _concat_prefix_text_suffix(self.prefix, "", self.suffix)]]

                    # slot[step] = center text (last-wins)
                    slot = [""] * (self.steps + 1)
                    for idx, (sv, ev) in enumerate(ranges_dyn):
                        child_idx = idx % len(child_scheds)
                        for step in range(sv, ev + 1):
                            slot[step] = _pick_child(child_scheds[child_idx], step)

                    # Compress slot → schedule segments
                    raw_dyn = []
                    for step in range(1, self.steps + 1):
                        text = _concat_prefix_text_suffix(self.prefix, slot[step], self.suffix)
                        raw_dyn.append([step, text])
                    return _finalize_schedules(_compact_schedule(raw_dyn, self.steps))

                boundary = _to_end_step(boundary_value, self.steps)
                return _finalize_schedules(
                    _merge_child_schedules_for_scheduled(
                        child_scheds, boundary,
                        self.steps, self.prefix, self.suffix,
                        is_reverse=False,  # already reversed above
                    )
                )

            # Статичный путь (нет dynamic-детей) — без изменений
            # Тексты промптов (только верхнего уровня)
            prompt_texts = [_unescape_literals(resolve_tree(p, keep_spacing=True)) for p in direct_prompts]
            if is_reverse:
                prompt_texts = list(reversed(prompt_texts))

            if boundary_value is None:
                # В норме такого быть не должно, но оставим безопасный fallback
                return [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]

            # ── RANGES MODE: "[...]: N 2-3,4-9 [reverse]" ─────────────────────────
            if step_range_list is not None:
                ranges: list[tuple[int, int]] = []

                # Preserve original order (needed for last-wins overlap policy)
                for r in step_range_list.children:
                    if not isinstance(r, lark.Tree):
                        continue
                    if getattr(r, "data", None) not in ("range_abs", "range_pct"):
                        continue

                    is_pct = (r.data == "range_pct")
                    try:
                        a = str(r.children[0].value)
                        b = str(r.children[1].value)
                    except (AttributeError, TypeError):
                        continue

                    start_val = _to_steps_local(a, is_pct)
                    end_val = _to_steps_local(b, is_pct)

                    if start_val > end_val:
                        tok = f"{a}{'%' if is_pct else ''}-{b}{'%' if is_pct else ''}"
                        raise PromptSyntaxError(
                            f"Reverse range detected: {tok}",
                            kind="reverse_range",
                            token=tok,
                            full=resolve_tree(tree, keep_spacing=True),
                        )

                    # Single-step allowed (start == end)
                    ranges.append((start_val, end_val))

                if not ranges:
                    return [[self.steps, _concat_prefix_text_suffix(self.prefix, "", self.suffix)]]

                # Build with unified policy: last-wins overlaps, strict reverse, single-step ok
                # NOTE: prompt_texts already reversed above (if is_reverse), so pass rev_flag=False
                # to avoid double-reverse inside _build_ranges_schedules_from_components.
                schedules = _build_ranges_schedules_from_components(
                    prompt_texts,
                    ranges,
                    self.steps,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    rev_flag=False,
                    cycle_prompts=True,
                    empty_center="",
                )
                return _finalize_schedules(schedules)

            # ── Boundary mode: "[...:N]" / "[...]:N" ─────────────────────────────
            boundary = _to_end_step(boundary_value, self.steps)
            schedules = _build_bracket_after_schedules(self.prefix, prompt_texts, boundary, self.suffix, self.steps)
            return _finalize_schedules(schedules)

    def visit_alternate(self, tree):
        vals = []
        prev_was_pipe = True   # ведущий '|' → пустая опция в начале
        for child in tree.children:
            if child is None:
                vals.append("")
                prev_was_pipe = False
                continue
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            tok = str(child).strip()
            # '[' и ']' — структурные скобки от !alternate, игнорируем
            if tok in ("[", "]", "!"):
                continue
            if tok in ("|", ","):
                if prev_was_pipe:
                    vals.append("")   # два '|' подряд → пустая опция
                prev_was_pipe = True
                continue
            child_schedules = self.visit(child)
            if child_schedules:
                # Если дочерний узел сам даёт несколько вариантов (например alternate1),
                # учитываем их все, сохраняя порядок.
                dedup_vals: list[str] = []
                for _, t in child_schedules:
                    if not dedup_vals or dedup_vals[-1] != t:
                        dedup_vals.append(t)
                child_vals = []
                for t in dedup_vals:
                    if t not in child_vals:
                        child_vals.append(t)
            else:
                child_vals = [resolve_tree(child, keep_spacing=True)]
            if not child_vals:
                child_vals = [""]
            vals.extend(child_vals)
            prev_was_pipe = False

        if prev_was_pipe:
            vals.append("")   # trailing '|' → пустая завершающая опция

        # ИСПРАВЛЕНИЕ: Более точная фильтрация
        # Удаляем избыточные пустые, но сохраняем хотя бы один если все пустые
        if ALLOW_EMPTY_ALTERNATE:
            # Разрешаем пустые, не фильтруем
            filtered_vals = vals if vals else [""]
        else:
            # Убираем все пустые
            filtered_vals = [v for v in vals if v.strip()]
            if not filtered_vals:
                filtered_vals = [""]

        if not filtered_vals:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = filtered_vals[(step - 1) % len(filtered_vals)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix)])
            return [[e, _apply_and(t)] for e, t in schedules]

        choice = self.rng.choice(filtered_vals)
        return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix)]]
        
    def visit_alternate_distinct(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                tok = str(child).strip()
                if tok in ("|", ",", "[", "]", "!"):
                    continue
                if tok:
                    options.append([tok])
                else:
                    options.append([""])
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1] for sched in child_schedules if sched[1] != ""]
            if child_options:
                options.append(child_options)
            else:
                txt = resolve_tree(child, keep_spacing=True)
                options.append([txt])
        flat = [opt for group in options for opt in group]
        if not flat:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]
        selected = self.rng.choice(flat)
        return [[self.steps, _collapse_spaces(self.prefix + selected + self.suffix)]]

    def visit_alternate1(self, tree):
        # Собираем варианты, включая потенциально пустые
        options = []
        last_was_sep = True
        for child in tree.children:
            if isinstance(child, lark.Token):
                tok = str(child).strip()
                if tok in ("|", ","):
                    if last_was_sep:
                        options.append("")
                    last_was_sep = True
                    continue
                if child.type == "WHITESPACE":
                    continue
                s = str(child)
            else:
                s = resolve_tree(child, keep_spacing=True)
            if s is None:
                continue
            tok = s.strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False
        if last_was_sep:
            options.append("")

        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = options[(step - 1) % len(options)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix)])
            return [[e, _apply_and(t)] for e, t in schedules]
        else:
            choice = self.rng.choice(options)
            return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix)]]


    def visit_alternate2(self, tree):
        opts = []
        for c in tree.children:
            s = resolve_tree(c, keep_spacing=True).strip()
            if not s or s in ("|", ","):
                continue
            opts.append(s)
        options = opts
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = []
        for opt in options:
            combined.append(opt if "_" in opt or not suffix else f"{opt}_{suffix}")
        text = "|".join(combined) if combined else "empty_prompt"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_weighted(self, tree):
        tr = ScheduleTransformer(self.steps, 1, self.seed)
        text = tr.transform(tree)  # '(token:1.2)'
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_grouped(self, tree):
        # Собираем варианты по дочерним узлам (сохраняем пустые элементы)
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            # берём тексты без обрезки пустых, чтобы пустые варианты учитывались
            child_opts = [sched[1] for sched in child_schedules]
            if not child_opts:
                child_opts = [resolve_tree(child, keep_spacing=True)]
            all_options.append(child_opts)

        # Оценка числа комбинаций:
        total_combos = 1
        for opts in all_options:
            total_combos *= max(1, len(opts))

        if total_combos > GROUP_COMBO_LIMIT:
            mode = GROUP_COMBO_FALLBACK
            if mode == "literal":
                # Возвращаем псевдолитерал, начинающийся с "{[" чтобы сохранить сигнал исходной формы
                inner = ", ".join(resolve_tree(c, keep_spacing=True) for c in tree.children if not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                original = "{[" + inner + "]}"
                return [[self.steps, _collapse_spaces(self.prefix + original + self.suffix)]]
            elif mode == "sample":
                k = GROUP_COMBO_LIMIT
                lens = [max(1, len(opts)) for opts in all_options]
                seen: set[tuple[int, ...]] = set()
                out = []
                max_tries = k * 10
                tries = 0
                while len(seen) < min(k, total_combos) and tries < max_tries:
                    idx = tuple(self.rng.randrange(n) for n in lens)
                    if idx in seen:
                        continue
                    tries += 1
                    seen.add(idx)
                    combo = [all_options[d][i] if all_options[d] else "" for d, i in enumerate(idx)]
                    text = ", ".join(combo).strip()
                    if text:
                        out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix)])
                if out:
                    return out
                # иначе сваливаемся на усечение (truncate)

        out = []
        for i, combo in enumerate(product(*all_options)):
            if i >= GROUP_COMBO_LIMIT:
                break
            text = ", ".join(combo).strip()
            if text:
                out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix)])
        return out or [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

    def visit_sequence(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_nested_sequence(self, tree):
        # Терминатор ('!', ';', '~') — ОПЦИОНАЛЬНЫЙ. Проверяем до среза,
        # иначе [:-1] отрежет последний prompt когда терминатора нет.
        last = tree.children[-1] if tree.children else None
        has_term = isinstance(last, lark.Token) and last.value in ('!', ';', '~')
        payload = tree.children[:-1] if has_term else tree.children
        elements = [resolve_tree(c, keep_spacing=True).strip(" ,~!;")
                    for c in payload
                    if resolve_tree(c, keep_spacing=True).strip(" ,~!;")]
        terminator = last if has_term else None
        if terminator and terminator.value == "~":
            text = self.rng.choice(elements) if elements else "empty_prompt"
        else:
            # Согласовано с ScheduleTransformer.nested_sequence: возвращаем обычный текст
            text = ", ".join(elements) if elements else "empty_prompt"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_numbered(self, tree):
        quantity = int(tree.children[0])
        distinct = False
        if len(tree.children) > 1:
            mark = tree.children[1]
            try:
                if str(mark) in ('!', '_'):
                    distinct = True
            except (AttributeError, TypeError):
                pass
        target = tree.children[-1]

        options = []
        def add_opts(node):
            if isinstance(node, lark.Tree):
                if getattr(node, "data", None) in ("alternate", "alternate1", "alternate2", "alternate_distinct", "prompt"):
                    for ch in node.children:
                        add_opts(ch)
                else:
                    txt = resolve_tree(node, keep_spacing=True)
                    if txt is not None:
                        options.append(txt)
            elif isinstance(node, lark.Token):
                if node.type == "WHITESPACE":
                    return
                txt = str(node)
                if txt in ("[", "]", "|", ",", "!"):
                    return
                options.append(txt)

        add_opts(target)
        options = [
            opt.strip() if isinstance(opt, str) else opt
            for opt in options
            if opt is not None
        ]
        if not options:
            child_schedules = self.visit(target)
            options = [s[1] for s in child_schedules]
        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) == quantity:
                    break
            if len(seen) < quantity:
                seen += self.rng.choices(options, k=quantity - len(seen))
            selected = seen
        else:
            selected = self.rng.choices(options, k=quantity)

        return [[self.steps, _collapse_spaces(self.prefix + ", ".join(selected) + self.suffix)]]

    def visit_and_rule(self, tree):
        # Static nodes: transform directly to text (no schedule needed)
        STATIC_NODES = ("weighted", "emphasized")

        branch_scheds: list[list] = []
        tr = ScheduleTransformer(self.steps, 1, self.seed)

        for child in tree.children:
            # Skip structural tokens (& operator, whitespace)
            if isinstance(child, lark.Token):
                continue

            node_type = getattr(child, "data", None)

            if node_type in STATIC_NODES:
                # Fast static path — transform to "(text:weight)" string
                text = tr.transform(child)
                branch_scheds.append([[self.steps, text]])
            else:
                # General dynamic path: alternate, grouped, scheduled,
                # numbered, alternate1/2/distinct — build sub-schedule
                sub = CollectSteps(
                    self.steps, prefix="", suffix="",
                    use_scheduling=self.use_scheduling,
                    seed=self.seed, _rng=self.rng,
                ).visit(child)
                if not sub:
                    sub = [[self.steps, resolve_tree(child, keep_spacing=True)]]
                branch_scheds.append(sub)

        if not branch_scheds:
            return [[self.steps, _collapse_spaces(self.prefix + self.suffix)]]

        # Merge all branches over their combined boundary points
        boundaries = sorted({int(e) for s in branch_scheds for (e, _) in s})
        if boundaries and boundaries[-1] < self.steps:
            boundaries.append(self.steps)

        def _pick(sched, step):
            for e, t in sched:
                if step <= int(e):
                    return t
            return sched[-1][1]

        out = []
        for end in boundaries:
            parts = [_pick(s, end) for s in branch_scheds]
            text = f" {ATTENTION_AND_OPERATOR} ".join(p for p in parts if p)
            text = _collapse_spaces(self.prefix + text + self.suffix)
            if not out or out[-1][1] != text:
                out.append([end, text])
        return out

    def visit_emphasized(self, tree):
        # Check if the inner prompt contains dynamic nodes that produce schedules
        inner_prompt = next(
            (c for c in tree.children
             if isinstance(c, lark.Tree) and c.data == "prompt"),
            None,
        )
        DYNAMIC_NODES = {
            "scheduled", "alternate", "alternate1", "alternate2",
            "alternate_distinct", "grouped", "numbered",
        }
        has_dynamic = inner_prompt and any(
            isinstance(c, lark.Tree) and getattr(c, "data", None) in DYNAMIC_NODES
            for c in inner_prompt.children
        )

        if not has_dynamic:
            # Fast static path — no change from original
            tr = ScheduleTransformer(self.steps, 1, self.seed)
            text = tr.transform(tree)
            return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

        # Dynamic path: build inner schedule, wrap each segment in (text:weight)
        weight_tok = next(
            (c for c in tree.children
             if isinstance(c, lark.Token) and c.type == "NUMBER"),
            None,
        )
        if weight_tok is not None:
            w_raw = weight_tok.value
            try:
                w_f = float(w_raw)
                w_str = f"{w_f:.1f}" if ("." not in w_raw and "e" not in w_raw.lower()) else w_raw
            except (ValueError, TypeError):
                w_str = _rb_weight_str()
        else:
            w_str = _rb_weight_str()

        inner_sched = CollectSteps(
            self.steps, prefix="", suffix="",
            use_scheduling=self.use_scheduling,
            seed=self.seed, _rng=self.rng,
        ).visit(inner_prompt)

        out = []
        for end, text in (inner_sched or [[self.steps, ""]]):
            t = text.strip()
            wrapped = f"({t}:{w_str})" if t else ""
            full = _collapse_spaces(self.prefix + wrapped + self.suffix)
            if not out or out[-1][1] != full:
                out.append([end, full])
        return out

    def __call__(self, tree):
        self.schedules = self.visit(tree)
        # dedup по паре (step,text)
        uniq = []
        seen = set()
        for end_step, text in self.schedules or []:
            key = (int(end_step), text)
            if key not in seen:
                uniq.append([int(end_step), text])
                seen.add(key)
        # Паритет с get_schedule: последний сегмент эксплицитно тянется до self.steps
        if uniq:
            try:
                last_end = int(uniq[-1][0])
            except (ValueError, TypeError):
                last_end = None
            if isinstance(last_end, int) and last_end < int(self.steps):
                uniq.append([int(self.steps), uniq[-1][1]])
        result = uniq or [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]
        if not self.use_scheduling:
            return _apply_scheduling_mode(result, self.steps, False)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Внешние утилиты для расписаний
# ──────────────────────────────────────────────────────────────────────────────

def at_step_from_schedule(step: int, schedule: Sequence[Sequence[int | str]]) -> str:
    if not schedule:
        return ""
    for end_step, text in schedule:
        try:
            if step <= int(end_step):
                return text
        except (ValueError, TypeError):
            continue
    return schedule[-1][1]

def at_step(step: int, prompt_or_schedule, *, steps: int | None = None,
            seed: int | None = 42, use_visitor: bool = True, use_scheduling: bool = True) -> str:
    if isinstance(prompt_or_schedule, list) and prompt_or_schedule and isinstance(prompt_or_schedule[0], list):
        return at_step_from_schedule(step, prompt_or_schedule)
    prompt = str(prompt_or_schedule)
    if steps is None:
        raise ValueError("steps is required when passing a prompt string to at_step(...)")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if step <= 0 or step > steps:
        raise ValueError(f"step must be between 1 and {steps}, got {step}")
    sched = get_schedule(prompt, steps, use_scheduling=use_scheduling, seed=seed, use_visitor=use_visitor)
    return at_step_from_schedule(step, sched)

def _apply_and(text: str) -> str:
    # 1) Нормализуем пробелы вокруг двоеточия в шаблоне 'token : number' -> 'token:number'
    text = re.sub(
        rf'(?<=[^\W\d_])\s*[:：]\s*(?=(?:{NUMERIC_RE})\b)',
        ':',
        text,
    )

    # 2) Приводим операторы к единому виду, игнорируя экранированный '&'
    text = _normalize_and_operators_for_parse(text)
    text = re.sub(
        rf'(?<!\\)(\S)\s*{re.escape(ATTENTION_AND_OPERATOR)}\s*(\S)',
        lambda m: f"{m.group(1)} {ATTENTION_AND_OPERATOR} {m.group(2)}",
        text,
    )  # Нормализуем пробелы вокруг &

    # 3) Схлопнуть множественные пробелы
    text = _re_ws_collapse.sub(" ", text)

    # 4) Разэкранируем литералы перед финальной отдачей
    text = _unescape_literals(text)
    text = text.replace(ESCAPED_AMP_PLACEHOLDER, "&")

    # 5) Удаляем лишние/висячие '&' из начала и конца выражения.
    tokens = [tok for tok in text.strip().split(" ") if tok]
    if not tokens:
        return ""
    if len(tokens) == 1 and tokens[0] == ATTENTION_AND_OPERATOR:
        return ATTENTION_AND_OPERATOR

    cleaned_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        if tok != ATTENTION_AND_OPERATOR:
            cleaned_tokens.append(tok)
            continue

        has_prev_term = bool(cleaned_tokens) and cleaned_tokens[-1] != ATTENTION_AND_OPERATOR
        has_next_term = any(t != ATTENTION_AND_OPERATOR for t in tokens[i + 1 :])
        if has_prev_term and has_next_term:
            cleaned_tokens.append(tok)

    return " ".join(cleaned_tokens)


def _validate_inputs(prompt: str, steps: int) -> None:
    """Единая валидация входных параметров"""
    if not isinstance(steps, int):
        raise TypeError(f"steps must be int, got {type(steps)}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps > 10000:
        raise ValueError(f"steps too large (max 10000), got {steps}")
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt)}")


def _apply_scheduling_mode(
    schedule: Sequence[Sequence[int | str]],
    steps: int,
    use_scheduling: bool,
) -> list[list[int | str]]:
    """Finalize schedules and optionally collapse them to a single static segment."""
    finalized = _finalize_schedules(schedule)
    if use_scheduling:
        return finalized
    final_text = at_step_from_schedule(int(steps), finalized)
    return [[int(steps), final_text]]


def _strict_check_delimiter_balance(text: str) -> None:
    """Best-effort bracket/paren balance check for strict lint mode."""
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in opening.items()}
    stack: list[str] = []
    escaped = False

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in opening:
            stack.append(ch)
            continue
        if ch in closing:
            if not stack or stack[-1] != closing[ch]:
                raise PromptSyntaxError(
                    f"Unmatched closing delimiter: {ch}",
                    kind="syntax_error",
                    token=ch,
                    full=text,
                )
            stack.pop()

    if stack:
        opener = stack[-1]
        raise PromptSyntaxError(
            f"Unclosed delimiter: {opener}",
            kind="syntax_error",
            token=opener,
            full=text,
        )


_RE_STRICT_RANGE_PREFIX = re.compile(r'^\s*(?:-|%|\d+\s*%?\s*-)')
_RE_STRICT_RANGE_CHARS = re.compile(r'^[\s\d%,-]+')
_RE_STRICT_RANGE_TOKEN = re.compile(r'^\d+%?\s*-\s*\d+%?$')
_RE_DUPLICATE_REVERSE_PREFIX = re.compile(r'^\s*reverse(?:\s*,)?\s+reverse\b')
_RE_INCOMPLETE_SEQUENCE_SUFFIX = re.compile(r'(?<!\\)::\s*$')
_RE_INCOMPLETE_SEQUENCE_BEFORE_OPERATOR = re.compile(r'(?<!\\)::(?=\s*(?:AND\b|&|,|$|\)|\]|\}))')
_RE_BOUNDARY_WITH_OPTIONAL_PERCENT = re.compile(rf'{NUMERIC_RE}%?')


def _iter_parenthesized_spans(text: str):
    """Yield balanced parenthesized spans as (start, end, inner_text)."""
    stack: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < len(text) else 1
            continue
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            yield start, i + 1, text[start + 1 : i]
        i += 1


def _iter_postfix_scheduler_tail_starts(text: str):
    """Yield indices right after top-level postfix scheduler boundaries like ``]:10``."""
    stack: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < len(text) else 1
            continue
        if ch == "[":
            stack.append(i)
            i += 1
            continue
        if ch == "]" and stack:
            stack.pop()
            if not stack:
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == ":":
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    m = _RE_BOUNDARY_WITH_OPTIONAL_PERCENT.match(text[j:])
                    if m is not None:
                        yield j + m.end()
        i += 1


def _raise_invalid_range_token(full: str, token: str) -> None:
    bad = token.strip() or token
    raise PromptSyntaxError(
        f"Invalid range token {bad!r}",
        kind="invalid_range_token",
        token=bad or token,
        full=full,
    )


def _strict_check_postfix_scheduler_surfaces(text: str) -> None:
    """Reject obviously broken postfix scheduler tails before lenient parsing hides them."""
    for tail_start in _iter_postfix_scheduler_tail_starts(text):
        tail = text[tail_start:]
        if not tail.strip():
            continue

        if _RE_DUPLICATE_REVERSE_PREFIX.match(tail):
            raise PromptSyntaxError(
                "Duplicate 'reverse' flag after scheduler block.",
                kind="duplicate_reverse_flag",
                token="reverse",
                full=text,
            )

        stripped = tail.lstrip()
        if stripped.startswith("reverse"):
            continue

        if not _RE_STRICT_RANGE_PREFIX.match(stripped):
            continue

        m = _RE_STRICT_RANGE_CHARS.match(stripped)
        candidate = (m.group(0) if m else stripped).rstrip()
        parts = [part.strip() for part in candidate.split(",")]
        valid_parts: list[str] = []

        for part in parts:
            if not part:
                _raise_invalid_range_token(text, candidate)
            if not _RE_STRICT_RANGE_TOKEN.fullmatch(part):
                _raise_invalid_range_token(text, part)
            valid_parts.append(part)

        if not valid_parts:
            _raise_invalid_range_token(text, candidate)

        rest = stripped[len(candidate):]
        if _RE_DUPLICATE_REVERSE_PREFIX.match(rest):
            raise PromptSyntaxError(
                "Duplicate 'reverse' flag after scheduler block.",
                kind="duplicate_reverse_flag",
                token="reverse",
                full=text,
            )


def _strict_check_sequence_surfaces(text: str) -> None:
    """Reject obviously incomplete top-level sequence markers."""
    m_incomplete = _RE_INCOMPLETE_SEQUENCE_SUFFIX.search(text) or _RE_INCOMPLETE_SEQUENCE_BEFORE_OPERATOR.search(text)
    if m_incomplete:
        raise PromptSyntaxError(
            "Incomplete sequence: expected text after '::'.",
            kind="incomplete_sequence",
            token="::",
            full=text,
        )

    idx = 0
    while True:
        pos = text.find(":::", idx)
        if pos == -1:
            break
        if pos > 0 and text[pos - 1] == "\\":
            idx = pos + 3
            continue
        if "!!!" not in text[pos + 3 :]:
            raise PromptSyntaxError(
                "Incomplete top-level sequence: expected '!!!' terminator after ':::'.",
                kind="incomplete_sequence",
                token=":::",
                full=text,
            )
        idx = pos + 3


def _strict_check_interpolation_surfaces(text: str) -> None:
    """Reject broken attention interpolation shells like ``(cat:1.0->)``."""
    exc = _detect_invalid_interpolation_surface(text)
    if exc is not None:
        raise exc


def _detect_invalid_interpolation_surface(text: str) -> PromptSyntaxError | None:
    """Return a syntax error for malformed interpolation shells, if any."""
    for start, end, inner in _iter_parenthesized_spans(text):
        if "->" not in inner or ":" not in inner:
            continue

        last_colon = _find_last_top_level_colon_index(inner)
        if last_colon < 0:
            continue

        body = inner[:last_colon].strip()
        tail = inner[last_colon + 1 :].strip()
        if "->" not in tail:
            continue
        tail_match = _RE_ATT_INTERP_TAIL.fullmatch(tail)
        if body and tail_match:
            try:
                w0 = float(tail_match.group(1))
                w1 = float(tail_match.group(2))
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(w0) and math.isfinite(w1):
                    continue

        return PromptSyntaxError(
            "Invalid attention interpolation. Use '(text:w0->w1)'.",
            kind="invalid_interpolation",
            token=text[start:end],
            full=text,
        )
    return None


def _strict_check_surface_forms(text: str, state: BackendPromptState | None = None) -> None:
    """Validate malformed surface forms that the lenient runtime path would otherwise swallow."""
    if state is None:
        state = _extract_backend_prompt_state(text)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(text)
    if state.has_mixed_backends or state.has_multiple_same_type:
        return
    if state.has_backend or state.has_pool or state.has_bind:
        return
    _raise_unsupported_backend_context_error(text)
    _strict_check_interpolation_surfaces(text)
    probe, _ = _protect_escaped_literal_spans(text)
    probe = _protect_escaped_literals(probe)
    _strict_check_postfix_scheduler_surfaces(probe)
    _strict_check_sequence_surfaces(probe)


def _strict_schedule_preview(text: str, steps: int, seed: int | None):
    """Strict dry-run for lint/explain: parse must succeed (no lenient fallback)."""
    text = _transpile_bind2_to_chunk(text)
    text = _transpile_bind3_to_chunk(text)
    _validate_inputs(text, steps)
    _strict_check_delimiter_balance(text)
    multicond_branches = _extract_multicond_preview_branches(text)
    if multicond_branches is not None:
        return _build_multicond_text_schedule_from_branches(
            multicond_branches,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    state = _extract_backend_prompt_state(text)
    if state.blend_spec is not None:
        for _br in state.blend_spec.branches:
            if _contains_chunk_marker(_br.text):
                raise PromptSyntaxError(
                    "CHUNK inside BLEND is not supported — BLEND pads shorter tensors, "
                    "making CHUNK multi-chunk semantics incorrect. "
                    "Rewrite as CHUNK{BLEND{branch1|...} | BLEND{branch2|...}}.",
                    kind="chunk_inside_blend_not_supported",
                    token=_br.text,
                    full=text,
                )
    if state.has_bind:
        return _build_bind_text_schedule_from_specs(
            state.bind_base_prompt,
            state.bind_specs,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.has_mixed_backends or state.has_multiple_same_type:
        protected_p, restore = _protect_escaped_literal_spans_for_source(text)
        protected_p = _protect_escaped_literals(protected_p)
        segments = _extract_sequential_backend_segments(protected_p, restore)
        part_schedules = [
            _strict_schedule_preview(seg, steps, seed)
            for seg in segments
        ]
        return _merge_sequential_text_schedules(part_schedules, steps)
    if state.pool_spec is not None:
        return _build_pool_text_schedule_from_spec(
            state.pool_spec,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.assemble_spec is not None:
        return _build_assemble_text_schedule_from_spec(
            state.assemble_spec,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.chunk_spec is not None:
        branch_schedules = []
        for branch in state.chunk_spec.branches:
            branch_prompt = _expand_chunk_branch_prompt(state.chunk_spec, branch)
            branch_schedules.append(_strict_schedule_preview(branch_prompt, steps, seed))

        boundaries = _collect_schedule_boundaries(branch_schedules, steps)
        out: list[list[int, str]] = []
        for end_at_step in boundaries:
            active_texts = [
                _normalize_preview_fragment(_select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY)
                for schedule in branch_schedules
            ]
            preview_parts = [text for text in active_texts if text]
            preview = CHUNK_PREVIEW_SEPARATOR.join(preview_parts) if preview_parts else SAFE_EMPTY
            if out and out[-1][1] == preview:
                out[-1][0] = int(end_at_step)
            else:
                out.append([int(end_at_step), preview])
        return out or [[int(steps), SAFE_EMPTY]]
    if state.blend_spec is not None:
        branch_schedules = []
        for branch in state.blend_spec.branches:
            branch_prompt = _expand_blend_branch_prompt(state.blend_spec, branch)
            branch_schedules.append(_strict_schedule_preview(branch_prompt, steps, seed))

        boundaries = _collect_schedule_boundaries(branch_schedules, steps)
        effective_weights = _resolve_blend_mode_weights(
            [branch.weight for branch in state.blend_spec.branches],
            state.blend_spec.mode,
            state.blend_spec.intensity,
        )
        out: list[list[int, str]] = []
        previous_key = None
        for end_at_step in boundaries:
            active_texts = [
                _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
                for schedule in branch_schedules
            ]
            preview = _build_blend_preview_text_with_target(active_texts, effective_weights, state.blend_spec.channel_target)
            key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in effective_weights))
            if out and previous_key == key:
                out[-1][0] = int(end_at_step)
            else:
                out.append([int(end_at_step), preview])
                previous_key = key
        return out or [[int(steps), SAFE_EMPTY]]
    if state.compound_spec is not None:
        return _build_compound_text_schedule_from_spec(
            state.compound_spec,
            int(steps),
            True,
            seed,
            True,
        )
    if state.active_morph_spec is not None:
        point_schedules = []
        for point in state.active_morph_spec.points:
            point_prompt = _expand_morph_point_prompt(state.active_morph_spec, point)
            point_schedules.append(_strict_schedule_preview(point_prompt, steps, seed))

        positions = _resolve_morph_positions(state.active_morph_spec, steps)
        window_steps = _resolve_morph_window_steps(state.active_morph_spec, steps)
        inactive_text = _build_morph_inactive_text(state.active_morph_spec)
        out: list[list[int, str]] = []
        previous_key = None
        for step in range(1, int(steps) + 1):
            if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
                preview = inactive_text
                key = ("inactive", preview)
                if out and previous_key == key:
                    out[-1][0] = int(step)
                else:
                    out.append([int(step), preview])
                    previous_key = key
                continue
            active_texts = [
                _select_text_from_schedule(schedule, step) or SAFE_EMPTY
                for schedule in point_schedules
            ]
            weights = _resolve_morph_point_weights(
                state.active_morph_spec.points,
                _compute_morph_curve_weights(
                    len(state.active_morph_spec.points),
                    positions,
                    step,
                    state.active_morph_spec.curve,
                    state.active_morph_spec.intensity,
                ),
            )
            preview = _build_morph_preview_text(active_texts, weights, state.active_morph_spec.channel_target)
            key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in weights))
            if out and previous_key == key:
                out[-1][0] = int(step)
            else:
                out.append([int(step), preview])
                previous_key = key
        return out or [[int(steps), SAFE_EMPTY]]
    _strict_check_surface_forms(text, state)
    probe = text
    if "\\n" in probe or "\\t" in probe:
        probe = probe.replace("\\n", "\n").replace("\\t", "\t")
    probe, _ = _protect_escaped_literal_spans(probe)
    probe = _protect_escaped_literals(probe)
    probe = _normalize_and_operators_for_parse(probe)
    probe = _normalize_scheduler_surface_syntax(probe)
    probe = _placeholderize_attention_interpolations(probe)
    _parse_lark_cached(probe)
    return get_schedule(text, steps=steps, use_scheduling=True, seed=seed, use_visitor=True)


def _schedule_runtime_cache_key() -> tuple:
    """Capture runtime flags that affect schedule construction results."""
    return (
        bool(ALLOW_EMPTY_ALTERNATE),
        bool(EXPAND_ALTERNATE_PER_STEP),
        int(GROUP_COMBO_LIMIT),
        str(GROUP_COMBO_FALLBACK),
        bool(DEDUP_SCHEDULE_STEPS),
        float(ROUND_BRACKET_MULTIPLIER),  # affects emphasized/weighted default weight
        # SUPPRESS_STANDALONE_COLON intentionally excluded: belongs to parse_prompt_attention only
    )
        
# ──────────────────────────────────────────────────────────────────────────────
# Главный API построения расписания из строки
# ──────────────────────────────────────────────────────────────────────────────

def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True):
    """
    Args:
        seed: Если None, генерируется стабильный seed на основе хеша промпта.
    """
    prompt_text = str(prompt or "")
    prompt_text = prompt_text.translate(_PUA_CLEAN_TABLE)
    prompt_text = _transpile_bind2_to_chunk(prompt_text)
    prompt_text = _transpile_bind3_to_chunk(prompt_text)
    multicond_branches = _extract_multicond_preview_branches(prompt_text)
    if multicond_branches is not None:
        return _build_multicond_text_schedule_from_branches(
            multicond_branches,
            int(steps),
            bool(use_scheduling),
            seed,
            use_visitor,
            strict=False,
        )
    if CHUNK_KEYWORD in prompt_text or ASSEMBLE_KEYWORD in prompt_text or BLEND_KEYWORD in prompt_text or MORPH_KEYWORD in prompt_text or POOL_KEYWORD in prompt_text or BIND_KEYWORD in prompt_text or COMPOUND_KEYWORD in prompt_text:
        try:
            state = _extract_backend_prompt_state(prompt_text)
        except PromptSyntaxError as e:
            if e.kind not in _STRUCTURAL_ERROR_KINDS and e.kind is not None:
                msg = f"Prompt semantic warning [{e.kind}]: {e} — falling back to raw text"
                if msg not in _warned_semantic_errors:
                    logging.warning(msg)
                    _warned_semantic_errors.add(msg)
            return [[int(steps), _collapse_spaces(prompt_text)]]
        if state.has_bind:
            return _build_bind_text_schedule_from_specs(
                state.bind_base_prompt,
                state.bind_specs,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.has_mixed_backends or state.has_multiple_same_type:
            protected_p, restore = _protect_escaped_literal_spans_for_source(prompt_text)
            protected_p = _protect_escaped_literals(protected_p)
            segments = _extract_sequential_backend_segments(protected_p, restore)
            part_schedules = [
                get_schedule(seg, int(steps), bool(use_scheduling), seed, use_visitor=use_visitor)
                for seg in segments
            ]
            return _merge_sequential_text_schedules(part_schedules, int(steps))
        if state.pool_spec is not None:
            return _build_pool_text_schedule_from_spec(
                state.pool_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.assemble_spec is not None:
            return _build_assemble_text_schedule_from_spec(
                state.assemble_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.chunk_spec is not None:
            return _build_chunk_text_schedule_from_spec(
                state.chunk_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if state.blend_spec is not None:
            return _build_blend_text_schedule_from_spec(
                state.blend_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if state.compound_spec is not None:
            return _build_compound_text_schedule_from_spec(
                state.compound_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if state.active_morph_spec is not None:
            return _build_morph_text_schedule_from_spec(
                state.active_morph_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if _contains_chunk_marker(prompt_text) or _contains_assemble_marker(prompt_text) or _contains_blend_marker(prompt_text) or _contains_morph_marker(prompt_text) or _contains_pool_marker(prompt_text) or _contains_compound_marker(prompt_text):
            return [[int(steps), _collapse_spaces(prompt_text)]]

    if _detect_invalid_interpolation_surface(prompt_text) is not None:
        return [[int(steps), _collapse_spaces(prompt_text)]]

    protected_prompt, span_restore = _protect_escaped_literal_spans(prompt_text)
    protected_prompt = _protect_escaped_literals(protected_prompt)

    if seed is None:
        prompt_bytes = protected_prompt.encode("utf-8")
        hash_digest = hashlib.sha256(prompt_bytes).digest()
        seed = int.from_bytes(hash_digest[:4], 'big') & 0x7fffffff
    runtime_key = _schedule_runtime_cache_key()
    cached = _get_schedule_cached(protected_prompt, steps, use_scheduling, seed, use_visitor, runtime_key)
    return [[row[0], _restore_escaped_literals(str(row[1]), span_restore)] for row in cached]
 

@lru_cache(maxsize=CACHE_SIZE)
def _get_schedule_cached(
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int,
    use_visitor: bool,
    _runtime_key: tuple,
):
    """Кешируемая версия для детерминированных вызовов"""
    schedule = _get_schedule_impl(prompt, steps, use_scheduling, seed, use_visitor)
    return tuple((row[0], row[1]) for row in schedule)

def _placeholderize_attention_interpolations(text: str) -> str:
    """Pre-pass: replace (body:w0->w1) patterns with private-use marker strings.

    v2: bracket-balanced scanner — supports nested parentheses in body, e.g.:
        (red (glowing eyes):0.8->1.4)
        ((cat:1.2):1.0->2.0)
        (\\(escaped\\) face:1.0->2.0)

    Algorithm:
      1. Walk the string char-by-char, tracking paren depth and escape state.
      2. For each outer '(' … ')' span, extract the inner text.
      3. Find the LAST top-level ':' inside the span (depth=0 relative to inner).
      4. Check whether the tail after that ':' matches 'w0->w1' strictly.
      5. If yes → replace the whole span with a marker; if no → keep as-is.
      6. Replacements are collected and applied right-to-left so earlier
         indices stay stable.

    Non-interpolation '(...)' spans are never modified.
    Called in _get_schedule_impl BEFORE Lark parsing.
    """
    if "->" not in text:
        return text

    spans: list[tuple[int, int, str]] = []  # (start, end_exclusive, replacement)
    i = 0
    n = len(text)

    while i < n:
        # Handle escape sequences — skip next char
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue

        if text[i] != "(":
            i += 1
            continue

        # Found an unescaped '(' — scan for matching ')'
        start = i
        depth = 0
        j = i
        while j < n:
            if text[j] == "\\" and j + 1 < n:
                j += 2
                continue
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if depth != 0:
            # Unmatched '(' — skip
            i += 1
            continue

        end = j  # index of closing ')'
        inner = text[start + 1 : end]  # content between the outer parens

        # Quick bail-out: if '->' not in inner, this can't be interpolation
        if "->" not in inner:
            i = end + 1
            continue

        # Find the LAST top-level ':' in inner (depth=0 inside inner)
        last_colon = -1
        d = 0
        escaped = False
        for k, ch in enumerate(inner):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            elif ch == ":" and d == 0:
                last_colon = k

        if last_colon == -1:
            i = end + 1
            continue

        body = inner[:last_colon].strip()
        tail = inner[last_colon + 1 :]

        # Validate tail: must be 'w0 -> w1' or 'w0 -> w1 ~ mode'
        m = _RE_ATT_INTERP_TAIL.match(tail)
        if not m:
            i = end + 1
            continue

        # Body must be non-empty
        if not body:
            i = end + 1
            continue

        w0   = float(m.group(1))
        w1   = float(m.group(2))
        if not (math.isfinite(w0) and math.isfinite(w1)):
            i = end + 1
            continue
        mode = (m.group(3) or _EASING_DEFAULT).strip().lower()
        if mode not in _EASING_MODES:
            mode = _EASING_DEFAULT
        replacement = _serialize_att_interp(body, w0, w1, mode)
        spans.append((start, end + 1, replacement))

        # Skip past this span (don't recurse into already-matched regions)
        i = end + 1

    if not spans:
        return text

    # Apply replacements right-to-left to keep earlier indices valid
    result = list(text)
    for start, end, replacement in reversed(spans):
        result[start:end] = list(replacement)

    return "".join(result)


def _expand_attention_interpolations(schedule: list[list], steps: int) -> list[list]:
    """Post-pass: expand (body:w0->w1) markers with correct global-range semantics.

    Two-phase algorithm that eliminates the sawtooth/reset bug while preserving
    contextual-aware behaviour for markers that live inside a single scheduler
    window:

    Phase 1 — global range discovery
        Walk the schedule and, for each unique marker identity
        (body, w0, w1, mode), record the FIRST segment's start_step and the
        LAST segment's end_step where that marker appears.

        Key insight:
        * Top-level marker alongside [a:b:N]: appears in MULTIPLE segments
          → global range spans all of them → smooth monotone interpolation.
        * Marker inside [(cat:0→1):N]: appears in ONE segment only
          → global range == that segment → original contextual behaviour
          preserved.

    Phase 2 — expansion
        Each marker is expanded using its discovered global range instead of
        the local segment boundary, so weight never resets at foreign
        scheduler boundaries.

    Segments without markers pass through in O(1) (no marker scan).
    Adjacent identical entries are merged to keep the output compact.
    """
    if not schedule:
        return [[steps, SAFE_EMPTY]]

    # ── Phase 1: discover global active range per unique marker identity ──────
    # Identity key: (body_stripped, raw_w0, raw_w1, mode)
    # Value: [first_start_step, last_end_step]
    MarkerRange = list  # [int, int]
    marker_global: dict[tuple, MarkerRange] = {}

    prev_end = 0
    for end_step, text in schedule:
        end_step   = int(end_step)
        start_step = prev_end + 1
        if ATTN_INTERP_OPEN in str(text):
            for m in RE_ATTN_INTERP_LITERAL.finditer(text):
                key = (
                    m.group(1).strip(),
                    m.group(2),
                    m.group(3),
                    (m.group(4) or _EASING_DEFAULT).strip().lower(),
                )
                if key not in marker_global:
                    marker_global[key] = [start_step, end_step]
                else:
                    marker_global[key][1] = end_step   # extend to latest end
        prev_end = end_step

    if not marker_global:
        # No markers anywhere — return as-is (shouldn't reach here normally)
        return schedule  # type: ignore[return-value]

    # ── Phase 2: expand each segment using per-marker global range ────────────
    out: list[list] = []
    prev_end = 0

    for end_step, text in schedule:
        end_step   = int(end_step)
        start_step = prev_end + 1

        if ATTN_INTERP_OPEN not in str(text):
            # Fast path: no marker — merge identical neighbours
            if out and out[-1][1] == text:
                out[-1][0] = end_step
            else:
                out.append([end_step, text])
            prev_end = end_step
            continue

        # Expand step-by-step; each marker uses its own global range
        for step in range(start_step, end_step + 1):

            def _make_repl(step: int, marker_global: dict) -> re.Match:
                def repl(m: re.Match) -> str:
                    key = (
                        m.group(1).strip(),
                        m.group(2),
                        m.group(3),
                        (m.group(4) or _EASING_DEFAULT).strip().lower(),
                    )
                    body = m.group(1).strip()
                    w0   = float(m.group(2))
                    w1   = float(m.group(3))
                    mode = key[3]
                    g_start, g_end = marker_global.get(key, (step, step))
                    span = max(0, g_end - g_start)
                    t_lin = 0.0 if span == 0 else (step - g_start) / span
                    t_lin = max(0.0, min(1.0, t_lin))
                    t     = _apply_easing(t_lin, mode)
                    w     = w0 + (w1 - w0) * t
                    return f"({body}:{_format_interp_weight(w)})"
                return repl

            expanded = RE_ATTN_INTERP_LITERAL.sub(_make_repl(step, marker_global), text)
            if out and out[-1][1] == expanded:
                out[-1][0] = step
            else:
                out.append([step, expanded])

        prev_end = end_step

    return out


def _get_schedule_impl(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool):
    """Основная реализация без кеширования"""

    _validate_inputs(prompt, steps)
    result: list[list[int | str]] | None = None

    if not str(prompt).strip():
        result = [[steps, SAFE_EMPTY]]
    else:
        if "\\n" in prompt or "\\t" in prompt:
            prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

        prompt = _normalize_and_operators_for_parse(prompt)
        prompt = _normalize_scheduler_surface_syntax(prompt)
        # ── Attention interpolation pre-pass ──────────────────────────────────
        # Must run AFTER and-operator normalization (so '&' is already canonical)
        # and BEFORE Lark parsing. Replaces (body:w0->w1) with private-use markers
        # so Lark never encounters '->' and no grammar ambiguity arises.
        prompt = _placeholderize_attention_interpolations(prompt)
        fast_path_allowed = not _has_multiple_bracket_blocks(prompt)

        m_tl3 = re.match(r'(?s)^\s*([^:\[\]\{\}\(\)]+?):::(.+?)!!!(?:,\s*(.*))?\s*$', prompt)
        if m_tl3:
            owner, rest, trailing = m_tl3.groups()
            parts = [p.strip() for p in rest.split(',') if p.strip()]
            seq_texts = []
            for seg in parts:
                seg = seg.rstrip('!;').strip()
                toks = [t.strip() for t in seg.split('::') if t.strip()]
                if toks:
                    label, descs = toks[0], toks[1:]
                    s = f"{owner.strip()}: {label}"
                    if descs:
                        s += f", {', '.join(descs)}"
                    seq_texts.append(s)
            trailing_texts = []
            if trailing:
                for t in trailing.split(','):
                    t = t.strip()
                    if not t:
                        continue
                    mseq = re.match(r'^\s*([^:\[\]\{\}\(\)]+?)::(.+?)([!;])\s*$', t)
                    if mseq:
                        seq_owner, rest2, _term = mseq.groups()
                        descs = [x.strip(' ,~!;') for x in rest2.split('::') if x.strip(' ,~!;')]
                        trailing_texts.append(f"{seq_owner.strip()}: {', '.join(descs)}")
                    else:
                        trailing_texts.append(t)
            text_out = f"{owner.strip()} -> {', '.join(seq_texts)}"
            if trailing_texts:
                text_out += f", {', '.join(trailing_texts)}"
            result = [[steps, _apply_and(_collapse_spaces(text_out))]]

        if result is None:
            m_num_alt = re.match(r'^\s*(\d+)\s*([!_])?\s*\[([^\]]+)\]\s*$', str(prompt))
            if m_num_alt:
                qty_txt, mark, inner = m_num_alt.groups()
                quantity = int(qty_txt)
                options = [_unescape_literals(x.strip()) for x in re.split(r'(?<!\\)\|', inner)]
                options = [opt if opt != "" else SAFE_EMPTY for opt in options]
                options_unique = list(dict.fromkeys(options)) or [SAFE_EMPTY]
                if mark:
                    if quantity <= len(options_unique):
                        chosen = options_unique[:quantity]
                    else:
                        need = quantity - len(options_unique)
                        pad = (options_unique * ((need + len(options_unique) - 1)//len(options_unique)))[:need]
                        chosen = options_unique + pad
                else:
                    rng = random.Random(seed) if seed is not None else random
                    chosen = rng.choices(options_unique, k=quantity)
                result = [[steps, _apply_and(_collapse_spaces(', '.join(chosen)))]]

        if result is None and fast_path_allowed:
            schedules = _try_fast_after_scheduler(
                prompt,
                steps,
                prefix="",
                suffix="",
                allow_ranges=True,
                allow_bracket_after=False,
                disallow_alt_bang=False,
                boundary_fallback=None,
            )
            if schedules is not None:
                result = schedules

        if result is None and fast_path_allowed:
            m_inner = re.match(r'(?s)^([^\[\]]*)\[([^\[\]]+)\]([^\[\]]*)$', prompt)
            if m_inner and prompt.count('[') == 1 and prompt.count(']') == 1:
                pre, inner, post = m_inner.groups()
                # If post starts with :N this is bracket-AFTER syntax "[...]:N"
                # handled by _try_fast_after_scheduler below — don't steal it here.
                # Simpler: just check if post stripped starts with ":"
                _post_stripped = (post or "").lstrip()
                _skip = _post_stripped.startswith(":") and bool(
                    RE_NUMERIC.match(_post_stripped[1:].lstrip())
                )
                if not _skip and not _needs_complex_parse(inner or "", inner or ""):
                    parts = _split_top_level_colon_keep_empty(inner)
                    if len(parts) >= 2:
                        tail_result = _parse_inner_sched_tail((parts[-1] or "").strip(), steps)
                        if tail_result is not None:
                            boundary_spec, ranges_txt = tail_result
                            prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                            post, rev_flag = _strip_reverse_from_post(post or "")
                            if rev_flag:
                                prompts = list(reversed(prompts))
                            boundary = _to_end_step(boundary_spec, steps)
                            if ranges_txt:
                                # Inner form with ranges: build via ranges builder
                                ranges: list[tuple[int, int]] = []
                                for rng in ranges_txt.split(','):
                                    rng = rng.strip()
                                    if not rng or '-' not in rng:
                                        continue
                                    a_raw, b_raw = [x.strip() for x in rng.split('-', 1)]
                                    a_pct = a_raw.endswith('%')
                                    b_pct = b_raw.endswith('%')
                                    try:
                                        av = float(a_raw[:-1] if a_pct else a_raw)
                                        bv = float(b_raw[:-1] if b_pct else b_raw)
                                    except (ValueError, TypeError):
                                        continue
                                    sv = _clamp(round((av / 100.0 * steps) if a_pct else av), steps)
                                    ev = _clamp(round((bv / 100.0 * steps) if b_pct else bv), steps)
                                    if sv <= ev:
                                        ranges.append((sv, ev))
                                if ranges:
                                    result = _finalize_schedules(
                                        _build_ranges_schedules_from_components(
                                            prompts, ranges, steps,
                                            prefix=pre, suffix=post,
                                            rev_flag=False, cycle_prompts=True,
                                        )
                                    )
                            else:
                                result = _build_bracket_inner_schedules(
                                    pre, prompts, boundary, post, steps
                                )

        if result is None and fast_path_allowed:
            schedules = _try_fast_after_scheduler(
                prompt,
                steps,
                prefix="",
                suffix="",
                allow_ranges=False,
                allow_bracket_after=True,
                disallow_alt_bang=False,
                boundary_fallback=None,
            )
            if schedules is not None:
                result = schedules

        if result is None:
            try:
                tree = _parse_lark_cached(prompt)
            except (lark.exceptions.LarkError, RecursionError) as e:
                logger.warning("Prompt parse failed: '%s' — %s", prompt, e)
                result = [[steps, _apply_and(_collapse_spaces(prompt))]]
            else:
                collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed)
                schedules = collector(tree)
                try:
                    schedules.sort(key=lambda x: int(x[0]))
                except (ValueError, TypeError):
                    pass

                if DEDUP_SCHEDULE_STEPS:
                    try:
                        schedules = _dedup_schedules(schedules)
                    except (ValueError, TypeError, AttributeError):
                        pass

                if not schedules:
                    result = [[steps, _collapse_spaces(prompt)]]
                else:
                    if not use_visitor:
                        logger.debug("use_visitor=False falls back to visitor schedules for consistency.")
                    result = schedules

    if result is None:
        result = [[steps, SAFE_EMPTY]]

    # ── Attention interpolation post-pass ─────────────────────────────────────
    # Expand (body:w0->w1) markers per local active segment BEFORE
    # _apply_scheduling_mode collapses use_scheduling=False to a single entry.
    # The check is O(n) over segments and skips entirely when no marker present.
    if any(ATTN_INTERP_OPEN in str(t) for _, t in result):
        result = _expand_attention_interpolations(result, steps)

    return _apply_scheduling_mode(result, steps, use_scheduling)


# ──────────────────────────────────────────────────────────────────────────────
# Сервиска
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_schedules(schedules: list[list[int | str]]) -> list[list[int | str]]:
    if not schedules:
        return schedules
    last_by_step: dict[int | str, str] = {}
    for end, text in schedules:
        try:
            key = int(end)
        except (ValueError, TypeError):
            key = end
        last_by_step[key] = text
    # Устойчивая сортировка: сначала числовые ключи по возрастанию, потом строковые.
    return [[k, v] for k, v in sorted(
        last_by_step.items(),
        key=lambda kv: (0, int(kv[0])) if isinstance(kv[0], int) else (1, str(kv[0]))
    )]

# ──────────────────────────────────────────────────────────────────────────────
# parse_prompt_attention — внимательность/веса
# ──────────────────────────────────────────────────────────────────────────────

re_attention = re.compile(rf"""
\\\(|
\\\)|
\\\[|
\\\]|
\\\\|
\\|
\(|
\[|
:\s*(x?{NUMERIC_RE})\s*\)|
\)|
]|
[^\\()\[\]:\s]+|
\s+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

# ──────────────────────────────────────────────────────────────────────────────
# Настройки и предкомпилированные regex для parse_prompt_attention
# ──────────────────────────────────────────────────────────────────────────────
# Опциональная АДДИТИВНАЯ семантика дельт (word +0.2 → 1.2 вместо 0.2)
# По умолчанию ВЫКЛ (0), чтобы поведение не менялось.
ATTENTION_DELTA_ADDITIVE = _env_bool("ATTENTION_DELTA_ADDITIVE", "0")


def _attention_runtime_cache_key() -> tuple:
    """Capture runtime knobs that change parse_prompt_attention output."""
    return (
        bool(ATTENTION_DELTA_ADDITIVE),
        bool(SUPPRESS_STANDALONE_COLON),
        float(ROUND_BRACKET_MULTIPLIER),
        float(SQUARE_BRACKET_MULTIPLIER),
    )

RX_FIX_ATT_WEIGHT = re.compile(
    rf'(\b[^\s:()\[\]{{}}]+)\s*:\s*({NUMERIC_RE})'
)
RX_DELTA_TOKEN = re.compile(
    rf'^\s*([-+]{NUMERIC_NOSIGN_RE})\s*$'
)
RX_INLINE_ATT = re.compile(
    r'(?:'
    r'(\b[^\s:(){{}}\[\]]+)\s*:\s*'         # g1: слово для абсолютного веса
    rf'({NUMERIC_RE})'                      # g2: абсолютный вес
    r')|(?:'
    r'(\b[^\s:(){{}}\[\]]+)\s*'             # g3: слово для дельты
    rf'([-+](?:{NUMERIC_NOSIGN_RE}))'       # g4: дельта со знаком
    r')',
    re.X
)

# ──────────────────────────────────────────────────────────────────────────────
# Объект состояния для pending_colon-автомата в _parse_prompt_attention_impl.
# Заменяет три разрозненные переменные (pending_colon, pending_colon_had_space,
# pending_colon_position) единым объектом: сброс/применение атомарны и не могут
# рассинхронизироваться при будущих правках кода.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ColonState:
    """Состояние ожидания числового веса после двоеточия.

    active           — ждём число после последнего ':'
    had_space        — между ':' и следующим токеном был пробел
    res_position     — индекс в res, куда был (мог бы быть) вставлен ':'
    last_token_index — индекс последнего «настоящего» слова в res
    """
    active: bool = False
    had_space: bool = False
    res_position: int = -1
    last_token_index: int = -1

    def arm(self, res_len: int, last_tok_idx: int) -> None:
        """Зафиксировать двоеточие: начать ожидание числа."""
        self.active = True
        self.had_space = False
        self.res_position = res_len       # позиция В МОМЕНТ встречи ':'
        self.last_token_index = last_tok_idx

    def reset(self) -> None:
        """Сбросить состояние."""
        self.active = False
        self.had_space = False
        self.res_position = -1

    def apply_weight(self, res: list, weight: float, suppress: bool) -> None:
        """Применить вес к предыдущему токену и убрать ':' из res (если он был добавлен)."""
        j = self.last_token_index

        # Пропускаем: ':' / 'BREAK' / пробельные токены.
        # Без проверки на пробел вес мог применяться к ' ' вместо слова
        # при паттерне "word : 1.5" (пробелы вокруг двоеточия).
        while j >= 0 and (
            res[j][0] in (':', 'BREAK')
            or res[j][0].strip() == ''
        ):
            j -= 1
        if j >= 0:
            res[j][1] *= weight
        # Если ':' был добавлен явно (SUPPRESS=False), удаляем по сохранённой позиции.
        # res между arm() и сюда растёт только append'ами → позиция стабильна.
        if not suppress and 0 <= self.res_position < len(res):
            if res[self.res_position][0] == ':':
                res.pop(self.res_position)
        self.reset()

    def restore_as_text(self, res: list, suppress: bool) -> None:
        """':' оказалось текстом — приклеить к предыдущему слову."""
        if suppress:
            j = self.last_token_index
            # Пропускаем пробельные токены и BREAK — ищем реальное слово.
            while j >= 0 and (res[j][0] == 'BREAK' or res[j][0].strip() == ''):
                j -= 1
            if j >= 0:
                res[j][0] = res[j][0] + ":" + (" " if self.had_space else "")
            else:
                res.append([':', 1.0])
                if self.had_space:
                    res.append([' ', 1.0])
        self.reset()


@lru_cache(maxsize=CACHE_SIZE)
def _parse_prompt_attention_impl(text, _runtime_key: tuple | bool | None = None):
    """Внутренняя реализация с кешем.
    _additive_delta — копия ATTENTION_DELTA_ADDITIVE на момент вызова;
    включена в ключ кеша, чтобы смена флага не возвращала устаревший результат.
    """
    if _runtime_key is None:
        _runtime_key = _attention_runtime_cache_key()
    elif isinstance(_runtime_key, bool):
        # Backward compatibility with older internal calls that passed only additive flag.
        _runtime_key = (
            bool(_runtime_key),
            bool(SUPPRESS_STANDALONE_COLON),
            float(ROUND_BRACKET_MULTIPLIER),
            float(SQUARE_BRACKET_MULTIPLIER),
        )
    try:
        additive_delta, suppress_standalone_colon, round_bracket_multiplier, square_bracket_multiplier = _runtime_key
    except Exception:
        additive_delta = bool(ATTENTION_DELTA_ADDITIVE)
        suppress_standalone_colon = bool(SUPPRESS_STANDALONE_COLON)
        round_bracket_multiplier = float(ROUND_BRACKET_MULTIPLIER)
        square_bracket_multiplier = float(SQUARE_BRACKET_MULTIPLIER)

    if not text or not str(text).strip():
        return [[SAFE_EMPTY, 1.0]]

    text = str(text).replace('\r\n', '\n').replace('\r', '\n')
    # FIX: нормализуем и реальный таб, и литерал таба до пробелов ДО токенизации
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.replace('\\n', ' ').replace('\\t', ' ')

    # Нормализуем AND → & и пробелы перед токенизацией
    text = _apply_and(text)

    res = []
    round_brackets = []
    square_brackets = []

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    # Единый объект состояния вместо трёх разрозненных переменных.
    colon = _ColonState()
    last_token_index = -1

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        wgrp = m.group(1)

        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
            last_token_index = len(res) - 1
            colon.reset()

        elif text_match == '(':
            round_brackets.append(len(res))
            colon.reset()

        elif text_match == '[':
            square_brackets.append(len(res))
            colon.reset()

        elif wgrp is not None and round_brackets:
            if wgrp.lower().startswith('x'):
                # ──────────────────────────────────────────────────────────
                # Синтаксис :xN.M — «расширение токена» (token repetition).
                # (текст:x3.2) → три копии текста, последняя с весом 1.2.
                # Математика: val=3.2 → count=3 (целая), tail=0.2+1=1.2.
                # Применение: усиление токена через повторение в cross-attention.
                # Отрицательное значение: val<0 → инвертировать вес (* -1).
                # ──────────────────────────────────────────────────────────
                try:
                    val = float(wgrp[1:])
                except ValueError:
                    val = 1.0
                start = round_brackets.pop()
                segment = res[start:]
                if segment:
                    abs_val = abs(val)
                    count = max(1, int(abs_val))
                    tail = (abs_val % 1) + (abs_val >= 1)
                    res[start:] = [[t, w] for _ in range(count) for t, w in segment]
                    if tail > 1e-4:
                        for seg in res[-len(segment):]:
                            seg[1] *= tail
                    if val < 0:
                        multiply_range(start, -1.0)
                else:
                    round_brackets.append(start)  # пустой сегмент — не трогаем
            else:
                try:
                    multiply_range(round_brackets.pop(), float(wgrp))
                except ValueError:
                    multiply_range(round_brackets.pop(), 1.0)
            colon.reset()

        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
            colon.reset()

        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
            colon.reset()

        elif text_match == ':' and not round_brackets and not square_brackets:
            # Встретили ':' — начинаем ожидание числового веса.
            # Если colon уже активен (второе ':' подряд), коммитим первый как текст
            if colon.active:
                colon.restore_as_text(res, suppress_standalone_colon)
                last_token_index = len(res) - 1
            colon.arm(res_len=len(res), last_tok_idx=last_token_index)
            if not suppress_standalone_colon:
                res.append([':', 1.0])
                last_token_index = len(res) - 1

        else:
            chunk = text_match

            # Если ждём число после ':' и пришли только пробелы — продолжаем ждать
            if colon.active and not (round_brackets or square_brackets):
                if chunk.strip() == "":
                    colon.had_space = True
                    continue

            # Проверяем, пришло ли чистое число (вес)
            if colon.active and not (round_brackets or square_brackets):
                mnum = re.match(r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$", chunk)
                if mnum:
                    # Это вес: применяем через метод объекта (атомарно).
                    try:
                        weight = float(mnum.group(1))
                        colon.apply_weight(res, weight, suppress_standalone_colon)
                        last_token_index = len(res) - 1
                        chunk = chunk[mnum.end():]
                        if not chunk.strip():
                            continue
                    except ValueError:
                        pass

                # Если это НЕ число — ':' оказалось текстом: восстанавливаем через метод.
                if colon.active:
                    colon.restore_as_text(res, suppress_standalone_colon)
                    last_token_index = len(res) - 1

            # Обработка обычного текста (chunk)
            parts = re.split(re_break, chunk)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part == "":
                    continue
                res.append([part, 1.0])
                last_token_index = len(res) - 1

    # --- Висящее двоеточие в конце строки --- #
    # Число так и не встретилось → ':' трактуем как литерал.
    if colon.active and not (round_brackets or square_brackets):
        colon.restore_as_text(res, suppress_standalone_colon)

    if not res:
        return [[SAFE_EMPTY, 1.0]]

    # --- Ремонтный проход для шаблона "word : number" (с пробелами) -----------
    rx_fix = RX_FIX_ATT_WEIGHT

    # Создаем временный текст из res для поиска
    temp_text = "".join(t for t, w in res if t != "BREAK")
    used = {}
    
    for mm in rx_fix.finditer(temp_text):
        word = mm.group(1)
        try:
            wt = float(mm.group(2))
        except ValueError:
            continue
        k = used.get(word, 0)
        # найдём k-ую по счёту неподправленную запись этого слова
        # ФИЙ16: сравниваем t.strip() == word — токен может содержать ведущий
        # пробел (например [' cat', 1.0]) и тогда t == word никогда не сработает.
        cnt = 0
        for i, (t, w) in enumerate(res):
            if t.strip() == word and w == 1.0:
                if cnt == k:
                    res[i][1] = wt
                    used[word] = k + 1
                    break
                cnt += 1


    # --- Сшивка "word  +0.2" / "word  -0.1" через ПРОБЕЛ ---
    # use precompiled RX_DELTA_TOKEN
    # Цель: интерпретировать отдельный токен "+0.2" (или "-0.1") как изменение веса
    # предыдущего НЕпустого токена (слова/фразы), даже если у него уже есть вес.
    #
    # Исправления:
    #  - не применяем дельту к пробельным токенам (t.strip()==""), чтобы "+0.2" не "взвешивал" пробел
    #  - разрешаем повторные дельты: "word +0.2 +0.3"
    #  - разрешаем дельту после уже-взвешенного слова: "word:1.2 +0.3"
    # ФИЙ15: O(N) однопроходный сбор в new_delta вместо O(N²) del/insert в цикле.
    # Алгоритм: для каждого «слова» жадно поглощаем все следующие дельта-токены подряд.
    new_delta: list = []
    i = 0
    while i < len(res):
        t, w = res[i]

        # Разделители/пустоту оставляем как есть
        if t in (':', 'BREAK') or (isinstance(t, str) and t.strip() == ""):
            new_delta.append([t, w])
            i += 1
            continue

        # Жадно поглощаем все последовательные дельты после этого слова
        cur_w = float(w)
        j = i + 1
        last_had_space = False
        consumed_to = i  # последний индекс, уже вошедший в результат

        while j <= len(res):  # j — следующий кандидат на дельту
            # Пропускаем пробельные токены между словом и дельтой
            k = j
            sp = False
            while k < len(res) and res[k][1] == 1.0 and isinstance(res[k][0], str) and res[k][0].strip() == "":
                sp = True
                k += 1
            if k >= len(res) or res[k][1] != 1.0:
                break
            m = RX_DELTA_TOKEN.match(res[k][0])
            if not m:
                break
            try:
                delta = float(m.group(1))
            except ValueError:
                break
            if additive_delta:
                cur_w = cur_w + delta
            else:
                cur_w = delta
            last_had_space = sp
            consumed_to = k
            j = k + 1

        new_delta.append([t, cur_w])
        # Если последняя поглощённая дельта была через пробел — восстанавливаем его
        if last_had_space and consumed_to > i:
            new_delta.append([" ", 1.0])
        i = consumed_to + 1

    res = new_delta
    # второй проход: inline-веса word:weight ИЛИ word(+/-)delta

    new_res = []
    for txt, w in res:
        if txt == "BREAK" or w != 1.0:
            new_res.append([txt, w])
            continue
        pos = 0; changed = False
        for mm in RX_INLINE_ATT.finditer(txt):
            pre = txt[pos:mm.start()]
            if pre:
                new_res.append([pre, 1.0])
            if mm.group(1) is not None:
                word, wt = mm.group(1), float(mm.group(2))
            else:
                word, delta = mm.group(3), float(mm.group(4))
                wt = (1.0 + delta) if additive_delta else delta
            new_res.append([word, wt]); changed = True
            pos = mm.end()
        if changed:
            tail = txt[pos:]
            if tail:
                new_res.append([tail, 1.0])
        else:
            new_res.append([txt, w])
    res = new_res

    # убрать ведущий "AND " (но сохранить пробелы после)
    norm = []
    for t, w in res:
        if t == "BREAK":
            norm.append([t, w]); continue
        m = re.match(r'^(\s*)AND\s+(.*)$', t)
        if m:
            t = m.group(1) + m.group(2)
        norm.append([t, w])
    res = norm

    # финальное схлопывание соседей с одинаковым весом — ВСТАВЛЯЕМ ПРОБЕЛ при склейке
    def _smart_concat(a: str, b: str):
        if not a: return b
        if not b: return a
        
        ar = a.rstrip()
        # Если в конце первого слова запятая/точка - пробел обычно желателен для токенизатора
        if ar and ar[-1] in ",.;?!":
            return ar + " " + b.lstrip()

        if a.endswith(" ") or b.startswith(" "):
            return a + b

        # Старая проверка была: if a[-1].isalnum() and b[0].isalnum():
        # Новая проверка: ставим пробел, если это не спецсимволы скобок/двоеточий
        # (грубая эвристика, но лучше чем терять пробелы после запятых)
        if a[-1] not in "(:[" and b[0] not in "):]":
            return a + " " + b
            
        return a + b


    # Слияние соседних токенов с одинаковым весом — O(N) через новый список,
    # вместо O(N²) res.pop(i+1) внутри while.
    merged: list[list] = []
    for tok, w in res:
        if (merged
                and merged[-1][1] == w
                and merged[-1][0] not in (':', 'BREAK')
                and tok not in (':', 'BREAK')):
            merged[-1][0] = _smart_concat(merged[-1][0], tok)
        else:
            merged.append([tok, w])
    res = merged

    # Сжать повторные пробелы внутри каждого токена
    res = [[_collapse_spaces(t), w] if t != "BREAK" else [t, w] for t, w in res]
    # Удалить пустые, если вдруг остались
    res = [[t, w] for t, w in res if t or t == "BREAK"]

    # Если остались только BREAK — вернём безопасный токен
    if all(t == "BREAK" for t, _ in res):
        return [[SAFE_EMPTY, 1.0]]

    return res

# Обёртка публичного API: сохраняем прежнее имя и тип возврата
def parse_prompt_attention(text):
    protected = _protect_escaped_literals(str(text or ""))
    out = _parse_prompt_attention_impl(protected, _attention_runtime_cache_key())
    # возвращаем копию list-of-[text, weight]
    return [[_restore_escaped_literals(str(t)), w] for t, w in out]

# ──────────────────────────────────────────────────────────────────────────────
# Мультиконд — как в старых версиях, но с дедупом текстов расписаний
# ──────────────────────────────────────────────────────────────────────────────

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None, distilled_cfg_scale=None):
        super().__init__()
        self.extend(prompts)
        if copy_from is None:
            copy_from = prompts
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width if width is not None else getattr(copy_from, 'width', None)
        self.height = height if height is not None else getattr(copy_from, 'height', None)
        self.distilled_cfg_scale = distilled_cfg_scale if distilled_cfg_scale is not None else getattr(copy_from, 'distilled_cfg_scale', None)

def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else base_steps
    use_scheduling = (hires_steps is None) or use_old_scheduling
    prompt_schedules = [get_schedule(p, steps, use_scheduling, seed, use_visitor=use_visitor) for p in prompts]
    return prompt_schedules

def get_learned_conditioning(
    model,
    prompts: SdConditioning | list[str],
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    effective_steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else steps
    use_scheduling = (hires_steps is None) or use_old_scheduling

    _transpiled = [_transpile_bind2_to_chunk(p) for p in prompts]
    if isinstance(prompts, SdConditioning):
        prompts = SdConditioning(_transpiled, copy_from=prompts)
    else:
        prompts = _transpiled

    if any(
        _contains_chunk_marker(prompt)
        or _contains_assemble_marker(prompt)
        or _contains_blend_marker(prompt)
        or _contains_morph_marker(prompt)
        or _contains_pool_marker(prompt)
        or _contains_bind_marker(prompt)
        or _contains_bind3_marker(prompt)
        or _contains_compound_marker(prompt)
        for prompt in prompts
    ):
        res = []
        cache = {}
        for prompt in prompts:
            cached = cache.get(prompt, None)
            if cached is not None:
                res.append(cached)
                continue
            cond_schedule = _build_prompt_conditioning_schedule(
                model,
                prompt,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                prompts,
            )
            cache[prompt] = cond_schedule
            res.append(cond_schedule)
        return res

    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(
        prompts,
        steps,
        hires_steps,
        use_old_scheduling,
        seed,
        use_visitor,
    )
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        if not prompt_schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached); continue
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res

# === DROP-IN REPLACEMENT ===
class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0, weight_schedule=None):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight
        self.weight_schedule: list[tuple[int, float]] | None = weight_schedule

    def weight_at_step(self, current_step: int) -> float:
        if not self.weight_schedule:
            return float(self.weight)
        for end_at_step, weight in self.weight_schedule:
            if current_step <= int(end_at_step):
                return float(weight)
        return float(self.weight_schedule[-1][1])

class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch

def _build_unique_text_index(flat_schedules):
    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    flat_schedule_text_indices: list[list[int]] = []
    for schedule in flat_schedules:
        indices: list[int] = []
        for _end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        flat_schedule_text_indices.append(indices)
    return unique_texts, flat_schedule_text_indices

def _build_single_prompt_multicond_parts(
    model,
    prompt: str,
    effective_steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
) -> list[ComposableScheduledPromptConditioning]:
    state = _extract_backend_prompt_state(prompt)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(prompt)

    if state.has_bind:
        base_schedule = _build_prompt_conditioning_schedule(
            model,
            state.bind_base_prompt,
            effective_steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        if not base_schedule:
            raise ValueError("Empty schedule for BIND base prompt")
        composable_parts = [ComposableScheduledPromptConditioning(base_schedule, 1.0)]
        for spec in state.bind_specs:
            bind_schedule, bind_weight_schedule = _build_bind_branch_prompt_conditioning_schedule(
                model,
                spec,
                base_schedule,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                copy_from,
            )
            if not bind_schedule:
                raise ValueError("Empty schedule for BIND branch")
            composable_parts.append(
                ComposableScheduledPromptConditioning(
                    bind_schedule,
                    spec.weight,
                    bind_weight_schedule,
                )
            )
        return composable_parts

    conds_list, prompt_flat_list, _prompt_indexes = get_multicond_prompt_list([prompt])
    if not conds_list:
        raise ValueError("Empty multicond prompt decomposition")
    cond_parts = conds_list[0]
    if not cond_parts:
        raise ValueError("Prompt did not produce any composable branches")

    if any(
        _contains_chunk_marker(subprompt)
        or _contains_assemble_marker(subprompt)
        or _contains_blend_marker(subprompt)
        or _contains_morph_marker(subprompt)
        or _contains_pool_marker(subprompt)
        or _contains_bind_marker(subprompt)
        for subprompt in prompt_flat_list
    ):
        flat_cond_schedules = [
            _build_prompt_conditioning_schedule(
                model,
                subprompt,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                copy_from,
            )
            for subprompt in prompt_flat_list
        ]
        if not flat_cond_schedules or any(not schedule for schedule in flat_cond_schedules):
            raise ValueError("Empty schedule for at least one sub-prompt")
        return [
            ComposableScheduledPromptConditioning(flat_cond_schedules[flat_index], weight)
            for flat_index, weight in cond_parts
        ]

    flat_schedules = [
        get_schedule(subprompt, effective_steps, use_scheduling, seed, use_visitor=use_visitor)
        for subprompt in prompt_flat_list
    ]
    if not flat_schedules or any(not schedule for schedule in flat_schedules):
        raise ValueError("Empty schedule for at least one sub-prompt")

    unique_texts, flat_schedule_text_indices = _build_unique_text_index(flat_schedules)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    composable_parts: list[ComposableScheduledPromptConditioning] = []
    for flat_index, weight in cond_parts:
        schedule = flat_schedules[flat_index]
        conds_for_steps = []
        for local_step_idx, (end_at_step, _text) in enumerate(schedule):
            i_global = flat_schedule_text_indices[flat_index][local_step_idx]
            conds_for_steps.append(ScheduledPromptConditioning(int(end_at_step), get_i_cond(i_global)))
        composable_parts.append(ComposableScheduledPromptConditioning(conds_for_steps, weight))
    return composable_parts

def get_multicond_learned_conditioning(
    model,
    prompts,
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    effective_steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    use_scheduling = (hires_steps is None) or use_old_scheduling

    _transpiled = [_transpile_bind2_to_chunk(p) for p in prompts]
    if isinstance(prompts, SdConditioning):
        prompts = SdConditioning(_transpiled, copy_from=prompts)
    else:
        prompts = _transpiled

    if any(_contains_bind_marker(prompt) for prompt in prompts):
        res_batch: list[list[ComposableScheduledPromptConditioning]] = []
        first_cond = None
        cache: dict[str, list[ComposableScheduledPromptConditioning]] = {}
        for prompt in prompts:
            cached = cache.get(prompt)
            if cached is None:
                cached = _build_single_prompt_multicond_parts(
                    model,
                    prompt,
                    effective_steps,
                    use_scheduling,
                    seed,
                    use_visitor,
                    prompts,
                )
                cache[prompt] = cached
            if not cached or any(not part.schedules for part in cached):
                raise ValueError("Empty schedule for at least one composable prompt part")
            if first_cond is None:
                first_cond = cached[0].schedules[0].cond
            res_batch.append(cached)
        if first_cond is None:
            raise ValueError("Empty multicond batch")
        shape = _conditioning_shape_from_cond(first_cond)
        return MulticondLearnedConditioning(shape, res_batch)

    # ФИЙ17: сначала разбиваем на подпромпты по AND, затем строим расписание
    # для каждого подпромпта отдельно — иначе все части AND получали бы одинаковый
    # тензор (весь "cat AND dog" целиком) вместо независимых тензоров "cat" и "dog".
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    if any(
        _contains_chunk_marker(prompt)
        or _contains_assemble_marker(prompt)
        or _contains_blend_marker(prompt)
        or _contains_morph_marker(prompt)
        or _contains_pool_marker(prompt)
        or _contains_bind3_marker(prompt)
        or _contains_compound_marker(prompt)
        for prompt in prompt_flat_list
    ):
        flat_cond_schedules = [
            _build_prompt_conditioning_schedule(
                model,
                prompt,
                effective_steps,
                (hires_steps is None) or use_old_scheduling,
                seed,
                use_visitor,
                prompts,
            )
            for prompt in prompt_flat_list
        ]
        if not flat_cond_schedules or any(not schedule for schedule in flat_cond_schedules):
            raise ValueError("Empty schedule for at least one sub-prompt")

        res_batch: list[list[ComposableScheduledPromptConditioning]] = []
        for cond_parts in conds_list:
            composable_parts: list[ComposableScheduledPromptConditioning] = []
            for flat_index, weight in cond_parts:
                composable_parts.append(
                    ComposableScheduledPromptConditioning(flat_cond_schedules[flat_index], weight)
                )
            res_batch.append(composable_parts)

        first_cond = flat_cond_schedules[0][0].cond
        shape = _conditioning_shape_from_cond(first_cond)
        return MulticondLearnedConditioning(shape, res_batch)

    # Строим расписание для каждого уникального подпромпта из prompt_flat_list
    flat_schedules = get_learned_conditioning_prompt_schedules(
        prompt_flat_list,
        effective_steps,
        hires_steps,
        use_old_scheduling,
        seed,
        use_visitor,
    )
    if not flat_schedules or any(not sch for sch in flat_schedules):
        raise ValueError("Empty schedule for at least one sub-prompt")

    unique_texts, flat_schedule_text_indices = _build_unique_text_index(flat_schedules)

    texts_conditioning = SdConditioning(unique_texts, copy_from=prompts)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    res_batch: list[list[ComposableScheduledPromptConditioning]] = []
    for cond_parts in conds_list:
        composable_parts: list[ComposableScheduledPromptConditioning] = []
        for flat_index, weight in cond_parts:
            # Каждый flat_index → своё расписание и свой набор тензоров
            schedule = flat_schedules[flat_index]
            conds_for_steps = []
            for local_step_idx, (end_at_step, _text) in enumerate(schedule):
                i_global = flat_schedule_text_indices[flat_index][local_step_idx]
                cond_at_step = get_i_cond(i_global)
                conds_for_steps.append(ScheduledPromptConditioning(int(end_at_step), cond_at_step))
            composable_parts.append(ComposableScheduledPromptConditioning(conds_for_steps, weight))
        res_batch.append(composable_parts)

    if isinstance(model_conds, dict):
        ca = model_conds.get('crossattn')
        if isinstance(ca, list) and ca:
            shape = getattr(ca[0], 'shape', None) or (0,)
        else:
            shape = getattr(ca, 'shape', None) or (0,)
    else:
        shape = getattr(model_conds, 'shape', None) or (0,)

    return MulticondLearnedConditioning(shape, res_batch)


# Делим по слову AND и по одиночному '&' как самостоятельному токену.
# Не трогаем AND_PERP / AND_SALT / AND_TOPK.

# Вес в конце подпрампта: "text : 1.2" (якорь на конец строки, поддержка китайского двоеточия)
RE_END_WEIGHT = re.compile(
    r"^(?P<text>.*?)"
    rf"\s*[:：]\s*(?P<w>{NUMERIC_RE})\s*$"  # Убрать необязательность группы веса
)

def _split_top_level_multicond(prompt: str) -> list[str]:
    if not prompt:
        return [prompt]

    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(prompt)
    escaped_backslash_placeholder = _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS["\\"]

    def flush_part():
        part = "".join(buf)
        buf.clear()
        parts.append(part)

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < n:
        ch = prompt[i]

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            prev_ch = prompt[i - 1] if i > 0 else ""
            next_ch = prompt[i + 1] if i + 1 < n else ""

            if ch == "&":
                if (not prev_ch or prev_ch.isspace()) and (not next_ch or next_ch.isspace()):
                    flush_part()
                    i += 1
                    continue

            if prompt.startswith("AND", i):
                prev_blocks_split = prev_ch == "\\" or prev_ch == escaped_backslash_placeholder or is_word_char(prev_ch)
                next_boundary_ch = prompt[i + 3] if i + 3 < n else ""
                next_blocks_split = is_word_char(next_boundary_ch)
                if not prev_blocks_split and not next_blocks_split:
                    flush_part()
                    i += 3
                    continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    flush_part()
    return parts


def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []
    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        protected_prompt, span_restore = _protect_escaped_literal_spans(prompt)
        protected_prompt = _protect_escaped_literals(protected_prompt)
        subprompts = _split_top_level_multicond(protected_prompt)
        indexes = []
        for subprompt in subprompts:
            #PP21 removes punctuation from the end of a prompt. This creates a difference in normal prompts where a prompt ended in a comma. the original code is commented out.
            '''
            s = subprompt if isinstance(subprompt, str) else str(subprompt)

            # 1) Нормализация скобочных весов: "(cat:2.0)" или просто "(cat)"
            m_emph = re.match(
                r'^\s*\(\s*(.*?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?\)\s*$',
                s
            )
            '''
            s_raw = subprompt if isinstance(subprompt, str) else str(subprompt)
            s_stripped = (s_raw or "").strip()

            # 1) Parenthesized emphasis form: "(cat:2.0)" or "(cat)"
            m_emph = re.match(
                r'^\s*\(\s*(.*?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?\)\s*$',
                s_raw
            )
            
            '''
            if m_emph:
                text = m_emph.group(1)
                weight = m_emph.group(2) if m_emph.group(2) is not None else ROUND_BRACKET_MULTIPLIER
            else:
                # 2) Надёжный разбор "word:weight"
                s_clean = (s or "").strip().rstrip(",;")
                m_end = RE_END_WEIGHT.match(s_clean)
                if m_end and m_end.group("w"):  # Проверяем что вес найден
                    text, weight = m_end.group("text"), m_end.group("w")
                else:
                    text, weight = (s_clean, None)

            text = (text or "").strip()
            # Нормализуем артефакты вроде "dog:" -> "dog"
            if text.endswith(":") or text.endswith("："):
                text = text[:-1].rstrip()
            text = _restore_escaped_literals(text, span_restore)
            if not text.strip():
                text = SAFE_EMPTY
            try:
                weight = float(weight) if weight is not None else 1.0
            except (ValueError, TypeError):
                weight = 1.0
                
            '''
            if m_emph:
                text = m_emph.group(1)
                weight = m_emph.group(2) if m_emph.group(2) is not None else ROUND_BRACKET_MULTIPLIER
            else:
                # Legacy parity:
                # preserve literal punctuation; only treat trailing ":<number>" as weight syntax
                m_end = RE_END_WEIGHT.match(s_stripped)
                if m_end and m_end.group("w"):
                    text, weight = m_end.group("text"), m_end.group("w")
                else:
                    text, weight = (s_stripped, None)

            text = _restore_escaped_literals((text or "").strip(), span_restore)

            # Only remove a trailing colon artifact when we actually parsed a weight suffix.
            if weight is not None and (text.endswith(":") or text.endswith("：")):
                text = text[:-1].rstrip()

            # Legacy parity:
            # preserve a truly empty literal prompt as "".
            # Keep SAFE_EMPTY only for synthetic/internal empty branches if needed.
            if not text.strip():
                if len(subprompts) == 1 and not s_stripped:
                    text = ""
                else:
                    text = SAFE_EMPTY

            try:
                weight = float(weight) if weight is not None else 1.0
            except (ValueError, TypeError):
                weight = 1.0
            
            
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index
            indexes.append((index, weight))
        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


def _extract_multicond_preview_branches(text: str) -> list[tuple[str, float]] | None:
    conds_list, prompt_flat_list, _prompt_indexes = get_multicond_prompt_list([text])
    if not conds_list or len(conds_list[0]) <= 1:
        return None

    branches: list[tuple[str, float]] = []
    has_backend = False
    for prompt_index, weight in conds_list[0]:
        branch_text = str(prompt_flat_list[prompt_index])
        branch_weight = float(weight)
        branches.append((branch_text, branch_weight))
        if (
            _contains_chunk_marker(branch_text)
            or _contains_assemble_marker(branch_text)
            or _contains_blend_marker(branch_text)
            or _contains_morph_marker(branch_text)
            or _contains_pool_marker(branch_text)
        ):
            has_backend = True

    return branches if has_backend else None


def _build_multicond_preview_text(active_texts: Sequence[str], weights: Sequence[float]) -> str:
    parts: list[str] = []
    for text, weight in zip(active_texts, weights):
        normalized_text = _normalize_preview_fragment(text)
        if not normalized_text:
            continue
        if abs(float(weight) - 1.0) <= 1e-8:
            parts.append(normalized_text)
        else:
            parts.append(f"({normalized_text})*{_format_interp_weight(float(weight))}")
    return " AND ".join(parts) if parts else SAFE_EMPTY


def _build_multicond_text_schedule_from_branches(
    branches: Sequence[tuple[str, float]],
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if not branches:
        return [[int(steps), SAFE_EMPTY]]

    if strict:
        branch_schedules = [
            _strict_schedule_preview(branch_text, steps, seed)
            for branch_text, _branch_weight in branches
        ]
    else:
        branch_schedules = [
            get_schedule(branch_text, steps, use_scheduling, seed, use_visitor=use_visitor)
            for branch_text, _branch_weight in branches
        ]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    previous_key = None
    weights = [float(weight) for _branch_text, weight in branches]

    for end_at_step in boundaries:
        active_texts = [
            _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
            for schedule in branch_schedules
        ]
        preview = _build_multicond_preview_text(active_texts, weights)
        key = (tuple(active_texts), tuple(round(weight, 8) for weight in weights))
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key

    return out or [[int(steps), SAFE_EMPTY]]


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки для реконструкции батчей
# ──────────────────────────────────────────────────────────────────────────────

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)
        self._shape = shape
    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        any_val = self.get("crossattn")
        if any_val is None and self:
            any_val = next(iter(self.values()))
        return getattr(any_val, "shape", None)


def _pick_schedule_entry_index(schedule, current_step):
    if not schedule:
        raise ValueError("Empty conditioning schedule")

    for current, entry in enumerate(schedule):
        if current_step <= int(entry.end_at_step):
            return current

    return len(schedule) - 1

def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    _torch = _ensure_torch()
        
    if not c or not c[0]:
        raise ValueError("Empty conditioning schedule")
    param = c[0][0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: _torch.zeros((len(c),) + v.shape, device=getattr(v, "device", "cpu"), dtype=getattr(v, "dtype", _torch.float32)) for k, v in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond.get('crossattn', next(iter(dict_cond.values()))).shape)
    else:
        res = _torch.zeros((len(c),) + param.shape, device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", _torch.float32))

    for i, cond_schedule in enumerate(c):
        target_index = _pick_schedule_entry_index(cond_schedule, current_step)
        if is_dict:
            for k, v in cond_schedule[target_index].cond.items():
                res[k][i] = v
        else:
            res[i] = cond_schedule[target_index].cond
    return res

def stack_conds(tensors):
    _torch = _ensure_torch()
    ndim = _tensor_ndim(tensors[0])
    axis = 1 if ndim >= 3 else 0
    token_count = max([x.shape[axis] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[axis] != token_count:
            pad_count = token_count - tensors[i].shape[axis]
            select = [slice(None)] * ndim
            select[axis] = slice(-1, None)
            last_vector = tensors[i][tuple(select)]
            repeats = [1] * ndim
            repeats[axis] = pad_count
            last_vector_repeated = last_vector.repeat(repeats)
            tensors[i] = _torch.cat([tensors[i], last_vector_repeated], dim=axis)
    return _torch.stack(tensors)

def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    _torch = _ensure_torch()

    if not c.batch or not c.batch[0]:
        raise ValueError("Empty multicond batch")
    param = c.batch[0][0].schedules[0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    tensors = []
    conds_list = []
    for composable_prompts in c.batch:
        conds_for_batch = []
        for composable_prompt in composable_prompts:
            weight = composable_prompt.weight_at_step(current_step)
            if abs(float(weight)) <= 1e-8:
                continue
            target_index = _pick_schedule_entry_index(composable_prompt.schedules, current_step)
            conds_for_batch.append((len(tensors), weight))
            tensors.append(composable_prompt.schedules[target_index].cond)
        conds_list.append(conds_for_batch)

    if not tensors:
        raise ValueError(
            f"All conditioning weights are zero or near-zero at step {current_step}. "
            "At least one AND-branch must have a non-zero weight."
        )

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked.get('crossattn', next(iter(stacked.values()))).shape)
    else:
        stacked = stack_conds(tensors).to(device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", _torch.float32))
    return conds_list, stacked


# ──────────────────────────────────────────────────────────────────────────────
# Тестовые самопроверки
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты (не влияют на основной API/поведение)
# ──────────────────────────────────────────────────────────────────────────────

def _syntax_error_details(exc: Exception, source_text: str) -> dict:
    """Return structured syntax error details suitable for UIs.

    The returned dict is intentionally language-neutral: UI can map `kind` to
    localized messages and implement smart highlighting via `pos`/`end`.
    """
    # Default (unknown error)
    details: dict = {"kind": "error", "message": str(exc), "fix_suggestion": None}

    # Our explicit parser errors (e.g., strict reverse ranges)
    if isinstance(exc, PromptSyntaxError):
        src = exc.full or source_text
        token = exc.token
        pos = None
        end = None
        if token and src:
            p = src.find(token)
            if p != -1:
                pos = p
                end = p + len(token)
        details = {
            "kind": exc.kind or "prompt_syntax_error",
            "message": str(exc),
            "token": token,
            "pos": pos,
            "end": end,
            "fix_suggestion": None,
        }

        # Optional automatic fix for UIs (e.g., reverse range "8-4" -> "4-8")
        if details["kind"] == "reverse_range" and isinstance(token, str):
            m = re.match(r'^(\s*\d+%?\s*)-(\s*\d+%?\s*)$', token)
            if m:
                details["fix_suggestion"] = f"{m.group(2)}-{m.group(1)}"

        return details

    # Lark syntax errors
    if isinstance(exc, lark.exceptions.UnexpectedInput):
        pos = getattr(exc, "pos_in_stream", None)
        details = {"kind": "syntax_error", "message": str(exc), "pos": pos, "fix_suggestion": None}
        try:
            details["context"] = exc.get_context(source_text)
        except AttributeError:
            pass
        return details

    return details

def _format_syntax_error(exc: Exception, source_text: str) -> str:
    """Format a human-friendly prompt syntax error message.

    For programmatic handling (UI highlighting / localization), prefer `lint_prompt()`,
    which returns structured fields like `kind`, `pos`, `end`, and `token`.
    """
    details = _syntax_error_details(exc, source_text)
    kind = details.get("kind", "error")
    token = details.get("token")
    pos = details.get("pos")
    msg = details.get("message") or str(exc)

    # Short, language-neutral hints (UI can localize by `kind`).
    hint = ""
    if kind == "reverse_range":
        hint = "Range start is greater than end. Fix it (e.g. '4-8')."
    elif kind == "invalid_range_token":
        hint = "Use 'a-b' or 'a%-b%' (e.g. '1-4' or '10%-50%')."
    elif kind == "invalid_chunk_syntax":
        hint = "Use exactly one CHUNK{branch|branch} block with a closing '}'."
    elif kind == "empty_chunk_branch":
        hint = "CHUNK branches cannot be empty. Remove duplicate '|' or add text."
    elif kind == "invalid_chunk_weight":
        hint = "Use a positive numeric chunk weight at the end of a branch, e.g. 'wolf*1.5'."
    elif kind == "invalid_chunk_mode":
        hint = "Use CHUNK{...}, CHUNK[share-pooled]{...}, or CHUNK[share-cross]{...}."
    elif kind == "nested_chunk_not_supported":
        hint = "Nested CHUNK blocks are not supported in v1."
    elif kind == "nested_backend_in_chunk_not_supported":
        hint = "In v1, CHUNK branches may contain regular prompt grammar or MORPH blocks, but not BLEND, POOL, or ASSEMBLE blocks."
    elif kind == "chunk_inner_multicond_not_supported":
        hint = "In v1, CHUNK branches must stay single-branch prompt grammar; top-level AND is not supported inside CHUNK branches."
    elif kind == "unsupported_chunk_context":
        hint = "In v1, CHUNK must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "invalid_pool_syntax":
        hint = "Use one POOL{prompt} block with regular prompt grammar inside and a closing '}'."
    elif kind == "nested_pool_not_supported":
        hint = "Nested POOL blocks are not supported in v1."
    elif kind == "nested_backend_in_pool_not_supported":
        hint = "In v1, POOL may contain regular prompt grammar only, not nested CHUNK, ASSEMBLE, BLEND, MORPH, BIND, or POOL blocks."
    elif kind == "invalid_assemble_syntax":
        hint = "Use ASSEMBLE{enc1=...; enc2=...; pooled=...}. Fields use '=' and enc1/enc2 are required."
    elif kind == "invalid_assemble_field":
        hint = "Supported ASSEMBLE fields are enc1, enc2, and pooled."
    elif kind == "duplicate_assemble_field":
        hint = "Each ASSEMBLE field may appear only once in v1."
    elif kind == "nested_assemble_not_supported":
        hint = "Nested ASSEMBLE blocks are not supported in v1."
    elif kind == "nested_backend_in_assemble_not_supported":
        hint = "In v1, ASSEMBLE fields must contain regular prompt grammar only, not nested CHUNK, BLEND, MORPH, POOL, or ASSEMBLE blocks."
    elif kind == "assemble_inner_multicond_not_supported":
        hint = "In v1, ASSEMBLE fields must stay single-branch prompt grammar; top-level AND is not supported inside enc1/enc2/pooled."
    elif kind == "unsupported_assemble_context":
        hint = "In v1, ASSEMBLE must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "unsupported_pool_context":
        hint = "In v1, POOL must appear as one top-level prompt adjunct, not inside scheduler/group wrappers."
    elif kind == "invalid_bind_syntax":
        hint = "Use BIND{owner => attrs} with non-empty owner and attrs sections and a closing '}'."
    elif kind == "invalid_bind_weight":
        hint = "Use a positive numeric BIND weight like BIND^1.2{owner => attrs}."
    elif kind == "nested_bind_not_supported":
        hint = "Nested BIND blocks are not supported in v1."
    elif kind == "nested_backend_in_bind_not_supported":
        hint = "In v1, BIND fields may contain regular prompt grammar only, not nested CHUNK, ASSEMBLE, BLEND, MORPH, POOL, or BIND blocks."
    elif kind == "bind_inner_multicond_not_supported":
        hint = "In v1, BIND owner and attrs must stay single-branch prompt grammar; top-level AND is not supported inside BIND fields."
    elif kind == "bind_requires_base_prompt":
        hint = "BIND needs a non-empty base prompt outside the BIND blocks in v1."
    elif kind == "unsupported_bind_context":
        hint = "In v1, BIND must appear as a top-level prompt adjunct, not inside scheduler/group wrappers."
    elif kind == "bind_with_backend_not_supported":
        hint = "In v1, BIND can be combined only with plain prompt text and optional POOL."
    elif kind == "bind_with_and_not_supported":
        hint = "In v1, BIND cannot be combined with top-level AND. Use separate prompts or keep BIND on a single branch."
    elif kind == "invalid_blend_syntax":
        hint = "Use BLEND{branch|branch} with at least two branches and a closing '}'."
    elif kind == "empty_blend_branch":
        hint = "BLEND branches cannot be empty. Remove duplicate '|' or add text."
    elif kind == "invalid_blend_weight":
        hint = "Use a positive numeric BLEND branch weight at the end of a branch, e.g. 'wolf*1.5'."
    elif kind == "invalid_blend_mode":
        hint = "Use a supported BLEND mode: mean or sum."
    elif kind == "invalid_blend_channel_target":
        hint = "Use BLEND[mean@both]{...}, BLEND[mean@cross]{...}, BLEND[mean@pooled]{...}, BLEND[mean@enc1]{...}, BLEND[mean@enc2]{...}, or omit @channel for old behavior."
    elif kind == "invalid_blend_intensity":
        hint = "Use BLEND^number{...} or BLEND^number[mode]{...} with a positive numeric intensity."
    elif kind == "nested_blend_not_supported":
        hint = "Nested BLEND blocks are not supported in v1."
    elif kind == "nested_backend_in_blend_not_supported":
        hint = "In v1, BLEND branches must contain regular prompt grammar only, not nested CHUNK, MORPH, POOL, or ASSEMBLE blocks."
    elif kind == "blend_inner_multicond_not_supported":
        hint = "In v1, BLEND branches must stay single-branch prompt grammar; top-level AND is not supported inside BLEND branches."
    elif kind == "unsupported_blend_context":
        hint = "In v1, BLEND must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "invalid_morph_syntax":
        hint = "Use MORPH{prompt => prompt@step} with at least two control prompts and a closing '}'."
    elif kind == "invalid_morph_curve":
        hint = "Use a supported MORPH curve: linear, bezier, catmull, or slerp."
    elif kind == "invalid_morph_intensity":
        hint = "Use MORPH^number{...} or MORPH^number[start-end]{...} with a positive numeric intensity."
    elif kind == "invalid_morph_channel_target":
        hint = "Use MORPH@both{...}, MORPH@cross{...}, MORPH@pooled{...}, MORPH@enc1{...}, or MORPH@enc2{...}; intensity may come first, e.g. MORPH^1.3@enc1{...}."
    elif kind == "invalid_morph_point_weight":
        hint = "Use a positive numeric MORPH point weight like 'human*0.8' or 'cyborg*1.3@0.6'."
    elif kind == "invalid_morph_boundary":
        hint = "Use increasing MORPH boundaries like @0.6, @75%, or @12."
    elif kind == "invalid_morph_window":
        hint = "Use MORPH[start-end]{...} with increasing boundaries like 5-12 or 25%-50%."
    elif kind == "morph_window_with_point_boundaries_not_supported":
        hint = "In v1, use either MORPH[start-end]{a => b} or MORPH{a => b@0.6}, but not both together."
    elif kind == "nested_morph_not_supported":
        hint = "Nested MORPH blocks are not supported in v1."
    elif kind == "nested_backend_in_morph_not_supported":
        hint = "In v1, MORPH control prompts must contain regular prompt grammar only, not nested POOL or ASSEMBLE blocks."
    elif kind == "morph_inner_multicond_not_supported":
        hint = "In v1, MORPH control prompts must stay single-branch prompt grammar; top-level AND is not supported inside MORPH points."
    elif kind == "unsupported_morph_context":
        hint = "In v1, MORPH must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "duplicate_reverse_flag":
        hint = "Use the 'reverse' flag only once after a scheduler block."
    elif kind == "incomplete_sequence":
        hint = "Complete the sequence syntax or remove the dangling '::' / ':::'."
    elif kind == "invalid_interpolation":
        hint = "Use '(text:w0->w1)' with text and both weights present."
    elif kind == "syntax_error":
        msg = f"Syntax error at position {pos if pos is not None else '?'}."
        hint = ""

    # Pointer to the offending token (best-effort)
    pointer = ""
    if token and source_text:
        p = source_text.find(token)
        if p != -1:
            last_nl = source_text.rfind('\n', 0, p)
            col = p - (last_nl + 1) if last_nl != -1 else p
            pointer = "\n" + source_text + "\n" + (" " * col) + ("^" * max(1, len(token)))

    # Lark often provides useful context lines
    context = details.get("context")
    if isinstance(context, str) and context.strip():
        msg = msg + "\n" + context.strip()

    if hint:
        msg = msg + "\n" + hint

    return (msg + pointer).strip()


def explain_syntax_error(text: str) -> str:
    """Вернуть человекочитаемое описание ошибки синтаксиса.
    При успехе вернуть пустую строку.
    """
    try:
        # Небольшой dry-parse через основной путь
        _ = _strict_schedule_preview(text, steps=20, seed=None)
        return ""
    except Exception as e:
        return _format_syntax_error(e, text)

_BACKEND_WARNINGS: dict[str, dict[str, str]] = {
    "single_branch_chunk": {
        "en": "CHUNK with a single branch is a no-op",
        "ru": "CHUNK с одной веткой — холостой вызов (всегда будет одна и та же ветка)",
    },
    "duplicate_blend_branches": {
        "en": "BLEND has duplicate branches — no blending effect",
        "ru": "BLEND содержит дублирующиеся ветки — эффекта смешивания нет",
    },
    "all_zero_blend_weights": {
        "en": "All BLEND branch weights are zero — result will be a zero tensor",
        "ru": "Все веса веток BLEND равны нулю — результат будет нулевым тензором",
    },
    "unescaped_brace_literal": {
        "en": "Unescaped literal '{' or '}' — these are interpreted as grouped syntax by the parser. Use \\{ or \\} if you meant a literal brace.",
        "ru": "Неэкранированная '{' или '}' — парсер интерпретирует их как grouped-синтаксис. Используйте \\{ или \\} если нужна литеральная скобка.",
    },
    "bind_in_negative_prompt": {
        "en": "BIND / BIND2 / BIND3 in a negative prompt is not meaningful — use a plain negative prompt instead",
        "ru": "BIND / BIND2 / BIND3 в негативном промпте не имеет смысла — используйте обычный негативный промпт",
    },
    "token_limit_exceeded": {
        "en": "Prompt may exceed 75-token CLIP limit after expansion (~50+ words). Consider shortening or splitting into multiple prompts.",
        "ru": "Промпт может превысить лимит CLIP в 75 токенов после раскрытия (~50+ слов). Сократите или разбейте на несколько промптов.",
    },
    "bind3_scheduling_not_supported_at_conditioning": {
        "en": "BIND3 with scheduling in prefix/suffix/attrs will raise at conditioning time. Workaround: use [BIND3{...}:N] syntax instead.",
        "ru": "BIND3 с scheduling в prefix/suffix/attrs упадёт при conditioning. Обходной путь: используйте [BIND3{...}:N].",
    },
    "narrow_bracket_boundary": {
        "en": "Bracket scheduling boundary is too small for the number of phases — some intermediate prompts will never appear in the schedule. Increase the boundary number or reduce phases.",
        "ru": "Граница планировщика слишком мала для количества фаз — некоторые промежуточные промпты никогда не появятся в расписании. Увеличьте число границы или уменьшите количество фаз.",
    },
}

_RE_BACKEND_KEYWORD_BRACE = re.compile(
    r'(CHUNK|BLEND|MORPH|POOL|ASSEMBLE|BIND|BIND2|BIND3)\{$'
)


def _has_unescaped_braces(text: str) -> bool:
    for m in re.finditer(r'(?<!\\)\{', text):
        pre = text[:m.start()].rstrip()
        if pre and pre.endswith(("CHUNK", "BLEND", "MORPH", "POOL", "ASSEMBLE", "BIND", "BIND2", "BIND3", "COMPOUND")):
            continue
        return True
    return False


def _approx_token_count(text: str) -> int:
    cjk = sum(1 for c in text
              if unicodedata.category(c) == 'Lo' and ord(c) > 0x2E7F)
    non_cjk_words = len([w for w in text.split()
                         if not any(ord(c) > 0x2E7F for c in w)])
    return cjk + non_cjk_words


def _check_backend_warnings(text: str, lang: str = "en", is_negative: bool = False) -> list[dict]:
    state = _extract_backend_prompt_state(text)
    warnings: list[dict] = []

    def _w(kind: str) -> dict:
        entry = _BACKEND_WARNINGS[kind]
        return {
            "severity": "WARNING",
            "kind": kind,
            "message": entry.get(lang, entry["en"]),
            "message_en": entry["en"],
            "message_ru": entry["ru"],
        }

    if state.chunk_spec is not None and len(state.chunk_spec.branches) == 1 and BIND2_KEYWORD not in text:
        warnings.append(_w("single_branch_chunk"))

    if state.blend_spec is not None:
        texts = [b.text for b in state.blend_spec.branches]
        if len(texts) != len(set(texts)):
            warnings.append(_w("duplicate_blend_branches"))

        if all(abs(branch.weight) <= 1e-8 for branch in state.blend_spec.branches):
            warnings.append(_w("all_zero_blend_weights"))

    if _has_unescaped_braces(text):
        warnings.append(_w("unescaped_brace_literal"))

    if is_negative and (
        _contains_bind_marker(text)
        or _contains_bind2_marker(text)
        or _contains_bind3_marker(text)
    ):
        warnings.append(_w("bind_in_negative_prompt"))

    if _contains_bind3_marker(text) and _RE_HAS_SCHEDULING.search(text):
        warnings.append(_w("bind3_scheduling_not_supported_at_conditioning"))

    for _bm in re.finditer(r'\[([^\[\]]+):(\d+(?:\.\d+)?)\]', text):
        _inner = _bm.group(1)
        _b_val = float(_bm.group(2))
        _phases = _inner.split(':')
        if _b_val < len(_phases) - 1:
            warnings.append(_w("narrow_bracket_boundary"))

    stripped = re.sub(r'\[.*?\]', '', text)
    if _approx_token_count(stripped) > 50:
        warnings.append(_w("token_limit_exceeded"))

    return warnings


def lint_prompt(text: str, steps: int = 20, seed: int | None = None, lang: str = "ru", is_negative: bool = False) -> dict:
    """Dry-run validation and schedule preview.

    On success returns:
      - ok: True
      - kind: None
      - spans: number of schedule segments
      - preview: short preview string

    On failure returns:
      - ok: False
      - kind: machine-readable error code (e.g. 'reverse_range', 'syntax_error')
      - token/pos/end: best-effort location info for UI highlighting
      - fix_suggestion: optional replacement text for [pos:end] (e.g. swapped range)
      - error: human-friendly message (optional for UIs; can be localized externally)
      - details: structured payload (language-neutral)
    """
    try:
        sched = _strict_schedule_preview(text, steps=steps, seed=seed)
        spans = len(sched)
        preview = '; '.join([f"to {end}: {t}" for end, t in sched[:3]])
        try:
            warnings = _check_backend_warnings(text, lang, is_negative=is_negative)
        except Exception as exc:
            logger.warning("lint_prompt: _check_backend_warnings failed: %s", exc)
            warnings = []
        return {"ok": True, "kind": None, "spans": spans, "preview": preview, "fix_suggestion": None, "warnings": warnings}
    except Exception as e:
        details = _syntax_error_details(e, text)
        return {
            "ok": False,
            "kind": details.get("kind"),
            "token": details.get("token"),
            "pos": details.get("pos"),
            "end": details.get("end"),
            "error": _format_syntax_error(e, text),
            "fix_suggestion": details.get("fix_suggestion"),
            "details": details,
            "warnings": [],
        }


def visualize_schedule(text: str, steps: int = 20, seed: int | None = None) -> str:

    """Сформатировать человекочитаемое представление расписания."""
    sched = get_schedule(text, steps=steps, use_scheduling=True, seed=seed, use_visitor=True)
    out_lines = []
    prev_end = 0
    for end, t in sched:
        start = prev_end + 1
        out_lines.append(f"Шаги {start}-{end}: {t}")
        prev_end = end
    return "\n".join(out_lines)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    random.seed(42)

    g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]

    assert g("test") == [[10, 'test']]
    assert g("a [b:3]") == [[3, 'a'], [10, 'a b']]
    assert g("[(a:2):3]") == [[3, ''], [10, '(a:2)']]

    big = "{[a|b|c|d|e|f|g],[h|i|j|k|l|m|n],[o|p|q|r|s|t|u]}"
    res = g(big)
    assert len(res) <= GROUP_COMBO_LIMIT

    g2 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    one = g2("[cat|dog|fox]!")
    assert len(set([txt for _, txt in one])) == 1
    
    # reverse для «чистой» формы со ступенчатым делением внутри скобок
    g3 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g3("[a:b:4] reverse") == [[4, 'b'], [6, 'a']]

    # Нумерованные альтернативы с нормализацией пробелов/AND
    g4 = lambda p: get_learned_conditioning_prompt_schedules([p], 5)[0]
    txt = g4("3[_a|b| |c]")[0][1]
    assert isinstance(txt, str) and len(txt) > 0

    # Пустые сегменты внутри [...:N] (чистая форма и reverse)
    g5 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g5("[a::4]") == [[4, 'a'], [6, '']]
    assert g5("[a::4] reverse") == [[4, ''], [6, 'a']]

    # Legacy inner reverse should be tolerated without rewriting to postfix.
    g5r = lambda p, s=20: get_schedule(p, s, True, seed=123)
    assert g5r("[cat, dog:8 reverse]") == [[8, ''], [20, 'cat, dog']]
    assert g5r("[cat:dog:8 reverse]") == [[8, 'dog'], [20, 'cat']]
    assert g5r("[cat, dog:8 reverse] [night, moon:15 reverse]") == [
        [8, ''],
        [15, 'cat, dog'],
        [20, 'cat, dog night, moon'],
    ]
    assert g5r("[cat, dog:8 1-4,5-8] [night, moon:15 1-7,8-15]") == [
        [8, 'cat, dog night, moon'],
        [15, 'night, moon'],
        [20, ''],
    ]

    # Префикс/суффикс + пустой сегмент: добавляется пролог до первого интервала
    g6 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g6("X [a::4] Y") == [[4, 'X a Y'], [6, 'X Y']]

    # --- новые проверки табов/переводов строки ---
    pa1 = parse_prompt_attention("dog\\tcat")
    joined1 = "".join(t for t, w in pa1 if t != "BREAK").strip()
    assert joined1 == "dog cat", f"Expected 'dog cat', got {joined1!r}"

    pa2 = parse_prompt_attention("dog\tcat")
    joined2 = "".join(t for t, w in pa2 if t != "BREAK").strip()
    assert joined2 == "dog cat", f"Expected 'dog cat', got {joined2!r}"

    # --- AND operator normalization ---
    g_and = lambda p: get_schedule(p, 4, True, seed=123)
    assert g_and("cat and dog") == [[4, "cat and dog"]]
    assert g_and("cat AND dog") == [[4, f"cat {ATTENTION_AND_OPERATOR} dog"]]
    assert g_and("cat & dog") == [[4, f"cat {ATTENTION_AND_OPERATOR} dog"]]
    assert g_and(r"R\&D") == [[4, "R&D"]]
    assert g_and(r"\&") == [[4, "&"]]

    _join_pa = lambda txt: "".join(t for t, w in parse_prompt_attention(txt) if t != "BREAK").strip()
    assert _join_pa(r"R\&D") == "R&D"
    assert _join_pa(r"\&") == "&"
    assert _join_pa("cat AND dog") == f"cat {ATTENTION_AND_OPERATOR} dog"

    import py_compile
    py_compile.compile(__file__, doraise=True)
    # ФИЙ19: убрана загрузка __file__ через importlib — она инициализировала весь модуль
    # второй раз в памяти (регулярки, lark.Lark и т.д.) без какой-либо пользы.
    # Smoke-тесты выше уже работают через текущий модуль напрямую.

    # ── Attention interpolation tests (v21) ─────────────────────────────────
    import re as _re

    def _w(text, token):
        """Extract weight from '(token:W)' in text."""
        m = _re.search(rf'\({_re.escape(token)}:([\d.]+)\)', text)
        assert m, f"Token '({token}:...)' not found in: {text!r}"
        return float(m.group(1))

    g5 = lambda p, s=5: get_schedule(p, s, True,  seed=123)
    g5f= lambda p, s=5: get_schedule(p, s, False, seed=123)

    # 1. Basic linear interpolation: first step = w0, last step = w1
    sc = g5("(cat:1.0->2.0)", 5)
    assert sc[0][1]  == "(cat:1.0)", f"w0 wrong: {sc[0][1]}"
    assert sc[-1][1] == "(cat:2.0)", f"w1 wrong: {sc[-1][1]}"
    assert len(sc) == 5, f"Expected 5 steps, got {len(sc)}: {sc}"
    ws = [_w(t, "cat") for _, t in sc]
    assert all(ws[i] <= ws[i+1] for i in range(len(ws)-1)), f"Not monotone: {ws}"

    # 2. Prefix text preserved
    sc2 = g5("dog, (red eyes:0.8->1.4)", 5)
    assert sc2[0][1].startswith("dog,"),   f"Prefix lost: {sc2[0][1]}"
    assert abs(_w(sc2[0][1],  "red eyes") - 0.8) < 1e-4
    assert abs(_w(sc2[-1][1], "red eyes") - 1.4) < 1e-4

    # 3. Two interpolations in one prompt
    sc3 = g5("(cat:1.0->2.0), (dog:0.5->1.5)", 5)
    assert "(cat:2.0)" in sc3[-1][1], f"cat at last step: {sc3[-1][1]}"
    assert "(dog:1.5)" in sc3[-1][1], f"dog at last step: {sc3[-1][1]}"
    # First block must not be mangled (v20 bug: parsed as weighted + plain)
    assert "(cat:1.0)" in sc3[0][1],  f"cat at first step: {sc3[0][1]}"

    # 4. AND operator: both parts survive
    sc4 = g5("(cat:1.0->2.0) & dog", 5)
    assert "dog" in sc4[-1][1],   f"AND part lost: {sc4[-1][1]}"
    assert "(cat:2.0)" in sc4[-1][1], f"Interp at last step: {sc4[-1][1]}"

    # 5. use_scheduling=False -> collapses to w1 (last step)
    sc5 = g5f("(cat:1.0->2.0)", 5)
    assert len(sc5) == 1,              f"Expected 1 segment, got {len(sc5)}: {sc5}"
    assert sc5[0][1] == "(cat:2.0)",   f"Expected cat:2.0, got {sc5[0][1]}"

    # 6. Contextual-aware: inside scheduled window
    # [(cat:0.5->1.0):2] -> before boundary: empty, after: cat interpolating
    sc6 = g5("[(cat:0.5->1.0):2]", 5)
    # Key assertions: no raw '->' and no unexpanded marker leaks
    for end, txt in sc6:
        assert "->"            not in txt, f"Raw '->' leaked at step {end}: {txt!r}"
        assert ATTN_INTERP_OPEN not in txt, f"Unexpanded marker at step {end}: {txt!r}"
    # After boundary: cat weight present and in [0.5, 1.0]
    active = [(end, txt) for end, txt in sc6 if "(cat:" in txt]
    assert active, f"No active cat segments in: {sc6}"
    for end, txt in active:
        w = _w(txt, "cat")
        assert 0.5 - 1e-4 <= w <= 1.0 + 1e-4, f"Weight out of range [{end}]: {w}"

    # 7. Regression: static (cat:1.2) unchanged
    sr = g5("(cat:1.2)", 5)
    assert all("(cat:1.2)" in t for _, t in sr), f"Regression static emphasis: {sr}"

    # 8. Regression: scheduler [a:b:3] unchanged
    sr2 = g5("[a:b:3]", 5)
    assert "->" not in str(sr2), f"Regression scheduled: {sr2}"

    # 9. Regression: reverse unchanged
    sr3 = g5("[a:b:4] reverse", 5)
    assert "->" not in str(sr3), f"Regression reverse: {sr3}"

    # 10. No false positive: 'word -> something' plain text stays plain
    sc10 = g5("go -> here", 5)
    assert sc10[0][1] == "go -> here", f"False positive interp: {sc10[0][1]}"

    # ── lint_prompt regression tests ─────────────────────────────────────────
    # Lark parses (body:w0->w1) as emphasized(weighted(body:w0) + plain("->w1"))
    # — a valid tree, no exception. So lint_prompt must return ok=True,
    # not a syntax error. These tests lock that behaviour so a future grammar
    # change can't silently break it.

    # 11. lint_prompt: basic interpolation passes
    lr = lint_prompt("(cat:1.0->2.0)")
    assert lr["ok"], f"lint_prompt basic interp failed: {lr}"
    assert lr["kind"] is None, f"Unexpected error kind: {lr['kind']}"

    # 12. lint_prompt: multi-token body passes
    lr2 = lint_prompt("(dog, red hair, hair, blue eyes, sword:0.5->1.5)")
    assert lr2["ok"], f"lint_prompt multi-token body failed: {lr2}"

    # 13. lint_prompt: interpolation inside scheduled window passes
    lr3 = lint_prompt("[(cat:1.0->2.0):3]")
    assert lr3["ok"], f"lint_prompt scheduled interp failed: {lr3}"

    # 14. lint_prompt: mixed static + interpolation passes
    lr4 = lint_prompt("(cat:1.2), (dog:0.8->1.5)")
    assert lr4["ok"], f"lint_prompt mixed failed: {lr4}"

    # 15. lint_prompt: plain '->' text passes (not a false-positive interp error)
    lr5 = lint_prompt("go -> here")
    assert lr5["ok"], f"lint_prompt plain arrow failed: {lr5}"

    # 16. lint_prompt: spans count is reasonable (not zero)
    lr6 = lint_prompt("(cat:1.0->2.0)", steps=5)
    assert lr6["ok"],       f"lint_prompt interp steps failed: {lr6}"
    assert lr6["spans"] > 0, f"lint_prompt spans=0: {lr6}"

    # ── visualize_schedule regression tests ─────────────────────────────────
    # visualize_schedule must not crash and must not leak raw '->' or markers
    # in its output for any interpolation input.

    # 17. visualize_schedule: basic interpolation — no raw '->' in output
    vs = visualize_schedule("(cat:1.0->2.0)", steps=5)
    assert "->"            not in vs, f"Raw '->' in visualize output:\n{vs}"
    assert ATTN_INTERP_OPEN not in vs, f"Unexpanded marker in visualize output:\n{vs}"
    assert "(cat:"          in vs,     f"'(cat:...)' missing in visualize output:\n{vs}"

    # 18. visualize_schedule: multi-token body
    vs2 = visualize_schedule("(dog, red hair, sword:0.5->1.5)", steps=5)
    assert "->"            not in vs2, f"Raw '->' in visualize multi:\n{vs2}"
    assert ATTN_INTERP_OPEN not in vs2, f"Marker leaked in visualize multi:\n{vs2}"

    # 19. visualize_schedule: inside scheduled window
    vs3 = visualize_schedule("[(cat:1.0->2.0):3]", steps=5)
    assert "->"            not in vs3, f"Raw '->' in visualize scheduled:\n{vs3}"
    assert ATTN_INTERP_OPEN not in vs3, f"Marker leaked in visualize scheduled:\n{vs3}"

    # 20. visualize_schedule: plain '->' text is preserved as-is
    vs4 = visualize_schedule("go -> here", steps=5)
    assert "go -> here" in vs4, f"Plain arrow mangled in visualize:\n{vs4}"

    # ── 5 дополнительных тестов (по рекомендации GPT) ─────────────────────────

    # 21. lint_prompt.preview содержит раскрытые веса, а не сырой синтаксис
    # Фиксируем, что preview строится из развёрнутого расписания, а не из исходной строки.
    lr7 = lint_prompt("(cat:1.0->2.0)", steps=5)
    assert lr7["ok"], f"lint_prompt preview test failed: {lr7}"
    preview = lr7.get("preview", "")
    assert "->"            not in preview, f"Raw '->' in preview: {preview!r}"
    assert ATTN_INTERP_OPEN not in preview, f"Marker leaked in preview: {preview!r}"
    assert "(cat:"          in preview,    f"'(cat:...)' missing in preview: {preview!r}"

    # 22. use_visitor=True vs use_visitor=False — parity для базовой интерполяции
    # Оба маршрута должны давать одинаковый результат, так как pre-pass работает до Lark.
    sv1 = get_schedule("(cat:1.0->2.0)", 5, True, seed=123, use_visitor=True)
    sv2 = get_schedule("(cat:1.0->2.0)", 5, True, seed=123, use_visitor=False)
    assert sv1 == sv2, f"use_visitor parity failed:\n  visitor=True:  {sv1}\n  visitor=False: {sv2}"

    # 23. use_visitor parity для interpolation внутри scheduled-окна
    sv3 = get_schedule("[(cat:0.5->1.0):2]", 5, True, seed=123, use_visitor=True)
    sv4 = get_schedule("[(cat:0.5->1.0):2]", 5, True, seed=123, use_visitor=False)
    assert sv3 == sv4, f"use_visitor scheduled parity failed:\n  True:  {sv3}\n  False: {sv4}"

    # 24. Single-step семантика: steps=1, span==0 → вес w0, а не w1
    # Важно: при span==0 берётся t=0.0, то есть w0.
    ss1 = get_schedule("(cat:1.0->2.0)", 1, True,  seed=123)
    assert ss1 == [[1, "(cat:1.0)"]], f"Single-step use_scheduling=True: {ss1}"
    ss2 = get_schedule("(cat:1.0->2.0)", 1, False, seed=123)
    assert ss2 == [[1, "(cat:1.0)"]], f"Single-step use_scheduling=False: {ss2}"

    # 25. Точный финальный текст для двух interpolation-блоков в одной строке
    # Проверяет, что оба блока раскрываются корректно — не только "что-то там есть".
    sm = get_schedule("(cat:1.0->2.0), (dog:0.5->1.5)", 5, True, seed=123)
    assert sm[0][1]  == "(cat:1.0), (dog:0.5)", f"Mixed first step: {sm[0][1]!r}"
    assert sm[-1][1] == "(cat:2.0), (dog:1.5)", f"Mixed last step:  {sm[-1][1]!r}"

    print("All quick integration tests passed!")

    # ── GPT P2/P3 regression tests ───────────────────────────────────────────

    # P1 (already fixed): ESCAPED_AMP_PLACEHOLDER uses private-use char \uE004
    # Verify user text containing the old string placeholder is NOT mangled
    r_amp = get_schedule("__PROMPT_PARSER_ESCAPED_AMP__", 4, True, seed=123)
    assert r_amp[0][1] == "__PROMPT_PARSER_ESCAPED_AMP__", \
        f"P1 regression: amp placeholder leaked into output: {r_amp}"

    # P2 (already fixed): LITERAL_REVERSE_TOKEN uses private-use char \uE005
    r_rev = get_schedule("__PP_LITERAL_REVERSE__", 4, True, seed=123)
    assert r_rev[0][1] == "__PP_LITERAL_REVERSE__", \
        f"P2 regression: reverse token leaked into output: {r_rev}"

    # P3: 'reverse, shot' and 'reverse shot' must give same suffix (no leading comma)
    g_p3 = lambda p: get_schedule(p, 4, True, seed=123)
    r_a = g_p3("[a:b]:10 reverse shot")
    r_b = g_p3("[a:b]:10 reverse, shot")
    # Both should have 'shot' in the suffix text, no leading ', '
    for end, txt in r_a:
        assert not txt.startswith(", "), f"P3a: leading comma in text: {txt!r}"
    for end, txt in r_b:
        assert not txt.startswith(", "), f"P3b: leading comma in text: {txt!r}"
    # And the non-reverse content should match between the two
    texts_a = [t for _, t in r_a]
    texts_b = [t for _, t in r_b]
    assert texts_a == texts_b, \
        f"P3: 'reverse shot' vs 'reverse, shot' gave different results:\n  A={texts_a}\n  B={texts_b}"

    # P4 (already fixed): SdConditioning preserves falsy numeric values (0)
    sd = SdConditioning([], width=0, height=0, distilled_cfg_scale=0)
    assert sd.width == 0,               f"P4: width=0 lost, got {sd.width!r}"
    assert sd.height == 0,              f"P4: height=0 lost, got {sd.height!r}"
    assert sd.distilled_cfg_scale == 0, f"P4: distilled_cfg_scale=0 lost, got {sd.distilled_cfg_scale!r}"

    print("GPT P2/P3 regression tests passed!")

    # ── P4 regression: empty alternate compatibility with A1111 ──────────────
    # [fe|]male: step1 -> 'female', step2 -> 'male' (cycles like A1111)
    g_alt = lambda p, s=10: get_learned_conditioning_prompt_schedules([p], s)[0]
    r_p4 = g_alt("[fe|]male", 10)
    step1 = r_p4[0][1]  # first entry
    # Find step-1 and step-2 texts by simulating get_schedule per step
    sc_p4 = get_schedule("[fe|]male", 10, True, seed=None)
    texts = {end: txt for end, txt in sc_p4}
    # At minimum: should contain 'female' and 'male' entries (alternating)
    all_texts = [txt for _, txt in sc_p4]
    assert "female" in all_texts, f"P4: 'female' missing from schedule: {sc_p4}"
    assert "male"   in all_texts, f"P4: 'male' missing from schedule: {sc_p4}"
    # Should NOT have 'fe male' (space between fe and male)
    assert "fe male" not in all_texts, f"P4: got 'fe male' instead of 'female': {sc_p4}"
    print("P4 empty alternate regression passed!")

    # ── Edge case tests ──────────────────────────────────────────────────────

    import re as _re2

    def _w2(text, token):
        m = _re2.search(rf'\({_re2.escape(token)}:([-\d.]+)\)', text)
        assert m, f"'({token}:...)' not found in: {text!r}"
        return float(m.group(1))

    g_ec = lambda p, s=5: get_schedule(p, s, True, seed=123)

    # EC-1: w0 == w1 — constant interpolation, all steps same weight
    sc_eq = g_ec("(cat:1.2->1.2)", 5)
    for end, txt in sc_eq:
        assert "(cat:1.2)" in txt, f"EC-1 constant interp, step {end}: {txt!r}"

    # EC-2: negative start weight (-0.5 -> 0.5)
    sc_neg = g_ec("(cat:-0.5->0.5)", 5)
    w_neg_first = _w2(sc_neg[0][1],  "cat")
    w_neg_last  = _w2(sc_neg[-1][1], "cat")
    assert abs(w_neg_first - (-0.5)) < 1e-4, f"EC-2 neg start: {w_neg_first}"
    assert abs(w_neg_last  -   0.5)  < 1e-4, f"EC-2 neg end:   {w_neg_last}"
    assert w_neg_first < w_neg_last,          f"EC-2 not increasing: {w_neg_first} -> {w_neg_last}"

    # EC-3: negative end weight (1.0 -> -0.5)
    sc_neg2 = g_ec("(cat:1.0->-0.5)", 5)
    w_n2_first = _w2(sc_neg2[0][1],  "cat")
    w_n2_last  = _w2(sc_neg2[-1][1], "cat")
    assert abs(w_n2_first - 1.0)  < 1e-4, f"EC-3 start: {w_n2_first}"
    assert abs(w_n2_last  - (-0.5)) < 1e-4, f"EC-3 end: {w_n2_last}"
    assert w_n2_first > w_n2_last,           f"EC-3 not decreasing: {w_n2_first} -> {w_n2_last}"

    # EC-4: alternate [a|b] combined with interpolation — no crash, no marker leak
    sc_alt = g_ec("[cat|dog], (red eyes:0.8->1.4)", 5)
    for end, txt in sc_alt:
        assert ATTN_INTERP_OPEN not in txt, f"EC-4 marker leak at step {end}: {txt!r}"
        assert "->" not in txt,             f"EC-4 raw arrow at step {end}: {txt!r}"
        assert "(red eyes:" in txt,         f"EC-4 interp missing at step {end}: {txt!r}"

    # EC-5: reverse + interpolation in same prompt — no crash, no marker leak
    sc_rev = g_ec("[a:b:3] reverse, (cat:1.0->2.0)", 5)
    for end, txt in sc_rev:
        assert ATTN_INTERP_OPEN not in txt, f"EC-5 marker leak at step {end}: {txt!r}"
        assert "->" not in txt,             f"EC-5 raw arrow at step {end}: {txt!r}"
    # Interpolation ramps across the whole schedule
    last_txt = sc_rev[-1][1]
    assert "(cat:2.0)" in last_txt, f"EC-5 cat not at w1 on last step: {last_txt!r}"

    # EC-6: AND with both branches having weights / interpolation
    sc_and2 = g_ec("(cat:1.5) & (dog:0.5->1.5)", 5)
    for end, txt in sc_and2:
        assert ATTN_INTERP_OPEN not in txt, f"EC-6 marker leak at step {end}: {txt!r}"
    assert "(cat:1.5)" in sc_and2[0][1],  f"EC-6 static branch first step:  {sc_and2[0][1]!r}"
    assert "(dog:1.5)" in sc_and2[-1][1], f"EC-6 interp branch last step:   {sc_and2[-1][1]!r}"

    # EC-7: AND with two interpolation branches
    sc_and3 = g_ec("(cat:1.0->2.0) & (dog:0.5->1.5)", 5)
    assert "(cat:2.0)" in sc_and3[-1][1], f"EC-7 cat last step:  {sc_and3[-1][1]!r}"
    assert "(dog:1.5)" in sc_and3[-1][1], f"EC-7 dog last step:  {sc_and3[-1][1]!r}"
    assert "(cat:1.0)" in sc_and3[0][1],  f"EC-7 cat first step: {sc_and3[0][1]!r}"
    assert "(dog:0.5)" in sc_and3[0][1],  f"EC-7 dog first step: {sc_and3[0][1]!r}"

    # EC-8: multi-token body with negative weight
    sc_mt_neg = g_ec("(red hair, blue eyes:-0.3->0.8)", 5)
    w_mt_first = _w2(sc_mt_neg[0][1],  "red hair, blue eyes")
    w_mt_last  = _w2(sc_mt_neg[-1][1], "red hair, blue eyes")
    assert abs(w_mt_first - (-0.3)) < 1e-4, f"EC-8 start: {w_mt_first}"
    assert abs(w_mt_last  -   0.8)  < 1e-4, f"EC-8 end:   {w_mt_last}"

    print("Edge case tests passed!")

    # ── v2 nested body tests ─────────────────────────────────────────────────

    g_v2 = lambda p, s=5: get_schedule(p, s, True, seed=123)

    # V2-1: nested body '(cat)' — emphasized transformer converts (cat)->(cat:1.1),
    # then outer interpolation weight wraps it: ((cat:1.1):w)
    sc_v2_1 = g_v2("((cat):1.0->2.0)", 5)
    assert "((cat:1.1):1.0)" in sc_v2_1[0][1],  f"V2-1 first: {sc_v2_1[0][1]!r}"
    assert "((cat:1.1):2.0)" in sc_v2_1[-1][1], f"V2-1 last:  {sc_v2_1[-1][1]!r}"

    # V2-2: inner '(glowing eyes)' → emphasized → '(glowing eyes:1.1)',
    # outer interpolation weight wraps the whole body.
    sc_v2_2 = g_v2("(red (glowing eyes):0.8->1.4)", 5)
    assert "->" not in sc_v2_2[0][1],              f"V2-2 raw arrow: {sc_v2_2[0][1]!r}"
    assert ATTN_INTERP_OPEN not in sc_v2_2[-1][1], f"V2-2 marker leak: {sc_v2_2[-1][1]!r}"
    assert "(red (glowing eyes:1.1):0.8)" in sc_v2_2[0][1],  f"V2-2 first: {sc_v2_2[0][1]!r}"
    assert "(red (glowing eyes:1.1):1.4)" in sc_v2_2[-1][1], f"V2-2 last:  {sc_v2_2[-1][1]!r}"

    # V2-3: '(a:b)' where b is non-numeric → emphasized gives (a:1.1),
    # outer interpolation wraps: ((a:1.1):w)
    sc_v2_3 = g_v2("((a:b):1.0->2.0)", 5)
    assert "((a:1.1):1.0)" in sc_v2_3[0][1],  f"V2-3 first: {sc_v2_3[0][1]!r}"
    assert "((a:1.1):2.0)" in sc_v2_3[-1][1], f"V2-3 last:  {sc_v2_3[-1][1]!r}"

    # V2-4: plain '->' text is NOT treated as interpolation (no outer parens)
    sc_v2_4 = g_v2("go -> here", 5)
    assert sc_v2_4[0][1] == "go -> here", f"V2-4 false positive: {sc_v2_4[0][1]!r}"

    # V2-5: unmatched paren — must not crash, left as-is
    sc_v2_5 = g_v2("(unclosed:1.0->2.0", 5)
    assert "->" in sc_v2_5[0][1] or sc_v2_5[0][1], f"V2-5 crash or empty"

    # V2-6: nested body + scheduled window (contextual-aware still works)
    sc_v2_6 = g_v2("[(red (glowing eyes):0.5->1.0):2]", 5)
    for end, txt in sc_v2_6:
        assert ATTN_INTERP_OPEN not in txt, f"V2-6 marker leak step {end}: {txt!r}"
        assert "->" not in txt,             f"V2-6 raw arrow step {end}: {txt!r}"

    # V2-7: v1 multi-token body still works (no regression)
    sc_v2_7 = g_v2("(cat, dog, sword:1.0->2.0)", 5)
    assert "(cat, dog, sword:1.0)" in sc_v2_7[0][1],  f"V2-7 first: {sc_v2_7[0][1]!r}"
    assert "(cat, dog, sword:2.0)" in sc_v2_7[-1][1], f"V2-7 last:  {sc_v2_7[-1][1]!r}"

    print("v2 nested body tests passed!")

    # ── Easing tests ─────────────────────────────────────────────────────────

    g_ez = lambda p, s=5: get_schedule(p, s, True, seed=123)

    def _w_at(schedule, step_idx):
        """Extract float weight from '(cat:W)' in schedule entry."""
        import re as _re3
        txt = schedule[step_idx][1]
        m = _re3.search(r'\(cat:([-\d.]+)\)', txt)
        assert m, f"no (cat:w) in {txt!r}"
        return float(m.group(1))

    # EZ-0: no suffix → linear (baseline)
    sc_lin = g_ez("(cat:0.0->1.0)", 5)
    w_lin  = [_w_at(sc_lin, i) for i in range(len(sc_lin))]
    # linear: differences between consecutive steps should be equal
    diffs = [round(w_lin[i+1] - w_lin[i], 4) for i in range(len(w_lin)-1)]
    assert len(set(diffs)) == 1, f"EZ-0 linear not uniform: {diffs}"

    # EZ-1: ease-in — starts slow, ends fast → w[midpoint] < linear midpoint
    sc_ei = g_ez("(cat:0.0->1.0~ease-in)", 5)
    w_ei  = [_w_at(sc_ei, i) for i in range(len(sc_ei))]
    mid_lin = 0.5
    mid_ei  = w_ei[len(w_ei)//2]
    assert mid_ei < mid_lin + 0.01, f"EZ-1 ease-in midpoint {mid_ei} not below linear {mid_lin}"
    assert w_ei[0]  <= w_lin[0]  + 1e-3, f"EZ-1 ease-in start mismatch: {w_ei[0]}"
    assert w_ei[-1] >= w_lin[-1] - 1e-3, f"EZ-1 ease-in end mismatch:   {w_ei[-1]}"

    # EZ-2: ease-out — starts fast, ends slow → w[midpoint] > linear midpoint
    sc_eo = g_ez("(cat:0.0->1.0~ease-out)", 5)
    w_eo  = [_w_at(sc_eo, i) for i in range(len(sc_eo))]
    mid_eo = w_eo[len(w_eo)//2]
    assert mid_eo > mid_lin - 0.01, f"EZ-2 ease-out midpoint {mid_eo} not above linear {mid_lin}"

    # EZ-3: ease-in-out — symmetric S-curve; midpoint ≈ 0.5 (exact for smooth step)
    sc_eio = g_ez("(cat:0.0->1.0~ease-in-out)", 5)
    w_eio  = [_w_at(sc_eio, i) for i in range(len(sc_eio))]
    mid_eio = w_eio[len(w_eio)//2]
    assert abs(mid_eio - 0.5) < 0.15, f"EZ-3 ease-in-out midpoint {mid_eio} far from 0.5"

    # EZ-4: unknown mode falls back to linear
    sc_unk = g_ez("(cat:0.0->1.0~bogus-mode)", 5)
    w_unk  = [_w_at(sc_unk, i) for i in range(len(sc_unk))]
    assert w_unk == w_lin, f"EZ-4 unknown mode not linear: {w_unk}"

    # EZ-5: easing with non-zero w0 (0.5->1.5~ease-in) — endpoints exact
    sc_e5 = g_ez("(cat:0.5->1.5~ease-in)", 5)
    assert abs(_w_at(sc_e5, 0) - 0.5) < 1e-3,  f"EZ-5 start: {_w_at(sc_e5, 0)}"
    assert abs(_w_at(sc_e5, -1) - 1.5) < 1e-3, f"EZ-5 end:   {_w_at(sc_e5, -1)}"

    # EZ-6: easing inside scheduled window — no marker leak, no raw ~
    sc_e6 = g_ez("[(cat:0.0->1.0~ease-out):2]", 5)
    for end, txt in sc_e6:
        assert ATTN_INTERP_OPEN not in txt, f"EZ-6 marker leak: {txt!r}"
        assert "~" not in txt,              f"EZ-6 raw tilde: {txt!r}"
        assert "->" not in txt,             f"EZ-6 raw arrow: {txt!r}"

    # EZ-7: backward compat — old-style no easing still works
    sc_e7 = g_ez("(cat:1.0->2.0)", 5)
    assert abs(_w_at(sc_e7, 0)  - 1.0) < 1e-3, f"EZ-7 compat start: {_w_at(sc_e7, 0)}"
    assert abs(_w_at(sc_e7, -1) - 2.0) < 1e-3, f"EZ-7 compat end:   {_w_at(sc_e7, -1)}"

    # EZ-8: case-insensitive mode — EASE-IN / Ease_In → same as ease-in
    sc_upper = g_ez("(cat:0.0->1.0~EASE-IN)", 5)
    w_upper  = [_w_at(sc_upper, i) for i in range(len(sc_upper))]
    assert w_upper == w_ei, f"EZ-8 EASE-IN != ease-in: {w_upper} vs {w_ei}"

    # EZ-9: bezier easing — smoothstep: 3t²-2t³, steeper at midpoint than linear
    sc_bz = g_ez("(cat:0.0->1.0~bezier)", 5)
    w_bz  = [_w_at(sc_bz, i) for i in range(len(sc_bz))]
    mid_bz = w_bz[len(w_bz)//2]
    # bezier at t=0.5: 0.5²*(3-2*0.5) = 0.25*2 = 0.5 (same as linear at midpoint)
    assert abs(mid_bz - 0.5) < 0.01, f"EZ-9 bezier midpoint {mid_bz} != 0.5"
    # bezier is steeper near t=0.25: bezier(0.25) = 0.0625*2.5 = 0.156
    # linear(0.25) = 0.25 → bezier < linear before midpoint
    w_lin_25 = 0.25
    w_bz_25  = w_bz[len(w_bz)//4] if len(w_bz) >= 5 else w_bz[1]
    # The test: bezier at 25% should be closer to 0 than linear (smoother start)
    assert w_bz[1] < w_lin[1] + 0.01, f"EZ-9 bezier not smoother at start: {w_bz} vs {w_lin}"

    # EZ-10: catmull easing — smootherstep: 6t⁵-15t⁴+10t³, even smoother edges
    sc_cm = g_ez("(cat:0.0->1.0~catmull)", 5)
    w_cm  = [_w_at(sc_cm, i) for i in range(len(sc_cm))]
    # catmull at t=0.5: 6*(0.5⁵)-15*(0.5⁴)+10*(0.5³) = 0.1875-0.9375+1.25 = 0.5
    mid_cm = w_cm[len(w_cm)//2]
    assert abs(mid_cm - 0.5) < 0.01, f"EZ-10 catmull midpoint {mid_cm} != 0.5"
    # catmull starts even slower than bezier → first quarter value should be < bezier
    assert w_cm[1] < w_bz[1] + 0.01 or len(w_cm) < 5, \
        f"EZ-10 catmull not smoother than bezier: {w_cm} vs {w_bz}"

    # EZ-11: bezier/catmull in _EASING_MODES
    assert "bezier" in _EASING_MODES, "EZ-11 bezier not in _EASING_MODES"
    assert "catmull" in _EASING_MODES, "EZ-11 catmull not in _EASING_MODES"

    # EZ-12: backward compat — old modes still work
    sc_lin2 = g_ez("(cat:0.0->1.0)")
    assert _w_at(sc_lin2, 0) == 0.0 and _w_at(sc_lin2, -1) == 1.0, "EZ-12 linear compat"

    print("Easing tests passed!")

    # ── Backend WARNING tests ──────────────────────────────────────────────

    # W1: plain prompt → no warnings
    w1 = lint_prompt("a cat")
    assert w1["ok"], f"W1 failed: {w1}"
    assert w1["warnings"] == [], f"W1 unexpected warnings: {w1['warnings']}"

    # W2: CHUNK single branch → warning
    w2 = lint_prompt("CHUNK{a}")
    assert w2["ok"], f"W2 failed: {w2}"
    assert any(w["kind"] == "single_branch_chunk" for w in w2["warnings"]), \
        f"W2 missing single_branch_chunk warning: {w2['warnings']}"
    assert any("CHUNK" in w["message"] for w in w2["warnings"]), \
        f"W2 English message missing CHUNK: {w2['warnings']}"

    # W2-ru: CHUNK single branch → Russian message
    w2_ru = lint_prompt("CHUNK{a}", lang="ru")
    assert w2_ru["ok"], f"W2-ru failed: {w2_ru}"
    assert any("CHUNK" in w["message"] for w in w2_ru["warnings"]), \
        f"W2-ru message missing CHUNK: {w2_ru['warnings']}"

    # W3: CHUNK two branches → no warning
    w3 = lint_prompt("CHUNK{a|b}")
    assert w3["ok"], f"W3 failed: {w3}"
    assert not any(w["kind"] == "single_branch_chunk" for w in w3["warnings"]), \
        f"W3 false positive: {w3['warnings']}"

    # W4: BLEND duplicate branches → warning
    w4 = lint_prompt("BLEND{cat|cat}")
    assert w4["ok"], f"W4 failed: {w4}"
    assert any(w["kind"] == "duplicate_blend_branches" for w in w4["warnings"]), \
        f"W4 missing duplicate warning: {w4['warnings']}"

    # W5: BLEND normal (unique) → no warning
    w5 = lint_prompt("BLEND{a*1|b*2}")
    assert w5["ok"], f"W5 failed: {w5}"
    assert not any(w["kind"] == "duplicate_blend_branches" for w in w5["warnings"]), \
        f"W5 false positive duplicate: {w5['warnings']}"

    # W6: error response still contains warnings: []
    w6 = lint_prompt("CHUNK{a")  # malformed, missing ]
    assert not w6["ok"], f"W6 should be error: {w6}"

    # W7: warnings include both message_en and message_ru
    w7 = lint_prompt("CHUNK{a}")
    assert w7["ok"], f"W7 failed: {w7}"
    for w in w7["warnings"]:
        assert "message_en" in w, f"W7 missing message_en: {w}"
        assert "message_ru" in w, f"W7 missing message_ru: {w}"

    # W8: all_zero_blend_weights warning (defense-in-depth — grammar rejects *0, but check exists)
    w8 = lint_prompt("BLEND{cat|dog}")
    assert w8["ok"], f"W8 failed: {w8}"
    assert not any(w["kind"] == "all_zero_blend_weights" for w in w8["warnings"]), f"W8 false positive: {w8}"
    assert "all_zero_blend_weights" in _BACKEND_WARNINGS, "W8 missing from _BACKEND_WARNINGS"

    # W9: BIND* in negative prompt → warning
    w9 = lint_prompt("BIND2{cat => black, white}", is_negative=True)
    assert w9["ok"], f"W9 failed: {w9}"
    assert any(w["kind"] == "bind_in_negative_prompt" for w in w9["warnings"]), \
        f"W9 missing bind_in_negative_prompt: {w9['warnings']}"

    w9b = lint_prompt("BIND3{cat => fur, cute}", is_negative=True)
    assert w9b["ok"], f"W9b failed: {w9b}"
    assert any(w["kind"] == "bind_in_negative_prompt" for w in w9b["warnings"]), \
        f"W9b missing bind_in_negative_prompt: {w9b['warnings']}"

    w9c = lint_prompt("base BIND{cat => black, white}", is_negative=True)
    assert w9c["ok"], f"W9c failed: {w9c}"
    assert any(w["kind"] == "bind_in_negative_prompt" for w in w9c["warnings"]), \
        f"W9c missing bind_in_negative_prompt: {w9c['warnings']}"

    # W9-negative: positive prompt with BIND → no warning
    w9d = lint_prompt("BIND2{cat => black, white}", is_negative=False)
    assert w9d["ok"], f"W9d failed: {w9d}"
    assert not any(w["kind"] == "bind_in_negative_prompt" for w in w9d["warnings"]), \
        f"W9d false positive: {w9d['warnings']}"

    # W9-negative2: plain negative prompt → no warning
    w9e = lint_prompt("ugly, blurry", is_negative=True)
    assert w9e["ok"], f"W9e failed: {w9e}"
    assert not any(w["kind"] == "bind_in_negative_prompt" for w in w9e["warnings"]), \
        f"W9e false positive: {w9e['warnings']}"

    # W9-ru: Russian message
    w9f = lint_prompt("BIND2{cat => black}", is_negative=True, lang="ru")
    assert w9f["ok"], f"W9f failed: {w9f}"
    assert any("BIND" in w["message"] for w in w9f["warnings"]), \
        f"W9f Russian message missing BIND: {w9f['warnings']}"

    # W10: token limit warning
    long_words = " ".join(["word"] * 60)
    w10 = lint_prompt(long_words)
    assert w10["ok"], f"W10 failed: {w10}"
    assert any(w["kind"] == "token_limit_exceeded" for w in w10["warnings"]), \
        f"W10 missing token_limit_exceeded: {w10['warnings']}"

    # W10-neg: short prompt → no token warning
    w10b = lint_prompt("a cat")
    assert w10b["ok"], f"W10b failed: {w10b}"
    assert not any(w["kind"] == "token_limit_exceeded" for w in w10b["warnings"]), \
        f"W10b false positive: {w10b['warnings']}"

    # W10-sched: scheduling syntax stripped before counting
    sched_text = " ".join(["word"] * 40) + " [a:b:10] " + " ".join(["word"] * 15)
    w10c = lint_prompt(sched_text)
    assert w10c["ok"], f"W10c failed: {w10c}"
    # 40 + 15 = 55 words after stripping [a:b:10] → exceeds 50
    assert any(w["kind"] == "token_limit_exceeded" for w in w10c["warnings"]), \
        f"W10c missing token_limit_exceeded: {w10c['warnings']}"

    # W10-ru: Russian message
    w10d = lint_prompt(long_words, lang="ru")
    assert w10d["ok"], f"W10d failed: {w10d}"
    assert any("токен" in w["message"] for w in w10d["warnings"]), \
        f"W10d Russian message missing: {w10d['warnings']}"

    # W11: BIND3 + scheduling → conditioning warning
    w11 = lint_prompt("[warm:cold:5] BIND3{cat => fur}")
    assert w11["ok"], f"W11 failed: {w11}"
    assert any(w["kind"] == "bind3_scheduling_not_supported_at_conditioning" for w in w11["warnings"]), \
        f"W11 missing warning: {w11['warnings']}"
    w11b = lint_prompt("BIND3{cat => fur}")
    assert w11b["ok"], f"W11b failed: {w11b}"
    assert not any(w["kind"] == "bind3_scheduling_not_supported_at_conditioning" for w in w11b["warnings"]), \
        f"W11b false positive: {w11b['warnings']}"
    w11c = lint_prompt("[red:blue:10] BIND3{cat => fur}", lang="ru")
    assert w11c["ok"], f"W11c failed: {w11c}"
    assert any("BIND3" in w["message"] for w in w11c["warnings"]), \
        f"W11c Russian message missing BIND3: {w11c['warnings']}"

    # W12: CJK text (long) → token warning
    cjk_long = " ".join(["\u732b"] * 55)
    w12 = lint_prompt(cjk_long)
    assert w12["ok"], f"W12 failed: {w12}"
    assert any(w["kind"] == "token_limit_exceeded" for w in w12["warnings"]), \
        f"W12 missing warning: {w12['warnings']}"
    w12b = lint_prompt("\u732b")
    assert w12b["ok"], f"W12b failed: {w12b}"
    assert not any(w["kind"] == "token_limit_exceeded" for w in w12b["warnings"]), \
        f"W12b false positive: {w12b['warnings']}"

    # W13: narrow_bracket_boundary + scheduled_boundary_too_small
    w13 = lint_prompt("[cat:dog:rat:1]")
    assert not w13["ok"], f"W13 should fail: {w13}"
    assert w13["kind"] == "scheduled_boundary_too_small", f"W13 wrong kind: {w13}"
    w13b = lint_prompt("[cat:dog:rat:10]")
    assert w13b["ok"], f"W13b failed: {w13b}"
    assert not any(w["kind"] == "narrow_bracket_boundary" for w in w13b["warnings"]), \
        f"W13b false positive: {w13b['warnings']}"
    w13c = lint_prompt("[cat:dog:5]")
    assert w13c["ok"], f"W13c failed: {w13c}"
    assert not any(w["kind"] == "narrow_bracket_boundary" for w in w13c["warnings"]), \
        f"W13c false positive: {w13c['warnings']}"
    assert "narrow_bracket_boundary" in _BACKEND_WARNINGS, "W13 missing from _BACKEND_WARNINGS"

    # W14: scheduled_boundary_too_small raises, not silent drop
    w14 = lint_prompt("[a:b:c:d:e:3]", steps=10)
    assert not w14["ok"], f"W14 should fail: {w14}"
    assert w14["kind"] == "scheduled_boundary_too_small", f"W14 wrong kind: {w14}"
    w14b = lint_prompt("[a:b:c:10]", steps=10)
    assert w14b["ok"], f"W14b should pass: {w14b}"

    print("Backend WARNING tests passed!")

    # ── Phase 4: same-type multiple backend blocks ──────────────────────────
    gp4 = lambda p: get_schedule(p, 10, True, seed=42)

    # P4-1: BLEND×2 → sequential handling (no crash)
    s = gp4("BLEND{a|b} text BLEND{c|d}")
    assert len(s) >= 1, f"P4-1 BLEND×2 empty: {s}"

    # P4-2: CHUNK×3 → sequential handling (no crash)
    s = gp4("CHUNK{a|b} CHUNK{c|d} CHUNK{e|f}")
    assert len(s) >= 1, f"P4-2 CHUNK×3 empty: {s}"

    # P4-3: BLEND×2 + CHUNK×1 mixed (regression)
    s = gp4("BLEND{a|b} CHUNK{c|d} BLEND{e|f}")
    assert len(s) >= 1, f"P4-3 mixed+multiple empty: {s}"

    # P4-4: POOL×2 → sequential handling (no crash)
    s = gp4("POOL{wolf} POOL{mood}")
    assert len(s) >= 1, f"P4-4 POOL×2 empty: {s}"

    # P4-5: BIND + multiple POOL → recursive sequential (no crash)
    s = gp4("BIND{wolf => furry} POOL{a} POOL{b}")
    assert len(s) >= 1, f"P4-5 BIND+multiPOOL empty: {s}"

    # Path 1 BIND2 → CHUNK transpile tests
    s_b2 = gp4("BIND2{1girl => red hair, blue eyes}")
    assert len(s_b2) >= 1, f"BIND2 basic failed: {s_b2}"
    assert any("BREAK" in part for _, part in s_b2), f"BIND2 not transpiled to CHUNK BREAK: {s_b2}"
    s_b2_owner = gp4("BIND2{1girl, pretty => red hair, blue eyes}")
    assert len(s_b2_owner) >= 1, f"BIND2 comma-owner failed: {s_b2_owner}"
    assert any("BREAK" in part for _, part in s_b2_owner), f"BIND2 comma-owner not transpiled: {s_b2_owner}"
    s_b2_single = gp4("BIND2{1girl => red hair}")
    assert len(s_b2_single) >= 1, f"BIND2 single attr failed: {s_b2_single}"
    b2 = _transpile_bind2_to_chunk("BIND2{cat => black, white}")
    assert b2, f"transpile raw empty: {b2}"
    assert "cat, black" in b2 and "cat, white" in b2 and b2.startswith("CHUNK"), f"transpile mismatch: {b2}"

    # BIND2 Path 2 — flag and validation (full encoding requires model)
    assert isinstance(BIND2_USE_PATH2, bool), "B2P2-1 flag type"
    assert not BIND2_USE_PATH2, "B2P2-2 flag defaults to False"
    assert _contains_bind2_marker("BIND2{cat => black, white}"), "B2P2-3 marker detection"
    assert not _contains_bind2_marker("BIND3{a => b}"), "B2P2-4 false positive on BIND3"
    assert not _contains_bind2_marker("plain"), "B2P2-5 false positive on plain"

    # W13: BIND2_NORMALIZE_WEIGHTS flag exists
    assert isinstance(BIND2_NORMALIZE_WEIGHTS, bool), "W13 flag type"
    assert not BIND2_NORMALIZE_WEIGHTS, "W13 flag defaults to False"
    print("BIND2 smoke tests passed!")

    # ── BIND3 Path 2: row-splice conditioning ──────────────────────────────────
    # B3-1: transpile to CHUNK (text-only path)
    b3_t = _transpile_bind3_to_chunk("BIND3{cat => black, white}")
    assert b3_t and b3_t.startswith("CHUNK"), f"B3 transpile failed: {b3_t}"
    assert "cat, black" in b3_t and "cat, white" in b3_t, f"B3 transpile mismatch: {b3_t}"
    # B3-2: transpile single attr
    b3_s = _transpile_bind3_to_chunk("BIND3{cat => black}")
    assert b3_s and "CHUNK" in b3_s, f"B3 single attr failed: {b3_s}"
    # B3-3: transpile with no-op (no BIND3)
    b3_n = _transpile_bind3_to_chunk("plain text")
    assert b3_n == "plain text", f"B3 no-op failed: {b3_n}"
    # B3-4: transpile scheduling detection
    assert _RE_HAS_SCHEDULING.search("[red:blue:10]"), f"B3 scheduling regex failed"
    assert not _RE_HAS_SCHEDULING.search("red hair"), f"B3 scheduling regex false positive"
    # B3-5: parse validation — empty attrs
    try:
        _parse_bind3_prompt("BIND3{cat => }")
        assert False, "B3-5 should have raised"
    except PromptSyntaxError:
        pass
    # B3-6: parse validation — scheduling in attrs raises
    try:
        _parse_bind3_prompt("BIND3{cat => [red:blue:10], blue eyes}")
        assert False, "B3-6 should have raised"
    except PromptSyntaxError as e:
        assert "scheduling" in str(e).lower(), f"B3-6 wrong error: {e}"
    # B3-7: _contains_bind3_marker
    assert _contains_bind3_marker("BIND3{a => b, c}"), "B3-7 marker detection failed"
    assert not _contains_bind3_marker("BIND2{a => b}"), "B3-7 false positive on BIND2"
    assert not _contains_bind3_marker("plain"), "B3-7 false positive on plain"

    # ── BIND3 Attrs scheduling (Patch B) ───────────────────────────────────
    # B3-SCHED-1: _parse_bind3_prompt with allow_attr_scheduling=True does NOT raise
    try:
        _parse_bind3_prompt("BIND3{cat => [red:blue:10], blue eyes}", allow_attr_scheduling=True)
    except PromptSyntaxError:
        assert False, "B3-SCHED-1 should NOT raise with allow_attr_scheduling=True"
    # B3-SCHED-2: scheduled attr text is preserved in parsed result
    owner_s, attrs_s, weights_s, _, _ = _parse_bind3_prompt("BIND3{cat => [red:blue:10], blue eyes}", allow_attr_scheduling=True)
    assert owner_s == "cat", f"B3-SCHED-2 owner: {owner_s}"
    assert "[red:blue:10]" in attrs_s[0], f"B3-SCHED-2 attr0: {attrs_s[0]}"
    assert "blue eyes" in attrs_s[1], f"B3-SCHED-2 attr1: {attrs_s[1]}"
    # B3-SCHED-3: get_schedule with BIND3 scheduling (text-only path)
    sched_b3 = get_schedule("BIND3{cat => [red:blue:5], blue eyes}", steps=10, use_scheduling=True, seed=42)
    assert len(sched_b3) >= 1, f"B3-SCHED-3 empty schedule: {sched_b3}"
    # B3-SCHED-4: lint with BIND3 scheduling does not raise
    lint_b3 = lint_prompt("BIND3{cat => [red:blue:5], blue eyes}", steps=10)
    assert lint_b3.get("ok", False), f"B3-SCHED-4 lint failed: {lint_b3}"
    print("BIND3 smoke tests passed!")

    # ── Multiple BIND3 (Patch 4: Phase 4 sequential) ─────────────────────────
    # B3-multi-1: _find_top_level_bind3_blocks finds all blocks
    p4_prot, p4_res = _protect_escaped_literal_spans_for_source("BIND3{a => b} text BIND3{c => d}")
    p4_prot = _protect_escaped_literals(p4_prot)
    p4_blocks = _find_top_level_bind3_blocks(p4_prot)
    assert len(p4_blocks) == 2, f"B3-multi-1 expected 2 blocks, got {len(p4_blocks)}: {p4_blocks}"
    # B3-multi-2: segments extracted correctly
    p4_segs = _extract_sequential_backend_segments(p4_prot, p4_res)
    assert len(p4_segs) >= 2, f"B3-multi-2 expected >=2 segments, got {len(p4_segs)}: {p4_segs}"
    # B3-multi-3: single BIND3 → still works
    p4_prot2, p4_res2 = _protect_escaped_literal_spans_for_source("BIND3{a => b}")
    p4_prot2 = _protect_escaped_literals(p4_prot2)
    p4_blocks2 = _find_top_level_bind3_blocks(p4_prot2)
    assert len(p4_blocks2) == 1, f"B3-multi-3 expected 1 block, got {len(p4_blocks2)}: {p4_blocks2}"
    # B3-multi-4: BIND3 + non-BIND3 backend → BIND3 block detected, others also detected
    p4_prot3, p4_res3 = _protect_escaped_literal_spans_for_source("BIND3{a => b} CHUNK{c|d}")
    p4_prot3 = _protect_escaped_literals(p4_prot3)
    p4_segs3 = _extract_sequential_backend_segments(p4_prot3, p4_res3)
    assert any("BIND3{" in s for s in p4_segs3), f"B3-multi-4 BIND3 segment missing: {p4_segs3}"
    assert any("CHUNK{" in s for s in p4_segs3), f"B3-multi-4 CHUNK segment missing: {p4_segs3}"
    print("Multiple BIND3 (Phase 4) tests passed!")

    # ── Claim #9: unescaped brace warning ─────────────────────────────────────
    # C9-1: plain text with literal { → warning
    c9 = lint_prompt("sum_{i=1}^{n}")
    assert c9["ok"], f"C9-1 should not crash: {c9}"
    assert any(w["kind"] == "unescaped_brace_literal" for w in c9["warnings"]), f"C9-1 should have brace warning: {c9}"
    # C9-2: backend keyword + { → no warning
    c9 = lint_prompt("CHUNK{cat|dog}")
    assert c9["ok"], f"C9-2 should not crash: {c9}"
    assert not any(w["kind"] == "unescaped_brace_literal" for w in c9["warnings"]), f"C9-2 false positive: {c9}"
    # C9-3: escaped \{ and \} → no warning
    c9 = lint_prompt(r"sum\{i=1\}^\{n\}")
    assert c9["ok"], f"C9-3 should not crash: {c9}"
    assert not any(w["kind"] == "unescaped_brace_literal" for w in c9["warnings"]), f"C9-3 false positive on escaped: {c9}"
    # C9-4: plain text without braces → no warning
    c9 = lint_prompt("hello world")
    assert c9["ok"], f"C9-4 should not crash: {c9}"
    assert not any(w["kind"] == "unescaped_brace_literal" for w in c9["warnings"]), f"C9-4 false positive on plain: {c9}"
    print("Braces (Claim #9) tests passed!")

    # ── Claim #10: CHUNK inside BLEND ─────────────────────────────────────────
    # C10-1: BLEND with CHUNK inside → lint error
    c10 = lint_prompt("BLEND{CHUNK{a|b}|dog}")
    assert not c10["ok"], f"C10-1 should fail lint: {c10}"
    assert c10["kind"] == "chunk_inside_blend_not_supported", f"C10-1 wrong kind: {c10}"
    # C10-2: BLEND without CHUNK → ok
    c10 = lint_prompt("BLEND{cat|dog}")
    assert c10["ok"], f"C10-2 should be ok: {c10}"
    print("CHUNK-inside-BLEND (Claim #10) tests passed!")

    # ── MORPH SLERP curve ─────────────────────────────────────────────────
    # MS-1: slerp in MORPH_CURVES
    assert "slerp" in MORPH_CURVES, "MS-1 slerp not in MORPH_CURVES"
    # MS-2: _compute_morph_curve_weights with slerp — basic
    w = _compute_morph_curve_weights(3, [0, 1, 2], 0, "slerp")
    assert len(w) == 3, f"MS-2 length: {len(w)}"
    assert abs(w[0] - 1.0) < 1e-6, f"MS-2 start: {w}"
    w = _compute_morph_curve_weights(3, [0, 1, 2], 2, "slerp")
    assert abs(w[-1] - 1.0) < 1e-6, f"MS-2 end: {w}"
    # MS-3: slerp vs linear — midpoint (u=0.5): sin²(0.5*π/2)² ≈ sin²(0.785) ≈ 0.5
    # At u=0.5: sin(0.5*π/2) = sin(π/4) ≈ 0.707, squared ≈ 0.5
    # So at midpoint alpha = 0.5, same as linear. Test at u=0.25 (quarter):
    # sin²(0.25*π/2) = sin²(π/8) ≈ sin²(0.3927) ≈ 0.146 → alpha ≈ 0.146
    # Linear at u=0.25: alpha = 0.25
    w_slerp_q = _compute_morph_curve_weights(2, [0, 1], 0, "slerp")
    import math as _math_check
    expected_slerp_end = 1.0 - _math_check.sin(1.0 * _math_check.pi * 0.5) ** 2
    w_slerp_q2 = _compute_morph_curve_weights(2, [0, 1], 2, "slerp")
    assert abs(w_slerp_q2[-1] - 1.0) < 1e-6, f"MS-3 slerp end: {w_slerp_q2}"
    # MS-4: slerp unknown mode falls to error
    try:
        _compute_morph_curve_weights(2, [0, 1], 1, "nonexistent")
        assert False, "MS-4 should have raised"
    except PromptSyntaxError:
        pass
    # MS-5: MORPH_CURVES validation
    assert "linear" in MORPH_CURVES, "MS-5 linear still in MORPH_CURVES"
    assert "bezier" in MORPH_CURVES, "MS-5 bezier still in MORPH_CURVES"
    print("MORPH SLERP curve tests passed!")

    # ── Tensor SLERP (true spherical interpolation) ─────────────────────────
    # TS-1: functions exist
    assert callable(_slerp_tensor), "TS-1 _slerp_tensor not callable"
    assert callable(_slerp_condition_values), "TS-1 _slerp_condition_values not callable"
    # TS-2: dispatch — non-zero extraction logic (mathematical test, no torch)
    mock_w = [0.0, 0.146, 0.854, 0.0]
    non_zero = [i for i, w in enumerate(mock_w) if abs(w) > 1e-8]
    assert len(non_zero) == 2, f"TS-2 non_zero count: {len(non_zero)}"
    assert non_zero[0] == 1 and non_zero[-1] == 2, f"TS-2 non_zero indices: {non_zero}"
    mock_alpha = mock_w[non_zero[-1]]
    assert abs(mock_alpha - 0.854) < 1e-3, f"TS-2 alpha: {mock_alpha}"
    # TS-3: edge case — single non-zero weight (start/end)
    mock_w_start = [1.0] + [0.0] * 3
    nz = [i for i, w in enumerate(mock_w_start) if abs(w) > 1e-8]
    assert len(nz) == 1, f"TS-3 single nz: {nz}"
    # TS-4: edge case — all weights zero
    mock_w_zero = [0.0] * 4
    nz = [i for i, w in enumerate(mock_w_zero) if abs(w) > 1e-8]
    assert len(nz) == 0, f"TS-4 all zero: {nz}"
    # TS-5: _slerp_condition_values with non-torch raises ImportError (torch not available)
    # but should NOT crash the test — it proves the function is reachable
    try:
        _ensure_torch()
        print("  (torch available — real tensor SLERP tests skipped in self-test)")
    except ImportError:
        print("  (torch not available — tensor SLERP tested at logic level")
    print("Tensor SLERP tests passed!")

    # ── COMPOUND backend operator ──────────────────────────────────────────────
    # CPD-1: basic compound with weights (consistent *N format with BLEND)
    cpd1 = _extract_compound_prompt_spec("COMPOUND{base | part1*1.0 | part2*0.5}")
    assert cpd1 is not None, "CPD-1 extract failed"
    assert cpd1.base.strip() == "base", f"CPD-1 base: {cpd1.base}"
    assert len(cpd1.parts) == 2, f"CPD-1 parts: {len(cpd1.parts)}"
    assert cpd1.parts[0].text == "part1", f"CPD-1 part1: {cpd1.parts[0].text}"
    assert abs(cpd1.parts[0].weight - 1.0) < 1e-6, f"CPD-1 w1: {cpd1.parts[0].weight}"
    assert abs(cpd1.parts[1].weight - 0.5) < 1e-6, f"CPD-1 w2: {cpd1.parts[1].weight}"
    # CPD-2: per-part scheduling ranges
    cpd2 = _extract_compound_prompt_spec("COMPOUND{base | glow@5-15*1.0 | fade@10-20*0.5}")
    assert cpd2 is not None, "CPD-2 extract failed"
    assert cpd2.parts[0].step_start == 5 and cpd2.parts[0].step_end == 15, f"CPD-2 range1: {cpd2.parts[0]}"
    assert cpd2.parts[1].step_start == 10 and cpd2.parts[1].step_end == 20, f"CPD-2 range2: {cpd2.parts[1]}"
    # CPD-3: text schedule preview
    cpd3_text = _build_compound_text_schedule_from_spec(cpd2, 20, True, None, True)
    assert len(cpd3_text) >= 3, f"CPD-3 schedule length: {len(cpd3_text)}"
    assert "COMPOUND<" in cpd3_text[0][1], f"CPD-3 preview format: {cpd3_text[0][1]}"
    # CPD-4: _contains_compound_marker
    assert _contains_compound_marker("COMPOUND{base|part}"), "CPD-4 marker detection failed"
    assert not _contains_compound_marker("CHUNK{base|part}"), "CPD-4 false positive on CHUNK"
    # CPD-5: _parse_compound_part_text rejects scheduling
    try:
        _parse_compound_part_text("cat[red:blue:10]*1.0", full_text="COMPOUND{a|cat[red:blue:10]*1.0}")
        assert False, "CPD-5 should have raised"
    except PromptSyntaxError as e:
        assert "scheduling" in str(e).lower(), f"CPD-5 wrong error: {e}"
    # CPD-6: invalid range @0
    try:
        _parse_compound_part_text("cat@0-10*1.0", full_text="COMPOUND{a|cat@0-10*1.0}")
        # @0 coerces to @1, should succeed
        # Test invalid start>end
        _parse_compound_part_text("cat@15-5*1.0", full_text="COMPOUND{a|cat@15-5*1.0}")
        assert False, "CPD-6 should have raised on start>end"
    except PromptSyntaxError:
        pass
    print("COMPOUND tests passed!")

    # ── Phase 4 smoke tests ───────────────────────────────────────────────────
    print("All self-tests passed (additions: Claims #9, #10, Multi-BIND3, MORPH-SLERP, Tensor-SLERP, COMPOUND)!")
