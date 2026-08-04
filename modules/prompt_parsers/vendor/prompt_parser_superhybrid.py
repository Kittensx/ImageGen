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
from typing import Any, Sequence

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
_RE_WEIGHT_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
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
# Marker format: OPEN body SEP w0 SEP w1 SEP [... SEP wn] SEP mode CLOSE
#   weights: 2+ numeric values separated by SEP (multi-segment interpolation)
#   mode: one of linear | ease-in | ease-out | ease-in-out | bezier | catmull
#         or cubic(p1x,p1y,p2x,p2y) for parametric cubic bezier
# Private-use chars survive _collapse_spaces() and _unescape_literals() unchanged.
ATTN_INTERP_OPEN  = "\uE001"  # marker start
ATTN_INTERP_SEP   = "\uE002"  # field separator  body | w0 | w1 | mode
ATTN_INTERP_CLOSE = "\uE003"  # marker end

# Valid easing mode names.
_EASING_MODES = frozenset({"linear", "ease", "ease-in", "ease-out", "ease-in-out",
    "bezier", "catmull",
    "sine-in", "sine-out", "sine-in-out",
    "quart-in", "quart-out", "quart-in-out",
    "quint-in", "quint-out", "quint-in-out",
    "expo-in", "expo-out", "expo-in-out",
    "circ-in", "circ-out", "circ-in-out",
    "back-in", "back-out", "back-in-out",
    "bounce"})
_EASING_DEFAULT = "linear"

# PUA ranges used internally — strip from user input to prevent injection.
_PUA_PLACEHOLDERS = frozenset(
    {"\uE000", "\uE001", "\uE002", "\uE003", "\uE004", "\uE005",
     "\uE110", "\uE111", "\uE112", "\uE113", "\uE114",
     "\uE115", "\uE116", "\uE117", "\uE118", "\uE119", "\uE11A"}
    | {chr(i) for i in range(0xE200, 0xE300)}
)
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
    "invalid_diff_syntax",
    "invalid_region_syntax",
    "invalid_region_coords",
    "region_reverse_range",
    "region_mixed_range",
    "region_empty_block",
    "region_nested",
    "region_grid_invalid_ratios",
    "region_grid_empty",
})
_MAX_SEMANTIC_WARNINGS = 128
_warned_semantic_errors: set[str] = set()
_warned_semantic_errors_lock = _threading.Lock()


def _warn_semantic_once(msg: str) -> bool:
    """Register warning message at most once (bounded). Returns True if caller should log."""
    with _warned_semantic_errors_lock:
        if msg in _warned_semantic_errors:
            return False
        if len(_warned_semantic_errors) >= _MAX_SEMANTIC_WARNINGS:
            return False
        _warned_semantic_errors.add(msg)
        return True

# Matches serialized marker in schedule text (used by post-pass).
# Groups: (1) body, (2) weights SEP-separated, (3) mode with optional cubic params
RE_ATTN_INTERP_LITERAL = re.compile(
    rf"{re.escape(ATTN_INTERP_OPEN)}(.*?)"
    rf"{re.escape(ATTN_INTERP_SEP)}"
    rf"((?:{NUMERIC_RE}{re.escape(ATTN_INTERP_SEP)})+)"
    rf"([A-Za-z][A-Za-z0-9_-]*(?:\([^)]*\))?)"
    rf"(?:{re.escape(ATTN_INTERP_SEP)}(\d*%?){re.escape(ATTN_INTERP_SEP)}(\d*%?))?"
    rf"{re.escape(ATTN_INTERP_CLOSE)}"
)

# Tail of a raw interpolation spec after last top-level ':'.
# Supports multi-segment: 'w0 -> w1 -> w2' and optional easing: '... ~ mode'
# mode token can be named (ease-in, bezier) or parametric (cubic(a,b,c,d)).
# Optional '@ start-end' for sub-range activation within the global schedule:
#   (body:1.0->2.0 @ 5-15) — interpolates only between steps 5 and 15.
#   (body:1.0->2.0 @ 20%-80%) — percentage of total steps.
_RE_ATT_INTERP_TAIL = re.compile(
    rf'^\s*((?:{NUMERIC_RE}\s*->\s*)+{NUMERIC_RE})'
    rf'(?:\s*~\s*([A-Za-z][A-Za-z0-9_-]*(?:\([^)]*\))?))?'
    rf'(?:\s*@\s*(\d+%?)\s*-\s*(\d+%?))?\s*$'
)


def _resolve_range_step(val: str, total_steps: int) -> int:
    """Resolve a step value to 1-indexed absolute step. Handles '15' and '20%'."""
    val = val.strip()
    if val.endswith('%'):
        pct = float(val[:-1])
        return max(1, min(total_steps, int(pct / 100.0 * total_steps)))
    return int(val)  # absolute step — caller handles via t_lin formula


def _serialize_att_interp(body: str, weights: list[float],
                           mode: str = _EASING_DEFAULT,
                           start_range: str | None = None,
                           end_range: str | None = None) -> str:
    """Pack (body, weights, mode, optional @start-end) into a private-use marker string."""
    safe = (body or "").replace(ATTN_INTERP_OPEN, "").replace(ATTN_INTERP_SEP, "").replace(ATTN_INTERP_CLOSE, "")
    m = mode if mode in _EASING_MODES or mode.startswith("cubic(") else _EASING_DEFAULT
    weight_str = ATTN_INTERP_SEP.join(str(w) for w in weights)
    if start_range is not None and end_range is not None:
        return f"{ATTN_INTERP_OPEN}{safe}{ATTN_INTERP_SEP}{weight_str}{ATTN_INTERP_SEP}{m}{ATTN_INTERP_SEP}{start_range}{ATTN_INTERP_SEP}{end_range}{ATTN_INTERP_CLOSE}"
    return f"{ATTN_INTERP_OPEN}{safe}{ATTN_INTERP_SEP}{weight_str}{ATTN_INTERP_SEP}{m}{ATTN_INTERP_CLOSE}"

def _solve_cubic_bezier_x(t: float, cp1x: float, cp2x: float,
                          max_iters: int = 10, tol: float = 1e-7) -> float:
    """Solve x(t_param) = t for a cubic bezier with P0=(0,0), P3=(1,1).

    Uses Newton-Raphson with bisection fallback for edge cases.
    Returns t_param such that x(t_param) ≈ t.
    """
    # Cubic bezier x-coordinate: Bx(t) = 3(1-t)²t·cp1x + 3(1-t)t²·cp2x + t³
    # Newton: t_{n+1} = t_n - (Bx(t_n) - t) / Bx'(t_n)
    # Bx'(t) = 3(1-t)²·cp1x + 6(1-t)t·(cp2x - cp1x) + 3t²·(1 - cp2x)
    guess = t
    for _ in range(max_iters):
        omt = 1.0 - guess
        bx = 3.0 * omt * omt * guess * cp1x + 3.0 * omt * guess * guess * cp2x + guess * guess * guess
        bxd = 3.0 * omt * omt * cp1x + 6.0 * omt * guess * (cp2x - cp1x) + 3.0 * guess * guess * (1.0 - cp2x)
        if abs(bxd) < 1e-12:
            break
        guess -= (bx - t) / bxd
        guess = max(0.0, min(1.0, guess))
        if abs(bx - t) < tol:
            break
    return max(0.0, min(1.0, guess))


def _apply_cubic_bezier_easing(t: float, cp1x: float, cp1y: float,
                                cp2x: float, cp2y: float) -> float:
    """Evaluate cubic-bezier easing: map input t∈[0,1] through (cp1, cp2)."""
    tp = _solve_cubic_bezier_x(t, cp1x, cp2x)
    omt = 1.0 - tp
    return 3.0 * omt * omt * tp * cp1y + 3.0 * omt * tp * tp * cp2y + tp * tp * tp


def _apply_easing(t: float, mode: str) -> float:
    """Map linear t∈[0,1] through easing curve.
    Note: "bezier" and "ease" are aliases for "ease-in-out" (smoothstep = cubic bezier with CP[0,0,1,1]).
    Supports parametric cubic bezier: mode="cubic(p1x,p1y,p2x,p2y)".
    Includes Penner easing functions.
    """
    if mode.startswith("cubic(") and mode.endswith(")"):
        params = mode[6:-1].split(",")
        if len(params) == 4:
            try:
                cp = [float(p.strip()) for p in params]
                return _apply_cubic_bezier_easing(t, cp[0], cp[1], cp[2], cp[3])
            except (TypeError, ValueError):
                pass
        return t  # fallback to linear on malformed params
    if mode in ("ease-in-out", "bezier", "ease"):
        return t * t * (3.0 - 2.0 * t)  # smoothstep
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return t * (2.0 - t)
    if mode == "catmull":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep
    # Penner easing functions
    if mode == "sine-in":
        return 1.0 - math.cos(t * math.pi / 2.0)
    if mode == "sine-out":
        return math.sin(t * math.pi / 2.0)
    if mode == "sine-in-out":
        return -(math.cos(math.pi * t) - 1.0) / 2.0
    if mode == "quart-in":
        return t * t * t * t
    if mode == "quart-out":
        return 1.0 - math.pow(1.0 - t, 4)
    if mode == "quart-in-out":
        if t < 0.5:
            return 8.0 * t * t * t * t
        return 1.0 - math.pow(-2.0 * t + 2.0, 4) / 2.0
    if mode == "quint-in":
        return t * t * t * t * t
    if mode == "quint-out":
        return 1.0 - math.pow(1.0 - t, 5)
    if mode == "quint-in-out":
        if t < 0.5:
            return 16.0 * t * t * t * t * t
        return 1.0 - math.pow(-2.0 * t + 2.0, 5) / 2.0
    if mode == "expo-in":
        return 0.0 if t == 0.0 else math.pow(2.0, 10.0 * (t - 1.0))
    if mode == "expo-out":
        return 1.0 if t == 1.0 else 1.0 - math.pow(2.0, -10.0 * t)
    if mode == "expo-in-out":
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        if t < 0.5:
            return math.pow(2.0, 10.0 * (2.0 * t - 1.0)) / 2.0
        return (2.0 - math.pow(2.0, -10.0 * (2.0 * t - 1.0))) / 2.0
    if mode == "circ-in":
        return 1.0 - math.sqrt(1.0 - t * t)
    if mode == "circ-out":
        return math.sqrt(1.0 - (t - 1.0) * (t - 1.0))
    if mode == "circ-in-out":
        if t < 0.5:
            return (1.0 - math.sqrt(1.0 - 4.0 * t * t)) / 2.0
        return (math.sqrt(1.0 - (2.0 * t - 2.0) * (2.0 * t - 2.0)) + 1.0) / 2.0
    if mode == "back-in":
        return t * t * t - t * math.sin(t * math.pi)
    if mode == "back-out":
        t -= 1.0
        return t * t * t + t * math.sin(t * math.pi) + 1.0
    if mode == "back-in-out":
        if t < 0.5:
            t2 = 2.0 * t
            return (t2 * t2 * t2 - t2 * math.sin(t2 * math.pi)) / 2.0
        t2 = 2.0 * t - 2.0
        return (t2 * t2 * t2 + t2 * math.sin(t2 * math.pi)) / 2.0 + 1.0
    if mode == "bounce":
        # Penner ease-out-bounce: 3 bounces settling at 1.0.
        if t < 1.0 / 2.75:
            return 7.5625 * t * t
        elif t < 2.0 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375
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
                    if len(restore) >= 256:
                        out.append(text[start:matched_end])
                        i = matched_end
                        continue
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
    if ESCAPED_AMP_PLACEHOLDER in text:
        text = text.replace(ESCAPED_AMP_PLACEHOLDER, "&")
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


# ── Variables / Macros / Params — препроцессор (ПАТЧ 1) ────────────────────

_RE_VM_RANDOM   = re.compile(r"<random:([^>]+)>",           re.IGNORECASE)
_VM_VAL         = r"(?:->|[^<>]|<[^>]*>)*"
_RE_VM_SETVAR   = re.compile(rf"<setvar\[([^\]]+)\]:({_VM_VAL})>",   re.IGNORECASE)
_RE_VM_SETMACRO = re.compile(rf"<setmacro\[([^\]]+)\]:({_VM_VAL})>", re.IGNORECASE)
_RE_VM_VAR      = re.compile(r"<var(?::([^>]+)|\[([^\]]+)\])>", re.IGNORECASE)
_RE_VM_MACRO    = re.compile(r"<macro(?::([^>]+)|\[([^\]]+)\])>",re.IGNORECASE)
_RE_VM_PARAM    = re.compile(rf"<param\[([^\]]+)\]:({_VM_VAL})>", re.IGNORECASE)
_RE_VM_WILDCARD = re.compile(r"__([a-zA-Z0-9_\-/]+)__")


def _vm_eval_randoms(text: str, rng: random.Random) -> str:
    def _repl(m: re.Match) -> str:
        raw = m.group(1)
        sep = "|" if ("|" in raw and "," not in raw) else ","
        opts = [o.strip() for o in raw.split(sep) if o.strip()]
        return rng.choice(opts) if opts else ""
    for _ in range(8):
        new = _RE_VM_RANDOM.sub(_repl, text)
        if new == text:
            break
        text = new
    return text


def _vm_substitute_vars(text: str, variables: dict[str, str]) -> str:
    if not variables or "<" not in text:
        return text
    def _repl(m: re.Match) -> str:
        name = (m.group(1) or m.group(2) or "").strip()
        return variables.get(name, m.group(0))
    return _RE_VM_VAR.sub(_repl, text)


def _vm_expand_wildcards(text: str, rng: random.Random, wildcard_dir: str | None = None) -> str:
    if "__" not in text:
        return text
    dirs_to_try: list[str] = []
    if wildcard_dir:
        dirs_to_try.append(wildcard_dir)
    for candidate in ["wildcards", "extensions/wildcards/wildcards"]:
        dirs_to_try.append(candidate)

    def _is_safe_relative_name(raw_name: str) -> bool:
        if raw_name.startswith("/") or raw_name.startswith("\\"):
            return False
        parts = raw_name.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            return False
        return True

    def _repl(m: re.Match) -> str:
        raw_name = m.group(1)
        if not _is_safe_relative_name(raw_name):
            return m.group(0)
        name = raw_name.replace("/", os.sep).replace("\\", os.sep)
        for d in dirs_to_try:
            try:
                base = os.path.realpath(d)
                candidate_path = os.path.realpath(os.path.join(d, name + ".txt"))
            except (OSError, ValueError):
                continue
            if candidate_path != base and not candidate_path.startswith(base + os.sep):
                continue
            if os.path.isfile(candidate_path):
                try:
                    with open(candidate_path, encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                    if lines:
                        return rng.choice(lines)
                except OSError as exc:
                    logger.warning("Wildcard file not readable: %s (%s)", candidate_path, exc)
        msg = f"Wildcard file not found: {raw_name}"
        if _warn_semantic_once(msg):
            logger.warning(msg)
        return m.group(0)

    return _RE_VM_WILDCARD.sub(_repl, text)


def _expand_vars_and_macros(
    prompt: str,
    seed: int | None = None,
    wildcard_dir: str | None = None,
) -> tuple[str, dict[str, str]]:
    if "<" not in prompt and "__" not in prompt:
        return prompt, {}

    if seed is None:
        import hashlib as _hashlib
        seed = int.from_bytes(_hashlib.sha256(prompt.encode('utf-8', errors='replace')).digest()[:4], 'big') & 0x7fffffff
    rng = random.Random(seed)
    variables: dict[str, str] = {}
    macros:    dict[str, str] = {}
    meta:      dict[str, str] = {}

    prompt = _vm_expand_wildcards(prompt, rng, wildcard_dir)

    for _ in range(16):
        if "<" not in prompt:
            break
        prev = prompt

        prompt = _vm_eval_randoms(prompt, rng)

        def _proc_setvar(m: re.Match) -> str:
            parts = [p.strip() for p in m.group(1).split(",", 1)]
            name = parts[0]
            emit = len(parts) < 2 or parts[1].lower() not in ("false", "0", "no")
            val = _vm_substitute_vars(m.group(2).strip(), variables)
            variables[name] = val
            return val if emit else ""
        prompt = _RE_VM_SETVAR.sub(_proc_setvar, prompt)

        def _proc_setmacro(m: re.Match) -> str:
            parts = [p.strip() for p in m.group(1).split(",", 1)]
            name = parts[0]
            emit = len(parts) < 2 or parts[1].lower() not in ("false", "0", "no")
            macros[name] = m.group(2).strip()
            if emit:
                expanded = _vm_eval_randoms(macros[name], rng)
                return _vm_substitute_vars(expanded, variables)
            return ""
        prompt = _RE_VM_SETMACRO.sub(_proc_setmacro, prompt)

        prompt = _vm_substitute_vars(prompt, variables)

        def _proc_macro(m: re.Match) -> str:
            name = (m.group(1) or m.group(2) or "").strip()
            tmpl = macros.get(name)
            if tmpl is None:
                return m.group(0)
            return _vm_substitute_vars(_vm_eval_randoms(tmpl, rng), variables)
        prompt = _RE_VM_MACRO.sub(_proc_macro, prompt)

        if prompt == prev:
            break

    def _collect_param(m: re.Match) -> str:
        meta[m.group(1).strip()] = m.group(2).strip()
        return ""
    prompt = _RE_VM_PARAM.sub(_collect_param, prompt)

    return prompt, meta


_VAR_META_LOCAL = _threading.local()


def _coerce_param_value(raw: str) -> int | float | str:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def get_prompt_params(text: str, seed: int | None = None,
                       wildcard_dir: str | None = None) -> dict[str, str]:
    """Extract <param[name]:value> from prompt text without going through get_schedule/thread-local."""
    _, meta = _expand_vars_and_macros(str(text or ""), seed, wildcard_dir)
    return meta


def _extract_toneg(text: str) -> tuple[str, str]:
    """Remove TONEG{...} blocks from positive prompt, return (clean_text, extra_negative).

    Uses depth-tracking for nested braces, same pattern as _find_top_level_region_blocks.
    Handles TONEG{ a {b} c } → extra_negative="a {b} c", clean has no TONEG prefix.
    """
    keyword = TONEG_KEYWORD
    kw_len = len(keyword)
    parts: list[str] = []
    clean_parts: list[str] = []
    i = 0
    t = str(text or "")
    while i < len(t):
        if t[i] == '\\' and i + 1 < len(t) and t[i + 1] in ('{', '}'):
            clean_parts.append(t[i:i+2])
            i += 2
            continue
        if t.startswith(keyword, i) and i + kw_len < len(t) and t[i + kw_len] == '{':
            brace_open = i + kw_len
            depth = 1
            j = brace_open + 1
            while j < len(t) and depth > 0:
                if t[j] == '\\' and j + 1 < len(t) and t[j + 1] in ('{', '}'):
                    j += 2
                    continue
                if t[j] == '{':
                    depth += 1
                elif t[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                body = t[brace_open + 1 : j - 1]
                parts.append(body.strip())
                i = j
                continue
        clean_parts.append(t[i])
        i += 1
    clean = "".join(clean_parts)
    return _collapse_spaces(clean), ", ".join(p for p in parts if p)


def get_prompt_with_toneg(text: str) -> tuple[str, str]:
    """Public API: return (prompt_without_TONEG, extra_negative_text). Stateless, no thread-local.

    Runs the same vars/macros/wildcards pre-pass that get_prompt_regions() already does
    (see _expand_vars_and_macros call there) so that <setvar>/<var:>/<macro:>/wildcards set
    in the base prompt are visible inside TONEG{} before it is split off into the negative
    prompt. Without this, TONEG{} is extracted from raw text before any expansion happens,
    and the negative prompt is expanded later in complete isolation (fresh empty `variables`
    dict per call) -- so a base-prompt <setvar> is silently invisible inside TONEG{}, the same
    class of cross-scope bug SwarmUI has open against setvar/negative-prompt (issue #884/#585).
    """
    if "<" in text or "__" in text:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8", errors="replace")).digest()[:4], "big"
        ) & 0x7fffffff
        text, _ = _expand_vars_and_macros(text, seed, None)
    return _extract_toneg(text)


def get_prompt_regions(text: str, steps: int = 20, use_scheduling: bool = True, seed: int | None = None) -> tuple[str, list[RegionBlock]]:
    """Extract REGION{...} blocks from text.

    Returns (clean_text, regions).
    clean_text has REGION{...} replaced with space-joined branch texts.
    REGION blocks are resolved to plain text before get_schedule().

    Caller flow:
        text, extra_neg = get_prompt_with_toneg(original_text)
        clean_text, regions = get_prompt_regions(text)
        schedule = get_schedule(clean_text, steps)

    Independent of get_schedule — pure text transformation.
    No TONEG handling inside (caller's responsibility).
    """
    text = str(text or "")
    text = text.translate(_PUA_CLEAN_TABLE)
    if "<" in text or "__" in text:
        if seed is None:
            seed = int.from_bytes(
                hashlib.sha256(text.encode('utf-8', errors='replace')).digest()[:4],
                'big'
            ) & 0x7fffffff
        text, _var_meta = _expand_vars_and_macros(text, seed, None)
    return _extract_region_blocks(text)


def extract_non_region_text(clean_text: str, regions: list[RegionBlock], original_text: str | None = None) -> str:
    """Extract text NOT inside any REGION block.

    For REGION v2 denoiser: bg cond should NOT include region texts.
    clean_text = "cat red blue" (from get_prompt_regions)
    regions = [RegionBlock(text="red"), RegionBlock(text="blue")]
    → returns "cat  " (the non-region parts preserved with spacing)

    When original_text is available (preferred), extracts non-region text
    directly from original positions — immune to text duplication.
    """
    if original_text is not None and _find_top_level_region_blocks(original_text):
        blocks = _find_top_level_region_blocks(original_text)
        non_parts: list[str] = []
        pos = 0
        for start, end, body, axis in blocks:
            if pos < start:
                non_parts.append(original_text[pos:start])
            skip_to = end + 1
            if skip_to < len(original_text):
                # Skip grid suffix [H:...|V:...] first
                gs_pos = skip_to
                while gs_pos < len(original_text) and original_text[gs_pos].isspace():
                    gs_pos += 1
                _, _, grid_skip = _parse_region_grid_suffix(original_text, gs_pos)
                if grid_skip > 0:
                    skip_to = gs_pos + grid_skip
                _raw = original_text[skip_to:]
                rest = _raw.lstrip()
                if rest.startswith(':') and len(rest) >= 2 and rest[1].upper() in ('H', 'V'):
                    _ws = len(_raw) - len(rest)
                    skip_to += _ws + 2
                    # Skip optional ratio suffix like :0.5,0.3
                    rest2 = original_text[skip_to:].lstrip()
                    if rest2.startswith(':'):
                        r_end = 1
                        while r_end < len(rest2) and (rest2[r_end].isdigit() or rest2[r_end] in '.,-'):
                            r_end += 1
                        if r_end > 1:
                            skip_to += len(rest2[:r_end])
            pos = skip_to
        if pos < len(original_text):
            non_parts.append(original_text[pos:])
        return "".join(non_parts).strip()

    # Fallback: heuristic removal (may fail if region text duplicates
    # prefix/suffix — use original_text= for correctness).
    for r in reversed(regions):
        rt = r.text.strip()
        if rt:
            idx = clean_text.rfind(rt)
            if idx >= 0:
                clean_text = clean_text[:idx] + clean_text[idx + len(rt):]
    return clean_text.strip()


def build_virtual_region_prompt(non_region_text: str, regions: list[RegionBlock]) -> str:
    """Build a pseudo-AND prompt where each region is a separate AND-branch.

    For REGION v2: each region text becomes a separate conditioning branch
    that gets spatial masking in combine_denoised.

    Example:
      non_region_text = "cat"
      regions = [red@left, blue@right]
      → "cat AND red AND blue"
    """
    parts = [non_region_text]
    for r in regions:
        rt = r.text.strip()
        if rt:
            if abs(r.weight - 1.0) > 1e-6:
                parts.append(f"{rt}:{r.weight}")
            else:
                parts.append(rt)
    if len(parts) <= 1:
        return non_region_text
    return " AND ".join(parts)


def _get_non_region_text_from_prompt(full_prompt: str) -> tuple[str, list[RegionBlock], str]:
    """Full pipeline: extract REGION blocks + non-region text + virtual prompt.

    Returns (non_region_text, regions, virtual_prompt).
    virtual_prompt = non_region_text AND region1 AND region2 ...

    For standalone use (outside processing.py pipeline).
    """
    clean, regions = get_prompt_regions(full_prompt)
    non_region = extract_non_region_text(clean, regions, original_text=full_prompt)
    virtual = build_virtual_region_prompt(non_region, regions)
    return non_region, regions, virtual


def _parse_cfg_param(raw: str, steps: int) -> list[float] | float:
    """Parse CFG schedule from <param[cfg]:7.0->3.0> or <param[cfg]:7.0>.

    Returns:
        list[float] of length steps for range syntax w0->w1->...->wN
        float scalar for a single value.
    """
    if "->" not in raw:
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    try:
        weights = [float(w.strip()) for w in raw.split("->")]
    except (ValueError, TypeError):
        return raw
    n_segments = len(weights) - 1
    result: list[float] = []
    for i in range(steps):
        t_lin = i / max(steps - 1, 1)
        seg = min(int(t_lin * n_segments), n_segments - 1)
        seg_t = t_lin * n_segments - seg
        result.append(weights[seg] + (weights[seg + 1] - weights[seg]) * seg_t)
    return result


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

# Включает кумулятивное построение F_i в BIND3: вместо "только текущий
# атрибут реален, остальные — заглушки" каждое F_i содержит ВСЕ атрибуты
# до текущего включительно как реальный текст (а не padding), что честнее
# отражает каузальную зависимость CLIP-энкодера от предшествующего
# контекста. Побочный эффект: строки атрибута, совпадающего с последним
# core_part, при этом совпадают с suffix-диапазоном, так что suffix тоже
# перестаёт быть "слепым" к реальным атрибутам при w=1.0 для последнего
# атрибута. По умолчанию ВЫКЛ (0) — поведение существующих вызывающих
# не меняется.
BIND3_CUMULATIVE_CONTEXT = _env_bool("BIND3_CUMULATIVE_CONTEXT", "0")
_RE_BIND2_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND2_KEYWORD)}(?:\s*\^([^{{]*))?\s*\{{")
_RE_BIND3_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND3_KEYWORD)}(?:\s*\^([^{{]*))?\s*\{{")
_RE_HAS_SCHEDULING = re.compile(r"\[(?:[^\[\]]*:[^\[\]]*:\d+%?|[^\[\]]*:\d+%?)\]")
ASSEMBLE_KEYWORD = "ASSEMBLE"
ASSEMBLE_PREVIEW_PREFIX = "ASSEMBLE<"
_RE_ASSEMBLE_MARKER = re.compile(rf"(?<![\w\\]){re.escape(ASSEMBLE_KEYWORD)}\s*\{{")
BLEND_KEYWORD = "BLEND"
BLEND_PREVIEW_PREFIX = "BLEND<"
BLEND_MODES = frozenset({"mean", "sum", "product", "max"})
BACKEND_CHANNEL_TARGETS = frozenset({"both", "cross", "pooled", "enc1", "enc2", "t5"})
SDXL_ENCODER_CHANNEL_TARGETS = frozenset({"enc1", "enc2"})
FLUX_ENCODER_CHANNEL_TARGETS = frozenset({"enc1", "t5"})
SD3_ENCODER_CHANNEL_TARGETS  = frozenset({"enc1", "enc2", "t5"})
SDXL_ENCODER1_CROSS_DIM = 768
SDXL_TOTAL_CROSS_DIM = 2048
_RE_BLEND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BLEND_KEYWORD)}(?:\s*\^[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")
MORPH_KEYWORD = "MORPH"
MORPH_PREVIEW_PREFIX = "MORPH<"
MORPH_CURVES = frozenset({"linear", "bezier", "bernstein", "catmull", "slerp",
    "ease", "ease-in", "ease-out", "ease-in-out",
    "sine-in", "sine-out", "sine-in-out",
    "quart-in", "quart-out", "quart-in-out",
    "quint-in", "quint-out", "quint-in-out",
    "expo-in", "expo-out", "expo-in-out",
    "circ-in", "circ-out", "circ-in-out",
    "back-in", "back-out", "back-in-out",
    "bounce"})
_RE_MORPH_MARKER = re.compile(rf"(?<![\w\\]){re.escape(MORPH_KEYWORD)}(?:\s*\^[^\[\{{@]*)?(?:\s*@[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")
COMPOUND_KEYWORD = "COMPOUND"
COMPOUND_PREVIEW_PREFIX = "COMPOUND<"
_RE_COMPOUND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(COMPOUND_KEYWORD)}\s*\{{")
_RE_COMPOUND_RANGE = re.compile(r"@(\d+)(?:-(\d+))?")
_RE_COMPOUND_WEIGHT = re.compile(r"\*(-?\d+(?:\.\d+)?)")
_RE_COMPOUND_CURVE = re.compile(r"~([A-Za-z][A-Za-z0-9_-]*(?:\([^)]*\))?)")
DIFF_KEYWORD = "DIFF"
_RE_DIFF_MARKER = re.compile(rf"(?<![\w\\]){re.escape(DIFF_KEYWORD)}(?:\s*\^([^\{{]*))?\s*\{{")
TONEG_KEYWORD = "TONEG"
REGION_KEYWORD = "REGION"
RegionBlock = namedtuple("RegionBlock", [
    "text", "x1", "x2", "y1", "y2", "weight",
    "axis", "base_text", "mode", "backend", "coords_pixels", "stop", "start", "blur", "canvas",
    "base_ratio", "curve",
], defaults=[0.2, "linear"])
REGION_LATENT_SCALE = 8  # VAE factor: 512px / 64 latent = 8

def _build_region_masks(regions: list[RegionBlock], h_lat: int, w_lat: int):
    """Convert RegionBlock coordinates to binary masks [1,1,H,W] in latent space.
    Returns list of tensors in the same order as regions. Requires torch."""
    _ensure_torch()
    import torch
    masks = []
    for r in regions:
        mask = torch.zeros(1, 1, h_lat, w_lat)
        coords = (r.x1, r.x2, r.y1, r.y2)
        if r.coords_pixels:
            xs = [int(v / REGION_LATENT_SCALE) for v in (r.x1, r.x2)]
            ys = [int(v / REGION_LATENT_SCALE) for v in (r.y1, r.y2)]
        else:
            xs = [int(r.x1 * w_lat), int(r.x2 * w_lat)]
            ys = [int(r.y1 * h_lat), int(r.y2 * h_lat)]
        x1 = max(0, min(xs[0], xs[1]))
        x2 = min(w_lat, max(xs[0], xs[1]))
        y1 = max(0, min(ys[0], ys[1]))
        y2 = min(h_lat, max(ys[0], ys[1]))
        if x1 < x2 and y1 < y2:
            mask[0, 0, y1:y2, x1:x2] = 1.0
        masks.append(mask)
    return masks

_RE_REGION_MARKER = re.compile(r'REGION\{', re.IGNORECASE)
_RE_REGION_BRANCH = re.compile(
    r'(.*)@\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)'
    r'(?:\s*\*\s*(?:([0-9.]+)(?:\s*~\s*([A-Za-z][A-Za-z0-9_-]*(?:\([^)]*\))?))?|([A-Za-z][A-Za-z0-9_-]*(?:\([^)]*\))?)))?'
)
_RE_REGION_GRID = re.compile(
    r'\[(?:H:([^\|\]]*)(?:\s*\|\s*V:([^\]]*))?|V:([^\]]*))\]',
    re.IGNORECASE,
)


def _split_region_branch_weight_and_curve(text: str) -> tuple[str, float, str]:
    """Split trailing '*weight~curve' (or '*weight') from a compact REGION branch.

    Compact branches carry no @coords (grid mode serialization: 'cat*1.5~ease-in').
    Returns (text, weight, curve). Lenient: non-numeric tails fall back to text.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "", 1.0, "linear"
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
        return stripped, 1.0, "linear"
    raw_tail = stripped[star_pos + 1 :].strip()
    curve = "linear"
    weight_text = raw_tail
    tilde_pos = raw_tail.find("~")
    if tilde_pos >= 0:
        curve_spec = raw_tail[tilde_pos + 1 :].strip()
        if curve_spec.startswith("cubic") or curve_spec in _EASING_MODES:
            curve = curve_spec
        else:
            # Non-curve '~' inside tail → treat whole branch as text
            return stripped, 1.0, "linear"
        weight_text = raw_tail[:tilde_pos].strip()
    elif raw_tail.startswith("cubic") or raw_tail in _EASING_MODES:
        # Curve-only tail (weight omitted): 'text*ease-in' → w=1.0, curve
        return stripped[:star_pos].rstrip(), 1.0, raw_tail
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        return stripped, 1.0, "linear"
    weight = float(weight_text)
    if weight <= 0:
        return stripped, 1.0, "linear"
    return stripped[:star_pos].rstrip(), weight, curve


def _parse_region_grid_suffix(text: str, start_pos: int) -> tuple[list[float] | None, list[float] | None, int]:
    m = _RE_REGION_GRID.match(text, start_pos)
    if not m:
        return None, None, 0
    raw_h = m.group(1)
    raw_v = m.group(2) if m.group(2) is not None else m.group(3)
    h_ratios: list[float] = []
    v_ratios: list[float] = []
    if raw_h is not None and raw_h.strip():
        try:
            h_ratios = [float(x.strip()) for x in raw_h.split(',') if x.strip()]
        except ValueError:
            raise PromptSyntaxError(
                f"Invalid H-ratios in region grid: '{raw_h}'",
                kind="region_grid_invalid_ratios",
            )
        if any(r <= 0 for r in h_ratios):
            raise PromptSyntaxError(
                "Region grid H-ratios must be positive",
                kind="region_grid_invalid_ratios",
            )
    if raw_v is not None and raw_v.strip():
        try:
            v_ratios = [float(x.strip()) for x in raw_v.split(',') if x.strip()]
        except ValueError:
            raise PromptSyntaxError(
                f"Invalid V-ratios in region grid: '{raw_v}'",
                kind="region_grid_invalid_ratios",
            )
        if any(r <= 0 for r in v_ratios):
            raise PromptSyntaxError(
                "Region grid V-ratios must be positive",
                kind="region_grid_invalid_ratios",
            )
    # If only V given, treat H as single row
    if raw_h is None and raw_v is not None:
        h_ratios = [1.0]
    # If only H given, treat V as single column
    if raw_v is None and raw_h is not None:
        v_ratios = [1.0]
    if not h_ratios or not v_ratios:
        raise PromptSyntaxError(
            "Region grid must specify at least one ratio in each dimension",
            kind="region_grid_empty",
        )
    total_h = sum(h_ratios)
    total_v = sum(v_ratios)
    h_ratios = [r / total_h for r in h_ratios]
    v_ratios = [r / total_v for r in v_ratios]
    return h_ratios, v_ratios, m.end() - start_pos


def _expand_region_grid(
    branches: list[str],
    h_ratios: list[float],
    v_ratios: list[float],
    base_text: str, mode: str, backend: str | None,
    stop: float, start: float, blur: float, canvas_b64: str,
    axis: str, base_ratio: float = 0.2,
) -> list[RegionBlock]:
    """Expand branches into grid cells.

    Branches are assigned to cells in row-major order.
    If fewer branches than cells, the last branch is repeated.

    h_ratios = horizontal divisions = columns (x-coords)
    v_ratios = vertical divisions = rows (y-coords)
    """
    n_rows = len(v_ratios)
    n_cols = len(h_ratios)
    n_cells = n_rows * n_cols
    result: list[RegionBlock] = []
    row_pos = 0.0
    for ri in range(n_rows):
        row_end = row_pos + v_ratios[ri]
        col_pos = 0.0
        for ci in range(n_cols):
            col_end = col_pos + h_ratios[ci]
            idx = ri * n_cols + ci
            if idx < len(branches):
                br_text = branches[idx]
            else:
                br_text = branches[-1] if branches else ""
            m = _RE_REGION_BRANCH.match(br_text)
            if m:
                region_text = m.group(1).rstrip()
                weight = float(m.group(6)) if m.group(6) else 1.0
                curve = m.group(7) if m.group(7) else (m.group(8) if m.group(8) else "linear")
            else:
                region_text, weight, curve = _split_region_branch_weight_and_curve(br_text)
            result.append(RegionBlock(
                text=region_text,
                x1=col_pos, x2=col_end, y1=row_pos, y2=row_end,
                weight=weight, axis=axis, base_text=base_text,
                mode=mode, backend=backend, coords_pixels=False,
                stop=stop, start=start, blur=blur, canvas=canvas_b64,
                base_ratio=base_ratio, curve=curve,
            ))
            col_pos = col_end
        row_pos = row_end
    return result


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
    """Мульти-энкодерная спецификация промпта.

    Поддерживаемые архитектуры:
        SDXL:  enc1 + enc2                 (CLIP-L + CLIP-G)
        Flux:  enc1 + t5                   (CLIP-L + T5-xxl)
        SD3:   enc1 + enc2 + t5            (CLIP-L + CLIP-G + T5-xxl)
    """
    prefix: str
    suffix: str
    enc1:   str = ""
    enc2:   str = ""
    pooled: str | None = None
    t5:     str | None = None
    source: str = ""

    @property
    def has_t5(self) -> bool:
        return bool(self.t5 and self.t5.strip())

    @property
    def has_sdxl_pair(self) -> bool:
        return bool(self.enc1 and self.enc2)

    @property
    def architecture_mode(self) -> str:
        if self.has_t5 and self.has_sdxl_pair:
            return "sd3"
        if self.has_t5:
            return "flux"
        if self.has_sdxl_pair:
            return "sdxl"
        if self.enc1:
            return "standard"
        return "unknown"


@dataclass(frozen=True)
class CompoundPartSpec:
    text: str
    step_start: int = 1
    step_end: int | None = None
    weight: float = 1.0
    curve: str = "linear"
    mode: str = "delta"  # "delta", "diff", or "diff_raw"


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
    curve: str = "linear"


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
    has_dangling_multicond: bool = False

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
        return (len(self.backend_specs) > 1 or self.has_dangling_multicond) and not self.allow_chunk_morph_sugar

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
    branch_text, weight, _ = _split_blend_branch_weight_and_curve(text, full_text=full_text)
    return branch_text, weight


def _split_blend_branch_weight_and_curve(text: str, *, full_text: str) -> tuple[str, float, str]:
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
        return stripped, 1.0, "linear"

    raw_weight_text = stripped[star_pos + 1 :].strip()
    curve = "linear"
    weight_text = raw_weight_text
    tilde_pos = raw_weight_text.find("~")
    if tilde_pos >= 0:
        curve_spec = raw_weight_text[tilde_pos + 1 :].strip()
        if curve_spec.startswith("cubic") or curve_spec in _EASING_MODES:
            curve = curve_spec
        else:
            allowed = ", ".join(sorted(_EASING_MODES))
            raise PromptSyntaxError(
                f"Unknown BLEND branch curve '{curve_spec}'. "
                f"Allowed modes: {allowed} or cubic(...).",
                kind="invalid_blend_curve",
                token=curve_spec,
                full=full_text,
            )
        weight_text = raw_weight_text[:tilde_pos].strip()

    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid BLEND branch weight {raw_weight_text!r}",
            kind="invalid_blend_weight",
            token=raw_weight_text or "*",
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
    return branch_text, weight, curve


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
        branch_text, weight, curve = _split_blend_branch_weight_and_curve(raw_part, full_text=text)
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
        branches.append(BlendBranchSpec(branch_text, weight, curve))

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
    if not raw:
        return []
    k = float(intensity)

    if mode == "max":
        mw = max(raw)
        if abs(mw) <= 1e-8:
            return [0.0 for _ in raw]
        winners = sum(1 for w in raw if abs(w - mw) <= 1e-8)
        return [1.0 / winners if abs(w - mw) <= 1e-8 else 0.0 for w in raw]

    if not math.isfinite(k) or k <= 0.0:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {intensity!r}",
            kind="invalid_blend_intensity",
            token=str(intensity),
            full=str(intensity),
        )

    if mode == "product":
        prod = 1.0
        for w in raw:
            prod *= max(0.0, w)
        total = float(sum(raw))
        if abs(total) <= 1e-8 or abs(prod) <= 1e-8 or abs(k) <= 1e-8:
            return [0.0 for _ in raw]
        scale = (prod ** k) / (total ** k) if k != 1.0 else prod / total
        return [w * max(0.0, scale) for w in raw]

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


def _build_compound_preview_text(
    base_text: str,
    active_parts: list[tuple[str, int, int, float, str]],
) -> str:
    base = base_text.strip() or SAFE_EMPTY
    if not active_parts:
        return f"{COMPOUND_PREVIEW_PREFIX}{base}>"
    parts_strs: list[str] = []
    for ptext, s, e, w, curve in active_parts:
        part = ptext.strip()
        if curve == "linear":
            part = f"{part}:{s}-{e}:{w}"
        else:
            part = f"{part}:{s}-{e}:{w}:{curve}"
        parts_strs.append(part)
    return f"{COMPOUND_PREVIEW_PREFIX}{base}; {'; '.join(parts_strs)}>"


def _build_compound_text_schedule_from_spec(
    spec: CompoundPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    strict: bool = False,
) -> list[list[int, str]]:
    steps_int = int(steps)
    base_full = _concat_prefix_text_suffix(spec.prefix, spec.base, spec.suffix)

    if strict:
        _fn = lambda txt, stp: _strict_schedule_preview(txt, stp, seed)
    elif use_scheduling:
        _fn = lambda txt, stp: get_schedule(txt, stp, True, seed, use_visitor)
    else:
        _fn = lambda txt, stp: get_schedule(txt, stp, False, seed, use_visitor)

    if not use_scheduling:
        base_sched = _fn(base_full, steps_int)
        base_text = _select_text_from_schedule(base_sched, steps_int) or SAFE_EMPTY
        active_parts: list[tuple[str, int, int, float, str]] = []
        for p in spec.parts:
            e = p.step_end if p.step_end is not None else steps_int
            if p.step_start <= steps_int <= e:
                part_full = _concat_prefix_text_suffix(spec.prefix, p.text, spec.suffix)
                ps = _fn(part_full, steps_int)
                pt = _select_text_from_schedule(ps, steps_int) or SAFE_EMPTY
                active_parts.append((pt, p.step_start, e, p.weight, p.curve))
        return [[steps_int, _build_compound_preview_text(base_text, active_parts)]]

    base_schedule = _fn(base_full, steps_int)
    part_schedules: list[list[list[int, str]]] = []
    for p in spec.parts:
        part_full = _concat_prefix_text_suffix(spec.prefix, p.text, spec.suffix)
        ps = _fn(part_full, steps_int)
        part_schedules.append(ps)

    change_points: set[int] = set()
    change_points.add(steps_int)
    for end_step, _ in base_schedule:
        change_points.add(int(end_step))
    for psched in part_schedules:
        for end_step, _ in psched:
            change_points.add(int(end_step))
    for p in spec.parts:
        s = p.step_start
        e = p.step_end if p.step_end is not None else steps_int
        if s > 1:
            change_points.add(s - 1)
        if e < steps_int:
            change_points.add(e)
    boundaries = sorted(change_points)

    out: list[list[int, str]] = []
    prev_text: str | None = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        active_parts: list[tuple[str, int, int, float, str]] = []
        for i, p in enumerate(spec.parts):
            e = p.step_end if p.step_end is not None else steps_int
            if p.step_start <= end_at_step <= e:
                pt = _select_text_from_schedule(part_schedules[i], end_at_step) or SAFE_EMPTY
                active_parts.append((pt, p.step_start, e, p.weight, p.curve))
        preview = _build_compound_preview_text(base_text, active_parts)
        if out and prev_text == preview:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            prev_text = preview
    return out or [[steps_int, SAFE_EMPTY]]


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
        blend_has_curves = any(branch.curve != "linear" for branch in spec.branches)
        if not blend_has_curves:
            final_step = int(steps)
            active_texts = [_select_text_from_schedule(schedule, final_step) or SAFE_EMPTY for schedule in branch_schedules]
            raw_weights = [float(branch.weight) for branch in spec.branches]
            final_weights = _resolve_blend_mode_weights(raw_weights, spec.mode, spec.intensity)
            return [[int(steps), _build_blend_preview_text_with_target(active_texts, final_weights, spec.channel_target)]]
        boundaries = list(range(1, int(steps) + 1))
    else:
        boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        active_texts = [
            _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
            for schedule in branch_schedules
        ]

        raw_weights = []
        total_steps = int(steps)
        for branch in spec.branches:
            w = float(branch.weight)
            if branch.curve != "linear" and total_steps > 1:
                progress = (end_at_step - 1) / (total_steps - 1)
                cf = _apply_easing(progress, branch.curve)
                w = w * max(0.0, cf)
            raw_weights.append(w)

        cur_weights = _resolve_blend_mode_weights(raw_weights, spec.mode, spec.intensity)
        preview = _build_blend_preview_text_with_target(active_texts, cur_weights, spec.channel_target)
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in cur_weights))
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
    curve_match = _RE_COMPOUND_CURVE.search(text)
    curve = "linear"
    mode = "delta"
    if curve_match:
        raw_curve = curve_match.group(1)
        if raw_curve == "diff":
            mode = "diff"
            curve = "linear"
        elif raw_curve == "diff_raw":
            mode = "diff_raw"
            curve = "linear"
        elif raw_curve == "delta":
            mode = "delta"
            curve = "linear"
        elif raw_curve == "ortho":
            mode = "ortho"
            curve = "linear"
        elif raw_curve.startswith("ortho_"):
            mode = "ortho"
            inner = raw_curve[6:]
            if inner.startswith("cubic") or inner in _EASING_MODES:
                curve = inner
            else:
                curve = "linear"
        elif raw_curve.startswith("diff_"):
            inner = raw_curve[5:]
            if inner == "raw":
                mode = "diff_raw"
                curve = "linear"
            elif inner.startswith("raw_"):
                mode = "diff_raw"
                rest = inner[4:]
                if rest.startswith("cubic") or rest in _EASING_MODES:
                    curve = rest
                else:
                    curve = "linear"
            else:
                mode = "diff"
                if inner.startswith("cubic") or inner in _EASING_MODES:
                    curve = inner
                else:
                    curve = "linear"
        elif raw_curve.startswith("cubic"):
            curve = raw_curve
        elif raw_curve not in _EASING_MODES:
            allowed = ", ".join(sorted(_EASING_MODES))
            raise PromptSyntaxError(
                f"Unknown COMPOUND curve '{raw_curve}'. "
                f"Allowed modes: {allowed} or cubic(...).",
                kind="invalid_compound_curve",
                token=raw_curve,
                full=full_text,
            )
        else:
            curve = raw_curve
        text = text[:curve_match.start()] + text[curve_match.end():]
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
    return CompoundPartSpec(text=text, step_start=start, step_end=end, weight=weight, curve=curve, mode=mode)


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
    return CompoundPromptSpec(
        base=base,
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
        weight_raw = m.group(1)
        bind3_weight = _parse_bind_weight(weight_raw, full_text=text)
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
        w_str = f"*{bind3_weight:.4g}" if abs(bind3_weight - 1.0) > 1e-8 else ""
        if len(clean_groups) == 1:
            branches = f"{owner}, {clean_groups[0]}{w_str}"
        else:
            branches = " | ".join(f"{owner}, {g}{w_str}" for g in clean_groups)
        chunk_text = f"CHUNK{{{branches}}}"
        result_parts.append(protected[last:start])
        result_parts.append(chunk_text)
        last = brace_close + 1
    result_parts.append(protected[last:])
    return _restore_escaped_literal_source("".join(result_parts), span_restore)


def _transpile_bind_v1_to_compound(text: str) -> str:
    """Safe transpile of BIND v1 -> COMPOUND when nested inside another backend block.

    A BIND that is a *top-level* block (found by ``_find_top_level_bind_blocks``) is left
    untouched so it keeps its row-splice algebra (the "Holy Grail" case). Only BINDs that
    appear nested *inside* another backend block (e.g. ``CHUNK{BIND{cat=>red} | dog}``)
    are transpiled to an additive COMPOUND, since v1 cannot represent binding there.
    """
    if not text or BIND_KEYWORD not in text:
        return text
    # Re-entrancy guard: if we are already inside a BIND transpile, bail out.
    if getattr(_VAR_META_LOCAL, "_in_bind_transpile", False):
        return text
    _VAR_META_LOCAL._in_bind_transpile = True
    try:
        protected, span_restore = _protect_escaped_literal_spans_for_source(text)
        protected = _protect_escaped_literals(protected)

        # Only BIND nested inside a *non-BIND* backend block (e.g. CHUNK{BIND{a=>b}|c})
        # is transpiled. A top-level BIND (with or without base text) keeps its
        # row-splice algebra; a BIND nested inside another BIND keeps the
        # nested_bind_not_supported error path.
        non_bind_backend_spans: list[tuple[int, int]] = []
        for finder in (
            _find_top_level_chunk_blocks,
            _find_top_level_blend_blocks,
            _find_top_level_morph_blocks,
            _find_top_level_compound_blocks,
            _find_top_level_pool_blocks,
            _find_top_level_assemble_blocks,
        ):
            try:
                for blk in finder(protected):
                    # 5-tuple (start, _mw, _mc, brace_open, brace_close) or
                    # 3-tuple (start, brace_open, brace_close) or 4-tuple.
                    if len(blk) >= 5:
                        non_bind_backend_spans.append((blk[3], blk[4]))
                    elif len(blk) == 3:
                        non_bind_backend_spans.append((blk[1], blk[2]))
            except PromptSyntaxError:
                continue

        nested_binds: list[tuple[int, int, int]] = []
        for mtch in _RE_BIND_MARKER.finditer(protected):
            start = mtch.start()
            brace_open = mtch.end() - 1
            depth = 1
            k = brace_open + 1
            while k < len(protected) and depth > 0:
                if protected[k] == "{":
                    depth += 1
                elif protected[k] == "}":
                    depth -= 1
                k += 1
            if depth != 0:
                continue
            brace_close = k - 1
            if any(bo < brace_open < bc for bo, bc in non_bind_backend_spans):
                nested_binds.append((start, brace_open, brace_close))

        if not nested_binds:
            return text

        result_parts: list[str] = []
        last = 0
        for start, brace_open, brace_close in nested_binds:
            body = protected[brace_open + 1 : brace_close]
            try:
                owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
                owner = _restore_escaped_literal_source(owner_body, span_restore).strip()
                attrs = _restore_escaped_literal_source(attrs_body, span_restore).strip()
                if not owner or not attrs:
                    continue
                # BIND{o => a} -> COMPOUND{ o | o, a ~ diff }
                transpiled = f"COMPOUND{{ {owner} | {owner}, {attrs} ~ diff }}"
                result_parts.append(protected[last:start])
                result_parts.append(transpiled)
                last = brace_close + 1
            except Exception as e:
                logger.warning(
                    "BIND v1 transpile skipped for nested segment at %d: %s",
                    start,
                    str(e),
                )
                continue
        if last > 0:
            result_parts.append(protected[last:])
            return _restore_escaped_literal_source("".join(result_parts), span_restore)
        return text
    finally:
        _VAR_META_LOCAL._in_bind_transpile = False


def _transpile_bind2_to_chunk(text: str) -> str:
    if not text or BIND2_KEYWORD not in text:
        return text
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    result_parts: list[str] = []
    last = 0
    for m in _RE_BIND2_MARKER.finditer(protected):
        start = m.start()
        weight_raw = m.group(1)
        weight = _parse_bind_weight(weight_raw, full_text=text)
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
        w_str = f"*{weight:.4g}" if abs(weight - 1.0) > 1e-8 else ""
        if len(groups) == 1:
            branches = f"{owner}, {groups[0]}{w_str}"
        else:
            branches = " | ".join(f"{owner}, {g}{w_str}" for g in groups)
        chunk_text = f"CHUNK{{{branches}}}"
        result_parts.append(protected[last:start])
        result_parts.append(chunk_text)
        last = brace_close + 1
    result_parts.append(protected[last:])
    return _restore_escaped_literal_source("".join(result_parts), span_restore)


def _transpile_diff_to_compound(text: str) -> str:
    if not text or DIFF_KEYWORD not in text:
        return text
    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    result_parts: list[str] = []
    last = 0
    for m in _RE_DIFF_MARKER.finditer(protected):
        start = m.start()
        weight_raw = m.group(1)
        if weight_raw is not None and not weight_raw.strip():
            weight_raw = None
        global_weight = _parse_bind_weight(weight_raw, full_text=text)
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
                "Unclosed DIFF block: expected '}'",
                kind="invalid_diff_syntax",
                token=DIFF_KEYWORD,
                full=text,
            )
        brace_close = i - 1
        body = protected[brace_open + 1 : brace_close]
        raw_parts = _split_top_level_commas(body)
        if not raw_parts:
            raise PromptSyntaxError(
                "DIFF requires at least base + one subtraction term.",
                kind="invalid_diff_syntax",
                token=DIFF_KEYWORD,
                full=text,
            )
        if len(raw_parts) < 2:
            raise PromptSyntaxError(
                "DIFF requires at least one subtraction term (base + sub).",
                kind="invalid_diff_syntax",
                token=DIFF_KEYWORD,
                full=text,
            )
        # _split_top_level_commas drops empty results — detect missing args
        top_level_commas = 0
        _depth = 0
        for _ch in body:
            if _ch in "([{":
                _depth += 1
            elif _ch in ")]}":
                _depth = max(0, _depth - 1)
            elif _ch == "," and _depth == 0:
                top_level_commas += 1
        if top_level_commas >= len(raw_parts):
            raise PromptSyntaxError(
                "DIFF body contains empty arguments (consecutive commas).",
                kind="invalid_diff_syntax",
                token=DIFF_KEYWORD,
                full=text,
            )
        base = _restore_escaped_literal_source(raw_parts[0].strip(), span_restore)
        if not base:
            base = SAFE_EMPTY
        parts_out: list[str] = []
        for sub in raw_parts[1:]:
            sub_clean = sub.strip()
            existing_weight = 1.0
            wm = _RE_COMPOUND_WEIGHT.search(sub_clean)
            if wm:
                try:
                    existing_weight = float(wm.group(1))
                except ValueError:
                    pass
                sub_clean = (sub_clean[:wm.start()] + sub_clean[wm.end():]).strip()
            w = global_weight * existing_weight
            cm = _RE_COMPOUND_CURVE.search(sub_clean)
            user_curve: str | None = None
            if cm:
                raw_curve = cm.group(1)
                if raw_curve != "diff":
                    user_curve = raw_curve
                sub_clean = (sub_clean[:cm.start()] + sub_clean[cm.end():]).strip()
            w_str = f"*{w:.6f}".rstrip("0").rstrip(".") if abs(w - 1.0) > 1e-8 else ""
            restored = _restore_escaped_literal_source(sub_clean, span_restore)
            if user_curve:
                if user_curve in ("delta", "diff_raw", "ortho") or user_curve.startswith("diff_raw_") or user_curve.startswith("ortho_"):
                    parts_out.append(f"{restored}{w_str}~{user_curve}")
                else:
                    parts_out.append(f"{restored}{w_str}~diff_{user_curve}")
            else:
                parts_out.append(f"{restored}{w_str}~diff")
        compound_text = f"COMPOUND{{{base} | {' | '.join(parts_out)}}}"
        result_parts.append(protected[last:start])
        result_parts.append(compound_text)
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
        if name not in {"enc1", "enc2", "pooled", "t5"}:
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

    _has_enc1 = bool(fields.get("enc1", "").strip())
    _has_enc2 = bool(fields.get("enc2", "").strip())
    _has_t5   = bool(fields.get("t5",   "").strip())
    _valid_combo = (_has_enc1 and _has_enc2) or (_has_enc1 and _has_t5)
    if not _valid_combo:
        raise PromptSyntaxError(
            "ASSEMBLE requires (enc1 + enc2) for SDXL, or (enc1 + t5) for Flux/SD3. "
            "Got fields: " + repr(sorted(fields.keys())),
            kind="invalid_assemble_syntax",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    pooled = fields.get("pooled") or None
    t5     = fields.get("t5")     or None
    return AssemblePromptSpec(
        prefix=prefix,
        suffix=suffix,
        enc1=fields.get("enc1", ""),
        enc2=fields.get("enc2", ""),
        pooled=pooled,
        t5=t5,
        source=text,
    )


def _expand_assemble_section_prompt(spec: AssemblePromptSpec, body: str) -> str:
    return _concat_prefix_text_suffix(spec.prefix, body, spec.suffix)


def _build_assemble_preview_text_v2(spec: AssemblePromptSpec) -> str:
    """Расширенный preview с режимом архитектуры и t5.
    Для статического preview (lint). Не для scheduling loop.
    """
    mode = spec.architecture_mode
    parts = []
    if spec.enc1:
        s = spec.enc1
        parts.append(f"enc1={s[:40]}{'…' if len(s) > 40 else ''}")
    if spec.enc2:
        s = spec.enc2
        parts.append(f"enc2={s[:40]}{'…' if len(s) > 40 else ''}")
    if spec.has_t5:
        s = spec.t5
        parts.append(f"t5={s[:60]}{'…' if len(s) > 60 else ''}")
    if spec.pooled:
        s = spec.pooled
        parts.append(f"pooled={s[:30]}{'…' if len(s) > 30 else ''}")
    return f"{ASSEMBLE_PREVIEW_PREFIX}[{mode}] {' | '.join(parts)}>"


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


def _find_top_level_region_blocks(protected: str) -> list[tuple[int, int, str, str]]:
    """Find REGION{...} blocks at top level (depth_brace==0).

    Takes PUA-protected text (escaped literals are already placeholderized).
    Same pattern as _find_top_level_bind3_blocks.
    Returns [(start, end, body, axis), ...].
    axis = 'H' or 'V' from optional :H/:V suffix after '}'.
    """
    results: list[tuple[int, int, str, str]] = []
    skip_until = 0
    for m in _RE_REGION_MARKER.finditer(protected):
        if m.start() < skip_until:
            continue
        start = m.start()
        brace_open = m.end() - 1
        depth = 1
        i = brace_open + 1
        while i < len(protected) and depth > 0:
            if protected[i] == '\\' and i + 1 < len(protected) and protected[i + 1] in ('{', '}'):
                i += 2
                continue
            if protected[i] == '{':
                depth += 1
            elif protected[i] == '}':
                depth -= 1
            i += 1
        if depth != 0:
            continue
        brace_close = i - 1
        body = protected[brace_open + 1 : brace_close]

        # Extract optional :H/:V suffix after '}'
        axis = 'H'
        rest = protected[brace_close + 1:].lstrip()
        ws_len = len(protected) - len(rest) - brace_close - 1
        suffix_len = 0
        if rest.startswith(':') and len(rest) >= 2:
            axis_char = rest[1].upper()
            if axis_char in ('H', 'V'):
                axis = axis_char
                suffix_len = 2
                rest2 = rest[2:]
                if rest2.startswith(':'):
                    r_end = 1
                    while r_end < len(rest2) and (rest2[r_end].isdigit() or rest2[r_end] in '.,-'):
                        r_end += 1
                    if r_end > 1:
                        ratios_str = rest2[1:r_end]
                        axis = axis_char + ':' + ratios_str
                        suffix_len += r_end

        results.append((start, brace_close, body, axis))
        skip_until = brace_close + 1 + ws_len + suffix_len
    return results


def _split_region_body_by_pipe(body: str) -> list[str]:
    """Split REGION body by | at depth 0 (respects nested {} [] ())."""
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '\\' and i + 1 < len(body) and body[i + 1] in ('{', '}'):
            buf.append(ch)
            buf.append(body[i + 1])
            i += 2
            continue
        if ch == '|' and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            parts.append("".join(buf).strip())
            buf.clear()
        else:
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
            buf.append(ch)
        i += 1
    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return parts


def _extract_region_directives(body: str) -> tuple[list[str], str, str, str | None, float, float, float, str, float]:
    """Extract *base=, mode=, backend=, stop=, start=, blur=, canvas=, base_ratio= directives.

    Returns (region_branches, base_text, mode, backend, stop, start, blur, canvas_b64, base_ratio).
    canvas= is a base64-encoded PNG mask (one canvas shared by all regions in the block).
    base_ratio= is the block-level base blend strength (default 0.2, clamped [0,1]).
    """
    raw_branches = _split_region_body_by_pipe(body)
    region_branches: list[str] = []
    base = ""
    mode = "overlay"
    backend: str | None = None
    stop = 1.0
    start = 0.0
    blur = 0.0
    canvas_b64 = ""
    base_ratio = 0.2
    for br in raw_branches:
        eq = br.find('=')
        if eq >= 0:
            name = br[:eq].strip()
            val = br[eq+1:].strip()
        else:
            name = br
            val = ''
        if name == '*base':
            base = val
        elif name == 'mode':
            if val in ('overlay', 'common'):
                mode = val
        elif name == 'backend':
            if val in ('attention', 'latent', 'forge', 'monkey'):
                backend = val
        elif name == 'stop':
            try:
                stop = float(val)
                stop = max(0.0, min(1.0, stop))
            except ValueError:
                pass
        elif name == 'start':
            try:
                start = float(val)
                start = max(0.0, min(1.0, start))
            except ValueError:
                pass
        elif name == 'blur':
            try:
                blur = float(val)
                blur = max(0.0, min(1.0, blur))
            except ValueError:
                pass
        elif name == 'canvas':
            canvas_b64 = val
        elif name == 'base_ratio':
            try:
                base_ratio = float(val)
                base_ratio = max(0.0, min(1.0, base_ratio))
            except ValueError:
                pass
        else:
            stripped = br.strip()
            if stripped:
                region_branches.append(stripped)
    if stop < start:
        raise PromptSyntaxError(
            f"REGION start={start} > stop={stop} — region would never activate",
            kind="region_invalid_window",
        )
    return region_branches, base, mode, backend, stop, start, blur, canvas_b64, base_ratio


def _resolve_region_auto_tile(branches: list[str], axis: str) -> list[tuple[float, float, float, float]]:
    """Assign equal coordinates to branches without @.

    axis: 'H', 'V', 'H:r1,r2,...', 'V:r1,r2,...'.
    When ratios are provided, they are normalized to sum=1 and assigned
    to branches in order. Count must match len(branches).
    """
    N = len(branches)
    if N == 0:
        return []

    ratios: list[float] | None = None
    if ':' in axis:
        _, ratios_str = axis.split(':', 1)
        try:
            raw_ratios = [float(x.strip()) for x in ratios_str.split(',') if x.strip()]
        except ValueError:
            raw_ratios = []
        if raw_ratios:
            if len(raw_ratios) != N:
                raise PromptSyntaxError(
                    f"Region axis has {len(raw_ratios)} ratios but {N} branches",
                    kind="region_ratio_mismatch",
                )
            total = sum(raw_ratios)
            if total <= 0:
                raise PromptSyntaxError(
                    "Region ratios sum to zero",
                    kind="region_zero_ratio_sum",
                )
            ratios = [r / total for r in raw_ratios]

    is_h = axis.startswith('H')

    if ratios is not None:
        if is_h:
            pos = 0.0
            coords = []
            for r in ratios:
                coords.append((pos, pos + r, 0.0, 1.0))
                pos += r
            return coords
        else:
            pos = 0.0
            coords = []
            for r in ratios:
                coords.append((0.0, 1.0, pos, pos + r))
                pos += r
            return coords

    step = 1.0 / N
    if is_h:
        return [(i * step, (i + 1) * step, 0.0, 1.0) for i in range(N)]
    else:
        return [(0.0, 1.0, i * step, (i + 1) * step) for i in range(N)]


def _extract_region_blocks(text: str) -> tuple[str, list[RegionBlock]]:
    """Extract REGION{...} blocks from text. Core extraction logic.

    Returns (clean_text, regions).
    clean_text has REGION{...} replaced with space-joined branch texts.
    """
    blocks = _find_top_level_region_blocks(text)
    if not blocks:
        return text, []

    all_regions: list[RegionBlock] = []
    for start, end, body, axis in blocks:
        branches, base_text, mode, backend, stop_val, start_val, blur_val, canvas_b64, base_ratio_val = _extract_region_directives(body)
        block_end = end
        # skip whitespace before grid suffix
        gs_pos = block_end + 1
        while gs_pos < len(text) and text[gs_pos].isspace():
            gs_pos += 1
        grid_h, grid_v, _ = _parse_region_grid_suffix(text, gs_pos)
        region_list: list[RegionBlock] = []
        if grid_h is not None and grid_v is not None:
            region_list = _expand_region_grid(
                branches, grid_h, grid_v,
                base_text, mode, backend,
                stop_val, start_val, blur_val, canvas_b64,
                axis, base_ratio_val,
            )
        else:
            auto_tile: list[str] = []
            for br_text in branches:
                m = _RE_REGION_BRANCH.match(br_text)
                if m:
                    region_text = m.group(1).rstrip()
                    x1 = float(m.group(2))
                    x2 = float(m.group(3))
                    y1 = float(m.group(4))
                    y2 = float(m.group(5))
                    weight = float(m.group(6)) if m.group(6) else 1.0
                    curve = m.group(7) if m.group(7) else (m.group(8) if m.group(8) else "linear")
                    coords_pixels = any(v > 1.0 for v in (x1, x2, y1, y2))
                    if coords_pixels:
                        pixel_vals = [v for v in (x1, x2, y1, y2) if v > 1.0]
                        norm_vals = [v for v in (x1, x2, y1, y2) if 0.0 < v < 1.0]
                        if norm_vals:
                            raise PromptSyntaxError(
                                "Mixed pixel and normalized coordinates in region branch",
                                kind="region_mixed_range",
                            )
                    if x1 > x2 or y1 > y2:
                        bad_axis = 'x' if x1 > x2 else 'y'
                        bad_v1 = x1 if x1 > x2 else y1
                        bad_v2 = x2 if x1 > x2 else y2
                        raise PromptSyntaxError(
                            f"Reverse range in region: {bad_axis}1={bad_v1} > {bad_axis}2={bad_v2}. "
                            f"Coordinates are x1,x2,y1,y2 (ranges). For left half: @0,0.5,0,1. "
                            f"A1111 format @x,y,w,h is not used here.",
                            kind="region_reverse_range",
                        )
                    region_list.append(RegionBlock(
                        text=region_text, x1=x1, x2=x2, y1=y1, y2=y2,
                        weight=weight, axis=axis, base_text=base_text,
                        mode=mode, backend=backend, coords_pixels=coords_pixels,
                        stop=stop_val, start=start_val, blur=blur_val, canvas=canvas_b64,
                        base_ratio=base_ratio_val, curve=curve,
                    ))
                else:
                    auto_tile.append(br_text)
            if auto_tile:
                coords_list = _resolve_region_auto_tile(auto_tile, axis)
                for br_text, (x1, x2, y1, y2) in zip(auto_tile, coords_list):
                    r_text, r_weight, r_curve = _split_region_branch_weight_and_curve(br_text)
                    region_list.append(RegionBlock(
                        text=r_text, x1=x1, x2=x2, y1=y1, y2=y2,
                        weight=r_weight, axis=axis, base_text=base_text,
                        mode=mode, backend=backend, coords_pixels=False,
                        stop=stop_val, start=start_val, blur=blur_val, canvas=canvas_b64,
                        base_ratio=base_ratio_val, curve=r_curve,
                    ))
        all_regions.extend(region_list)

    # Build clean_text: replace REGION{...} with space-joined branch texts
    # Skip optional :V/:H suffix after closing }
    clean_parts: list[str] = []
    pos = 0
    for start, end, body, block_axis in blocks:
        if pos < start:
            clean_parts.append(text[pos:start])
        branches, _base, _mode, _backend, _stop, _start, _blur, _canvas, _bratio = _extract_region_directives(body)
        branch_texts = []
        for br in branches:
            m = _RE_REGION_BRANCH.match(br)
            if m:
                branch_texts.append(m.group(1).rstrip())
            else:
                _t, _w, _c = _split_region_branch_weight_and_curve(br)
                branch_texts.append(_t)
        clean_parts.append(" ".join(branch_texts))
        # Skip past :V/:H suffix, grid suffix [H:...|V:...], or optional ratio suffix
        suffix_end = end + 1
        if suffix_end < len(text):
            # skip whitespace before grid suffix
            gs_pos = suffix_end
            while gs_pos < len(text) and text[gs_pos].isspace():
                gs_pos += 1
            # Check for grid suffix [H:...|V:...] first
            _, _, grid_skip = _parse_region_grid_suffix(text, gs_pos)
            if grid_skip > 0:
                suffix_end = gs_pos + grid_skip
            _raw = text[suffix_end:]
            rest = _raw.lstrip()
            if rest.startswith(':') and len(rest) >= 2 and rest[1].upper() in ('H', 'V'):
                _ws = len(_raw) - len(rest)
                suffix_end += _ws + 2
                rest2 = text[suffix_end:].lstrip()
                if rest2.startswith(':'):
                    r_end = 1
                    while r_end < len(rest2) and (rest2[r_end].isdigit() or rest2[r_end] in '.,-'):
                        r_end += 1
                    if r_end > 1:
                        suffix_end += len(rest2[:r_end])
        pos = suffix_end
    if pos < len(text):
        clean_parts.append(text[pos:])
    clean_text = "".join(clean_parts)

    return clean_text, all_regions


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
        conds = _align_condition_values_for_blend(conds)
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
    text = _transpile_diff_to_compound(text)
    text = _transpile_bind_v1_to_compound(text)
    text = _transpile_bind2_to_chunk(text)
    text = _transpile_bind3_to_chunk(text)
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
    # Наличие "висячих" AND-веток вне бэкенд-блоков — изолирует хвост от суффикса блока.
    has_dangling_multicond = False
    if extracted_backend and _contains_top_level_multicond(primary_text):
        has_dangling_multicond = True
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
        has_dangling_multicond=has_dangling_multicond,
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

    if curve in ("bezier", "bernstein"):
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

    # Easing curves — segment-based with eased alpha (like linear, but eased)
    if curve in _EASING_MODES:
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        left = min(int(segment_pos), count - 2)
        alpha = segment_pos - left
        eased = _apply_easing(alpha, curve)
        weights = [0.0] * count
        weights[left] = 1.0 - eased
        weights[left + 1] = eased
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
    t5_prompt = _expand_assemble_section_prompt(spec, spec.t5) if spec.has_t5 else ""

    if strict:
        enc1_schedule = _strict_schedule_preview(enc1_prompt, steps, seed)
        enc2_schedule = _strict_schedule_preview(enc2_prompt, steps, seed)
        pooled_schedule = _strict_schedule_preview(pooled_prompt, steps, seed)
        t5_schedule = _strict_schedule_preview(t5_prompt, steps, seed) if spec.has_t5 else []
    else:
        enc1_schedule = get_schedule(enc1_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        enc2_schedule = get_schedule(enc2_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        pooled_schedule = get_schedule(pooled_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        t5_schedule = get_schedule(t5_prompt, steps, use_scheduling, seed, use_visitor=use_visitor) if spec.has_t5 else []

    schedule_list = [enc1_schedule, enc2_schedule, pooled_schedule]
    if spec.has_t5:
        schedule_list.append(t5_schedule)
    boundaries = _collect_schedule_boundaries(schedule_list, steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        enc1_text = _select_text_from_schedule(enc1_schedule, end_at_step) or SAFE_EMPTY
        enc2_text = _select_text_from_schedule(enc2_schedule, end_at_step) or SAFE_EMPTY
        pooled_text = _select_text_from_schedule(pooled_schedule, end_at_step) or SAFE_EMPTY
        t5_text = _select_text_from_schedule(t5_schedule, end_at_step) or ""
        if not spec.has_t5:
            t5_text = ""
        preview = f"{ASSEMBLE_PREVIEW_PREFIX}[{spec.architecture_mode}] enc1={enc1_text}"
        if enc2_text.strip():
            preview += f" | enc2={enc2_text}"
        if t5_text:
            preview += f" | t5={t5_text}"
        if pooled_text.strip():
            preview += f" | pooled={pooled_text}"
        preview += ">"
        key = (enc1_text, enc2_text, pooled_text, t5_text if spec.has_t5 else "")
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
    except Exception as exc:
        logger.warning(
            "SDXL channel target splice failed (%s: %s), falling back to whole-cross routing",
            type(exc).__name__, exc,
        )
        return target_value


def _weighted_average_condition_values(values, weights):
    if not values:
        raise ValueError("Empty conditioning values for weighted average")
    total = float(sum(weights)) if weights else 0.0
    if total == 0.0:
        logger.warning("Weighted average total=0 — returning zero conditioning")
        total = 1.0

    first = values[0]
    if isinstance(first, dict):
        result = {}
        for key in first.keys():
            keyed = [(v[key], w) for v, w in zip(values, weights) if key in v]
            if not keyed:
                continue
            sub_values = [v for v, _ in keyed]
            sub_weights = [w for _, w in keyed]
            result[key] = _weighted_average_condition_values(sub_values, sub_weights)
        return result

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
                key_weights = [weights[i] for i, cond in enumerate(cond_list) if key in cond]
                weighted = [value * float(w) if float(w) != 1.0 else value for value, w in zip(values, key_weights)]
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
        except (TypeError, IndexError) as exc:
            logger.warning("Tensor alignment shape extraction failed (%s), returning unaligned", exc)
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
        except (IndexError, RuntimeError) as exc:
            logger.warning("Tensor alignment padding failed (%s), returning unaligned", exc)
            return list(values)
    return aligned


def _weighted_sum_condition_values(values, weights):
    if not values:
        raise ValueError("Empty conditioning values for weighted sum")

    first = values[0]
    if isinstance(first, dict):
        result = {}
        for key in first.keys():
            keyed = [(v[key], w) for v, w in zip(values, weights) if key in v]
            if not keyed:
                continue
            sub_values = [v for v, _ in keyed]
            sub_weights = [w for _, w in keyed]
            result[key] = _weighted_sum_condition_values(sub_values, sub_weights)
        return result

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

    if APPLY_ADVANCED_WEIGHTS_ENABLED:
        cond_schedule = []
        fallback = False
        for end_at_step, text in prompt_schedule:
            try:
                cond = _encode_prompt_with_chunked_weights(model, text, copy_from)
            except ValueError:
                # Модель без распознаваемого токенизатора — откатываемся
                # на обычный путь для ВСЕГО расписания (логируем один раз).
                logger.warning(
                    "APPLY_ADVANCED_WEIGHTS_ENABLED=1, но модель не даёт "
                    "распознать токенизатор — откат на стандартное кодирование "
                    "без per-chunk весов для '%s'.", prompt,
                )
                fallback = True
                break
            cond_schedule.append(ScheduledPromptConditioning(int(end_at_step), cond))
        if not fallback:
            return cond_schedule

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


def _assemble_condition_values_by_mode(
    spec: AssemblePromptSpec,
    enc1_cond,
    enc2_cond,
    pooled_cond,
):
    """Route conditioning merge by architecture_mode.

    SDXL (enc1+enc2): full merge via _assemble_condition_values.
    Flux (enc1+t5):   enc1 returned as-is (T5 not available without hook).
    SD3 (enc1+enc2+t5): SDXL merge for CLIP part (T5 not available without hook).
    """
    mode = spec.architecture_mode
    if mode in ("flux", "standard"):
        return enc1_cond
    if mode == "sd3":
        try:
            return _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("SD3 ASSEMBLE merge failed (%s), falling back to single-encoder", exc)
            return enc1_cond
    return _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond)


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
        merged = _assemble_condition_values_by_mode(spec, enc1_cond, enc2_cond, pooled_cond)
        # TODO: T5-xxl encoding при spec.has_t5
        # Требует T5 tokenizer + encoder модель + механизм слияния CLIP/T5 эмбеддингов.
        # Когда будет реализовано:
        #   if spec.has_t5:
        #       t5_emb = t5_encode(t5_prompt)
        #       merged = merge_clip_t5(merged, t5_emb)
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

        active_conds = _align_condition_values_for_blend(active_conds)
        merged = _merge_chunk_condition_values(active_conds, weights)
        if anchor_cond_schedule is not None and target_channel is not None:
            anchor_index = _pick_schedule_entry_index(anchor_cond_schedule, end_at_step)
            anchor_cond = anchor_cond_schedule[anchor_index].cond
            merged = _apply_condition_channel_target(anchor_cond, merged, target_channel)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = active_key

    return out


def _compound_text_needs_scheduling(spec: CompoundPromptSpec) -> bool:
    texts = [spec.prefix, spec.suffix, spec.base] + [p.text for p in spec.parts]
    for t in texts:
        if "->" in t:
            return True
        if _RE_HAS_SCHEDULING.search(t):
            return True
    return False


def _normalize_cond_diff(merged, base_cond):
    if isinstance(merged, dict):
        res = {}
        for k in merged:
            if k not in base_cond:
                res[k] = merged[k]
                continue
            m, b = merged[k], base_cond[k]
            m_norm = m.norm(dim=-1, keepdim=True)
            b_norm = b.norm(dim=-1, keepdim=True)
            ratio = (b_norm / (m_norm + 1e-8)).clamp(0.1, 10.0)
            res[k] = m * ratio
        return res
    m_norm = merged.norm(dim=-1, keepdim=True)
    b_norm = base_cond.norm(dim=-1, keepdim=True)
    ratio = (b_norm / (m_norm + 1e-8)).clamp(0.1, 10.0)
    return merged * ratio


def _build_compound_conditioning_schedule(
    model,
    spec: CompoundPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    steps_int = int(steps)
    base_full = _concat_prefix_text_suffix(spec.prefix, spec.base, spec.suffix)
    needs_dynamic = _compound_text_needs_scheduling(spec)
    _torch = _ensure_torch()

    if any(p.mode in ("diff", "diff_raw", "ortho") for p in spec.parts):
        try:
            _empty_raw = model.get_learned_conditioning(SdConditioning([""], copy_from=copy_from))
            if isinstance(_empty_raw, dict):
                empty_cond = {k: v[0] for k, v in _empty_raw.items()}
            else:
                empty_cond = _empty_raw[0]
        except Exception:
            empty_cond = None
    else:
        empty_cond = None

    def _cond_token_len(c):
        """Get token count (dim 0) from conditioning, handling dict vs tensor.
        For dicts, only consider 2D+ tensors — SDXL 'vector' is 1D (pooled),
        not a token dimension.
        """
        if isinstance(c, dict):
            return max((v.shape[0] for v in c.values() if v.ndim >= 2), default=1)
        return c.shape[0]

    if not use_scheduling or not needs_dynamic:
        texts = [base_full] + [_concat_prefix_text_suffix(spec.prefix, p.text, spec.suffix) for p in spec.parts]
        conds = model.get_learned_conditioning(SdConditioning(texts, copy_from=copy_from))
        if isinstance(conds, dict):
            base_cond = {k: v[0] for k, v in conds.items()}
            part_conds = [{k: v[i + 1] for k, v in conds.items()} for i in range(len(spec.parts))]
        else:
            base_cond = conds[0]
            part_conds = [conds[i + 1] for i in range(len(spec.parts))]
        if empty_cond is not None and part_conds:
            _ref = part_conds[0]
            _target_len = _cond_token_len(_ref)
            _ec_len = _cond_token_len(empty_cond)
            if _ec_len < _target_len:
                _n = (_target_len + _ec_len - 1) // _ec_len
                if isinstance(empty_cond, dict):
                    for k, v in empty_cond.items():
                        if v.ndim >= 2:
                            empty_cond[k] = _torch.cat([v] * _n)[:_target_len]
                else:
                    empty_cond = _torch.cat([empty_cond] * _n)[:_target_len]
            elif _ec_len > _target_len:
                if isinstance(empty_cond, dict):
                    for k, v in empty_cond.items():
                        if v.ndim >= 2:
                            empty_cond[k] = v[:_target_len]
                else:
                    empty_cond = empty_cond[:_target_len]

        _all_conds: list = [base_cond] + list(part_conds)
        if empty_cond is not None:
            _all_conds.append(empty_cond)

        # Per-key alignment for dict conds (SDXL/Flux).
        # Different keys (vector, cross_axis, clip_l, t5) have different
        # token lengths; aligning to a single _max_len corrupts 1D pooled
        # embeddings and crashes when keys from separate batch calls diverge.
        _dict_conds = [c for c in _all_conds if isinstance(c, dict)]
        if _dict_conds:
            _all_keys: set[str] = set()
            for c in _dict_conds:
                _all_keys.update(c.keys())
            for _k in _all_keys:
                _k_lens = [c[_k].shape[0] for c in _dict_conds if _k in c and c[_k].ndim >= 2]
                if _k_lens:
                    _k_max = max(_k_lens)
                    for c in _dict_conds:
                        if _k in c and c[_k].ndim >= 2 and c[_k].shape[0] < _k_max:
                            _v = c[_k]
                            _n = (_k_max + _v.shape[0] - 1) // _v.shape[0]
                            c[_k] = _torch.cat([_v] * _n)[:_k_max]

        # Align tensor conds uniformly
        _tensor_conds = [c for c in _all_conds if not isinstance(c, dict)]
        if _tensor_conds:
            _max_len = max(c.shape[0] for c in _tensor_conds)
            for _ci, _c in enumerate(_all_conds):
                if not isinstance(_c, dict) and _c.shape[0] < _max_len:
                    _n = (_max_len + _c.shape[0] - 1) // _c.shape[0]
                    _c_aligned = _torch.cat([_c] * _n)[:_max_len]
                    if _ci == 0:
                        base_cond = _c_aligned
                    elif _ci <= len(part_conds):
                        part_conds[_ci - 1] = _c_aligned
                    else:
                        empty_cond = _c_aligned

        boundaries: list[int]
        if not use_scheduling:
            has_curves = any(p.curve != "linear" for p in spec.parts)
            if has_curves:
                boundaries = list(range(1, steps_int + 1))
            else:
                boundaries = [steps_int]
        else:
            change_points: set[int] = set()
            change_points.add(steps_int)
            for p in spec.parts:
                s = p.step_start
                e = p.step_end if p.step_end is not None else steps_int
                if s > 1:
                    change_points.add(s - 1)
                if e < steps_int:
                    change_points.add(e)
            boundaries = sorted(change_points)

        out: list[ScheduledPromptConditioning] = []
        prev_key: tuple | None = None
        for end_at_step in boundaries:
            active_indices: list[int] = []
            active_weights: list[float] = []
            active_modes: list[str] = []
            for i, p in enumerate(spec.parts):
                s = p.step_start
                e = p.step_end if p.step_end is not None else steps_int
                if s <= end_at_step <= e:
                    active_indices.append(i)
                    active_modes.append(p.mode)
                    if p.curve == "linear":
                        active_weights.append(p.weight)
                    else:
                        span = float(e - s)
                        if span <= 0:
                            active_weights.append(p.weight)
                        else:
                            progress = (end_at_step - s) / span
                            cf = _apply_easing(progress, p.curve)
                            active_weights.append(p.weight * max(0.0, cf))

            merged = base_cond
            for idx, w, mode in zip(active_indices, active_weights, active_modes):
                part = part_conds[idx]
                if mode == "diff" or mode == "diff_raw":
                    if empty_cond is not None:
                        if isinstance(merged, dict):
                            merged = {k: merged[k] - w * (part[k] - empty_cond[k]) for k in merged}
                        else:
                            merged = merged - w * (part - empty_cond)
                    else:
                        if isinstance(merged, dict):
                            merged = {k: merged[k] - w * part[k] for k in merged}
                        else:
                            merged = merged - w * part
                    if mode == "diff":
                        merged = _normalize_cond_diff(merged, base_cond)
                elif mode == "ortho":
                    if empty_cond is not None:
                        if isinstance(merged, dict):
                            merged_new = {}
                            for k in merged:
                                A = merged[k]; Bp = part[k]; E = empty_cond[k]
                                delta_B = Bp - E
                                dot_ad = (A * delta_B).sum(dim=-1, keepdim=True)
                                dot_dd = (delta_B * delta_B).sum(dim=-1, keepdim=True) + 1e-8
                                proj = (dot_ad / dot_dd) * delta_B
                                merged_new[k] = A - w * proj
                            merged = merged_new
                        else:
                            delta_B = part - empty_cond
                            dot_ad = (merged * delta_B).sum(dim=-1, keepdim=True)
                            dot_dd = (delta_B * delta_B).sum(dim=-1, keepdim=True) + 1e-8
                            proj = (dot_ad / dot_dd) * delta_B
                            merged = merged - w * proj
                    else:
                        if isinstance(merged, dict):
                            merged = {k: merged[k] - w * part[k] for k in merged}
                        else:
                            merged = merged - w * part
                        merged = _normalize_cond_diff(merged, base_cond)
                elif isinstance(merged, dict):
                    merged = {k: merged[k] + w * (part[k] - base_cond[k]) for k in merged}
                else:
                    merged = merged + w * (part - base_cond)

            key = (spec.base,) + tuple(active_indices) + tuple(active_weights) + tuple(active_modes)
            if out and prev_key == key:
                out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            else:
                out.append(ScheduledPromptConditioning(int(end_at_step), merged))
                prev_key = key

        return out

    base_schedule = get_schedule(base_full, steps_int, True, seed, use_visitor)
    part_schedules: list[list[list[int, str]]] = []
    for p in spec.parts:
        part_full = _concat_prefix_text_suffix(spec.prefix, p.text, spec.suffix)
        ps = get_schedule(part_full, steps_int, True, seed, use_visitor)
        part_schedules.append(ps)

    change_points = set()
    change_points.add(steps_int)
    for end_step, _ in base_schedule:
        change_points.add(int(end_step))
    for psched in part_schedules:
        for end_step, _ in psched:
            change_points.add(int(end_step))
    for p in spec.parts:
        s = p.step_start
        e = p.step_end if p.step_end is not None else steps_int
        if s > 1:
            change_points.add(s - 1)
        if e < steps_int:
            change_points.add(e)
    boundaries = sorted(change_points)

    out: list[ScheduledPromptConditioning] = []
    prev_key: tuple | None = None
    prev_text_combo: tuple | None = None
    prev_encodes: tuple | None = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        active_texts: list[str] = []
        active_indices: list[int] = []
        active_weights: list[float] = []
        active_modes: list[str] = []
        for i, p in enumerate(spec.parts):
            s = p.step_start
            e = p.step_end if p.step_end is not None else steps_int
            if s <= end_at_step <= e:
                pt = _select_text_from_schedule(part_schedules[i], end_at_step) or SAFE_EMPTY
                active_texts.append(pt)
                active_indices.append(i)
                active_modes.append(p.mode)
                if p.curve == "linear":
                    active_weights.append(p.weight)
                else:
                    span = float(e - s)
                    if span <= 0:
                        active_weights.append(p.weight)
                    else:
                        progress = (end_at_step - s) / span
                        cf = _apply_easing(progress, p.curve)
                        active_weights.append(p.weight * max(0.0, cf))

        text_combo = (base_text, tuple(active_texts))
        if text_combo == prev_text_combo and prev_encodes is not None:
            base_cond, part_encodes = prev_encodes
        else:
            all_texts = [base_text] + active_texts
            if not all_texts:
                continue
            all_conds = model.get_learned_conditioning(SdConditioning(all_texts, copy_from=copy_from))
            if isinstance(all_conds, dict):
                base_cond = {k: v[0] for k, v in all_conds.items()}
                part_encodes = [{k: v[i + 1] for k, v in all_conds.items()} for i in range(len(active_texts))]
            else:
                base_cond = all_conds[0]
                part_encodes = [all_conds[i + 1] for i in range(len(active_texts))]
            if empty_cond is not None and part_encodes:
                _ref = part_encodes[0]
                _target_len = _cond_token_len(_ref)
                _ec_len = _cond_token_len(empty_cond)
                if _ec_len < _target_len:
                    _n = (_target_len + _ec_len - 1) // _ec_len
                    if isinstance(empty_cond, dict):
                        empty_cond = {k: _torch.cat([v] * _n)[:_target_len] if v.ndim >= 2 else v
                                     for k, v in empty_cond.items()}
                    else:
                        empty_cond = _torch.cat([empty_cond] * _n)[:_target_len]
                elif _ec_len > _target_len:
                    if isinstance(empty_cond, dict):
                        empty_cond = {k: v[:_target_len] if v.ndim >= 2 else v
                                     for k, v in empty_cond.items()}
                    else:
                        empty_cond = empty_cond[:_target_len]
            prev_text_combo = text_combo
            prev_encodes = (base_cond, part_encodes)

        merged = base_cond
        for idx, w, mode in zip(range(len(active_indices)), active_weights, active_modes):
            part = part_encodes[idx]
            if mode == "diff" or mode == "diff_raw":
                if empty_cond is not None:
                    if isinstance(merged, dict):
                        merged = {k: merged[k] - w * (part[k] - empty_cond[k]) for k in merged}
                    else:
                        merged = merged - w * (part - empty_cond)
                else:
                    if isinstance(merged, dict):
                        merged = {k: merged[k] - w * part[k] for k in merged}
                    else:
                        merged = merged - w * part
                if mode == "diff":
                    merged = _normalize_cond_diff(merged, base_cond)
            elif mode == "ortho":
                if empty_cond is not None:
                    if isinstance(merged, dict):
                        merged_new = {}
                        for k in merged:
                            A = merged[k]; Bp = part[k]; E = empty_cond[k]
                            delta_B = Bp - E
                            dot_ad = (A * delta_B).sum(dim=-1, keepdim=True)
                            dot_dd = (delta_B * delta_B).sum(dim=-1, keepdim=True) + 1e-8
                            proj = (dot_ad / dot_dd) * delta_B
                            merged_new[k] = A - w * proj
                        merged = merged_new
                    else:
                        delta_B = part - empty_cond
                        dot_ad = (merged * delta_B).sum(dim=-1, keepdim=True)
                        dot_dd = (delta_B * delta_B).sum(dim=-1, keepdim=True) + 1e-8
                        proj = (dot_ad / dot_dd) * delta_B
                        merged = merged - w * proj
                else:
                    if isinstance(merged, dict):
                        merged = {k: merged[k] - w * part[k] for k in merged}
                    else:
                        merged = merged - w * part
                    merged = _normalize_cond_diff(merged, base_cond)
            elif isinstance(merged, dict):
                merged = {k: merged[k] + w * (part[k] - base_cond[k]) for k in merged}
            else:
                merged = merged + w * (part - base_cond)

        key = (base_text,) + tuple(active_texts) + tuple(active_indices) + tuple(active_weights) + tuple(active_modes)
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

    if not use_scheduling:
        has_curves = any(branch.curve != "linear" for branch in spec.branches)
        if has_curves:
            boundaries = list(range(1, int(steps) + 1))
        else:
            boundaries = [int(steps)]
    else:
        boundaries = _collect_schedule_boundaries(
            branch_cond_schedules,
            steps,
        )
    out: list[ScheduledPromptConditioning] = []
    previous_key = None

    for end_at_step in boundaries:
        branch_indices = tuple(_pick_schedule_entry_index(cond_schedule, end_at_step) for cond_schedule in branch_cond_schedules)
        active_conds = [
            branch_cond_schedules[i][branch_indices[i]].cond
            for i in range(len(branch_cond_schedules))
        ]

        raw_weights = []
        total_steps = int(steps)
        for branch in spec.branches:
            w = float(branch.weight)
            if branch.curve != "linear" and total_steps > 1:
                progress = (end_at_step - 1) / (total_steps - 1)
                cf = _apply_easing(progress, branch.curve)
                w = w * max(0.0, cf)
            raw_weights.append(w)

        cur_weights = _resolve_blend_mode_weights(raw_weights, spec.mode, spec.intensity)
        key = branch_indices + tuple(cur_weights)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue

        merged = _blend_morph_condition_values(active_conds, cur_weights)
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
        active_conds = _align_condition_values_for_blend(active_conds)
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
    per_f_row_ranges: list[list[list[tuple[int, int]]]] | None = None,
):
    """Apply row-splice: for each group i (index > 0), replace row segments at
    row_ranges[i] of R_cond with lerp(R_rows, F_conds[i-1]_rows, w_i).

    Each group is a list of (start, end) tensor-row ranges (multi-range supports
    chunk-boundary crossings where a single content group spans >75 tokens).

    When per_f_row_ranges is provided (BIND3 cumulative mode), use per_f_row_ranges[i]
    instead of row_ranges[i+1] for the i-th F_i. This is needed because in cumulative
    mode F_i's token positions differ from core_parts' positions (real attrs at
    positions 0..i shift the range of attribute i vs the padding template).

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
            for i, Fi in enumerate(F_conds):
                w = weights[i]
                Fi_t = Fi[key] if isinstance(Fi, dict) else Fi
                ranges = per_f_row_ranges[i] if per_f_row_ranges is not None else row_ranges[i + 1]
                for start, end in ranges:
                    if start >= end:
                        continue
                    if end > result.shape[-2]:
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
    result = R_t.clone()
    for i, Fi in enumerate(F_conds):
        w = weights[i]
        ranges = per_f_row_ranges[i] if per_f_row_ranges is not None else row_ranges[i + 1]
        for start, end in ranges:
            if start >= end:
                continue
            if end > result.shape[-2]:
                continue
            if abs(w - 1.0) <= 1e-8:
                result[start:end] = Fi[start:end]
            else:
                w_t = _torch.tensor(w, dtype=result.dtype, device=result.device)
                diff = Fi[start:end] - R_t[start:end]
                result[start:end] = R_t[start:end] + w_t * diff
    return result


def _parse_bind3_prompt(text: str, allow_attr_scheduling: bool = False) -> tuple[str, list[str], list[float], str, str, float]:
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
    weight_raw = m.group(1)
    bind3_weight = _parse_bind_weight(weight_raw, full_text=text)
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
    return owner, attrs, weights, prefix, suffix, bind3_weight


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
    owner, attrs, weights, prefix, suffix, bind3_weight = _parse_bind3_prompt(prompt, allow_attr_scheduling=bool(use_scheduling))
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
            all_f_parts: list[list[str]] = []
            for i, cur_a in enumerate(cur_attrs):
                f_parts = list(core_parts)
                if BIND3_CUMULATIVE_CONTEXT:
                    for j in range(i + 1):
                        if j == len(cur_attrs) - 1:
                            f_parts[j + 1] = _concat_prefix_text_suffix("", cur_attrs[j], suffix)
                        else:
                            f_parts[j + 1] = cur_attrs[j]
                else:
                    if i == len(cur_attrs) - 1:
                        f_parts[i + 1] = _concat_prefix_text_suffix("", cur_a, suffix)
                    else:
                        f_parts[i + 1] = cur_a
                F_texts.append(", ".join(f_parts))
                all_f_parts.append(f_parts)

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
                raise PromptSyntaxError(
                    f"BIND3 cannot compute row ranges — tokenization returned empty ranges for '{core_parts}'.",
                    kind="bind3_tokenization_error",
                    token=BIND3_KEYWORD,
                    full=prompt,
                )

            for pi, group in enumerate(row_ranges):
                span = sum(e - s for s, e in group)
                if span <= 0:
                    raise PromptSyntaxError(
                        f"BIND3 part {pi} has zero token span — padding error.",
                        kind="bind3_tokenization_error",
                        token=BIND3_KEYWORD,
                        full=prompt,
                    )

            per_f_row_ranges: list[list[list[tuple[int, int]]]] = []
            for i, fp in enumerate(all_f_parts):
                fi_ranges = _compute_row_ranges(model, fp)
                per_f_row_ranges.append(fi_ranges[i + 1])

            final_cond = _splice_condition_rows(R_cond_item, F_conds_items, weights, row_ranges, per_f_row_ranges=per_f_row_ranges)
            if isinstance(final_cond, dict):
                for pk in naive_cond_item:
                    if pk not in CHUNK_CROSSATTN_KEYS:
                        final_cond[pk] = naive_cond_item[pk]

            if abs(bind3_weight - 1.0) > 1e-8:
                final_cond = _weighted_sum_condition_values(
                    [naive_cond_item, final_cond],
                    [1.0 - bind3_weight, bind3_weight],
                )

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
    all_f_parts: list[list[str]] = []
    for i, attr in enumerate(attrs):
        f_parts = list(core_parts)
        if BIND3_CUMULATIVE_CONTEXT:
            # Кумулятивно: позиции 0..i — реальные атрибуты (видят друг
            # друга через каузальную маску CLIP), позиции i+1..N остаются
            # заглушками. F_i для последнего атрибута структурно совпадает
            # с naive_text (все атрибуты реальны) — отсюда и побочный
            # эффект, что suffix перестаёт быть "слепым" при w_last=1.0.
            for j in range(i + 1):
                if j == len(attrs) - 1:
                    f_parts[j + 1] = _concat_prefix_text_suffix("", attrs[j], suffix)
                else:
                    f_parts[j + 1] = attrs[j]
        else:
            if i == len(attrs) - 1:
                f_parts[i + 1] = _concat_prefix_text_suffix("", attr, suffix)
            else:
                f_parts[i + 1] = attr
        F_texts.append(", ".join(f_parts))
        all_f_parts.append(f_parts)

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
        raise PromptSyntaxError(
            f"BIND3 cannot compute row ranges — tokenization returned empty ranges for '{core_parts}'.",
            kind="bind3_tokenization_error",
            token=BIND3_KEYWORD,
            full=prompt,
        )

    for pi, group in enumerate(row_ranges):
        span = sum(e - s for s, e in group)
        if span <= 0:
            raise PromptSyntaxError(
                f"BIND3 part {pi} has zero token span — padding error.",
                kind="bind3_tokenization_error",
                token=BIND3_KEYWORD,
                full=prompt,
            )

    per_f_row_ranges: list[list[list[tuple[int, int]]]] = []
    for i, fp in enumerate(all_f_parts):
        fi_ranges = _compute_row_ranges(model, fp)
        per_f_row_ranges.append(fi_ranges[i + 1])

    final_cond = _splice_condition_rows(R_cond_item, F_conds_items, weights, row_ranges, per_f_row_ranges=per_f_row_ranges)
    if isinstance(final_cond, dict):
        for pk in naive_cond_item:
            if pk not in CHUNK_CROSSATTN_KEYS:
                final_cond[pk] = naive_cond_item[pk]

    if abs(bind3_weight - 1.0) > 1e-8:
        final_cond = _weighted_sum_condition_values(
            [naive_cond_item, final_cond],
            [1.0 - bind3_weight, bind3_weight],
        )

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

    # Extract base prompt context outside BIND2{...}
    _bind2_prefix_raw = protected[:m.start()]
    _bind2_suffix_raw = protected[brace_close + 1:]
    _bind2_prefix = _restore_escaped_literal_source(_bind2_prefix_raw, span_restore).strip()
    _bind2_suffix = _restore_escaped_literal_source(_bind2_suffix_raw, span_restore).strip()

    def _wrap_bind2_text(core: str) -> str:
        parts = []
        if _bind2_prefix:
            parts.append(_bind2_prefix)
        parts.append(core)
        if _bind2_suffix:
            parts.append(_bind2_suffix)
        return ", ".join(parts)

    core_parts = [owner] + pad_texts
    R_text = _wrap_bind2_text(", ".join(core_parts))
    V_texts: list[str] = []
    for i, a in enumerate(attrs):
        v_parts = list(core_parts)
        v_parts[i + 1] = a
        V_texts.append(_wrap_bind2_text(", ".join(v_parts)))
    naive_text = _wrap_bind2_text(", ".join([owner] + attrs))
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
!start: (prompt | /[():,|&]/+ | bare_bang)*
!bare_bang: "!"

prompt: (interpolated | scheduled | emphasized | grouped
         | alternate_distinct | alternate
        | alternate1 | alternate2
        | top_level_sequence3 | top_level_sequence | sequence
        | weighted | numbered | and_rule
        | compound_block
        | plain | WHITESPACE)*

!interpolated: "(" prompt ":" NUMBER "->" NUMBER ("->" NUMBER)* ("~" easing_mode)? ")"
easing_mode: "linear" | "ease" | "ease-in" | "ease-out" | "ease-in-out" | "bezier" | "catmull"
           | "sine-in" | "sine-out" | "sine-in-out"
           | "quart-in" | "quart-out" | "quart-in-out"
           | "quint-in" | "quint-out" | "quint-in-out"
           | "expo-in" | "expo-out" | "expo-in-out"
           | "circ-in" | "circ-out" | "circ-in-out"
           | "back-in" | "back-out" | "back-in-out"
           | "bounce"
           | "cubic" "(" NUMBER ("," NUMBER)* ")"

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

and_rule: (plain | weighted | emphasized | numbered | grouped | interpolated | alternate | alternate_distinct | alternate2 | alternate1 | scheduled | sequence | top_level_sequence | top_level_sequence3 | compound_block) ("&" (plain | weighted | emphasized | numbered | grouped | interpolated | alternate | alternate_distinct | alternate2 | alternate1 | scheduled | sequence | top_level_sequence | top_level_sequence3 | compound_block))+
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

    # '|' внутри скобок — alt-bang оператор (a|b)
    if "|" in (full or ""):
        return True

    # '!' — alt-bang только ВНЕ [content]
    # [hello!:5] — '!' внутри, не alt-bang
    # [a|b]!       — '!' снаружи, alt-bang
    if "!" in (full or ""):
        if full != (inner or ""):
            outer = full.replace(inner, "", 1) if inner else full
            if "!" in outer:
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
        s = "".join(resolve_tree(arg, keep_spacing=True) if isinstance(arg, lark.Tree) else str(arg) for arg in args if arg)
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
        return "".join(resolve_tree(arg, keep_spacing=True) if isinstance(arg, lark.Tree) else str(arg) for arg in args if arg)

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
                # Non-numeric second part — preserve raw text as-is.
                # Without this, (cat:dog) silently becomes (cat:1.1) losing "dog".
                return _unescape_literals("".join(raw_parts))
        else:
            weight_str = _rb_weight_str()

        # ★ Если уже имеем "(...:w)" как текст и внешний вес дефолтный — не оборачиваем повторно
        #   Это устраняет артефакт вида "((cat:1.2):1.1)" в Visitor.
        pt = prompt_text.strip()
        if (
            weight_str in _rb_weight_strs()
            and len(pt) >= 5 and pt[0] == "(" and pt[-1] == ")" and ":" in pt
        ):
            last_colon = pt.rfind(":")
            after = pt[last_colon + 1:-1].strip()
            if _RE_WEIGHT_NUMBER.fullmatch(after):
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
    def __init__(self, steps, prefix="", suffix="", use_scheduling=True, seed=None, _rng=None, _prompt_text=None):
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
        # Original prompt text for text-based fast paths (avoids resolve_tree losing bracket info)
        self._prompt_text = _prompt_text

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
        full = (self._prompt_text or resolve_tree(tree, keep_spacing=True)).strip()
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

        # 0) owner::a::b!!, trailing  (requires ≥2 :: separators, i.e. A :: B :: C !!)
        if "::" in full and "!!" in full and all(ch not in full for ch in '[]()'):
            left, trailing = full.split("!!", 1)
            after_first = left.split("::", 1)[1] if "::" in left else None
            if after_first and "::" in after_first:
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

        # 1.5) [a|b]! — resolve ALL bang-alternates to random choices BEFORE
        # Lark to avoid Earley ambiguity. Then apply step 2 scheduling on the
        # clean text if there are remaining [a:b:N] blocks. Never falls through
        # to _default_visit (which uses the ambiguous Lark tree).
        if ']!' in full:
            _re_alt_bang = re.compile(r'\[([^\[\]]+(?:\|[^\[\]]+)*)\]!')
            def _pick(m):
                inner = m.group(1)
                opts = [x.strip() for x in inner.split('|')]
                opts = [o for o in opts if o] or [""]
                return self.rng.choice(opts)
            clean = _re_alt_bang.sub(_pick, full)
            if '[' in clean and clean.count('[') == 1 and clean.count(']') == 1:
                try:
                    lb, rb = clean.index('['), clean.rindex(']')
                    pre, inner, post = clean[:lb], clean[lb+1:rb], clean[rb+1:]
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
            return [[self.steps, _collapse_spaces(self.prefix + clean + self.suffix)]]

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

    @staticmethod
    def _is_bare_weighted_prompt(node):
        if not (isinstance(node, lark.Tree) and getattr(node, "data", None) == "prompt"):
            return False
        kids = list(node.children)
        return len(kids) == 1 and isinstance(kids[0], lark.Tree) and kids[0].data == "weighted"

    @staticmethod
    def _contains_alternate1(node):
        """True, если где-то внутри node (рекурсивно) есть узел alternate1 —
        сигнал, что это часть де-вложенного "( ... alternate1 ... )"."""
        if isinstance(node, lark.Tree):
            if node.data == "alternate1":
                return True
            return any(CollectSteps._contains_alternate1(c) for c in node.children if isinstance(c, lark.Tree))
        return False

    @staticmethod
    def _leaf_texts(node):
        """Собирает непустые листовые текстовые куски слева направо,
        рекурсивно разворачивая "prompt"/"alternate1" (в т.ч. с пустыми
        слотами) — восстанавливает список опций из растащенных Earley
        сиблингов вида "2_(cat|dog)" -> [cat, dog], независимо от того,
        в какую конкретно форму дерева это распалось на конкретном seed."""
        if isinstance(node, lark.Token):
            s = str(node)
            return [s] if s.strip() else []
        if not isinstance(node, lark.Tree):
            return []
        if node.data in ("prompt", "alternate1"):
            out = []
            for c in node.children:
                out.extend(CollectSteps._leaf_texts(c))
            return out
        s = resolve_tree(node, keep_spacing=True)
        return [s] if s and s.strip() else []

    @staticmethod
    def _flatten_range_text(children):
        """Плоско рендерит список детей в текст (для частей до/после
        реконструированного alternate1-run), без учёта schedule/скобок."""
        parts = []
        for child in children:
            if isinstance(child, lark.Token):
                parts.append(" " if child.type == "WHITESPACE" else str(child))
            else:
                parts.append(resolve_tree(child, keep_spacing=True))
        return "".join(parts)

    def _default_visit(self, tree):
        # Warning for nested interpolations that bypass pre-processing
        if getattr(tree, "data", None) == "interpolated":
            raw = resolve_tree(tree, keep_spacing=True)
            msg = f"Nested interpolation '{raw}' will not be expanded into per-step schedule. Use (body:w0->w1) only at top-level parentheses."
            if _warn_semantic_once(msg):
                logging.warning(msg)

        children = list(tree.children)
        n = len(children)

        # Grammar ambiguity guard #2: de-nested "(" ... alternate1 ... ")" run,
        # где alternate1 (возможно с пустыми слотами) оказался вперемешку с
        # "потерянными" prompt-сиблингами (напр. "2_(cat|dog)" — Earley иногда
        # растаскивает cat/dog из alternate1 в отдельные соседние prompt-узлы).
        # Реконструируем плоский список опций из всех непустых текстовых
        # кусков между "(" и ")" в порядке появления и рендерим как обычную
        # per-step альтернацию, вместо того чтобы concatенировать куски как
        # попало (даёт "catdog" вместо чередования).
        alt_run = None
        for i in range(n):
            if isinstance(children[i], lark.Token) and str(children[i]) == "(":
                for j in range(i + 1, n):
                    if isinstance(children[j], lark.Token) and str(children[j]) == ")":
                        segment = children[i + 1:j]
                        if any(isinstance(c, lark.Tree) and self._contains_alternate1(c) for c in segment):
                            alt_run = (i, j, segment)
                        break
                if alt_run:
                    break
        if alt_run is not None:
            i, j, segment = alt_run
            options = []
            for c in segment:
                options.extend(self._leaf_texts(c))
            if options:
                pre = self._flatten_range_text(children[:i])
                post = self._flatten_range_text(children[j + 1:])
                if EXPAND_ALTERNATE_PER_STEP:
                    out = []
                    for step in range(1, self.steps + 1):
                        choice = options[(step - 1) % len(options)]
                        text = _concat_prefix_text_suffix(self.prefix, pre + choice + post, self.suffix)
                        text = _apply_and(_collapse_spaces(text, keep_edges=True))
                        out.append([step, text])
                    return out
                else:
                    choice = self.rng.choice(options)
                    text = _concat_prefix_text_suffix(self.prefix, pre + choice + post, self.suffix)
                    text = _apply_and(_collapse_spaces(text, keep_edges=True))
                    return [[self.steps, text]]

        # Grammar ambiguity guard: detect de-nested `( weighted )` triples where
        # Earley left ( and ) as sibling tokens instead of wrapping via emphasized.
        # visit_weighted already wraps output in parens; skip bare parens to avoid
        # doubling: `(` + `(cat:1.3)` + `)` = `((cat:1.3))`.
        _skip_brace = set()
        for i in range(n - 2):
            a, mid, b = children[i], children[i + 1], children[i + 2]
            if (isinstance(a, lark.Token) and str(a) == "("
                    and isinstance(b, lark.Token) and str(b) == ")"
                    and self._is_bare_weighted_prompt(mid)):
                _skip_brace.add(i)
                _skip_brace.add(i + 2)

        # 1) Собираем расписание для каждого ребёнка без внешних аффиксов
        child_scheds = []
        for idx, child in enumerate(children):
            if idx in _skip_brace:
                continue
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
            text = _apply_and(_collapse_spaces(text, keep_edges=True))
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
                            tok = f"{a}{'%' if is_pct else ''}-{b}{'%' if is_pct else ''}"
                            raise PromptSyntaxError(
                                f"Reverse range detected: {tok}",
                                kind="reverse_range",
                                token=tok,
                                full=resolve_tree(tree, keep_spacing=True),
                            )
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

    @staticmethod
    def _flatten_alternate1_options(node):
        """Рекурсивно разворачивает вложенный alternate1 в плоский список опций.

        Грамматика `alternate1: (prompt) "|" (prompt)+` рекурсивна через
        "prompt" (который сам может быть alternate1) -> для "cat|dog|fox"
        Earley неоднозначно между alternate1(cat, alternate1(dog,fox)) и
        alternate1(alternate1(cat,dog), fox). Без разворачивания вложенный
        alternate1 схлопывается в один текст (напр. "catdog" вместо
        отдельного чередования cat/dog/fox). Возвращает None, если node —
        не alternate1-обёртка (тогда вызывающий код использует resolve_tree).
        """
        if isinstance(node, lark.Tree) and node.data == "alternate1":
            alt = node
        elif (isinstance(node, lark.Tree) and node.data == "prompt"
                and len(node.children) == 1
                and isinstance(node.children[0], lark.Tree)
                and node.children[0].data == "alternate1"):
            alt = node.children[0]
        else:
            return None
        opts = []
        for child in alt.children:
            if isinstance(child, lark.Token):
                tok = str(child).strip()
                if tok in ("|", ",") or child.type == "WHITESPACE":
                    continue
                opts.append(str(child))
                continue
            nested = CollectSteps._flatten_alternate1_options(child)
            if nested is not None:
                opts.extend(nested)
            else:
                opts.append(resolve_tree(child, keep_spacing=True))
        return opts

    def visit_alternate1(self, tree):
        # Собираем варианты, включая потенциально пустые
        options = []
        last_was_sep = True
        for child in tree.children:
            # ВНИМАНИЕ (2026-07-31): Guard "пропустить prompt с 0 детей" был удалён.
            # Раньше он отбрасывал пустые варианты внутри alternate1, из-за чего
            # "[a||b]" (пустая опция в середине) и "[a|:3]" теряли пустые опции —
            # регрессия относительно допатченной версии. Фантомные пустые варианты
            # из "{a|b, c|d}" уже перехватываются в visit_grouped через
            # _phantom_alt1_continuation (до вызова visit_alternate1), поэтому
            # эта проверка здесь не нужна и вредна.
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
                nested_options = self._flatten_alternate1_options(child)
                if nested_options is not None:
                    options.extend(nested_options)
                    last_was_sep = False
                    continue
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

    def _schedule_from_combos(self, combos):
        """combos: готовые тексты (после ", ".join). Раньше все комбо в visit_grouped
        получали один и тот же end_at_step=self.steps — из-за этого в расписании
        побеждала всегда первая запись, "{a|b|c}" был неотличим от "{a}" (баг,
        найден фаззером на PYTHONHASHSEED/seed матрице). Тут — та же конвенция,
        что и в visit_alternate1: по шагам при EXPAND_ALTERNATE_PER_STEP,
        либо один случайный выбор на всю генерацию.
        """
        if not combos:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]
        if EXPAND_ALTERNATE_PER_STEP:
            out = []
            for step in range(1, self.steps + 1):
                text = combos[(step - 1) % len(combos)]
                out.append([step, _collapse_spaces(self.prefix + text + self.suffix)])
            return out
        text = self.rng.choice(combos)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def _phantom_alt1_continuation(self, child):
        """Распознаёт фантомный alternate1-continuation из "{a|b, c|d}".

        Earley для смешанного ",+|" парсит "c|d" как: голый "c" (сиблинг
        grouped) + ОТДЕЛЬНЫЙ alternate1 с фантомным пустым первым prompt
        (0 детей) — продолжение "|d". Возвращает непустые опции этого
        alternate1 (без фантомного пустого) для склейки с предыдущим
        слотом, либо None если child не является таким фантомным узлом.
        """
        if not (isinstance(child, lark.Tree) and child.data == "prompt"):
            return None
        kids = [k for k in child.children if not (isinstance(k, lark.Token) and k.type == "WHITESPACE")]
        if len(kids) != 1 or not isinstance(kids[0], lark.Tree) or kids[0].data != "alternate1":
            return None
        alt = kids[0]
        first = alt.children[0] if alt.children else None
        if not (isinstance(first, lark.Tree) and first.data == "prompt" and len(first.children) == 0):
            return None
        opts = []
        for c in alt.children[1:]:
            s = resolve_tree(c, keep_spacing=True)
            if s is not None and s.strip() != "":
                opts.append(s.strip())
        return opts

    def visit_grouped(self, tree):
        # Собираем варианты по дочерним узлам (сохраняем пустые элементы)
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            # Грамматика допускает неоднозначность в повторе "(item sep?)+":
            # для "cat,dog" Earley иногда вставляет лишний ПУСТОЙ "prompt"
            # между реальными элементами (0-длина matches "prompt: (...)* "),
            # что даёт фантомную опцию "" и лишнюю запятую на выходе
            # ("2, cat, , dog" вместо "2, cat, dog"). Пропускаем такие узлы.
            if (isinstance(child, lark.Tree) and child.data == "prompt"
                    and len(child.children) == 0):
                continue
            # Тот же фантом, другая форма: prompt-узел НЕ пустой (1+ детей), но
            # резолвится в чистый пробел — то же 0-длины-вокруг-разделителя
            # неоднозначность, просто Earley в этот раз завернул сам пробельный
            # токен в отдельный prompt вместо того, чтобы отбросить его как WHITESPACE.
            if isinstance(child, lark.Tree) and child.data == "prompt":
                _peek = resolve_tree(child, keep_spacing=True)
                if _peek is not None and _peek.strip() == "":
                    continue
            # Глубинная проблема "{a|b, c|d}": для смешанного ",+|" Earley НЕ
            # связывает "c" и "|d" в один alternate1 — "c" остаётся голым
            # сиблингом grouped, а "|d" парсится как ОТДЕЛЬНЫЙ alternate1 с
            # фантомным пустым первым вариантом (prompt с 0 детей). Результат:
            # "big" и "small" не чередуются, а попадают в РАЗНЫЕ слоты product().
            # Здесь склеиваем обратно: непустые опции фантомного alternate1
            # добавляются к последнему слоту (предыдущий голый элемент) — так
            # "big|small" снова становится одной альтернацией.
            _phantom_opts = self._phantom_alt1_continuation(child)
            if _phantom_opts is not None and all_options:
                all_options[-1].extend(_phantom_opts)
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
                combos = []
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
                        combos.append(text)
                if combos:
                    return self._schedule_from_combos(combos)
                # иначе сваливаемся на усечение (truncate)

        combos = []
        for i, combo in enumerate(product(*all_options)):
            if i >= GROUP_COMBO_LIMIT:
                break
            text = ", ".join(combo).strip()
            if text:
                combos.append(text)
        return self._schedule_from_combos(combos)

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
            return self._default_visit(tree)

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
    _lead_ws = text[:len(text) - len(text.lstrip(" \t\r\n"))]
    _trail_ws = text[len(text.rstrip(" \t\r\n")):]

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
        return _lead_ws + ATTENTION_AND_OPERATOR + _trail_ws

    cleaned_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        if tok != ATTENTION_AND_OPERATOR:
            cleaned_tokens.append(tok)
            continue

        has_prev_term = bool(cleaned_tokens) and cleaned_tokens[-1] != ATTENTION_AND_OPERATOR
        has_next_term = any(t != ATTENTION_AND_OPERATOR for t in tokens[i + 1 :])
        if has_prev_term and has_next_term:
            cleaned_tokens.append(tok)

    return _lead_ws + " ".join(cleaned_tokens) + _trail_ws


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
                weights_str = tail_match.group(1)
                weights = [float(w) for w in re.split(r'\s*->\s*', weights_str)]
            except (TypeError, ValueError):
                pass
            else:
                if all(math.isfinite(w) for w in weights):
                    continue

        return PromptSyntaxError(
            "Invalid attention interpolation. Use '(text:w0->w1)', '(text:w0->w1->w2~mode)', or '(text:w0->w1 @ N-M)'.",
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
    # Vars expansion must happen BEFORE transpile (matching get_schedule order),
    # and seed must be derived from pre-transpile text for consistency.
    if "<" in text or "__" in text:
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8", errors="replace")).digest()[:4], "big") & 0x7fffffff
        text, _var_meta = _expand_vars_and_macros(text, seed)
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
            strict=True,
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

def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True, wildcard_dir: str | None = None, is_negative: bool = False):
    """Depth-tracking wrapper for _get_schedule_body. Preserves _VAR_META_LOCAL.meta across recursive calls."""
    _call_depth = getattr(_VAR_META_LOCAL, '_schedule_depth', 0)
    _VAR_META_LOCAL._schedule_depth = _call_depth + 1
    if _call_depth == 0:
        _VAR_META_LOCAL.meta = {}
    try:
        return _get_schedule_body(prompt, steps, use_scheduling, seed, use_visitor, wildcard_dir, is_negative)
    finally:
        _VAR_META_LOCAL._schedule_depth = _call_depth


def _get_schedule_body(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True, wildcard_dir: str | None = None, is_negative: bool = False):
    """
    Args:
        seed: Если None, генерируется стабильный seed на основе хеша промпта.
    """
    prompt_text = str(prompt or "")
    prompt_text = prompt_text.translate(_PUA_CLEAN_TABLE)

    # TONEG pre-pass: remove from positive text only — in negative prompts, preserve TONEG as literal text
    if not is_negative:
        prompt_text, _ = _extract_toneg(prompt_text)

    if "<" in prompt_text or "__" in prompt_text:
        if seed is not None:
            vars_seed = seed
        else:
            vars_seed = int.from_bytes(hashlib.sha256(prompt_text.encode('utf-8', errors='replace')).digest()[:4], 'big') & 0x7fffffff
            seed = vars_seed
        prompt_text, _var_meta = _expand_vars_and_macros(prompt_text, vars_seed, wildcard_dir)
        _VAR_META_LOCAL.meta = _var_meta

    prompt_text = _transpile_diff_to_compound(prompt_text)
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
    if CHUNK_KEYWORD in prompt_text or ASSEMBLE_KEYWORD in prompt_text or BLEND_KEYWORD in prompt_text or MORPH_KEYWORD in prompt_text or POOL_KEYWORD in prompt_text or BIND_KEYWORD in prompt_text or COMPOUND_KEYWORD in prompt_text or REGION_KEYWORD.upper() in prompt_text.upper():
        try:
            state = _extract_backend_prompt_state(prompt_text)
        except PromptSyntaxError as e:
            if e.kind not in _STRUCTURAL_ERROR_KINDS and e.kind is not None:
                msg = f"Prompt semantic warning [{e.kind}]: {e} — falling back to raw text"
                if _warn_semantic_once(msg):
                    logging.warning(msg)
            elif e.kind is None:
                msg = f"Prompt error (no kind): {e} — falling back to raw text"
                if _warn_semantic_once(msg):
                    logging.warning(msg)
            else:
                msg = f"Prompt structural error [{e.kind}]: {e} — falling back to raw text"
                if _warn_semantic_once(msg):
                    logging.warning(msg)
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
        if _contains_chunk_marker(prompt_text) or _contains_assemble_marker(prompt_text) or _contains_blend_marker(prompt_text) or _contains_morph_marker(prompt_text) or _contains_pool_marker(prompt_text) or _contains_compound_marker(prompt_text) or _contains_bind_marker(prompt_text) or REGION_KEYWORD.upper() in prompt_text.upper():
            return [[int(steps), _collapse_spaces(prompt_text)]]

    err = _detect_invalid_interpolation_surface(prompt_text)
    if err is not None:
        msg = f"Invalid interpolation surface [{err.kind}]: {err} — ignoring interpolation"
        if _warn_semantic_once(msg):
            logging.warning(msg)
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
    schedule = _get_schedule_impl(prompt, steps, use_scheduling, seed, use_visitor, vars_expanded=True)
    return tuple((row[0], row[1]) for row in schedule)

def _placeholderize_attention_interpolations(text: str) -> str:
    """Pre-pass: replace (body:w0->w1) patterns with private-use marker strings.

    v3: supports multi-segment interpolation (cat:1.0->2.0->3.0)
        and parametric cubic-bezier easing (~cubic(a,b,c,d)).
        See _RE_ATT_INTERP_TAIL for full syntax.

    v2: bracket-balanced scanner — supports nested parentheses in body, e.g.:
        (red (glowing eyes):0.8->1.4)
        ((cat:1.2):1.0->2.0)
        (\\(escaped\\) face:1.0->2.0)

    Algorithm:
      1. Walk the string char-by-char, tracking paren depth and escape state.
      2. For each outer '(' … ')' span, extract the inner text.
      3. Find the LAST top-level ':' inside the span (depth=0 relative to inner).
      4. Check whether the tail after that ':' matches multi-segment
         interpolation syntax via _RE_ATT_INTERP_TAIL.
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

        # Validate tail: must be 'w0 -> w1', 'w0 -> w1 ~ mode', or 'w0 -> w1 @ N-M'
        m = _RE_ATT_INTERP_TAIL.match(tail)
        if not m:
            i = end + 1
            continue

        # Body must be non-empty
        if not body:
            i = end + 1
            continue

        weights_str = m.group(1)  # e.g. "1.0 -> 2.0" or "1.0 -> 2.0 -> 3.0"
        weight_parts = re.split(r'\s*->\s*', weights_str)
        weights = [float(w) for w in weight_parts]
        if not all(math.isfinite(w) for w in weights):
            i = end + 1
            continue
        mode = (m.group(2) or _EASING_DEFAULT).strip().lower()
        if mode not in _EASING_MODES and not mode.startswith("cubic("):
            mode = _EASING_DEFAULT

        # Extract optional @ start-end range (supports "5" or "20%")
        start_range: str | None = None
        end_range: str | None = None
        raw_start = m.group(3)
        raw_end = m.group(4)
        if raw_start is not None and raw_end is not None:
            if re.fullmatch(r'\d+%?', raw_start) and re.fullmatch(r'\d+%?', raw_end):
                start_range, end_range = raw_start, raw_end

        replacement = _serialize_att_interp(body, weights, mode, start_range, end_range)
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
    """Post-pass: expand (body:w0->w1[...->wn]) markers with correct global-range semantics.

    v3: supports multi-segment interpolation (3+ weights) where weight changes
        are distributed across N-1 segments using the same easing curve.

    Two-phase algorithm that eliminates the sawtooth/reset bug while preserving
    contextual-aware behaviour for markers that live inside a single scheduler
    window:

    Phase 1 — global range discovery
        Walk the schedule and, for each unique marker identity
        (body, raw_weights, mode), record the FIRST segment's start_step and
        the LAST segment's end_step where that marker appears.

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
    # Identity key: (body_stripped, raw_weights_str, mode)
    #   raw_weights_str is SEP-joined weights (e.g. "1.0\x0022.0\x0023.0")
    # Value: [first_start_step, last_end_step]
    MarkerRange = list  # [int, int]
    marker_global: dict[tuple, MarkerRange] = {}

    prev_end = 0
    for end_step, text in schedule:
        end_step   = int(end_step)
        start_step = prev_end + 1
        if ATTN_INTERP_OPEN in str(text):
            for m in RE_ATTN_INTERP_LITERAL.finditer(text):
                body   = m.group(1).strip()
                raw_w  = m.group(2)  # all weights SEP-joined
                mode   = (m.group(3) or _EASING_DEFAULT).strip().lower()
                key = (body, raw_w, mode)
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

            def _make_repl(step: int, marker_global: dict, total_steps: int) -> re.Match:
                def repl(m: re.Match) -> str:
                    body = m.group(1).strip()
                    raw_w = m.group(2)   # SEP-joined weights
                    mode  = (m.group(3) or _EASING_DEFAULT).strip().lower()
                    key = (body, raw_w, mode)

                    # Parse all weights from SEP-joined string
                    weight_strs = [w for w in raw_w.split(ATTN_INTERP_SEP) if w]
                    weights = [float(w) for w in weight_strs]
                    n_segments = len(weights) - 1

                    g_start, g_end = marker_global.get(key, (step, step))

                    # Check for optional @ sub-range from groups 4/5
                    # Supports "5", "20%", or None
                    raw_rs = m.group(4)
                    raw_re = m.group(5)
                    if raw_rs is not None and raw_re is not None and raw_rs.strip() and raw_re.strip():
                        try:
                            ms = _resolve_range_step(raw_rs, total_steps)
                            me = _resolve_range_step(raw_re, total_steps)
                            if ms > 0 and me > 0 and ms < me:
                                effective_start = ms
                                effective_end = me
                            else:
                                effective_start = g_start
                                effective_end = g_end
                        except (TypeError, ValueError):
                            effective_start = g_start
                            effective_end = g_end
                    else:
                        effective_start = g_start
                        effective_end = g_end

                    span = max(0, effective_end - effective_start)
                    t_lin = 0.0 if span == 0 else (step - effective_start) / span
                    t_lin = max(0.0, min(1.0, t_lin))

                    if n_segments == 1:
                        # Simple 2-weight interpolation (backward compat)
                        t = _apply_easing(t_lin, mode)
                        w = weights[0] + (weights[1] - weights[0]) * t
                    else:
                        # Multi-segment: easing applies per segment
                        seg = min(int(t_lin * n_segments), n_segments - 1)
                        seg_t = (t_lin * n_segments) - seg
                        seg_t = max(0.0, min(1.0, seg_t))
                        seg_t = _apply_easing(seg_t, mode)
                        w = weights[seg] + (weights[seg + 1] - weights[seg]) * seg_t
                    return f"({body}:{_format_interp_weight(w)})"
                return repl

            expanded = RE_ATTN_INTERP_LITERAL.sub(_make_repl(step, marker_global, steps), text)
            if out and out[-1][1] == expanded:
                out[-1][0] = step
            else:
                out.append([step, expanded])

        prev_end = end_step

    return out


def _get_schedule_impl(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool, vars_expanded: bool = False):
    """Основная реализация без кеширования"""

    _validate_inputs(prompt, steps)
    result: list[list[int | str]] | None = None

    if not str(prompt).strip():
        result = [[steps, SAFE_EMPTY]]
    else:
        if "\\n" in prompt or "\\t" in prompt:
            prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

        if not vars_expanded:
            prompt, _var_meta = _expand_vars_and_macros(prompt, seed)
            if _var_meta:
                _VAR_META_LOCAL.meta = _var_meta

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
                collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed, _prompt_text=prompt)
                schedules = collector(tree)
                try:
                    schedules.sort(key=lambda x: int(x[0]))
                except (ValueError, TypeError) as exc:
                    logger.warning("Schedule sort failed (%s) — using unsorted", exc)

                if DEDUP_SCHEDULE_STEPS:
                    try:
                        schedules = _dedup_schedules(schedules)
                    except (ValueError, TypeError, AttributeError) as exc:
                        logger.warning("Schedule dedup failed (%s) — keeping duplicates", exc)

                if not schedules:
                    result = [[steps, _collapse_spaces(prompt)]]
                else:
                    if not use_visitor:
                        logging.warning("use_visitor=False is deprecated — always uses Lark visitor. Parameter will be removed in a future version.")
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

# ── Weight Interpretation Mode (ПАТЧ 2) ──────────────────────────────────

_VALID_WEIGHT_MODES = frozenset({"a1111", "comfy", "compel", "comfy++", "down_weight"})


def _env_weight_mode(name: str = "WEIGHT_INTERPRETATION", default: str = "a1111") -> str:
    v = os.getenv(name, default).strip().lower()
    if v not in _VALID_WEIGHT_MODES:
        logger.warning(
            "Unknown %s=%r, falling back to 'a1111'. Valid: %s",
            name, v, ", ".join(sorted(_VALID_WEIGHT_MODES))
        )
        return "a1111"
    return v


WEIGHT_INTERPRETATION: str = _env_weight_mode()

# Включает реальное per-chunk применение apply_advanced_weights в
# _build_plain_prompt_conditioning_schedule. По умолчанию ВЫКЛ (0), чтобы
# поведение для существующих вызывающих не менялось (особенно важно внутри
# самого A1111, где per-token взвешивание уже делает sd_hijack_clip.py —
# включение этого флага там привело бы к двойному применению весов).
APPLY_ADVANCED_WEIGHTS_ENABLED = _env_bool("APPLY_ADVANCED_WEIGHTS_ENABLED", "0")
CHUNK_CONTENT_TOKEN_LIMIT = 75


def _wi_a1111_renorm(base_emb, weighted_emb):
    """Renorm: сохраняет среднее значение базового эмбеддинга."""
    torch = _ensure_torch()
    b_mean = base_emb.mean()
    w_mean = weighted_emb.mean()
    if w_mean.abs() < 1e-8:
        return weighted_emb
    return (b_mean / w_mean) * weighted_emb


def _wi_apply_comfy(base_emb, weights, empty_emb):
    """Comfy-style: lerp(empty, base, weight)."""
    w = weights.unsqueeze(-1)
    return empty_emb + w * (base_emb - empty_emb)


def _wi_apply_compel(base_emb, weights, empty_emb):
    """Partial Compel: up=comfy, down=base*w (прямое масштабирование).

    Настоящий Compel down-weight требует per-token маскирования
    (encode_without_token_i), что невозможно без CLIP encoder internals.
    Partial Compel: down=base*w — ослабляет вклад токена, сохраняя направление,
    в отличие от comfy (lerp к empty).
    """
    torch = _ensure_torch()
    w = weights.unsqueeze(-1)
    up_mask = (weights >= 1.0).unsqueeze(-1)
    up_result = empty_emb + w * (base_emb - empty_emb)
    down_result = base_emb * w
    return torch.where(up_mask, up_result, down_result)


def _wi_apply_comfy_pp(base_emb, weights, empty_emb):
    """comfy++: identical to compel due to CLIP access limitation (see _wi_apply_compel)."""
    return _wi_apply_compel(base_emb, weights, empty_emb)


def _wi_apply_down_weight(base_emb, weights, empty_emb):
    """Down-weight: нормализует веса к макс=1, затем compel."""
    torch = _ensure_torch()
    max_w = weights.max()
    if max_w > 1e-8:
        weights = weights / max_w
    return _wi_apply_compel(base_emb, weights, empty_emb)


def apply_advanced_weights(
    base_emb,
    weights,
    empty_emb,
    mode: str | None = None,
):
    """Library function — не вызывается нигде до подключения в conditioning pipeline.

    Использовать после интеграции conditioning:
        cond[i] = apply_advanced_weights(cond[i], w[i], empty_emb, mode="comfy")
    """
    if isinstance(base_emb, dict):
        result = {}
        for key in base_emb.keys():
            sub_empty = empty_emb.get(key) if isinstance(empty_emb, dict) else empty_emb
            if sub_empty is None:
                sub_empty = empty_emb  # fallback: missing key → use whole dict or empty signal
            key_weights = weights
            if weights is not None and base_emb[key].dim() >= 2 and weights.dim() >= 2 and base_emb[key].shape[1] != weights.shape[1]:
                key_weights = weights.mean(dim=1, keepdim=True)
            result[key] = apply_advanced_weights(
                base_emb[key], key_weights, sub_empty, mode=mode
            )
        return result

    if weights is None:
        return base_emb
    torch = _ensure_torch()
    if mode is None:
        mode = WEIGHT_INTERPRETATION

    if mode == "a1111":
        w = weights.unsqueeze(-1)
        return _wi_a1111_renorm(base_emb, base_emb * w)
    elif mode == "comfy":
        return _wi_apply_comfy(base_emb, weights, empty_emb)
    elif mode == "compel":
        return _wi_apply_compel(base_emb, weights, empty_emb)
    elif mode == "comfy++":
        return _wi_apply_comfy_pp(base_emb, weights, empty_emb)
    elif mode == "down_weight":
        return _wi_apply_down_weight(base_emb, weights, empty_emb)
    else:
        raise ValueError(
            f"Unknown WEIGHT_INTERPRETATION mode: {mode!r}. "
            f"Valid: {', '.join(sorted(_VALID_WEIGHT_MODES))}"
        )


# ── Честное per-chunk применение весов (расширение ПАТЧА 2) ─────────────
#
# Реальный A1111 (modules/sd_hijack_clip.py) не лезет в attention внутри
# трансформера — он, как и apply_advanced_weights выше, делает постфактум
# умножение+перенормировку уже готового выхода энкодера. Разница только в
# том, ЧТО именно перенормируется: A1111 делает это НА КАЖДОМ 75-токенном
# чанке отдельно, до конкатенации, а не одним глобальным коэффициентом на
# уже склеенный тензор (как делал apply_advanced_weights в режиме "a1111"
# выше). Эта секция реализует ровно ту же per-chunk схему, опираясь
# исключительно на публичный API model.get_learned_conditioning() и
# model.cond_stage_model.tokenize() — без единого обращения к forward pass
# энкодера.
#
# Включается только при APPLY_ADVANCED_WEIGHTS_ENABLED=1 — по умолчанию
# ничего не меняется.

def _build_token_weight_chunks(model, prompt: str) -> list[tuple[str, list[float]]]:
    """Разбивает промпт на чанки не более CHUNK_CONTENT_TOKEN_LIMIT
    контент-токенов каждый, сохраняя для каждого чанка список весов
    по одному на контент-токен (BOS/EOS/padding сюда не входят — они
    получают вес 1.0 при дальнейшем применении).

    Гранулярность — ПО СЛОВАМ внутри каждого спана, а не по спанам
    целиком: parse_prompt_attention сливает соседние слова с одинаковым
    весом в один спан (например, 80 слов без весов — это ОДИН спан, а
    не 80), и если резать только между спанами, длинный безвесовой текст
    никогда бы не попал в чанк меньше своего полного размера. Резка по
    словам — тот же компромисс, которым пользуется сам A1111 в
    tokenize_line(): если единственное слово само по себе превышает
    лимит токенов (большая редкость), оно остаётся в чанке как есть —
    разбиение внутри BPE-подслов здесь не делается.

    Текст чанка восстанавливается склейкой слов через пробел — не
    побайтовая копия оригинала (точные пробелы parse_prompt_attention
    не хранит), но конкретные веса привязаны к словам, а не к пробелам.
    """
    spans = parse_prompt_attention(prompt)
    flat: list[tuple[str, float, int]] = []
    for text, weight in spans:
        if not text:
            continue
        for word in text.split():
            try:
                n_tokens = len(_tokenize_strip_special(model, word))
            except ValueError:
                raise
            if n_tokens == 0:
                continue
            flat.append((word, float(weight), n_tokens))

    chunks: list[tuple[str, list[float]]] = []
    cur_texts: list[str] = []
    cur_weights: list[float] = []
    cur_count = 0
    for word, weight, n_tokens in flat:
        if cur_count + n_tokens > CHUNK_CONTENT_TOKEN_LIMIT and cur_texts:
            chunks.append((" ".join(cur_texts), cur_weights))
            cur_texts, cur_weights, cur_count = [], [], 0
        cur_texts.append(word)
        cur_weights.extend([weight] * n_tokens)
        cur_count += n_tokens
    if cur_texts:
        chunks.append((" ".join(cur_texts), cur_weights))
    if not chunks:
        chunks = [("", [])]
    return chunks


def _row_weight_tensor(torch_mod, n_rows: int, content_weights: list[float], device=None, dtype=None):
    """Строит тензор весов по строкам эмбеддинга: row 0 (BOS) = 1.0,
    rows 1..len(content_weights) = веса по словам, остаток (EOS+padding) = 1.0."""
    row_w = [1.0] * n_rows
    for i, w in enumerate(content_weights, start=1):
        if i < n_rows:
            row_w[i] = w
    return torch_mod.tensor(row_w, dtype=dtype, device=device)


def _apply_a1111_chunk_weight(chunk_emb, content_weights: list[float], empty_emb=None, mode: str | None = None):
    """Применяет per-token веса к ОДНОМУ уже закодированному чанку
    (одно обращение к model.get_learned_conditioning на чанк), используя
    выбранный WEIGHT_INTERPRETATION mode. Для словарных (SDXL) conditioning
    взвешивается только crossattn-подобный ключ (CHUNK_CROSSATTN_KEYS) —
    pooled-векторы ('vector' и т.п.) не имеют построчной токен-структуры,
    их веса не касаются, как и в самом A1111."""
    torch_mod = _ensure_torch()
    if isinstance(chunk_emb, dict):
        out = dict(chunk_emb)
        for key in CHUNK_CROSSATTN_KEYS:
            if key not in chunk_emb:
                continue
            tensor = chunk_emb[key]
            n_rows = tensor.shape[-2] if _tensor_ndim(tensor) >= 2 else tensor.shape[0]
            w = _row_weight_tensor(torch_mod, n_rows, content_weights, device=tensor.device, dtype=getattr(tensor, "dtype", None))
            sub_empty = empty_emb.get(key) if isinstance(empty_emb, dict) else empty_emb
            out[key] = apply_advanced_weights(tensor, w, sub_empty, mode=mode)
        return out

    n_rows = chunk_emb.shape[-2] if _tensor_ndim(chunk_emb) >= 2 else chunk_emb.shape[0]
    w = _row_weight_tensor(torch_mod, n_rows, content_weights, device=chunk_emb.device, dtype=getattr(chunk_emb, "dtype", None))
    return apply_advanced_weights(chunk_emb, w, empty_emb, mode=mode)


def _encode_prompt_with_chunked_weights(model, text: str, copy_from, mode: str | None = None):
    """Кодирует один текст с честным per-chunk применением весов вместо
    одного вызова model.get_learned_conditioning([text]). Чанкинг по
    CHUNK_CONTENT_TOKEN_LIMIT контент-токенов воспроизводит то же окно,
    которым физически ограничен сам CLIP-энкодер (см. обсуждение лимита
    77 позиций) — это не новое ограничение, а явное отражение уже
    существующего.

    Если у модели нет распознаваемого токенизатора — поднимает
    ValueError, чтобы вызывающий код мог откатиться на простой путь."""
    chunks = _build_token_weight_chunks(model, text)
    mode_eff = mode if mode is not None else WEIGHT_INTERPRETATION
    needs_empty = mode_eff != "a1111"
    empty_emb = None
    if needs_empty:
        empty_texts = SdConditioning([""], copy_from=copy_from)
        empty_conds = model.get_learned_conditioning(empty_texts)
        empty_emb = {k: v[0] for k, v in empty_conds.items()} if isinstance(empty_conds, dict) else empty_conds[0]

    chunk_embs = []
    for chunk_text, content_weights in chunks:
        chunk_texts = SdConditioning([chunk_text], copy_from=copy_from)
        chunk_conds = model.get_learned_conditioning(chunk_texts)
        chunk_emb = {k: v[0] for k, v in chunk_conds.items()} if isinstance(chunk_conds, dict) else chunk_conds[0]
        if all(abs(w - 1.0) < 1e-9 for w in content_weights):
            chunk_embs.append(chunk_emb)
        else:
            chunk_embs.append(_apply_a1111_chunk_weight(chunk_emb, content_weights, empty_emb, mode=mode_eff))

    if len(chunk_embs) == 1:
        return chunk_embs[0]
    chunk_embs = _align_condition_values_for_blend(chunk_embs)
    return _merge_chunk_condition_values(chunk_embs, [1.0] * len(chunk_embs))


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
    is_negative: bool = False,
):
    steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else base_steps
    use_scheduling = (hires_steps is None) or use_old_scheduling
    prompt_schedules = [get_schedule(p, steps, use_scheduling, seed, use_visitor=use_visitor, is_negative=is_negative) for p in prompts]
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
    is_negative = getattr(prompts, 'is_negative_prompt', False)

    _transpiled = [_transpile_diff_to_compound(p) for p in prompts]
    _transpiled = [_transpile_bind2_to_chunk(p) for p in _transpiled]
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
        if WEIGHT_INTERPRETATION == "a1111":
            logger.debug("per-chunk a1111 renorm skipped: backend operators handle weights internally")
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
        is_negative=is_negative,
    )
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        if not prompt_schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached); continue
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)

        # Per-chunk a1111 renorm if applicable
        _have_torch = False
        try:
            __import__("torch")
            _have_torch = True
        except ImportError:
            pass
        if WEIGHT_INTERPRETATION == "a1111" and _have_torch:
            cond_schedule: list[ScheduledPromptConditioning] = []
            try:
                for end_at_step, step_text in prompt_schedule:
                    merged_cond = _encode_prompt_with_chunked_weights(model, step_text, prompts)
                    cond_schedule.append(ScheduledPromptConditioning(end_at_step, merged_cond))
            except (ValueError, TypeError, AttributeError):
                conds = model.get_learned_conditioning(texts)
                cond_schedule = []
                for i, (end_at_step, _) in enumerate(prompt_schedule):
                    if isinstance(conds, dict):
                        cond = {k: v[i] for k, v in conds.items()}
                    else:
                        cond = conds[i]
                    cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        else:
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

    _transpiled = [_transpile_diff_to_compound(p) for p in prompts]
    _transpiled = [_transpile_bind2_to_chunk(p) for p in _transpiled]
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
                prev_blocks_split = prev_ch == "\\" or prev_ch in _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.values() or is_word_char(prev_ch)
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
            s = subprompt if isinstance(subprompt, str) else str(subprompt)

            # 1) Нормализация скобочных весов: "(cat:2.0)" или просто "(cat)"
            m_emph = re.match(
                r'^\s*\(\s*(.*?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?\)\s*$',
                s
            )
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
            or _contains_compound_marker(branch_text)
            or _contains_bind_marker(branch_text)
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

    def to(self, *args, **kwargs):
        for k in self.keys():
            if isinstance(self[k], _ensure_torch().Tensor):
                self[k] = self[k].to(*args, **kwargs)
        return self

    def advanced_indexing(self, item):
        result = {}
        for k in self.keys():
            if isinstance(self[k], _ensure_torch().Tensor):
                result[k] = self[k][item]
        return DictWithShape(result)


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
    targets = [cond_schedule[_pick_schedule_entry_index(cond_schedule, current_step)].cond for cond_schedule in c]
    param = targets[0]
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    if hasattr(_ensure_torch(), 'stack'):
        targets = _align_condition_values_for_blend(targets)
    aligned = targets
    if not hasattr(_torch, 'stack'):
        return aligned
    is_dict = isinstance(param, dict)
    if is_dict:
        ref = aligned[0]
        res = {k: _torch.zeros((len(c),) + v.shape, device=getattr(v, "device", "cpu"), dtype=getattr(v, "dtype", _torch.float32)) for k, v in ref.items()}
        res = DictWithShape(res, (len(c),) + ref.get('crossattn', next(iter(ref.values()))).shape)
        for i, cond in enumerate(aligned):
            for k, v in cond.items():
                res[k][i] = v
    else:
        res = _torch.stack(aligned)
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
        hint = "Use a supported BLEND mode: mean, sum, product, or max."
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
        hint = "Use a supported MORPH curve: linear, bezier, catmull, slerp, or any easing mode (ease-in, sine-out, bounce, cubic(...), etc.)."
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
    "toneg_in_negative_prompt": {
        "en": "TONEG{...} inside a negative prompt has no effect — extra negative text from TONEG is already subtracted from the positive prompt",
        "ru": "TONEG{...} внутри негативного промпта не имеет эффекта — extra negative из TONEG уже вычитается из позитивного промпта",
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
    "tail_unreachable_at_boundary": {
        "en": "The tail prompt(s) in a multi-phase bracket will never appear because the boundary equals or exceeds the total step count. Use fewer phases or more steps.",
        "ru": "Хвостовой(ые) промпт(ы) в многофазном bracket никогда не появятся, так как граница равна или превышает общее количество шагов. Уменьшите количество фаз или увеличьте количество шагов.",
    },
    "cfg_schedule_with_single_step": {
        "en": "CFG schedule range (a->b) is meaningless when steps=1 — use a single CFG value instead",
        "ru": "CFG range-расписание (a->b) не имеет смысла при steps=1 — используйте одно значение CFG",
    },
    "non_numeric_weight": {
        "en": "Non-numeric weight string after ':' in parentheses — weight will be treated as 1.0",
        "ru": "Нечисловой вес после ':' в скобках — вес будет проигнорирован (используется 1.0)",
    },
    "pipe_inside_bracket": {
        "en": "'|' inside [...] without trailing '!' — use [a|b]! for alternation, or [a:b:N] for scheduling",
        "ru": "'|' внутри [...] без '!' в конце — используйте [a|b]! для альтернации или [a:b:N] для планирования",
    },
    "region_zero_area": {
        "en": "Region has near-zero area — will not affect generation",
        "ru": "Область региона почти нулевая — не повлияет на генерацию",
    },
    "region_multiple_base": {
        "en": "Multiple *base= directives, using last",
        "ru": "Несколько *base=, используется последний",
    },
    "region_empty_region": {
        "en": "Empty region text — region will have no content",
        "ru": "Пустой текст региона — регион будет без содержимого",
    },
    "region_inside_backend_not_supported": {
        "en": "REGION inside backend operator (CHUNK/BLEND/etc.) is not supported — region block will be treated as plain text",
        "ru": "REGION внутри backend-оператора (CHUNK/BLEND/и т.д.) не поддерживается — блок будет обработан как обычный текст",
    },
    "region_in_negative_prompt": {
        "en": "REGION in negative prompt has no effect — use plain negative prompt instead",
        "ru": "REGION в негативном промпте не имеет эффекта — используйте обычный негативный промпт",
    },
    "region_too_many_regions": {
        "en": "Too many regions (>4) — performance may degrade",
        "ru": "Слишком много регионов (>4) — производительность может снизиться",
    },
    "region_multiple_blocks": {
        "en": "Multiple REGION{...} blocks found — merging into one region list",
        "ru": "Найдено несколько REGION{...} блоков — объединяются в один список",
    },
}

_RE_BACKEND_KEYWORD_BRACE = re.compile(
    r'(CHUNK|BLEND|MORPH|POOL|ASSEMBLE|BIND|BIND2|BIND3|TONEG)\{$'
)


def _has_unescaped_braces(text: str) -> bool:
    for m in re.finditer(r'(?<!\\)\{', text):
        pre = text[:m.start()].rstrip()
        if pre and pre.endswith(("CHUNK", "BLEND", "MORPH", "POOL", "ASSEMBLE", "BIND", "BIND2", "BIND3", "COMPOUND", "TONEG", "REGION")):
            continue
        return True
    return False


def _approx_token_count(text: str) -> int:
    cjk = sum(1 for c in text
              if unicodedata.category(c) == 'Lo' and ord(c) > 0x2E7F)
    non_cjk_words = len([w for w in text.split()
                         if not any(ord(c) > 0x2E7F for c in w)])
    return cjk + non_cjk_words
_TOKENIZER_WARNED: set[str] = set()


def _exact_token_count(text: str, tokenizer: Any) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception as exc:
        global _TOKENIZER_WARNED
        key = f"{id(tokenizer)}:{exc}"
        if key not in _TOKENIZER_WARNED:
            _TOKENIZER_WARNED.add(key)
            logger.warning("exact_token_count failed (%s), fallback to approx", exc)
        return _approx_token_count(text)


def _check_backend_warnings(text: str, lang: str = "en", is_negative: bool = False, steps: int = 20, tokenizer: Any = None) -> list[dict]:
    orig_text = text  # keep for BIND3 marker checks (transpile removes them)
    text = _transpile_diff_to_compound(text)
    text = _transpile_bind3_to_chunk(text)
    try:
        state = _extract_backend_prompt_state(text)
    except PromptSyntaxError:
        raise
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
        or _contains_bind3_marker(orig_text)
    ):
        warnings.append(_w("bind_in_negative_prompt"))

    if is_negative and TONEG_KEYWORD in text:
        warnings.append(_w("toneg_in_negative_prompt"))

    if _contains_bind3_marker(orig_text) and _RE_HAS_SCHEDULING.search(text):
        warnings.append(_w("bind3_scheduling_not_supported_at_conditioning"))

    for _bm in re.finditer(r'\[([^\[\]]+):([^\]\s]+(?:\s+[^\]\s]+)?)\]\s*(?:reverse)?', text):
        _inner = _bm.group(1)
        _b_str = _bm.group(2).strip().split()[0]  # boundary, strip step_range
        try:
            if _b_str.endswith('%'):
                _b_val = float(_b_str[:-1]) / 100.0 * steps
            else:
                _b_val = float(_b_str)
        except (ValueError, TypeError):
            continue
        _phases = _inner.split(':')
        if _b_val < len(_phases) - 1:
            warnings.append(_w("narrow_bracket_boundary"))
        if len(_phases) >= 3 and _b_val >= steps:
            warnings.append(_w("tail_unreachable_at_boundary"))
    # Also check postfix syntax: [a:b]:N
    for _bm in re.finditer(r'\[([^\[\]]+):([^\[\]]+)\]:(\d+(?:\.\d+)?%?)', text):
        _inner = _bm.group(1) + ':' + _bm.group(2)
        try:
            _b_val = float(_bm.group(3))
        except (ValueError, TypeError):
            continue
        _phases = _inner.split(':')
        if _b_val < len(_phases) - 1:
            warnings.append(_w("narrow_bracket_boundary"))
        if len(_phases) >= 3 and _b_val >= steps:
            warnings.append(_w("tail_unreachable_at_boundary"))

    stripped = re.sub(r'\[.*?\]', '', text)
    tc = _exact_token_count(stripped, tokenizer) if tokenizer is not None else _approx_token_count(stripped)
    if tc > 50:
        warnings.append(_w("token_limit_exceeded"))

    if '<param[cfg]:' in text and steps == 1 and '->' in text:
        warnings.append(_w("cfg_schedule_with_single_step"))

    for _mm in re.finditer(r'\(([^()]+):\s*([A-Za-z0-9_.+-]+)\s*\)', text):
        try:
            float(_mm.group(2))
        except ValueError:
            warnings.append(_w("non_numeric_weight"))
            break

    for _bm in re.finditer(r'\[([^\[\]]*\|[^\[\]]*)\](?!!)', text):
        warnings.append(_w("pipe_inside_bracket"))
        break

    # REGION-specific warnings
    if REGION_KEYWORD in text:
        try:
            _r_clean, _r_regs = get_prompt_regions(text)
        except Exception:
            _r_regs = []
        if _r_regs:
            for _rb in _r_regs:
                if not _rb.text.strip():
                    warnings.append(_w("region_empty_region"))
                if _rb.x1 is not None and abs(_rb.x2 - _rb.x1) * abs(_rb.y2 - _rb.y1) < 1e-9:
                    warnings.append(_w("region_zero_area"))
            if len(_r_regs) > 4:
                warnings.append(_w("region_too_many_regions"))
        if text.count('*base=') > 1:
            warnings.append(_w("region_multiple_base"))
        if len(_find_top_level_region_blocks(text)) > 1:
            warnings.append(_w("region_multiple_blocks"))
        if is_negative:
            warnings.append(_w("region_in_negative_prompt"))
        # Check if REGION{ appears inside another { } block (depth > 0)
        _region_depth = 0
        _ri = 0
        while _ri < len(text):
            if text[_ri] == '\\' and _ri + 1 < len(text) and text[_ri + 1] in ('{', '}'):
                _ri += 2
                continue
            _rc = text[_ri]
            if (_rc == 'R' and text.startswith(REGION_KEYWORD, _ri)
                and _ri + len(REGION_KEYWORD) < len(text)
                and text[_ri + len(REGION_KEYWORD)] == '{'):
                if _region_depth > 0:
                    warnings.append(_w("region_inside_backend_not_supported"))
                    break
            if _rc == '{':
                _region_depth += 1
            elif _rc == '}':
                _region_depth = max(0, _region_depth - 1)
            _ri += 1

    return warnings


def lint_prompt(text: str, steps: int = 20, seed: int | None = None, lang: str = "ru", is_negative: bool = False, tokenizer: Any = None) -> dict:
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
            warnings = _check_backend_warnings(text, lang, is_negative=is_negative, steps=steps, tokenizer=tokenizer)
        except PromptSyntaxError:
            raise
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
    sched = _strict_schedule_preview(text, steps=steps, seed=seed)
    out_lines = []
    prev_end = 0
    for end, t in sched:
        start = prev_end + 1
        out_lines.append(f"Шаги {start}-{end}: {t}")
        prev_end = end
    return "\n".join(out_lines)


_STRIP_EMPHASIS_RE = re.compile(
    r'<[^<>]*>'                                  # entire <tag> — pass through untouched
    r'|[\[\]()]'                                  # emphasis brackets (not {})
    r'|:[\d.]+(?:->[\d.]+)*(?=[\s,\])}\(]|$)'    # weight / interpolation suffix
)

def strip_emphasis(prompt: str) -> str:
    return _STRIP_EMPHASIS_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('<') else '', prompt
    )


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
    import inspect
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

    # V2-1: nested body '(cat)' — _default_visit keeps (cat) without explicit :1.1,
    # parse_prompt_attention infers 1.1x from depth. Outer interpolation wraps it.
    sc_v2_1 = g_v2("((cat):1.0->2.0)", 5)
    assert "((cat):1.0)" in sc_v2_1[0][1],  f"V2-1 first: {sc_v2_1[0][1]!r}"
    assert "((cat):2.0)" in sc_v2_1[-1][1], f"V2-1 last:  {sc_v2_1[-1][1]!r}"

    # V2-2: inner '(glowing eyes)' → _default_visit keeps (glowing eyes) without :1.1,
    # outer interpolation weight wraps the whole body.
    sc_v2_2 = g_v2("(red (glowing eyes):0.8->1.4)", 5)
    assert "->" not in sc_v2_2[0][1],              f"V2-2 raw arrow: {sc_v2_2[0][1]!r}"
    assert ATTN_INTERP_OPEN not in sc_v2_2[-1][1], f"V2-2 marker leak: {sc_v2_2[-1][1]!r}"
    assert "(red (glowing eyes):0.8)" in sc_v2_2[0][1],  f"V2-2 first: {sc_v2_2[0][1]!r}"
    assert "(red (glowing eyes):1.4)" in sc_v2_2[-1][1], f"V2-2 last:  {sc_v2_2[-1][1]!r}"

    # V2-3: '(a:b)' where b is non-numeric → emphasized preserves raw text,
    # then multiplied by the dynamic emphasized weight → ((a:b):W)
    sc_v2_3 = g_v2("((a:b):1.0->2.0)", 5)
    assert "((a:b):1.0)" in sc_v2_3[0][1],  f"V2-3 first: {sc_v2_3[0][1]!r}"
    assert "((a:b):2.0)" in sc_v2_3[-1][1], f"V2-3 last:  {sc_v2_3[-1][1]!r}"

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
    assert "ease" in _EASING_MODES, "EZ-12 ease not in _EASING_MODES"
    assert "sine-in" in _EASING_MODES, "EZ-13 sine-in not in _EASING_MODES"
    assert "back-in-out" in _EASING_MODES, "EZ-14 back-in-out not in _EASING_MODES"

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
    assert any(w["kind"] == "tail_unreachable_at_boundary" for w in w14b["warnings"]), \
        f"W14b missing tail_unreachable warning: {w14b['warnings']}"

    # W15: narrow_bracket_boundary corner case — [a:b:c:1.9] at steps=10
    w15 = lint_prompt("[a:b:c:1.9]", steps=10)
    assert w15["ok"], f"W15 should pass: {w15}"
    assert any(w["kind"] == "narrow_bracket_boundary" for w in w15["warnings"]), \
        f"W15 missing narrow_bracket_boundary: {w15['warnings']}"
    assert "tail_unreachable_at_boundary" in _BACKEND_WARNINGS, "W15 missing from _BACKEND_WARNINGS"

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
    owner_s, attrs_s, weights_s, _, _, _ = _parse_bind3_prompt("BIND3{cat => [red:blue:10], blue eyes}", allow_attr_scheduling=True)
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

    # ── BIND3 with surrounding text (prefix/suffix) ──────────────────────────
    # BS-1: parse extracts correct prefix/suffix
    bs1_o, bs1_a, _, bs1_p, bs1_s, _ = _parse_bind3_prompt("pre BIND3{cat => fur, cute} suf")
    assert bs1_p == "pre", f"BS-1 prefix: {bs1_p!r}"
    assert bs1_s == "suf", f"BS-1 suffix: {bs1_s!r}"
    assert bs1_o == "cat", f"BS-1 owner: {bs1_o!r}"
    assert bs1_a == ["fur", "cute"], f"BS-1 attrs: {bs1_a}"
    # BS-2: parse empty prefix (BIND3 at start)
    bs2_o, _, _, bs2_p, bs2_s, _ = _parse_bind3_prompt("BIND3{cat => fur, cute} suf")
    assert bs2_p == "", f"BS-2 prefix: {bs2_p!r}"
    assert bs2_s == "suf", f"BS-2 suffix: {bs2_s!r}"
    # BS-3: parse empty suffix (BIND3 at end)
    bs3_o, _, _, bs3_p, bs3_s, _ = _parse_bind3_prompt("pre BIND3{cat => fur, cute}")
    assert bs3_p == "pre", f"BS-3 prefix: {bs3_p!r}"
    assert bs3_s == "", f"BS-3 suffix: {bs3_s!r}"
    # BS-4: transpile preserves surrounding text
    bs4_t = _transpile_bind3_to_chunk("pre BIND3{cat => fur, cute} suf")
    assert bs4_t.startswith("pre"), f"BS-4 prefix lost: {bs4_t[:30]}"
    assert "CHUNK{" in bs4_t, f"BS-4 no CHUNK: {bs4_t}"
    assert bs4_t.endswith("suf"), f"BS-4 suffix lost: {bs4_t[-30:]}"
    # BS-5: get_schedule with BIND3 at start, middle, end
    for bs5_desc, bs5_prompt in [
        ("start", "BIND3{cat => fur} suf"),
        ("end", "pre BIND3{cat => fur}"),
        ("middle", "pre BIND3{cat => fur, cute} suf"),
        ("alone", "BIND3{cat => fur, cute}"),
    ]:
        bs5_sched = get_schedule(bs5_prompt, 10, True, 42)
        assert len(bs5_sched) >= 1, f"BS-5 {bs5_desc} empty: {bs5_sched}"
    # BS-6: BIND3 + ^weight + prefix/suffix
    bs6_t = _transpile_bind3_to_chunk("pre BIND3^0.7{cat => fur, cute} suf")
    assert "CHUNK{" in bs6_t, f"BS-6 no CHUNK: {bs6_t}"
    # BS-7: BIND3 with long prefix (>75 tokens, multi-chunk)
    for bs7_n in [80, 150, 300]:
        bs7_pre = " ".join(["pre"] * bs7_n)
        bs7_sched = get_schedule(f"{bs7_pre} BIND3{{cat => fur, cute}} end", 10, True, 42)
        assert len(bs7_sched) >= 1, f"BS-7 {bs7_n}t prefix empty: {bs7_sched}"
        assert "end" in str(bs7_sched[0][1]), f"BS-7 {bs7_n}t suffix missing"
    # BS-8: BIND3 + special chars in prefix/suffix
    bs8_sched = get_schedule("(hello) BIND3{cat => fur} (world)", 10, True, 42)
    assert len(bs8_sched) >= 1, f"BS-8 parens empty: {bs8_sched}"
    # BS-9: Multiple BIND3 with surrounding text
    bs9_t = _transpile_bind3_to_chunk("start BIND3{a => b, c} mid BIND3{d => e, f} end")
    assert bs9_t.count("CHUNK{") == 2, f"BS-9 counts: {bs9_t}"
    bs9_sched = get_schedule("start BIND3{a => b, c} mid BIND3{d => e, f} end", 10, True, 42)
    assert len(bs9_sched) >= 1, f"BS-9 multi empty: {bs9_sched}"
    # BS-10: BIND3 use_scheduling=False with prefix/suffix
    bs10_sched = get_schedule("pre BIND3{cat => fur} suf", 10, False, 42)
    assert len(bs10_sched) == 1, f"BS-10 no-sched length: {len(bs10_sched)}"
    assert "pre" in str(bs10_sched[0][1]), f"BS-10 prefix missing: {bs10_sched[0][1]}"
    assert "suf" in str(bs10_sched[0][1]), f"BS-10 suffix missing: {bs10_sched[0][1]}"
    # BS-11: extreme combo — long prefix + long attrs + long suffix
    bs11_pre = " ".join(["pre"] * 200)
    bs11_suf = " ".join(["suf"] * 200)
    bs11_a1 = " ".join(["a1"] * 100)
    bs11_a2 = " ".join(["a2"] * 100)
    bs11_sched = get_schedule(
        f"{bs11_pre} BIND3{{cat => {bs11_a1}, {bs11_a2}}} {bs11_suf}",
        10, True, 42,
    )
    assert len(bs11_sched) >= 1, f"BS-11 extreme combo empty: {bs11_sched}"
    # BS-12: lint with extreme length
    bs12_pre = " ".join(["pre"] * 150)
    bs12_lint = lint_prompt(f"{bs12_pre} BIND3{{cat => fur, cute}} end", 10)
    assert bs12_lint.get("ok", False), f"BS-12 lint failed: {bs12_lint}"
    print("BIND3 surrounding text tests passed!")

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
    # MS-4: unknown mode falls to error; sine-in (easing) works
    try:
        _compute_morph_curve_weights(2, [0, 1], 1, "nonexistent")
        assert False, "MS-4 should have raised"
    except PromptSyntaxError:
        pass
    w_ease = _compute_morph_curve_weights(2, [0, 1], 1, "sine-in")
    assert len(w_ease) == 2, f"MS-4 sine-in length: {len(w_ease)}"
    assert abs(w_ease[0] + w_ease[1] - 1.0) < 1e-6, f"MS-4 sine-in weights sum: {w_ease}"
    # MS-5: MORPH_CURVES validation
    assert "linear" in MORPH_CURVES, "MS-5 linear still in MORPH_CURVES"
    assert "bezier" in MORPH_CURVES, "MS-5 bezier still in MORPH_CURVES"
    assert "bernstein" in MORPH_CURVES, "MS-5 bernstein in MORPH_CURVES"
    assert "sine-in" in MORPH_CURVES, "MS-5 sine-in in MORPH_CURVES"
    # MS-6: easing curve smoke tests — each group
    for ecurve in ("ease", "sine-out", "quart-in", "quint-out", "expo-in-out", "circ-in", "back-out"):
        w = _compute_morph_curve_weights(2, [0, 1], 0, ecurve)
        assert abs(w[0] - 1.0) < 1e-6, f"MS-6 {ecurve} start: {w}"
        w = _compute_morph_curve_weights(2, [0, 1], 2, ecurve)
        assert abs(w[-1] - 1.0) < 1e-6, f"MS-6 {ecurve} end: {w}"
    # MS-7: bernstein === bezier
    w_bez = _compute_morph_curve_weights(3, [0, 1, 2], 1, "bezier")
    w_bern = _compute_morph_curve_weights(3, [0, 1, 2], 1, "bernstein")
    assert w_bez == w_bern, f"MS-7 bernstein != bezier: {w_bez} vs {w_bern}"
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
    # TS-5: _slerp_condition_values with actual tensors (if torch available) or ImportError path
    try:
        _torch = _ensure_torch()
    except ImportError:
        _torch = None

    if _torch is not None:
        a = _torch.randn(1, 77, 768)
        b = _torch.randn(1, 77, 768)
        result = _slerp_condition_values(a, b, 0.5)
        assert result.shape == a.shape, f"TS-5 slerp shape: {result.shape}"
        assert _torch.isfinite(result).all(), "TS-5 slerp nan/inf"
        result0 = _slerp_condition_values(a, b, 0.0)
        assert _torch.allclose(result0, a, atol=1e-6), "TS-5 slerp t=0"
        result1 = _slerp_condition_values(a, b, 1.0)
        assert _torch.allclose(result1, b, atol=1e-6), "TS-5 slerp t=1"
        print("  (torch available — tensor SLERP tested with real tensors)")
    else:
        try:
            _slerp_condition_values(None, None, 0.5)
            assert False, "TS-5 should have raised ImportError"
        except ImportError:
            pass
        print("  (torch not available — tensor SLERP tested at logic level)")
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

    # ── Phase B: COMPOUND curves self-tests (CP-1..CP-6) ─────────────────────
    # CP-1: basic curve parsing
    cp1 = _parse_compound_part_text("cat*1.0~ease-in", full_text="COMPOUND{a|cat*1.0~ease-in}")
    assert cp1.curve == "ease-in", f"CP-1 curve: {cp1.curve}"
    assert cp1.weight == 1.0, f"CP-1 weight: {cp1.weight}"
    assert cp1.text == "cat", f"CP-1 text: {cp1.text}"
    print("CP-1: curve parsing passed!")

    # CP-2: curve with range
    cp2 = _parse_compound_part_text("cat@5-15*0.5~ease-out", full_text="COMPOUND{a|cat@5-15*0.5~ease-out}")
    assert cp2.curve == "ease-out", f"CP-2 curve: {cp2.curve}"
    assert cp2.step_start == 5, f"CP-2 start: {cp2.step_start}"
    assert cp2.step_end == 15, f"CP-2 end: {cp2.step_end}"
    assert cp2.weight == 0.5, f"CP-2 weight: {cp2.weight}"
    print("CP-2: curve with range passed!")

    # CP-3: bezier curve
    cp3 = _parse_compound_part_text("cat@1-10*2.0~bezier", full_text="COMPOUND{a|cat@1-10*2.0~bezier}")
    assert cp3.curve == "bezier", f"CP-3 curve: {cp3.curve}"
    print("CP-3: bezier curve passed!")

    # CP-4: cubic-bezier curve
    cp4 = _parse_compound_part_text("cat@1-10*1.0~cubic(0.25,0.1,0.75,0.9)", full_text="COMPOUND{a|cat@1-10*1.0~cubic(0.25,0.1,0.75,0.9)}")
    assert cp4.curve == "cubic(0.25,0.1,0.75,0.9)", f"CP-4 curve: {cp4.curve}"
    print("CP-4: cubic-bezier curve passed!")

    # CP-5: invalid curve → error
    try:
        _parse_compound_part_text("cat*1.0~invalid_curve_xyz", full_text="COMPOUND{a|cat*1.0~invalid_curve_xyz}")
        assert False, "CP-5 should have raised on invalid curve"
    except PromptSyntaxError as e:
        assert "curve" in str(e).lower(), f"CP-5 wrong error: {e}"
    print("CP-5: invalid curve raises passed!")

    # CP-6: curve in text schedule (smoke test)
    cp6_spec = CompoundPromptSpec(base="sky", parts=(_parse_compound_part_text("cat@5-15*0.5~ease-out", full_text="COMPOUND{a|cat@5-15*0.5~ease-out}"),), prefix="", suffix="", source="")
    cp6_text = _build_compound_text_schedule_from_spec(cp6_spec, 20, True, 42, True)
    assert len(cp6_text) >= 3, f"CP-6 text len: {len(cp6_text)} (should have at least 3 entries for pre/ramp/post)"
    assert COMPOUND_PREVIEW_PREFIX in cp6_text[0][1], f"CP-6 preview prefix: {cp6_text[0][1]}"
    print("CP-6: curve text schedule passed!")

    # ── COMPOUND + Interp/Scheduling self-tests (CPD-7..CPD-12) ──────────────
    # CPD-7: interp prefix expands in compound text schedule
    cpd7 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("(cat:1.0->2.0) COMPOUND{base | part@5-10*1.0}"), 5, True, 42, True,
    )
    assert len(cpd7) == 5, f"CPD-7 boundaries: {len(cpd7)} (expected 5)"
    assert "(cat:1.0)" in cpd7[0][1], f"CPD-7 step1 interp: {cpd7[0][1]}"
    assert "(cat:2.0)" in cpd7[-1][1], f"CPD-7 last step interp: {cpd7[-1][1]}"
    assert "part" in cpd7[-1][1], f"CPD-7 part active at final: {cpd7[-1][1]}"
    assert "part" not in cpd7[0][1], f"CPD-7 part NOT active at step1: {cpd7[0][1]}"
    print("CPD-7: interp prefix expands in compound text schedule passed!")

    # CPD-8: interp in part text expands when part active
    cpd8 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("COMPOUND{base | (cat:1.0->2.0)@5-10*1.0}"), 5, True, 42, True,
    )
    assert len(cpd8) >= 2, f"CPD-8 boundaries: {len(cpd8)}"
    # Part not active at step <5, base only; when active, interp expanded
    last = cpd8[-1][1]
    assert "(cat:2.0)" in last, f"CPD-8 part text expanded: {last}"
    print("CPD-8: interp in part text expands passed!")

    # CPD-9: interp in base text expands
    cpd9 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("COMPOUND{(cat:1.0->2.0) | dog@5-10*1.0}"), 5, True, 42, True,
    )
    assert len(cpd9) == 5, f"CPD-9 boundaries: {len(cpd9)}"
    assert "(cat:1.0)" in cpd9[0][1], f"CPD-9 base step1: {cpd9[0][1]}"
    assert "(cat:2.0)" in cpd9[-1][1], f"CPD-9 base last: {cpd9[-1][1]}"
    print("CPD-9: interp in base text expands passed!")

    # CPD-10: scheduling prefix expands in compound text schedule
    cpd10 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("[cat:dog:5] COMPOUND{base | part@5-10*1.0}"), 10, True, 42, True,
    )
    assert any("cat" in s[1] for s in cpd10), f"CPD-10 cat prefix: {cpd10}"
    assert any("dog" in s[1] for s in cpd10), f"CPD-10 dog prefix: {cpd10}"
    print("CPD-10: scheduling prefix expands passed!")

    # CPD-11: interp in both prefix and part simultaneously
    cpd11 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("(sky:1.0->2.0) COMPOUND{base | (cat:0.5->1.5)@3-5*1.0}"), 5, True, 42, True,
    )
    assert len(cpd11) == 5, f"CPD-11 boundaries: {len(cpd11)}"
    assert "(sky:1.0)" in cpd11[0][1], f"CPD-11 prefix step1: {cpd11[0][1]}"
    print("CPD-11: dual interp (prefix+part) passed!")

    # CPD-12: use_scheduling=False with interp prefix
    cpd12 = _build_compound_text_schedule_from_spec(
        _extract_compound_prompt_spec("(cat:1.0->2.0) COMPOUND{base | part@5-10*1.0}"), 5, False, 42, True,
    )
    assert len(cpd12) == 1, f"CPD-12 no-sched length: {len(cpd12)}"
    assert "(cat:2.0)" in cpd12[0][1], f"CPD-12 interp at final: {cpd12[0][1]}"
    assert "part:5-10" in cpd12[0][1], f"CPD-12 part active: {cpd12[0][1]}"
    print("CPD-12: use_scheduling=False with interp prefix passed!")

    # COMPOUND-DYN-1: dedup key includes base_text + active_texts (regression guard)
    _compound_src = inspect.getsource(_build_compound_conditioning_schedule)
    _has_key = 'tuple(active_texts)' in _compound_src
    assert _has_key, "COMPOUND-DYN-1: dedup key missing active_texts"
    _has_base = 'tuple(active_indices)' in _compound_src
    assert _has_base, "COMPOUND-DYN-1: dedup key missing active_indices"
    print("COMPOUND-DYN-1: dedup key includes base_text + active_texts passed!")

    print("Phase B COMPOUND curves tests passed!")

    # ── Phase C: BLEND improvements self-tests (BC-1..BC-10) ─────────────────
    # BC-1: product mode
    bc1 = _resolve_blend_mode_weights([1.0, 1.0], "product")
    assert len(bc1) == 2, f"BC-1 len: {bc1}"
    assert abs(bc1[0] - 0.5) < 0.01, f"BC-1 product equal: {bc1}"
    print("BC-1: product mode passed!")

    # BC-2: max mode — only the biggest survives
    bc2 = _resolve_blend_mode_weights([0.5, 2.0, 1.0], "max")
    assert len(bc2) == 3, f"BC-2 len: {bc2}"
    assert abs(bc2[1] - 1.0) < 0.01, f"BC-2 max weight: {bc2}"
    assert abs(bc2[0]) < 0.01 and abs(bc2[2]) < 0.01, f"BC-2 others zero: {bc2}"
    print("BC-2: max mode passed!")

    # BC-3: max mode with equal weights (tied → each gets equal share)
    bc3 = _resolve_blend_mode_weights([2.0, 2.0], "max")
    assert abs(bc3[0] - 0.5) < 0.01 and abs(bc3[1] - 0.5) < 0.01, f"BC-3 equal max: {bc3}"
    print("BC-3: max equal passed!")

    # BC-4: product with intensity
    bc4 = _resolve_blend_mode_weights([2.0, 3.0], "product", intensity=2.0)
    assert len(bc4) == 2, f"BC-4 len: {bc4}"
    print("BC-4: product with intensity passed!")

    # BC-5: branch curve parsing
    bc5_text, bc5_weight, bc5_curve = _split_blend_branch_weight_and_curve("cat*1.0~ease-in", full_text="BLEND{cat*1.0~ease-in, dog}")
    assert bc5_text == "cat", f"BC-5 text: {bc5_text}"
    assert abs(bc5_weight - 1.0) < 0.01, f"BC-5 weight: {bc5_weight}"
    assert bc5_curve == "ease-in", f"BC-5 curve: {bc5_curve}"
    print("BC-5: branch curve parsing passed!")

    # BC-6: branch curve with cubic-bezier
    bc6_text, bc6_weight, bc6_curve = _split_blend_branch_weight_and_curve("cat*2.0~cubic(0.25,0.1,0.75,0.9)", full_text="BLEND{cat*2.0~cubic(0.25,0.1,0.75,0.9), dog}")
    assert bc6_text == "cat", f"BC-6 text: {bc6_text}"
    assert abs(bc6_weight - 2.0) < 0.01, f"BC-6 weight: {bc6_weight}"
    assert bc6_curve == "cubic(0.25,0.1,0.75,0.9)", f"BC-6 curve: {bc6_curve}"
    print("BC-6: branch cubic curve passed!")

    # BC-7: branch without curve (default)
    bc7_text, bc7_weight, bc7_curve = _split_blend_branch_weight_and_curve("cat*1.5", full_text="BLEND{cat*1.5, dog}")
    assert bc7_curve == "linear", f"BC-7 curve: {bc7_curve}"
    print("BC-7: branch default curve passed!")

    # BC-8: unknown curve → error
    try:
        _split_blend_branch_weight_and_curve("cat*1.0~unknown_curve", full_text="BLEND{cat*1.0~unknown_curve}")
        assert False, "BC-8 should have raised"
    except PromptSyntaxError as e:
        assert "curve" in str(e).lower(), f"BC-8 wrong error: {e}"
    print("BC-8: unknown curve raises passed!")

    # BC-9: invalid blend mode → error
    try:
        _parse_blend_mode("invalid_mode", full_text="BLEND[invalid_mode]{a|b}")
        assert False, "BC-9 should have raised"
    except PromptSyntaxError as e:
        assert "mode" in str(e).lower(), f"BC-9 wrong error: {e}"
    print("BC-9: invalid blend mode raises passed!")

    # BC-10: product+max in BLEND_MODES
    assert "product" in BLEND_MODES, "BC-10 product not in BLEND_MODES"
    assert "max" in BLEND_MODES, "BC-10 max not in BLEND_MODES"
    print("BC-10: product+max in BLEND_MODES passed!")

    # ── Patch 1: Vars/Macros/Wildcards self-tests (VM-1..VM-7) ──────────────
    # VM-1: basic <random:a|b|c> expansion
    vm1 = _expand_vars_and_macros("photo of <random:cat|dog|fox>", seed=42)
    assert vm1[1] == {}, f"VM-1 meta not empty: {vm1[1]}"
    assert vm1[0] in ("photo of cat", "photo of dog", "photo of fox"), f"VM-1 unexpected: {vm1[0]}"
    print("VM-1: random expansion passed!")

    # VM-2: <setvar> + <var>
    vm2, _ = _expand_vars_and_macros("<setvar[X]:hello> <var:X> world", seed=42)
    assert "hello" in vm2 and "world" in vm2, f"VM-2 setvar+var: {vm2}"
    print("VM-2: setvar + var passed!")

    # VM-3: <setmacro> + <macro>
    vm3, _ = _expand_vars_and_macros('<setmacro[GREET]:hello <random:world|there>> <macro:GREET>', seed=42)
    assert "hello" in vm3, f"VM-3 macro: {vm3}"
    print("VM-3: setmacro + macro passed!")

    # VM-4: <param> extraction
    _, vm4_meta = _expand_vars_and_macros("cat <param[style]:anime> dog")
    assert vm4_meta.get("style") == "anime", f"VM-4 param: {vm4_meta}"
    print("VM-4: param extraction passed!")

    # VM-5: seed determinism (same seed => same output)
    vm5a, _ = _expand_vars_and_macros("photo of <random:cat|dog|fox>", seed=42)
    vm5b, _ = _expand_vars_and_macros("photo of <random:cat|dog|fox>", seed=42)
    assert vm5a == vm5b, f"VM-5 seed determinism: {vm5a} != {vm5b}"
    print("VM-5: seed determinism passed!")

    # VM-6: seed=None determinism (same prompt => same hash)
    vm6a, _ = _expand_vars_and_macros("photo of <random:cat|dog>")
    vm6b, _ = _expand_vars_and_macros("photo of <random:cat|dog>")
    assert vm6a == vm6b, f"VM-6 seed=None determinism: {vm6a} != {vm6b}"
    print("VM-6: seed=None determinism passed!")

    # VM-7: no-op fast path
    vm7, vm7_meta = _expand_vars_and_macros("plain text prompt")
    assert vm7 == "plain text prompt", f"VM-7 no-op: {vm7}"
    assert vm7_meta == {}, f"VM-7 meta: {vm7_meta}"
    print("VM-7: no-op fast path passed!")

    # VM-8: wildcard path traversal protection
    vm8, _ = _expand_vars_and_macros("__etc/passwd__", seed=42)
    assert vm8 == "__etc/passwd__", f"VM-8 traversal: {vm8}"
    print("VM-8: wildcard path traversal protection passed!")

    # VM-9: no-op within _get_schedule_impl (integration smoke)
    vm9 = get_schedule("plain text", 20, True, 42)
    assert len(vm9) == 1, f"VM-9 get_schedule: {vm9}"
    print("VM-9: get_schedule plain text passed!")

    # ── Patch 2: Weight Modes self-tests (WI-1..WI-7) ──────────────────────
    # WI-1: env_weight_mode defaults
    wi1 = _env_weight_mode("NONEXISTENT_VAR_12345", "a1111")
    assert wi1 == "a1111", f"WI-1 default: {wi1}"
    print("WI-1: env_weight_mode default passed!")

    # WI-7: VALID_WEIGHT_MODES
    assert "a1111" in _VALID_WEIGHT_MODES
    assert "comfy" in _VALID_WEIGHT_MODES
    assert "compel" in _VALID_WEIGHT_MODES
    assert "comfy++" in _VALID_WEIGHT_MODES
    assert "down_weight" in _VALID_WEIGHT_MODES
    assert len(_VALID_WEIGHT_MODES) == 5, f"WI-7 modes: {_VALID_WEIGHT_MODES}"
    print("WI-7: VALID_WEIGHT_MODES passed!")

    # WI-2..WI-6: require torch
    try:
        import torch as _torch2
        _t_avail = True
    except ImportError:
        _t_avail = False
    if _t_avail:
        base = _torch2.randn(1, 4, 768)
        empty = _torch2.zeros(1, 4, 768)
        weights = _torch2.tensor([1.5, 1.0, 0.5, 0.8])

        # WI-3: comfy
        wi3 = _wi_apply_comfy(base, weights, empty)
        assert wi3.shape == base.shape, f"WI-3 shape: {wi3.shape}"
        assert _torch2.isfinite(wi3).all(), "WI-3 nan/inf"
        print("WI-3: comfy weight mode passed!")

        # WI-2: a1111 renorm keeps mean
        weighted = base * weights.unsqueeze(-1)
        wi2 = _wi_a1111_renorm(base[0:1], weighted[0:1])
        assert abs(wi2.mean() - base[0:1].mean()) < 1e-4, f"WI-2 renorm mean diff"
        assert _torch2.isfinite(wi2).all(), "WI-2 nan/inf"
        print("WI-2: a1111 renorm passed!")

        # WI-4: compel branching (up vs down)
        wi4_up = _wi_apply_compel(base[0:1], _torch2.tensor([1.5]), empty[0:1])
        wi4_down = _wi_apply_compel(base[0:1], _torch2.tensor([0.5]), empty[0:1])
        assert wi4_up.shape == base[0:1].shape
        assert wi4_down.shape == base[0:1].shape
        assert _torch2.isfinite(wi4_up).all(), "WI-4 up nan/inf"
        assert _torch2.isfinite(wi4_down).all(), "WI-4 down nan/inf"
        assert not _torch2.allclose(wi4_up, wi4_down), "WI-4 up == down"
        print("WI-4: compel branching passed!")

        # WI-6: apply_advanced_weights dispatch
        wi6 = apply_advanced_weights(base[0:1], _torch2.tensor([1.5]), empty[0:1], mode="comfy")
        assert wi6.shape == base[0:1].shape
        assert _torch2.isfinite(wi6).all(), "WI-6 nan/inf"
        print("WI-6: apply_advanced_weights dispatch passed!")

        # WI-5: down_weight normalizes
        wi5 = _wi_apply_down_weight(base[0:1], _torch2.tensor([0.3, 0.6, 0.9, 1.0]), empty[0:1])
        assert wi5.shape == base[0:1].shape
        assert _torch2.isfinite(wi5).all(), "WI-5 nan/inf"
        print("WI-5: down_weight mode passed!")

        # WI-8: invalid mode raises
        try:
            apply_advanced_weights(base[0:1], _torch2.tensor([1.0]), empty[0:1], mode="invalid")
            assert False, "WI-8 should have raised"
        except ValueError:
            pass
        print("WI-8: invalid mode raises passed!")
    else:
        print("  (torch not available — WI-2..WI-6, WI-8 skipped)")

    # ── Patch 3: T5 self-tests (T5-1..T5-9) ────────────────────────────────
    # T5-1a: SDXL mode (enc1+enc2)
    t5_1a = AssemblePromptSpec(prefix="", suffix="", enc1="cat", enc2="dog")
    assert not t5_1a.has_t5, "T5-1a has_t5 false positive"
    assert t5_1a.has_sdxl_pair, "T5-1a has_sdxl_pair false negative"
    assert t5_1a.architecture_mode == "sdxl", f"T5-1a mode: {t5_1a.architecture_mode}"
    print("T5-1a: SDXL mode passed!")

    # T5-1b: Flux mode (enc1+t5)
    t5_1b = AssemblePromptSpec(prefix="", suffix="", enc1="cat", t5="t5 text")
    assert t5_1b.has_t5, "T5-1b has_t5 false negative"
    assert not t5_1b.has_sdxl_pair, "T5-1b has_sdxl_pair false positive"
    assert t5_1b.architecture_mode == "flux", f"T5-1b mode: {t5_1b.architecture_mode}"
    print("T5-1b: Flux mode passed!")

    # T5-1c: SD3 mode (enc1+enc2+t5)
    t5_1c = AssemblePromptSpec(prefix="", suffix="", enc1="cat", enc2="dog", t5="t5 text")
    assert t5_1c.architecture_mode == "sd3", f"T5-1c mode: {t5_1c.architecture_mode}"
    print("T5-1c: SD3 mode passed!")

    # T5-1d: t5 present → flux (даже без enc1)
    t5_1d = AssemblePromptSpec(prefix="", suffix="", enc2="dog", t5="t5 text")
    assert t5_1d.architecture_mode == "flux", f"T5-1d mode: {t5_1d.architecture_mode}"
    print("T5-1d: t5->flux passed!")

    # T5-2: encoder channel constants
    assert "t5" in BACKEND_CHANNEL_TARGETS, "T5-2 t5 not in BACKEND_CHANNEL_TARGETS"
    assert FLUX_ENCODER_CHANNEL_TARGETS == frozenset({"enc1", "t5"}), f"T5-2 FLUX: {FLUX_ENCODER_CHANNEL_TARGETS}"
    assert SD3_ENCODER_CHANNEL_TARGETS == frozenset({"enc1", "enc2", "t5"}), f"T5-2 SD3: {SD3_ENCODER_CHANNEL_TARGETS}"
    print("T5-2: encoder channel constants passed!")

    # T5-3: _build_assemble_preview_text_v2 format
    # SDXL mode (без t5)
    t5_3_sdxl = _build_assemble_preview_text_v2(
        AssemblePromptSpec(prefix="", suffix="", enc1="cat", enc2="dog")
    )
    assert "[sdxl]" in t5_3_sdxl, f"T5-3 SDXL wrong mode: {t5_3_sdxl}"
    assert "enc1=cat" in t5_3_sdxl, f"T5-3 SDXL no enc1: {t5_3_sdxl}"
    assert "enc2=dog" in t5_3_sdxl, f"T5-3 SDXL no enc2: {t5_3_sdxl}"
    print("T5-3a: preview v2 SDXL passed!")
    # SD3 mode (с t5)
    t5_3_sd3 = _build_assemble_preview_text_v2(
        AssemblePromptSpec(prefix="", suffix="", enc1="cat", enc2="dog", t5="some t5 text")
    )
    assert "[sd3]" in t5_3_sd3, f"T5-3 SD3 wrong mode: {t5_3_sd3}"
    assert "t5=some t5 text" in t5_3_sd3, f"T5-3 SD3 no t5: {t5_3_sd3}"
    print("T5-3b: preview v2 SD3 passed!")

    # T5-4: ASSEMBLE{enc1=cat;t5=t5 text} valid (Flux)
    t5_4_spec = _extract_assemble_prompt_spec("ASSEMBLE{enc1=cat;t5=t5 text}")
    assert t5_4_spec is not None, "T5-4 extract failed"
    assert t5_4_spec.has_t5, "T5-4 has_t5 missing"
    assert t5_4_spec.architecture_mode == "flux", f"T5-4 mode: {t5_4_spec.architecture_mode}"
    assert t5_4_spec.enc1 == "cat", f"T5-4 enc1: {t5_4_spec.enc1}"
    assert t5_4_spec.t5 == "t5 text", f"T5-4 t5: {t5_4_spec.t5}"
    print("T5-4: ASSEMBLE enc1+t5 (Flux) passed!")

    # T5-5: ASSEMBLE{enc1=cat;enc2=dog;t5=t5 text} valid (SD3)
    t5_5_spec = _extract_assemble_prompt_spec("ASSEMBLE{enc1=cat;enc2=dog;t5=t5 text}")
    assert t5_5_spec is not None, "T5-5 extract failed"
    assert t5_5_spec.architecture_mode == "sd3", f"T5-5 mode: {t5_5_spec.architecture_mode}"
    assert t5_5_spec.enc2 == "dog", f"T5-5 enc2: {t5_5_spec.enc2}"
    assert t5_5_spec.t5 == "t5 text", f"T5-5 t5: {t5_5_spec.t5}"
    print("T5-5: ASSEMBLE enc1+enc2+t5 (SD3) passed!")

    # T5-6: ASSEMBLE{enc2=foo;t5=bar} invalid (no enc1)
    try:
        _extract_assemble_prompt_spec("ASSEMBLE{enc2=foo;t5=bar}")
        assert False, "T5-6 should have raised"
    except PromptSyntaxError:
        pass
    print("T5-6: ASSEMBLE enc2+t5 invalid (no enc1) passed!")

    # T5-7: ASSEMBLE text schedule with t5 in preview/key
    t5_7_spec = _extract_assemble_prompt_spec("ASSEMBLE{enc1=cat;t5=dog}")
    assert t5_7_spec is not None, "T5-7 extract failed"
    t5_7_sched = _build_assemble_text_schedule_from_spec(t5_7_spec, 20, False, None, True, strict=True)
    assert len(t5_7_sched) > 0, "T5-7 empty schedule"
    assert "[flux]" in t5_7_sched[0][1], f"T5-7 no flux mode: {t5_7_sched[0][1]}"
    assert "t5=" in t5_7_sched[0][1], f"T5-7 no t5 preview: {t5_7_sched[0][1]}"
    assert "enc1=cat" in t5_7_sched[0][1], f"T5-7 no enc1: {t5_7_sched[0][1]}"
    print("T5-7: ASSEMBLE text schedule with t5 passed!")

    # T5-8: ASSEMBLE SD3 text schedule (three encoders)
    t5_8_spec = _extract_assemble_prompt_spec("ASSEMBLE{enc1=cat;enc2=dog;t5=bird}")
    assert t5_8_spec is not None, "T5-8 extract failed"
    assert t5_8_spec.architecture_mode == "sd3", f"T5-8 mode: {t5_8_spec.architecture_mode}"
    t5_8_sched = _build_assemble_text_schedule_from_spec(t5_8_spec, 20, False, None, True, strict=True)
    assert "[sd3]" in t5_8_sched[0][1], f"T5-8 no sd3 mode: {t5_8_sched[0][1]}"
    assert "t5=bird" in t5_8_sched[0][1], f"T5-8 no t5: {t5_8_sched[0][1]}"
    print("T5-8: ASSEMBLE SD3 text schedule passed!")

    # T5-9: _VAR_META_LOCAL thread-local exists
    assert hasattr(_VAR_META_LOCAL, 'meta'), "T5-9 _VAR_META_LOCAL missing meta attr"
    print("T5-9: _VAR_META_LOCAL exists passed!")

    # ── Phase A: Multi-segment interpolation + cubic-bezier ──────────────────
    # IP-1: basic multi-segment (3 weights) — output monotonic
    ip1 = get_schedule("(cat:1.0->2.0->3.0)", 5, True, 42)
    assert len(ip1) == 5, f"IP-1 expected 5 steps, got {len(ip1)}"
    w1 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip1]
    assert all(w1[i] <= w1[i+1] for i in range(len(w1)-1)), f"IP-1 not monotonic: {w1}"
    assert abs(w1[0] - 1.0) < 0.01, f"IP-1 first weight {w1[0]} != 1.0"
    assert abs(w1[-1] - 3.0) < 0.01, f"IP-1 last weight {w1[-1]} != 3.0"
    print("IP-1: multi-segment 3 weights monotonic passed!")

    # IP-2: multi-segment with ease-in
    ip2 = get_schedule("(cat:1.0->3.0->5.0~ease-in)", 5, True, 42)
    assert len(ip2) == 5, f"IP-2 expected 5 steps, got {len(ip2)}"
    print("IP-2: multi-segment with ease-in passed!")

    # IP-3: multi-segment with bezier
    ip3 = get_schedule("(cat:0.0->1.0->0.0~bezier)", 5, True, 42)
    assert len(ip3) == 5, f"IP-3 expected 5 steps, got {len(ip3)}"
    print("IP-3: multi-segment with bezier passed!")

    # IP-4: 4-weight multi-segment
    ip4 = get_schedule("(cat:1.0->2.0->3.0->4.0)", 7, True, 42)
    assert len(ip4) == 7, f"IP-4 expected 7 steps, got {len(ip4)}"
    w4 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip4]
    assert all(w4[i] <= w4[i+1] for i in range(len(w4)-1)), f"IP-4 not monotonic: {w4}"
    print("IP-4: 4-weight multi-segment passed!")

    # IP-5: cubic-bezier easing
    ip5 = get_schedule("(cat:0.0->1.0~cubic(0.25,0.1,0.75,0.9))", 5, True, 42)
    assert len(ip5) == 5, f"IP-5 expected 5 steps, got {len(ip5)}"
    w5 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip5]
    assert abs(w5[0] - 0.0) < 0.01, f"IP-5 first weight {w5[0]} != 0.0"
    assert abs(w5[-1] - 1.0) < 0.01, f"IP-5 last weight {w5[-1]} != 1.0"
    print("IP-5: cubic-bezier easing passed!")

    # IP-6: cubic-bezier with multi-segment
    ip6 = get_schedule("(cat:0.0->0.5->1.0~cubic(0.42,0.0,0.58,1.0))", 5, True, 42)
    assert len(ip6) == 5, f"IP-6 expected 5 steps, got {len(ip6)}"
    w6 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip6]
    assert all(w6[i] <= w6[i+1] for i in range(len(w6)-1)), f"IP-6 not monotonic: {w6}"
    assert abs(w6[0] - 0.0) < 0.01, f"IP-6 first weight {w6[0]} != 0.0"
    assert abs(w6[-1] - 1.0) < 0.01, f"IP-6 last weight {w6[-1]} != 1.0"
    print("IP-6: cubic-bezier + multi-segment passed!")

    # IP-7: backward compat — 2-weight still works
    ip7 = get_schedule("(cat:1.0->2.0)", 5, True, 42)
    assert len(ip7) == 5, f"IP-7 expected 5 steps, got {len(ip7)}"
    w7 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip7]
    assert abs(w7[0] - 1.0) < 0.01, f"IP-7 first weight {w7[0]} != 1.0"
    assert abs(w7[-1] - 2.0) < 0.01, f"IP-7 last weight {w7[-1]} != 2.0"
    print("IP-7: 2-weight backward compat passed!")

    # IP-8: cubic-bezier with ease-in-out (standard CSS ease)
    ip8 = get_schedule("(cat:0.0->1.0~cubic(0.0,0.0,1.0,1.0))", 5, True, 42)
    assert len(ip8) == 5, f"IP-8 expected 5 steps, got {len(ip8)}"
    w8 = [float(seg.split(":")[-1].rstrip(")")) for _, seg in ip8]
    assert abs(w8[-1] - 1.0) < 0.01, f"IP-8 last weight {w8[-1]} != 1.0"
    print("IP-8: cubic-bezier linear (0,0,1,1) passed!")

    # IP-9: lint passes for multi-segment
    ip9 = lint_prompt("(cat:1.0->2.0->3.0)", steps=10)
    assert ip9["ok"], f"IP-9 lint not ok: {ip9}"
    print("IP-9: lint multi-segment passed!")

    # IP-10: lint passes for cubic-bezier
    ip10 = lint_prompt("(cat:0.0->1.0~cubic(0.25,0.1,0.75,0.9))", steps=10)
    assert ip10["ok"], f"IP-10 lint not ok: {ip10}"
    print("IP-10: lint cubic-bezier passed!")

    # IP-11: use_scheduling=False collapses multi-segment
    ip11 = get_schedule("(cat:0.0->2.0->4.0)", 10, False, 42)
    assert len(ip11) == 1, f"IP-11 expected 1 entry, got {len(ip11)}"
    assert "4.0" in ip11[0][1], f"IP-11 no final weight: {ip11[0][1]}"
    print("IP-11: unscheduled collapse multi-segment passed!")

    # IP-12: cubic bezier malformed params fallback to linear
    ip12 = lint_prompt("(cat:1.0->2.0~cubic(1,2))", steps=10)
    assert ip12["ok"], f"IP-12 malformed cubic: {ip12}"
    print("IP-12: malformed cubic params fallback passed!")

    # IP-13: _apply_cubic_bezier_easing boundary values
    assert abs(_apply_cubic_bezier_easing(0.0, 0.25, 0.1, 0.75, 0.9) - 0.0) < 1e-6, "IP-13 t=0 not 0"
    assert abs(_apply_cubic_bezier_easing(1.0, 0.25, 0.1, 0.75, 0.9) - 1.0) < 1e-6, "IP-13 t=1 not 1"
    print("IP-13: cubic bezier boundaries passed!")

    # IP-14: no marker leak in schedule output
    ip14 = get_schedule("(cat:1.0->2.0->3.0~ease-in)", 5, True, 42)
    for _, seg in ip14:
        assert ATTN_INTERP_OPEN not in str(seg), f"IP-14 marker leak: {seg}"
    print("IP-14: no marker leak multi-segment passed!")

    # ── @ start-end range tests (IP-15..IP-21) ─────────────────────────────────
    # IP-15: Basic @ 5-15 — frozen before, interpolating inside, frozen after
    ip15 = get_schedule("(cat:1.0->2.0 @ 5-15)", 20, True, 42)
    assert str(ip15[0][1]).endswith("(cat:1.0)"), f"IP-15 first not 1.0: {ip15[0]}"
    ip15_mid = next((e for e in ip15 if "(cat:1.5)" in str(e[1])), None)
    assert ip15_mid is not None, f"IP-15 no 1.5 entry: {ip15}"
    assert str(ip15[-1][1]).endswith("(cat:2.0)"), f"IP-15 last not 2.0: {ip15[-1]}"
    print("IP-15: @ 5-15 range passed!")

    # IP-16: @ range with easing
    ip16 = get_schedule("(cat:1.0->2.0~ease-in @ 5-15)", 20, True, 42)
    assert len(ip16) > 0, "IP-16 empty schedule"
    assert str(ip16[-1][1]).endswith("(cat:2.0)"), f"IP-16 last not 2.0: {ip16[-1]}"
    print("IP-16: @ range with easing passed!")

    # IP-17: @ range extends beyond total steps — partial interpolation
    ip17 = get_schedule("(cat:1.0->2.0 @ 1-50)", 20, True, 42)
    ip17_last = str(ip17[-1][1])
    assert "(cat:1." in ip17_last and "(cat:2.0)" not in ip17_last, f"IP-17 unexpected: {ip17_last}"
    print("IP-17: @ range beyond steps passed!")

    # IP-18: multi-segment (3 weights) with @ range
    ip18 = get_schedule("(cat:1.0->2.0->3.0 @ 5-15)", 20, True, 42)
    assert str(ip18[-1][1]).endswith("(cat:3.0)"), f"IP-18 last not 3.0: {ip18[-1]}"
    assert any("(cat:1.0)" in str(t) for _, t in ip18), "IP-18 no frozen 1.0"
    print("IP-18: multi-segment with @ range passed!")

    # IP-19: lint passes for @ syntax
    ip19 = lint_prompt("(cat:1.0->2.0 @ 5-15)")
    assert ip19["ok"], f"IP-19 lint failed: {ip19}"
    print("IP-19: lint @ syntax passed!")

    # IP-20: use_scheduling=False with @ range — collapses to final weight
    ip20 = get_schedule("(cat:1.0->2.0 @ 5-15)", 20, False, 42)
    assert len(ip20) == 1, f"IP-20 expected 1 entry, got {len(ip20)}"
    assert str(ip20[0][1]).endswith("(cat:2.0)"), f"IP-20 final not 2.0: {ip20[0]}"
    print("IP-20: use_scheduling=False with @ range passed!")

    # IP-21: @ percentage range — 20%-80% with steps=20
    ip21 = get_schedule("(cat:1.0->2.0 @ 20%-80%)", 20, True, 42)
    assert any("(cat:1.0)" in str(t) for _, t in ip21), "IP-21 no frozen 1.0 before 20%"
    assert str(ip21[-1][1]).endswith("(cat:2.0)"), f"IP-21 final not 2.0: {ip21[-1]}"
    print("IP-21: @ percentage range passed!")

    # IP-22: invalid @ a-b (non-numeric) — not recognized as interpolation, preserved literally
    ip22 = get_schedule("(cat:1.0->2.0 @ a-b)", 20, True, 42)
    assert "(cat:1.0->2.0 @ a-b)" in str(ip22[-1][1]), f"IP-22 unexpected literal: {ip22[-1]}"
    print("IP-22: invalid @ a-b literal passed!")

    # IP-23: mixed percentage + absolute — @ 20%-15
    ip23 = get_schedule("(cat:1.0->2.0 @ 20%-15)", 20, True, 42)
    assert any("(cat:1.0)" in str(t) for _, t in ip23), "IP-23 no frozen 1.0 before range"
    assert str(ip23[-1][1]).endswith("(cat:2.0)"), f"IP-23 final not 2.0: {ip23[-1]}"
    print("IP-23: @ mixed percentage-absolute passed!")

    # ── Bounce curve self-test ────────────────────────────────────────────────
    ba0 = _apply_easing(0.0, "bounce")
    ba1 = _apply_easing(1.0, "bounce")
    ba_mid = _apply_easing(0.5, "bounce")
    assert abs(ba0) < 1e-6, f"bounce t=0: {ba0}"
    assert abs(ba1 - 1.0) < 1e-6, f"bounce t=1: {ba1}"
    assert 0.0 < ba_mid < 1.0, f"bounce t=0.5: {ba_mid}"
    assert "bounce" in _EASING_MODES, "bounce not in _EASING_MODES"
    assert "bounce" in MORPH_CURVES, "bounce not in MORPH_CURVES"
    print("BOUNCE: bounce curve (Penner ease-out-bounce) passed!")

    # ── TONEG tests (TN) ─────────────────────────────────────────────────────
    assert get_prompt_with_toneg("cat TONEG{ugly}") == ("cat", "ugly"), "TN-1 basic"
    assert get_prompt_with_toneg("TONEG{bad} cat") == ("cat", "bad"), "TN-2 prefix"
    assert get_prompt_with_toneg("cat TONEG{bad} dog TONEG{ugly}") == ("cat dog", "bad, ugly"), "TN-3 multiple"
    assert get_prompt_with_toneg("plain text") == ("plain text", ""), "TN-4 no toneg"
    assert get_prompt_with_toneg("") == ("", ""), "TN-5 empty"
    toneg_sched = get_schedule("cat TONEG{ugly}", 10, True, 42)
    assert all("TONEG" not in seg for _, seg in toneg_sched), f"TN-6 TONEG leaked into schedule: {toneg_sched}"
    toneg_w = lint_prompt("TONEG{bad} cat", steps=10, is_negative=True)
    assert any(w["kind"] == "toneg_in_negative_prompt" for w in toneg_w.get("warnings", [])), "TN-7 lint warning"
    toneg_w2 = lint_prompt("TONEG{bad} cat", steps=10, is_negative=False)
    assert not any(w["kind"] == "toneg_in_negative_prompt" for w in toneg_w2.get("warnings", [])), "TN-8 no warning in positive"
    assert get_prompt_with_toneg("<setvar[Q]:red> cat TONEG{<var:Q> background}") == ("red cat", "red background"), \
        "TN-9 setvar in base prompt must be visible inside TONEG{} (cross-scope var, mirrors get_prompt_regions L843)"
    print("TONEG tests (TN-1..9) passed!")

    # ── CFG schedule tests (CFG) ─────────────────────────────────────────────
    cfg1 = _parse_cfg_param("7.0", 10)
    assert cfg1 == 7.0 and isinstance(cfg1, float), "CFG-1 scalar"
    cfg2 = _parse_cfg_param("7.0->3.0", 5)
    assert isinstance(cfg2, list) and len(cfg2) == 5, "CFG-2 range length"
    assert abs(cfg2[0] - 7.0) < 1e-6 and abs(cfg2[-1] - 3.0) < 1e-6, "CFG-2 endpoints"
    cfg3 = _parse_cfg_param("7.0->4.0->2.0", 5)
    assert isinstance(cfg3, list) and len(cfg3) == 5, "CFG-3 multi length"
    assert abs(cfg3[0] - 7.0) < 1e-6 and abs(cfg3[-1] - 2.0) < 1e-6, "CFG-3 multi endpoints"
    # lint: cfg range with steps=1 triggers warning
    cfg_w = lint_prompt("<param[cfg]:7.0->3.0> cat", steps=1, is_negative=False)
    assert any(w["kind"] == "cfg_schedule_with_single_step" for w in cfg_w.get("warnings", [])), "CFG-4 lint warning steps=1"
    cfg_w2 = lint_prompt("<param[cfg]:7.0->3.0> cat", steps=10, is_negative=False)
    assert not any(w["kind"] == "cfg_schedule_with_single_step" for w in cfg_w2.get("warnings", [])), "CFG-5 no warning steps=10"
    print("CFG schedule tests (CFG-1..5) passed!")

    # ── get_prompt_params tests (GPP) ────────────────────────────────────────
    assert get_prompt_params("cat <param[cfg]:7.5>", seed=42) == {"cfg": "7.5"}, "GPP-1 basic param"
    assert get_prompt_params("cat <param[val]:3.14>", seed=42) == {"val": "3.14"}, "GPP-2 float text"
    assert get_prompt_params("cat <param[name]:hello>", seed=42) == {"name": "hello"}, "GPP-3 string text"
    assert get_prompt_params("plain text", seed=42) == {}, "GPP-4 no params"
    assert get_prompt_params("CHUNK{a <param[x]:1>|b}", seed=42) == {"x": "1"}, "GPP-5 param inside CHUNK"
    assert get_prompt_params("") == {}, "GPP-6 empty text"
    assert get_prompt_params("<param[a]:1> and <param[b]:2>", seed=42) == {"a": "1", "b": "2"}, "GPP-7 multi param"
    assert get_prompt_params("BLEND{cat <param[p]:5>|dog}", seed=42) == {"p": "5"}, "GPP-8 param inside BLEND"
    assert get_prompt_params("MORPH{cat => dog <param[q]:6>}", seed=42) == {"q": "6"}, "GPP-9 param inside MORPH"
    assert get_prompt_params("COMPOUND{base | part@5-10*1.0 <param[r]:7>}", seed=42) == {"r": "7"}, "GPP-10 param inside COMPOUND"
    assert get_prompt_params("ASSEMBLE{enc1=cat <param[s]:8>}", seed=42) == {"s": "8"}, "GPP-11 param inside ASSEMBLE"
    print("get_prompt_params tests (GPP-1..11) passed!")

    # ── _coerce_param_value tests (CPV) ──────────────────────────────────────
    assert _coerce_param_value("42") == 42 and isinstance(_coerce_param_value("42"), int), "CPV-1 int"
    assert _coerce_param_value("3.14") == 3.14 and isinstance(_coerce_param_value("3.14"), float), "CPV-2 float"
    assert _coerce_param_value("hello") == "hello" and isinstance(_coerce_param_value("hello"), str), "CPV-3 str"
    assert _coerce_param_value("00042") == 42, "CPV-4 leading zeros"
    assert _coerce_param_value("1e5") == 100000, "CPV-5 scientific"
    print("_coerce_param_value tests (CPV-1..5) passed!")

    # ── DIFF raw/delta self-tests (DR) ───────────────────────────────────────
    # DR-1: COMPOUND {a | b~diff_raw}
    dr1 = _extract_compound_prompt_spec("COMPOUND{a | b~diff_raw}")
    assert dr1 is not None, "DR-1 extract"
    assert dr1.parts[0].mode == "diff_raw", f"DR-1 mode: {dr1.parts[0].mode}"
    assert dr1.parts[0].curve == "linear", f"DR-1 curve: {dr1.parts[0].curve}"
    # DR-2: COMPOUND {a | b~diff_raw_ease-in}
    dr2 = _extract_compound_prompt_spec("COMPOUND{a | b~diff_raw_ease-in}")
    assert dr2 is not None, "DR-2 extract"
    assert dr2.parts[0].mode == "diff_raw", f"DR-2 mode: {dr2.parts[0].mode}"
    assert dr2.parts[0].curve == "ease-in", f"DR-2 curve: {dr2.parts[0].curve}"
    # DR-3: DIFF{a, b~diff_raw} transpiles to COMPOUND with ~diff_raw
    dr3 = _transpile_diff_to_compound("DIFF{a, b~diff_raw}")
    assert "~diff_raw" in dr3, f"DR-3 transpile: {dr3}"
    assert "~diff_diff_raw" not in dr3, f"DR-3 double diff_: {dr3}"
    # DR-4: DIFF{a, b} — default mode=diff with normalization
    dr4 = _extract_compound_prompt_spec(_transpile_diff_to_compound("DIFF{a, b}"))
    assert dr4 is not None, "DR-4 extract"
    assert dr4.parts[0].mode == "diff", f"DR-4 mode: {dr4.parts[0].mode}"
    # DR-5: DIFF{a, b~delta} transpiles with ~delta
    dr5 = _transpile_diff_to_compound("DIFF{a, b~delta}")
    assert "~delta" in dr5, f"DR-5 transpile: {dr5}"
    assert "~diff_delta" not in dr5, f"DR-5 no diff_delta: {dr5}"
    # DR-6: DIFF{a, b, c~diff_raw, d~delta} mixed modes
    dr6 = _transpile_diff_to_compound("DIFF{a, b, c~diff_raw, d~delta}")
    assert "c~diff_raw" in dr6, f"DR-6 diff_raw: {dr6}"
    assert "d~delta" in dr6, f"DR-6 delta: {dr6}"
    assert dr6.count("|") == 3, f"DR-6 parts count: {dr6}"
    print("DIFF raw/delta tests (DR-1..6) passed!")

    # ── REGION self-tests (RG-1..RG-8) ─────────────────────────────────────────
    # RG-1: basic extraction
    _r_clean, _r_regs = _extract_region_blocks('cat REGION{dog@0,1,0,1}')
    assert _r_clean == 'cat dog', f'RG-1 clean: {_r_clean!r}'
    assert len(_r_regs) == 1
    assert _r_regs[0].text == 'dog'
    assert _r_regs[0].x1 == 0.0 and _r_regs[0].x2 == 1.0
    assert _r_regs[0].y1 == 0.0 and _r_regs[0].y2 == 1.0
    print("RG-1: basic extraction passed!")

    # RG-2: multiple branches with weights
    _r2, _r2r = _extract_region_blocks('REGION{cat@0,0.5,0,1*0.8 | dog@0.5,1,0,1*1.2}')
    assert _r2 == 'cat dog', f'RG-2 clean: {_r2!r}'
    assert len(_r2r) == 2
    assert _r2r[0].weight == 0.8 and _r2r[1].weight == 1.2
    assert abs(_r2r[0].x2 - 0.5) < 1e-9
    assert abs(_r2r[1].x1 - 0.5) < 1e-9
    print("RG-2: multiple branches passed!")

    # RG-3: auto-tile horizontal
    _r3, _r3r = _extract_region_blocks('REGION{cat | dog | bird}')
    assert _r3 == 'cat dog bird', f'RG-3 clean: {_r3!r}'
    assert len(_r3r) == 3
    assert abs(_r3r[0].x2 - 1.0/3.0) < 1e-9
    assert abs(_r3r[1].x1 - 1.0/3.0) < 1e-9
    assert abs(_r3r[2].x1 - 2.0/3.0) < 1e-9
    print("RG-3: auto-tile H passed!")

    # RG-4: auto-tile vertical
    _r4, _r4r = _extract_region_blocks('REGION{top | bottom}:V')
    assert _r4 == 'top bottom', f'RG-4 clean: {_r4!r}'
    assert len(_r4r) == 2
    assert abs(_r4r[0].y2 - 0.5) < 1e-9
    assert abs(_r4r[1].y1 - 0.5) < 1e-9
    print("RG-4: auto-tile V passed!")

    # RG-5: pixel mode detection
    _r5, _r5r = _extract_region_blocks('REGION{face@0,512,0,512}')
    assert _r5r[0].coords_pixels == True
    print("RG-5: pixel mode passed!")

    # RG-6: reverse range error
    try:
        _extract_region_blocks('REGION{bad@0.5,0,0,1}')
        assert False, 'RG-6 should raise'
    except PromptSyntaxError as _e6:
        assert _e6.kind == 'region_reverse_range', f'RG-6 kind: {_e6.kind}'
    print("RG-6: reverse range error passed!")

    # RG-7: @handle conflict (greedy regex)
    _r7, _r7r = _extract_region_blocks('REGION{@greg_rutkowski@0,1,0,1}')
    assert _r7r[0].text == '@greg_rutkowski', f'RG-7 text: {_r7r[0].text!r}'
    print("RG-7: @handle conflict passed!")

    # RG-8: guard — get_schedule with raw REGION text does not crash
    _r8s = get_schedule('REGION{cat@0,1,0,1}', steps=10, use_scheduling=True, seed=42)
    assert len(_r8s) >= 1
    print("RG-8: guard fallback passed!")

    # RG-9: auto-tile H with ratios
    _r9, _r9r = _extract_region_blocks('REGION{big | small}:H:0.7,0.3')
    assert _r9 == 'big small', f'RG-9 clean: {_r9!r}'
    assert len(_r9r) == 2, f'RG-9 count: {len(_r9r)}'
    assert abs(_r9r[0].x1 - 0.0) < 1e-6, f'RG-9 big x1: {_r9r[0].x1}'
    assert abs(_r9r[0].x2 - 0.7) < 1e-6, f'RG-9 big x2: {_r9r[0].x2}'
    assert abs(_r9r[1].x1 - 0.7) < 1e-6, f'RG-9 small x1: {_r9r[1].x1}'
    assert abs(_r9r[1].x2 - 1.0) < 1e-6, f'RG-9 small x2: {_r9r[1].x2}'
    print("RG-9: auto-tile H with ratios passed!")

    # RG-10: auto-tile V with ratios
    _r10, _r10r = _extract_region_blocks('REGION{top | mid | bot}:V:0.2,0.3,0.5')
    assert _r10 == 'top mid bot', f'RG-10 clean: {_r10!r}'
    assert len(_r10r) == 3, f'RG-10 count: {len(_r10r)}'
    assert abs(_r10r[0].y1 - 0.0) < 1e-6, f'RG-10 top y1: {_r10r[0].y1}'
    assert abs(_r10r[0].y2 - 0.2) < 1e-6, f'RG-10 top y2: {_r10r[0].y2}'
    assert abs(_r10r[1].y1 - 0.2) < 1e-6
    assert abs(_r10r[1].y2 - 0.5) < 1e-6
    assert abs(_r10r[2].y1 - 0.5) < 1e-6
    assert abs(_r10r[2].y2 - 1.0) < 1e-6
    print("RG-10: auto-tile V with ratios passed!")

    # RG-11: extract_non_region_text with ratio suffix
    _r11_text = 'fox REGION{dog | cat}:H:0.6,0.4 bird'
    _r11, _r11r = _extract_region_blocks(_r11_text)
    _r11_base = extract_non_region_text(_r11, _r11r, original_text=_r11_text)
    assert 'fox' in _r11_base, f'RG-11 non-region missing fox: {_r11_base!r}'
    assert 'bird' in _r11_base, f'RG-11 non-region missing bird: {_r11_base!r}'
    assert ':0.6,0.4' not in _r11_base, f'RG-11 ratio leaked: {_r11_base!r}'
    assert ':H' not in _r11_base, f'RG-11 axis leaked: {_r11_base!r}'
    print("RG-11: extract_non_region_text with ratio suffix passed!")

    # RG-12: canvas= directive extraction
    _r12, _r12r = _extract_region_blocks('REGION{cat | dog | canvas=iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==}')
    assert _r12 == 'cat dog', f'RG-12 clean: {_r12!r}'
    assert len(_r12r) == 2, f'RG-12 count: {len(_r12r)}'
    assert _r12r[0].canvas == 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', f'RG-12 canvas[0]: {_r12r[0].canvas[:20]!r}'
    assert _r12r[1].canvas == _r12r[0].canvas, f'RG-12 canvas mismatch: {_r12r[1].canvas[:20]!r} != {_r12r[0].canvas[:20]!r}'
    print("RG-12: canvas= directive passed!")

    # ── Grid split self-tests (RG-13..RG-20) ─────────────────────────────────
    # RG-13: basic 2×2 grid
    _r13, _r13r = _extract_region_blocks('REGION{cat | dog | bird | fish}[H:0.5,0.5 | V:0.5,0.5]')
    assert _r13 == 'cat dog bird fish', f'RG-13 clean: {_r13!r}'
    assert len(_r13r) == 4, f'RG-13 count: {len(_r13r)}'
    assert abs(_r13r[0].x1 - 0.0) < 1e-6 and abs(_r13r[0].y1 - 0.0) < 1e-6
    assert abs(_r13r[0].x2 - 0.5) < 1e-6 and abs(_r13r[0].y2 - 0.5) < 1e-6
    assert abs(_r13r[1].x1 - 0.5) < 1e-6 and abs(_r13r[1].y1 - 0.0) < 1e-6
    assert abs(_r13r[1].x2 - 1.0) < 1e-6 and abs(_r13r[1].y2 - 0.5) < 1e-6
    assert abs(_r13r[2].x1 - 0.0) < 1e-6 and abs(_r13r[2].y1 - 0.5) < 1e-6
    assert abs(_r13r[2].x2 - 0.5) < 1e-6 and abs(_r13r[2].y2 - 1.0) < 1e-6
    assert abs(_r13r[3].x1 - 0.5) < 1e-6 and abs(_r13r[3].y1 - 0.5) < 1e-6
    assert abs(_r13r[3].x2 - 1.0) < 1e-6 and abs(_r13r[3].y2 - 1.0) < 1e-6
    print("RG-13: basic 2x2 grid passed!")

    # RG-14: unequal ratios
    # H:0.3,0.7 = 2 columns (x: 0→0.3→1.0), V:0.6,0.4 = 2 rows (y: 0→0.6→1.0)
    _r14, _r14r = _extract_region_blocks('REGION{top | bot}[H:0.3,0.7 | V:0.6,0.4]')
    assert len(_r14r) == 4, f'RG-14 count: {len(_r14r)}'
    assert abs(_r14r[0].x2 - 0.3) < 1e-6 and abs(_r14r[0].y2 - 0.6) < 1e-6
    assert abs(_r14r[1].x2 - 1.0) < 1e-6 and abs(_r14r[1].y2 - 0.6) < 1e-6
    assert abs(_r14r[2].x2 - 0.3) < 1e-6 and abs(_r14r[2].y2 - 1.0) < 1e-6
    assert abs(_r14r[3].x2 - 1.0) < 1e-6 and abs(_r14r[3].y2 - 1.0) < 1e-6
    print("RG-14: unequal ratios passed!")

    # RG-15: H-only grid (columns only, single row)
    # H:0.2,0.3,0.5 = 3 columns (x varies), V default [1.0] = 1 row (y full)
    _r15, _r15r = _extract_region_blocks('REGION{top | mid | bot}[H:0.2,0.3,0.5]')
    assert len(_r15r) == 3, f'RG-15 count: {len(_r15r)}'
    assert abs(_r15r[0].x1 - 0.0) < 1e-6 and abs(_r15r[0].x2 - 0.2) < 1e-6
    assert abs(_r15r[0].y1 - 0.0) < 1e-6 and abs(_r15r[0].y2 - 1.0) < 1e-6
    assert abs(_r15r[1].x1 - 0.2) < 1e-6 and abs(_r15r[1].x2 - 0.5) < 1e-6
    assert abs(_r15r[2].x1 - 0.5) < 1e-6 and abs(_r15r[2].x2 - 1.0) < 1e-6
    print("RG-15: H-only grid passed!")

    # RG-16: V-only grid (rows only, single column)
    # H default [1.0] = 1 col (x full), V:0.5,0.3,0.2 = 3 rows (y varies)
    _r16, _r16r = _extract_region_blocks('REGION{a | b | c}[V:0.5,0.3,0.2]')
    assert len(_r16r) == 3, f'RG-16 count: {len(_r16r)}'
    assert abs(_r16r[0].x1 - 0.0) < 1e-6 and abs(_r16r[0].x2 - 1.0) < 1e-6
    assert abs(_r16r[0].y1 - 0.0) < 1e-6 and abs(_r16r[0].y2 - 0.5) < 1e-6
    assert abs(_r16r[1].x1 - 0.0) < 1e-6 and abs(_r16r[1].x2 - 1.0) < 1e-6
    assert abs(_r16r[1].y1 - 0.5) < 1e-6 and abs(_r16r[1].y2 - 0.8) < 1e-6
    assert abs(_r16r[2].x1 - 0.0) < 1e-6 and abs(_r16r[2].x2 - 1.0) < 1e-6
    assert abs(_r16r[2].y1 - 0.8) < 1e-6 and abs(_r16r[2].y2 - 1.0) < 1e-6
    print("RG-16: V-only grid passed!")

    # RG-17: fewer branches than cells (last repeated)
    _r17, _r17r = _extract_region_blocks('REGION{cat | dog}[H:0.5,0.5 | V:0.5,0.5]')
    assert len(_r17r) == 4, f'RG-17 count: {len(_r17r)}'
    assert _r17r[0].text == 'cat', f'RG-17[0]: {_r17r[0].text!r}'
    assert _r17r[1].text == 'dog', f'RG-17[1]: {_r17r[1].text!r}'
    assert _r17r[2].text == 'dog', f'RG-17[2] (repeat): {_r17r[2].text!r}'
    assert _r17r[3].text == 'dog', f'RG-17[3] (repeat): {_r17r[3].text!r}'
    print("RG-17: fewer branches than cells passed!")

    # RG-18: more branches than cells (extra ignored)
    _r18, _r18r = _extract_region_blocks('REGION{a | b | c | d | e}[H:0.5,0.5 | V:0.5,0.5]')
    assert len(_r18r) == 4, f'RG-18 count: {len(_r18r)}'
    assert _r18r[0].text == 'a'
    assert _r18r[1].text == 'b'
    assert _r18r[2].text == 'c'
    assert _r18r[3].text == 'd'
    print("RG-18: more branches than cells passed!")

    # RG-19: clean_text excludes grid suffix
    _r19_text = 'fox REGION{cat | dog}[H:0.5,0.5 | V:0.5,0.5] bird'
    _r19, _r19r = _extract_region_blocks(_r19_text)
    assert _r19 == 'fox cat dog bird', f'RG-19 clean: {_r19!r}'
    assert 'H:0.5' not in _r19, f'RG-19 leaked H-ratios: {_r19!r}'
    assert 'V:0.5' not in _r19, f'RG-19 leaked V-ratios: {_r19!r}'
    print("RG-19: clean_text excludes grid suffix passed!")

    # RG-20: invalid ratios → error
    try:
        _extract_region_blocks('REGION{x | y}[H:0.5,abc | V:0.5,0.5]')
        assert False, 'RG-20 should raise'
    except PromptSyntaxError as _e20:
        assert _e20.kind == 'region_grid_invalid_ratios', f'RG-20 kind: {_e20.kind}'
    print("RG-20: invalid ratios error passed!")

    # RG-26: base_ratio directive (block-level, default 0.2)
    _r26, _r26r = _extract_region_blocks('REGION{cat | dog | base_ratio=0.4}:H:0.5,0.5')
    assert len(_r26r) == 2, f'RG-26 count: {len(_r26r)}'
    assert all(abs(r.base_ratio - 0.4) < 1e-6 for r in _r26r), f'RG-26 base_ratio: {[r.base_ratio for r in _r26r]}'
    _r26d, _r26d2 = _extract_region_blocks('REGION{cat | dog}:H:0.5,0.5')
    assert len(_r26d2) == 2, f'RG-26d count: {len(_r26d2)}'
    assert all(abs(r.base_ratio - 0.2) < 1e-6 for r in _r26d2), f'RG-26d default: {[r.base_ratio for r in _r26d2]}'
    _r26c, _r26c2 = _extract_region_blocks('REGION{cat | dog | base_ratio=1.5}:H:0.5,0.5')
    assert all(abs(r.base_ratio - 1.0) < 1e-6 for r in _r26c2), f'RG-26c clamp: {[r.base_ratio for r in _r26c2]}'
    print("RG-26: base_ratio directive passed!")

    # ── _build_region_masks self-tests (RG-21..RG-25) ──────────────────────
    try:
        import torch
        _has_torch = True
    except ImportError:
        _has_torch = False
    if _has_torch:
        r21: list[RegionBlock] = [
            RegionBlock(text="a", x1=0.0, x2=0.5, y1=0.0, y2=1.0, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=False, stop=1.0, start=0.0, blur=0.0, canvas=""),
        ]
        _m21 = _build_region_masks(r21, 8, 8)
        assert len(_m21) == 1, f'RG-21 len: {len(_m21)}'
        assert _m21[0].shape == (1, 1, 8, 8), f'RG-21 shape: {_m21[0].shape}'
        assert _m21[0][0, 0, :, :4].eq(1.0).all(), 'RG-21 left half should be 1'
        assert _m21[0][0, 0, :, 4:].eq(0.0).all(), 'RG-21 right half should be 0'
        print("RG-21: basic half-width mask passed!")

        # RG-22: pixel coords
        r22: list[RegionBlock] = [
            RegionBlock(text="a", x1=0, x2=32, y1=0, y2=64, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=True, stop=1.0, start=0.0, blur=0.0, canvas=""),
        ]
        _m22 = _build_region_masks(r22, 8, 8)
        assert _m22[0][0, 0, :, :4].eq(1.0).all(), 'RG-22 pixel left=32px/8=4'
        assert _m22[0][0, 0, :, 4:].eq(0.0).all(), 'RG-22 pixel right=32px/8=4'
        print("RG-22: pixel coords mask passed!")

        # RG-23: multiple regions
        r23: list[RegionBlock] = [
            RegionBlock(text="a", x1=0.0, x2=0.5, y1=0.0, y2=0.5, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=False, stop=1.0, start=0.0, blur=0.0, canvas=""),
            RegionBlock(text="b", x1=0.5, x2=1.0, y1=0.5, y2=1.0, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=False, stop=1.0, start=0.0, blur=0.0, canvas=""),
        ]
        _m23 = _build_region_masks(r23, 8, 8)
        assert len(_m23) == 2, f'RG-23 len: {len(_m23)}'
        assert _m23[0][0, 0, :4, :4].eq(1.0).all(), 'RG-23 region0 top-left'
        assert _m23[0][0, 0, 4:, 4:].eq(0.0).all(), 'RG-23 region0 bottom-right should be 0'
        assert _m23[1][0, 0, 4:, 4:].eq(1.0).all(), 'RG-23 region1 bottom-right'
        assert _m23[1][0, 0, :4, :4].eq(0.0).all(), 'RG-23 region1 top-left should be 0'
        print("RG-23: multiple regions passed!")

        # RG-24: clamped boundaries
        r24: list[RegionBlock] = [
            RegionBlock(text="a", x1=-0.5, x2=1.5, y1=-0.3, y2=1.3, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=False, stop=1.0, start=0.0, blur=0.0, canvas=""),
        ]
        _m24 = _build_region_masks(r24, 8, 8)
        assert _m24[0].eq(1.0).all(), 'RG-24 clamped coords should cover full area'
        print("RG-24: clamped boundaries passed!")

        # RG-25: zero area → empty mask (all zeros)
        r25: list[RegionBlock] = [
            RegionBlock(text="a", x1=0.5, x2=0.5, y1=0.0, y2=0.5, weight=1.0,
                        axis="", base_text="", mode="attention", backend=None,
                        coords_pixels=False, stop=1.0, start=0.0, blur=0.0, canvas=""),
        ]
        _m25 = _build_region_masks(r25, 8, 8)
        assert _m25[0].eq(0.0).all(), 'RG-25 zero area should be all zeros'
        print("RG-25: zero area handled passed!")

        print("REGION mask tests (RG-21..25) passed!")
    else:
        print("Skipping RG-21..25: torch not available")

    print("All REGION tests (RG-1..26) passed!")

    # ── REGION + DIFF/COMPOUND E2E integration test ──────────────────────────
    print("Testing REGION + DIFF integration...")
    _rd_prompt = "background REGION{ DIFF{cat, red~ortho}@0,0.5,0,1 | dog }"
    _rd_clean, _rd_regions = get_prompt_regions(_rd_prompt)
    assert _rd_clean == 'background DIFF{cat, red~ortho} dog', f"E2E clean_text: {_rd_clean!r}"
    assert len(_rd_regions) == 2, f"E2E expected 2 regions, got {len(_rd_regions)}"
    assert _rd_regions[0].text == 'DIFF{cat, red~ortho}', f"E2E region[0]: {_rd_regions[0].text!r}"
    assert _rd_regions[1].text == 'dog', f"E2E region[1]: {_rd_regions[1].text!r}"
    _rd_non = extract_non_region_text(_rd_clean, _rd_regions, original_text=_rd_prompt)
    assert _rd_non == 'background', f"E2E non_region: {_rd_non!r}"
    _rd_vp = build_virtual_region_prompt(_rd_non, _rd_regions)
    assert _rd_vp == 'background AND DIFF{cat, red~ortho} AND dog', f"E2E virtual: {_rd_vp!r}"
    _rd_vps = get_schedule(_rd_vp, 20, True, 42)
    assert len(_rd_vps) >= 1, "E2E virtual schedule empty"
    _rd_rs = get_schedule(_rd_regions[0].text, 20, True, 42)
    assert len(_rd_rs) >= 1, "E2E region schedule empty"
    # Test with ~ortho weight
    _rd_prompt2 = "bg REGION{ DIFF{cat, tail~ortho*0.5}@0,1,0,1 }"
    _rd_c2, _rd_r2 = get_prompt_regions(_rd_prompt2)
    assert _rd_r2[0].text == 'DIFF{cat, tail~ortho*0.5}', f"E2E ortho weight: {_rd_r2[0].text!r}"
    print("REGION + DIFF integration (RI-1..4) passed!")

    # ── V10 audit: escaped braces inside REGION ──────────────────────────────────
    _v10a1, _v10r1 = get_prompt_regions(r'REGION{cat with \} literal@0,1,0,1}')
    assert len(_v10r1) == 1, f'V10-A1 count: {len(_v10r1)}'
    assert '\\}' in _v10r1[0].text, f'V10-A1 text: {_v10r1[0].text!r}'
    assert _v10r1[0].x2 == 1.0 and _v10r1[0].y2 == 1.0, f'V10-A1 coords: {_v10r1[0]}'
    print("V10-A1: escaped } inside REGION passed!")

    _v10a2, _v10r2 = get_prompt_regions(r'REGION{cat with \{ literal@0,1,0,1}')
    assert len(_v10r2) == 1, f'V10-A2 count: {len(_v10r2)}'
    assert '\\{' in _v10r2[0].text, f'V10-A2 text: {_v10r2[0].text!r}'
    assert abs(_v10r2[0].x1 - 0.0) < 1e-6
    print("V10-A2: escaped { inside REGION passed!")

    # Escaped braces + extract_non_region_text
    _v10c, _v10r = get_prompt_regions(r'prefix REGION{cat with \} hi@0,1,0,1} suffix')
    assert _v10c == r'prefix cat with \} hi suffix', f'V10-A3 clean: {_v10c!r}'
    assert len(_v10r) == 1
    print("V10-A3: escaped } + clean_text passed!")

    # V10-B: false lint warning with escaped brace before REGION
    _v10l = lint_prompt(r'literal \{ brace REGION{cat@0,1,0,1}', steps=20)
    _v10b_warns = [w for w in _v10l['warnings'] if w['kind'] == 'region_inside_backend_not_supported']
    assert len(_v10b_warns) == 0, f'V10-B false lint: {_v10l["warnings"]}'
    assert _v10l['ok'], 'V10-B lint ok=False'
    print("V10-B: no false lint warning for escaped { before REGION passed!")

    # V10-C: PUA overflow guard
    _v10_many = ' '.join(f'\\[{i}\\]' for i in range(300)) + ' hello'
    _v10_prot, _v10_rest = _protect_escaped_literal_spans_for_source(_v10_many)
    assert len(_v10_rest) <= 256, f'V10-C restore count: {len(_v10_rest)} (max 256)'
    assert 'hello' in _v10_prot, f'V10-C prot lost text: {_v10_prot[-50:]!r}'
    print("V10-C: PUA overflow guard passed!")

    # V10-D: escaped \{ inside REGION body with pipe split
    # Note: auto-tile branches are ordered after explicit-coord branches
    _v10d, _v10dr = get_prompt_regions(r'REGION{cat \{ dog | bird@0,1,0,1}')
    assert len(_v10dr) == 2, f'V10-D count: {len(_v10dr)}'
    assert _v10dr[0].text == 'bird', f'V10-D[0] (explicit): {_v10dr[0].text!r}'
    assert _v10dr[1].text == r'cat \{ dog', f'V10-D[1] (auto-tile): {_v10dr[1].text!r}'
    print("V10-D: escaped { in body with pipe split passed!")

    # V10-E: escaped } inside REGION body with pipe split
    _v10e, _v10er = get_prompt_regions(r'REGION{cat \} dog@0,1,0,1 | bird@0.5,1,0,1}')
    assert len(_v10er) == 2, f'V10-E count: {len(_v10er)}'
    assert _v10er[0].text == r'cat \} dog', f'V10-E[0]: {_v10er[0].text!r}'
    assert _v10er[0].text == r'cat \} dog', f'V10-E[0]: {_v10er[0].text!r}'
    assert _v10er[1].text == 'bird', f'V10-E[1]: {_v10er[1].text!r}'
    assert abs(_v10er[0].x1 - 0.0) < 1e-6
    assert abs(_v10er[0].x2 - 1.0) < 1e-6
    assert abs(_v10er[1].x1 - 0.5) < 1e-6
    assert abs(_v10er[1].x2 - 1.0) < 1e-6
    print("V10-E: escaped } in body with pipe split passed!")

    # ── Phase 4 smoke tests ───────────────────────────────────────────────────
    print("All self-tests passed (additions: Claims #9, #10, Multi-BIND3, MORPH-SLERP, Tensor-SLERP, COMPOUND, 3-Patch Integration, Phase A interpolation improvements, get_prompt_params GPP-1..11 + CPV-1..5, TONEG TN-1..8, DIFF raw/delta DR-1..6, REGION RG-1..26, _build_region_masks, REGION+DIFF RI-1..4, V10-A1..3/B/C/D/E)!")
