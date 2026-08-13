from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

THEME_TOKEN_SCHEMA_VERSION = 1
MIN_TEXT_CONTRAST_RATIO = 4.5
MIN_UI_CONTRAST_RATIO = 3.0

_HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{6})$")

SEMANTIC_THEME_TOKEN_DEFAULTS: dict[str, str] = {
    "color.accent": "#179ee7",
    "color.surface.primary": "#111d29",
    "color.surface.secondary": "#172431",
    "color.component.surface": "#111d29",
    "color.component.border": "#2b4358",
    "color.component.accent": "#179ee7",
    "color.text.primary": "#f4f9fd",
    "color.text.secondary": "#9db2c4",
    "color.border": "#2b4358",
    "color.success": "#35c978",
    "color.warning": "#f5b94b",
    "color.error": "#ef6262",
    "typography.family": "Inter",
    "buttons.primary.text": "#ffffff",
    "buttons.secondary.text": "#d5f1ff",
    "radius.panel": "12px",
    "radius.control": "7px",
    "shadow.panel": "0 12px 30px rgba(0, 0, 0, 0.26)",
}

SEMANTIC_PALETTE_KEYS: dict[str, str] = {
    "surface_secondary": "color.surface.secondary",
    "component_surface": "color.component.surface",
    "component_border": "color.component.border",
    "component_accent": "color.component.accent",
    "text_primary": "color.text.primary",
    "text_secondary": "color.text.secondary",
}


def normalize_hex_color(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    short = re.fullmatch(r"#?([0-9a-fA-F]{3})", text)
    if short:
        expanded = "".join(char * 2 for char in short.group(1))
        return f"#{expanded.lower()}"
    full = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
    if full:
        return f"#{full.group(1).lower()}"
    return str(fallback).lower()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    color = normalize_hex_color(hex_color, "#000000")[1:]
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def mix_hex(source: str, target: str, target_weight: float) -> str:
    left = _rgb(source)
    right = _rgb(target)
    weight = max(0.0, min(1.0, float(target_weight)))
    channels = tuple(round(a + ((b - a) * weight)) for a, b in zip(left, right))
    return "#" + "".join(f"{value:02x}" for value in channels)


def relative_luminance(hex_color: str) -> float:
    channels = []
    for value in _rgb(hex_color):
        normalized = value / 255.0
        channels.append(
            normalized / 12.92
            if normalized <= 0.03928
            else ((normalized + 0.055) / 1.055) ** 2.4
        )
    return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2])


def contrast_ratio(left: str, right: str) -> float:
    a = relative_luminance(left)
    b = relative_luminance(right)
    lighter = max(a, b)
    darker = min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def derive_semantic_palette_defaults(*, accent: str, surface: str) -> dict[str, str]:
    accent_color = normalize_hex_color(accent, SEMANTIC_THEME_TOKEN_DEFAULTS["color.accent"])
    surface_color = normalize_hex_color(surface, SEMANTIC_THEME_TOKEN_DEFAULTS["color.surface.primary"])
    return {
        "surface_secondary": mix_hex(surface_color, "#ffffff", 0.08),
        "component_surface": surface_color,
        "component_border": mix_hex(surface_color, "#ffffff", 0.22),
        "component_accent": accent_color,
        "text_primary": SEMANTIC_THEME_TOKEN_DEFAULTS["color.text.primary"],
        "text_secondary": SEMANTIC_THEME_TOKEN_DEFAULTS["color.text.secondary"],
    }


def normalize_semantic_palette(
    value: Any,
    *,
    accent: str,
    surface: str,
) -> dict[str, str]:
    defaults = derive_semantic_palette_defaults(accent=accent, surface=surface)
    source = value if isinstance(value, Mapping) else {}
    return {
        key: normalize_hex_color(source.get(key), fallback)
        for key, fallback in defaults.items()
    }


def legacy_palette_to_semantic_tokens(value: Mapping[str, Any]) -> dict[str, str]:
    accent_block = value.get("accent") if isinstance(value.get("accent"), Mapping) else {}
    surface_block = value.get("surface") if isinstance(value.get("surface"), Mapping) else {}
    typography = value.get("typography") if isinstance(value.get("typography"), Mapping) else {}
    accent = normalize_hex_color(accent_block.get("color"), SEMANTIC_THEME_TOKEN_DEFAULTS["color.accent"])
    surface = normalize_hex_color(surface_block.get("color"), SEMANTIC_THEME_TOKEN_DEFAULTS["color.surface.primary"])
    semantic = normalize_semantic_palette(value.get("semantic"), accent=accent, surface=surface)
    output = dict(SEMANTIC_THEME_TOKEN_DEFAULTS)
    output.update(
        {
            "color.accent": accent,
            "color.surface.primary": surface,
            "typography.family": str(typography.get("font_family") or output["typography.family"]),
            "buttons.primary.text": normalize_hex_color(
                typography.get("primary_button_text"), output["buttons.primary.text"]
            ),
            "buttons.secondary.text": normalize_hex_color(
                typography.get("secondary_button_text"), output["buttons.secondary.text"]
            ),
        }
    )
    for palette_key, token_key in SEMANTIC_PALETTE_KEYS.items():
        output[token_key] = semantic[palette_key]
    output["color.border"] = semantic["component_border"]
    return output


def _flatten_tokens(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        token_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(_flatten_tokens(item, token_key))
        else:
            output[token_key] = item
    return output


def normalize_semantic_tokens(value: Any, *, base: Mapping[str, str] | None = None) -> dict[str, str]:
    output = dict(base or SEMANTIC_THEME_TOKEN_DEFAULTS)
    if not isinstance(value, Mapping):
        return output
    source = _flatten_tokens(value)
    for key, raw in source.items():
        if key not in SEMANTIC_THEME_TOKEN_DEFAULTS:
            continue
        default = SEMANTIC_THEME_TOKEN_DEFAULTS[key]
        if key.startswith("color.") or key.startswith("buttons."):
            output[key] = normalize_hex_color(raw, output.get(key, default))
        else:
            text = str(raw or "").strip()
            output[key] = text or output.get(key, default)
    return output


def semantic_tokens_to_legacy_palette(
    tokens: Mapping[str, Any],
    *,
    accent_name: str = "Custom",
    surface_name: str = "Custom",
) -> dict[str, Any]:
    normalized = normalize_semantic_tokens(tokens)
    return {
        "accent": {"name": accent_name, "color": normalized["color.accent"]},
        "surface": {"name": surface_name, "color": normalized["color.surface.primary"]},
        "typography": {
            "font_family": normalized["typography.family"],
            "primary_button_text": normalized["buttons.primary.text"],
            "secondary_button_text": normalized["buttons.secondary.text"],
        },
        "semantic": {
            palette_key: normalized[token_key]
            for palette_key, token_key in SEMANTIC_PALETTE_KEYS.items()
        },
    }


@dataclass(frozen=True)
class ContrastCheck:
    foreground_token: str
    background_token: str
    ratio: float
    required_ratio: float
    blocking: bool
    valid: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "foregroundToken": self.foreground_token,
            "backgroundToken": self.background_token,
            "ratio": round(self.ratio, 2),
            "requiredRatio": self.required_ratio,
            "blocking": self.blocking,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ThemeContrastValidationResult:
    valid: bool
    checks: tuple[ContrastCheck, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": [check.to_dict() for check in self.checks],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_semantic_theme_contrast(tokens: Mapping[str, Any]) -> ThemeContrastValidationResult:
    normalized = normalize_semantic_tokens(tokens)
    checks: list[ContrastCheck] = []
    errors: list[str] = []
    warnings: list[str] = []

    readability_pairs = [
        ("color.text.primary", "color.surface.primary", MIN_TEXT_CONTRAST_RATIO),
        ("color.text.primary", "color.surface.secondary", MIN_TEXT_CONTRAST_RATIO),
        ("color.text.primary", "color.component.surface", MIN_TEXT_CONTRAST_RATIO),
        ("color.text.secondary", "color.surface.primary", MIN_TEXT_CONTRAST_RATIO),
        ("color.text.secondary", "color.surface.secondary", MIN_TEXT_CONTRAST_RATIO),
        ("color.text.secondary", "color.component.surface", MIN_TEXT_CONTRAST_RATIO),
    ]
    advisory_pairs = [
        ("color.component.border", "color.component.surface", MIN_UI_CONTRAST_RATIO),
        ("color.component.accent", "color.component.surface", MIN_UI_CONTRAST_RATIO),
        ("color.accent", "color.surface.primary", MIN_UI_CONTRAST_RATIO),
        ("buttons.primary.text", "color.accent", MIN_UI_CONTRAST_RATIO),
        ("buttons.secondary.text", "color.surface.secondary", MIN_UI_CONTRAST_RATIO),
    ]

    def evaluate(foreground: str, background: str, required: float, *, blocking: bool) -> None:
        fg = normalized[foreground]
        bg = normalized[background]
        ratio = contrast_ratio(fg, bg)
        identical = fg.casefold() == bg.casefold()
        valid = (not identical) and ratio >= required
        reason = ""
        if identical:
            reason = "Foreground and background use the same color and may make text or controls unreadable."
        elif not valid:
            reason = f"Contrast is {ratio:.2f}:1; minimum is {required:.1f}:1."
        checks.append(
            ContrastCheck(
                foreground_token=foreground,
                background_token=background,
                ratio=ratio,
                required_ratio=required,
                blocking=blocking,
                valid=valid,
                reason=reason,
            )
        )
        if valid:
            return
        message = f"{foreground} vs {background}: {reason}"
        if blocking:
            errors.append(message)
        else:
            warnings.append(message)

    for foreground, background, required in readability_pairs:
        evaluate(foreground, background, required, blocking=False)
    for foreground, background, required in advisory_pairs:
        evaluate(foreground, background, required, blocking=False)

    return ThemeContrastValidationResult(
        valid=not errors,
        checks=tuple(checks),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_palette_contrast(value: Mapping[str, Any]) -> ThemeContrastValidationResult:
    return validate_semantic_theme_contrast(legacy_palette_to_semantic_tokens(value))
