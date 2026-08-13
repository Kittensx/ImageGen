from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from image_gen.webui.theme.contracts import known_theme_capabilities

PROHIBITED_THEME_PACKAGE_SUFFIXES = frozenset(
    {
        ".js",
        ".mjs",
        ".cjs",
        ".py",
        ".bat",
        ".cmd",
        ".ps1",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
    }
)

_SVG_UNSAFE_PATTERNS = (
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"<\s*foreignObject\b", re.IGNORECASE),
    re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE),
    re.compile(r"(?:href|xlink:href)\s*=\s*['\"]\s*(?:https?:|//|javascript:)", re.IGNORECASE),
)

_CSS_UNSAFE_PATTERNS = (
    re.compile(r"@\s*import\b", re.IGNORECASE),
    re.compile(r"url\s*\(\s*['\"]?\s*(?:https?:|//|javascript:|data:text/html)", re.IGNORECASE),
    re.compile(r"expression\s*\(", re.IGNORECASE),
    re.compile(r"(?:behavior|-moz-binding)\s*:", re.IGNORECASE),
)


@dataclass(frozen=True)
class ThemePackageValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_theme_package_contract(
    member_paths: Iterable[str],
    *,
    declared_capabilities: Iterable[str] = (),
) -> ThemePackageValidationResult:
    errors: list[str] = []
    for raw_path in member_paths:
        text = str(raw_path or "").replace("\\", "/").strip()
        if not text:
            errors.append("Theme package contains an empty member path.")
            continue
        path = PurePosixPath(text)
        first_part = path.parts[0] if path.parts else ""
        if "\x00" in text or path.is_absolute() or ".." in path.parts or ":" in first_part:
            errors.append(f"Theme package member escapes package root: {text}")
            continue
        suffix = path.suffix.lower()
        if suffix in PROHIBITED_THEME_PACKAGE_SUFFIXES:
            errors.append(f"Executable theme package content is prohibited: {text}")

    known = known_theme_capabilities()
    unknown = sorted({str(value) for value in declared_capabilities} - known)
    if unknown:
        errors.append(f"Unknown theme capabilities are not permitted: {', '.join(unknown)}")

    return ThemePackageValidationResult(valid=not errors, errors=tuple(errors))


def validate_svg_visual_content(svg_text: str) -> ThemePackageValidationResult:
    text = str(svg_text or "")
    errors = [
        "SVG contains executable or externally referenced content."
        for pattern in _SVG_UNSAFE_PATTERNS
        if pattern.search(text)
    ]
    return ThemePackageValidationResult(valid=not errors, errors=tuple(dict.fromkeys(errors)))


def validate_scoped_css_visual_content(css_text: str) -> ThemePackageValidationResult:
    text = str(css_text or "")
    errors = [
        "CSS contains executable or externally referenced content."
        for pattern in _CSS_UNSAFE_PATTERNS
        if pattern.search(text)
    ]
    return ThemePackageValidationResult(valid=not errors, errors=tuple(dict.fromkeys(errors)))
