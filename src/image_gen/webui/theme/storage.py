from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

THEME_STORAGE_SETTING_KEYS = (
    "theme_library_root",
    "theme_user_root",
    "theme_cache_root",
    "theme_preview_root",
)

_THEME_STORAGE_ENV_KEYS = {
    "theme_library_root": "IMAGE_GEN_THEME_LIBRARY_ROOT",
    "theme_user_root": "IMAGE_GEN_THEME_USER_ROOT",
    "theme_cache_root": "IMAGE_GEN_THEME_CACHE_ROOT",
    "theme_preview_root": "IMAGE_GEN_THEME_PREVIEW_ROOT",
}

_THEME_STORAGE_DEFAULT_DIRS = {
    "theme_library_root": "library",
    "theme_user_root": "user",
    "theme_cache_root": "cache",
    "theme_preview_root": "preview",
}


class ThemeStorageConfigurationError(ValueError):
    """Raised when a configured theme storage root violates the TM-01 boundary."""


def normalize_theme_storage_settings(value: Any) -> dict[str, str]:
    stored = value if isinstance(value, Mapping) else {}
    return {key: str(stored.get(key) or "").strip() for key in THEME_STORAGE_SETTING_KEYS}


def _resolve_configured_path(value: str, *, project_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ThemeStorageRoots:
    theme_library_root: Path
    theme_user_root: Path
    theme_cache_root: Path
    theme_preview_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        project_root: str | Path,
        settings: Mapping[str, Any] | None = None,
        environment: Mapping[str, str] | None = None,
        user_home: str | Path | None = None,
    ) -> "ThemeStorageRoots":
        source_root = Path(project_root).expanduser().resolve()
        normalized = normalize_theme_storage_settings(settings)
        env = environment if environment is not None else os.environ
        home = Path(user_home).expanduser().resolve() if user_home is not None else Path.home().resolve()
        auto_base = (home / ".image_gen" / "themes").resolve()

        resolved: dict[str, Path] = {}
        for key in THEME_STORAGE_SETTING_KEYS:
            configured = normalized.get(key, "")
            env_value = str(env.get(_THEME_STORAGE_ENV_KEYS[key]) or "").strip()
            if configured:
                candidate = _resolve_configured_path(configured, project_root=source_root)
            elif env_value:
                candidate = _resolve_configured_path(env_value, project_root=source_root)
            else:
                candidate = (auto_base / _THEME_STORAGE_DEFAULT_DIRS[key]).resolve()

            if _is_within(candidate, source_root):
                raise ThemeStorageConfigurationError(
                    f"{key} must resolve outside the IMAGE_GEN source tree: {candidate}"
                )
            resolved[key] = candidate

        return cls(**resolved)

    def to_dict(self) -> dict[str, str]:
        return {
            "theme_library_root": str(self.theme_library_root),
            "theme_user_root": str(self.theme_user_root),
            "theme_cache_root": str(self.theme_cache_root),
            "theme_preview_root": str(self.theme_preview_root),
        }
