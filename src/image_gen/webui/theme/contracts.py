from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from image_gen.webui.theme.tokens import (
    legacy_palette_to_semantic_tokens,
    normalize_semantic_palette,
)

THEME_CONTRACT_SCHEMA_VERSION = 1
THEME_PACKAGE_SCHEMA_VERSION = 1

DEFAULT_LEGACY_THEME_PALETTE: dict[str, Any] = {
    "accent": {"name": "Sky Blue", "color": "#179ee7"},
    "surface": {"name": "Charcoal", "color": "#111d29"},
    "typography": {
        "font_family": "Inter",
        "primary_button_text": "#ffffff",
        "secondary_button_text": "#d5f1ff",
    },
    "semantic": {
        "surface_secondary": "#172431",
        "component_surface": "#111d29",
        "component_border": "#2b4358",
        "component_accent": "#179ee7",
        "text_primary": "#f4f9fd",
        "text_secondary": "#9db2c4",
    },
}


class ThemePackageClass(str, Enum):
    THEME = "theme"
    PAGE_SKIN = "page_skin"
    ASSET_PACK = "asset_pack"
    THEME_SUITE = "theme_suite"


class ThemeCapability(str, Enum):
    GLOBAL_TOKENS = "global_tokens"
    PAGE_SKIN = "page_skin"
    PAGE_BANNER = "page_banner"
    PAGE_BACKGROUND = "page_background"
    ICON_PACK = "icon_pack"
    TEXTURE_PACK = "texture_pack"
    COMPONENT_SURFACE_TOKENS = "component_surface_tokens"
    SCOPED_COMPONENT_CSS = "scoped_component_css"


class ThemeCompatibilityState(str, Enum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"
    INCOMPATIBLE_IMAGEGEN_VERSION = "incompatible_imagegen_version"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    MISSING_CAPABILITY = "missing_capability"
    MISSING_DEPENDENCY = "missing_dependency"
    CONFLICT = "conflict"
    INVALID_PACKAGE = "invalid_package"


class ThemeSourceKind(str, Enum):
    BUNDLED = "bundled"
    LOCAL = "local"
    USER = "user"
    REMOTE = "remote"
    LEGACY_PALETTE = "legacy_palette"


class ThemePreviewMode(str, Enum):
    STATIC = "static"
    LIVE = "live"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_legacy_theme_palette(value: Any) -> dict[str, Any]:
    """Preserve the historical palette while adding TM-02 semantic roles."""

    stored = _mapping(value)
    accent = {**DEFAULT_LEGACY_THEME_PALETTE["accent"], **_mapping(stored.get("accent"))}
    surface = {**DEFAULT_LEGACY_THEME_PALETTE["surface"], **_mapping(stored.get("surface"))}
    typography = {
        **DEFAULT_LEGACY_THEME_PALETTE["typography"],
        **_mapping(stored.get("typography")),
    }
    semantic = normalize_semantic_palette(
        stored.get("semantic"),
        accent=str(accent.get("color") or DEFAULT_LEGACY_THEME_PALETTE["accent"]["color"]),
        surface=str(surface.get("color") or DEFAULT_LEGACY_THEME_PALETTE["surface"]["color"]),
    )
    return {
        "accent": accent,
        "surface": surface,
        "typography": typography,
        "semantic": semantic,
    }


@dataclass(frozen=True)
class ThemeTokenSet:
    """Canonical TM-01 token representation for the legacy palette surface."""

    accent_name: str
    accent_color: str
    surface_name: str
    surface_color: str
    font_family: str
    primary_button_text: str
    secondary_button_text: str
    semantic_tokens: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = THEME_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_legacy_palette(cls, value: Any) -> "ThemeTokenSet":
        palette = normalize_legacy_theme_palette(value)
        return cls(
            accent_name=str(palette["accent"]["name"]),
            accent_color=str(palette["accent"]["color"]),
            surface_name=str(palette["surface"]["name"]),
            surface_color=str(palette["surface"]["color"]),
            font_family=str(palette["typography"]["font_family"]),
            primary_button_text=str(palette["typography"]["primary_button_text"]),
            secondary_button_text=str(palette["typography"]["secondary_button_text"]),
            semantic_tokens=legacy_palette_to_semantic_tokens(palette),
        )

    def to_legacy_palette(self) -> dict[str, Any]:
        return {
            "accent": {"name": self.accent_name, "color": self.accent_color},
            "surface": {"name": self.surface_name, "color": self.surface_color},
            "typography": {
                "font_family": self.font_family,
                "primary_button_text": self.primary_button_text,
                "secondary_button_text": self.secondary_button_text,
            },
            "semantic": normalize_legacy_theme_palette({
                "accent": {"name": self.accent_name, "color": self.accent_color},
                "surface": {"name": self.surface_name, "color": self.surface_color},
                "typography": {
                    "font_family": self.font_family,
                    "primary_button_text": self.primary_button_text,
                    "secondary_button_text": self.secondary_button_text,
                },
                "semantic": {
                    "surface_secondary": self.semantic_tokens.get("color.surface.secondary", ""),
                    "component_surface": self.semantic_tokens.get("color.component.surface", ""),
                    "component_border": self.semantic_tokens.get("color.component.border", ""),
                    "component_accent": self.semantic_tokens.get("color.component.accent", ""),
                    "text_primary": self.semantic_tokens.get("color.text.primary", ""),
                    "text_secondary": self.semantic_tokens.get("color.text.secondary", ""),
                },
            })["semantic"],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_tokens": dict(self.semantic_tokens or {}),
            "accent": {"name": self.accent_name, "color": self.accent_color},
            "surface": {"name": self.surface_name, "color": self.surface_color},
            "typography": {
                "font_family": self.font_family,
                "primary_button_text": self.primary_button_text,
                "secondary_button_text": self.secondary_button_text,
            },
        }


@dataclass(frozen=True)
class ThemeSourceDescriptor:
    kind: ThemeSourceKind
    source_id: str
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "source_id": self.source_id, "reference": self.reference}


@dataclass(frozen=True)
class ThemeAssetDescriptor:
    asset_id: str
    relative_path: str
    slot: str
    media_type: str = ""
    supported_pages: tuple[str, ...] = ()
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "slot": self.slot,
            "media_type": self.media_type,
            "supported_pages": list(self.supported_pages),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ThemeDescriptor:
    theme_id: str
    display_name: str
    token_set: ThemeTokenSet
    capabilities: tuple[ThemeCapability, ...] = (ThemeCapability.GLOBAL_TOKENS,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "display_name": self.display_name,
            "token_set": self.token_set.to_dict(),
            "capabilities": [item.value for item in self.capabilities],
        }


@dataclass(frozen=True)
class PageSkinDescriptor:
    skin_id: str
    display_name: str
    supported_pages: tuple[str, ...]
    assets: tuple[ThemeAssetDescriptor, ...] = ()
    capabilities: tuple[ThemeCapability, ...] = (ThemeCapability.PAGE_SKIN,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skin_id": self.skin_id,
            "display_name": self.display_name,
            "supported_pages": list(self.supported_pages),
            "assets": [asset.to_dict() for asset in self.assets],
            "capabilities": [item.value for item in self.capabilities],
        }


@dataclass(frozen=True)
class ThemeSuiteDescriptor:
    suite_id: str
    display_name: str
    package_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "display_name": self.display_name,
            "package_ids": list(self.package_ids),
        }


@dataclass(frozen=True)
class ThemePackageDescriptor:
    package_id: str
    version: str
    package_class: ThemePackageClass
    display_name: str
    schema_version: int = THEME_PACKAGE_SCHEMA_VERSION
    min_imagegen_contract: str = ""
    max_imagegen_contract: str = ""
    supported_pages: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    source: ThemeSourceDescriptor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "version": self.version,
            "package_class": self.package_class.value,
            "display_name": self.display_name,
            "min_imagegen_contract": self.min_imagegen_contract,
            "max_imagegen_contract": self.max_imagegen_contract,
            "supported_pages": list(self.supported_pages),
            "required_capabilities": list(self.required_capabilities),
            "optional_dependencies": list(self.optional_dependencies),
            "conflicts": list(self.conflicts),
            "source": self.source.to_dict() if self.source else None,
        }


@dataclass(frozen=True)
class ThemeInstallRecord:
    package_id: str
    version: str
    install_root: str
    source: ThemeSourceDescriptor
    enabled: bool = False
    verified_sha256: str = ""
    installed_at: str = ""
    verification_state: str = "unverified"
    previous_version: str = ""
    local_modification_state: str = "clean"

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "version": self.version,
            "install_root": self.install_root,
            "source": self.source.to_dict(),
            "enabled": self.enabled,
            "verified_sha256": self.verified_sha256,
            "installed_at": self.installed_at,
            "verification_state": self.verification_state,
            "previous_version": self.previous_version,
            "local_modification_state": self.local_modification_state,
        }


@dataclass(frozen=True)
class ThemeActivationState:
    global_theme_package_id: str = ""
    theme_suite_package_id: str = ""
    page_skin_package_ids: Mapping[str, str] = field(default_factory=dict)
    workspace_appearance_preset_ids: Mapping[str, str] = field(default_factory=dict)
    user_overrides_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_theme_package_id": self.global_theme_package_id,
            "theme_suite_package_id": self.theme_suite_package_id,
            "page_skin_package_ids": dict(self.page_skin_package_ids),
            "workspace_appearance_preset_ids": dict(self.workspace_appearance_preset_ids),
            "user_overrides_enabled": self.user_overrides_enabled,
        }


@dataclass(frozen=True)
class ThemePreviewSession:
    session_id: str
    package_id: str
    mode: ThemePreviewMode
    preview_root: str
    installed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "package_id": self.package_id,
            "mode": self.mode.value,
            "preview_root": self.preview_root,
            "installed": self.installed,
        }


@dataclass(frozen=True)
class ThemeCompatibilityResult:
    state: ThemeCompatibilityState
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.state in {
            ThemeCompatibilityState.COMPATIBLE,
            ThemeCompatibilityState.COMPATIBLE_WITH_WARNINGS,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "usable": self.usable,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def known_theme_capabilities() -> frozenset[str]:
    return frozenset(item.value for item in ThemeCapability)


def evaluate_theme_compatibility(
    package: ThemePackageDescriptor,
    *,
    supported_schema_versions: Iterable[int] = (THEME_PACKAGE_SCHEMA_VERSION,),
    available_capabilities: Iterable[str] | None = None,
    installed_package_ids: Iterable[str] = (),
) -> ThemeCompatibilityResult:
    supported_schemas = {int(value) for value in supported_schema_versions}
    if package.schema_version not in supported_schemas:
        return ThemeCompatibilityResult(
            ThemeCompatibilityState.INCOMPATIBLE_SCHEMA,
            reasons=(f"Unsupported theme package schema version: {package.schema_version}",),
        )

    known = known_theme_capabilities()
    declared = {str(value) for value in package.required_capabilities}
    unknown = sorted(declared - known)
    if unknown:
        return ThemeCompatibilityResult(
            ThemeCompatibilityState.MISSING_CAPABILITY,
            reasons=(f"Unknown or unsupported theme capabilities: {', '.join(unknown)}",),
        )

    if available_capabilities is not None:
        available = {str(value) for value in available_capabilities}
        missing = sorted(declared - available)
        if missing:
            return ThemeCompatibilityResult(
                ThemeCompatibilityState.MISSING_CAPABILITY,
                reasons=(f"Required theme capabilities are unavailable: {', '.join(missing)}",),
            )

    installed = {str(value) for value in installed_package_ids}
    missing_dependencies = sorted(set(package.optional_dependencies) - installed)
    conflicts = sorted(set(package.conflicts) & installed)
    if conflicts:
        return ThemeCompatibilityResult(
            ThemeCompatibilityState.CONFLICT,
            reasons=(f"Conflicting theme packages are installed: {', '.join(conflicts)}",),
        )
    if missing_dependencies:
        return ThemeCompatibilityResult(
            ThemeCompatibilityState.COMPATIBLE_WITH_WARNINGS,
            warnings=(f"Optional theme dependencies are not installed: {', '.join(missing_dependencies)}",),
        )

    return ThemeCompatibilityResult(ThemeCompatibilityState.COMPATIBLE)
