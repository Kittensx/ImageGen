from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from image_gen.webui.theme.contracts import (
    THEME_PACKAGE_SCHEMA_VERSION,
    ThemeCapability,
    ThemePackageClass,
    ThemeSourceDescriptor,
    ThemeSourceKind,
    normalize_legacy_theme_palette,
)
from image_gen.webui.theme.security import (
    validate_scoped_css_visual_content,
    validate_svg_visual_content,
    validate_theme_package_contract,
)
from image_gen.webui.theme.storage import ThemeStorageRoots
from image_gen.webui.theme.tokens import (
    SEMANTIC_THEME_TOKEN_DEFAULTS,
    legacy_palette_to_semantic_tokens,
    normalize_semantic_tokens,
    semantic_tokens_to_legacy_palette,
    validate_semantic_theme_contrast,
)

THEME_PACKAGE_SCHEMA = "imagegen.theme-package"
THEME_LIBRARY_SCHEMA_VERSION = 1
LEGACY_PALETTE_PACKAGE_ID = "imagegen.local.legacy-palette"
MAX_THEME_PACKAGE_FILES = 2048
MAX_THEME_PACKAGE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_THEME_PACKAGE_MEMBER_BYTES = 64 * 1024 * 1024
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")


class ThemeLibraryError(ValueError):
    def __init__(self, message: str, *, code: str = "theme_library_error") -> None:
        super().__init__(message)
        self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


@dataclass(frozen=True)
class ParsedThemePackage:
    manifest: dict[str, Any]
    package_id: str
    version: str
    package_class: ThemePackageClass
    display_name: str
    capabilities: tuple[str, ...]
    tokens: dict[str, str]
    assets: tuple[dict[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(text)
    first_part = path.parts[0] if path.parts else ""
    if (
        not text
        or "\x00" in text
        or path.is_absolute()
        or ".." in path.parts
        or ":" in first_part
    ):
        raise ThemeLibraryError(f"Unsafe theme package path: {value}", code="theme_package_path_invalid")
    return str(path)


def _manifest_value(manifest: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in manifest:
            return manifest[name]
    return default


def _validate_manifest(manifest: Any) -> tuple[str, str, ThemePackageClass, str, tuple[str, ...]]:
    if not isinstance(manifest, Mapping):
        raise ThemeLibraryError("Theme package manifest.json must contain a JSON object.", code="theme_manifest_invalid")
    schema = str(manifest.get("schema") or "").strip()
    if schema != THEME_PACKAGE_SCHEMA:
        raise ThemeLibraryError(
            f"Unsupported theme package schema '{schema or '<missing>'}'. Expected {THEME_PACKAGE_SCHEMA}.",
            code="theme_manifest_schema_invalid",
        )
    try:
        schema_version = int(_manifest_value(manifest, "schemaVersion", "schema_version", default=0) or 0)
    except (TypeError, ValueError) as exc:
        raise ThemeLibraryError("Theme package schemaVersion must be an integer.", code="theme_manifest_invalid") from exc
    if schema_version != THEME_PACKAGE_SCHEMA_VERSION:
        raise ThemeLibraryError(
            f"Unsupported theme package schema version: {schema_version}", code="theme_manifest_schema_unsupported"
        )
    package_id = str(_manifest_value(manifest, "packageId", "package_id", default="") or "").strip().lower()
    if not _PACKAGE_ID.fullmatch(package_id):
        raise ThemeLibraryError("Theme packageId is missing or invalid.", code="theme_manifest_invalid")
    version = str(manifest.get("version") or "").strip()
    if not version or len(version) > 64 or any(char in version for char in "/\\"):
        raise ThemeLibraryError("Theme package version is missing or invalid.", code="theme_manifest_invalid")
    display_name = str(manifest.get("name") or "").strip()
    if not display_name:
        raise ThemeLibraryError("Theme package name is required.", code="theme_manifest_invalid")
    try:
        package_class = ThemePackageClass(str(manifest.get("type") or ""))
    except ValueError as exc:
        raise ThemeLibraryError("Theme package type is missing or unsupported.", code="theme_manifest_invalid") from exc
    publisher = manifest.get("publisher") or manifest.get("author")
    if not publisher:
        raise ThemeLibraryError("Theme package publisher/author metadata is required.", code="theme_manifest_invalid")
    if not str(manifest.get("license") or "").strip():
        raise ThemeLibraryError("Theme package license metadata is required.", code="theme_manifest_invalid")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ThemeLibraryError("Theme package compatibility must be a JSON object.", code="theme_manifest_invalid")
    capabilities_raw = manifest.get("capabilities")
    if not isinstance(capabilities_raw, list):
        raise ThemeLibraryError("Theme package capabilities must be a JSON array.", code="theme_manifest_invalid")
    if not isinstance(manifest.get("contents"), (Mapping, list)):
        raise ThemeLibraryError("Theme package contents metadata is required.", code="theme_manifest_invalid")
    capabilities = tuple(str(value).strip() for value in capabilities_raw if str(value).strip())
    return package_id, version, package_class, display_name, capabilities


def _read_token_files(root: Path) -> dict[str, str]:
    tokens = dict(SEMANTIC_THEME_TOKEN_DEFAULTS)
    token_root = root / "tokens"
    if not token_root.is_dir():
        return tokens
    for path in sorted(token_root.glob("*.json")):
        payload = _read_json(path, None)
        if not isinstance(payload, Mapping):
            raise ThemeLibraryError(f"Theme token file is not a JSON object: {path.name}", code="theme_tokens_invalid")
        tokens = normalize_semantic_tokens(payload, base=tokens)
    return tokens


def _index_assets(root: Path, package_id: str, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    declared: dict[str, Mapping[str, Any]] = {}
    contents = manifest.get("contents")
    if isinstance(contents, Mapping):
        assets = contents.get("assets")
        if isinstance(assets, list):
            for item in assets:
                if not isinstance(item, Mapping):
                    continue
                path = _safe_relative_path(str(item.get("path") or ""))
                declared[path] = item

    excluded_prefixes = ("tokens/", "styles/")
    output: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json" or relative.startswith(excluded_prefixes):
            continue
        metadata = declared.get(relative, {})
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        output.append(
            {
                "assetId": str(metadata.get("assetId") or metadata.get("asset_id") or f"{package_id}:{relative}"),
                "packageId": package_id,
                "type": str(metadata.get("type") or path.parent.name or "asset"),
                "path": relative,
                "variant": str(metadata.get("variant") or ""),
                "pageCompatibility": list(metadata.get("pages") or metadata.get("pageCompatibility") or []),
                "width": int(metadata.get("width") or 0),
                "height": int(metadata.get("height") or 0),
                "mimeType": media_type,
                "lightDarkIntent": str(metadata.get("intent") or metadata.get("lightDarkIntent") or ""),
                "previewPath": str(metadata.get("preview") or metadata.get("previewPath") or ""),
            }
        )
    return tuple(output)


def parse_theme_package_directory(root: Path) -> ParsedThemePackage:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ThemeLibraryError("Theme package is missing manifest.json.", code="theme_manifest_missing")
    manifest = _read_json(manifest_path, None)
    package_id, version, package_class, display_name, capabilities = _validate_manifest(manifest)
    members = [item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()]
    contract = validate_theme_package_contract(members, declared_capabilities=capabilities)
    if not contract.valid:
        raise ThemeLibraryError(" ".join(contract.errors), code="theme_package_security_rejected")
    has_css = any(PurePosixPath(member).suffix.lower() == ".css" for member in members)
    if has_css and ThemeCapability.SCOPED_COMPONENT_CSS.value not in capabilities:
        raise ThemeLibraryError(
            "Theme package CSS requires the scoped_component_css capability.",
            code="theme_package_capability_required",
        )
    for path in root.rglob("*.svg"):
        try:
            svg = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ThemeLibraryError(f"Unable to read SVG asset: {path.name}", code="theme_package_svg_invalid") from exc
        result = validate_svg_visual_content(svg)
        if not result.valid:
            raise ThemeLibraryError(f"Unsafe SVG asset '{path.name}': {' '.join(result.errors)}", code="theme_package_svg_invalid")
    for path in root.rglob("*.css"):
        try:
            css = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ThemeLibraryError(f"Unable to read CSS asset: {path.name}", code="theme_package_css_invalid") from exc
        result = validate_scoped_css_visual_content(css)
        if not result.valid:
            raise ThemeLibraryError(f"Unsafe CSS asset '{path.name}': {' '.join(result.errors)}", code="theme_package_css_invalid")
    tokens = _read_token_files(root)
    return ParsedThemePackage(
        manifest=dict(manifest),
        package_id=package_id,
        version=version,
        package_class=package_class,
        display_name=display_name,
        capabilities=capabilities,
        tokens=tokens,
        assets=_index_assets(root, package_id, manifest),
    )


class ThemePackageLibrary:
    def __init__(
        self,
        roots: ThemeStorageRoots,
        *,
        legacy_palette_provider: Callable[[], Mapping[str, Any]],
    ) -> None:
        self.roots = roots
        self._legacy_palette_provider = legacy_palette_provider
        self._lock = threading.RLock()
        for path in (
            roots.theme_library_root,
            roots.theme_user_root,
            roots.theme_cache_root,
            roots.theme_preview_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._index_path = roots.theme_library_root / "installed-library.json"
        self._activation_path = roots.theme_user_root / "activation.json"
        self._diagnostic_path = roots.theme_user_root / "diagnostics.json"
        self.recover_active_package()

    def _load_index(self) -> dict[str, Any]:
        payload = _read_json(self._index_path, {})
        if not isinstance(payload, Mapping):
            payload = {}
        packages = payload.get("packages") if isinstance(payload.get("packages"), Mapping) else {}
        return {
            "schemaVersion": THEME_LIBRARY_SCHEMA_VERSION,
            "packages": {str(key): dict(value) for key, value in packages.items() if isinstance(value, Mapping)},
        }

    def _save_index(self, index: Mapping[str, Any]) -> None:
        _atomic_write_json(self._index_path, dict(index))

    def _load_activation(self) -> dict[str, Any]:
        payload = _read_json(self._activation_path, {})
        return {
            "globalThemePackageId": str(payload.get("globalThemePackageId") or "") if isinstance(payload, Mapping) else "",
        }

    def _save_activation(self, payload: Mapping[str, Any]) -> None:
        _atomic_write_json(self._activation_path, {"schemaVersion": 1, **dict(payload)})

    def _record_diagnostic(self, message: str, *, package_id: str = "") -> None:
        current = _read_json(self._diagnostic_path, [])
        entries = list(current) if isinstance(current, list) else []
        entries.append({"timestamp": _utc_now(), "packageId": package_id, "message": message})
        _atomic_write_json(self._diagnostic_path, entries[-50:])

    def _legacy_record(self) -> dict[str, Any]:
        active = self._load_activation()["globalThemePackageId"]
        palette = normalize_legacy_theme_palette(self._legacy_palette_provider())
        return {
            "packageId": LEGACY_PALETTE_PACKAGE_ID,
            "installedVersion": "local",
            "name": "Custom / legacy palette",
            "type": ThemePackageClass.THEME.value,
            "installLocation": "",
            "source": {"kind": ThemeSourceKind.LEGACY_PALETTE.value, "source_id": LEGACY_PALETTE_PACKAGE_ID, "reference": "application.theme_palette"},
            "installedAt": "",
            "enabledState": not bool(active),
            "verificationState": "synthetic",
            "previousVersion": "",
            "localModificationState": "user_owned",
            "assets": [],
            "manifest": {"synthetic": True},
            "semanticTokens": legacy_palette_to_semantic_tokens(palette),
        }

    def list_packages(self) -> list[dict[str, Any]]:
        with self._lock:
            index = self._load_index()
            output = [self._legacy_record()]
            for package_id in sorted(index["packages"], key=str.casefold):
                output.append(dict(index["packages"][package_id]))
            return output

    def library_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "contractVersion": THEME_LIBRARY_SCHEMA_VERSION,
                "storage": self.roots.to_dict(),
                "activation": self._load_activation(),
                "packages": self.list_packages(),
                "effectivePalette": self.resolve_effective_palette(),
                "diagnostics": list(_read_json(self._diagnostic_path, []) or [])[-10:],
            }

    def _validate_zip_limits(self, archive: zipfile.ZipFile) -> list[str]:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise ThemeLibraryError("Theme package archive is empty.", code="theme_package_empty")
        if len(infos) > MAX_THEME_PACKAGE_FILES:
            raise ThemeLibraryError("Theme package contains too many files.", code="theme_package_too_large")
        total = 0
        members: list[str] = []
        for info in infos:
            member = _safe_relative_path(info.filename)
            members.append(member)
            total += int(info.file_size or 0)
            if int(info.file_size or 0) > MAX_THEME_PACKAGE_MEMBER_BYTES:
                raise ThemeLibraryError(f"Theme package member is too large: {member}", code="theme_package_too_large")
            if total > MAX_THEME_PACKAGE_UNCOMPRESSED_BYTES:
                raise ThemeLibraryError("Theme package exceeds the uncompressed size limit.", code="theme_package_too_large")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ThemeLibraryError(f"Theme package symbolic links are prohibited: {member}", code="theme_package_security_rejected")
        return members

    def install_archive(self, archive_path: str | Path) -> dict[str, Any]:
        source_path = Path(archive_path).expanduser().resolve()
        if not source_path.is_file():
            raise ThemeLibraryError("Theme package file does not exist.", code="theme_package_missing")
        archive_hash = _sha256_file(source_path)
        with self._lock:
            try:
                archive = zipfile.ZipFile(source_path)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ThemeLibraryError("Theme package must be a valid ZIP-compatible archive.", code="theme_package_archive_invalid") from exc
            with archive:
                members = self._validate_zip_limits(archive)
                contract = validate_theme_package_contract(members)
                if not contract.valid:
                    raise ThemeLibraryError(" ".join(contract.errors), code="theme_package_security_rejected")
                staging_parent = self.roots.theme_cache_root / "imports"
                staging_parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="theme-import-", dir=staging_parent) as temp_dir:
                    staging = Path(temp_dir)
                    archive.extractall(staging)
                    parsed = parse_theme_package_directory(staging)
                    target = self.roots.theme_library_root / "packages" / parsed.package_id / parsed.version
                    index = self._load_index()
                    existing = index["packages"].get(parsed.package_id)
                    if existing and str(existing.get("installedVersion")) == parsed.version and target.exists():
                        raise ThemeLibraryError(
                            f"Theme package {parsed.package_id} {parsed.version} is already installed.",
                            code="theme_package_duplicate",
                        )
                    previous_version = str(existing.get("installedVersion") or "") if existing else ""
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(staging, target)
                    contrast = validate_semantic_theme_contrast(parsed.tokens)
                    readability_warnings = [
                        f"{check.foreground_token} vs {check.background_token}: {check.reason}"
                        for check in contrast.checks
                        if (not check.valid) and check.foreground_token.startswith("color.text.")
                    ]
                    record = {
                        "packageId": parsed.package_id,
                        "installedVersion": parsed.version,
                        "name": parsed.display_name,
                        "type": parsed.package_class.value,
                        "installLocation": str(target),
                        "source": ThemeSourceDescriptor(
                            kind=ThemeSourceKind.LOCAL,
                            source_id=parsed.package_id,
                            reference=source_path.name,
                        ).to_dict(),
                        "installedAt": _utc_now(),
                        "enabledState": False,
                        "verificationState": "local_validated",
                        "verifiedSha256": archive_hash,
                        "previousVersion": previous_version,
                        "localModificationState": "clean",
                        "assets": list(parsed.assets),
                        "manifest": parsed.manifest,
                        "semanticTokens": parsed.tokens,
                        "contrastWarnings": readability_warnings,
                    }
                    index["packages"][parsed.package_id] = record
                    self._save_index(index)
                    return dict(record)

    def enable(self, package_id: str) -> dict[str, Any]:
        package_id = str(package_id or "").strip().lower()
        if package_id == LEGACY_PALETTE_PACKAGE_ID:
            self.deactivate_global_theme()
            return self._legacy_record()
        with self._lock:
            index = self._load_index()
            record = index["packages"].get(package_id)
            if not isinstance(record, Mapping):
                raise ThemeLibraryError(f"Theme package is not installed: {package_id}", code="theme_package_not_found")
            self._validate_installed_record(record)
            for other_id, other in index["packages"].items():
                if isinstance(other, dict) and str(other.get("type")) == ThemePackageClass.THEME.value:
                    other["enabledState"] = other_id == package_id
            if str(record.get("type")) == ThemePackageClass.THEME.value:
                self._save_activation({"globalThemePackageId": package_id})
            else:
                record["enabledState"] = True
            self._save_index(index)
            return dict(index["packages"][package_id])

    def disable(self, package_id: str) -> dict[str, Any]:
        package_id = str(package_id or "").strip().lower()
        with self._lock:
            if package_id == LEGACY_PALETTE_PACKAGE_ID:
                return self._legacy_record()
            index = self._load_index()
            record = index["packages"].get(package_id)
            if not isinstance(record, dict):
                raise ThemeLibraryError(f"Theme package is not installed: {package_id}", code="theme_package_not_found")
            record["enabledState"] = False
            activation = self._load_activation()
            if activation["globalThemePackageId"] == package_id:
                self._save_activation({"globalThemePackageId": ""})
            self._save_index(index)
            return dict(record)

    def deactivate_global_theme(self) -> None:
        with self._lock:
            activation = self._load_activation()
            active = activation["globalThemePackageId"]
            if not active:
                return
            index = self._load_index()
            record = index["packages"].get(active)
            if isinstance(record, dict):
                record["enabledState"] = False
                self._save_index(index)
            self._save_activation({"globalThemePackageId": ""})

    def remove(self, package_id: str) -> bool:
        package_id = str(package_id or "").strip().lower()
        if package_id == LEGACY_PALETTE_PACKAGE_ID:
            raise ThemeLibraryError("The synthetic legacy palette cannot be removed.", code="theme_package_protected")
        with self._lock:
            index = self._load_index()
            record = index["packages"].pop(package_id, None)
            if not isinstance(record, Mapping):
                return False
            activation = self._load_activation()
            if activation["globalThemePackageId"] == package_id:
                self._save_activation({"globalThemePackageId": ""})
            install_root = Path(str(record.get("installLocation") or ""))
            package_root = self.roots.theme_library_root / "packages" / package_id
            if package_root.exists():
                shutil.rmtree(package_root)
            elif install_root.exists() and self.roots.theme_library_root in install_root.parents:
                shutil.rmtree(install_root)
            self._save_index(index)
            return True

    def _validate_installed_record(self, record: Mapping[str, Any]) -> ParsedThemePackage:
        root = Path(str(record.get("installLocation") or ""))
        if not root.is_dir():
            raise ThemeLibraryError("Installed theme package files are missing.", code="theme_package_corrupt")
        parsed = parse_theme_package_directory(root)
        if parsed.package_id != str(record.get("packageId") or ""):
            raise ThemeLibraryError("Installed theme package identity does not match its library record.", code="theme_package_corrupt")
        return parsed

    def recover_active_package(self) -> bool:
        with self._lock:
            activation = self._load_activation()
            package_id = activation["globalThemePackageId"]
            if not package_id:
                return True
            index = self._load_index()
            record = index["packages"].get(package_id)
            try:
                if not isinstance(record, Mapping):
                    raise ThemeLibraryError("Active theme package is missing from installed-library metadata.")
                self._validate_installed_record(record)
                return True
            except ThemeLibraryError as exc:
                if isinstance(record, dict):
                    record["enabledState"] = False
                    self._save_index(index)
                self._save_activation({"globalThemePackageId": ""})
                self._record_diagnostic(f"Disabled corrupt/missing active package: {exc}", package_id=package_id)
                return False

    def resolve_effective_palette(self) -> dict[str, Any]:
        with self._lock:
            fallback = normalize_legacy_theme_palette(self._legacy_palette_provider())
            activation = self._load_activation()
            package_id = activation["globalThemePackageId"]
            if not package_id:
                return fallback
            index = self._load_index()
            record = index["packages"].get(package_id)
            try:
                if not isinstance(record, Mapping):
                    raise ThemeLibraryError("Active theme package metadata is missing.")
                parsed = self._validate_installed_record(record)
                return semantic_tokens_to_legacy_palette(
                    parsed.tokens,
                    accent_name=parsed.display_name,
                    surface_name=parsed.display_name,
                )
            except ThemeLibraryError as exc:
                if isinstance(record, dict):
                    record["enabledState"] = False
                    self._save_index(index)
                self._save_activation({"globalThemePackageId": ""})
                self._record_diagnostic(f"Fell back from corrupt/missing active package: {exc}", package_id=package_id)
                return fallback
