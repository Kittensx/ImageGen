from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from modules.project_context import ProjectContext
from modules.registry.asset_registry import AssetRegistry
from modules.registry.component_snapshot import SafetensorsComponentSnapshotter


ROLE_ALIASES: Mapping[str, str] = {
    "clip-l": "clip_l",
    "clip-g": "clip_g",
    "t5": "t5xxl",
    "t5_xxl": "t5xxl",
    "t5-xxl": "t5xxl",
}

CANONICAL_TEXT_ENCODER_SUBDIRS: Mapping[str, Path] = {
    "clip_l": Path("clip"),
    "clip_g": Path("clip"),
    "t5xxl": Path("t5"),
}

CANONICAL_TEXT_ENCODER_FILENAMES: Mapping[str, str] = {
    "clip_l": "clip_l.safetensors",
    "clip_g": "clip_g.safetensors",
    "t5xxl": "t5xxl_fp8_e4m3fn.safetensors",
}

COMPONENT_ROLE_BY_TEXT_ENCODER_ROLE: Mapping[str, str] = {
    "clip_l": "text_encoder",
    "clip_g": "text_encoder_2",
    "t5xxl": "text_encoder_3",
}

TEXT_ENCODER_ROLE_BY_FILENAME: Mapping[str, str] = {
    filename: role for role, filename in CANONICAL_TEXT_ENCODER_FILENAMES.items()
}


@dataclass(frozen=True)
class SharedTextEncoderResolution:
    role: str
    selected: Path | None
    source_layout: str
    checked: tuple[Path, ...]
    registered_sha256: str | None = None
    matched_component_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "selected": str(self.selected) if self.selected else None,
            "source_layout": self.source_layout,
            "checked": [str(path) for path in self.checked],
            "registered_sha256": self.registered_sha256,
            "matched_component_sha256": self.matched_component_sha256,
        }


def normalize_text_encoder_role(role: str) -> str:
    key = str(role or "").strip().lower()
    return ROLE_ALIASES.get(key, key)


def text_encoder_component_role(role: str) -> str:
    key = normalize_text_encoder_role(role)
    try:
        return COMPONENT_ROLE_BY_TEXT_ENCODER_ROLE[key]
    except KeyError as exc:
        raise ValueError(f"Unknown shared text-encoder role: {role!r}") from exc


def infer_text_encoder_role_from_path(path: str | Path) -> str:
    filename = Path(path).name
    try:
        return TEXT_ENCODER_ROLE_BY_FILENAME[filename]
    except KeyError as exc:
        raise ValueError(f"Unknown shared text-encoder filename: {filename!r}") from exc


def text_encoder_root(context: ProjectContext) -> Path:
    return (Path(context.models_root) / "TextEncoders").resolve()


def canonical_text_encoder_path(context: ProjectContext, role: str) -> Path:
    key = normalize_text_encoder_role(role)
    try:
        subdir = CANONICAL_TEXT_ENCODER_SUBDIRS[key]
        filename = CANONICAL_TEXT_ENCODER_FILENAMES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown shared text-encoder role: {role!r}") from exc
    return (text_encoder_root(context) / subdir / filename).resolve()


def shared_text_encoder_candidates(context: ProjectContext, role: str) -> tuple[tuple[str, Path], ...]:
    """Return filesystem bootstrap candidates only.

    Bootstrap discovery intentionally supports only the canonical family folder
    and a flat TextEncoders filename. Architecture-specific subfolders are not
    part of the lookup contract. Registered SHA-256 recovery is handled
    separately and may locate a moved file anywhere inside TextEncoders.
    """

    key = normalize_text_encoder_role(role)
    filename = CANONICAL_TEXT_ENCODER_FILENAMES.get(key)
    if filename is None:
        raise ValueError(f"Unknown shared text-encoder role: {role!r}")
    root = text_encoder_root(context)
    canonical = canonical_text_encoder_path(context, key)
    flat = (root / filename).resolve()
    return (
        ("canonical_family", canonical),
        ("flat_filename", flat),
    )


def _file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _registry_for_context(context: ProjectContext) -> AssetRegistry | None:
    raw = getattr(context, "registry_db_path", None)
    if raw is None:
        return None
    db_path = Path(raw).resolve()
    # Resolver reads the established registry. It must not create a new database
    # merely because an encoder lookup occurred.
    if not db_path.is_file():
        return None
    return AssetRegistry(str(db_path))


def _registered_records(registry: AssetRegistry, filename: str):
    return registry.find_assets_by_filename(filename, limit=256)


def _resolve_registered_component_match(
    *,
    registry: AssetRegistry,
    component_role: str,
    component_sha256: str,
    root: Path,
    canonical: Path,
) -> tuple[Path | None, str | None, str | None]:
    snapshots = [
        item
        for item in registry.find_components_by_sha256(component_sha256, limit=256)
        if item.component_role == component_role
    ]
    if not snapshots:
        return None, None, None

    ready: list[tuple[Path, str | None]] = []
    for snapshot in snapshots:
        asset = registry.get_asset_by_id(snapshot.asset_id)
        if asset is None:
            continue
        candidate = Path(asset.path).resolve()
        if _file_ready(candidate):
            ready.append((candidate, asset.sha256))

    if not ready:
        return None, None, None

    for candidate, sha256 in ready:
        if candidate == canonical:
            return candidate, sha256, component_sha256

    inside_root = []
    for candidate, sha256 in ready:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        inside_root.append((candidate, sha256))

    unique_inside_root = {
        (str(path).casefold(), str(sha256 or "").lower()): (path, sha256)
        for path, sha256 in inside_root
    }
    if len(unique_inside_root) == 1:
        path, sha256 = next(iter(unique_inside_root.values()))
        return path, sha256, component_sha256

    unique_ready = {
        (str(path).casefold(), str(sha256 or "").lower()): (path, sha256)
        for path, sha256 in ready
    }
    if len(unique_ready) == 1:
        path, sha256 = next(iter(unique_ready.values()))
        return path, sha256, component_sha256
    return None, None, None


def _resolve_registered_path(
    *,
    records,
    root: Path,
    canonical: Path,
) -> tuple[Path | None, str | None]:
    ready = [Path(record.path).resolve() for record in records if _file_ready(Path(record.path).resolve())]
    if not ready:
        return None, None

    # A currently registered canonical location wins. Otherwise prefer a
    # registered path inside the managed TextEncoders tree. If several distinct
    # registered files remain, do not guess between them.
    for path in ready:
        if path == canonical:
            record = next(record for record in records if Path(record.path).resolve() == path)
            return path, record.sha256

    inside_root: list[tuple[Path, str | None]] = []
    for record in records:
        path = Path(record.path).resolve()
        if not _file_ready(path):
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        inside_root.append((path, record.sha256))

    unique = {(str(path).casefold(), sha256): (path, sha256) for path, sha256 in inside_root}
    if len(unique) == 1:
        return next(iter(unique.values()))

    # A uniquely registered external path is also a valid registry-owned asset.
    # Multiple same-name registered files remain ambiguous without a stronger
    # role binding, so never choose arbitrarily.
    unique_ready = {}
    for record in records:
        path = Path(record.path).resolve()
        if _file_ready(path):
            unique_ready[(str(path).casefold(), record.sha256)] = (path, record.sha256)
    if len(unique_ready) == 1:
        return next(iter(unique_ready.values()))
    return None, None


def _resolve_registered_hash_relocation(
    *,
    registry: AssetRegistry,
    records,
    root: Path,
    filename: str,
) -> tuple[Path | None, str | None, tuple[Path, ...]]:
    expected_hashes = {str(record.sha256).lower() for record in records if record.sha256}
    if not expected_hashes or not root.is_dir():
        return None, None, ()

    checked: list[Path] = []
    matches: list[tuple[Path, str]] = []
    for candidate in sorted(root.rglob(filename), key=lambda path: str(path).casefold()):
        resolved = candidate.resolve()
        if not _file_ready(resolved):
            continue
        checked.append(resolved)
        digest = registry.fingerprinter.compute_sha256(resolved).lower()
        if digest in expected_hashes:
            matches.append((resolved, digest))

    # SHA-256 is the strong identity, but multiple physical copies with the same
    # hash are still ambiguous as a path choice. Prefer exactly one match.
    unique_paths = {(str(path).casefold(), digest): (path, digest) for path, digest in matches}
    if len(unique_paths) == 1:
        path, digest = next(iter(unique_paths.values()))
        return path, digest, tuple(checked)
    return None, None, tuple(checked)


def register_shared_text_encoder_asset(
    context: ProjectContext,
    path: str | Path,
    *,
    role: str | None = None,
    registry: AssetRegistry | None = None,
):
    """Register a standalone shared text encoder and its content-based snapshot.

    The asset row tracks the physical file. The component snapshot establishes
    identity from normalized tensor structure plus payload bytes so the same
    encoder can later match an embedded checkpoint component regardless of file
    name, packaging, or folder placement.
    """
    encoder_path = Path(path).expanduser().resolve()
    resolved_role = normalize_text_encoder_role(role or infer_text_encoder_role_from_path(encoder_path))
    component_role = text_encoder_component_role(resolved_role)
    active_registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
    root = Path(context.models_root).resolve()
    asset = active_registry.register_file(
        str(encoder_path),
        compute_sha256=True,
        library_root=str(root),
        managed_category="TextEncoders",
        path_kind="managed",
    )
    snapshotter = SafetensorsComponentSnapshotter()
    snapshot = snapshotter.snapshot_standalone_component(
        encoder_path,
        component_role=component_role,
    )
    records = active_registry.store_component_snapshots(
        asset.id,
        {component_role: snapshot},
        source_file_sha256=asset.sha256,
        source_quick_fingerprint=asset.quick_fingerprint,
    )
    return asset, tuple(records)


def resolve_shared_text_encoder(
    context: ProjectContext,
    role: str,
    *,
    expected_component_sha256: str | None = None,
) -> SharedTextEncoderResolution:
    key = normalize_text_encoder_role(role)
    filename = CANONICAL_TEXT_ENCODER_FILENAMES.get(key)
    if filename is None:
        raise ValueError(f"Unknown shared text-encoder role: {role!r}")

    root = text_encoder_root(context)
    canonical = canonical_text_encoder_path(context, key)
    component_role = text_encoder_component_role(key)
    checked: list[Path] = []

    registry = _registry_for_context(context)
    if registry is not None and expected_component_sha256:
        selected, sha256, matched = _resolve_registered_component_match(
            registry=registry,
            component_role=component_role,
            component_sha256=str(expected_component_sha256).strip().lower(),
            root=root,
            canonical=canonical,
        )
        if selected is not None:
            checked.append(selected)
            return SharedTextEncoderResolution(
                role=key,
                selected=selected,
                source_layout="component_sha256_match",
                checked=tuple(checked),
                registered_sha256=sha256,
                matched_component_sha256=matched,
            )

    records = _registered_records(registry, filename) if registry is not None else []
    if registry is not None and records:
        selected, sha256 = _resolve_registered_path(records=records, root=root, canonical=canonical)
        if selected is not None:
            checked.append(selected)
            return SharedTextEncoderResolution(
                role=key,
                selected=selected,
                source_layout="asset_registry",
                checked=tuple(checked),
                registered_sha256=sha256,
            )

        selected, sha256, hashed = _resolve_registered_hash_relocation(
            registry=registry,
            records=records,
            root=root,
            filename=filename,
        )
        checked.extend(hashed)
        if selected is not None:
            return SharedTextEncoderResolution(
                role=key,
                selected=selected,
                source_layout="asset_registry_sha256",
                checked=tuple(checked),
                registered_sha256=sha256,
            )

    ready_candidates: list[tuple[str, Path]] = []
    for layout, candidate in shared_text_encoder_candidates(context, key):
        if candidate not in checked:
            checked.append(candidate)
        if _file_ready(candidate):
            ready_candidates.append((layout, candidate))

    if ready_candidates:
        canonical_matches = [item for item in ready_candidates if item[0] == "canonical_family"]
        if canonical_matches:
            return SharedTextEncoderResolution(
                role=key,
                selected=canonical_matches[0][1],
                source_layout="canonical_family",
                checked=tuple(checked),
            )
        if len(ready_candidates) == 1:
            return SharedTextEncoderResolution(
                role=key,
                selected=ready_candidates[0][1],
                source_layout=ready_candidates[0][0],
                checked=tuple(checked),
            )
        return SharedTextEncoderResolution(
            role=key,
            selected=None,
            source_layout="ambiguous",
            checked=tuple(checked),
        )

    return SharedTextEncoderResolution(
        role=key,
        selected=None,
        source_layout="missing",
        checked=tuple(checked),
    )
