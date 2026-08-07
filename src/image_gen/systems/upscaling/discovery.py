from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from image_gen.systems.upscaling.cache import (
    UpscalerCacheRecord,
    UpscalerScanCache,
    canonical_path_key,
)
from image_gen.systems.upscaling.classifier import (
    CANONICAL_LOADER_BACKEND,
    UpscalerClassification,
    inspect_upscaler_file,
    loader_backend_version,
)
from image_gen.systems.upscaling.contracts import (
    BUILTIN_LATENT_UPSCALERS,
    UPSCALER_SCAN_SCHEMA_VERSION,
    SUPPORTED_UPSCALER_EXTENSIONS,
    UpscalerDescriptor,
    UpscalerDiscoveryDiagnostic,
    UpscalerDiscoveryResult,
    build_upscaler_id,
)
from image_gen.systems.upscaling.diagnostics import bounded_error_text

SCAN_MODES = frozenset({"unidentified", "all", "selected"})
UPSCALER_DISCOVERY_RECURSIVE = True
UPSCALER_CATALOG_FLATTENED = True


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(max(64 * 1024, int(chunk_size)))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _configured_additional_roots(context: Any) -> tuple[Path, ...]:
    config = getattr(context, "config", {}) or {}
    if not isinstance(config, Mapping):
        return ()
    section = config.get("upscaling") or {}
    if not isinstance(section, Mapping):
        return ()
    raw_roots = section.get("additional_roots") or ()
    if isinstance(raw_roots, (str, os.PathLike)):
        raw_roots = (raw_roots,)
    output: list[Path] = []
    for value in raw_roots:
        if value is None or not str(value).strip():
            continue
        resolver = getattr(context, "resolve_project_path", None)
        if callable(resolver):
            output.append(Path(resolver(value)))
        else:
            root = Path(getattr(context, "project_root", Path.cwd()))
            selected = Path(os.path.expandvars(os.path.expanduser(str(value))))
            output.append((selected if selected.is_absolute() else root / selected).resolve())
    return tuple(output)


def configured_upscaler_roots(context: Any) -> tuple[Path, ...]:
    raw_roots = (
        Path(getattr(context, "esrgan_dir")),
        Path(getattr(context, "realesrgan_dir")),
        *_configured_additional_roots(context),
    )
    output: list[Path] = []
    seen: set[str] = set()
    for raw_root in raw_roots:
        root = raw_root.expanduser().resolve()
        key = canonical_path_key(root)
        if key in seen:
            continue
        seen.add(key)
        output.append(root)
    return tuple(output)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _source_location(path: Path, roots: Iterable[Path]) -> tuple[str, str]:
    resolved = path.expanduser().resolve()
    matches: list[tuple[int, Path]] = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if _is_within(resolved, root):
            matches.append((len(root.parts), root))
    if not matches:
        return "", path.name
    _, root = max(matches, key=lambda item: item[0])
    return str(root), resolved.relative_to(root).as_posix()


def _with_location_metadata(
    descriptor: UpscalerDescriptor,
    *,
    path: Path,
    roots: Iterable[Path],
) -> UpscalerDescriptor:
    source_root, relative_path = _source_location(path, roots)
    return replace(
        descriptor,
        source_root=source_root,
        relative_path=relative_path,
    )


def _canonical_descriptor_sort_key(
    descriptor: UpscalerDescriptor,
    *,
    roots: tuple[Path, ...],
) -> tuple[int, int, str, str]:
    root_priority = len(roots)
    descriptor_root = str(descriptor.source_root or "")
    for index, root in enumerate(roots):
        if descriptor_root == str(root):
            root_priority = index
            break
    relative_path = str(descriptor.relative_path or descriptor.file_name or "").replace("\\", "/")
    parts = Path(relative_path).parts if relative_path else ()
    return (
        int(root_priority),
        len(parts),
        relative_path.casefold(),
        str(descriptor.path).casefold(),
    )


def _discover_files(
    roots: Iterable[Path],
    *,
    diagnostics: list[UpscalerDiscoveryDiagnostic],
) -> tuple[Path, ...]:
    files: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            diagnostics.append(
                UpscalerDiscoveryDiagnostic(
                    "warning",
                    "upscaler_root_missing",
                    "Configured upscaler root does not exist; discovery did not create it.",
                    str(root),
                )
            )
            continue
        if not root.is_dir():
            diagnostics.append(
                UpscalerDiscoveryDiagnostic(
                    "warning",
                    "upscaler_root_not_directory",
                    "Configured upscaler root is not a directory.",
                    str(root),
                )
            )
            continue
        resolved_root = root.resolve()
        try:
            candidates = root.rglob("*")
            for candidate in candidates:
                if candidate.suffix.casefold() not in SUPPORTED_UPSCALER_EXTENSIONS:
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError as exc:
                    diagnostics.append(
                        UpscalerDiscoveryDiagnostic(
                            "warning",
                            "upscaler_file_unavailable",
                            bounded_error_text(exc),
                            str(candidate),
                        )
                    )
                    continue
                if not _is_within(resolved, resolved_root):
                    diagnostics.append(
                        UpscalerDiscoveryDiagnostic(
                            "warning",
                            "upscaler_symlink_escape_blocked",
                            "A model path resolved outside its configured root and was ignored.",
                            str(candidate),
                        )
                    )
                    continue
                if not resolved.is_file():
                    continue
                files.setdefault(canonical_path_key(resolved), resolved)
        except OSError as exc:
            diagnostics.append(
                UpscalerDiscoveryDiagnostic(
                    "warning",
                    "upscaler_root_scan_failed",
                    bounded_error_text(exc),
                    str(root),
                )
            )
    return tuple(files[key] for key in sorted(files))


def _selected_file(
    selected_file: str | os.PathLike[str] | None,
    *,
    roots: tuple[Path, ...],
    diagnostics: list[UpscalerDiscoveryDiagnostic],
) -> tuple[Path, ...]:
    if selected_file is None or not str(selected_file).strip():
        raise ValueError("selected_file is required when mode='selected'.")
    selected = Path(selected_file).expanduser().resolve()
    if selected.suffix.casefold() not in SUPPORTED_UPSCALER_EXTENSIONS:
        diagnostics.append(
            UpscalerDiscoveryDiagnostic(
                "error",
                "unsupported_upscaler_extension",
                "Only configured .pth upscaler files are eligible for Phase 14N discovery.",
                str(selected),
            )
        )
        return ()
    if not any(_is_within(selected, root.resolve()) for root in roots):
        diagnostics.append(
            UpscalerDiscoveryDiagnostic(
                "error",
                "selected_upscaler_outside_roots",
                "The selected file is outside every configured upscaler root.",
                str(selected),
            )
        )
        return ()
    if not selected.is_file():
        diagnostics.append(
            UpscalerDiscoveryDiagnostic(
                "error",
                "selected_upscaler_missing",
                "The selected upscaler file does not exist.",
                str(selected),
            )
        )
        return ()
    return (selected,)


def _descriptor_from_classification(
    path: Path,
    *,
    roots: Iterable[Path],
    sha256: str,
    classification: UpscalerClassification,
    cache_status: str,
) -> UpscalerDescriptor:
    stat = path.stat()
    architecture = classification.architecture
    native_scale = classification.native_scale
    upscaler_id = build_upscaler_id(
        loader_backend=classification.loader_backend or CANONICAL_LOADER_BACKEND,
        architecture=architecture,
        native_scale=native_scale,
        sha256=sha256,
    )
    source_root, relative_path = _source_location(path, roots)
    return UpscalerDescriptor(
        upscaler_id=upscaler_id,
        display_name=path.stem,
        path=str(path),
        file_name=path.name,
        sha256=sha256,
        file_size_bytes=int(stat.st_size),
        modified_time_ns=int(stat.st_mtime_ns),
        architecture=architecture,
        architecture_confidence=classification.architecture_confidence,
        native_scale=native_scale,
        input_channels=classification.input_channels,
        output_channels=classification.output_channels,
        supports_half=classification.supports_half,
        supports_bfloat16=classification.supports_bfloat16,
        tile_supported=classification.tile_supported,
        load_status=classification.load_status,
        scan_cache_status=cache_status,
        loader_backend=classification.loader_backend,
        compatibility_notes=classification.compatibility_notes,
        bounded_error=bounded_error_text(classification.bounded_error),
        source_root=source_root,
        relative_path=relative_path,
    )


def _collapse_duplicate_content(
    descriptors: Iterable[UpscalerDescriptor],
    *,
    roots: tuple[Path, ...],
) -> tuple[UpscalerDescriptor, ...]:
    grouped: dict[str, list[UpscalerDescriptor]] = {}
    for descriptor in descriptors:
        grouped.setdefault(descriptor.upscaler_id, []).append(descriptor)

    collapsed: list[UpscalerDescriptor] = []
    for key in sorted(grouped):
        group = sorted(
            grouped[key],
            key=lambda item: _canonical_descriptor_sort_key(item, roots=roots),
        )
        primary = group[0]
        alias_paths = tuple(str(item.path) for item in group[1:])
        alias_relative_paths = tuple(str(item.relative_path or item.file_name) for item in group[1:])
        if alias_paths or alias_relative_paths:
            primary = replace(
                primary,
                alias_paths=alias_paths,
                alias_relative_paths=alias_relative_paths,
            )
        collapsed.append(primary)
    return tuple(collapsed)


def discover_upscalers(
    context: Any,
    *,
    mode: str = "unidentified",
    selected_file: str | os.PathLike[str] | None = None,
    persist_cache: bool = True,
) -> UpscalerDiscoveryResult:
    normalized_mode = str(mode or "unidentified").strip().casefold()
    if normalized_mode not in SCAN_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(SCAN_MODES))}")

    diagnostics: list[UpscalerDiscoveryDiagnostic] = []
    roots = configured_upscaler_roots(context)
    if normalized_mode == "selected":
        files = _selected_file(selected_file, roots=roots, diagnostics=diagnostics)
    else:
        files = _discover_files(roots, diagnostics=diagnostics)

    cache_root = Path(getattr(context, "cache_root"))
    cache = UpscalerScanCache(cache_root)
    cache.records()
    backend_version = loader_backend_version()
    descriptors: list[UpscalerDescriptor] = []
    cache_changed = False

    if cache.load_error:
        diagnostics.append(
            UpscalerDiscoveryDiagnostic(
                "warning",
                "upscaler_cache_read_failed",
                bounded_error_text(cache.load_error),
                str(cache.path),
            )
        )

    for path in files:
        record = cache.get(path)
        current = record is not None and record.is_current(
            path,
            loader_backend_version=backend_version,
        )
        if current and normalized_mode != "all":
            cached = record.descriptor.with_cache_status("hit")
            descriptors.append(
                _with_location_metadata(cached, path=path, roots=roots)
            )
            continue

        prior_exists = record is not None
        try:
            digest = sha256_file(path)
            classification = inspect_upscaler_file(path)
            cache_status = "refresh" if normalized_mode == "all" else ("stale" if prior_exists else "miss")
            descriptor = _descriptor_from_classification(
                path,
                roots=roots,
                sha256=digest,
                classification=classification,
                cache_status=cache_status,
            )
            descriptors.append(descriptor)
            cache.put(
                UpscalerCacheRecord(
                    schema_version=UPSCALER_SCAN_SCHEMA_VERSION,
                    path=str(path),
                    file_size_bytes=descriptor.file_size_bytes,
                    modified_time_ns=descriptor.modified_time_ns,
                    sha256=descriptor.sha256,
                    loader_backend_version=backend_version,
                    scan_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    descriptor=descriptor.with_cache_status("cached"),
                )
            )
            cache_changed = True
        except OSError as exc:
            diagnostics.append(
                UpscalerDiscoveryDiagnostic(
                    "error",
                    "upscaler_scan_failed",
                    bounded_error_text(exc),
                    str(path),
                )
            )

    descriptors = list(_collapse_duplicate_content(descriptors, roots=roots))
    descriptors.sort(
        key=lambda item: (
            item.catalog_name.casefold(),
            item.sha256,
            item.path,
        )
    )
    if persist_cache and cache_changed:
        try:
            cache.save()
        except OSError as exc:
            diagnostics.append(
                UpscalerDiscoveryDiagnostic(
                    "warning",
                    "upscaler_cache_write_failed",
                    bounded_error_text(exc),
                    str(cache.path),
                )
            )

    return UpscalerDiscoveryResult(
        mode=normalized_mode,
        roots=tuple(str(root) for root in roots),
        built_in_latent=BUILTIN_LATENT_UPSCALERS,
        neural_descriptors=tuple(descriptors),
        diagnostics=tuple(diagnostics),
        cache_path=str(cache.path),
    )
