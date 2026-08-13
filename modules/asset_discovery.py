from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import os


DEFAULT_EXCLUDED_DIR_NAMES = frozenset({
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".cache",
    "venv",
    "node_modules",
})


class AssetResolutionError(RuntimeError):
    """Base error for deterministic recursive asset resolution."""


class AmbiguousAssetError(AssetResolutionError):
    def __init__(self, requested: str, matches: Sequence[Path]) -> None:
        self.requested = requested
        self.matches = tuple(matches)
        details = "\n  - ".join(str(path) for path in self.matches)
        super().__init__(
            f"Asset reference {requested!r} matched multiple files. "
            f"Use a relative or absolute path to disambiguate:\n  - {details}"
        )


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str] | None:
    if extensions is None:
        return None
    output: set[str] = set()
    for value in extensions:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = "." + text
        output.add(text)
    return output


def _is_excluded(path: Path, root: Path, excluded_dir_names: set[str]) -> bool:
    try:
        relative = path.relative_to(root)
        parts = relative.parts[:-1] if path.is_file() else relative.parts
    except ValueError:
        parts = path.parts
    return any(part in excluded_dir_names for part in parts)


def iter_asset_files(
    root: str | os.PathLike[str],
    *,
    extensions: Iterable[str] | None = None,
    recursive: bool = True,
    excluded_dir_names: Iterable[str] | None = None,
) -> Iterable[Path]:
    """Yield files under ``root`` deterministically, at arbitrary nesting depth."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return

    allowed = _normalize_extensions(extensions)
    excluded = set(DEFAULT_EXCLUDED_DIR_NAMES)
    if excluded_dir_names is not None:
        excluded.update(str(value) for value in excluded_dir_names)

    walker = base.rglob("*") if recursive else base.glob("*")
    candidates: list[Path] = []
    for path in walker:
        try:
            if not path.is_file():
                continue
            if _is_excluded(path, base, excluded):
                continue
            if allowed is not None and path.suffix.lower() not in allowed:
                continue
            candidates.append(path.resolve())
        except OSError:
            continue

    for path in sorted(candidates, key=lambda item: str(item).casefold()):
        yield path


def find_named_files(
    root: str | os.PathLike[str],
    filename: str,
    *,
    extensions: Iterable[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Find files whose basename matches ``filename`` case-insensitively."""

    wanted = Path(str(filename or "").strip()).name.casefold()
    if not wanted:
        return []
    return [
        path
        for path in iter_asset_files(root, extensions=extensions, recursive=recursive)
        if path.name.casefold() == wanted
    ]


def resolve_nested_asset(
    root: str | os.PathLike[str],
    requested: str | os.PathLike[str],
    *,
    extensions: Iterable[str] | None = None,
    allow_stem_match: bool = True,
) -> Path | None:
    """Resolve a file beneath ``root`` without assuming a fixed nesting layout.

    Resolution order is deliberately conservative:
      1. exact absolute path
      2. exact ``root / requested`` path
      3. exact relative-suffix match anywhere below root
      4. exact basename match anywhere below root
      5. optional stem match when ``requested`` has no suffix

    Any tier with multiple matches raises ``AmbiguousAssetError`` rather than
    silently selecting the first file.
    """

    text = str(requested or "").strip()
    if not text:
        return None

    base = Path(root).expanduser().resolve()
    allowed = _normalize_extensions(extensions)

    direct = Path(text).expanduser()
    if direct.is_absolute() and direct.is_file():
        resolved = direct.resolve()
        if allowed is None or resolved.suffix.lower() in allowed:
            return resolved

    exact = (base / direct).resolve()
    try:
        exact.relative_to(base)
    except ValueError:
        exact = base / "__outside_root__"
    if exact.is_file() and (allowed is None or exact.suffix.lower() in allowed):
        return exact.resolve()

    files = list(iter_asset_files(base, extensions=allowed, recursive=True))
    normalized_request = Path(text.replace("\\", "/")).as_posix().lstrip("./").casefold()

    if "/" in normalized_request:
        suffix_matches: list[Path] = []
        for path in files:
            try:
                relative = path.relative_to(base).as_posix().casefold()
            except ValueError:
                continue
            if relative == normalized_request or relative.endswith("/" + normalized_request):
                suffix_matches.append(path)
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise AmbiguousAssetError(text, suffix_matches)

    basename = Path(text).name.casefold()
    basename_matches = [path for path in files if path.name.casefold() == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise AmbiguousAssetError(text, basename_matches)

    requested_path = Path(text)
    if allow_stem_match and not requested_path.suffix:
        stem = requested_path.name.casefold()
        stem_matches = [path for path in files if path.stem.casefold() == stem]
        if len(stem_matches) == 1:
            return stem_matches[0]
        if len(stem_matches) > 1:
            raise AmbiguousAssetError(text, stem_matches)

    return None


def resolve_unique_named_file(
    root: str | os.PathLike[str],
    filename: str,
    *,
    extensions: Iterable[str] | None = None,
) -> Path:
    """Resolve exactly one nested file with ``filename`` or raise a clear error."""

    matches = find_named_files(root, filename, extensions=extensions, recursive=True)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousAssetError(filename, matches)
    raise FileNotFoundError(f"Could not find {filename!r} anywhere under: {Path(root).expanduser().resolve()}")
