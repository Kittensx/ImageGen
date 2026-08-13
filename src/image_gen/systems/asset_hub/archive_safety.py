from __future__ import annotations

import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from image_gen.systems.asset_hub.providers.base import AssetHubError

ARCHIVE_MEMBER_LIMIT = 512
ARCHIVE_EXPANDED_LIMIT = 16 * 1024 * 1024 * 1024
ARCHIVE_MEMBER_SIZE_LIMIT = 8 * 1024 * 1024 * 1024
ARCHIVE_COMPRESSION_RATIO_LIMIT = 250.0
_EXECUTABLE_SUFFIXES = {
    ".exe", ".dll", ".msi", ".com", ".scr", ".bat", ".cmd", ".ps1", ".psm1",
    ".py", ".pyw", ".sh", ".so", ".dylib", ".jar", ".app", ".reg",
}
_ALLOWED_INSTALL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".json"}


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    size_bytes: int
    compressed_bytes: int
    suffix: str
    install_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sizeBytes": self.size_bytes,
            "compressedBytes": self.compressed_bytes,
            "suffix": self.suffix,
            "installCandidate": self.install_candidate,
        }


@dataclass(frozen=True)
class ArchiveInspection:
    members: tuple[ArchiveMember, ...]
    install_candidates: tuple[str, ...]
    expanded_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [item.to_dict() for item in self.members],
            "installCandidates": list(self.install_candidates),
            "expandedSizeBytes": self.expanded_size_bytes,
        }


def _safe_member_name(raw: str) -> str:
    name = str(raw or "").replace("\\", "/")
    pure = PurePosixPath(name)
    if not name or name.startswith(("/", "//")) or pure.is_absolute():
        raise AssetHubError("archive_path_unsafe", "Archive contains an absolute path.", status_code=422)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise AssetHubError("archive_path_unsafe", "Archive contains parent traversal or an invalid member path.", status_code=422)
    first = pure.parts[0] if pure.parts else ""
    if len(first) >= 2 and first[1] == ":":
        raise AssetHubError("archive_path_unsafe", "Archive contains a Windows drive path.", status_code=422)
    return pure.as_posix()


def _reject_special_member(info: zipfile.ZipInfo) -> None:
    mode = (int(info.external_attr) >> 16) & 0xFFFF
    if mode:
        kind = stat.S_IFMT(mode)
        if kind in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK}:
            raise AssetHubError("archive_special_file", "Archive contains a link or special device entry.", status_code=422)
    if int(info.flag_bits) & 0x1:
        raise AssetHubError("archive_encrypted", "Encrypted ZIP entries are not supported.", status_code=422)


def inspect_zip(path: str | os.PathLike[str]) -> ArchiveInspection:
    selected = Path(path).expanduser().resolve()
    try:
        archive = zipfile.ZipFile(selected, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise AssetHubError("archive_invalid", "ZIP archive could not be inspected safely.", status_code=422) from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > ARCHIVE_MEMBER_LIMIT:
            raise AssetHubError("archive_member_limit", "ZIP archive contains too many members.", status_code=422)
        members: list[ArchiveMember] = []
        candidates: list[str] = []
        expanded = 0
        for info in infos:
            name = _safe_member_name(info.filename)
            _reject_special_member(info)
            if info.is_dir():
                continue
            expanded += max(0, int(info.file_size))
            if expanded > ARCHIVE_EXPANDED_LIMIT or int(info.file_size) > ARCHIVE_MEMBER_SIZE_LIMIT:
                raise AssetHubError("archive_size_limit", "ZIP archive exceeds the allowed expanded size.", status_code=422)
            compressed = max(0, int(info.compress_size))
            if int(info.file_size) > 1024 * 1024 and compressed == 0:
                raise AssetHubError("archive_compression_bomb", "ZIP archive contains an unsafe compression ratio.", status_code=422)
            if compressed > 0 and (int(info.file_size) / compressed) > ARCHIVE_COMPRESSION_RATIO_LIMIT:
                raise AssetHubError("archive_compression_bomb", "ZIP archive contains an unsafe compression ratio.", status_code=422)
            suffix = Path(name).suffix.lower()
            if suffix in _EXECUTABLE_SUFFIXES:
                raise AssetHubError("archive_executable_content", "ZIP archive contains executable or script content.", status_code=422)
            install_candidate = suffix in _ALLOWED_INSTALL_SUFFIXES
            if install_candidate:
                candidates.append(name)
            members.append(ArchiveMember(name, int(info.file_size), compressed, suffix, install_candidate))
        return ArchiveInspection(tuple(members), tuple(candidates), expanded)


def extract_member(path: str | os.PathLike[str], member_name: str, destination: str | os.PathLike[str]) -> Path:
    selected = Path(path).expanduser().resolve()
    inspection = inspect_zip(selected)
    normalized = _safe_member_name(member_name)
    if normalized not in inspection.install_candidates:
        raise AssetHubError("archive_member_invalid", "Selected archive member is not an installable asset candidate.", status_code=400)
    target_root = Path(destination).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target = (target_root / Path(normalized).name).resolve()
    try:
        target.relative_to(target_root)
    except ValueError as exc:
        raise AssetHubError("archive_path_unsafe", "Archive extraction escaped the staging directory.", status_code=422) from exc
    try:
        with zipfile.ZipFile(selected, "r") as archive, archive.open(normalized, "r") as source, target.open("wb") as sink:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise AssetHubError("archive_extract_failed", "Selected archive member could not be staged safely.", status_code=422) from exc
    return target
