from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import os


@dataclass
class FingerprintResult:
    path: str
    file_size: int
    modified_time: float
    created_time: float
    quick_fingerprint: str
    sha256: Optional[str] = None
    blake3: Optional[str] = None


class FileFingerprint:
    """
    Computes quick and strong file fingerprints.

    Quick fingerprint:
    - path-independent
    - based on file size, mtime, and sample bytes

    Strong fingerprint:
    - full-file SHA256
    - optional BLAKE3 if package is installed
    """

    def __init__(self, sample_size: int = 65536):
        self.sample_size = sample_size

    def fingerprint_file(
        self,
        path: str,
        compute_sha256: bool = False,
        compute_blake3: bool = False,
    ) -> FingerprintResult:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()
        quick_fingerprint = self._compute_quick_fingerprint(file_path, stat.st_size, stat.st_mtime)

        sha256 = self.compute_sha256(file_path) if compute_sha256 else None
        blake3 = self.compute_blake3(file_path) if compute_blake3 else None

        return FingerprintResult(
            path=str(file_path),
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            created_time=getattr(stat, "st_ctime", stat.st_mtime),
            quick_fingerprint=quick_fingerprint,
            sha256=sha256,
            blake3=blake3,
        )

    def compute_sha256(self, path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        file_path = Path(path)
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)

        return hasher.hexdigest()

    def compute_blake3(self, path: str | Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
        try:
            import blake3  # type: ignore
        except ImportError:
            return None

        file_path = Path(path)
        hasher = blake3.blake3()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)

        return hasher.hexdigest()

    def _compute_quick_fingerprint(self, file_path: Path, file_size: int, modified_time: float) -> str:
        sample_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            head = f.read(self.sample_size)

            tail = b""
            if file_size > self.sample_size:
                try:
                    f.seek(max(0, file_size - self.sample_size), os.SEEK_SET)
                    tail = f.read(self.sample_size)
                except OSError:
                    tail = b""

        sample_hash.update(str(file_size).encode("utf-8"))
        sample_hash.update(str(modified_time).encode("utf-8"))
        sample_hash.update(head)
        sample_hash.update(tail)

        return sample_hash.hexdigest()