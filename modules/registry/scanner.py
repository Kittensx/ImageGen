from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os

from .asset_registry import AssetRegistry
from .models import ScanResult


DEFAULT_SUPPORTED_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
}


class AssetScanner:
    """
    Filesystem scanner for registering model-like assets.

    This scanner only registers files and fingerprints them.
    It does not deeply inspect checkpoint keys by itself.
    """

    def __init__(
        self,
        registry: AssetRegistry,
        supported_extensions: Optional[set[str]] = None,
    ):
        self.registry = registry
        self.supported_extensions = supported_extensions or set(DEFAULT_SUPPORTED_EXTENSIONS)

    def scan_directory(
        self,
        root_dir: str,
        recursive: bool = True,
        compute_sha256: bool = False,
        compute_blake3: bool = False,
        skip_hidden: bool = True,
        exclude_dir_names: Optional[set[str]] = None,
    ) -> ScanResult:
        result = ScanResult()
        root = Path(root_dir)

        if not root.exists():
            result.errors.append(f"Directory does not exist: {root}")
            return result

        excluded = exclude_dir_names or {
            "__pycache__",
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
        }

        walker = root.rglob("*") if recursive else root.glob("*")

        for path in walker:
            result.scanned_paths += 1

            try:
                if path.is_dir():
                    continue

                if skip_hidden and self._is_hidden(path):
                    result.skipped_files += 1
                    continue

                if any(part in excluded for part in path.parts):
                    result.skipped_files += 1
                    continue

                if path.suffix.lower() not in self.supported_extensions:
                    result.skipped_files += 1
                    continue

                result.matched_files += 1

                existing = self.registry.get_asset_by_path(str(path.resolve()))
                asset = self.registry.register_file(
                    str(path.resolve()),
                    compute_sha256=compute_sha256,
                    compute_blake3=compute_blake3,
                )

                if existing is None:
                    result.inserted_assets += 1
                else:
                    result.updated_assets += 1

            except Exception as e:
                result.errors.append(f"{path}: {e}")

        return result

    def scan_files(
        self,
        file_paths: Iterable[str],
        compute_sha256: bool = False,
        compute_blake3: bool = False,
    ) -> ScanResult:
        result = ScanResult()

        for file_path in file_paths:
            result.scanned_paths += 1
            path = Path(file_path)

            try:
                if not path.exists() or not path.is_file():
                    result.errors.append(f"Missing file: {path}")
                    continue

                if path.suffix.lower() not in self.supported_extensions:
                    result.skipped_files += 1
                    continue

                result.matched_files += 1

                existing = self.registry.get_asset_by_path(str(path.resolve()))
                self.registry.register_file(
                    str(path.resolve()),
                    compute_sha256=compute_sha256,
                    compute_blake3=compute_blake3,
                )

                if existing is None:
                    result.inserted_assets += 1
                else:
                    result.updated_assets += 1

            except Exception as e:
                result.errors.append(f"{path}: {e}")

        return result

    def _is_hidden(self, path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts if part not in (os.sep, ""))