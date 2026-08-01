from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class AssetRecord:
    id: int
    path: str
    filename: str
    extension: str
    file_size: int
    modified_time: float
    created_time: float
    first_seen_at: str
    last_seen_at: str
    exists_on_disk: bool

    quick_fingerprint: Optional[str] = None
    sha256: Optional[str] = None
    blake3: Optional[str] = None

    asset_type: str = "unknown"
    format_type: str = "other"
    architecture: str = "unknown"

    checkpoint_kind: str = "unknown"
    has_unet: bool = False
    has_vae: bool = False
    has_text_encoder: bool = False
    has_text_encoder_2: bool = False

    library_root: Optional[str] = None
    managed_category: Optional[str] = None
    path_kind: str = "external"

    key_count: Optional[int] = None
    metadata_json: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class InspectionRecord:
    id: int
    asset_id: int
    inspected_at: str
    inspector_version: Optional[str] = None
    key_count: Optional[int] = None
    prefix_summary_json: Optional[str] = None
    example_keys_json: Optional[str] = None
    dtype_summary_json: Optional[str] = None
    tensor_shape_summary_json: Optional[str] = None
    result_json: Optional[str] = None


@dataclass
class LoadHistoryRecord:
    id: int
    asset_id: int
    loaded_at: str
    status: str
    device: Optional[str] = None
    precision: Optional[str] = None
    load_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    context_json: Optional[str] = None


@dataclass
class ScanResult:
    scanned_paths: int = 0
    matched_files: int = 0
    inserted_assets: int = 0
    updated_assets: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_paths": self.scanned_paths,
            "matched_files": self.matched_files,
            "inserted_assets": self.inserted_assets,
            "updated_assets": self.updated_assets,
            "skipped_files": self.skipped_files,
            "errors": list(self.errors),
        }