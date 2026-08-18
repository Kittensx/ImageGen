from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Mapping


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

    asset_type: str = "unclassified_asset"
    format_type: str = "other"
    architecture: str = ""
    architecture_state: str = "observed_unclassified"

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
    location_state: str = "available"


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

@dataclass
class ComponentSnapshotRecord:
    id: int
    asset_id: int
    snapshot_at: str
    snapshot_version: str
    component_role: str
    source_prefixes_json: Optional[str]
    tensor_count: int
    total_bytes: int
    component_sha256: str
    structure_sha256: str
    dtype_summary_json: Optional[str] = None
    tensor_manifest_json: Optional[str] = None
    metadata_json: Optional[str] = None



@dataclass
class ComponentIdentityRecord:
    component_sha256: str
    structure_sha256: Optional[str]
    total_bytes: Optional[int]
    tensor_count: Optional[int]
    first_seen_at: str
    last_seen_at: str
    metadata_json: Optional[str] = None


@dataclass
class ComponentSourceRecord:
    id: int
    component_sha256: str
    asset_id: int
    component_role: str
    source_form: str
    embedded_state: str
    provider_family: Optional[str]
    provider_version: Optional[str]
    availability_state: str
    locator_json: Optional[str]
    scan_timestamp: Optional[str]
    scanner_version: Optional[str]
    snapshot_version: str
    metadata_json: Optional[str] = None



LOCATION_STATE_AVAILABLE = "available"
LOCATION_STATE_ARCHIVED = "archived"
LOCATION_STATE_MISSING = "missing"
LOCATION_STATE_MOVED_RELINKED = "moved_relinked"
LOCATION_STATE_INACCESSIBLE = "inaccessible"

SCAN_SCOPE_SINGLE_ASSET = "single_asset"
SCAN_SCOPE_SELECTED_ASSETS = "selected_assets"
SCAN_SCOPE_LIBRARY_REFRESH = "library_refresh"
SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH = "external_repository_refresh"
SCAN_STRENGTH_QUICK = "quick"
SCAN_STRENGTH_STRUCTURAL = "structural"
SCAN_STRENGTH_FULL = "full"
ANALYSIS_STRENGTH_NONE = "none"
ANALYSIS_STRENGTH_LAYOUT = "layout"
ANALYSIS_STRENGTH_EXACT = "exact"


@dataclass(frozen=True)
class ComponentScanRequest:
    scope: str = SCAN_SCOPE_LIBRARY_REFRESH
    strength: str = SCAN_STRENGTH_STRUCTURAL
    analysis_strength: str = ANALYSIS_STRENGTH_NONE
    paths: tuple[str, ...] = ()
    repository_roots: tuple[str, ...] = ()
    force: bool = False
    dry_run: bool = False

    def normalized(self) -> "ComponentScanRequest":
        def _norm_many(values: tuple[str, ...]) -> tuple[str, ...]:
            result: list[str] = []
            seen: set[str] = set()
            for value in values:
                item = str(value or "").strip()
                if not item:
                    continue
                token = item.casefold()
                if token in seen:
                    continue
                seen.add(token)
                result.append(item)
            return tuple(result)

        scope = str(self.scope or SCAN_SCOPE_LIBRARY_REFRESH).strip().lower()
        if scope not in {
            SCAN_SCOPE_SINGLE_ASSET,
            SCAN_SCOPE_SELECTED_ASSETS,
            SCAN_SCOPE_LIBRARY_REFRESH,
            SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH,
        }:
            raise ValueError(f"Unsupported component scan scope: {self.scope!r}")
        strength = str(self.strength or SCAN_STRENGTH_STRUCTURAL).strip().lower()
        if strength not in {SCAN_STRENGTH_QUICK, SCAN_STRENGTH_STRUCTURAL, SCAN_STRENGTH_FULL}:
            raise ValueError(f"Unsupported component scan strength: {self.strength!r}")
        analysis_strength = str(self.analysis_strength or ANALYSIS_STRENGTH_NONE).strip().lower()
        if analysis_strength not in {ANALYSIS_STRENGTH_NONE, ANALYSIS_STRENGTH_LAYOUT, ANALYSIS_STRENGTH_EXACT}:
            raise ValueError(f"Unsupported analytical strength: {self.analysis_strength!r}")
        paths = _norm_many(tuple(self.paths or ()))
        repository_roots = _norm_many(tuple(self.repository_roots or ()))
        if scope == SCAN_SCOPE_SINGLE_ASSET and len(paths) != 1:
            raise ValueError("single_asset scans require exactly one path.")
        if scope == SCAN_SCOPE_SELECTED_ASSETS and not paths:
            raise ValueError("selected_assets scans require at least one path.")
        if scope == SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH and not repository_roots:
            raise ValueError("external_repository_refresh scans require at least one repository root.")
        return ComponentScanRequest(
            scope=scope,
            strength=strength,
            analysis_strength=analysis_strength,
            paths=paths,
            repository_roots=repository_roots,
            force=bool(self.force),
            dry_run=bool(self.dry_run),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ComponentScanRequest":
        if payload is None:
            return cls().normalized()
        values = dict(payload)
        paths_raw = values.get("paths") or ()
        repository_raw = values.get("repository_roots") or ()
        if isinstance(paths_raw, (str, bytes)):
            paths = (str(paths_raw),)
        else:
            paths = tuple(str(item) for item in paths_raw)
        if isinstance(repository_raw, (str, bytes)):
            repository_roots = (str(repository_raw),)
        else:
            repository_roots = tuple(str(item) for item in repository_raw)
        return cls(
            scope=str(values.get("scope") or SCAN_SCOPE_LIBRARY_REFRESH),
            strength=str(values.get("strength") or SCAN_STRENGTH_STRUCTURAL),
            analysis_strength=str(values.get("analysis_strength") or ANALYSIS_STRENGTH_NONE),
            paths=paths,
            repository_roots=repository_roots,
            force=bool(values.get("force")),
            dry_run=bool(values.get("dry_run")),
        ).normalized()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "strength": self.strength,
            "analysis_strength": self.analysis_strength,
            "paths": list(self.paths),
            "repository_roots": list(self.repository_roots),
            "force": self.force,
            "dry_run": self.dry_run,
        }
