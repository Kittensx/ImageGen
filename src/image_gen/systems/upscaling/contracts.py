from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from image_gen.systems.upscaling.capabilities import discovery_disposition, runtime_disposition

UPSCALER_SCAN_SCHEMA_VERSION = 1
UPSCALER_ID_HASH_PREFIX_LENGTH = 16
SUPPORTED_UPSCALER_EXTENSIONS = frozenset({".pth"})
SUPPORTED_NATIVE_SCALES = frozenset({2, 4, 8})
SUPPORTED_CLASSIFICATION_STATES = frozenset(
    {
        "supported",
        "deferred_architecture",
        "deferred_scale",
        "deferred_hardware_validation",
        "unsupported_architecture",
        "unsupported_channels",
        "unsupported_scale",
        "inspection_failed",
        "corrupt",
        "unclassified",
    }
)

SUPPORTED_RUNTIME_QUALIFICATION_STATES = frozenset(
    {
        "unqualified",
        "qualified_cpu",
        "qualified_cuda",
        "backend_unavailable",
        "backend_unqualified",
        "hash_mismatch",
        "metadata_mismatch",
        "load_failed",
        "runtime_contract_failed",
        "security_boundary_failed",
        "deferred_not_tested",
        "deferred_hardware_limit",
        "deferred_hardware_unavailable",
        "deferred_backend_support",
        "deferred_architecture",
        "deferred_scale",
    }
)


_STABLE_TOKEN = re.compile(r"[^a-z0-9_.-]+")


def normalize_architecture_token(value: Any) -> str:
    token = str(value or "unclassified").strip().casefold().replace(" ", "_")
    token = _STABLE_TOKEN.sub("_", token).strip("_.-")
    return token or "unclassified"


def build_upscaler_id(
    *,
    loader_backend: str,
    architecture: str,
    native_scale: int,
    sha256: str,
) -> str:
    digest = str(sha256 or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A full lowercase or uppercase SHA-256 digest is required.")
    backend = normalize_architecture_token(loader_backend)
    architecture_token = normalize_architecture_token(architecture)
    scale_token = f"x{max(0, int(native_scale))}"
    return (
        f"upscaler.{backend}.{architecture_token}.{scale_token}."
        f"{digest[:UPSCALER_ID_HASH_PREFIX_LENGTH]}"
    )


def _string_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(str(value) for value in values or ())


@dataclass(frozen=True)
class BuiltinLatentUpscaler:
    """Compatibility record type; active latent entries were retired in Phase 14N-11."""

    upscaler_id: str
    display_name: str
    interpolation: str
    strategy: str = "retired"

    def to_dict(self) -> dict[str, Any]:
        return {
            "upscaler_id": self.upscaler_id,
            "display_name": self.display_name,
            "interpolation": self.interpolation,
            "strategy": self.strategy,
            "selectable": False,
            "retired": True,
        }


BUILTIN_LATENT_UPSCALERS: tuple[BuiltinLatentUpscaler, ...] = ()


@dataclass(frozen=True)
class UpscalerDescriptor:
    upscaler_id: str
    display_name: str
    path: str
    file_name: str
    sha256: str
    file_size_bytes: int
    modified_time_ns: int
    architecture: str
    architecture_confidence: str
    native_scale: int
    input_channels: int
    output_channels: int
    supports_half: bool
    supports_bfloat16: bool
    tile_supported: bool
    load_status: str
    scan_cache_status: str
    loader_backend: str
    compatibility_notes: tuple[str, ...] = field(default_factory=tuple)
    bounded_error: str = ""
    source_root: str = ""
    relative_path: str = ""
    alias_paths: tuple[str, ...] = field(default_factory=tuple)
    alias_relative_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        status = str(self.load_status or "unclassified").strip().casefold()
        if status not in SUPPORTED_CLASSIFICATION_STATES:
            raise ValueError(f"Unsupported upscaler load_status: {self.load_status!r}")
        object.__setattr__(self, "load_status", status)
        object.__setattr__(self, "compatibility_notes", _string_tuple(self.compatibility_notes))
        object.__setattr__(self, "alias_paths", _string_tuple(self.alias_paths))
        object.__setattr__(self, "alias_relative_paths", _string_tuple(self.alias_relative_paths))

    @property
    def selectable(self) -> bool:
        return self.load_status == "supported"

    @property
    def disposition(self) -> str:
        return discovery_disposition(self.load_status)

    @property
    def deferred(self) -> bool:
        return self.disposition == "deferred"

    @property
    def failed(self) -> bool:
        return self.disposition == "failed"

    @property
    def nested(self) -> bool:
        value = str(self.relative_path or self.file_name or "").replace("\\", "/")
        return "/" in value.strip("/")

    @property
    def catalog_name(self) -> str:
        value = str(self.relative_path or self.file_name or "").replace("\\", "/").strip("/")
        if value:
            suffix = Path(value).suffix
            return value[: -len(suffix)] if suffix else value
        return self.display_name

    @property
    def alias_catalog_names(self) -> tuple[str, ...]:
        values: list[str] = []
        for raw_value in self.alias_relative_paths:
            value = str(raw_value or "").replace("\\", "/").strip("/")
            if not value:
                continue
            suffix = Path(value).suffix
            values.append(value[: -len(suffix)] if suffix else value)
        return tuple(values)

    @property
    def all_paths(self) -> tuple[str, ...]:
        return (self.path, *self.alias_paths)

    @property
    def physical_instance_count(self) -> int:
        return 1 + len(self.alias_paths)

    def with_cache_status(self, value: str) -> "UpscalerDescriptor":
        return replace(self, scan_cache_status=str(value or "unknown"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "upscaler_id": self.upscaler_id,
            "display_name": self.display_name,
            "path": self.path,
            "file_name": self.file_name,
            "sha256": self.sha256,
            "file_size_bytes": int(self.file_size_bytes),
            "modified_time_ns": int(self.modified_time_ns),
            "architecture": self.architecture,
            "architecture_confidence": self.architecture_confidence,
            "native_scale": int(self.native_scale),
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "supports_half": bool(self.supports_half),
            "supports_bfloat16": bool(self.supports_bfloat16),
            "tile_supported": bool(self.tile_supported),
            "load_status": self.load_status,
            "scan_cache_status": self.scan_cache_status,
            "loader_backend": self.loader_backend,
            "compatibility_notes": list(self.compatibility_notes),
            "bounded_error": self.bounded_error,
            "source_root": self.source_root,
            "relative_path": self.relative_path,
            "catalog_name": self.catalog_name,
            "nested": self.nested,
            "alias_paths": list(self.alias_paths),
            "alias_relative_paths": list(self.alias_relative_paths),
            "alias_catalog_names": list(self.alias_catalog_names),
            "physical_instance_count": int(self.physical_instance_count),
            "all_paths": list(self.all_paths),
            "disposition": self.disposition,
            "deferred": self.deferred,
            "failed": self.failed,
            "selectable": self.selectable,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UpscalerDescriptor":
        payload = dict(value)
        return cls(
            upscaler_id=str(payload.get("upscaler_id") or ""),
            display_name=str(payload.get("display_name") or ""),
            path=str(payload.get("path") or ""),
            file_name=str(payload.get("file_name") or ""),
            sha256=str(payload.get("sha256") or ""),
            file_size_bytes=int(payload.get("file_size_bytes") or 0),
            modified_time_ns=int(payload.get("modified_time_ns") or 0),
            architecture=str(payload.get("architecture") or "unclassified"),
            architecture_confidence=str(payload.get("architecture_confidence") or "none"),
            native_scale=int(payload.get("native_scale") or 0),
            input_channels=int(payload.get("input_channels") or 0),
            output_channels=int(payload.get("output_channels") or 0),
            supports_half=bool(payload.get("supports_half", False)),
            supports_bfloat16=bool(payload.get("supports_bfloat16", False)),
            tile_supported=bool(payload.get("tile_supported", False)),
            load_status=str(payload.get("load_status") or "unclassified"),
            scan_cache_status=str(payload.get("scan_cache_status") or "unknown"),
            loader_backend=str(payload.get("loader_backend") or "spandrel"),
            compatibility_notes=_string_tuple(payload.get("compatibility_notes")),
            bounded_error=str(payload.get("bounded_error") or ""),
            source_root=str(payload.get("source_root") or ""),
            relative_path=str(payload.get("relative_path") or ""),
            alias_paths=_string_tuple(payload.get("alias_paths")),
            alias_relative_paths=_string_tuple(payload.get("alias_relative_paths")),
        )


@dataclass(frozen=True)
class UpscalerDiscoveryDiagnostic:
    severity: str
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class UpscalerDiscoveryResult:
    mode: str
    roots: tuple[str, ...]
    built_in_latent: tuple[BuiltinLatentUpscaler, ...]
    neural_descriptors: tuple[UpscalerDescriptor, ...]
    diagnostics: tuple[UpscalerDiscoveryDiagnostic, ...]
    cache_path: str
    recursive: bool = True
    flattened_catalog: bool = True

    @property
    def supported_neural(self) -> tuple[UpscalerDescriptor, ...]:
        return tuple(item for item in self.neural_descriptors if item.selectable)

    @property
    def unavailable_neural(self) -> tuple[UpscalerDescriptor, ...]:
        return tuple(item for item in self.neural_descriptors if not item.selectable)

    @property
    def deferred_neural(self) -> tuple[UpscalerDescriptor, ...]:
        return tuple(item for item in self.neural_descriptors if item.deferred)

    @property
    def failed_neural(self) -> tuple[UpscalerDescriptor, ...]:
        return tuple(item for item in self.neural_descriptors if item.failed)

    def descriptor_by_id(self, upscaler_id: str) -> UpscalerDescriptor | None:
        selected = str(upscaler_id or "").strip()
        for descriptor in self.neural_descriptors:
            if descriptor.upscaler_id == selected:
                return descriptor
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "roots": list(self.roots),
            "cache_path": self.cache_path,
            "recursive": bool(self.recursive),
            "flattened_catalog": bool(self.flattened_catalog),
            "nested_neural_count": sum(1 for item in self.neural_descriptors if item.nested),
            "built_in_latent": [item.to_dict() for item in self.built_in_latent],
            "neural_descriptors": [item.to_dict() for item in self.neural_descriptors],
            "supported_neural_count": len(self.supported_neural),
            "unavailable_neural_count": len(self.unavailable_neural),
            "deferred_neural_count": len(self.deferred_neural),
            "failed_neural_count": len(self.failed_neural),
            "physical_neural_instance_count": sum(item.physical_instance_count for item in self.neural_descriptors),
            "duplicate_content_alias_count": sum(len(item.alias_paths) for item in self.neural_descriptors),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def descriptor_path(descriptor: UpscalerDescriptor) -> Path:
    return Path(descriptor.path)


@dataclass(frozen=True)
class UpscalerRuntimeQualification:
    upscaler_id: str
    descriptor_sha256: str
    status: str
    loader_backend: str
    loader_backend_version: str
    architecture_id: str = ""
    native_scale: int = 0
    input_channels: int = 0
    output_channels: int = 0
    supports_half: bool = False
    supports_bfloat16: bool = False
    device: str = "cpu"
    dtype: str = "torch.float32"
    load_duration_ms: float = 0.0
    qualified_at_utc: str = ""
    bounded_error: str = ""

    def __post_init__(self) -> None:
        normalized = str(self.status or "unqualified").strip().casefold()
        if normalized not in SUPPORTED_RUNTIME_QUALIFICATION_STATES:
            raise ValueError(
                f"Unsupported upscaler runtime qualification status: {self.status!r}"
            )
        object.__setattr__(self, "status", normalized)

    @property
    def qualified(self) -> bool:
        return self.status in {"qualified_cpu", "qualified_cuda"}

    @property
    def disposition(self) -> str:
        return runtime_disposition(self.status)

    @property
    def deferred(self) -> bool:
        return self.disposition == "deferred"

    def to_dict(self) -> dict[str, Any]:
        return {
            "upscaler_id": self.upscaler_id,
            "descriptor_sha256": self.descriptor_sha256,
            "status": self.status,
            "qualified": self.qualified,
            "disposition": self.disposition,
            "deferred": self.deferred,
            "loader_backend": self.loader_backend,
            "loader_backend_version": self.loader_backend_version,
            "architecture_id": self.architecture_id,
            "native_scale": int(self.native_scale),
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "supports_half": bool(self.supports_half),
            "supports_bfloat16": bool(self.supports_bfloat16),
            "device": self.device,
            "dtype": self.dtype,
            "load_duration_ms": float(self.load_duration_ms),
            "qualified_at_utc": self.qualified_at_utc,
            "bounded_error": self.bounded_error,
        }



SUPPORTED_UPSCALE_DTYPE_POLICIES = frozenset({
    "auto",
    "fp32",
    "fp16_if_qualified",
    "bf16_if_qualified",
})
SUPPORTED_UPSCALE_DEVICE_POLICIES = frozenset({"auto", "cpu", "cuda"})
SUPPORTED_EXACT_RESIZE_FILTERS = frozenset({"nearest", "bilinear", "bicubic", "area"})


@dataclass(frozen=True)
class UpscaleRequest:
    source_images: Any
    upscaler_id: str
    target_width: int = 0
    target_height: int = 0
    scale: float | None = None
    tile_size: int = 0
    tile_overlap: int = 16
    tile_batch_size: int = 1
    dtype_policy: str = "auto"
    device_policy: str = "auto"
    exact_resize_filter: str = "bicubic"
    allow_tiling: bool = True
    allow_oom_retry: bool = True
    minimum_retry_tile_size: int = 64
    host_transfer_non_blocking: bool = False

    def normalized(self) -> "UpscaleRequest":
        selected_id = str(self.upscaler_id or "").strip()
        if not selected_id:
            raise ValueError("An upscaler_id is required.")
        target_width = int(self.target_width or 0)
        target_height = int(self.target_height or 0)
        scale = None if self.scale is None else float(self.scale)
        if target_width <= 0 or target_height <= 0:
            if scale is None or scale <= 0:
                raise ValueError(
                    "Either explicit target_width/target_height or a positive scale is required."
                )
        tile_size = int(self.tile_size or 0)
        tile_overlap = int(self.tile_overlap or 0)
        tile_batch_size = int(self.tile_batch_size or 1)
        dtype_policy = str(self.dtype_policy or "auto").strip().casefold()
        device_policy = str(self.device_policy or "auto").strip().casefold()
        exact_resize_filter = str(self.exact_resize_filter or "bicubic").strip().casefold()
        if dtype_policy not in SUPPORTED_UPSCALE_DTYPE_POLICIES:
            raise ValueError(
                f"Unsupported upscaler dtype_policy: {self.dtype_policy!r}."
            )
        if device_policy not in SUPPORTED_UPSCALE_DEVICE_POLICIES:
            raise ValueError(
                f"Unsupported upscaler device_policy: {self.device_policy!r}."
            )
        if exact_resize_filter not in SUPPORTED_EXACT_RESIZE_FILTERS:
            raise ValueError(
                f"Unsupported upscaler exact_resize_filter: {self.exact_resize_filter!r}."
            )
        if tile_size < 0:
            raise ValueError("Upscaler tile_size must be zero or positive.")
        if tile_overlap < 0:
            raise ValueError("Upscaler tile_overlap must be non-negative.")
        if tile_size > 0 and tile_overlap >= tile_size:
            raise ValueError("Upscaler tile_overlap must be smaller than tile_size.")
        if tile_batch_size <= 0:
            raise ValueError("Upscaler tile_batch_size must be positive.")
        minimum_retry_tile_size = max(1, int(self.minimum_retry_tile_size or 64))
        return UpscaleRequest(
            source_images=self.source_images,
            upscaler_id=selected_id,
            target_width=target_width,
            target_height=target_height,
            scale=scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            dtype_policy=dtype_policy,
            device_policy=device_policy,
            exact_resize_filter=exact_resize_filter,
            allow_tiling=bool(self.allow_tiling),
            allow_oom_retry=bool(self.allow_oom_retry),
            minimum_retry_tile_size=minimum_retry_tile_size,
            host_transfer_non_blocking=bool(self.host_transfer_non_blocking),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "upscaler_id": normalized.upscaler_id,
            "target_width": int(normalized.target_width),
            "target_height": int(normalized.target_height),
            "scale": normalized.scale,
            "tile_size": int(normalized.tile_size),
            "tile_overlap": int(normalized.tile_overlap),
            "tile_batch_size": int(normalized.tile_batch_size),
            "dtype_policy": normalized.dtype_policy,
            "device_policy": normalized.device_policy,
            "exact_resize_filter": normalized.exact_resize_filter,
            "allow_tiling": bool(normalized.allow_tiling),
            "allow_oom_retry": bool(normalized.allow_oom_retry),
            "minimum_retry_tile_size": int(normalized.minimum_retry_tile_size),
            "host_transfer_non_blocking": bool(normalized.host_transfer_non_blocking),
        }


@dataclass(frozen=True)
class UpscaleProgress:
    completed_tiles: int
    total_tiles: int
    batch_index: int = 0
    batch_size: int = 0
    tile_index: int = 0
    tile_coordinates: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_tiles": int(self.completed_tiles),
            "total_tiles": int(self.total_tiles),
            "batch_index": int(self.batch_index),
            "batch_size": int(self.batch_size),
            "tile_index": int(self.tile_index),
            "tile_coordinates": [int(value) for value in self.tile_coordinates],
        }


@dataclass(frozen=True)
class UpscaleResult:
    images: Any
    metadata: Mapping[str, Any]


StandaloneUpscaleRequest = UpscaleRequest
