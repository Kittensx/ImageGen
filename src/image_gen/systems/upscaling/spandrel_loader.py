from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from image_gen.systems.upscaling.classifier import (
    CANONICAL_LOADER_BACKEND,
    extract_tensor_state_dict,
)
from image_gen.systems.upscaling.contracts import (
    UpscalerDescriptor,
    UpscalerRuntimeQualification,
)
from image_gen.systems.upscaling.diagnostics import bounded_error_text
from image_gen.systems.upscaling.discovery import sha256_file
from image_gen.systems.upscaling.tiling import RuntimeModelMetadata, normalize_spandrel_metadata

QUALIFIED_SPANDREL_VERSION = "0.4.2"
QUALIFIED_DESCRIPTOR_ARCHITECTURES = frozenset(
    {
        "esrgan_rrdbnet",
        "realesrgan_rrdbnet",
        "realesrgan_srvggnetcompact",
    }
)
QUALIFIED_SPANDREL_ARCHITECTURE_IDS: dict[str, frozenset[str]] = {
    "esrgan_rrdbnet": frozenset({"esrgan"}),
    "realesrgan_rrdbnet": frozenset({"esrgan"}),
    "realesrgan_srvggnetcompact": frozenset(
        {
            "realesrgancompact",
            "realesrgan_compact",
            "srvggnetcompact",
            "srvggnet_compact",
        }
    ),
}


class UpscalerRuntimeLoadError(RuntimeError):
    def __init__(self, message: str, *, status: str = "load_failed") -> None:
        super().__init__(bounded_error_text(message))
        self.status = str(status or "load_failed")


class SpandrelSecurityBoundaryError(UpscalerRuntimeLoadError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status="backend_unqualified")


class UpscalerHashMismatchError(UpscalerRuntimeLoadError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status="hash_mismatch")


class UpscalerMetadataMismatchError(UpscalerRuntimeLoadError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status="metadata_mismatch")


@dataclass(frozen=True)
class SpandrelRuntimeAudit:
    installed: bool
    version: str
    qualified_version: bool
    model_loader_available: bool
    state_dict_loader_available: bool
    image_descriptor_available: bool
    safe_state_dict_boundary: bool
    bounded_error: str = ""

    @property
    def qualified(self) -> bool:
        return bool(
            self.installed
            and self.qualified_version
            and self.model_loader_available
            and self.state_dict_loader_available
            and self.image_descriptor_available
            and self.safe_state_dict_boundary
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "qualified_version": self.qualified_version,
            "model_loader_available": self.model_loader_available,
            "state_dict_loader_available": self.state_dict_loader_available,
            "image_descriptor_available": self.image_descriptor_available,
            "safe_state_dict_boundary": self.safe_state_dict_boundary,
            "qualified": self.qualified,
            "bounded_error": self.bounded_error,
        }


@dataclass
class LoadedSpandrelUpscaler:
    descriptor: UpscalerDescriptor
    model_descriptor: Any
    qualification: UpscalerRuntimeQualification

    @property
    def module(self) -> torch.nn.Module:
        module = getattr(self.model_descriptor, "model", None)
        if not isinstance(module, torch.nn.Module):
            raise UpscalerRuntimeLoadError(
                "Spandrel returned an image descriptor without a torch.nn.Module model."
            )
        return module

    @property
    def architecture_id(self) -> str:
        architecture = getattr(self.model_descriptor, "architecture", None)
        return str(getattr(architecture, "id", "") or "")

    def eval(self) -> "LoadedSpandrelUpscaler":
        evaluator = getattr(self.model_descriptor, "eval", None)
        if callable(evaluator):
            evaluator()
        else:
            self.module.eval()
        return self

    def to(
        self,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "LoadedSpandrelUpscaler":
        mover = getattr(self.model_descriptor, "to", None)
        if callable(mover):
            mover(device=device, dtype=dtype)
        else:
            self.module.to(device=device, dtype=dtype)
        return self



@dataclass
class SpandrelCapabilityProbe:
    descriptor: UpscalerDescriptor
    model_descriptor: Any
    runtime_metadata: RuntimeModelMetadata
    loader_backend_version: str
    load_duration_ms: float

    @property
    def module(self) -> torch.nn.Module:
        module = getattr(self.model_descriptor, "model", None)
        if not isinstance(module, torch.nn.Module):
            raise UpscalerRuntimeLoadError(
                "Spandrel returned an image descriptor without a torch.nn.Module model."
            )
        return module

    def to_dict(self) -> dict[str, Any]:
        return {
            "upscaler_id": self.descriptor.upscaler_id,
            "descriptor_sha256": self.descriptor.sha256,
            "loader_backend": CANONICAL_LOADER_BACKEND,
            "loader_backend_version": self.loader_backend_version,
            "load_duration_ms": float(self.load_duration_ms),
            "runtime_metadata": self.runtime_metadata.to_dict(),
        }



def _normalize_architecture_id(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum() or character == "_")


def _package_version(spandrel_module: Any) -> str:
    direct = str(getattr(spandrel_module, "__version__", "") or "").strip()
    if direct:
        return direct
    try:
        return importlib.metadata.version("spandrel")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def audit_spandrel_loading_path(
    *,
    spandrel_module: Any | None = None,
) -> SpandrelRuntimeAudit:
    try:
        module = spandrel_module or importlib.import_module("spandrel")
    except Exception as exc:
        return SpandrelRuntimeAudit(
            installed=False,
            version="not-installed",
            qualified_version=False,
            model_loader_available=False,
            state_dict_loader_available=False,
            image_descriptor_available=False,
            safe_state_dict_boundary=False,
            bounded_error=bounded_error_text(exc),
        )

    version = _package_version(module)
    model_loader = getattr(module, "ModelLoader", None)
    image_descriptor = getattr(module, "ImageModelDescriptor", None)
    state_dict_loader = getattr(model_loader, "load_from_state_dict", None)
    state_dict_loader_available = callable(state_dict_loader)
    safe_boundary = False
    error = ""
    if state_dict_loader_available:
        try:
            signature = inspect.signature(state_dict_loader)
            safe_boundary = "state_dict" in signature.parameters
            if not safe_boundary:
                error = "Spandrel load_from_state_dict does not expose the expected state_dict parameter."
        except (TypeError, ValueError) as exc:
            error = bounded_error_text(exc)
    else:
        error = "Spandrel ModelLoader.load_from_state_dict is unavailable."

    return SpandrelRuntimeAudit(
        installed=True,
        version=version,
        qualified_version=version == QUALIFIED_SPANDREL_VERSION,
        model_loader_available=model_loader is not None,
        state_dict_loader_available=state_dict_loader_available,
        image_descriptor_available=image_descriptor is not None,
        safe_state_dict_boundary=safe_boundary,
        bounded_error=bounded_error_text(error),
    )


def _load_weights_only_state_dict(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise SpandrelSecurityBoundaryError(
            "The active PyTorch build does not support torch.load(weights_only=True); "
            "unsafe fallback is forbidden."
        ) from exc
    except Exception as exc:
        raise UpscalerRuntimeLoadError(
            f"Tensor-only upscaler loading failed: {type(exc).__name__}: {exc}"
        ) from exc

    state_dict = extract_tensor_state_dict(payload)
    if state_dict is None:
        raise UpscalerRuntimeLoadError(
            "The upscaler file did not contain a supported tensor state dictionary."
        )
    if not state_dict:
        raise UpscalerRuntimeLoadError("The upscaler state dictionary was empty.")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise SpandrelSecurityBoundaryError(
            "The extracted upscaler state dictionary contained a non-tensor value."
        )
    return state_dict


def _validate_descriptor_for_runtime(descriptor: UpscalerDescriptor) -> None:
    if not descriptor.selectable:
        raise UpscalerRuntimeLoadError(
            f"Upscaler {descriptor.upscaler_id!r} is recognized but not selectable: "
            f"{descriptor.load_status}."
        )
    if descriptor.loader_backend != CANONICAL_LOADER_BACKEND:
        raise UpscalerRuntimeLoadError(
            f"Unsupported upscaler loader backend: {descriptor.loader_backend!r}."
        )
    if descriptor.architecture not in QUALIFIED_DESCRIPTOR_ARCHITECTURES:
        raise UpscalerRuntimeLoadError(
            f"Architecture {descriptor.architecture!r} is outside the qualified Phase 14N-2 scope."
        )
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise UpscalerMetadataMismatchError(
            "Phase 14N-2 requires three-channel RGB input and output."
        )


def _validate_spandrel_metadata(
    descriptor: UpscalerDescriptor,
    model_descriptor: Any,
    *,
    image_descriptor_type: type[Any],
) -> tuple[str, int, int, int, bool, bool]:
    if not isinstance(model_descriptor, image_descriptor_type):
        raise UpscalerMetadataMismatchError(
            "Spandrel did not return an ImageModelDescriptor for the selected upscaler."
        )

    architecture = getattr(model_descriptor, "architecture", None)
    architecture_id = str(getattr(architecture, "id", "") or "")
    normalized_id = _normalize_architecture_id(architecture_id)
    allowed_ids = QUALIFIED_SPANDREL_ARCHITECTURE_IDS.get(descriptor.architecture, frozenset())
    if normalized_id not in allowed_ids:
        raise UpscalerMetadataMismatchError(
            f"Spandrel detected architecture {architecture_id!r}, which does not match "
            f"the Phase 14N-1 classification {descriptor.architecture!r}."
        )

    native_scale = int(getattr(model_descriptor, "scale", 0) or 0)
    input_channels = int(getattr(model_descriptor, "input_channels", 0) or 0)
    output_channels = int(getattr(model_descriptor, "output_channels", 0) or 0)
    if native_scale != int(descriptor.native_scale):
        raise UpscalerMetadataMismatchError(
            f"Native scale mismatch: discovery={descriptor.native_scale}x, "
            f"Spandrel={native_scale}x."
        )
    if input_channels != int(descriptor.input_channels) or output_channels != int(
        descriptor.output_channels
    ):
        raise UpscalerMetadataMismatchError(
            "Channel metadata mismatch between discovery and Spandrel runtime loading."
        )
    purpose = str(getattr(model_descriptor, "purpose", "") or "")
    if purpose not in {"SR", "FaceSR"}:
        raise UpscalerMetadataMismatchError(
            f"Spandrel reported unsupported model purpose {purpose!r}; super-resolution is required."
        )

    return (
        architecture_id,
        native_scale,
        input_channels,
        output_channels,
        bool(getattr(model_descriptor, "supports_half", False)),
        bool(getattr(model_descriptor, "supports_bfloat16", False)),
    )


def probe_spandrel_upscaler(
    descriptor: UpscalerDescriptor,
    *,
    spandrel_module: Any | None = None,
    require_qualified_version: bool = True,
) -> SpandrelCapabilityProbe:
    """Safely construct any Spandrel-recognized image model for capability qualification.

    This function deliberately does not apply the Phase 14N-2 architecture whitelist.
    It retains the same hash verification and ``weights_only=True`` boundary, then
    records metadata without promoting the model into the production registry.
    """

    if descriptor.loader_backend != CANONICAL_LOADER_BACKEND:
        raise UpscalerRuntimeLoadError(
            f"Unsupported upscaler loader backend: {descriptor.loader_backend!r}."
        )
    audit = audit_spandrel_loading_path(spandrel_module=spandrel_module)
    if not audit.installed:
        raise UpscalerRuntimeLoadError(
            audit.bounded_error or "Spandrel is not installed.",
            status="backend_unavailable",
        )
    if not audit.safe_state_dict_boundary:
        raise SpandrelSecurityBoundaryError(
            audit.bounded_error
            or "Spandrel does not expose a qualified state-dictionary-only loading boundary."
        )
    if require_qualified_version and not audit.qualified_version:
        raise SpandrelSecurityBoundaryError(
            f"Spandrel {audit.version!r} is not the qualified version "
            f"{QUALIFIED_SPANDREL_VERSION!r}."
        )

    path = Path(descriptor.path).expanduser().resolve()
    if not path.is_file():
        raise UpscalerRuntimeLoadError(f"Upscaler file does not exist: {path}")
    current_hash = sha256_file(path)
    if current_hash.casefold() != descriptor.sha256.casefold():
        raise UpscalerHashMismatchError(
            "The upscaler file hash changed after discovery; rescan before qualification."
        )

    pre_load_stat = path.stat()
    pre_load_identity = (
        int(pre_load_stat.st_size),
        int(pre_load_stat.st_mtime_ns),
        int(getattr(pre_load_stat, "st_dev", 0)),
        int(getattr(pre_load_stat, "st_ino", 0)),
    )
    started = time.perf_counter()
    state_dict = _load_weights_only_state_dict(path)
    post_load_stat = path.stat()
    post_load_identity = (
        int(post_load_stat.st_size),
        int(post_load_stat.st_mtime_ns),
        int(getattr(post_load_stat, "st_dev", 0)),
        int(getattr(post_load_stat, "st_ino", 0)),
    )
    if post_load_identity != pre_load_identity:
        state_dict.clear()
        raise UpscalerHashMismatchError(
            "The upscaler file changed while it was being loaded; rescan before retrying."
        )

    module = spandrel_module or importlib.import_module("spandrel")
    try:
        loader = module.ModelLoader(device="cpu")
        model_descriptor = loader.load_from_state_dict(state_dict)
    except Exception as exc:
        raise UpscalerRuntimeLoadError(
            f"Spandrel state-dictionary capability probe failed: {type(exc).__name__}: {exc}",
            status="deferred_backend_support",
        ) from exc
    finally:
        state_dict.clear()

    if not isinstance(model_descriptor, module.ImageModelDescriptor):
        raise UpscalerMetadataMismatchError(
            "Spandrel did not return an ImageModelDescriptor during capability qualification."
        )
    runtime_metadata = normalize_spandrel_metadata(model_descriptor)
    if runtime_metadata.input_channels <= 0 or runtime_metadata.output_channels <= 0:
        raise UpscalerMetadataMismatchError(
            "Spandrel capability metadata did not report valid input/output channels."
        )
    evaluator = getattr(model_descriptor, "eval", None)
    if callable(evaluator):
        evaluator()
    else:
        model_descriptor.model.eval()
    return SpandrelCapabilityProbe(
        descriptor=descriptor,
        model_descriptor=model_descriptor,
        runtime_metadata=runtime_metadata,
        loader_backend_version=audit.version,
        load_duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def load_spandrel_upscaler(
    descriptor: UpscalerDescriptor,
    *,
    spandrel_module: Any | None = None,
    require_qualified_version: bool = True,
) -> LoadedSpandrelUpscaler:
    _validate_descriptor_for_runtime(descriptor)
    audit = audit_spandrel_loading_path(spandrel_module=spandrel_module)
    if not audit.installed:
        raise UpscalerRuntimeLoadError(
            audit.bounded_error or "Spandrel is not installed.",
            status="backend_unavailable",
        )
    if not audit.safe_state_dict_boundary:
        raise SpandrelSecurityBoundaryError(
            audit.bounded_error
            or "Spandrel does not expose a qualified state-dictionary-only loading boundary."
        )
    if require_qualified_version and not audit.qualified_version:
        raise SpandrelSecurityBoundaryError(
            f"Spandrel {audit.version!r} is not the qualified version "
            f"{QUALIFIED_SPANDREL_VERSION!r}."
        )

    path = Path(descriptor.path).expanduser().resolve()
    if not path.is_file():
        raise UpscalerRuntimeLoadError(f"Upscaler file does not exist: {path}")

    current_hash = sha256_file(path)
    if current_hash.casefold() != descriptor.sha256.casefold():
        raise UpscalerHashMismatchError(
            "The upscaler file hash changed after discovery; rescan before loading or replay."
        )

    # Capture file identity after the immediate pre-load hash.  This catches a
    # replacement or rewrite during deserialization without weakening the
    # weights-only boundary or performing a second expensive full-file hash.
    pre_load_stat = path.stat()
    pre_load_identity = (
        int(pre_load_stat.st_size),
        int(pre_load_stat.st_mtime_ns),
        int(getattr(pre_load_stat, "st_dev", 0)),
        int(getattr(pre_load_stat, "st_ino", 0)),
    )

    started = time.perf_counter()
    state_dict = _load_weights_only_state_dict(path)
    post_load_stat = path.stat()
    post_load_identity = (
        int(post_load_stat.st_size),
        int(post_load_stat.st_mtime_ns),
        int(getattr(post_load_stat, "st_dev", 0)),
        int(getattr(post_load_stat, "st_ino", 0)),
    )
    if post_load_identity != pre_load_identity:
        state_dict.clear()
        raise UpscalerHashMismatchError(
            "The upscaler file changed while it was being loaded; rescan before retrying."
        )
    module = spandrel_module or importlib.import_module("spandrel")
    try:
        loader = module.ModelLoader(device="cpu")
        model_descriptor = loader.load_from_state_dict(state_dict)
    except Exception as exc:
        raise UpscalerRuntimeLoadError(
            f"Spandrel state-dictionary loading failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        state_dict.clear()

    (
        architecture_id,
        native_scale,
        input_channels,
        output_channels,
        supports_half,
        supports_bfloat16,
    ) = _validate_spandrel_metadata(
        descriptor,
        model_descriptor,
        image_descriptor_type=module.ImageModelDescriptor,
    )
    loaded = LoadedSpandrelUpscaler(
        descriptor=descriptor,
        model_descriptor=model_descriptor,
        qualification=UpscalerRuntimeQualification(
            upscaler_id=descriptor.upscaler_id,
            descriptor_sha256=descriptor.sha256,
            status="qualified_cpu",
            loader_backend=CANONICAL_LOADER_BACKEND,
            loader_backend_version=audit.version,
            architecture_id=architecture_id,
            native_scale=native_scale,
            input_channels=input_channels,
            output_channels=output_channels,
            supports_half=supports_half,
            supports_bfloat16=supports_bfloat16,
            device="cpu",
            dtype=str(next(model_descriptor.model.parameters()).dtype),
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            qualified_at_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    loaded.eval()
    return loaded
