from __future__ import annotations

import gc
import threading
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import torch

from image_gen.systems.upscaling.contracts import (
    BUILTIN_LATENT_UPSCALERS,
    BuiltinLatentUpscaler,
    UpscalerDescriptor,
    UpscalerDiscoveryResult,
    UpscalerRuntimeQualification,
)
from image_gen.systems.upscaling.diagnostics import bounded_error_text
from image_gen.systems.upscaling.spandrel_loader import (
    LoadedSpandrelUpscaler,
    UpscalerRuntimeLoadError,
    audit_spandrel_loading_path,
    load_spandrel_upscaler,
)

UPSCALER_COMPONENT_ID = "upscaler"
UPSCALER_COMPONENT_KIND = "neural_upscaler"
UPSCALER_STAGE_ID = "neural_upscale"


class UpscalerRegistryError(RuntimeError):
    pass


class UpscalerModelLease(AbstractContextManager[LoadedSpandrelUpscaler]):
    def __init__(
        self,
        registry: "UpscalerModelRegistry",
        *,
        upscaler_id: str,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        self.registry = registry
        self.upscaler_id = str(upscaler_id)
        self.device = device
        self.dtype = dtype
        self.loaded: LoadedSpandrelUpscaler | None = None
        self._component_lease: AbstractContextManager[Any] | None = None

    def __enter__(self) -> LoadedSpandrelUpscaler:
        loaded, component_lease = self.registry._acquire(
            self.upscaler_id,
            device=self.device,
            dtype=self.dtype,
        )
        self.loaded = loaded
        self._component_lease = component_lease
        return loaded

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if self._component_lease is not None:
                self._component_lease.__exit__(exc_type, exc, traceback)
        finally:
            self.registry._release(self.upscaler_id)
            self.loaded = None
            self._component_lease = None
        return False


class UpscalerModelRegistry:
    """Stable-ID registry and scoped runtime owner for neural upscalers.

    Recognition remains a Phase 14N-1 descriptor property. Runtime qualification
    is recorded separately and only changes after the exact content hash has
    passed the qualified Spandrel state-dictionary loading path.
    """

    def __init__(
        self,
        *,
        neural_descriptors: Iterable[UpscalerDescriptor] = (),
        built_in_latent: Iterable[BuiltinLatentUpscaler] = BUILTIN_LATENT_UPSCALERS,
        memory_manager: Any | None = None,
        loader: Callable[[UpscalerDescriptor], LoadedSpandrelUpscaler] | None = None,
    ) -> None:
        latent_items = tuple(built_in_latent)
        descriptor_items = tuple(neural_descriptors)
        self._built_in = {item.upscaler_id: item for item in latent_items}
        self._descriptors = {item.upscaler_id: item for item in descriptor_items}
        if len(self._built_in) != len(latent_items):
            raise UpscalerRegistryError("Duplicate built-in upscaler IDs are not allowed.")
        if len(self._descriptors) != len(descriptor_items):
            raise UpscalerRegistryError("Duplicate neural upscaler IDs are not allowed.")
        overlap = set(self._built_in).intersection(self._descriptors)
        if overlap:
            raise UpscalerRegistryError(
                f"Built-in and neural upscaler IDs overlap: {sorted(overlap)!r}"
            )

        self.memory_manager = memory_manager
        self._loader = loader or load_spandrel_upscaler
        self._lock = threading.RLock()
        self._loaded_id: str | None = None
        self._loaded: LoadedSpandrelUpscaler | None = None
        self._active_leases: dict[str, int] = {}
        self._last_errors: dict[str, str] = {}
        self._qualifications: dict[str, UpscalerRuntimeQualification] = {
            descriptor.upscaler_id: self._unqualified(descriptor)
            for descriptor in self._descriptors.values()
        }

    @classmethod
    def from_discovery(
        cls,
        result: UpscalerDiscoveryResult,
        *,
        memory_manager: Any | None = None,
        loader: Callable[[UpscalerDescriptor], LoadedSpandrelUpscaler] | None = None,
    ) -> "UpscalerModelRegistry":
        return cls(
            neural_descriptors=result.neural_descriptors,
            built_in_latent=result.built_in_latent,
            memory_manager=memory_manager,
            loader=loader,
        )

    @staticmethod
    def _unqualified(descriptor: UpscalerDescriptor) -> UpscalerRuntimeQualification:
        audit = audit_spandrel_loading_path()
        return UpscalerRuntimeQualification(
            upscaler_id=descriptor.upscaler_id,
            descriptor_sha256=descriptor.sha256,
            status="unqualified",
            loader_backend=descriptor.loader_backend,
            loader_backend_version=audit.version,
            native_scale=descriptor.native_scale,
            input_channels=descriptor.input_channels,
            output_channels=descriptor.output_channels,
            supports_half=descriptor.supports_half,
            supports_bfloat16=descriptor.supports_bfloat16,
        )

    @property
    def built_in_latent(self) -> tuple[BuiltinLatentUpscaler, ...]:
        return tuple(self._built_in[key] for key in sorted(self._built_in))

    @property
    def neural_descriptors(self) -> tuple[UpscalerDescriptor, ...]:
        return tuple(
            sorted(
                self._descriptors.values(),
                key=lambda item: (item.display_name.casefold(), item.sha256, item.path),
            )
        )

    def resolve(self, upscaler_id: str) -> BuiltinLatentUpscaler | UpscalerDescriptor:
        selected = str(upscaler_id or "").strip()
        if selected in self._built_in:
            return self._built_in[selected]
        descriptor = self._descriptors.get(selected)
        if descriptor is None:
            raise KeyError(f"Unknown upscaler ID: {selected!r}")
        return descriptor

    def resolve_neural(self, upscaler_id: str) -> UpscalerDescriptor:
        resolved = self.resolve(upscaler_id)
        if isinstance(resolved, BuiltinLatentUpscaler):
            raise UpscalerRegistryError(
                f"Built-in interpolation entry {resolved.upscaler_id!r} has no neural model."
            )
        return resolved

    def qualification(self, upscaler_id: str) -> UpscalerRuntimeQualification:
        descriptor = self.resolve_neural(upscaler_id)
        return self._qualifications[descriptor.upscaler_id]

    def lease(
        self,
        upscaler_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> UpscalerModelLease:
        return UpscalerModelLease(
            self,
            upscaler_id=upscaler_id,
            device=device,
            dtype=dtype,
        )

    def _load_locked(self, descriptor: UpscalerDescriptor) -> LoadedSpandrelUpscaler:
        if self._loaded_id == descriptor.upscaler_id and self._loaded is not None:
            return self._loaded
        if self._loaded_id is not None:
            active = self._active_leases.get(self._loaded_id, 0)
            if active:
                raise UpscalerRegistryError(
                    "A different neural upscaler cannot replace the active scoped component "
                    "while its lease is held."
                )
            self._unload_locked(self._loaded_id)

        try:
            loaded = self._loader(descriptor)
        except UpscalerRuntimeLoadError as exc:
            self._record_error(descriptor, exc.status, str(exc))
            raise
        except Exception as exc:
            error = bounded_error_text(f"{type(exc).__name__}: {exc}")
            self._record_error(descriptor, "load_failed", error)
            raise UpscalerRuntimeLoadError(error) from exc

        if loaded.descriptor.upscaler_id != descriptor.upscaler_id:
            error = "The runtime loader returned a different stable upscaler ID."
            self._record_error(descriptor, "metadata_mismatch", error)
            raise UpscalerRegistryError(error)
        self._loaded_id = descriptor.upscaler_id
        self._loaded = loaded
        self._qualifications[descriptor.upscaler_id] = loaded.qualification
        self._last_errors.pop(descriptor.upscaler_id, None)

        if self.memory_manager is not None:
            try:
                self.memory_manager.register_scoped_component(
                    component_id=UPSCALER_COMPONENT_ID,
                    component_kind=UPSCALER_COMPONENT_KIND,
                    model_identity=descriptor.upscaler_id,
                    module=loaded.module,
                    preferred_dtype=str(next(loaded.module.parameters()).dtype),
                    required_by_stages={UPSCALER_STAGE_ID},
                )
            except Exception as exc:
                with suppress(Exception):
                    self.memory_manager.remove_scoped_component(
                        UPSCALER_COMPONENT_ID,
                        call_unload=False,
                    )
                self._loaded = None
                self._loaded_id = None
                loaded.model_descriptor = None
                error = bounded_error_text(
                    f"Scoped upscaler component registration failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_error(descriptor, "load_failed", error)
                gc.collect()
                raise UpscalerRegistryError(error) from exc
        return loaded

    def _acquire(
        self,
        upscaler_id: str,
        *,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[LoadedSpandrelUpscaler, AbstractContextManager[Any]]:
        descriptor = self.resolve_neural(upscaler_id)
        with self._lock:
            loaded = self._load_locked(descriptor)
            self._active_leases[descriptor.upscaler_id] = (
                self._active_leases.get(descriptor.upscaler_id, 0) + 1
            )

        component_lease: AbstractContextManager[Any] | None = None
        component_lease_entered = False
        try:
            if self.memory_manager is not None:
                target_device = torch.device(
                    device if device is not None else self.memory_manager.target_device
                )
                manager_target = torch.device(self.memory_manager.target_device)
                if target_device.type != manager_target.type:
                    raise UpscalerRegistryError(
                        "A scoped upscaler lease must use the memory manager target device."
                    )
                component_lease = self.memory_manager.lease(
                    stage=UPSCALER_STAGE_ID,
                    required=(UPSCALER_COMPONENT_ID,),
                    requested_profile_override=(
                        "cpu_fallback" if target_device.type == "cpu" else None
                    ),
                )
                component_lease.__enter__()
                component_lease_entered = True
                if dtype is not None:
                    loaded.to(device=target_device, dtype=dtype)
            else:
                target_device = torch.device(device or "cpu")
                loaded.to(device=target_device, dtype=dtype)
                component_lease = nullcontext()
                component_lease.__enter__()
                component_lease_entered = True
            loaded.eval()
            self._require_requested_device(loaded, target_device)
            self._record_device_qualification(loaded)
            if component_lease is None:
                raise UpscalerRegistryError("Failed to create an upscaler component lease.")
            return loaded, component_lease
        except Exception:
            if component_lease is not None and component_lease_entered:
                component_lease.__exit__(None, None, None)
            with self._lock:
                self._active_leases[descriptor.upscaler_id] = max(
                    0, self._active_leases.get(descriptor.upscaler_id, 1) - 1
                )
                if self._active_leases[descriptor.upscaler_id] == 0:
                    self._unload_locked(descriptor.upscaler_id)
            raise

    @staticmethod
    def _require_requested_device(
        loaded: LoadedSpandrelUpscaler,
        target_device: torch.device,
    ) -> None:
        parameter = next(loaded.module.parameters())
        if parameter.device.type != target_device.type:
            raise UpscalerRegistryError(
                f"The upscaler lease requested {target_device.type!r}, but the model "
                f"remained on {parameter.device.type!r}. Runtime qualification was not recorded."
            )
        if target_device.type == "cuda" and target_device.index is not None:
            actual_index = parameter.device.index
            if actual_index is not None and actual_index != target_device.index:
                raise UpscalerRegistryError(
                    f"The upscaler lease requested CUDA device {target_device.index}, "
                    f"but the model was placed on CUDA device {actual_index}."
                )

    def _record_device_qualification(self, loaded: LoadedSpandrelUpscaler) -> None:
        parameter = next(loaded.module.parameters())
        device = str(parameter.device)
        status = "qualified_cuda" if parameter.device.type == "cuda" else "qualified_cpu"
        qualification = replace(
            loaded.qualification,
            status=status,
            device=device,
            dtype=str(parameter.dtype),
            qualified_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        loaded.qualification = qualification
        self._qualifications[loaded.descriptor.upscaler_id] = qualification

    def _release(self, upscaler_id: str) -> None:
        selected = str(upscaler_id)
        with self._lock:
            count = max(0, self._active_leases.get(selected, 0) - 1)
            self._active_leases[selected] = count
            if count == 0:
                self._unload_locked(selected)

    def unload(self, upscaler_id: str | None = None) -> bool:
        with self._lock:
            selected = str(upscaler_id or self._loaded_id or "")
            if not selected or self._loaded_id != selected:
                return False
            if self._active_leases.get(selected, 0):
                raise UpscalerRegistryError(
                    f"Cannot unload leased upscaler {selected!r}."
                )
            self._unload_locked(selected)
            return True

    def _unload_locked(self, upscaler_id: str) -> None:
        if self._loaded_id != upscaler_id:
            return
        if self._active_leases.get(upscaler_id, 0):
            raise UpscalerRegistryError(
                f"Cannot unload leased upscaler {upscaler_id!r}."
            )
        if self.memory_manager is not None:
            self.memory_manager.remove_scoped_component(
                UPSCALER_COMPONENT_ID,
                call_unload=False,
            )
        loaded = self._loaded
        self._loaded = None
        self._loaded_id = None
        if loaded is not None:
            loaded.model_descriptor = None
        gc.collect()

    def _record_error(
        self,
        descriptor: UpscalerDescriptor,
        status: str,
        error: str,
    ) -> None:
        bounded = bounded_error_text(error)
        prior = self._qualifications.get(descriptor.upscaler_id) or self._unqualified(
            descriptor
        )
        self._qualifications[descriptor.upscaler_id] = replace(
            prior,
            status=status,
            bounded_error=bounded,
            qualified_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._last_errors[descriptor.upscaler_id] = bounded

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latent = [item.to_dict() for item in self.built_in_latent]
            neural: list[dict[str, Any]] = []
            for descriptor in self.neural_descriptors:
                qualification = self._qualifications[descriptor.upscaler_id]
                neural.append(
                    {
                        **descriptor.to_dict(),
                        "runtime_qualification": qualification.to_dict(),
                        "runtime_loaded": self._loaded_id == descriptor.upscaler_id,
                        "active_leases": self._active_leases.get(
                            descriptor.upscaler_id, 0
                        ),
                        "last_runtime_error": self._last_errors.get(
                            descriptor.upscaler_id, ""
                        ),
                    }
                )
            return {
                "built_in_latent": latent,
                "neural": neural,
                "loaded_upscaler_id": self._loaded_id,
                "component_id": UPSCALER_COMPONENT_ID,
            }
