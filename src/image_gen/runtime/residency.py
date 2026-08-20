from __future__ import annotations

import gc
import time
from typing import Any

import torch

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.systems.memory.policy import normalize_policy
from modules.component_placement import place_component
from modules.txt2img.request_loader import payload_to_generation_request


class ResidencyMixin:
    @staticmethod
    def _runtime_component_entries(components: Any) -> list[tuple[str, Any]]:
        if components is None:
            return []
        denoiser_kind = str(getattr(components, "denoiser_kind", "unet") or "unet").strip().lower()
        entries: list[tuple[str, Any]] = []
        if denoiser_kind == "transformer":
            entries.append(("transformer", getattr(components, "denoiser", None)))
        else:
            entries.append(("unet", getattr(components, "unet", None)))
        entries.extend([
            ("text_encoder", getattr(components, "text_encoder", None)),
            ("text_encoder_2", getattr(components, "text_encoder_2", None)),
            ("text_encoder_3", getattr(components, "text_encoder_3", None)),
            ("vae", getattr(components, "vae", None)),
        ])
        seen: set[int] = set()
        unique: list[tuple[str, Any]] = []
        for name, module in entries:
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            unique.append((name, module))
        return unique

    def clear_model_cache(self, *, move_components_to_cpu: bool = True) -> dict[str, Any]:
        """Release cached checkpoint components and return before/after telemetry.

        CUDA's process context and third-party allocations may remain visible in system
        monitors, but IMAGE_GEN-owned module parameters are moved to CPU and allocator
        caches are explicitly released before the status response is produced.
        """
        started = time.perf_counter()
        cached_entries = len(self._loaded_model_cache)
        previous = self.resident_model_status()
        cuda_before = dict(previous.get("cuda_memory") or {})
        released_components: list[str] = []
        directly_released_components: list[str] = []
        placement_errors: list[dict[str, str]] = []

        loaded_objects: list[Any] = list(self._loaded_model_cache.values())
        if self.last_loaded_model is not None:
            loaded_objects.append(self.last_loaded_model)
        seen: set[int] = set()
        for loaded in loaded_objects:
            if id(loaded) in seen:
                continue
            seen.add(id(loaded))
            components = getattr(loaded, "components", None)
            if components is None:
                continue
            for name, module in self._runtime_component_entries(components):
                if not callable(getattr(module, "to", None)):
                    continue
                if not move_components_to_cpu:
                    directly_released_components.append(name)
                    continue
                try:
                    module.to(device=torch.device("cpu"))
                    released_components.append(name)
                except Exception as exc:
                    placement_errors.append(
                        {"component": name, "error_type": type(exc).__name__, "error": str(exc)}
                    )

        self._loaded_model_cache.clear()
        self.last_loaded_model = None
        self.lora_runtime_manager.reset()
        gc.collect()
        cuda_cleanup_errors: list[str] = []
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                cuda_cleanup_errors.append(f"synchronize: {type(exc).__name__}: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cuda_cleanup_errors.append(f"empty_cache: {type(exc).__name__}: {exc}")
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception as exc:
                    cuda_cleanup_errors.append(f"ipc_collect: {type(exc).__name__}: {exc}")

        after = self.resident_model_status()
        return {
            "cached_entries_released": cached_entries,
            "previous_model_path": previous.get("model_path"),
            "components_moved_to_cpu": sorted(set(released_components)),
            "components_released_without_cpu_stage": sorted(set(directly_released_components)),
            "move_components_to_cpu": bool(move_components_to_cpu),
            "component_release_errors": placement_errors,
            "cuda_cleanup_errors": cuda_cleanup_errors,
            "cuda_memory_before": cuda_before,
            "cuda_memory_after": dict(after.get("cuda_memory") or {}),
            "note": (
                f"System GPU monitors may still show the CUDA context or non-{PRODUCT_NAME} allocations; "
                f"allocated_bytes is the authoritative {PRODUCT_NAME}/PyTorch tensor allocation value."
            ),
            "unload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def resident_model_status(self) -> dict[str, Any]:
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        report = getattr(getattr(loaded, "load_plan", None), "report", None)
        model_path = str(getattr(report, "model_path", "") or "")
        component_devices: dict[str, str] = {}
        if components is not None:
            for name, module in self._runtime_component_entries(components):
                device = "unknown"
                try:
                    parameter = next(module.parameters())
                    device = str(parameter.device)
                except (StopIteration, AttributeError, TypeError):
                    device = str(getattr(module, "device", "unknown"))
                component_devices[name] = device
        known_component_devices = [
            value for value in component_devices.values() if value and value != "unknown"
        ]
        gpu_loaded = bool(known_component_devices) and all(
            value.startswith("cuda") for value in known_component_devices
        )
        cpu_loaded = any(value.startswith("cpu") for value in known_component_devices)
        cuda_memory = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
            "device_name": None,
        }
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                cuda_memory = {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                }
            except Exception:
                pass
        resident = loaded is not None and bool(self._loaded_model_cache)
        architecture = str(getattr(components, "architecture", "") or "") if components is not None else ""
        staged_runtime = architecture.strip().lower() in {"sd3", "sd3.x", "stable-diffusion-3.x"}
        generation_ready = bool(
            resident
            and known_component_devices
            and all(value != "unknown" and not value.startswith("meta") for value in known_component_devices)
        )
        return {
            "resident": resident,
            "architecture": architecture,
            "staged_runtime": staged_runtime,
            "generation_ready": generation_ready,
            "model_path": model_path or None,
            "model_identity": str(getattr(components, "model_identity", "") or "") if components is not None else "",
            "composition_sha256": (
                str(getattr(components, "model_identity", "") or "").removeprefix("advanced:")
                if components is not None and str(getattr(components, "model_identity", "") or "").startswith("advanced:")
                else ""
            ),
            "runtime_load_variant": dict(getattr(components, "runtime_load_variant", {}) or {}) if components is not None else {},
            "runtime_load_variant_fingerprint": str(getattr(components, "runtime_load_variant_fingerprint", "") or "") if components is not None else "",
            "cache_entries": len(self._loaded_model_cache),
            "cpu_loaded": cpu_loaded,
            "gpu_loaded": gpu_loaded,
            "component_devices": component_devices,
            "cuda_memory": cuda_memory,
        }

    def apply_resident_retention(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Keep the selected checkpoint components resident between jobs."""
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        if components is None:
            return {"applied": False, "reason": "no cached model"}
        values = dict(settings or {})
        architecture = str(getattr(components, "architecture", "") or "").strip().lower()
        staged_sd3 = architecture in {"sd3", "sd3.x", "stable-diffusion-3.x"}
        checkpoint_retain = bool(values.get("memory_retain_checkpoint_between_jobs", True))
        vae_retain = bool(values.get("memory_retain_vae_between_jobs", True))
        text_retain = bool(values.get("model_runtime_retain_text_encoder_between_jobs", True))
        memory_policy = normalize_policy(values.get("memory_policy"))
        advanced_models_enabled = bool(values.get("advanced_models_enabled"))
        retain = {}
        for name, _module in self._runtime_component_entries(components):
            if name in {"unet", "transformer"}:
                retain[name] = checkpoint_retain
            elif name == "vae":
                retain[name] = vae_retain
            elif name.startswith("text_encoder"):
                retain[name] = text_retain
        retention_policy = str(values.get("model_runtime_retention_device") or "cuda_preferred").strip().lower()
        execution_device, fallback_reason = self._resolve_execution_device(values)
        staged_advanced = bool(advanced_models_enabled and memory_policy != "high_vram")
        staged_low_vram = memory_policy in {"low_vram", "cpu_fallback"}
        if staged_sd3:
            # SD3 is generation-qualified with component-at-a-time CUDA residency.
            # Keep hydrated modules cached on CPU between jobs and let the memory
            # lifecycle stage only the component needed by the active phase.
            target_device = torch.device("cpu")
            retention_policy = "staged_cpu"
        elif staged_advanced:
            # Advanced Models composes independently selected components, often
            # from several digital checkpoint donors. Unless high-VRAM mode is
            # explicitly requested, keeping every retained component on CUDA here
            # defeats the stage memory manager and can freeze low-memory systems
            # before generation starts. Keep the selected modules hydrated on CPU;
            # leases will promote only the conditioning/denoising/decode working set.
            target_device = torch.device("cpu")
            retention_policy = "advanced_staged_cpu"
        elif staged_low_vram:
            target_device = torch.device("cpu")
            retention_policy = "low_vram_staged_cpu"
        elif retention_policy == "cpu":
            target_device = torch.device("cpu")
        elif retention_policy == "cuda":
            if not torch.cuda.is_available():
                target_device = torch.device("cpu")
                fallback_reason = "GPU retention was requested, but CUDA is unavailable; retained on CPU."
            else:
                target_device = torch.device("cuda")
        else:
            target_device = execution_device
        moves: list[dict[str, Any]] = []
        for name, module in self._runtime_component_entries(components):
            keep_on_target = bool(retain.get(name, False))
            if not hasattr(module, "to"):
                continue
            target = target_device if keep_on_target else torch.device("cpu")
            before = "unknown"
            try:
                before = str(next(module.parameters()).device)
            except (StopIteration, AttributeError, TypeError):
                pass
            if before == str(target) or (before.startswith("cuda") and target.type == "cuda"):
                continue
            module.to(target)
            moves.append({"component": name, "from": before, "to": str(target)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "applied": True,
            "retain": retain,
            "retention_device": str(target_device),
            "staged_runtime": staged_sd3,
            "advanced_staged_runtime": staged_advanced,
            "memory_policy": memory_policy,
            "execution_device": str(execution_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_fallback_reason": fallback_reason,
            "moves": moves,
            "status": self.resident_model_status(),
        }

    def preload_model(
        self,
        model_path: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load and retain checkpoint components without running a generation."""
        preload_extras = dict(extras or {})
        preload_extras["model_path"] = str(model_path)
        defaults = dict(self.project_context.generation_defaults() or {})
        defaults.update({
            "positive_prompt": defaults.get("positive_prompt") or "warmup",
            "negative_prompt": defaults.get("negative_prompt") or "",
            "model_path": str(model_path),
            "save_images": False,
        })
        request, payload_extras = payload_to_generation_request(defaults)
        payload_extras.update(preload_extras)
        effective_config_fn = getattr(self.project_context, "effective_config", None)
        effective_config = effective_config_fn() if callable(effective_config_fn) else {
            "project_root": str(getattr(self.project_context, "project_root", "."))
        }
        session = self.diagnostics_system.start(
            request,
            effective_config=effective_config,
            request_extras=payload_extras,
        )
        started = time.perf_counter()
        try:
            request.device = str(self.device)
            self._configure_runtime_state(payload_extras, session)
            # Model activation must apply architecture/profile mutations before registry
            # resolution. SDXL preflight may change the sampler/scheduler and clears
            # descriptors that were resolved for the previous/default request values.
            # Running it here keeps WebUI preload ordering identical to generation.
            self._apply_sdxl_runtime_preflight(request, payload_extras)
            self._apply_sd3_runtime_preflight(request, payload_extras)
            request, payload_extras = self.registry_system.apply_resolution(request, payload_extras)
            self._build_pipeline(request, payload_extras, session)
            self.diagnostics_system.complete(session)
            status = self.resident_model_status()
            status.update({
                "preload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "model_provenance": dict(payload_extras.get("model_provenance") or {}),
            })
            return status
        except Exception as exc:
            raise self.diagnostics_system.fail_unassigned(
                session, exc, system="model_loading", operation="preload_model"
            ) from exc
        finally:
            self.reset_runtime_state()

    def _resolve_execution_device(self, extras: dict[str, Any]) -> tuple[torch.device, str | None]:
        policy = str(extras.get("model_runtime_execution_device") or "cuda_preferred").strip().lower()
        if policy == "cpu":
            return torch.device("cpu"), "CPU execution was explicitly selected."
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        if policy == "cuda_required":
            raise RuntimeError("CUDA execution is required, but torch.cuda.is_available() is false in this worker.")
        return torch.device("cpu"), "CUDA is unavailable in the model runtime; CPU fallback was activated."

    def _place_loaded_components(
        self,
        loaded: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, Any]]:
        components = getattr(loaded, "components", None)
        if components is None:
            return []
        reports: list[dict[str, Any]] = []
        architecture = str(getattr(components, "architecture", "") or "").strip().lower()
        sequential_cpu_first = (
            architecture in {"sdxl", "sd3.x", "sd3", "stable-diffusion-3.x"}
            and device.type == "cuda"
        )
        denoiser_kind = str(getattr(components, "denoiser_kind", "unet") or "unet").strip().lower()
        denoiser_name = "unet" if denoiser_kind == "unet" else "denoiser"
        component_names = (
            denoiser_name,
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        )
        seen_modules: set[int] = set()
        for name in component_names:
            module = getattr(components, name, None)
            if module is None or id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            target_device = torch.device("cpu") if sequential_cpu_first else device
            target_dtype = dtype
            if name == "vae" and bool(getattr(components, "vae_force_upcast", False)):
                target_dtype = torch.float32
            reports.append(
                place_component(
                    module,
                    device=target_device,
                    dtype=target_dtype,
                    owner=(
                        "Txt2ImgRunner.sdxl_cached_cpu_first"
                        if sequential_cpu_first and architecture == "sdxl"
                        else "Txt2ImgRunner.sd3_sequential_cpu_first"
                        if sequential_cpu_first
                        else "Txt2ImgRunner.execution_promotion"
                    ),
                    component_name=name,
                ).to_dict()
            )
        return reports
