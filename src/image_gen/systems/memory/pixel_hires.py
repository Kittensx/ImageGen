from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


PIXEL_HIRES_PREFLIGHT_SCHEMA_VERSION = "phase14n6-pixel-hires-preflight-v1"
PIXEL_HIRES_HOST_STAGING_SCHEMA_VERSION = "phase14n6-host-staging-v1"


class PixelHiresAdmissionError(RuntimeError):
    """Raised before base denoising when a pixel-hires job is clearly impossible."""


class PixelHiresCancelled(RuntimeError):
    """Raised at an owned Phase 14N-6 cancellation boundary."""

    def __init__(self, stage: str) -> None:
        self.stage = str(stage)
        super().__init__(f"Pixel-neural hires generation was cancelled during {self.stage}.")


@dataclass(frozen=True)
class PixelHiresPreflightReport:
    schema_version: str
    admitted: bool
    rejection_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    base_shape: tuple[int, int, int, int]
    target_shape: tuple[int, int, int, int]
    native_scale: int
    requested_tile_size: int
    requested_tile_overlap: int
    estimated_host_bytes: int
    available_host_bytes: int | None
    estimated_disk_bytes: int
    available_disk_bytes: int | None
    explicit_disk_budget_bytes: int | None
    estimated_vram_by_stage: Mapping[str, int]
    available_vram_bytes: int | None
    total_vram_bytes: int | None
    safety_margin_bytes: int
    major_contributors: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["warnings"] = list(self.warnings)
        payload["base_shape"] = list(self.base_shape)
        payload["target_shape"] = list(self.target_shape)
        payload["estimated_vram_by_stage"] = dict(self.estimated_vram_by_stage)
        payload["major_contributors"] = dict(self.major_contributors)
        return payload


@dataclass(frozen=True)
class HostStagingReport:
    schema_version: str
    label: str
    requested_policy: str
    effective_policy: str
    tensor_bytes: int
    cap_bytes: int
    source_device: str
    output_device: str
    output_pinned: bool
    non_blocking_copy: bool
    benchmark_requested: bool
    duration_ms: float | None
    fallback_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_bytes(batch: int, channels: int, width: int, height: int, bytes_per_value: int) -> int:
    return max(0, int(batch)) * max(0, int(channels)) * max(0, int(width)) * max(0, int(height)) * max(1, int(bytes_per_value))


def _snapshot_values(memory_manager: Any | None) -> tuple[int | None, int | None, int | None]:
    snapshot: dict[str, Any] = {}
    if memory_manager is not None:
        capture = getattr(memory_manager, "capture", None)
        if callable(capture):
            try:
                snapshot = dict(capture("pixel_hires_preflight") or {})
            except Exception:
                snapshot = {}
    cuda = dict(snapshot.get("cuda") or {})
    system = dict(snapshot.get("system") or {})
    available_vram = cuda.get("free_bytes")
    total_vram = cuda.get("total_bytes")
    available_host = system.get("available_bytes")
    return (
        int(available_vram) if available_vram is not None else None,
        int(total_vram) if total_vram is not None else None,
        int(available_host) if available_host is not None else None,
    )


def estimate_pixel_hires_preflight(
    *,
    request: Any,
    base_width: int,
    base_height: int,
    target_width: int,
    target_height: int,
    native_scale: int,
    model_file_size_bytes: int,
    memory_manager: Any | None,
    output_dir: str | Path | None = None,
) -> PixelHiresPreflightReport:
    """Conservatively estimate Phase 14N-6 host, disk, and stage VRAM admission.

    The gate rejects only physically impossible jobs. Current free VRAM is recorded
    for comparison, but total VRAM is the hard admission boundary because the
    stage-owned lifecycle can offload inactive components before each lease.
    """

    batch = max(1, int(getattr(request, "batch_size", 1) or 1))
    tile_size = max(0, int(getattr(request, "hires_tile_size", 0) or 0))
    overlap = max(0, int(getattr(request, "hires_tile_overlap", 0) or 0))
    native_scale = max(1, int(native_scale or 1))
    safety_margin = int(getattr(getattr(memory_manager, "settings", None), "safety_margin_bytes", 0) or 0)

    base_rgb = _tensor_bytes(batch, 3, base_width, base_height, 4)
    target_rgb = _tensor_bytes(batch, 3, target_width, target_height, 4)
    base_latent = _tensor_bytes(batch, 4, max(1, base_width // 8), max(1, base_height // 8), 4)
    target_latent = _tensor_bytes(batch, 4, max(1, target_width // 8), max(1, target_height // 8), 4)
    effective_tile = tile_size or max(base_width, base_height)
    tile_native_width = min(target_width, effective_tile * native_scale)
    tile_native_height = min(target_height, effective_tile * native_scale)
    tile_workspace = _tensor_bytes(1, 4, tile_native_width, tile_native_height, 4)
    model_runtime = int(max(0, model_file_size_bytes) * 1.35)

    component_bytes: dict[str, int] = {}
    if memory_manager is not None:
        resolver = getattr(memory_manager, "component_bytes", None)
        if callable(resolver):
            try:
                component_bytes = dict(resolver() or {})
            except Exception:
                component_bytes = {}
    unet_bytes = int(component_bytes.get("unet", 0) or 0)
    vae_bytes = int(component_bytes.get("vae", 0) or 0)

    stage_vram = {
        "base_denoise": unet_bytes + (base_latent * 5),
        "base_decode": vae_bytes + (base_rgb * 3),
        "neural_upscale": model_runtime + target_rgb + tile_workspace,
        "vae_encode": vae_bytes + (target_rgb * 2) + (target_latent * 3),
        "hires_second_pass": unet_bytes + (target_latent * 5),
        "final_decode": vae_bytes + (target_rgb * 3),
    }
    diagnostics = dict(getattr(request, "diagnostics", None) or {})
    save_images = bool(getattr(request, "save_images", False))
    roundtrip_captured = bool(getattr(request, "hires_save_vae_roundtrip", False))
    pre_denoise_captured = bool(
        getattr(request, "hires_save_upscaled_pre_denoise", False)
    )
    lowres_captured = bool(
        getattr(request, "hires_save_lowres", False)
        and not bool(getattr(request, "return_latents", False))
    )
    roundtrip_persisted = bool(save_images and roundtrip_captured)
    pre_denoise_persisted = bool(save_images and pre_denoise_captured)
    lowres_persisted = bool(save_images and lowres_captured)
    final_count = 1 if save_images else 0
    roles_per_image = (
        final_count
        + int(roundtrip_persisted)
        + int(pre_denoise_persisted)
        + int(lowres_persisted)
    )
    image_count = batch * roles_per_image

    host_contributors = {
        "decoded_base_rgb": base_rgb,
        "exact_target_rgb": target_rgb,
        "base_latent_retry_boundary": base_latent,
        "hires_latent_retry_boundary": target_latent,
        "optional_pre_denoise_rgb": target_rgb if pre_denoise_captured else 0,
        "optional_roundtrip_rgb": target_rgb if roundtrip_captured else 0,
        "optional_lowres_rgb": base_rgb if lowres_captured else 0,
    }
    estimated_host = int(sum(host_contributors.values()) * 1.20)
    # PNG can approach raw RGB size for noisy diffusion output. Add JSON/log and
    # atomic temporary-file headroom rather than assuming compression.
    disk_image_bytes = (
        base_width * base_height * 3 * batch if lowres_persisted else 0
    )
    disk_image_bytes += target_width * target_height * 3 * batch * (
        final_count + int(roundtrip_persisted) + int(pre_denoise_persisted)
    )
    estimated_disk = int(
        disk_image_bytes * 2.20 + image_count * 2 * 1024 * 1024
    )

    available_vram, total_vram, available_host = _snapshot_values(memory_manager)
    destination = Path(output_dir or getattr(request, "output_dir", None) or ".").expanduser()
    try:
        destination.mkdir(parents=True, exist_ok=True)
        available_disk = int(shutil.disk_usage(destination).free)
    except OSError:
        available_disk = None

    explicit_budget_mb = int(getattr(request, "hires_artifact_disk_budget_mb", 0) or 0)
    explicit_budget = explicit_budget_mb * 1024 * 1024 if explicit_budget_mb > 0 else None
    reasons: list[str] = []
    warnings: list[str] = []
    hardest_stage = max(stage_vram.items(), key=lambda item: item[1])
    physical_limit = None if total_vram is None else max(0, total_vram - safety_margin)
    if physical_limit is not None and hardest_stage[1] > physical_limit:
        reasons.append(
            f"Estimated {hardest_stage[0]} VRAM requirement {hardest_stage[1]} exceeds the physical admission limit {physical_limit}."
        )
    elif available_vram is not None and hardest_stage[1] > max(0, available_vram - safety_margin):
        warnings.append(
            "The current free VRAM is below the conservative estimate; stage-owned offloading and the one bounded tile retry will be required."
        )
    if available_host is not None and estimated_host > available_host:
        reasons.append(
            f"Estimated host staging requirement {estimated_host} exceeds available system memory {available_host}."
        )
    if available_disk is not None and estimated_disk > available_disk:
        reasons.append(
            f"Estimated output and diagnostic disk requirement {estimated_disk} exceeds free disk space {available_disk}."
        )
    if explicit_budget is not None and estimated_disk > explicit_budget:
        reasons.append(
            f"Estimated output and diagnostic disk requirement {estimated_disk} exceeds the configured budget {explicit_budget}."
        )
    if diagnostics.get("capture_hires_memory_preflight", False):
        warnings.append("Detailed preflight capture was requested for this diagnostic run.")

    return PixelHiresPreflightReport(
        schema_version=PIXEL_HIRES_PREFLIGHT_SCHEMA_VERSION,
        admitted=not reasons,
        rejection_reasons=tuple(reasons),
        warnings=tuple(warnings),
        base_shape=(batch, 3, int(base_height), int(base_width)),
        target_shape=(batch, 3, int(target_height), int(target_width)),
        native_scale=native_scale,
        requested_tile_size=tile_size,
        requested_tile_overlap=overlap,
        estimated_host_bytes=estimated_host,
        available_host_bytes=available_host,
        estimated_disk_bytes=estimated_disk,
        available_disk_bytes=available_disk,
        explicit_disk_budget_bytes=explicit_budget,
        estimated_vram_by_stage=stage_vram,
        available_vram_bytes=available_vram,
        total_vram_bytes=total_vram,
        safety_margin_bytes=safety_margin,
        major_contributors={**host_contributors, "upscaler_model_runtime": model_runtime},
    )


def stage_tensor_to_host(
    tensor: torch.Tensor,
    *,
    label: str,
    policy: str = "pageable",
    cap_bytes: int = 0,
    benchmark: bool = False,
) -> tuple[torch.Tensor, HostStagingReport]:
    requested = str(policy or "pageable").strip().casefold().replace("-", "_")
    if requested not in {"pageable", "pinned", "auto"}:
        raise ValueError("hires_host_staging_policy must be pageable, pinned, or auto.")
    tensor_bytes = int(tensor.numel() * tensor.element_size())
    cap = max(0, int(cap_bytes))
    use_pinned = requested in {"pinned", "auto"}
    fallback_reason = ""
    if use_pinned and cap and tensor_bytes > cap:
        use_pinned = False
        fallback_reason = "tensor_exceeds_host_staging_cap"
    if use_pinned and not torch.cuda.is_available():
        use_pinned = False
        fallback_reason = "cuda_unavailable"
    if use_pinned and tensor.device.type == "cpu" and tensor.is_pinned():
        output = tensor.detach()
        duration_ms = 0.0 if benchmark else None
        return output, HostStagingReport(
            schema_version=PIXEL_HIRES_HOST_STAGING_SCHEMA_VERSION,
            label=str(label), requested_policy=requested, effective_policy="pinned",
            tensor_bytes=tensor_bytes, cap_bytes=cap, source_device=str(tensor.device),
            output_device="cpu", output_pinned=True, non_blocking_copy=False,
            benchmark_requested=bool(benchmark), duration_ms=duration_ms, fallback_reason="",
        )

    started = time.perf_counter() if benchmark else None
    non_blocking = False
    if use_pinned:
        try:
            output = torch.empty_like(tensor, device="cpu", pin_memory=True)
            non_blocking = tensor.device.type == "cuda"
            output.copy_(tensor.detach(), non_blocking=non_blocking)
            if non_blocking:
                # This is an explicit stage boundary. Synchronization is not
                # performed anywhere inside a tile loop.
                torch.cuda.current_stream(tensor.device).synchronize()
            effective = "pinned"
        except (RuntimeError, OSError) as exc:
            output = tensor.detach().to(device="cpu")
            effective = "pageable"
            non_blocking = False
            fallback_reason = f"pinned_allocation_failed:{type(exc).__name__}"
    else:
        output = tensor.detach().to(device="cpu")
        effective = "pageable"
    duration_ms = ((time.perf_counter() - started) * 1000.0) if started is not None else None
    report = HostStagingReport(
        schema_version=PIXEL_HIRES_HOST_STAGING_SCHEMA_VERSION,
        label=str(label), requested_policy=requested, effective_policy=effective,
        tensor_bytes=tensor_bytes, cap_bytes=cap, source_device=str(tensor.device),
        output_device=str(output.device), output_pinned=bool(output.is_pinned()),
        non_blocking_copy=non_blocking, benchmark_requested=bool(benchmark),
        duration_ms=duration_ms, fallback_reason=fallback_reason,
    )
    return output, report


def cancellation_requested(*, request: Any, state: Any | None = None) -> bool:
    diagnostics = dict(getattr(request, "diagnostics", None) or {})
    callback = diagnostics.get("cancellation_check")
    if callable(callback):
        try:
            if bool(callback()):
                return True
        except Exception:
            return True
    extra = getattr(state, "extra", None)
    if isinstance(extra, dict):
        callback = extra.get("generation_cancellation_check")
        if callable(callback):
            try:
                if bool(callback()):
                    return True
            except Exception:
                return True
        for key in ("cancel_requested", "interrupted", "skip_requested"):
            if bool(extra.get(key, False)):
                return True
    return False


def raise_if_pixel_hires_cancelled(stage: str, *, request: Any, state: Any | None = None) -> None:
    if cancellation_requested(request=request, state=state):
        raise PixelHiresCancelled(stage)


def compare_preflight_to_actual(
    preflight: Mapping[str, Any] | PixelHiresPreflightReport,
    memory_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    forecast = preflight.to_dict() if isinstance(preflight, PixelHiresPreflightReport) else dict(preflight or {})
    actual = dict(memory_summary or {})
    peaks = dict(actual.get("peak_vram_by_stage") or {})
    comparisons: dict[str, Any] = {}
    for stage, estimate in dict(forecast.get("estimated_vram_by_stage") or {}).items():
        peak = dict(peaks.get(stage) or {}).get("peak_allocated_vram_bytes")
        comparisons[str(stage)] = {
            "estimated_bytes": int(estimate or 0),
            "actual_peak_allocated_bytes": None if peak is None else int(peak),
            "actual_minus_estimate_bytes": None if peak is None else int(peak) - int(estimate or 0),
        }
    return {
        "schema_version": "phase14n6-preflight-actual-comparison-v1",
        "stages": comparisons,
        "transfer_count": len(list(actual.get("transfers") or [])),
    }
