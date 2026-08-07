from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from image_gen.systems.upscaling.contracts import (
    UpscaleProgress,
    UpscaleRequest,
    UpscaleResult,
    UpscalerRuntimeQualification,
)
from image_gen.systems.upscaling.registry import UpscalerModelRegistry
from image_gen.systems.upscaling.resize import resize_exact, resolve_target_dimensions
from image_gen.systems.upscaling.tiling import (
    RuntimeModelMetadata,
    TilePlan,
    linear_blend_weight,
    normalize_spandrel_metadata,
    plan_tiles,
)

ProgressCallback = Callable[[Mapping[str, Any]], None]
CancellationCheck = Callable[[], bool]


class UpscaleRuntimeError(RuntimeError):
    pass


class UpscaleCancelled(UpscaleRuntimeError):
    pass


@dataclass
class _ProgressState:
    total_tiles: int
    completed_tiles: int = 0
    callback: ProgressCallback | None = None

    def emit(self, *, event: str, **extra: Any) -> None:
        if self.callback is None:
            return
        payload = {
            "event": event,
            "completed_tiles": int(self.completed_tiles),
            "total_tiles": int(self.total_tiles),
            "progress_percent": (
                (self.completed_tiles / self.total_tiles) * 100.0
                if self.total_tiles
                else 100.0
            ),
            **extra,
        }
        self.callback(payload)


class StandaloneNeuralUpscaler:
    """Phase 14N-3 standalone pixel neural inference runtime.

    The retained Spandrel ImageModelDescriptor is the only inference call
    boundary. The runtime never reopens a model file and never calls the raw
    underlying torch module directly.
    """

    def __init__(self, registry: UpscalerModelRegistry) -> None:
        self.registry = registry

    def upscale(
        self,
        request: UpscaleRequest,
        *,
        cancellation_check: CancellationCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> UpscaleResult:
        request = request.normalized()
        if int(request.tile_batch_size) != 1:
            raise ValueError(
                "Spandrel ImageModelDescriptor.__call__ is qualified for one image/tile at a time; "
                "tile_batch_size must remain 1 in Phase 14N-3."
            )
        self._validate_source(request.source_images)
        source = request.source_images
        source_height = int(source.shape[-2])
        source_width = int(source.shape[-1])
        target_width, target_height = resolve_target_dimensions(
            source_width=source_width,
            source_height=source_height,
            target_width=request.target_width,
            target_height=request.target_height,
            scale=request.scale,
        )
        target_device = self._resolve_device(request.device_policy, source)
        requested_dtype = self._resolve_dtype(
            upscaler_id=request.upscaler_id,
            device=target_device,
            dtype_policy=request.dtype_policy,
        )

        started = time.perf_counter()
        retry_count = 0
        oom_cleanup_records: list[dict[str, Any]] = []
        model_lease_started = time.perf_counter()
        initial_tile_size = int(request.tile_size)
        effective_tile_size = initial_tile_size if request.allow_tiling else 0
        retry_reason = ""

        # Keep one registry lease and one Spandrel model instance for the entire
        # standalone operation, including the optional local tile retry. The
        # retry must never reopen the .pth file or construct a second model.
        with self.registry.lease(
            request.upscaler_id,
            device=target_device,
            dtype=requested_dtype,
        ) as loaded:
            model_load_duration_ms = (time.perf_counter() - model_lease_started) * 1000.0
            model_metadata = normalize_spandrel_metadata(loaded.model_descriptor)
            self._validate_runtime_metadata(model_metadata)
            source_for_model = source.to(
                device=target_device,
                dtype=next(loaded.module.parameters()).dtype,
                non_blocking=bool(
                    request.host_transfer_non_blocking
                    and source.device.type == "cpu"
                    and bool(source.is_pinned())
                    and target_device.type == "cuda"
                ),
            )
            while True:
                try:
                    plan = plan_tiles(
                        source_width=source_width,
                        source_height=source_height,
                        native_scale=model_metadata.native_scale,
                        requested_tile_size=effective_tile_size,
                        overlap_source_pixels=request.tile_overlap,
                        metadata=model_metadata,
                    )
                    progress = _ProgressState(
                        total_tiles=int(source_for_model.shape[0]) * len(plan.regions),
                        callback=progress_callback,
                    )
                    progress.emit(
                        event="started",
                        upscaler_id=request.upscaler_id,
                        device=str(target_device),
                        dtype=str(source_for_model.dtype),
                        tile_size=int(effective_tile_size),
                        attempt=retry_count + 1,
                    )
                    native_output = self._execute_batch(
                        loaded.model_descriptor,
                        source_for_model,
                        plan=plan,
                        metadata=model_metadata,
                        cancellation_check=cancellation_check,
                        progress=progress,
                    )
                    break
                except Exception as exc:
                    genuine_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                        target_device.type == "cuda" and self._is_cuda_oom(exc)
                    )
                    if not genuine_oom or not request.allow_oom_retry or retry_count >= 1:
                        raise
                    previous_tile_size = effective_tile_size
                    next_tile_size = self._retry_tile_size(
                        source_width=source_width,
                        source_height=source_height,
                        current_tile_size=effective_tile_size,
                        minimum_tile_size=request.minimum_retry_tile_size,
                    )
                    if next_tile_size == effective_tile_size:
                        raise
                    retry_count += 1
                    retry_reason = f"{type(exc).__name__}: {exc}"
                    effective_tile_size = next_tile_size
                    cleanup_record = {
                        "schema_version": "phase14n6-neural-oom-cleanup-v1",
                        "attempt": int(retry_count),
                        "failed_output_references_released": True,
                        "blend_canvas_released": True,
                        "blend_weights_released": True,
                        "traceback_frames_cleared": True,
                        "retry_source": "original_exact_source_tensor",
                        "previous_tile_size": int(previous_tile_size),
                        "retry_tile_size": int(effective_tile_size),
                        "reason": retry_reason[:512],
                    }
                    oom_cleanup_records.append(cleanup_record)
                    # The traceback can retain the failed tile output and blend
                    # accumulators. Clear it before cache release so the retry
                    # receives a genuinely clean standalone attempt.
                    exc.__traceback__ = None
                    gc.collect()
                    manager = getattr(self.registry, "memory_manager", None)
                    recorder = getattr(manager, "record_external_stage_telemetry", None)
                    if callable(recorder):
                        recorder("neural_upscale", {"event": "oom_retry", **cleanup_record})
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "oom_retry",
                                "retry_count": retry_count,
                                "previous_tile_size": previous_tile_size,
                                "retry_tile_size": effective_tile_size,
                                "reason": retry_reason[:512],
                            }
                        )
                    cache_release = getattr(
                        getattr(self.registry, "memory_manager", None),
                        "release_cuda_cache",
                        None,
                    )
                    if callable(cache_release):
                        cache_release(
                            stage="neural_upscale",
                            reason="bounded standalone tile OOM retry",
                        )
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()

            qualification = self.registry.qualification(request.upscaler_id)
            runtime_dtype = str(next(loaded.module.parameters()).dtype)
            runtime_device = str(next(loaded.module.parameters()).device)

        resized = resize_exact(
            native_output,
            target_width=target_width,
            target_height=target_height,
            resize_filter=request.exact_resize_filter,
        )
        if not bool(torch.isfinite(resized).all()):
            raise UpscaleRuntimeError("Neural upscaler output contains NaN or Inf.")
        resized = resized.clamp(0.0, 1.0)
        duration_ms = (time.perf_counter() - started) * 1000.0
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "completed",
                    "completed_tiles": int(source.shape[0]) * len(plan.regions),
                    "total_tiles": int(source.shape[0]) * len(plan.regions),
                    "progress_percent": 100.0,
                    "duration_ms": duration_ms,
                }
            )

        metadata = {
            "schema_version": "phase14n3-standalone-upscale-v1",
            "inference_boundary": "spandrel.ImageModelDescriptor.__call__",
            "upscaler_id": request.upscaler_id,
            "upscaler_display_name": str(getattr(getattr(loaded, "descriptor", None), "display_name", request.upscaler_id)),
            "upscaler_architecture": str(getattr(getattr(loaded, "descriptor", None), "architecture", qualification.architecture_id)),
            "upscaler_native_scale": int(getattr(getattr(loaded, "descriptor", None), "native_scale", qualification.native_scale) or 0),
            "upscaler_load_status": str(getattr(getattr(loaded, "descriptor", None), "load_status", "runtime_qualified")),
            "upscaler_sha256": qualification.descriptor_sha256,
            "loader_backend": qualification.loader_backend,
            "loader_backend_version": qualification.loader_backend_version,
            "runtime_architecture_id": qualification.architecture_id,
            "runtime_qualification": qualification.to_dict(),
            "runtime_device": runtime_device,
            "runtime_dtype": runtime_dtype,
            "model_metadata": model_metadata.to_dict(),
            "source_shape": [int(item) for item in source.shape],
            "native_output_shape": [int(item) for item in native_output.shape],
            "output_shape": [int(item) for item in resized.shape],
            "target_width": target_width,
            "target_height": target_height,
            "exact_resize_filter": request.exact_resize_filter,
            "tile_batch_size": request.tile_batch_size,
            "tile_plan": plan.to_dict(include_regions=True),
            "tile_count": int(source.shape[0]) * len(plan.regions),
            "tile_size": int(plan.effective_tile_size),
            "tile_overlap": int(plan.overlap_source_pixels),
            "initial_tile_size": initial_tile_size,
            "effective_tile_size": effective_tile_size,
            "allow_oom_retry": bool(request.allow_oom_retry),
            "allow_tiling": bool(request.allow_tiling),
            "oom_retry_count": retry_count,
            "oom_retry_reason": retry_reason[:512],
            "deterministic_tile_order": "batch_then_source_row_major",
            "duration_ms": duration_ms,
            "model_load_duration_ms": model_load_duration_ms,
            "inference_duration_ms": max(0.0, duration_ms - model_load_duration_ms),
            "oom_retry_cleanup": list(oom_cleanup_records),
            "host_transfer_non_blocking": bool(request.host_transfer_non_blocking),
        }
        recorder = getattr(
            getattr(self.registry, "memory_manager", None),
            "record_external_stage_telemetry",
            None,
        )
        if callable(recorder):
            recorder(
                "neural_upscale",
                {
                    "event": "completed",
                    "upscaler_id": request.upscaler_id,
                    "model_load_duration_ms": model_load_duration_ms,
                    "inference_duration_ms": metadata["inference_duration_ms"],
                    "tile_count": int(source.shape[0]) * len(plan.regions),
                    "initial_tile_size": int(initial_tile_size),
                    "effective_tile_size": int(effective_tile_size),
                    "tile_overlap": int(request.tile_overlap),
                    "oom_retry_count": int(retry_count),
                    "oom_retry_reason": retry_reason[:512],
                },
            )
        return UpscaleResult(images=resized, metadata=metadata)

    def _execute_batch(
        self,
        model_descriptor: Any,
        source: torch.Tensor,
        *,
        plan: TilePlan,
        metadata: RuntimeModelMetadata,
        cancellation_check: CancellationCheck | None,
        progress: _ProgressState,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for batch_index in range(int(source.shape[0])):
            self._check_cancelled(cancellation_check)
            image = source[batch_index : batch_index + 1]
            if plan.tiled:
                output = self._execute_tiled_image(
                    model_descriptor,
                    image,
                    plan=plan,
                    metadata=metadata,
                    cancellation_check=cancellation_check,
                    progress=progress,
                    batch_index=batch_index,
                )
            else:
                output = self._call_descriptor(model_descriptor, image, metadata=metadata)
                progress.completed_tiles += 1
                progress.emit(
                    event="tile_completed",
                    batch_index=batch_index,
                    tile_index=0,
                )
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def _execute_tiled_image(
        self,
        model_descriptor: Any,
        image: torch.Tensor,
        *,
        plan: TilePlan,
        metadata: RuntimeModelMetadata,
        cancellation_check: CancellationCheck | None,
        progress: _ProgressState,
        batch_index: int,
    ) -> torch.Tensor:
        scale = metadata.native_scale
        canvas: torch.Tensor | None = torch.zeros(
            (
                1,
                metadata.output_channels,
                plan.source_height * scale,
                plan.source_width * scale,
            ),
            device=image.device,
            dtype=torch.float32,
        )
        weights: torch.Tensor | None = torch.zeros_like(canvas[:, :1])
        output_tile: torch.Tensor | None = None
        blend: torch.Tensor | None = None

        try:
            for region in plan.regions:
                self._check_cancelled(cancellation_check)
                source_tile = image[
                    ...,
                    region.source_y : region.source_y + region.source_height,
                    region.source_x : region.source_x + region.source_width,
                ]
                output_tile = self._call_descriptor(
                    model_descriptor,
                    source_tile,
                    metadata=metadata,
                )
                expected_shape = (region.output_height, region.output_width)
                if tuple(output_tile.shape[-2:]) != expected_shape:
                    raise UpscaleRuntimeError(
                        "Spandrel descriptor returned an unexpected tile shape: "
                        f"expected {expected_shape}, received {tuple(output_tile.shape[-2:])}."
                    )
                blend = linear_blend_weight(
                    region.output_height,
                    region.output_width,
                    top=region.top_overlap,
                    bottom=region.bottom_overlap,
                    left=region.left_overlap,
                    right=region.right_overlap,
                    device=output_tile.device,
                    dtype=torch.float32,
                )[None, None]
                y0 = region.output_y
                x0 = region.output_x
                y1 = y0 + region.output_height
                x1 = x0 + region.output_width
                canvas[..., y0:y1, x0:x1] += output_tile.float() * blend
                weights[..., y0:y1, x0:x1] += blend
                progress.completed_tiles += 1
                progress.emit(
                    event="tile_completed",
                    batch_index=batch_index,
                    tile_index=region.tile_index,
                    row_index=region.row_index,
                    column_index=region.column_index,
                )
            if bool((weights <= 0).any()):
                raise UpscaleRuntimeError("Tiled upscaling left uncovered output pixels.")
            return (canvas / weights).to(dtype=image.dtype).clamp(0.0, 1.0)
        except Exception:
            output_tile = None
            blend = None
            weights = None
            canvas = None
            raise

    @staticmethod
    def _call_descriptor(
        model_descriptor: Any,
        image: torch.Tensor,
        *,
        metadata: RuntimeModelMetadata,
    ) -> torch.Tensor:
        output = model_descriptor(image)
        if not torch.is_tensor(output) or output.ndim != 4:
            raise UpscaleRuntimeError("Spandrel ImageModelDescriptor must return a BCHW tensor.")
        if int(output.shape[0]) != 1 or int(output.shape[1]) != metadata.output_channels:
            raise UpscaleRuntimeError("Spandrel descriptor returned an invalid batch/channel shape.")

        # Spandrel's ImageModelDescriptor.__call__ executes under inference mode.
        # The returned value can therefore be an inference tensor, which cannot
        # be mutated by ImageGen outside that context. Clone immediately to
        # establish an ImageGen-owned normal tensor before finite checks,
        # blending, resizing, or clamping.
        output = output.clone()
        if not bool(torch.isfinite(output).all()):
            raise UpscaleRuntimeError("Spandrel descriptor returned NaN or Inf.")
        return output.clamp(0.0, 1.0)

    @staticmethod
    def _validate_source(images: torch.Tensor) -> None:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("Standalone neural upscaling requires a BCHW tensor.")
        if int(images.shape[0]) < 1:
            raise ValueError("Standalone neural upscaling requires at least one image.")
        if int(images.shape[1]) != 3:
            raise ValueError("Standalone neural upscaling currently requires RGB BCHW input.")
        if not images.is_floating_point():
            raise ValueError("Standalone neural upscaling requires floating-point RGB input.")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("Standalone neural upscaling input contains NaN or Inf.")
        minimum = float(images.min().detach().cpu())
        maximum = float(images.max().detach().cpu())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError("Standalone neural upscaling expects RGB values in [0, 1].")

    def _resolve_device(self, policy: str, source: torch.Tensor) -> torch.device:
        selected = str(policy or "auto").strip().casefold()
        if selected not in {"auto", "cpu", "cuda"}:
            raise ValueError("device_policy must be auto, cpu, or cuda.")
        if selected == "cpu":
            return torch.device("cpu")
        if selected == "cuda":
            if not torch.cuda.is_available():
                raise UpscaleRuntimeError("CUDA neural upscaling was requested, but CUDA is unavailable.")
            return torch.device("cuda")
        manager = getattr(self.registry, "memory_manager", None)
        manager_device = getattr(manager, "target_device", None)
        if manager_device is not None:
            return torch.device(manager_device)
        if source.device.type == "cuda":
            return source.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_dtype(
        self,
        *,
        upscaler_id: str,
        device: torch.device,
        dtype_policy: str,
    ) -> torch.dtype:
        policy = str(dtype_policy or "auto").strip().casefold()
        if policy not in {"auto", "fp32", "fp16_if_qualified", "bf16_if_qualified"}:
            raise ValueError(
                "dtype_policy must be auto, fp32, fp16_if_qualified, or bf16_if_qualified."
            )
        qualification = self.registry.qualification(upscaler_id)
        matching_status = "qualified_cuda" if device.type == "cuda" else "qualified_cpu"

        if policy == "fp32":
            return torch.float32
        if policy == "fp16_if_qualified":
            if device.type != "cuda":
                raise UpscaleRuntimeError("Phase 14N-3 FP16 execution is qualified only on CUDA.")
            if qualification.status != matching_status or not qualification.supports_half:
                raise UpscaleRuntimeError(
                    "FP16 requires a successful matching Phase 14N-2 CUDA runtime qualification."
                )
            return torch.float16
        if policy == "bf16_if_qualified":
            if qualification.status != matching_status or not qualification.supports_bfloat16:
                raise UpscaleRuntimeError(
                    "BF16 requires a successful matching Phase 14N-2 runtime qualification."
                )
            return torch.bfloat16

        if (
            device.type == "cuda"
            and qualification.status == "qualified_cuda"
            and qualification.supports_half
        ):
            return torch.float16
        return torch.float32

    @staticmethod
    def _validate_runtime_metadata(metadata: RuntimeModelMetadata) -> None:
        if metadata.purpose not in {"SR", "FaceSR"}:
            raise UpscaleRuntimeError(
                f"Standalone neural upscaling requires SR purpose, received {metadata.purpose!r}."
            )
        if metadata.native_scale <= 0:
            raise UpscaleRuntimeError("Runtime upscaler metadata reported an invalid native scale.")
        if metadata.input_channels != 3 or metadata.output_channels != 3:
            raise UpscaleRuntimeError("Standalone neural upscaling currently requires RGB models.")

    @staticmethod
    def _check_cancelled(cancellation_check: CancellationCheck | None) -> None:
        if cancellation_check is not None and bool(cancellation_check()):
            raise UpscaleCancelled("Neural upscaling was cancelled between tiles.")

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        text = str(exc).casefold()
        return "cuda" in text and "out of memory" in text

    @staticmethod
    def _retry_tile_size(
        *,
        source_width: int,
        source_height: int,
        current_tile_size: int,
        minimum_tile_size: int,
    ) -> int:
        minimum = max(8, int(minimum_tile_size))
        maximum_dimension = max(int(source_width), int(source_height))
        if int(current_tile_size) <= 0:
            candidate = max(minimum, min(256, max(minimum, maximum_dimension // 2)))
        else:
            candidate = max(minimum, int(current_tile_size) // 2)
        if candidate >= maximum_dimension and maximum_dimension > minimum:
            candidate = max(minimum, maximum_dimension // 2)
        return int(candidate)


__all__ = [
    "CancellationCheck",
    "ProgressCallback",
    "StandaloneNeuralUpscaler",
    "UpscaleCancelled",
    "UpscaleRequest",
    "UpscaleResult",
    "UpscaleRuntimeError",
]


def _resolve_dtype(
    dtype_policy: str,
    qualification: UpscalerRuntimeQualification,
    device: torch.device,
) -> torch.dtype:
    selected = str(dtype_policy or "auto").strip().casefold()
    if selected == "fp32" or device.type != "cuda":
        return torch.float32
    if selected == "bf16_if_qualified":
        if qualification.status == "qualified_cuda" and qualification.supports_bfloat16:
            return torch.bfloat16
        return torch.float32
    if selected == "fp16_if_qualified":
        if qualification.status == "qualified_cuda" and qualification.supports_half:
            return torch.float16
        return torch.float32
    if qualification.status == "qualified_cuda":
        if qualification.supports_half:
            return torch.float16
        if qualification.supports_bfloat16:
            return torch.bfloat16
    dtype_name = str(qualification.dtype or "").strip()
    return {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16}.get(dtype_name, torch.float32)


class StandaloneUpscaleRuntime(StandaloneNeuralUpscaler):
    def execute(
        self,
        request: UpscaleRequest,
        *,
        progress_callback: Callable[[UpscaleProgress], None] | None = None,
        cancel_callback: CancellationCheck | None = None,
    ) -> UpscaleResult:
        adapter = None
        if progress_callback is not None:
            def adapter(event):
                if isinstance(event, dict) and "completed_tiles" in event and "total_tiles" in event:
                    progress_callback(
                        UpscaleProgress(
                            completed_tiles=int(event.get("completed_tiles", 0)),
                            total_tiles=int(event.get("total_tiles", 0)),
                            batch_index=int(event.get("batch_index", 0)),
                            batch_size=int(event.get("batch_size", 0)),
                            tile_index=int(event.get("tile_index", 0)),
                            tile_coordinates=tuple(int(value) for value in event.get("tile_coordinates", (0, 0, 0, 0))),
                        )
                    )
            adapter = adapter
        return self.upscale(request, cancellation_check=cancel_callback, progress_callback=adapter)


def run_standalone_neural_upscale(
    registry: UpscalerModelRegistry,
    request: UpscaleRequest,
    *,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: Callable[[UpscaleProgress], None] | None = None,
) -> UpscaleResult:
    return StandaloneUpscaleRuntime(registry).execute(
        request,
        cancel_callback=cancellation_check,
        progress_callback=progress_callback,
    )


execute_upscale_request = run_standalone_neural_upscale

__all__ = [
    "StandaloneNeuralUpscaler",
    "StandaloneUpscaleRuntime",
    "UpscaleCancelled",
    "UpscaleRuntimeError",
    "_resolve_dtype",
    "execute_upscale_request",
    "run_standalone_neural_upscale",
]
