from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from PIL import Image

from modules.pipeline.live_preview import LivePreviewFrame


_FAST_LATENT_RGB = torch.tensor(
    [
        [0.2980, 0.2070, 0.2080],
        [0.1870, 0.2860, 0.1730],
        [-0.1580, 0.1890, 0.2640],
        [-0.1840, -0.2710, -0.4730],
    ],
    dtype=torch.float32,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _percentile(values: list[float], percentile: float) -> float:
    cleaned = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not cleaned:
        return 0.0
    position = (len(cleaned) - 1) * min(max(float(percentile), 0.0), 1.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return cleaned[lower]
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def _decoder_vae(decoder: Callable[[torch.Tensor], torch.Tensor] | None) -> Any | None:
    owner = getattr(decoder, "__self__", None)
    vae = getattr(owner, "vae", None)
    if vae is None:
        vae = getattr(getattr(owner, "base", None), "vae", None)
    return vae


def move_latent_to_decoder(
    latent: torch.Tensor,
    decoder: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Move a detached latent to the VAE's device/dtype when discoverable."""

    value = latent.detach()
    vae = _decoder_vae(decoder)
    if vae is None:
        return value
    try:
        parameter = next(vae.parameters())
    except (StopIteration, AttributeError, TypeError):
        return value
    return value.to(device=parameter.device, dtype=parameter.dtype)


def tensor_to_pil_images(images: torch.Tensor) -> list[Image.Image]:
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(images.shape)}")
    batch = images.detach().float().cpu().clamp(0.0, 1.0)
    output: list[Image.Image] = []
    for image in batch:
        channel_count = int(image.shape[0])
        if channel_count == 1:
            image_uint8 = (image[0] * 255.0).round().to(torch.uint8).numpy()
            pil_image = Image.fromarray(image_uint8, mode="L")
        else:
            image_hwc = image[:4].permute(1, 2, 0)
            image_uint8 = (image_hwc * 255.0).round().to(torch.uint8).numpy()
            pil_image = Image.fromarray(image_uint8)
        output.append(pil_image.convert("RGB"))
    return output


def select_batch_item(tensor: torch.Tensor, batch_index: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Live preview expects BCHW tensor data, got {tuple(tensor.shape)}.")
    if tensor.shape[0] <= 0:
        raise ValueError("Live preview tensor has an empty batch.")
    index = min(max(int(batch_index), 0), int(tensor.shape[0]) - 1)
    return tensor[index : index + 1]


def fast_latent_to_image(latent: torch.Tensor, *, batch_index: int = 0) -> Image.Image:
    """Create a cheap RGB approximation from one latent batch item."""

    selected = select_batch_item(latent, batch_index).detach().float().cpu()
    channels = int(selected.shape[1])
    if channels < 4:
        padding = torch.zeros(
            (selected.shape[0], 4 - channels, selected.shape[2], selected.shape[3]),
            dtype=selected.dtype,
        )
        selected = torch.cat([selected, padding], dim=1)
    selected = selected[:, :4]
    matrix = _FAST_LATENT_RGB.to(dtype=selected.dtype)
    rgb = torch.einsum("bchw,cr->brhw", selected, matrix)
    rgb = (rgb + 0.5).clamp(0.0, 1.0)
    return tensor_to_pil_images(rgb)[0]


def decode_latent_to_pil_images(
    decoder: Callable[[torch.Tensor], torch.Tensor],
    latent: torch.Tensor,
    *,
    batch_index: int | None = None,
) -> list[Image.Image]:
    """Decode detached latent data with the existing pipeline VAE."""

    if decoder is None:
        raise RuntimeError("No VAE decoder was supplied for live preview.")
    source = latent.detach()
    if batch_index is not None:
        source = select_batch_item(source, batch_index)
    decode_latent = move_latent_to_decoder(source, decoder)
    with torch.no_grad():
        decoded = decoder(decode_latent)
    return tensor_to_pil_images(decoded)


def resize_preserving_aspect(image: Image.Image, target_long_edge: int) -> Image.Image:
    target = max(64, int(target_long_edge))
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("Preview image has invalid dimensions.")
    scale = target / float(max(width, height))
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    if new_size == source.size:
        return source
    return source.resize(new_size, Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class LivePreviewSettings:
    enabled: bool = True
    mode: str = "fast"
    interval: int = 1
    width: int = 384
    image_format: str = "webp"
    keep_history: str = "current_job"
    batch_index: int = 0
    quality: int = 78
    adaptive_throttle: bool = True
    adaptive_target_ratio: float = 0.75
    adaptive_recovery_ratio: float = 0.40
    adaptive_max_interval: int = 8
    adaptive_window: int = 6
    adaptive_suspend_on_overhead: bool = True
    adaptive_suspend_ratio: float = 0.55
    adaptive_suspend_min_work_ms: float = 1000.0
    adaptive_suspend_min_samples: int = 2
    preview_policy: str = "normal"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "LivePreviewSettings":
        raw = dict(values or {})
        mode = str(raw.get("live_preview_mode", "fast")).strip().lower()
        if mode not in {"fast", "balanced", "accurate"}:
            mode = "fast"
        image_format = str(raw.get("live_preview_format", "webp")).strip().lower().lstrip(".")
        if image_format not in {"webp", "png"}:
            image_format = "webp"
        keep_history_value = raw.get("live_preview_keep_history", "current_job")
        if isinstance(keep_history_value, bool):
            keep_history = "current_job" if keep_history_value else "latest_only"
        else:
            keep_history = str(keep_history_value or "current_job").strip().lower()
        if keep_history not in {"current_job", "latest_only"}:
            keep_history = "current_job"
        preview_policy = str(raw.get("preview_policy", "normal") or "normal").strip().lower().replace("-", "_")
        if preview_policy not in {"normal", "suspend_on_pressure", "disable_during_hires", "disabled"}:
            preview_policy = "normal"
        return cls(
            enabled=_coerce_bool(raw.get("live_preview_enabled", True), True),
            mode=mode,
            interval=max(1, _coerce_int(raw.get("live_preview_interval", 1), 1)),
            width=min(640, max(128, _coerce_int(raw.get("live_preview_width", 384), 384))),
            image_format=image_format,
            keep_history=keep_history,
            batch_index=max(0, _coerce_int(raw.get("live_preview_batch_index", 0), 0)),
            quality=min(95, max(35, _coerce_int(raw.get("live_preview_quality", 78), 78))),
            adaptive_throttle=_coerce_bool(
                raw.get("live_preview_adaptive_throttle", True), True
            ),
            adaptive_target_ratio=min(1.50, max(0.20, _coerce_float(
                raw.get("live_preview_adaptive_target_ratio", 0.75), 0.75
            ))),
            adaptive_recovery_ratio=min(1.00, max(0.05, _coerce_float(
                raw.get("live_preview_adaptive_recovery_ratio", 0.40), 0.40
            ))),
            adaptive_max_interval=min(32, max(1, _coerce_int(
                raw.get("live_preview_adaptive_max_interval", 8), 8
            ))),
            adaptive_window=min(30, max(2, _coerce_int(
                raw.get("live_preview_adaptive_window", 6), 6
            ))),
            adaptive_suspend_on_overhead=_coerce_bool(
                raw.get("live_preview_adaptive_suspend_on_overhead", True), True
            ),
            adaptive_suspend_ratio=min(2.0, max(0.10, _coerce_float(
                raw.get("live_preview_adaptive_suspend_ratio", 0.55), 0.55
            ))),
            adaptive_suspend_min_work_ms=max(0.0, _coerce_float(
                raw.get("live_preview_adaptive_suspend_min_work_ms", 1000.0), 1000.0
            )),
            adaptive_suspend_min_samples=min(10, max(1, _coerce_int(
                raw.get("live_preview_adaptive_suspend_min_samples", 2), 2
            ))),
            preview_policy=preview_policy,
        )

    @property
    def extension(self) -> str:
        return ".png" if self.image_format == "png" else ".webp"

    @property
    def performance_warning(self) -> str | None:
        if self.mode != "accurate":
            return None
        return (
            "Accurate live preview uses the checkpoint VAE and may noticeably increase "
            "generation time, VRAM activity, and overhead at larger preview sizes."
        )


class AdaptivePreviewThrottle:
    """Adjust preview cadence from measured preview work versus sampler-step time."""

    def __init__(self, settings: LivePreviewSettings) -> None:
        self.enabled = bool(settings.adaptive_throttle)
        self.target_ratio = float(settings.adaptive_target_ratio)
        self.recovery_ratio = min(float(settings.adaptive_recovery_ratio), self.target_ratio)
        self.max_interval = max(1, int(settings.adaptive_max_interval))
        self.window = max(2, int(settings.adaptive_window))
        self.suspend_on_overhead = bool(settings.adaptive_suspend_on_overhead)
        self.suspend_ratio = float(settings.adaptive_suspend_ratio)
        self.suspend_min_work_ms = float(settings.adaptive_suspend_min_work_ms)
        self.suspend_min_samples = max(1, int(settings.adaptive_suspend_min_samples))
        self.effective_interval = 1
        self.ratios: deque[float] = deque(maxlen=self.window)
        self.sampler_step_ms: deque[float] = deque(maxlen=self.window)
        self.preview_work_ms: deque[float] = deque(maxlen=self.window)
        self.skipped_frames = 0
        self.adjustment_count = 0
        self.throttle_events: list[dict[str, Any]] = []
        self._recovery_streak = 0

    def should_process(self, step_number: int, total_steps: int) -> bool:
        if not self.enabled or self.effective_interval <= 1 or step_number >= total_steps:
            return True
        allowed = step_number % self.effective_interval == 0
        if not allowed:
            self.skipped_frames += 1
        return allowed

    def observe(
        self,
        *,
        step_number: int,
        sampler_step_ms: float | None,
        preview_work_ms: float,
    ) -> dict[str, Any]:
        if sampler_step_ms is None or sampler_step_ms <= 0 or preview_work_ms < 0:
            return self.snapshot()
        sampler_ms = float(sampler_step_ms)
        preview_ms = float(preview_work_ms)
        ratio = preview_ms / max(sampler_ms, 0.001)
        self.sampler_step_ms.append(sampler_ms)
        self.preview_work_ms.append(preview_ms)
        self.ratios.append(ratio)

        if self.enabled and len(self.ratios) >= 2:
            average_ratio = sum(self.ratios) / len(self.ratios)
            previous = self.effective_interval
            reason = ""
            if average_ratio >= self.target_ratio and previous < self.max_interval:
                multiplier = max(2, int(math.ceil(average_ratio / max(self.target_ratio, 0.01))))
                self.effective_interval = min(self.max_interval, max(previous + 1, previous * multiplier))
                self._recovery_streak = 0
                reason = "preview_work_near_sampler_step"
            elif average_ratio <= self.recovery_ratio and previous > 1:
                self._recovery_streak += 1
                if self._recovery_streak >= 2:
                    self.effective_interval = max(1, previous - 1)
                    self._recovery_streak = 0
                    reason = "preview_headroom_recovered"
            else:
                self._recovery_streak = 0

            if self.effective_interval != previous:
                self.adjustment_count += 1
                self.throttle_events.append({
                    "step": int(step_number),
                    "reason": reason,
                    "previous_interval": int(previous),
                    "new_interval": int(self.effective_interval),
                    "rolling_ratio": round(average_ratio, 4),
                    "target_ratio": round(self.target_ratio, 4),
                })
        return self.snapshot()

    def suspension_recommendation(self) -> dict[str, Any] | None:
        if not self.enabled or not self.suspend_on_overhead:
            return None
        if len(self.ratios) < self.suspend_min_samples:
            return None
        average_ratio = sum(self.ratios) / len(self.ratios)
        average_work_ms = sum(self.preview_work_ms) / len(self.preview_work_ms)
        if average_ratio < self.suspend_ratio or average_work_ms < self.suspend_min_work_ms:
            return None
        return {
            "reason": "live preview image work exceeded the adaptive performance budget",
            "rolling_ratio": round(average_ratio, 4),
            "average_preview_work_ms": round(average_work_ms, 3),
            "ratio_threshold": round(self.suspend_ratio, 4),
            "work_threshold_ms": round(self.suspend_min_work_ms, 3),
            "sample_count": len(self.ratios),
        }

    def snapshot(self) -> dict[str, Any]:
        average_ratio = sum(self.ratios) / len(self.ratios) if self.ratios else 0.0
        return {
            "adaptive_throttle_enabled": bool(self.enabled),
            "adaptive_effective_interval": int(self.effective_interval),
            "adaptive_target_ratio": round(self.target_ratio, 4),
            "adaptive_recovery_ratio": round(self.recovery_ratio, 4),
            "adaptive_rolling_overhead_ratio": round(average_ratio, 4),
            "adaptive_skipped_frames": int(self.skipped_frames),
            "adaptive_adjustment_count": int(self.adjustment_count),
        }

    def summary(self) -> dict[str, Any]:
        payload = self.snapshot()
        payload.update({
            "adaptive_max_interval": int(self.max_interval),
            "adaptive_window": int(self.window),
            "adaptive_throttle_events": list(self.throttle_events),
            "adaptive_suspend_on_overhead": bool(self.suspend_on_overhead),
            "adaptive_suspend_ratio": round(self.suspend_ratio, 4),
            "adaptive_suspend_min_work_ms": round(self.suspend_min_work_ms, 3),
            "adaptive_suspend_min_samples": int(self.suspend_min_samples),
        })
        return payload


class LivePreviewFrameWriter:
    """Synchronously decode and atomically persist current-job preview frames."""

    def __init__(
        self,
        root: str | Path,
        *,
        decoder: Callable[[torch.Tensor], torch.Tensor] | None,
        settings: LivePreviewSettings,
        event_callback: Callable[[dict[str, Any]], Any] | None = None,
        memory_event_callback: Callable[[str], Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.decoder = decoder
        self.settings = settings
        self.event_callback = event_callback
        self.memory_event_callback = memory_event_callback
        self.records: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None
        self.finalized = False
        self.last_step_number = 0
        self.frames_emitted = 0
        self.frames_failed = 0
        self.decode_time_total_ms = 0.0
        self.encode_time_total_ms = 0.0
        self.frame_latencies_ms: list[float] = []
        self.queue_latencies_ms: list[float] = []
        self.sampler_step_durations_ms: list[float] = []
        self.preview_overhead_ratios: list[float] = []
        self.image_decode_suspended = False
        self.image_decode_suspension_reason = ""
        self.image_decode_suspension_source = ""
        self.image_decode_decoder_released = False
        self.telemetry_only_frames = 0
        self.performance_suspension: dict[str, Any] | None = None
        self.adaptive_throttle = AdaptivePreviewThrottle(settings)
        if self.settings.preview_policy == "disabled":
            self.suspend_image_decode(
                reason="Preview policy disabled image decoding for this job.",
                source="policy_disabled",
            )


    def _memory_event(self, stage: str) -> None:
        callback = self.memory_event_callback
        if callable(callback):
            try:
                callback(stage)
            except Exception:
                pass

    def suspend_image_decode(
        self,
        *,
        reason: str,
        source: str = "automatic",
        release_decoder: bool = True,
    ) -> None:
        # Suspension is one-way for the remainder of this writer/job. Keeping the
        # first reason/source prevents later stages from implying that image
        # decoding was restored and suspended again.
        if self.image_decode_suspended:
            return
        self.image_decode_suspended = True
        self.image_decode_suspension_reason = str(reason or "memory policy")
        self.image_decode_suspension_source = str(source or "automatic")
        if release_decoder and self.decoder is not None:
            self.decoder = None
            self.image_decode_decoder_released = True

    def release_nonfinal_history(self) -> int:
        removed = 0
        for path in self.root.glob("step_*.*"):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        if self.latest is not None and self.latest.get("is_final"):
            self.records = [dict(self.latest)]
        else:
            self.records = []
        return removed

    def _write_telemetry_only_record(
        self,
        *,
        frame: LivePreviewFrame,
        step_number: int,
        decode_mode: str,
    ) -> dict[str, Any]:
        metadata = dict(frame.metadata or {})
        record = {
            "step": int(step_number),
            "step_index": max(0, int(step_number) - 1),
            "total_steps": max(1, int(frame.total_steps)),
            "progress_percent": min(max((int(step_number) / max(1, int(frame.total_steps))) * 100.0, 0.0), 100.0),
            "filename": "",
            "preview_path": "",
            "decode_mode": str(decode_mode),
            "preview_mode": self.settings.mode,
            "preview_width": int(self.settings.width),
            "image_format": self.settings.image_format,
            "batch_index": int(self.settings.batch_index),
            "image_width": 0,
            "image_height": 0,
            "sigma": frame.sigma,
            "model_timestep": frame.model_timestep,
            "sampler_name": metadata.get("sampler_name", ""),
            "scheduler_name": metadata.get("scheduler_name", ""),
            "requested_cfg_scale": metadata.get("requested_cfg_scale"),
            "effective_cfg_scale": metadata.get("effective_cfg_scale"),
            "guidance_mode": metadata.get("guidance_mode") or metadata.get("cfg_guidance_mode") or "flat",
            "cfg_rescale": metadata.get("cfg_rescale", 0.0),
            "cfg_rescale_applied": bool(metadata.get("cfg_rescale_applied", False)),
            "override_source": metadata.get("override_source", "base_request"),
            "transition_id": metadata.get("transition_id"),
            "is_final": False,
            "updated_at": _utc_now(),
            "preview_image_suspended": True,
            "preview_image_suspension_reason": self.image_decode_suspension_reason,
            "preview_image_suspension_source": self.image_decode_suspension_source,
            "preview_decoder_released": bool(self.image_decode_decoder_released),
            "cfg_telemetry_continues": True,
            "decode_time_ms": 0.0,
            "encode_time_ms": 0.0,
            "preview_work_time_ms": 0.0,
        }
        self.telemetry_only_frames += 1
        self.last_step_number = max(self.last_step_number, int(step_number))
        callback = self.event_callback
        if callable(callback):
            try:
                callback(record)
            except Exception:
                pass
        return record

    def _save_kwargs(self) -> dict[str, Any]:
        if self.settings.image_format == "webp":
            return {"format": "WEBP", "quality": self.settings.quality, "method": 0}
        return {"format": "PNG", "compress_level": 2}

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    def _atomic_image(self, path: Path, image: Image.Image) -> float:
        started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=path.stem + ".",
            suffix=path.suffix + ".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            image.save(temp_path, **self._save_kwargs())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return (time.perf_counter() - started) * 1000.0

    def _cleanup_old_steps(self, keep: Path) -> None:
        if self.settings.keep_history != "latest_only":
            return
        for path in self.root.glob(f"step_*{self.settings.extension}"):
            if path != keep:
                path.unlink(missing_ok=True)

    def _write_record(
        self,
        *,
        image: Image.Image,
        filename: str,
        step_number: int,
        total_steps: int,
        decode_mode: str,
        frame: LivePreviewFrame | None = None,
        is_final: bool = False,
        decode_time_ms: float = 0.0,
    ) -> dict[str, Any]:
        encode_started = time.perf_counter()
        resized = resize_preserving_aspect(image, self.settings.width)
        path = self.root / filename
        atomic_write_ms = self._atomic_image(path, resized)
        encode_time_ms = (time.perf_counter() - encode_started) * 1000.0
        self.decode_time_total_ms += max(0.0, float(decode_time_ms))
        self.encode_time_total_ms += max(0.0, float(encode_time_ms))
        if filename.startswith("step_"):
            self._cleanup_old_steps(path)

        metadata = dict(frame.metadata if frame is not None else {})
        now_monotonic = time.perf_counter()
        emitted_monotonic = metadata.get("preview_emitted_monotonic")
        processing_started = metadata.get("preview_processing_started_monotonic")
        try:
            emitted_monotonic = float(emitted_monotonic)
        except (TypeError, ValueError):
            emitted_monotonic = None
        try:
            processing_started = float(processing_started)
        except (TypeError, ValueError):
            processing_started = None
        queue_latency_ms = (
            max(0.0, (processing_started - emitted_monotonic) * 1000.0)
            if emitted_monotonic is not None and processing_started is not None
            else 0.0
        )
        frame_latency_ms = (
            max(0.0, (now_monotonic - emitted_monotonic) * 1000.0)
            if emitted_monotonic is not None
            else max(0.0, float(decode_time_ms) + float(encode_time_ms))
        )
        sampler_step_duration_ms = metadata.get("sampler_step_duration_ms")
        try:
            sampler_step_duration_ms = float(sampler_step_duration_ms)
            if sampler_step_duration_ms <= 0:
                sampler_step_duration_ms = None
        except (TypeError, ValueError):
            sampler_step_duration_ms = None
        preview_work_ms = max(0.0, float(decode_time_ms) + float(encode_time_ms))
        throttle_snapshot = self.adaptive_throttle.observe(
            step_number=int(step_number),
            sampler_step_ms=sampler_step_duration_ms,
            preview_work_ms=preview_work_ms,
        )
        suspension = None if is_final else self.adaptive_throttle.suspension_recommendation()
        if suspension is not None and not self.image_decode_suspended:
            self.performance_suspension = dict(suspension)
            self.suspend_image_decode(
                reason=(
                    "Live preview image decoding was suspended for the remainder of this job because "
                    f"preview work averaged {suspension['average_preview_work_ms']} ms "
                    f"({suspension['rolling_ratio']}x sampler-step time)."
                ),
                source="adaptive_performance",
                release_decoder=True,
            )
        overhead_ratio = (
            preview_work_ms / sampler_step_duration_ms
            if sampler_step_duration_ms is not None
            else 0.0
        )
        self.frame_latencies_ms.append(frame_latency_ms)
        self.queue_latencies_ms.append(queue_latency_ms)
        if sampler_step_duration_ms is not None:
            self.sampler_step_durations_ms.append(sampler_step_duration_ms)
            self.preview_overhead_ratios.append(overhead_ratio)
        record = {
            "step": int(step_number),
            "step_index": max(0, int(step_number) - 1),
            "total_steps": max(1, int(total_steps)),
            "progress_percent": min(max((int(step_number) / max(1, int(total_steps))) * 100.0, 0.0), 100.0),
            "filename": filename,
            "decode_mode": str(decode_mode),
            "preview_mode": self.settings.mode,
            "preview_width": int(self.settings.width),
            "image_format": self.settings.image_format,
            "batch_index": int(self.settings.batch_index),
            "image_width": int(resized.width),
            "image_height": int(resized.height),
            "sigma": None if frame is None else frame.sigma,
            "model_timestep": None if frame is None else frame.model_timestep,
            "sampler_name": metadata.get("sampler_name", ""),
            "scheduler_name": metadata.get("scheduler_name", ""),
            "requested_cfg_scale": metadata.get("requested_cfg_scale"),
            "effective_cfg_scale": metadata.get("effective_cfg_scale"),
            "guidance_mode": metadata.get("guidance_mode") or metadata.get("cfg_guidance_mode") or "flat",
            "cfg_rescale": metadata.get("cfg_rescale", 0.0),
            "cfg_rescale_applied": bool(metadata.get("cfg_rescale_applied", False)),
            "override_source": metadata.get("override_source", "base_request"),
            "transition_id": metadata.get("transition_id"),
            "is_final": bool(is_final),
            "updated_at": _utc_now(),
            "decode_time_ms": round(max(0.0, float(decode_time_ms)), 3),
            "encode_time_ms": round(max(0.0, float(encode_time_ms)), 3),
            "atomic_write_time_ms": round(max(0.0, float(atomic_write_ms)), 3),
            "preview_work_time_ms": round(preview_work_ms, 3),
            "queue_latency_ms": round(queue_latency_ms, 3),
            "frame_latency_ms": round(frame_latency_ms, 3),
            "sampler_step_duration_ms": (
                None if sampler_step_duration_ms is None else round(sampler_step_duration_ms, 3)
            ),
            "preview_to_sampler_ratio": round(overhead_ratio, 4),
            "adaptive_image_decode_suspended_after_frame": bool(suspension is not None),
            "adaptive_performance_suspension": dict(suspension or {}),
            **throttle_snapshot,
        }
        self.frames_emitted += 1
        self.latest = record
        self.records.append(record)
        self._atomic_json(self.root / "latest.json", record)
        self.last_step_number = max(self.last_step_number, int(step_number))
        callback = self.event_callback
        if callable(callback):
            try:
                callback({**record, "preview_path": str(path)})
            except Exception:
                pass
        return record

    def _decode_mode_for_step(self, step_number: int, total_steps: int) -> str | None:
        if self.settings.mode == "fast":
            if step_number % self.settings.interval != 0 and step_number != total_steps:
                return None
            return "fast"
        if self.settings.mode == "balanced":
            if step_number % self.settings.interval == 0 or step_number == total_steps:
                return "vae"
            return "fast"
        if step_number % self.settings.interval != 0 and step_number != total_steps:
            return None
        return "vae"

    def __call__(self, frame: LivePreviewFrame) -> dict[str, Any] | None:
        if not self.settings.enabled:
            return None
        step_number = int(frame.step_index) + 1
        decode_mode = self._decode_mode_for_step(step_number, frame.total_steps)
        if decode_mode is None:
            return None
        if not self.adaptive_throttle.should_process(step_number, frame.total_steps):
            return None
        if self.image_decode_suspended:
            return self._write_telemetry_only_record(
                frame=frame,
                step_number=step_number,
                decode_mode="telemetry_only",
            )

        frame.metadata.setdefault("preview_processing_started_monotonic", time.perf_counter())
        source = frame.predicted_x0 if frame.predicted_x0 is not None else frame.latent
        decode_started = time.perf_counter()
        self._memory_event("before_live_preview_decode")
        try:
            if decode_mode == "vae":
                image = decode_latent_to_pil_images(
                    self.decoder,
                    source,
                    batch_index=self.settings.batch_index,
                )[0]
            else:
                image = fast_latent_to_image(source, batch_index=self.settings.batch_index)
            decode_time_ms = (time.perf_counter() - decode_started) * 1000.0
            return self._write_record(
                image=image,
                filename=f"step_{step_number:03d}{self.settings.extension}",
                step_number=step_number,
                total_steps=frame.total_steps,
                decode_mode=decode_mode,
                frame=frame,
                decode_time_ms=decode_time_ms,
            )
        except Exception:
            self.frames_failed += 1
            raise
        finally:
            self._memory_event("after_live_preview_decode")
            source = None

    def write_final(self, images: torch.Tensor, *, total_steps: int) -> dict[str, Any]:
        decode_started = time.perf_counter()
        try:
            selected = select_batch_item(images.detach(), self.settings.batch_index)
            image = tensor_to_pil_images(selected)[0]
            decode_time_ms = (time.perf_counter() - decode_started) * 1000.0
            final_name = f"final{self.settings.extension}"
            record = self._write_record(
                image=image,
                filename=final_name,
                step_number=max(1, int(total_steps)),
                total_steps=max(1, int(total_steps)),
                decode_mode="final",
                is_final=True,
                decode_time_ms=decode_time_ms,
            )
        except Exception:
            self.frames_failed += 1
            raise
        # Replace the last approximate/interval frame with the true final output.
        step_name = f"step_{max(1, int(total_steps)):03d}{self.settings.extension}"
        resized = resize_preserving_aspect(image, self.settings.width)
        replacement_write_ms = self._atomic_image(self.root / step_name, resized)
        self.encode_time_total_ms += max(0.0, float(replacement_write_ms))
        record["final_step_replace_time_ms"] = round(
            max(0.0, float(replacement_write_ms)), 3
        )
        self.latest = record
        if self.records:
            self.records[-1] = record
        self._atomic_json(self.root / "latest.json", record)
        self._cleanup_old_steps(self.root / step_name)
        self.finalized = True
        return record

    def summary(self) -> dict[str, Any]:
        frame_count = max(self.frames_emitted, 1)
        latency_count = max(len(self.frame_latencies_ms), 1)
        sampler_count = max(len(self.sampler_step_durations_ms), 1)
        ratio_count = max(len(self.preview_overhead_ratios), 1)
        return {
            "enabled": self.settings.enabled,
            "mode": self.settings.mode,
            "interval": self.settings.interval,
            "width": self.settings.width,
            "format": self.settings.image_format,
            "keep_history": self.settings.keep_history,
            "batch_index": self.settings.batch_index,
            "root": str(self.root),
            "frame_count": len(self.records),
            "preview_frames_emitted": int(self.frames_emitted),
            "preview_frames_failed": int(self.frames_failed),
            "preview_decode_time_total_ms": round(self.decode_time_total_ms, 3),
            "preview_encode_time_total_ms": round(self.encode_time_total_ms, 3),
            "preview_work_time_total_ms": round(
                self.decode_time_total_ms + self.encode_time_total_ms, 3
            ),
            "preview_last_step": int(self.last_step_number),
            "preview_decode_time_average_ms": round(
                self.decode_time_total_ms / frame_count, 3
            ),
            "preview_encode_time_average_ms": round(
                self.encode_time_total_ms / frame_count, 3
            ),
            "frame_latency_average_ms": round(sum(self.frame_latencies_ms) / latency_count, 3),
            "frame_latency_p95_ms": round(_percentile(self.frame_latencies_ms, 0.95), 3),
            "frame_latency_max_ms": round(max(self.frame_latencies_ms, default=0.0), 3),
            "queue_latency_average_ms": round(sum(self.queue_latencies_ms) / latency_count, 3),
            "queue_latency_p95_ms": round(_percentile(self.queue_latencies_ms, 0.95), 3),
            "sampler_step_duration_average_ms": round(
                sum(self.sampler_step_durations_ms) / sampler_count, 3
            ),
            "sampler_step_duration_p95_ms": round(
                _percentile(self.sampler_step_durations_ms, 0.95), 3
            ),
            "preview_to_sampler_ratio_average": round(
                sum(self.preview_overhead_ratios) / ratio_count, 4
            ),
            "preview_to_sampler_ratio_p95": round(
                _percentile(self.preview_overhead_ratios, 0.95), 4
            ),
            **self.adaptive_throttle.summary(),
            "finalized": self.finalized,
            "preview_policy": self.settings.preview_policy,
            "image_decode_suspended": bool(self.image_decode_suspended),
            "image_decode_suspension_reason": self.image_decode_suspension_reason,
            "image_decode_suspension_source": self.image_decode_suspension_source,
            "preview_decoder_released": bool(self.image_decode_decoder_released),
            "preview_suspension_one_way_for_job": True,
            "telemetry_only_frames": int(self.telemetry_only_frames),
            "performance_suspension": dict(self.performance_suspension or {}),
            "cfg_telemetry_continues_during_preview_suspension": True,
            "latest": dict(self.latest or {}),
            "performance_warning": self.settings.performance_warning,
        }



class CoalescingLivePreviewWriter:
    """Background preview writer that keeps only the newest pending frame.

    The sampling loop hands frames to this adapter synchronously, but decoding and
    filesystem writes happen on a daemon worker thread. If the worker falls
    behind, older pending frames are replaced with the most recent frame so slow
    preview consumers never apply backpressure to sampling.
    """

    def __init__(
        self,
        writer: LivePreviewFrameWriter,
        *,
        warning_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.writer = writer
        self.warning_callback = warning_callback
        self._condition = threading.Condition()
        self._pending: LivePreviewFrame | None = None
        self._processing = False
        self._stopped = False
        self._latest_record: dict[str, Any] | None = None
        self.frames_enqueued = 0
        self.frames_processed = 0
        self.frames_replaced = 0
        self.worker_failures = 0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name='live-preview-writer',
            daemon=True,
        )
        self._worker.start()

    def _emit_warning(self, operation: str, exc: Exception, *, step_index: int | None = None) -> None:
        self.worker_failures += 1
        payload = {
            'operation': operation,
            'step_index': None if step_index is None else int(step_index),
            'error_type': type(exc).__name__,
            'error': str(exc),
            'failure_count': int(self.worker_failures),
            'disabled': False,
            'message': 'Live preview worker failure was isolated and generation continued.',
            'metadata': {},
        }
        callback = self.warning_callback
        if callable(callback):
            try:
                callback(payload)
            except Exception:
                pass

    def __call__(self, frame: LivePreviewFrame) -> None:
        with self._condition:
            if self._stopped:
                return
            self.frames_enqueued += 1
            if self._pending is not None:
                self.frames_replaced += 1
            self._pending = frame
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stopped and self._pending is None:
                    self._condition.wait()
                if self._stopped and self._pending is None:
                    self._condition.notify_all()
                    return
                frame = self._pending
                self._pending = None
                self._processing = True
            try:
                if frame is not None:
                    self._latest_record = self.writer(frame)
                    self.frames_processed += 1
            except Exception as exc:
                step_index = getattr(frame, 'step_index', None) if frame is not None else None
                self._emit_warning('worker_callback', exc, step_index=step_index)
            finally:
                with self._condition:
                    self._processing = False
                    self._condition.notify_all()

    def drain(self, timeout: float | None = None) -> None:
        with self._condition:
            if timeout is None:
                while self._pending is not None or self._processing:
                    self._condition.wait()
                return
            import time
            deadline = time.monotonic() + max(0.0, float(timeout))
            while self._pending is not None or self._processing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

    def suspend_image_decode(
        self,
        *,
        reason: str,
        source: str = "automatic",
        release_decoder: bool = True,
    ) -> None:
        # Discard queued image work and wait for any active decode before the
        # decoder reference is released. This prevents a hires transition or
        # pressure response from racing the background preview worker.
        with self._condition:
            if self._pending is not None:
                self._pending = None
                self.frames_replaced += 1
            while self._processing:
                self._condition.wait()
            self.writer.suspend_image_decode(
                reason=reason,
                source=source,
                release_decoder=release_decoder,
            )
            self._condition.notify_all()

    def release_nonfinal_history(self) -> int:
        self.drain()
        return self.writer.release_nonfinal_history()

    def write_final(self, images: torch.Tensor, *, total_steps: int) -> dict[str, Any]:
        self.drain()
        return self.writer.write_final(images, total_steps=total_steps)

    def summary(self) -> dict[str, Any]:
        self.drain()
        payload = dict(self.writer.summary())
        payload.update({
            'queue_mode': 'latest_only',
            'frames_enqueued': int(self.frames_enqueued),
            'frames_processed': int(self.frames_processed),
            'frames_replaced': int(self.frames_replaced),
            'coalesced_frames': int(self.frames_replaced),
            'worker_failures': int(self.worker_failures),
        })
        return payload

    def close(self) -> None:
        self.drain()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._worker.join(timeout=1.0)


def create_live_preview_writer(
    values: Mapping[str, Any] | None,
    *,
    decoder: Callable[[torch.Tensor], torch.Tensor] | None,
) -> LivePreviewFrameWriter | CoalescingLivePreviewWriter | None:
    raw = dict(values or {})
    settings = LivePreviewSettings.from_mapping(raw)
    root = raw.get("live_preview_root")
    if not settings.enabled or not root:
        return None
    event_callback = raw.get('live_preview_event_callback')
    warning_callback = raw.get('live_preview_warning_callback')
    memory_event_callback = raw.get('live_preview_memory_event_callback')
    writer = LivePreviewFrameWriter(
        root,
        decoder=decoder,
        settings=settings,
        event_callback=event_callback if callable(event_callback) else None,
        memory_event_callback=(
            memory_event_callback if callable(memory_event_callback) else None
        ),
    )
    if _coerce_bool(raw.get('live_preview_async', True), True):
        return CoalescingLivePreviewWriter(
            writer,
            warning_callback=warning_callback if callable(warning_callback) else None,
        )
    return writer
