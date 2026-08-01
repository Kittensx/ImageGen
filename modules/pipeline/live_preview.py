from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

import torch


@dataclass
class LivePreviewFrame:
    """Sampler-independent live-preview frame payload.

    The payload intentionally carries tensors directly so later phases can decide
    whether to decode, encode, cache, or stream them. Tensors are detached and
    cloned before delivery so preview consumers cannot mutate sampler-owned
    working state.
    """

    step_index: int
    total_steps: int
    latent: torch.Tensor
    predicted_x0: torch.Tensor | None = None
    sigma: float | None = None
    model_timestep: float | int | None = None
    batch_index: int = 0
    progress_percent: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "total_steps": int(self.total_steps),
            "latent": self.latent,
            "predicted_x0": self.predicted_x0,
            "sigma": self.sigma,
            "model_timestep": self.model_timestep,
            "batch_index": int(self.batch_index),
            "progress_percent": float(self.progress_percent),
            "metadata": dict(self.metadata),
        }


class LivePreviewSink:
    """Failure-isolated sink for per-step preview frames."""

    def __init__(
        self,
        callback: Callable[[LivePreviewFrame], Any] | None = None,
        *,
        enabled: bool = True,
        sampler_name: str | None = None,
        scheduler_name: str | None = None,
        warning_callback: Callable[[dict[str, Any]], Any] | None = None,
        max_failures: int = 2,
        clone_tensors: bool = True,
    ) -> None:
        self.callback = callback
        self.enabled = bool(enabled)
        self.sampler_name = str(sampler_name or "")
        self.scheduler_name = str(scheduler_name or "")
        self.warning_callback = warning_callback
        self.max_failures = max(1, int(max_failures))
        self.clone_tensors = bool(clone_tensors)
        self.failure_count = 0
        self.disabled = False
        self.emitted_frames = 0
        self.warnings: list[dict[str, Any]] = []
        self.latest_frame: LivePreviewFrame | None = None
        self._last_step_observed_at: float | None = None

    @staticmethod
    def _scalar(value: Any) -> float | int | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            value = value.detach().flatten()[0].item()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _tensor_payload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        detached = tensor.detach()
        if self.clone_tensors:
            detached = detached.clone()
        return detached

    def _warn(self, warning: dict[str, Any]) -> None:
        payload = dict(warning)
        self.warnings.append(payload)
        callback = self.warning_callback
        if callable(callback):
            try:
                callback(payload)
            except Exception:
                # Preview warnings must never escalate into generation failures.
                pass

    def _handle_failure(
        self,
        operation: str,
        exc: Exception,
        *,
        step_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.failure_count += 1
        warning = {
            "operation": str(operation),
            "step_index": int(step_index),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_count": int(self.failure_count),
            "disabled": False,
            "message": "Live preview sink failure was isolated and generation continued.",
            "metadata": dict(metadata or {}),
        }
        if self.failure_count >= self.max_failures:
            self.disabled = True
            warning["disabled"] = True
            warning["message"] = (
                "Live preview sink disabled itself after repeated failures; generation continued."
            )
        self._warn(warning)

    def on_step(
        self,
        *,
        step_index: int,
        total_steps: int,
        latent: torch.Tensor,
        predicted_x0: torch.Tensor | None = None,
        sigma: Any = None,
        model_timestep: Any = None,
        batch_index: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.disabled or self.callback is None:
            return

        step_number = int(step_index) + 1
        total = max(1, int(total_steps))
        progress_percent = min(max((step_number / total) * 100.0, 0.0), 100.0)
        observed_at = time.perf_counter()
        frame_metadata = dict(metadata or {})
        if self._last_step_observed_at is not None:
            frame_metadata.setdefault(
                "sampler_step_duration_ms",
                max(0.0, (observed_at - self._last_step_observed_at) * 1000.0),
            )
        self._last_step_observed_at = observed_at
        frame_metadata.setdefault("preview_emitted_monotonic", observed_at)
        frame_metadata.setdefault("preview_emitted_unix", time.time())
        if self.sampler_name and not frame_metadata.get("sampler_name"):
            frame_metadata["sampler_name"] = self.sampler_name
        if self.scheduler_name and not frame_metadata.get("scheduler_name"):
            frame_metadata["scheduler_name"] = self.scheduler_name

        try:
            frame = LivePreviewFrame(
                step_index=int(step_index),
                total_steps=total,
                latent=self._tensor_payload(latent),
                predicted_x0=self._tensor_payload(predicted_x0),
                sigma=self._scalar(sigma),
                model_timestep=self._scalar(model_timestep),
                batch_index=int(batch_index),
                progress_percent=progress_percent,
                metadata=frame_metadata,
            )
        except Exception as exc:
            self._handle_failure(
                "prepare_frame",
                exc,
                step_index=int(step_index),
                metadata=frame_metadata,
            )
            return

        try:
            self.callback(frame)
            self.emitted_frames += 1
            self.latest_frame = frame
        except Exception as exc:
            self._handle_failure(
                "callback",
                exc,
                step_index=int(step_index),
                metadata=frame_metadata,
            )


def _supports_live_preview_sink(candidate: Any) -> bool:
    return candidate is not None and callable(getattr(candidate, "on_step", None))


def get_live_preview_sink(state: Any | None) -> Any | None:
    if state is None:
        return None
    extra = getattr(state, "extra", None)
    if not isinstance(extra, dict):
        return None
    sink = extra.get("live_preview_sink")
    return sink if _supports_live_preview_sink(sink) else None


def build_live_preview_sink(
    *,
    state: Any | None,
    request: Any | None = None,
    schedule: Any | None = None,
    warning_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> Any | None:
    """Create or adopt the per-run live preview sink on shared state.

    The pipeline owns sink construction. Direct sampler tests may still inject an
    already-built sink into ``state.extra['live_preview_sink']``.
    """

    if state is None:
        return None
    extra = getattr(state, "extra", None)
    if not isinstance(extra, dict):
        return None

    existing = extra.get("live_preview_sink")
    callback = extra.get("live_preview_callback")
    enabled = bool(extra.get("live_preview_enabled", True))
    factory = extra.get("live_preview_sink_factory")
    max_failures = extra.get("live_preview_max_failures", 2)
    clone_tensors = extra.get("live_preview_clone_tensors", True)

    sampler_name = getattr(request, "sampler_name", None) or extra.get("sampler_name")
    scheduler_name = getattr(request, "scheduler_name", None) or extra.get("scheduler_name")
    if not scheduler_name and schedule is not None:
        schedule_extra = getattr(schedule, "extra", None)
        if isinstance(schedule_extra, Mapping):
            scheduler_name = schedule_extra.get("scheduler_name")

    sink = None
    if callable(factory):
        sink = factory(
            state=state,
            request=request,
            schedule=schedule,
            warning_callback=warning_callback,
        )
    elif callable(callback):
        sink = LivePreviewSink(
            callback=callback,
            enabled=enabled,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            warning_callback=warning_callback,
            max_failures=max_failures,
            clone_tensors=clone_tensors,
        )
    elif _supports_live_preview_sink(existing):
        sink = existing

    if sink is not None:
        extra["live_preview_sink"] = sink
        extra["live_preview_warnings"] = getattr(sink, "warnings", [])
    else:
        extra.pop("live_preview_sink", None)
        extra.setdefault("live_preview_warnings", [])
    return sink
