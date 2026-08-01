from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import json
import math
import os
import tempfile


@dataclass
class TraceRequest:
    enabled: bool = True
    export_json: bool = False
    export_csv: bool = False
    export_txt_summary: bool = True
    keep_in_memory: bool = True
    cleanup_after_export: bool = False

    capture_latents: bool = False
    capture_latent_every_n: int = 0
    capture_first_step: bool = True
    capture_last_step: bool = True

    capture_predicted_x0_previews: bool = False
    predicted_x0_preview_steps: tuple[int, ...] = ()

    run_name: str = "generation_run"
    output_dir: str = "modules/pipeline/image_generation_data"


@dataclass
class StepTraceRecord:
    step_index: int
    sigma: Optional[float] = None
    sigma_next: Optional[float] = None
    sigma_delta: Optional[float] = None
    timestep: Optional[float] = None
    latent_norm_before: Optional[float] = None
    latent_norm_after: Optional[float] = None
    noise_pred_norm: Optional[float] = None
    guided_noise_norm: Optional[float] = None
    cfg_scale: Optional[float] = None
    requested_cfg_scale: Optional[float] = None
    effective_cfg_scale: Optional[float] = None
    guidance_owner: Optional[str] = None
    unconditional_output: Optional[Dict[str, Any]] = None
    conditional_output: Optional[Dict[str, Any]] = None
    guidance_delta: Optional[Dict[str, Any]] = None
    guided_output: Optional[Dict[str, Any]] = None
    predicted_x0: Optional[Dict[str, Any]] = None
    latent_before: Optional[Dict[str, Any]] = None
    latent_after: Optional[Dict[str, Any]] = None
    stopping_candidate: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSummary:
    run_name: str
    total_steps_recorded: int
    requested_steps: Optional[int] = None
    effective_steps: Optional[int] = None
    stopping_index: Optional[int] = None
    sampler_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    schedule_extra: Dict[str, Any] = field(default_factory=dict)
    exported_files: Dict[str, str] = field(default_factory=dict)
    predicted_x0_preview_steps: list[int] = field(default_factory=list)
    predicted_x0_preview_errors: list[dict[str, Any]] = field(default_factory=list)
    guidance_owners: list[str] = field(default_factory=list)
    requested_cfg_scales: list[float] = field(default_factory=list)
    effective_cfg_scales: list[float] = field(default_factory=list)
    trace_schema: str = "image-gen-step-trace-v2"


class StepTraceRecorder:
    """
    Shared runtime trace recorder for pipeline-managed, sampler-fed step logging.

    Intended flow:
    1. Pipeline creates one recorder per generation run.
    2. Pipeline passes recorder into the sampler.
    3. Sampler calls record_step(...) inside its iteration loop.
    4. Pipeline calls export_requested_artifacts(...) after sampling.

    Notes:
    - This class is intentionally in-memory first.
    - CSV export is optional and disabled by default.
    - Full latent retention is opt-in and should be used sparingly.
    """

    def __init__(
        self,
        request: Optional[TraceRequest] = None,
        schedule_extra: Optional[Dict[str, Any]] = None,
        sampler_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
    ) -> None:
        self.request = request or TraceRequest()
        self.schedule_extra: Dict[str, Any] = dict(schedule_extra or {})
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name

        self.records: list[StepTraceRecord] = []
        self.latents: dict[int, Any] = {}
        self.predicted_x0_latents: dict[int, Any] = {}
        self.predicted_x0_preview_errors: list[dict[str, Any]] = []
        self.exported_files: Dict[str, str] = {}

        self.requested_steps: Optional[int] = self._coerce_int(self.schedule_extra.get("requested_steps"))
        self.effective_steps: Optional[int] = self._coerce_int(self.schedule_extra.get("effective_steps"))
        self.stopping_index: Optional[int] = self._coerce_int(self.schedule_extra.get("predicted_stop_step"))

    @property
    def enabled(self) -> bool:
        return bool(self.request.enabled)

    def recovery_checkpoint(self) -> dict[str, Any]:
        """Capture list/dict boundaries before a retryable sampling pass."""

        return {
            "records_length": len(self.records),
            "latent_keys": sorted(int(key) for key in self.latents),
            "predicted_x0_keys": sorted(
                int(key) for key in self.predicted_x0_latents
            ),
            "preview_error_length": len(self.predicted_x0_preview_errors),
            "exported_files": dict(self.exported_files),
        }

    def restore_recovery_checkpoint(
        self, checkpoint: Optional[Dict[str, Any]]
    ) -> dict[str, Any]:
        """Remove partial trace data written by a failed sampling attempt."""

        marker = dict(checkpoint or {})
        records_length = max(0, int(marker.get("records_length", 0)))
        del self.records[records_length:]

        latent_keys = {int(key) for key in marker.get("latent_keys", [])}
        predicted_keys = {
            int(key) for key in marker.get("predicted_x0_keys", [])
        }
        self.latents = {
            int(key): value
            for key, value in self.latents.items()
            if int(key) in latent_keys
        }
        self.predicted_x0_latents = {
            int(key): value
            for key, value in self.predicted_x0_latents.items()
            if int(key) in predicted_keys
        }
        preview_error_length = max(
            0, int(marker.get("preview_error_length", 0))
        )
        del self.predicted_x0_preview_errors[preview_error_length:]
        self.exported_files = dict(marker.get("exported_files") or {})
        return {
            "records_length": len(self.records),
            "latent_count": len(self.latents),
            "predicted_x0_count": len(self.predicted_x0_latents),
            "preview_error_count": len(self.predicted_x0_preview_errors),
        }

    def configure_predicted_x0_preview_steps(self, step_indices: Iterable[int]) -> None:
        cleaned = sorted({int(value) for value in step_indices if int(value) >= 0})
        self.request.predicted_x0_preview_steps = tuple(cleaned)

    def add_predicted_x0_preview_error(
        self,
        *,
        step_index: int,
        error: BaseException,
    ) -> None:
        self.predicted_x0_preview_errors.append(
            {
                "step_index": int(step_index),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def seed_schedule_metadata(self, schedule_extra: Optional[Dict[str, Any]]) -> None:
        if not schedule_extra:
            return
        self.schedule_extra.update(schedule_extra)

        if self.requested_steps is None:
            self.requested_steps = self._coerce_int(schedule_extra.get("requested_steps"))
        if self.effective_steps is None:
            self.effective_steps = self._coerce_int(schedule_extra.get("effective_steps"))
        if self.stopping_index is None:
            self.stopping_index = self._coerce_int(schedule_extra.get("predicted_stop_step"))

    def record_step(
        self,
        *,
        step_index: int,
        sigma: Any = None,
        sigma_next: Any = None,
        timestep: Any = None,
        latent_before: Any = None,
        latent_after: Any = None,
        noise_pred: Any = None,
        guided_noise: Any = None,
        cfg_scale: Any = None,
        requested_cfg_scale: Any = None,
        effective_cfg_scale: Any = None,
        guidance_owner: Optional[str] = None,
        unconditional_output: Any = None,
        conditional_output: Any = None,
        guidance_delta: Any = None,
        guided_output: Any = None,
        predicted_x0: Any = None,
        stopping_candidate: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
        latent_snapshot: Any = None,
        predicted_x0_snapshot: Any = None,
    ) -> None:
        if not self.enabled:
            return

        sigma_value = self._coerce_float(sigma)
        sigma_next_value = self._coerce_float(sigma_next)
        sigma_delta = None
        if sigma_value is not None and sigma_next_value is not None:
            sigma_delta = sigma_next_value - sigma_value

        requested_cfg = self._coerce_float(
            requested_cfg_scale if requested_cfg_scale is not None else cfg_scale
        )
        effective_cfg = self._coerce_float(
            effective_cfg_scale if effective_cfg_scale is not None else requested_cfg
        )
        record = StepTraceRecord(
            step_index=int(step_index),
            sigma=sigma_value,
            sigma_next=sigma_next_value,
            sigma_delta=sigma_delta,
            timestep=self._coerce_float(timestep),
            latent_norm_before=self._tensor_norm(latent_before),
            latent_norm_after=self._tensor_norm(latent_after),
            noise_pred_norm=self._tensor_norm(noise_pred),
            guided_noise_norm=self._tensor_norm(guided_noise),
            cfg_scale=requested_cfg,
            requested_cfg_scale=requested_cfg,
            effective_cfg_scale=effective_cfg,
            guidance_owner=str(guidance_owner) if guidance_owner else None,
            unconditional_output=self._tensor_statistics(unconditional_output),
            conditional_output=self._tensor_statistics(conditional_output),
            guidance_delta=self._tensor_statistics(guidance_delta),
            guided_output=self._tensor_statistics(
                guided_output if guided_output is not None else guided_noise
            ),
            predicted_x0=self._tensor_statistics(predicted_x0),
            latent_before=self._tensor_statistics(latent_before),
            latent_after=self._tensor_statistics(latent_after),
            stopping_candidate=stopping_candidate,
            extra=dict(extra or {}),
        )
        self.records.append(record)

        snapshot = latent_snapshot if latent_snapshot is not None else latent_after
        if self._should_capture_latent(step_index):
            self.latents[int(step_index)] = self._detach_for_storage(snapshot)

        if self._should_capture_predicted_x0(step_index):
            stored = self._detach_for_storage(predicted_x0_snapshot)
            if stored is not None:
                self.predicted_x0_latents[int(step_index)] = stored

    def set_runtime_summary(
        self,
        *,
        requested_steps: Optional[int] = None,
        effective_steps: Optional[int] = None,
        stopping_index: Optional[int] = None,
    ) -> None:
        if requested_steps is not None:
            self.requested_steps = int(requested_steps)
        if effective_steps is not None:
            self.effective_steps = int(effective_steps)
        if stopping_index is not None:
            self.stopping_index = int(stopping_index)

    def build_summary(self) -> TraceSummary:
        return TraceSummary(
            run_name=self.request.run_name,
            total_steps_recorded=len(self.records),
            requested_steps=self.requested_steps,
            effective_steps=self.effective_steps,
            stopping_index=self.stopping_index,
            sampler_name=self.sampler_name,
            scheduler_name=self.scheduler_name,
            schedule_extra=dict(self.schedule_extra),
            exported_files=dict(self.exported_files),
            predicted_x0_preview_steps=sorted(self.predicted_x0_latents),
            predicted_x0_preview_errors=list(self.predicted_x0_preview_errors),
            guidance_owners=sorted(
                {record.guidance_owner for record in self.records if record.guidance_owner}
            ),
            requested_cfg_scales=sorted(
                {record.requested_cfg_scale for record in self.records if record.requested_cfg_scale is not None}
            ),
            effective_cfg_scales=sorted(
                {record.effective_cfg_scale for record in self.records if record.effective_cfg_scale is not None}
            ),
        )

    def export_requested_artifacts(self) -> Dict[str, str]:
        if not self.enabled:
            return {}

        output_dir = Path(self.request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = self._safe_run_name(self.request.run_name)

        if self.request.export_json:
            json_path = output_dir / f"{base_name}_step_trace.json"
            self._export_json(json_path)
            self.exported_files["json"] = str(json_path)

        if self.request.export_csv:
            csv_path = output_dir / f"{base_name}_step_trace.csv"
            self._export_csv(csv_path)
            self.exported_files["csv"] = str(csv_path)

        if self.request.export_txt_summary:
            txt_path = output_dir / f"{base_name}_trace_summary.txt"
            self._export_txt_summary(txt_path)
            self.exported_files["summary"] = str(txt_path)

        if self.request.cleanup_after_export:
            self.cleanup()

        return dict(self.exported_files)

    def cleanup(self) -> None:
        self.records.clear()
        self.latents.clear()
        self.predicted_x0_latents.clear()
        self.predicted_x0_preview_errors.clear()
        if not self.request.keep_in_memory:
            self.schedule_extra.clear()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        # Import lazily to avoid a module-import cycle: DiagnosticsSystem imports
        # StepTraceRecorder while the diagnostics package is still initializing.
        from image_gen.systems.diagnostics.serialization import json_safe

        return json_safe(value)

    def _export_json(self, path: Path) -> None:
        payload = self._json_safe(
            {
                "format": "image-gen-step-trace-v2",
                "summary": asdict(self.build_summary()),
                "records": [asdict(record) for record in self.records],
            }
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _export_csv(self, path: Path) -> None:
        import csv

        fieldnames = [
            "step_index",
            "sigma",
            "sigma_next",
            "sigma_delta",
            "timestep",
            "latent_norm_before",
            "latent_norm_after",
            "noise_pred_norm",
            "guided_noise_norm",
            "cfg_scale",
            "requested_cfg_scale",
            "effective_cfg_scale",
            "guidance_owner",
            "unconditional_output_json",
            "conditional_output_json",
            "guidance_delta_json",
            "guided_output_json",
            "predicted_x0_json",
            "latent_before_json",
            "latent_after_json",
            "stopping_candidate",
            "extra_json",
        ]

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "step_index": record.step_index,
                        "sigma": record.sigma,
                        "sigma_next": record.sigma_next,
                        "sigma_delta": record.sigma_delta,
                        "timestep": record.timestep,
                        "latent_norm_before": record.latent_norm_before,
                        "latent_norm_after": record.latent_norm_after,
                        "noise_pred_norm": record.noise_pred_norm,
                        "guided_noise_norm": record.guided_noise_norm,
                        "cfg_scale": record.cfg_scale,
                        "requested_cfg_scale": record.requested_cfg_scale,
                        "effective_cfg_scale": record.effective_cfg_scale,
                        "guidance_owner": record.guidance_owner,
                        "unconditional_output_json": json.dumps(self._json_safe(record.unconditional_output), ensure_ascii=False),
                        "conditional_output_json": json.dumps(self._json_safe(record.conditional_output), ensure_ascii=False),
                        "guidance_delta_json": json.dumps(self._json_safe(record.guidance_delta), ensure_ascii=False),
                        "guided_output_json": json.dumps(self._json_safe(record.guided_output), ensure_ascii=False),
                        "predicted_x0_json": json.dumps(self._json_safe(record.predicted_x0), ensure_ascii=False),
                        "latent_before_json": json.dumps(self._json_safe(record.latent_before), ensure_ascii=False),
                        "latent_after_json": json.dumps(self._json_safe(record.latent_after), ensure_ascii=False),
                        "stopping_candidate": record.stopping_candidate,
                        "extra_json": json.dumps(
                            self._json_safe(record.extra),
                            ensure_ascii=False,
                        ),
                    }
                )

    def _export_txt_summary(self, path: Path) -> None:
        summary = self.build_summary()
        lines = [
            f"run_name: {summary.run_name}",
            f"sampler_name: {summary.sampler_name}",
            f"scheduler_name: {summary.scheduler_name}",
            f"requested_steps: {summary.requested_steps}",
            f"effective_steps: {summary.effective_steps}",
            f"stopping_index: {summary.stopping_index}",
            f"total_steps_recorded: {summary.total_steps_recorded}",
            f"guidance_owners: {summary.guidance_owners}",
            f"requested_cfg_scales: {summary.requested_cfg_scales}",
            f"effective_cfg_scales: {summary.effective_cfg_scales}",
            "schedule_extra:",
        ]

        if summary.schedule_extra:
            for key, value in summary.schedule_extra.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  <none>")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _should_capture_latent(self, step_index: int) -> bool:
        if not self.request.capture_latents:
            return False

        if step_index == 0 and self.request.capture_first_step:
            return True

        if (
            self.effective_steps is not None
            and step_index == max(0, self.effective_steps - 1)
            and self.request.capture_last_step
        ):
            return True

        every_n = int(self.request.capture_latent_every_n or 0)
        return every_n > 0 and step_index % every_n == 0

    def _should_capture_predicted_x0(self, step_index: int) -> bool:
        if not self.request.capture_predicted_x0_previews:
            return False
        return int(step_index) in set(self.request.predicted_x0_preview_steps)

    @staticmethod
    def _safe_run_name(value: str) -> str:
        cleaned = value.strip().replace(" ", "_")
        return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"}) or "generation_run"

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_norm(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "norm"):
                norm_value = value.norm()
                if hasattr(norm_value, "item"):
                    norm_value = norm_value.item()
                return float(norm_value)
        except Exception:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_statistics(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        try:
            if not hasattr(value, "detach"):
                return None
            from image_gen.systems.diagnostics.serialization import tensor_summary

            return tensor_summary(value, include_statistics=True)
        except Exception:
            return None

    @staticmethod
    def _detach_for_storage(value: Any) -> Any:
        if value is None:
            return None

        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return value
        except Exception:
            return None


def create_step_trace_recorder(
    *,
    enabled: bool = True,
    run_name: str = "generation_run",
    output_dir: str = "modules/pipeline/image_generation_data",
    schedule_extra: Optional[Dict[str, Any]] = None,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    export_json: bool = False,
    export_csv: bool = False,
    export_txt_summary: bool = True,
    capture_latents: bool = False,
    capture_latent_every_n: int = 0,
    cleanup_after_export: bool = False,
    capture_predicted_x0_previews: bool = False,
    predicted_x0_preview_steps: Iterable[int] = (),
) -> StepTraceRecorder:
    request = TraceRequest(
        enabled=enabled,
        export_json=export_json,
        export_csv=export_csv,
        export_txt_summary=export_txt_summary,
        run_name=run_name,
        output_dir=output_dir,
        capture_latents=capture_latents,
        capture_latent_every_n=capture_latent_every_n,
        cleanup_after_export=cleanup_after_export,
        capture_predicted_x0_previews=capture_predicted_x0_previews,
        predicted_x0_preview_steps=tuple(int(value) for value in predicted_x0_preview_steps),
    )
    return StepTraceRecorder(
        request=request,
        schedule_extra=schedule_extra,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
    )
