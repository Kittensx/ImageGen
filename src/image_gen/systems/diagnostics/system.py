from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import torch

from image_gen.contracts import GenerationRequest, SamplerOutput, SchedulerOutput
from image_gen.systems.diagnostics.failure_bundle import write_failure_bundle
from image_gen.systems.diagnostics.models import (
    DiagnosticConfig,
    DiagnosticEvent,
    DiagnosticSession,
    PipelineStageError,
    SEVERITY_LEVEL,
    StageTiming,
)
from image_gen.systems.diagnostics.serialization import json_safe, tensor_summary
from modules.pipeline.step_trace_recorder import create_step_trace_recorder
from modules.pipeline.live_preview_decode import decode_latent_to_pil_images


T = TypeVar("T")


class DiagnosticsSystem:
    """Structured events, timings, tensor summaries, traces, and failures.

    The default configuration records enough in memory to create a failure
    bundle while emitting no informational console chatter. Nothing is written
    on a successful normal run unless event export or sampler tracing is enabled.
    """

    def __init__(
        self,
        config: DiagnosticConfig | Mapping[str, Any] | None = None,
        *,
        project_context: Any | None = None,
        console: Any | None = None,
    ) -> None:
        self.project_context = project_context
        self.console = console or sys.stderr
        if isinstance(config, DiagnosticConfig):
            self.config = config
        else:
            configured_root = None
            if project_context is not None:
                configured_root = getattr(project_context, "diagnostics_root", None)
                if configured_root is None:
                    project_root = Path(getattr(project_context, "project_root", Path.cwd()))
                    configured_root = project_root / "artifacts" / "diagnostics"
            self.config = DiagnosticConfig.from_mapping(config, artifacts_root=configured_root)

    @classmethod
    def from_project_context(cls, project_context: Any) -> "DiagnosticsSystem":
        config = getattr(project_context, "config", {}) or {}
        raw = config.get("diagnostics") or {} if isinstance(config, Mapping) else {}
        return cls(raw, project_context=project_context)

    def _request_config(self, request: GenerationRequest) -> DiagnosticConfig:
        override = getattr(request, "diagnostics", None) or {}
        return self.config.merged(override)

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    def start(
        self,
        request: GenerationRequest,
        schedule: SchedulerOutput | None = None,
        *,
        effective_config: Mapping[str, Any] | None = None,
        request_extras: Mapping[str, Any] | None = None,
        components: Any | None = None,
    ) -> DiagnosticSession:
        session = DiagnosticSession(
            run_id=self._new_run_id(),
            config=self._request_config(request),
            request=request,
            effective_config=dict(json_safe(effective_config or {}, redact_secrets=True)),
            request_extras=dict(json_safe(request_extras or {}, redact_secrets=True)),
        )
        session.schedule_report = {
            "requested_scheduler": getattr(request, "scheduler_name", None),
            "requested_steps": getattr(request, "steps", None),
            "scheduler_kwargs": json_safe(
                getattr(request, "scheduler_kwargs", {}) or {}, redact_secrets=True
            ),
        }
        session.sampler_report = {
            "requested_sampler": getattr(request, "sampler_name", None),
            "sampler_kwargs": json_safe(
                getattr(request, "sampler_kwargs", {}) or {}, redact_secrets=True
            ),
        }
        if components is not None:
            self.update_components(session, components)
        if schedule is not None:
            self.update_schedule(session, schedule)
        self.emit(session, "DEBUG", "runtime", "start", "diagnostic session started")
        self._start_sampler_trace(session)
        return session

    def emit(
        self,
        session: DiagnosticSession,
        severity: str,
        system: str,
        operation: str,
        message: str,
        **details: Any,
    ) -> DiagnosticEvent:
        severity_name = severity.upper()
        if severity_name not in SEVERITY_LEVEL:
            raise ValueError(f"Unknown diagnostic severity: {severity}")
        event = DiagnosticEvent(
            sequence=len(session.events) + 1,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=(time.perf_counter() - session.started_perf) * 1000.0,
            run_id=session.run_id,
            severity=severity_name,
            system=str(system),
            operation=str(operation),
            message=str(message),
            details=dict(json_safe(details, redact_secrets=True)),
        )
        session.events.append(event)
        if SEVERITY_LEVEL[severity_name] >= session.config.console_level:
            detail_text = " ".join(
                f"{key}={json.dumps(value, ensure_ascii=False)}"
                for key, value in event.details.items()
            )
            line = (
                f"[{severity_name}] run_id={session.run_id} system={event.system} "
                f"operation={event.operation} message={json.dumps(event.message)}"
            )
            if detail_text:
                line += " " + detail_text
            print(line, file=self.console)
        return event

    def run_stage(
        self,
        session: DiagnosticSession,
        system: str,
        operation: str,
        action: Callable[[], T],
    ) -> T:
        occurrence = session.next_occurrence(system, operation)
        self.emit(
            session,
            "DEBUG",
            system,
            operation,
            "stage started",
            occurrence=occurrence,
        )
        started = time.perf_counter()
        try:
            result = action()
        except PipelineStageError:
            duration_ms = (time.perf_counter() - started) * 1000.0
            session.timings.append(
                StageTiming(system, operation, occurrence, "failed_nested", duration_ms)
            )
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            session.timings.append(StageTiming(system, operation, occurrence, "failed", duration_ms))
            self.emit(
                session,
                "ERROR",
                system,
                operation,
                "stage failed",
                occurrence=occurrence,
                duration_ms=round(duration_ms, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            bundle_path = self._capture_failure(session, system, operation, exc)
            raise PipelineStageError(
                run_id=session.run_id,
                system=system,
                operation=operation,
                cause=exc,
                bundle_path=bundle_path,
            ) from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        session.timings.append(StageTiming(system, operation, occurrence, "passed", duration_ms))
        self.emit(
            session,
            "INFO",
            system,
            operation,
            "stage completed",
            occurrence=occurrence,
            duration_ms=round(duration_ms, 3),
        )
        return result

    def wrap_callable(
        self,
        session: DiagnosticSession,
        system: str,
        operation: str,
        function: Callable[..., T],
    ) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return self.run_stage(
                session,
                system,
                operation,
                lambda: function(*args, **kwargs),
            )

        # Preserve optional diagnostic side channels exposed by bound system
        # methods. Samplers may consume these after a call without changing the
        # callable's ordinary return contract.
        owner = getattr(function, "__self__", None)
        configure_guidance_trace = getattr(owner, "set_guidance_trace_enabled", None)
        if callable(configure_guidance_trace):
            configure_guidance_trace(
                bool(
                    session.trace_recorder is not None
                    and getattr(session.trace_recorder, "enabled", False)
                )
            )
        consume_guidance_trace = getattr(owner, "consume_guidance_trace", None)
        if callable(consume_guidance_trace):
            setattr(wrapped, "consume_guidance_trace", consume_guidance_trace)

        return wrapped

    def record_tensor(
        self,
        session: DiagnosticSession,
        name: str,
        value: torch.Tensor | None,
        *,
        system: str,
        operation: str,
    ) -> None:
        if not session.config.tensor_summaries or value is None:
            return
        summary = tensor_summary(
            value,
            include_statistics=session.config.tensor_statistics,
        )
        session.tensor_summaries[name] = summary
        self.emit(
            session,
            "DEBUG",
            system,
            operation,
            "tensor summary",
            tensor=name,
            summary=summary,
        )

    def update_components(self, session: DiagnosticSession, components: Any) -> None:
        report: dict[str, Any] = {}
        for name in ("unet", "vae", "text_encoder", "tokenizer"):
            value = getattr(components, name, None)
            if value is None:
                report[name] = None
                continue
            item: dict[str, Any] = {
                "runtime_type": f"{type(value).__module__}.{type(value).__qualname__}"
            }
            if isinstance(value, torch.nn.Module):
                parameters = list(value.parameters())
                item["parameter_count"] = int(sum(parameter.numel() for parameter in parameters))
                if parameters:
                    item["device"] = str(parameters[0].device)
                    item["dtype"] = str(parameters[0].dtype)
                item["training"] = bool(value.training)
            report[name] = item
        session.component_report = report

    def update_schedule(self, session: DiagnosticSession, schedule: SchedulerOutput | None) -> None:
        if schedule is None:
            return
        session.schedule_report.update(dict(schedule.to_serializable_dict()))
        if session.trace_recorder is not None:
            session.trace_recorder.seed_schedule_metadata(dict(schedule.extra or {}))
            recorder = session.trace_recorder
            if recorder.request.capture_predicted_x0_previews:
                configured = tuple(recorder.request.predicted_x0_preview_steps)
                if configured:
                    selected = sorted(
                        value
                        for value in {int(item) for item in configured}
                        if 0 <= value < schedule.sigma_transitions
                    )
                    selection = {
                        "mode": "explicit",
                        "selected_steps": selected,
                    }
                else:
                    selected, selection = self._select_predicted_x0_preview_steps(schedule)
                recorder.configure_predicted_x0_preview_steps(selected)
                recorder.seed_schedule_metadata(
                    {"phase08b_predicted_x0_preview_selection": selection}
                )
        self.record_tensor(
            session,
            "schedule.sigmas",
            schedule.sigmas,
            system="scheduling",
            operation="build",
        )
        self.record_tensor(
            session,
            "schedule.timesteps",
            schedule.timesteps,
            system="scheduling",
            operation="build",
        )

    def update_sampler(self, session: DiagnosticSession, sampler: SamplerOutput | None) -> None:
        if sampler is None:
            return
        session.sampler_report.update(dict(sampler.to_serializable_dict()))
        self.record_tensor(
            session,
            "sampler.latents",
            sampler.latents,
            system="sampling",
            operation="sample",
        )

    def finish(
        self,
        session: DiagnosticSession,
        sampler: SamplerOutput,
        *,
        decoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        diagnostic_decode_enabled: bool = True,
    ) -> dict[str, Any]:
        self.update_sampler(session, sampler)
        recorder = session.trace_recorder
        if recorder is None:
            return {}
        extra = sampler.extra or {}
        recorder.set_runtime_summary(
            requested_steps=extra.get("requested_steps"),
            effective_steps=extra.get("effective_steps"),
            stopping_index=extra.get("stopping_index"),
        )

        exports: dict[str, Any] = {}
        if recorder.request.capture_predicted_x0_previews:
            if diagnostic_decode_enabled:
                exports.update(
                    self._export_predicted_x0_previews(
                        session,
                        recorder,
                        decoder=decoder,
                    )
                )
            else:
                exports["predicted_x0_decode"] = {
                    "enabled": False,
                    "reason": "diagnostic_decode_disabled",
                    "captured_latent_count": len(recorder.predicted_x0_latents),
                }
        try:
            exports.update(recorder.export_requested_artifacts())
        except Exception as exc:
            exports.setdefault("errors", []).append(
                {
                    "operation": "sampler_trace_export",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            self.emit(
                session,
                "WARNING",
                "diagnostics",
                "sampler_trace",
                "sampler trace export failed without blocking generation",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        session.trace_exports = exports
        self.emit(
            session,
            "INFO",
            "diagnostics",
            "sampler_trace",
            "sampler trace finalized",
            exported=session.trace_exports,
        )
        return dict(session.trace_exports)

    @staticmethod
    def _select_predicted_x0_preview_steps(
        schedule: SchedulerOutput,
    ) -> tuple[list[int], dict[str, Any]]:
        transitions = int(schedule.sigma_transitions)
        if transitions <= 0:
            return [], {"mode": "automatic", "selected_steps": [], "labels": {}}

        timesteps = schedule.timesteps
        values: list[float] = []
        if timesteps is not None:
            flat = timesteps.detach().to(device="cpu", dtype=torch.float32).flatten()
            values = [float(value) for value in flat[:transitions].tolist()]

        labels: dict[int, list[str]] = {}

        def add(index: int, label: str) -> None:
            if 0 <= index < transitions:
                labels.setdefault(int(index), []).append(label)

        add(0, "first_step")
        if values:
            initial = values[0]
            plateau_end = 0
            for index, value in enumerate(values[1:], start=1):
                if abs(value - initial) > 1e-6:
                    break
                plateau_end = index
            if plateau_end > 0:
                add(plateau_end, "last_repeated_initial_timestep")
                if plateau_end + 1 < transitions:
                    add(plateau_end + 1, "first_distinct_timestep_after_plateau")
        add(transitions // 2, "middle_step")
        add(max(0, transitions - 2), "penultimate_step")

        selected = sorted(labels)
        return selected, {
            "mode": "automatic",
            "selected_steps": selected,
            "labels": {str(index): names for index, names in labels.items()},
            "timestep_values": values,
        }

    def _export_predicted_x0_previews(
        self,
        session: DiagnosticSession,
        recorder: Any,
        *,
        decoder: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> dict[str, Any]:
        trace_output_dir = Path(recorder.request.output_dir).expanduser().resolve()
        run_root = (
            trace_output_dir.parent
            if trace_output_dir.name.lower() == "traces"
            else session.config.artifacts_root / "phase08b_diagnostics" / session.run_id
        )
        preview_root = run_root / "predicted_x0"
        manifest_root = run_root / "manifests"
        preview_root.mkdir(parents=True, exist_ok=True)
        manifest_root.mkdir(parents=True, exist_ok=True)

        exported: dict[str, Any] = {}
        preview_records: list[dict[str, Any]] = []
        if decoder is None:
            error = RuntimeError("No decoder was supplied for predicted-x0 previews.")
            recorder.add_predicted_x0_preview_error(step_index=-1, error=error)
        else:
            for step_index, latent in sorted(recorder.predicted_x0_latents.items()):
                step_dir = preview_root / f"step_{int(step_index):03d}"
                try:
                    step_dir.mkdir(parents=True, exist_ok=True)
                    images = decode_latent_to_pil_images(decoder, latent)
                    paths: list[str] = []
                    for image_index, image in enumerate(images):
                        path = step_dir / f"predicted_x0_{image_index:02d}.png"
                        image.save(path)
                        paths.append(str(path))
                    preview_records.append(
                        {
                            "step_index": int(step_index),
                            "status": "saved",
                            "paths": paths,
                        }
                    )
                except Exception as exc:
                    recorder.add_predicted_x0_preview_error(
                        step_index=int(step_index), error=exc
                    )
                    preview_records.append(
                        {
                            "step_index": int(step_index),
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    self.emit(
                        session,
                        "WARNING",
                        "diagnostics",
                        "predicted_x0_preview",
                        "predicted-x0 preview failed without blocking generation",
                        step_index=int(step_index),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

        manifest_path = manifest_root / "predicted_x0_previews.json"
        payload = json_safe(
            {
                "format": "image-gen-phase08b-predicted-x0-previews-v1",
                "run_id": session.run_id,
                "selection": recorder.schedule_extra.get(
                    "phase08b_predicted_x0_preview_selection", {}
                ),
                "records": preview_records,
                "errors": list(recorder.predicted_x0_preview_errors),
            }
        )
        try:
            manifest_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            exported["predicted_x0_manifest"] = str(manifest_path)
            exported["predicted_x0_root"] = str(preview_root)
        except Exception as exc:
            exported.setdefault("errors", []).append(
                {
                    "operation": "predicted_x0_manifest_export",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            self.emit(
                session,
                "WARNING",
                "diagnostics",
                "predicted_x0_preview",
                "predicted-x0 manifest export failed without blocking generation",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return exported

    def complete(self, session: DiagnosticSession, *, result: Any | None = None) -> dict[str, Any]:
        if session.completed:
            return self.summary(session)
        session.completed = True
        self.emit(session, "DEBUG", "runtime", "complete", "diagnostic session completed")
        if session.config.export_events or session.config.verbosity == "trace":
            self._write_success_artifacts(session, result=result)
        return self.summary(session)

    def summary(self, session: DiagnosticSession) -> dict[str, Any]:
        return {
            "run_id": session.run_id,
            "started_utc": session.started_utc,
            "verbosity": session.config.verbosity,
            "timings": [item.to_dict() for item in session.timings],
            "tensor_summaries": dict(session.tensor_summaries),
            "trace_exports": dict(session.trace_exports),
            "failure_bundle": session.failure_bundle,
        }

    def fail_unassigned(
        self,
        session: DiagnosticSession,
        error: BaseException,
        *,
        system: str = "runtime",
        operation: str = "run",
    ) -> PipelineStageError:
        if isinstance(error, PipelineStageError):
            return error
        self.emit(
            session,
            "ERROR",
            system,
            operation,
            "unassigned failure captured",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        bundle_path = self._capture_failure(session, system, operation, error)
        return PipelineStageError(
            run_id=session.run_id,
            system=system,
            operation=operation,
            cause=error,
            bundle_path=bundle_path,
        )

    def _start_sampler_trace(self, session: DiagnosticSession) -> None:
        request = session.request
        request_config = getattr(request, "diagnostics", None) or {}
        trace_override = request_config.get("sampler_trace") or {}
        legacy_trace = dict(getattr(request, "sampler_kwargs", {}) or {}).get("trace") or {}
        trace = dict(legacy_trace)
        trace.update(dict(trace_override))
        enabled = bool(trace.get("enabled", session.config.sampler_trace_enabled))
        if not enabled:
            return

        predicted_x0_previews = bool(
            trace.get(
                "predicted_x0_previews",
                session.config.sampler_trace_predicted_x0_previews,
            )
        )
        if predicted_x0_previews:
            output_dir = (
                session.config.artifacts_root
                / "phase08b_diagnostics"
                / session.run_id
                / "traces"
            )
        else:
            output_dir = session.config.artifacts_root / "traces" / session.run_id
        configured_output = trace.get("output_dir")
        if configured_output:
            candidate = Path(str(configured_output)).expanduser()
            if not candidate.is_absolute() and self.project_context is not None:
                candidate = Path(self.project_context.project_root) / candidate
            candidate = candidate.resolve()
            if self._is_source_path(candidate):
                self.emit(
                    session,
                    "WARNING",
                    "diagnostics",
                    "sampler_trace",
                    "trace output was relocated outside source folders",
                    requested_path=str(candidate),
                    effective_path=str(output_dir),
                )
            else:
                output_dir = candidate

        recorder = create_step_trace_recorder(
            enabled=True,
            run_name=trace.get("run_name") or session.run_id,
            output_dir=str(output_dir),
            schedule_extra={},
            sampler_name=getattr(request, "sampler_name", None),
            scheduler_name=getattr(request, "scheduler_name", None),
            export_json=bool(trace.get("export_json", session.config.sampler_trace_json)),
            export_csv=bool(trace.get("export_csv", session.config.sampler_trace_csv)),
            export_txt_summary=bool(
                trace.get("export_txt_summary", session.config.sampler_trace_summary)
            ),
            capture_latents=bool(
                trace.get("capture_latents", session.config.sampler_trace_capture_latents)
            ),
            capture_latent_every_n=int(
                trace.get(
                    "capture_latent_every_n",
                    session.config.sampler_trace_capture_every_n,
                )
                or 0
            ),
            cleanup_after_export=bool(trace.get("cleanup_after_export", False)),
            capture_predicted_x0_previews=predicted_x0_previews,
            predicted_x0_preview_steps=tuple(
                int(value)
                for value in (
                    trace.get("predicted_x0_steps")
                    or session.config.sampler_trace_predicted_x0_steps
                    or ()
                )
            ),
        )
        request.sampler_kwargs = dict(request.sampler_kwargs or {})
        request.sampler_kwargs["trace_recorder"] = recorder
        session.trace_recorder = recorder

    def _is_source_path(self, path: Path) -> bool:
        if self.project_context is None:
            return False
        root = Path(self.project_context.project_root).resolve()
        source_roots = [
            root / name
            for name in ("src", "modules", "tests", "docs", "app", "image_gen")
        ]
        return any(path == item or item in path.parents for item in source_roots)

    def _capture_failure(
        self,
        session: DiagnosticSession,
        system: str,
        operation: str,
        error: BaseException,
    ) -> str | None:
        if session.failure_bundle:
            return session.failure_bundle
        if not session.config.failure_bundles:
            return None
        try:
            bundle = write_failure_bundle(
                session,
                system=system,
                operation=operation,
                error=error,
            )
            session.failure_bundle = str(bundle)
            return session.failure_bundle
        except Exception as bundle_error:
            # Diagnostics must never replace the pipeline exception they were
            # attempting to record. Persist a minimal fallback when possible,
            # then return control to the original error path.
            fallback_dir = (
                session.config.artifacts_root
                / "failures"
                / f"{session.run_id}_failure_bundle_fallback"
            )
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                payload = json_safe(
                    {
                        "format": "image-gen-failure-bundle-fallback-v1",
                        "run_id": session.run_id,
                        "system": system,
                        "operation": operation,
                        "original_error_type": type(error).__name__,
                        "original_error_message": str(error),
                        "bundle_error_type": type(bundle_error).__name__,
                        "bundle_error_message": str(bundle_error),
                    },
                    redact_secrets=True,
                )
                (fallback_dir / "failure.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (fallback_dir / "original_traceback.txt").write_text(
                    "".join(
                        traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    ),
                    encoding="utf-8",
                )
                (fallback_dir / "bundle_traceback.txt").write_text(
                    "".join(
                        traceback.format_exception(
                            type(bundle_error),
                            bundle_error,
                            bundle_error.__traceback__,
                        )
                    ),
                    encoding="utf-8",
                )
                session.failure_bundle = str(fallback_dir)
            except Exception:
                session.failure_bundle = None
            print(
                "[WARNING] failure bundle capture failed; preserving original "
                f"{type(error).__name__}: {error}. "
                f"Bundle error: {type(bundle_error).__name__}: {bundle_error}",
                file=self.console,
            )
            return session.failure_bundle

    def _write_success_artifacts(self, session: DiagnosticSession, *, result: Any | None) -> None:
        output = session.run_artifact_dir
        output.mkdir(parents=True, exist_ok=True)
        (output / "events.jsonl").write_text(
            "".join(json.dumps(event.to_dict(), ensure_ascii=False) + "\n" for event in session.events),
            encoding="utf-8",
        )
        payload = self.summary(session)
        if result is not None:
            payload["result"] = json_safe(result)
        (output / "run_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
