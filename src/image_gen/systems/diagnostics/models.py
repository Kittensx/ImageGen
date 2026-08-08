from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


VERBOSITY_CONSOLE_LEVEL = {
    "trace": 10,
    "verbose": 20,
    "normal": 30,
    "quiet": 40,
}
SEVERITY_LEVEL = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


@dataclass(frozen=True)
class DiagnosticConfig:
    """Runtime diagnostics policy.

    Failure bundles remain enabled by default while normal console output stays
    quiet. Diagnostic files are written under ``artifacts/diagnostics`` unless
    an explicit artifacts root is supplied.
    """

    verbosity: str = "normal"
    artifacts_root: Path = Path("artifacts/diagnostics")
    failure_bundles: bool = True
    export_events: bool = False
    tensor_summaries: bool = False
    tensor_statistics: bool = False
    progress: bool = True
    sampler_trace_enabled: bool = False
    sampler_trace_json: bool = True
    sampler_trace_csv: bool = False
    sampler_trace_summary: bool = True
    sampler_trace_capture_latents: bool = False
    sampler_trace_capture_every_n: int = 0
    sampler_trace_predicted_x0_previews: bool = False
    sampler_trace_predicted_x0_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        verbosity = self.verbosity.strip().lower()
        if verbosity not in VERBOSITY_CONSOLE_LEVEL:
            raise ValueError(
                "diagnostics verbosity must be one of: quiet, normal, verbose, trace"
            )
        object.__setattr__(self, "verbosity", verbosity)
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root).expanduser().resolve())
        if self.sampler_trace_capture_every_n < 0:
            raise ValueError("sampler_trace_capture_every_n cannot be negative")
        preview_steps = tuple(int(value) for value in self.sampler_trace_predicted_x0_steps)
        if any(value < 0 for value in preview_steps):
            raise ValueError("sampler_trace_predicted_x0_steps cannot contain negative values")
        object.__setattr__(self, "sampler_trace_predicted_x0_steps", preview_steps)

    @property
    def console_level(self) -> int:
        return VERBOSITY_CONSOLE_LEVEL[self.verbosity]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        artifacts_root: Path | str | None = None,
    ) -> "DiagnosticConfig":
        raw = dict(value or {})
        trace = raw.get("sampler_trace") or {}
        if not isinstance(trace, Mapping):
            raise TypeError("diagnostics.sampler_trace must be a mapping")
        selected_root = artifacts_root or raw.get("artifacts_root") or "artifacts/diagnostics"
        return cls(
            verbosity=str(raw.get("verbosity", "normal")),
            artifacts_root=Path(selected_root),
            failure_bundles=bool(raw.get("failure_bundles", True)),
            export_events=bool(raw.get("export_events", False)),
            tensor_summaries=bool(raw.get("tensor_summaries", False)),
            tensor_statistics=bool(raw.get("tensor_statistics", False)),
            progress=bool(raw.get("progress", True)),
            sampler_trace_enabled=bool(trace.get("enabled", False)),
            sampler_trace_json=bool(trace.get("export_json", True)),
            sampler_trace_csv=bool(trace.get("export_csv", False)),
            sampler_trace_summary=bool(trace.get("export_txt_summary", True)),
            sampler_trace_capture_latents=bool(trace.get("capture_latents", False)),
            sampler_trace_capture_every_n=int(trace.get("capture_latent_every_n", 0) or 0),
            sampler_trace_predicted_x0_previews=bool(
                trace.get("predicted_x0_previews", False)
            ),
            sampler_trace_predicted_x0_steps=tuple(
                int(value) for value in (trace.get("predicted_x0_steps") or ())
            ),
        )

    def merged(self, override: Mapping[str, Any] | None) -> "DiagnosticConfig":
        if not override:
            return self
        raw = self.to_dict()
        trace = dict(raw.pop("sampler_trace"))
        supplied = dict(override)
        supplied_trace = supplied.pop("sampler_trace", None)
        raw.update({key: value for key, value in supplied.items() if value is not None})
        if supplied_trace is not None:
            if not isinstance(supplied_trace, Mapping):
                raise TypeError("request diagnostics sampler_trace must be a mapping")
            trace.update({key: value for key, value in supplied_trace.items() if value is not None})
        raw["sampler_trace"] = trace
        return DiagnosticConfig.from_mapping(raw, artifacts_root=raw.get("artifacts_root"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbosity": self.verbosity,
            "artifacts_root": str(self.artifacts_root),
            "failure_bundles": self.failure_bundles,
            "export_events": self.export_events,
            "tensor_summaries": self.tensor_summaries,
            "tensor_statistics": self.tensor_statistics,
            "progress": self.progress,
            "sampler_trace": {
                "enabled": self.sampler_trace_enabled,
                "export_json": self.sampler_trace_json,
                "export_csv": self.sampler_trace_csv,
                "export_txt_summary": self.sampler_trace_summary,
                "capture_latents": self.sampler_trace_capture_latents,
                "capture_latent_every_n": self.sampler_trace_capture_every_n,
                "predicted_x0_previews": self.sampler_trace_predicted_x0_previews,
                "predicted_x0_steps": list(self.sampler_trace_predicted_x0_steps),
            },
        }


@dataclass(frozen=True)
class DiagnosticEvent:
    sequence: int
    timestamp_utc: str
    elapsed_ms: float
    run_id: str
    severity: str
    system: str
    operation: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "run_id": self.run_id,
            "severity": self.severity,
            "system": self.system,
            "operation": self.operation,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class StageTiming:
    system: str
    operation: str
    occurrence: int
    status: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "operation": self.operation,
            "occurrence": self.occurrence,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class DiagnosticSession:
    run_id: str
    config: DiagnosticConfig
    request: Any
    started_perf: float = field(default_factory=time.perf_counter)
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    effective_config: dict[str, Any] = field(default_factory=dict)
    request_extras: dict[str, Any] = field(default_factory=dict)
    component_report: dict[str, Any] = field(default_factory=dict)
    schedule_report: dict[str, Any] = field(default_factory=dict)
    sampler_report: dict[str, Any] = field(default_factory=dict)
    events: list[DiagnosticEvent] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    tensor_summaries: dict[str, Any] = field(default_factory=dict)
    trace_recorder: Any | None = None
    trace_exports: dict[str, Any] = field(default_factory=dict)
    failure_bundle: str | None = None
    completed: bool = False

    @property
    def run_artifact_dir(self) -> Path:
        return self.config.artifacts_root / "runs" / self.run_id

    def next_occurrence(self, system: str, operation: str) -> int:
        return 1 + sum(
            1
            for item in self.timings
            if item.system == system and item.operation == operation
        )


class PipelineStageError(RuntimeError):
    """Failure assigned to one named system and operation."""

    def __init__(
        self,
        *,
        run_id: str,
        system: str,
        operation: str,
        cause: BaseException,
        bundle_path: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.system = system
        self.operation = operation
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)
        self.bundle_path = bundle_path
        message = (
            f"{system}.{operation} failed during run {run_id}: "
            f"{self.cause_type}: {self.cause_message}"
        )
        if bundle_path:
            message += f" (failure bundle: {bundle_path})"
        super().__init__(message)
