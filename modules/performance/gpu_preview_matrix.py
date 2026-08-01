from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from image_gen.runtime_options import build_cuda_allocator_diagnostics

try:
    import yaml
except Exception:  # pragma: no cover - validated at runtime
    yaml = None

try:
    import psutil
except Exception:  # pragma: no cover - optional until the matrix runner is used
    psutil = None


_GENERATION_TIME_RE = re.compile(r"^Generation time \(sec\):\s*([0-9.]+)\s*$")
_PREVIEW_SUMMARY_RE = re.compile(r"^LIVE_PREVIEW_SUMMARY_JSON:\s*(\{.*\})\s*$")
_MODEL_DIAGNOSTIC_RE = re.compile(r"^MODEL_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_RUNTIME_DIAGNOSTIC_RE = re.compile(r"^RUNTIME_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_PERFORMANCE_PHASE_RE = re.compile(r"^PERFORMANCE_PHASE_JSON:\s*(\{.*\})\s*$")
_MEMORY_STATUS_RE = re.compile(r"MEMORY_STATUS_JSON:\s*(\{.*\})\s*$")


@dataclass(frozen=True)
class MatrixCase:
    checkpoint_label: str
    checkpoint_path: str
    sampler_label: str
    sampler_name: str
    scheduler_name: str
    sampler_kwargs: dict[str, Any]
    scheduler_kwargs: dict[str, Any]
    preview_label: str
    preview: dict[str, Any]
    memory_label: str
    memory: dict[str, Any]
    resolution_label: str
    width: int
    height: int

    @property
    def baseline_key(self) -> tuple[str, str, str, str, str, int, int]:
        return (
            self.checkpoint_path,
            self.sampler_name,
            self.scheduler_name,
            self.memory_label,
            self.resolution_label,
            self.width,
            self.height,
        )

    @property
    def case_id(self) -> str:
        return "__".join(
            _safe_name(value)
            for value in (
                self.checkpoint_label,
                self.sampler_label,
                self.preview_label,
                self.memory_label,
                self.resolution_label,
            )
        )


@dataclass
class TelemetrySample:
    timestamp_sec: float
    cpu_process_percent: float | None = None
    cpu_system_percent: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    process_vram_mb: float | None = None


@dataclass
class TelemetrySummary:
    sample_count: int = 0
    cpu_process_average_percent: float | None = None
    cpu_process_peak_percent: float | None = None
    cpu_system_average_percent: float | None = None
    cpu_system_peak_percent: float | None = None
    gpu_utilization_average_percent: float | None = None
    gpu_utilization_peak_percent: float | None = None
    gpu_memory_used_average_mb: float | None = None
    gpu_memory_used_peak_mb: float | None = None
    process_vram_average_mb: float | None = None
    process_vram_peak_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    nvidia_smi_available: bool = False
    psutil_available: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class ProcessTelemetryMonitor:
    """Sample CPU and NVIDIA GPU telemetry while one generation subprocess runs."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        sample_interval_sec: float = 0.25,
        gpu_index: int = 0,
        nvidia_smi_path: str | None = None,
    ) -> None:
        self.process = process
        self.sample_interval_sec = max(0.10, float(sample_interval_sec))
        self.gpu_index = max(0, int(gpu_index))
        self.nvidia_smi_path = nvidia_smi_path or shutil.which("nvidia-smi")
        self.samples: list[TelemetrySample] = []
        self.warnings: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_process = None
        self._logical_cpu_count = max(1, os.cpu_count() or 1)
        if psutil is not None:
            try:
                self._root_process = psutil.Process(process.pid)
                self._root_process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception as exc:
                self.warnings.append(f"psutil process initialization failed: {exc}")
                self._root_process = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name="gpu-preview-matrix-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.sample_interval_sec * 4.0))
        return summarize_telemetry(
            self.samples,
            nvidia_smi_available=bool(self.nvidia_smi_path),
            psutil_available=self._root_process is not None,
            warnings=self.warnings,
        )

    def _loop(self) -> None:
        while not self._stop_event.is_set() and self.process.poll() is None:
            started = time.perf_counter()
            self.samples.append(self._sample())
            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.0, self.sample_interval_sec - elapsed))
        if not self._stop_event.is_set():
            self.samples.append(self._sample())

    def _process_tree(self) -> list[Any]:
        if self._root_process is None:
            return []
        processes = [self._root_process]
        try:
            processes.extend(self._root_process.children(recursive=True))
        except Exception:
            pass
        return processes

    def _sample_cpu(self) -> tuple[float | None, float | None, set[int]]:
        if self._root_process is None or psutil is None:
            return None, None, {int(self.process.pid)}
        raw_percent = 0.0
        pids: set[int] = set()
        for process in self._process_tree():
            try:
                pids.add(int(process.pid))
                raw_percent += max(0.0, float(process.cpu_percent(interval=None)))
            except Exception:
                continue
        try:
            system_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            system_percent = None
        normalized = min(100.0, raw_percent / self._logical_cpu_count)
        return normalized, system_percent, pids or {int(self.process.pid)}

    def _run_nvidia_smi(self, arguments: Sequence[str]) -> str:
        if not self.nvidia_smi_path:
            return ""
        completed = subprocess.run(
            [self.nvidia_smi_path, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
            creationflags=_no_window_creation_flags(),
        )
        if completed.returncode != 0:
            message = completed.stderr.strip()
            if message and message not in self.warnings:
                self.warnings.append(f"nvidia-smi: {message}")
            return ""
        return completed.stdout.strip()

    def _sample_gpu(self, pids: set[int]) -> tuple[float | None, float | None, float | None, float | None]:
        if not self.nvidia_smi_path:
            return None, None, None, None
        device_output = self._run_nvidia_smi(
            [
                f"--id={self.gpu_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        gpu_util = memory_used = memory_total = None
        if device_output:
            first = device_output.splitlines()[0]
            values = [_float_or_none(item.strip()) for item in first.split(",")]
            if len(values) >= 3:
                gpu_util, memory_used, memory_total = values[:3]

        process_output = self._run_nvidia_smi(
            [
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_vram = 0.0
        matched = False
        for line in process_output.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 2:
                continue
            try:
                pid = int(fields[0])
            except ValueError:
                continue
            memory = _float_or_none(fields[1])
            if pid in pids and memory is not None:
                matched = True
                process_vram += memory
        return gpu_util, memory_used, memory_total, process_vram if matched else None

    def _sample(self) -> TelemetrySample:
        cpu_process, cpu_system, pids = self._sample_cpu()
        gpu_util, gpu_used, gpu_total, process_vram = self._sample_gpu(pids)
        return TelemetrySample(
            timestamp_sec=time.time(),
            cpu_process_percent=cpu_process,
            cpu_system_percent=cpu_system,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            process_vram_mb=process_vram,
        )


def _no_window_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _mean(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def _max(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return max(cleaned) if cleaned else None


def summarize_telemetry(
    samples: Sequence[TelemetrySample],
    *,
    nvidia_smi_available: bool,
    psutil_available: bool,
    warnings: Sequence[str] = (),
) -> TelemetrySummary:
    return TelemetrySummary(
        sample_count=len(samples),
        cpu_process_average_percent=_round_or_none(_mean(item.cpu_process_percent for item in samples), 3),
        cpu_process_peak_percent=_round_or_none(_max(item.cpu_process_percent for item in samples), 3),
        cpu_system_average_percent=_round_or_none(_mean(item.cpu_system_percent for item in samples), 3),
        cpu_system_peak_percent=_round_or_none(_max(item.cpu_system_percent for item in samples), 3),
        gpu_utilization_average_percent=_round_or_none(_mean(item.gpu_utilization_percent for item in samples), 3),
        gpu_utilization_peak_percent=_round_or_none(_max(item.gpu_utilization_percent for item in samples), 3),
        gpu_memory_used_average_mb=_round_or_none(_mean(item.gpu_memory_used_mb for item in samples), 3),
        gpu_memory_used_peak_mb=_round_or_none(_max(item.gpu_memory_used_mb for item in samples), 3),
        process_vram_average_mb=_round_or_none(_mean(item.process_vram_mb for item in samples), 3),
        process_vram_peak_mb=_round_or_none(_max(item.process_vram_mb for item in samples), 3),
        gpu_memory_total_mb=_round_or_none(_max(item.gpu_memory_total_mb for item in samples), 3),
        nvidia_smi_available=bool(nvidia_smi_available),
        psutil_available=bool(psutil_available),
        warnings=list(dict.fromkeys(str(value) for value in warnings if value)),
    )


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return text.strip("-._") or "unnamed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for the GPU preview matrix configuration.")
        payload = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported configuration extension: {suffix}")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return dict(payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _expand_checkpoints(project_root: Path, entries: Sequence[Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in entries:
        entry = {"path": raw} if isinstance(raw, str) else dict(raw or {})
        path_value = str(entry.get("path") or entry.get("glob") or "").strip()
        if not path_value:
            continue
        absolute_pattern = Path(path_value).expanduser()
        if not absolute_pattern.is_absolute():
            absolute_pattern = project_root / absolute_pattern
        matches = sorted(glob.glob(str(absolute_pattern), recursive=True))
        if not matches and not glob.has_magic(str(absolute_pattern)):
            matches = [str(absolute_pattern)]
        for match in matches:
            path = str(Path(match).resolve())
            if path in seen:
                continue
            seen.add(path)
            label = str(entry.get("label") or Path(path).stem)
            if len(matches) > 1 and entry.get("label"):
                label = f"{entry['label']} - {Path(path).stem}"
            output.append({"label": label, "path": path})
    return output


def expand_matrix_cases(config: Mapping[str, Any], project_root: Path) -> list[MatrixCase]:
    checkpoints = _expand_checkpoints(project_root, list(config.get("checkpoints") or []))
    samplers = [dict(value or {}) for value in list(config.get("samplers") or [])]
    previews = [dict(value or {}) for value in list(config.get("preview_modes") or [])]
    memory_profiles = [dict(value or {}) for value in list(config.get("memory_profiles") or [])]
    if not memory_profiles:
        memory_profiles = [{"label": "Auto", "policy": "auto"}]
    resolutions = [dict(value or {}) for value in list(config.get("resolutions") or [])]
    if not checkpoints:
        raise ValueError("The matrix requires at least one checkpoint path or glob.")
    if not samplers:
        raise ValueError("The matrix requires at least one sampler entry.")
    if not resolutions:
        raise ValueError("The matrix requires at least one resolution entry.")
    if not any(not bool(item.get("enabled", True)) for item in previews):
        previews.insert(0, {"label": "baseline-disabled", "enabled": False})

    cases: list[MatrixCase] = []
    for checkpoint in checkpoints:
        for sampler in samplers:
            sampler_name = str(sampler.get("name") or "").strip()
            scheduler_name = str(sampler.get("scheduler") or sampler.get("scheduler_name") or "").strip()
            if not sampler_name or not scheduler_name:
                raise ValueError("Every sampler entry requires name and scheduler.")
            for resolution in resolutions:
                width = int(resolution.get("width") or 0)
                height = int(resolution.get("height") or 0)
                if width <= 0 or height <= 0:
                    raise ValueError("Every resolution entry requires positive width and height.")
                for memory in memory_profiles:
                    memory_label = str(memory.get("label") or memory.get("policy") or "Auto")
                    for preview in previews:
                        preview_enabled = bool(preview.get("enabled", True))
                        preview_label = str(
                            preview.get("label")
                            or (preview.get("mode") if preview_enabled else "baseline-disabled")
                        )
                        cases.append(
                            MatrixCase(
                                checkpoint_label=checkpoint["label"],
                                checkpoint_path=checkpoint["path"],
                                sampler_label=str(sampler.get("label") or sampler_name),
                                sampler_name=sampler_name,
                                scheduler_name=scheduler_name,
                                sampler_kwargs=dict(sampler.get("sampler_kwargs") or {}),
                                scheduler_kwargs=dict(sampler.get("scheduler_kwargs") or {}),
                                preview_label=preview_label,
                                preview=dict(preview),
                                memory_label=memory_label,
                                memory=dict(memory),
                                resolution_label=str(
                                    resolution.get("label") or f"{width}x{height}"
                                ),
                                width=width,
                                height=height,
                            )
                        )
    return sorted(
        cases,
        key=lambda item: (
            item.baseline_key,
            bool(item.preview.get("enabled", True)),
            item.preview_label,
        ),
    )


def _preview_payload(case: MatrixCase, preview_root: Path) -> dict[str, Any]:
    preview = dict(case.preview)
    enabled = bool(preview.get("enabled", True))
    return {
        "live_preview_enabled": enabled,
        "live_preview_mode": str(preview.get("mode") or "fast"),
        "live_preview_interval": max(1, int(preview.get("interval") or 1)),
        "live_preview_width": max(128, int(preview.get("width") or 384)),
        "live_preview_format": str(preview.get("format") or "webp"),
        "live_preview_keep_history": str(preview.get("keep_history") or "latest_only"),
        "live_preview_batch_index": max(0, int(preview.get("batch_index") or 0)),
        "live_preview_quality": max(35, min(95, int(preview.get("quality") or 78))),
        "live_preview_root": str(preview_root),
        "live_preview_async": bool(preview.get("async", True)),
        "live_preview_clone_tensors": bool(preview.get("clone_tensors", False)),
        "live_preview_adaptive_throttle": bool(preview.get("adaptive_throttle", enabled)),
        "live_preview_adaptive_target_ratio": float(preview.get("adaptive_target_ratio") or 0.75),
        "live_preview_adaptive_recovery_ratio": float(preview.get("adaptive_recovery_ratio") or 0.40),
        "live_preview_adaptive_max_interval": max(1, int(preview.get("adaptive_max_interval") or 8)),
        "live_preview_adaptive_window": max(2, int(preview.get("adaptive_window") or 6)),
    }


def _memory_payload(case: MatrixCase) -> dict[str, Any]:
    memory = dict(case.memory or {})
    return {
        "memory_policy": str(memory.get("policy") or "auto"),
        "memory_vram_safety_margin_mb": max(128, int(memory.get("safety_margin_mb") or 1024)),
        "memory_retain_checkpoint_between_jobs": bool(memory.get("retain_checkpoint", True)),
        "memory_retain_vae_between_jobs": bool(memory.get("retain_vae", False)),
        "memory_pinned_cpu_memory": bool(memory.get("pinned_cpu_memory", False)),
        "memory_allow_tiled_vae_fallback": bool(memory.get("allow_tiled_vae_fallback", True)),
        "memory_allow_preview_suspension_on_oom": bool(memory.get("allow_preview_suspension_on_oom", True)),
    }


def build_case_request(
    base_request: Mapping[str, Any],
    case: MatrixCase,
    run_root: Path,
) -> dict[str, Any]:
    payload = dict(base_request)
    payload.update(
        {
            "model_path": case.checkpoint_path,
            "sampler_name": case.sampler_name,
            "scheduler_name": case.scheduler_name,
            "sampler_kwargs": dict(case.sampler_kwargs),
            "scheduler_kwargs": dict(case.scheduler_kwargs),
            "width": case.width,
            "height": case.height,
            "batch_size": 1,
            "batch_count": 1,
            "unlimited": False,
            "save_images": False,
            "output_dir": str(run_root / "output"),
        }
    )
    payload.update(_preview_payload(case, run_root / "live-preview"))
    payload.update(_memory_payload(case))
    return payload


def parse_generation_console(text: str) -> dict[str, Any]:
    generation_time = None
    preview_summary: dict[str, Any] = {}
    model_diagnostic: dict[str, Any] = {}
    runtime_diagnostic: dict[str, Any] = {}
    performance_phase: dict[str, float] = {}
    memory_status: dict[str, Any] = {}
    for line in text.splitlines():
        generation_match = _GENERATION_TIME_RE.match(line.strip())
        if generation_match:
            generation_time = float(generation_match.group(1))
            continue
        phase_match = _PERFORMANCE_PHASE_RE.match(line.strip())
        if phase_match:
            try:
                phase_payload = json.loads(phase_match.group(1))
            except json.JSONDecodeError:
                phase_payload = {}
            event = str(phase_payload.get("event") or "")
            timestamp = _float_or_none(phase_payload.get("timestamp_unix"))
            if event and timestamp is not None:
                performance_phase[event] = timestamp
            continue
        memory_match = _MEMORY_STATUS_RE.search(line.strip())
        if memory_match:
            try:
                memory_payload = json.loads(memory_match.group(1))
            except json.JSONDecodeError:
                memory_payload = {}
            if isinstance(memory_payload, dict):
                memory_status = dict(memory_payload.get("status") or {})
                memory_status["event"] = memory_payload.get("event")
                memory_status["stage"] = memory_payload.get("stage")
            continue
        for pattern, target_name in (
            (_PREVIEW_SUMMARY_RE, "preview"),
            (_MODEL_DIAGNOSTIC_RE, "model"),
            (_RUNTIME_DIAGNOSTIC_RE, "runtime"),
        ):
            match = pattern.match(line.strip())
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                payload = {}
            if target_name == "preview":
                preview_summary = dict(payload or {})
            elif target_name == "model":
                model_diagnostic = dict(payload or {})
            else:
                runtime_diagnostic = dict(payload or {})
    return {
        "generation_time_sec": generation_time,
        "preview_summary": preview_summary,
        "model_diagnostic": model_diagnostic,
        "runtime_diagnostic": runtime_diagnostic,
        "performance_phase": performance_phase,
        "memory_status": memory_status,
    }


def _flatten_preview(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "preview_frames_emitted",
        "preview_frames_failed",
        "preview_decode_time_total_ms",
        "preview_encode_time_total_ms",
        "preview_work_time_total_ms",
        "preview_decode_time_average_ms",
        "preview_encode_time_average_ms",
        "frame_latency_average_ms",
        "frame_latency_p95_ms",
        "frame_latency_max_ms",
        "queue_latency_average_ms",
        "queue_latency_p95_ms",
        "sampler_step_duration_average_ms",
        "sampler_step_duration_p95_ms",
        "preview_to_sampler_ratio_average",
        "preview_to_sampler_ratio_p95",
        "frames_enqueued",
        "frames_processed",
        "frames_replaced",
        "coalesced_frames",
        "adaptive_throttle_enabled",
        "adaptive_effective_interval",
        "adaptive_skipped_frames",
        "adaptive_adjustment_count",
        "adaptive_rolling_overhead_ratio",
    )
    return {key: summary.get(key) for key in keys}


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=_no_window_creation_flags(),
            )
        else:
            process.terminate()
            process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def execute_case_run(
    *,
    project_root: Path,
    python_executable: Path,
    base_request: Mapping[str, Any],
    case: MatrixCase,
    run_root: Path,
    repeat_index: int,
    is_warmup: bool,
    timeout_seconds: float,
    sample_interval_sec: float,
    gpu_index: int,
    nvidia_smi_path: str | None,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    request_payload = build_case_request(base_request, case, run_root)
    request_path = run_root / "request.json"
    _write_json(request_path, request_payload)
    console_path = run_root / "console.txt"
    command = [
        str(python_executable),
        "-m",
        "modules.txt2img.cli",
        "run",
        "--project-root",
        str(project_root),
        "--config",
        str(request_path),
        "--verbose",
        "--no-progress",
        "--no-txt",
        "--no-json",
    ]
    env = os.environ.copy()
    env["IMAGE_GEN_PERF_MATRIX"] = "1"
    source_root = str(project_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, str(project_root), env.get("PYTHONPATH", "")) if item
    )
    started_at = _utc_now()
    wall_started = time.perf_counter()
    timed_out = False
    with console_path.open("w", encoding="utf-8", newline="\n") as console:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=_no_window_creation_flags(),
        )
        monitor = ProcessTelemetryMonitor(
            process,
            sample_interval_sec=sample_interval_sec,
            gpu_index=gpu_index,
            nvidia_smi_path=nvidia_smi_path,
        )
        monitor.start()
        try:
            return_code = process.wait(timeout=max(1.0, timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(process)
            return_code = process.wait(timeout=30)
        telemetry = monitor.stop()
    wall_time_sec = time.perf_counter() - wall_started
    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_generation_console(console_text)
    allocator_diagnostic = dict(
        (parsed.get("runtime_diagnostic") or {}).get("cuda_allocator") or {}
    )
    phase = dict(parsed.get("performance_phase") or {})
    phase_start = _float_or_none(phase.get("generation_start"))
    phase_end = _float_or_none(phase.get("generation_end"))
    telemetry_scope = "full_process"
    if phase_start is not None and phase_end is not None and phase_end >= phase_start:
        scoped_samples = [
            sample for sample in monitor.samples
            if phase_start <= sample.timestamp_sec <= phase_end
        ]
        if scoped_samples:
            telemetry = summarize_telemetry(
                scoped_samples,
                nvidia_smi_available=bool(monitor.nvidia_smi_path),
                psutil_available=monitor._root_process is not None,
                warnings=monitor.warnings,
            )
            telemetry_scope = "generation_phase"
    preview_summary = dict(parsed["preview_summary"] or {})
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "run_id": run_root.name,
        "started_at": started_at,
        "is_warmup": bool(is_warmup),
        "repeat_index": int(repeat_index),
        "status": "timeout" if timed_out else ("passed" if return_code == 0 else "failed"),
        "return_code": int(return_code),
        "checkpoint": case.checkpoint_label,
        "checkpoint_path": case.checkpoint_path,
        "sampler": case.sampler_label,
        "sampler_name": case.sampler_name,
        "scheduler_name": case.scheduler_name,
        "preview_mode": case.preview_label,
        "preview_enabled": bool(case.preview.get("enabled", True)),
        "memory_profile": case.memory_label,
        "memory_policy": str(case.memory.get("policy") or "auto"),
        "memory_effective_policy": parsed["memory_status"].get("effective_policy"),
        "memory_component_transfer_count": parsed["memory_status"].get("component_transfer_count"),
        "memory_oom_recovery_count": parsed["memory_status"].get("oom_recovery_count"),
        "memory_preview_suspended": parsed["memory_status"].get("preview_image_decode_suspended"),
        "cuda_allocator_config": allocator_diagnostic.get("effective_config", ""),
        "cuda_allocator_fingerprint": allocator_diagnostic.get("fingerprint"),
        "cuda_expandable_segments_enabled": allocator_diagnostic.get(
            "expandable_segments_enabled"
        ),
        "cuda_allocator_applied_before_cuda_initialization": (
            (allocator_diagnostic.get("bootstrap") or {}).get(
                "applied_before_cuda_initialization"
            )
        ),
        "memory_peak_allocated_vram_mb": (
            round(float(parsed["memory_status"].get("peak_allocated_vram_bytes")) / (1024 * 1024), 3)
            if parsed["memory_status"].get("peak_allocated_vram_bytes") is not None
            else None
        ),
        "resolution": case.resolution_label,
        "width": case.width,
        "height": case.height,
        "generation_time_sec": parsed["generation_time_sec"],
        "process_wall_time_sec": round(wall_time_sec, 3),
        "telemetry_scope": telemetry_scope,
        "generation_phase_started_unix": phase_start,
        "generation_phase_ended_unix": phase_end,
        "console_path": str(console_path),
        "request_path": str(request_path),
        "loaded_model_path": parsed["model_diagnostic"].get("loaded_path"),
        **telemetry.to_dict(),
        **_flatten_preview(preview_summary),
    }
    _write_json(
        run_root / "run-result.json",
        {
            "row": row,
            "command": command,
            "request": request_payload,
            "preview_summary": preview_summary,
            "model_diagnostic": parsed["model_diagnostic"],
            "runtime_diagnostic": parsed["runtime_diagnostic"],
            "cuda_allocator_diagnostic": allocator_diagnostic,
            "memory_status": parsed["memory_status"],
            "telemetry_samples": [vars(sample) for sample in monitor.samples],
        },
    )
    return row


def _numeric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return _round_or_none(_mean(_float_or_none(row.get(key)) for row in rows), 4)


def _numeric_peak(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return _round_or_none(_max(_float_or_none(row.get(key)) for row in rows), 4)


def aggregate_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_warmup") or row.get("status") != "passed":
            continue
        key = (
            row.get("checkpoint"),
            row.get("checkpoint_path"),
            row.get("sampler"),
            row.get("sampler_name"),
            row.get("scheduler_name"),
            row.get("preview_mode"),
            bool(row.get("preview_enabled")),
            row.get("memory_profile"),
            row.get("memory_policy"),
            row.get("cuda_allocator_fingerprint"),
            row.get("cuda_allocator_config"),
            row.get("resolution"),
            int(row.get("width") or 0),
            int(row.get("height") or 0),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        (
            checkpoint,
            checkpoint_path,
            sampler,
            sampler_name,
            scheduler_name,
            preview_mode,
            preview_enabled,
            memory_profile,
            memory_policy,
            cuda_allocator_fingerprint,
            cuda_allocator_config,
            resolution,
            width,
            height,
        ) = key
        summaries.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_path": checkpoint_path,
                "sampler": sampler,
                "sampler_name": sampler_name,
                "scheduler_name": scheduler_name,
                "preview_mode": preview_mode,
                "preview_enabled": preview_enabled,
                "memory_profile": memory_profile,
                "memory_policy": memory_policy,
                "cuda_allocator_fingerprint": cuda_allocator_fingerprint,
                "cuda_allocator_config": cuda_allocator_config,
                "resolution": resolution,
                "width": width,
                "height": height,
                "successful_runs": len(group),
                "generation_time_average_sec": _numeric_mean(group, "generation_time_sec"),
                "generation_time_min_sec": _round_or_none(
                    min(
                        [
                            value
                            for value in (_float_or_none(row.get("generation_time_sec")) for row in group)
                            if value is not None
                        ],
                        default=None,
                    ),
                    4,
                ),
                "process_wall_time_average_sec": _numeric_mean(group, "process_wall_time_sec"),
                "process_vram_peak_mb": _numeric_peak(group, "process_vram_peak_mb"),
                "gpu_memory_used_peak_mb": _numeric_peak(group, "gpu_memory_used_peak_mb"),
                "gpu_utilization_average_percent": _numeric_mean(group, "gpu_utilization_average_percent"),
                "gpu_utilization_peak_percent": _numeric_peak(group, "gpu_utilization_peak_percent"),
                "cpu_process_average_percent": _numeric_mean(group, "cpu_process_average_percent"),
                "cpu_process_peak_percent": _numeric_peak(group, "cpu_process_peak_percent"),
                "frame_latency_average_ms": _numeric_mean(group, "frame_latency_average_ms"),
                "frame_latency_p95_ms": _numeric_mean(group, "frame_latency_p95_ms"),
                "coalesced_frames_average": _numeric_mean(group, "coalesced_frames"),
                "adaptive_skipped_frames_average": _numeric_mean(group, "adaptive_skipped_frames"),
                "adaptive_effective_interval_peak": _numeric_peak(group, "adaptive_effective_interval"),
                "preview_work_time_average_ms": _numeric_mean(group, "preview_work_time_total_ms"),
                "preview_to_sampler_ratio_average": _numeric_mean(group, "preview_to_sampler_ratio_average"),
                "memory_component_transfer_count_average": _numeric_mean(group, "memory_component_transfer_count"),
                "memory_oom_recovery_count_average": _numeric_mean(group, "memory_oom_recovery_count"),
                "memory_peak_allocated_vram_mb": _numeric_peak(group, "memory_peak_allocated_vram_mb"),
                "memory_preview_suspension_runs": sum(1 for row in group if row.get("memory_preview_suspended")),
            }
        )

    baseline: dict[tuple[Any, ...], dict[str, Any]] = {}
    for summary in summaries:
        if summary["preview_enabled"]:
            continue
        key = (
            summary["checkpoint_path"],
            summary["sampler_name"],
            summary["scheduler_name"],
            summary["memory_profile"],
            summary["memory_policy"],
            summary.get("cuda_allocator_fingerprint"),
            summary["resolution"],
            summary["width"],
            summary["height"],
        )
        baseline[key] = summary

    for summary in summaries:
        key = (
            summary["checkpoint_path"],
            summary["sampler_name"],
            summary["scheduler_name"],
            summary["memory_profile"],
            summary["memory_policy"],
            summary.get("cuda_allocator_fingerprint"),
            summary["resolution"],
            summary["width"],
            summary["height"],
        )
        baseline_row = baseline.get(key)
        baseline_time = _float_or_none(
            baseline_row.get("generation_time_average_sec") if baseline_row else None
        )
        generation_time = _float_or_none(summary.get("generation_time_average_sec"))
        summary["baseline_generation_time_sec"] = baseline_time
        if baseline_time is None or generation_time is None:
            summary["preview_overhead_sec"] = None
            summary["preview_overhead_percent"] = None
        else:
            overhead = generation_time - baseline_time
            summary["preview_overhead_sec"] = round(overhead, 4)
            summary["preview_overhead_percent"] = round(
                (overhead / baseline_time) * 100.0 if baseline_time > 0 else 0.0,
                3,
            )
    return sorted(
        summaries,
        key=lambda item: (
            str(item["checkpoint"]),
            str(item["sampler"]),
            str(item["resolution"]),
            bool(item["preview_enabled"]),
            str(item["memory_profile"]),
            str(item["preview_mode"]),
        ),
    )


def _format_value(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Checkpoint",
        "Sampler",
        "Preview",
        "Memory",
        "Resolution",
        "Gen sec",
        "Overhead %",
        "VRAM peak MB",
        "GPU avg %",
        "CPU avg %",
        "Frame p95 ms",
        "Coalesced",
        "Adaptive skip",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row.get("checkpoint"),
            row.get("sampler"),
            row.get("preview_mode"),
            row.get("memory_profile"),
            row.get("resolution"),
            _format_value(row.get("generation_time_average_sec"), 3),
            _format_value(row.get("preview_overhead_percent"), 2),
            _format_value(row.get("process_vram_peak_mb") or row.get("gpu_memory_used_peak_mb"), 1),
            _format_value(row.get("gpu_utilization_average_percent"), 1),
            _format_value(row.get("cpu_process_average_percent"), 1),
            _format_value(row.get("frame_latency_p95_ms"), 1),
            _format_value(row.get("coalesced_frames_average"), 1),
            _format_value(row.get("adaptive_skipped_frames_average"), 1),
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _group_report(title: str, dimension: str, summaries: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[str(row.get(dimension) or "unknown")].append(row)
    lines = [f"# {title}", "", f"Generated: {_utc_now()}", ""]
    for name in sorted(grouped):
        lines.extend([f"## {name}", "", _markdown_table(grouped[name]), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_reports(output_root: Path, rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> None:
    _write_json(output_root / "raw-runs.json", list(rows))
    _write_csv(output_root / "raw-runs.csv", list(rows))
    _write_json(output_root / "summary-by-case.json", list(summaries))
    _write_csv(output_root / "summary-by-case.csv", list(summaries))
    report = [
        "# Real GPU Preview Performance Matrix",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Generation time is the canonical pipeline generation timer. Preview overhead is measured against the preview-disabled baseline for the same checkpoint, sampler, scheduler, memory profile, and resolution.",
        "Each run result records the effective PYTORCH_CUDA_ALLOC_CONF value and its stable fingerprint so allocator changes are visible in benchmark comparisons.",
        "Allocator tuning may reduce fragmentation, but it cannot satisfy a single allocation larger than available VRAM.",
        "",
        _markdown_table(summaries),
        "",
        "## Output files",
        "",
        "- `raw-runs.csv` / `raw-runs.json`: every warmup and measured execution.",
        "- `summary-by-case.csv` / `summary-by-case.json`: averaged comparisons and baseline overhead.",
        "- `by-sampler.md`, `by-preview-mode.md`, `by-memory-profile.md`, `by-checkpoint.md`, `by-resolution.md`: grouped comparison reports.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for filename, title, dimension in (
        ("by-sampler.md", "Preview Performance by Sampler", "sampler"),
        ("by-preview-mode.md", "Preview Performance by Preview Mode", "preview_mode"),
        ("by-memory-profile.md", "Performance by Memory Profile", "memory_profile"),
        ("by-checkpoint.md", "Preview Performance by Checkpoint", "checkpoint"),
        ("by-resolution.md", "Preview Performance by Resolution", "resolution"),
    ):
        (output_root / filename).write_text(
            _group_report(title, dimension, summaries), encoding="utf-8"
        )


def _resolve_python(project_root: Path, configured: Any = None) -> Path:
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
    elif os.name == "nt":
        candidate = project_root / "venv" / "Scripts" / "python.exe"
    else:
        candidate = Path(sys.executable)
    if not candidate.exists():
        raise FileNotFoundError(f"Python executable was not found: {candidate}")
    return candidate.resolve()


def run_matrix(config_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = _load_mapping(config_file)
    root = Path(project_root or config.get("project_root") or config_file.parents[2]).expanduser().resolve()
    base_request_value = config.get("base_request") or "configs/generation_config.yml"
    base_request_path = Path(str(base_request_value)).expanduser()
    if not base_request_path.is_absolute():
        base_request_path = root / base_request_path
    base_request = _load_mapping(base_request_path.resolve())
    python_executable = _resolve_python(root, config.get("python_executable"))
    output_value = config.get("output_root") or "artifacts/performance/gpu-preview-matrix"
    output_root = Path(str(output_value)).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    cases = expand_matrix_cases(config, root)
    warmup_runs = max(0, int(config.get("warmup_runs") or 0))
    repetitions = max(1, int(config.get("repetitions") or 1))
    timeout_seconds = max(1.0, float(config.get("timeout_seconds") or 7200))
    sample_interval_sec = max(0.10, float(config.get("sample_interval_ms") or 250) / 1000.0)
    gpu_index = max(0, int(config.get("gpu_index") or 0))
    nvidia_smi_path = str(config.get("nvidia_smi_path") or "").strip() or None

    _write_json(
        output_root / "matrix-plan.json",
        {
            "created_at": _utc_now(),
            "config_path": str(config_file),
            "base_request_path": str(base_request_path.resolve()),
            "project_root": str(root),
            "python_executable": str(python_executable),
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
            "case_count": len(cases),
            "cuda_allocator": build_cuda_allocator_diagnostics(),
            "cases": [vars(case) for case in cases],
        },
    )

    rows: list[dict[str, Any]] = []
    total_runs = len(cases) * (warmup_runs + repetitions)
    run_number = 0
    for case in cases:
        for local_index in range(warmup_runs + repetitions):
            run_number += 1
            is_warmup = local_index < warmup_runs
            repeat_index = local_index + 1 if is_warmup else local_index - warmup_runs + 1
            lane = "warmup" if is_warmup else "measured"
            run_root = output_root / "runs" / case.case_id / f"{lane}-{repeat_index:02d}"
            print(
                f"[{run_number}/{total_runs}] {case.checkpoint_label} | {case.sampler_label} | "
                f"{case.preview_label} | {case.resolution_label} | {lane} {repeat_index}",
                flush=True,
            )
            row = execute_case_run(
                project_root=root,
                python_executable=python_executable,
                base_request=base_request,
                case=case,
                run_root=run_root,
                repeat_index=repeat_index,
                is_warmup=is_warmup,
                timeout_seconds=timeout_seconds,
                sample_interval_sec=sample_interval_sec,
                gpu_index=gpu_index,
                nvidia_smi_path=nvidia_smi_path,
            )
            rows.append(row)
            summaries = aggregate_runs(rows)
            write_reports(output_root, rows, summaries)
            print(
                f"  {row['status']} | generation={row.get('generation_time_sec')} sec | "
                f"VRAM peak={row.get('process_vram_peak_mb') or row.get('gpu_memory_used_peak_mb')} MB",
                flush=True,
            )

    summaries = aggregate_runs(rows)
    write_reports(output_root, rows, summaries)
    print(f"Matrix report: {output_root / 'report.md'}", flush=True)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Windows/CUDA real-checkpoint live-preview performance matrix."
    )
    parser.add_argument(
        "--config",
        default="configs/performance/gpu_preview_matrix.yml",
        help="Matrix YAML or JSON configuration.",
    )
    parser.add_argument("--project-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_matrix(args.config, project_root=args.project_root)
    except KeyboardInterrupt:
        print("Performance matrix cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Performance matrix failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
