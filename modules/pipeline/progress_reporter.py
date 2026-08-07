from __future__ import annotations

import json
import math
import shutil
import sys
import time
from typing import Any, Mapping, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _coerce_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _format_duration_ms(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
        return "--"
    milliseconds = float(value)
    if milliseconds < 1000.0:
        return f"{milliseconds:.0f}ms"
    seconds = milliseconds / 1000.0
    if seconds < 10.0:
        return f"{seconds:.2f}s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return _format_clock_ms(milliseconds)


def _format_clock_ms(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
        return "--:--"
    total_seconds = max(0, int(round(float(value) / 1000.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_gib(value: int | None) -> str:
    if value is None:
        return "--"
    return f"{float(value) / (1024.0 ** 3):.1f}G"


class ProgressReporter:
    def __init__(
        self,
        total: int = 0,
        *,
        enabled: bool = True,
        desc: str = "Sampling",
        unit: str = "step",
        machine_readable: bool = False,
        single_line: bool = True,
    ):
        self.total = int(total or 0)
        self.enabled = bool(enabled)
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.machine_readable = bool(machine_readable)
        self.single_line = bool(single_line)
        self.phase_index = 0
        self._bar = None
        self._sampling_started_monotonic: float | None = None
        self._last_update_monotonic: float | None = None
        self._step_durations_ms: list[float] = []
        self._latest_step_duration_ms: float | None = None
        self._memory_summary: dict[str, int] = {}
        self._last_plain_width = 0

    def _timing_snapshot(self) -> dict[str, float | int | None]:
        now = time.perf_counter()
        elapsed_ms = (
            max(0.0, (now - self._sampling_started_monotonic) * 1000.0)
            if self._sampling_started_monotonic is not None
            else 0.0
        )
        average_step_ms = (
            sum(self._step_durations_ms) / len(self._step_durations_ms)
            if self._step_durations_ms
            else None
        )
        rolling_window = self._step_durations_ms[-5:]
        rolling_average_step_ms = (
            sum(rolling_window) / len(rolling_window)
            if rolling_window
            else None
        )
        estimate_basis = rolling_average_step_ms or average_step_ms
        remaining_steps = max(0, int(self.total) - int(self.current))
        estimated_remaining_ms = (
            float(estimate_basis) * remaining_steps
            if estimate_basis is not None
            else None
        )
        steps_per_second = (
            1000.0 / float(average_step_ms)
            if average_step_ms is not None and average_step_ms > 0.0
            else None
        )
        return {
            "elapsed_ms": elapsed_ms,
            "average_step_ms": average_step_ms,
            "rolling_average_step_ms": rolling_average_step_ms,
            "estimated_remaining_ms": estimated_remaining_ms,
            "steps_per_second": steps_per_second,
            "timed_step_count": len(self._step_durations_ms),
        }

    @staticmethod
    def _short_description(value: str) -> str:
        text = str(value or "Sampling").strip() or "Sampling"
        if text.lower().endswith(" sampling"):
            text = text[: -len(" sampling")].rstrip()
        return text or "Sampling"

    @staticmethod
    def _terminal_columns() -> int:
        try:
            return max(40, int(shutil.get_terminal_size(fallback=(120, 24)).columns))
        except (AttributeError, OSError, TypeError, ValueError):
            return 120

    def _memory_segments(self) -> list[str]:
        if not self._memory_summary:
            return []
        used = self._memory_summary.get("used_bytes")
        total = self._memory_summary.get("total_bytes")
        requested = self._memory_summary.get("requested_bytes")
        available = self._memory_summary.get("available_bytes")
        segments: list[str] = []
        if used is not None or total is not None:
            segments.append(f"VRAM {_format_gib(used)}/{_format_gib(total)}")
        if requested is not None:
            segments.append(f"req {_format_gib(requested)}")
        if available is not None:
            segments.append(f"avail {_format_gib(available)}")
        return segments

    def _human_status_segments(self) -> tuple[list[str], list[str]]:
        timing = self._timing_snapshot()
        rate = timing["steps_per_second"]
        required = [
            f"step {_format_duration_ms(self._latest_step_duration_ms)}",
            f"avg {_format_duration_ms(timing['average_step_ms'])}",
            f"ETA {_format_clock_ms(timing['estimated_remaining_ms'])}",
        ]
        optional = self._memory_segments()
        if rate is not None:
            optional.append(f"{float(rate):.2f}/s")
        optional.append(f"t {_format_clock_ms(timing['elapsed_ms'])}")
        return required, optional

    def _human_status_text(self) -> str:
        required, optional = self._human_status_segments()
        return " | ".join(required + optional)

    def _single_line_text(self) -> str:
        total = max(0, int(self.total))
        current = max(0, int(self.current))
        percent = (current / total) * 100.0 if total > 0 else 0.0
        prefix = (
            f"{self._short_description(self.desc)} "
            f"{current}/{total} {percent:.0f}%"
        )
        required, optional = self._human_status_segments()
        columns = self._terminal_columns()
        maximum = max(1, columns - 1)
        line = " | ".join([prefix] + required)
        for segment in optional:
            candidate = f"{line} | {segment}"
            if len(candidate) <= maximum:
                line = candidate
        if len(line) > maximum:
            line = line[:maximum]
        return line

    def _render_plain_progress(self) -> None:
        if not self.enabled:
            return
        line = self._single_line_text()
        width = max(self._last_plain_width, len(line))
        sys.stdout.write("\r" + line.ljust(width))
        sys.stdout.flush()
        self._last_plain_width = len(line)

    def _refresh_human_progress(self, *, refresh: bool = True) -> None:
        if not self.enabled:
            return
        if self.single_line:
            if self._sampling_started_monotonic is not None:
                self._render_plain_progress()
        elif self._bar is not None:
            self._bar.set_postfix_str(self._human_status_text(), refresh=refresh)
        elif self._sampling_started_monotonic is not None:
            self._render_plain_progress()

    def update_memory_status(self, payload: Mapping[str, Any] | None) -> None:
        """Update the compact memory figures shown beside the sampling bar."""

        source = dict(payload or {})
        status = dict(source.get("status") or source)
        snapshot = dict(status.get("latest_snapshot") or {})
        cuda = dict(snapshot.get("cuda") or {})
        estimate = dict(status.get("latest_estimate") or {})

        used = _coerce_nonnegative_int(cuda.get("physical_used_vram_bytes"))
        if used is None:
            used = _coerce_nonnegative_int(cuda.get("allocated_vram_bytes"))
        total = _coerce_nonnegative_int(cuda.get("physical_total_vram_bytes"))
        if total is None:
            total = _coerce_nonnegative_int(cuda.get("total_vram_bytes"))
        requested = _coerce_nonnegative_int(
            estimate.get("safety_adjusted_required_bytes")
        )
        if requested is None:
            requested = _coerce_nonnegative_int(estimate.get("estimated_expected_bytes"))
        available = _coerce_nonnegative_int(estimate.get("available_bytes"))
        if available is None:
            available = _coerce_nonnegative_int(cuda.get("physical_free_vram_bytes"))
        if available is None:
            available = _coerce_nonnegative_int(cuda.get("free_vram_bytes"))

        updates = {
            "used_bytes": used,
            "total_bytes": total,
            "requested_bytes": requested,
            "available_bytes": available,
        }
        for key, value in updates.items():
            if value is not None:
                self._memory_summary[key] = value
        self._refresh_human_progress(refresh=True)

    def _emit_machine_progress(self, *, step_duration_ms: float | None = None) -> None:
        if not self.machine_readable:
            return
        total = max(0, int(self.total))
        current = max(0, int(self.current))
        percent = (current / total) * 100.0 if total > 0 else 0.0
        timing = self._timing_snapshot()
        payload = {
            "schema_version": 2,
            "phase_index": int(self.phase_index),
            "description": str(self.desc),
            "unit": str(self.unit),
            "step": current,
            "total_steps": total,
            "progress_percent": min(max(percent, 0.0), 100.0),
            "step_duration_ms": round(float(step_duration_ms), 3) if step_duration_ms is not None else None,
            "average_step_ms": round(float(timing["average_step_ms"]), 3) if timing["average_step_ms"] is not None else None,
            "rolling_average_step_ms": round(float(timing["rolling_average_step_ms"]), 3) if timing["rolling_average_step_ms"] is not None else None,
            "sampling_elapsed_ms": round(float(timing["elapsed_ms"]), 3),
            "estimated_remaining_ms": round(float(timing["estimated_remaining_ms"]), 3) if timing["estimated_remaining_ms"] is not None else None,
            "timed_step_count": int(timing["timed_step_count"] or 0),
            "updated_at_unix": time.time(),
        }
        print(
            "STEP_PROGRESS_JSON: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )

    def start(self, total: Optional[int] = None, desc: Optional[str] = None):
        if total is not None:
            self.total = int(total)
        if desc is not None:
            self.desc = desc

        self.phase_index += 1
        self.current = 0
        self._sampling_started_monotonic = time.perf_counter()
        self._last_update_monotonic = self._sampling_started_monotonic
        self._step_durations_ms = []
        self._latest_step_duration_ms = None
        self._last_plain_width = 0
        if self.enabled:
            if not self.single_line and tqdm is not None:
                self._bar = tqdm(
                    total=self.total,
                    desc=self.desc,
                    unit=self.unit,
                    dynamic_ncols=True,
                    leave=True,
                    bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar}| "
                        "{n_fmt}/{total_fmt} [{postfix}]"
                    ),
                )
                self._refresh_human_progress(refresh=True)
            else:
                self._render_plain_progress()
        self._emit_machine_progress()
        return self

    def update(self, n: int = 1):
        increment = int(n)
        now = time.perf_counter()
        step_duration_ms = None
        if self._last_update_monotonic is not None and increment > 0:
            elapsed_ms = max(0.0, (now - self._last_update_monotonic) * 1000.0)
            per_step_ms = elapsed_ms / max(1, increment)
            self._step_durations_ms.extend([per_step_ms] * increment)
            step_duration_ms = per_step_ms
            self._latest_step_duration_ms = per_step_ms
        self._last_update_monotonic = now
        self.current += increment
        if self.enabled:
            if self._bar is not None:
                self._refresh_human_progress(refresh=False)
                self._bar.update(increment)
            else:
                self._render_plain_progress()
        self._emit_machine_progress(step_duration_ms=step_duration_ms)

    def set_total(self, total: int):
        self.total = int(total)
        if self._bar is not None:
            self._bar.total = self.total
            self._refresh_human_progress(refresh=False)
            self._bar.refresh()
        elif self.enabled:
            self._render_plain_progress()
        self._emit_machine_progress()

    def close(self):
        if self._bar is not None:
            self._refresh_human_progress(refresh=False)
            self._bar.close()
            self._bar = None
        elif self.enabled and self.total:
            self._render_plain_progress()
            print()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
