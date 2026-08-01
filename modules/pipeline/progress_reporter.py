from __future__ import annotations

import json
import time
from typing import Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class ProgressReporter:
    def __init__(
        self,
        total: int = 0,
        *,
        enabled: bool = True,
        desc: str = "Sampling",
        unit: str = "step",
        machine_readable: bool = False,
    ):
        self.total = int(total or 0)
        self.enabled = bool(enabled)
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.machine_readable = bool(machine_readable)
        self.phase_index = 0
        self._bar = None
        self._sampling_started_monotonic: float | None = None
        self._last_update_monotonic: float | None = None
        self._step_durations_ms: list[float] = []

    def _emit_machine_progress(self, *, step_duration_ms: float | None = None) -> None:
        if not self.machine_readable:
            return
        total = max(0, int(self.total))
        current = max(0, int(self.current))
        percent = (current / total) * 100.0 if total > 0 else 0.0
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
        remaining_steps = max(0, total - current)
        estimated_remaining_ms = (
            float(estimate_basis) * remaining_steps
            if estimate_basis is not None
            else None
        )
        payload = {
            "schema_version": 2,
            "phase_index": int(self.phase_index),
            "description": str(self.desc),
            "unit": str(self.unit),
            "step": current,
            "total_steps": total,
            "progress_percent": min(max(percent, 0.0), 100.0),
            "step_duration_ms": round(float(step_duration_ms), 3) if step_duration_ms is not None else None,
            "average_step_ms": round(float(average_step_ms), 3) if average_step_ms is not None else None,
            "rolling_average_step_ms": round(float(rolling_average_step_ms), 3) if rolling_average_step_ms is not None else None,
            "sampling_elapsed_ms": round(float(elapsed_ms), 3),
            "estimated_remaining_ms": round(float(estimated_remaining_ms), 3) if estimated_remaining_ms is not None else None,
            "timed_step_count": len(self._step_durations_ms),
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
        if self.enabled:
            if tqdm is not None:
                self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
            else:
                print(f"{self.desc}: 0/{self.total}")
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
        self._last_update_monotonic = now
        self.current += increment
        if self.enabled:
            if self._bar is not None:
                self._bar.update(increment)
            else:
                print(f"\r{self.desc}: {self.current}/{self.total}", end="", flush=True)
        self._emit_machine_progress(step_duration_ms=step_duration_ms)

    def set_total(self, total: int):
        self.total = int(total)
        if self._bar is not None:
            self._bar.total = self.total
            self._bar.refresh()
        self._emit_machine_progress()

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif self.enabled and self.total:
            print()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
