from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from image_gen.webui.diagnostics import write_webui_failure_bundle
from image_gen.webui.job_request_normalization import _coerce_boolean
from image_gen.webui.job_store import _ACTIVE_JOB_STATUSES, _utc_now


def _timestamp_from_iso(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


class JobWatchdogMixin:
    def _watchdog_settings(self) -> dict[str, Any]:
        settings = self._application_settings()
        enabled = _coerce_boolean(settings.get("queue_watchdog_enabled", True), True)
        try:
            interval = max(2.0, float(settings.get("queue_watchdog_interval_seconds", 5) or 5.0))
        except (TypeError, ValueError):
            interval = 5.0
        try:
            running_timeout = max(30.0, float(settings.get("queue_watchdog_running_stall_timeout_seconds", 180) or 180.0))
        except (TypeError, ValueError):
            running_timeout = 180.0
        try:
            transition_timeout = max(20.0, float(settings.get("queue_watchdog_transition_stall_timeout_seconds", 120) or 120.0))
        except (TypeError, ValueError):
            transition_timeout = 120.0
        try:
            model_transition_timeout = max(
                transition_timeout,
                float(
                    settings.get(
                        "queue_watchdog_model_transition_stall_timeout_seconds",
                        settings.get("queue_watchdog_advanced_model_transition_stall_timeout_seconds", 600),
                    )
                    or 600.0
                ),
            )
        except (TypeError, ValueError):
            model_transition_timeout = max(transition_timeout, 600.0)
        try:
            advanced_model_transition_timeout = max(
                model_transition_timeout,
                float(settings.get("queue_watchdog_advanced_model_transition_stall_timeout_seconds", 600) or 600.0),
            )
        except (TypeError, ValueError):
            advanced_model_transition_timeout = max(model_transition_timeout, 600.0)
        try:
            finalizing_timeout = max(60.0, float(settings.get("queue_watchdog_finalizing_stall_timeout_seconds", 600) or 600.0))
        except (TypeError, ValueError):
            finalizing_timeout = 600.0
        default_gap_threshold = max(30.0, interval * 3.0)
        try:
            suspension_gap_threshold = max(
                interval * 2.0,
                float(
                    settings.get(
                        "queue_watchdog_suspension_gap_threshold_seconds",
                        default_gap_threshold,
                    )
                    or default_gap_threshold
                ),
            )
        except (TypeError, ValueError):
            suspension_gap_threshold = default_gap_threshold
        self._watchdog_report.update(
            {
                "enabled": enabled,
                "interval_seconds": interval,
                "running_stall_timeout_seconds": running_timeout,
                "transition_stall_timeout_seconds": transition_timeout,
                "model_transition_stall_timeout_seconds": model_transition_timeout,
                "advanced_model_transition_stall_timeout_seconds": advanced_model_transition_timeout,
                "finalizing_stall_timeout_seconds": finalizing_timeout,
                "suspension_gap_threshold_seconds": suspension_gap_threshold,
            }
        )
        return {
            "enabled": enabled,
            "interval_seconds": interval,
            "running_stall_timeout_seconds": running_timeout,
            "transition_stall_timeout_seconds": transition_timeout,
            "model_transition_stall_timeout_seconds": model_transition_timeout,
            "advanced_model_transition_stall_timeout_seconds": advanced_model_transition_timeout,
            "finalizing_stall_timeout_seconds": finalizing_timeout,
            "suspension_gap_threshold_seconds": suspension_gap_threshold,
        }

    def _job_last_activity_timestamp(self, job: GenerationJob) -> float:
        candidates = [
            _timestamp_from_iso(job.last_progress_at),
            _timestamp_from_iso(job.last_runtime_line_at),
            _timestamp_from_iso(job.updated_at),
            _timestamp_from_iso(job.status_changed_at),
            _timestamp_from_iso(job.started_at),
        ]
        if not any(value is not None for value in candidates):
            candidates.append(_timestamp_from_iso(job.created_at))
        return max((value for value in candidates if value is not None), default=datetime.now(timezone.utc).timestamp())

    def _job_stall_reason(
        self,
        job: GenerationJob,
        *,
        now_timestamp: float,
        runtime_status: Mapping[str, Any] | None,
        settings: Mapping[str, Any],
    ) -> str | None:
        if job.status not in _ACTIVE_JOB_STATUSES or job.status == "paused":
            return None
        if job.status == "running":
            timeout = float(settings.get("running_stall_timeout_seconds"))
        elif job.status == "finalizing":
            timeout = float(settings.get("finalizing_stall_timeout_seconds"))
        else:
            timeout = float(settings.get("transition_stall_timeout_seconds"))
        last_activity_timestamp = self._job_last_activity_timestamp(job)
        observation_floor = getattr(self, "_watchdog_observation_floor_timestamp", None)
        if observation_floor is not None:
            try:
                last_activity_timestamp = max(last_activity_timestamp, float(observation_floor))
            except (TypeError, ValueError):
                pass
        stale_for = max(0.0, now_timestamp - last_activity_timestamp)
        if job.execution_mode == "resident_model":
            worker_job_id = str((runtime_status or {}).get("current_job_id") or "").strip()
            worker_stage = str((runtime_status or {}).get("stage") or "idle").strip().lower()
            worker_online = bool((runtime_status or {}).get("online", True))
            if not worker_online and stale_for >= min(timeout, 15.0):
                return f"Model runtime went offline while {job.job_id} remained {job.status}."
            if worker_job_id and worker_job_id == job.job_id:
                owned_timeout = timeout
                model_transition_stages = {
                    "preparing_model",
                    "loading_tokenizer",
                    "loading_checkpoint",
                    "moving_to_gpu",
                    "reusing_checkpoint",
                    "model_ready",
                    "applying_retention_policy",
                }
                job_stage = str(getattr(job, "worker_stage", "") or "").strip().lower()
                if (
                    job.status in {"preparing_model", "warming_model"}
                    or job_stage in model_transition_stages
                    or worker_stage in model_transition_stages
                ):
                    owned_timeout = max(
                        owned_timeout,
                        float(settings.get("model_transition_stall_timeout_seconds") or owned_timeout),
                    )
                if bool(job.request.get("advanced_models_enabled")):
                    owned_timeout = max(
                        owned_timeout,
                        float(settings.get("advanced_model_transition_stall_timeout_seconds") or owned_timeout),
                    )
                return None if stale_for < owned_timeout else (
                    f"Model runtime still reports {job.job_id} active, but no runtime activity was observed for {stale_for:.1f} seconds."
                )
            if stale_for >= min(timeout, 15.0) and not worker_job_id and worker_stage in {"idle", "ready"}:
                if job.status == "finalizing" and (job.output_paths or (job.total_steps and job.current_step >= job.total_steps)):
                    return None
                return (
                    "Model runtime no longer reports an active job, but the queue still marked "
                    f"{job.job_id} as {job.status}."
                )
            if stale_for >= min(timeout, 15.0) and worker_stage in {"failed", "offline"}:
                return f"Model runtime entered {worker_stage} while {job.job_id} remained {job.status}."
        elif job.process is None and stale_for >= min(timeout, 15.0):
            return f"The isolated generation process disappeared while {job.job_id} remained {job.status}."
        if stale_for >= timeout:
            return f"No runtime activity was observed for {stale_for:.1f} seconds while {job.job_id} remained {job.status}."
        return None

    def _capture_watchdog_failure_bundle(
        self,
        job: GenerationJob,
        *,
        reason: str,
        source: str,
    ) -> str | None:
        if str(job.failure_bundle_path or "").strip():
            return str(job.failure_bundle_path)
        diagnostics = self._diagnostics_request_settings(self._application_settings())
        if not bool(diagnostics.get("failure_bundles")):
            return None
        try:
            request_path = None
            if job.job_root:
                candidate = Path(job.job_root) / "request.json"
                if candidate.is_file():
                    request_path = str(candidate)
            bundle = write_webui_failure_bundle(
                project_root=self.context.project_root,
                stage=f"generation_{str(source or 'watchdog').strip().lower()}_recovery",
                error=RuntimeError(reason),
                payload=dict(job.request or {}),
                request_path=request_path,
                extra={
                    "job_id": job.job_id,
                    "job_status": job.status,
                    "worker_stage": job.worker_stage,
                    "execution_mode": job.execution_mode,
                    "model_runtime_status": self.model_runtime.status(),
                    "model_runtime_diagnostics": dict(job.model_runtime_diagnostics or {}),
                    "log_tail": list(job.log_lines[-80:]),
                },
            )
        except Exception as exc:  # pragma: no cover - failure capture must not mask the original failure
            job.log_lines.append(
                f"{str(source or 'watchdog').upper()} FAILURE BUNDLE CAPTURE FAILED: {type(exc).__name__}: {exc}"
            )
            return None
        job.failure_bundle_path = str(bundle)
        job.log_lines.append(f"(failure bundle: {bundle})")
        return str(bundle)

    async def _recover_terminal_job(
        self,
        job: GenerationJob,
        *,
        reason: str,
        source: str,
    ) -> dict[str, Any]:
        timestamp = _utc_now()
        entry = {
            "timestamp": timestamp,
            "source": source,
            "reason": reason,
        }
        job.log_lines.append(f"{source.upper()} RECOVERY: {reason}")
        job.model_runtime_diagnostics.setdefault("recovery_actions", []).append(entry)
        job.model_diagnostics.setdefault("recovery", []).append(entry)
        if job.execution_mode == "resident_model":
            try:
                await self.model_runtime.stop()
                entry["model_runtime_stopped"] = True
            except Exception as exc:  # pragma: no cover - best effort
                entry["model_runtime_stop_error"] = f"{type(exc).__name__}: {exc}"
        elif job.process is not None:
            try:
                job.process.terminate()
                entry["process_terminated"] = True
            except Exception as exc:  # pragma: no cover - best effort
                entry["process_terminate_error"] = f"{type(exc).__name__}: {exc}"
        job.error = reason
        job.return_code = 130 if job.status == "cancelling" else 1
        terminal_status = "cancelled" if job.status == "cancelling" else "failed"
        self._transition_job(job, status=terminal_status, worker_stage=terminal_status)
        if terminal_status == "failed":
            self._capture_watchdog_failure_bundle(job, reason=reason, source=source)
        job.completed_at = timestamp
        job.process = None
        self._watchdog_report["recoveries"] = int(self._watchdog_report.get("recoveries", 0) or 0) + 1
        self._watchdog_report["last_recovery_at"] = timestamp
        self._watchdog_report["last_recovery_reason"] = reason
        self._watchdog_report["last_recovery_job_id"] = job.job_id
        if job.execution_mode == "resident_model":
            self._finalize_resident_job(job)
        else:
            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
            if job.job_root:
                (Path(job.job_root) / "model-diagnostics.json").write_text(
                    json.dumps(job.model_diagnostics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._persist_job(job)
            terminal_event = {
                "completed": "job-completed",
                "cancelled": "job-cancelled",
                "failed": "job-failed",
            }.get(job.status, "job-progress")
            self._publish_terminal_once(job, terminal_event)
            self.cleanup_preview_directories()
        return entry

    async def _run_watchdog_check(self) -> None:
        settings = self._watchdog_settings()
        self._watchdog_report["last_check_at"] = _utc_now()
        self._watchdog_report["checks"] = int(self._watchdog_report.get("checks", 0) or 0) + 1
        if not settings["enabled"]:
            return
        now_timestamp = datetime.now(timezone.utc).timestamp()
        runtime_status = self.model_runtime.status()
        for job in list(self.jobs.values()):
            if (
                job.execution_mode == "resident_model"
                and job.status == "finalizing"
                and not str(runtime_status.get("current_job_id") or "").strip()
                and str(runtime_status.get("stage") or "").lower() in {"ready", "idle"}
                and (job.output_paths or (job.total_steps and job.current_step >= job.total_steps))
            ):
                job.return_code = 0
                self._transition_job(job, status="completed", worker_stage="completed")
                self._finalize_resident_job(job)
                continue
            reason = self._job_stall_reason(job, now_timestamp=now_timestamp, runtime_status=runtime_status, settings=settings)
            if reason:
                await self._recover_terminal_job(job, reason=reason, source="watchdog")
                break

    def _record_watchdog_observation_gap(
        self,
        *,
        observed_gap: float,
        resumed_timestamp: float,
        threshold: float,
    ) -> bool:
        gap_seconds = max(0.0, float(observed_gap))
        if gap_seconds < max(0.0, float(threshold)):
            return False
        self._watchdog_observation_floor_timestamp = float(resumed_timestamp)
        self._watchdog_report["observation_gap_count"] = int(
            self._watchdog_report.get("observation_gap_count", 0) or 0
        ) + 1
        self._watchdog_report["last_observation_gap_seconds"] = round(gap_seconds, 3)
        self._watchdog_report["last_observation_resumed_at"] = datetime.fromtimestamp(
            float(resumed_timestamp), timezone.utc
        ).isoformat()
        return True

    async def _watchdog_loop(self) -> None:
        while not self._stopping:
            settings = self._watchdog_settings()
            self._watchdog_report["running"] = True
            try:
                sleep_started_timestamp = datetime.now(timezone.utc).timestamp()
                await asyncio.sleep(float(settings["interval_seconds"]))
                resumed_timestamp = datetime.now(timezone.utc).timestamp()
                observed_gap = max(0.0, resumed_timestamp - sleep_started_timestamp)
                gap_threshold = float(settings.get("suspension_gap_threshold_seconds") or 30.0)
                # We cannot call time spent while this loop was suspended worker
                # inactivity. Restart observation from resume and require a fresh
                # full stall window before recovery.
                self._record_watchdog_observation_gap(
                    observed_gap=observed_gap,
                    resumed_timestamp=resumed_timestamp,
                    threshold=gap_threshold,
                )
                await self._run_watchdog_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive watchdog logging
                self._watchdog_report["last_error"] = f"{type(exc).__name__}: {exc}"
        self._watchdog_report["running"] = False

    async def force_stop_generation(
        self,
        *,
        reason: str = "Force stop requested from the WebUI.",
    ) -> dict[str, Any]:
        """Immediately stop active generation work and clear queued jobs as cancellation."""
        active = self._active_generation_job()
        if active is not None and active.status != "cancelling":
            self._transition_job(active, status="cancelling", worker_stage="force_stopping")
            active.pause_after_current_requested = False
            active.skip_current_requested = False
            self._persist_job(active)
            self._publish_event(active, "job-progress", force_stop_requested=True)
        return await self.recover_worker(
            clear_active=True,
            clear_queue=True,
            reason=reason,
        )

    async def recover_worker(
        self,
        *,
        clear_active: bool = True,
        clear_queue: bool = False,
        reason: str = "Manual recovery requested from the WebUI.",
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "reason": reason,
            "clear_active": bool(clear_active),
            "clear_queue": bool(clear_queue),
            "active_job_id": None,
            "worker_stopped": False,
            "queue": None,
        }
        active = next((job for job in self.jobs.values() if job.status in _ACTIVE_JOB_STATUSES), None)
        if clear_active and active is not None:
            report["active_job_id"] = active.job_id
            await self._recover_terminal_job(active, reason=reason, source="manual")
            report["worker_stopped"] = True
        else:
            try:
                await self.model_runtime.stop()
                report["worker_stopped"] = True
            except Exception as exc:
                report["worker_stop_error"] = f"{type(exc).__name__}: {exc}"
        if clear_queue:
            report["queue"] = self.clear_queued_jobs(reason="Queued jobs were cleared during manual recovery.")
        return report
