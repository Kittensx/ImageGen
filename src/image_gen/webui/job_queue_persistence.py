from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

_QUEUE_STATE_SCHEMA_VERSION = 1
_TERMINAL_JOB_STATUSES = {"completed", "cancelled", "failed"}
_INTERRUPTED_JOB_STATUSES = {
    "preparing_model",
    "warming_model",
    "running",
    "finalizing",
}

_JobT = TypeVar("_JobT")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_non_negative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


class JobQueuePersistence:
    """Persist and restore recoverable WebUI generation queue state.

    Job payloads continue to live in each job directory's ``job.json``. This
    collaborator stores only scheduler-level state that cannot be reconstructed
    reliably from those files alone: explicit queue ordering, global queue hold,
    and the active job that should be recoverable after a graceful shutdown.
    """

    def __init__(self, context: Any) -> None:
        self.path = Path(context.data_root) / "webui" / "job-queue.json"
        self._recover_on_restart: set[str] = set()
        self._last_save_report: dict[str, Any] = {}
        self._last_restore_report: dict[str, Any] = {}

    @property
    def last_save_report(self) -> dict[str, Any]:
        return dict(self._last_save_report)

    @property
    def last_restore_report(self) -> dict[str, Any]:
        return dict(self._last_restore_report)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": _QUEUE_STATE_SCHEMA_VERSION,
            "path": str(self.path),
            "exists": self.path.is_file(),
            "last_save": self.last_save_report,
            "last_restore": self.last_restore_report,
        }

    def _state_payload(self, manager: Any) -> dict[str, Any]:
        return {
            "schema_version": _QUEUE_STATE_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "queue_order": list(manager._queued_order()),
            "queue_pause_requested": not manager._queue_resume_event.is_set(),
            "queue_pause_requested_at": manager._queue_pause_requested_at,
            "queue_pause_owner_job_id": manager._queue_pause_owner_job_id,
            "recover_on_restart": sorted(self._recover_on_restart),
        }

    def save(self, manager: Any) -> dict[str, Any]:
        payload = self._state_payload(manager)
        report: dict[str, Any] = {
            "ok": False,
            "path": str(self.path),
            "queue_order": list(payload["queue_order"]),
            "queue_pause_requested": bool(payload["queue_pause_requested"]),
            "recover_on_restart": list(payload["recover_on_restart"]),
            "saved_at": payload["updated_at"],
            "error": None,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)
            report["ok"] = True
        except OSError as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        self._last_save_report = report
        return dict(report)

    def mark_shutdown_recovery(self, manager: Any) -> dict[str, Any]:
        queued_ids = set(manager._queued_order())
        recoverable: set[str] = set()
        for job_id, job in manager.jobs.items():
            status = str(getattr(job, "status", "") or "").strip().lower()
            if status in _INTERRUPTED_JOB_STATUSES:
                recoverable.add(job_id)
            elif status == "queued" and job_id not in queued_ids:
                # Covers the very small dequeue-before-status-transition window.
                recoverable.add(job_id)
        self._recover_on_restart = recoverable
        return self.save(manager)

    def clear_shutdown_recovery(self, manager: Any) -> dict[str, Any]:
        self._recover_on_restart.clear()
        return self.save(manager)

    def _read_state(self) -> tuple[dict[str, Any], str | None]:
        if not self.path.is_file():
            return {}, None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"{type(exc).__name__}: {exc}"
        if not isinstance(payload, dict):
            return {}, "Queue persistence payload is not a JSON object."
        return payload, None

    @staticmethod
    def _job_init_values(job_type: type[_JobT], payload: Mapping[str, Any]) -> dict[str, Any]:
        init_names = {
            item.name
            for item in fields(job_type)
            if item.init and item.name != "process"
        }
        return {name: payload[name] for name in init_names if name in payload}

    @staticmethod
    def _interruption_progress(job: Any) -> tuple[int, int]:
        diagnostics = getattr(job, "model_runtime_diagnostics", {}) or {}
        orchestration = diagnostics.get("batch_orchestration") if isinstance(diagnostics, Mapping) else {}
        if not isinstance(orchestration, Mapping):
            orchestration = {}
        attempted = max(
            _coerce_non_negative_int(getattr(job, "resume_image_index", 0)),
            _coerce_non_negative_int(orchestration.get("attempted_images")),
        )
        completed = max(
            _coerce_non_negative_int(getattr(job, "resume_completed_images", 0)),
            _coerce_non_negative_int(orchestration.get("completed_images")),
        )
        return attempted, min(completed, attempted)

    @classmethod
    def _normalize_interrupted_job(cls, job: Any, *, source_status: str) -> None:
        attempted, completed = cls._interruption_progress(job)
        recovery_at = _utc_now()
        job.resume_image_index = attempted
        job.resume_completed_images = completed
        job.status = "queued"
        job.worker_stage = "queued_after_session_restart"
        job.completed_at = None
        job.return_code = None
        job.error = None
        job.pause_after_current_requested = False
        job.pause_requested_at = None
        job.paused_at = None
        job.queue_paused_from_status = None
        job.scheduler_suspended = False
        job.skip_current_requested = False
        job.skip_requested_at = None
        job.sse_clients_connected = 0
        job.process = None
        job.updated_at = recovery_at
        job.status_changed_at = recovery_at
        diagnostics = dict(getattr(job, "model_runtime_diagnostics", {}) or {})
        diagnostics["session_recovery"] = {
            "recovered_at": recovery_at,
            "source_status": source_status,
            "resume_image_index": attempted,
            "resume_completed_images": completed,
            "requires_queue_resume": True,
        }
        job.model_runtime_diagnostics = diagnostics
        log_lines = list(getattr(job, "log_lines", []) or [])
        log_lines.append(
            "SESSION RECOVERY: generation was interrupted by application shutdown/restart; "
            "the queue is held until explicitly resumed."
        )
        job.log_lines = log_lines[-80:]

    @staticmethod
    def _normalize_paused_job(job: Any) -> None:
        job.sse_clients_connected = 0
        job.process = None
        paused_from = str(getattr(job, "queue_paused_from_status", "") or "").strip().lower()
        if paused_from and paused_from != "queued":
            # A production between-images pause previously depended on the old
            # worker coroutine. After restart it must re-enter the scheduler.
            job.scheduler_suspended = True

    @staticmethod
    def _created_at_key(job: Any) -> tuple[str, str]:
        return (str(getattr(job, "created_at", "") or ""), str(getattr(job, "job_id", "") or ""))

    def restore(self, manager: Any, job_type: type[_JobT]) -> dict[str, Any]:
        state, state_error = self._read_state()
        raw_order = state.get("queue_order") if isinstance(state.get("queue_order"), list) else []
        saved_order = [str(value) for value in raw_order if str(value).strip()]
        saved_recovery = state.get("recover_on_restart") if isinstance(state.get("recover_on_restart"), list) else []
        recover_on_restart = {str(value) for value in saved_recovery if str(value).strip()}

        jobs_root = Path(manager.jobs_root)
        jobs_root.mkdir(parents=True, exist_ok=True)
        restored: dict[str, _JobT] = {}
        restored_queued_ids: set[str] = set()
        restored_paused_ids: set[str] = set()
        interrupted_ids: list[str] = []
        ignored_terminal_ids: list[str] = []
        corrupt_job_ids: list[str] = []

        for job_root in sorted((path for path in jobs_root.iterdir() if path.is_dir()), key=lambda item: item.name):
            job_file = job_root / "job.json"
            if not job_file.is_file():
                continue
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("job.json is not a JSON object")
                init_values = self._job_init_values(job_type, payload)
                init_values.setdefault("job_id", job_root.name)
                init_values.setdefault("request", {})
                job = job_type(**init_values)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                corrupt_job_ids.append(job_root.name)
                continue

            job_id = str(getattr(job, "job_id", "") or job_root.name)
            job.job_id = job_id
            job.job_root = str(job_root)
            job.process = None
            job.sse_clients_connected = 0
            source_status = str(getattr(job, "status", "") or "queued").strip().lower()
            recover_cancelled = source_status == "cancelled" and job_id in recover_on_restart
            recover_dequeued = source_status == "queued" and job_id in recover_on_restart

            if source_status == "completed" or source_status == "cancelling" or (source_status in {"cancelled", "failed"} and not recover_cancelled):
                ignored_terminal_ids.append(job_id)
                continue

            if source_status in _INTERRUPTED_JOB_STATUSES or recover_cancelled or recover_dequeued:
                self._normalize_interrupted_job(job, source_status=source_status)
                interrupted_ids.append(job_id)
                restored_queued_ids.add(job_id)
            elif source_status == "paused":
                self._normalize_paused_job(job)
                restored_paused_ids.add(job_id)
            else:
                job.status = "queued"
                job.worker_stage = str(getattr(job, "worker_stage", "") or "queued")
                restored_queued_ids.add(job_id)

            restored[job_id] = job

        manager.jobs.update(restored)

        queue_order: list[str] = []
        for job_id in interrupted_ids:
            if job_id in restored_queued_ids and job_id not in queue_order:
                queue_order.append(job_id)
        for job_id in saved_order:
            if job_id in restored_queued_ids and job_id not in queue_order:
                queue_order.append(job_id)
        orphan_jobs = sorted(
            (restored[job_id] for job_id in restored_queued_ids if job_id not in queue_order),
            key=self._created_at_key,
        )
        queue_order.extend(str(job.job_id) for job in orphan_jobs)

        manager._queue.clear()
        manager._queue.extend(queue_order)
        if queue_order:
            manager._queue_available.set()
        else:
            manager._queue_available.clear()

        saved_pause = bool(state.get("queue_pause_requested", False))
        force_pause = bool(interrupted_ids)
        queue_pause_requested = saved_pause or force_pause
        if queue_pause_requested:
            manager._queue_resume_event.clear()
        else:
            manager._queue_resume_event.set()
        manager._queue_pause_requested_at = (
            str(state.get("queue_pause_requested_at") or "") or (_utc_now() if force_pause else None)
        )
        manager._queue_pause_owner_job_id = (
            str(state.get("queue_pause_owner_job_id") or "") or (interrupted_ids[0] if interrupted_ids else None)
        )

        for job in restored.values():
            persist = getattr(manager, "_persist_job", None)
            if callable(persist):
                persist(job)

        self._recover_on_restart.clear()
        report = {
            "ok": state_error is None and not corrupt_job_ids,
            "path": str(self.path),
            "state_error": state_error,
            "restored_job_ids": sorted(restored),
            "restored_count": len(restored),
            "queue_order": list(queue_order),
            "paused_job_ids": sorted(restored_paused_ids),
            "interrupted_job_ids": list(interrupted_ids),
            "queue_pause_requested": queue_pause_requested,
            "ignored_terminal_job_ids": sorted(ignored_terminal_ids),
            "corrupt_job_ids": sorted(corrupt_job_ids),
            "restored_at": _utc_now(),
        }
        self._last_restore_report = report
        self.save(manager)
        return dict(report)
