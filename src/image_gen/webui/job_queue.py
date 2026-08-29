from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from image_gen.webui.job_store import _ACTIVE_JOB_STATUSES, _utc_now

_CANCELLABLE_JOB_STATUSES = {"preparing_model", "warming_model", "running", "paused"}


class _ReorderableJobQueue(deque[str]):
    """Deque-backed scheduler queue with the legacy qsize() test/debug helper."""

    def qsize(self) -> int:
        return len(self)


class JobQueueControlMixin:
    def clear_queued_jobs(self, *, reason: str = "Queued jobs were cleared from the WebUI.") -> dict[str, Any]:
        report = {
            "cleared_job_ids": [],
            "cleared_count": 0,
            "reason": reason,
        }
        for job in self.jobs.values():
            if job.status != "queued":
                continue
            self._remove_queued_job_id(job.job_id)
            timestamp = self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.completed_at = timestamp
            job.error = reason
            job.log_lines.append(f"MANUAL CLEAR: {reason}")
            self._persist_job(job)
            self._publish_terminal_once(job, "job-cancelled")
            report["cleared_job_ids"].append(job.job_id)
        report["cleared_count"] = len(report["cleared_job_ids"])
        self._queue_persistence.save(self)
        return report

    def dismiss_terminal_jobs(self) -> dict[str, Any]:
        terminal = {"completed", "cancelled", "failed"}
        removed: list[str] = []
        for job_id, job in list(self.jobs.items()):
            if job.status not in terminal:
                continue
            subscribers = self._event_subscribers.pop(job_id, set())
            for queue in list(subscribers):
                self._offer_event(queue, None)
            self.jobs.pop(job_id, None)
            self._terminal_events_emitted.discard(job_id)
            removed.append(job_id)
        return {"removed_job_ids": removed, "removed_count": len(removed)}

    async def cancel(self, job_id: str) -> GenerationJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            self._remove_queued_job_id(job_id)
            completed = self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.completed_at = completed
            self._persist_job(job)
            self._queue_persistence.save(self)
            self._publish_terminal_once(job, "job-cancelled")
        elif job.status in _CANCELLABLE_JOB_STATUSES:
            was_paused = job.status == "paused"
            self._transition_job(job, status="cancelling", worker_stage="cancelling")
            job.pause_after_current_requested = False
            job.skip_current_requested = False
            resume_event = self._job_resume_events.get(job.job_id)
            if resume_event is not None:
                resume_event.set()
            paused_without_live_worker = was_paused and (
                job.scheduler_suspended or job.queue_paused_from_status == "queued"
            )
            if paused_without_live_worker:
                # Individually paused queued jobs and interrupted jobs recovered after
                # restart have no live worker coroutine waiting on their resume event.
                # Cancellation must therefore become terminal here instead of waiting
                # for an execution path that no longer exists.
                job.scheduler_suspended = False
                job.queue_paused_from_status = None
                self._transition_job(job, status="cancelled", worker_stage="cancelled")
                job.return_code = 130
                self._job_resume_events.pop(job.job_id, None)
                if job.execution_mode == "resident_model":
                    self._finalize_resident_job(job)
                else:
                    job.completed_at = _utc_now()
                    self._persist_job(job)
                    self._publish_terminal_once(job, "job-cancelled")
            else:
                if job.execution_mode == "resident_model" and not was_paused:
                    await self.model_runtime.cancel_active(job.job_id)
                elif job.process is not None:
                    job.process.terminate()
                self._persist_job(job)
                self._publish_event(job, "job-progress")
        return job

    def _active_generation_job(self) -> GenerationJob | None:
        return next(
            (job for job in self.jobs.values() if job.status in _ACTIVE_JOB_STATUSES and job.status != "paused"),
            None,
        )

    def _enqueue_job_id(self, job_id: str, *, front: bool = False) -> None:
        self._remove_queued_job_id(job_id)
        if front:
            self._queue.appendleft(job_id)
        else:
            self._queue.append(job_id)
        self._queue_available.set()
        self._queue_persistence.save(self)

    def _remove_queued_job_id(self, job_id: str) -> bool:
        removed = False
        if not self._queue:
            return False
        kept = _ReorderableJobQueue()
        for candidate in self._queue:
            if candidate == job_id:
                removed = True
                continue
            kept.append(candidate)
        self._queue = kept
        if not self._queue:
            self._queue_available.clear()
        if removed:
            self._queue_persistence.save(self)
        return removed

    def _queued_order(self) -> list[str]:
        return [job_id for job_id in self._queue if self.jobs.get(job_id) is not None and self.jobs[job_id].status == "queued"]

    def reorder_job(self, job_id: str, direction: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "queued":
            raise ValueError("Only queued jobs can be reordered. Pause the active job before changing what runs next.")
        order = self._queued_order()
        if job_id not in order:
            raise ValueError("The job is not currently in the schedulable queue.")
        index = order.index(job_id)
        token = str(direction or "").strip().lower()
        target = index - 1 if token in {"up", "higher", "-1"} else index + 1 if token in {"down", "lower", "1"} else index
        target = max(0, min(len(order) - 1, target))
        if target != index:
            order.pop(index)
            order.insert(target, job_id)
            self._queue = _ReorderableJobQueue(order)
            self._queue_available.set()
            self._queue_persistence.save(self)
        return self.status()

    async def pause_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "queued":
            self._remove_queued_job_id(job_id)
            job.queue_paused_from_status = "queued"
            job.pause_requested_at = _utc_now()
            job.paused_at = job.pause_requested_at
            self._transition_job(job, status="paused", worker_stage="paused_in_queue")
            self._persist_job(job)
            self._queue_persistence.save(self)
            self._publish_event(job, "job-paused", paused_at=job.paused_at, queue_item_paused=True)
            return self.status()
        active = self._active_generation_job()
        if active is not None and active.job_id == job_id:
            return await self.pause_after_current(job_id, hold_queue=False)
        if job.status == "paused":
            return self.status()
        raise ValueError("This job cannot be paused in its current state.")

    async def resume_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "paused":
            raise ValueError("Only paused jobs can be moved back into the queue.")
        paused_from = job.queue_paused_from_status
        job.pause_after_current_requested = False
        job.pause_requested_at = None
        job.resumed_at = _utc_now()
        job.resume_count += 1
        job.queue_paused_from_status = None
        resume_event = self._resume_event_for_job(job.job_id)
        if job.scheduler_suspended or paused_from == "queued":
            job.scheduler_suspended = False
            self._transition_job(job, status="queued", worker_stage="queued_after_resume")
            self._persist_job(job)
            self._enqueue_job_id(job_id)
        else:
            self._transition_job(job, status="running", worker_stage="resuming_queue")
            self._persist_job(job)
            resume_event.set()
        self._publish_event(job, "job-progress", resumed_at=job.resumed_at, queue_item_resumed=True)
        return self.status()

    def _resume_event_for_job(self, job_id: str) -> asyncio.Event:
        event = self._job_resume_events.get(job_id)
        if event is None:
            event = asyncio.Event()
            event.set()
            self._job_resume_events[job_id] = event
        return event

    async def pause_after_current(
        self,
        job_id: str | None = None,
        *,
        hold_queue: bool = True,
    ) -> dict[str, Any]:
        active = self._active_generation_job()
        if job_id and (active is None or active.job_id != str(job_id)):
            raise ValueError("The requested generation is not the active queue item.")
        if active is None:
            if hold_queue:
                self._queue_pause_requested_at = _utc_now()
                self._queue_pause_owner_job_id = None
                self._queue_resume_event.clear()
                self._queue_persistence.save(self)
            return self.status()
        if active.status in {"finalizing", "cancelling"}:
            raise ValueError("The active generation can no longer be paused between images.")

        requested_at = _utc_now()
        if hold_queue:
            self._queue_pause_requested_at = requested_at
            self._queue_pause_owner_job_id = active.job_id
            self._queue_resume_event.clear()
        active.pause_after_current_requested = True
        active.pause_requested_at = requested_at
        active.resumed_at = None
        self._resume_event_for_job(active.job_id).clear()
        self._transition_job(active, worker_stage="pause_after_current_requested")
        self._persist_job(active)
        self._queue_persistence.save(self)
        self._publish_event(
            active,
            "job-progress",
            pause_after_current_requested=True,
            queue_pause_requested=True,
            queue_item_pause_requested=True,
        )
        return self.status()

    async def resume_queue(self) -> dict[str, Any]:
        """Compatibility action: cancel a pending active pause and requeue all held jobs."""
        resumed_at = _utc_now()
        active = self._active_generation_job()
        if active is not None and active.pause_after_current_requested:
            active.pause_after_current_requested = False
            active.pause_requested_at = None
            active.resumed_at = resumed_at
            active.resume_count += 1
            self._transition_job(active, worker_stage="running")
            self._persist_job(active)
            self._publish_event(
                active,
                "job-progress",
                queue_pause_requested=False,
                resumed_at=resumed_at,
            )

        for paused in list(self.jobs.values()):
            if paused.status != "paused":
                continue
            paused_from = paused.queue_paused_from_status
            paused.pause_after_current_requested = False
            paused.pause_requested_at = None
            paused.resumed_at = resumed_at
            paused.resume_count += 1
            paused.queue_paused_from_status = None
            resume_event = self._resume_event_for_job(paused.job_id)
            if paused.scheduler_suspended or paused_from == "queued":
                paused.scheduler_suspended = False
                self._transition_job(paused, status="queued", worker_stage="queued_after_resume")
                self._persist_job(paused)
                self._enqueue_job_id(paused.job_id)
            else:
                self._transition_job(paused, status="running", worker_stage="resuming_queue")
                self._persist_job(paused)
                resume_event.set()
            self._publish_event(
                paused,
                "job-progress",
                resumed_at=resumed_at,
                queue_item_resumed=True,
            )

        self._queue_pause_requested_at = None
        self._queue_pause_owner_job_id = None
        # Kept set for compatibility with older diagnostics that inspect this event.
        self._queue_resume_event.set()
        self._queue_persistence.save(self)
        return self.status()

    async def skip_current(self, job_id: str) -> GenerationJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "running":
            raise ValueError("Skip is only available while an image is actively generating.")
        runtime_status = self.model_runtime.status()
        if str(runtime_status.get("current_job_id") or "") != job.job_id:
            raise ValueError("The resident runtime is not currently sampling this generation.")
        if job.skip_current_requested:
            return job

        job.skip_current_requested = True
        job.skip_requested_at = _utc_now()
        self._transition_job(job, worker_stage="skipping_current_image")
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            skip_current_requested=True,
            skip_requested_at=job.skip_requested_at,
        )
        await self.model_runtime.cancel_active(job.job_id)
        return job
