from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from modules.txt2img.output_saver import SavedImageRecord




@dataclass(frozen=True)
class OutputSaveQueueSnapshot:
    event: str
    job_id: str
    batch_number: int
    pending_batches: int
    queue_depth: int
    active_batch_number: int | None
    completed_batches: int
    failed_batches: int
    expected_saved_count: int
    saved_count: int
    enqueued_at_unix: float
    started_at_unix: float | None
    completed_at_unix: float | None
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "job_id": self.job_id,
            "batch_number": self.batch_number,
            "pending_batches": self.pending_batches,
            "queue_depth": self.queue_depth,
            "active_batch_number": self.active_batch_number,
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "expected_saved_count": self.expected_saved_count,
            "saved_count": self.saved_count,
            "enqueued_at_unix": self.enqueued_at_unix,
            "started_at_unix": self.started_at_unix,
            "completed_at_unix": self.completed_at_unix,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass
class OutputSaveTicket:
    job_id: str
    batch_number: int
    prepared_request: Any
    enqueued_at_unix: float = field(default_factory=time.time)
    started_at_unix: float | None = None
    completed_at_unix: float | None = None
    records: list[SavedImageRecord] = field(default_factory=list)
    error: BaseException | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def mark_completed(self, records: list[SavedImageRecord]) -> None:
        self.records = list(records)
        self.completed_at_unix = time.time()
        self._done.set()

    def mark_failed(self, error: BaseException) -> None:
        self.error = error
        self.completed_at_unix = time.time()
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result(self) -> list[SavedImageRecord]:
        self._done.wait()
        if self.error is not None:
            raise RuntimeError(
                f"Async output save failed for job {self.job_id} batch {self.batch_number}: {self.error}"
            ) from self.error
        return list(self.records)


class AsyncOutputSaveQueue:
    """Single-worker CPU/disk save pipeline for resident generation.

    The worker serializes all output writes so filename sequencing remains stable
    while allowing the main generation loop to begin the next batch as soon as
    CPU images and metadata are handed off.
    """

    def __init__(
        self,
        save_callable: Callable[[Any], list[SavedImageRecord]],
        *,
        on_enqueued: Callable[[OutputSaveTicket, OutputSaveQueueSnapshot], None] | None = None,
        on_started: Callable[[OutputSaveTicket, OutputSaveQueueSnapshot], None] | None = None,
        on_saved: Callable[[OutputSaveTicket, list[SavedImageRecord], OutputSaveQueueSnapshot], None] | None = None,
        on_error: Callable[[OutputSaveTicket, BaseException, OutputSaveQueueSnapshot], None] | None = None,
    ) -> None:
        self._save_callable = save_callable
        self._on_enqueued = on_enqueued
        self._on_started = on_started
        self._on_saved = on_saved
        self._on_error = on_error
        self._queue: queue.Queue[OutputSaveTicket | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._shutdown = False
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending_batches = 0
        self._completed_batches = 0
        self._failed_batches = 0
        self._active_batch_number: int | None = None

    def _expected_saved_count(self, ticket: OutputSaveTicket) -> int:
        expected = getattr(ticket.prepared_request, "expected_count", None)
        try:
            return max(0, int(expected or 0))
        except (TypeError, ValueError):
            return 0

    def _snapshot(
        self,
        ticket: OutputSaveTicket,
        *,
        event: str,
        saved_count: int = 0,
        error: BaseException | None = None,
    ) -> OutputSaveQueueSnapshot:
        with self._state_lock:
            pending_batches = int(self._pending_batches)
            completed_batches = int(self._completed_batches)
            failed_batches = int(self._failed_batches)
            active_batch_number = self._active_batch_number
            queue_depth = max(0, pending_batches - (1 if active_batch_number is not None else 0))
        return OutputSaveQueueSnapshot(
            event=event,
            job_id=str(ticket.job_id),
            batch_number=int(ticket.batch_number),
            pending_batches=pending_batches,
            queue_depth=queue_depth,
            active_batch_number=active_batch_number,
            completed_batches=completed_batches,
            failed_batches=failed_batches,
            expected_saved_count=self._expected_saved_count(ticket),
            saved_count=max(0, int(saved_count or 0)),
            enqueued_at_unix=float(ticket.enqueued_at_unix),
            started_at_unix=ticket.started_at_unix,
            completed_at_unix=ticket.completed_at_unix,
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
        )

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown = False
            self._thread = threading.Thread(
                target=self._worker_loop,
                name='imagegen-output-save-worker',
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, prepared_request: Any, *, job_id: str, batch_number: int) -> OutputSaveTicket:
        if self._shutdown:
            raise RuntimeError('The async output save queue has been shut down.')
        self._ensure_started()
        ticket = OutputSaveTicket(
            job_id=str(job_id),
            batch_number=int(batch_number),
            prepared_request=prepared_request,
        )
        with self._state_lock:
            self._pending_batches += 1
        self._queue.put(ticket)
        if self._on_enqueued is not None:
            self._on_enqueued(ticket, self._snapshot(ticket, event="enqueued"))
        return ticket

    def _worker_loop(self) -> None:
        while True:
            ticket = self._queue.get()
            try:
                if ticket is None:
                    return
                ticket.started_at_unix = time.time()
                with self._state_lock:
                    self._active_batch_number = int(ticket.batch_number)
                if self._on_started is not None:
                    self._on_started(ticket, self._snapshot(ticket, event="started"))
                try:
                    records = list(self._save_callable(ticket.prepared_request))
                except BaseException as exc:  # pragma: no cover - surfaced to waiter.
                    ticket.mark_failed(exc)
                    with self._state_lock:
                        self._pending_batches = max(0, self._pending_batches - 1)
                        self._failed_batches += 1
                        self._active_batch_number = None
                    if self._on_error is not None:
                        self._on_error(ticket, exc, self._snapshot(ticket, event="failed", error=exc))
                else:
                    ticket.mark_completed(records)
                    with self._state_lock:
                        self._pending_batches = max(0, self._pending_batches - 1)
                        self._completed_batches += 1
                        self._active_batch_number = None
                    if self._on_saved is not None:
                        self._on_saved(
                            ticket,
                            records,
                            self._snapshot(ticket, event="completed", saved_count=len(records)),
                        )
            finally:
                self._queue.task_done()

    def shutdown(self, *, wait: bool = True) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._thread is None:
            return
        self._queue.put(None)
        if wait and self._thread.is_alive():
            self._thread.join(timeout=15.0)
        self._thread = None
