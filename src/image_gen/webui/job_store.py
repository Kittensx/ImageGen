from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ACTIVE_JOB_STATUSES = {
    "preparing_model",
    "warming_model",
    "running",
    "paused",
    "finalizing",
    "cancelling",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStoreMixin:
    @property
    def jobs_root(self) -> Path:
        return self.context.data_root / "webui" / "jobs"

    def clear_job_cache(
        self,
        *,
        preserve_active: bool = True,
        startup: bool = False,
    ) -> dict[str, Any]:
        """Delete session-only WebUI job data without touching final output images."""

        root = self.jobs_root
        root.mkdir(parents=True, exist_ok=True)
        active_statuses = {"queued", *_ACTIVE_JOB_STATUSES}
        preserved_ids = {
            job.job_id
            for job in self.jobs.values()
            if preserve_active and job.status in active_statuses
        }
        report: dict[str, Any] = {
            "startup": bool(startup),
            "root": str(root),
            "removed_job_ids": [],
            "removed_files": [],
            "removed_bytes": 0,
            "preserved_active": sorted(preserved_ids),
            "final_outputs_deleted": 0,
        }

        for item in list(root.iterdir()):
            if item.name in preserved_ids:
                continue
            size = self._directory_size(item) if item.is_dir() else 0
            if item.is_file():
                try:
                    size = int(item.stat().st_size)
                except OSError:
                    size = 0
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    report["removed_job_ids"].append(item.name)
                else:
                    item.unlink()
                    report["removed_files"].append(item.name)
            except OSError:
                continue
            report["removed_bytes"] += size

        for job_id, job in list(self.jobs.items()):
            if job_id in preserved_ids:
                continue
            subscribers = self._event_subscribers.pop(job_id, set())
            for queue in list(subscribers):
                self._offer_event(queue, None)
            self.jobs.pop(job_id, None)
            self._terminal_events_emitted.discard(job_id)

        report["removed_count"] = len(report["removed_job_ids"]) + len(report["removed_files"])
        report["remaining_job_directories"] = sum(1 for item in root.iterdir() if item.is_dir())
        self._last_job_cache_report = report
        return report

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return total
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += int(item.stat().st_size)
            except OSError:
                continue
        return total

    @staticmethod
    def _persist_job(job: GenerationJob) -> None:
        if not job.job_root:
            return
        root = Path(job.job_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "job.json").write_text(
            json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if job.failure_bundle_path:
            bundle = Path(job.failure_bundle_path)
            (root / "failure-link.json").write_text(
                json.dumps(
                    {
                        "failure_bundle_path": str(bundle),
                        "exists": bundle.exists(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def list_jobs(self) -> list[dict[str, Any]]:
        queue_order = self._queued_order()
        queued_ids = set(queue_order)
        active = [
            job for job in self.jobs.values()
            if job.status in _ACTIVE_JOB_STATUSES and job.status != "paused"
        ]
        active.sort(key=lambda item: item.created_at)
        queued = [self.jobs[job_id] for job_id in queue_order if job_id in self.jobs]
        paused = sorted(
            (job for job in self.jobs.values() if job.status == "paused"),
            key=lambda item: item.updated_at or item.created_at,
            reverse=True,
        )
        claimed = {job.job_id for job in active + paused} | queued_ids
        terminal = sorted(
            (job for job in self.jobs.values() if job.job_id not in claimed),
            key=lambda item: item.created_at,
            reverse=True,
        )
        return [item.to_dict() for item in active + queued + paused + terminal]
