from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from image_gen.webui.job_request_normalization import _coerce_boolean
from image_gen.webui.job_store import _ACTIVE_JOB_STATUSES, _utc_now


class JobPreviewMixin:
    def cleanup_preview_directories(self, *, now_timestamp: float | None = None) -> dict[str, Any]:
        """Remove only old job preview directories, never final txt2img outputs."""

        settings = self._application_settings()
        enabled = _coerce_boolean(settings.get("live_preview_cleanup_enabled", True), True)
        jobs_root = self.jobs_root
        report = {
            "enabled": enabled,
            "removed": [],
            "removed_bytes": 0,
            "preserved_active": [],
            "remaining_preview_directories": 0,
        }
        if not enabled or not jobs_root.exists():
            self._last_cleanup_report = report
            return report

        try:
            max_age_days = max(0, int(settings.get("live_preview_retention_days", 7)))
        except (TypeError, ValueError):
            max_age_days = 7
        try:
            max_jobs = max(1, int(settings.get("live_preview_retention_jobs", 24)))
        except (TypeError, ValueError):
            max_jobs = 24
        try:
            max_bytes = max(1, int(float(settings.get("live_preview_disk_budget_mb", 1024)) * 1024 * 1024))
        except (TypeError, ValueError):
            max_bytes = 1024 * 1024 * 1024

        now_value = float(now_timestamp if now_timestamp is not None else datetime.now(timezone.utc).timestamp())
        active_ids = {
            job.job_id for job in self.jobs.values()
            if job.status == "queued" or job.status in _ACTIVE_JOB_STATUSES
        }
        entries: list[dict[str, Any]] = []
        for job_root in jobs_root.iterdir():
            preview_root = job_root / "live-preview"
            if not preview_root.is_dir():
                continue
            try:
                modified = max(item.stat().st_mtime for item in [preview_root, *preview_root.glob("*")])
            except (OSError, ValueError):
                modified = preview_root.stat().st_mtime
            entries.append({
                "job_id": job_root.name,
                "path": preview_root,
                "modified": float(modified),
                "bytes": self._directory_size(preview_root),
            })

        entries.sort(key=lambda item: item["modified"], reverse=True)
        keep: list[dict[str, Any]] = []
        remove: list[dict[str, Any]] = []
        cutoff = now_value - (max_age_days * 86400) if max_age_days > 0 else None
        for entry in entries:
            if entry["job_id"] in active_ids:
                keep.append(entry)
                report["preserved_active"].append(entry["job_id"])
            elif cutoff is not None and entry["modified"] < cutoff:
                remove.append(entry)
            else:
                keep.append(entry)

        non_active_keep = [item for item in keep if item["job_id"] not in active_ids]
        for entry in non_active_keep[max_jobs:]:
            keep.remove(entry)
            remove.append(entry)

        total_bytes = sum(int(item["bytes"]) for item in keep)
        for entry in sorted(
            [item for item in keep if item["job_id"] not in active_ids],
            key=lambda item: item["modified"],
        ):
            if total_bytes <= max_bytes:
                break
            keep.remove(entry)
            remove.append(entry)
            total_bytes -= int(entry["bytes"])

        seen: set[Path] = set()
        for entry in remove:
            path = Path(entry["path"])
            if path in seen or entry["job_id"] in active_ids:
                continue
            seen.add(path)
            try:
                shutil.rmtree(path)
            except OSError:
                continue
            report["removed"].append(entry["job_id"])
            report["removed_bytes"] += int(entry["bytes"])

        report["remaining_preview_directories"] = sum(
            1 for job_root in jobs_root.iterdir() if (job_root / "live-preview").is_dir()
        )
        report["disk_budget_bytes"] = max_bytes
        report["retention_days"] = max_age_days
        report["retention_jobs"] = max_jobs
        self._last_cleanup_report = report
        return report

    def _preview_step_url(self, job: GenerationJob, step_number: int, *, updated_at: str | None = None) -> str:
        version = updated_at or job.updated_at or _utc_now()
        return f"/api/jobs/{job.job_id}/preview/{int(step_number)}?v={version}"

    def _preview_latest_url(self, job: GenerationJob, *, updated_at: str | None = None) -> str:
        version = updated_at or job.updated_at or _utc_now()
        return f"/api/jobs/{job.job_id}/preview/latest?v={version}"

    def _output_url_for_path(self, value: str | Path) -> str | None:
        try:
            output_root = self.context.txt2img_output_root.resolve()
            resolved = Path(value).expanduser().resolve()
            relative = resolved.relative_to(output_root).as_posix()
        except Exception:
            return None
        return f"/outputs/{quote(relative, safe='/')}"

    def _safe_within(self, root: Path, path: Path) -> Path | None:
        try:
            resolved_root = root.resolve()
            resolved_path = path.resolve()
            resolved_path.relative_to(resolved_root)
            return resolved_path
        except Exception:
            return None

    def live_preview_root_path(self, job: GenerationJob) -> Path | None:
        if not job.live_preview_root:
            return None
        root = Path(job.live_preview_root)
        if not root.exists():
            return root
        return self._safe_within(root, root)

    def live_preview_step_path(self, job: GenerationJob, step_number: int) -> Path | None:
        root = self.live_preview_root_path(job)
        if root is None:
            return None
        for item in reversed(job.live_preview_history):
            if int(item.get("step", 0)) != int(step_number):
                continue
            preview_path = item.get("preview_path")
            if preview_path:
                candidate = self._safe_within(root, Path(preview_path))
                if candidate is not None and candidate.is_file():
                    return candidate
            filename = item.get("filename")
            if filename:
                candidate = self._safe_within(root, root / str(filename))
                if candidate is not None and candidate.is_file() and candidate.name.startswith(f"step_{int(step_number):03d}"):
                    return candidate
        for candidate in sorted(root.glob(f"step_{int(step_number):03d}.*")):
            safe = self._safe_within(root, candidate)
            if safe is not None and safe.is_file():
                return safe
        return None

    def live_preview_latest_file(self, job: GenerationJob) -> Path | None:
        root = self.live_preview_root_path(job)
        if root is None:
            return None
        latest_json = root / "latest.json"
        if latest_json.is_file():
            try:
                latest = json.loads(latest_json.read_text(encoding="utf-8"))
            except Exception:
                latest = {}
            filename = latest.get("filename")
            if filename:
                candidate = self._safe_within(root, root / str(filename))
                if candidate is not None and candidate.is_file():
                    return candidate
        if job.live_preview_path:
            candidate = self._safe_within(root, Path(job.live_preview_path))
            if candidate is not None and candidate.is_file():
                return candidate
        return None
