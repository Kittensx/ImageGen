from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.systems.asset_hub.contracts import ProviderDownloadSource
from image_gen.systems.asset_hub.diagnostics import DownloadReportWriter, redact_text, strip_url_query, write_json_atomic
from image_gen.systems.asset_hub.download_settings import DownloadRuntimeSettings
from image_gen.systems.asset_hub.filenames import sanitize_filename
from image_gen.systems.asset_hub.providers.base import AssetHubError, AssetProvider
from image_gen.systems.asset_hub.repository import DownloadJobRecord, DownloadRepository, utc_now
from image_gen.systems.asset_hub.secrets import AssetHubSecretStore

DEFAULT_MAX_ACTIVE_DOWNLOADS = 2
DEFAULT_MAX_HASH_WORKERS = 1
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 6
DEFAULT_RETRIES = 3
DEFAULT_STALE_PARTIAL_AGE_SECONDS = 7 * 24 * 60 * 60
PROGRESS_INTERVAL_SECONDS = 0.25
_DISK_RESERVE_BYTES = 64 * 1024 * 1024
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_RESUMABLE_STATUSES = {"paused", "failed"}
_ACTIVE_STATUSES = {"queued", "resolving", "downloading", "verifying", "pausing", "cancelling"}
_TRANSIENT_CODES = {"download_network_error", "provider_timeout", "provider_unavailable", "provider_rate_limited"}
_UNRECOVERABLE_PARTIAL_CODES = {
    "download_hash_mismatch",
    "download_identity_changed",
    "download_redirect_blocked",
    "download_size_limit",
    "download_staging_error",
    "download_content_length_mismatch",
    "provider_bad_response",
    "provider_not_found",
}


@dataclass(frozen=True)
class DownloadPlan:
    plan_id: str
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    file_name: str
    expected_bytes: int = 0
    expected_sha256: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "planId": self.plan_id,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "fileName": self.file_name,
            "expectedBytes": self.expected_bytes,
            "expectedSha256": self.expected_sha256 or None,
            "createdAt": self.created_at,
            "installCommitted": False,
        }


class DownloadCancelled(RuntimeError):
    pass


class DownloadPaused(RuntimeError):
    pass


class _DynamicConcurrencyGate:
    def __init__(self, limit: int) -> None:
        self.limit = max(1, min(int(limit), 8))
        self.active = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self) -> None:
        async with self._condition:
            self.active = max(0, self.active - 1)
            self._condition.notify_all()

    async def set_limit(self, value: int) -> None:
        async with self._condition:
            self.limit = max(1, min(int(value), 8))
            self._condition.notify_all()


class _BandwidthLimiter:
    def __init__(self, mib_per_second: float = 0.0) -> None:
        self._rate = max(0.0, float(mib_per_second)) * 1024.0 * 1024.0
        self._next_at = 0.0
        self._lock = asyncio.Lock()

    def set_rate(self, mib_per_second: float) -> None:
        self._rate = max(0.0, float(mib_per_second)) * 1024.0 * 1024.0
        if self._rate <= 0:
            self._next_at = 0.0

    async def throttle(self, byte_count: int) -> None:
        rate = self._rate
        if rate <= 0 or byte_count <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            delay = max(0.0, start - now)
            self._next_at = start + (float(byte_count) / rate)
        if delay > 0:
            await asyncio.sleep(delay)


def _safe_error_message(value: Any, *, secret: str | None = None) -> str:
    return redact_text(value, secrets=((secret or ""),))[:512]


def _parse_content_range(value: str) -> tuple[int, int, int]:
    """Return (start, end, total) for a bytes Content-Range header.

    Invalid or wildcard components are reported as -1/-1/0 instead of being
    trusted. Resume callers must validate the returned start against the local
    partial-file size before appending any bytes.
    """
    token = str(value or "").strip()
    if not token.casefold().startswith("bytes ") or "/" not in token:
        return -1, -1, 0
    range_token, total_token = token[6:].split("/", 1)
    range_token = range_token.strip()
    total_token = total_token.strip()
    total = int(total_token) if total_token.isdigit() else 0
    if range_token == "*" or "-" not in range_token:
        return -1, -1, total
    start_token, end_token = range_token.split("-", 1)
    if not start_token.isdigit() or not end_token.isdigit():
        return -1, -1, total
    start = int(start_token)
    end = int(end_token)
    if end < start:
        return -1, -1, total
    return start, end, total


def _parse_total_from_content_range(value: str) -> int:
    return _parse_content_range(value)[2]


def _merge_note(existing: str, note: str) -> str:
    current = str(existing or "").strip()
    addition = str(note or "").strip()
    if not addition:
        return current[:512]
    if not current:
        return addition[:512]
    if addition in current:
        return current[:512]
    return f"{current} {addition}"[:512]


def _hash_existing(path: Path) -> hashlib._Hash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest


def _age_seconds(timestamp: str, *, now: datetime | None = None) -> float:
    token = str(timestamp or "").strip()
    if not token:
        return float("inf")
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (current - parsed.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return float("inf")


class AssetHubDownloadManager:
    """Persistent, bounded Asset Hub download/staging manager.

    Phase 02 deliberately stops at a verified staged payload. It never classifies,
    extracts, or commits files into live model/library roots.
    """

    def __init__(
        self,
        providers: Mapping[str, AssetProvider],
        *,
        secret_store: AssetHubSecretStore,
        repository: DownloadRepository,
        temporary_root: str | os.PathLike[str],
        report_root: str | os.PathLike[str],
        transport: httpx.AsyncBaseTransport | None = None,
        max_active_downloads: int = DEFAULT_MAX_ACTIVE_DOWNLOADS,
        max_hash_workers: int = DEFAULT_MAX_HASH_WORKERS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        retries: int = DEFAULT_RETRIES,
        timeout_seconds: float = 60.0,
        host_resolver: Callable[[str], list[str]] | None = None,
        settings: DownloadRuntimeSettings | None = None,
    ) -> None:
        self.providers = {str(key).casefold(): value for key, value in providers.items()}
        self.secret_store = secret_store
        self.repository = repository
        self.staging_root = Path(temporary_root).resolve() / "asset-hub"
        self.reports = DownloadReportWriter(report_root)
        self.transport = transport
        self.max_file_bytes = max(1024 * 1024, int(max_file_bytes))
        self.max_redirects = max(1, min(int(max_redirects), 12))
        initial = settings or DownloadRuntimeSettings(
            max_active_downloads=max_active_downloads,
            retry_attempts=retries,
        )
        self.settings = initial
        self.retries = max(0, min(int(initial.retry_attempts), 5))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self._download_slots = _DynamicConcurrencyGate(initial.max_active_downloads)
        self._bandwidth = _BandwidthLimiter(initial.bandwidth_limit_mib_per_second)
        self._hash_slots = asyncio.Semaphore(max(1, min(int(max_hash_workers), 4)))
        self._plans: dict[str, DownloadPlan] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._pause: dict[str, asyncio.Event] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._last_progress_event: dict[str, float] = {}
        self._transfer_speed: dict[str, float] = {}
        self._transfer_started: dict[str, tuple[float, int]] = {}
        self._provider_request_lock = asyncio.Lock()
        self._provider_last_request: dict[str, float] = {}
        self._host_resolver = host_resolver or self._resolve_host
        self._completion_handler: Callable[[str], Awaitable[Any]] | None = None
        self.repository.recover_interrupted()

    @staticmethod
    def _resolve_host(host: str) -> list[str]:
        try:
            return sorted({str(item[4][0]) for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
        except OSError:
            # The actual HTTP request will report a network failure. Literal private
            # addresses are still rejected below even when DNS is unavailable.
            return []

    def _provider(self, provider_id: str) -> AssetProvider:
        provider = self.providers.get(str(provider_id or "").strip().casefold())
        if provider is None:
            raise AssetHubError("provider_not_found", "Unknown asset provider.", status_code=404)
        return provider

    def set_completion_handler(self, handler: Callable[[str], Awaitable[Any]] | None) -> None:
        self._completion_handler = handler

    @staticmethod
    def _partial_path(record: DownloadJobRecord) -> Path:
        return Path(record.staging_directory) / "payload.part"

    @classmethod
    def _partial_bytes(cls, record: DownloadJobRecord) -> int:
        try:
            path = cls._partial_path(record)
            return int(path.stat().st_size) if path.is_file() else 0
        except OSError:
            return 0

    @classmethod
    def _can_resume_record(cls, record: DownloadJobRecord) -> bool:
        if record.status == "paused":
            return True
        if record.status != "failed":
            return False
        return str(record.error_code or "") not in _UNRECOVERABLE_PARTIAL_CODES

    def settings_payload(self) -> dict[str, Any]:
        payload = self.settings.to_dict()
        payload.update({
            "activeDownloads": self._download_slots.active,
            "queuedDownloads": sum(1 for item in self.repository.list(limit=500) if item.status == "queued"),
        })
        return payload

    async def update_settings(self, values: Mapping[str, Any]) -> DownloadRuntimeSettings:
        aliases = {
            "maxActiveDownloads": "max_active_downloads",
            "maxQueuedDownloads": "max_queued_downloads",
            "bandwidthLimitMiBPerSecond": "bandwidth_limit_mib_per_second",
            "providerMinRequestIntervalSeconds": "provider_min_request_interval_seconds",
            "retryAttempts": "retry_attempts",
        }
        normalized = {aliases.get(str(key), str(key)): value for key, value in dict(values or {}).items()}
        updated = self.settings.updated(normalized)
        self.settings = updated
        self.retries = updated.retry_attempts
        self._bandwidth.set_rate(updated.bandwidth_limit_mib_per_second)
        await self._download_slots.set_limit(updated.max_active_downloads)
        return updated

    async def _provider_request_turn(self, provider_id: str) -> None:
        interval = max(0.0, float(self.settings.provider_min_request_interval_seconds))
        if interval <= 0:
            return
        token = str(provider_id or "").casefold()
        async with self._provider_request_lock:
            now = time.monotonic()
            previous = self._provider_last_request.get(token, 0.0)
            delay = max(0.0, interval - (now - previous))
            if delay > 0:
                await asyncio.sleep(delay)
            self._provider_last_request[token] = time.monotonic()

    def job_payload(self, record: DownloadJobRecord) -> dict[str, Any]:
        payload = record.to_dict()
        expected = int(record.expected_bytes or 0)
        received = int(record.received_bytes or 0)
        payload["progress"] = (min(1.0, received / expected) if expected > 0 else None)
        payload["bytesPerSecond"] = round(max(0.0, self._transfer_speed.get(record.job_id, 0.0)), 2)
        partial_bytes = self._partial_bytes(record)
        payload["partialBytes"] = partial_bytes
        payload["canPause"] = record.status in {"queued", "resolving", "downloading", "verifying"}
        payload["canResume"] = self._can_resume_record(record)
        payload["canCancel"] = record.status not in _TERMINAL_STATUSES
        payload["cleanupEligible"] = bool(record.status == "failed" and partial_bytes > 0 and not payload["canResume"])
        if record.status == "queued":
            queue = [item for item in reversed(self.repository.list(limit=500)) if item.status == "queued"]
            try:
                payload["queuePosition"] = next(i for i, item in enumerate(queue, 1) if item.job_id == record.job_id)
            except StopIteration:
                payload["queuePosition"] = None
        else:
            payload["queuePosition"] = None
        return payload

    @staticmethod
    def _identity(value: Any, *, label: str) -> str:
        token = str(value or "").strip()
        if not token or len(token) > 256:
            raise AssetHubError("download_plan_invalid", f"{label} is required.", status_code=400)
        return token

    async def create_plan(
        self,
        *,
        provider_id: str,
        remote_model_id: str,
        remote_version_id: str,
        remote_file_id: str,
        file_name_hint: str = "",
        expected_bytes_hint: int = 0,
        expected_sha256_hint: str = "",
    ) -> DownloadPlan:
        provider_id = self._identity(provider_id, label="providerId").casefold()
        model_id = self._identity(remote_model_id, label="remoteModelId")
        version_id = self._identity(remote_version_id, label="remoteVersionId")
        file_id = self._identity(remote_file_id, label="remoteFileId")
        provider = self._provider(provider_id)

        # Planning normally resolves the provider once so the UI gets an
        # authoritative filename/size/hash. A transient provider outage must not
        # prevent a first-class download job from entering the persistent queue,
        # though. The worker resolves the exact provider source again before any
        # bytes are transferred and validates the provider/model/version/file IDs.
        # Browser-supplied metadata below is therefore display/progress hinting
        # only; it is never trusted as the delivery URL or final integrity proof.
        source = None
        try:
            await self._provider_request_turn(provider_id)
            source = await provider.resolve_download_source(
                model_id,
                version_id,
                file_id,
                secret=self.secret_store.get(provider_id),
            )
        except AssetHubError as exc:
            if exc.code not in _TRANSIENT_CODES:
                raise

        if source is not None and source.expected_bytes and source.expected_bytes > self.max_file_bytes:
            raise AssetHubError(
                "download_size_limit",
                "The selected provider file exceeds the configured Asset Hub download size limit.",
                status_code=413,
            )

        hint_bytes = 0
        try:
            hint_bytes = max(0, int(expected_bytes_hint or 0))
        except (TypeError, ValueError):
            hint_bytes = 0
        if hint_bytes > self.max_file_bytes:
            hint_bytes = 0
        hint_sha = str(expected_sha256_hint or "").strip().lower()
        if len(hint_sha) != 64 or any(ch not in "0123456789abcdef" for ch in hint_sha):
            hint_sha = ""

        plan = DownloadPlan(
            plan_id=str(uuid.uuid4()),
            provider_id=provider_id,
            remote_model_id=model_id,
            remote_version_id=version_id,
            remote_file_id=file_id,
            file_name=sanitize_filename(
                source.file_name if source is not None else file_name_hint,
                fallback=f"asset-{file_id}.bin",
            ),
            expected_bytes=max(0, int(source.expected_bytes or 0)) if source is not None else hint_bytes,
            expected_sha256=(str(source.expected_sha256 or "").strip().lower() if source is not None else hint_sha),
            created_at=utc_now(),
        )
        self._plans[plan.plan_id] = plan
        # Keep the in-memory plan set bounded. Plans are intentionally ephemeral;
        # persistent jobs store only normalized remote identity, never delivery URLs.
        if len(self._plans) > 256:
            for key in list(self._plans)[:64]:
                self._plans.pop(key, None)
        return plan

    def _plan(self, plan_id: str) -> DownloadPlan:
        plan = self._plans.get(str(plan_id or "").strip())
        if plan is None:
            raise AssetHubError(
                "download_plan_not_found",
                "The download plan is missing or expired. Recreate it from the provider file selection.",
                status_code=404,
            )
        return plan

    async def enqueue(self, plan_id: str) -> DownloadJobRecord:
        plan = self._plan(plan_id)
        pending = sum(1 for item in self.repository.list(limit=500) if item.status in _ACTIVE_STATUSES or item.status == "paused")
        if pending >= self.settings.max_queued_downloads:
            raise AssetHubError(
                "download_queue_full",
                f"Asset Hub download queue is at its configured limit of {self.settings.max_queued_downloads} jobs.",
                status_code=429,
            )
        job_id = str(uuid.uuid4())
        staging = self.staging_root / job_id
        staging.mkdir(parents=True, exist_ok=False)
        record = self.repository.create(DownloadJobRecord(
            job_id=job_id,
            provider_id=plan.provider_id,
            remote_model_id=plan.remote_model_id,
            remote_version_id=plan.remote_version_id,
            remote_file_id=plan.remote_file_id,
            file_name=plan.file_name,
            staging_directory=str(staging),
            status="queued",
            expected_bytes=plan.expected_bytes,
            expected_sha256=plan.expected_sha256,
        ))
        self._write_transaction_manifest(record)
        self._schedule(job_id)
        await self._publish(record, event="queued", force=True)
        return record

    def list_jobs(self, *, limit: int = 100) -> list[DownloadJobRecord]:
        return self.repository.list(limit=limit)

    def get_job(self, job_id: str) -> DownloadJobRecord:
        record = self.repository.get(str(job_id or ""))
        if record is None:
            raise AssetHubError("download_job_not_found", "Download job not found.", status_code=404)
        return record

    async def pause(self, job_id: str) -> DownloadJobRecord:
        record = self.get_job(job_id)
        if record.status in _TERMINAL_STATUSES or record.status == "paused":
            return record
        event = self._pause.setdefault(record.job_id, asyncio.Event())
        event.set()
        if record.status == "queued":
            record = self.repository.update(
                record.job_id,
                status="paused",
                error_code="",
                error_message="",
                resume_note="Paused by user before transfer started.",
            )
            await self._publish(record, event="paused", force=True)
            return record
        record = self.repository.update(record.job_id, status="pausing", error_code="", error_message="")
        await self._publish(record, event="pausing", force=True)
        return record

    async def cancel(self, job_id: str) -> DownloadJobRecord:
        record = self.get_job(job_id)
        if record.status in _TERMINAL_STATUSES:
            return record
        event = self._cancel.setdefault(record.job_id, asyncio.Event())
        event.set()
        self._pause.pop(record.job_id, None)
        if record.status in {"queued", "paused"}:
            self._discard_partial(record)
            record = self.repository.update(
                record.job_id,
                status="cancelled",
                received_bytes=0,
                actual_sha256="",
                etag="",
                last_modified="",
                resume_note="Cancelled by user; any staged partial payload was removed.",
                error_code="download_cancelled",
                error_message="Download was cancelled by the user.",
                completed_at=utc_now(),
            )
            self._write_transaction_manifest(record)
            self._write_report(record, redirect_chain=[], result="cancelled", duration_seconds=0.0, secret="")
            await self._publish(record, event="cancelled", force=True)
            await self._close_subscribers(record.job_id)
            return record
        record = self.repository.update(record.job_id, status="cancelling")
        await self._publish(record, event="cancelling", force=True)
        return record

    async def resume(self, job_id: str) -> DownloadJobRecord:
        record = self.get_job(job_id)
        if not self._can_resume_record(record):
            raise AssetHubError(
                "download_not_resumable",
                f"Download job is not resumable from status {record.status!r} ({record.error_code or 'no recoverable transfer state'}).",
                status_code=409,
            )
        self._cancel.pop(record.job_id, None)
        self._pause.pop(record.job_id, None)
        record = self.repository.update(
            record.job_id,
            status="queued",
            error_code="",
            error_message="",
            completed_at="",
        )
        self._schedule(record.job_id)
        await self._publish(record, event="queued", force=True)
        return record

    async def _finalize_completed(self, job_id: str) -> None:
        if self._completion_handler is None:
            return
        record = self.get_job(job_id)
        if record.status != "completed":
            return
        try:
            await self._completion_handler(job_id)
        except Exception as exc:
            current = self.get_job(job_id)
            note = _merge_note(current.resume_note, f"Automatic library finalization is pending: {_safe_error_message(exc)}")
            current = self.repository.update(job_id, resume_note=note)
            self._write_transaction_manifest(current)
            await self._publish(current, event="finalize_pending", force=True)

    async def reconcile_completed(self, *, limit: int = 500) -> dict[str, int]:
        attempted = 0
        for record in reversed(self.repository.list(limit=limit)):
            if record.status != "completed" or self._partial_bytes(record) <= 0:
                continue
            attempted += 1
            await self._finalize_completed(record.job_id)
        return {"attempted": attempted}

    def cleanup_job_staging(self, job_id: str) -> bool:
        record = self.get_job(job_id)
        if record.status in _ACTIVE_STATUSES or record.status == "paused":
            return False
        target = Path(record.staging_directory)
        try:
            if target.exists():
                shutil.rmtree(target)
                return True
        except OSError:
            return False
        return False

    def cleanup_stale_partials(
        self,
        *,
        max_age_seconds: float = DEFAULT_STALE_PARTIAL_AGE_SECONDS,
        include_recent_unrecoverable: bool = False,
    ) -> dict[str, Any]:
        threshold = max(0.0, float(max_age_seconds))
        now = datetime.now(timezone.utc)
        records = self.repository.list(limit=500)
        known = {str(Path(item.staging_directory).resolve()) for item in records}
        removed_files = 0
        removed_bytes = 0
        removed_orphans = 0
        preserved_resumable = 0
        cleaned_jobs: list[str] = []
        for record in records:
            partial = self._partial_path(record)
            partial_bytes = self._partial_bytes(record)
            if partial_bytes <= 0:
                continue
            if record.status in _ACTIVE_STATUSES or record.status == "paused":
                preserved_resumable += 1
                continue
            if record.status == "failed" and self._can_resume_record(record):
                preserved_resumable += 1
                continue
            if record.status == "completed":
                continue
            old_enough = _age_seconds(record.updated_at, now=now) >= threshold
            if not old_enough and not (include_recent_unrecoverable and record.status == "failed"):
                continue
            try:
                partial.unlink()
            except OSError:
                continue
            removed_files += 1
            removed_bytes += partial_bytes
            cleaned_jobs.append(record.job_id)
            self.repository.update(
                record.job_id, received_bytes=0, actual_sha256="", etag="", last_modified="",
                resume_note=_merge_note(record.resume_note, "Unrecoverable stale partial payload was cleaned."),
            )
        if self.staging_root.exists():
            for child in self.staging_root.iterdir():
                if not child.is_dir() or str(child.resolve()) in known:
                    continue
                try:
                    age = max(0.0, time.time() - child.stat().st_mtime)
                except OSError:
                    continue
                if age < threshold:
                    continue
                try:
                    size = sum(item.stat().st_size for item in child.rglob("*") if item.is_file())
                except OSError:
                    size = 0
                try:
                    shutil.rmtree(child)
                except OSError:
                    continue
                removed_orphans += 1
                removed_bytes += int(size)
        return {
            "removedFiles": removed_files, "removedOrphanDirectories": removed_orphans, "removedBytes": removed_bytes,
            "preservedResumable": preserved_resumable, "cleanedJobIds": cleaned_jobs,
        }

    def clear_history(self, *, status: str = "inactive") -> dict[str, int]:
        requested = str(status or "inactive").strip().casefold()
        if requested not in {"inactive", "completed", "failed", "cancelled"}:
            raise AssetHubError("download_clear_filter_invalid", "Unsupported download history clear filter.", status_code=400)
        removable: list[str] = []
        skipped = 0
        for record in self.repository.list(limit=500):
            if record.status not in _TERMINAL_STATUSES:
                continue
            if requested != "inactive" and record.status != requested:
                continue
            partial_bytes = self._partial_bytes(record)
            if record.status == "completed" and partial_bytes > 0:
                skipped += 1
                continue
            if record.status == "failed" and partial_bytes > 0 and self._can_resume_record(record):
                skipped += 1
                continue
            if record.staging_directory:
                try:
                    target = Path(record.staging_directory)
                    if target.exists():
                        shutil.rmtree(target)
                except OSError:
                    skipped += 1
                    continue
            removable.append(record.job_id)
        removed = self.repository.delete_many(removable)
        return {"removed": removed, "skippedRecoverable": skipped}

    async def bulk_action(self, action: str) -> dict[str, int]:
        token = str(action or "").strip().casefold()
        if token not in {"pause", "resume", "cancel"}:
            raise AssetHubError("download_bulk_action_invalid", "Unsupported download bulk action.", status_code=400)
        affected = 0
        skipped = 0
        for record in list(reversed(self.repository.list(limit=500))):
            try:
                if token == "pause":
                    if not self.job_payload(record).get("canPause"):
                        skipped += 1
                        continue
                    await self.pause(record.job_id)
                elif token == "resume":
                    if not self._can_resume_record(record):
                        skipped += 1
                        continue
                    await self.resume(record.job_id)
                else:
                    if record.status in _TERMINAL_STATUSES:
                        skipped += 1
                        continue
                    await self.cancel(record.job_id)
                affected += 1
            except AssetHubError:
                skipped += 1
        return {"affected": affected, "skipped": skipped}

    def _schedule(self, job_id: str) -> None:
        current = self._tasks.get(job_id)
        if current is not None and not current.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise AssetHubError("download_runtime_unavailable", "Download queue requires an active async runtime.", status_code=503) from exc
        task = loop.create_task(self._run(job_id), name=f"asset-hub-download-{job_id[:8]}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _task, key=job_id: self._tasks.pop(key, None))

    async def _run(self, job_id: str) -> None:
        await self._download_slots.acquire()
        try:
            secret = ""
            try:
                record = self.get_job(job_id)
                if record.status in {"cancelled", "paused"}:
                    return
                secret = self.secret_store.get(record.provider_id) or ""
                last_error: AssetHubError | None = None
                for attempt in range(self.retries + 1):
                    try:
                        await self._download_once(job_id, secret=secret)
                        await self._finalize_completed(job_id)
                        return
                    except DownloadPaused:
                        await self._finish_paused(job_id)
                        return
                    except DownloadCancelled:
                        await self._finish_cancelled(job_id)
                        return
                    except AssetHubError as exc:
                        last_error = exc
                        if exc.code not in _TRANSIENT_CODES or attempt >= self.retries:
                            raise
                        retry_after = float(exc.retry_after_seconds or 0) if exc.retry_after_seconds is not None else 0.0
                        await asyncio.sleep(max(retry_after, min(8.0, 0.5 * (2 ** attempt))))
                if last_error is not None:
                    raise last_error
            except AssetHubError as exc:
                await self._finish_failed(job_id, exc.code, _safe_error_message(exc.message, secret=secret))
            except Exception as exc:  # defensive boundary: never leak raw network/provider details
                await self._finish_failed(job_id, "download_failed", _safe_error_message(exc, secret=secret) or "Download failed.")
        finally:
            await self._download_slots.release()

    async def _download_once(self, job_id: str, *, secret: str) -> None:
        record = self.get_job(job_id)
        provider = self._provider(record.provider_id)
        record = self.repository.update(record.job_id, status="resolving", error_code="", error_message="")
        await self._publish(record, event="resolving", force=True)
        self._check_interrupted(job_id)
        await self._provider_request_turn(record.provider_id)

        source = await provider.resolve_download_source(
            record.remote_model_id,
            record.remote_version_id,
            record.remote_file_id,
            secret=secret or None,
        )
        self._validate_source_identity(record, source)
        expected_bytes = int(source.expected_bytes or record.expected_bytes or 0)
        expected_sha256 = str(source.expected_sha256 or record.expected_sha256 or "").lower()
        if expected_bytes > self.max_file_bytes:
            raise AssetHubError("download_size_limit", "Provider file exceeds the configured download size limit.", status_code=413)
        if record.expected_bytes and expected_bytes and record.expected_bytes != expected_bytes:
            self._restart_partial(record, "Provider-reported file size changed; previous partial data was discarded.")
            record = self.get_job(job_id)
        if record.expected_sha256 and expected_sha256 and record.expected_sha256 != expected_sha256:
            self._restart_partial(record, "Provider-reported SHA-256 changed; previous partial data was discarded.")
            record = self.get_job(job_id)

        record = self.repository.update(
            job_id,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            file_name=sanitize_filename(source.file_name, fallback=record.file_name),
        )
        self._ensure_disk_space(record, expected_bytes)
        self._write_transaction_manifest(record)
        await self._transfer(record, source, secret=secret)

    def _validate_source_identity(self, record: DownloadJobRecord, source: ProviderDownloadSource) -> None:
        if (
            source.provider_id.casefold() != record.provider_id.casefold()
            or source.remote_model_id != record.remote_model_id
            or source.remote_version_id != record.remote_version_id
            or source.remote_file_id != record.remote_file_id
        ):
            raise AssetHubError("download_identity_changed", "Provider download identity changed after the job was created.", status_code=409)

    def _ensure_disk_space(self, record: DownloadJobRecord, expected_bytes: int) -> None:
        target = Path(record.staging_directory)
        target.mkdir(parents=True, exist_ok=True)
        partial = target / "payload.part"
        current = partial.stat().st_size if partial.exists() else 0
        required = max(0, expected_bytes - current) if expected_bytes else min(self.max_file_bytes, 512 * 1024 * 1024)
        try:
            free = shutil.disk_usage(target).free
        except OSError as exc:
            raise AssetHubError("download_disk_check_failed", "Unable to verify staging disk capacity.", status_code=507) from exc
        if free < required + _DISK_RESERVE_BYTES:
            raise AssetHubError("download_insufficient_space", "Insufficient free disk space for the staged download.", status_code=507)

    @staticmethod
    def _discard_partial(record: DownloadJobRecord) -> None:
        partial = Path(record.staging_directory) / "payload.part"
        try:
            if partial.exists():
                partial.unlink()
        except OSError as exc:
            raise AssetHubError("download_staging_error", "Unable to remove the staged partial file safely.", status_code=500) from exc

    def _restart_partial(self, record: DownloadJobRecord, note: str) -> None:
        self._discard_partial(record)
        self.repository.update(
            record.job_id,
            received_bytes=0,
            actual_sha256="",
            etag="",
            last_modified="",
            resume_note=str(note)[:512],
        )

    async def _transfer(self, record: DownloadJobRecord, source: ProviderDownloadSource, *, secret: str) -> None:
        staging = Path(record.staging_directory)
        staging.mkdir(parents=True, exist_ok=True)
        partial = staging / "payload.part"
        existing = partial.stat().st_size if partial.exists() else 0
        # The provider/model/version/file IDs plus the provider's current source
        # metadata are the durable asset identity. ETag/Last-Modified strengthen
        # a Range continuation when available, but many CDNs (including some
        # CivitAI delivery paths) omit them. A partial transfer is still worth
        # attempting with Range because Content-Range is validated below and the
        # final provider SHA-256 remains authoritative whenever one is available.
        can_attempt_resume = bool(existing > 0)
        if existing > self.max_file_bytes:
            self._restart_partial(record, "Partial file exceeded the configured maximum and was discarded.")
            existing = 0
            can_attempt_resume = False

        headers: dict[str, str] = {"Accept": "application/octet-stream", "Accept-Encoding": "identity"}
        if can_attempt_resume:
            headers["Range"] = f"bytes={existing}-"
            validator = record.etag or record.last_modified
            if validator:
                headers["If-Range"] = validator

        response, redirect_chain, client = await self._request_with_redirects(
            source.url,
            source=source,
            secret=secret,
            headers=headers,
        )
        try:
            if response.status_code == 416 and existing:
                self._restart_partial(record, "Provider rejected the saved Range; transfer restarted from zero.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if response.status_code not in {200, 206}:
                self._raise_download_status(response)

            response_etag = str(response.headers.get("ETag") or "")[:512]
            response_modified = str(response.headers.get("Last-Modified") or "")[:512]
            range_start, range_end, total_from_range = _parse_content_range(response.headers.get("Content-Range", ""))
            resume_accepted = bool(can_attempt_resume and response.status_code == 206)
            if resume_accepted and range_start != existing:
                self._restart_partial(
                    record,
                    f"Provider returned an invalid resume range starting at {range_start}; expected byte {existing}. Transfer restarted from zero.",
                )
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if resume_accepted and range_end >= 0 and range_end < range_start:
                self._restart_partial(record, "Provider returned an invalid Content-Range; transfer restarted from zero.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if resume_accepted and record.etag and response_etag and response_etag != record.etag:
                self._restart_partial(record, "ETag changed; previous partial data was discarded and transfer restarted.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if resume_accepted and record.last_modified and response_modified and response_modified != record.last_modified:
                self._restart_partial(record, "Last-Modified changed; previous partial data was discarded and transfer restarted.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if can_attempt_resume and not resume_accepted:
                # A 200 response to a Range request explicitly means the current
                # delivery path did not honor resume. Restarting is safe because
                # this response is the complete object, but record the decision so
                # the UI/report never makes a zero-byte restart look like a resume.
                self._restart_partial(record, "Provider did not honor the Range request; transfer restarted from zero using the complete response.")
                record = self.get_job(record.job_id)
                existing = 0
            elif resume_accepted:
                validation_basis = "saved HTTP validator" if (record.etag or record.last_modified) else (
                    "provider SHA-256 + Content-Range" if (record.expected_sha256 or source.expected_sha256) else "stable provider identity + Content-Range"
                )
                record = self.repository.update(
                    record.job_id,
                    resume_count=record.resume_count + 1,
                    resume_note=f"Resumed partial transfer at byte {existing} using Range ({validation_basis}).",
                )

            content_encoding = str(response.headers.get("Content-Encoding") or "").strip().casefold()
            content_length = int(response.headers.get("Content-Length") or 0) if str(response.headers.get("Content-Length") or "").isdigit() else 0
            # Content-Length is only directly comparable to the bytes written when
            # the response is identity encoded. Range totals remain useful because
            # we explicitly request identity encoding for provider binaries.
            declared_total = total_from_range or (
                (existing + content_length) if content_length and content_encoding in {"", "identity"} else 0
            )
            metadata_expected = int(record.expected_bytes or source.expected_bytes or 0)
            if declared_total and declared_total > self.max_file_bytes:
                raise AssetHubError("download_size_limit", "HTTP response exceeds the configured Asset Hub size limit.", status_code=413)

            # Provider API file-size metadata is useful for display and disk-space
            # planning, but it can lag the delivery object. The current HTTP
            # representation is a stronger byte-count signal, while the provider
            # SHA-256 remains the authoritative integrity check when available.
            expected = declared_total or metadata_expected
            if declared_total and metadata_expected and declared_total != metadata_expected:
                record = self.repository.update(
                    record.job_id,
                    expected_bytes=declared_total,
                    resume_note=_merge_note(
                        record.resume_note,
                        f"Provider metadata size ({metadata_expected} bytes) differed from the current HTTP transfer size ({declared_total} bytes); the HTTP size was adopted and SHA-256 verification remains authoritative.",
                    ),
                )
                expected = declared_total
            self._ensure_disk_space(record, expected)

            if existing and resume_accepted:
                async with self._hash_slots:
                    digest = await asyncio.to_thread(_hash_existing, partial)
                mode = "ab"
            else:
                digest = hashlib.sha256()
                mode = "wb"
                existing = 0

            record = self.repository.update(
                record.job_id,
                status="downloading",
                received_bytes=existing,
                etag=response_etag or record.etag,
                last_modified=response_modified or record.last_modified,
            )
            await self._publish(record, event="downloading", force=True)
            started = time.monotonic()
            received = existing
            self._transfer_started[record.job_id] = (started, existing)
            self._transfer_speed[record.job_id] = 0.0
            try:
                with partial.open(mode) as stream:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        self._check_interrupted(record.job_id)
                        if not chunk:
                            continue
                        await self._bandwidth.throttle(len(chunk))
                        self._check_interrupted(record.job_id)
                        received += len(chunk)
                        if received > self.max_file_bytes:
                            raise AssetHubError("download_size_limit", "Download exceeded the configured maximum file size.", status_code=413)
                        stream.write(chunk)
                        digest.update(chunk)
                        elapsed = max(0.001, time.monotonic() - started)
                        self._transfer_speed[record.job_id] = max(0.0, (received - existing) / elapsed)
                        current = self.repository.update(record.job_id, received_bytes=received)
                        await self._publish(current, event="progress")
                    stream.flush()
                    os.fsync(stream.fileno())
            except DownloadPaused:
                raise
            except DownloadCancelled:
                raise
            except httpx.TimeoutException as exc:
                self._check_interrupted(record.job_id)
                current = self.repository.update(
                    record.job_id,
                    received_bytes=max(received, partial.stat().st_size if partial.exists() else 0),
                    resume_note=_merge_note(record.resume_note, "Provider transfer timed out; partial data was preserved for automatic/manual Range resume."),
                )
                await self._publish(current, event="interrupted", force=True)
                raise AssetHubError(
                    "provider_timeout",
                    "Provider transfer timed out before the file finished. Partial data was preserved and can be resumed.",
                    status_code=504,
                ) from exc
            except httpx.HTTPError as exc:
                self._check_interrupted(record.job_id)
                current = self.repository.update(
                    record.job_id,
                    received_bytes=max(received, partial.stat().st_size if partial.exists() else 0),
                    resume_note=_merge_note(record.resume_note, "Provider connection closed/interrupted; partial data was preserved for automatic/manual Range resume."),
                )
                await self._publish(current, event="interrupted", force=True)
                raise AssetHubError(
                    "download_network_error",
                    "Provider connection closed before the file finished. Partial data was preserved and can be resumed.",
                    status_code=502,
                ) from exc

            actual_sha256 = digest.hexdigest()
            expected_sha256 = str(record.expected_sha256 or source.expected_sha256 or "").lower()
            record = self.repository.update(record.job_id, status="verifying", received_bytes=received, actual_sha256=actual_sha256)
            await self._publish(record, event="verifying", force=True)

            hash_verified = bool(expected_sha256 and actual_sha256 == expected_sha256)
            if expected_sha256 and not hash_verified:
                raise AssetHubError("download_hash_mismatch", "Downloaded file SHA-256 does not match the provider hash.", status_code=422)

            size_reference = declared_total or expected
            if size_reference and received != size_reference:
                if not hash_verified:
                    raise AssetHubError("download_content_length_mismatch", "Received byte count does not match the current provider transfer size.", status_code=502)
                # A matching provider SHA-256 proves that the selected file arrived
                # intact even if stale metadata or an incorrect HTTP length was
                # reported. Reconcile future progress/install metadata to reality.
                record = self.repository.update(
                    record.job_id,
                    expected_bytes=received,
                    resume_note=_merge_note(
                        record.resume_note,
                        f"Transfer byte count ({received}) differed from the advertised size ({size_reference}); the matching provider SHA-256 verified the payload and the recorded size was reconciled.",
                    ),
                )
            elif int(record.expected_bytes or 0) != received and received > 0:
                record = self.repository.update(record.job_id, expected_bytes=received)

            completed = self.repository.update(
                record.job_id,
                status="completed",
                expected_bytes=received if received > 0 else int(record.expected_bytes or 0),
                received_bytes=received,
                actual_sha256=actual_sha256,
                completed_at=utc_now(),
                error_code="",
                error_message="",
            )
            self._write_transaction_manifest(completed)
            self._write_report(
                completed,
                redirect_chain=redirect_chain,
                result="verified",
                duration_seconds=max(0.0, time.monotonic() - started),
                secret=secret,
            )
            await self._publish(completed, event="completed", force=True)
            await self._close_subscribers(record.job_id)
        finally:
            await response.aclose()
            await client.aclose()

    async def _request_with_redirects(
        self,
        url: str,
        *,
        source: ProviderDownloadSource,
        secret: str,
        headers: Mapping[str, str],
    ) -> tuple[httpx.Response, list[str], httpx.AsyncClient]:
        current = str(url or "").strip()
        chain: list[str] = []
        seen: set[str] = set()
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(20.0, self.timeout_seconds))
        client = httpx.AsyncClient(
            transport=self.transport,
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": f"{PRODUCT_NAME}-AssetHub/2"},
        )
        try:
            for redirect_index in range(self.max_redirects + 1):
                self._assert_safe_url(current)
                canonical = strip_url_query(current)
                if canonical in seen:
                    raise AssetHubError("download_redirect_loop", "Provider delivery redirect loop was rejected.", status_code=502)
                seen.add(canonical)
                chain.append(canonical)
                parsed = urlsplit(current)
                request_headers = dict(headers)
                host = str(parsed.hostname or "").casefold()
                if secret and host in {item.casefold() for item in source.auth_hosts}:
                    request_headers["Authorization"] = f"Bearer {secret}"
                else:
                    request_headers.pop("Authorization", None)
                request = client.build_request("GET", current, headers=request_headers)
                try:
                    response = await client.send(request, stream=True)
                except httpx.TimeoutException as exc:
                    raise AssetHubError("provider_timeout", "Provider download request timed out.", status_code=504) from exc
                except httpx.HTTPError as exc:
                    raise AssetHubError("download_network_error", "Provider download connection failed.", status_code=502) from exc
                if response.status_code not in {301, 302, 303, 307, 308}:
                    return response, chain, client
                location = str(response.headers.get("Location") or "").strip()
                await response.aclose()
                if not location:
                    raise AssetHubError("download_redirect_invalid", "Provider returned a redirect without a Location.", status_code=502)
                if redirect_index >= self.max_redirects:
                    raise AssetHubError("download_redirect_limit", "Provider delivery exceeded the redirect limit.", status_code=502)
                current = urljoin(current, location)
        except Exception:
            await client.aclose()
            raise
        await client.aclose()
        raise AssetHubError("download_redirect_limit", "Provider delivery exceeded the redirect limit.", status_code=502)

    def _assert_safe_url(self, value: str) -> None:
        parsed = urlsplit(str(value or ""))
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise AssetHubError("download_redirect_blocked", "Only HTTPS provider delivery URLs are allowed.", status_code=400)
        if parsed.username or parsed.password:
            raise AssetHubError("download_redirect_blocked", "Credential-bearing delivery URLs are not allowed.", status_code=400)
        host = str(parsed.hostname).strip().casefold().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            raise AssetHubError("download_redirect_blocked", "Local/private delivery hosts are not allowed.", status_code=400)
        addresses: list[str] = []
        try:
            addresses.append(str(ipaddress.ip_address(host)))
        except ValueError:
            addresses.extend(self._host_resolver(host))
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise AssetHubError("download_redirect_blocked", "Private or non-routable provider delivery address was rejected.", status_code=400)

    @staticmethod
    def _raise_download_status(response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise AssetHubError("provider_auth_required", "Provider rejected download authentication.", status_code=401)
        if response.status_code == 404:
            raise AssetHubError("provider_not_found", "Provider download file was not found.", status_code=404)
        if response.status_code == 429:
            try:
                retry_after = max(0, int(float(response.headers.get("Retry-After", "0") or 0)))
            except (TypeError, ValueError):
                retry_after = None
            raise AssetHubError(
                "provider_rate_limited",
                "Provider rate-limited the download request.",
                status_code=429,
                retry_after_seconds=retry_after,
            )
        if response.status_code >= 500:
            raise AssetHubError("provider_unavailable", "Provider download service is temporarily unavailable.", status_code=502)
        raise AssetHubError("provider_bad_response", f"Provider download returned HTTP {response.status_code}.", status_code=502)

    def _check_interrupted(self, job_id: str) -> None:
        pause = self._pause.get(job_id)
        if pause is not None and pause.is_set():
            raise DownloadPaused()
        cancel = self._cancel.get(job_id)
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()

    async def _finish_paused(self, job_id: str) -> None:
        record = self.get_job(job_id)
        partial = Path(record.staging_directory) / "payload.part"
        try:
            partial_bytes = partial.stat().st_size if partial.exists() else 0
        except OSError:
            partial_bytes = 0
        record = self.repository.update(
            job_id,
            status="paused",
            received_bytes=max(int(record.received_bytes or 0), int(partial_bytes)),
            error_code="",
            error_message="",
            resume_note="Paused by user. Partial data is preserved and will be identity-checked before Range resume.",
            completed_at="",
        )
        self._write_transaction_manifest(record)
        await self._publish(record, event="paused", force=True)
        await self._close_subscribers(job_id)

    async def _finish_cancelled(self, job_id: str) -> None:
        record = self.get_job(job_id)
        self._discard_partial(record)
        record = self.repository.update(
            job_id,
            status="cancelled",
            received_bytes=0,
            actual_sha256="",
            etag="",
            last_modified="",
            resume_note="Cancelled by user; any staged partial payload was removed.",
            error_code="download_cancelled",
            error_message="Download was cancelled by the user.",
            completed_at=utc_now(),
        )
        self._write_transaction_manifest(record)
        self._write_report(record, redirect_chain=[], result="cancelled", duration_seconds=0.0, secret="")
        await self._publish(record, event="cancelled", force=True)
        await self._close_subscribers(job_id)

    async def _finish_failed(self, job_id: str, code: str, message: str) -> None:
        try:
            record = self.repository.update(
                job_id,
                status="failed",
                error_code=str(code or "download_failed")[:128],
                error_message=str(message or "Download failed.")[:512],
                completed_at=utc_now(),
            )
        except KeyError:
            return
        self._write_transaction_manifest(record)
        self._write_report(record, redirect_chain=[], result="failed", duration_seconds=0.0, secret="")
        await self._publish(record, event="failed", force=True)
        await self._close_subscribers(job_id)

    def _write_transaction_manifest(self, record: DownloadJobRecord) -> None:
        write_json_atomic(Path(record.staging_directory) / "transaction.json", {
            "schemaVersion": 1,
            "jobId": record.job_id,
            "providerId": record.provider_id,
            "remoteModelId": record.remote_model_id,
            "remoteVersionId": record.remote_version_id,
            "remoteFileId": record.remote_file_id,
            "fileName": record.file_name,
            "status": record.status,
            "expectedBytes": record.expected_bytes,
            "receivedBytes": record.received_bytes,
            "expectedSha256": record.expected_sha256 or None,
            "actualSha256": record.actual_sha256 or None,
            "updatedAt": record.updated_at,
            "installCommitted": False,
        })

    def _write_report(
        self,
        record: DownloadJobRecord,
        *,
        redirect_chain: list[str],
        result: str,
        duration_seconds: float,
        secret: str,
    ) -> None:
        self.reports.write(record.job_id, {
            "schemaVersion": 1,
            "providerIdentity": {
                "providerId": record.provider_id,
                "remoteModelId": record.remote_model_id,
                "remoteVersionId": record.remote_version_id,
                "remoteFileId": record.remote_file_id,
            },
            "fileName": record.file_name,
            "redirectHostChain": [strip_url_query(item) for item in redirect_chain],
            "expectedBytes": record.expected_bytes,
            "receivedBytes": record.received_bytes,
            "expectedHashes": {"SHA256": record.expected_sha256} if record.expected_sha256 else {},
            "actualSha256": record.actual_sha256 or None,
            "resumeBehavior": {"count": record.resume_count, "note": record.resume_note or None},
            "startedAt": record.created_at,
            "endedAt": record.completed_at or record.updated_at,
            "durationSeconds": round(max(0.0, float(duration_seconds)), 3),
            "result": result,
            "error": {"code": record.error_code, "message": record.error_message} if record.error_code else None,
            "installCommitted": False,
        }, secrets=(secret,))

    async def _publish(self, record: DownloadJobRecord, *, event: str, force: bool = False) -> None:
        now = time.monotonic()
        if event == "progress" and not force:
            last = self._last_progress_event.get(record.job_id, 0.0)
            if now - last < PROGRESS_INTERVAL_SECONDS:
                return
        self._last_progress_event[record.job_id] = now
        payload = {"event": event, "job": self.job_payload(record)}
        for queue in tuple(self._subscribers.get(record.job_id, ())):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        record = self.get_job(job_id)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=16)
        self._subscribers.setdefault(record.job_id, set()).add(queue)
        await queue.put({"event": "snapshot", "job": self.job_payload(record)})
        if record.status in _TERMINAL_STATUSES:
            await queue.put(None)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        subscribers = self._subscribers.get(job_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(job_id, None)

    async def _close_subscribers(self, job_id: str) -> None:
        for queue in tuple(self._subscribers.get(job_id, ())):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
