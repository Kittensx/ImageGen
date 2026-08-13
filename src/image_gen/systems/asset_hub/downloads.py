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
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.systems.asset_hub.contracts import ProviderDownloadSource
from image_gen.systems.asset_hub.diagnostics import DownloadReportWriter, redact_text, strip_url_query, write_json_atomic
from image_gen.systems.asset_hub.filenames import sanitize_filename
from image_gen.systems.asset_hub.providers.base import AssetHubError, AssetProvider
from image_gen.systems.asset_hub.repository import DownloadJobRecord, DownloadRepository, utc_now
from image_gen.systems.asset_hub.secrets import AssetHubSecretStore

DEFAULT_MAX_ACTIVE_DOWNLOADS = 2
DEFAULT_MAX_HASH_WORKERS = 1
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 6
DEFAULT_RETRIES = 3
PROGRESS_INTERVAL_SECONDS = 0.25
_DISK_RESERVE_BYTES = 64 * 1024 * 1024
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_RESUMABLE_STATUSES = {"paused", "failed"}
_TRANSIENT_CODES = {"download_network_error", "provider_timeout", "provider_unavailable", "provider_rate_limited"}


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


def _safe_error_message(value: Any, *, secret: str | None = None) -> str:
    return redact_text(value, secrets=((secret or ""),))[:512]


def _parse_total_from_content_range(value: str) -> int:
    token = str(value or "").strip()
    if "/" not in token:
        return 0
    tail = token.rsplit("/", 1)[-1].strip()
    if not tail.isdigit():
        return 0
    return int(tail)


def _hash_existing(path: Path) -> hashlib._Hash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest


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
    ) -> None:
        self.providers = {str(key).casefold(): value for key, value in providers.items()}
        self.secret_store = secret_store
        self.repository = repository
        self.staging_root = Path(temporary_root).resolve() / "asset-hub"
        self.reports = DownloadReportWriter(report_root)
        self.transport = transport
        self.max_file_bytes = max(1024 * 1024, int(max_file_bytes))
        self.max_redirects = max(1, min(int(max_redirects), 12))
        self.retries = max(0, min(int(retries), 5))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self._download_slots = asyncio.Semaphore(max(1, min(int(max_active_downloads), 8)))
        self._hash_slots = asyncio.Semaphore(max(1, min(int(max_hash_workers), 4)))
        self._plans: dict[str, DownloadPlan] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._last_progress_event: dict[str, float] = {}
        self._host_resolver = host_resolver or self._resolve_host
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
    ) -> DownloadPlan:
        provider_id = self._identity(provider_id, label="providerId").casefold()
        model_id = self._identity(remote_model_id, label="remoteModelId")
        version_id = self._identity(remote_version_id, label="remoteVersionId")
        file_id = self._identity(remote_file_id, label="remoteFileId")
        provider = self._provider(provider_id)
        source = await provider.resolve_download_source(
            model_id,
            version_id,
            file_id,
            secret=self.secret_store.get(provider_id),
        )
        if source.expected_bytes and source.expected_bytes > self.max_file_bytes:
            raise AssetHubError(
                "download_size_limit",
                "The selected provider file exceeds the configured Asset Hub download size limit.",
                status_code=413,
            )
        plan = DownloadPlan(
            plan_id=str(uuid.uuid4()),
            provider_id=provider_id,
            remote_model_id=model_id,
            remote_version_id=version_id,
            remote_file_id=file_id,
            file_name=sanitize_filename(source.file_name, fallback=f"asset-{file_id}.bin"),
            expected_bytes=max(0, int(source.expected_bytes or 0)),
            expected_sha256=str(source.expected_sha256 or "").strip().lower(),
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

    async def cancel(self, job_id: str) -> DownloadJobRecord:
        record = self.get_job(job_id)
        if record.status in _TERMINAL_STATUSES:
            return record
        event = self._cancel.setdefault(record.job_id, asyncio.Event())
        event.set()
        if record.status in {"queued", "paused"}:
            record = self.repository.update(
                record.job_id,
                status="cancelled",
                error_code="download_cancelled",
                error_message="Download was cancelled before transfer resumed.",
                completed_at=utc_now(),
            )
        else:
            record = self.repository.update(record.job_id, status="cancelling")
        await self._publish(record, event="cancelling", force=True)
        return record

    async def resume(self, job_id: str) -> DownloadJobRecord:
        record = self.get_job(job_id)
        if record.status not in _RESUMABLE_STATUSES:
            raise AssetHubError(
                "download_not_resumable",
                f"Download job is not resumable from status {record.status!r}.",
                status_code=409,
            )
        self._cancel.pop(record.job_id, None)
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
        async with self._download_slots:
            secret = ""
            try:
                record = self.get_job(job_id)
                if record.status == "cancelled":
                    return
                secret = self.secret_store.get(record.provider_id) or ""
                last_error: AssetHubError | None = None
                for attempt in range(self.retries + 1):
                    try:
                        await self._download_once(job_id, secret=secret)
                        return
                    except DownloadCancelled:
                        await self._finish_cancelled(job_id)
                        return
                    except AssetHubError as exc:
                        last_error = exc
                        if exc.code not in _TRANSIENT_CODES or attempt >= self.retries:
                            raise
                        await asyncio.sleep(min(4.0, 0.5 * (2 ** attempt)))
                if last_error is not None:
                    raise last_error
            except AssetHubError as exc:
                await self._finish_failed(job_id, exc.code, _safe_error_message(exc.message, secret=secret))
            except Exception as exc:  # defensive boundary: never leak raw network/provider details
                await self._finish_failed(job_id, "download_failed", _safe_error_message(exc, secret=secret) or "Download failed.")

    async def _download_once(self, job_id: str, *, secret: str) -> None:
        record = self.get_job(job_id)
        provider = self._provider(record.provider_id)
        record = self.repository.update(record.job_id, status="resolving", error_code="", error_message="")
        await self._publish(record, event="resolving", force=True)
        self._check_cancelled(job_id)

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

    def _restart_partial(self, record: DownloadJobRecord, note: str) -> None:
        partial = Path(record.staging_directory) / "payload.part"
        try:
            if partial.exists():
                partial.unlink()
        except OSError as exc:
            raise AssetHubError("download_staging_error", "Unable to reset the staged partial file safely.", status_code=500) from exc
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
        can_attempt_resume = bool(existing > 0 and (record.etag or record.last_modified))
        if existing > self.max_file_bytes:
            self._restart_partial(record, "Partial file exceeded the configured maximum and was discarded.")
            existing = 0
            can_attempt_resume = False

        headers: dict[str, str] = {"Accept": "application/octet-stream"}
        if can_attempt_resume:
            headers["Range"] = f"bytes={existing}-"
            headers["If-Range"] = record.etag or record.last_modified

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
            resume_accepted = bool(can_attempt_resume and response.status_code == 206)
            if resume_accepted and record.etag and response_etag and response_etag != record.etag:
                self._restart_partial(record, "ETag changed; previous partial data was discarded and transfer restarted.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if resume_accepted and record.last_modified and response_modified and response_modified != record.last_modified:
                self._restart_partial(record, "Last-Modified changed; previous partial data was discarded and transfer restarted.")
                return await self._transfer(self.get_job(record.job_id), source, secret=secret)
            if can_attempt_resume and not resume_accepted:
                self._restart_partial(record, "Provider did not honor Range/identity validation; transfer restarted from zero.")
                record = self.get_job(record.job_id)
                existing = 0
            elif resume_accepted:
                record = self.repository.update(
                    record.job_id,
                    resume_count=record.resume_count + 1,
                    resume_note="Resumed a verified partial transfer using Range and saved remote identity.",
                )

            content_length = int(response.headers.get("Content-Length") or 0) if str(response.headers.get("Content-Length") or "").isdigit() else 0
            total_from_range = _parse_total_from_content_range(response.headers.get("Content-Range", ""))
            declared_total = total_from_range or ((existing + content_length) if content_length else 0)
            expected = int(record.expected_bytes or source.expected_bytes or 0)
            if declared_total and declared_total > self.max_file_bytes:
                raise AssetHubError("download_size_limit", "HTTP response exceeds the configured Asset Hub size limit.", status_code=413)
            if expected and declared_total and declared_total != expected:
                raise AssetHubError("download_content_length_mismatch", "Provider response size does not match the selected file metadata.", status_code=502)
            self._ensure_disk_space(record, expected or declared_total)

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
            with partial.open(mode) as stream:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    self._check_cancelled(record.job_id)
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > self.max_file_bytes:
                        raise AssetHubError("download_size_limit", "Download exceeded the configured maximum file size.", status_code=413)
                    stream.write(chunk)
                    digest.update(chunk)
                    current = self.repository.update(record.job_id, received_bytes=received)
                    await self._publish(current, event="progress")
                stream.flush()
                os.fsync(stream.fileno())

            actual_sha256 = digest.hexdigest()
            expected_sha256 = str(record.expected_sha256 or source.expected_sha256 or "").lower()
            if expected and received != expected:
                raise AssetHubError("download_content_length_mismatch", "Received byte count does not match the selected provider file.", status_code=502)
            record = self.repository.update(record.job_id, status="verifying", received_bytes=received, actual_sha256=actual_sha256)
            await self._publish(record, event="verifying", force=True)
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise AssetHubError("download_hash_mismatch", "Downloaded file SHA-256 does not match the provider hash.", status_code=422)

            completed = self.repository.update(
                record.job_id,
                status="completed",
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
            raise AssetHubError("provider_rate_limited", "Provider rate-limited the download request.", status_code=429)
        if response.status_code >= 500:
            raise AssetHubError("provider_unavailable", "Provider download service is temporarily unavailable.", status_code=502)
        raise AssetHubError("provider_bad_response", f"Provider download returned HTTP {response.status_code}.", status_code=502)

    def _check_cancelled(self, job_id: str) -> None:
        event = self._cancel.get(job_id)
        if event is not None and event.is_set():
            raise DownloadCancelled()

    async def _finish_cancelled(self, job_id: str) -> None:
        record = self.get_job(job_id)
        partial = Path(record.staging_directory) / "payload.part"
        partial_bytes = 0
        try:
            partial_bytes = partial.stat().st_size if partial.exists() else 0
        except OSError:
            partial_bytes = 0
        received_bytes = max(int(record.received_bytes or 0), int(partial_bytes))
        if received_bytes > 0:
            status = "paused"
            message = "Download was cancelled; verified remote identity is required before resuming the partial file."
            completed_at = ""
        else:
            status = "cancelled"
            message = "Download was cancelled before resumable data was staged."
            completed_at = utc_now()
        record = self.repository.update(
            job_id,
            status=status,
            received_bytes=received_bytes,
            error_code="download_cancelled",
            error_message=message,
            completed_at=completed_at,
        )
        self._write_transaction_manifest(record)
        self._write_report(record, redirect_chain=[], result=status, duration_seconds=0.0, secret="")
        await self._publish(record, event=status, force=True)
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
        payload = {"event": event, "job": record.to_dict()}
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
        await queue.put({"event": "snapshot", "job": record.to_dict()})
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
