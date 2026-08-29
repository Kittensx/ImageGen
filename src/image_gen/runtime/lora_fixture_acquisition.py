from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from image_gen.runtime.adapters.compatibility import AdapterCompatibilityService
from image_gen.runtime.lora_inspector import inspect_lora_file
from image_gen.systems.asset_hub.contracts import ProviderFile, ProviderModel, ProviderSearchPage, ProviderSearchRequest, ProviderVersion
from image_gen.systems.asset_hub.downloads import AssetHubDownloadManager
from image_gen.systems.asset_hub.providers.base import AssetHubError, AssetProvider
from image_gen.systems.asset_hub.providers.civitai import CivitaiProvider
from image_gen.systems.asset_hub.repository import DownloadJobRecord, DownloadRepository
from image_gen.systems.asset_hub.secrets import AssetHubSecretStore
from image_gen.webui.civitai_asset_metadata import (
    CivitaiCredentialError,
    read_civitai_api_key,
    sync_civitai_api_key_to_secret_store,
)
from modules.project_context import ProjectContext

FIXTURE_MANIFEST_SCHEMA_VERSION = 1
CACHE_INDEX_SCHEMA_VERSION = 1
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "paused"}


@dataclass
class FixtureSearchSpec:
    query: str = ""
    creator: str = ""
    base_models: list[str] = field(default_factory=list)
    safe_content: bool = True
    sort: str = "Highest Rated"
    period: str = "AllTime"
    limit: int = 12

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FixtureSearchSpec":
        payload = dict(value or {})
        return cls(
            query=str(payload.get("query") or "").strip(),
            creator=str(payload.get("creator") or "").strip(),
            base_models=[str(item).strip() for item in payload.get("base_models") or [] if str(item).strip()],
            safe_content=bool(payload.get("safe_content", True)),
            sort=str(payload.get("sort") or "Highest Rated").strip(),
            period=str(payload.get("period") or "AllTime").strip(),
            limit=max(1, min(int(payload.get("limit") or 12), 100)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "creator": self.creator,
            "base_models": list(self.base_models),
            "safe_content": self.safe_content,
            "sort": self.sort,
            "period": self.period,
            "limit": self.limit,
        }


@dataclass
class FixtureSelection:
    remote_model_id: str = ""
    remote_version_id: str = ""
    remote_file_id: str = ""
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FixtureSelection":
        payload = dict(value or {})
        return cls(
            remote_model_id=str(payload.get("remote_model_id") or "").strip(),
            remote_version_id=str(payload.get("remote_version_id") or "").strip(),
            remote_file_id=str(payload.get("remote_file_id") or "").strip(),
            notes=str(payload.get("notes") or "").strip(),
        )

    def is_pinned(self) -> bool:
        return bool(self.remote_model_id and self.remote_version_id and self.remote_file_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "remote_model_id": self.remote_model_id,
            "remote_version_id": self.remote_version_id,
            "remote_file_id": self.remote_file_id,
            "notes": self.notes,
        }


@dataclass
class FixtureManifestEntry:
    fixture_id: str
    display_name: str
    provider: str = "civitai"
    asset_kind: str = "lora"
    model_family: str = ""
    adapter_format: str = ""
    target_profile: str = ""
    required: bool = True
    notes: str = ""
    search: FixtureSearchSpec = field(default_factory=FixtureSearchSpec)
    selection: FixtureSelection = field(default_factory=FixtureSelection)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixtureManifestEntry":
        payload = dict(value or {})
        fixture_id = str(payload.get("fixture_id") or payload.get("id") or "").strip()
        display_name = str(payload.get("display_name") or payload.get("name") or fixture_id).strip()
        if not fixture_id:
            raise ValueError("fixture_id is required in the fixture manifest")
        return cls(
            fixture_id=fixture_id,
            display_name=display_name or fixture_id,
            provider=str(payload.get("provider") or "civitai").strip().casefold(),
            asset_kind=str(payload.get("asset_kind") or "lora").strip().casefold(),
            model_family=str(payload.get("model_family") or "").strip().casefold(),
            adapter_format=str(payload.get("adapter_format") or "").strip(),
            target_profile=str(payload.get("target_profile") or "").strip(),
            required=bool(payload.get("required", True)),
            notes=str(payload.get("notes") or "").strip(),
            search=FixtureSearchSpec.from_mapping(payload.get("search")),
            selection=FixtureSelection.from_mapping(payload.get("selection")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "display_name": self.display_name,
            "provider": self.provider,
            "asset_kind": self.asset_kind,
            "model_family": self.model_family,
            "adapter_format": self.adapter_format,
            "target_profile": self.target_profile,
            "required": self.required,
            "notes": self.notes,
            "search": self.search.to_dict(),
            "selection": self.selection.to_dict(),
        }


@dataclass
class FixtureManifest:
    path: Path
    fixtures: list[FixtureManifestEntry]
    schema_version: int = FIXTURE_MANIFEST_SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "FixtureManifest":
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Fixture manifest must be a mapping at the document root.")
        fixtures = [FixtureManifestEntry.from_mapping(item) for item in payload.get("fixtures") or []]
        return cls(
            path=source,
            schema_version=int(payload.get("schema_version") or FIXTURE_MANIFEST_SCHEMA_VERSION),
            fixtures=fixtures,
        )

    def save(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "fixtures": [entry.to_dict() for entry in self.fixtures],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def get(self, fixture_id: str) -> FixtureManifestEntry:
        token = str(fixture_id or "").strip()
        for entry in self.fixtures:
            if entry.fixture_id == token:
                return entry
        raise KeyError(token)


@dataclass
class AcquiredFixtureRecord:
    fixture_id: str
    display_name: str
    provider: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    local_path: str
    file_name: str
    sha256: str
    size_bytes: int
    acquired_at: str
    source_page_url: str = ""
    model_name: str = ""
    version_name: str = ""
    model_family: str = ""
    adapter_format: str = ""
    inspection: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    selection_notes: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquiredFixtureRecord":
        payload = dict(value)
        return cls(
            fixture_id=str(payload.get("fixture_id") or "").strip(),
            display_name=str(payload.get("display_name") or "").strip(),
            provider=str(payload.get("provider") or "").strip().casefold(),
            remote_model_id=str(payload.get("remote_model_id") or "").strip(),
            remote_version_id=str(payload.get("remote_version_id") or "").strip(),
            remote_file_id=str(payload.get("remote_file_id") or "").strip(),
            local_path=str(payload.get("local_path") or "").strip(),
            file_name=str(payload.get("file_name") or "").strip(),
            sha256=str(payload.get("sha256") or "").strip().lower(),
            size_bytes=max(0, int(payload.get("size_bytes") or 0)),
            acquired_at=str(payload.get("acquired_at") or "").strip(),
            source_page_url=str(payload.get("source_page_url") or "").strip(),
            model_name=str(payload.get("model_name") or "").strip(),
            version_name=str(payload.get("version_name") or "").strip(),
            model_family=str(payload.get("model_family") or "").strip().casefold(),
            adapter_format=str(payload.get("adapter_format") or "").strip(),
            inspection=dict(payload.get("inspection") or {}),
            compatibility=dict(payload.get("compatibility") or {}),
            selection_notes=str(payload.get("selection_notes") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixtureStatusRow:
    fixture_id: str
    display_name: str
    provider: str
    model_family: str
    adapter_format: str
    pinned: bool
    acquired: bool
    local_path: str = ""
    sha256: str = ""
    selection_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixtureCacheIndex:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, AcquiredFixtureRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        fixtures = payload.get("fixtures") or {}
        results: dict[str, AcquiredFixtureRecord] = {}
        for fixture_id, raw in fixtures.items():
            try:
                record = AcquiredFixtureRecord.from_mapping(raw)
            except Exception:
                continue
            if record.fixture_id:
                results[record.fixture_id] = record
            elif str(fixture_id).strip():
                record.fixture_id = str(fixture_id).strip()
                results[record.fixture_id] = record
        return results

    def save(self, records: Mapping[str, AcquiredFixtureRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_INDEX_SCHEMA_VERSION,
            "fixtures": {key: value.to_dict() for key, value in sorted(records.items())},
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class FixtureAcquisitionService:
    def __init__(
        self,
        context: ProjectContext,
        *,
        providers: Mapping[str, AssetProvider] | None = None,
        secret_store: AssetHubSecretStore | None = None,
        transport: Any | None = None,
    ) -> None:
        self.context = context
        self.secret_store = secret_store or AssetHubSecretStore()
        if not self.secret_store.get("civitai"):
            try:
                sync_civitai_api_key_to_secret_store(self.context, self.secret_store)
            except CivitaiCredentialError:
                pass
        self.cache_root = Path(self.context.cache_root) / "qualification" / "lora_fixtures"
        self.file_root = self.cache_root / "files"
        self.index = FixtureCacheIndex(self.cache_root / "fixture_index.json")
        self.transport = transport
        self.providers = dict(providers or self._default_providers())

    def _default_providers(self) -> dict[str, AssetProvider]:
        cache_root = Path(self.context.cache_root) / "asset_hub" / "providers" / "civitai"
        store = self.secret_store

        def _civitai_secret() -> str | None:
            value = store.get("civitai")
            if value:
                return value
            try:
                _path, fallback = read_civitai_api_key(self.context)
            except CivitaiCredentialError:
                return None
            return fallback or None

        return {
            "civitai": CivitaiProvider(
                cache_root=cache_root,
                transport=self.transport,
                secret_provider=_civitai_secret,
            )
        }

    def _provider(self, provider_id: str) -> AssetProvider:
        token = str(provider_id or "").strip().casefold()
        provider = self.providers.get(token)
        if provider is None:
            raise AssetHubError("provider_not_found", f"Unknown fixture provider: {provider_id}", status_code=404)
        return provider

    def _download_manager(self) -> AssetHubDownloadManager:
        return AssetHubDownloadManager(
            self.providers,
            secret_store=self.secret_store,
            repository=DownloadRepository(self.cache_root / "downloads.sqlite3"),
            temporary_root=Path(self.context.temporary_root) / "qualification" / "lora_fixtures",
            report_root=self.cache_root / "download_reports",
            transport=self.transport,
        )

    def status_rows(self, manifest: FixtureManifest) -> list[FixtureStatusRow]:
        cache = self.index.load()
        rows: list[FixtureStatusRow] = []
        for entry in manifest.fixtures:
            record = cache.get(entry.fixture_id)
            acquired = False
            local_path = ""
            sha256 = ""
            if record is not None:
                candidate = Path(record.local_path)
                if candidate.is_file():
                    acquired = True
                    local_path = str(candidate)
                    sha256 = record.sha256
            selection_summary = ""
            if entry.selection.is_pinned():
                selection_summary = f"{entry.selection.remote_model_id}/{entry.selection.remote_version_id}/{entry.selection.remote_file_id}"
            rows.append(FixtureStatusRow(
                fixture_id=entry.fixture_id,
                display_name=entry.display_name,
                provider=entry.provider,
                model_family=entry.model_family,
                adapter_format=entry.adapter_format,
                pinned=entry.selection.is_pinned(),
                acquired=acquired,
                local_path=local_path,
                sha256=sha256,
                selection_summary=selection_summary,
            ))
        return rows

    async def search(self, entry: FixtureManifestEntry, *, limit: int | None = None, refresh: bool = False) -> ProviderSearchPage:
        provider = self._provider(entry.provider)
        request = ProviderSearchRequest(
            query=entry.search.query or entry.display_name,
            asset_kind=entry.asset_kind or "lora",
            base_models=tuple(entry.search.base_models),
            creator=entry.search.creator,
            sort=entry.search.sort,
            period=entry.search.period,
            safe_content=entry.search.safe_content,
            limit=max(1, min(int(limit or entry.search.limit or 12), 100)),
            refresh=refresh,
        )
        return await provider.search(request)

    async def get_model(self, provider_id: str, remote_model_id: str) -> ProviderModel:
        return await self._provider(provider_id).get_model(remote_model_id)

    async def get_version(self, provider_id: str, remote_model_id: str, remote_version_id: str) -> ProviderVersion:
        return await self._provider(provider_id).get_version(remote_model_id, remote_version_id)

    async def acquire(
        self,
        manifest: FixtureManifest,
        fixture_id: str,
        *,
        remote_model_id: str = "",
        remote_version_id: str = "",
        remote_file_id: str = "",
        persist_selection: bool = False,
        force: bool = False,
    ) -> AcquiredFixtureRecord:
        entry = manifest.get(fixture_id)
        selection = FixtureSelection(
            remote_model_id=remote_model_id or entry.selection.remote_model_id,
            remote_version_id=remote_version_id or entry.selection.remote_version_id,
            remote_file_id=remote_file_id or entry.selection.remote_file_id,
            notes=entry.selection.notes,
        )
        if not selection.is_pinned():
            raise ValueError(
                f"Fixture '{entry.fixture_id}' does not have a pinned provider selection. "
                "Use the search command to inspect candidates and pass remote ids to acquire."
            )

        cached = self.index.load().get(entry.fixture_id)
        if cached and not force:
            local = Path(cached.local_path)
            if local.is_file() and cached.sha256 == self._sha256(local):
                return cached

        provider = self._provider(entry.provider)
        version = await provider.get_version(selection.remote_model_id, selection.remote_version_id)
        file_entry = self._select_file(version, selection.remote_file_id)

        manager = self._download_manager()
        plan = await manager.create_plan(
            provider_id=entry.provider,
            remote_model_id=selection.remote_model_id,
            remote_version_id=selection.remote_version_id,
            remote_file_id=selection.remote_file_id,
        )
        job = await manager.enqueue(plan.plan_id)
        completed = await self._wait_for_job(manager, job.job_id)
        if completed.status != "completed":
            reason = completed.error_message or completed.error_code or completed.status
            raise RuntimeError(f"Fixture acquisition failed for '{entry.fixture_id}': {reason}")

        payload = Path(completed.staging_directory) / "payload.part"
        if not payload.is_file():
            raise FileNotFoundError(f"Expected staged payload is missing: {payload}")
        destination_dir = self.file_root / entry.fixture_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / completed.file_name
        shutil.copy2(payload, destination_path)
        sha256 = self._sha256(destination_path)
        inspection = inspect_lora_file(destination_path)
        compatibility = AdapterCompatibilityService().evaluate(inspection, active_checkpoint_family=entry.model_family).to_dict()
        record = AcquiredFixtureRecord(
            fixture_id=entry.fixture_id,
            display_name=entry.display_name,
            provider=entry.provider,
            remote_model_id=selection.remote_model_id,
            remote_version_id=selection.remote_version_id,
            remote_file_id=selection.remote_file_id,
            local_path=str(destination_path),
            file_name=completed.file_name,
            sha256=sha256,
            size_bytes=destination_path.stat().st_size,
            acquired_at=self._utc_now(),
            source_page_url=file_entry.source_page_url,
            model_name=getattr(version, "name", "") or entry.display_name,
            version_name=version.name,
            model_family=entry.model_family,
            adapter_format=entry.adapter_format,
            inspection=dict(inspection),
            compatibility=compatibility,
            selection_notes=selection.notes,
        )
        cache = self.index.load()
        cache[record.fixture_id] = record
        self.index.save(cache)

        if persist_selection:
            entry.selection = selection
            manifest.save()
        return record

    @staticmethod
    def _select_file(version: ProviderVersion, remote_file_id: str) -> ProviderFile:
        selected = str(remote_file_id or "").strip()
        for file_entry in version.files:
            if file_entry.remote_file_id == selected:
                return file_entry
        raise ValueError(f"Provider version '{version.remote_version_id}' does not expose file id '{selected}'.")

    @staticmethod
    async def _wait_for_job(manager: AssetHubDownloadManager, job_id: str) -> DownloadJobRecord:
        queue = await manager.subscribe(job_id)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                job = dict(event.get("job") or {})
                if str(job.get("status") or "") in _TERMINAL_JOB_STATUSES:
                    break
            return manager.get_job(job_id)
        finally:
            manager.unsubscribe(job_id, queue)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


def default_manifest_path(project_root: str | os.PathLike[str]) -> Path:
    return Path(project_root) / "testing" / "test_validations" / "lora" / "aq03_provider_fixture_manifest.yaml"


async def _async_main(args: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Acquire and inspect provider-backed LoRA qualification fixtures.")
    parser.add_argument("command", choices=("status", "search", "show", "acquire", "inspect"))
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--fixture-id", default="")
    parser.add_argument("--provider", default="civitai")
    parser.add_argument("--remote-model-id", default="")
    parser.add_argument("--remote-version-id", default="")
    parser.add_argument("--remote-file-id", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--persist-selection", action="store_true")
    parser.add_argument("--force", action="store_true")
    parsed = parser.parse_args(args=args)

    project_root = Path(parsed.project_root)
    context = ProjectContext.load(project_root=str(project_root))
    manifest_path = Path(parsed.manifest) if parsed.manifest else default_manifest_path(project_root)
    manifest = FixtureManifest.load(manifest_path)
    service = FixtureAcquisitionService(context)

    if parsed.command == "status":
        rows = [row.to_dict() for row in service.status_rows(manifest)]
        print(json.dumps({"manifest": str(manifest_path), "fixtures": rows}, indent=2))
        return 0

    if not parsed.fixture_id:
        raise SystemExit("--fixture-id is required for this command")

    if parsed.command == "search":
        entry = manifest.get(parsed.fixture_id)
        page = await service.search(entry, limit=parsed.limit)
        output = {
            "fixture": entry.to_dict(),
            "results": [item.to_dict() for item in page.items],
            "next_cursor": page.next_cursor or None,
        }
        print(json.dumps(output, indent=2))
        return 0

    if parsed.command == "show":
        model = await service.get_model(parsed.provider, parsed.remote_model_id)
        print(json.dumps(model.to_dict(), indent=2))
        return 0

    if parsed.command == "acquire":
        record = await service.acquire(
            manifest,
            parsed.fixture_id,
            remote_model_id=parsed.remote_model_id,
            remote_version_id=parsed.remote_version_id,
            remote_file_id=parsed.remote_file_id,
            persist_selection=parsed.persist_selection,
            force=parsed.force,
        )
        print(json.dumps(record.to_dict(), indent=2))
        return 0

    if parsed.command == "inspect":
        cache = service.index.load()
        record = cache.get(parsed.fixture_id)
        if record is None:
            raise SystemExit(f"No cached fixture record exists for '{parsed.fixture_id}'.")
        print(json.dumps(record.to_dict(), indent=2))
        return 0

    raise SystemExit(f"Unsupported command: {parsed.command}")


def main(args: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(args=args))


if __name__ == "__main__":
    raise SystemExit(main())
