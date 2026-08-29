from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from image_gen.runtime.lora_inspector import canonical_model_family, inspect_lora_file
from image_gen.systems.asset_hub.contracts import ProviderFile, ProviderModel, ProviderSearchRequest, ProviderVersion
from image_gen.systems.asset_hub.diagnostics import write_json_atomic
from image_gen.systems.asset_hub.downloads import AssetHubDownloadManager
from image_gen.systems.asset_hub.policy import normalize_architecture
from image_gen.systems.asset_hub.providers.base import AssetHubError
from image_gen.systems.asset_hub.service import AssetHubService


FIXTURE_MANIFEST_SCHEMA_VERSION = 1
FIXTURE_CACHE_SCHEMA_VERSION = 1
_TERMINAL_DOWNLOAD_STATES = {"completed", "failed", "cancelled", "paused"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_id(value: Any) -> str:
    return str(value or "").strip()


def _clean_sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _algorithm_from_inspection(adapter_format: str) -> str:
    token = str(adapter_format or "").strip().lower()
    if token.startswith("standard_"):
        return "standard_lora"
    return {
        "lycoris_locon": "locon",
        "lycoris_loha": "loha",
        "lycoris_lokr": "lokr",
        "lycoris_other": "lycoris_other",
        "unknown_adapter": "unknown_adapter",
        "non_adapter_full_model": "non_adapter_full_model",
        "inspection_restricted": "inspection_restricted",
        "invalid": "invalid",
    }.get(token, token or "unknown_adapter")


def _normalized_architecture_hint(value: Any) -> str:
    normalized = normalize_architecture(value)
    if normalized:
        return normalized
    token = str(value or "").strip().lower().replace("_", ".")
    if token in {"sd1", "sd1.x"}:
        return "sd1.x"
    if token in {"sd2", "sd2.x"}:
        return "sd2.x"
    if token in {"sd3", "sd3.x", "sd3.5", "sd3.5.x", "sd3.5m", "sd3.5-medium"}:
        return "sd3.x"
    if token == "sdxl":
        return "sdxl"
    return ""


def _inspection_architecture(value: Any) -> str:
    family = canonical_model_family(str(value or ""))
    return {"sd1": "sd1.x", "sd2": "sd2.x", "sd3": "sd3.x", "sdxl": "sdxl"}.get(family, family)


@dataclass(frozen=True)
class FixtureManifestEntry:
    fixture_id: str
    provider: str
    source_asset_id: str = ""
    source_version_id: str = ""
    alternate_source_version_ids: tuple[str, ...] = ()
    source_file_id: str = ""
    browser_selection_purpose: str = ""
    expected_sha256: str = ""
    expected_architecture_hint: str = ""
    expected_algorithm_hint: str = ""
    expected_targets_hint: tuple[str, ...] = ()
    search_terms: str = ""
    training_type_hint: str = ""
    version_hint: str = ""
    license_review: str = "required"
    notes: str = ""
    enabled: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FixtureManifestEntry":
        fixture_id = _clean_id(payload.get("id") or payload.get("fixture_id"))
        provider = _clean_id(payload.get("provider")).casefold()
        if not fixture_id:
            raise ValueError("Fixture manifest entry is missing id.")
        if not provider:
            raise ValueError(f"Fixture {fixture_id!r} is missing provider.")
        targets = payload.get("expected_targets_hint") or ()
        if isinstance(targets, str):
            targets = [targets]
        alternate_versions = payload.get("alternate_source_version_ids") or ()
        if isinstance(alternate_versions, str):
            alternate_versions = [alternate_versions]
        return cls(
            fixture_id=fixture_id,
            provider=provider,
            source_asset_id=_clean_id(payload.get("source_asset_id")),
            source_version_id=_clean_id(payload.get("source_version_id")),
            alternate_source_version_ids=tuple(_clean_id(item) for item in alternate_versions if _clean_id(item)),
            source_file_id=_clean_id(payload.get("source_file_id")),
            browser_selection_purpose=_clean_id(payload.get("browser_selection_purpose")).casefold(),
            expected_sha256=_clean_sha256(payload.get("expected_sha256")),
            expected_architecture_hint=_clean_id(payload.get("expected_architecture_hint")),
            expected_algorithm_hint=_clean_id(payload.get("expected_algorithm_hint")).casefold(),
            expected_targets_hint=tuple(str(item).strip().lower() for item in targets if str(item).strip()),
            search_terms=_clean_id(payload.get("search_terms")),
            training_type_hint=_clean_id(payload.get("training_type_hint")),
            version_hint=_clean_id(payload.get("version_hint")),
            license_review=_clean_id(payload.get("license_review") or "required"),
            notes=_clean_id(payload.get("notes")),
            enabled=bool(payload.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fixture_id,
            "provider": self.provider,
            "source_asset_id": self.source_asset_id,
            "source_version_id": self.source_version_id,
            "alternate_source_version_ids": list(self.alternate_source_version_ids),
            "source_file_id": self.source_file_id,
            "browser_selection_purpose": self.browser_selection_purpose,
            "expected_sha256": self.expected_sha256,
            "expected_architecture_hint": self.expected_architecture_hint,
            "expected_algorithm_hint": self.expected_algorithm_hint,
            "expected_targets_hint": list(self.expected_targets_hint),
            "search_terms": self.search_terms,
            "training_type_hint": self.training_type_hint,
            "version_hint": self.version_hint,
            "license_review": self.license_review,
            "notes": self.notes,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class FixtureManifest:
    entries: tuple[FixtureManifestEntry, ...] = ()
    schema_version: int = FIXTURE_MANIFEST_SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "FixtureManifest":
        manifest_path = Path(path).expanduser().resolve()
        try:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8-sig")) or {}
        except OSError as exc:
            raise ValueError(f"Unable to read fixture manifest: {manifest_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Fixture manifest is invalid YAML: {manifest_path}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("Fixture manifest must be a YAML mapping.")
        try:
            version = int(raw.get("schema_version") or FIXTURE_MANIFEST_SCHEMA_VERSION)
        except (TypeError, ValueError) as exc:
            raise ValueError("Fixture manifest schema_version must be an integer.") from exc
        if version != FIXTURE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported fixture manifest schema_version: {version}")
        items = raw.get("fixtures") or []
        if not isinstance(items, list):
            raise ValueError("Fixture manifest 'fixtures' must be a list.")
        entries = tuple(FixtureManifestEntry.from_mapping(item) for item in items if isinstance(item, Mapping))
        seen: set[str] = set()
        for entry in entries:
            key = entry.fixture_id.casefold()
            if key in seen:
                raise ValueError(f"Duplicate fixture manifest id: {entry.fixture_id}")
            seen.add(key)
        return cls(entries=entries, schema_version=version)

    def get(self, fixture_id: str) -> FixtureManifestEntry:
        selected = str(fixture_id or "").strip().casefold()
        for entry in self.entries:
            if entry.fixture_id.casefold() == selected:
                return entry
        raise KeyError(fixture_id)


@dataclass(frozen=True)
class ResolvedFixtureIdentity:
    entry: FixtureManifestEntry
    model: ProviderModel
    version: ProviderVersion
    file: ProviderFile

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.entry.to_dict(),
            "source_provider": self.file.provider_id,
            "source_asset_id": self.file.remote_model_id,
            "source_version_id": self.file.remote_version_id,
            "source_file_id": self.file.remote_file_id,
            "source_url": self.file.source_page_url,
            "provider_base_model_hint": self.version.base_model,
            "provider_architecture_hint": self.version.architecture,
            "file_name": self.file.file_name,
            "file_size_bytes": self.file.size_bytes,
            "provider_hashes": self.file.hash_map(),
            "model": self.model.to_dict(),
            "version": self.version.to_dict(),
        }


class QualificationFixtureService:
    """Provider-neutral acquisition service for qualification fixtures.

    The service delegates discovery to ``AssetHubService`` and all byte transfer,
    redirect policy, secret handling, resume, and hash verification to
    ``AssetHubDownloadManager``. It only promotes a *verified staged payload* into
    a qualification cache after a second local SHA-256 check and bounded adapter
    inspection.
    """

    def __init__(
        self,
        *,
        asset_hub: AssetHubService,
        downloads: AssetHubDownloadManager,
        cache_root: str | os.PathLike[str],
    ) -> None:
        self.asset_hub = asset_hub
        self.downloads = downloads
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.index_path = self.cache_root / "fixture_cache_report.json"

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema_version": FIXTURE_CACHE_SCHEMA_VERSION, "fixtures": []}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": FIXTURE_CACHE_SCHEMA_VERSION, "fixtures": []}
        if not isinstance(payload, Mapping):
            return {"schema_version": FIXTURE_CACHE_SCHEMA_VERSION, "fixtures": []}
        fixtures = payload.get("fixtures") if isinstance(payload.get("fixtures"), list) else []
        return {"schema_version": FIXTURE_CACHE_SCHEMA_VERSION, "fixtures": [dict(item) for item in fixtures if isinstance(item, Mapping)]}

    def _save_index_record(self, record: Mapping[str, Any]) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        index = self._load_index()
        identity = (
            str(record.get("source_provider") or "").casefold(),
            str(record.get("source_asset_id") or ""),
            str(record.get("source_version_id") or ""),
            str(record.get("source_file_id") or ""),
        )
        new_hash = _clean_sha256(record.get("sha256"))
        for item in index["fixtures"]:
            item_identity = (
                str(item.get("source_provider") or "").casefold(),
                str(item.get("source_asset_id") or ""),
                str(item.get("source_version_id") or ""),
                str(item.get("source_file_id") or ""),
            )
            if item_identity != identity:
                continue
            old_hash = _clean_sha256(item.get("sha256"))
            if old_hash and new_hash and old_hash != new_hash:
                raise AssetHubError(
                    "fixture_identity_changed",
                    "Provider served different bytes for an already-pinned fixture identity; existing qualification evidence was preserved.",
                    status_code=409,
                )

        existing: list[dict[str, Any]] = []
        replaced = False
        fixture_id = str(record.get("fixture_id") or "").casefold()
        for item in index["fixtures"]:
            if str(item.get("fixture_id") or "").casefold() == fixture_id:
                existing.append(dict(record))
                replaced = True
            else:
                existing.append(item)
        if not replaced:
            existing.append(dict(record))
        write_json_atomic(self.index_path, {
            "schema_version": FIXTURE_CACHE_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "fixtures": existing,
        })

    def list_missing(self, manifest: FixtureManifest) -> list[dict[str, Any]]:
        index = self._load_index()
        by_id = {str(item.get("fixture_id") or "").casefold(): item for item in index["fixtures"]}
        output: list[dict[str, Any]] = []
        for entry in manifest.entries:
            if not entry.enabled:
                continue
            cached = by_id.get(entry.fixture_id.casefold())
            cache_path = Path(str((cached or {}).get("local_cache_path") or "")) if cached else None
            present = bool(cached and cache_path and cache_path.is_file())
            output.append({
                "fixture_id": entry.fixture_id,
                "provider": entry.provider,
                "missing": not present,
                "cached": cached or None,
                "expectation": entry.to_dict(),
            })
        return output

    async def search_candidates(
        self,
        entry: FixtureManifestEntry,
        *,
        query: str = "",
        limit: int = 24,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        architecture = _normalized_architecture_hint(entry.expected_architecture_hint)
        request = ProviderSearchRequest(
            query=(query or entry.search_terms),
            asset_kind="lora",
            base_models=(architecture,) if architecture else (),
            limit=max(1, min(int(limit), 50)),
            refresh=refresh,
        )
        page = await self.asset_hub.search(entry.provider, request)
        candidates: list[dict[str, Any]] = []
        for model in page.items:
            for version in model.versions:
                for file in version.files:
                    candidates.append({
                        "fixture_id": entry.fixture_id,
                        "source_provider": file.provider_id,
                        "source_asset_id": file.remote_model_id,
                        "source_version_id": file.remote_version_id,
                        "source_file_id": file.remote_file_id,
                        "source_url": file.source_page_url,
                        "provider_base_model_hint": version.base_model,
                        "provider_architecture_hint": version.architecture,
                        "provider_type_hint": model.provider_type,
                        "training_type_hint_requested": entry.training_type_hint,
                        "version_hint_requested": entry.version_hint,
                        "file_name": file.file_name,
                        "file_size_bytes": file.size_bytes,
                        "provider_hashes": file.hash_map(),
                        "primary": file.primary,
                        "trained_words": list(file.trained_words),
                    })
        return candidates

    async def resolve_identity(self, entry: FixtureManifestEntry, *, refresh: bool = False) -> ResolvedFixtureIdentity:
        version: ProviderVersion | None = None
        pinned_version_ids = tuple(
            dict.fromkeys(
                item
                for item in (entry.source_version_id, *entry.alternate_source_version_ids)
                if str(item or "").strip()
            )
        )
        if pinned_version_ids:
            unavailable: list[str] = []
            for version_id in pinned_version_ids:
                try:
                    version = await self.asset_hub.get_version(entry.provider, version_id, refresh=refresh)
                    break
                except AssetHubError as exc:
                    if exc.code != "provider_not_found":
                        raise
                    unavailable.append(version_id)
            if version is None:
                raise AssetHubError(
                    "fixture_provider_identity_unavailable",
                    f"None of the pinned provider versions are currently available for fixture {entry.fixture_id!r}: "
                    + ", ".join(unavailable),
                    status_code=404,
                )
        elif entry.expected_sha256:
            version = await self.asset_hub.lookup_hash(entry.provider, entry.expected_sha256, refresh=refresh)
        else:
            raise AssetHubError(
                "fixture_identity_incomplete",
                "Fixture acquisition requires source_version_id, alternate_source_version_ids, or expected_sha256 to resolve a pinned provider version.",
                status_code=400,
            )

        model_id = entry.source_asset_id or version.remote_model_id
        if not model_id:
            raise AssetHubError("fixture_identity_incomplete", "Provider version did not resolve a model identity.", status_code=502)
        if entry.source_asset_id and version.remote_model_id and entry.source_asset_id != version.remote_model_id:
            raise AssetHubError("fixture_identity_changed", "Provider version model identity does not match the fixture manifest.", status_code=409)
        model = await self.asset_hub.get_model(entry.provider, model_id, refresh=refresh)

        selected: ProviderFile | None = None
        if entry.source_file_id:
            selected = next((item for item in version.files if item.remote_file_id == entry.source_file_id), None)
            if selected is None:
                raise AssetHubError("fixture_file_not_found", "Pinned provider file is no longer present in the selected version.", status_code=404)
        if selected is None and entry.expected_sha256:
            selected = next((item for item in version.files if _clean_sha256(item.hash_map().get("SHA256")) == entry.expected_sha256), None)
        if selected is None:
            primary = [item for item in version.files if item.primary]
            if len(primary) == 1:
                selected = primary[0]
        if selected is None and len(version.files) == 1:
            selected = version.files[0]
        if selected is None:
            raise AssetHubError(
                "fixture_file_ambiguous",
                "Fixture manifest did not identify one provider file. Pin source_file_id or expected_sha256 before acquisition.",
                status_code=409,
            )
        if entry.expected_sha256:
            provider_sha = _clean_sha256(selected.hash_map().get("SHA256"))
            if provider_sha and provider_sha != entry.expected_sha256:
                raise AssetHubError("fixture_identity_changed", "Provider file hash disagrees with the fixture manifest pin.", status_code=409)
        return ResolvedFixtureIdentity(entry=entry, model=model, version=version, file=selected)

    async def _wait_download(self, job_id: str, *, timeout_seconds: float = 3600.0) -> Any:
        deadline = asyncio.get_running_loop().time() + max(5.0, float(timeout_seconds))
        while True:
            record = self.downloads.get_job(job_id)
            if record.status in _TERMINAL_DOWNLOAD_STATES:
                return record
            if asyncio.get_running_loop().time() >= deadline:
                await self.downloads.cancel(job_id)
                raise AssetHubError("fixture_download_timeout", "Timed out waiting for the fixture download to complete.", status_code=504)
            await asyncio.sleep(0.10)

    def _cache_destination(self, identity: ResolvedFixtureIdentity) -> Path:
        safe_name = Path(identity.file.file_name).name or f"fixture-{identity.file.remote_file_id}.bin"
        return (
            self.cache_root
            / identity.file.provider_id
            / identity.file.remote_model_id
            / identity.file.remote_version_id
            / identity.file.remote_file_id
            / safe_name
        )

    @staticmethod
    def _classification_disagreements(entry: FixtureManifestEntry, inspection: Mapping[str, Any], identity: ResolvedFixtureIdentity) -> list[dict[str, str]]:
        disagreements: list[dict[str, str]] = []
        local_arch = _inspection_architecture(inspection.get("detected_model_family"))
        expected_arch = _normalized_architecture_hint(entry.expected_architecture_hint)
        provider_arch = _normalized_architecture_hint(identity.version.architecture or identity.version.base_model)
        if expected_arch and local_arch and expected_arch != local_arch:
            disagreements.append({
                "kind": "manifest_vs_local_architecture",
                "expected": expected_arch,
                "observed": local_arch,
            })
        if provider_arch and local_arch and provider_arch != local_arch:
            disagreements.append({
                "kind": "provider_vs_local_architecture",
                "expected": provider_arch,
                "observed": local_arch,
            })
        expected_algorithm = entry.expected_algorithm_hint.casefold()
        local_algorithm = _algorithm_from_inspection(str(inspection.get("adapter_format") or ""))
        if expected_algorithm and local_algorithm and expected_algorithm != local_algorithm:
            disagreements.append({
                "kind": "manifest_vs_local_algorithm",
                "expected": expected_algorithm,
                "observed": local_algorithm,
            })
        expected_targets = set(entry.expected_targets_hint)
        local_targets = {str(item).strip().lower() for item in (inspection.get("target_scopes") or []) if str(item).strip()}
        missing_targets = sorted(expected_targets - local_targets)
        if missing_targets:
            disagreements.append({
                "kind": "manifest_vs_local_targets",
                "expected": ",".join(sorted(expected_targets)),
                "observed": ",".join(sorted(local_targets)),
            })
        return disagreements

    async def acquire(
        self,
        entry: FixtureManifestEntry,
        *,
        refresh: bool = False,
        timeout_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        identity = await self.resolve_identity(entry, refresh=refresh)
        plan = await self.downloads.create_plan(
            provider_id=identity.file.provider_id,
            remote_model_id=identity.file.remote_model_id,
            remote_version_id=identity.file.remote_version_id,
            remote_file_id=identity.file.remote_file_id,
        )
        job = await self.downloads.enqueue(plan.plan_id)
        completed = await self._wait_download(job.job_id, timeout_seconds=timeout_seconds)
        if completed.status != "completed":
            raise AssetHubError(
                completed.error_code or "fixture_download_failed",
                completed.error_message or f"Fixture download ended with status {completed.status}.",
                status_code=502,
            )
        staged = Path(completed.staging_directory) / "payload.part"
        if not staged.is_file():
            raise AssetHubError("fixture_staging_missing", "Verified fixture payload is missing from Asset Hub staging.", status_code=500)
        staged_hash = _sha256_file(staged)
        if completed.actual_sha256 and staged_hash != completed.actual_sha256:
            raise AssetHubError("fixture_staging_hash_mismatch", "Staged fixture changed after Asset Hub verification.", status_code=409)
        if entry.expected_sha256 and staged_hash != entry.expected_sha256:
            raise AssetHubError("fixture_manifest_hash_mismatch", "Downloaded fixture SHA-256 does not match the manifest pin.", status_code=409)

        destination = self._cache_destination(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            existing_hash = _sha256_file(destination)
            if existing_hash != staged_hash:
                raise AssetHubError(
                    "fixture_identity_changed",
                    "A cached fixture already exists for this provider identity with different bytes; it was not replaced.",
                    status_code=409,
                )
        else:
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copyfile(staged, temporary)
            copied_hash = _sha256_file(temporary)
            if copied_hash != staged_hash:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise AssetHubError("fixture_cache_copy_mismatch", "Fixture cache copy failed SHA-256 verification.", status_code=500)
            temporary.replace(destination)

        inspection = inspect_lora_file(destination, include_compatibility_hash=False)
        provider_snapshot = identity.to_dict()
        provider_snapshot_digest = _hash_mapping(provider_snapshot)
        disagreements = self._classification_disagreements(entry, inspection, identity)
        record = {
            "schema_version": FIXTURE_CACHE_SCHEMA_VERSION,
            "fixture_id": entry.fixture_id,
            "source_provider": identity.file.provider_id,
            "source_asset_id": identity.file.remote_model_id,
            "source_version_id": identity.file.remote_version_id,
            "source_file_id": identity.file.remote_file_id,
            "source_url": identity.file.source_page_url,
            "provider_base_model_hint": identity.version.base_model,
            "provider_architecture_hint": identity.version.architecture,
            "provider_metadata_snapshot_sha256": provider_snapshot_digest,
            "provider_metadata_snapshot": provider_snapshot,
            "license_review": entry.license_review,
            "license_source_note": {
                "source_page_url": identity.model.source_page_url,
                "permissions": identity.model.permissions.to_dict(),
            },
            "file_name": identity.file.file_name,
            "size_bytes": destination.stat().st_size,
            "sha256": staged_hash,
            "retrieved_at": completed.completed_at or _utc_now(),
            "download_job_id": completed.job_id,
            "local_cache_path": str(destination),
            "inspection": inspection,
            "classification_disagreements": disagreements,
            "qualification_state": "candidate_unqualified",
        }
        self._save_index_record(record)
        write_json_atomic(destination.with_suffix(destination.suffix + ".fixture.json"), record)
        # The Asset Hub keeps payload.part as resumable staging during transfer. Once
        # the qualification cache copy and evidence record are durably committed, the
        # duplicate payload is no longer needed. Keep transaction/report evidence.
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            record["staging_cleanup_warning"] = f"{type(exc).__name__}: {exc}"
            self._save_index_record(record)
            write_json_atomic(destination.with_suffix(destination.suffix + ".fixture.json"), record)
        return record

    def verify_cached(self, entry: FixtureManifestEntry) -> dict[str, Any]:
        index = self._load_index()
        record = next((item for item in index["fixtures"] if str(item.get("fixture_id") or "").casefold() == entry.fixture_id.casefold()), None)
        if record is None:
            raise AssetHubError("fixture_not_cached", "Fixture is not present in the local qualification cache.", status_code=404)
        path = Path(str(record.get("local_cache_path") or ""))
        if not path.is_file():
            raise AssetHubError("fixture_cache_missing", "Fixture cache record points to a missing local file.", status_code=404)
        actual = _sha256_file(path)
        recorded = _clean_sha256(record.get("sha256"))
        if not recorded or actual != recorded:
            raise AssetHubError("fixture_cache_hash_mismatch", "Cached fixture SHA-256 no longer matches the acquisition record.", status_code=409)
        if entry.expected_sha256 and actual != entry.expected_sha256:
            raise AssetHubError("fixture_manifest_hash_mismatch", "Cached fixture no longer matches the source-controlled manifest pin.", status_code=409)
        inspection = inspect_lora_file(path, include_compatibility_hash=False)
        return {
            "fixture_id": entry.fixture_id,
            "verified": True,
            "sha256": actual,
            "local_cache_path": str(path),
            "inspection": inspection,
            "record": record,
        }


__all__ = [
    "FIXTURE_CACHE_SCHEMA_VERSION",
    "FIXTURE_MANIFEST_SCHEMA_VERSION",
    "FixtureManifest",
    "FixtureManifestEntry",
    "QualificationFixtureService",
    "ResolvedFixtureIdentity",
]
