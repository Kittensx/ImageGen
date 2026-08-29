from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.asset_hub.install_planner import AssetHubInstallPlanner, InstallPlan, sha256_file
from image_gen.systems.asset_hub.metadata_index import InstalledAssetMetadataIndex
from image_gen.systems.asset_hub.metadata_sidecar import provenance_sidecar_path, read_provenance_sidecar, write_provenance_sidecar
from image_gen.systems.asset_hub.provenance import build_provenance, utc_now
from image_gen.systems.asset_hub.providers.base import AssetHubError
from image_gen.systems.asset_hub.repository import InstallRecord, InstallRepository
from image_gen.systems.upscaling.discovery import discover_upscalers
from image_gen.systems.sd21_support import SD21SupportManager
from modules.registry.asset_registry import AssetRegistry

INSTALLER_VERSION = "image-gen-asset-hub-installer-v1"


class AssetHubInstaller:
    def __init__(
        self,
        *,
        context: Any,
        service: Any,
        downloads: Any,
        catalog: Any,
        upscaler_catalog: Any,
        registry: AssetRegistry,
        repository: InstallRepository,
        discovery_index: Any | None = None,
    ) -> None:
        self.context = context
        self.service = service
        self.downloads = downloads
        self.catalog = catalog
        self.upscaler_catalog = upscaler_catalog
        self.registry = registry
        self.repository = repository
        self.discovery_index = discovery_index
        self.planner = AssetHubInstallPlanner(context=context, service=service, downloads=downloads)
        self.metadata_index = InstalledAssetMetadataIndex(context.cache_root)
        self._plans: dict[str, InstallPlan] = {}

    async def create_plan(self, download_job_id: str, *, conflict_policy: str = "hash_suffix", archive_member: str = "") -> InstallPlan:
        plan = await self.planner.create_plan(download_job_id, conflict_policy=conflict_policy, archive_member=archive_member)
        prior = next((
            item for item in self.repository.list(limit=1000)
            if item.status == "installed"
            and item.provider_id == plan.provider_id
            and item.remote_model_id == plan.remote_model_id
            and item.remote_version_id == plan.remote_version_id
            and item.remote_file_id == plan.remote_file_id
            and item.verified_sha256 != plan.verified_sha256
        ), None)
        if prior is not None:
            plan = replace(
                plan,
                requires_confirmation=True,
                warnings=plan.warnings + (
                    "This provider file identity is already installed with locally different content; review the conflict policy before installing.",
                ),
            )
        self._plans[plan.plan_id] = plan
        if len(self._plans) > 256:
            for key in list(self._plans)[:64]:
                self._plans.pop(key, None)
        return plan

    def _plan(self, plan_id: str) -> InstallPlan:
        plan = self._plans.get(str(plan_id or "").strip())
        if plan is None:
            raise AssetHubError("install_plan_not_found", "Install plan is missing or expired; create a fresh plan.", status_code=404)
        return plan

    def _managed_roots(self) -> dict[str, str]:
        return {
            "checkpoint": str(self.context.checkpoints_dir),
            "vae": str(self.context.vae_dir),
            "lora": str(self.context.lora_dir),
            "esrgan": str(self.context.esrgan_dir),
            "realesrgan": str(self.context.realesrgan_dir),
            "controlnet": str(self.context.controlnet_dir),
            "embedding": str(self.context.embeddings_dir),
        }

    @staticmethod
    def _atomic_copy(source: Path, destination: Path, expected_sha256: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.imagegen-{uuid.uuid4().hex}.tmp"
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if sha256_file(temporary) != expected_sha256:
                raise AssetHubError("install_copy_hash_mismatch", "Atomic install copy failed SHA-256 verification.", status_code=500)
            os.replace(temporary, destination)
        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

    def _quarantine(self, plan: InstallPlan, install_id: str, install_job_id: str) -> InstallRecord:
        source = Path(plan.source_path).resolve()
        root = (Path(self.context.data_root) / "asset-hub" / "quarantine" / plan.download_job_id).resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = root / plan.source_filename
        self._atomic_copy(source, target, plan.verified_sha256 if not plan.source_member else sha256_file(source))
        record_payload = {
            "verified_hash": plan.verified_sha256,
            "provider_identity": {
                "provider_id": plan.provider_id,
                "remote_model_id": plan.remote_model_id,
                "remote_version_id": plan.remote_version_id,
                "remote_file_id": plan.remote_file_id,
            },
            "reason_code": plan.quarantine_reason or "install_review_required",
            "safe_inspection_result": dict(plan.classification),
            "original_sanitized_filename": plan.source_filename,
            "quarantined_at_utc": utc_now(),
        }
        (root / "quarantine.json").write_text(json.dumps(record_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.repository.create(InstallRecord(
            install_id=install_id,
            install_job_id=install_job_id,
            download_job_id=plan.download_job_id,
            provider_id=plan.provider_id,
            remote_model_id=plan.remote_model_id,
            remote_version_id=plan.remote_version_id,
            remote_file_id=plan.remote_file_id,
            installed_path=str(target),
            verified_sha256=plan.verified_sha256,
            asset_kind=plan.proposed_asset_kind,
            status="quarantined",
            source_metadata_json=json.dumps(record_payload, sort_keys=True),
            installed_at=utc_now(),
        ))

    def _resolve_conflict(self, destination: Path, source_hash: str, plan: InstallPlan, install_id: str) -> tuple[Path, bool, Path | None]:
        if not destination.exists():
            return destination, False, None
        existing_hash = sha256_file(destination)
        if existing_hash == source_hash:
            return destination, True, None
        if plan.conflict_policy == "cancel":
            raise AssetHubError("install_conflict", "Destination already contains different content.", status_code=409)
        if plan.conflict_policy == "hash_suffix":
            candidate = destination.with_name(f"{destination.stem}-{source_hash[:8]}{destination.suffix}")
            if candidate.exists() and sha256_file(candidate) != source_hash:
                candidate = destination.with_name(f"{destination.stem}-{source_hash[:12]}{destination.suffix}")
            return candidate, candidate.exists() and sha256_file(candidate) == source_hash, None
        backup_root = (Path(self.context.data_root) / "asset-hub" / "backups" / install_id).resolve()
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / destination.name
        shutil.copy2(destination, backup)
        sidecar = provenance_sidecar_path(destination)
        if sidecar.exists():
            shutil.copy2(sidecar, backup_root / sidecar.name)
        return destination, False, backup

    def _refresh_after_install(self, kind: str, installed_path: Path) -> dict[str, Any]:
        if kind == "lora":
            self.catalog.refresh_asset_type("lora")
            return self.catalog.scan_loras(mode="missing")
        if kind in {"checkpoint", "vae", "textual_inversion"}:
            catalog_kind = "textual_inversion" if kind == "textual_inversion" else kind
            return self.catalog.refresh_asset_type(catalog_kind)
        if kind == "upscaler":
            discover_upscalers(self.context, mode="selected", selected_file=installed_path)
            return self.upscaler_catalog.refresh(mode="selected", selected_file=str(installed_path))
        return {}

    @staticmethod
    def _provider_discovery_payload(model: Any, version: Any) -> dict[str, Any]:
        payload = dict(model.to_dict()) if hasattr(model, "to_dict") else dict(model or {})
        version_payload = dict(version.to_dict()) if hasattr(version, "to_dict") else dict(version or {})
        version_id = str(version_payload.get("remoteVersionId") or "")
        versions = [dict(item) for item in (payload.get("versions") or []) if isinstance(item, Mapping)]
        if version_id:
            replaced = False
            for index, item in enumerate(versions):
                if str(item.get("remoteVersionId") or "") == version_id:
                    versions[index] = {**item, **version_payload}
                    replaced = True
                    break
            if not replaced:
                versions.append(version_payload)
        payload["versions"] = versions
        return payload

    def _sync_provider_discovery(self, model: Any, version: Any) -> dict[str, Any]:
        if self.discovery_index is None:
            return {"status": "unavailable", "indexed": 0}
        try:
            payload = self._provider_discovery_payload(model, version)
            provider_id = str(payload.get("providerId") or getattr(model, "provider_id", "") or "")
            indexed = int(self.discovery_index.ingest_model(provider_id, payload) or 0)
            return {"status": "synced", "indexed": indexed}
        except Exception as exc:
            # A downloaded asset is already safely installed at this point. A
            # discovery-index failure must remain repairable metadata drift, not
            # roll the user's file back out of the library.
            return {
                "status": "stale",
                "indexed": 0,
                "error": f"{type(exc).__name__}: {exc}"[:512],
            }

    async def install(self, plan_id: str, *, confirmed: bool = False) -> InstallRecord:
        plan = self._plan(plan_id)
        if plan.requires_confirmation and not confirmed:
            raise AssetHubError("install_confirmation_required", "Install plan requires explicit review/confirmation.", status_code=409)
        install_id = str(uuid.uuid4())
        install_job_id = str(uuid.uuid4())
        if plan.quarantine_reason or not plan.proposed_destination:
            return self._quarantine(plan, install_id, install_job_id)

        source = Path(plan.source_path).resolve()
        source_hash = sha256_file(source)
        expected = source_hash if plan.source_member else plan.verified_sha256
        if source_hash != expected:
            raise AssetHubError("install_staging_hash_changed", "Install source no longer matches its verified hash.", status_code=409)
        destination = Path(plan.proposed_destination).resolve()
        roots = [Path(value).resolve() for value in self._managed_roots().values()]
        if not any(destination == root or root in destination.parents for root in roots):
            raise AssetHubError("install_destination_unsafe", "Install destination is outside configured managed asset roots.", status_code=500)

        target, deduplicated, backup = self._resolve_conflict(destination, source_hash, plan, install_id)
        created_target = not target.exists()
        try:
            if not deduplicated:
                self._atomic_copy(source, target, source_hash)
            classification = dict(plan.classification)
            classification.setdefault("installer_version", INSTALLER_VERSION)
            provenance = build_provenance(
                model=plan.provider_model,
                version=plan.provider_version,
                provider_file=plan.provider_file,
                installed_path=target,
                verified_sha256=source_hash,
                asset_kind=plan.proposed_asset_kind,
                classification=classification,
                download_job_id=plan.download_job_id,
                install_id=install_id,
            )
            provenance["deduplicated_existing_file"] = bool(deduplicated)
            provenance["archive_member"] = plan.source_member or ""
            sidecar_payload = write_provenance_sidecar(target, provenance, hydrate_card_fields=True)
            library_root, managed_category, path_kind = self.registry.classify_path(str(target), self._managed_roots())
            registry_record = self.registry.register_file(
                str(target),
                compute_sha256=True,
                compute_blake3=False,
                library_root=library_root,
                managed_category=managed_category,
                path_kind=path_kind,
            )
            self._refresh_after_install(plan.proposed_asset_kind, target)
            discovery_sync = self._sync_provider_discovery(plan.provider_model, plan.provider_version)
            provenance["post_install_sync"] = {
                "catalog": {"status": "synced"},
                "discovery_index": discovery_sync,
                "registry_asset_id": str(registry_record.id),
                "metadata_index": {"status": "synced"},
            }
            sidecar_payload = write_provenance_sidecar(target, provenance, hydrate_card_fields=True)
            record = self.repository.create(InstallRecord(
                install_id=install_id,
                install_job_id=install_job_id,
                download_job_id=plan.download_job_id,
                provider_id=plan.provider_id,
                remote_model_id=plan.remote_model_id,
                remote_version_id=plan.remote_version_id,
                remote_file_id=plan.remote_file_id,
                installed_path=str(target),
                verified_sha256=source_hash,
                asset_kind=plan.proposed_asset_kind,
                status="installed",
                source_metadata_json=json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                sidecar_path=str(provenance_sidecar_path(target)),
                registry_asset_id=str(registry_record.id),
                installed_at=utc_now(),
            ))
            self.metadata_index.upsert(install_id, provenance)
            if plan.proposed_asset_kind == "checkpoint":
                architecture = str(classification.get("architecture") or "").strip()
                if architecture.casefold() == "sd2.x":
                    try:
                        SD21SupportManager(self.context).ensure_for_architecture(
                            architecture,
                            launch_if_missing=True,
                            reason="asset_hub_sd2_checkpoint_install",
                        )
                    except Exception:
                        # Installing the user's checkpoint must remain successful even if the
                        # standalone support-installer launch itself cannot be started. Model
                        # activation performs the same readiness check and surfaces the reason.
                        pass
            return record
        except Exception:
            if backup is not None and backup.exists():
                try:
                    os.replace(backup, target)
                except OSError:
                    pass
            elif created_target:
                try:
                    if target.exists():
                        target.unlink()
                    sidecar = provenance_sidecar_path(target)
                    if sidecar.exists():
                        sidecar.unlink()
                except OSError:
                    pass
            raise

    async def auto_install_download(self, download_job_id: str) -> InstallRecord:
        existing = self.get_by_download_job(download_job_id)
        if existing is not None and existing.status in {"installed", "quarantined"}:
            self.downloads.cleanup_job_staging(download_job_id)
            return existing
        plan = await self.create_plan(download_job_id, conflict_policy="hash_suffix")
        record = await self.install(plan.plan_id, confirmed=True)
        if record.status in {"installed", "quarantined"}:
            self.downloads.cleanup_job_staging(download_job_id)
        return record

    def list_installed(self) -> list[InstallRecord]:
        return self.repository.list(limit=1000)

    def get_by_download_job(self, download_job_id: str) -> InstallRecord | None:
        return self.repository.get_by_download_job(download_job_id)

    def open_install_folder(self, install_id: str) -> str:
        record = self.repository.get(install_id)
        if record is None:
            raise AssetHubError("install_not_found", "Installed Asset Hub record not found.", status_code=404)
        path = Path(record.installed_path).resolve()
        target = path if path.is_dir() else path.parent
        allowed_roots = [Path(value).resolve() for value in self._managed_roots().values()]
        if not any(root == target or root in target.parents for root in allowed_roots):
            raise AssetHubError("install_destination_unsafe", "Installed asset folder is outside configured managed roots.", status_code=409)
        if not target.exists():
            raise AssetHubError("installed_file_missing", "Installed asset folder no longer exists.", status_code=404)
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(target)], start_new_session=True)
        except OSError as exc:
            raise AssetHubError("open_folder_failed", f"Unable to open installed asset folder: {exc}", status_code=500) from exc
        return str(target)

    def get_install_job(self, install_job_id: str) -> InstallRecord:
        record = self.repository.get_by_job(install_job_id)
        if record is None:
            raise AssetHubError("install_job_not_found", "Install job not found.", status_code=404)
        return record

    async def refresh_metadata(self, install_id: str) -> InstallRecord:
        record = self.repository.get(install_id)
        if record is None:
            raise AssetHubError("install_not_found", "Installed Asset Hub record not found.", status_code=404)
        if record.status != "installed":
            raise AssetHubError("install_not_active", "Only installed assets can refresh provider metadata.", status_code=409)
        path = Path(record.installed_path).resolve()
        if not path.is_file():
            return self.repository.update(install_id, status="missing", error_code="installed_file_missing", error_message="Installed file is no longer present.")
        try:
            model = await self.service.get_model(
                record.provider_id,
                record.remote_model_id,
                refresh=True,
                include_unsupported=True,
            )
            version = await self.service.get_version(
                record.provider_id,
                record.remote_version_id,
                refresh=True,
                include_unsupported=True,
            )
            provider_file = next((item for item in version.files if item.remote_file_id == record.remote_file_id), None)
            if provider_file is None:
                raise AssetHubError("provider_not_found", "Provider file is no longer available.", status_code=404)
            existing = read_provenance_sidecar(path)
            provenance = build_provenance(
                model=model,
                version=version,
                provider_file=provider_file,
                installed_path=path,
                verified_sha256=record.verified_sha256,
                asset_kind=record.asset_kind,
                classification=existing.get("classification_result") or existing.get("imagegen_classification") or {},
                download_job_id=record.download_job_id,
                install_id=record.install_id,
                source_metadata=existing,
            )
            provenance["metadata_created_at_utc"] = existing.get("metadata_created_at_utc") or provenance["metadata_created_at_utc"]
            write_provenance_sidecar(path, provenance, hydrate_card_fields=True)
            self._refresh_after_install(record.asset_kind, path)
            discovery_sync = self._sync_provider_discovery(model, version)
            provenance["post_install_sync"] = {
                "catalog": {"status": "synced"},
                "discovery_index": discovery_sync,
                "registry_asset_id": record.registry_asset_id,
                "metadata_index": {"status": "synced"},
            }
            write_provenance_sidecar(path, provenance, hydrate_card_fields=True)
            self.metadata_index.upsert(record.install_id, provenance)
            return self.repository.update(
                install_id,
                source_metadata_json=json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                error_code="",
                error_message="",
            )
        except AssetHubError as exc:
            existing = read_provenance_sidecar(path)
            if existing:
                existing["provider_sync_status"] = "stale"
                existing["provider_sync_error"] = exc.message[:512]
                write_provenance_sidecar(path, existing)
                self.metadata_index.upsert(record.install_id, existing)
            raise

    def uninstall(self, install_id: str) -> InstallRecord:
        record = self.repository.get(install_id)
        if record is None:
            raise AssetHubError("install_not_found", "Installed Asset Hub record not found.", status_code=404)
        if record.status != "installed":
            return record
        path = Path(record.installed_path).resolve()
        allowed_roots = [Path(value).resolve() for value in self._managed_roots().values()]
        if not any(root == path or root in path.parents for root in allowed_roots):
            raise AssetHubError("uninstall_path_unsafe", "Installed path is outside configured managed asset roots.", status_code=409)

        # Phase 03 never permanently deletes a user asset. Until the Phase 06 OS
        # Recycle Bin lifecycle lands, uninstall moves only the Asset Hub-owned file
        # and its sidecar into a bounded recovery location outside model scan roots.
        recovery = (Path(self.context.data_root) / "asset-hub" / "uninstalled" / install_id).resolve()
        recovery.mkdir(parents=True, exist_ok=True)
        if path.exists():
            os.replace(path, recovery / path.name)
        sidecar = provenance_sidecar_path(path)
        if sidecar.exists():
            os.replace(sidecar, recovery / sidecar.name)
        self.metadata_index.remove(install_id)
        try:
            self._refresh_after_install(record.asset_kind, path)
        except Exception:
            pass
        return self.repository.update(install_id, status="removed", installed_path=str(recovery / path.name))
