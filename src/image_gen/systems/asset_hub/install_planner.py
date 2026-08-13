from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from image_gen.runtime.lora_inspector import inspect_lora_file
from image_gen.runtime.adapters.compatibility import AdapterCompatibilityService
from image_gen.runtime.adapters.contracts import AdapterInspectionRecord
from image_gen.systems.asset_hub.archive_safety import ArchiveInspection, extract_member, inspect_zip
from image_gen.systems.asset_hub.contracts import ProviderFile, ProviderModel, ProviderVersion
from image_gen.systems.asset_hub.providers.base import AssetHubError
from image_gen.systems.upscaling.classifier import INSPECTOR_VERSION, inspect_upscaler_file, loader_backend_version
from modules.checkpoint_inspector import CheckpointInspector

INSTALL_PLANNER_VERSION = "image-gen-asset-hub-install-planner-v1"
_DIRECT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InstallPlan:
    plan_id: str
    download_job_id: str
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    source_path: str
    source_member: str
    source_filename: str
    verified_sha256: str
    proposed_asset_kind: str
    proposed_destination: str
    classification_strategy: str
    classification: Mapping[str, Any] = field(default_factory=dict)
    conflict_policy: str = "hash_suffix"
    security_decision: str = "allow"
    warnings: tuple[str, ...] = ()
    requires_confirmation: bool = False
    quarantine_reason: str = ""
    archive: ArchiveInspection | None = None
    provider_file: ProviderFile | None = field(default=None, repr=False, compare=False)
    provider_model: ProviderModel | None = field(default=None, repr=False, compare=False)
    provider_version: ProviderVersion | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planId": self.plan_id,
            "downloadJobId": self.download_job_id,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "sourceFilename": self.source_filename,
            "sourceMember": self.source_member or None,
            "verifiedSha256": self.verified_sha256,
            "proposedAssetKind": self.proposed_asset_kind,
            "proposedDestination": self.proposed_destination or None,
            "classificationStrategy": self.classification_strategy,
            "classification": dict(self.classification),
            "conflictPolicy": self.conflict_policy,
            "securityDecision": self.security_decision,
            "warnings": list(self.warnings),
            "requiresConfirmation": self.requires_confirmation,
            "quarantineReason": self.quarantine_reason or None,
            "archive": self.archive.to_dict() if self.archive else None,
            "plannerVersion": INSTALL_PLANNER_VERSION,
        }


class AssetHubInstallPlanner:
    def __init__(self, *, context: Any, service: Any, downloads: Any) -> None:
        self.context = context
        self.service = service
        self.downloads = downloads
        self._checkpoint_inspector = CheckpointInspector()
        self._adapter_compatibility = AdapterCompatibilityService()

    def _destination(self, kind: str) -> Path | None:
        mapping = {
            "checkpoint": Path(self.context.checkpoints_dir),
            "lora": Path(self.context.lora_dir),
            "vae": Path(self.context.vae_dir),
            "textual_inversion": Path(self.context.embeddings_dir),
            "controlnet": Path(self.context.controlnet_dir),
            "esrgan": Path(self.context.esrgan_dir),
            "realesrgan": Path(self.context.realesrgan_dir),
        }
        return mapping.get(kind)

    @staticmethod
    def _find_provider_file(version: ProviderVersion, remote_file_id: str) -> ProviderFile:
        selected = next((item for item in version.files if str(item.remote_file_id) == str(remote_file_id)), None)
        if selected is None:
            raise AssetHubError("install_provider_file_missing", "Provider file metadata is no longer available.", status_code=404)
        return selected

    @staticmethod
    def _safe_json(path: Path) -> dict[str, Any]:
        try:
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("JSON workflow is too large")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AssetHubError("install_json_invalid", "JSON workflow candidate could not be validated safely.", status_code=422) from exc
        if not isinstance(payload, (dict, list)):
            raise AssetHubError("install_json_invalid", "JSON workflow must contain an object or array.", status_code=422)
        return {"json_type": type(payload).__name__, "size_bytes": path.stat().st_size}

    @staticmethod
    def _inspect_vae(path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".safetensors":
            raise AssetHubError("install_vae_unsafe_format", "Automatic VAE installation requires safetensors.", status_code=422)
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise AssetHubError("install_vae_inspection_failed", "VAE safetensors header could not be inspected.", status_code=422) from exc
        signals = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.", "first_stage_model.")
        if not keys or not any(str(key).startswith(signals) for key in keys):
            raise AssetHubError("install_vae_unrecognized", "Safetensors file does not expose a recognized VAE key layout.", status_code=422)
        return {"tensor_key_count": len(keys), "format": "safetensors", "scan_version": "asset-hub-vae-header-v1"}

    @staticmethod
    def _inspect_embedding(path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".safetensors":
            raise AssetHubError("install_embedding_unsafe_format", "Automatic textual-inversion installation requires safetensors.", status_code=422)
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise AssetHubError("install_embedding_inspection_failed", "Embedding safetensors header could not be inspected.", status_code=422) from exc
        if not keys:
            raise AssetHubError("install_embedding_unrecognized", "Embedding safetensors contains no tensors.", status_code=422)
        return {"tensor_key_count": len(keys), "format": "safetensors", "scan_version": "asset-hub-embedding-header-v1"}

    def classify(self, path: Path, provider_file: ProviderFile, provider_model: ProviderModel) -> tuple[str, Path | None, str, dict[str, Any], str]:
        suffix = path.suffix.lower()
        hint = str(provider_model.asset_kind or provider_file.file_type or "").strip().casefold()
        if suffix not in _DIRECT_SUFFIXES:
            return "other", None, "unsupported_extension", {"suffix": suffix}, "unsupported_file_type"

        if suffix == ".json":
            details = self._safe_json(path)
            return "workflow", None, "json_workflow_inspector", details, "workflow_destination_requires_review"

        if suffix == ".pth":
            result = inspect_upscaler_file(path)
            payload = {
                "architecture": result.architecture,
                "native_scale": result.native_scale,
                "input_channels": result.input_channels,
                "output_channels": result.output_channels,
                "load_status": result.load_status,
                "loader_family": result.loader_backend,
                "compatibility": list(result.compatibility_notes),
                "classifier_version": INSPECTOR_VERSION,
                "loader_backend_version": loader_backend_version(),
                "error": result.bounded_error,
            }
            if result.load_status != "supported":
                return "upscaler", None, "upscaler_safe_classifier", payload, "upscaler_not_qualified"
            if result.architecture.startswith("esrgan_"):
                return "upscaler", self._destination("esrgan"), "upscaler_safe_classifier", payload, ""
            if result.architecture.startswith("realesrgan_"):
                return "upscaler", self._destination("realesrgan"), "upscaler_safe_classifier", payload, ""
            return "upscaler", None, "upscaler_safe_classifier", payload, "upscaler_family_unroutable"

        if hint == "lora":
            details = inspect_lora_file(path)
            inspection = AdapterInspectionRecord.from_mapping(details.get("adapter_inspection"))
            support = self._adapter_compatibility.evaluate(inspection, active_checkpoint_family="")
            details = {
                **details,
                "runtime_support_state": support.overall_support_state,
                "runtime_loadable": support.runtime_loadable,
                "support_reason": support.blocking_reason,
                "loader_id": support.loader_id,
            }
            if inspection.adapter_format == "non_adapter_full_model":
                return "lora", None, "lora_inspector", details, "lora_misclassified_full_model"
            if inspection.adapter_format == "invalid" or details.get("inspection_error") or int(details.get("tensor_key_count") or 0) <= 0:
                return "lora", None, "lora_inspector", details, "lora_inspection_failed"
            return "lora", self._destination("lora"), "lora_inspector", details, ""

        if hint == "checkpoint":
            if suffix != ".safetensors":
                return "checkpoint", None, "checkpoint_inspector", {"format": suffix}, "checkpoint_unsafe_format"
            try:
                report = self._checkpoint_inspector.inspect(str(path))
                details = report.to_dict()
            except Exception as exc:
                return "checkpoint", None, "checkpoint_inspector", {"error": f"{type(exc).__name__}: {exc}"}, "checkpoint_inspection_failed"
            if report.checkpoint_kind not in {"full", "partial"} or not report.has_unet:
                return "checkpoint", None, "checkpoint_inspector", details, "checkpoint_unrecognized"
            return "checkpoint", self._destination("checkpoint"), "checkpoint_inspector", details, ""

        if hint == "vae":
            try:
                details = self._inspect_vae(path)
            except AssetHubError as exc:
                return "vae", None, "vae_header_inspector", {"error": exc.message}, exc.code
            return "vae", self._destination("vae"), "vae_header_inspector", details, ""

        if hint == "textual_inversion":
            try:
                details = self._inspect_embedding(path)
            except AssetHubError as exc:
                return "textual_inversion", None, "embedding_header_inspector", {"error": exc.message}, exc.code
            return "textual_inversion", self._destination("textual_inversion"), "embedding_header_inspector", details, ""

        return hint or "other", None, "provider_hint_insufficient", {"provider_hint": hint, "suffix": suffix}, "unclassified_asset"

    @staticmethod
    def _inspection_path(source: Path, staging: Path, filename: str) -> Path:
        suffix = Path(filename).suffix.lower()
        if source.suffix.lower() == suffix and suffix:
            return source
        inspection_dir = staging / "inspection"
        inspection_dir.mkdir(parents=True, exist_ok=True)
        target = inspection_dir / Path(filename).name
        if target.exists():
            try:
                if target.stat().st_size == source.stat().st_size:
                    return target
                target.unlink()
            except OSError:
                pass
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        return target

    async def create_plan(
        self,
        download_job_id: str,
        *,
        conflict_policy: str = "hash_suffix",
        archive_member: str = "",
    ) -> InstallPlan:
        record = self.downloads.get_job(download_job_id)
        if record.status != "completed" or not record.actual_sha256:
            raise AssetHubError("install_download_not_verified", "Install planning requires a completed verified download.", status_code=409)
        staging = Path(record.staging_directory).resolve()
        source = staging / "payload.part"
        if not source.is_file():
            raise AssetHubError("install_staging_missing", "Verified staged payload is missing.", status_code=409)
        actual = sha256_file(source)
        if actual != str(record.actual_sha256).lower():
            raise AssetHubError("install_staging_hash_changed", "Staged payload changed after download verification.", status_code=409)

        model = await self.service.get_model(record.provider_id, record.remote_model_id, refresh=False)
        version = await self.service.get_version(record.provider_id, record.remote_version_id, refresh=False)
        provider_file = self._find_provider_file(version, record.remote_file_id)
        selected_source = source
        selected_member = ""
        archive = None
        filename = record.file_name
        warnings: list[str] = []
        requires_confirmation = False

        if Path(filename).suffix.lower() == ".zip":
            archive = inspect_zip(source)
            if archive_member:
                selected_source = extract_member(source, archive_member, staging / "extracted")
                selected_member = archive_member
                filename = selected_source.name
            elif len(archive.install_candidates) == 1:
                selected_member = archive.install_candidates[0]
                selected_source = extract_member(source, selected_member, staging / "extracted")
                filename = selected_source.name
            else:
                requires_confirmation = True
                warnings.append("Archive requires an explicit install member selection before live installation.")

        if not selected_member and Path(filename).suffix.lower() != ".zip":
            selected_source = self._inspection_path(source, staging, filename)

        if requires_confirmation:
            kind, destination, strategy, classification, quarantine = "archive", None, "archive_review", {}, ""
        else:
            kind, destination, strategy, classification, quarantine = self.classify(selected_source, provider_file, model)
        if quarantine:
            warnings.append(f"Automatic install is blocked: {quarantine}.")

        normalized_conflict = str(conflict_policy or "hash_suffix").strip().casefold()
        if normalized_conflict not in {"hash_suffix", "cancel", "replace"}:
            raise AssetHubError("install_conflict_policy_invalid", "Unsupported install conflict policy.", status_code=400)
        proposed = (destination / filename).resolve() if destination else None
        return InstallPlan(
            plan_id=str(uuid.uuid4()),
            download_job_id=record.job_id,
            provider_id=record.provider_id,
            remote_model_id=record.remote_model_id,
            remote_version_id=record.remote_version_id,
            remote_file_id=record.remote_file_id,
            source_path=str(selected_source),
            source_member=selected_member,
            source_filename=filename,
            verified_sha256=record.actual_sha256,
            proposed_asset_kind=kind,
            proposed_destination=str(proposed) if proposed else "",
            classification_strategy=strategy,
            classification=classification,
            conflict_policy=normalized_conflict,
            security_decision="quarantine" if quarantine else ("review" if requires_confirmation else "allow"),
            warnings=tuple(warnings),
            requires_confirmation=requires_confirmation,
            quarantine_reason=quarantine,
            archive=archive,
            provider_file=provider_file,
            provider_model=model,
            provider_version=version,
        )
