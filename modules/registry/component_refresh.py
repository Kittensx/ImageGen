from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import threading
import time

from modules.checkpoint_inspector import CheckpointInspector, CheckpointReport
from modules.project_context import ProjectContext

from .asset_registry import AssetRegistry
from .component_snapshot import COMPONENT_SNAPSHOT_VERSION, SafetensorsComponentSnapshotter
from .family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from .standalone_component_evidence import (
    StandaloneComponentEvidence,
    classify_standalone_text_encoder,
    classify_standalone_vae,
)
from .models import (
    AssetRecord,
    ComponentSnapshotRecord,
    SCAN_STRENGTH_FULL,
    SCAN_STRENGTH_QUICK,
    SCAN_STRENGTH_STRUCTURAL,
)


REGISTRY_REFRESH_POLICY_VERSION = "component-registry-refresh-v4-cnrr08"
_SUPPORTED_COMPONENT_KINDS = {"checkpoint", "vae", "text_encoder"}


@dataclass
class _DiscoveryFlight:
    event: threading.Event
    error: BaseException | None = None


_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_INFLIGHT: dict[tuple[str, str, str, str, str], _DiscoveryFlight] = {}


def _run_discovery_once(
    key: tuple[str, str, str, str, str],
    operation,
):
    """Serialize expensive component discovery for one exact source version.

    The registry remains the authority; this is only an in-process duplicate-work
    guard.  Waiters never consume another thread's in-memory snapshots.  They wait
    for the leader's transaction to finish and then read the committed registry
    rows normally.
    """
    with _DISCOVERY_LOCK:
        flight = _DISCOVERY_INFLIGHT.get(key)
        if flight is None:
            flight = _DiscoveryFlight(event=threading.Event())
            _DISCOVERY_INFLIGHT[key] = flight
            leader = True
        else:
            leader = False

    if not leader:
        started = time.perf_counter()
        flight.event.wait()
        if flight.error is not None:
            raise RuntimeError("Concurrent component discovery failed before commit.") from flight.error
        return False, None, round((time.perf_counter() - started) * 1000.0, 3)

    try:
        result = operation()
        return True, result, 0.0
    except BaseException as exc:
        flight.error = exc
        raise
    finally:
        with _DISCOVERY_LOCK:
            _DISCOVERY_INFLIGHT.pop(key, None)
            flight.event.set()


@dataclass(frozen=True)
class ComponentRefreshAssessment:
    asset_id: int
    path: str
    asset_kind: str
    architecture: str
    checkpoint_kind: str
    quick_fingerprint: str
    expected_roles: tuple[str, ...]
    stored_roles: tuple[str, ...]
    stored_snapshot_versions: tuple[str, ...]
    refresh_required: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": int(self.asset_id),
            "path": self.path,
            "asset_kind": self.asset_kind,
            "architecture": self.architecture,
            "checkpoint_kind": self.checkpoint_kind,
            "quick_fingerprint": self.quick_fingerprint,
            "expected_roles": list(self.expected_roles),
            "stored_roles": list(self.stored_roles),
            "stored_snapshot_versions": list(self.stored_snapshot_versions),
            "refresh_required": bool(self.refresh_required),
            "reasons": list(self.reasons),
            "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
            "snapshot_version": COMPONENT_SNAPSHOT_VERSION,
        }


class ComponentRegistryRefresher:
    """Keep component-content snapshots complete for individual loaded assets.

    The refresher is intentionally targeted. It never walks the whole model library
    unless a caller explicitly asks it to assess registered assets. A checkpoint is
    considered cache-complete only when:

    * component snapshots exist;
    * every stored snapshot is bound to the asset's current quick fingerprint;
    * all snapshots use the current component snapshot version; and
    * the stored learned-component role set exactly matches the roles discoverable
      from the current Safetensors header under the detected architecture.

    This closes the partial-cache hole where an earlier SD3 encoder-only snapshot
    could be mistaken for a fully decomposed checkpoint merely because the source
    file itself had not changed.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        registry: AssetRegistry | None = None,
        inspector: CheckpointInspector | None = None,
        snapshotter: SafetensorsComponentSnapshotter | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.inspector = inspector or CheckpointInspector()
        self.snapshotter = snapshotter or SafetensorsComponentSnapshotter()

    @staticmethod
    def _snapshot_metadata(record: ComponentSnapshotRecord) -> dict[str, Any]:
        try:
            value = json.loads(record.metadata_json or "{}")
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def _snapshot_source_quick_fingerprint(cls, record: ComponentSnapshotRecord) -> str:
        return str(cls._snapshot_metadata(record).get("source_quick_fingerprint") or "").strip().lower()

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def classify_path(
        self,
        path: str | Path,
        *,
        asset: AssetRecord | None = None,
        explicit_kind: str | None = None,
    ) -> str:
        if explicit_kind:
            value = str(explicit_kind).strip().lower()
            if value in _SUPPORTED_COMPONENT_KINDS:
                return value
            raise ValueError(f"Unsupported targeted component-refresh asset kind: {explicit_kind!r}")

        if asset is not None and str(asset.asset_type or "").strip().lower() in _SUPPORTED_COMPONENT_KINDS:
            return str(asset.asset_type).strip().lower()

        resolved = Path(path).expanduser().resolve()
        roots = (
            (Path(self.context.checkpoints_dir).resolve(), "checkpoint"),
            (Path(self.context.vae_dir).resolve(), "vae"),
            ((Path(self.context.models_root) / "TextEncoders").resolve(), "text_encoder"),
        )
        for root, kind in roots:
            if self._is_under(resolved, root):
                return kind
        return "unknown"

    def _standalone_text_encoder_role(self, path: Path) -> str:
        return classify_standalone_text_encoder(path).component_role

    @staticmethod
    def _standalone_component_evidence(path: Path, asset_kind: str) -> StandaloneComponentEvidence | None:
        if asset_kind == "text_encoder":
            return classify_standalone_text_encoder(path)
        if asset_kind == "vae":
            return classify_standalone_vae(path)
        return None

    def _expected_roles(
        self,
        path: Path,
        *,
        asset_kind: str,
        report: CheckpointReport | None,
    ) -> tuple[str, ...]:
        if asset_kind == "checkpoint":
            if report is None:
                raise ValueError("Checkpoint refresh assessment requires an inspection report.")
            roles = self.snapshotter.discover_checkpoint_roles(
                path,
                architecture=report.architecture,
                include_extras=False,
            )
            return tuple(sorted(roles))
        if asset_kind == "vae":
            return ("vae",)
        if asset_kind == "text_encoder":
            return (self._standalone_text_encoder_role(path),)
        return ()

    def checkpoint_snapshot_eligibility(
        self,
        path: str | Path,
        *,
        report: CheckpointReport,
    ) -> dict[str, Any]:
        """Return whether a checkpoint container exposes provider-supported components.

        Component snapshot eligibility is intentionally based on the components that
        are structurally present, not on whether the outer checkpoint is a complete
        generation package. This allows supported partial checkpoints such as the SDXL
        refiner to materialize exact UNet/VAE/text-encoder component snapshots while
        continuing to reject unknown or unrelated partial Safetensors.
        """
        expected_roles = self._expected_roles(
            Path(path).expanduser().resolve(),
            asset_kind="checkpoint",
            report=report,
        )

        if str(report.checkpoint_kind or "").strip().lower() == "full":
            return {
                "eligible": True,
                "reason": "full_checkpoint",
                "family_id": DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(report.architecture),
                "expected_roles": list(expected_roles),
                "provider_roles": [],
                "supported_roles": list(expected_roles),
            }

        if str(report.checkpoint_kind or "").strip().lower() != "partial":
            return {
                "eligible": False,
                "reason": f"checkpoint_kind={report.checkpoint_kind or 'unknown'}",
                "family_id": "",
                "expected_roles": list(expected_roles),
                "provider_roles": [],
                "supported_roles": [],
            }

        architecture_token = str(report.architecture or "").strip().lower()
        if not architecture_token or architecture_token == "unknown" or "_or_" in architecture_token:
            return {
                "eligible": False,
                "reason": "partial_checkpoint_provider_unresolved",
                "family_id": "",
                "expected_roles": list(expected_roles),
                "provider_roles": [],
                "supported_roles": [],
            }

        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(report.architecture)
        if provider is None:
            return {
                "eligible": False,
                "reason": "partial_checkpoint_provider_unresolved",
                "family_id": "",
                "expected_roles": list(expected_roles),
                "provider_roles": [],
                "supported_roles": [],
            }

        provider_roles = tuple(sorted(item.canonical_role_id for item in provider.role_definitions()))
        provider_role_set = set(provider_roles)
        supported_roles = tuple(sorted(role for role in expected_roles if role in provider_role_set))
        if not supported_roles:
            return {
                "eligible": False,
                "reason": "partial_checkpoint_has_no_provider_supported_components",
                "family_id": provider.family_id,
                "expected_roles": list(expected_roles),
                "provider_roles": list(provider_roles),
                "supported_roles": [],
            }

        return {
            "eligible": True,
            "reason": "supported_partial_checkpoint_components",
            "family_id": provider.family_id,
            "expected_roles": list(expected_roles),
            "provider_roles": list(provider_roles),
            "supported_roles": list(supported_roles),
        }

    def assess(
        self,
        asset: AssetRecord,
        *,
        asset_kind: str,
        report: CheckpointReport | None = None,
        force: bool = False,
    ) -> ComponentRefreshAssessment:
        path = Path(asset.path).resolve()
        if path.suffix.lower() != ".safetensors":
            return ComponentRefreshAssessment(
                asset_id=asset.id,
                path=str(path),
                asset_kind=asset_kind,
                architecture=str(getattr(report, "architecture", None) or asset.architecture or ""),
                checkpoint_kind=str(getattr(report, "checkpoint_kind", None) or asset.checkpoint_kind or "unknown"),
                quick_fingerprint=str(asset.quick_fingerprint or ""),
                expected_roles=(),
                stored_roles=(),
                stored_snapshot_versions=(),
                refresh_required=False,
                reasons=("non_safetensors_not_supported",),
            )

        current = self.registry.get_component_snapshots(asset.id)
        expected_roles = self._expected_roles(path, asset_kind=asset_kind, report=report)
        stored_roles = tuple(sorted({item.component_role for item in current if item.component_role != "extras"}))
        versions = tuple(sorted({str(item.snapshot_version or "") for item in current}))
        quick = str(asset.quick_fingerprint or "").strip().lower()
        reasons: list[str] = []

        if force:
            reasons.append("forced")
        if not current:
            reasons.append("missing_component_snapshots")
        if not quick:
            reasons.append("asset_quick_fingerprint_missing")
        elif current and any(self._snapshot_source_quick_fingerprint(item) != quick for item in current):
            reasons.append("source_quick_fingerprint_changed_or_unbound")
        if current and any(item.snapshot_version != COMPONENT_SNAPSHOT_VERSION for item in current):
            reasons.append("component_snapshot_version_stale")

        expected_set = set(expected_roles)
        stored_set = set(stored_roles)
        missing = sorted(expected_set - stored_set)
        unexpected = sorted(stored_set - expected_set)
        if missing:
            reasons.append("missing_expected_roles:" + ",".join(missing))
        if unexpected:
            reasons.append("unexpected_stored_roles:" + ",".join(unexpected))

        return ComponentRefreshAssessment(
            asset_id=asset.id,
            path=str(path),
            asset_kind=asset_kind,
            architecture=str(getattr(report, "architecture", None) or asset.architecture or ""),
            checkpoint_kind=str(getattr(report, "checkpoint_kind", None) or asset.checkpoint_kind or "unknown"),
            quick_fingerprint=quick,
            expected_roles=expected_roles,
            stored_roles=stored_roles,
            stored_snapshot_versions=versions,
            refresh_required=bool(reasons and reasons != ["non_safetensors_not_supported"]),
            reasons=tuple(reasons),
        )

    def ensure_path(
        self,
        path: str | Path,
        *,
        explicit_kind: str | None = None,
        force: bool = False,
        dry_run: bool = False,
        source: str = "manual_targeted_refresh",
        precomputed_report: CheckpointReport | None = None,
        library_root: str | None = None,
        managed_category: str | None = None,
        path_kind: str | None = None,
        strength: str = SCAN_STRENGTH_STRUCTURAL,
    ) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Cannot refresh missing asset: {resolved}")

        requested_strength = str(strength or SCAN_STRENGTH_STRUCTURAL).strip().lower()
        if requested_strength not in {SCAN_STRENGTH_QUICK, SCAN_STRENGTH_STRUCTURAL, SCAN_STRENGTH_FULL}:
            raise ValueError(f"Unsupported component refresh strength: {strength!r}")

        existing = self.registry.get_asset_by_path(str(resolved))
        kind = self.classify_path(resolved, asset=existing, explicit_kind=explicit_kind)
        if kind not in _SUPPORTED_COMPONENT_KINDS:
            return {
                "path": str(resolved),
                "asset_kind": kind,
                "status": "skipped_unsupported_asset_kind",
                "changed": False,
                "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                "strength": requested_strength,
            }

        if requested_strength == SCAN_STRENGTH_FULL:
            quick_only = self.registry.fingerprinter.fingerprint_file(str(resolved), compute_sha256=False)
            existing_quick = str(existing.quick_fingerprint or "").strip().lower() if existing is not None else ""
            current_quick = str(quick_only.quick_fingerprint or "").strip().lower()
            needs_strong = force or existing is None or not existing.sha256 or existing_quick != current_quick
            fingerprint = (
                self.registry.fingerprinter.fingerprint_file(str(resolved), compute_sha256=True)
                if needs_strong
                else quick_only
            )
            asset = self.registry.upsert_asset_from_fingerprint(
                fingerprint,
                library_root=(library_root if library_root is not None else (existing.library_root if existing else None)),
                managed_category=(
                    managed_category if managed_category is not None else (existing.managed_category if existing else None)
                ),
                path_kind=(path_kind if path_kind is not None else (existing.path_kind if existing else "external")),
            )
        else:
            asset = self.registry.register_file(
                str(resolved),
                compute_sha256=False,
                library_root=(library_root if library_root is not None else (existing.library_root if existing else None)),
                managed_category=(
                    managed_category if managed_category is not None else (existing.managed_category if existing else None)
                ),
                path_kind=(path_kind if path_kind is not None else (existing.path_kind if existing else "external")),
            )

        report = precomputed_report
        if kind == "checkpoint":
            report = report or self.inspector.inspect(str(resolved), compute_sha256=False)
            eligibility = self.checkpoint_snapshot_eligibility(resolved, report=report)
            if not eligibility["eligible"]:
                return {
                    "path": str(resolved),
                    "asset_id": int(asset.id),
                    "asset_kind": kind,
                    "status": "skipped_ineligible_checkpoint_container",
                    "changed": False,
                    "architecture": report.architecture,
                    "checkpoint_kind": report.checkpoint_kind,
                    "checkpoint_snapshot_eligibility": eligibility,
                    "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                    "strength": requested_strength,
                }

        before = self.registry.get_component_snapshots(asset.id)
        assessment = self.assess(asset, asset_kind=kind, report=report, force=force)

        def _metrics(**updates: Any) -> dict[str, Any]:
            payload = {
                "registry_lookup_hit": bool(before),
                "component_hash_required": False,
                "bytes_hashed": 0,
                "roles_hashed": [],
                "hash_reused_from_registry": bool(before),
                "source_occurrence_upserted_count": 0,
                "duplicate_discovery_avoided": False,
                "duplicate_discovery_wait_ms": 0.0,
                "extra_disk_pass_required": False,
                "hash_scope": "lookup_only" if before else "none",
            }
            payload.update(updates)
            return payload

        if dry_run or not assessment.refresh_required:
            payload = {
                "path": str(resolved),
                "asset_id": int(asset.id),
                "asset_kind": kind,
                "status": "would_refresh" if dry_run and assessment.refresh_required else "cached_complete",
                "changed": False,
                "assessment": assessment.to_dict(),
                "before": self._snapshot_summary(before),
                "after": self._snapshot_summary(before),
                "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                "strength": requested_strength,
                "discovery_metrics": _metrics(
                    component_hash_required=bool(dry_run and assessment.refresh_required),
                    hash_reused_from_registry=bool(before and not assessment.refresh_required),
                ),
            }
            if kind == "checkpoint" and report is not None:
                payload["checkpoint_snapshot_eligibility"] = self.checkpoint_snapshot_eligibility(
                    resolved,
                    report=report,
                )
            if requested_strength == SCAN_STRENGTH_FULL:
                payload["whole_file_sha256"] = asset.sha256
                payload["whole_file_identity_state"] = "reused_existing" if asset.sha256 else "missing"
            return payload

        if requested_strength == SCAN_STRENGTH_QUICK and before:
            stored_roles = tuple(sorted(item.component_role for item in before if item.component_role != "extras"))
            self._store_refresh_inspection(
                asset_id=asset.id,
                asset_kind=kind,
                path=resolved,
                report=report,
                expected_roles=assessment.expected_roles,
                stored_roles=stored_roles,
                source=source,
            )
            refreshed_asset = self.registry.get_asset_by_id(asset.id) or asset
            return {
                "path": str(resolved),
                "asset_id": int(asset.id),
                "asset_kind": kind,
                "status": self._refresh_status(assessment),
                "changed": False,
                "assessment": assessment.to_dict(),
                "after_assessment": assessment.to_dict(),
                "before": self._snapshot_summary(before),
                "after": self._snapshot_summary(before),
                "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                "strength": requested_strength,
                "whole_file_sha256": refreshed_asset.sha256,
                "discovery_metrics": _metrics(hash_scope="quick_refresh_only"),
            }

        reason_set = set(assessment.reasons)
        missing_roles = tuple(sorted(set(assessment.expected_roles) - set(assessment.stored_roles)))
        targeted_missing_only = bool(
            kind == "checkpoint"
            and before
            and missing_roles
            and reason_set
            and all(reason.startswith("missing_expected_roles:") for reason in reason_set)
        )
        roles_to_hash = missing_roles if targeted_missing_only else assessment.expected_roles
        discovery_key = (
            str(Path(self.registry.db_path).resolve()),
            str(resolved),
            str(asset.quick_fingerprint or "").strip().lower(),
            COMPONENT_SNAPSHOT_VERSION,
            kind,
        )

        def _perform_discovery() -> dict[str, Any]:
            if kind == "checkpoint":
                assert report is not None
                snapshots = self.snapshotter.snapshot_checkpoint(
                    resolved,
                    architecture=report.architecture,
                    include_extras=not targeted_missing_only,
                    include_roles=set(roles_to_hash) if targeted_missing_only else None,
                )
            elif kind == "vae":
                snapshots = {
                    "vae": self.snapshotter.snapshot_standalone_component(
                        resolved,
                        component_role="vae",
                    )
                }
            else:
                role = assessment.expected_roles[0]
                snapshots = {
                    role: self.snapshotter.snapshot_standalone_component(
                        resolved,
                        component_role=role,
                    )
                }

            standalone_evidence = self._standalone_component_evidence(resolved, kind)
            metadata_extra = {
                "registry_refresh_policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                "registry_refresh_source": str(source),
                "requested_strength": requested_strength,
                "expected_component_roles": list(assessment.expected_roles),
                "cnrr08_discovery_scope": "missing_roles_only" if targeted_missing_only else "full_required_roles",
            }
            if standalone_evidence is not None:
                metadata_extra["standalone_component_evidence"] = standalone_evidence.to_metadata()
                metadata_extra["provider_family_evidence"] = list(standalone_evidence.provider_family_evidence)
                standalone_snapshot = next(iter(snapshots.values()))
                metadata_extra["standalone_component_identity"] = {
                    "component_sha256": standalone_snapshot.component_sha256,
                    "structure_sha256": standalone_snapshot.structure_sha256,
                    "identity_basis": "fingerprinted_tensor_content_and_structure",
                }

            store_snapshots = (
                self.registry.merge_component_snapshots
                if targeted_missing_only
                else self.registry.store_component_snapshots
            )
            stored = store_snapshots(
                asset.id,
                snapshots,
                source_file_sha256=asset.sha256,
                source_quick_fingerprint=asset.quick_fingerprint,
                metadata_extra=metadata_extra,
            )
            self._store_refresh_inspection(
                asset_id=asset.id,
                asset_kind=kind,
                path=resolved,
                report=report,
                expected_roles=assessment.expected_roles,
                stored_roles=tuple(sorted(item.component_role for item in stored if item.component_role != "extras")),
                source=source,
            )
            refreshed_asset = self.registry.get_asset_by_id(asset.id) or asset
            after_assessment = self.assess(
                refreshed_asset,
                asset_kind=kind,
                report=report,
                force=False,
            )
            if after_assessment.refresh_required:
                raise RuntimeError(
                    "Targeted component refresh completed but the registry is still incomplete: "
                    + "; ".join(after_assessment.reasons)
                )
            bytes_hashed = sum(int(snapshot.total_bytes) for snapshot in snapshots.values())
            hashed_roles = sorted(str(role) for role in snapshots if role != "extras")
            return {
                "path": str(resolved),
                "asset_id": int(asset.id),
                "asset_kind": kind,
                "status": self._refresh_status(assessment),
                "changed": True,
                "assessment": assessment.to_dict(),
                "after_assessment": after_assessment.to_dict(),
                "before": self._snapshot_summary(before),
                "after": self._snapshot_summary(stored),
                "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                "strength": requested_strength,
                "whole_file_sha256": refreshed_asset.sha256,
                "discovery_metrics": _metrics(
                    component_hash_required=True,
                    bytes_hashed=bytes_hashed,
                    roles_hashed=hashed_roles,
                    hash_reused_from_registry=False,
                    source_occurrence_upserted_count=len(hashed_roles),
                    extra_disk_pass_required=True,
                    hash_scope="missing_roles_only" if targeted_missing_only else "full_required_roles",
                ),
                **(
                    {
                        "checkpoint_snapshot_eligibility": self.checkpoint_snapshot_eligibility(
                            resolved,
                            report=report,
                        )
                    }
                    if kind == "checkpoint" and report is not None
                    else {}
                ),
            }

        leader, result, wait_ms = _run_discovery_once(discovery_key, _perform_discovery)
        if leader:
            assert result is not None
            return result

        # Another request performed the expensive discovery. Read the authoritative
        # committed rows instead of sharing in-memory snapshots between loaders.
        refreshed_asset = self.registry.get_asset_by_path(str(resolved)) or asset
        current = self.registry.get_component_snapshots(refreshed_asset.id)
        after_assessment = self.assess(
            refreshed_asset,
            asset_kind=kind,
            report=report,
            force=False,
        )
        if after_assessment.refresh_required:
            raise RuntimeError(
                "Concurrent component discovery completed without satisfying the requested registry evidence: "
                + "; ".join(after_assessment.reasons)
            )
        return {
            "path": str(resolved),
            "asset_id": int(refreshed_asset.id),
            "asset_kind": kind,
            "status": "cached_after_concurrent_discovery",
            "changed": False,
            "assessment": assessment.to_dict(),
            "after_assessment": after_assessment.to_dict(),
            "before": self._snapshot_summary(before),
            "after": self._snapshot_summary(current),
            "policy_version": REGISTRY_REFRESH_POLICY_VERSION,
            "strength": requested_strength,
            "whole_file_sha256": refreshed_asset.sha256,
            "discovery_metrics": _metrics(
                registry_lookup_hit=True,
                component_hash_required=False,
                hash_reused_from_registry=True,
                duplicate_discovery_avoided=True,
                duplicate_discovery_wait_ms=wait_ms,
                hash_scope="concurrent_lookup_after_wait",
            ),
            **(
                {
                    "checkpoint_snapshot_eligibility": self.checkpoint_snapshot_eligibility(
                        resolved,
                        report=report,
                    )
                }
                if kind == "checkpoint" and report is not None
                else {}
            ),
        }

    def ensure_checkpoint(
        self,
        *,
        asset: AssetRecord,
        checkpoint_path: str | Path,
        report: CheckpointReport,
        force: bool = False,
        source: str = "model_load",
    ) -> tuple[tuple[ComponentSnapshotRecord, ...], dict[str, Any]]:
        result = self.ensure_path(
            checkpoint_path,
            explicit_kind="checkpoint",
            force=force,
            dry_run=False,
            source=source,
            precomputed_report=report,
            library_root=asset.library_root,
            managed_category=asset.managed_category,
            path_kind=asset.path_kind,
        )
        current = tuple(self.registry.get_component_snapshots(asset.id))
        return current, result

    def refresh_registered(
        self,
        *,
        paths: Iterable[str | Path] = (),
        force: bool = False,
        dry_run: bool = False,
        source: str = "manual_targeted_refresh",
        progress_callback=None,
        strength: str = SCAN_STRENGTH_STRUCTURAL,
    ) -> list[dict[str, Any]]:
        explicit_paths = [Path(item).expanduser().resolve() for item in paths]
        if explicit_paths:
            targets = explicit_paths
        else:
            targets = []
            for asset in self.registry.list_assets(limit=1_000_000):
                path = Path(asset.path)
                if not asset.exists_on_disk or not path.is_file() or path.suffix.lower() != ".safetensors":
                    continue
                if self.classify_path(path, asset=asset) in _SUPPORTED_COMPONENT_KINDS:
                    targets.append(path.resolve())

        results: list[dict[str, Any]] = []
        total = len(targets)
        for index, target in enumerate(targets, start=1):
            if progress_callback is not None:
                progress_callback(index, total, target)
            results.append(
                self.ensure_path(
                    target,
                    force=force,
                    dry_run=dry_run,
                    source=source,
                    strength=strength,
                )
            )
        return results

    @staticmethod
    def _refresh_status(assessment: ComponentRefreshAssessment) -> str:
        reasons = set(assessment.reasons)
        if "forced" in reasons:
            return "refreshed_forced"
        if "missing_component_snapshots" in reasons:
            return "hashed_new"
        if any(reason.startswith("missing_expected_roles:") or reason.startswith("unexpected_stored_roles:") for reason in reasons):
            return "refreshed_incomplete_cache"
        if "source_quick_fingerprint_changed_or_unbound" in reasons:
            return "refreshed_changed_file"
        if "component_snapshot_version_stale" in reasons:
            return "refreshed_snapshot_version"
        return "refreshed"

    @staticmethod
    def _snapshot_summary(records: Iterable[ComponentSnapshotRecord]) -> list[dict[str, Any]]:
        return [
            {
                "id": int(item.id),
                "component_role": item.component_role,
                "snapshot_version": item.snapshot_version,
                "component_sha256": item.component_sha256,
                "structure_sha256": item.structure_sha256,
                "tensor_count": int(item.tensor_count),
                "total_bytes": int(item.total_bytes),
            }
            for item in records
        ]

    def _store_refresh_inspection(
        self,
        *,
        asset_id: int,
        asset_kind: str,
        path: Path,
        report: CheckpointReport | None,
        expected_roles: Iterable[str],
        stored_roles: Iterable[str],
        source: str,
    ) -> None:
        if report is not None:
            payload = {
                "asset_type": "checkpoint",
                "format_type": "safetensors",
                "architecture": report.architecture,
                "checkpoint_kind": report.checkpoint_kind,
                "has_unet": report.has_unet,
                "has_vae": report.has_vae,
                "has_text_encoder": report.has_text_encoder,
                "has_text_encoder_2": report.has_sdxl_text_encoder_2 or report.has_clip_g,
                "key_count": report.total_keys,
                "prefix_summary": {"prefixes": report.key_prefixes},
                "example_keys": report.example_keys,
                "dtype_summary": dict(report.dtype_summary),
                "tensor_shape_summary": dict(report.tensor_shape_summary),
                "metadata": {
                    "registry_refresh": True,
                    "registry_refresh_source": str(source),
                    "registry_refresh_policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                    "expected_component_roles": sorted(set(expected_roles)),
                    "stored_component_roles": sorted(set(stored_roles)),
                    "file_name": path.name,
                    "architecture_variant": report.architecture_variant,
                },
                "inspector_version": REGISTRY_REFRESH_POLICY_VERSION,
            }
        else:
            payload = {
                "asset_type": asset_kind,
                "format_type": "safetensors",
                "architecture": "",
                "architecture_state": "observed_unclassified",
                "checkpoint_kind": "standalone_component",
                "has_unet": False,
                "has_vae": asset_kind == "vae",
                "has_text_encoder": asset_kind == "text_encoder",
                "has_text_encoder_2": False,
                "key_count": None,
                "prefix_summary": {},
                "example_keys": [],
                "dtype_summary": {},
                "tensor_shape_summary": {},
                "metadata": {
                    "registry_refresh": True,
                    "registry_refresh_source": str(source),
                    "registry_refresh_policy_version": REGISTRY_REFRESH_POLICY_VERSION,
                    "expected_component_roles": sorted(set(expected_roles)),
                    "stored_component_roles": sorted(set(stored_roles)),
                    "file_name": path.name,
                },
                "inspector_version": REGISTRY_REFRESH_POLICY_VERSION,
            }
        self.registry.store_inspection(asset_id, payload)
