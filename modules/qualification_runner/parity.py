from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from modules.project_context import ProjectContext
from modules.registry.asset_registry import AssetRegistry
from modules.registry.component_selection import canonical_model_family
from modules.registry.evidence_contracts import (
    SPLIT_ELIGIBILITY_BLOCKED,
    SPLIT_ELIGIBILITY_ELIGIBLE,
    SPLIT_ELIGIBILITY_INCONCLUSIVE,
    SPLIT_ELIGIBILITY_UNTESTED,
    SPLIT_GATE_RECOMMENDED,
    VALIDATION_ADVISORY,
    VALIDATION_RESULT_FAIL,
    VALIDATION_RESULT_PASS,
    VALIDATION_STAGE_PARITY,
)
from modules.registry.family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY

from .service import ComponentQualificationRunner


PARITY_QUEUE_SCHEMA_VERSION = 1
PARITY_EVIDENCE_VERSION = "model-component-digital-parity-v1"
PARITY_REVIEW_MATCH = "parity_match"
PARITY_REVIEW_MISMATCH = "parity_mismatch"
PARITY_REVIEW_INCONCLUSIVE = "inconclusive"
PARITY_REVIEW_PENDING = "pending"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: Any) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    return "-".join(part for part in text.split("-") if part) or "queue"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return dict(payload)


def _queue_item_id(asset_id: int, model_sha256: str, components: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "asset_id": int(asset_id),
        "model_sha256": str(model_sha256 or ""),
        "components": [
            {
                "role": str(item.get("role") or ""),
                "component_sha256": str(item.get("component_sha256") or ""),
            }
            for item in components
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class ModelComponentParityRunner:
    """Database-driven digital component parity qualification queue.

    The queue is intentionally diagnostic-first. It discovers exact component hashes reused by
    multiple distinct checkpoint contents, then validates one checkpoint at a time using an
    untouched blueprint control plus forced digital extraction from the checkpoint itself and a
    same-hash peer checkpoint. Human parity review remains authoritative during the experimental
    phase; exact pixel equality is recorded as supporting evidence only.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        registry: AssetRegistry | None = None,
        qualification_runner: ComponentQualificationRunner | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.runner = qualification_runner or ComponentQualificationRunner(
            context,
            registry=self.registry,
            python_executable=python_executable,
        )

    def queue_root(self) -> Path:
        return self.runner.default_output_root() / "parity_queues"

    def _eligible_lookup(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        output: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in self.registry.list_model_split_eligibility(limit=1_000_000):
            key = (
                str(record.get("model_sha256") or "").strip().lower(),
                str(record.get("component_role") or "").strip(),
                str(record.get("component_sha256") or "").strip().lower(),
            )
            if all(key):
                output[key] = dict(record)
        return output

    def duplicate_component_groups(
        self,
        *,
        family: str = "",
        include_base_weights: bool = False,
    ) -> list[dict[str, Any]]:
        requested_family = canonical_model_family(family)
        raw_models = self.runner.list_models(family=requested_family)
        models, _ = self.runner.dedupe_models_by_registry_hash(raw_models)
        by_asset = {int(item["asset_id"]): dict(item) for item in models if item.get("asset_id") is not None}
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

        for asset_id, model in by_asset.items():
            model_family = canonical_model_family(model.get("family") or model.get("architecture"))
            if requested_family and model_family != requested_family:
                continue
            provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(model_family)
            base_role = str(provider.base_weight_role if provider is not None else "")
            for snapshot in self.registry.get_component_snapshots(asset_id):
                role = str(snapshot.component_role or "").strip()
                digest = str(snapshot.component_sha256 or "").strip().lower()
                if not role or not digest:
                    continue
                if not include_base_weights and base_role and role == base_role:
                    continue
                key = (model_family, role, digest)
                identity = self.registry.get_component_identity(digest)
                group = grouped.setdefault(
                    key,
                    {
                        "family": model_family,
                        "role": role,
                        "component_sha256": digest,
                        "component_bytes": int(getattr(identity, "total_bytes", 0) or snapshot.total_bytes or 0),
                        "models": [],
                    },
                )
                group["models"].append(
                    {
                        "asset_id": asset_id,
                        "filename": model.get("filename"),
                        "path": model.get("path"),
                        "model_sha256": model.get("sha256"),
                        "registry_location_count": int(model.get("registry_location_count") or 1),
                        "duplicate_aliases": list(model.get("duplicate_aliases") or []),
                    }
                )

        result: list[dict[str, Any]] = []
        for group in grouped.values():
            model_hashes = {str(item.get("model_sha256") or "") for item in group["models"] if item.get("model_sha256")}
            # Strong duplicate-component evidence requires the same component in at least two
            # different checkpoint contents, not two filesystem aliases of one checkpoint.
            if len(group["models"]) < 2 or len(model_hashes) < 2:
                continue
            group["models"].sort(key=lambda item: (str(item.get("filename") or "").casefold(), str(item.get("path") or "").casefold()))
            group["model_count"] = len(group["models"])
            group["estimated_redundant_bytes"] = max(0, len(group["models"]) - 1) * int(group["component_bytes"] or 0)
            group["identity_status"] = "exact_component_sha256"
            result.append(group)

        result.sort(
            key=lambda item: (
                -int(item.get("estimated_redundant_bytes") or 0),
                -int(item.get("model_count") or 0),
                str(item.get("family") or ""),
                str(item.get("role") or ""),
                str(item.get("component_sha256") or ""),
            )
        )
        return result

    def suggest_queue(
        self,
        *,
        family: str = "",
        include_base_weights: bool = False,
        include_blocked: bool = False,
        include_eligible: bool = False,
    ) -> list[dict[str, Any]]:
        groups = self.duplicate_component_groups(
            family=family,
            include_base_weights=include_base_weights,
        )
        eligibility = self._eligible_lookup()
        per_model: dict[tuple[str, str], dict[str, Any]] = {}

        for group in groups:
            members = list(group["models"])
            for member in members:
                model_sha = str(member.get("model_sha256") or "").strip().lower()
                if not model_sha:
                    # Do not guess model identity for destructive-work recommendations.
                    continue
                existing = eligibility.get((model_sha, group["role"], group["component_sha256"]))
                state = str((existing or {}).get("eligibility_state") or SPLIT_ELIGIBILITY_UNTESTED)
                if state == SPLIT_ELIGIBILITY_ELIGIBLE and not include_eligible:
                    continue
                if state == SPLIT_ELIGIBILITY_BLOCKED and not include_blocked:
                    continue
                peers = [item for item in members if int(item["asset_id"]) != int(member["asset_id"])]
                if not peers:
                    continue
                # Prefer a peer with the fewest duplicate filesystem aliases to keep source evidence simple.
                peers.sort(
                    key=lambda item: (
                        int(item.get("registry_location_count") or 1),
                        str(item.get("filename") or "").casefold(),
                    )
                )
                peer = peers[0]
                model_key = (model_sha, str(member.get("path") or ""))
                item = per_model.setdefault(
                    model_key,
                    {
                        "asset_id": int(member["asset_id"]),
                        "model_sha256": model_sha,
                        "filename": str(member.get("filename") or ""),
                        "path": str(member.get("path") or ""),
                        "family": str(group.get("family") or ""),
                        "registry_location_count": int(member.get("registry_location_count") or 1),
                        "components": [],
                        "smart_score": 0,
                    },
                )
                item["components"].append(
                    {
                        "role": str(group["role"]),
                        "component_sha256": str(group["component_sha256"]),
                        "component_bytes": int(group.get("component_bytes") or 0),
                        "duplicate_model_count": int(group.get("model_count") or 0),
                        "estimated_redundant_bytes": int(group.get("estimated_redundant_bytes") or 0),
                        "peer_asset_id": int(peer["asset_id"]),
                        "peer_filename": str(peer.get("filename") or ""),
                        "peer_path": str(peer.get("path") or ""),
                        "peer_model_sha256": str(peer.get("model_sha256") or ""),
                        "prior_eligibility_state": state,
                    }
                )
                item["smart_score"] += int(group.get("estimated_redundant_bytes") or 0)

        items = list(per_model.values())
        for item in items:
            item["components"].sort(
                key=lambda component: (
                    -int(component.get("estimated_redundant_bytes") or 0),
                    str(component.get("role") or ""),
                )
            )
            item["queue_item_id"] = _queue_item_id(
                int(item["asset_id"]),
                str(item["model_sha256"]),
                item["components"],
            )
        items.sort(
            key=lambda item: (
                -int(item.get("smart_score") or 0),
                -len(item.get("components") or []),
                str(item.get("family") or ""),
                str(item.get("filename") or "").casefold(),
            )
        )
        for index, item in enumerate(items, start=1):
            item["position"] = index
            item["status"] = "pending"
            item["run_dir"] = None
            item["recommendation"] = None
        return items

    def create_queue(
        self,
        *,
        family: str = "",
        include_base_weights: bool = False,
        include_blocked: bool = False,
        include_eligible: bool = False,
    ) -> Path:
        items = self.suggest_queue(
            family=family,
            include_base_weights=include_base_weights,
            include_blocked=include_blocked,
            include_eligible=include_eligible,
        )
        queue_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + _slug(canonical_model_family(family) or "all-families")
        )
        queue_dir = self.queue_root() / queue_id
        queue_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": PARITY_QUEUE_SCHEMA_VERSION,
            "queue_id": queue_id,
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "registry_db_path": str(self.context.registry_db_path),
            "family_filter": canonical_model_family(family) or None,
            "include_base_weights": bool(include_base_weights),
            "soft_split_gate": {
                "mode": SPLIT_GATE_RECOMMENDED,
                "override_allowed": True,
                "note": "Digital parity is highly recommended before physical splitting but is not a hard prohibition.",
            },
            "current_index": 0,
            "total_items": len(items),
            "items": items,
        }
        _write_yaml(queue_dir / "parity_queue.yaml", payload)
        return queue_dir

    def load_queue(self, queue_dir: str | Path) -> dict[str, Any]:
        root = Path(queue_dir).expanduser().resolve()
        path = root / "parity_queue.yaml"
        if not path.is_file():
            raise ValueError(f"Not a parity queue directory: {root}")
        payload = _load_yaml(path)
        if int(payload.get("schema_version") or 0) != PARITY_QUEUE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported parity queue schema: {payload.get('schema_version')}")
        return payload

    def _save_queue(self, queue_dir: Path, queue: Mapping[str, Any]) -> None:
        payload = dict(queue)
        payload["updated_at_utc"] = _utc_now()
        _write_yaml(queue_dir / "parity_queue.yaml", payload)

    @staticmethod
    def _next_item_index(queue: Mapping[str, Any]) -> int | None:
        items = list(queue.get("items") or [])
        for index, item in enumerate(items):
            if str(item.get("status") or "pending") in {"pending", "waiting_review"}:
                return index
        return None

    def status(self, queue_dir: str | Path) -> dict[str, Any]:
        root = Path(queue_dir).expanduser().resolve()
        queue = self.load_queue(root)
        items = list(queue.get("items") or [])
        next_index = self._next_item_index(queue)
        counts: dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "pending")
            counts[status] = counts.get(status, 0) + 1
        current = dict(items[next_index]) if next_index is not None else None
        return {
            "queue_dir": str(root),
            "queue_id": queue.get("queue_id"),
            "total_items": len(items),
            "counts": counts,
            "next_position": (next_index + 1) if next_index is not None else None,
            "current": current,
            "complete": next_index is None,
        }

    def run_current(
        self,
        queue_dir: str | Path,
        *,
        profile_path: str | Path | None = None,
        stop_after_control_failure: bool = True,
    ) -> Path | None:
        root = Path(queue_dir).expanduser().resolve()
        queue = self.load_queue(root)
        items = list(queue.get("items") or [])
        index = self._next_item_index(queue)
        if index is None:
            return None
        item = dict(items[index])
        if str(item.get("status") or "") == "waiting_review" and item.get("run_dir"):
            return Path(str(item["run_dir"]))

        if profile_path is None:
            profile_path = (
                self.runner.model_output_root(item["filename"])
                / "profiles"
                / f"{Path(str(item['filename'])).stem}.yaml"
            )
            profile_path = Path(profile_path)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            if not profile_path.is_file():
                self.runner.write_profile_template(item["path"], profile_path)

        plan = self.runner.build_component_source_parity_plan(
            model_path=item["path"],
            components=item["components"],
            profile_path=profile_path,
        )
        run_dir = self.runner.run_plan(
            plan,
            output_root=self.runner.default_output_root(),
            stop_after_control_failure=stop_after_control_failure,
        )
        item["status"] = "waiting_review"
        item["run_dir"] = str(run_dir)
        item["started_at_utc"] = _utc_now()
        item["position"] = index + 1
        items[index] = item
        queue["items"] = items
        queue["current_index"] = index
        self._save_queue(root, queue)
        return run_dir

    @staticmethod
    def _read_case_review(run_dir: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        case_dir = run_dir / "cases" / case_id
        review_path = case_dir / "review.yaml"
        case_path = case_dir / "case.yaml"
        if not review_path.is_file() or not case_path.is_file():
            raise ValueError(f"Parity case {case_id!r} is missing review/case evidence under {run_dir}.")
        return _load_yaml(review_path), _load_yaml(case_path)

    def _component_recommendation(
        self,
        *,
        item: Mapping[str, Any],
        component: Mapping[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        role = str(component["role"])
        self_case_id = f"{_slug(role)}-digital-self"
        peer_case_id = f"{_slug(role)}-digital-peer"
        self_review, self_case = self._read_case_review(run_dir, self_case_id)
        peer_review, peer_case = self._read_case_review(run_dir, peer_case_id)
        self_choice = str((self_review.get("review") or {}).get("choice") or PARITY_REVIEW_PENDING).strip().lower()
        peer_choice = str((peer_review.get("review") or {}).get("choice") or PARITY_REVIEW_PENDING).strip().lower()
        technical_ok = (
            str(self_case.get("technical_status") or "") == "success"
            and str(peer_case.get("technical_status") or "") == "success"
        )

        if self_choice == PARITY_REVIEW_PENDING or peer_choice == PARITY_REVIEW_PENDING:
            state = SPLIT_ELIGIBILITY_INCONCLUSIVE
            parity_status = "pending_review"
            reason = "Manual parity review is incomplete."
        elif not technical_ok:
            state = SPLIT_ELIGIBILITY_BLOCKED
            parity_status = "technical_failure"
            reason = "Digital extraction parity could not be established because one or more parity cases failed technically."
        elif PARITY_REVIEW_MISMATCH in {self_choice, peer_choice}:
            state = SPLIT_ELIGIBILITY_BLOCKED
            parity_status = "failed"
            reason = "Manual review found a parity mismatch; investigate digital extraction/hydration before physical splitting."
        elif self_choice == PARITY_REVIEW_MATCH and peer_choice == PARITY_REVIEW_MATCH:
            state = SPLIT_ELIGIBILITY_ELIGIBLE
            parity_status = "validated"
            reason = "Self-source and same-hash peer digital extraction both matched the untouched blueprint control."
        else:
            state = SPLIT_ELIGIBILITY_INCONCLUSIVE
            parity_status = "inconclusive"
            reason = "Manual parity review did not establish a pass or a definite mismatch."

        automatic = {
            "self": str(self_case.get("automatic_parity") or "unavailable"),
            "peer": str(peer_case.get("automatic_parity") or "unavailable"),
        }
        return {
            "role": role,
            "component_sha256": str(component["component_sha256"]),
            "eligibility_state": state,
            "digital_parity_status": parity_status,
            "reason": reason,
            "self_review": self_choice,
            "peer_review": peer_choice,
            "automatic_pixel_parity": automatic,
            "self_case_id": self_case_id,
            "peer_case_id": peer_case_id,
            "peer_asset_id": int(component["peer_asset_id"]),
            "peer_filename": str(component.get("peer_filename") or ""),
        }

    def collect_current(self, queue_dir: str | Path) -> dict[str, Any]:
        root = Path(queue_dir).expanduser().resolve()
        queue = self.load_queue(root)
        items = list(queue.get("items") or [])
        index = self._next_item_index(queue)
        if index is None:
            return {"complete": True, "recommendation": None, "next_position": None}
        item = dict(items[index])
        if str(item.get("status") or "") != "waiting_review" or not item.get("run_dir"):
            raise ValueError("Current parity queue item has not been generated yet.")
        run_dir = Path(str(item["run_dir"])).expanduser().resolve()
        run_record = _load_yaml(run_dir / "run.yaml")
        blueprint = dict(run_record.get("blueprint") or {})
        base_role = str(blueprint.get("base_weight_role") or "")
        base_sha = str((blueprint.get("components") or {}).get(base_role) or "")
        composition_sha = ""
        cases = list(run_record.get("cases") or [])
        control_case = next((case for case in cases if case.get("case_id") == "control"), None)
        if control_case:
            composition_sha = str((control_case.get("resolved_composition") or {}).get("composition_sha256") or "")

        recommendations = [
            self._component_recommendation(item=item, component=component, run_dir=run_dir)
            for component in item.get("components") or []
        ]
        pending = [rec for rec in recommendations if rec["digital_parity_status"] == "pending_review"]
        if pending:
            return {
                "complete": False,
                "waiting_for_review": True,
                "position": index + 1,
                "total_items": len(items),
                "model": item["filename"],
                "pending_roles": [rec["role"] for rec in pending],
                "run_dir": str(run_dir),
            }

        stored_records: list[dict[str, Any]] = []
        for rec in recommendations:
            validation: dict[str, Any] | None = None
            if rec["eligibility_state"] in {SPLIT_ELIGIBILITY_ELIGIBLE, SPLIT_ELIGIBILITY_BLOCKED}:
                validation_result = (
                    VALIDATION_RESULT_PASS
                    if rec["eligibility_state"] == SPLIT_ELIGIBILITY_ELIGIBLE
                    else VALIDATION_RESULT_FAIL
                )
                validation = self.registry.record_component_validation(
                    component_sha256=rec["component_sha256"],
                    validation_stage=VALIDATION_STAGE_PARITY,
                    validation_result=validation_result,
                    family_id=str(item.get("family") or ""),
                    base_component_sha256=base_sha or None,
                    composition_sha256=composition_sha or None,
                    component_role=rec["role"],
                    blocking_state=VALIDATION_ADVISORY,
                    evidence_type="digital_component_source_parity",
                    evidence={
                        "evidence_version": PARITY_EVIDENCE_VERSION,
                        "model_asset_id": int(item["asset_id"]),
                        "model_sha256": str(item.get("model_sha256") or ""),
                        "model_filename": str(item.get("filename") or ""),
                        "manual_reviews": {
                            "self": rec["self_review"],
                            "peer": rec["peer_review"],
                        },
                        "automatic_pixel_parity": dict(rec["automatic_pixel_parity"]),
                        "peer_asset_id": rec["peer_asset_id"],
                        "peer_filename": rec["peer_filename"],
                        "soft_split_gate": True,
                    },
                    evidence_artifact=str(run_dir),
                )
            eligibility = self.registry.record_model_split_eligibility(
                asset_id=int(item["asset_id"]),
                model_sha256=str(item.get("model_sha256") or ""),
                family_id=str(item.get("family") or ""),
                component_role=rec["role"],
                component_sha256=rec["component_sha256"],
                eligibility_state=rec["eligibility_state"],
                digital_parity_status=rec["digital_parity_status"],
                parity_validation_id=int(validation["id"]) if validation else None,
                gate_mode=SPLIT_GATE_RECOMMENDED,
                evidence_artifact=str(run_dir),
                recommendation_reason=rec["reason"],
                evidence={
                    "evidence_version": PARITY_EVIDENCE_VERSION,
                    "run_dir": str(run_dir),
                    "manual_reviews": {"self": rec["self_review"], "peer": rec["peer_review"]},
                    "automatic_pixel_parity": dict(rec["automatic_pixel_parity"]),
                    "peer_asset_id": rec["peer_asset_id"],
                    "peer_filename": rec["peer_filename"],
                    "override_allowed": True,
                },
            )
            stored_records.append(eligibility)

        eligible_roles = [rec["role"] for rec in recommendations if rec["eligibility_state"] == SPLIT_ELIGIBILITY_ELIGIBLE]
        blocked_roles = [rec["role"] for rec in recommendations if rec["eligibility_state"] == SPLIT_ELIGIBILITY_BLOCKED]
        inconclusive_roles = [rec["role"] for rec in recommendations if rec["eligibility_state"] == SPLIT_ELIGIBILITY_INCONCLUSIVE]
        recommendation = {
            "model": str(item.get("filename") or ""),
            "asset_id": int(item["asset_id"]),
            "model_sha256": str(item.get("model_sha256") or ""),
            "eligible_for_split": bool(eligible_roles),
            "eligible_roles": eligible_roles,
            "blocked_roles": blocked_roles,
            "inconclusive_roles": inconclusive_roles,
            "soft_rule": True,
            "override_allowed": True,
            "note": (
                "Physical split is recommended only for eligible roles. The future splitter may allow an explicit user override "
                "for untested/blocked roles, but diagnostic parity is the development default."
            ),
            "components": recommendations,
            "run_dir": str(run_dir),
        }
        _write_yaml(run_dir / "split_recommendation.yaml", recommendation)

        if inconclusive_roles:
            item["status"] = "inconclusive"
        elif blocked_roles and not eligible_roles:
            item["status"] = "blocked"
        else:
            item["status"] = "completed"
        item["completed_at_utc"] = _utc_now()
        item["recommendation"] = recommendation
        items[index] = item
        queue["items"] = items
        queue["current_index"] = min(index + 1, len(items))
        self._save_queue(root, queue)
        next_index = self._next_item_index(queue)
        return {
            "complete": next_index is None,
            "waiting_for_review": False,
            "position": index + 1,
            "total_items": len(items),
            "recommendation": recommendation,
            "next_position": (next_index + 1) if next_index is not None else None,
        }


__all__ = [
    "ModelComponentParityRunner",
    "PARITY_EVIDENCE_VERSION",
    "PARITY_QUEUE_SCHEMA_VERSION",
]
