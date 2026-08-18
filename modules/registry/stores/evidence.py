from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
import json

from ..evidence_contracts import (
    POLICY_ACTION_DISABLE,
    POLICY_SCOPE_BASE,
    POLICY_SCOPE_GLOBAL,
    POLICY_SOURCES,
    RELATIONSHIP_SOURCE_EXACT_ANALYSIS,
    RELATIONSHIP_SOURCE_RECORDED,
    RELATIONSHIP_STATUS_ACTIVE,
    RELATIONSHIP_STATUS_SUPERSEDED,
    RELATIONSHIP_STATUSES,
    RelationshipParticipant,
    VALIDATION_ADVISORY,
    VALIDATION_BLOCKING_STATES,
    VALIDATION_RESULTS,
    VALIDATION_STAGES,
    SPLIT_ELIGIBILITY_STATES,
    SPLIT_ELIGIBILITY_UNTESTED,
    SPLIT_GATE_MODES,
    SPLIT_GATE_RECOMMENDED,
    canonical_json,
    normalize_relationship_participants,
    normalized_blocking_state,
    relationship_key,
)


class EvidenceStore:
    """Relationship, policy, validation, and split-eligibility persistence operations."""

    def upsert_component_relationship(
        self,
        *,
        source_component_sha256: str,
        target_component_sha256: str,
        relationship_type: str,
        evidence_kind: str,
        evidence_version: str,
        evidence_json: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        source = str(source_component_sha256 or "").strip().lower()
        target = str(target_component_sha256 or "").strip().lower()
        if not source or not target:
            raise ValueError("Component relationship endpoints must not be empty.")
        if source == target:
            raise ValueError("Pairwise component relationships must use distinct component identities.")
        source, target = sorted((source, target))
        relationship = str(relationship_type or "").strip()
        kind = str(evidence_kind or "").strip()
        version = str(evidence_version or "").strip()
        if not relationship or not kind or not version:
            raise ValueError("Relationship type, evidence kind, and evidence version are required.")
        if isinstance(evidence_json, str):
            serialized = evidence_json
        else:
            serialized = json.dumps(
                dict(evidence_json or {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM component_relationships
                WHERE source_component_sha256 = ?
                  AND target_component_sha256 = ?
                  AND relationship_type = ?
                  AND evidence_kind = ?
                  AND evidence_version = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (source, target, relationship, kind, version),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    """
                    INSERT INTO component_relationships (
                        source_component_sha256, target_component_sha256,
                        relationship_type, evidence_kind, evidence_version, evidence_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (source, target, relationship, kind, version, serialized, now, now),
                )
                row_id = int(cursor.lastrowid)
            else:
                row_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE component_relationships
                    SET evidence_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (serialized, now, row_id),
                )
                # Normalize any historical duplicates without requiring a schema bump.
                conn.execute(
                    """
                    DELETE FROM component_relationships
                    WHERE source_component_sha256 = ?
                      AND target_component_sha256 = ?
                      AND relationship_type = ?
                      AND evidence_kind = ?
                      AND evidence_version = ?
                      AND id <> ?
                    """,
                    (source, target, relationship, kind, version, row_id),
                )
            stored = conn.execute(
                "SELECT * FROM component_relationships WHERE id = ?",
                (row_id,),
            ).fetchone()
        if stored is None:
            raise RuntimeError("Stored component relationship could not be read back.")
        result = dict(stored)
        evidence_source = (
            RELATIONSHIP_SOURCE_EXACT_ANALYSIS
            if "exact" in kind.casefold() or "analysis" in kind.casefold()
            else RELATIONSHIP_SOURCE_RECORDED
        )
        self.upsert_relationship_evidence(
            relationship_type=relationship,
            participants=(
                RelationshipParticipant(component_sha256=source, participant_role="left", position=0),
                RelationshipParticipant(component_sha256=target, participant_role="right", position=1),
            ),
            evidence_source=evidence_source,
            evidence_kind=kind,
            evidence_version=version,
            evidence_json=serialized,
            supersede_previous=True,
        )
        return result

    def list_component_relationships(
        self,
        *,
        component_sha256: str | None = None,
        relationship_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if component_sha256:
            digest = str(component_sha256).strip().lower()
            clauses.append("(source_component_sha256 = ? OR target_component_sha256 = ?)")
            params.extend([digest, digest])
        if relationship_type:
            clauses.append("relationship_type = ?")
            params.append(str(relationship_type))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM component_relationships {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_relationship_evidence(
        self,
        *,
        relationship_type: str,
        participants: Iterable[RelationshipParticipant | Mapping[str, Any]],
        evidence_source: str,
        evidence_kind: str,
        evidence_version: str,
        provider_id: str = "",
        family_id: str = "",
        algorithm_id: str = "",
        algorithm_version: str = "",
        layout_version: int | None = None,
        authoritative: bool = False,
        confidence: float | None = None,
        status: str = RELATIONSHIP_STATUS_ACTIVE,
        evidence_json: Mapping[str, Any] | str | None = None,
        supersede_previous: bool = True,
    ) -> dict[str, Any]:
        normalized = normalize_relationship_participants(participants)
        relationship = str(relationship_type or "").strip()
        source = str(evidence_source or "").strip().lower()
        kind = str(evidence_kind or "").strip()
        version = str(evidence_version or "").strip()
        normalized_status = str(status or RELATIONSHIP_STATUS_ACTIVE).strip().lower()
        if source not in {
            "recorded", "exact_analysis", "inferred", "runtime_validated"
        }:
            raise ValueError(f"Unsupported relationship evidence source: {evidence_source!r}")
        if normalized_status not in RELATIONSHIP_STATUSES:
            raise ValueError(f"Unsupported relationship status: {status!r}")
        if not relationship or not kind or not version:
            raise ValueError("Relationship type, evidence kind, and evidence version are required.")
        rel_key = relationship_key(
            relationship_type=relationship,
            participants=normalized,
            family_id=family_id,
            provider_id=provider_id,
        )
        serialized = evidence_json if isinstance(evidence_json, str) else canonical_json(dict(evidence_json or {}))
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id, created_at FROM component_relationship_evidence
                WHERE relationship_key = ? AND evidence_source = ?
                  AND evidence_kind = ? AND evidence_version = ?
                LIMIT 1
                """,
                (rel_key, source, kind, version),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO component_relationship_evidence (
                        relationship_key, relationship_type, evidence_source,
                        evidence_kind, evidence_version, provider_id, family_id,
                        algorithm_id, algorithm_version, layout_version,
                        authoritative, confidence, status, evidence_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel_key,
                        relationship,
                        source,
                        kind,
                        version,
                        str(provider_id or "") or None,
                        str(family_id or "") or None,
                        str(algorithm_id or "") or None,
                        str(algorithm_version or "") or None,
                        (int(layout_version) if layout_version is not None else None),
                        1 if authoritative else 0,
                        (float(confidence) if confidence is not None else None),
                        normalized_status,
                        serialized,
                        now,
                        now,
                    ),
                )
                evidence_id = int(cursor.lastrowid)
            else:
                evidence_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE component_relationship_evidence
                    SET relationship_type = ?, provider_id = ?, family_id = ?,
                        algorithm_id = ?, algorithm_version = ?, layout_version = ?,
                        authoritative = ?, confidence = ?, status = ?, evidence_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        relationship,
                        str(provider_id or "") or None,
                        str(family_id or "") or None,
                        str(algorithm_id or "") or None,
                        str(algorithm_version or "") or None,
                        (int(layout_version) if layout_version is not None else None),
                        1 if authoritative else 0,
                        (float(confidence) if confidence is not None else None),
                        normalized_status,
                        serialized,
                        now,
                        evidence_id,
                    ),
                )
            conn.execute(
                "DELETE FROM component_relationship_participants WHERE relationship_evidence_id = ?",
                (evidence_id,),
            )
            for item in normalized:
                conn.execute(
                    """
                    INSERT INTO component_relationship_participants (
                        relationship_evidence_id, position, participant_role,
                        component_sha256, composition_id, blueprint_id, weight, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        int(item.position),
                        item.participant_role,
                        item.component_sha256 or None,
                        item.composition_id or None,
                        item.blueprint_id or None,
                        item.weight,
                        canonical_json(dict(item.metadata)),
                    ),
                )
            if supersede_previous and normalized_status == RELATIONSHIP_STATUS_ACTIVE:
                conn.execute(
                    """
                    UPDATE component_relationship_evidence
                    SET status = ?, superseded_by_id = ?, updated_at = ?
                    WHERE relationship_key = ? AND evidence_source = ? AND evidence_kind = ?
                      AND evidence_version <> ? AND id <> ? AND status = ?
                    """,
                    (
                        RELATIONSHIP_STATUS_SUPERSEDED,
                        evidence_id,
                        now,
                        rel_key,
                        source,
                        kind,
                        version,
                        evidence_id,
                        RELATIONSHIP_STATUS_ACTIVE,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM component_relationship_evidence WHERE id = ?",
                (evidence_id,),
            ).fetchone()
            participant_rows = conn.execute(
                """
                SELECT * FROM component_relationship_participants
                WHERE relationship_evidence_id = ? ORDER BY position ASC, id ASC
                """,
                (evidence_id,),
            ).fetchall()
        if row is None:
            raise RuntimeError("Stored relationship evidence could not be read back.")
        result = dict(row)
        result["authoritative"] = bool(result.get("authoritative"))
        result["participants"] = [dict(item) for item in participant_rows]
        return result

    def list_relationship_evidence(
        self,
        *,
        component_sha256: str | None = None,
        relationship_type: str | None = None,
        evidence_source: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        join = ""
        if component_sha256:
            join = "JOIN component_relationship_participants p ON p.relationship_evidence_id = e.id"
            clauses.append("p.component_sha256 = ?")
            params.append(str(component_sha256).strip().lower())
        if relationship_type:
            clauses.append("e.relationship_type = ?")
            params.append(str(relationship_type))
        if evidence_source:
            clauses.append("e.evidence_source = ?")
            params.append(str(evidence_source).strip().lower())
        if status:
            clauses.append("e.status = ?")
            params.append(str(status).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT e.* FROM component_relationship_evidence e {join} {where} ORDER BY e.updated_at DESC, e.id DESC LIMIT ?",
                params,
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["authoritative"] = bool(item.get("authoritative"))
                participants = conn.execute(
                    """
                    SELECT * FROM component_relationship_participants
                    WHERE relationship_evidence_id = ? ORDER BY position ASC, id ASC
                    """,
                    (int(row["id"]),),
                ).fetchall()
                item["participants"] = [dict(value) for value in participants]
                results.append(item)
        return results

    def set_component_policy(
        self,
        *,
        component_sha256: str,
        policy_scope: str,
        base_component_sha256: str | None = None,
        component_role: str | None = None,
        policy_action: str = POLICY_ACTION_DISABLE,
        policy_source: str = "user",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        component = str(component_sha256 or "").strip().lower()
        if len(component) != 64 or any(ch not in "0123456789abcdef" for ch in component):
            raise ValueError("component_sha256 must be a 64-character hexadecimal SHA-256.")
        scope = str(policy_scope or "").strip().lower()
        if scope not in {POLICY_SCOPE_GLOBAL, POLICY_SCOPE_BASE}:
            raise ValueError("policy_scope must be 'global' or 'base'.")
        base = str(base_component_sha256 or "").strip().lower()
        if scope == POLICY_SCOPE_BASE:
            if len(base) != 64 or any(ch not in "0123456789abcdef" for ch in base):
                raise ValueError("Per-base policy requires a 64-character base_component_sha256.")
        else:
            base = ""
        action = str(policy_action or POLICY_ACTION_DISABLE).strip().lower()
        if action != POLICY_ACTION_DISABLE:
            raise ValueError("Phase 05 currently supports only the 'disable' policy action; re-enable removes that policy.")
        source = str(policy_source or "user").strip().lower()
        if source not in POLICY_SOURCES:
            raise ValueError(f"Unsupported policy source: {policy_source!r}")
        role = str(component_role or "").strip() or None
        now = datetime.now(timezone.utc).isoformat()
        serialized = canonical_json(dict(metadata or {}))
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM component_policies
                WHERE policy_scope = ? AND COALESCE(base_component_sha256, '') = ?
                  AND component_sha256 = ? AND COALESCE(component_role, '') = ?
                  AND policy_action = ?
                LIMIT 1
                """,
                (scope, base, component, role or "", action),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO component_policies (
                        policy_scope, base_component_sha256, component_sha256,
                        component_role, policy_action, policy_source, reason,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (scope, base or None, component, role, action, source, str(reason or ""), serialized, now, now),
                )
                row_id = int(cursor.lastrowid)
            else:
                row_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE component_policies
                    SET policy_source = ?, reason = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (source, str(reason or ""), serialized, now, row_id),
                )
            row = conn.execute("SELECT * FROM component_policies WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise RuntimeError("Stored component policy could not be read back.")
        return dict(row)

    def clear_component_policy(
        self,
        *,
        component_sha256: str,
        policy_scope: str,
        base_component_sha256: str | None = None,
        component_role: str | None = None,
    ) -> int:
        scope = str(policy_scope or "").strip().lower()
        base = str(base_component_sha256 or "").strip().lower()
        role = str(component_role or "").strip()
        clauses = ["policy_scope = ?", "component_sha256 = ?", "policy_action = ?"]
        params: list[Any] = [scope, str(component_sha256 or "").strip().lower(), POLICY_ACTION_DISABLE]
        if scope == POLICY_SCOPE_BASE:
            clauses.append("base_component_sha256 = ?")
            params.append(base)
        else:
            clauses.append("base_component_sha256 IS NULL")
        if role:
            clauses.append("component_role = ?")
            params.append(role)
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM component_policies WHERE {' AND '.join(clauses)}", params)
            return int(cursor.rowcount or 0)

    def list_component_policies(
        self,
        *,
        component_sha256: str | None = None,
        base_component_sha256: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if component_sha256:
            clauses.append("component_sha256 = ?")
            params.append(str(component_sha256).strip().lower())
        if base_component_sha256:
            clauses.append("base_component_sha256 = ?")
            params.append(str(base_component_sha256).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM component_policies {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def record_component_validation(
        self,
        *,
        component_sha256: str,
        validation_stage: str,
        validation_result: str,
        family_id: str = "",
        provider_version: str = "",
        base_component_sha256: str | None = None,
        composition_sha256: str | None = None,
        component_role: str = "",
        blocking_state: str = VALIDATION_ADVISORY,
        evidence_type: str = "runtime_validation",
        evidence: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        evidence_artifact: str = "",
        error_category: str = "",
        error_message: str = "",
        runtime_version: str = "",
    ) -> dict[str, Any]:
        component = str(component_sha256 or "").strip().lower()
        if len(component) != 64 or any(ch not in "0123456789abcdef" for ch in component):
            raise ValueError("component_sha256 must be a 64-character hexadecimal SHA-256.")
        base = str(base_component_sha256 or "").strip().lower()
        if base and (len(base) != 64 or any(ch not in "0123456789abcdef" for ch in base)):
            raise ValueError("base_component_sha256 must be empty or a 64-character hexadecimal SHA-256.")
        composition = str(composition_sha256 or "").strip().lower()
        if composition and (len(composition) != 64 or any(ch not in "0123456789abcdef" for ch in composition)):
            raise ValueError("composition_sha256 must be empty or a 64-character hexadecimal SHA-256.")
        stage = str(validation_stage or "").strip().lower()
        result = str(validation_result or "").strip().lower()
        if stage not in VALIDATION_STAGES:
            raise ValueError(f"Unsupported validation stage: {validation_stage!r}")
        if result not in VALIDATION_RESULTS:
            raise ValueError(f"Unsupported validation result: {validation_result!r}")
        requested_blocking = str(blocking_state or VALIDATION_ADVISORY).strip().lower()
        if requested_blocking not in VALIDATION_BLOCKING_STATES:
            raise ValueError(f"Unsupported validation blocking state: {blocking_state!r}")
        effective_blocking = normalized_blocking_state(
            result=result,
            requested=requested_blocking,
            error_category=error_category,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO component_validations (
                    family_id, provider_version, base_component_sha256,
                    component_sha256, composition_sha256, component_role,
                    validation_state, validation_stage, validation_result,
                    blocking_state, evidence_type, evidence_json, environment_json,
                    evidence_artifact, error_category, error_message, runtime_version,
                    validated_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(family_id or "") or None,
                    str(provider_version or "") or None,
                    base or None,
                    component,
                    composition or None,
                    str(component_role or "") or None,
                    result,
                    stage,
                    result,
                    effective_blocking,
                    str(evidence_type or "runtime_validation"),
                    canonical_json(dict(evidence or {})),
                    canonical_json(dict(environment or {})),
                    str(evidence_artifact or "") or None,
                    str(error_category or "") or None,
                    str(error_message or "") or None,
                    str(runtime_version or "") or None,
                    now,
                    now,
                    now,
                ),
            )
            row_id = int(cursor.lastrowid)
            row = conn.execute("SELECT * FROM component_validations WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise RuntimeError("Stored component validation could not be read back.")
        return dict(row)

    def list_component_validations(
        self,
        *,
        component_sha256: str | None = None,
        base_component_sha256: str | None = None,
        composition_sha256: str | None = None,
        validation_stage: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("component_sha256", component_sha256),
            ("base_component_sha256", base_component_sha256),
            ("composition_sha256", composition_sha256),
            ("validation_stage", validation_stage),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value).strip().lower() if "sha256" in column else str(value).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM component_validations {where} ORDER BY validated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_component_validations(
        self,
        *,
        component_sha256: str | None = None,
        base_component_sha256: str | None = None,
        composition_sha256: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("component_sha256", component_sha256),
            ("base_component_sha256", base_component_sha256),
            ("composition_sha256", composition_sha256),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value).strip().lower())
        if not clauses:
            raise ValueError("Clearing validation evidence requires a component, base, or composition fingerprint filter.")
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM component_validations WHERE {' AND '.join(clauses)}", params)
            return int(cursor.rowcount or 0)

    def record_model_split_eligibility(
        self,
        *,
        asset_id: int,
        component_role: str,
        component_sha256: str,
        eligibility_state: str,
        digital_parity_status: str,
        family_id: str = "",
        model_sha256: str = "",
        parity_validation_id: int | None = None,
        gate_mode: str = SPLIT_GATE_RECOMMENDED,
        evidence_artifact: str = "",
        recommendation_reason: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        asset = self.get_asset_by_id(int(asset_id))
        if asset is None:
            raise ValueError(f"Unknown asset_id={asset_id} for split eligibility.")
        role = str(component_role or "").strip()
        if not role:
            raise ValueError("component_role is required for split eligibility.")
        component = str(component_sha256 or "").strip().lower()
        if len(component) != 64 or any(ch not in "0123456789abcdef" for ch in component):
            raise ValueError("component_sha256 must be a 64-character hexadecimal SHA-256.")
        state = str(eligibility_state or "").strip().lower()
        if state not in SPLIT_ELIGIBILITY_STATES:
            raise ValueError(f"Unsupported split eligibility state: {eligibility_state!r}")
        gate = str(gate_mode or SPLIT_GATE_RECOMMENDED).strip().lower()
        if gate not in SPLIT_GATE_MODES:
            raise ValueError(f"Unsupported split gate mode: {gate_mode!r}")
        parity_status = str(digital_parity_status or SPLIT_ELIGIBILITY_UNTESTED).strip().lower()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_split_eligibility (
                    asset_id, model_sha256, family_id, component_role, component_sha256,
                    eligibility_state, gate_mode, digital_parity_status, parity_validation_id,
                    evidence_artifact, recommendation_reason, evidence_json,
                    validated_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, component_role, component_sha256) DO UPDATE SET
                    model_sha256 = excluded.model_sha256,
                    family_id = excluded.family_id,
                    eligibility_state = excluded.eligibility_state,
                    gate_mode = excluded.gate_mode,
                    digital_parity_status = excluded.digital_parity_status,
                    parity_validation_id = excluded.parity_validation_id,
                    evidence_artifact = excluded.evidence_artifact,
                    recommendation_reason = excluded.recommendation_reason,
                    evidence_json = excluded.evidence_json,
                    validated_at = excluded.validated_at,
                    updated_at = excluded.updated_at
                """,
                (
                    int(asset_id),
                    str(model_sha256 or asset.sha256 or "") or None,
                    str(family_id or asset.architecture or "") or None,
                    role,
                    component,
                    state,
                    gate,
                    parity_status,
                    int(parity_validation_id) if parity_validation_id is not None else None,
                    str(evidence_artifact or "") or None,
                    str(recommendation_reason or "") or None,
                    canonical_json(dict(evidence or {})),
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM model_split_eligibility WHERE asset_id = ? AND component_role = ? AND component_sha256 = ?",
                (int(asset_id), role, component),
            ).fetchone()
        if row is None:
            raise RuntimeError("Stored split eligibility could not be read back.")
        return dict(row)

    def list_model_split_eligibility(
        self,
        *,
        asset_id: int | None = None,
        model_sha256: str | None = None,
        component_sha256: str | None = None,
        component_role: str | None = None,
        eligibility_state: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(int(asset_id))
        if model_sha256:
            clauses.append("model_sha256 = ?")
            params.append(str(model_sha256).strip().lower())
        if component_sha256:
            clauses.append("component_sha256 = ?")
            params.append(str(component_sha256).strip().lower())
        if component_role:
            clauses.append("component_role = ?")
            params.append(str(component_role).strip())
        if eligibility_state:
            clauses.append("eligibility_state = ?")
            params.append(str(eligibility_state).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM model_split_eligibility {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def split_preflight(
        self,
        *,
        asset_id: int,
        component_role: str,
        component_sha256: str,
    ) -> dict[str, Any]:
        rows = self.list_model_split_eligibility(
            asset_id=int(asset_id),
            component_role=component_role,
            component_sha256=component_sha256,
            limit=1,
        )
        if not rows:
            asset = self.get_asset_by_id(int(asset_id))
            if asset is not None and str(asset.sha256 or "").strip():
                rows = self.list_model_split_eligibility(
                    model_sha256=str(asset.sha256).strip().lower(),
                    component_role=component_role,
                    component_sha256=component_sha256,
                    limit=1,
                )
        if rows:
            record = dict(rows[0])
            return {
                "recommended": str(record.get("eligibility_state") or "") == "eligible",
                "eligibility_state": str(record.get("eligibility_state") or "untested"),
                "digital_parity_status": str(record.get("digital_parity_status") or "untested"),
                "gate_mode": str(record.get("gate_mode") or SPLIT_GATE_RECOMMENDED),
                "override_allowed": True,
                "record": record,
            }
        return {
            "recommended": False,
            "eligibility_state": SPLIT_ELIGIBILITY_UNTESTED,
            "digital_parity_status": SPLIT_ELIGIBILITY_UNTESTED,
            "gate_mode": SPLIT_GATE_RECOMMENDED,
            "override_allowed": True,
            "record": None,
        }

    def add_relationship(
        self,
        source_asset_id: int,
        target_asset_id: int,
        relationship_type: str,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO asset_relationships (
                    source_asset_id,
                    target_asset_id,
                    relationship_type,
                    confidence,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_asset_id,
                    target_asset_id,
                    relationship_type,
                    confidence,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def clear_asset_relationships(self, *, asset_ids: Iterable[int] | None = None) -> int:
        with self._connect() as conn:
            if asset_ids is None:
                cursor = conn.execute("DELETE FROM asset_relationships")
                return int(cursor.rowcount or 0)
            ids = sorted({int(item) for item in asset_ids})
            if not ids:
                return 0
            placeholders = ", ".join("?" for _ in ids)
            cursor = conn.execute(
                f"DELETE FROM asset_relationships WHERE source_asset_id IN ({placeholders}) OR target_asset_id IN ({placeholders})",
                (*ids, *ids),
            )
            return int(cursor.rowcount or 0)

    def list_asset_relationships(
        self,
        *,
        asset_id: int | None = None,
        relationship_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset_id is not None:
            clauses.append("(source_asset_id = ? OR target_asset_id = ?)")
            params.extend([int(asset_id), int(asset_id)])
        if relationship_type:
            clauses.append("relationship_type = ?")
            params.append(str(relationship_type))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM asset_relationships {where} ORDER BY source_asset_id, target_asset_id, id LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = row["metadata_json"]
            results.append(
                {
                    "id": int(row["id"]),
                    "source_asset_id": int(row["source_asset_id"]),
                    "target_asset_id": int(row["target_asset_id"]),
                    "relationship_type": row["relationship_type"],
                    "confidence": row["confidence"],
                    "metadata": metadata,
                }
            )
        return results
