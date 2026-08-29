from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.asset_hub.providers.base import AssetHubError

SEARCH_SESSION_SCHEMA_VERSION = 3
_ACTIVE_STATUSES = {"queued", "running", "stopping"}
_ALLOWED_STATUSES = {"idle", "queued", "running", "stopping", "paused", "stopped", "completed", "failed"}
_ALLOWED_MODES = {"search", "browse"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().casefold()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_mode(value: Any, default: str = "browse") -> str:
    mode = _text(value).casefold()
    return mode if mode in _ALLOWED_MODES else default


def _safe_list(value: Any, *, item_limit: int = 128, total_limit: int = 16) -> list[str]:
    if not isinstance(value, (list, tuple)):
        if isinstance(value, str) and value.strip():
            value = [value]
        else:
            return []
    items: list[str] = []
    for part in value:
        token = _text(part)[:item_limit]
        if token:
            items.append(token)
        if len(items) >= total_limit:
            break
    return items


def _safe_discovery_criteria(value: Any, *, default_provider: str = "civitai", default_mode: str = "browse") -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    provider_id = _text(raw.get("providerId") or raw.get("provider") or default_provider).casefold()[:64] or "civitai"
    mode = _safe_mode(raw.get("mode"), default=default_mode)
    criteria: dict[str, Any] = {
        "providerId": provider_id,
        "mode": mode,
        "query": _text(raw.get("query"))[:512] if mode == "search" else "",
        "assetType": _text(raw.get("assetType") or raw.get("type") or "any")[:64] or "any",
        "providerSort": _text(raw.get("providerSort") or raw.get("sort"))[:64],
        "period": _text(raw.get("period"))[:64],
        # This is a provider transport/safety flag, not a local rating facet.
        "safeContent": _safe_bool(raw.get("safeContent") if "safeContent" in raw else raw.get("safe_content"), True),
    }
    # Older sessions may have used creator as a provider-side discovery term.
    # Preserve it so existing tabs remain reproducible, but new DSV2-02 UI uses
    # resultFilters.creators for local creator refinement instead.
    creator = _text(raw.get("creator"))[:256]
    if creator:
        criteria["creator"] = creator
    limit = raw.get("limit")
    if isinstance(limit, (int, float)):
        criteria["limit"] = max(1, min(int(limit), 500))
    return criteria


def _safe_result_filters(value: Any) -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    output: dict[str, Any] = {}

    keyword_terms = _safe_list(raw.get("keywordTerms"), item_limit=512, total_limit=32)
    legacy_keywords = _text(raw.get("keywords") or raw.get("keyword"))[:512]
    if not keyword_terms and legacy_keywords:
        keyword_terms = [legacy_keywords]
    if keyword_terms:
        output["keywordTerms"] = keyword_terms
        # Keep the legacy aggregate for older callers and persisted sessions.
        output["keywords"] = " ".join(keyword_terms)[:512]
    keyword_mode = _text(raw.get("keywordMode") or "all_words").casefold().replace("-", "_")
    if keyword_mode not in {"all_words", "any_word", "exact_phrase"}:
        keyword_mode = "all_words"
    output["keywordMode"] = keyword_mode

    architectures = _safe_list(raw.get("architectures") or raw.get("baseModels") or raw.get("architecture"), total_limit=32)
    if architectures:
        output["architectures"] = architectures
    asset_kinds = _safe_list(raw.get("assetKinds") or raw.get("assetKind"), total_limit=16)
    if asset_kinds:
        output["assetKinds"] = asset_kinds
    ratings = _safe_list(raw.get("ratings") or raw.get("ratingLevels"), total_limit=16)
    if ratings:
        output["ratings"] = ratings

    rating_basis = _text(raw.get("ratingBasis") or "strictest").casefold().replace("-", "_")
    if rating_basis not in {"model", "author_previews", "strictest"}:
        rating_basis = "strictest"
    output["ratingBasis"] = rating_basis

    for source_key, target_key in (
        ("supportStates", "supportStates"),
        ("libraryStates", "libraryStates"),
        ("previewSources", "previewSources"),
        ("creators", "creators"),
        ("categories", "categories"),
    ):
        values = _safe_list(raw.get(source_key), item_limit=256, total_limit=32)
        if values:
            output[target_key] = values

    # Legacy scalar result filters normalize into the DSV2 multi-value facets.
    support = _text(raw.get("supportFilter"))
    if support and support.casefold() not in {"", "any", "all", "*"} and "supportStates" not in output:
        output["supportStates"] = [support]
    library = _text(raw.get("libraryFilter"))
    if library and library.casefold() not in {"", "any", "all", "*"} and "libraryStates" not in output:
        output["libraryStates"] = [library]
    preview = _text(raw.get("previewFilter"))
    if preview and preview.casefold() not in {"", "any", "all", "*"} and "previewSources" not in output:
        aliases = {"provider": "provider", "local": "local", "both": "both"}
        output["previewSources"] = [aliases.get(preview.casefold(), preview)]
    mature_mode = _text(raw.get("maturePreviewMode") or raw.get("maturePreviews") or "show").casefold()
    if mature_mode not in {"show", "blur", "hide"}:
        mature_mode = "show"
    output["maturePreviewMode"] = mature_mode

    local_sort = _text(raw.get("localSort") or raw.get("sortMode") or "candidate_order").casefold().replace("-", "_")
    if local_sort not in {"candidate_order", "safest", "most_mature", "newest", "title"}:
        local_sort = "candidate_order"
    output["localSort"] = local_sort
    return output


def _legacy_filters_to_discovery_and_results(value: Any, *, default_provider: str = "civitai", default_mode: str = "browse") -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    discovery = _safe_discovery_criteria(raw, default_provider=default_provider, default_mode=default_mode)
    # Legacy creator was discovery-side; keep it there. Legacy architecture,
    # support/library/preview state move into local result filters.
    results = _safe_result_filters(raw)
    if "safeContent" in raw or "safe_content" in raw:
        discovery["safeContent"] = _safe_bool(raw.get("safeContent") if "safeContent" in raw else raw.get("safe_content"), True)
    return discovery, results


def _combined_filters(discovery: Mapping[str, Any] | None, results: Mapping[str, Any] | None) -> dict[str, Any]:
    discovery = dict(discovery or {})
    results = dict(results or {})
    combined: dict[str, Any] = {
        "provider": _text(discovery.get("providerId") or "civitai") or "civitai",
        "query": _text(discovery.get("query")),
        "type": _text(discovery.get("assetType") or "any") or "any",
        "creator": _text(discovery.get("creator")),
        "sort": _text(discovery.get("providerSort")),
        "period": _text(discovery.get("period")),
        "safeContent": bool(discovery.get("safeContent", True)),
        "mode": _safe_mode(discovery.get("mode"), default="browse"),
    }
    if "limit" in discovery:
        try:
            combined["limit"] = max(1, min(int(discovery.get("limit") or 0), 500))
        except (TypeError, ValueError):
            pass
    # Keep enough legacy keys for older JS/builds to read a DSV2 session.
    architectures = _safe_list(results.get("architectures"), total_limit=32)
    if architectures:
        combined["baseModels"] = architectures
    support_states = _safe_list(results.get("supportStates"), total_limit=16)
    combined["supportFilter"] = support_states[0] if len(support_states) == 1 else "any"
    library_states = _safe_list(results.get("libraryStates"), total_limit=16)
    combined["libraryFilter"] = library_states[0] if len(library_states) == 1 else "any"
    preview_sources = _safe_list(results.get("previewSources"), total_limit=16)
    combined["previewFilter"] = preview_sources[0] if len(preview_sources) == 1 else "any"
    return combined


class AssetSearchSessionStore:
    """Persistent Asset Browser search-tab/session metadata.

    Provider payloads live in ``AssetDiscoveryIndex``. This store records only
    the user's search workspace state, counts, cursors, and freshness timestamps.
    Interrupted provider work is recovered as paused/partial on startup so
    reopening IMAGE_GEN never silently resumes network requests.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()
        self.recover_interrupted()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"] or "") for row in rows}

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, ddl: str) -> None:
        if column_name in self._table_columns(connection, table_name):
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS asset_search_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_closed INTEGER NOT NULL DEFAULT 0,
                    partial INTEGER NOT NULL DEFAULT 0,
                    next_cursor TEXT NOT NULL DEFAULT '',
                    result_count INTEGER NOT NULL DEFAULT 0,
                    cached_result_count INTEGER NOT NULL DEFAULT 0,
                    provider_result_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at_unix REAL NOT NULL,
                    updated_at_unix REAL NOT NULL,
                    queued_at_unix REAL NOT NULL DEFAULT 0,
                    started_at_unix REAL NOT NULL DEFAULT 0,
                    stopped_at_unix REAL NOT NULL DEFAULT 0,
                    completed_at_unix REAL NOT NULL DEFAULT 0,
                    last_local_refresh_at_unix REAL NOT NULL DEFAULT 0,
                    last_provider_refresh_at_unix REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_asset_search_sessions_open
                    ON asset_search_sessions(is_closed, updated_at_unix DESC);
                CREATE TABLE IF NOT EXISTS asset_search_session_candidates (
                    session_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    remote_model_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    discovered_at_unix REAL NOT NULL,
                    PRIMARY KEY(session_id, provider_id, remote_model_id)
                );
                CREATE INDEX IF NOT EXISTS idx_asset_search_session_candidates_sequence
                    ON asset_search_session_candidates(session_id, sequence_no ASC);
                """
            )
            self._ensure_column(connection, "asset_search_sessions", "discovery_criteria_json", "TEXT NOT NULL DEFAULT '{}' ")
            self._ensure_column(connection, "asset_search_sessions", "result_filters_json", "TEXT NOT NULL DEFAULT '{}' ")

    @staticmethod
    def _decode_json(value: Any) -> dict[str, Any]:
        try:
            decoded = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}

    def _normalized_from_row(self, row: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
        legacy_filters = self._decode_json(row["filters_json"])
        raw_discovery = self._decode_json(row["discovery_criteria_json"])
        raw_results = self._decode_json(row["result_filters_json"])
        migrated = False

        if not raw_discovery:
            discovery, derived_results = _legacy_filters_to_discovery_and_results(
                legacy_filters,
                default_provider=str(row["provider_id"] or "civitai"),
                default_mode=str(row["mode"] or "browse"),
            )
            results = _safe_result_filters(raw_results) if raw_results else derived_results
            migrated = True
        else:
            discovery = _safe_discovery_criteria(
                raw_discovery,
                default_provider=str(row["provider_id"] or "civitai"),
                default_mode=str(row["mode"] or "browse"),
            )
            results = _safe_result_filters(raw_results)
            if not raw_results and legacy_filters:
                _, results = _legacy_filters_to_discovery_and_results(
                    legacy_filters,
                    default_provider=str(row["provider_id"] or "civitai"),
                    default_mode=str(row["mode"] or "browse"),
                )
                migrated = True

        # DSV2-01 stored safeContent inside resultFilters. DSV2-02 treats it as
        # provider transport state; migrate it without changing the user's choice.
        if "safeContent" not in discovery:
            discovery["safeContent"] = _safe_bool(raw_results.get("safeContent") if raw_results else legacy_filters.get("safeContent"), True)
            migrated = True
        if "safeContent" in results:
            results.pop("safeContent", None)
            migrated = True

        combined = _combined_filters(discovery, results)
        return discovery, results, combined, migrated

    def _persist_split_filters(self, connection: sqlite3.Connection, row: sqlite3.Row, discovery: Mapping[str, Any], results: Mapping[str, Any], combined: Mapping[str, Any]) -> None:
        connection.execute(
            """
            UPDATE asset_search_sessions
            SET provider_id=?, mode=?, filters_json=?, discovery_criteria_json=?, result_filters_json=?, updated_at_unix=?
            WHERE session_id=?
            """,
            (
                _text(discovery.get("providerId") or row["provider_id"] or "civitai").casefold()[:64] or "civitai",
                _safe_mode(discovery.get("mode"), default=str(row["mode"] or "browse")),
                json.dumps(dict(combined), ensure_ascii=False, separators=(",", ":")),
                json.dumps(dict(discovery), ensure_ascii=False, separators=(",", ":")),
                json.dumps(dict(results), ensure_ascii=False, separators=(",", ":")),
                time.time(),
                str(row["session_id"]),
            ),
        )

    def _candidate_count_connection(self, connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM asset_search_session_candidates WHERE session_id=?",
            (_text(session_id),),
        ).fetchone()
        return int(row["total"] or 0) if row else 0

    def _row_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        discovery, results, combined, migrated = self._normalized_from_row(row)
        with self._lock, self._connect() as connection:
            if migrated:
                self._persist_split_filters(connection, row, discovery, results, combined)
            candidate_count = self._candidate_count_connection(connection, str(row["session_id"]))
        return {
            "schemaVersion": SEARCH_SESSION_SCHEMA_VERSION,
            "sessionId": str(row["session_id"]),
            "title": str(row["title"] or "Search"),
            "providerId": str(row["provider_id"] or discovery.get("providerId") or "civitai"),
            "mode": str(row["mode"] or discovery.get("mode") or "browse"),
            "filters": dict(combined),
            "discoveryCriteria": dict(discovery),
            "resultFilters": dict(results),
            "providerState": {
                "status": str(row["status"] or "idle"),
                "nextCursor": str(row["next_cursor"] or "") or None,
                "partial": bool(row["partial"]),
                "errorMessage": str(row["error_message"] or "") or None,
            },
            "candidateCount": candidate_count,
            "status": str(row["status"] or "idle"),
            "closed": bool(row["is_closed"]),
            "partial": bool(row["partial"]),
            "nextCursor": str(row["next_cursor"] or "") or None,
            "resultCount": int(row["result_count"] or 0),
            "cachedResultCount": int(row["cached_result_count"] or 0),
            "providerResultCount": int(row["provider_result_count"] or 0),
            "errorMessage": str(row["error_message"] or "") or None,
            "createdAtUnix": float(row["created_at_unix"] or 0.0),
            "updatedAtUnix": float(row["updated_at_unix"] or 0.0),
            "queuedAtUnix": float(row["queued_at_unix"] or 0.0),
            "startedAtUnix": float(row["started_at_unix"] or 0.0),
            "stoppedAtUnix": float(row["stopped_at_unix"] or 0.0),
            "completedAtUnix": float(row["completed_at_unix"] or 0.0),
            "lastLocalRefreshAtUnix": float(row["last_local_refresh_at_unix"] or 0.0),
            "lastProviderRefreshAtUnix": float(row["last_provider_refresh_at_unix"] or 0.0),
        }

    def recover_interrupted(self) -> int:
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE asset_search_sessions
                SET status='paused', partial=1, stopped_at_unix=?, updated_at_unix=?,
                    error_message=''
                WHERE is_closed=0 AND status IN ('queued','running','stopping')
                """,
                (now, now),
            )
            return int(cursor.rowcount or 0)

    def create(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = dict(payload or {})
        now = time.time()
        session_id = str(uuid.uuid4())
        fallback_mode = _safe_mode(body.get("mode"), default="browse")
        discovery, results = _legacy_filters_to_discovery_and_results(
            body.get("filters"),
            default_provider=_text(body.get("providerId")).casefold()[:64] or "civitai",
            default_mode=fallback_mode,
        )
        if "discoveryCriteria" in body:
            discovery = _safe_discovery_criteria(
                body.get("discoveryCriteria"),
                default_provider=_text(body.get("providerId") or discovery.get("providerId")).casefold()[:64] or "civitai",
                default_mode=fallback_mode,
            )
        if "resultFilters" in body:
            results = _safe_result_filters(body.get("resultFilters"))
        mode = _safe_mode(body.get("mode") or discovery.get("mode"), default="browse")
        discovery["mode"] = mode
        provider = _text(body.get("providerId") or discovery.get("providerId") or "civitai").casefold()[:64] or "civitai"
        discovery["providerId"] = provider
        combined = _combined_filters(discovery, results)
        title = _text(body.get("title"))[:120] or "New search"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO asset_search_sessions(
                    session_id,title,provider_id,mode,filters_json,discovery_criteria_json,result_filters_json,status,
                    created_at_unix,updated_at_unix
                ) VALUES(?,?,?,?,?,?,?,'idle',?,?)
                """,
                (
                    session_id,
                    title,
                    provider,
                    mode,
                    json.dumps(combined, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(discovery, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(results, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )
        return self.get(session_id)

    def get(self, session_id: str) -> dict[str, Any]:
        selected = _text(session_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM asset_search_sessions WHERE session_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise AssetHubError("search_session_not_found", "Asset Browser search session was not found.", status_code=404)
        return self._row_payload(row)

    def list(self, *, include_closed: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM asset_search_sessions"
        args: list[Any] = []
        if not include_closed:
            sql += " WHERE is_closed=0"
        sql += " ORDER BY updated_at_unix DESC LIMIT ?"
        args.append(max(1, min(int(limit or 100), 500)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, tuple(args)).fetchall()
        return [self._row_payload(row) for row in rows]

    def update(self, session_id: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        current = self.get(session_id)
        body = dict(payload or {})
        now = time.time()
        title = _text(body.get("title"))[:120] if "title" in body else current["title"]
        title = title or "Search"
        mode = _safe_mode(body.get("mode"), default=current["mode"]) if "mode" in body else current["mode"]

        discovery = dict(current.get("discoveryCriteria") or {})
        results = dict(current.get("resultFilters") or {})
        if "filters" in body:
            discovery, results = _legacy_filters_to_discovery_and_results(
                body.get("filters"),
                default_provider=current["providerId"],
                default_mode=mode,
            )
        if "discoveryCriteria" in body:
            discovery = _safe_discovery_criteria(body.get("discoveryCriteria"), default_provider=current["providerId"], default_mode=mode)
        if "resultFilters" in body:
            results = _safe_result_filters(body.get("resultFilters"))
        discovery["mode"] = mode
        provider = _text(body.get("providerId") or discovery.get("providerId") or current["providerId"]).casefold()[:64] or current["providerId"]
        discovery["providerId"] = provider
        combined = _combined_filters(discovery, results)

        status = _text(body.get("status")).casefold() if "status" in body else current["status"]
        if status not in _ALLOWED_STATUSES:
            status = current["status"]
        partial = bool(body.get("partial")) if "partial" in body else bool(current["partial"])
        next_cursor = _text(body.get("nextCursor")) if "nextCursor" in body else _text(current.get("nextCursor"))
        result_count = max(0, int(body.get("resultCount") if "resultCount" in body else current["resultCount"]))
        cached_count = max(0, int(body.get("cachedResultCount") if "cachedResultCount" in body else current["cachedResultCount"]))
        provider_count = max(0, int(body.get("providerResultCount") if "providerResultCount" in body else current["providerResultCount"]))
        error_message = _text(body.get("errorMessage"))[:2000] if "errorMessage" in body else _text(current.get("errorMessage"))
        closed = bool(body.get("closed")) if "closed" in body else bool(current["closed"])

        queued_at = float(current["queuedAtUnix"] or 0.0)
        started_at = float(current["startedAtUnix"] or 0.0)
        stopped_at = float(current["stoppedAtUnix"] or 0.0)
        completed_at = float(current["completedAtUnix"] or 0.0)
        local_at = float(current["lastLocalRefreshAtUnix"] or 0.0)
        provider_at = float(current["lastProviderRefreshAtUnix"] or 0.0)
        if status == "queued" and current["status"] != "queued":
            queued_at = now
        if status == "running" and current["status"] != "running":
            started_at = now
        if status == "stopped" and current["status"] != "stopped":
            stopped_at = now
        if status == "completed" and current["status"] != "completed":
            completed_at = now
            partial = False
            next_cursor = ""
        if bool(body.get("touchLocal")):
            local_at = now
        if bool(body.get("touchProvider")):
            provider_at = now

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE asset_search_sessions SET
                    title=?, provider_id=?, mode=?, filters_json=?, discovery_criteria_json=?, result_filters_json=?, status=?, is_closed=?, partial=?,
                    next_cursor=?, result_count=?, cached_result_count=?, provider_result_count=?,
                    error_message=?, updated_at_unix=?, queued_at_unix=?, started_at_unix=?,
                    stopped_at_unix=?, completed_at_unix=?, last_local_refresh_at_unix=?,
                    last_provider_refresh_at_unix=?
                WHERE session_id=?
                """,
                (
                    title,
                    provider,
                    mode,
                    json.dumps(combined, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(discovery, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(results, ensure_ascii=False, separators=(",", ":")),
                    status,
                    1 if closed else 0,
                    1 if partial else 0,
                    next_cursor,
                    result_count,
                    cached_count,
                    provider_count,
                    error_message,
                    now,
                    queued_at,
                    started_at,
                    stopped_at,
                    completed_at,
                    local_at,
                    provider_at,
                    _text(session_id),
                ),
            )
            if bool(body.get("resetCandidates")):
                connection.execute("DELETE FROM asset_search_session_candidates WHERE session_id=?", (_text(session_id),))
        return self.get(session_id)

    def pause(self, session_id: str) -> dict[str, Any]:
        return self.update(session_id, {"status": "paused", "partial": True})

    def resume(self, session_id: str) -> dict[str, Any]:
        return self.update(session_id, {"status": "queued", "partial": True, "errorMessage": ""})

    def stop(self, session_id: str) -> dict[str, Any]:
        return self.update(session_id, {"status": "stopped", "partial": True})

    def close(self, session_id: str) -> dict[str, Any]:
        current = self.get(session_id)
        payload: dict[str, Any] = {"closed": True}
        if current["status"] in _ACTIVE_STATUSES:
            payload.update({"status": "stopped", "partial": True})
        return self.update(session_id, payload)

    def clear_candidates(self, session_id: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM asset_search_session_candidates WHERE session_id=?",
                (_text(session_id),),
            )
            return int(cursor.rowcount or 0)

    def candidate_count(self, session_id: str) -> int:
        with self._lock, self._connect() as connection:
            return self._candidate_count_connection(connection, session_id)

    def add_candidates(self, session_id: str, provider_id: str, remote_model_ids: list[str] | tuple[str, ...], *, source_kind: str = "provider_search") -> int:
        selected_session_id = _text(session_id)
        selected_provider = _text(provider_id).casefold()[:64] or "civitai"
        model_ids = [token for token in (_text(item) for item in (remote_model_ids or ())) if token]
        if not selected_session_id or not model_ids:
            return 0
        now = time.time()
        inserted = 0
        with self._lock, self._connect() as connection:
            start_row = connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) AS maximum FROM asset_search_session_candidates WHERE session_id=?",
                (selected_session_id,),
            ).fetchone()
            sequence = int(start_row["maximum"] or 0) if start_row else 0
            seen: set[str] = set()
            for model_id in model_ids:
                if model_id in seen:
                    continue
                seen.add(model_id)
                existing = connection.execute(
                    "SELECT 1 FROM asset_search_session_candidates WHERE session_id=? AND provider_id=? AND remote_model_id=?",
                    (selected_session_id, selected_provider, model_id),
                ).fetchone()
                if existing is not None:
                    continue
                sequence += 1
                connection.execute(
                    """
                    INSERT INTO asset_search_session_candidates(
                        session_id, provider_id, remote_model_id, sequence_no, source_kind, discovered_at_unix
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (selected_session_id, selected_provider, model_id, sequence, _text(source_kind)[:64] or "provider_search", now),
                )
                inserted += 1
        return inserted
