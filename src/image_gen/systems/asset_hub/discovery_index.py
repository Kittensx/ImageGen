from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from image_gen.systems.asset_hub.policy import ArchitectureCompatibilityPolicy, normalize_architecture, normalize_asset_kind
from image_gen.systems.asset_hub.service import LocalPresenceResolver

DISCOVERY_INDEX_SCHEMA_VERSION = 4


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).casefold()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _architectures(item: Mapping[str, Any]) -> tuple[str, ...]:
    output: list[str] = []
    for version in _list(item.get("versions")):
        if not isinstance(version, Mapping):
            continue
        architecture = normalize_architecture(version.get("architecture") or version.get("baseModel"))
        if architecture and architecture not in output:
            output.append(architecture)
    return tuple(output)


_RATING_ORDER = {"Unknown": -1, "PG": 0, "PG13": 1, "R": 2, "X": 3, "XXX": 4, "Blocked": 5}


def _rating_levels(value: Any) -> tuple[str, ...]:
    data = _mapping(value)
    levels = tuple(str(level or "").strip() for level in _list(data.get("levels")) if str(level or "").strip())
    state = _lower(data.get("state"))
    known = tuple(level for level in levels if level in _RATING_ORDER and level != "Unknown")
    if state == "known" and known:
        return known
    if known:
        return known
    return ("Unknown",)


def _highest_rating(levels: Iterable[str]) -> str:
    normalized = [str(level or "").strip() for level in levels if str(level or "").strip() in _RATING_ORDER]
    known = [level for level in normalized if level != "Unknown"]
    if known:
        return max(known, key=lambda level: _RATING_ORDER[level])
    return "Unknown"


def _rating_values(item: Mapping[str, Any]) -> tuple[str, str, str]:
    model_rating = _highest_rating(_rating_levels(item.get("maturity")))
    preview_levels: list[str] = []
    for version in _list(item.get("versions")):
        if not isinstance(version, Mapping):
            continue
        author = _mapping(version.get("authorPreviewMaturity"))
        items = _list(author.get("items"))
        if items:
            for preview in items:
                if isinstance(preview, Mapping):
                    preview_levels.extend(_rating_levels(preview.get("maturity")))
        else:
            for preview in _list(version.get("previews")):
                if isinstance(preview, Mapping):
                    preview_levels.extend(_rating_levels(preview.get("maturity")))
    preview_rating = _highest_rating(preview_levels)
    strictest = _highest_rating((model_rating, preview_rating))
    return model_rating, preview_rating, strictest


def _search_blob(item: Mapping[str, Any]) -> str:
    values: list[str] = [
        _text(item.get("name")),
        _text(item.get("creator")),
        _text(item.get("description")),
        *(_text(value) for value in _list(item.get("tags"))),
    ]
    for version in _list(item.get("versions")):
        if not isinstance(version, Mapping):
            continue
        values.append(_text(version.get("name")))
        values.extend(_text(value) for value in _list(version.get("trainedWords")))
    return "\n".join(value.casefold() for value in values if value)


def _provider_category(item: Mapping[str, Any]) -> str:
    for key in ("category", "categoryName", "providerCategory"):
        value = _text(item.get(key))
        if value:
            return value
    return ""


def _rank(item: Mapping[str, Any], query: str) -> int:
    phrase = " ".join(_lower(query).split())
    if not phrase:
        return 0
    words = tuple(dict.fromkeys(part for part in phrase.split() if part))
    name = _lower(item.get("name"))
    creator = _lower(item.get("creator"))
    description = _lower(item.get("description"))
    tags = tuple(_lower(value) for value in _list(item.get("tags")))
    score = 0
    if name == phrase:
        score += 1200
    elif name.startswith(phrase):
        score += 900
    elif phrase in name:
        score += 760
    if any(tag == phrase for tag in tags):
        score += 520
    elif any(phrase in tag for tag in tags):
        score += 420
    if phrase in creator:
        score += 260
    if phrase in description:
        score += 180
    for word in words:
        if word in name:
            score += 130
        if any(word in tag for tag in tags):
            score += 80
        if word in description:
            score += 24
    return score


class AssetDiscoveryIndex:
    """Persistent, provider-neutral search acceleration for Asset Browser.

    Provider responses are cached as normalized public model summaries. Current
    library presence is *not* trusted from the cache: every read overlays the
    live local catalog through ``LocalPresenceResolver``. This makes prior
    provider searches immediately reusable while installed assets and user
    preview replacements stay current.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        presence: LocalPresenceResolver,
        policy: ArchitectureCompatibilityPolicy | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.presence = presence
        self.policy = policy or ArchitectureCompatibilityPolicy()
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS discovery_models (
                    provider_id TEXT NOT NULL,
                    remote_model_id TEXT NOT NULL,
                    asset_kind TEXT NOT NULL,
                    name_text TEXT NOT NULL,
                    creator_text TEXT NOT NULL,
                    description_text TEXT NOT NULL,
                    tags_text TEXT NOT NULL,
                    architectures_text TEXT NOT NULL,
                    nsfw INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    indexed_at_unix REAL NOT NULL,
                    first_seen_at_unix REAL NOT NULL DEFAULT 0,
                    last_refreshed_at_unix REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (provider_id, remote_model_id)
                );
                CREATE INDEX IF NOT EXISTS idx_discovery_models_kind
                    ON discovery_models(provider_id, asset_kind, indexed_at_unix DESC);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_name
                    ON discovery_models(provider_id, name_text);
                CREATE TABLE IF NOT EXISTS discovery_search_snapshots (
                    search_key TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    model_ids_json TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    is_complete INTEGER NOT NULL DEFAULT 0,
                    updated_at_unix REAL NOT NULL,
                    last_used_at_unix REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_discovery_search_snapshots_provider
                    ON discovery_search_snapshots(provider_id, updated_at_unix DESC);
                CREATE TABLE IF NOT EXISTS discovery_model_architectures (
                    provider_id TEXT NOT NULL,
                    remote_model_id TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    PRIMARY KEY(provider_id, remote_model_id, architecture)
                );
                CREATE INDEX IF NOT EXISTS idx_discovery_model_architectures_value
                    ON discovery_model_architectures(provider_id, architecture, remote_model_id);
                CREATE TABLE IF NOT EXISTS discovery_index_meta (
                    meta_key TEXT PRIMARY KEY,
                    meta_value TEXT NOT NULL
                );
                """
            )
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(discovery_models)").fetchall()}
            additions = {
                "first_seen_at_unix": "REAL NOT NULL DEFAULT 0",
                "last_refreshed_at_unix": "REAL NOT NULL DEFAULT 0",
                "search_text": "TEXT NOT NULL DEFAULT ''",
                "creator_value": "TEXT NOT NULL DEFAULT ''",
                "support_state": "TEXT NOT NULL DEFAULT 'unknown'",
                "model_rating": "TEXT NOT NULL DEFAULT 'Unknown'",
                "preview_rating": "TEXT NOT NULL DEFAULT 'Unknown'",
                "strictest_rating": "TEXT NOT NULL DEFAULT 'Unknown'",
                "provider_preview": "INTEGER NOT NULL DEFAULT 0",
                "provider_category": "TEXT NOT NULL DEFAULT ''",
            }
            for column_name, ddl in additions.items():
                if column_name not in columns:
                    connection.execute(f"ALTER TABLE discovery_models ADD COLUMN {column_name} {ddl}")
            connection.execute(
                """
                UPDATE discovery_models
                SET first_seen_at_unix=CASE WHEN first_seen_at_unix <= 0 THEN indexed_at_unix ELSE first_seen_at_unix END,
                    last_refreshed_at_unix=CASE WHEN last_refreshed_at_unix <= 0 THEN indexed_at_unix ELSE last_refreshed_at_unix END
                """
            )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_discovery_models_creator_value
                    ON discovery_models(provider_id, creator_value);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_support
                    ON discovery_models(provider_id, support_state);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_model_rating
                    ON discovery_models(provider_id, model_rating);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_preview_rating
                    ON discovery_models(provider_id, preview_rating);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_strictest_rating
                    ON discovery_models(provider_id, strictest_rating);
                CREATE INDEX IF NOT EXISTS idx_discovery_models_provider_preview
                    ON discovery_models(provider_id, provider_preview);
                """
            )
            self._backfill_dsv2_facets(connection)

    def _backfill_dsv2_facets(self, connection: sqlite3.Connection) -> None:
        marker = connection.execute(
            "SELECT meta_value FROM discovery_index_meta WHERE meta_key='dsv2_facets_v1'"
        ).fetchone()
        if marker is not None and str(marker["meta_value"] or "") == "1":
            return
        rows = connection.execute(
            "SELECT provider_id, remote_model_id, payload_json FROM discovery_models"
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, Mapping):
                payload = {}
            model_rating, preview_rating, strictest_rating = _rating_values(payload)
            architectures = _architectures(payload)
            connection.execute(
                """
                UPDATE discovery_models
                SET search_text=?, creator_value=?, support_state=?, model_rating=?,
                    preview_rating=?, strictest_rating=?, provider_preview=?, provider_category=?
                WHERE provider_id=? AND remote_model_id=?
                """,
                (
                    _search_blob(payload),
                    _text(payload.get("creator")),
                    _lower(payload.get("supportState")) or "unknown",
                    model_rating,
                    preview_rating,
                    strictest_rating,
                    1 if _text(payload.get("providerPreviewUrl")) else 0,
                    _provider_category(payload),
                    str(row["provider_id"] or ""),
                    str(row["remote_model_id"] or ""),
                ),
            )
            connection.execute(
                "DELETE FROM discovery_model_architectures WHERE provider_id=? AND remote_model_id=?",
                (str(row["provider_id"] or ""), str(row["remote_model_id"] or "")),
            )
            if architectures:
                connection.executemany(
                    "INSERT OR IGNORE INTO discovery_model_architectures(provider_id,remote_model_id,architecture) VALUES(?,?,?)",
                    [(str(row["provider_id"] or ""), str(row["remote_model_id"] or ""), architecture) for architecture in architectures],
                )
        connection.execute(
            "INSERT OR REPLACE INTO discovery_index_meta(meta_key,meta_value) VALUES('dsv2_facets_v1','1')"
        )

    @staticmethod
    def _snapshot_filters(
        *,
        provider_id: str,
        query: str = "",
        creator: str = "",
        asset_kind: str = "any",
        base_models: tuple[str, ...] = (),
        safe_content: bool = True,
        support_filter: str = "any",
        library_filter: str = "any",
        preview_filter: str = "any",
        sort: str = "",
        period: str = "",
        mode: str = "search",
    ) -> dict[str, Any]:
        """Normalize only *primary discovery* semantics into a snapshot key.

        Secondary result filters intentionally do not participate in persistent
        replay membership. DSV2-01 separates the first-pass provider discovery
        from later local refinement so changes like support/library filters do
        not invalidate or fragment the same provider search snapshot.
        """
        kind = normalize_asset_kind(asset_kind) if _lower(asset_kind) not in {"", "any", "all", "*"} else "any"
        return {
            "provider": _lower(provider_id),
            "query": " ".join(_lower(query).split()) if _lower(mode) == "search" else "",
            "creator": " ".join(_lower(creator).split()),
            "assetKind": kind,
            "sort": " ".join(_lower(sort).split()),
            "period": " ".join(_lower(period).split()),
            "mode": "search" if _lower(mode) == "search" else "browse",
        }

    @classmethod
    def _snapshot_key(cls, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        filters = cls._snapshot_filters(**kwargs)
        serialized = json.dumps(filters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), filters

    def remember_search_page(
        self,
        *,
        provider_id: str,
        items: Iterable[Mapping[str, Any]],
        query: str = "",
        creator: str = "",
        asset_kind: str = "any",
        base_models: tuple[str, ...] = (),
        safe_content: bool = True,
        support_filter: str = "any",
        library_filter: str = "any",
        preview_filter: str = "any",
        sort: str = "",
        period: str = "",
        mode: str = "search",
        first_page: bool = False,
        complete: bool = False,
    ) -> int:
        """Persist provider result membership/order for instant repeated searches.

        A refresh never erases the older snapshot before replacement results are
        available. The first refreshed page is promoted to the front and later
        pages append. This keeps prior searches immediately replayable even if
        ImageGen or the browser was restarted.
        """
        provider = _lower(provider_id)
        if not provider:
            return 0
        incoming_ids = list(dict.fromkeys(
            _text(item.get("remoteModelId"))
            for item in items
            if isinstance(item, Mapping) and _text(item.get("remoteModelId"))
        ))
        search_key, filters = self._snapshot_key(
            provider_id=provider,
            query=query,
            creator=creator,
            asset_kind=asset_kind,
            base_models=base_models,
            safe_content=safe_content,
            support_filter=support_filter,
            library_filter=library_filter,
            preview_filter=preview_filter,
            sort=sort,
            period=period,
            mode=mode,
        )
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT model_ids_json, is_complete FROM discovery_search_snapshots WHERE search_key=?",
                (search_key,),
            ).fetchone()
            existing_ids: list[str] = []
            prior_complete = False
            if row is not None:
                try:
                    parsed = json.loads(str(row["model_ids_json"] or "[]"))
                    existing_ids = [_text(value) for value in parsed if _text(value)] if isinstance(parsed, list) else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    existing_ids = []
                prior_complete = bool(row["is_complete"])

            incoming_set = set(incoming_ids)
            if first_page:
                merged_ids = incoming_ids + [model_id for model_id in existing_ids if model_id not in incoming_set]
            else:
                merged_ids = list(existing_ids)
                known = set(merged_ids)
                for model_id in incoming_ids:
                    if model_id not in known:
                        merged_ids.append(model_id)
                        known.add(model_id)
            is_complete = bool(complete or (prior_complete and not incoming_ids))
            connection.execute(
                """
                INSERT INTO discovery_search_snapshots(
                    search_key, provider_id, filters_json, model_ids_json, result_count,
                    is_complete, updated_at_unix, last_used_at_unix
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(search_key) DO UPDATE SET
                    provider_id=excluded.provider_id,
                    filters_json=excluded.filters_json,
                    model_ids_json=excluded.model_ids_json,
                    result_count=excluded.result_count,
                    is_complete=excluded.is_complete,
                    updated_at_unix=excluded.updated_at_unix
                """,
                (
                    search_key,
                    provider,
                    json.dumps(filters, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(merged_ids, ensure_ascii=False, separators=(",", ":")),
                    len(merged_ids),
                    1 if is_complete else 0,
                    now,
                    now,
                ),
            )
        return len(merged_ids)

    def _snapshot_record(self, **kwargs: Any) -> tuple[list[str], bool] | None:
        search_key, _filters = self._snapshot_key(**kwargs)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT model_ids_json, is_complete FROM discovery_search_snapshots WHERE search_key=?",
                (search_key,),
            ).fetchone()
            if row is None:
                return None
            try:
                parsed = json.loads(str(row["model_ids_json"] or "[]"))
                model_ids = [_text(value) for value in parsed if _text(value)] if isinstance(parsed, list) else []
            except (TypeError, ValueError, json.JSONDecodeError):
                model_ids = []
            connection.execute(
                "UPDATE discovery_search_snapshots SET last_used_at_unix=? WHERE search_key=?",
                (time.time(), search_key),
            )
        return model_ids, bool(row["is_complete"])

    @staticmethod
    def _row_item(row: sqlite3.Row) -> dict[str, Any] | None:
        try:
            payload = json.loads(str(row["payload_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        item = dict(payload)
        item["indexSource"] = "provider_cache"
        item["indexedAtUnix"] = float(row["indexed_at_unix"] or 0.0)
        item["firstSeenAtUnix"] = float(row["first_seen_at_unix"] or row["indexed_at_unix"] or 0.0)
        item["lastRefreshedAtUnix"] = float(row["last_refreshed_at_unix"] or row["indexed_at_unix"] or 0.0)
        return item

    @staticmethod
    def _merge_payload(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
        old = dict(existing)
        new = dict(incoming)
        old_versions = {str(item.get("remoteVersionId") or ""): dict(item) for item in _list(old.get("versions")) if isinstance(item, Mapping)}
        for raw in _list(new.get("versions")):
            if not isinstance(raw, Mapping):
                continue
            version = dict(raw)
            version_id = str(version.get("remoteVersionId") or "")
            if not version_id:
                continue
            prior = old_versions.get(version_id, {})
            # Prefer the newest response, but retain files/previews if a filtered
            # provider page omitted them.
            merged = {**prior, **version}
            if not _list(merged.get("files")) and _list(prior.get("files")):
                merged["files"] = prior["files"]
            if not _list(merged.get("previews")) and _list(prior.get("previews")):
                merged["previews"] = prior["previews"]
            old_versions[version_id] = merged
        merged_model = {**old, **new}
        merged_model["versions"] = list(old_versions.values())
        if not _text(merged_model.get("providerPreviewUrl")):
            merged_model["providerPreviewUrl"] = _text(old.get("providerPreviewUrl"))
        return merged_model

    def ingest_items(self, provider_id: str, items: Iterable[Mapping[str, Any]]) -> int:
        provider = _lower(provider_id)
        if not provider:
            return 0
        now = time.time()
        count = 0
        with self._lock, self._connect() as connection:
            for raw in items:
                item = dict(raw) if isinstance(raw, Mapping) else {}
                model_id = _text(item.get("remoteModelId"))
                if not model_id:
                    continue
                prior_row = connection.execute(
                    "SELECT payload_json, first_seen_at_unix FROM discovery_models WHERE provider_id=? AND remote_model_id=?",
                    (provider, model_id),
                ).fetchone()
                prior: dict[str, Any] = {}
                if prior_row is not None:
                    try:
                        parsed = json.loads(str(prior_row["payload_json"] or "{}"))
                        prior = dict(parsed) if isinstance(parsed, Mapping) else {}
                    except (TypeError, ValueError, json.JSONDecodeError):
                        prior = {}
                merged = self._merge_payload(prior, item)
                first_seen = float(prior_row["first_seen_at_unix"] or 0.0) if prior_row is not None else 0.0
                if first_seen <= 0:
                    first_seen = now
                tags_text = "\n".join(_text(value).casefold() for value in _list(merged.get("tags")) if _text(value))
                architectures = _architectures(merged)
                arch_text = "|" + "|".join(architectures) + "|"
                model_rating, preview_rating, strictest_rating = _rating_values(merged)
                connection.execute(
                    """
                    INSERT INTO discovery_models(
                        provider_id, remote_model_id, asset_kind, name_text,
                        creator_text, description_text, tags_text,
                        architectures_text, nsfw, payload_json, indexed_at_unix,
                        first_seen_at_unix, last_refreshed_at_unix, search_text,
                        creator_value, support_state, model_rating, preview_rating,
                        strictest_rating, provider_preview, provider_category
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(provider_id, remote_model_id) DO UPDATE SET
                        asset_kind=excluded.asset_kind,
                        name_text=excluded.name_text,
                        creator_text=excluded.creator_text,
                        description_text=excluded.description_text,
                        tags_text=excluded.tags_text,
                        architectures_text=excluded.architectures_text,
                        nsfw=excluded.nsfw,
                        payload_json=excluded.payload_json,
                        indexed_at_unix=excluded.indexed_at_unix,
                        last_refreshed_at_unix=excluded.last_refreshed_at_unix,
                        search_text=excluded.search_text,
                        creator_value=excluded.creator_value,
                        support_state=excluded.support_state,
                        model_rating=excluded.model_rating,
                        preview_rating=excluded.preview_rating,
                        strictest_rating=excluded.strictest_rating,
                        provider_preview=excluded.provider_preview,
                        provider_category=excluded.provider_category
                    """,
                    (
                        provider,
                        model_id,
                        normalize_asset_kind(merged.get("assetKind")),
                        _lower(merged.get("name")),
                        _lower(merged.get("creator")),
                        _lower(merged.get("description")),
                        tags_text,
                        arch_text,
                        1 if bool(merged.get("nsfw")) else 0,
                        json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                        now,
                        first_seen,
                        now,
                        _search_blob(merged),
                        _text(merged.get("creator")),
                        _lower(merged.get("supportState")) or "unknown",
                        model_rating,
                        preview_rating,
                        strictest_rating,
                        1 if _text(merged.get("providerPreviewUrl")) else 0,
                        _provider_category(merged),
                    ),
                )
                connection.execute(
                    "DELETE FROM discovery_model_architectures WHERE provider_id=? AND remote_model_id=?",
                    (provider, model_id),
                )
                if architectures:
                    connection.executemany(
                        "INSERT OR IGNORE INTO discovery_model_architectures(provider_id,remote_model_id,architecture) VALUES(?,?,?)",
                        [(provider, model_id, architecture) for architecture in architectures],
                    )
                count += 1
        return count

    def ingest_model(self, provider_id: str, model: Mapping[str, Any]) -> int:
        return self.ingest_items(provider_id, [model])

    def _local_summaries(
        self,
        provider_id: str,
        *,
        records: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        source_records = tuple(records) if records is not None else self.presence.provider_linked_records(provider_id)
        for record in source_records:
            model_id = _text(record.get("model_id"))
            if not model_id:
                continue
            kind = normalize_asset_kind(record.get("local_asset_type"))
            architecture = normalize_architecture(record.get("local_base_model"))
            if kind == "upscaler":
                support = "supported"
            elif kind != "unknown" and architecture and architecture in self.policy.supported_architectures(kind):
                support = "supported"
            elif kind != "unknown" and architecture:
                support = "unsupported"
            else:
                support = "unknown"
            version_id = _text(record.get("version_id"))
            file_id = _text(record.get("file_id"))
            summary = grouped.setdefault(model_id, {
                "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
                "providerId": provider_id,
                "remoteModelId": model_id,
                "name": _text(record.get("local_name")) or f"Provider model {model_id}",
                "assetKind": kind,
                "providerType": _text(record.get("local_provider_type")) or kind,
                "creator": _text(record.get("local_creator")),
                "description": _text(record.get("local_description")),
                "tags": list(record.get("local_tags") or ()),
                "nsfw": False,
                "versions": [],
                "supportState": support,
                "supportReason": "Compatibility inferred from the current local asset metadata.",
                "searchRank": 0,
                "searchMatches": ["local library"],
                "libraryStatus": "installed",
                "localAssetId": _text(record.get("local_asset_id")) or None,
                "localAssetType": kind or None,
                "providerPreviewUrl": _text(record.get("provider_preview_url")) or None,
                "localPreviewUrl": _text(record.get("local_preview_url")) or None,
                "localPreviewSource": _text(record.get("local_preview_source")) or None,
                "indexSource": "local_library",
            })
            if not summary.get("providerPreviewUrl") and _text(record.get("provider_preview_url")):
                summary["providerPreviewUrl"] = _text(record.get("provider_preview_url"))
            if not summary.get("localPreviewUrl") and _text(record.get("local_preview_url")):
                summary["localPreviewUrl"] = _text(record.get("local_preview_url"))
                summary["localPreviewSource"] = _text(record.get("local_preview_source")) or None
            if version_id:
                version = next((v for v in summary["versions"] if str(v.get("remoteVersionId")) == version_id), None)
                if version is None:
                    version = {
                        "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
                        "providerId": provider_id,
                        "remoteModelId": model_id,
                        "remoteVersionId": version_id,
                        "name": _text(record.get("local_version_name")) or f"Version {version_id}",
                        "baseModel": _text(record.get("local_base_model")),
                        "architecture": architecture,
                        "description": "",
                        "supportState": support,
                        "supportReason": "Compatibility inferred from the current local asset metadata.",
                        "trainedWords": [],
                        "publishedAt": "",
                        "updatedAt": "",
                        "files": [],
                        "previews": ([{"schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION, "url": _text(record.get("provider_preview_url")), "kind": "image"}]
                                     if _text(record.get("provider_preview_url")) else []),
                        "stats": {},
                        "libraryStatus": "installed",
                        "localAssetId": _text(record.get("local_asset_id")) or None,
                        "localAssetType": kind or None,
                    }
                    summary["versions"].append(version)
                if file_id and not any(str(f.get("remoteFileId")) == file_id for f in version.get("files", [])):
                    version["files"].append({
                        "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
                        "providerId": provider_id,
                        "remoteModelId": model_id,
                        "remoteVersionId": version_id,
                        "remoteFileId": file_id,
                        "fileName": _text(record.get("local_file_name")),
                        "fileType": "",
                        "format": "",
                        "sizeBytes": int(_text(record.get("local_size_bytes")) or 0),
                        "baseModel": _text(record.get("local_base_model")),
                        "architecture": architecture,
                        "trainedWords": [],
                        "hashes": {"SHA256": _text(record.get("sha256"))} if _text(record.get("sha256")) else {},
                        "primary": len(version["files"]) == 0,
                        "libraryStatus": "installed",
                        "localAssetId": _text(record.get("local_asset_id")) or None,
                        "localAssetType": kind or None,
                    })
        return list(grouped.values())

    @staticmethod
    def _overlay_local(
        item: dict[str, Any],
        overlay: Mapping[str, Any],
        installed_records: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        records = tuple(record for record in installed_records if isinstance(record, Mapping))
        if not overlay and not records:
            item.setdefault("libraryStatus", "not_installed")
            item.setdefault("localPreviewUrl", None)
            item.setdefault("localPreviewSource", None)
            for version in _list(item.get("versions")):
                if isinstance(version, dict):
                    version.setdefault("libraryStatus", "not_installed")
                    for provider_file in _list(version.get("files")):
                        if isinstance(provider_file, dict):
                            provider_file.setdefault("libraryStatus", "not_installed")
            return item

        item["libraryStatus"] = "installed"
        item["localAssetId"] = _text(overlay.get("local_asset_id")) or item.get("localAssetId")
        item["localAssetType"] = _text(overlay.get("local_asset_type")) or item.get("localAssetType")
        item["localPreviewUrl"] = _text(overlay.get("local_preview_url")) or item.get("localPreviewUrl")
        item["localPreviewSource"] = _text(overlay.get("local_preview_source")) or item.get("localPreviewSource")
        item["providerPreviewUrl"] = _text(item.get("providerPreviewUrl")) or _text(overlay.get("provider_preview_url")) or None

        installed_files = {( _text(record.get("version_id")), _text(record.get("file_id")) ) for record in records if _text(record.get("file_id"))}
        installed_hashes = {_lower(record.get("sha256")) for record in records if _lower(record.get("sha256"))}
        installed_versions = {_text(record.get("version_id")) for record in records if _text(record.get("version_id"))}
        for version in _list(item.get("versions")):
            if not isinstance(version, dict):
                continue
            version_id = _text(version.get("remoteVersionId"))
            version_installed = False
            for provider_file in _list(version.get("files")):
                if not isinstance(provider_file, dict):
                    continue
                file_id = _text(provider_file.get("remoteFileId"))
                hashes = _mapping(provider_file.get("hashes"))
                sha256 = _lower(hashes.get("SHA256") or hashes.get("sha256"))
                matched = (bool(file_id) and (version_id, file_id) in installed_files) or (bool(sha256) and sha256 in installed_hashes)
                provider_file["libraryStatus"] = "installed" if matched else "not_installed"
                if matched:
                    version_installed = True
                    matching = next((record for record in records if (_text(record.get("version_id")), _text(record.get("file_id"))) == (version_id, file_id)), None)
                    if matching:
                        provider_file["localAssetId"] = _text(matching.get("local_asset_id")) or None
                        provider_file["localAssetType"] = _text(matching.get("local_asset_type")) or None
            if not _list(version.get("files")) and version_id in installed_versions:
                version_installed = True
            version["libraryStatus"] = "installed" if version_installed else "not_installed"
            if version_installed:
                version["localAssetId"] = item.get("localAssetId")
                version["localAssetType"] = item.get("localAssetType")
        return item

    @staticmethod
    def _matches_preview(item: Mapping[str, Any], mode: str) -> bool:
        selected = _lower(mode) or "any"
        local_url = _text(item.get("localPreviewUrl"))
        local_source = _lower(item.get("localPreviewSource"))
        provider = bool(_text(item.get("providerPreviewUrl")) or (local_url and local_source == "civitai_cache"))
        local = bool(local_url and local_source != "civitai_cache")
        if selected in {"any", "best", "all"}:
            return True
        if selected in {"provider", "civitai"}:
            return provider
        if selected in {"local", "user", "user_local"}:
            return local
        if selected == "both":
            return provider and local
        return True

    def get_model(self, provider_id: str, remote_model_id: str) -> dict[str, Any] | None:
        provider = _lower(provider_id)
        model_id = _text(remote_model_id)
        if not provider or not model_id:
            return None
        cached: dict[str, Any] | None = None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM discovery_models WHERE provider_id=? AND remote_model_id=?",
                (provider, model_id),
            ).fetchone()
        if row is not None:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, Mapping):
                cached = dict(payload)
                cached["indexSource"] = "provider_cache"
                cached["indexedAtUnix"] = float(row["indexed_at_unix"] or 0.0)
                cached["firstSeenAtUnix"] = float(row["first_seen_at_unix"] or row["indexed_at_unix"] or 0.0)
                cached["lastRefreshedAtUnix"] = float(row["last_refreshed_at_unix"] or row["indexed_at_unix"] or 0.0)

        linked_records = tuple(
            record for record in self.presence.provider_linked_records(provider)
            if _text(record.get("model_id")) == model_id
        )
        local_items = self._local_summaries(provider, records=linked_records)
        local = next((item for item in local_items if _text(item.get("remoteModelId")) == model_id), None)
        if cached is None:
            return dict(local) if local is not None else None
        overlay = ({
            "local_asset_id": local.get("localAssetId"),
            "local_asset_type": local.get("localAssetType"),
            "local_preview_url": local.get("localPreviewUrl"),
            "local_preview_source": local.get("localPreviewSource"),
            "provider_preview_url": local.get("providerPreviewUrl"),
        } if local else {})
        return self._overlay_local(cached, overlay, linked_records)

    @staticmethod
    def _normalize_facet_filters(value: Mapping[str, Any] | None) -> dict[str, Any]:
        raw = dict(value or {})

        def values(*keys: str, limit: int = 32) -> list[str]:
            selected: Any = None
            for key in keys:
                if key in raw:
                    selected = raw.get(key)
                    break
            if isinstance(selected, str):
                selected = [selected]
            if not isinstance(selected, (list, tuple)):
                return []
            output: list[str] = []
            for item in selected:
                token = _text(item)
                if token and token not in output:
                    output.append(token)
                if len(output) >= limit:
                    break
            return output

        keyword_mode = _lower(raw.get("keywordMode") or "all_words").replace("-", "_")
        if keyword_mode not in {"all_words", "any_word", "exact_phrase"}:
            keyword_mode = "all_words"
        rating_basis = _lower(raw.get("ratingBasis") or "strictest").replace("-", "_")
        if rating_basis not in {"model", "author_previews", "strictest"}:
            rating_basis = "strictest"
        mature_mode = _lower(raw.get("maturePreviewMode") or "show")
        if mature_mode not in {"show", "blur", "hide"}:
            mature_mode = "show"
        local_sort = _lower(raw.get("localSort") or "candidate_order").replace("-", "_")
        if local_sort not in {"candidate_order", "safest", "most_mature", "newest", "title"}:
            local_sort = "candidate_order"
        ratings = []
        for rating in values("ratings", "ratingLevels", limit=16):
            token = rating.upper().replace("-", "")
            canonical = {"PG": "PG", "PG13": "PG13", "R": "R", "X": "X", "XXX": "XXX", "UNKNOWN": "Unknown"}.get(token)
            if canonical and canonical not in ratings:
                ratings.append(canonical)
        keyword_terms = values("keywordTerms", limit=32)
        legacy_keywords = _text(raw.get("keywords") or raw.get("keyword"))[:512]
        if not keyword_terms and legacy_keywords:
            keyword_terms = [legacy_keywords]
        return {
            "keywordTerms": keyword_terms,
            "keywords": " ".join(keyword_terms)[:512] if keyword_terms else legacy_keywords,
            "keywordMode": keyword_mode,
            "architectures": [normalize_architecture(item) for item in values("architectures", "baseModels") if normalize_architecture(item)],
            "assetKinds": [normalize_asset_kind(item) for item in values("assetKinds", "assetKind") if normalize_asset_kind(item) != "unknown"],
            "ratings": ratings,
            "ratingBasis": rating_basis,
            "supportStates": [_lower(item) for item in values("supportStates")],
            "libraryStates": [_lower(item).replace("in_library", "installed").replace("not_in_library", "not_installed") for item in values("libraryStates")],
            "previewSources": [_lower(item) for item in values("previewSources")],
            "creators": [item for item in values("creators", limit=16)],
            "categories": [item for item in values("categories", limit=16)],
            "maturePreviewMode": mature_mode,
            "localSort": local_sort,
        }

    def _prepare_local_presence(self, connection: sqlite3.Connection, provider_id: str) -> tuple[tuple[dict[str, Any], ...], dict[str, list[dict[str, Any]]]]:
        connection.execute("DROP TABLE IF EXISTS temp.temp_dsv2_local_presence")
        connection.execute(
            """
            CREATE TEMP TABLE temp_dsv2_local_presence (
                remote_model_id TEXT PRIMARY KEY,
                installed INTEGER NOT NULL DEFAULT 1,
                local_preview INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        records = tuple(self.presence.provider_linked_records(provider_id))
        by_model: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            model_id = _text(record.get("model_id"))
            if not model_id:
                continue
            by_model.setdefault(model_id, []).append(dict(record))
        if by_model:
            connection.executemany(
                "INSERT OR REPLACE INTO temp_dsv2_local_presence(remote_model_id,installed,local_preview) VALUES(?,?,?)",
                [
                    (
                        model_id,
                        1,
                        1 if any(_text(item.get("local_preview_url")) and _lower(item.get("local_preview_source")) != "civitai_cache" for item in model_records) else 0,
                    )
                    for model_id, model_records in by_model.items()
                ],
            )
        return records, by_model

    @staticmethod
    def _rating_column(basis: str) -> str:
        if basis == "model":
            return "d.model_rating"
        if basis == "author_previews":
            return "d.preview_rating"
        return "d.strictest_rating"

    @classmethod
    def _facet_filter_sql(cls, filters: Mapping[str, Any], *, exclude: str = "") -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        args: list[Any] = []
        keyword_terms = [
            " ".join(_text(item).casefold().split())
            for item in list(filters.get("keywordTerms") or [])
            if _text(item)
        ]
        if not keyword_terms:
            legacy_keywords = _text(filters.get("keywords"))
            if legacy_keywords:
                keyword_terms = [" ".join(legacy_keywords.casefold().split())]
        if keyword_terms and exclude != "keywords":
            mode = _lower(filters.get("keywordMode")) or "all_words"
            if mode == "exact_phrase":
                clauses: list[str] = []
                for phrase in keyword_terms:
                    clauses.append("d.search_text LIKE ?")
                    args.append(f"%{phrase}%")
                if clauses:
                    # Multiple committed phrase chips are cumulative refinements.
                    conditions.append("(" + " AND ".join(clauses) + ")")
            else:
                words = tuple(dict.fromkeys(
                    word
                    for phrase in keyword_terms
                    for word in phrase.split()
                    if word
                ))
                clauses: list[str] = []
                for word in words:
                    clauses.append("d.search_text LIKE ?")
                    args.append(f"%{word}%")
                if clauses:
                    joiner = " AND " if mode == "all_words" else " OR "
                    conditions.append("(" + joiner.join(clauses) + ")")

        architectures = list(filters.get("architectures") or [])
        if architectures and exclude != "architecture":
            placeholders = ",".join("?" for _ in architectures)
            conditions.append(
                f"EXISTS (SELECT 1 FROM discovery_model_architectures fa WHERE fa.provider_id=d.provider_id AND fa.remote_model_id=d.remote_model_id AND fa.architecture IN ({placeholders}))"
            )
            args.extend(architectures)

        asset_kinds = list(filters.get("assetKinds") or [])
        if asset_kinds and exclude != "asset_kind":
            placeholders = ",".join("?" for _ in asset_kinds)
            conditions.append(f"d.asset_kind IN ({placeholders})")
            args.extend(asset_kinds)

        ratings = list(filters.get("ratings") or [])
        if ratings and exclude != "rating":
            column = cls._rating_column(str(filters.get("ratingBasis") or "strictest"))
            placeholders = ",".join("?" for _ in ratings)
            conditions.append(f"{column} IN ({placeholders})")
            args.extend(ratings)

        support_states = list(filters.get("supportStates") or [])
        if support_states and exclude != "support":
            placeholders = ",".join("?" for _ in support_states)
            conditions.append(f"d.support_state IN ({placeholders})")
            args.extend(support_states)

        library_states = list(filters.get("libraryStates") or [])
        if library_states and exclude != "library":
            placeholders = ",".join("?" for _ in library_states)
            conditions.append(f"(CASE WHEN lp.remote_model_id IS NULL THEN 'not_installed' ELSE 'installed' END) IN ({placeholders})")
            args.extend(library_states)

        preview_sources = list(filters.get("previewSources") or [])
        if preview_sources and exclude != "preview":
            placeholders = ",".join("?" for _ in preview_sources)
            conditions.append(
                "(CASE "
                "WHEN d.provider_preview=1 AND COALESCE(lp.local_preview,0)=1 THEN 'both' "
                "WHEN d.provider_preview=1 THEN 'provider' "
                "WHEN COALESCE(lp.local_preview,0)=1 THEN 'local' ELSE 'none' END) "
                f"IN ({placeholders})"
            )
            args.extend(preview_sources)

        creators = [item.casefold() for item in list(filters.get("creators") or []) if _text(item)]
        if creators and exclude != "creator":
            placeholders = ",".join("?" for _ in creators)
            conditions.append(f"d.creator_text IN ({placeholders})")
            args.extend(creators)

        categories = [item.casefold() for item in list(filters.get("categories") or []) if _text(item)]
        if categories and exclude != "category":
            placeholders = ",".join("?" for _ in categories)
            conditions.append(f"LOWER(d.provider_category) IN ({placeholders})")
            args.extend(categories)
        return conditions, args

    @staticmethod
    def _facet_base_from(*, mode: str, session_id: str) -> tuple[str, list[Any]]:
        if mode == "search" and session_id:
            return (
                "FROM discovery_models d "
                "JOIN asset_search_session_candidates c ON c.provider_id=d.provider_id AND c.remote_model_id=d.remote_model_id "
                "LEFT JOIN temp_dsv2_local_presence lp ON lp.remote_model_id=d.remote_model_id "
                "WHERE d.provider_id=? AND c.session_id=?",
                [],
            )
        return (
            "FROM discovery_models d "
            "LEFT JOIN temp_dsv2_local_presence lp ON lp.remote_model_id=d.remote_model_id "
            "WHERE d.provider_id=?",
            [],
        )

    def query_facets(
        self,
        *,
        provider_id: str,
        session_id: str = "",
        mode: str = "browse",
        filters: Mapping[str, Any] | None = None,
        sort: str = "",
        offset: int = 0,
        limit: int = 50,
        facets: Iterable[str] = (),
    ) -> dict[str, Any]:
        provider = _lower(provider_id) or "civitai"
        selected_session_id = _text(session_id)
        selected_mode = "search" if _lower(mode) == "search" else "browse"
        normalized = self._normalize_facet_filters(filters)
        if sort:
            normalized["localSort"] = self._normalize_facet_filters({"localSort": sort})["localSort"]
        page_limit = max(1, min(int(limit or 50), 500))
        page_offset = max(0, int(offset or 0))
        requested_facets = tuple(dict.fromkeys(_lower(value).replace("-", "_") for value in facets if _text(value)))
        if not requested_facets:
            requested_facets = ("architecture", "asset_kind", "rating", "support", "library", "preview", "creator", "category")

        with self._lock, self._connect() as connection:
            linked_records, local_records_by_id = self._prepare_local_presence(connection, provider)
            base_from, _ = self._facet_base_from(mode=selected_mode, session_id=selected_session_id)
            base_args: list[Any] = [provider]
            if selected_mode == "search" and selected_session_id:
                base_args.append(selected_session_id)

            candidate_row = connection.execute(f"SELECT COUNT(DISTINCT d.remote_model_id) AS total {base_from}", tuple(base_args)).fetchone()
            candidate_count = int(candidate_row["total"] or 0) if candidate_row else 0

            conditions, condition_args = self._facet_filter_sql(normalized)
            where_suffix = "" if not conditions else " AND " + " AND ".join(conditions)
            match_row = connection.execute(
                f"SELECT COUNT(DISTINCT d.remote_model_id) AS total {base_from}{where_suffix}",
                tuple([*base_args, *condition_args]),
            ).fetchone()
            match_count = int(match_row["total"] or 0) if match_row else 0

            rating_column = self._rating_column(normalized["ratingBasis"])
            local_sort = normalized["localSort"]
            if local_sort == "title":
                order_sql = "d.name_text ASC, d.remote_model_id ASC"
            elif local_sort == "newest":
                order_sql = "d.last_refreshed_at_unix DESC, d.remote_model_id ASC"
            elif local_sort == "safest":
                order_sql = f"CASE {rating_column} WHEN 'PG' THEN 0 WHEN 'PG13' THEN 1 WHEN 'R' THEN 2 WHEN 'X' THEN 3 WHEN 'XXX' THEN 4 WHEN 'Blocked' THEN 5 ELSE 99 END ASC, d.name_text ASC"
            elif local_sort == "most_mature":
                order_sql = f"CASE {rating_column} WHEN 'Blocked' THEN 6 WHEN 'XXX' THEN 5 WHEN 'X' THEN 4 WHEN 'R' THEN 3 WHEN 'PG13' THEN 2 WHEN 'PG' THEN 1 ELSE 0 END DESC, d.name_text ASC"
            elif selected_mode == "search" and selected_session_id:
                order_sql = "c.sequence_no ASC"
            else:
                order_sql = "d.indexed_at_unix DESC, d.remote_model_id ASC"

            rows = connection.execute(
                f"SELECT d.* {base_from}{where_suffix} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                tuple([*base_args, *condition_args, page_limit, page_offset]),
            ).fetchall()

            facet_payload: dict[str, list[dict[str, Any]]] = {}
            for facet in requested_facets:
                exclude = facet
                if facet in {"assetkind", "model_type", "type"}:
                    facet = "asset_kind"
                    exclude = "asset_kind"
                elif facet in {"base_model", "architectures"}:
                    facet = "architecture"
                    exclude = "architecture"
                elif facet in {"ratings"}:
                    facet = "rating"
                    exclude = "rating"
                elif facet in {"support_state"}:
                    facet = "support"
                    exclude = "support"
                elif facet in {"library_state"}:
                    facet = "library"
                    exclude = "library"
                elif facet in {"preview_source"}:
                    facet = "preview"
                    exclude = "preview"

                facet_conditions, facet_args = self._facet_filter_sql(normalized, exclude=exclude)
                facet_suffix = "" if not facet_conditions else " AND " + " AND ".join(facet_conditions)
                query = ""
                if facet == "architecture":
                    if selected_mode == "search" and selected_session_id:
                        arch_base = (
                            "FROM discovery_models d "
                            "JOIN asset_search_session_candidates c ON c.provider_id=d.provider_id AND c.remote_model_id=d.remote_model_id "
                            "JOIN discovery_model_architectures a ON a.provider_id=d.provider_id AND a.remote_model_id=d.remote_model_id "
                            "LEFT JOIN temp_dsv2_local_presence lp ON lp.remote_model_id=d.remote_model_id "
                            "WHERE d.provider_id=? AND c.session_id=?"
                        )
                    else:
                        arch_base = (
                            "FROM discovery_models d "
                            "JOIN discovery_model_architectures a ON a.provider_id=d.provider_id AND a.remote_model_id=d.remote_model_id "
                            "LEFT JOIN temp_dsv2_local_presence lp ON lp.remote_model_id=d.remote_model_id "
                            "WHERE d.provider_id=?"
                        )
                    query = (
                        "SELECT a.architecture AS value, COUNT(DISTINCT d.remote_model_id) AS count "
                        f"{arch_base}{facet_suffix} GROUP BY a.architecture HAVING count > 0 ORDER BY count DESC, value ASC"
                    )
                elif facet == "asset_kind":
                    query = f"SELECT d.asset_kind AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} GROUP BY d.asset_kind HAVING count > 0 ORDER BY count DESC, value ASC"
                elif facet == "rating":
                    column = self._rating_column(normalized["ratingBasis"])
                    query = f"SELECT {column} AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} AND {column} != 'Blocked' GROUP BY {column} HAVING count > 0 ORDER BY CASE {column} WHEN 'PG' THEN 0 WHEN 'PG13' THEN 1 WHEN 'R' THEN 2 WHEN 'X' THEN 3 WHEN 'XXX' THEN 4 ELSE 5 END ASC"
                elif facet == "support":
                    query = f"SELECT d.support_state AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} GROUP BY d.support_state HAVING count > 0 ORDER BY count DESC, value ASC"
                elif facet == "library":
                    expression = "CASE WHEN lp.remote_model_id IS NULL THEN 'not_installed' ELSE 'installed' END"
                    query = f"SELECT {expression} AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} GROUP BY {expression} HAVING count > 0 ORDER BY count DESC, value ASC"
                elif facet == "preview":
                    expression = "CASE WHEN d.provider_preview=1 AND COALESCE(lp.local_preview,0)=1 THEN 'both' WHEN d.provider_preview=1 THEN 'provider' WHEN COALESCE(lp.local_preview,0)=1 THEN 'local' ELSE 'none' END"
                    query = f"SELECT {expression} AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} GROUP BY {expression} HAVING count > 0 ORDER BY count DESC, value ASC"
                elif facet == "creator":
                    query = f"SELECT d.creator_value AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} AND d.creator_value != '' GROUP BY d.creator_text, d.creator_value HAVING count > 0 ORDER BY count DESC, value COLLATE NOCASE ASC LIMIT 200"
                elif facet == "category":
                    query = f"SELECT d.provider_category AS value, COUNT(DISTINCT d.remote_model_id) AS count {base_from}{facet_suffix} AND d.provider_category != '' GROUP BY d.provider_category HAVING count > 0 ORDER BY count DESC, value COLLATE NOCASE ASC LIMIT 200"
                if not query:
                    continue
                facet_rows = connection.execute(query, tuple([*base_args, *facet_args])).fetchall()
                facet_payload[facet] = [
                    {"value": str(row["value"] or ""), "count": int(row["count"] or 0)}
                    for row in facet_rows
                    if _text(row["value"])
                ]

        items: list[dict[str, Any]] = []
        for row in rows:
            item = self._row_item(row)
            if item is None:
                continue
            model_id = _text(item.get("remoteModelId"))
            model_records = local_records_by_id.get(model_id, [])
            overlay = {}
            if model_records:
                local_preview = next((record for record in model_records if _text(record.get("local_preview_url")) and _lower(record.get("local_preview_source")) != "civitai_cache"), model_records[0])
                provider_preview = next((record for record in model_records if _text(record.get("provider_preview_url"))), model_records[0])
                overlay = {
                    "local_asset_id": local_preview.get("local_asset_id"),
                    "local_asset_type": local_preview.get("local_asset_type"),
                    "local_preview_url": local_preview.get("local_preview_url"),
                    "local_preview_source": local_preview.get("local_preview_source"),
                    "provider_preview_url": provider_preview.get("provider_preview_url"),
                }
            items.append(self._overlay_local(item, overlay, model_records))

        next_offset = page_offset + len(items) if page_offset + len(items) < match_count else None
        return {
            "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
            "providerId": provider,
            "sessionId": selected_session_id or None,
            "candidateCount": candidate_count,
            "matchCount": match_count,
            "items": items,
            "facets": facet_payload,
            "filters": normalized,
            "sort": normalized["localSort"],
            "offset": page_offset,
            "nextOffset": next_offset,
            "limit": page_limit,
            "source": "local_faceted_query",
        }

    def search(
        self,
        *,
        provider_id: str,
        query: str = "",
        creator: str = "",
        asset_kind: str = "any",
        base_models: tuple[str, ...] = (),
        safe_content: bool = True,
        support_filter: str = "any",
        library_filter: str = "any",
        preview_filter: str = "any",
        sort: str = "",
        period: str = "",
        mode: str = "search",
        limit: int = 48,
        offset: int = 0,
        candidate_session_id: str = "",
    ) -> dict[str, Any]:
        """Return local candidate results for the requested discovery scope.

        DSV2-01 separates the *candidate pool* from later refinements. The
        first pass uses only primary discovery semantics (provider, search text,
        creator, asset kind, provider sort/period, browse vs. search). When a
        search session has explicit candidate membership, this method returns
        that stable ordered pool; otherwise it falls back to the global indexed
        catalog and any persistent snapshot for the same primary discovery key.
        """
        provider = _lower(provider_id)
        kind = normalize_asset_kind(asset_kind) if _lower(asset_kind) not in {"", "any", "all", "*"} else "any"
        phrase = " ".join(_lower(query).split()) if _lower(mode) == "search" else ""
        creator_filter = " ".join(_lower(creator).split())
        requested_architectures = {
            normalize_architecture(value)
            for value in base_models
            if normalize_architecture(value)
        }
        normalized_support = _lower(support_filter) or "any"
        normalized_library = _lower(library_filter) or "any"
        normalized_library = {
            "in_library": "installed",
            "already_in_library": "installed",
            "not_in_library": "not_installed",
        }.get(normalized_library, normalized_library)
        page_limit = max(1, min(int(limit or 48), 500))
        page_offset = max(0, int(offset or 0))
        selected_session_id = _text(candidate_session_id)

        def apply_local_overlay(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            linked_records = self.presence.provider_linked_records(provider)
            local_items = self._local_summaries(provider, records=linked_records)
            local_by_id = {
                str(item.get("remoteModelId") or ""): item
                for item in local_items
                if item.get("remoteModelId")
            }
            local_records_by_id: dict[str, list[dict[str, Any]]] = {}
            for record in linked_records:
                model_id = _text(record.get("model_id"))
                if model_id:
                    local_records_by_id.setdefault(model_id, []).append(dict(record))
            output: list[dict[str, Any]] = []
            for item in items:
                model_id = str(item.get("remoteModelId") or "")
                local = local_by_id.get(model_id)
                overlay = ({
                    "local_asset_id": local.get("localAssetId"),
                    "local_asset_type": local.get("localAssetType"),
                    "local_preview_url": local.get("localPreviewUrl"),
                    "local_preview_source": local.get("localPreviewSource"),
                    "provider_preview_url": local.get("providerPreviewUrl"),
                } if local else {})
                output.append(self._overlay_local(dict(item), overlay, local_records_by_id.get(model_id, ())))
            return output

        session_candidate_count = 0
        session_candidate_items: list[dict[str, Any]] = []
        if selected_session_id:
            with self._lock, self._connect() as connection:
                count_row = connection.execute(
                    "SELECT COUNT(*) AS total FROM asset_search_session_candidates WHERE session_id=? AND provider_id=?",
                    (selected_session_id, provider),
                ).fetchone()
                session_candidate_count = int(count_row["total"] or 0) if count_row else 0
                if session_candidate_count:
                    rows = connection.execute(
                        """
                        SELECT d.*, c.sequence_no AS candidate_sequence_no
                        FROM asset_search_session_candidates c
                        JOIN discovery_models d
                          ON d.provider_id = c.provider_id AND d.remote_model_id = c.remote_model_id
                        WHERE c.session_id=? AND c.provider_id=?
                        ORDER BY c.sequence_no ASC
                        """,
                        (selected_session_id, provider),
                    ).fetchall()
            for row in rows if session_candidate_count else ():
                item = self._row_item(row)
                if item is not None:
                    item["candidateSequence"] = int(row["candidate_sequence_no"] or 0)
                    session_candidate_items.append(item)

        snapshot = None
        snapshot_ids: list[str] = []
        snapshot_complete = False
        snapshot_hit = False
        snapshot_order: dict[str, int] = {}
        cached: list[dict[str, Any]] = []

        if not session_candidate_items:
            snapshot = self._snapshot_record(
                provider_id=provider,
                query=query,
                creator=creator,
                asset_kind=asset_kind,
                base_models=base_models,
                safe_content=safe_content,
                support_filter=support_filter,
                library_filter=library_filter,
                preview_filter=preview_filter,
                sort=sort,
                period=period,
                mode=mode,
            )
            snapshot_ids = list(snapshot[0]) if snapshot else []
            snapshot_complete = bool(snapshot[1]) if snapshot else False
            snapshot_hit = bool(snapshot_ids)
            snapshot_order = {model_id: index for index, model_id in enumerate(snapshot_ids)}

            if snapshot_hit:
                rows_by_id: dict[str, sqlite3.Row] = {}
                with self._lock, self._connect() as connection:
                    for start in range(0, len(snapshot_ids), 400):
                        chunk = snapshot_ids[start:start + 400]
                        placeholders = ",".join("?" for _ in chunk)
                        rows = connection.execute(
                            f"SELECT * FROM discovery_models WHERE provider_id=? AND remote_model_id IN ({placeholders})",
                            (provider, *chunk),
                        ).fetchall()
                        for row in rows:
                            rows_by_id[_text(row["remote_model_id"])] = row
                for model_id in snapshot_ids:
                    row = rows_by_id.get(model_id)
                    if row is None:
                        continue
                    item = self._row_item(row)
                    if item is not None:
                        cached.append(item)
            else:
                sql = "SELECT * FROM discovery_models WHERE provider_id=?"
                args: list[Any] = [provider]
                if kind != "any":
                    sql += " AND asset_kind=?"
                    args.append(kind)
                if creator_filter:
                    sql += " AND creator_text LIKE ?"
                    args.append(f"%{creator_filter}%")
                if phrase:
                    terms = tuple(dict.fromkeys([phrase, *phrase.split()]))
                    term_clauses: list[str] = []
                    for term in terms:
                        term_clauses.append("(name_text LIKE ? OR creator_text LIKE ? OR description_text LIKE ? OR tags_text LIKE ?)")
                        like = f"%{term}%"
                        args.extend([like, like, like, like])
                    sql += " AND (" + " OR ".join(term_clauses) + ")"
                sql += " ORDER BY indexed_at_unix DESC"
                with self._lock, self._connect() as connection:
                    for row in connection.execute(sql, tuple(args)).fetchall():
                        item = self._row_item(row)
                        if item is not None:
                            cached.append(item)

        base_items = session_candidate_items or cached
        output: list[dict[str, Any]] = []
        for item in apply_local_overlay(base_items):
            if kind != "any" and normalize_asset_kind(item.get("assetKind")) != kind:
                continue
            if creator_filter and creator_filter not in _lower(item.get("creator")):
                continue
            # Compatibility lane: callers of the legacy local-index search API
            # may still supply secondary filters. DSV2-02's primary UI path uses
            # query_facets(), but preserving these semantics keeps older callers,
            # tests, and startup replay behavior deterministic during migration.
            if safe_content and bool(item.get("nsfw")):
                continue
            if requested_architectures and not requested_architectures.intersection(set(_architectures(item))):
                continue
            support_state = _lower(item.get("supportState")) or "unknown"
            if normalized_support != "any" and support_state != normalized_support:
                continue
            library_state = _lower(item.get("libraryStatus")) or "not_installed"
            if normalized_library != "any" and library_state != normalized_library:
                continue
            if not self._matches_preview(item, preview_filter):
                continue
            score = _rank(item, phrase)
            if phrase and score <= 0:
                continue
            item["searchRank"] = score
            output.append(item)

        if session_candidate_items:
            output.sort(key=lambda item: int(item.get("candidateSequence") or 0))
            source = "search_session_candidates"
        elif snapshot_hit:
            output.sort(key=lambda item: snapshot_order.get(str(item.get("remoteModelId") or ""), len(snapshot_order)))
            source = "persistent_search_snapshot"
        else:
            output.sort(key=lambda item: (int(item.get("searchRank") or 0), float(item.get("indexedAtUnix") or 0.0)), reverse=True)
            source = "local_discovery_index"

        visible = output[page_offset: page_offset + page_limit]
        next_offset = page_offset + len(visible) if page_offset + len(visible) < len(output) else None
        return {
            "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
            "providerId": provider,
            "items": visible,
            "totalItems": len(output),
            "indexHitCount": len(visible),
            "offset": page_offset,
            "nextOffset": next_offset,
            "source": source,
            "snapshotHit": snapshot_hit,
            "snapshotComplete": snapshot_complete if snapshot_hit else False,
            "snapshotResultCount": len(snapshot_ids) if snapshot_hit else 0,
            "candidateSessionId": selected_session_id or None,
            "candidateCount": session_candidate_count if selected_session_id else len(output),
        }

    def status(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, MAX(last_refreshed_at_unix) AS latest FROM discovery_models"
            ).fetchone()
            snapshot_row = connection.execute(
                "SELECT COUNT(*) AS total FROM discovery_search_snapshots"
            ).fetchone()
        return {
            "schemaVersion": DISCOVERY_INDEX_SCHEMA_VERSION,
            "databasePath": str(self.database_path),
            "providerModels": int(row["total"] or 0) if row else 0,
            "persistentSearchSnapshots": int(snapshot_row["total"] or 0) if snapshot_row else 0,
            "latestIndexedAtUnix": float(row["latest"] or 0.0) if row else 0.0,
            "localProviderLinkedAssets": len(self.presence.provider_linked_records()),
        }
