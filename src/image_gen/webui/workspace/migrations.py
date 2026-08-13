from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .models import WORKSPACE_SCHEMA, WORKSPACE_SCHEMA_VERSION


class WorkspaceMigrationError(ValueError):
    pass


Migration = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class WorkspaceMigrationResult:
    payload: dict[str, Any]
    migrated_from_version: int | None = None


class WorkspaceMigrationRegistry:
    def __init__(self, current_version: int = WORKSPACE_SCHEMA_VERSION) -> None:
        self.current_version = int(current_version)
        self._migrations: dict[int, Migration] = {}

    def register(self, from_version: int, migration: Migration) -> None:
        version = int(from_version)
        if version in self._migrations:
            raise WorkspaceMigrationError(f"Migration from workspace schema version {version} is already registered.")
        self._migrations[version] = migration

    def migrate(self, raw: Mapping[str, Any]) -> WorkspaceMigrationResult:
        payload = deepcopy(dict(raw))
        schema = str(payload.get("schema") or "").strip()
        if schema and schema != WORKSPACE_SCHEMA:
            raise WorkspaceMigrationError(f"Unsupported workspace schema '{schema}'.")
        raw_version = payload.get("schemaVersion")
        if raw_version is None:
            raw_version = 0
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise WorkspaceMigrationError("schemaVersion must be an integer.") from exc
        if version > self.current_version:
            raise WorkspaceMigrationError(
                f"Workspace schema version {version} is newer than supported version {self.current_version}."
            )
        original = version
        while version < self.current_version:
            migration = self._migrations.get(version)
            if migration is None:
                raise WorkspaceMigrationError(
                    f"No workspace migration is registered from schema version {version}."
                )
            payload = migration(payload)
            version += 1
            payload["schema"] = WORKSPACE_SCHEMA
            payload["schemaVersion"] = version
        payload.setdefault("schema", WORKSPACE_SCHEMA)
        payload.setdefault("schemaVersion", self.current_version)
        return WorkspaceMigrationResult(payload, original if original != self.current_version else None)


def migrate_legacy_layout_v0(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the Phase-01 browser/localStorage shape into canonical v1 geometry."""

    source = deepcopy(payload)
    page_id = str(source.get("pageId") or "").strip()
    workspace_id = str(source.get("workspaceId") or "").strip() or f"legacy.{page_id or 'workspace'}"
    name = str(source.get("name") or "").strip() or "Imported legacy workspace"
    migrated: list[dict[str, Any]] = []
    for index, item in enumerate(source.get("components") or []):
        if not isinstance(item, Mapping):
            continue
        span = item.get("span", item.get("columnSpan", 12))
        order = item.get("order", index)
        try:
            span_value = int(span)
        except (TypeError, ValueError):
            span_value = 12
        try:
            order_value = int(order)
        except (TypeError, ValueError):
            order_value = index
        migrated.append(
            {
                "componentId": str(item.get("componentId") or ""),
                "instanceId": str(item.get("instanceId") or ""),
                "position": {
                    "column": 1,
                    "row": max(1, order_value + 1),
                    "columnSpan": span_value,
                    "heightUnits": 1,
                },
                "variant": str(item.get("variant") or "standard"),
                "settings": dict(item.get("settings") or {}) if isinstance(item.get("settings"), Mapping) else {},
                "responsive": dict(item.get("responsive") or {}) if isinstance(item.get("responsive"), Mapping) else {},
                "visible": item.get("visible", True) is not False,
                "shellState": str(item.get("shellState") or "expanded"),
            }
        )
    return {
        "schema": WORKSPACE_SCHEMA,
        "schemaVersion": 1,
        "workspaceId": workspace_id,
        "name": name,
        "pageId": page_id,
        "components": migrated,
    }


def default_workspace_migrations() -> WorkspaceMigrationRegistry:
    registry = WorkspaceMigrationRegistry()
    registry.register(0, migrate_legacy_layout_v0)
    return registry
