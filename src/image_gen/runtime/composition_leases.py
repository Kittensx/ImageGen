from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import threading
import time
from typing import Any, Mapping

from image_gen.runtime.component_residency import (
    ResidentComponentEntry,
    build_resident_component_inventory,
)
from modules.registry.family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY


COMPOSITION_LEASE_SCHEMA_VERSION = 1


def _digest(value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _path_key(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        return os.path.normcase(str(Path(token).expanduser().resolve(strict=False)))
    except OSError:
        return os.path.normcase(token)


def _module_device_dtype(module: Any) -> tuple[str, str]:
    if module is None:
        return "missing", "unknown"
    try:
        parameter = next(module.parameters())
        return str(parameter.device), str(parameter.dtype)
    except (StopIteration, AttributeError, TypeError):
        return str(getattr(module, "device", "unknown")), str(getattr(module, "dtype", "unknown"))


def _module_for_role(container: Any, role: str) -> Any:
    if container is None:
        return None
    if role == "transformer":
        denoiser = getattr(container, "denoiser", None)
        return denoiser if denoiser is not None else getattr(container, "transformer", None)
    return getattr(container, role, None)


@dataclass
class LeasedComponentEntry:
    component_sha256: str
    role: str
    module: Any
    family: str
    provider_version: str
    source_model_path: str
    consumers: tuple[int, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)
    origin: str = "resident"
    warmed_at_unix: float | None = None
    reuse_eligible: bool = True
    reuse_block_reason: str = ""

    @property
    def device(self) -> str:
        return _module_device_dtype(self.module)[0]

    @property
    def dtype(self) -> str:
        return _module_device_dtype(self.module)[1]

    def public_dict(self) -> dict[str, Any]:
        return {
            "component_sha256": self.component_sha256,
            "role": self.role,
            "family": self.family,
            "provider_version": self.provider_version,
            "source_model_path": self.source_model_path,
            "consumers": list(self.consumers),
            "source": dict(self.source),
            "origin": self.origin,
            "warmed_at_unix": self.warmed_at_unix,
            "runtime_object_id": id(self.module) if self.module is not None else None,
            "device": self.device,
            "dtype": self.dtype,
            "reuse_eligible": bool(self.reuse_eligible),
            "reuse_block_reason": self.reuse_block_reason,
        }


@dataclass
class PreparedCompositionEntry:
    index: int
    model_path: str
    family: str
    provider_version: str
    composition_sha256: str
    components: dict[str, str]
    runtime_settings: dict[str, Any] = field(default_factory=dict)
    load_plan: Any = None
    built_components: Any = None
    loaded_model: Any = None
    prepared_at_unix: float = field(default_factory=time.time)
    origin: str = "warm_stage"

    def public_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "model_path": self.model_path,
            "family": self.family,
            "provider_version": self.provider_version,
            "composition_sha256": self.composition_sha256,
            "component_roles": sorted(self.components),
            "prepared_at_unix": self.prepared_at_unix,
            "origin": self.origin,
            "has_load_plan": self.load_plan is not None,
            "has_built_components": self.built_components is not None,
            "has_loaded_model": self.loaded_model is not None,
        }


class CompositionExecutionLease:
    """Process-local lifetime contract for a planned same-family model sequence.

    The lease owns *references*, not serialized model objects. Exact component SHA is
    the deduplication key. A component needed by several planned compositions is held
    once and its consumer indexes describe how long it remains useful.
    """

    def __init__(
        self,
        *,
        generation: int,
        schedule: list[dict[str, Any]],
        active_index: int,
        family: str,
        provider_version: str,
    ) -> None:
        self.schema_version = COMPOSITION_LEASE_SCHEMA_VERSION
        self.generation = int(generation)
        self.schedule = [dict(item) for item in schedule]
        self.active_index = int(active_index)
        self.family = str(family or "")
        self.provider_version = str(provider_version or "")
        self.created_unix = time.time()
        self.updated_unix = self.created_unix
        self.state = "active"
        self.reason = "established"
        self.component_pool: dict[str, LeasedComponentEntry] = {}
        self._prepared_compositions: dict[int, PreparedCompositionEntry] = {}
        self.warm_states: dict[int, dict[str, Any]] = {}
        self.event_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._consumer_map: dict[str, tuple[int, ...]] = self._build_consumer_map()

    @staticmethod
    def _pool_key(role: str, digest: str) -> str:
        return f"{str(role)}:{_digest(digest)}"

    def _build_consumer_map(self) -> dict[str, tuple[int, ...]]:
        result: dict[str, list[int]] = {}
        for index, item in enumerate(self.schedule):
            for role, digest in dict(item.get("components") or {}).items():
                token = _digest(digest)
                if token:
                    result.setdefault(self._pool_key(str(role), token), []).append(index)
        return {key: tuple(indexes) for key, indexes in result.items()}

    def _event(self, kind: str, **payload: Any) -> None:
        self.updated_unix = time.time()
        self.event_history.append({
            "timestamp_unix": self.updated_unix,
            "kind": str(kind),
            **payload,
        })
        self.event_history = self.event_history[-100:]

    def register_active_loaded(self, loaded: Any, *, adapter_state_dirty: bool = False) -> None:
        inventory = build_resident_component_inventory(
            loaded,
            adapter_state_dirty=adapter_state_dirty,
        )
        with self._lock:
            for role, entry in inventory.items():
                digest = _digest(entry.component_sha256)
                if not digest or entry.module is None:
                    continue
                pool_key = self._pool_key(role, digest)
                existing = self.component_pool.get(pool_key)
                if existing is not None and existing.module is not entry.module:
                    # Exact SHA should converge on one live object inside one lease.
                    # Prefer the currently active object and drop the stale pool ref.
                    pass
                self.component_pool[pool_key] = LeasedComponentEntry(
                    component_sha256=digest,
                    role=role,
                    module=entry.module,
                    family=entry.family,
                    provider_version=entry.provider_version,
                    source_model_path=str(getattr(getattr(loaded, "load_plan", None), "report", None).model_path if getattr(getattr(loaded, "load_plan", None), "report", None) is not None else ""),
                    consumers=self._consumer_map.get(pool_key, ()),
                    source=dict(entry.source),
                    origin="active_resident",
                    reuse_eligible=bool(entry.reuse_eligible),
                    reuse_block_reason=str(entry.reuse_block_reason or ""),
                )
            active_item = dict(self.schedule[self.active_index] or {}) if 0 <= self.active_index < len(self.schedule) else {}
            self._prepared_compositions[self.active_index] = PreparedCompositionEntry(
                index=self.active_index,
                model_path=str(active_item.get("model_path") or ""),
                family=self.family,
                provider_version=self.provider_version,
                composition_sha256=str(active_item.get("composition_sha256") or ""),
                components={str(role): _digest(digest) for role, digest in dict(active_item.get("components") or {}).items() if _digest(digest)},
                runtime_settings=dict(active_item.get("runtime_settings") or {}),
                load_plan=getattr(loaded, "load_plan", None),
                built_components=getattr(loaded, "built_components", None),
                loaded_model=loaded,
                origin="active_resident",
            )
            self._event("active_components_registered", count=len(inventory), active_index=self.active_index)

    def reusable_bundle(self, loaded: Any | None, *, adapter_state_dirty: bool = False) -> dict[str, Any]:
        current_entries = build_resident_component_inventory(
            loaded,
            adapter_state_dirty=adapter_state_dirty,
        ) if loaded is not None else {}
        with self._lock:
            return {
                "schema_version": COMPOSITION_LEASE_SCHEMA_VERSION,
                "family": self.family,
                "provider_version": self.provider_version,
                "composition_sha256": str(
                    getattr(getattr(loaded, "components", None), "composition_sha256", "") or ""
                ),
                "entries": dict(current_entries),
                "lease_entries_by_sha": dict(self.component_pool),
                "public_inventory": {
                    role: entry.public_dict() for role, entry in current_entries.items()
                },
                "lease_public_inventory": {
                    digest: entry.public_dict() for digest, entry in self.component_pool.items()
                },
                "lease_generation": self.generation,
            }

    def mark_queued(self, index: int, model_path: str) -> None:
        with self._lock:
            current = dict(self.warm_states.get(int(index)) or {})
            if str(current.get("state") or "") in {"queued", "warming", "warm"}:
                return
            self.warm_states[int(index)] = {
                "state": "queued",
                "model_path": str(model_path),
                "queued_unix": time.time(),
            }
            self._event("warm_queued", index=int(index), model_path=str(model_path))

    def mark_warming(self, index: int, model_path: str) -> None:
        with self._lock:
            previous = dict(self.warm_states.get(int(index)) or {})
            self.warm_states[int(index)] = {
                **previous,
                "state": "warming",
                "model_path": str(model_path),
                "started_unix": time.time(),
            }
            self._event("warm_started", index=int(index), model_path=str(model_path))

    def register_warmed_components(
        self,
        *,
        index: int,
        model_path: str,
        components: Any,
        target_components: Mapping[str, str],
        load_plan: Any = None,
        runtime_settings: Mapping[str, Any] | None = None,
        loaded_model: Any = None,
    ) -> dict[str, Any]:
        newly_registered: list[str] = []
        reused_existing: list[str] = []
        with self._lock:
            for role, raw_digest in dict(target_components or {}).items():
                digest = _digest(raw_digest)
                if not digest:
                    continue
                module = _module_for_role(components, role)
                if module is None:
                    continue
                pool_key = self._pool_key(str(role), digest)
                existing = self.component_pool.get(pool_key)
                if existing is not None:
                    reused_existing.append(role)
                    # If the builder returned another object for the same exact SHA,
                    # keep the lease's canonical live object. The extra object falls
                    # out of scope when the warm build result is released.
                    continue
                self.component_pool[pool_key] = LeasedComponentEntry(
                    component_sha256=digest,
                    role=str(role),
                    module=module,
                    family=self.family,
                    provider_version=self.provider_version,
                    source_model_path=str(model_path),
                    consumers=self._consumer_map.get(pool_key, ()),
                    source={"source_kind": "lease_warm_cpu", "source_path": str(model_path)},
                    origin="background_warm",
                    warmed_at_unix=time.time(),
                )
                newly_registered.append(str(role))
            schedule_item = dict(self.schedule[int(index)] or {}) if 0 <= int(index) < len(self.schedule) else {}
            self._prepared_compositions[int(index)] = PreparedCompositionEntry(
                index=int(index),
                model_path=str(model_path),
                family=self.family,
                provider_version=self.provider_version,
                composition_sha256=str(schedule_item.get("composition_sha256") or ""),
                components={str(role): _digest(digest) for role, digest in dict(target_components or {}).items() if _digest(digest)},
                runtime_settings=dict(runtime_settings or schedule_item.get("runtime_settings") or {}),
                load_plan=load_plan,
                built_components=components,
                loaded_model=loaded_model,
                origin="background_warm",
            )
            previous_state = dict(self.warm_states.get(int(index)) or {})
            self.warm_states[int(index)] = {
                **previous_state,
                "state": "warm",
                "model_path": str(model_path),
                "completed_unix": time.time(),
                "new_component_roles": newly_registered,
                "already_leased_roles": reused_existing,
            }
            self._event(
                "warm_completed",
                index=int(index),
                model_path=str(model_path),
                new_component_roles=newly_registered,
                already_leased_roles=reused_existing,
            )
        return {
            "new_component_roles": newly_registered,
            "already_leased_roles": reused_existing,
        }

    def prepared_composition(self, index: int) -> PreparedCompositionEntry | None:
        index = int(index)
        with self._lock:
            entry = self._prepared_compositions.get(index)
            if entry is not None:
                return entry
            if index < 0 or index >= len(self.schedule):
                return None
            requested = dict(self.schedule[index] or {})
            requested_path = _path_key(requested.get("model_path"))
            requested_sha = str(requested.get("composition_sha256") or "")
            requested_components = {
                str(role): _digest(digest)
                for role, digest in dict(requested.get("components") or {}).items()
                if _digest(digest)
            }
            for prepared in self._prepared_compositions.values():
                if requested_sha and prepared.composition_sha256 == requested_sha:
                    return prepared
                if requested_path and _path_key(prepared.model_path) == requested_path and prepared.components == requested_components:
                    return prepared
            return None

    def prepared_composition_status(self, index: int) -> dict[str, Any]:
        entry = self.prepared_composition(index)
        return entry.public_dict() if entry is not None else {}

    def mark_warm_failed(self, index: int, model_path: str, exc: BaseException) -> None:
        with self._lock:
            previous_state = dict(self.warm_states.get(int(index)) or {})
            self.warm_states[int(index)] = {
                **previous_state,
                "state": "failed",
                "model_path": str(model_path),
                "completed_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._event(
                "warm_failed",
                index=int(index),
                model_path=str(model_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def next_warm_index(self) -> int | None:
        with self._lock:
            for index in range(self.active_index + 1, len(self.schedule)):
                state = str(dict(self.warm_states.get(index) or {}).get("state") or "")
                if state in {"queued", "warming", "warm"}:
                    continue
                target = dict(self.schedule[index].get("components") or {})
                missing = [
                    digest for role, digest in target.items()
                    if _digest(digest) and self._pool_key(str(role), _digest(digest)) not in self.component_pool
                ]
                if missing:
                    return index
                self.warm_states[index] = {
                    "state": "warm",
                    "model_path": str(self.schedule[index].get("model_path") or ""),
                    "completed_unix": time.time(),
                    "new_component_roles": [],
                    "already_leased_roles": sorted(target),
                    "reason": "all_components_already_leased",
                }
            return None

    def set_active_index(self, index: int, loaded: Any | None = None) -> None:
        with self._lock:
            self.active_index = int(index)
            self._event("active_index_changed", active_index=self.active_index)
        if loaded is not None:
            self.register_active_loaded(loaded)
        self.release_expired_components()

    def release_expired_components(self) -> list[str]:
        released: list[str] = []
        with self._lock:
            for digest, entry in list(self.component_pool.items()):
                future = [index for index in entry.consumers if index >= self.active_index]
                if future:
                    continue
                self.component_pool.pop(digest, None)
                released.append(digest)
            if released:
                self._event("expired_components_released", count=len(released), component_sha256=released)
        return released

    def index_for_model(self, model_path: str, *, after_current: bool = True) -> int | None:
        key = _path_key(model_path)
        start = self.active_index + 1 if after_current else 0
        with self._lock:
            for index in range(start, len(self.schedule)):
                if _path_key(self.schedule[index].get("model_path")) == key:
                    return index
            if not after_current:
                return None
            # Returning to an earlier model later in a cyclic test/profile may be
            # represented by a repeated schedule entry, not by wrapping implicitly.
            return None

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self.state = "invalidated"
            self.reason = str(reason or "invalidated")
            self.component_pool.clear()
            self._prepared_compositions.clear()
            self._event("lease_invalidated", reason=self.reason)

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            entries = {
                digest: entry.public_dict()
                for digest, entry in self.component_pool.items()
            }
            shared = [
                digest for digest, entry in self.component_pool.items()
                if len(entry.consumers) > 1
            ]
            return {
                "schema_version": self.schema_version,
                "generation": self.generation,
                "state": self.state,
                "reason": self.reason,
                "family": self.family,
                "provider_version": self.provider_version,
                "active_index": self.active_index,
                "schedule_length": len(self.schedule),
                "schedule": [
                    {
                        "index": index,
                        "model_path": str(item.get("model_path") or ""),
                        "family": str(item.get("family") or ""),
                        "composition_sha256": str(item.get("composition_sha256") or ""),
                        "component_roles": sorted(dict(item.get("components") or {})),
                    }
                    for index, item in enumerate(self.schedule)
                ],
                "component_pool": entries,
                "component_pool_count": len(entries),
                "shared_component_count": len(shared),
                "warm_states": {str(index): dict(value) for index, value in self.warm_states.items()},
                "prepared_compositions": {
                    str(index): entry.public_dict()
                    for index, entry in self._prepared_compositions.items()
                },
                "created_unix": self.created_unix,
                "updated_unix": self.updated_unix,
                "events": list(self.event_history[-30:]),
            }


def normalize_planned_schedule(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    source = raw if isinstance(raw, (list, tuple)) else [raw]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        if isinstance(item, str):
            payload = {"model_path": item}
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            continue
        model_path = str(payload.get("model_path") or payload.get("path") or "").strip()
        if not model_path:
            continue
        payload["model_path"] = model_path
        payload.setdefault("segment_index", index)
        result.append(payload)
    return result


__all__ = [
    "COMPOSITION_LEASE_SCHEMA_VERSION",
    "LeasedComponentEntry",
    "PreparedCompositionEntry",
    "CompositionExecutionLease",
    "normalize_planned_schedule",
]
