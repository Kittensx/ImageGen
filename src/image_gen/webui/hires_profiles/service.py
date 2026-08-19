from __future__ import annotations

import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION,
    HiresDefaultAssignment,
    HiresProfile,
    HiresProfileSaveManifest,
    HiresProfileValidationError,
)
from .builtins import build_builtin_auto_profiles
from .schema import HiresProfileSchemaRegistry


_SAFE_ID = re.compile(r"[^a-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    token = _SAFE_ID.sub("-", str(value or "").strip().casefold()).strip("-._")
    return token or "profile"


class HiresProfileService:
    """JSON-backed hires-profile persistence with serializer-driven inspection."""

    def __init__(
        self,
        webui_root: str | Path,
        *,
        builtin_profiles: Iterable[HiresProfile | Mapping[str, Any]] = (),
        schema_registry: HiresProfileSchemaRegistry | None = None,
    ) -> None:
        self.root = Path(webui_root).expanduser().resolve() / "hires-profiles"
        self.user_dir = self.root / "user"
        self.assignments_path = self.root / "default-assignments.json"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema_registry or HiresProfileSchemaRegistry()
        self._builtins: dict[str, HiresProfile] = {}
        resolved_builtins = tuple(builtin_profiles) or tuple(build_builtin_auto_profiles(self.schema))
        for item in resolved_builtins:
            profile = item if isinstance(item, HiresProfile) else HiresProfile.from_mapping(item)
            if not profile.read_only or profile.source != "builtin":
                profile = HiresProfile(
                    **{
                        **profile.__dict__,
                        "source": "builtin",
                        "read_only": True,
                    }
                )
            self._builtins[profile.profile_id] = profile

    @staticmethod
    def _read(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            temp_path = Path(handle.name)
        temp_path.replace(path)

    def _profile_path(self, profile_id: str) -> Path:
        return self.user_dir / f"{_slug(profile_id)}.json"

    def _new_profile_id(self, name: str) -> str:
        return f"user.{_slug(name)}.{uuid.uuid4().hex[:12]}"

    def list_profiles(self) -> list[HiresProfile]:
        profiles = list(self._builtins.values())
        for path in sorted(self.user_dir.glob("*.json"), key=lambda item: item.name.casefold()):
            payload = self._read(path, {})
            if not isinstance(payload, Mapping):
                continue
            try:
                profiles.append(HiresProfile.from_mapping(payload))
            except ValueError:
                continue
        return sorted(profiles, key=lambda item: (item.source != "builtin", item.name.casefold(), item.profile_id))

    def get_profile(self, profile_id: str) -> HiresProfile | None:
        selected = str(profile_id or "").strip()
        if selected in self._builtins:
            return self._builtins[selected]
        path = self._profile_path(selected)
        payload = self._read(path, {})
        if not isinstance(payload, Mapping) or not payload:
            return None
        try:
            profile = HiresProfile.from_mapping(payload)
        except ValueError:
            return None
        return profile if profile.profile_id == selected else None

    def _baseline_values(self, baseline_profile_id: str) -> dict[str, Any]:
        selected = str(baseline_profile_id or "").strip()
        if not selected:
            return self.schema.default_values()
        baseline = self.get_profile(selected)
        if baseline is None:
            raise ValueError(f"Unknown hires baseline profile: {selected!r}")
        return {**self.schema.default_values(), **baseline.values}

    def preview_save(
        self,
        *,
        name: str,
        values: Mapping[str, Any],
        included_fields: Sequence[str] | None = None,
        profile_id: str = "",
        baseline_profile_id: str = "",
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> tuple[dict[str, Any], HiresProfileSaveManifest]:
        profile_name = str(name or "").strip()
        if not profile_name:
            raise ValueError("Hires profile name is required.")
        resolved_id = str(profile_id or "").strip() or self._new_profile_id(profile_name)
        baseline_values = self._baseline_values(baseline_profile_id)
        try:
            normalized, rejected = self.schema.normalize_values(
                values,
                included_fields=included_fields,
                choice_overrides=choice_overrides,
            )
        except ValueError as exc:
            requested = tuple(str(key) for key in (included_fields if included_fields is not None else values.keys()))
            rejected = tuple(sorted(key for key in set(values) | set(requested) if key not in self.schema.eligible_keys))
            manifest = self.schema.build_save_manifest(
                profile_id=resolved_id,
                profile_name=profile_name,
                values={},
                included_fields=(),
                baseline_profile_id=baseline_profile_id,
                baseline_values=baseline_values,
                rejected_fields=rejected,
                incoming_values=values,
                choice_overrides=choice_overrides,
            )
            raise HiresProfileValidationError(str(exc), manifest=manifest) from exc

        manifest = self.schema.build_save_manifest(
            profile_id=resolved_id,
            profile_name=profile_name,
            values=normalized,
            included_fields=tuple(normalized),
            baseline_profile_id=baseline_profile_id,
            baseline_values=baseline_values,
            rejected_fields=rejected,
            incoming_values=values,
            choice_overrides=choice_overrides,
        )
        if rejected:
            raise HiresProfileValidationError(
                "Hires profile contains fields outside the hires persistence contract: "
                + ", ".join(rejected),
                manifest=manifest,
            )
        return normalized, manifest

    def save_profile(
        self,
        *,
        name: str,
        values: Mapping[str, Any],
        included_fields: Sequence[str] | None = None,
        profile_id: str = "",
        description: str = "",
        compatibility: Mapping[str, Any] | None = None,
        baseline_profile_id: str = "",
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> tuple[HiresProfile, HiresProfileSaveManifest]:
        selected_id = str(profile_id or "").strip()
        existing = self.get_profile(selected_id) if selected_id else None
        if existing is not None and existing.read_only:
            raise ValueError("Built-in hires profiles are read-only; duplicate the profile before editing it.")
        normalized, manifest = self.preview_save(
            name=name,
            values=values,
            included_fields=included_fields,
            profile_id=selected_id,
            baseline_profile_id=baseline_profile_id,
            choice_overrides=choice_overrides,
        )
        now = _utc_now()
        created_at = existing.created_at if existing and existing.created_at else now
        profile = HiresProfile(
            profile_id=manifest.profile_id,
            name=str(name).strip(),
            description=str(description or ""),
            source="user",
            read_only=False,
            included_fields=tuple(sorted(normalized)),
            values=normalized,
            compatibility=dict(compatibility or {}),
            baseline_profile_id=str(baseline_profile_id or ""),
            created_at=created_at,
            updated_at=now,
        )
        # The manifest above was produced from the exact normalized values and field
        # ownership immediately before this persistent write.
        self._write(self._profile_path(profile.profile_id), profile.to_dict())
        return profile, manifest

    def inspect_profile(
        self,
        profile_id: str,
        *,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> HiresProfileSaveManifest:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown hires profile: {profile_id}")
        rejected = tuple(
            sorted(key for key in profile.values if key not in self.schema.eligible_keys)
        )
        return self.schema.build_save_manifest(
            profile_id=profile.profile_id,
            profile_name=profile.name,
            values=profile.values,
            included_fields=profile.included_fields,
            baseline_profile_id=profile.baseline_profile_id,
            baseline_values=self._baseline_values(profile.baseline_profile_id),
            rejected_fields=rejected,
            incoming_values=profile.values,
            choice_overrides=choice_overrides,
        )

    def duplicate_profile(
        self,
        profile_id: str,
        *,
        name: str,
        choice_overrides: Mapping[str, Sequence[Any]] | None = None,
    ) -> tuple[HiresProfile, HiresProfileSaveManifest]:
        source = self.get_profile(profile_id)
        if source is None:
            raise KeyError(f"Unknown hires profile: {profile_id}")
        return self.save_profile(
            name=name,
            values=source.values,
            included_fields=source.included_fields,
            description=source.description,
            compatibility=source.compatibility,
            baseline_profile_id=source.baseline_profile_id,
            choice_overrides=choice_overrides,
        )

    def delete_profile(self, profile_id: str) -> dict[str, Any]:
        selected = str(profile_id or "").strip()
        profile = self.get_profile(selected)
        if profile is None:
            return {"deleted": False, "assignments_removed": 0}
        if profile.read_only:
            raise ValueError("Built-in hires profiles cannot be deleted.")
        try:
            self._profile_path(selected).unlink()
        except OSError:
            return {"deleted": False, "assignments_removed": 0}
        assignments = self.list_default_assignments()
        retained = [item for item in assignments if item.profile_id != selected]
        removed = len(assignments) - len(retained)
        if removed:
            self._write_assignments(retained)
        return {"deleted": True, "assignments_removed": removed}

    def _read_assignments(self) -> list[HiresDefaultAssignment]:
        payload = self._read(
            self.assignments_path,
            {"schema_version": HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION, "assignments": []},
        )
        if not isinstance(payload, Mapping):
            return []
        rows = payload.get("assignments") if isinstance(payload.get("assignments"), list) else []
        output: list[HiresDefaultAssignment] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                output.append(HiresDefaultAssignment.from_mapping(row))
            except ValueError:
                continue
        return output

    def _write_assignments(self, assignments: Sequence[HiresDefaultAssignment]) -> None:
        self._write(
            self.assignments_path,
            {
                "schema_version": HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION,
                "assignments": [item.to_dict() for item in assignments],
            },
        )

    def list_default_assignments(self) -> list[HiresDefaultAssignment]:
        return sorted(self._read_assignments(), key=lambda item: item.assignment_key)

    def save_default_assignment(self, assignment: HiresDefaultAssignment | Mapping[str, Any]) -> HiresDefaultAssignment:
        incoming = assignment if isinstance(assignment, HiresDefaultAssignment) else HiresDefaultAssignment.from_mapping(assignment)
        if self.get_profile(incoming.profile_id) is None:
            raise ValueError(f"Default assignment references unknown hires profile {incoming.profile_id!r}.")
        existing = {item.assignment_key: item for item in self._read_assignments()}
        now = _utc_now()
        prior = existing.get(incoming.assignment_key)
        normalized = HiresDefaultAssignment(
            scope=incoming.scope,
            profile_id=incoming.profile_id,
            model_family=incoming.model_family,
            checkpoint_sha256=incoming.checkpoint_sha256,
            upscaler_sha256=incoming.upscaler_sha256,
            created_at=(prior.created_at if prior and prior.created_at else now),
            updated_at=now,
        )
        existing[normalized.assignment_key] = normalized
        self._write_assignments(list(existing.values()))
        return normalized

    def resolve_auto(self, context, *, choice_overrides=None):
        from .resolver import HiresAutoResolver, HiresResolutionContext

        choices = dict(choice_overrides or {})
        resolved_context = context if isinstance(context, HiresResolutionContext) else HiresResolutionContext.from_mapping(
            context,
            available_upscalers=choices.get("upscalers", ()),
            available_samplers=choices.get("samplers", ()),
            available_schedulers=choices.get("schedulers", ()),
        )
        return HiresAutoResolver(self).resolve(resolved_context)

    def delete_default_assignment(self, assignment_key: str) -> bool:
        selected = str(assignment_key or "").strip().casefold()
        assignments = self._read_assignments()
        retained = [item for item in assignments if item.assignment_key.casefold() != selected]
        if len(retained) == len(assignments):
            return False
        self._write_assignments(retained)
        return True


__all__ = ["HiresProfileService"]
