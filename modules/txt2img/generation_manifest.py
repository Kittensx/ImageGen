from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar


def _field_payload(value: Any) -> dict[str, Any]:
    """Return dataclass fields without deepcopying live runtime objects."""
    return {item.name: getattr(value, item.name) for item in fields(value)}


@dataclass
class AssetCandidateMatch:
    """
    A possible local or online match for a requested asset.
    Useful for warning panels, manual selection, and future auto-resolution.
    """

    display_name: str = ""
    path: str = ""
    provider: str = "local"
    match_type: str = "unknown"
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _field_payload(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AssetCandidateMatch":
        data = data or {}
        return cls(
            display_name=str(data.get("display_name", "")),
            path=str(data.get("path", "")),
            provider=str(data.get("provider", "local")),
            match_type=str(data.get("match_type", "unknown")),
            confidence=(
                None if data.get("confidence") is None
                else float(data.get("confidence"))
            ),
            notes=[str(x) for x in data.get("notes", [])] if isinstance(data.get("notes"), list) else [],
            extra=dict(data.get("extra", {})) if isinstance(data.get("extra"), dict) else {},
        )


@dataclass
class AssetReference:
    """
    Tracks an asset as requested by the original generation and how it was
    resolved on the current system.
    """

    asset_type: str = ""
    provider: str = "local"

    # Requested/original asset details
    requested_display_name: str = ""
    requested_filename: str = ""
    requested_path: str = ""
    requested_identifier: str = ""
    requested_version: str = ""
    requested_hash: str = ""
    requested_hash_type: str = ""

    # Resolved/current asset details
    resolved_display_name: str = ""
    resolved_filename: str = ""
    resolved_path: str = ""
    resolved_identifier: str = ""
    resolved_version: str = ""
    resolved_hash: str = ""
    resolved_hash_type: str = ""

    # Resolution state
    resolution_status: str = "unresolved"
    resolution_method: str = ""
    action_taken: str = ""
    was_found: bool = False
    was_used_for_generation: bool = False

    # Behavior flags
    is_required_for_rerun: bool = False
    is_optional: bool = False
    should_autoload: bool = True
    allow_ui_fallback: bool = True
    allow_online_search: bool = False
    user_override_allowed: bool = True

    # Provenance / search support
    source_url: str = ""
    repo_id: str = ""
    repo_revision: str = ""
    search_locations: list[str] = field(default_factory=list)

    # UI/debug support
    warning_messages: list[str] = field(default_factory=list)
    info_messages: list[str] = field(default_factory=list)
    candidate_matches: list[AssetCandidateMatch] = field(default_factory=list)

    # Future-safe overflow
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _field_payload(self)
        payload["candidate_matches"] = [m.to_dict() for m in self.candidate_matches]
        return payload
    
    @staticmethod
    def create_asset(asset_type: str) -> AssetReference:
        return AssetReference(asset_type=asset_type)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AssetReference":
        data = data or {}
        matches_raw = data.get("candidate_matches", [])
        candidate_matches = [
            AssetCandidateMatch.from_dict(item)
            for item in matches_raw
            if isinstance(item, dict)
        ]

        return cls(
            asset_type=str(data.get("asset_type", "")),
            provider=str(data.get("provider", "local")),

            requested_display_name=str(data.get("requested_display_name", "")),
            requested_filename=str(data.get("requested_filename", "")),
            requested_path=str(data.get("requested_path", "")),
            requested_identifier=str(data.get("requested_identifier", "")),
            requested_version=str(data.get("requested_version", "")),
            requested_hash=str(data.get("requested_hash", "")),
            requested_hash_type=str(data.get("requested_hash_type", "")),

            resolved_display_name=str(data.get("resolved_display_name", "")),
            resolved_filename=str(data.get("resolved_filename", "")),
            resolved_path=str(data.get("resolved_path", "")),
            resolved_identifier=str(data.get("resolved_identifier", "")),
            resolved_version=str(data.get("resolved_version", "")),
            resolved_hash=str(data.get("resolved_hash", "")),
            resolved_hash_type=str(data.get("resolved_hash_type", "")),

            resolution_status=str(data.get("resolution_status", "unresolved")),
            resolution_method=str(data.get("resolution_method", "")),
            action_taken=str(data.get("action_taken", "")),
            was_found=bool(data.get("was_found", False)),
            was_used_for_generation=bool(data.get("was_used_for_generation", False)),

            is_required_for_rerun=bool(data.get("is_required_for_rerun", False)),
            is_optional=bool(data.get("is_optional", False)),
            should_autoload=bool(data.get("should_autoload", True)),
            allow_ui_fallback=bool(data.get("allow_ui_fallback", True)),
            allow_online_search=bool(data.get("allow_online_search", False)),
            user_override_allowed=bool(data.get("user_override_allowed", True)),

            source_url=str(data.get("source_url", "")),
            repo_id=str(data.get("repo_id", "")),
            repo_revision=str(data.get("repo_revision", "")),
            search_locations=[
                str(x) for x in data.get("search_locations", [])
            ] if isinstance(data.get("search_locations"), list) else [],

            warning_messages=[
                str(x) for x in data.get("warning_messages", [])
            ] if isinstance(data.get("warning_messages"), list) else [],
            info_messages=[
                str(x) for x in data.get("info_messages", [])
            ] if isinstance(data.get("info_messages"), list) else [],
            candidate_matches=candidate_matches,

            extra=dict(data.get("extra", {})) if isinstance(data.get("extra"), dict) else {},
        )

    @property
    def requested_label(self) -> str:
        return (
            self.requested_display_name
            or self.requested_filename
            or self.requested_identifier
            or self.requested_path
        )

    @property
    def resolved_label(self) -> str:
        return (
            self.resolved_display_name
            or self.resolved_filename
            or self.resolved_identifier
            or self.resolved_path
        )

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status not in {"unresolved", "missing"} and (
            self.was_found or bool(self.resolved_path or self.resolved_filename or self.resolved_identifier)
        )

    def mark_missing(self, warning: str | None = None) -> None:
        self.was_found = False
        self.resolution_status = "missing"
        self.action_taken = self.action_taken or "not_used"
        if warning:
            self.warning_messages.append(warning)

    def mark_resolved(
        self,
        resolved_path: str = "",
        resolved_filename: str = "",
        resolved_display_name: str = "",
        method: str = "",
        status: str = "resolved",
    ) -> None:
        if resolved_path:
            self.resolved_path = resolved_path
        if resolved_filename:
            self.resolved_filename = resolved_filename
        if resolved_display_name:
            self.resolved_display_name = resolved_display_name

        self.was_found = True
        self.resolution_method = method or self.resolution_method
        self.resolution_status = status

    def add_candidate_match(self, match: AssetCandidateMatch) -> None:
        self.candidate_matches.append(match)

    def add_warning(self, message: str) -> None:
        if message:
            self.warning_messages.append(message)

    def add_info(self, message: str) -> None:
        if message:
            self.info_messages.append(message)

    def choose_ui_fallback(
        self,
        resolved_display_name: str = "",
        resolved_path: str = "",
    ) -> None:
        self.resolved_display_name = resolved_display_name or self.resolved_display_name
        self.resolved_path = resolved_path or self.resolved_path
        self.resolution_status = "missing_used_ui_default"
        self.resolution_method = "ui_fallback"
        self.action_taken = "used_ui_default"
        self.was_found = False
        self.was_used_for_generation = True


    
    
@dataclass
class RequiredForRerun:
    """
    Core fields required to recreate a generation request in a meaningful,
    reproducible way.
    """

    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    batch_size: int = 1
    batch_count: int = 1
    sampler_name: str = ""
    scheduler_name: str = ""
    model_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _field_payload(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RequiredForRerun":
        data = data or {}
        return cls(
            prompt=str(data.get("prompt", "")),
            negative_prompt=str(data.get("negative_prompt", "")),
            seed=int(data.get("seed", -1)),
            width=int(data.get("width", 512)),
            height=int(data.get("height", 512)),
            steps=int(data.get("steps", 20)),
            cfg_scale=float(data.get("cfg_scale", 7.0)),
            batch_size=int(data.get("batch_size", 1)),
            batch_count=int(data.get("batch_count", 1)),
            sampler_name=str(data.get("sampler_name", "")),
            scheduler_name=str(data.get("scheduler_name", "")),
            model_path=str(data.get("model_path", "")),
        )


@dataclass
class OptionalForRerun:
    """
    Optional inputs that may be used for rerun if the caller wants a closer
    recreation of the original generation.

    Keep these as flexible dictionaries so scheduler/sampler-specific settings
    can evolve without forcing constant dataclass rewrites.
    """

    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)

    compatibility_mode: str | None = None
    clip_skip: int | None = None
    guidance_rescale: float | None = None
    tiling: bool | None = None

    # Safe place for future fields that do not yet deserve first-class members.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _field_payload(self)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OptionalForRerun":
        data = data or {}
        known_keys = {
            "scheduler_kwargs",
            "sampler_kwargs",
            "compatibility_mode",
            "clip_skip",
            "guidance_rescale",
            "tiling",
            "extra",
        }

        extra = dict(data.get("extra", {}))
        for key, value in data.items():
            if key not in known_keys:
                extra[key] = value

        scheduler_kwargs = data.get("scheduler_kwargs", {})
        sampler_kwargs = data.get("sampler_kwargs", {})

        return cls(
            scheduler_kwargs=dict(scheduler_kwargs) if isinstance(scheduler_kwargs, dict) else {},
            sampler_kwargs=dict(sampler_kwargs) if isinstance(sampler_kwargs, dict) else {},
            compatibility_mode=(
                None if data.get("compatibility_mode") is None
                else str(data.get("compatibility_mode"))
            ),
            clip_skip=(
                None if data.get("clip_skip") is None
                else int(data.get("clip_skip"))
            ),
            guidance_rescale=(
                None if data.get("guidance_rescale") is None
                else float(data.get("guidance_rescale"))
            ),
            tiling=(
                None if data.get("tiling") is None
                else bool(data.get("tiling"))
            ),
            extra=extra,
        )


@dataclass
class RuntimeInfo:
    """
    Information observed during generation. These values are useful for audit,
    debugging, and analysis, but should generally not be fed back into a rerun
    request automatically.
    """

    effective_steps: int | None = None
    scheduler_step_override_applied: bool | None = None
    active_blend_methods: list[str] = field(default_factory=list)
    active_blend_weights: list[float] = field(default_factory=list)
    tail_features_used: dict[str, Any] = field(default_factory=dict)
    predicted_stop_step: int | None = None

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    device: str | None = None
    generation_time_sec: float | None = None
    output_image_path: str | None = None
    output_txt_path: str | None = None
    output_json_path: str | None = None

    # Safe overflow bucket for future runtime/debug fields.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _field_payload(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeInfo":
        data = data or {}
        known_keys = {
            "effective_steps",
            "scheduler_step_override_applied",
            "active_blend_methods",
            "active_blend_weights",
            "tail_features_used",
            "predicted_stop_step",
            "timestamp",
            "device",
            "generation_time_sec",
            "output_image_path",
            "output_txt_path",
            "output_json_path",
            "extra",
        }

        extra = dict(data.get("extra", {}))
        for key, value in data.items():
            if key not in known_keys:
                extra[key] = value

        active_blend_methods = data.get("active_blend_methods", [])
        active_blend_weights = data.get("active_blend_weights", [])
        tail_features_used = data.get("tail_features_used", {})

        return cls(
            effective_steps=(
                None if data.get("effective_steps") is None
                else int(data.get("effective_steps"))
            ),
            scheduler_step_override_applied=(
                None if data.get("scheduler_step_override_applied") is None
                else bool(data.get("scheduler_step_override_applied"))
            ),
            active_blend_methods=[
                str(item) for item in active_blend_methods
            ] if isinstance(active_blend_methods, list) else [],
            active_blend_weights=[
                float(item) for item in active_blend_weights
            ] if isinstance(active_blend_weights, list) else [],
            tail_features_used=(
                dict(tail_features_used) if isinstance(tail_features_used, dict) else {}
            ),
            predicted_stop_step=(
                None if data.get("predicted_stop_step") is None
                else int(data.get("predicted_stop_step"))
            ),
            timestamp=str(data.get("timestamp", datetime.now().isoformat(timespec="seconds"))),
            device=(
                None if data.get("device") is None
                else str(data.get("device"))
            ),
            generation_time_sec=(
                None if data.get("generation_time_sec") is None
                else float(data.get("generation_time_sec"))
            ),
            output_image_path=(
                None if data.get("output_image_path") is None
                else str(data.get("output_image_path"))
            ),
            output_txt_path=(
                None if data.get("output_txt_path") is None
                else str(data.get("output_txt_path"))
            ),
            output_json_path=(
                None if data.get("output_json_path") is None
                else str(data.get("output_json_path"))
            ),
            extra=extra,
        )


@dataclass
class GenerationManifest:
    """
    Main manifest object for generation reproducibility and auditing.

    Design goals:
    - Strong typing for required rerun fields
    - Flexible expansion points for scheduler/sampler evolution
    - Safe distinction between inputs and runtime outputs
    """

    CURRENT_VERSION: ClassVar[str] = "1.0"

    manifest_version: str = CURRENT_VERSION
    manifest_type: str = "generation_manifest"

    required_for_rerun: RequiredForRerun = field(default_factory=RequiredForRerun)
    optional_for_rerun: OptionalForRerun = field(default_factory=OptionalForRerun)
    runtime_info: RuntimeInfo = field(default_factory=RuntimeInfo)    
    base_model: AssetReference = field(default_factory=lambda: AssetReference(asset_type="base_model"))
    vae: AssetReference = field(default_factory=lambda: AssetReference(asset_type="vae"))
    loras: list[AssetReference] = field(default_factory=list)
    embeddings: list[AssetReference] = field(default_factory=list)
    hypernetworks: list[AssetReference] = field(default_factory=list)
    extras: list[AssetReference] = field(default_factory=list)

    # Top-level overflow for future compatibility or cross-cutting metadata.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        include_optional_for_rerun: bool = True,
        include_runtime_info: bool = True,
        include_assets: bool = True,
        include_top_level_extra: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "manifest_type": self.manifest_type,
            "required_for_rerun": self.required_for_rerun.to_dict(),
        }

        if include_optional_for_rerun:
            payload["optional_for_rerun"] = self.optional_for_rerun.to_dict()

        if include_runtime_info:
            payload["runtime_info"] = self.runtime_info.to_dict()

        if include_assets:
            payload["base_model"] = self.base_model.to_dict()
            payload["vae"] = self.vae.to_dict()
            payload["loras"] = self._serialize_asset_list(self.loras)
            payload["embeddings"] = self._serialize_asset_list(self.embeddings)
            payload["hypernetworks"] = self._serialize_asset_list(self.hypernetworks)
            payload["extras"] = self._serialize_asset_list(self.extras)

        if include_top_level_extra and self.extra:
            payload["extra"] = dict(self.extra)

        return payload

    
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationManifest":
        data = data or {}
        known_keys = {
            "manifest_version",
            "manifest_type",
            "required_for_rerun",
            "optional_for_rerun",
            "runtime_info",
            "base_model",
            "vae",
            "loras",
            "embeddings",
            "hypernetworks",
            "extras",
            "extra",
        }

        extra = dict(data.get("extra", {}))
        for key, value in data.items():
            if key not in known_keys:
                extra[key] = value

        return cls(
            manifest_version=str(data.get("manifest_version", cls.CURRENT_VERSION)),
            manifest_type=str(data.get("manifest_type", "generation_manifest")),
            required_for_rerun=RequiredForRerun.from_dict(
                data.get("required_for_rerun", {})
            ),
            optional_for_rerun=OptionalForRerun.from_dict(
                data.get("optional_for_rerun", {})
            ),
            runtime_info=RuntimeInfo.from_dict(
                data.get("runtime_info", {})
            ),
            base_model=AssetReference.from_dict(data.get("base_model", {"asset_type": "base_model"})),
            vae=AssetReference.from_dict(data.get("vae", {"asset_type": "vae"})),
            loras=cls._deserialize_asset_list(data.get("loras", [])),
            embeddings=cls._deserialize_asset_list(data.get("embeddings", [])),
            hypernetworks=cls._deserialize_asset_list(data.get("hypernetworks", [])),
            extras=cls._deserialize_asset_list(data.get("extras", [])),
            extra=extra,
        )
    
    def _serialize_asset_list(self, assets: list[AssetReference]) -> list[dict[str, Any]]:
        return [asset.to_dict() for asset in assets]

    @staticmethod
    def _deserialize_asset_list(data: Any) -> list[AssetReference]:
        if not isinstance(data, list):
            return []
        return [
            AssetReference.from_dict(item)
            for item in data
            if isinstance(item, dict)
        ]
    
    def to_rerun_payload(
        self,
        include_optional_for_rerun: bool = True,
    ) -> dict[str, Any]:
        """
        Build a safe rerun payload.

        This intentionally excludes runtime_info.
        """
        payload = self.required_for_rerun.to_dict()

        if include_optional_for_rerun:
            optional_dict = self.optional_for_rerun.to_dict()

            scheduler_kwargs = optional_dict.pop("scheduler_kwargs", {})
            sampler_kwargs = optional_dict.pop("sampler_kwargs", {})
            extra = optional_dict.pop("extra", {})

            if scheduler_kwargs:
                payload["scheduler_kwargs"] = dict(scheduler_kwargs)

            if sampler_kwargs:
                payload["sampler_kwargs"] = dict(sampler_kwargs)

            for key, value in optional_dict.items():
                if value is not None:
                    payload[key] = value

            for key, value in extra.items():
                payload[key] = value

        return payload

    def update_runtime_paths(
        self,
        image_path: str | None = None,
        txt_path: str | None = None,
        json_path: str | None = None,
    ) -> None:
        if image_path is not None:
            self.runtime_info.output_image_path = image_path
        if txt_path is not None:
            self.runtime_info.output_txt_path = txt_path
        if json_path is not None:
            self.runtime_info.output_json_path = json_path

    def add_optional_field(self, key: str, value: Any) -> None:
        self.optional_for_rerun.extra[key] = value

    def add_runtime_field(self, key: str, value: Any) -> None:
        self.runtime_info.extra[key] = value

    def add_top_level_field(self, key: str, value: Any) -> None:
        self.extra[key] = value


