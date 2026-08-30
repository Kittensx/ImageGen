from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from image_gen.runtime.residency_policy import normalize_model_residency_mode
from image_gen.webui.default_assets import default_document, normalize_document
from image_gen.webui.theme.contracts import normalize_legacy_theme_palette
from image_gen.webui.theme.storage import normalize_theme_storage_settings

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")

DEFAULT_PANEL_SCALES: dict[str, int] = {
    "controls": 100,
    "output_viewer": 100,
    "recent_outputs": 100,
    "live_preview": 100,
    "active_prompt_assets": 100,
    "memory_status": 100,
    "runtime_status": 100,
    "queue": 100,
    "recent_runs": 100,
    "prompt_presets": 100,
    "model_refresh": 100,
    "maintenance": 100,
    "startup_defaults": 100,
}

FAILSAFE_APPLICATION_DEFAULTS: dict[str, Any] = {
    "restore_last_session": True,
    "ui_scale": 100,
    "density": "comfortable",
    "theme": "dark",
    "theme_palette": {
        "accent": {"name": "Sky Blue", "color": "#179ee7"},
        "surface": {"name": "Charcoal", "color": "#111d29"},
        "typography": {
            "font_family": "Inter",
            "primary_button_text": "#ffffff",
            "secondary_button_text": "#d5f1ff",
        },
        "semantic": {
            "surface_secondary": "#172431",
            "component_surface": "#111d29",
            "component_border": "#2b4358",
            "component_accent": "#179ee7",
            "text_primary": "#f4f9fd",
            "text_secondary": "#9db2c4",
        },
    },
    "theme_storage": {
        "theme_library_root": "",
        "theme_user_root": "",
        "theme_cache_root": "",
        "theme_preview_root": "",
    },
    "theme_contrast_warnings_enabled": True,
    "live_preview_enabled": True,
    "live_preview_mode": "fast",
    "live_preview_interval": 1,
    "live_preview_width": 384,
    "live_preview_format": "webp",
    "live_preview_keep_history": "current_job",
    "live_preview_batch_index": 0,
    "live_preview_quality": 78,
    "live_preview_adaptive_throttle": True,
    "live_preview_adaptive_target_ratio": 0.75,
    "live_preview_adaptive_recovery_ratio": 0.40,
    "live_preview_adaptive_max_interval": 8,
    "live_preview_adaptive_window": 6,
    "live_preview_adaptive_suspend_on_overhead": False,
    "cfg_lab_enabled": False,
    "live_preview_cfg_visual_enabled": False,
    "preferred_hires_upscaler_id": "",
    "cfg_lab_user_presets": {},
    "diagnostics_mode": "failures_only",
    "diagnostic_decode_enabled": False,
    "live_preview_cleanup_enabled": True,
    "live_preview_retention_days": 7,
    "live_preview_retention_jobs": 24,
    "live_preview_disk_budget_mb": 1024,
    "memory_policy": "auto",
    "model_residency_mode": "managed",
    "memory_vram_safety_margin_mb": 1024,
    "memory_retain_checkpoint_between_jobs": True,
    "memory_retain_vae_between_jobs": True,
    "model_runtime_retain_text_encoder_between_jobs": True,
    "memory_pinned_cpu_memory": False,
    "memory_allow_tiled_vae_fallback": True,
    "memory_allow_preview_suspension_on_oom": True,
    "checkpoint_startup_mode": "last_used",
    "checkpoint_startup_path": "",
    "checkpoint_preload_on_startup": True,
    "lora_prompt_integration_mode": "visual",
    "lora_auto_scan_unknown_on_startup": True,
    "attention_backend": "auto",
    "allocator_options": {"PYTORCH_CUDA_ALLOC_CONF": ""},
    "runtime_job_overrides": {},
    "mslk_fmha": {
        "policy": "blackwell_safe",
        "debug": "",
        "block_n": "",
        "block_m": "",
        "num_warps": "",
        "num_stages": "",
        "experimental_head_dims": "",
    },
    "phase13_troubleshooting_lock_removed": True,
    "recent_outputs_background_refresh_enabled": True,
    "recent_outputs_refresh_ms_active": 4000,
    "recent_outputs_refresh_ms_idle": 12000,
    "ui_layout": {
        "workspace_layout_version": 1,
        "left_column_width": 330,
        "right_column_width": 360,
        "gallery_panel_height": 132,
        "live_preview_panel_height": 360,
        "live_preview_collapsed": False,
        "follow_newest_output": False,
        "startup_defaults_open": False,
        "startup_defaults_pinned": False,
        "startup_defaults_width": 300,
        "panel_zones": {
            "left": ["generation_controls"],
            "center": ["output_viewer", "recent_outputs"],
            "right": [
                "live_preview",
                "active_prompt_assets",
                "memory_status",
                "runtime_status",
                "queue",
                "recent_runs",
                "prompt_presets",
                "model_refresh",
                "maintenance",
            ],
        },
        "collapsed_panels": [],
        "panel_scales": dict(DEFAULT_PANEL_SCALES),
    },
    "ui_scale_layout_defaults": {},
    "recent_outputs_browser": {
        "time_window": "72",
        "custom_hours": 24,
        "include_subfolders": True,
        "source_paths": [],
        "require_metadata_for_external": True,
    },
}

FORCED_LIVE_PREVIEW_MODE = "fast"
DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL = 10


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value or "").strip()).strip(" .")
    return cleaned or "Untitled"


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        output = {key: _deep_merge(value, override[key]) if key in override else value for key, value in base.items()}
        for key, value in override.items():
            if key not in output:
                output[key] = value
        return output
    return override


class WebUIStore:
    """Small JSON-backed store for sessions, profiles, and prompt presets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.session_dir = self.root / "session"
        self.profile_dir = self.root / "profiles"
        self.prompt_dir = self.root / "prompt-presets"
        self.prompt_shortcut_profile_dir = self.root / "prompt-shortcut-profiles"
        self.prompt_parser_preset_dir = self.root / "prompt-parser-presets"
        self.settings_dir = self.root / "settings"
        self.recent_outputs_dir = self.root / "recent-outputs"
        self.default_assets_dir = self.root / "default-assets"
        self._hires_profile_service = None
        for directory in (
            self.session_dir,
            self.profile_dir,
            self.prompt_dir,
            self.prompt_shortcut_profile_dir,
            self.prompt_parser_preset_dir,
            self.settings_dir,
            self.recent_outputs_dir,
            self.default_assets_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def hires_profile_service(self):
        # WebUIStore is imported during process bootstrap before CUDA allocator
        # options are applied. Keep the hires-profile subsystem lazy because its
        # GenerationRequest schema imports Torch.
        if self._hires_profile_service is None:
            from image_gen.webui.hires_profiles import HiresProfileService, build_builtin_auto_profiles

            self._hires_profile_service = HiresProfileService(
                self.root,
                builtin_profiles=build_builtin_auto_profiles(),
            )
        return self._hires_profile_service

    @staticmethod
    def _read(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
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

    @staticmethod
    def _packaged_defaults_path() -> Path:
        return Path(__file__).resolve().parent / "defaults" / "application.json"

    @staticmethod
    def _normalize_panel_scales(value: Any) -> dict[str, int]:
        stored = value if isinstance(value, dict) else {}
        return {
            **DEFAULT_PANEL_SCALES,
            **{key: stored[key] for key in stored if key in DEFAULT_PANEL_SCALES},
        }

    @classmethod
    def _normalize_layout(cls, value: Any) -> dict[str, Any]:
        stored = value if isinstance(value, dict) else {}
        panel_zones = stored.get("panel_zones") if isinstance(stored.get("panel_zones"), dict) else {}

        def _zone_items(name: str, fallback: list[str]) -> list[str]:
            raw = panel_zones.get(name)
            return list(raw) if isinstance(raw, list) else list(fallback)

        normalized_zones = {
            "left": _zone_items("left", ["generation_controls"]),
            "center": _zone_items("center", ["output_viewer", "recent_outputs"]),
            "right": _zone_items("right", [
                "live_preview",
                "active_prompt_assets",
                "memory_status",
                "runtime_status",
                "queue",
                "recent_runs",
                "prompt_presets",
                "model_refresh",
                "maintenance",
            ]),
        }
        return {
            "left_column_width": 330,
            "right_column_width": 360,
            "gallery_panel_height": 132,
            "live_preview_panel_height": 360,
            "live_preview_collapsed": False,
            "follow_newest_output": False,
            "startup_defaults_open": False,
            "startup_defaults_pinned": False,
            "startup_defaults_width": 300,
            **stored,
            "workspace_layout_version": 1,
            "startup_defaults_open": bool(stored.get("startup_defaults_open", False)),
            "startup_defaults_pinned": bool(stored.get("startup_defaults_pinned", False)),
            "panel_zones": normalized_zones,
            "collapsed_panels": list(stored.get("collapsed_panels") or []),
            "panel_scales": cls._normalize_panel_scales(stored.get("panel_scales")),
        }

    @classmethod
    def _normalize_recent_outputs_browser(cls, value: Any) -> dict[str, Any]:
        stored = value if isinstance(value, dict) else {}
        return {
            "time_window": "72",
            "custom_hours": 24,
            "include_subfolders": True,
            "source_paths": [],
            "require_metadata_for_external": True,
            **stored,
        }

    @classmethod
    def _normalize_theme_palette(cls, value: Any) -> dict[str, Any]:
        return normalize_legacy_theme_palette(value)

    @classmethod
    def _normalize_scale_layout_defaults(cls, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        output: dict[str, dict[str, Any]] = {}
        for key, layout in value.items():
            try:
                scale_key = str(int(round(float(key))))
            except (TypeError, ValueError):
                continue
            if not isinstance(layout, dict):
                continue
            output[scale_key] = cls._normalize_layout(layout)
        return output

    @classmethod
    def _normalize_application_settings(cls, value: Any) -> dict[str, Any]:
        merged = _deep_merge(FAILSAFE_APPLICATION_DEFAULTS, value if isinstance(value, dict) else {})
        merged["live_preview_mode"] = FORCED_LIVE_PREVIEW_MODE
        merged["live_preview_mode_locked"] = True
        try:
            merged["live_preview_interval"] = max(1, int(merged.get("live_preview_interval", DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL) or DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL))
        except (TypeError, ValueError):
            merged["live_preview_interval"] = DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL
        merged["theme_palette"] = cls._normalize_theme_palette(merged.get("theme_palette"))
        merged["theme_storage"] = normalize_theme_storage_settings(merged.get("theme_storage"))
        merged["ui_layout"] = cls._normalize_layout(merged.get("ui_layout"))
        merged["ui_scale_layout_defaults"] = cls._normalize_scale_layout_defaults(merged.get("ui_scale_layout_defaults"))
        merged["recent_outputs_browser"] = cls._normalize_recent_outputs_browser(merged.get("recent_outputs_browser"))
        merged["runtime_job_overrides"] = (
            dict(merged.get("runtime_job_overrides"))
            if isinstance(merged.get("runtime_job_overrides"), dict)
            else {}
        )
        merged["model_residency_mode"] = normalize_model_residency_mode(
            merged.get("model_residency_mode")
        )
        for legacy_key in (
            "startup_model_behavior",
            "generation_worker_mode",
            "warm_worker_enabled",
            "warm_worker_auto_warm_on_model_selection",
            "warm_worker_execution_device",
            "warm_worker_retention_device",
            "model_preload_policy",
            "warm_worker_idle_unload_seconds",
            "warm_worker_allow_isolated_fallback",
            "warm_worker_retain_text_encoder_between_jobs",
        ):
            merged.pop(legacy_key, None)
        startup_mode = str(merged.get("checkpoint_startup_mode") or "last_used").strip().lower()
        if startup_mode not in {"last_used", "pinned_default", "none"}:
            startup_mode = "last_used"
        merged["checkpoint_startup_mode"] = startup_mode
        merged["checkpoint_startup_path"] = str(merged.get("checkpoint_startup_path") or "").strip()
        merged["checkpoint_preload_on_startup"] = bool(merged.get("checkpoint_preload_on_startup", True))
        lora_mode = str(merged.get("lora_prompt_integration_mode") or "visual").strip().lower()
        merged["lora_prompt_integration_mode"] = lora_mode if lora_mode in {"visual", "inline"} else "visual"
        merged["lora_auto_scan_unknown_on_startup"] = bool(merged.get("lora_auto_scan_unknown_on_startup", True))
        merged["cfg_lab_enabled"] = bool(merged.get("cfg_lab_enabled", False))
        if not merged["cfg_lab_enabled"]:
            merged["live_preview_cfg_visual_enabled"] = False
        diagnostics_mode = str(merged.get("diagnostics_mode") or "failures_only").strip().lower()
        merged["diagnostics_mode"] = diagnostics_mode if diagnostics_mode in {"off", "failures_only", "every_run", "deep_tensor"} else "failures_only"
        merged["diagnostic_decode_enabled"] = bool(merged.get("diagnostic_decode_enabled", False))
        merged["recent_outputs_background_refresh_enabled"] = bool(merged.get("recent_outputs_background_refresh_enabled", True))
        try:
            merged["recent_outputs_refresh_ms_active"] = max(1000, int(merged.get("recent_outputs_refresh_ms_active", 4000) or 4000))
        except (TypeError, ValueError):
            merged["recent_outputs_refresh_ms_active"] = 4000
        try:
            merged["recent_outputs_refresh_ms_idle"] = max(2000, int(merged.get("recent_outputs_refresh_ms_idle", 12000) or 12000))
        except (TypeError, ValueError):
            merged["recent_outputs_refresh_ms_idle"] = 12000
        try:
            current_scale = str(int(round(float(merged.get("ui_scale", 100) or 100))))
        except (TypeError, ValueError):
            current_scale = "100"
        merged["ui_layout_defaults"] = cls._normalize_layout(
            _deep_merge(
                merged.get("ui_layout"),
                merged["ui_scale_layout_defaults"].get(current_scale, {}),
            )
        )
        return merged

    @classmethod
    def _migrate_phase13_troubleshooting_settings(cls, value: Any) -> tuple[dict[str, Any], bool]:
        stored = dict(value) if isinstance(value, dict) else {}
        changed = False
        for legacy_key in (
            "external_vae_override_enabled",
            "startup_model_behavior",
            "generation_worker_mode",
            "warm_worker_enabled",
            "warm_worker_auto_warm_on_model_selection",
            "warm_worker_execution_device",
            "warm_worker_retention_device",
            "model_preload_policy",
            "warm_worker_idle_unload_seconds",
            "warm_worker_allow_isolated_fallback",
            "warm_worker_retain_text_encoder_between_jobs",
        ):
            if legacy_key in stored:
                stored.pop(legacy_key, None)
                changed = True
        if not bool(stored.get("phase13_troubleshooting_lock_removed", False)):
            stored["phase13_troubleshooting_lock_removed"] = True
            changed = True
        return stored, changed

    @classmethod
    def _migrate_hmr01_residency_settings(cls, value: Any) -> tuple[dict[str, Any], bool]:
        stored = dict(value) if isinstance(value, dict) else {}
        if "model_residency_mode" not in stored:
            return stored, False
        normalized = normalize_model_residency_mode(stored.get("model_residency_mode"))
        if stored.get("model_residency_mode") == normalized:
            return stored, False
        stored["model_residency_mode"] = normalized
        return stored, True

    def load_packaged_application_defaults(self) -> dict[str, Any]:
        packaged = self._read(self._packaged_defaults_path(), {})
        return self._normalize_application_settings(packaged)

    def load_session(self) -> dict[str, Any]:
        return dict(self._read(self.session_dir / "last-session.json", {}) or {})

    def save_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload or {})
        self._write(self.session_dir / "last-session.json", value)
        return value

    def load_application_settings(self) -> dict[str, Any]:
        packaged = self.load_packaged_application_defaults()
        settings_path = self.settings_dir / "application.json"
        stored, changed = self._migrate_phase13_troubleshooting_settings(self._read(settings_path, {}))
        stored, residency_changed = self._migrate_hmr01_residency_settings(stored)
        if changed or residency_changed:
            self._write(settings_path, stored)
        return self._normalize_application_settings(_deep_merge(packaged, stored))

    def inherit_runtime_startup_profile(self) -> dict[str, Any]:
        """Clear explicit per-job runtime overrides without changing startup settings."""

        settings_path = self.settings_dir / "application.json"
        stored = self._read(settings_path, {})
        persisted = dict(stored) if isinstance(stored, dict) else {}
        persisted["runtime_job_overrides"] = {}
        persisted["phase13_troubleshooting_lock_removed"] = True
        self._write(settings_path, persisted)
        return self.load_application_settings()

    def save_application_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        incoming = dict(payload or {})
        if "model_residency_mode" in incoming:
            incoming["model_residency_mode"] = normalize_model_residency_mode(
                incoming.get("model_residency_mode")
            )
        if "theme_palette" in incoming:
            incoming["theme_palette"] = self._normalize_theme_palette(incoming.get("theme_palette"))
        stored = self._read(self.settings_dir / "application.json", {})
        persisted = _deep_merge(stored if isinstance(stored, dict) else {}, incoming)
        if isinstance(persisted, dict):
            persisted.pop("ui_layout_defaults", None)
            persisted.pop("external_vae_override_enabled", None)
            persisted["phase13_troubleshooting_lock_removed"] = True
        self._write(self.settings_dir / "application.json", persisted)
        return self.load_application_settings()

    def load_default_asset_profiles(self) -> dict[str, Any]:
        payload = self._read(self.default_assets_dir / "profiles.json", default_document())
        return normalize_document(payload if isinstance(payload, dict) else {})

    def save_default_asset_profiles(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_document(payload)
        self._write(self.default_assets_dir / "profiles.json", normalized)
        return normalized

    def load_recent_output_visibility(self) -> dict[str, Any]:
        payload = self._read(
            self.recent_outputs_dir / "visibility.json",
            {"cleared_through_modified_ns": 0},
        )
        if not isinstance(payload, dict):
            payload = {}
        try:
            cleared_through = max(0, int(payload.get("cleared_through_modified_ns", 0) or 0))
        except (TypeError, ValueError):
            cleared_through = 0
        return {"cleared_through_modified_ns": cleared_through}

    def clear_recent_outputs_through(self, modified_ns: int) -> dict[str, Any]:
        current = self.load_recent_output_visibility()
        value = {
            "cleared_through_modified_ns": max(
                int(current.get("cleared_through_modified_ns", 0) or 0),
                max(0, int(modified_ns or 0)),
            )
        }
        self._write(self.recent_outputs_dir / "visibility.json", value)
        return value

    def restore_recent_outputs_visibility(self) -> dict[str, Any]:
        value = {"cleared_through_modified_ns": 0}
        self._write(self.recent_outputs_dir / "visibility.json", value)
        return value

    def _profile_root(self, kind: str, plugin_id: str | None = None) -> Path:
        safe_kind = _safe_name(kind).lower().replace(" ", "-")
        root = self.profile_dir / safe_kind
        if plugin_id:
            root = root / _safe_name(plugin_id).lower().replace(" ", "-")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list_profiles(self, kind: str, plugin_id: str | None = None) -> list[dict[str, Any]]:
        root = self._profile_root(kind, plugin_id)
        output: list[dict[str, Any]] = []
        for path in root.glob("*.json"):
            payload = self._read(path, {})
            if isinstance(payload, dict):
                output.append({**payload, "file_name": path.name})
        if str(kind or "").strip().casefold() == "generation":
            output.sort(
                key=lambda item: (
                    str(item.get("name") or Path(str(item.get("file_name") or "")).stem).casefold(),
                    str(item.get("file_name") or "").casefold(),
                )
            )
        else:
            output.sort(key=lambda item: str(item.get("file_name") or "").casefold())
        return output

    def save_profile(
        self,
        kind: str,
        name: str,
        payload: dict[str, Any],
        plugin_id: str | None = None,
        *,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        root = self._profile_root(kind, plugin_id)
        clean_name = str(name).strip() or "Untitled"
        safe_name = f"{_safe_name(clean_name)}.json"
        existing = {
            str(item.get("name") or "").strip().casefold(): item
            for item in self.list_profiles(kind, plugin_id)
            if isinstance(item, dict)
        }
        if not overwrite and clean_name.casefold() in existing:
            raise ValueError(f"A {kind} profile named '{clean_name}' already exists.")
        record = {
            "name": clean_name,
            "plugin_id": plugin_id,
            "kind": kind,
            "values": dict(payload or {}),
        }
        self._write(root / safe_name, record)
        return {**record, "file_name": safe_name}

    def delete_profile(self, kind: str, name: str, plugin_id: str | None = None) -> bool:
        root = self._profile_root(kind, plugin_id)
        safe_name = f"{_safe_name(name)}.json"
        path = root / safe_name
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def list_hires_profiles(self) -> list[dict[str, Any]]:
        return [profile.to_dict() for profile in self.hires_profile_service.list_profiles()]

    def get_hires_profile(self, profile_id: str) -> dict[str, Any] | None:
        profile = self.hires_profile_service.get_profile(profile_id)
        return profile.to_dict() if profile is not None else None

    def inspect_hires_profile(
        self,
        profile_id: str,
        *,
        choice_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.hires_profile_service.inspect_profile(
            profile_id,
            choice_overrides=choice_overrides,
        ).to_dict()

    def save_hires_profile(
        self,
        *,
        name: str,
        values: Mapping[str, Any],
        included_fields: list[str] | tuple[str, ...] | None = None,
        profile_id: str = "",
        description: str = "",
        compatibility: Mapping[str, Any] | None = None,
        baseline_profile_id: str = "",
        choice_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile, manifest = self.hires_profile_service.save_profile(
            name=name,
            values=values,
            included_fields=included_fields,
            profile_id=profile_id,
            description=description,
            compatibility=compatibility,
            baseline_profile_id=baseline_profile_id,
            choice_overrides=choice_overrides,
        )
        return {"profile": profile.to_dict(), "manifest": manifest.to_dict()}

    def duplicate_hires_profile(
        self,
        profile_id: str,
        *,
        name: str,
        choice_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile, manifest = self.hires_profile_service.duplicate_profile(
            profile_id,
            name=name,
            choice_overrides=choice_overrides,
        )
        return {"profile": profile.to_dict(), "manifest": manifest.to_dict()}

    def delete_hires_profile(self, profile_id: str) -> dict[str, Any]:
        return self.hires_profile_service.delete_profile(profile_id)

    def list_hires_default_assignments(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.hires_profile_service.list_default_assignments()]

    def save_hires_default_assignment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.hires_profile_service.save_default_assignment(payload).to_dict()

    def delete_hires_default_assignment(self, assignment_key: str) -> bool:
        return self.hires_profile_service.delete_default_assignment(assignment_key)

    def resolve_hires_auto(self, context: Mapping[str, Any], *, choice_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.hires_profile_service.resolve_auto(
            context, choice_overrides=choice_overrides
        ).to_dict()

    def list_prompt_presets(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for path in sorted(self.prompt_dir.glob("*.json"), key=lambda item: item.stem.casefold()):
            payload = self._read(path, {})
            if isinstance(payload, dict):
                output.append(payload)
        return output

    def save_prompt_preset(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "preset_version": 1,
            "name": str(name).strip() or "Untitled",
            **dict(payload or {}),
        }
        self._write(self.prompt_dir / f"{_safe_name(record['name'])}.json", record)
        return record

    def delete_prompt_preset(self, name: str) -> bool:
        try:
            (self.prompt_dir / f"{_safe_name(name)}.json").unlink()
            return True
        except OSError:
            return False


    def list_prompt_shortcut_profiles(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for path in sorted(self.prompt_shortcut_profile_dir.glob("*.json"), key=lambda item: item.stem.casefold()):
            payload = self._read(path, {})
            if isinstance(payload, dict):
                output.append({**payload, "file_name": path.name})
        return output

    def save_prompt_shortcut_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        profile_id = str(record.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("Shortcut profile_id is required.")
        record["profile_id"] = profile_id
        record["builtin"] = False
        record["source"] = "user"
        self._write(self.prompt_shortcut_profile_dir / f"{_safe_name(profile_id)}.json", record)
        return record

    def delete_prompt_shortcut_profile(self, profile_id: str) -> bool:
        try:
            (self.prompt_shortcut_profile_dir / f"{_safe_name(profile_id)}.json").unlink()
            return True
        except OSError:
            return False

    def list_prompt_parser_presets(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for path in sorted(self.prompt_parser_preset_dir.glob("*.json"), key=lambda item: item.stem.casefold()):
            payload = self._read(path, {})
            if isinstance(payload, dict):
                output.append({**payload, "file_name": path.name})
        return output

    def save_prompt_parser_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        preset_id = str(record.get("preset_id") or record.get("name") or "").strip()
        if not preset_id:
            raise ValueError("Parser preset_id or name is required.")
        record["preset_id"] = preset_id
        record["name"] = str(record.get("name") or preset_id).strip()
        record["builtin"] = False
        self._write(self.prompt_parser_preset_dir / f"{_safe_name(preset_id)}.json", record)
        return record

    def delete_prompt_parser_preset(self, preset_id: str) -> bool:
        try:
            (self.prompt_parser_preset_dir / f"{_safe_name(preset_id)}.json").unlink()
            return True
        except OSError:
            return False
