from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from image_gen.program_metadata import APPLICATION_VERSION, PRODUCT_NAME
from image_gen.webui.discord_native import DiscordNativeBridge


PROFILE_SCHEMA = "image-gen-profile-v1"
PROFILE_SHARING_SCHEMA = "image-gen-profile-sharing-v1"
DISCORD_ACTIVITY_SCHEMA = "image-gen-discord-activity-v1"
DISCORD_INTRO_SCHEMA = "image-gen-discord-intro-v1"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


class ImageGenProfileService:
    """Persistent, privacy-safe local profile and shareable usage statistics."""

    def __init__(
        self,
        project_root: str | Path,
        data_root: str | Path,
        output_root: str | Path,
        *,
        startup_timestamp: float | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.profile_root = Path(data_root).expanduser().resolve() / "webui" / "profile"
        self.output_root = Path(output_root).expanduser().resolve()
        self.profile_path = self.profile_root / "profile.json"
        self.discord_bridge = DiscordNativeBridge(self.project_root)
        self.startup_timestamp = startup_timestamp
        self._lock = threading.RLock()
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self.ensure_profile()

    def _detect_install_date(self) -> tuple[str, str]:
        candidates: list[tuple[float, str]] = []
        install_root = self.project_root / "artifacts" / "install"
        if install_root.is_dir():
            for path in install_root.iterdir():
                try:
                    candidates.append((path.stat().st_mtime, "install_artifact"))
                except OSError:
                    continue
        venv_marker = self.project_root / ".venv" / "pyvenv.cfg"
        if venv_marker.is_file():
            try:
                candidates.append((venv_marker.stat().st_mtime, "environment_marker"))
            except OSError:
                pass
        if candidates:
            timestamp, source = min(candidates, key=lambda item: item[0])
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(), source
        if self.startup_timestamp is not None:
            try:
                return datetime.fromtimestamp(float(self.startup_timestamp), tz=timezone.utc).isoformat(), "app_startup"
            except (TypeError, ValueError, OSError, OverflowError):
                pass
        return _utc_now(), "profile_initialized"

    def _existing_output_count(self) -> int:
        if not self.output_root.is_dir():
            return 0
        count = 0
        try:
            for path in self.output_root.rglob("*"):
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
                    count += 1
        except OSError:
            return count
        return count

    @staticmethod
    def _sharing_defaults() -> dict[str, bool]:
        return {
            "discord_rich_presence_enabled": False,
            "discord_intro_card_enabled": False,
            "share_install_date": True,
            "share_image_count": True,
            "share_bug_stats": True,
        }

    @staticmethod
    def _discord_defaults() -> dict[str, Any]:
        return {
            "linked": False,
            "user_id": None,
            "username": None,
            "display_name": None,
            "avatar_url": None,
            "linked_at": None,
            "server_member": False,
            "server_name": None,
            "server_verified_at": None,
            "intro_card_shared_at": None,
        }

    def _default_profile(self) -> dict[str, Any]:
        installed_at, source = self._detect_install_date()
        return {
            "schema": PROFILE_SCHEMA,
            "installation_id": str(uuid.uuid4()),
            "installed_at": installed_at,
            "install_date_source": source,
            "first_seen_version": str(APPLICATION_VERSION),
            "last_seen_version": str(APPLICATION_VERSION),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "usage": {
                "images_generated": self._existing_output_count(),
                "image_counter_source": "existing_output_scan",
                "image_counter_started_at": _utc_now(),
            },
            "sharing": self._sharing_defaults(),
            "discord": self._discord_defaults(),
        }

    def ensure_profile(self) -> dict[str, Any]:
        with self._lock:
            payload = _read_json(self.profile_path)
            if payload.get("schema") != PROFILE_SCHEMA:
                payload = self._default_profile()
                self._write(payload)
                return payload

            changed = False
            if not payload.get("installation_id"):
                payload["installation_id"] = str(uuid.uuid4())
                changed = True
            if not payload.get("installed_at"):
                payload["installed_at"], payload["install_date_source"] = self._detect_install_date()
                changed = True
            payload.setdefault("first_seen_version", str(APPLICATION_VERSION))
            if payload.get("last_seen_version") != str(APPLICATION_VERSION):
                payload["last_seen_version"] = str(APPLICATION_VERSION)
                changed = True
            usage = payload.setdefault("usage", {})
            if "images_generated" not in usage:
                usage["images_generated"] = self._existing_output_count()
                usage["image_counter_source"] = "existing_output_scan"
                usage["image_counter_started_at"] = _utc_now()
                changed = True
            sharing = payload.setdefault("sharing", {})
            defaults = self._sharing_defaults()
            for key, value in defaults.items():
                if key not in sharing:
                    sharing[key] = value
                    changed = True
            discord = payload.setdefault("discord", {})
            for key, value in self._discord_defaults().items():
                if key not in discord:
                    discord[key] = value
                    changed = True
            if changed:
                self._write(payload)
            return payload

    def _write(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        value["schema"] = PROFILE_SCHEMA
        value["updated_at"] = _utc_now()
        self.profile_root.mkdir(parents=True, exist_ok=True)
        temp = self.profile_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.profile_path)
        return value

    def record_generated_image(self, image_path: str | Path) -> dict[str, Any]:
        """Increment the lifetime image count after a newly persisted final output."""

        with self._lock:
            payload = self.ensure_profile()
            usage = payload.setdefault("usage", {})
            usage["images_generated"] = _safe_int(usage.get("images_generated")) + 1
            usage["last_generated_at"] = _utc_now()
            # Deliberately do not retain prompts, paths, filenames, hashes, or image contents.
            return self._write(payload)

    def update_sharing(self, values: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "discord_rich_presence_enabled",
            "discord_intro_card_enabled",
            "share_install_date",
            "share_image_count",
            "share_bug_stats",
        }
        with self._lock:
            payload = self.ensure_profile()
            sharing = payload.setdefault("sharing", {})
            for key in allowed:
                if key in values:
                    sharing[key] = bool(values[key])
            return self._write(payload)

    def disconnect_discord(self) -> dict[str, Any]:
        with self._lock:
            payload = self.ensure_profile()
            payload["discord"] = self._discord_defaults()
            payload["sharing"]["discord_rich_presence_enabled"] = False
            payload["sharing"]["discord_intro_card_enabled"] = False
            return self._write(payload)

    def discord_capabilities(self) -> dict[str, Any]:
        return self.discord_bridge.capabilities()

    def discord_community_status(self, *, force: bool = False) -> dict[str, Any]:
        return self.discord_bridge.community_status(force=force)

    def connect_discord(self) -> dict[str, Any]:
        result = self.discord_bridge.link_account()
        if not bool(result.get("ok")):
            return result
        user = result.get("user") if isinstance(result.get("user"), Mapping) else {}
        user_id = str(user.get("id") or "").strip()
        if not user_id:
            return {"ok": False, "state": "user_unavailable", "message": "Discord did not return a user identity."}
        with self._lock:
            payload = self.ensure_profile()
            discord = payload.setdefault("discord", {})
            discord.update(
                {
                    "linked": True,
                    "user_id": user_id,
                    "username": str(user.get("username") or "").strip() or None,
                    "display_name": str(user.get("display_name") or "").strip() or None,
                    "avatar_url": str(user.get("avatar_url") or "").strip() or None,
                    "linked_at": _utc_now(),
                }
            )
            self._write(payload)
        return {"ok": True, "state": "linked", "user": dict(user)}

    def _bug_stats(self, bug_profile: Mapping[str, Any] | None) -> dict[str, Any]:
        source = bug_profile if isinstance(bug_profile, Mapping) else {}
        return {
            "reported": _safe_int(source.get("reported")),
            "open": _safe_int(source.get("open")),
            "resolved": _safe_int(source.get("resolved")),
            "pending": _safe_int(source.get("pending")),
            "resolution_rate": float(source.get("resolution_rate") or 0.0),
        }

    def public_stats(self, bug_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = self.ensure_profile()
        sharing = payload.get("sharing") or {}
        stats: dict[str, Any] = {
            "schema": PROFILE_SHARING_SCHEMA,
            "images_generated": _safe_int((payload.get("usage") or {}).get("images_generated")),
            "bugs": self._bug_stats(bug_profile),
        }
        if bool(sharing.get("share_install_date", True)):
            stats["installed_at"] = payload.get("installed_at")
        if not bool(sharing.get("share_image_count", True)):
            stats.pop("images_generated", None)
        if not bool(sharing.get("share_bug_stats", True)):
            stats.pop("bugs", None)
        return stats

    def discord_activity_payload(self, bug_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = self.ensure_profile()
        sharing = payload.get("sharing") or {}
        stats = self.public_stats(bug_profile)
        state_bits: list[str] = []
        if "images_generated" in stats:
            state_bits.append(f"{stats['images_generated']:,} images created")
        bugs = stats.get("bugs") if isinstance(stats.get("bugs"), Mapping) else None
        if bugs and _safe_int(bugs.get("reported")):
            state_bits.append(f"{_safe_int(bugs.get('reported'))} bugs reported")
        return {
            "schema": DISCORD_ACTIVITY_SCHEMA,
            "enabled": bool(sharing.get("discord_rich_presence_enabled", False)),
            "activity_type": "playing",
            "details": f"Currently using {PRODUCT_NAME}",
            "state": " · ".join(state_bits),
            "large_image_key": "imagegen",
            "large_image_text": PRODUCT_NAME,
            "privacy": "aggregate_stats_only",
        }

    def discord_intro_payload(self, bug_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = self.ensure_profile()
        sharing = payload.get("sharing") or {}
        return {
            "schema": DISCORD_INTRO_SCHEMA,
            "enabled": bool(sharing.get("discord_intro_card_enabled", False)),
            "stats": self.public_stats(bug_profile),
            "privacy": "aggregate_stats_only",
        }

    def publish_presence(self, bug_profile: Mapping[str, Any] | None = None, *, active: bool = True) -> dict[str, Any]:
        """Publish or clear aggregate Rich Presence through the native helper."""

        capabilities = self.discord_capabilities()
        activity = self.discord_activity_payload(bug_profile)
        if not capabilities.get("rich_presence_ready"):
            return {"published": False, "state": capabilities.get("rich_presence_state"), "activity": activity}
        if not activity["enabled"] and active:
            return {"published": False, "state": "disabled_by_user", "activity": activity}

        result = self.discord_bridge.set_activity(activity) if active else self.discord_bridge.clear_activity()
        return {
            "published": bool(result.get("ok")),
            "state": str(result.get("state") or ("published" if result.get("ok") else "helper_error")),
            "message": str(result.get("message") or "")[:1000],
            "activity": activity,
        }

    def snapshot(self, bug_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = self.ensure_profile()
        return {
            "schema": PROFILE_SCHEMA,
            "installation_id": payload.get("installation_id"),
            "installed_at": payload.get("installed_at"),
            "install_date_source": payload.get("install_date_source"),
            "first_seen_version": payload.get("first_seen_version"),
            "last_seen_version": payload.get("last_seen_version"),
            "usage": {
                "images_generated": _safe_int((payload.get("usage") or {}).get("images_generated")),
                "last_generated_at": (payload.get("usage") or {}).get("last_generated_at"),
            },
            "bugs": self._bug_stats(bug_profile),
            "sharing": dict(payload.get("sharing") or {}),
            "discord": dict(payload.get("discord") or {}),
            "discord_capabilities": self.discord_capabilities(),
            "discord_activity_preview": self.discord_activity_payload(bug_profile),
            "discord_intro_preview": self.discord_intro_payload(bug_profile),
        }
