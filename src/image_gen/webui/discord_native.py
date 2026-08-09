from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_IMAGEGEN_DISCORD_GUILD_ID = "1535734882494316746"
DISCORD_CONFIG_SCHEMA = "image-gen-discord-integration-v1"


class DiscordNativeBridge:
    """Small process boundary around the bundled Discord Social SDK helper.

    The helper owns native SDK interaction. The WebUI only exchanges bounded JSON and
    never receives or persists Discord OAuth access/refresh tokens.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self._widget_cache_lock = threading.Lock()
        self._widget_cache: dict[str, Any] | None = None
        self._widget_cache_expires_at = 0.0

    def _config(self) -> dict[str, Any]:
        path = self.project_root / "configs" / "discord_integration.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema") != DISCORD_CONFIG_SCHEMA:
            return {}
        return payload

    @property
    def application_id(self) -> str:
        configured = self._config().get("application_id")
        return str(os.environ.get("IMAGE_GEN_DISCORD_APPLICATION_ID") or configured or "").strip()

    @property
    def guild_id(self) -> str:
        configured = self._config().get("guild_id")
        return str(os.environ.get("IMAGE_GEN_DISCORD_GUILD_ID") or configured or DEFAULT_IMAGEGEN_DISCORD_GUILD_ID).strip()

    @staticmethod
    def _discord_url(value: Any) -> str:
        text = str(value or "").strip()
        if text.startswith("https://discord.com/") or text.startswith("https://discord.gg/"):
            return text
        return ""

    @property
    def server_url(self) -> str:
        configured = self._discord_url(self._config().get("server_url"))
        return configured or f"https://discord.com/channels/{self.guild_id}/"

    @property
    def invite_url(self) -> str:
        return self._discord_url(os.environ.get("IMAGE_GEN_DISCORD_INVITE_URL") or self._config().get("invite_url"))

    @property
    def widget_url(self) -> str:
        guild_id = self.guild_id
        if not guild_id.isdigit():
            return ""
        return f"https://discord.com/api/guilds/{guild_id}/widget.json"

    def community_status(self, *, force: bool = False, timeout: float = 3.0) -> dict[str, Any]:
        """Return the public Discord server presence count without member identities."""

        url = self.widget_url
        if not url:
            return {
                "ok": False,
                "state": "invalid_guild_id",
                "guild_id": self.guild_id,
                "online_count": None,
                "privacy": "count_only",
            }

        now = time.monotonic()
        with self._widget_cache_lock:
            cached = dict(self._widget_cache) if self._widget_cache is not None else None
            if cached is not None and not force and now < self._widget_cache_expires_at:
                cached["cached"] = True
                return cached

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ImageGen-WebUI Discord-Community-Status",
            },
        )
        try:
            with urlopen(request, timeout=max(0.5, float(timeout))) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Discord widget response was not an object.")
            raw_count = payload.get("presence_count")
            if isinstance(raw_count, bool):
                raise ValueError("Discord widget presence count was invalid.")
            online_count = int(raw_count)
            if online_count < 0:
                raise ValueError("Discord widget presence count was invalid.")
            result = {
                "ok": True,
                "state": "ready",
                "guild_id": self.guild_id,
                "server_name": str(payload.get("name") or "ImageGen")[:120],
                "online_count": online_count,
                "community_url": self.invite_url or self.server_url,
                "privacy": "count_only",
                "cached": False,
            }
            with self._widget_cache_lock:
                self._widget_cache = dict(result)
                self._widget_cache_expires_at = time.monotonic() + 60.0
            return result
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            with self._widget_cache_lock:
                stale = dict(self._widget_cache) if self._widget_cache is not None else None
            if stale is not None:
                stale.update({"ok": True, "state": "stale", "cached": True, "stale": True})
                return stale
            return {
                "ok": False,
                "state": "unavailable",
                "guild_id": self.guild_id,
                "online_count": None,
                "privacy": "count_only",
                "message": str(exc)[:240],
            }

    def helper_path(self) -> Path | None:
        configured = str(
            os.environ.get("IMAGE_GEN_DISCORD_HELPER")
            or os.environ.get("IMAGE_GEN_DISCORD_PRESENCE_HELPER")
            or ""
        ).strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured).expanduser())
        app_dir = self.project_root / "app" / "discord"
        if os.name == "nt":
            candidates.append(app_dir / "imagegen_discord_helper.exe")
        candidates.append(app_dir / "imagegen_discord_helper")
        for path in candidates:
            try:
                if path.is_file():
                    return path.resolve()
            except OSError:
                continue
        return None

    def _invoke(
        self,
        action: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        helper = self.helper_path()
        if helper is None:
            return {"ok": False, "state": "helper_required", "message": "Discord native helper is not installed."}
        request = {
            "application_id": self.application_id,
            "guild_id": self.guild_id,
            **dict(payload or {}),
        }
        try:
            completed = subprocess.run(
                [str(helper), action],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "state": "timeout", "message": "Discord did not finish the request in time."}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "state": "helper_error", "message": str(exc)}

        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        try:
            result = json.loads(output) if output else {}
        except json.JSONDecodeError:
            result = {}
        if not isinstance(result, dict):
            result = {}
        result.setdefault("ok", completed.returncode == 0)
        if completed.returncode != 0:
            result.setdefault("state", "helper_error")
            result.setdefault("message", error or output or f"Discord helper exited with {completed.returncode}.")
        return result

    def capabilities(self) -> dict[str, Any]:
        helper = self.helper_path()
        configured = bool(self.application_id)
        base = {
            "provider": "discord_social_sdk",
            "application_id_configured": configured,
            "guild_id": self.guild_id,
            "server_url": self.server_url,
            "invite_url": self.invite_url or None,
            "community_url": self.invite_url or self.server_url,
            "invite_configured": bool(self.invite_url),
            "helper_installed": helper is not None,
            "account_linking_supported": True,
            "rich_presence_supported": True,
            "server_membership_supported": True,
            "server_membership_state": "bot_required",
            "intro_card_supported": True,
            "intro_card_state": "bot_required",
        }
        if not configured:
            return {
                **base,
                "helper_ready": False,
                "account_linking_ready": False,
                "account_linking_state": "discord_application_required",
                "rich_presence_ready": False,
                "rich_presence_state": "discord_application_required",
            }
        if helper is None:
            return {
                **base,
                "helper_ready": False,
                "account_linking_ready": False,
                "account_linking_state": "native_helper_required",
                "rich_presence_ready": False,
                "rich_presence_state": "native_helper_required",
            }
        status = self._invoke("--status", timeout=3.0)
        sdk_ready = bool(status.get("ok") and status.get("sdk_available", True))
        state = "ready" if sdk_ready else str(status.get("state") or "sdk_unavailable")
        return {
            **base,
            "helper_ready": sdk_ready,
            "helper_version": status.get("helper_version"),
            "account_linking_ready": sdk_ready,
            "account_linking_state": state,
            "rich_presence_ready": sdk_ready,
            "rich_presence_state": state,
        }

    def link_account(self) -> dict[str, Any]:
        if not self.application_id:
            return {
                "ok": False,
                "state": "discord_application_required",
                "message": "Discord application ID is not configured.",
            }
        return self._invoke("--link-account", timeout=180.0)

    def set_activity(self, activity: Mapping[str, Any]) -> dict[str, Any]:
        return self._invoke("--set-activity", {"activity": dict(activity)}, timeout=10.0)

    def clear_activity(self) -> dict[str, Any]:
        return self._invoke("--clear-activity", timeout=10.0)
