from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DownloadRuntimeSettings:
    max_active_downloads: int = 2
    max_queued_downloads: int = 64
    bandwidth_limit_mib_per_second: float = 0.0
    provider_min_request_interval_seconds: float = 0.25
    retry_attempts: int = 3

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DownloadRuntimeSettings":
        root = dict(config or {})
        asset_hub = root.get("asset_hub") if isinstance(root.get("asset_hub"), Mapping) else {}
        raw = asset_hub.get("downloads") if isinstance(asset_hub.get("downloads"), Mapping) else {}
        return cls().updated(raw)

    def updated(self, values: Mapping[str, Any] | None) -> "DownloadRuntimeSettings":
        raw = dict(values or {})

        def integer(name: str, current: int, low: int, high: int) -> int:
            value = raw.get(name, current)
            try:
                number = int(value)
            except (TypeError, ValueError):
                return current
            return max(low, min(number, high))

        def decimal(name: str, current: float, low: float, high: float) -> float:
            value = raw.get(name, current)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return current
            return max(low, min(number, high))

        return replace(
            self,
            max_active_downloads=integer("max_active_downloads", self.max_active_downloads, 1, 8),
            max_queued_downloads=integer("max_queued_downloads", self.max_queued_downloads, 1, 500),
            bandwidth_limit_mib_per_second=decimal(
                "bandwidth_limit_mib_per_second",
                self.bandwidth_limit_mib_per_second,
                0.0,
                10240.0,
            ),
            provider_min_request_interval_seconds=decimal(
                "provider_min_request_interval_seconds",
                self.provider_min_request_interval_seconds,
                0.0,
                60.0,
            ),
            retry_attempts=integer("retry_attempts", self.retry_attempts, 0, 5),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxActiveDownloads": self.max_active_downloads,
            "maxQueuedDownloads": self.max_queued_downloads,
            "bandwidthLimitMiBPerSecond": self.bandwidth_limit_mib_per_second,
            "providerMinRequestIntervalSeconds": self.provider_min_request_interval_seconds,
            "retryAttempts": self.retry_attempts,
        }

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "max_active_downloads": self.max_active_downloads,
            "max_queued_downloads": self.max_queued_downloads,
            "bandwidth_limit_mib_per_second": self.bandwidth_limit_mib_per_second,
            "provider_min_request_interval_seconds": self.provider_min_request_interval_seconds,
            "retry_attempts": self.retry_attempts,
        }


_TOP_LEVEL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*:\s*(?:#.*)?$")
_TWO_SPACE_KEY = re.compile(r"^  [A-Za-z0-9_][A-Za-z0-9_-]*:\s*(?:#.*)?$")


def _download_block(settings: DownloadRuntimeSettings, *, indent: str = "  ") -> list[str]:
    values = settings.to_config_dict()
    return [
        f"{indent}downloads:",
        f"{indent}  max_active_downloads: {values['max_active_downloads']}",
        f"{indent}  max_queued_downloads: {values['max_queued_downloads']}",
        f"{indent}  bandwidth_limit_mib_per_second: {values['bandwidth_limit_mib_per_second']:g}",
        f"{indent}  provider_min_request_interval_seconds: {values['provider_min_request_interval_seconds']:g}",
        f"{indent}  retry_attempts: {values['retry_attempts']}",
    ]


def persist_download_settings(config_path: str | Path, settings: DownloadRuntimeSettings) -> None:
    """Persist only ``asset_hub.downloads`` while preserving unrelated YAML text/comments.

    The canonical user config is intentionally user-owned. This targeted writer avoids
    round-tripping the whole document through a YAML emitter just to change downloader
    controls.
    """

    path = Path(config_path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    asset_start = next((i for i, line in enumerate(lines) if line.strip() == "asset_hub:" and not line.startswith((" ", "\t"))), None)

    if asset_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("asset_hub:")
        lines.extend(_download_block(settings))
    else:
        asset_end = len(lines)
        for i in range(asset_start + 1, len(lines)):
            line = lines[i]
            if line and not line.startswith((" ", "\t")) and _TOP_LEVEL.match(line):
                asset_end = i
                break
        download_start = None
        for i in range(asset_start + 1, asset_end):
            if lines[i].startswith("  downloads:") and lines[i].strip().startswith("downloads:"):
                download_start = i
                break
        block = _download_block(settings)
        if download_start is None:
            insert_at = asset_end
            while insert_at > asset_start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines[insert_at:insert_at] = block
        else:
            download_end = asset_end
            for i in range(download_start + 1, asset_end):
                line = lines[i]
                if line.startswith("  ") and not line.startswith("    ") and _TWO_SPACE_KEY.match(line):
                    download_end = i
                    break
            lines[download_start:download_end] = block

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
