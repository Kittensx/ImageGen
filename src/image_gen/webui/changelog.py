from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from image_gen.program_metadata import PRODUCT_NAME


CHANGELOG_ENTRY_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\.md$")
GITHUB_REPOSITORY = "Kittensx/ImageGen"
GITHUB_BRANCH = "main"
GITHUB_CHANGELOG_DIRECTORY_URL = f"https://github.com/{GITHUB_REPOSITORY}/tree/{GITHUB_BRANCH}/changelog"
GITHUB_CONTENTS_API_URL = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/changelog?ref={GITHUB_BRANCH}"
GITHUB_RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{GITHUB_BRANCH}/changelog"
GITHUB_BLOB_BASE_URL = f"https://github.com/{GITHUB_REPOSITORY}/blob/{GITHUB_BRANCH}/changelog"


class ChangelogService:
    """Read the public ImageGen changelog with a bundled offline fallback.

    GitHub is authoritative while it is reachable. The packaged ``changelog``
    directory is used only when the public directory cannot be queried, so the
    Home page can still show release notes without a network connection.
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        cache_ttl_seconds: float = 900.0,
        request_timeout_seconds: float = 3.0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.local_root = self.project_root / "changelog"
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.request_timeout_seconds = max(0.25, float(request_timeout_seconds))
        self._catalog_cache: dict[str, Any] | None = None
        self._catalog_cached_at = 0.0
        self._entry_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _date_token(value: str) -> str:
        token = str(value or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            raise ValueError("Changelog entry must use YYYY-MM-DD format.")
        try:
            parsed = date.fromisoformat(token)
        except ValueError as exc:
            raise ValueError("Changelog entry date is invalid.") from exc
        if parsed.isoformat() != token:
            raise ValueError("Changelog entry must use canonical YYYY-MM-DD format.")
        return token

    @staticmethod
    def _github_url(entry_date: str) -> str:
        return f"{GITHUB_BLOB_BASE_URL}/{entry_date}.md"

    @staticmethod
    def _extract_title(markdown: str, entry_date: str) -> str:
        for raw_line in str(markdown or "").splitlines():
            match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", raw_line)
            if match:
                title = match.group(1).strip().strip("#").strip()
                if title:
                    return title
        return f"{PRODUCT_NAME} update - {entry_date}"

    def _request(self, url: str, *, accept: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": f"{PRODUCT_NAME}-WebUI-Changelog/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
            return response.read()

    def _github_entries(self) -> list[dict[str, Any]]:
        payload = json.loads(self._request(GITHUB_CONTENTS_API_URL, accept="application/vnd.github+json").decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("GitHub changelog directory response was not a file list.")

        entries: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict) or str(item.get("type") or "") != "file":
                continue
            filename = str(item.get("name") or "").strip()
            match = CHANGELOG_ENTRY_PATTERN.fullmatch(filename)
            if not match:
                continue
            entry_date = self._date_token(match.group("date"))
            entries.append(
                {
                    "date": entry_date,
                    "filename": filename,
                    "github_url": str(item.get("html_url") or self._github_url(entry_date)),
                    "source": "github",
                }
            )
        entries.sort(key=lambda item: item["date"], reverse=True)
        return entries

    def _local_entries(self) -> list[dict[str, Any]]:
        if not self.local_root.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for path in self.local_root.iterdir():
            if not path.is_file():
                continue
            match = CHANGELOG_ENTRY_PATTERN.fullmatch(path.name)
            if not match:
                continue
            try:
                entry_date = self._date_token(match.group("date"))
            except ValueError:
                continue
            entries.append(
                {
                    "date": entry_date,
                    "filename": path.name,
                    "github_url": self._github_url(entry_date),
                    "source": "bundled",
                }
            )
        entries.sort(key=lambda item: item["date"], reverse=True)
        return entries

    def catalog(self, *, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._catalog_cache
            if (
                not refresh
                and cached is not None
                and now - self._catalog_cached_at < self.cache_ttl_seconds
            ):
                return dict(cached)

        remote_error = ""
        try:
            entries = self._github_entries()
            source = "github"
            remote_available = True
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            entries = self._local_entries()
            source = "bundled"
            remote_available = False
            remote_error = str(exc)

        payload = {
            "schema": "image-gen-changelog-v1",
            "entries": entries,
            "source": source,
            "remote_available": remote_available,
            "github_directory_url": GITHUB_CHANGELOG_DIRECTORY_URL,
        }
        if remote_error:
            payload["remote_error"] = remote_error

        with self._lock:
            self._catalog_cache = dict(payload)
            self._catalog_cached_at = time.monotonic()
        return payload

    def _github_markdown(self, entry_date: str) -> str:
        url = f"{GITHUB_RAW_BASE_URL}/{entry_date}.md"
        return self._request(url, accept="text/plain; charset=utf-8").decode("utf-8")

    def _local_markdown(self, entry_date: str) -> str:
        path = self.local_root / f"{entry_date}.md"
        if not path.is_file():
            raise FileNotFoundError(f"Changelog entry not found: {entry_date}")
        return path.read_text(encoding="utf-8")

    def entry(self, entry_date: str, *, refresh: bool = False) -> dict[str, Any]:
        token = self._date_token(entry_date)
        now = time.monotonic()
        with self._lock:
            cached_item = self._entry_cache.get(token)
            if (
                not refresh
                and cached_item is not None
                and now - cached_item[0] < self.cache_ttl_seconds
            ):
                return dict(cached_item[1])

        remote_error = ""
        try:
            markdown = self._github_markdown(token)
            source = "github"
        except (OSError, UnicodeDecodeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            remote_error = str(exc)
            markdown = self._local_markdown(token)
            source = "bundled"

        payload = {
            "schema": "image-gen-changelog-entry-v1",
            "date": token,
            "filename": f"{token}.md",
            "title": self._extract_title(markdown, token),
            "markdown": markdown,
            "source": source,
            "github_url": self._github_url(token),
        }
        if remote_error:
            payload["remote_error"] = remote_error

        with self._lock:
            self._entry_cache[token] = (time.monotonic(), dict(payload))
        return payload
