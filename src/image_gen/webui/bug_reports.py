from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from image_gen.program_metadata import APPLICATION_VERSION, build_program_metadata


BUG_REPORT_SCHEMA = "image-gen-bug-report-v1"
BUG_LEDGER_SCHEMA = "image-gen-bug-ledger-v1"
GITHUB_OWNER = "Kittensx"
GITHUB_REPOSITORY = "ImageGen"
GITHUB_REPOSITORY_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}"
GITHUB_ATTACHMENT_LIMIT_BYTES = 25 * 1024 * 1024
# Leave headroom for ZIP metadata and small differences in browser-side size display.
TARGET_BUNDLE_LIMIT_BYTES = 24 * 1024 * 1024
MAX_GITHUB_ISSUE_PAGES = 5

_FINGERPRINT_MARKER_RE = re.compile(
    r"<!--\s*imagegen-bug-fingerprint:\s*([0-9a-f]{64})\s*-->", re.IGNORECASE
)
_REPORTER_MARKER_RE = re.compile(
    r"<!--\s*imagegen-reporter-record:\s*([0-9a-f-]{32,36})\s*-->", re.IGNORECASE
)
_VERSION_MARKER_RE = re.compile(
    r"<!--\s*imagegen-bug-version:\s*([^>]+?)\s*-->", re.IGNORECASE
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+|(?:api[_-]?key|access[_-]?token|secret|password|passwd)\s*[:=]\s*)"
    r"([^\s,;\]\[}\{\"']+)"
)
_TOKEN_RE = re.compile(
    r"(?i)\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"
)
_HEX_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
_LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
_LINE_NUMBER_RE = re.compile(r"(?i)(?:line\s+|:)(\d{1,6})(?=\D|$)")
_VERSION_TOKEN_RE = re.compile(r"\d+|[A-Za-z]+")


class BugReportError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return path.name


def _version_key(value: Any) -> tuple:
    """Return a SemVer-like sortable key without adding a packaging dependency."""

    text = str(value or "").strip()
    if not text or text.casefold() == "unknown":
        return (-1, -1, -1, -1, -1, ())

    normalized = text.lstrip("vV")
    main, separator, prerelease = normalized.partition("-")
    main_numbers = [int(token) for token in re.findall(r"\d+", main)[:4]]
    while len(main_numbers) < 4:
        main_numbers.append(0)

    if not separator:
        prerelease_rank = 100
        prerelease_tokens: tuple[Any, ...] = ()
    else:
        lowered = prerelease.casefold()
        label = re.split(r"[._+-]", lowered, maxsplit=1)[0]
        prerelease_rank = {
            "dev": 0,
            "snapshot": 5,
            "alpha": 10,
            "a": 10,
            "beta": 20,
            "b": 20,
            "rc": 30,
        }.get(label, 15)
        prerelease_tokens = tuple(
            (1, int(token)) if token.isdigit() else (0, token.casefold())
            for token in _VERSION_TOKEN_RE.findall(prerelease)
        )
    return (*main_numbers, prerelease_rank, prerelease_tokens)


def _normalized_error_message(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = _HEX_ADDRESS_RE.sub("<addr>", text)
    text = _LONG_NUMBER_RE.sub("<n>", text)
    text = re.sub(r"[A-Za-z]:[/\\][^\s\"']+", "<path>", text)
    text = re.sub(r"/(?:[^\s/]+/){2,}[^\s\"']+", "<path>", text)
    return re.sub(r"\s+", " ", text)[:1200]


def _trace_signature(traceback_text: str) -> list[str]:
    frames: list[str] = []
    for line in str(traceback_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("File "):
            continue
        match = re.search(r'File ["\'](.+?)["\'], line \d+, in (.+)$', stripped)
        if not match:
            continue
        filename = Path(match.group(1).replace("\\", "/")).name.casefold()
        function = match.group(2).strip().casefold()
        frames.append(f"{filename}:{function}")
    if frames:
        return frames[-8:]

    # Some diagnostics contain stack-like text without standard Python frames.
    fallback = []
    for line in str(traceback_text or "").splitlines():
        line = _LINE_NUMBER_RE.sub("<line>", line.strip().casefold())
        if line and not line.startswith("traceback"):
            fallback.append(line[:240])
    return fallback[-6:]


def _redact_text(text: str, project_root: Path) -> str:
    output = str(text)
    replacements: list[tuple[str, str]] = []
    try:
        replacements.append((str(project_root.resolve()), "<PROJECT_ROOT>"))
    except OSError:
        pass
    try:
        replacements.append((str(Path.home().resolve()), "<USER_HOME>"))
    except OSError:
        pass
    for source, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if source:
            output = output.replace(source, replacement)
            output = output.replace(source.replace("\\", "/"), replacement)
            output = output.replace(source.replace("/", "\\"), replacement)
    output = _SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}<redacted>", output)
    output = _TOKEN_RE.sub("<redacted-token>", output)
    return output


def _redact_json_value(value: Any, project_root: Path, *, key: str = "", depth: int = 0) -> Any:
    if depth > 10:
        return "<maximum report depth reached>"
    lowered = str(key or "").casefold()
    if any(marker in lowered for marker in ("password", "passwd", "secret", "token", "api_key", "apikey", "authorization", "cookie")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_json_value(item, project_root, key=str(item_key), depth=depth + 1)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json_value(item, project_root, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return _redact_text(value, project_root)
    return value


class BugReportService:
    """Discover, deduplicate, package, and reconcile local failure reports."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.bug_root = self.project_root / "artifacts" / "bugs"
        self.bundle_root = self.bug_root / "bundles"
        self.ledger_path = self.bug_root / "ledger.json"
        self.failure_root = self.project_root / "artifacts" / "diagnostics" / "failures"
        self.bug_root.mkdir(parents=True, exist_ok=True)
        self.bundle_root.mkdir(parents=True, exist_ok=True)

    def _default_ledger(self) -> dict[str, Any]:
        return {
            "schema": BUG_LEDGER_SCHEMA,
            "installation_id": str(uuid.uuid4()),
            "updated_at": _utc_now(),
            "reports": {},
            "github_cache": {
                "updated_at": None,
                "status": "not_synced",
                "message": "GitHub issue state has not been synchronized yet.",
            },
        }

    def load_ledger(self) -> dict[str, Any]:
        payload = _read_json(self.ledger_path)
        if payload.get("schema") != BUG_LEDGER_SCHEMA or not isinstance(payload.get("reports"), dict):
            payload = self._default_ledger()
        if not payload.get("installation_id"):
            payload["installation_id"] = str(uuid.uuid4())
        payload.setdefault("github_cache", {})
        return payload

    def save_ledger(self, ledger: dict[str, Any]) -> dict[str, Any]:
        ledger = dict(ledger)
        ledger["schema"] = BUG_LEDGER_SCHEMA
        ledger["updated_at"] = _utc_now()
        self.bug_root.mkdir(parents=True, exist_ok=True)
        temp = self.ledger_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.ledger_path)
        return ledger

    def _candidate_directories(self) -> list[Path]:
        if not self.failure_root.is_dir():
            return []
        output: list[Path] = []
        for path in self.failure_root.iterdir():
            if not path.is_dir():
                continue
            if (path / "failure.json").is_file() or (path / "report.json").is_file():
                output.append(path)
        return sorted(output, key=lambda item: item.stat().st_mtime_ns)

    def _candidate_payload(self, directory: Path) -> dict[str, Any] | None:
        failure = _read_json(directory / "failure.json")
        report = _read_json(directory / "report.json")
        payload = failure or report
        if not payload:
            return None

        kind = "generation" if failure else "webui"
        application = payload.get("application") if isinstance(payload.get("application"), dict) else {}
        version = str(application.get("version") or payload.get("application_version") or "unknown").strip() or "unknown"
        build_display = str((application.get("build") or {}).get("display") or "").strip() if isinstance(application, dict) else ""
        created_at = str(
            payload.get("started_utc")
            or payload.get("created_at")
            or datetime.fromtimestamp(directory.stat().st_mtime, tz=timezone.utc).isoformat()
        )

        if failure:
            component = str(failure.get("system") or "generation")
            operation = str(failure.get("operation") or "unknown")
            error_type = str(failure.get("error_type") or "Error")
            error_message = str(failure.get("error_message") or "")
            stage = f"{component}.{operation}"
        else:
            component = "webui"
            operation = str(report.get("stage") or "unknown")
            exception = report.get("exception") if isinstance(report.get("exception"), dict) else {}
            error_type = str(exception.get("type") or "Error")
            error_message = str(exception.get("message") or exception.get("summary") or "")
            stage = operation

        traceback_path = directory / "traceback.txt"
        try:
            traceback_text = traceback_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            traceback_text = ""
        signature = {
            "kind": kind,
            "component": component.casefold(),
            "operation": operation.casefold(),
            "error_type": error_type.casefold(),
            "error_message": _normalized_error_message(error_message),
            "frames": _trace_signature(traceback_text),
        }
        fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return {
            "fingerprint": fingerprint,
            "source_path": str(directory),
            "source_relative": _safe_relative(directory, self.project_root),
            "kind": kind,
            "component": component,
            "operation": operation,
            "stage": stage,
            "error_type": error_type,
            "error_message": error_message,
            "normalized_error": signature["error_message"],
            "frames": signature["frames"],
            "version": version,
            "build": build_display,
            "created_at": created_at,
            "modified_ns": directory.stat().st_mtime_ns,
        }

    def discover(self) -> list[dict[str, Any]]:
        candidates = []
        for directory in self._candidate_directories():
            payload = self._candidate_payload(directory)
            if payload:
                candidates.append(payload)
        return candidates

    @staticmethod
    def _winner_key(candidate: Mapping[str, Any]) -> tuple:
        return (
            _version_key(candidate.get("version")),
            str(candidate.get("created_at") or ""),
            int(candidate.get("modified_ns") or 0),
        )

    def _group_candidates(self, candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate["fingerprint"], []).append(candidate)

        output: dict[str, dict[str, Any]] = {}
        for fingerprint, occurrences in grouped.items():
            ordered = sorted(occurrences, key=self._winner_key, reverse=True)
            winner = dict(ordered[0])
            winner["occurrences"] = [
                {
                    "version": str(item.get("version") or "unknown"),
                    "build": str(item.get("build") or ""),
                    "created_at": str(item.get("created_at") or ""),
                    "source_relative": str(item.get("source_relative") or ""),
                }
                for item in ordered
            ]
            winner["versions_seen"] = list(
                dict.fromkeys(str(item.get("version") or "unknown") for item in ordered)
            )
            output[fingerprint] = winner
        return output

    def _reporter_record_id(self, ledger: dict[str, Any], fingerprint: str) -> str:
        existing = (ledger.get("reports") or {}).get(fingerprint) or {}
        value = str(existing.get("reporter_record_id") or "").strip()
        if value:
            return value
        return str(uuid.uuid4())

    def _metadata_document(self, candidate: Mapping[str, Any], reporter_record_id: str) -> dict[str, Any]:
        return {
            "schema": BUG_REPORT_SCHEMA,
            "fingerprint": candidate["fingerprint"],
            "reporter_record_id": reporter_record_id,
            "repository": f"{GITHUB_OWNER}/{GITHUB_REPOSITORY}",
            "selected_occurrence": {
                "version": candidate.get("version"),
                "build": candidate.get("build"),
                "created_at": candidate.get("created_at"),
                "kind": candidate.get("kind"),
                "component": candidate.get("component"),
                "operation": candidate.get("operation"),
                "error_type": candidate.get("error_type"),
                "error_message": _redact_text(str(candidate.get("error_message") or ""), self.project_root),
            },
            "deduplication": {
                "occurrence_count": len(candidate.get("occurrences") or []),
                "versions_seen": list(candidate.get("versions_seen") or []),
                "policy": "Highest application version wins; ties prefer the newest occurrence.",
                "occurrences": list(candidate.get("occurrences") or []),
            },
            "privacy": {
                "redaction_applied": True,
                "notes": [
                    "Known secret-like fields and token patterns are redacted.",
                    "Project-root and user-home absolute paths are replaced in text files.",
                    "Generation prompts/settings may remain because they can be required to reproduce the bug.",
                ],
            },
        }

    def _iter_source_files(self, source_dir: Path) -> list[Path]:
        return [path for path in sorted(source_dir.rglob("*")) if path.is_file() and not path.is_symlink()]

    def _write_sanitized_file(self, archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
        suffix = path.suffix.casefold()
        if suffix == ".json":
            payload = _read_json(path)
            if payload:
                redacted = _redact_json_value(payload, self.project_root)
                archive.writestr(arcname, json.dumps(redacted, indent=2, ensure_ascii=False) + "\n")
                return
        if suffix in {".txt", ".log", ".jsonl", ".csv", ".md", ".yml", ".yaml"}:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            archive.writestr(arcname, _redact_text(text, self.project_root))
            return
        archive.write(path, arcname)

    def _build_zip(self, candidate: Mapping[str, Any], reporter_record_id: str, *, compact: bool = False) -> Path:
        fingerprint = str(candidate["fingerprint"])
        version = re.sub(r"[^A-Za-z0-9._-]+", "-", str(candidate.get("version") or "unknown"))[:80]
        report_dir = self.bundle_root / fingerprint[:16]
        report_dir.mkdir(parents=True, exist_ok=True)
        target = report_dir / f"imagegen-bug-{fingerprint[:12]}-v{version}.zip"

        for existing in report_dir.glob("imagegen-bug-*.zip"):
            if existing != target:
                try:
                    existing.unlink()
                except OSError:
                    pass

        source_dir = Path(str(candidate["source_path"]))
        metadata = self._metadata_document(candidate, reporter_record_id)
        omitted: list[str] = []
        with tempfile.NamedTemporaryFile(prefix="imagegen-bug-", suffix=".zip", dir=report_dir, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as archive:
                archive.writestr(
                    "imagegen_bug_report.json",
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                )
                files = self._iter_source_files(source_dir)
                for path in files:
                    relative = str(path.relative_to(source_dir)).replace("\\", "/")
                    if compact and path.suffix.casefold() not in {
                        ".json", ".jsonl", ".txt", ".log", ".csv", ".md", ".yml", ".yaml"
                    }:
                        omitted.append(relative)
                        continue
                    self._write_sanitized_file(archive, path, f"diagnostics/{relative}")
                if omitted:
                    archive.writestr(
                        "COMPACT_BUNDLE_OMITTED_FILES.txt",
                        "The full diagnostic folder exceeded GitHub's attachment size target.\n"
                        "The following non-text files were omitted from this compact report bundle:\n\n"
                        + "\n".join(omitted)
                        + "\n",
                    )
            temp_path.replace(target)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        return target

    def _ensure_bundle(self, candidate: Mapping[str, Any], reporter_record_id: str) -> tuple[Path, bool]:
        path = self._build_zip(candidate, reporter_record_id, compact=False)
        compact = False
        if path.stat().st_size > TARGET_BUNDLE_LIMIT_BYTES:
            path = self._build_zip(candidate, reporter_record_id, compact=True)
            compact = True
        return path, compact

    def _issue_title(self, candidate: Mapping[str, Any]) -> str:
        error_type = str(candidate.get("error_type") or "Error").strip()
        component = str(candidate.get("component") or candidate.get("stage") or "runtime").strip()
        message = re.sub(r"\s+", " ", str(candidate.get("error_message") or "").strip())
        if len(message) > 90:
            message = message[:87].rstrip() + "..."
        title = f"[Bug] {component}: {error_type}"
        if message:
            title += f" - {message}"
        return title[:220]

    def _issue_body(self, candidate: Mapping[str, Any], reporter_record_id: str, bundle_path: Path) -> str:
        versions = [str(value) for value in candidate.get("versions_seen") or []]
        frames = candidate.get("frames") or []
        frame_text = "\n".join(f"- `{frame}`" for frame in frames) if frames else "- No normalized Python frames were available."
        message = _redact_text(str(candidate.get("error_message") or ""), self.project_root)
        return (
            "## ImageGen automated bug report\n\n"
            "> This issue was prepared locally by ImageGen and reviewed by the user before submission.\n\n"
            f"**Latest affected ImageGen version:** `{candidate.get('version') or 'unknown'}`\n\n"
            f"**Build:** `{candidate.get('build') or 'unknown'}`\n\n"
            f"**Failure area:** `{candidate.get('stage') or candidate.get('component') or 'unknown'}`\n\n"
            f"**Exception:** `{candidate.get('error_type') or 'Error'}`\n\n"
            "### Error\n\n"
            f"```text\n{message[:4000]}\n```\n\n"
            "### Deduplication\n\n"
            f"- Unique local occurrences: **{len(candidate.get('occurrences') or [])}**\n"
            f"- Versions observed: {', '.join(f'`{item}`' for item in versions) if versions else '`unknown`'}\n"
            "- The reporter selected the highest application version for the attached diagnostic bundle.\n\n"
            "### Stable stack signature\n\n"
            f"{frame_text}\n\n"
            "### Diagnostic ZIP\n\n"
            f"Attach `{bundle_path.name}` to this issue before submitting. The ZIP is generated under "
            "`artifacts/bugs/` and may contain prompts and generation settings needed for reproduction.\n\n"
            f"<!-- imagegen-bug-fingerprint: {candidate['fingerprint']} -->\n"
            f"<!-- imagegen-reporter-record: {reporter_record_id} -->\n"
            f"<!-- imagegen-bug-version: {candidate.get('version') or 'unknown'} -->\n"
        )

    def _issue_url(self, candidate: Mapping[str, Any], reporter_record_id: str, bundle_path: Path) -> str:
        query = urllib.parse.urlencode(
            {
                "title": self._issue_title(candidate),
                "body": self._issue_body(candidate, reporter_record_id, bundle_path),
            }
        )
        return f"{GITHUB_REPOSITORY_URL}/issues/new?{query}"

    def refresh_local(self) -> dict[str, Any]:
        ledger = self.load_ledger()
        reports: dict[str, Any] = dict(ledger.get("reports") or {})
        grouped = self._group_candidates(self.discover())
        now = _utc_now()

        for fingerprint, candidate in grouped.items():
            previous = dict(reports.get(fingerprint) or {})
            reporter_record_id = self._reporter_record_id(ledger, fingerprint)
            bundle_path, compact = self._ensure_bundle(candidate, reporter_record_id)
            record = {
                **previous,
                "fingerprint": fingerprint,
                "reporter_record_id": reporter_record_id,
                "kind": candidate.get("kind"),
                "component": candidate.get("component"),
                "operation": candidate.get("operation"),
                "stage": candidate.get("stage"),
                "error_type": candidate.get("error_type"),
                "error_message": _redact_text(str(candidate.get("error_message") or ""), self.project_root),
                "version": candidate.get("version"),
                "build": candidate.get("build"),
                "created_at": candidate.get("created_at"),
                "occurrence_count": len(candidate.get("occurrences") or []),
                "occurrences": candidate.get("occurrences") or [],
                "versions_seen": candidate.get("versions_seen") or [],
                "source_relative": candidate.get("source_relative"),
                "bundle_path": str(bundle_path),
                "bundle_relative": _safe_relative(bundle_path, self.project_root),
                "bundle_filename": bundle_path.name,
                "bundle_size": bundle_path.stat().st_size,
                "bundle_within_github_limit": bundle_path.stat().st_size <= GITHUB_ATTACHMENT_LIMIT_BYTES,
                "compact_bundle": compact,
                "issue_title": self._issue_title(candidate),
                "issue_body": self._issue_body(candidate, reporter_record_id, bundle_path),
                "new_issue_url": self._issue_url(candidate, reporter_record_id, bundle_path),
                "last_local_scan_at": now,
            }
            reports[fingerprint] = record

        # Keep previously submitted/linked records even if the original failure artifact was removed.
        for fingerprint, record in list(reports.items()):
            record["local_artifact_present"] = fingerprint in grouped

        ledger["reports"] = reports
        self.save_ledger(ledger)
        return self.payload(ledger)

    def _github_request(self, url: str) -> tuple[Any, Mapping[str, str]]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": f"ImageGen-BugReporter/{APPLICATION_VERSION}",
            },
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, dict(response.headers.items())

    def _list_github_issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for page in range(1, MAX_GITHUB_ISSUE_PAGES + 1):
            url = (
                f"{GITHUB_API_ROOT}/issues?state=all&sort=updated&direction=desc"
                f"&per_page=100&page={page}"
            )
            payload, _headers = self._github_request(url)
            if not isinstance(payload, list):
                raise BugReportError("GitHub returned an unexpected issue-list response.")
            current = [item for item in payload if isinstance(item, dict) and "pull_request" not in item]
            issues.extend(current)
            if len(payload) < 100:
                break
        return issues

    def _search_github_issues_by_fingerprint(self, fingerprint: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "q": f'repo:{GITHUB_OWNER}/{GITHUB_REPOSITORY} is:issue in:body "{fingerprint}"',
                "per_page": 20,
            }
        )
        payload, _headers = self._github_request(f"https://api.github.com/search/issues?{query}")
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise BugReportError("GitHub returned an unexpected issue-search response.")
        return [item for item in payload["items"] if isinstance(item, dict) and "pull_request" not in item]

    @staticmethod
    def _issue_markers(issue: Mapping[str, Any]) -> tuple[str, str, str]:
        body = str(issue.get("body") or "")
        fingerprint_match = _FINGERPRINT_MARKER_RE.search(body)
        reporter_match = _REPORTER_MARKER_RE.search(body)
        version_match = _VERSION_MARKER_RE.search(body)
        return (
            fingerprint_match.group(1).casefold() if fingerprint_match else "",
            reporter_match.group(1).casefold() if reporter_match else "",
            version_match.group(1).strip() if version_match else "",
        )

    def sync_github(self) -> dict[str, Any]:
        ledger = self.load_ledger()
        reports = dict(ledger.get("reports") or {})
        try:
            issues = self._list_github_issues()
        except (BugReportError, OSError, TimeoutError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            ledger["github_cache"] = {
                "updated_at": _utc_now(),
                "status": "unavailable",
                "message": f"GitHub issue synchronization is temporarily unavailable: {exc}",
            }
            self.save_ledger(ledger)
            return self.payload(ledger)

        by_fingerprint: dict[str, list[dict[str, Any]]] = {}
        for issue in issues:
            fingerprint, reporter_record_id, reported_version = self._issue_markers(issue)
            if not fingerprint:
                continue
            normalized = {
                "number": int(issue.get("number") or 0),
                "url": str(issue.get("html_url") or ""),
                "state": str(issue.get("state") or "open"),
                "state_reason": issue.get("state_reason"),
                "title": str(issue.get("title") or ""),
                "created_at": issue.get("created_at"),
                "updated_at": issue.get("updated_at"),
                "closed_at": issue.get("closed_at"),
                "reporter_record_id": reporter_record_id,
                "reported_version": reported_version,
            }
            by_fingerprint.setdefault(fingerprint, []).append(normalized)

        for fingerprint, record in reports.items():
            matches = by_fingerprint.get(fingerprint.casefold(), [])
            if not matches:
                record.pop("github_issue", None)
                record["github_match"] = "none"
                record["confirmed_reported"] = False
                continue
            local_record_id = str(record.get("reporter_record_id") or "").casefold()
            owned = [item for item in matches if item.get("reporter_record_id") == local_record_id]
            candidates = owned or matches
            selected = sorted(
                candidates,
                key=lambda item: (item.get("state") == "open", int(item.get("number") or 0)),
                reverse=True,
            )[0]
            record["github_issue"] = selected
            record["github_match"] = "local_report" if owned else "known_issue"
            record["confirmed_reported"] = bool(owned)

        ledger["reports"] = reports
        ledger["github_cache"] = {
            "updated_at": _utc_now(),
            "status": "ready",
            "message": f"Synchronized {len(issues)} public GitHub issues.",
            "issue_count_scanned": len(issues),
        }
        self.save_ledger(ledger)
        return self.payload(ledger)

    def mark_issue_opened(self, fingerprint: str) -> dict[str, Any]:
        fingerprint = str(fingerprint or "").strip().casefold()
        ledger = self.load_ledger()
        record = (ledger.get("reports") or {}).get(fingerprint)
        if not isinstance(record, dict):
            raise KeyError("Bug report not found.")

        # A previously synchronized match is authoritative enough to reuse.
        if record.get("github_match") in {"known_issue", "local_report"} and record.get("github_issue"):
            return {
                "fingerprint": fingerprint,
                "known_issue": True,
                "url": record["github_issue"].get("url"),
                "issue": record["github_issue"],
            }

        # Perform an exact, last-moment GitHub search before opening a new issue.
        # This protects against duplicates that are older than the normal sync window
        # or that were created by another user after the last background refresh.
        try:
            search_results = self._search_github_issues_by_fingerprint(fingerprint)
        except (BugReportError, OSError, TimeoutError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise BugReportError(
                "GitHub could not be checked for an existing copy of this bug. "
                "IMAGE_GEN will not open a new issue until duplicate verification succeeds."
            ) from exc

        matches: list[dict[str, Any]] = []
        local_record_id = str(record.get("reporter_record_id") or "").casefold()
        for issue in search_results:
            issue_fingerprint, reporter_record_id, reported_version = self._issue_markers(issue)
            if issue_fingerprint != fingerprint:
                continue
            matches.append(
                {
                    "number": int(issue.get("number") or 0),
                    "url": str(issue.get("html_url") or ""),
                    "state": str(issue.get("state") or "open"),
                    "state_reason": issue.get("state_reason"),
                    "title": str(issue.get("title") or ""),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "closed_at": issue.get("closed_at"),
                    "reporter_record_id": reporter_record_id,
                    "reported_version": reported_version,
                }
            )

        if matches:
            owned = [item for item in matches if item.get("reporter_record_id") == local_record_id]
            selected = sorted(
                owned or matches,
                key=lambda item: (item.get("state") == "open", int(item.get("number") or 0)),
                reverse=True,
            )[0]
            record["github_issue"] = selected
            record["github_match"] = "local_report" if owned else "known_issue"
            record["confirmed_reported"] = bool(owned)
            self.save_ledger(ledger)
            return {
                "fingerprint": fingerprint,
                "known_issue": True,
                "url": selected.get("url"),
                "issue": selected,
            }

        record["issue_opened_at"] = _utc_now()
        self.save_ledger(ledger)
        return {
            "fingerprint": fingerprint,
            "known_issue": False,
            "url": record.get("new_issue_url"),
            "title": record.get("issue_title"),
            "body": record.get("issue_body"),
            "bundle_filename": record.get("bundle_filename"),
        }

    def bundle_path(self, fingerprint: str) -> Path:
        ledger = self.load_ledger()
        record = (ledger.get("reports") or {}).get(str(fingerprint or "").strip().casefold())
        if not isinstance(record, dict):
            raise KeyError("Bug report not found.")
        path = Path(str(record.get("bundle_path") or ""))
        try:
            resolved = path.expanduser().resolve()
        except OSError as exc:
            raise FileNotFoundError("Bug-report bundle is unavailable.") from exc
        if not resolved.is_file() or self.bundle_root.resolve() not in resolved.parents:
            raise FileNotFoundError("Bug-report bundle is unavailable.")
        return resolved

    def reveal_bundle(self, fingerprint: str) -> Path:
        path = self.bundle_path(fingerprint)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(path)], start_new_session=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(path.parent)], start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BugReportError(f"Unable to reveal the bug-report ZIP: {exc}") from exc
        return path

    def payload(self, ledger: dict[str, Any] | None = None) -> dict[str, Any]:
        current = ledger or self.load_ledger()
        reports = list((current.get("reports") or {}).values())
        reports.sort(
            key=lambda item: (
                item.get("local_artifact_present", False),
                _version_key(item.get("version")),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        confirmed = [item for item in reports if item.get("confirmed_reported")]
        resolved = [
            item for item in confirmed
            if str((item.get("github_issue") or {}).get("state") or "").casefold() == "closed"
        ]
        open_reported = [
            item for item in confirmed
            if str((item.get("github_issue") or {}).get("state") or "").casefold() == "open"
        ]
        known_existing = [item for item in reports if item.get("github_match") == "known_issue"]
        pending = [
            item for item in reports
            if item.get("local_artifact_present")
            and item.get("github_match") not in {"known_issue", "local_report"}
        ]
        total_reported = len(confirmed)
        resolution_rate = (len(resolved) / total_reported) if total_reported else 0.0
        public_reports = []
        for item in reports:
            public_reports.append({key: value for key, value in item.items() if key != "bundle_path"})
        return {
            "schema": BUG_REPORT_SCHEMA,
            "repository": {
                "owner": GITHUB_OWNER,
                "name": GITHUB_REPOSITORY,
                "url": GITHUB_REPOSITORY_URL,
                "issues_url": f"{GITHUB_REPOSITORY_URL}/issues",
            },
            "storage": {
                "root": _safe_relative(self.bug_root, self.project_root),
                "ledger": _safe_relative(self.ledger_path, self.project_root),
                "attachment_limit_bytes": GITHUB_ATTACHMENT_LIMIT_BYTES,
            },
            "github": dict(current.get("github_cache") or {}),
            "profile": {
                "badge": f"Bug Reporter · {total_reported}",
                "reported": total_reported,
                "open": len(open_reported),
                "resolved": len(resolved),
                "resolution_rate": resolution_rate,
                "known_existing": len(known_existing),
                "pending": len(pending),
            },
            "reports": public_reports,
        }


__all__ = [
    "BUG_LEDGER_SCHEMA",
    "BUG_REPORT_SCHEMA",
    "BugReportError",
    "BugReportService",
    "GITHUB_ATTACHMENT_LIMIT_BYTES",
    "GITHUB_OWNER",
    "GITHUB_REPOSITORY",
]
