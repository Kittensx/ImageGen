from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SENSITIVE_QUERY = re.compile(r"(?i)(token|api[_-]?key|authorization|signature|sig|expires)=([^&\s]+)")


def strip_url_query(value: str) -> str:
    try:
        parsed = urlsplit(str(value or ""))
    except Exception:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return str(value or "")[:2048]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:2048]


def redact_text(value: Any, *, secrets: Iterable[str] = ()) -> str:
    text = str(value or "")
    for secret in secrets:
        token = str(secret or "")
        if token:
            text = text.replace(token, "[REDACTED]")
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _SENSITIVE_QUERY.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text[:1024]


def _sanitize_payload(value: Any, *, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.casefold() in {"authorization", "cookie", "cookies", "api_key", "apikey", "token", "secret", "signed_url", "delivery_url"}:
                continue
            if name in {"redirectHostChain", "redirect_host_chain"} and isinstance(item, (list, tuple)):
                output[name] = [strip_url_query(str(entry)) for entry in item]
            else:
                output[name] = _sanitize_payload(item, secrets=secrets)
        return output
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    return value


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any], *, secrets: Iterable[str] = ()) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_payload(dict(payload), secrets=tuple(str(item or "") for item in secrets if str(item or "")))
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


class DownloadReportWriter:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def report_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def write(self, job_id: str, payload: Mapping[str, Any], *, secrets: Iterable[str] = ()) -> Path:
        return write_json_atomic(self.report_path(job_id), payload, secrets=secrets)
