from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_SECRET_MARKERS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "session",
}


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.isoformat(), now.strftime("%Y%m%dT%H%M%S%fZ")


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "<maximum diagnostic depth reached>"
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key or "")
            lowered = key.casefold()
            if any(marker in lowered for marker in _SECRET_MARKERS):
                output[key] = "<redacted>"
            else:
                output[key] = _safe_value(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        text = value
        if isinstance(text, str) and len(text) > 20000:
            return text[:20000] + "\n<truncated>"
        return text
    return repr(value)


def _module_status() -> dict[str, Any]:
    modules = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "multipart": "python-multipart",
        "pydantic": "pydantic",
    }
    output: dict[str, Any] = {}
    for module_name, package_name in modules.items():
        try:
            installed = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        output[module_name] = {
            "installed": installed,
            "package": package_name,
            "install_command": f'"{sys.executable}" -m pip install {package_name}',
        }
    return output


def _project_root(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    # diagnostics.py -> webui -> image_gen -> src -> project root
    return Path(__file__).resolve().parents[3]


def write_webui_failure_bundle(
    *,
    project_root: str | Path | None,
    stage: str,
    error: BaseException,
    payload: Any = None,
    request_path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local, sanitized bundle for failures that occur before a job exists.

    Generation jobs already have their own diagnostics. This helper closes the
    startup/request-validation gap, including missing WebUI dependencies such as
    ``python-multipart`` that prevent FastAPI from creating the application.
    """

    root = _project_root(project_root)
    created_at, stamp = _utc_stamp()
    bundle_root = root / "artifacts" / "diagnostics" / "failures"
    bundle = bundle_root / f"webui-{stage.replace('_', '-')}-{stamp}-{uuid.uuid4().hex[:8]}"
    bundle.mkdir(parents=True, exist_ok=False)

    tb_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    module_status = _module_status()
    missing = [name for name, item in module_status.items() if not item["installed"]]
    hints: list[str] = []
    error_text = f"{type(error).__name__}: {error}"
    lowered = error_text.casefold()
    if "python-multipart" in lowered or ("form data requires" in lowered and "multipart" in lowered):
        hints.append(f'Install the required upload parser with: "{sys.executable}" -m pip install python-multipart')
    for name in missing:
        hints.append(module_status[name]["install_command"])

    report = {
        "schema": "image-gen-webui-failure-v1",
        "created_at": created_at,
        "stage": str(stage),
        "request_path": request_path,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "summary": error_text,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_prefix": sys.prefix,
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            "cwd": os.getcwd(),
            "project_root": str(root),
            "platform": platform.platform(),
            "module_status": module_status,
        },
        "submitted_payload": _safe_value(payload),
        "extra": _safe_value(dict(extra or {})),
        "reproduction": {
            "server_command": f'"{sys.executable}" -m image_gen.webui.server --project-root "{root}"',
            "install_hints": list(dict.fromkeys(hints)),
        },
        "files": {
            "traceback": "traceback.txt",
            "report": "report.json",
            "reproduction": "reproduction.txt",
        },
    }

    (bundle / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (bundle / "traceback.txt").write_text(tb_text, encoding="utf-8")
    reproduction_lines = [
        "IMAGE_GEN WebUI failure reproduction",
        "",
        f"Stage: {stage}",
        f"Error: {error_text}",
        f"Project root: {root}",
        f"Python: {sys.executable}",
        "",
        "Start command:",
        report["reproduction"]["server_command"],
    ]
    if hints:
        reproduction_lines.extend(["", "Suggested dependency repair:", *list(dict.fromkeys(hints))])
    (bundle / "reproduction.txt").write_text("\n".join(reproduction_lines) + "\n", encoding="utf-8")
    return bundle
