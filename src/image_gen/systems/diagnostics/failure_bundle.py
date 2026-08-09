from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

from image_gen.program_metadata import build_program_metadata

from image_gen.systems.diagnostics.models import DiagnosticSession
from image_gen.systems.diagnostics.serialization import json_safe


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._-") or "unknown"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value, redact_secrets=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _prompt_parser_failure_payload(
    *,
    error: BaseException,
    error_details: Any,
    request_payload: Any,
    request_extras: Any,
) -> dict[str, Any] | None:
    details = dict(error_details or {}) if isinstance(error_details, dict) else {}
    extras = dict(request_extras or {}) if isinstance(request_extras, dict) else {}
    nested = extras.get("prompt_parser_failure")
    if not details and isinstance(nested, dict):
        details = dict(nested)
    parser_id = str(details.get("parser_id") or details.get("parser") or "")
    error_kind = str(details.get("error_kind") or "")
    diagnostics = dict(details.get("diagnostics") or {}) if isinstance(details.get("diagnostics"), dict) else {}
    looks_like_parser_failure = bool(
        parser_id
        or error_kind.startswith("prompt_")
        or error_kind.startswith("shortcut_")
        or "parser" in type(error).__name__.lower()
        or "shortcut" in type(error).__name__.lower()
    )
    if not looks_like_parser_failure:
        return None

    request = dict(request_payload or {}) if isinstance(request_payload, dict) else {}
    prompt_role = str(details.get("prompt_role") or diagnostics.get("prompt_role") or "positive")
    translation_key = "negative_translation" if prompt_role == "negative" else "positive_translation"
    translation = diagnostics.get(translation_key)
    if not isinstance(translation, dict):
        translation = {}
    raw_prompt = str(
        translation.get("raw_prompt")
        or request.get("negative_prompt" if prompt_role == "negative" else "positive_prompt")
        or ""
    )
    parser_input = str(translation.get("parser_input") or raw_prompt)
    canonical_prompt = str(translation.get("canonical_prompt") or "")
    shortcut_profile = diagnostics.get("shortcut_profile")
    if not isinstance(shortcut_profile, dict):
        shortcut_profile = {}

    token = str(diagnostics.get("token") or diagnostics.get("error_token") or diagnostics.get("alias") or "")
    position = diagnostics.get("position", diagnostics.get("error_start"))
    try:
        error_start = max(0, int(position)) if position is not None else None
    except (TypeError, ValueError):
        error_start = None
    error_end_value = diagnostics.get("error_end")
    try:
        error_end = max(error_start or 0, int(error_end_value)) if error_end_value is not None else None
    except (TypeError, ValueError):
        error_end = None
    if error_start is None and token:
        found = parser_input.find(token)
        error_start = found if found >= 0 else None
    if error_start is not None and error_end is None:
        error_end = error_start + max(1, len(token))

    excerpt = ""
    caret_excerpt = ""
    if parser_input:
        if error_start is None:
            excerpt = parser_input[:240]
        else:
            window_start = max(0, error_start - 80)
            window_end = min(len(parser_input), max(error_end or error_start + 1, error_start + 1) + 80)
            excerpt = parser_input[window_start:window_end]
            caret_offset = max(0, error_start - window_start)
            caret_width = max(1, (error_end or error_start + 1) - error_start)
            caret_excerpt = " " * caret_offset + "^" * min(caret_width, max(1, len(excerpt) - caret_offset))

    return {
        "format": "image-gen-prompt-parser-failure-v1",
        "parser": {
            "id": parser_id or str(request.get("prompt_parser_name") or "unknown"),
            "version": str(details.get("parser_version") or ""),
        },
        "prompt_role": prompt_role,
        "shortcut_profile": shortcut_profile,
        "raw_prompt": raw_prompt,
        "expanded_prompt": parser_input,
        "canonical_prompt": canonical_prompt,
        "parser_input": parser_input,
        "parser_options": dict(request.get("prompt_parser_kwargs") or {}),
        "error_type": type(error).__name__,
        "error_kind": error_kind or "prompt_parse_error",
        "error_message": str(details.get("message") or error),
        "error_token": token,
        "error_start": error_start,
        "error_end": error_end,
        "prompt_excerpt": excerpt,
        "caret_excerpt": caret_excerpt,
        "translation_stages": {
            "positive": diagnostics.get("positive_translation") or {},
            "negative": diagnostics.get("negative_translation") or {},
        },
        "diagnostics": diagnostics,
        "fallback_attempted": bool(diagnostics.get("fallback_attempted", False)),
    }


def _prompt_parser_failure_text(payload: dict[str, Any]) -> str:
    parser = dict(payload.get("parser") or {})
    lines = [
        "IMAGE_GEN Prompt Parser Failure",
        "=" * 31,
        f"Parser: {parser.get('id') or 'unknown'}{(' v' + str(parser.get('version'))) if parser.get('version') else ''}",
        f"Prompt role: {str(payload.get('prompt_role') or 'unknown').title()}",
        f"Error kind: {payload.get('error_kind') or 'prompt_parse_error'}",
        f"Error: {payload.get('error_message') or ''}",
        "",
        "Exact parser input:",
        str(payload.get("parser_input") or ""),
    ]
    if payload.get("prompt_excerpt"):
        lines.extend(["", "Failure location:", str(payload.get("prompt_excerpt"))])
        if payload.get("caret_excerpt"):
            lines.append(str(payload.get("caret_excerpt")))
    lines.extend([
        "",
        "Raw user prompt:",
        str(payload.get("raw_prompt") or ""),
        "",
        "Canonical prompt available before failure:",
        str(payload.get("canonical_prompt") or "(not available)"),
        "",
        "See prompt_parser_failure.json for the complete structured diagnostics.",
    ])
    return "\n".join(lines) + "\n"


def write_failure_bundle(
    session: DiagnosticSession,
    *,
    system: str,
    operation: str,
    error: BaseException,
) -> Path:
    """Persist a self-contained, sanitized failure-reproduction bundle."""

    bundle_dir = (
        session.config.artifacts_root
        / "failures"
        / f"{session.run_id}_{_safe_name(system)}_{_safe_name(operation)}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)

    request_payload = (
        session.request.to_serializable_dict()
        if hasattr(session.request, "to_serializable_dict")
        else json_safe(session.request)
    )
    reproduction_payload = dict(request_payload or {})
    reproduction_payload.update(json_safe(session.request_extras, redact_secrets=True) or {})

    error_details = None
    if hasattr(error, "to_dict") and callable(error.to_dict):
        try:
            error_details = error.to_dict()
        except Exception:
            error_details = None
    prompt_failure = _prompt_parser_failure_payload(
        error=error,
        error_details=error_details,
        request_payload=request_payload,
        request_extras=session.request_extras,
    )

    project_root = session.effective_config.get("project_root") or "."
    failure = {
        "format": "image-gen-failure-bundle-v1",
        "application": build_program_metadata(project_root),
        "run_id": session.run_id,
        "started_utc": session.started_utc,
        "system": system,
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_details": error_details,
        "diagnostics": session.config.to_dict(),
        "bundle_files": [
            "failure.json",
            "request.json",
            "request_extras.json",
            "effective_config.json",
            "components.json",
            "schedule.json",
            "sampler.json",
            "timings.json",
            "tensor_summaries.json",
            "events.jsonl",
            "traceback.txt",
            "reproduction_request.json",
            "reproduce_command.txt",
            *(["prompt_parser_failure.json", "prompt_parser_failure.txt"] if prompt_failure else []),
        ],
    }

    _write_json(bundle_dir / "failure.json", failure)
    _write_json(bundle_dir / "request.json", request_payload)
    _write_json(bundle_dir / "request_extras.json", session.request_extras)
    if prompt_failure:
        _write_json(bundle_dir / "prompt_parser_failure.json", prompt_failure)
        (bundle_dir / "prompt_parser_failure.txt").write_text(
            _prompt_parser_failure_text(prompt_failure),
            encoding="utf-8",
        )
    _write_json(bundle_dir / "effective_config.json", session.effective_config)
    _write_json(bundle_dir / "components.json", session.component_report)
    _write_json(bundle_dir / "schedule.json", session.schedule_report)
    _write_json(bundle_dir / "sampler.json", session.sampler_report)
    _write_json(bundle_dir / "timings.json", [item.to_dict() for item in session.timings])
    _write_json(bundle_dir / "tensor_summaries.json", session.tensor_summaries)
    _write_json(bundle_dir / "reproduction_request.json", reproduction_payload)

    events_path = bundle_dir / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event.to_dict(), ensure_ascii=False) + "\n" for event in session.events),
        encoding="utf-8",
    )
    (bundle_dir / "traceback.txt").write_text(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        encoding="utf-8",
    )

    config_path = session.effective_config.get("config_path")
    context_args = f'--project-root "{project_root}" '
    if config_path:
        context_args += f'--project-config "{config_path}" '
    command = (
        f'cd /d "{project_root}"\n'
        'set "PYTHON_EXE="\n'
        'if exist ".venv\\Scripts\\python.exe" set "PYTHON_EXE=.venv\\Scripts\\python.exe"\n'
        'if not defined PYTHON_EXE if exist "venv\\Scripts\\python.exe" set "PYTHON_EXE=venv\\Scripts\\python.exe"\n'
        'if not defined PYTHON_EXE set "PYTHON_EXE=py -3.10"\n'
        f'%PYTHON_EXE% -m modules.txt2img.cli run '
        f'{context_args}'
        f'--config "{bundle_dir / "reproduction_request.json"}" '
        f'--diagnostics verbose\n'
    )
    (bundle_dir / "reproduce_command.txt").write_text(command, encoding="utf-8")
    return bundle_dir
