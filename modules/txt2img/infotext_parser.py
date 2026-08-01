from __future__ import annotations

import json
import re
from typing import Any

from modules.txt2img.field_aliases import FIELD_ALIASES, SPECIAL_FIELD_HANDLERS


_FIELD_SPLIT_RE = re.compile(r",\s*(?=[A-Za-z][A-Za-z0-9 _+./()-]*\s*:)")
_KNOWN_PARAM_KEYS = {
    "steps",
    "sampler",
    "sampler name",
    "schedule type",
    "scheduler",
    "cfg scale",
    "seed",
    "size",
    "model",
    "model hash",
    "vae",
    "vae hash",
    "clip skip",
    "ensd",
    "version",
}

# Local compatibility aliases for infotext import. These intentionally sit on top
# of field_aliases.py so older alias tables can still parse A1111-style exports.
_INFO_ALIAS_OVERRIDES = {
    "sampler": "sampler_label",
    "sampler name": "sampler_label",
    "schedule type": "scheduler_label",
    "scheduler": "scheduler_label",
    "model": "model_name",
}
_IGNORE_INFOTEXT_KEYS = {"version"}


def _clean_key(value: str) -> str:
    return str(value or "").strip().lower()



def _coerce_scalar(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        pass

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except Exception:
            return text

    return text



def _looks_like_parameter_line(line: str) -> bool:
    lowered = line.lower()
    return any(lowered.startswith(f"{key}:") for key in _KNOWN_PARAM_KEYS)



def _resolve_alias(key: str) -> str:
    if key in _INFO_ALIAS_OVERRIDES:
        return _INFO_ALIAS_OVERRIDES[key]
    return FIELD_ALIASES.get(key, key)



def _parse_parameter_line(line: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for token in _FIELD_SPLIT_RE.split(line.strip()):
        if ":" not in token:
            continue
        raw_key, raw_value = token.split(":", 1)
        key = _clean_key(raw_key)
        if key in _IGNORE_INFOTEXT_KEYS:
            continue

        value = raw_value.strip()
        alias = _resolve_alias(_clean_key(key))
        
        if alias in SPECIAL_FIELD_HANDLERS:
            handled = SPECIAL_FIELD_HANDLERS[alias](value)
            if handled:
                # Explicitly override size-related keys
                payload.update(handled)
            continue

        if alias in payload:
            # Preserve first occurrence (A1111-style behavior)
            continue

        coerced = _coerce_scalar(value)

        if alias in {"sampler_label", "scheduler_label", "model_name"}:
            coerced = str(coerced)

        if alias in payload:
            continue

        payload[alias] = coerced
    return payload



def parse_infotext(text: str) -> dict[str, Any]:
    """
    Parse A1111-style infotext into normalized request/extras payload keys.

    Rules:
    - The first unlabeled block is the positive prompt.
    - 'Negative prompt:' starts the negative prompt block.
    - Parameter lines such as 'Steps: ...' are parsed into fields.
    - A1111 'Version' is ignored for manifest generation.
    """
    raw = str(text or "").replace("", "").strip()
    if not raw:
        return {}

    lines = [line.rstrip() for line in raw.split("") if line.strip()]
    payload: dict[str, Any] = {}

    positive_prompt_lines: list[str] = []
    negative_prompt_lines: list[str] = []
    tail_parameter_lines: list[str] = []

    current_section = "positive"

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if lowered.startswith("negative prompt:"):
            current_section = "negative"
            value = stripped.split(":", 1)[1].strip()
            if value:
                negative_prompt_lines.append(value)
            continue

        if _looks_like_parameter_line(stripped):
            tail_parameter_lines.append(stripped)
            current_section = "params"
            continue

        if current_section == "positive":
            positive_prompt_lines.append(stripped)
        elif current_section == "negative":
            negative_prompt_lines.append(stripped)
        else:
            # tolerate stray non-parameter lines before/after params by folding
            # them back into the positive prompt rather than discarding them.
            positive_prompt_lines.append(stripped)

    if positive_prompt_lines:
        payload["positive_prompt"] = "".join(positive_prompt_lines).strip()
    if negative_prompt_lines:
        payload["negative_prompt"] = "".join(negative_prompt_lines).strip()
    else:
        payload["negative_prompt"] = ""

    for line in tail_parameter_lines:
        payload.update(_parse_parameter_line(line))

    return payload
