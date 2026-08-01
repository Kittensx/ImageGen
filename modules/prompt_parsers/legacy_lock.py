from __future__ import annotations

from typing import Any, Mapping

from modules.prompt_shortcuts import default_prompt_shortcut_registry

LEGACY_PROMPT_PARSER_ID = "legacy"
LEGACY_SHORTCUT_PROFILE_ID = "legacy_default"


def legacy_shortcut_profile_snapshot() -> dict[str, Any]:
    try:
        return dict(default_prompt_shortcut_registry().get(LEGACY_SHORTCUT_PROFILE_ID).snapshot())
    except Exception:
        return {}


def force_legacy_prompt_mode(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compatibility shim retained for callers from the Phase 13F-5 branch.

    Parser selection is active again. This function intentionally performs no
    normalization so an older extension importing it cannot silently reapply
    the retired legacy-parser lock.
    """

    return dict(payload or {})
