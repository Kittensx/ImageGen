from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from modules.prompt_shortcuts.builtins import builtin_prompt_shortcut_profiles
from modules.prompt_shortcuts.contracts import PromptShortcutProfileDescriptor
from modules.prompt_shortcuts.validation import validate_prompt_shortcut_profile


class PromptShortcutProfileRegistry:
    def __init__(self, profiles: Iterable[PromptShortcutProfileDescriptor] | None = None) -> None:
        self._profiles: dict[str, PromptShortcutProfileDescriptor] = {}
        for profile in profiles or builtin_prompt_shortcut_profiles():
            self.register(profile)

    @staticmethod
    def _token(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def register(self, profile: PromptShortcutProfileDescriptor, *, replace: bool = False) -> None:
        validation = validate_prompt_shortcut_profile(profile)
        if not validation.valid:
            messages = "; ".join(issue.message for issue in validation.errors)
            raise ValueError(f"Invalid prompt shortcut profile {profile.profile_id!r}: {messages}")
        profile_id = self._token(profile.profile_id)
        if not profile_id:
            raise ValueError("Prompt shortcut profile must define profile_id.")
        if profile_id in self._profiles and not replace:
            raise ValueError(f"Prompt shortcut profile {profile_id!r} is already registered.")
        self._profiles[profile_id] = profile

    def register_payload(self, payload: dict[str, Any], *, replace: bool = False) -> PromptShortcutProfileDescriptor:
        profile = PromptShortcutProfileDescriptor.from_dict(payload, builtin=False)
        self.register(profile, replace=replace)
        return profile

    def get(self, value: Any = "legacy_default") -> PromptShortcutProfileDescriptor:
        profile_id = self._token(value or "legacy_default")
        if profile_id not in self._profiles:
            available = ", ".join(sorted(self._profiles))
            raise ValueError(f"Unknown prompt shortcut profile {value!r}. Available profiles: {available}.")
        return self._profiles[profile_id]

    def has(self, value: Any) -> bool:
        try:
            self.get(value)
            return True
        except ValueError:
            return False

    def descriptors(self, *, parser_id: str | None = None) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key in sorted(self._profiles):
            profile = self._profiles[key]
            payload = profile.to_dict(parser_id=parser_id)
            payload["valid"] = True
            output.append(payload)
        return output


_DEFAULT_REGISTRY: PromptShortcutProfileRegistry | None = None


def default_prompt_shortcut_registry() -> PromptShortcutProfileRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PromptShortcutProfileRegistry()
    return _DEFAULT_REGISTRY
