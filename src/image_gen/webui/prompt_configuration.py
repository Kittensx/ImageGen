from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from image_gen.webui.store import WebUIStore
from modules.prompt_parsers import (
    PromptParserRegistry,
    PromptProcessingPreflight,
    default_prompt_parser_registry,
)
from modules.prompt_shortcuts import (
    BUILTIN_PARSER_PRESETS,
    PromptShortcutProfileDescriptor,
    PromptShortcutProfileRegistry,
    PromptShortcutTranslator,
    validate_prompt_shortcut_profile,
)


class PromptConfigurationService:
    """WebUI-facing prompt parser, shortcut profile, and preset coordination."""

    def __init__(
        self,
        store: WebUIStore,
        *,
        parser_registry: PromptParserRegistry | None = None,
    ) -> None:
        self.store = store
        self.parser_registry = parser_registry or default_prompt_parser_registry()
        self.translator = PromptShortcutTranslator()

    @staticmethod
    def _token(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def profile_registry(self) -> PromptShortcutProfileRegistry:
        registry = PromptShortcutProfileRegistry()
        for payload in self.store.list_prompt_shortcut_profiles():
            record = dict(payload)
            record.pop("file_name", None)
            try:
                registry.register_payload(record, replace=False)
            except ValueError:
                # Invalid user profiles remain inspectable through the raw store,
                # but are not activatable until the user fixes and validates them.
                continue
        return registry

    def list_profiles(self) -> list[dict[str, Any]]:
        registry = self.profile_registry()
        output: list[dict[str, Any]] = []
        for payload in registry.descriptors():
            payload["palettes"] = {
                parser["parser_id"]: registry.get(payload["profile_id"]).palette_for_parser(parser["parser_id"])
                for parser in self.parser_registry.descriptors()
                if parser.get("available")
            }
            validation = validate_prompt_shortcut_profile(registry.get(payload["profile_id"]))
            payload["validation"] = validation.to_dict()
            output.append(payload)

        registered = {item["profile_id"] for item in output}
        for raw in self.store.list_prompt_shortcut_profiles():
            profile_id = str(raw.get("profile_id") or "").strip()
            if not profile_id or profile_id in registered:
                continue
            validation = validate_prompt_shortcut_profile(raw)
            output.append({
                **{key: value for key, value in raw.items() if key != "file_name"},
                "profile_id": profile_id,
                "builtin": False,
                "source": "user",
                "valid": False,
                "validation": validation.to_dict(),
                "palettes": {},
                "palette": [],
            })
        return sorted(output, key=lambda item: (not bool(item.get("builtin")), str(item.get("label") or item.get("profile_id")).casefold()))

    def validate_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        profile = PromptShortcutProfileDescriptor.from_dict(dict(payload or {}), builtin=False)
        result = validate_prompt_shortcut_profile(profile)
        return {**result.to_dict(), "profile": profile.snapshot()}

    def save_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        profile = PromptShortcutProfileDescriptor.from_dict(dict(payload or {}), builtin=False)
        builtins = PromptShortcutProfileRegistry()
        if builtins.has(profile.profile_id) and builtins.get(profile.profile_id).builtin:
            raise ValueError("Built-in shortcut profiles are read-only. Duplicate it with a new profile_id first.")
        validation = validate_prompt_shortcut_profile(profile)
        if not validation.valid:
            raise ValueError("Shortcut profile validation failed: " + " | ".join(issue.message for issue in validation.errors))
        return self.store.save_prompt_shortcut_profile(profile.snapshot())

    def delete_profile(self, profile_id: str) -> bool:
        builtins = PromptShortcutProfileRegistry()
        if builtins.has(profile_id) and builtins.get(profile_id).builtin:
            raise ValueError("Built-in shortcut profiles cannot be deleted.")
        return self.store.delete_prompt_shortcut_profile(profile_id)

    def parser_presets(self) -> list[dict[str, Any]]:
        output = [copy.deepcopy(item) for item in BUILTIN_PARSER_PRESETS]
        builtins = {self._token(item.get("preset_id")) for item in output}
        for item in self.store.list_prompt_parser_presets():
            record = {key: value for key, value in item.items() if key != "file_name"}
            preset_id = self._token(record.get("preset_id") or record.get("name"))
            if not preset_id or preset_id in builtins:
                continue
            record["preset_id"] = preset_id
            record["builtin"] = False
            output.append(record)
        return sorted(output, key=lambda item: (not bool(item.get("builtin")), str(item.get("name") or item.get("preset_id")).casefold()))

    def resolve_preset(self, value: Any) -> dict[str, Any] | None:
        token = self._token(value)
        if not token:
            return None
        for preset in self.parser_presets():
            if token in {self._token(preset.get("preset_id")), self._token(preset.get("name"))}:
                return copy.deepcopy(preset)
        raise ValueError(f"Unknown prompt parser preset {value!r}.")

    def save_preset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        preset_id = self._token(record.get("preset_id") or record.get("name"))
        if not preset_id:
            raise ValueError("Prompt parser preset name is required.")
        if any(self._token(item.get("preset_id")) == preset_id and item.get("builtin") for item in BUILTIN_PARSER_PRESETS):
            raise ValueError("Built-in prompt parser presets are read-only.")
        parser_id = self.parser_registry.resolve_id(record.get("prompt_parser_name") or "legacy")
        fallback = "legacy_default" if parser_id == "legacy" else ("parser21_native" if parser_id == "parser21" else ("superhybrid_native" if parser_id == "superhybrid" else "canonical"))
        profile = self.profile_registry().get(record.get("shortcut_profile_name") or fallback)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
        )
        if not compatible:
            raise ValueError(f"Shortcut profile {profile.profile_id!r} is not compatible with parser {parser_id!r}.")
        record.update({
            "preset_id": preset_id,
            "name": str(record.get("name") or preset_id).strip(),
            "prompt_parser_name": parser_id,
            "shortcut_profile_name": profile.profile_id,
            "prompt_parser_kwargs": dict(record.get("prompt_parser_kwargs") or {}),
            "fallback_policy": str(record.get("fallback_policy") or "fail"),
            "hires_inheritance": str(record.get("hires_inheritance") or "same_as_base"),
            "builtin": False,
        })
        return self.store.save_prompt_parser_preset(record)

    def delete_preset(self, preset_id: str) -> bool:
        token = self._token(preset_id)
        if any(self._token(item.get("preset_id")) == token and item.get("builtin") for item in BUILTIN_PARSER_PRESETS):
            raise ValueError("Built-in prompt parser presets cannot be deleted.")
        return self.store.delete_prompt_parser_preset(token)

    def resolve_profile(
        self,
        profile_name: Any,
        *,
        parser_id: str,
        snapshot: Mapping[str, Any] | None = None,
    ) -> PromptShortcutProfileDescriptor:
        if snapshot:
            profile = PromptShortcutProfileDescriptor.from_dict(dict(snapshot), builtin=bool(snapshot.get("builtin", False)))
            validation = validate_prompt_shortcut_profile(profile)
            if not validation.valid:
                raise ValueError("Embedded shortcut profile snapshot is invalid: " + " | ".join(issue.message for issue in validation.errors))
        else:
            fallback = "legacy_default" if parser_id == "legacy" else ("parser21_native" if parser_id == "parser21" else ("superhybrid_native" if parser_id == "superhybrid" else "canonical"))
            profile = self.profile_registry().get(profile_name or fallback)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
        )
        if not compatible:
            raise ValueError(f"Shortcut profile {profile.profile_id!r} is not compatible with parser {parser_id!r}.")
        return profile

    def _apply_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset_name = payload.get("prompt_parser_preset_name")
        preset = self.resolve_preset(preset_name) if preset_name else None
        if not preset:
            return payload
        output = dict(payload)
        output.setdefault("prompt_parser_name", preset.get("prompt_parser_name"))
        output.setdefault("prompt_shortcut_profile_name", preset.get("shortcut_profile_name"))
        if not output.get("prompt_parser_kwargs"):
            output["prompt_parser_kwargs"] = dict(preset.get("prompt_parser_kwargs") or {})
        output["prompt_parser_preset_name"] = str(preset.get("preset_id") or preset.get("name") or preset_name)
        return output

    def preflight_report(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        prepared = self._apply_preset(dict(payload or {}))
        validator = PromptProcessingPreflight(
            parser_registry=self.parser_registry,
            profile_registry=self.profile_registry(),
        )
        report = validator.validate(prepared)
        report["prompt_parser_preset_name"] = str(prepared.get("prompt_parser_preset_name") or "")
        return report

    def prepare_generation_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        output = self._apply_preset(dict(payload or {}))
        report = self.preflight_report(output)
        if not report.get("valid"):
            messages = " | ".join(
                str(item.get("message") or "Prompt validation failed.")
                for item in report.get("blocking_errors") or []
            )
            raise ValueError(f"Prompt preflight failed: {messages}")
        output.update(copy.deepcopy(report.get("normalized_fields") or {}))
        output["prompt_parser_preset_name"] = str(output.get("prompt_parser_preset_name") or "")
        output["prompt_preflight"] = report
        return output

    def translate_preview(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        report = self.preflight_report(payload)
        base = dict(report.get("base") or {})
        hires = dict(report.get("hires") or {})
        return {
            "valid": bool(report.get("valid")),
            "contract_version": report.get("contract_version"),
            "prompt_parser": base.get("parser") or {},
            "shortcut_profile": base.get("shortcut_profile") or {},
            "prompt_parser_preset_name": report.get("prompt_parser_preset_name", ""),
            "positive": base.get("positive") or {},
            "negative": base.get("negative") or {},
            "hires": hires,
            "messages": list(report.get("messages") or []),
            "blocking_errors": list(report.get("blocking_errors") or []),
            "behavior_warnings": list(report.get("behavior_warnings") or []),
            "informational_notices": list(report.get("informational_notices") or []),
            "warnings": [
                str(item.get("message") or "")
                for item in [
                    *(report.get("behavior_warnings") or []),
                    *(report.get("informational_notices") or []),
                ]
                if item.get("message")
            ],
            "normalized_fields": dict(report.get("normalized_fields") or {}),
        }

    def bootstrap_payload(self) -> dict[str, Any]:
        return {
            "prompt_parsers": self.parser_registry.descriptors(),
            "prompt_shortcut_profiles": self.list_profiles(),
            "prompt_parser_presets": self.parser_presets(),
        }


def safe_profile_id_from_label(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or "").strip()).strip("._-")
    return value.lower() or "custom_profile"
