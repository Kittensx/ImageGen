from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PROMPT_SHORTCUT_CONTRACT_VERSION = "image-gen-prompt-shortcut-v1"

CANONICAL_OPERATOR_TOKENS: dict[str, str] = {
    "AND": "AND",
    "GROUP_OPEN": "{",
    "GROUP_CLOSE": "}",
    "SEQUENCE": "::",
    "DEEP_SEQUENCE": ":::",
    "CLOSE": "!!",
    "TOP_CLOSE": "!!!",
    "CHUNK": "CHUNK",
    "BLEND": "BLEND",
    "BIND": "BIND",
    "POOL": "POOL",
    "MORPH": "MORPH",
    "ASSEMBLE": "ASSEMBLE",
    "COMPOUND": "COMPOUND",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def stable_mapping_hash(payload: dict[str, Any]) -> str:
    normalized = dict(payload or {})
    normalized.pop("mapping_hash", None)
    encoded = json.dumps(_json_safe(normalized), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptShortcutProfileDescriptor:
    profile_id: str
    label: str
    version: str
    aliases: dict[str, tuple[str, ...]]
    parser_emitters: dict[str, dict[str, str]]
    compatible_parsers: tuple[str, ...] = ("legacy", "parser21", "superhybrid")
    escape_character: str = "\\"
    builtin: bool = True
    credit: str = ""
    description: str = ""
    source: str = "builtin"
    contract_version: str = PROMPT_SHORTCUT_CONTRACT_VERSION
    palette: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, builtin: bool | None = None) -> "PromptShortcutProfileDescriptor":
        aliases = {
            str(operator).strip().upper(): tuple(str(item) for item in (values or []) if str(item) != "")
            for operator, values in dict(payload.get("aliases") or payload.get("mappings") or {}).items()
        }
        parser_emitters = {
            str(parser).strip().lower(): {
                str(operator).strip().upper(): str(value)
                for operator, value in dict(values or {}).items()
            }
            for parser, values in dict(payload.get("parser_emitters") or {}).items()
        }
        profile_id = str(payload.get("profile_id") or payload.get("id") or "").strip()
        resolved_builtin = bool(payload.get("builtin", False)) if builtin is None else bool(builtin)
        return cls(
            profile_id=profile_id,
            label=str(payload.get("label") or profile_id or "Untitled Shortcut Profile").strip(),
            version=str(payload.get("version") or "1"),
            aliases=aliases,
            parser_emitters=parser_emitters,
            compatible_parsers=tuple(str(item).strip().lower() for item in (payload.get("compatible_parsers") or ("legacy", "parser21", "superhybrid")) if str(item).strip()),
            escape_character=str(payload.get("escape_character") or "\\"),
            builtin=resolved_builtin,
            credit=str(payload.get("credit") or ""),
            description=str(payload.get("description") or ""),
            source=str(payload.get("source") or ("builtin" if resolved_builtin else "user")),
            contract_version=str(payload.get("contract_version") or PROMPT_SHORTCUT_CONTRACT_VERSION),
            palette=tuple(dict(item) for item in (payload.get("palette") or []) if isinstance(item, dict)),
        )

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "contract_version": self.contract_version,
            "profile_id": self.profile_id,
            "label": self.label,
            "version": self.version,
            "aliases": {key: list(values) for key, values in sorted(self.aliases.items())},
            "parser_emitters": {
                parser: dict(sorted(values.items()))
                for parser, values in sorted(self.parser_emitters.items())
            },
            "compatible_parsers": list(self.compatible_parsers),
            "escape_character": self.escape_character,
            "builtin": self.builtin,
            "credit": self.credit,
            "description": self.description,
            "source": self.source,
            "palette": [_json_safe(item) for item in self.palette],
        }
        payload["mapping_hash"] = stable_mapping_hash(payload)
        return payload

    @property
    def mapping_hash(self) -> str:
        return str(self.snapshot()["mapping_hash"])

    def to_dict(self, *, parser_id: str | None = None) -> dict[str, Any]:
        payload = self.snapshot()
        if parser_id:
            payload["palette"] = self.palette_for_parser(parser_id)
        return payload

    def palette_for_parser(self, parser_id: str) -> list[dict[str, Any]]:
        parser = str(parser_id or "legacy").strip().lower()
        output: list[dict[str, Any]] = []
        for item in self.palette:
            parsers = [str(value).strip().lower() for value in (item.get("parsers") or [])]
            if parsers and parser not in parsers and not (parser == "combined" and "parser21" in parsers):
                continue
            operator = str(item.get("operator") or "").strip().upper()
            aliases = list(self.aliases.get(operator) or ())
            value = dict(item)
            if aliases:
                value.setdefault("alias", aliases[0])
                value["label"] = str(value.get("label") or aliases[0])
            value["operator"] = operator
            output.append(value)
        return output


@dataclass(frozen=True)
class PromptShortcutValidationIssue:
    severity: str
    code: str
    message: str
    operator: str = ""
    alias: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "operator": self.operator,
            "alias": self.alias,
        }


@dataclass
class PromptShortcutValidationResult:
    valid: bool
    issues: list[PromptShortcutValidationIssue] = field(default_factory=list)
    mapping_hash: str = ""

    @property
    def errors(self) -> list[PromptShortcutValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[PromptShortcutValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "mapping_hash": self.mapping_hash,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class PromptTranslationResult:
    raw_prompt: str
    parser_input: str
    canonical_prompt: str
    canonical_structure: dict[str, Any]
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "raw_prompt": self.raw_prompt,
            "parser_input": self.parser_input,
            "canonical_prompt": self.canonical_prompt,
            "canonical_structure": _json_safe(self.canonical_structure),
            "substitutions": _json_safe(self.substitutions),
            "warnings": list(self.warnings),
            "diagnostics": _json_safe(self.diagnostics),
        }


class PromptShortcutError(ValueError):
    def __init__(self, message: str, *, error_kind: str = "prompt_shortcut_error", diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.diagnostics = dict(diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_kind": self.error_kind,
            "message": str(self),
            "diagnostics": _json_safe(self.diagnostics),
        }
