from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

PROMPT_PARSER_CONTRACT_VERSION = "image-gen-prompt-parser-v1"
CANONICAL_PROMPT_CONTRACT_VERSION = "image-gen-canonical-prompt-v1"


@dataclass(frozen=True)
class PromptParserDescriptor:
    parser_id: str
    label: str
    version: str
    contract_version: str = PROMPT_PARSER_CONTRACT_VERSION
    aliases: tuple[str, ...] = ()
    capabilities: dict[str, Any] = field(default_factory=dict)
    experimental: bool = False
    credit: str = ""
    source_url: str = ""
    settings_schema: dict[str, Any] = field(default_factory=dict)
    process_scoped_settings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "label": self.label,
            "version": self.version,
            "contract_version": self.contract_version,
            "aliases": list(self.aliases),
            "capabilities": dict(self.capabilities),
            "experimental": bool(self.experimental),
            "credit": self.credit,
            "source_url": self.source_url,
            "settings_schema": dict(self.settings_schema),
            "process_scoped_settings": list(self.process_scoped_settings),
        }


@dataclass
class PromptParseRequest:
    raw_prompt: str
    prompt_role: str
    steps: int
    hires_steps: int | None
    parser_options: dict[str, Any]
    model_context: Any
    shared_state: Any
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    recorded_route_plan: dict[str, Any] | None = None


@dataclass
class PromptParseResult:
    parser_id: str
    parser_version: str
    parser_contract_version: str
    raw_prompt: str
    canonical_prompt: str
    canonical_structure: dict[str, Any]
    schedules: Any
    conditioning_source: Any
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    directives: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "parser_contract_version": self.parser_contract_version,
            "raw_prompt": self.raw_prompt,
            "canonical_prompt": self.canonical_prompt,
            "canonical_structure": self.canonical_structure,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "directives": dict(self.directives),
        }


class PromptParserError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        parser_id: str,
        prompt_role: str,
        error_kind: str = "prompt_parse_error",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.parser_id = parser_id
        self.prompt_role = prompt_role
        self.error_kind = error_kind
        self.diagnostics = dict(diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "prompt_role": self.prompt_role,
            "error_kind": self.error_kind,
            "message": str(self),
            "diagnostics": dict(self.diagnostics),
        }


@runtime_checkable
class PromptParserProtocol(Protocol):
    descriptor: PromptParserDescriptor

    def parse(self, request: PromptParseRequest) -> PromptParseResult:
        ...
