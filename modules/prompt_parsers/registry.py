from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from modules.prompt_parsers.adapters.legacy import LegacyPromptParserAdapter
from modules.prompt_parsers.adapters.parser21 import Parser21PromptParserAdapter
from modules.prompt_parsers.adapters.combined import CombinedPromptParserAdapter
from modules.prompt_parsers.contracts import PromptParserDescriptor, PromptParserProtocol


class PromptParserRegistry:
    def __init__(self, adapters: Iterable[PromptParserProtocol] | None = None) -> None:
        self._adapters: dict[str, PromptParserProtocol] = {}
        self._aliases: dict[str, str] = {}
        for adapter in adapters or (LegacyPromptParserAdapter(), Parser21PromptParserAdapter(), CombinedPromptParserAdapter()):
            self.register(adapter)

    @staticmethod
    def _token(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def register(self, adapter: PromptParserProtocol) -> None:
        descriptor = adapter.descriptor
        parser_id = self._token(descriptor.parser_id)
        if not parser_id:
            raise ValueError("Prompt parser descriptor must define parser_id.")
        if parser_id in self._adapters:
            raise ValueError(f"Prompt parser {parser_id!r} is already registered.")
        self._adapters[parser_id] = adapter
        for alias in (descriptor.parser_id, descriptor.label, *descriptor.aliases):
            token = self._token(alias)
            if token:
                self._aliases[token] = parser_id

    def resolve_id(self, value: Any) -> str:
        token = self._token(value or "legacy")
        parser_id = self._aliases.get(token, token)
        if parser_id not in self._adapters:
            available = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown prompt parser {value!r}. Available prompt parsers: {available}."
            )
        return parser_id

    def get(self, value: Any = "legacy") -> PromptParserProtocol:
        return self._adapters[self.resolve_id(value)]

    def descriptor(self, value: Any = "legacy") -> PromptParserDescriptor:
        return self.get(value).descriptor

    def availability(self, value: Any) -> tuple[bool, str]:
        try:
            adapter = self.get(value)
        except ValueError as exc:
            return False, str(exc)
        checker = getattr(adapter, "availability", None)
        if callable(checker):
            try:
                available, reason = checker()
                return bool(available), str(reason or "")
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"
        return True, ""

    def is_available(self, value: Any) -> bool:
        return self.availability(value)[0]

    def descriptors(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key in self._adapters:
            payload = self._adapters[key].descriptor.to_dict()
            available, reason = self.availability(key)
            payload["available"] = available
            payload["availability_error"] = reason
            output.append(payload)
        return output

    def has(self, value: Any, *, require_available: bool = False) -> bool:
        try:
            self.resolve_id(value)
            return self.is_available(value) if require_available else True
        except ValueError:
            return False


_DEFAULT_REGISTRY: PromptParserRegistry | None = None


def default_prompt_parser_registry() -> PromptParserRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PromptParserRegistry()
    return _DEFAULT_REGISTRY
