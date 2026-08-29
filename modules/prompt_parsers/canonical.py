from __future__ import annotations

import json
import re
from typing import Any, Mapping

from modules.prompt_parsers.contracts import (
    CANONICAL_PROMPT_CONTRACT_VERSION,
    LEGACY_CANONICAL_PROMPT_CONTRACT_VERSION,
)
from modules.parser.legacy_structured_prompt import normalize_legacy_structured_source
from modules.prompt_parsers.classic_semantic_contract import classic_semantic_nodes
from modules.prompt_parsers.ir import (
    PromptIR,
    parse_prompt_ir,
    prompt_ir_from_dict,
)

_WHITESPACE = re.compile(r"[ \t]+")
_EXTENSION_PATTERNS = {
    "BIND": re.compile(r"(?<!\\)\bBIND(?:2|3)?\b|\bBIND\s*\{", re.IGNORECASE),
    "CHUNK": re.compile(r"(?<!\\)\bCHUNK\b", re.IGNORECASE),
    "BLEND": re.compile(r"(?<!\\)\bBLEND\b", re.IGNORECASE),
    "MORPH": re.compile(r"(?<!\\)\bMORPH\b", re.IGNORECASE),
    "ASSEMBLE": re.compile(r"(?<!\\)\bASSEMBLE\b", re.IGNORECASE),
    "POOL": re.compile(r"(?<!\\)\bPOOL\b", re.IGNORECASE),
    "COMPOUND": re.compile(r"(?<!\\)\bCOMPOUND\b", re.IGNORECASE),
}


def normalize_prompt_source(prompt: str) -> str:
    lines = [line.rstrip() for line in str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(_WHITESPACE.sub(" ", line).strip() for line in lines).strip()


def _legacy_nodes(prompt: str) -> list[dict[str, Any]]:
    source = normalize_prompt_source(prompt)
    if not source:
        return [{"type": "text", "value": ""}]
    return classic_semantic_nodes(source)


def _extension_nodes(source: str, parser_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for operator, pattern in _EXTENSION_PATTERNS.items():
        for match in pattern.finditer(source):
            output.append({
                "type": "extension",
                "namespace": parser_id,
                "operator": operator,
                "source": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
    return output


def canonicalize_prompt(
    prompt: str,
    *,
    parser_id: str,
    semantic_ir: PromptIR | None = None,
) -> tuple[str, dict[str, Any], list[str]]:
    source = normalize_prompt_source(prompt)
    parser_namespace = str(parser_id or "legacy").strip().lower()
    classic_source, compatibility_warnings = normalize_legacy_structured_source(source)
    warnings: list[str] = list(compatibility_warnings)
    prompt_ir = semantic_ir or parse_prompt_ir(classic_source, parser_namespace=parser_namespace)
    for warning in prompt_ir.warnings:
        if warning not in warnings:
            warnings.append(warning)

    # Keep the proven PPSR-01 node surface for routing/UI compatibility while
    # canonical-v2 adds the complete parser-neutral semantic_ir payload.
    if parser_namespace == "legacy":
        nodes = _legacy_nodes(classic_source)
    elif parser_namespace == "combined":
        nodes = [*_legacy_nodes(classic_source), *_extension_nodes(classic_source, "parser21")]
    else:
        extension_nodes = _extension_nodes(classic_source, parser_namespace)
        nodes = [*_legacy_nodes(classic_source), *extension_nodes]

    structure = {
        "contract": CANONICAL_PROMPT_CONTRACT_VERSION,
        "compatibility_contracts": [LEGACY_CANONICAL_PROMPT_CONTRACT_VERSION],
        "parser_namespace": parser_namespace,
        "lossless_source": source,
        "classic_normalized_source": classic_source,
        "semantic_ir": prompt_ir.to_dict(),
        "nodes": nodes,
    }
    canonical = json.dumps(structure, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return canonical, structure, warnings


def canonical_ir_from_structure(value: Mapping[str, Any] | str) -> PromptIR:
    """Load Prompt IR from canonical v2 or migrate canonical v1 in memory.

    Old manifests remain replay-readable without rewriting the user's stored
    metadata.  V1 did not carry a live IR, so migration reparses its recorded
    normalized/lossless source through the locked Classic semantic contract.
    """

    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = dict(value or {})
    contract = str(data.get("contract") or "")
    parser_namespace = str(data.get("parser_namespace") or "legacy")

    if contract == CANONICAL_PROMPT_CONTRACT_VERSION:
        semantic = data.get("semantic_ir")
        if not isinstance(semantic, Mapping):
            raise ValueError("Canonical prompt v2 is missing semantic_ir.")
        return prompt_ir_from_dict(semantic)

    if contract == LEGACY_CANONICAL_PROMPT_CONTRACT_VERSION:
        source = str(
            data.get("classic_normalized_source")
            or data.get("lossless_source")
            or data.get("parser_input")
            or ""
        )
        return parse_prompt_ir(source, parser_namespace=parser_namespace)

    raise ValueError(f"Unsupported canonical prompt contract: {contract!r}")


def migrate_canonical_structure(value: Mapping[str, Any] | str) -> dict[str, Any]:
    """Return a canonical-v2 structure for v1/v2 recorded metadata."""

    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = dict(value or {})
    prompt_ir = canonical_ir_from_structure(data)
    source = str(data.get("lossless_source") or prompt_ir.raw_source or prompt_ir.normalized_source)
    parser_namespace = str(data.get("parser_namespace") or prompt_ir.parser_namespace or "legacy")
    _, structure, _ = canonicalize_prompt(source, parser_id=parser_namespace, semantic_ir=prompt_ir)
    return structure
