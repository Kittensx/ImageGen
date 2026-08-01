from __future__ import annotations

import json
import re
from typing import Any

from modules.prompt_parsers.contracts import CANONICAL_PROMPT_CONTRACT_VERSION

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
    nodes: list[dict[str, Any]] = []
    source = normalize_prompt_source(prompt)
    if not source:
        return [{"type": "text", "value": ""}]

    if " AND " in f" {source} ":
        branches = [item.strip() for item in re.split(r"\s+AND\s+", source) if item.strip()]
        nodes.append({"type": "conjunction", "branches": [{"type": "text", "value": item} for item in branches]})
    if re.search(r"\[[^\]]*:[^\]]*:[^\]]*\]", source):
        nodes.append({"type": "scheduled_text", "source": source})
    if re.search(r"\[[^\]|]+(?:\|[^\]]+)+\]", source):
        nodes.append({"type": "alternate_text", "source": source})
    if re.search(r"\([^()]+:[-+]?\d+(?:\.\d+)?\)", source):
        nodes.append({"type": "weighted_text", "source": source})
    elif "(" in source or "[" in source:
        nodes.append({"type": "attention_group", "source": source})
    if ":::" in source:
        nodes.append({"type": "deep_sequence", "source": ":::"})
    elif "::" in source:
        nodes.append({"type": "sequence", "source": "::"})
    return nodes or [{"type": "text", "value": source}]


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


def canonicalize_prompt(prompt: str, *, parser_id: str) -> tuple[str, dict[str, Any], list[str]]:
    source = normalize_prompt_source(prompt)
    parser_namespace = str(parser_id or "legacy").strip().lower()
    warnings: list[str] = []
    if parser_namespace == "legacy":
        nodes = _legacy_nodes(source)
    elif parser_namespace == "combined":
        nodes = [*_legacy_nodes(source), *_extension_nodes(source, "parser21")]
    else:
        extension_nodes = _extension_nodes(source, parser_namespace)
        nodes = [*_legacy_nodes(source), *extension_nodes]

    structure = {
        "contract": CANONICAL_PROMPT_CONTRACT_VERSION,
        "parser_namespace": parser_namespace,
        "lossless_source": source,
        "nodes": nodes,
    }
    canonical = json.dumps(structure, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return canonical, structure, warnings
