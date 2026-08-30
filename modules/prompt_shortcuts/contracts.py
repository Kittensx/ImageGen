from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any

PROMPT_SHORTCUT_CONTRACT_VERSION = "image-gen-prompt-shortcut-v2"
PROMPT_SHORTCUT_LEGACY_CONTRACT_VERSION = "image-gen-prompt-shortcut-v1"
PROMPT_STYLE_PROFILE_SCHEMA_VERSION = 2

# Phase 13D parser-emission vocabulary. These names remain supported so old
# profiles and replay snapshots can be loaded without rewriting their aliases.
CANONICAL_OPERATOR_TOKENS: dict[str, str] = {
    "AND": "AND",
    "AVERAGE_SET": "||",
    "BREAK": "BREAK",
    "GROUP_OPEN": "{",
    "GROUP_CLOSE": "}",
    "SEQUENCE": "::",
    "PARENT_CHILD": "PARENT_CHILD",
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

# Stable semantic vocabulary for the prompt-style architecture. Not every item
# is active in the production parser yet; later compatibility phases bind these
# IDs to qualified compiler/runtime algorithms without changing replay meaning.
CANONICAL_SEMANTIC_OPERATORS: tuple[str, ...] = (
    "TEXT",
    "ATTENTION_DEFAULT",
    "ATTENTION_WEIGHTED",
    "ATTENTION_DECREASE",
    "SCHEDULE_SWITCH",
    "SCHEDULE_ADD",
    "SCHEDULE_REMOVE",
    "ALTERNATE_STEP",
    "COMPOSABLE_AND",
    "CHUNK_BREAK",
    "DYNAMIC_CHOICE",
    "COMMENT_LINE",
    "COMMENT_BLOCK",
    "EMBEDDING_REF",
    "LITERAL_TEXT_SCOPE",
    "SEMANTIC_SCOPE",
    "COHESIVE_GROUP",
    "AVERAGE_SET",
    "TARGET_BIND",
    "SUBTREE_BIND",
    "RELATION",
    "OWNER_RELATION",
    "RELATION_CLOSE",
    "OWNER_CLOSE",
    "LITERAL_ESCAPE",
    # Compatibility IDs describe still-supported pre-cutover semantics without
    # falsely claiming that they already execute future A1111/Comfy math.
    "LEGACY_CONJUNCTION",
    "PARSER_CHUNK",
    "PARSER_EXTENSION",
)

DEFAULT_PROFILE_PRECEDENCE: tuple[str, ...] = (
    "ESCAPES",
    "PREPROCESSING",
    "PAIRED_DELIMITERS",
    "NUMERIC_FORMS",
    "LOCAL_BINDINGS",
    "GROUPS",
    "RELATIONS",
    "BRANCH_COMPOSITION",
    "TEXT",
)

KNOWN_PRECEDENCE_STAGES = frozenset(DEFAULT_PROFILE_PRECEDENCE)

# The defaults intentionally describe the current pre-cutover behavior. A v1
# replay/profile loaded under the v2 descriptor therefore keeps its established
# semantic intent until a later phase explicitly selects another mode.
DEFAULT_LEGACY_SEMANTIC_MODES: dict[str, str] = {
    "attention_algorithm": "parser_owned_legacy",
    "and_composition": "legacy_normalized_average_v1",
    "group_composition": "branch_average_v1",
    "average_composition": "branch_average_v1",
    "cohesive_group": "shared_context_focus_v1_available",
    "dynamic_choice": "disabled",
    "break_mode": "parser_owned_legacy",
    "target_binding": "target_only_v1",
    "subtree_binding": "subtree_inheritance_v1",
    "relation_mode": "classic_structured_v1",
}

DEFAULT_PROFILE_PREPROCESSING: dict[str, Any] = {
    "pipeline": "none",
    "dynamic_choice": "disabled",
    "comments": "preserve",
}

# ``average_surface`` is a PPSR-09E opt-in mode.  It is intentionally known to
# profile validation without being injected into every legacy profile.  Keeping
# it out of DEFAULT_LEGACY_SEMANTIC_MODES preserves the exact mapping hashes of
# Phase-04/earlier built-ins and therefore avoids invalidating recorded profile
# snapshots merely because the experimental operator exists.
KNOWN_SEMANTIC_MODE_KEYS = frozenset(
    (*DEFAULT_LEGACY_SEMANTIC_MODES, "average_surface", "double_quote_scope", "single_quote_scope", "schedule_algorithm", "alternate_algorithm", "clip_chunking")
)

_OPERATOR_MODE_KEYS: dict[str, str] = {
    "AND": "and_composition",
    "AVERAGE_SET": "average_composition",
    "BREAK": "break_mode",
    "GROUP_OPEN": "group_composition",
    "GROUP_CLOSE": "group_composition",
    "CHUNK": "break_mode",
    "SEQUENCE": "relation_mode",
    "DEEP_SEQUENCE": "relation_mode",
    "PARENT_CHILD": "relation_mode",
    "CLOSE": "relation_mode",
    "TOP_CLOSE": "relation_mode",
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


class PromptShortcutSnapshotHashError(ValueError):
    """Raised when a persisted prompt-style snapshot fails integrity checks."""


def verify_prompt_style_profile_snapshot_hash(
    payload: dict[str, Any],
    *,
    require_hash: bool | None = None,
) -> str:
    """Verify a persisted prompt-style snapshot before migration or execution.

    PPSR-10A treats v2 snapshots as strict replay/runtime records: if the snapshot
    declares schema v2 (or newer), it must carry the exact SHA-256 mapping hash
    produced when it was recorded. Historical v1 payloads are allowed to omit the
    hash so pre-hash manifests can still migrate, but when a v1 hash is present it
    is verified against the original v1 payload *before* migration changes fields.
    """

    snapshot = dict(payload or {})
    source_contract = str(snapshot.get("contract_version") or PROMPT_SHORTCUT_LEGACY_CONTRACT_VERSION)
    try:
        source_schema = int(
            snapshot.get("profile_schema_version")
            or (1 if source_contract == PROMPT_SHORTCUT_LEGACY_CONTRACT_VERSION else PROMPT_STYLE_PROFILE_SCHEMA_VERSION)
        )
    except (TypeError, ValueError) as exc:
        raise PromptShortcutSnapshotHashError("Prompt-style snapshot has an invalid profile_schema_version.") from exc

    if require_hash is None:
        require_hash = source_schema >= PROMPT_STYLE_PROFILE_SCHEMA_VERSION

    recorded = str(snapshot.get("mapping_hash") or "").strip().lower()
    if not recorded:
        if require_hash:
            raise PromptShortcutSnapshotHashError(
                f"Prompt-style profile snapshot schema v{source_schema} is missing mapping_hash."
            )
        return ""

    calculated = stable_mapping_hash(snapshot)
    if not hmac.compare_digest(recorded, calculated):
        raise PromptShortcutSnapshotHashError(
            "Prompt-style profile snapshot mapping_hash does not match its recorded contents."
        )
    return calculated


def ordered_profile_alias_entries(aliases: dict[str, tuple[str, ...]] | dict[str, list[str]]) -> tuple[tuple[str, str], ...]:
    """Return the canonical longest-match alias order used by translation.

    Centralizing the ordering keeps prefix-sensitive forms such as ``:::``/``::``,
    ``!!``/``!`` and ``||``/``|`` deterministic instead of relying on dictionary
    insertion order or duplicated regex ordering in callers.
    """

    entries: list[tuple[str, str]] = []
    for operator, values in dict(aliases or {}).items():
        canonical_operator = str(operator or "").strip().upper()
        for value in values or ():
            alias = str(value)
            if alias:
                entries.append((alias, canonical_operator))
    return tuple(sorted(entries, key=lambda item: (-len(item[0]), item[0], item[1])))


def semantic_algorithm_for_operator(operator: str, semantic_modes: dict[str, str] | None = None) -> str:
    key = _OPERATOR_MODE_KEYS.get(str(operator or "").strip().upper(), "")
    if not key:
        return ""
    return str(dict(semantic_modes or {}).get(key) or "")


def semantic_operator_id_for_alias(operator: str, semantic_modes: dict[str, str] | None = None) -> str:
    """Resolve a Phase-13D operator name to the v2 semantic vocabulary.

    This is diagnostic/contract metadata only in this phase. It deliberately
    distinguishes the currently executing legacy conjunction from future native
    A1111 composable guidance so the inspector/replay layer cannot mislabel it.
    """

    op = str(operator or "").strip().upper()
    modes = dict(semantic_modes or {})
    if op == "AND":
        return (
            "COMPOSABLE_AND"
            if str(modes.get("and_composition") or "").startswith("a1111_composable")
            else "LEGACY_CONJUNCTION"
        )
    if op == "AVERAGE_SET":
        return "AVERAGE_SET"
    if op == "BREAK":
        return "CHUNK_BREAK" if modes.get("break_mode") == "encoder_chunk_break_v1" else "TEXT"
    if op in {"GROUP_OPEN", "GROUP_CLOSE"}:
        group_mode = modes.get("group_composition")
        if group_mode == "shared_context_focus_v1":
            return "COHESIVE_GROUP"
        if group_mode == "branch_average_v1":
            return "AVERAGE_SET"
        return "PARSER_EXTENSION"
    if op == "CHUNK":
        return "CHUNK_BREAK" if modes.get("break_mode") == "encoder_chunk_break_v1" else "PARSER_CHUNK"
    if op == "SEQUENCE" or op == "PARENT_CHILD":
        return "RELATION"
    if op == "DEEP_SEQUENCE":
        return "OWNER_RELATION"
    if op == "CLOSE":
        return "RELATION_CLOSE"
    if op == "TOP_CLOSE":
        return "OWNER_CLOSE"
    if op == "BIND":
        return "PARSER_EXTENSION"
    if op in {"BLEND", "POOL", "MORPH", "ASSEMBLE", "COMPOUND"}:
        return "PARSER_EXTENSION"
    return "TEXT"


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
    semantic_modes: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LEGACY_SEMANTIC_MODES))
    preprocessing: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROFILE_PREPROCESSING))
    precedence: tuple[str, ...] = DEFAULT_PROFILE_PRECEDENCE
    reserved_syntax: tuple[str, ...] = ()
    profile_schema_version: int = PROMPT_STYLE_PROFILE_SCHEMA_VERSION
    migrated_from_contract: str = ""

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
        source_contract = str(payload.get("contract_version") or PROMPT_SHORTCUT_LEGACY_CONTRACT_VERSION)
        source_schema = int(payload.get("profile_schema_version") or (1 if source_contract == PROMPT_SHORTCUT_LEGACY_CONTRACT_VERSION else PROMPT_STYLE_PROFILE_SCHEMA_VERSION))
        semantic_modes = dict(DEFAULT_LEGACY_SEMANTIC_MODES)
        semantic_modes.update({
            str(key).strip(): str(value).strip()
            for key, value in dict(payload.get("semantic_modes") or {}).items()
            if str(key).strip() and str(value).strip()
        })
        preprocessing = dict(DEFAULT_PROFILE_PREPROCESSING)
        preprocessing.update(_json_safe(dict(payload.get("preprocessing") or {})))
        precedence_values = payload.get("precedence") or DEFAULT_PROFILE_PRECEDENCE
        precedence = tuple(str(item).strip().upper() for item in precedence_values if str(item).strip())
        reserved_syntax = tuple(str(item) for item in (payload.get("reserved_syntax") or []) if str(item) != "")
        migrated_from_contract = str(payload.get("migrated_from_contract") or "")
        if source_schema < PROMPT_STYLE_PROFILE_SCHEMA_VERSION and not migrated_from_contract:
            migrated_from_contract = source_contract
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
            contract_version=PROMPT_SHORTCUT_CONTRACT_VERSION,
            palette=tuple(dict(item) for item in (payload.get("palette") or []) if isinstance(item, dict)),
            semantic_modes=semantic_modes,
            preprocessing=preprocessing,
            precedence=precedence or DEFAULT_PROFILE_PRECEDENCE,
            reserved_syntax=reserved_syntax,
            profile_schema_version=PROMPT_STYLE_PROFILE_SCHEMA_VERSION,
            migrated_from_contract=migrated_from_contract,
        )


    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        *,
        builtin: bool | None = None,
        require_hash: bool | None = None,
    ) -> "PromptShortcutProfileDescriptor":
        """Load a persisted profile snapshot with PPSR-10A integrity checks."""

        verify_prompt_style_profile_snapshot_hash(dict(payload or {}), require_hash=require_hash)
        return cls.from_dict(dict(payload or {}), builtin=builtin)

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "contract_version": self.contract_version,
            "profile_schema_version": self.profile_schema_version,
            "profile_id": self.profile_id,
            "label": self.label,
            "version": self.version,
            "aliases": {key: list(values) for key, values in sorted(self.aliases.items())},
            "parser_emitters": {
                parser: dict(sorted(values.items()))
                for parser, values in sorted(self.parser_emitters.items())
            },
            "semantic_modes": dict(sorted(self.semantic_modes.items())),
            "preprocessing": _json_safe(self.preprocessing),
            "precedence": list(self.precedence),
            "reserved_syntax": list(self.reserved_syntax),
            "compatible_parsers": list(self.compatible_parsers),
            "escape_character": self.escape_character,
            "builtin": self.builtin,
            "credit": self.credit,
            "description": self.description,
            "source": self.source,
            "palette": [_json_safe(item) for item in self.palette],
        }
        if self.migrated_from_contract:
            payload["migrated_from_contract"] = self.migrated_from_contract
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

    def semantic_mode(self, key: str, default: str = "") -> str:
        return str(self.semantic_modes.get(str(key or "").strip()) or default)

    def semantic_operator_id(self, operator: str) -> str:
        return semantic_operator_id_for_alias(operator, self.semantic_modes)

    def semantic_algorithm(self, operator: str) -> str:
        return semantic_algorithm_for_operator(operator, self.semantic_modes)

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
            value["semantic_operator_id"] = self.semantic_operator_id(operator)
            value["semantic_algorithm"] = self.semantic_algorithm(operator)
            output.append(value)
        return output


@dataclass(frozen=True)
class PromptShortcutValidationIssue:
    severity: str
    code: str
    message: str
    operator: str = ""
    alias: str = ""
    collision_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "operator": self.operator,
            "alias": self.alias,
            "collision_kind": self.collision_kind,
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
