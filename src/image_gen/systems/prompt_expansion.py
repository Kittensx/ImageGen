from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPERHYBRID_EXPANSION_CONTRACT_VERSION = "image-gen-superhybrid-expansion-v1"
SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION = "image-gen-superhybrid-expansion-batch-v1"
SUPERHYBRID_EXPANSION_SCOPES = {"per_batch", "per_image"}

_RANDOM_RE = re.compile(r"(?<!\\)<random:([^>]+)>", re.IGNORECASE)
_VALUE_PATTERN = r"(?:->|[^<>]|<[^>]*>)*"
_SETVAR_RE = re.compile(rf"(?<!\\)<setvar\[([^\]]+)\]:({_VALUE_PATTERN})>", re.IGNORECASE)
_SETMACRO_RE = re.compile(rf"(?<!\\)<setmacro\[([^\]]+)\]:({_VALUE_PATTERN})>", re.IGNORECASE)
_VAR_RE = re.compile(r"(?<!\\)<var(?::([^>]+)|\[([^\]]+)\])>", re.IGNORECASE)
_MACRO_RE = re.compile(r"(?<!\\)<macro(?::([^>]+)|\[([^\]]+)\])>", re.IGNORECASE)
_WILDCARD_RE = re.compile(r"(?<!\\)__([^_\n][^\n]*?)__")
_TONEG_MARKER_RE = re.compile(r"(?<!\\)\bTONEG\s*\{", re.IGNORECASE)
_UNRESOLVED_RE = re.compile(
    r"(?<!\\)<(?:random:|setvar\[|setmacro\[|var(?::|\[)|macro(?::|\[))",
    re.IGNORECASE,
)
_VENDOR_EXPANSION_MARKER_RE = re.compile(
    r"(?:<(?:random:|setvar\[|setmacro\[|var(?::|\[)|macro(?::|\[))|__[A-Za-z0-9_\-/]+__|\bTONEG\s*\{)",
    re.IGNORECASE,
)


class PromptExpansionError(ValueError):
    pass


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _collapse_spaces(value: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", str(value or "")).strip()


def _derive_seed(positive: str, negative: str, seed: int | None) -> tuple[int, str]:
    if seed is not None:
        try:
            numeric = int(seed)
        except (TypeError, ValueError):
            numeric = -1
        if numeric >= 0:
            return numeric & 0x7FFFFFFF, "generation_seed"
    digest = hashlib.sha256(
        (str(positive or "") + "\0" + str(negative or "")).encode("utf-8", errors="replace")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF, "prompt_hash_fallback"


def _parse_definition_header(raw: str) -> tuple[str, bool]:
    parts = [part.strip() for part in str(raw or "").split(",", 1)]
    name = parts[0]
    if not name or len(name) > 128:
        raise PromptExpansionError("Variable and macro names must contain 1 to 128 characters.")
    emit = len(parts) < 2 or parts[1].lower() not in {"false", "0", "no"}
    return name, emit


def _extract_toneg(text: str) -> tuple[str, list[str]]:
    source = str(text or "")
    additions: list[str] = []
    output: list[str] = []
    index = 0
    while index < len(source):
        if source[index] == "\\" and index + 1 < len(source):
            output.append(source[index : index + 2])
            index += 2
            continue
        marker = _TONEG_MARKER_RE.match(source, index)
        if marker is None:
            output.append(source[index])
            index += 1
            continue
        open_index = marker.end() - 1
        depth = 1
        cursor = open_index + 1
        while cursor < len(source) and depth:
            if source[cursor] == "\\" and cursor + 1 < len(source):
                cursor += 2
                continue
            if source[cursor] == "{":
                depth += 1
            elif source[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise PromptExpansionError("Unclosed SuperHybrid TONEG block.")
        body = source[open_index + 1 : cursor - 1].strip()
        if body:
            additions.append(body)
        index = cursor
    return _collapse_spaces("".join(output)), additions


@dataclass
class _ExpansionLimits:
    maximum_depth: int = 16
    maximum_expanded_length: int = 65536
    maximum_selections: int = 256
    maximum_wildcard_bytes: int = 1024 * 1024
    maximum_wildcard_lines: int = 10000

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_depth": self.maximum_depth,
            "maximum_expanded_length": self.maximum_expanded_length,
            "maximum_selections": self.maximum_selections,
            "maximum_wildcard_bytes": self.maximum_wildcard_bytes,
            "maximum_wildcard_lines": self.maximum_wildcard_lines,
        }


@dataclass
class _ExpansionContext:
    seed: int
    wildcard_root: Path
    limits: _ExpansionLimits
    rng: random.Random = field(init=False)
    variables: dict[str, str] = field(default_factory=dict)
    macros: dict[str, str] = field(default_factory=dict)
    wildcard_selections: list[dict[str, Any]] = field(default_factory=list)
    random_selections: list[dict[str, Any]] = field(default_factory=list)
    variable_definitions: list[dict[str, Any]] = field(default_factory=list)
    macro_definitions: list[dict[str, Any]] = field(default_factory=list)
    macro_expansions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _selection_guard(self) -> None:
        count = len(self.wildcard_selections) + len(self.random_selections) + len(self.macro_expansions)
        if count >= self.limits.maximum_selections:
            raise PromptExpansionError(
                f"Prompt expansion exceeded the {self.limits.maximum_selections}-selection limit."
            )

    def _check_length(self, text: str) -> None:
        if len(text) > self.limits.maximum_expanded_length:
            raise PromptExpansionError(
                f"Expanded prompt exceeds {self.limits.maximum_expanded_length} characters."
            )

    def _resolve_wildcard_path(self, raw_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_\-/]+", raw_name):
            raise PromptExpansionError(f"Unsafe wildcard name: {raw_name!r}.")
        if raw_name.startswith(("/", "\\")) or "\\" in raw_name:
            raise PromptExpansionError(f"Unsafe wildcard name: {raw_name!r}.")
        parts = raw_name.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise PromptExpansionError(f"Unsafe wildcard name: {raw_name!r}.")
        candidate = (self.wildcard_root / (raw_name + ".txt")).resolve()
        root = self.wildcard_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PromptExpansionError(f"Wildcard path escapes the configured root: {raw_name!r}.") from exc
        return candidate

    def _wildcard(self, match: re.Match[str], *, prompt_role: str) -> str:
        self._selection_guard()
        raw_name = match.group(1)
        path = self._resolve_wildcard_path(raw_name)
        if not path.is_file():
            raise PromptExpansionError(
                f"Wildcard file not found beneath {self.wildcard_root}: {raw_name}.txt"
            )
        size = path.stat().st_size
        if size > self.limits.maximum_wildcard_bytes:
            raise PromptExpansionError(
                f"Wildcard file {raw_name}.txt exceeds the {self.limits.maximum_wildcard_bytes}-byte limit."
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise PromptExpansionError(f"Wildcard file {raw_name}.txt could not be read as UTF-8: {exc}") from exc
        lines: list[tuple[int, str]] = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append((line_number, stripped))
            if len(lines) > self.limits.maximum_wildcard_lines:
                raise PromptExpansionError(
                    f"Wildcard file {raw_name}.txt exceeds the {self.limits.maximum_wildcard_lines}-line limit."
                )
        if not lines:
            raise PromptExpansionError(f"Wildcard file {raw_name}.txt contains no selectable values.")
        selected_index = self.rng.randrange(len(lines))
        line_number, selected = lines[selected_index]
        self.wildcard_selections.append(
            {
                "occurrence": len(self.wildcard_selections),
                "prompt_role": prompt_role,
                "name": raw_name,
                "file": path.relative_to(self.wildcard_root).as_posix(),
                "file_sha256": _text_hash(content),
                "line_number": line_number,
                "choice_index": selected_index,
                "choice_count": len(lines),
                "selected": selected,
            }
        )
        return selected

    def _random(self, match: re.Match[str], *, prompt_role: str) -> str:
        self._selection_guard()
        raw = match.group(1)
        separator = "|" if "|" in raw and "," not in raw else ","
        options = [item.strip() for item in raw.split(separator) if item.strip()]
        if not options:
            raise PromptExpansionError("SuperHybrid <random:...> requires at least one non-empty option.")
        selected_index = self.rng.randrange(len(options))
        selected = options[selected_index]
        self.random_selections.append(
            {
                "occurrence": len(self.random_selections),
                "prompt_role": prompt_role,
                "expression": match.group(0),
                "options": options,
                "choice_index": selected_index,
                "selected": selected,
            }
        )
        return selected

    def _substitute_variables(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = (match.group(1) or match.group(2) or "").strip()
            return self.variables.get(name, match.group(0))

        return _VAR_RE.sub(replace, text)

    def expand(self, text: str, *, prompt_role: str) -> str:
        expanded = str(text or "")
        self._check_length(expanded)
        for depth in range(self.limits.maximum_depth):
            previous = expanded
            expanded = _WILDCARD_RE.sub(
                lambda match: self._wildcard(match, prompt_role=prompt_role), expanded
            )
            expanded = _RANDOM_RE.sub(
                lambda match: self._random(match, prompt_role=prompt_role), expanded
            )

            def setvar(match: re.Match[str]) -> str:
                name, emit = _parse_definition_header(match.group(1))
                value = self._substitute_variables(match.group(2).strip())
                self.variables[name] = value
                self.variable_definitions.append(
                    {
                        "occurrence": len(self.variable_definitions),
                        "prompt_role": prompt_role,
                        "name": name,
                        "value": value,
                        "emit": emit,
                    }
                )
                return value if emit else ""

            expanded = _SETVAR_RE.sub(setvar, expanded)

            def setmacro(match: re.Match[str]) -> str:
                name, emit = _parse_definition_header(match.group(1))
                template = match.group(2).strip()
                self.macros[name] = template
                self.macro_definitions.append(
                    {
                        "occurrence": len(self.macro_definitions),
                        "prompt_role": prompt_role,
                        "name": name,
                        "template": template,
                        "emit": emit,
                    }
                )
                if not emit:
                    return ""
                value = _RANDOM_RE.sub(
                    lambda nested: self._random(nested, prompt_role=prompt_role), template
                )
                return self._substitute_variables(value)

            expanded = _SETMACRO_RE.sub(setmacro, expanded)
            expanded = self._substitute_variables(expanded)

            def macro(match: re.Match[str]) -> str:
                self._selection_guard()
                name = (match.group(1) or match.group(2) or "").strip()
                template = self.macros.get(name)
                if template is None:
                    return match.group(0)
                value = _RANDOM_RE.sub(
                    lambda nested: self._random(nested, prompt_role=prompt_role), template
                )
                value = self._substitute_variables(value)
                self.macro_expansions.append(
                    {
                        "occurrence": len(self.macro_expansions),
                        "prompt_role": prompt_role,
                        "name": name,
                        "template": template,
                        "expanded": value,
                    }
                )
                return value

            expanded = _MACRO_RE.sub(macro, expanded)
            self._check_length(expanded)
            if expanded == previous:
                break
        else:
            raise PromptExpansionError(
                f"Prompt expansion exceeded the {self.limits.maximum_depth}-pass recursion limit."
            )

        unresolved = _UNRESOLVED_RE.search(expanded)
        if unresolved:
            raise PromptExpansionError(
                f"Unresolved SuperHybrid expansion directive remains near {unresolved.group(0)!r}."
            )
        return _collapse_spaces(expanded)


def _resolve_wildcard_root(
    wildcard_directory: str | None,
    *,
    project_root: str | Path | None,
) -> Path:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
    configured = str(wildcard_directory or "wildcards").strip() or "wildcards"
    relative = Path(configured)
    if (
        relative.is_absolute()
        or configured in {".", "./"}
        or not relative.parts
        or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.parts)
    ):
        raise PromptExpansionError(
            "wildcard_directory must be a non-hidden project-relative subdirectory without '.' or '..'."
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PromptExpansionError("wildcard_directory must remain inside the IMAGE_GEN project root.") from exc
    return resolved


def _fingerprint_source(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"fingerprint", "replay_locked", "replay_source"}
    return {key: value for key, value in dict(record).items() if key not in excluded}


def _attach_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    record["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_json_hash(_fingerprint_source(record)),
    }
    return record


def validate_recorded_prompt_expansion(
    recorded: Mapping[str, Any],
    *,
    raw_positive: str,
    raw_negative: str,
    pass_name: str,
    expected_slot_index: int | None = None,
    expected_image_seed: int | None = None,
) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != SUPERHYBRID_EXPANSION_CONTRACT_VERSION:
        raise PromptExpansionError("Recorded prompt expansion uses an unsupported contract version.")
    if str(record.get("pass") or "") != str(pass_name):
        raise PromptExpansionError("Recorded prompt expansion belongs to a different generation pass.")
    if expected_slot_index is not None and int(record.get("slot_index", expected_slot_index)) != int(expected_slot_index):
        raise PromptExpansionError("Recorded prompt expansion belongs to a different batch slot.")
    if expected_image_seed is not None and int(record.get("image_seed", expected_image_seed)) != int(expected_image_seed):
        raise PromptExpansionError("Recorded prompt expansion belongs to a different image seed.")
    if record.get("raw_positive_sha256") != _text_hash(raw_positive):
        raise PromptExpansionError("Recorded prompt expansion positive-prompt hash does not match the request.")
    if record.get("raw_negative_sha256") != _text_hash(raw_negative):
        raise PromptExpansionError("Recorded prompt expansion negative-prompt hash does not match the request.")
    fingerprint = dict(record.get("fingerprint") or {})
    if fingerprint.get("algorithm") != "sha256":
        raise PromptExpansionError("Recorded prompt expansion fingerprint is missing or unsupported.")
    expected = _stable_json_hash(_fingerprint_source(record))
    if fingerprint.get("digest") != expected:
        raise PromptExpansionError("Recorded prompt expansion fingerprint validation failed.")
    expanded_positive = str(record.get("expanded_positive") or "")
    expanded_negative = str(record.get("expanded_negative") or "")
    if len(expanded_positive) > 65536 or len(expanded_negative) > 65536:
        raise PromptExpansionError("Recorded prompt expansion exceeds the supported prompt length.")
    record["replay_locked"] = True
    record["replay_source"] = "recorded_exact"
    return record


def expand_superhybrid_prompt_pair(
    positive_prompt: str,
    negative_prompt: str,
    *,
    seed: int | None,
    pass_name: str,
    parser_version: str,
    wildcard_directory: str | None = "wildcards",
    project_root: str | Path | None = None,
    recorded: Mapping[str, Any] | None = None,
    replay_mode: str = "reconstruct",
    slot_index: int | None = None,
    image_seed: int | None = None,
) -> dict[str, Any]:
    positive = str(positive_prompt or "")
    negative = str(negative_prompt or "")
    mode = str(replay_mode or "reconstruct").strip().lower()
    if mode not in {"reconstruct", "recorded_exact"}:
        raise PromptExpansionError("prompt_expansion_replay_mode must be reconstruct or recorded_exact.")
    if mode == "recorded_exact":
        if not recorded:
            raise PromptExpansionError(
                f"Exact prompt expansion replay was requested, but no recorded {pass_name} expansion was supplied."
            )
        return validate_recorded_prompt_expansion(
            recorded,
            raw_positive=positive,
            raw_negative=negative,
            pass_name=pass_name,
            expected_slot_index=slot_index,
            expected_image_seed=image_seed,
        )

    if _TONEG_MARKER_RE.search(negative):
        raise PromptExpansionError("SuperHybrid TONEG is only valid in the positive prompt.")

    resolved_seed, seed_source = _derive_seed(positive, negative, seed)
    limits = _ExpansionLimits()
    wildcard_root = _resolve_wildcard_root(
        wildcard_directory,
        project_root=project_root,
    )
    context = _ExpansionContext(seed=resolved_seed, wildcard_root=wildcard_root, limits=limits)
    expanded_positive_with_toneg = context.expand(positive, prompt_role=f"{pass_name}_positive")
    expanded_positive, toneg_additions = _extract_toneg(expanded_positive_with_toneg)
    expanded_negative = context.expand(negative, prompt_role=f"{pass_name}_negative")
    if toneg_additions:
        joined_toneg = ", ".join(toneg_additions)
        expanded_negative = ", ".join(
            item for item in (expanded_negative.strip(" ,"), joined_toneg.strip(" ,")) if item
        )
    context._check_length(expanded_negative)

    record = {
        "contract_version": SUPERHYBRID_EXPANSION_CONTRACT_VERSION,
        "parser_id": "superhybrid",
        "parser_version": str(parser_version or ""),
        "pass": str(pass_name),
        "slot_index": None if slot_index is None else int(slot_index),
        "image_seed": resolved_seed if image_seed is None else int(image_seed),
        "seed": resolved_seed,
        "seed_source": seed_source,
        "raw_positive": positive,
        "raw_negative": negative,
        "raw_positive_sha256": _text_hash(positive),
        "raw_negative_sha256": _text_hash(negative),
        "expanded_positive": expanded_positive,
        "expanded_negative": expanded_negative,
        "toneg_additions": toneg_additions,
        "wildcard_directory": str(wildcard_directory or "wildcards"),
        "wildcard_selections": context.wildcard_selections,
        "random_selections": context.random_selections,
        "variable_definitions": context.variable_definitions,
        "variables": dict(context.variables),
        "macro_definitions": context.macro_definitions,
        "macros": dict(context.macros),
        "macro_expansions": context.macro_expansions,
        "warnings": context.warnings,
        "limits": limits.to_dict(),
        "replay_locked": False,
        "replay_source": "reconstruct",
    }
    return _attach_fingerprint(record)


def prompt_has_superhybrid_expansion_syntax(text: str) -> bool:
    # This intentionally catches escaped expansion markers too. The vendored
    # SuperHybrid parser expands them before its normal escape-protection pass,
    # so forwarding a residual marker would bypass IMAGE_GEN's replay record.
    return bool(_VENDOR_EXPANSION_MARKER_RE.search(str(text or "")))


def _normalize_expansion_scope(value: str | None) -> str:
    scope = str(value or "per_batch").strip().lower()
    aliases = {
        "batch": "per_batch",
        "shared": "per_batch",
        "per_request": "per_batch",
        "image": "per_image",
        "slot": "per_image",
        "per_seed": "per_image",
    }
    scope = aliases.get(scope, scope)
    if scope not in SUPERHYBRID_EXPANSION_SCOPES:
        raise PromptExpansionError(
            "prompt_expansion_scope must be per_batch or per_image."
        )
    return scope


def _batch_fingerprint_source(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"fingerprint", "replay_locked", "replay_source"}
    return {key: value for key, value in dict(record).items() if key not in excluded}


def _attach_batch_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    record["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_json_hash(_batch_fingerprint_source(record)),
    }
    return record


def validate_recorded_prompt_expansion_batch(
    recorded: Mapping[str, Any],
    *,
    raw_positive: str,
    raw_negative: str,
    pass_name: str,
    resolved_seeds: Sequence[int],
    scope: str | None = None,
) -> dict[str, Any]:
    record = dict(recorded or {})
    # Phase 3 single-pair records remain valid and are normalized into a shared batch view.
    if record.get("contract_version") == SUPERHYBRID_EXPANSION_CONTRACT_VERSION:
        pair = validate_recorded_prompt_expansion(
            record,
            raw_positive=raw_positive,
            raw_negative=raw_negative,
            pass_name=pass_name,
        )
        seeds = [int(value) for value in resolved_seeds] or [int(pair.get("image_seed", pair.get("seed", 0)))]
        return _attach_batch_fingerprint({
            "contract_version": SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION,
            "slot_contract_version": SUPERHYBRID_EXPANSION_CONTRACT_VERSION,
            "parser_id": str(pair.get("parser_id") or "superhybrid"),
            "parser_version": str(pair.get("parser_version") or ""),
            "pass": str(pass_name),
            "scope": "per_batch",
            "slot_count": len(seeds),
            "resolved_seeds": seeds,
            "raw_positive": str(raw_positive or ""),
            "raw_negative": str(raw_negative or ""),
            "raw_positive_sha256": _text_hash(raw_positive),
            "raw_negative_sha256": _text_hash(raw_negative),
            "expanded_positive": str(pair.get("expanded_positive") or ""),
            "expanded_negative": str(pair.get("expanded_negative") or ""),
            "expanded_positive_by_slot": [str(pair.get("expanded_positive") or "") for _ in seeds],
            "expanded_negative_by_slot": [str(pair.get("expanded_negative") or "") for _ in seeds],
            "slot_records": [pair],
            "all_slots_identical": True,
            "shared_record": True,
            "replay_locked": True,
            "replay_source": "recorded_exact_legacy_v1",
        })

    if record.get("contract_version") != SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION:
        raise PromptExpansionError("Recorded prompt expansion uses an unsupported batch contract version.")
    if str(record.get("pass") or "") != str(pass_name):
        raise PromptExpansionError("Recorded prompt expansion belongs to a different generation pass.")
    if record.get("raw_positive_sha256") != _text_hash(raw_positive):
        raise PromptExpansionError("Recorded prompt expansion positive-prompt hash does not match the request.")
    if record.get("raw_negative_sha256") != _text_hash(raw_negative):
        raise PromptExpansionError("Recorded prompt expansion negative-prompt hash does not match the request.")
    fingerprint = dict(record.get("fingerprint") or {})
    if fingerprint.get("algorithm") != "sha256":
        raise PromptExpansionError("Recorded prompt expansion batch fingerprint is missing or unsupported.")
    expected = _stable_json_hash(_batch_fingerprint_source(record))
    if fingerprint.get("digest") != expected:
        raise PromptExpansionError("Recorded prompt expansion batch fingerprint validation failed.")

    requested_scope = _normalize_expansion_scope(scope or record.get("scope"))
    if requested_scope != str(record.get("scope") or "per_batch"):
        raise PromptExpansionError("Recorded prompt expansion scope does not match the request.")
    seeds = [int(value) for value in resolved_seeds]
    recorded_seeds = [int(value) for value in list(record.get("resolved_seeds") or [])]
    if seeds and recorded_seeds != seeds:
        raise PromptExpansionError("Recorded prompt expansion image seeds do not match the request.")
    slot_count = int(record.get("slot_count", len(recorded_seeds)) or 0)
    if slot_count != len(recorded_seeds):
        raise PromptExpansionError("Recorded prompt expansion slot count is inconsistent.")
    positive_by_slot = [str(value or "") for value in list(record.get("expanded_positive_by_slot") or [])]
    negative_by_slot = [str(value or "") for value in list(record.get("expanded_negative_by_slot") or [])]
    if len(positive_by_slot) != slot_count or len(negative_by_slot) != slot_count:
        raise PromptExpansionError("Recorded prompt expansion slot materializations are incomplete.")
    if any(len(value) > 65536 for value in positive_by_slot + negative_by_slot):
        raise PromptExpansionError("Recorded prompt expansion exceeds the supported prompt length.")

    slots = [dict(item or {}) for item in list(record.get("slot_records") or [])]
    expected_records = 1 if requested_scope == "per_batch" else slot_count
    if len(slots) != expected_records:
        raise PromptExpansionError("Recorded prompt expansion slot records are incomplete.")
    validated_slots: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        seed = recorded_seeds[0] if requested_scope == "per_batch" else recorded_seeds[index]
        validated_slots.append(validate_recorded_prompt_expansion(
            slot,
            raw_positive=raw_positive,
            raw_negative=raw_negative,
            pass_name=pass_name,
            expected_slot_index=0 if requested_scope == "per_batch" else index,
            expected_image_seed=seed,
        ))
    record["slot_records"] = validated_slots
    record["replay_locked"] = True
    record["replay_source"] = "recorded_exact"
    return record


def expand_superhybrid_prompt_batch(
    positive_prompt: str,
    negative_prompt: str,
    *,
    resolved_seeds: Sequence[int],
    pass_name: str,
    parser_version: str,
    scope: str = "per_batch",
    selection_seeds: Sequence[int] | None = None,
    wildcard_directory: str | None = "wildcards",
    project_root: str | Path | None = None,
    recorded: Mapping[str, Any] | None = None,
    replay_mode: str = "reconstruct",
) -> dict[str, Any]:
    normalized_scope = _normalize_expansion_scope(scope)
    seeds = [int(value) for value in resolved_seeds]
    if not seeds:
        raise PromptExpansionError("At least one resolved image seed is required for prompt expansion.")
    rng_seeds = [int(value) for value in (selection_seeds or seeds)]
    if len(rng_seeds) != len(seeds):
        raise PromptExpansionError("Prompt expansion selection seed count must match the image seed count.")
    mode = str(replay_mode or "reconstruct").strip().lower()
    if mode not in {"reconstruct", "recorded_exact"}:
        raise PromptExpansionError("prompt_expansion_replay_mode must be reconstruct or recorded_exact.")
    if mode == "recorded_exact":
        if not recorded:
            raise PromptExpansionError(
                f"Exact prompt expansion replay was requested, but no recorded {pass_name} expansion was supplied."
            )
        return validate_recorded_prompt_expansion_batch(
            recorded,
            raw_positive=positive_prompt,
            raw_negative=negative_prompt,
            pass_name=pass_name,
            resolved_seeds=seeds,
            scope=normalized_scope,
        )

    slot_records: list[dict[str, Any]] = []
    if normalized_scope == "per_batch":
        slot_records.append(expand_superhybrid_prompt_pair(
            positive_prompt,
            negative_prompt,
            seed=rng_seeds[0],
            pass_name=pass_name,
            parser_version=parser_version,
            wildcard_directory=wildcard_directory,
            project_root=project_root,
            slot_index=0,
            image_seed=seeds[0],
        ))
        positive_by_slot = [slot_records[0]["expanded_positive"] for _ in seeds]
        negative_by_slot = [slot_records[0]["expanded_negative"] for _ in seeds]
    else:
        for index, seed in enumerate(seeds):
            slot_records.append(expand_superhybrid_prompt_pair(
                positive_prompt,
                negative_prompt,
                seed=rng_seeds[index],
                pass_name=pass_name,
                parser_version=parser_version,
                wildcard_directory=wildcard_directory,
                project_root=project_root,
                slot_index=index,
                image_seed=seed,
            ))
        positive_by_slot = [str(item.get("expanded_positive") or "") for item in slot_records]
        negative_by_slot = [str(item.get("expanded_negative") or "") for item in slot_records]

    primary = slot_records[0]
    all_identical = len(set(zip(positive_by_slot, negative_by_slot))) == 1
    record = {
        "contract_version": SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION,
        "slot_contract_version": SUPERHYBRID_EXPANSION_CONTRACT_VERSION,
        "parser_id": "superhybrid",
        "parser_version": str(parser_version or ""),
        "pass": str(pass_name),
        "scope": normalized_scope,
        "slot_count": len(seeds),
        "resolved_seeds": seeds,
        "selection_seeds": rng_seeds,
        "raw_positive": str(positive_prompt or ""),
        "raw_negative": str(negative_prompt or ""),
        "raw_positive_sha256": _text_hash(positive_prompt),
        "raw_negative_sha256": _text_hash(negative_prompt),
        "expanded_positive": positive_by_slot[0],
        "expanded_negative": negative_by_slot[0],
        "expanded_positive_by_slot": positive_by_slot,
        "expanded_negative_by_slot": negative_by_slot,
        "slot_records": slot_records,
        "all_slots_identical": all_identical,
        "shared_record": normalized_scope == "per_batch",
        # Compatibility summaries retain Phase 3 output-detail fields.
        "toneg_additions": list(primary.get("toneg_additions") or []),
        "wildcard_selections": list(primary.get("wildcard_selections") or []),
        "random_selections": list(primary.get("random_selections") or []),
        "variable_definitions": list(primary.get("variable_definitions") or []),
        "variables": dict(primary.get("variables") or {}),
        "macro_definitions": list(primary.get("macro_definitions") or []),
        "macros": dict(primary.get("macros") or {}),
        "macro_expansions": list(primary.get("macro_expansions") or []),
        "warnings": list(primary.get("warnings") or []),
        "wildcard_directory": str(wildcard_directory or "wildcards"),
        "replay_locked": False,
        "replay_source": "reconstruct",
    }
    return _attach_batch_fingerprint(record)


def select_prompt_expansion_slot(recorded: Mapping[str, Any], slot_index: int) -> dict[str, Any]:
    """Project a Phase 4 batch expansion record into a one-image replay record."""
    record = dict(recorded or {})
    if record.get("contract_version") != SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION:
        return record
    index = int(slot_index)
    count = int(record.get("slot_count", 0) or 0)
    if index < 0 or index >= count:
        raise PromptExpansionError("Prompt expansion slot index is outside the recorded batch.")
    if count <= 1:
        return record

    scope = str(record.get("scope") or "per_batch")
    slots = [dict(item or {}) for item in list(record.get("slot_records") or [])]
    expected_records = 1 if scope == "per_batch" else count
    if len(slots) != expected_records:
        raise PromptExpansionError("Prompt expansion slot records are incomplete.")

    selected = dict(slots[0] if scope == "per_batch" else slots[index])
    seed = int((record.get("resolved_seeds") or [])[index])
    selection_seed_values = list(record.get("selection_seeds") or record.get("resolved_seeds") or [])
    selection_seed_index = 0 if scope == "per_batch" else index
    selection_seed = (
        int(selection_seed_values[selection_seed_index])
        if selection_seed_values
        else seed
    )
    selected["slot_index"] = 0
    selected["image_seed"] = seed
    selected["seed"] = selection_seed
    selected["seed_source"] = "projected_recorded_selection_seed"
    selected["replay_locked"] = False
    selected["replay_source"] = "reconstruct"
    selected = _attach_fingerprint({
        key: value for key, value in selected.items() if key != "fingerprint"
    })

    positive = str((record.get("expanded_positive_by_slot") or [])[index] or "")
    negative = str((record.get("expanded_negative_by_slot") or [])[index] or "")
    projected = {
        "contract_version": SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION,
        "slot_contract_version": SUPERHYBRID_EXPANSION_CONTRACT_VERSION,
        "parser_id": str(record.get("parser_id") or "superhybrid"),
        "parser_version": str(record.get("parser_version") or ""),
        "pass": str(record.get("pass") or "base"),
        "scope": scope,
        "slot_count": 1,
        "resolved_seeds": [seed],
        "selection_seeds": [selection_seed],
        "raw_positive": str(record.get("raw_positive") or ""),
        "raw_negative": str(record.get("raw_negative") or ""),
        "raw_positive_sha256": str(record.get("raw_positive_sha256") or ""),
        "raw_negative_sha256": str(record.get("raw_negative_sha256") or ""),
        "expanded_positive": positive,
        "expanded_negative": negative,
        "expanded_positive_by_slot": [positive],
        "expanded_negative_by_slot": [negative],
        "slot_records": [selected],
        "all_slots_identical": True,
        "shared_record": scope == "per_batch",
        "toneg_additions": list(selected.get("toneg_additions") or []),
        "wildcard_selections": list(selected.get("wildcard_selections") or []),
        "random_selections": list(selected.get("random_selections") or []),
        "variable_definitions": list(selected.get("variable_definitions") or []),
        "variables": dict(selected.get("variables") or {}),
        "macro_definitions": list(selected.get("macro_definitions") or []),
        "macros": dict(selected.get("macros") or {}),
        "macro_expansions": list(selected.get("macro_expansions") or []),
        "warnings": list(selected.get("warnings") or []),
        "wildcard_directory": str(record.get("wildcard_directory") or "wildcards"),
        "replay_locked": False,
        "replay_source": "projected_image_manifest",
        "source_batch_slot_index": index,
        "source_batch_scope": scope,
        "source_batch_fingerprint": dict(record.get("fingerprint") or {}),
    }
    return _attach_batch_fingerprint(projected)
