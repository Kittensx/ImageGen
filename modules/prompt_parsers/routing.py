from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from modules.prompt_parsers.canonical import canonicalize_prompt, normalize_prompt_source
from modules.prompt_parsers.contracts import PromptParserError

PROMPT_ROUTE_CONTRACT_VERSION = "image-gen-prompt-route-v1"
PROMPT_SHADOW_CONTRACT_VERSION = "image-gen-prompt-shadow-v1"
PROMPT_MERGE_CONTRACT_VERSION = "image-gen-prompt-merge-v1"

_ROUTE_STRATEGIES = {
    "prefer_legacy",
    "prefer_parser21",
    "strict_by_capability",
    "single_parser_only",
    "auto_split",
}
_FALLBACK_POLICIES = {
    "fail",
    "warn_and_literalize",
    "warn_and_use_legacy",
    "warn_and_use_parser21",
}

_FEATURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bind", re.compile(r"(?<!\\)\bBIND(?:2|3)?\b", re.IGNORECASE)),
    ("chunk", re.compile(r"(?<!\\)\bCHUNK\b", re.IGNORECASE)),
    ("blend", re.compile(r"(?<!\\)\bBLEND\b", re.IGNORECASE)),
    ("pool", re.compile(r"(?<!\\)\bPOOL\b", re.IGNORECASE)),
    ("morph", re.compile(r"(?<!\\)\bMORPH\b", re.IGNORECASE)),
    ("assemble", re.compile(r"(?<!\\)\bASSEMBLE\b", re.IGNORECASE)),
    ("compound", re.compile(r"(?<!\\)\bCOMPOUND\b", re.IGNORECASE)),
    ("attention_interpolation", re.compile(r"\([^()]*(?:->|~)[^()]*\)")),
    ("scheduled_prompts", re.compile(r"\[[^\]]*:[^\]]*:[^\]]*\]")),
    ("alternates", re.compile(r"\[[^\]|]+(?:\|[^\]]+)+\]")),
    ("attention_weights", re.compile(r"\([^()]+:[-+]?\d+(?:\.\d+)?\)")),
    ("and_composition", re.compile(r"(?<![A-Za-z0-9_])AND(?![A-Za-z0-9_])")),
    ("deep_sequence", re.compile(r":::")),
    ("sequence", re.compile(r"(?<!:)::(?!:)")),
)

_CAPABILITY_ALIASES = {
    "scheduled_prompts": ("scheduled_prompts", "scheduling"),
    "alternates": ("alternates", "alternation"),
    "and_composition": ("and_composition", "composable_prompts"),
    "attention_weights": ("attention_weights", "attention"),
    "plain_text": ("plain_text", "positive_prompt"),
    "sequence": ("sequence", "scheduling"),
    "deep_sequence": ("deep_sequence", "scheduling"),
}

_PARSER21_ONLY = {
    "bind", "chunk", "blend", "pool", "morph", "assemble", "compound",
    "attention_interpolation",
}


def normalize_route_strategy(value: Any) -> str:
    strategy = str(value or "prefer_legacy").strip().lower().replace("-", "_")
    if strategy not in _ROUTE_STRATEGIES:
        raise ValueError(f"Unknown combined-dispatch strategy {value!r}.")
    return strategy


def normalize_fallback_policy(value: Any) -> str:
    policy = str(value or "fail").strip().lower().replace("-", "_")
    if policy not in _FALLBACK_POLICIES:
        raise ValueError(f"Unknown prompt parser fallback policy {value!r}.")
    return policy


def _json_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _capability_enabled(capabilities: Mapping[str, Any], feature: str) -> bool:
    names = _CAPABILITY_ALIASES.get(feature, (feature,))
    return any(bool(capabilities.get(name)) for name in names)


@dataclass(frozen=True)
class CapabilityAnalysis:
    canonical_source: str
    features: tuple[str, ...]
    nodes: tuple[dict[str, Any], ...]
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_source": self.canonical_source,
            "features": list(self.features),
            "nodes": [_json_copy(item) for item in self.nodes],
            "source_hash": self.source_hash,
        }



_NODE_FEATURES = {
    "text": "plain_text",
    "conjunction": "and_composition",
    "scheduled_text": "scheduled_prompts",
    "alternate_text": "alternates",
    "weighted_text": "attention_weights",
    "attention_group": "attention_weights",
    "sequence": "sequence",
    "deep_sequence": "deep_sequence",
}


@dataclass(frozen=True)
class PromptMergeOperation:
    operation_id: str
    certified: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "certified": self.certified,
            "description": self.description,
        }


class PromptMergeRegistry:
    """Registry of explicit cross-parser merge operations.

    Phase 13F deliberately certifies no cross-parser tensor merge. Combined
    dispatch may choose one parser for the full prompt; any actual split is
    rejected until a later phase adds shape- and schedule-aware certification.
    """

    def __init__(self) -> None:
        self._operations = {
            "unsupported": PromptMergeOperation(
                operation_id="unsupported",
                certified=False,
                description="Cross-parser conditioning merge is not certified.",
            ),
        }

    def descriptors(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._operations.values()]

    def require_certified(self, operation_id: str) -> PromptMergeOperation:
        operation = self._operations.get(str(operation_id))
        if operation is None or not operation.certified:
            raise PromptParserError(
                "The requested cross-parser conditioning merge is not certified.",
                parser_id="combined",
                prompt_role="unknown",
                error_kind="unsupported_prompt_merge",
                diagnostics={
                    "merge_contract": PROMPT_MERGE_CONTRACT_VERSION,
                    "operation_id": str(operation_id),
                    "available_operations": self.descriptors(),
                },
            )
        return operation


class PromptCapabilityAnalyzer:
    def analyze(self, canonical_structure_or_source: Mapping[str, Any] | str) -> CapabilityAnalysis:
        if isinstance(canonical_structure_or_source, Mapping):
            source = str(
                canonical_structure_or_source.get("lossless_source")
                or canonical_structure_or_source.get("canonical_source")
                or canonical_structure_or_source.get("parser_input")
                or ""
            )
        else:
            source = str(canonical_structure_or_source or "")
        source = normalize_prompt_source(source)
        features: list[str] = []
        nodes: list[dict[str, Any]] = []
        canonical_nodes = []
        if isinstance(canonical_structure_or_source, Mapping):
            canonical_nodes = list(canonical_structure_or_source.get("nodes") or [])
        for index, raw_node in enumerate(canonical_nodes):
            node = dict(raw_node or {})
            node_type = str(node.get("type") or "").strip().lower()
            feature = _NODE_FEATURES.get(node_type)
            if node_type == "extension":
                feature = str(node.get("operator") or "").strip().lower()
            if not feature:
                continue
            if feature not in features:
                features.append(feature)
            nodes.append({
                "type": "capability_node",
                "feature": feature,
                "canonical_node_index": index,
                "source": node.get("source") or node.get("value") or "",
                "start": node.get("start"),
                "end": node.get("end"),
            })
        # Canonical node inspection is authoritative. The bounded pattern pass
        # fills gaps for lossless text constructs not yet represented as nodes.
        for feature, pattern in _FEATURE_PATTERNS:
            matches = list(pattern.finditer(source))
            if not matches:
                continue
            if feature not in features:
                features.append(feature)
            if not any(item.get("feature") == feature for item in nodes):
                nodes.extend({
                    "type": "capability_node",
                    "feature": feature,
                    "source": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "derived_from": "canonical_lossless_source",
                } for match in matches)
        if not features:
            features.append("plain_text")
        return CapabilityAnalysis(
            canonical_source=source,
            features=tuple(features),
            nodes=tuple(nodes),
            source_hash=_stable_hash({"source": source, "features": features}),
        )


@dataclass
class PromptRoutePlan:
    strategy: str
    selected_parser: str
    analysis: CapabilityAnalysis
    candidates: dict[str, dict[str, Any]]
    segments: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    ambiguities: list[dict[str, Any]] = field(default_factory=list)
    merge_operations: list[dict[str, Any]] = field(default_factory=list)
    fallback_policy: str = "fail"
    fail_on_ambiguous_route: bool = False
    exact_replay_supported: bool = True
    planner_duration_ms: float = 0.0
    contract: str = PROMPT_ROUTE_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": self.contract,
            "strategy": self.strategy,
            "selected_parser": self.selected_parser,
            "analysis": self.analysis.to_dict(),
            "candidates": _json_copy(self.candidates),
            "segments": _json_copy(self.segments),
            "warnings": list(self.warnings),
            "ambiguities": _json_copy(self.ambiguities),
            "merge_operations": _json_copy(self.merge_operations),
            "fallback_policy": self.fallback_policy,
            "fail_on_ambiguous_route": bool(self.fail_on_ambiguous_route),
            "exact_replay_supported": bool(self.exact_replay_supported),
            "planner_duration_ms": float(self.planner_duration_ms),
        }
        payload["route_hash"] = _stable_hash({
            "contract": self.contract,
            "strategy": self.strategy,
            "selected_parser": self.selected_parser,
            "analysis": self.analysis.to_dict(),
            "segments": self.segments,
            "merge_operations": self.merge_operations,
            "fallback_policy": self.fallback_policy,
        })
        return payload


class PromptRoutePlanner:
    def __init__(self, descriptors: Mapping[str, Mapping[str, Any]]) -> None:
        self.descriptors = {str(key): dict(value) for key, value in descriptors.items()}
        self.analyzer = PromptCapabilityAnalyzer()

    def _candidate(self, parser_id: str, analysis: CapabilityAnalysis) -> dict[str, Any]:
        descriptor = dict(self.descriptors.get(parser_id) or {})
        capabilities = dict(descriptor.get("capabilities") or {})
        unsupported = [feature for feature in analysis.features if not _capability_enabled(capabilities, feature)]
        return {
            "parser_id": parser_id,
            "label": descriptor.get("label") or parser_id,
            "supported": not unsupported,
            "unsupported_features": unsupported,
            "capabilities": capabilities,
            "experimental": bool(descriptor.get("experimental")),
        }

    def plan(
        self,
        canonical_structure_or_source: Mapping[str, Any] | str,
        *,
        strategy: Any = "prefer_legacy",
        fallback_policy: Any = "fail",
        fail_on_ambiguous_route: bool = False,
        preferred_parser: Any = "legacy",
    ) -> PromptRoutePlan:
        started = time.perf_counter()
        strategy_name = normalize_route_strategy(strategy)
        fallback_name = normalize_fallback_policy(fallback_policy)
        preferred = str(preferred_parser or "legacy").strip().lower()
        if preferred not in {"legacy", "parser21"}:
            preferred = "legacy"
        analysis = self.analyzer.analyze(canonical_structure_or_source)
        candidates = {
            parser_id: self._candidate(parser_id, analysis)
            for parser_id in ("legacy", "parser21")
        }
        supported = [parser_id for parser_id, item in candidates.items() if item["supported"]]
        warnings: list[str] = []
        ambiguities: list[dict[str, Any]] = []
        selected = ""

        if not supported:
            warnings.append("No installed prompt parser advertises all canonical capabilities required by this prompt.")
        elif strategy_name == "prefer_legacy":
            selected = "legacy" if "legacy" in supported else supported[0]
        elif strategy_name == "prefer_parser21":
            selected = "parser21" if "parser21" in supported else supported[0]
        elif strategy_name == "strict_by_capability":
            if len(supported) == 1:
                selected = supported[0]
            else:
                ambiguity = {
                    "kind": "multiple_full_capability_routes",
                    "candidate_parsers": supported,
                    "tie_break": preferred,
                    "message": "Both installed parsers advertise support for the complete canonical prompt.",
                }
                ambiguities.append(ambiguity)
                if not fail_on_ambiguous_route:
                    selected = preferred if preferred in supported else supported[0]
                    warnings.append(
                        f"Strict routing found multiple valid parsers; deterministic tie-break selected {selected}."
                    )
        elif strategy_name == "single_parser_only":
            selected = preferred if preferred in supported else supported[0]
        elif strategy_name == "auto_split":
            # Cross-parser tensor merging remains deliberately uncertified. If one
            # parser can own the complete prompt, choose it and avoid a split.
            if any(feature in _PARSER21_ONLY for feature in analysis.features) and "parser21" in supported:
                selected = "parser21"
            else:
                selected = preferred if preferred in supported else supported[0]
            warnings.append("Auto-split selected a single full-capability parser; no cross-parser merge was required.")

        segments: list[dict[str, Any]] = []
        if selected:
            parser_only = sorted(feature for feature in analysis.features if feature in _PARSER21_ONLY)
            reason = (
                "contains Parser 21-specific canonical capabilities: " + ", ".join(parser_only)
                if selected == "parser21" and parser_only
                else f"selected by {strategy_name} strategy"
            )
            segments.append({
                "segment_id": "root",
                "node_range": [0, len(analysis.nodes)],
                "source_range": [0, len(analysis.canonical_source)],
                "parser": selected,
                "reason": reason,
                "features": list(analysis.features),
            })

        plan = PromptRoutePlan(
            strategy=strategy_name,
            selected_parser=selected,
            analysis=analysis,
            candidates=candidates,
            segments=segments,
            warnings=warnings,
            ambiguities=ambiguities,
            merge_operations=[],
            fallback_policy=fallback_name,
            fail_on_ambiguous_route=bool(fail_on_ambiguous_route),
            exact_replay_supported=bool(selected and not (ambiguities and fail_on_ambiguous_route)),
            planner_duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return plan


def assert_recorded_route_matches(recorded: Mapping[str, Any] | None, current: Mapping[str, Any]) -> None:
    if not recorded:
        return
    recorded_contract = str(recorded.get("contract") or "")
    if recorded_contract and recorded_contract != PROMPT_ROUTE_CONTRACT_VERSION:
        raise PromptParserError(
            f"Recorded prompt route contract {recorded_contract!r} is not supported.",
            parser_id="combined",
            prompt_role="unknown",
            error_kind="route_contract_mismatch",
            diagnostics={"recorded_route_plan": dict(recorded), "current_route_plan": dict(current)},
        )
    fields = ("strategy", "selected_parser", "fallback_policy")
    differences = {
        key: {"recorded": recorded.get(key), "current": current.get(key)}
        for key in fields
        if recorded.get(key) not in (None, "") and recorded.get(key) != current.get(key)
    }
    recorded_hash = str(recorded.get("route_hash") or "")
    current_hash = str(current.get("route_hash") or "")
    if recorded_hash and current_hash and recorded_hash != current_hash:
        differences["route_hash"] = {"recorded": recorded_hash, "current": current_hash}
    if differences:
        raise PromptParserError(
            "The recorded combined-dispatch route cannot be reproduced exactly.",
            parser_id="combined",
            prompt_role="unknown",
            error_kind="recorded_route_mismatch",
            diagnostics={
                "differences": differences,
                "recorded_route_plan": dict(recorded),
                "current_route_plan": dict(current),
            },
        )


def shadow_compare_parsers(
    *,
    raw_prompt: str,
    prompt_role: str,
    steps: int,
    hires_steps: int | None,
    seed: int | None,
    legacy_adapter: Any,
    parser21_adapter: Any,
    parser21_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for parser_id, adapter, options in (
        ("legacy", legacy_adapter, {}),
        ("parser21", parser21_adapter, dict(parser21_options or {})),
    ):
        parser_started = time.perf_counter()
        canonical, structure, canonical_warnings = canonicalize_prompt(raw_prompt, parser_id=parser_id)
        try:
            validation = adapter.validate_syntax(
                raw_prompt,
                prompt_role=prompt_role,
                steps=int(steps),
                hires_steps=hires_steps,
                parser_options=options,
                seed=seed,
            )
            valid = bool(validation.get("valid", True))
            error = None
        except PromptParserError as exc:
            valid = False
            validation = {"valid": False}
            error = exc.to_dict()
        results[parser_id] = {
            "parser_id": parser_id,
            "valid": valid,
            "canonical_prompt": canonical,
            "canonical_structure": structure,
            "canonical_structure_hash": _stable_hash(structure),
            "schedule_count": int(validation.get("schedule_count") or 0),
            "branch_count": int(validation.get("branch_count") or 0),
            "flat_prompt_count": int(validation.get("flat_prompt_count") or 0),
            "warnings": [*canonical_warnings, *(validation.get("warnings") or [])],
            "error": error,
            "duration_ms": round((time.perf_counter() - parser_started) * 1000.0, 3),
        }

    legacy = results["legacy"]
    parser21 = results["parser21"]
    analysis = PromptCapabilityAnalyzer().analyze(
        canonicalize_prompt(raw_prompt, parser_id="combined")[1]
    )
    parser_specific_features = sorted(set(analysis.features) & _PARSER21_ONLY)
    if parser_specific_features:
        if parser21["valid"]:
            classification = "Parser-specific"
        else:
            classification = "Bug candidate"
    elif legacy["valid"] and parser21["valid"]:
        equivalent_metrics = all(
            legacy[key] == parser21[key]
            for key in ("schedule_count", "branch_count", "flat_prompt_count")
        )
        classification = "Equivalent" if equivalent_metrics else "Compatible but different"
    elif legacy["valid"] != parser21["valid"]:
        classification = "Parser-specific"
    elif not legacy["valid"] and not parser21["valid"]:
        classification = "Unsupported"
    else:  # pragma: no cover - defensive
        classification = "Bug candidate"

    differences = {
        key: {"legacy": legacy[key], "parser21": parser21[key]}
        for key in ("schedule_count", "branch_count", "flat_prompt_count", "canonical_structure_hash")
        if legacy[key] != parser21[key]
    }
    return {
        "contract": PROMPT_SHADOW_CONTRACT_VERSION,
        "classification": classification,
        "prompt_role": prompt_role,
        "source_hash": _stable_hash(raw_prompt),
        "results": results,
        "differences": differences,
        "capability_analysis": analysis.to_dict(),
        "parser_specific_features": parser_specific_features,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
