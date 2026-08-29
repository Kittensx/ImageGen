from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION = (
    "image-gen-superhybrid-semantic-batch-v1"
)


class PromptSemanticReplayError(ValueError):
    """Raised when a recorded SuperHybrid semantic contract cannot be trusted."""


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_source(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(record or {}).items()
        if key not in {"fingerprint", "replay_locked", "replay_source"}
    }


def _validated_semantic_fingerprints(
    values: Sequence[Mapping[str, Any] | None],
    *,
    role: str,
    slot_count: int,
) -> list[dict[str, Any]]:
    items = [dict(value or {}) for value in values]
    if len(items) != int(slot_count):
        raise PromptSemanticReplayError(
            f"SuperHybrid {role} semantic fingerprint count does not match the prompt slot count."
        )
    for index, item in enumerate(items):
        if item.get("algorithm") != "sha256" or not str(item.get("digest") or ""):
            raise PromptSemanticReplayError(
                f"SuperHybrid {role} semantic fingerprint for slot {index} is missing or unsupported."
            )
    return items


def build_superhybrid_semantic_record(
    *,
    parser_version: str,
    pass_name: str,
    scope: str,
    resolved_seeds: Sequence[int],
    selection_seeds: Sequence[int],
    positive_fingerprints: Sequence[Mapping[str, Any] | None],
    negative_fingerprints: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    seeds = [int(value) for value in resolved_seeds]
    parser_seeds = [int(value) for value in selection_seeds]
    if not seeds:
        raise PromptSemanticReplayError(
            "At least one resolved seed is required for a SuperHybrid semantic record."
        )
    if len(parser_seeds) != len(seeds):
        raise PromptSemanticReplayError(
            "SuperHybrid semantic selection seed count does not match the image seed count."
        )
    positive = _validated_semantic_fingerprints(
        positive_fingerprints,
        role="positive",
        slot_count=len(seeds),
    )
    negative = _validated_semantic_fingerprints(
        negative_fingerprints,
        role="negative",
        slot_count=len(seeds),
    )
    record = {
        "contract_version": SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION,
        "parser_id": "superhybrid",
        "parser_version": str(parser_version or ""),
        "pass": str(pass_name or "base"),
        "scope": str(scope or "per_batch"),
        "slot_count": len(seeds),
        "resolved_seeds": seeds,
        "selection_seeds": parser_seeds,
        "positive_fingerprints": positive,
        "negative_fingerprints": negative,
        "replay_locked": False,
        "replay_source": "reconstruct",
    }
    record["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(record)),
    }
    return record


def validate_recorded_superhybrid_semantic_record(
    recorded: Mapping[str, Any],
    *,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(recorded or {})
    current_record = dict(current or {})
    if record.get("contract_version") != SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION:
        raise PromptSemanticReplayError(
            "Recorded SuperHybrid semantic data uses an unsupported contract version."
        )
    fingerprint = dict(record.get("fingerprint") or {})
    if fingerprint.get("algorithm") != "sha256":
        raise PromptSemanticReplayError(
            "Recorded SuperHybrid semantic fingerprint is missing or unsupported."
        )
    if fingerprint.get("digest") != _stable_hash(_fingerprint_source(record)):
        raise PromptSemanticReplayError(
            "Recorded SuperHybrid semantic contract fingerprint validation failed."
        )

    for key in (
        "parser_id",
        "parser_version",
        "pass",
        "scope",
        "slot_count",
        "resolved_seeds",
        "selection_seeds",
    ):
        if record.get(key) != current_record.get(key):
            raise PromptSemanticReplayError(
                f"Recorded SuperHybrid semantic {key.replace('_', ' ')} does not match the current request."
            )

    for role_key, role_label in (
        ("positive_fingerprints", "positive"),
        ("negative_fingerprints", "negative"),
    ):
        recorded_items = [dict(value or {}) for value in list(record.get(role_key) or [])]
        current_items = [dict(value or {}) for value in list(current_record.get(role_key) or [])]
        if len(recorded_items) != len(current_items):
            raise PromptSemanticReplayError(
                f"Recorded SuperHybrid {role_label} semantic slot count does not match."
            )
        for index, (recorded_item, current_item) in enumerate(
            zip(recorded_items, current_items)
        ):
            if recorded_item.get("algorithm") != "sha256":
                raise PromptSemanticReplayError(
                    f"Recorded SuperHybrid {role_label} semantic fingerprint for slot {index} is unsupported."
                )
            if recorded_item.get("digest") != current_item.get("digest"):
                raise PromptSemanticReplayError(
                    f"Recorded SuperHybrid {role_label} semantics changed for prompt slot {index}."
                )

    record["replay_locked"] = True
    record["replay_source"] = "recorded_exact"
    return record


def select_superhybrid_semantic_slot(
    recorded: Mapping[str, Any],
    slot_index: int,
) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION:
        return record
    count = int(record.get("slot_count", 0) or 0)
    index = int(slot_index)
    if index < 0 or index >= count:
        raise PromptSemanticReplayError(
            "SuperHybrid semantic slot index is outside the recorded batch."
        )
    if count <= 1:
        return record
    scope = str(record.get("scope") or "per_batch")
    selection_seed_values = list(record.get("selection_seeds") or [])
    selection_seed_index = 0 if scope == "per_batch" else index
    if not selection_seed_values:
        raise PromptSemanticReplayError(
            "SuperHybrid semantic selection seeds are incomplete."
        )

    projected = {
        "contract_version": SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION,
        "parser_id": str(record.get("parser_id") or "superhybrid"),
        "parser_version": str(record.get("parser_version") or ""),
        "pass": str(record.get("pass") or "base"),
        "scope": scope,
        "slot_count": 1,
        "resolved_seeds": [int((record.get("resolved_seeds") or [])[index])],
        "selection_seeds": [int(selection_seed_values[selection_seed_index])],
        "positive_fingerprints": [
            dict((record.get("positive_fingerprints") or [])[index] or {})
        ],
        "negative_fingerprints": [
            dict((record.get("negative_fingerprints") or [])[index] or {})
        ],
        "replay_locked": False,
        "replay_source": "projected_image_manifest",
        "source_batch_slot_index": index,
        "source_batch_fingerprint": dict(record.get("fingerprint") or {}),
    }
    projected["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(projected)),
    }
    return projected


# PPSR-08 parser-neutral semantic replay -------------------------------------
PPSR_SEMANTIC_RECORD_CONTRACT_VERSION = "image-gen-ppsr-semantic-record-v1"
PPSR_SEMANTIC_PASS_CONTRACT_VERSION = "image-gen-prompt-semantic-pass-v2"
PPSR_SEMANTIC_COMPILER_VERSION = "ppsr-09"


def _normalize_semantic_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _semantic_payload(value: Any) -> Any:
    """Strip source-location/reformatting data while preserving semantic intent."""
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            token = str(key)
            if token in {
                "raw_source",
                "normalized_source",
                "lossless_source",
                "classic_normalized_source",
                "source_text",
                "source",
                "source_start",
                "source_end",
                "start",
                "end",
                "token",
            }:
                continue
            if token in {"value", "text", "owner_text", "relation_parent", "relation_child"} and isinstance(item, str):
                output[token] = _normalize_semantic_text(item)
            else:
                output[token] = _semantic_payload(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_semantic_payload(item) for item in value]
    if isinstance(value, str):
        return _normalize_semantic_text(value)
    return value


def semantic_digest(
    semantic_ir: Any,
    conditioning_plan: Any | None = None,
    *,
    degradation: Any | None = None,
) -> dict[str, str]:
    """Return a stable semantic digest independent of harmless source whitespace."""
    ir_payload = semantic_ir.to_dict() if hasattr(semantic_ir, "to_dict") else dict(semantic_ir or {})
    plan_payload = (
        conditioning_plan.to_dict()
        if hasattr(conditioning_plan, "to_dict")
        else dict(conditioning_plan or {})
    )
    payload = {
        "semantic_ir": _semantic_payload(ir_payload),
        "conditioning_plan": _semantic_payload(plan_payload),
        "degradation": _semantic_payload(degradation or []),
    }
    return {"algorithm": "sha256", "digest": _stable_hash(payload)}


def semantic_structure_digest(semantic_ir: Any) -> dict[str, str]:
    ir_payload = semantic_ir.to_dict() if hasattr(semantic_ir, "to_dict") else dict(semantic_ir or {})
    return {
        "algorithm": "sha256",
        "digest": _stable_hash({"semantic_ir": _semantic_payload(ir_payload)}),
    }


def build_ppsr_semantic_record(
    *,
    parser_id: str,
    parser_version: str,
    parser_contract_version: str,
    prompt_role: str,
    raw_prompt: str,
    canonical_structure: Mapping[str, Any] | None,
    semantic_ir: Any,
    conditioning_plan: Any,
    parser_seed: int | None = None,
    replay_source: str = "reconstruct",
    migration_path: str = "none",
    shared_classic_semantics: bool = True,
    model_family_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ir_payload = semantic_ir.to_dict() if hasattr(semantic_ir, "to_dict") else semantic_ir
    plan_payload = conditioning_plan.to_dict() if hasattr(conditioning_plan, "to_dict") else conditioning_plan
    degradation = list((plan_payload or {}).get("fallbacks") or []) if isinstance(plan_payload, Mapping) else []
    structure = dict(canonical_structure or {})
    record = {
        "contract_version": PPSR_SEMANTIC_RECORD_CONTRACT_VERSION,
        "compiler_version": PPSR_SEMANTIC_COMPILER_VERSION,
        "parser_id": str(parser_id or "legacy"),
        "parser_version": str(parser_version or ""),
        "parser_contract_version": str(parser_contract_version or ""),
        "canonical_contract_version": str(structure.get("contract") or ""),
        "prompt_ir_contract_version": str((ir_payload or {}).get("contract") or "") if isinstance(ir_payload, Mapping) else "",
        "conditioning_plan_contract_version": str((plan_payload or {}).get("contract") or "") if isinstance(plan_payload, Mapping) else "",
        "prompt_role": str(prompt_role or "positive"),
        "raw_prompt": str(raw_prompt or ""),
        "canonical_structure": structure,
        "semantic_ir": ir_payload,
        "conditioning_plan": plan_payload,
        "semantic_digest": semantic_digest(ir_payload or {}, plan_payload or {}, degradation=degradation),
        "structure_digest": semantic_structure_digest(ir_payload or {}),
        "fallbacks": degradation,
        "model_family_semantics": dict(model_family_semantics or {}),
        "parser_seed": None if parser_seed is None else int(parser_seed),
        "shared_classic_semantics": bool(shared_classic_semantics),
        "replay_locked": replay_source == "recorded_exact",
        "replay_source": str(replay_source or "reconstruct"),
        "migration_path": str(migration_path or "none"),
    }
    record["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(record)),
    }
    return record


def validate_ppsr_semantic_record(
    recorded: Mapping[str, Any],
    *,
    parser_id: str,
    parser_contract_version: str,
    prompt_role: str,
) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != PPSR_SEMANTIC_RECORD_CONTRACT_VERSION:
        raise PromptSemanticReplayError("Recorded PPSR semantic data uses an unsupported contract version.")
    fingerprint = dict(record.get("fingerprint") or {})
    if fingerprint.get("algorithm") != "sha256" or fingerprint.get("digest") != _stable_hash(_fingerprint_source(record)):
        raise PromptSemanticReplayError("Recorded PPSR semantic record fingerprint validation failed.")
    if str(record.get("parser_id") or "") != str(parser_id or ""):
        raise PromptSemanticReplayError("Recorded PPSR semantic parser ID does not match the active parser.")
    if str(record.get("parser_contract_version") or "") != str(parser_contract_version or ""):
        raise PromptSemanticReplayError("Recorded PPSR parser contract version does not match the active parser contract.")
    if str(record.get("prompt_role") or "") != str(prompt_role or ""):
        raise PromptSemanticReplayError("Recorded PPSR semantic prompt role does not match the active prompt role.")
    semantic_ir = record.get("semantic_ir")
    if not isinstance(semantic_ir, Mapping):
        raise PromptSemanticReplayError("Recorded PPSR semantic record is missing semantic_ir.")
    return record


def replay_prompt_ir(
    recorded: Mapping[str, Any],
    *,
    parser_id: str,
    parser_contract_version: str,
    prompt_role: str,
):
    from modules.prompt_parsers.ir import prompt_ir_from_dict

    record = validate_ppsr_semantic_record(
        recorded,
        parser_id=parser_id,
        parser_contract_version=parser_contract_version,
        prompt_role=prompt_role,
    )
    return prompt_ir_from_dict(record["semantic_ir"]), record


def validate_replayed_ppsr_result(
    recorded: Mapping[str, Any],
    *,
    semantic_ir: Any,
    conditioning_plan: Any,
) -> dict[str, Any]:
    record = dict(recorded or {})
    expected_structure = str((record.get("structure_digest") or {}).get("digest") or "")
    current_structure = semantic_structure_digest(semantic_ir)["digest"]
    if expected_structure and expected_structure != current_structure:
        raise PromptSemanticReplayError("Recorded PPSR semantic structure changed during exact replay.")
    expected_semantics = str((record.get("semantic_digest") or {}).get("digest") or "")
    plan_payload = conditioning_plan.to_dict() if hasattr(conditioning_plan, "to_dict") else dict(conditioning_plan or {})
    current_semantics = semantic_digest(
        semantic_ir,
        conditioning_plan,
        degradation=list(plan_payload.get("fallbacks") or []),
    )["digest"]
    if expected_semantics and expected_semantics != current_semantics:
        raise PromptSemanticReplayError(
            "Recorded PPSR semantic conditioning changed under the current compiler/runtime contract."
        )
    output = dict(record)
    output["replay_locked"] = True
    output["replay_source"] = "recorded_exact"
    output["validated_structure_digest"] = current_structure
    output["validated_semantic_digest"] = current_semantics
    return output


def build_ppsr_semantic_pass_record(
    *,
    parser_id: str,
    parser_version: str,
    pass_name: str,
    positive_records: Sequence[Mapping[str, Any]],
    negative_records: Sequence[Mapping[str, Any]],
    legacy_superhybrid_record: Mapping[str, Any] | None = None,
    replay_source: str = "reconstruct",
) -> dict[str, Any]:
    positive = [dict(item or {}) for item in positive_records]
    negative = [dict(item or {}) for item in negative_records]
    if len(positive) != len(negative):
        raise PromptSemanticReplayError("PPSR positive/negative semantic slot counts do not match.")
    record = {
        "contract_version": PPSR_SEMANTIC_PASS_CONTRACT_VERSION,
        "compiler_version": PPSR_SEMANTIC_COMPILER_VERSION,
        "parser_id": str(parser_id or "legacy"),
        "parser_version": str(parser_version or ""),
        "pass": str(pass_name or "base"),
        "slot_count": len(positive),
        "positive": positive,
        "negative": negative,
        "superhybrid_record": dict(legacy_superhybrid_record or {}),
        "replay_locked": replay_source == "recorded_exact",
        "replay_source": str(replay_source or "reconstruct"),
    }
    record["fingerprint"] = {"algorithm": "sha256", "digest": _stable_hash(_fingerprint_source(record))}
    return record


def validate_ppsr_semantic_pass_record(
    recorded: Mapping[str, Any],
    *,
    parser_id: str,
    pass_name: str,
    slot_count: int,
) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != PPSR_SEMANTIC_PASS_CONTRACT_VERSION:
        raise PromptSemanticReplayError("Recorded prompt semantic pass data uses an unsupported contract version.")
    fingerprint = dict(record.get("fingerprint") or {})
    if fingerprint.get("algorithm") != "sha256" or fingerprint.get("digest") != _stable_hash(_fingerprint_source(record)):
        raise PromptSemanticReplayError("Recorded prompt semantic pass fingerprint validation failed.")
    if str(record.get("parser_id") or "") != str(parser_id or ""):
        raise PromptSemanticReplayError("Recorded prompt semantic pass parser does not match the active parser.")
    if str(record.get("pass") or "") != str(pass_name or ""):
        raise PromptSemanticReplayError("Recorded prompt semantic pass name does not match the active pass.")
    if int(record.get("slot_count", -1)) != int(slot_count):
        raise PromptSemanticReplayError("Recorded prompt semantic slot count does not match the current request.")
    if len(list(record.get("positive") or [])) != int(slot_count) or len(list(record.get("negative") or [])) != int(slot_count):
        raise PromptSemanticReplayError("Recorded prompt semantic role records are incomplete.")
    return record


def resolve_request_prompt_ir(
    request: Any,
    *,
    source: str,
    parser_namespace: str,
    parser_contract_version: str,
):
    """Prefer recorded PPSR semantic IR for exact replay; otherwise parse source."""
    from modules.prompt_parsers.ir import parse_prompt_ir

    mode = str(getattr(request, "semantic_replay_mode", "reconstruct") or "reconstruct").strip().lower()
    recorded = dict(getattr(request, "recorded_semantic_replay", {}) or {})
    if mode not in {"reconstruct", "recorded_exact"}:
        raise PromptSemanticReplayError("PPSR semantic replay mode must be reconstruct or recorded_exact.")
    if mode == "recorded_exact":
        if not recorded:
            raise PromptSemanticReplayError("Exact PPSR semantic replay was requested without a recorded semantic record.")
        ir, record = replay_prompt_ir(
            recorded,
            parser_id=parser_namespace,
            parser_contract_version=parser_contract_version,
            prompt_role=str(getattr(request, "prompt_role", "positive") or "positive"),
        )
        return ir, {
            "mode": mode,
            "source": "recorded_exact",
            "migration_path": str(record.get("migration_path") or "none"),
            "recorded_semantic_digest": dict(record.get("semantic_digest") or {}),
            "recorded_structure_digest": dict(record.get("structure_digest") or {}),
        }
    return parse_prompt_ir(str(source or ""), parser_namespace=parser_namespace), {
        "mode": mode,
        "source": "reconstruct",
        "migration_path": "none",
    }


def select_ppsr_semantic_pass_slot(recorded: Mapping[str, Any], slot_index: int) -> dict[str, Any]:
    record = dict(recorded or {})
    if record.get("contract_version") != PPSR_SEMANTIC_PASS_CONTRACT_VERSION:
        return record
    count = int(record.get("slot_count", 0) or 0)
    index = int(slot_index)
    if index < 0 or index >= count:
        raise PromptSemanticReplayError("PPSR semantic slot index is outside the recorded batch.")
    if count <= 1:
        return record
    legacy = dict(record.get("superhybrid_record") or {})
    if legacy.get("contract_version") == SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION:
        legacy = select_superhybrid_semantic_slot(legacy, index)
    projected = build_ppsr_semantic_pass_record(
        parser_id=str(record.get("parser_id") or "legacy"),
        parser_version=str(record.get("parser_version") or ""),
        pass_name=str(record.get("pass") or "base"),
        positive_records=[dict((record.get("positive") or [])[index] or {})],
        negative_records=[dict((record.get("negative") or [])[index] or {})],
        legacy_superhybrid_record=legacy,
        replay_source="projected_image_manifest",
    )
    projected["source_batch_slot_index"] = index
    projected["source_batch_fingerprint"] = dict(record.get("fingerprint") or {})
    # Recompute because the source-batch annotations are part of this projected record.
    projected["fingerprint"] = {
        "algorithm": "sha256",
        "digest": _stable_hash(_fingerprint_source(projected)),
    }
    return projected
