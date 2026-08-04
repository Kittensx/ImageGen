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
