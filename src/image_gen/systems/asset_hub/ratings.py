from __future__ import annotations

from typing import Any, Iterable, Mapping

from image_gen.systems.asset_hub.contracts import (
    MATURITY_LEVELS,
    MATURITY_UNKNOWN_LEVEL,
    PREVIEW_MATURITY_COMPLETENESS,
)

RATING_INDEX_SCHEMA_VERSION = 1
RATING_POLICY_SCHEMA_VERSION = 1
RATING_SEVERITY = {level: index for index, level in enumerate(MATURITY_LEVELS)}
RATING_BASES = ("asset", "author_previews", "strictest")
RATING_SORTS = ("provider", "safest_first", "most_mature")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def unknown_maturity(provider_id: str = "") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "providerId": _text(provider_id) or None,
        "raw": None,
        "mask": None,
        "levels": [MATURITY_UNKNOWN_LEVEL],
        "state": "unknown",
        "sourceField": None,
        "unknownBits": 0,
    }


def maturity_is_known(value: Any) -> bool:
    maturity = _mapping(value)
    if _text(maturity.get("state")).casefold() != "known":
        return False
    levels = [str(level) for level in _list(maturity.get("levels"))]
    return any(level in RATING_SEVERITY for level in levels)


def maturity_information_score(value: Any) -> tuple[int, int, int, int]:
    """Rank rating payloads by preserved provider information.

    Known normalized truth always outranks Unknown. Among equal states, retain the
    payload carrying more provider provenance rather than replacing it with an
    emptier response from a different endpoint.
    """
    maturity = _mapping(value)
    return (
        1 if maturity_is_known(maturity) else 0,
        1 if maturity.get("raw") is not None else 0,
        1 if _text(maturity.get("sourceField")) else 0,
        len(_list(maturity.get("levels"))),
    )


def preserve_maturity(existing: Any, incoming: Any, *, provider_id: str = "") -> dict[str, Any]:
    old = _mapping(existing)
    new = _mapping(incoming)
    if not old and not new:
        return unknown_maturity(provider_id)
    if not old:
        return dict(new) if new else unknown_maturity(provider_id)
    if not new:
        return dict(old)
    if maturity_information_score(new) >= maturity_information_score(old):
        return dict(new)
    return dict(old)


def maturity_summary(value: Any) -> dict[str, Any]:
    maturity = _mapping(value)
    levels = [str(level) for level in _list(maturity.get("levels")) if str(level) in RATING_SEVERITY]
    known = maturity_is_known(maturity)
    severity = max((RATING_SEVERITY[level] for level in levels), default=-1)
    unknown_bits = max(0, _integer(maturity.get("unknownBits"), 0))
    return {
        "state": "known" if known else "unknown",
        "levels": levels if known else [MATURITY_UNKNOWN_LEVEL],
        "severity": severity if known else -1,
        "blocked": bool(known and "Blocked" in levels),
        "unknownBits": unknown_bits,
        "indeterminate": (not known) or unknown_bits > 0,
    }


def _preview_identity(item: Mapping[str, Any]) -> str:
    provider_image_id = _text(item.get("providerImageId"))
    if provider_image_id:
        return f"id:{provider_image_id}"
    url = _text(item.get("url"))
    if url:
        return f"url:{url}"
    return ""


def merge_preview_items(existing: Any, incoming: Any, *, completeness: str = "unknown", provider_id: str = "") -> list[dict[str, Any]]:
    old_items = [dict(item) for item in _list(existing) if isinstance(item, Mapping)]
    new_items = [dict(item) for item in _list(incoming) if isinstance(item, Mapping)]
    old_by_id = {_preview_identity(item): item for item in old_items if _preview_identity(item)}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in new_items:
        identity = _preview_identity(item)
        prior = old_by_id.get(identity, {}) if identity else {}
        combined = {**prior, **item}
        combined["maturity"] = preserve_maturity(
            prior.get("maturity"), item.get("maturity"), provider_id=provider_id
        )
        merged.append(combined)
        if identity:
            seen.add(identity)

    # Unless the provider explicitly proves this is a complete preview set, a
    # shorter/filtered response cannot erase preview-rating knowledge learned
    # earlier from another provider endpoint.
    if completeness != "complete":
        for item in old_items:
            identity = _preview_identity(item)
            if identity and identity in seen:
                continue
            merged.append(dict(item))
            if identity:
                seen.add(identity)
    return merged


def merge_preview_completeness(existing: Any, incoming: Any) -> str:
    old = _text(existing).casefold()
    new = _text(incoming).casefold()
    if old not in PREVIEW_MATURITY_COMPLETENESS:
        old = "unknown"
    if new not in PREVIEW_MATURITY_COMPLETENESS:
        new = "unknown"
    # Complete is positive proof and should not be lost to a later filtered or
    # metadata-light endpoint. provider_filtered is more informative than an
    # unqualified unknown response.
    rank = {"unknown": 0, "provider_filtered": 1, "complete": 2}
    return new if rank[new] >= rank[old] else old


def normalize_preview_completeness(value: Any) -> str:
    selected = _text(value).casefold()
    return selected if selected in PREVIEW_MATURITY_COMPLETENESS else "unknown"


def ensure_rating_contract(item: Mapping[str, Any], *, provider_id: str = "") -> dict[str, Any]:
    """Make legacy cached provider payloads explicitly Unknown without guessing."""
    output = dict(item)
    provider = _text(output.get("providerId") or provider_id)
    output["maturity"] = preserve_maturity(None, output.get("maturity"), provider_id=provider)
    versions: list[dict[str, Any]] = []
    for raw_version in _list(output.get("versions")):
        if not isinstance(raw_version, Mapping):
            continue
        version = dict(raw_version)
        version["maturity"] = preserve_maturity(None, version.get("maturity"), provider_id=provider)
        previews: list[dict[str, Any]] = []
        for raw_preview in _list(version.get("previews")):
            if not isinstance(raw_preview, Mapping):
                continue
            preview = dict(raw_preview)
            preview["maturity"] = preserve_maturity(None, preview.get("maturity"), provider_id=provider)
            previews.append(preview)
        version["previews"] = previews

        author = _mapping(version.get("authorPreviewMaturity"))
        completeness = normalize_preview_completeness(author.get("completeness"))
        author_items = [dict(value) for value in _list(author.get("items")) if isinstance(value, Mapping)]
        if author_items:
            normalized_items = []
            for author_item in author_items:
                normalized_item = dict(author_item)
                normalized_item["maturity"] = preserve_maturity(
                    None, normalized_item.get("maturity"), provider_id=provider
                )
                normalized_items.append(normalized_item)
        else:
            normalized_items = [
                {
                    "providerImageId": preview.get("providerImageId"),
                    "kind": preview.get("kind") or "image",
                    "maturity": dict(preview["maturity"]),
                }
                for preview in previews
            ]
        version["authorPreviewMaturity"] = {
            "completeness": completeness,
            "items": normalized_items,
        }
        versions.append(version)
    output["versions"] = versions
    return output


def _version_preview_contract(version: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    author = _mapping(version.get("authorPreviewMaturity"))
    completeness = _text(author.get("completeness")).casefold()
    if completeness not in PREVIEW_MATURITY_COMPLETENESS:
        completeness = _text(version.get("previewMaturityCompleteness")).casefold()
    if completeness not in PREVIEW_MATURITY_COMPLETENESS:
        completeness = "unknown"

    items = [dict(item) for item in _list(author.get("items")) if isinstance(item, Mapping)]
    if not items:
        items = [
            {
                "providerImageId": preview.get("providerImageId"),
                "kind": preview.get("kind") or "image",
                "maturity": _mapping(preview.get("maturity")),
            }
            for preview in _list(version.get("previews"))
            if isinstance(preview, Mapping)
        ]
    return completeness, items


def _preview_aggregate(items: Iterable[Mapping[str, Any]], completeness: str) -> dict[str, Any]:
    summaries = [maturity_summary(item.get("maturity")) for item in items]
    known = [summary for summary in summaries if summary["state"] == "known"]
    levels: list[str] = []
    for summary in known:
        for level in summary["levels"]:
            if level not in levels:
                levels.append(level)
    levels.sort(key=lambda level: RATING_SEVERITY.get(level, -1))
    severity = max((int(summary["severity"]) for summary in known), default=-1)
    any_item_indeterminate = any(bool(summary["indeterminate"]) for summary in summaries)
    complete = completeness == "complete"
    return {
        "state": "known" if known else "unknown",
        "levels": levels if known else [MATURITY_UNKNOWN_LEVEL],
        "severity": severity,
        "blocked": any(bool(summary["blocked"]) for summary in known),
        "itemCount": len(summaries),
        "knownItemCount": len(known),
        "completeness": completeness,
        "indeterminate": (not complete) or any_item_indeterminate or not known,
    }


def build_rating_index(item: Mapping[str, Any]) -> dict[str, Any]:
    model = maturity_summary(item.get("maturity"))
    versions_output: list[dict[str, Any]] = []
    all_preview_items: list[dict[str, Any]] = []
    version_completeness: list[str] = []
    version_summaries: list[dict[str, Any]] = []

    for raw_version in _list(item.get("versions")):
        if not isinstance(raw_version, Mapping):
            continue
        version = dict(raw_version)
        version_summary = maturity_summary(version.get("maturity"))
        version_summaries.append(version_summary)
        completeness, preview_items = _version_preview_contract(version)
        version_completeness.append(completeness)
        all_preview_items.extend(preview_items)
        versions_output.append({
            "remoteVersionId": _text(version.get("remoteVersionId")) or None,
            "maturity": version_summary,
            "authorPreviews": _preview_aggregate(preview_items, completeness),
        })

    if not version_completeness:
        overall_completeness = "unknown"
    elif "unknown" in version_completeness:
        overall_completeness = "unknown"
    elif "provider_filtered" in version_completeness:
        overall_completeness = "provider_filtered"
    else:
        overall_completeness = "complete"

    version_known = [summary for summary in version_summaries if summary["state"] == "known"]
    version_levels: list[str] = []
    for summary in version_known:
        for level in summary["levels"]:
            if level not in version_levels:
                version_levels.append(level)
    version_levels.sort(key=lambda level: RATING_SEVERITY.get(level, -1))
    versions_aggregate = {
        "state": "known" if version_known else "unknown",
        "levels": version_levels if version_known else [MATURITY_UNKNOWN_LEVEL],
        "severity": max((int(summary["severity"]) for summary in version_known), default=-1),
        "blocked": any(bool(summary["blocked"]) for summary in version_known),
        "indeterminate": (not version_known) or any(bool(summary["indeterminate"]) for summary in version_summaries),
    }
    author_previews = _preview_aggregate(all_preview_items, overall_completeness)
    return {
        "schemaVersion": RATING_INDEX_SCHEMA_VERSION,
        "model": model,
        "versions": versions_output,
        "versionAggregate": versions_aggregate,
        "authorPreviews": author_previews,
    }


def normalize_rating_policy(value: Any) -> dict[str, Any]:
    policy = _mapping(value)
    allowed_raw = policy.get("allowed")
    raw_levels = [str(level) for level in _list(allowed_raw)]
    allowed = [level for level in raw_levels if level in MATURITY_LEVELS and level != "Blocked"]
    if allowed_raw is None:
        allowed = [level for level in MATURITY_LEVELS if level != "Blocked"]
    allowed = sorted(set(allowed), key=lambda level: RATING_SEVERITY[level])

    basis = _text(policy.get("basis")).casefold().replace("-", "_")
    aliases = {"author": "author_previews", "previews": "author_previews", "both": "strictest"}
    basis = aliases.get(basis, basis)
    if basis not in RATING_BASES:
        basis = "strictest"

    sort = _text(policy.get("sort")).casefold().replace("-", "_")
    if sort not in RATING_SORTS:
        sort = "provider"

    include_unknown = bool(
        policy.get("includeUnknown", policy.get("include_unknown", False))
        or MATURITY_UNKNOWN_LEVEL in raw_levels
    )
    return {
        "schemaVersion": RATING_POLICY_SCHEMA_VERSION,
        "allowed": allowed,
        "includeUnknown": include_unknown,
        "basis": basis,
        "sort": sort,
    }


def evaluate_rating_index(rating_index: Any, policy_value: Any) -> dict[str, Any]:
    """Evaluate a persisted rating index without contacting the provider.

    Incomplete/partially unknown provider truth remains Unknown for eligibility.
    A known disallowed level or Blocked flag can still reject deterministically;
    otherwise the user's explicit Unknown preference decides indeterminate data.
    """
    index = _mapping(rating_index)
    policy = normalize_rating_policy(policy_value)
    basis = policy["basis"]
    components: list[dict[str, Any]] = []
    if basis in {"asset", "strictest"}:
        components.append(_mapping(index.get("model")))
    if basis in {"author_previews", "strictest"}:
        components.append(_mapping(index.get("authorPreviews")))

    known_levels: list[str] = []
    severity = -1
    blocked = False
    indeterminate = not components
    for component in components:
        state = _text(component.get("state")).casefold()
        component_levels = [str(level) for level in _list(component.get("levels")) if str(level) in RATING_SEVERITY]
        if state != "known" or bool(component.get("indeterminate")):
            indeterminate = True
        for level in component_levels:
            if level not in known_levels:
                known_levels.append(level)
            severity = max(severity, RATING_SEVERITY[level])
            if level == "Blocked":
                blocked = True

    known_levels.sort(key=lambda level: RATING_SEVERITY.get(level, -1))
    disallowed = [
        level for level in known_levels
        if level == "Blocked" or level not in set(policy["allowed"])
    ]
    if blocked:
        eligible = False
        reason = "blocked"
    elif disallowed:
        eligible = False
        reason = "disallowed_known_level"
    elif indeterminate:
        eligible = bool(policy["includeUnknown"])
        reason = "unknown_included" if eligible else "unknown_excluded"
    else:
        eligible = True
        reason = "allowed"

    return {
        "schemaVersion": RATING_POLICY_SCHEMA_VERSION,
        "basis": basis,
        "eligible": eligible,
        "state": "unknown" if indeterminate else "known",
        "levels": known_levels if known_levels else [MATURITY_UNKNOWN_LEVEL],
        "severity": severity,
        "indeterminate": indeterminate,
        "blocked": blocked,
        "reason": reason,
    }


def rating_sort_key(evaluation: Any, sort: str) -> tuple[int, int]:
    selected = _text(sort).casefold().replace("-", "_")
    result = _mapping(evaluation)
    indeterminate = bool(result.get("indeterminate")) or _text(result.get("state")).casefold() != "known"
    severity = _integer(result.get("severity"), -1)
    if selected == "safest_first":
        return (1, 0) if indeterminate else (0, severity)
    if selected == "most_mature":
        return (1, 0) if indeterminate else (0, -severity)
    return (0, 0)
