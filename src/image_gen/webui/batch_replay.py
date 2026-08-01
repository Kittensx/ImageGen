from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.model_selection import WebUIModelSelectionState
from image_gen.webui.replay import ReplayService, _BATCH_OVERRIDE_FIELDS

MAX_SELECTED_OUTPUTS = 100
MAX_COMPOSER_JOBS = 250
_TOKEN_TTL_SECONDS = 15 * 60
_REMAP_FIELDS = {"model_path", "vae_path", "sampler_name", "scheduler_name"}


@dataclass
class BatchReplayPreflight:
    valid: bool
    jobs: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]
    remap_groups: list[dict[str, Any]]
    preflight_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _StoredBatchPreflight:
    token: str
    specification: dict[str, Any]
    created_monotonic: float = field(default_factory=time.monotonic)


class BatchReplayService:
    """Phase 10C multi-output replay preflight and ordered FIFO submission."""

    def __init__(
        self,
        replay: ReplayService,
        jobs: GenerationJobManager,
        model_selection: WebUIModelSelectionState,
    ) -> None:
        self.replay = replay
        self.jobs = jobs
        self.model_selection = model_selection
        self._tokens: dict[str, _StoredBatchPreflight] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _unique_ids(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            token = str(value or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            output.append(token)
        return output

    @staticmethod
    def _safe_remap(value: Any) -> dict[str, Any]:
        source = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            key: copy.deepcopy(item)
            for key, item in source.items()
            if key in _REMAP_FIELDS and item not in (None, "")
        }

    def _safe_specification(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        source = dict(payload or {})
        output_ids = self._unique_ids(source.get("output_ids"))
        if not output_ids:
            raise ValueError("Select at least one output before composing a queue.")
        if len(output_ids) > MAX_SELECTED_OUTPUTS:
            raise ValueError(f"A queue composition may select at most {MAX_SELECTED_OUTPUTS} outputs.")

        order = self._unique_ids(source.get("order") or output_ids)
        unknown = [item for item in order if item not in output_ids]
        if unknown:
            raise ValueError("Queue order contains output IDs that are not selected.")
        # Removing an item in preview is represented by omitting it from order.
        if not order:
            raise ValueError("The composed queue contains no jobs.")
        if len(order) > MAX_COMPOSER_JOBS:
            raise ValueError(f"A queue composition may contain at most {MAX_COMPOSER_JOBS} jobs.")

        mode = str(source.get("mode") or "exact_or_override").strip().lower()
        if mode not in {"exact", "exact_or_override", "override"}:
            raise ValueError("Batch replay mode must be exact or exact_or_override.")

        overrides = copy.deepcopy(dict(source.get("overrides") or {}))
        override_fields = [str(item) for item in source.get("override_fields") or []]
        unsupported = sorted(set(override_fields) - _BATCH_OVERRIDE_FIELDS)
        if unsupported:
            raise ValueError("Unsupported common override field(s): " + ", ".join(unsupported))
        missing_values = [field for field in override_fields if field not in overrides]
        if missing_values:
            raise ValueError("Enabled override field(s) have no value: " + ", ".join(missing_values))

        seed_source = dict(source.get("seed_policy") or {})
        seed_mode = str(seed_source.get("mode") or "keep_original").strip().lower()
        if seed_mode not in {"keep_original", "random", "sequential"}:
            raise ValueError("Seed policy must be keep_original, random, or sequential.")
        seed_start: int | None = None
        if seed_mode == "sequential":
            try:
                seed_start = int(seed_source.get("start"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Sequential seed policy requires an integer start value.") from exc

        common_remap = self._safe_remap(source.get("common_remap"))
        item_source = source.get("item_remaps") or {}
        item_remaps = {
            str(output_id): self._safe_remap(remap)
            for output_id, remap in dict(item_source).items()
            if str(output_id) in output_ids
        } if isinstance(item_source, Mapping) else {}

        return {
            "output_ids": output_ids,
            "order": order,
            "mode": mode,
            "overrides": overrides,
            "override_fields": override_fields,
            "seed_policy": {"mode": seed_mode, "start": seed_start},
            "common_remap": common_remap,
            "item_remaps": item_remaps,
        }

    def _cleanup_tokens(self) -> None:
        cutoff = time.monotonic() - _TOKEN_TTL_SECONDS
        with self._lock:
            for token in [
                token for token, stored in self._tokens.items()
                if stored.created_monotonic < cutoff
            ]:
                self._tokens.pop(token, None)

    def _issue_token(self, specification: dict[str, Any]) -> str:
        self._cleanup_tokens()
        token = uuid.uuid4().hex
        with self._lock:
            self._tokens[token] = _StoredBatchPreflight(
                token=token,
                specification=copy.deepcopy(specification),
            )
        return token

    def _consume_specification(self, token: str) -> dict[str, Any]:
        self._cleanup_tokens()
        with self._lock:
            stored = self._tokens.get(str(token or ""))
        if stored is None:
            raise ValueError("Batch replay preflight expired or was not found. Run preflight again.")
        return copy.deepcopy(stored.specification)

    @staticmethod
    def _prompt_summary(request: Mapping[str, Any], limit: int = 96) -> str:
        prompt = " ".join(str(request.get("positive_prompt") or "").split())
        return prompt if len(prompt) <= limit else prompt[: limit - 1].rstrip() + "…"

    def _item_specification(
        self,
        specification: Mapping[str, Any],
        output_id: str,
        index: int,
    ) -> dict[str, Any]:
        request_overrides = copy.deepcopy(dict(specification.get("overrides") or {}))
        override_fields = list(specification.get("override_fields") or [])
        seed_policy = dict(specification.get("seed_policy") or {})
        seed_mode = seed_policy.get("mode")
        replay_seed_mode = "original"
        if seed_mode == "random":
            replay_seed_mode = "random"
        elif seed_mode == "sequential":
            request_overrides["seed"] = int(seed_policy.get("start")) + index
            if "seed" not in override_fields:
                override_fields.append("seed")

        remap = dict(specification.get("common_remap") or {})
        remap.update(dict((specification.get("item_remaps") or {}).get(output_id) or {}))
        return {
            "output_id": output_id,
            "mode": "exact",
            "selected_fields": [],
            "current_values": {},
            "seed_mode": replay_seed_mode,
            "model_mode": "original",
            "remap": remap,
            "request_overrides": request_overrides,
            "override_fields": override_fields,
        }

    @staticmethod
    def _remap_groups(items: list[dict[str, Any]], specification: Mapping[str, Any]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        common_remap = dict(specification.get("common_remap") or {})
        for item in items:
            for missing in item.get("missing_assets") or []:
                field_name = str(missing.get("field") or "")
                group = groups.setdefault(field_name, {
                    "field": field_name,
                    "kind": missing.get("kind"),
                    "affected_output_ids": [],
                    "requested_values": [],
                    "unresolved_output_ids": [],
                    "common_replacement": common_remap.get(field_name),
                    "common_replacement_resolves_all": False,
                    "item_specific_replacement_supported": True,
                })
                output_id = item["output_id"]
                group["affected_output_ids"].append(output_id)
                group["unresolved_output_ids"].append(output_id)
                requested = missing.get("requested")
                if requested not in group["requested_values"]:
                    group["requested_values"].append(requested)
        for group in groups.values():
            replacement = group.get("common_replacement")
            group["common_replacement_resolves_all"] = bool(replacement) and not group["unresolved_output_ids"]
            group["requires_item_specific_review"] = len(group["requested_values"]) > 1
        return list(groups.values())

    def _evaluate(self, specification: dict[str, Any], *, issue_token: bool) -> BatchReplayPreflight:
        items: list[dict[str, Any]] = []
        aggregate_warnings: list[str] = []
        aggregate_errors: list[str] = []
        quality_counts = {"exact_request": 0, "best_available": 0}

        for index, output_id in enumerate(specification["order"]):
            item_spec = self._item_specification(specification, output_id, index)
            try:
                preflight = self.replay.evaluate_specification(item_spec, issue_token=False)
                quality = str(preflight.completeness.get("quality") or "best_available")
                quality_counts[quality if quality in quality_counts else "best_available"] += 1
                request = copy.deepcopy(preflight.request)
                item = {
                    "order": index + 1,
                    "output_id": output_id,
                    "valid": preflight.valid,
                    "request": request,
                    "summary": {
                        **copy.deepcopy(preflight.summary),
                        "prompt_summary": self._prompt_summary(request),
                        "advanced_warning_count": len(preflight.warnings)
                        + len(preflight.unsupported_settings),
                    },
                    "errors": list(preflight.errors),
                    "warnings": list(preflight.warnings),
                    "missing_assets": copy.deepcopy(preflight.missing_assets),
                    "completeness": copy.deepcopy(preflight.completeness),
                    "field_results": copy.deepcopy(preflight.field_results),
                }
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
                item = {
                    "order": index + 1,
                    "output_id": output_id,
                    "valid": False,
                    "request": {},
                    "summary": {"prompt_summary": "", "advanced_warning_count": 0},
                    "errors": [str(exc)],
                    "warnings": [],
                    "missing_assets": [],
                    "completeness": {
                        "quality": "best_available",
                        "label": "Best Available Replay",
                    },
                    "field_results": [],
                }
                quality_counts["best_available"] += 1
            items.append(item)
            aggregate_warnings.extend(item["warnings"])
            aggregate_errors.extend(f"{output_id}: {message}" for message in item["errors"])

        valid_count = sum(1 for item in items if item["valid"])
        invalid_count = len(items) - valid_count
        remap_groups = self._remap_groups(items, specification)
        summary = {
            "selected_count": len(specification["output_ids"]),
            "job_count": len(items),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "quality_counts": quality_counts,
            "exact_request_count": quality_counts["exact_request"],
            "best_available_count": quality_counts["best_available"],
            "seed_policy": copy.deepcopy(specification["seed_policy"]),
            "override_fields": list(specification["override_fields"]),
            "common_remap_fields": sorted(specification["common_remap"]),
            "requires_item_specific_remap": any(
                group.get("requires_item_specific_review") for group in remap_groups
            ),
        }
        result = BatchReplayPreflight(
            valid=invalid_count == 0,
            jobs=items,
            errors=list(dict.fromkeys(aggregate_errors)),
            warnings=list(dict.fromkeys(aggregate_warnings)),
            summary=summary,
            remap_groups=remap_groups,
        )
        if issue_token:
            result.preflight_token = self._issue_token(specification)
        return result

    def preflight(self, payload: Mapping[str, Any] | None) -> BatchReplayPreflight:
        return self._evaluate(self._safe_specification(payload), issue_token=True)

    async def submit(
        self,
        preflight_token: str,
        *,
        queue_valid_only: bool = False,
    ) -> tuple[BatchReplayPreflight, list[Any], list[dict[str, Any]]]:
        specification = self._consume_specification(preflight_token)
        result = self._evaluate(specification, issue_token=False)
        if not result.valid and not queue_valid_only:
            raise ValueError(
                "The composed queue contains invalid jobs. Choose Queue Valid Jobs Only or fix the errors."
            )

        submitted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for item in result.jobs:
            if not item["valid"]:
                rejected.append({
                    "output_id": item["output_id"],
                    "errors": list(item["errors"]),
                })
                continue
            request = copy.deepcopy(item["request"])
            try:
                selection = self.model_selection.authorize(
                    request.get("model_path"), source="batch_replay_submission"
                )
                request["model_path"] = selection.resolved_path
                job = await self.jobs.submit(request, model_selection=selection.to_dict())
                submitted.append(job)
            except (OSError, TypeError, ValueError) as exc:
                rejected.append({"output_id": item["output_id"], "errors": [str(exc)]})

        with self._lock:
            self._tokens.pop(preflight_token, None)
        return result, submitted, rejected


__all__ = [
    "BatchReplayPreflight",
    "BatchReplayService",
    "MAX_SELECTED_OUTPUTS",
    "MAX_COMPOSER_JOBS",
]
