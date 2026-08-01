from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch


OOM_RECOVERY_PROFILES = ("disabled", "cleanup", "low_vram", "maximum")
_PROFILE_STRENGTH = {
    "disabled": 0,
    "cleanup": 1,
    "low_vram": 2,
    "maximum": 3,
}
_SAMPLER_STAGES = {"sampling", "hires_second_pass"}


def normalize_oom_recovery_profile(value: Any) -> str:
    selected = str(value or "disabled").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "off": "disabled",
        "none": "disabled",
        "no_retry": "disabled",
        "retry": "cleanup",
        "on": "cleanup",
        "low": "low_vram",
        "lowvram": "low_vram",
        "max": "maximum",
        "maximum_memory_savings": "maximum",
    }
    selected = aliases.get(selected, selected)
    if selected not in OOM_RECOVERY_PROFILES:
        raise ValueError(
            "oom_retry_profile must be one of: disabled, cleanup, low_vram, maximum."
        )
    return selected


def is_cuda_oom(error: BaseException) -> bool:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(error, cuda_oom_type):
        return True
    text = str(error).lower()
    return "cuda out of memory" in text or ("out of memory" in text and "cuda" in text)


@dataclass
class StageRecoveryContract:
    """Explicit valid-boundary contract for safely restarting one failed stage.

    Sampling retries must provide this contract. ``operation_factory`` receives
    the zero-based attempt index and must return an operation that starts from
    the saved boundary rather than from partially mutated sampler state.
    """

    boundary_id: str
    operation_factory: Callable[[int], Callable[[], Any]]
    restart_mode: str = "same_stage_from_saved_boundary"
    prepare_retry: Callable[[str, int], Mapping[str, Any] | None] | None = None
    release_boundary: Callable[[], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def operation_for_attempt(self, attempt_index: int) -> Callable[[], Any]:
        operation = self.operation_factory(int(attempt_index))
        if not callable(operation):
            raise TypeError("OOM recovery operation_factory must return a callable.")
        return operation

    def prepare(self, profile: str, retry_index: int) -> dict[str, Any]:
        callback = self.prepare_retry
        if not callable(callback):
            return {}
        payload = callback(str(profile), int(retry_index))
        return dict(payload or {})

    def release(self) -> None:
        callback = self.release_boundary
        if callable(callback):
            callback()

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": str(self.boundary_id),
            "restart_mode": str(self.restart_mode),
            "metadata": dict(self.metadata),
        }


@dataclass
class OOMRecoveryState:
    configured_profile: str = "disabled"
    retry_limit: int = 1
    retry_count_by_stage: dict[str, int] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_result_by_stage: dict[str, str] = field(default_factory=dict)
    runtime_path_changed: bool = False

    def __post_init__(self) -> None:
        self.configured_profile = normalize_oom_recovery_profile(
            self.configured_profile
        )
        self.retry_limit = max(0, int(self.retry_limit))

    @property
    def enabled(self) -> bool:
        return self.configured_profile != "disabled" and self.retry_limit > 0

    @property
    def total_retry_count(self) -> int:
        return sum(int(value) for value in self.retry_count_by_stage.values())

    @staticmethod
    def sampler_stage(stage: str) -> bool:
        return str(stage) in _SAMPLER_STAGES

    def may_retry(
        self,
        stage: str,
        *,
        has_valid_boundary: bool = True,
    ) -> bool:
        target = str(stage)
        if not self.enabled:
            return False
        if self.total_retry_count >= self.retry_limit:
            return False
        # A stage is retried at most once. A second OOM after the fallback ends
        # the job even when the global job retry limit is greater than one.
        if int(self.retry_count_by_stage.get(target, 0)) >= 1:
            return False
        if self.sampler_stage(target) and not has_valid_boundary:
            return False
        return True

    def record_retry(
        self,
        stage: str,
        *,
        original_profile: str,
        fallback_profile: str,
        boundary: StageRecoveryContract | None,
        error: BaseException,
    ) -> dict[str, Any]:
        target = str(stage)
        self.retry_count_by_stage[target] = int(
            self.retry_count_by_stage.get(target, 0)
        ) + 1
        self.runtime_path_changed = True
        attempt = {
            "stage": target,
            "retry_index": int(self.retry_count_by_stage[target]),
            "job_retry_index": int(self.total_retry_count),
            "original_profile": str(original_profile),
            "fallback_profile": str(fallback_profile),
            "profile_strength_increased": (
                _PROFILE_STRENGTH.get(str(fallback_profile), 0)
                > _PROFILE_STRENGTH.get(str(original_profile), 0)
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "boundary": boundary.to_dict() if boundary is not None else None,
            "result": "retry_pending",
            "actions": [],
        }
        self.attempts.append(attempt)
        return attempt

    def current_attempt(self, stage: str) -> dict[str, Any] | None:
        target = str(stage)
        for attempt in reversed(self.attempts):
            if str(attempt.get("stage")) == target:
                return attempt
        return None

    def record_action(self, stage: str, action: str, **details: Any) -> None:
        payload = {"stage": str(stage), "action": str(action), **details}
        self.actions.append(payload)
        attempt = self.current_attempt(stage)
        if attempt is not None and attempt.get("result") == "retry_pending":
            attempt.setdefault("actions", []).append(dict(payload))

    def record_result(
        self,
        stage: str,
        result: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        target = str(stage)
        selected = str(result)
        self.final_result_by_stage[target] = selected
        attempt = self.current_attempt(target)
        if attempt is not None:
            attempt["result"] = selected
            if error is not None:
                attempt["retry_error_type"] = type(error).__name__
                attempt["retry_error"] = str(error)

    def summary(self) -> dict[str, Any]:
        fallback_profiles = [
            str(item.get("fallback_profile") or "")
            for item in self.attempts
            if item.get("fallback_profile")
        ]
        return {
            "schema_version": 1,
            "configured_profile": self.configured_profile,
            "retry_limit": int(self.retry_limit),
            "enabled": self.enabled,
            "total_retry_count": int(self.total_retry_count),
            "retry_count_by_stage": dict(self.retry_count_by_stage),
            "attempts": [dict(item) for item in self.attempts],
            "actions": [dict(item) for item in self.actions],
            "final_result_by_stage": dict(self.final_result_by_stage),
            "runtime_path_changed": bool(self.runtime_path_changed),
            "fallback_profiles_applied": fallback_profiles,
            "deterministic_replay": {
                "fallback_changed_runtime_path": bool(self.runtime_path_changed),
                "exact_replay_requires_recorded_fallback_path": bool(
                    self.runtime_path_changed
                ),
                "note": (
                    "An OOM fallback changed component residency and/or memory controls. "
                    "Creative request values were preserved, but exact runtime-path replay "
                    "requires reproducing the recorded fallback."
                    if self.runtime_path_changed
                    else "No OOM fallback changed the runtime path."
                ),
            },
            "safety": {
                "per_stage_retry_limit": 1,
                "global_job_retry_limit": int(self.retry_limit),
                "sampler_retry_requires_valid_saved_boundary": True,
                "partially_mutated_sampler_state_is_never_continued": True,
            },
        }
