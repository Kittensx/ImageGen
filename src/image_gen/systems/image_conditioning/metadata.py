from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Mapping

import torch

from image_gen.contracts import SchedulerOutput
from .contracts import (
    ImageConditionedSchedule,
    ImageConditionedStepPlan,
    ScheduleRehydrationResult,
)

SCHEDULE_REPLAY_SCHEMA_VERSION = 1
SCHEDULE_REPLAY_FORMAT = "image-gen-schedule-replay-v1"
SCHEDULE_FINGERPRINT_SCHEMA_VERSION = 1
SCHEDULE_FINGERPRINT_FORMAT = "image-gen-schedule-fingerprint-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _tensor_payload(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    np_value = tensor.numpy()
    encoded = base64.b64encode(np_value.tobytes()).decode("ascii")
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(dim) for dim in tensor.shape],
        "values": [float(item) for item in tensor.reshape(-1).tolist()],
        "encoding": "base64_raw_bytes",
        "bytes_base64": encoded,
    }


def _tensor_sha256(value: torch.Tensor | None) -> str:
    if value is None:
        return ""
    tensor = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(tensor.tobytes()).hexdigest()


def build_schedule_replay_record(
    schedule: ImageConditionedSchedule,
    *,
    scheduler_identifier: str = "",
    scheduler_configuration: Mapping[str, Any] | None = None,
    sampler_name: str = "",
    requires_terminal_zero: bool | None = None,
) -> dict[str, Any]:
    full = schedule.full_schedule
    active = schedule.active_schedule
    full_timesteps = full.timesteps
    active_timesteps = active.timesteps
    terminal_zero = bool(abs(float(full.sigmas[-1].detach().cpu().item())) <= 1.0e-8)
    return {
        "schema_version": SCHEDULE_REPLAY_SCHEMA_VERSION,
        "format": SCHEDULE_REPLAY_FORMAT,
        "scheduler_identifier": str(scheduler_identifier or ""),
        "scheduler_configuration": dict(scheduler_configuration or {}),
        "sampler_name": str(sampler_name or ""),
        "step_policy": str(schedule.step_policy),
        "requested_refinement_steps": int(schedule.requested_refinement_steps),
        "planned_internal_schedule_steps": int(schedule.step_plan.internal_schedule_steps),
        "internal_schedule_steps": int(schedule.internal_schedule_steps),
        "effective_refinement_steps": int(schedule.effective_refinement_steps),
        "requested_denoising_strength": float(schedule.step_plan.requested_denoising_strength),
        "denoising_strength": float(schedule.denoising_strength),
        "safe_denoising_strength": float(schedule.step_plan.safe_denoising_strength),
        "step_plan": schedule.step_plan.to_serializable_dict(),
        "start_index": int(schedule.start_index),
        "start_sigma": float(schedule.start_sigma),
        "start_timestep": (
            float(schedule.start_timestep) if schedule.start_timestep is not None else None
        ),
        "full_schedule": {
            "requested_steps": int(full.requested_steps),
            "effective_steps": int(full.effective_steps),
            "scheduler_step_override_applied": bool(full.scheduler_step_override_applied),
            "compatibility_mode": full.compatibility_mode,
            "sigmas": _tensor_payload(full.sigmas),
            "timesteps": _tensor_payload(full_timesteps),
        },
        "active_schedule": {
            "requested_steps": int(active.requested_steps),
            "effective_steps": int(active.effective_steps),
            "scheduler_step_override_applied": bool(active.scheduler_step_override_applied),
            "compatibility_mode": active.compatibility_mode,
            "sigmas": _tensor_payload(active.sigmas),
            "timesteps": _tensor_payload(active_timesteps),
        },
        "terminal_zero": {
            "required": requires_terminal_zero,
            "present_in_full_schedule": terminal_zero,
        },
    }


def build_schedule_fingerprint_record(
    schedule: ImageConditionedSchedule,
    *,
    replay_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    full = schedule.full_schedule
    active = schedule.active_schedule
    replay = dict(replay_record or {})
    contract_snapshot = {
        "step_policy": str(schedule.step_policy),
        "requested_refinement_steps": int(schedule.requested_refinement_steps),
        "planned_internal_schedule_steps": int(schedule.step_plan.internal_schedule_steps),
        "internal_schedule_steps": int(schedule.internal_schedule_steps),
        "effective_refinement_steps": int(schedule.effective_refinement_steps),
        "requested_denoising_strength": float(schedule.step_plan.requested_denoising_strength),
        "denoising_strength": float(schedule.denoising_strength),
        "safe_denoising_strength": float(schedule.step_plan.safe_denoising_strength),
        "start_index": int(schedule.start_index),
        "start_sigma": float(schedule.start_sigma),
        "start_timestep": (
            float(schedule.start_timestep) if schedule.start_timestep is not None else None
        ),
    }
    full_schedule_snapshot = {
        "sigmas_sha256": _tensor_sha256(full.sigmas),
        "timesteps_sha256": _tensor_sha256(full.timesteps),
        "requested_steps": int(full.requested_steps),
        "effective_steps": int(full.effective_steps),
    }
    active_schedule_snapshot = {
        "sigmas_sha256": _tensor_sha256(active.sigmas),
        "timesteps_sha256": _tensor_sha256(active.timesteps),
        "requested_steps": int(active.requested_steps),
        "effective_steps": int(active.effective_steps),
    }
    snapshot = {
        "contract": contract_snapshot,
        "full_schedule": full_schedule_snapshot,
        "active_schedule": active_schedule_snapshot,
        "replay_format": replay.get("format"),
    }
    return {
        "schema_version": SCHEDULE_FINGERPRINT_SCHEMA_VERSION,
        "format": SCHEDULE_FINGERPRINT_FORMAT,
        "sha256": hashlib.sha256(_canonical_json(snapshot).encode("ascii")).hexdigest(),
        "snapshot": snapshot,
    }


_TORCH_DTYPE_BY_NAME = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _tensor_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("Recorded schedule tensor payload must be a mapping.")
    encoding = str(payload.get("encoding") or "")
    if encoding != "base64_raw_bytes":
        raise ValueError(
            "Recorded schedule tensor payload uses an unsupported encoding: "
            f"{encoding!r}."
        )
    dtype_name = str(payload.get("dtype") or "")
    dtype = _TORCH_DTYPE_BY_NAME.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Recorded schedule tensor dtype is unsupported: {dtype_name!r}.")
    shape = tuple(int(value) for value in (payload.get("shape") or []))
    if not shape or any(value < 0 for value in shape):
        raise ValueError("Recorded schedule tensor shape is invalid.")
    encoded = str(payload.get("bytes_base64") or "")
    if not encoded:
        raise ValueError("Recorded schedule tensor payload is missing bytes_base64.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Recorded schedule tensor bytes_base64 is invalid.") from exc

    element_size = torch.empty((), dtype=dtype).element_size()
    expected_elements = 1
    for dimension in shape:
        expected_elements *= int(dimension)
    expected_bytes = expected_elements * element_size
    if len(raw) != expected_bytes:
        raise ValueError(
            "Recorded schedule tensor byte length does not match dtype and shape: "
            f"expected {expected_bytes}, received {len(raw)}."
        )

    buffer = bytearray(raw)
    tensor = torch.frombuffer(buffer, dtype=dtype, count=expected_elements).clone().reshape(shape)
    if device is not None:
        tensor = tensor.to(device=torch.device(device))
    return tensor


def _step_plan_from_record(record: Mapping[str, Any]) -> ImageConditionedStepPlan:
    step_plan = dict(record.get("step_plan") or {})
    requested_steps = int(
        step_plan.get("requested_refinement_steps", record.get("requested_refinement_steps", 0))
    )
    requested_strength = float(
        step_plan.get("requested_denoising_strength", record.get("requested_denoising_strength", record.get("denoising_strength", 1.0)))
    )
    normalized_strength = float(
        step_plan.get("normalized_denoising_strength", record.get("denoising_strength", requested_strength))
    )
    safe_strength = float(
        step_plan.get("safe_denoising_strength", record.get("safe_denoising_strength", normalized_strength))
    )
    internal_steps = int(
        step_plan.get("internal_schedule_steps", record.get("planned_internal_schedule_steps", record.get("internal_schedule_steps", 0)))
    )
    effective_steps = int(
        step_plan.get("effective_refinement_steps", record.get("effective_refinement_steps", 0))
    )
    if requested_steps < 1 or internal_steps < 1 or effective_steps < 1:
        raise ValueError("Recorded schedule step plan contains invalid step counts.")
    return ImageConditionedStepPlan(
        step_policy=str(step_plan.get("step_policy") or record.get("step_policy") or ""),
        requested_refinement_steps=requested_steps,
        requested_denoising_strength=requested_strength,
        normalized_denoising_strength=normalized_strength,
        safe_denoising_strength=safe_strength,
        internal_schedule_steps=internal_steps,
        effective_refinement_steps=effective_steps,
        minimum_supported_strength=float(step_plan.get("minimum_supported_strength", 0.01)),
        maximum_internal_schedule_steps=int(step_plan.get("maximum_internal_schedule_steps", 20_000)),
        denoising_strength_was_clamped=bool(step_plan.get("denoising_strength_was_clamped", False)),
    )


def rehydrate_schedule_replay_record(
    replay_record: Mapping[str, Any],
    *,
    expected_fingerprint: Mapping[str, Any] | str | None = None,
    device: torch.device | str | None = None,
    strict_fingerprint: bool = True,
) -> ScheduleRehydrationResult:
    """Restore exact full and active schedules from recorded tensor bytes.

    The caller receives a fully validated ``ImageConditionedSchedule``.  When
    an expected fingerprint is supplied, the rehydrated schedule is compared
    against it before the result is returned.  Strict mode rejects any
    mismatch so an exact replay cannot silently substitute a different
    schedule.
    """

    record = dict(replay_record or {})
    if record.get("format") != SCHEDULE_REPLAY_FORMAT:
        raise ValueError(
            "Recorded schedule replay format is unsupported: "
            f"{record.get('format')!r}."
        )
    if int(record.get("schema_version", 0) or 0) != SCHEDULE_REPLAY_SCHEMA_VERSION:
        raise ValueError(
            "Recorded schedule replay schema version is unsupported: "
            f"{record.get('schema_version')!r}."
        )

    full_record = dict(record.get("full_schedule") or {})
    active_record = dict(record.get("active_schedule") or {})
    full_sigmas = _tensor_from_payload(full_record.get("sigmas"), device=device)
    full_timesteps = _tensor_from_payload(full_record.get("timesteps"), device=device)
    active_sigmas = _tensor_from_payload(active_record.get("sigmas"), device=device)
    active_timesteps = _tensor_from_payload(active_record.get("timesteps"), device=device)
    if full_sigmas is None or active_sigmas is None:
        raise ValueError("Recorded schedule replay must contain full and active sigma tensors.")
    if full_timesteps is None or active_timesteps is None:
        raise ValueError("Recorded schedule replay must contain full and active timestep tensors.")

    scheduler_identifier = str(record.get("scheduler_identifier") or "")
    scheduler_configuration = dict(record.get("scheduler_configuration") or {})
    common_metadata = {
        "scheduler_name": scheduler_identifier,
        "validated_settings": scheduler_configuration,
        "recorded_schedule_rehydrated": True,
        "recorded_schedule_replay_format": SCHEDULE_REPLAY_FORMAT,
    }
    full_schedule = SchedulerOutput(
        sigmas=full_sigmas,
        timesteps=full_timesteps,
        requested_steps=int(full_record.get("requested_steps", full_sigmas.numel() - 1)),
        effective_steps=int(full_record.get("effective_steps", full_sigmas.numel() - 1)),
        scheduler_step_override_applied=bool(full_record.get("scheduler_step_override_applied", False)),
        compatibility_mode=full_record.get("compatibility_mode"),
        metadata=common_metadata,
    )

    step_plan = _step_plan_from_record(record)
    start_index = int(record.get("start_index", 0))
    start_sigma = float(record.get("start_sigma"))
    start_timestep_value = record.get("start_timestep")
    start_timestep = float(start_timestep_value) if start_timestep_value is not None else None
    active_metadata = {
        **common_metadata,
        "hires_second_pass": True,
        "hires_step_policy": step_plan.step_policy,
        "hires_requested_steps": int(step_plan.requested_refinement_steps),
        "hires_planned_internal_schedule_steps": int(step_plan.internal_schedule_steps),
        "hires_full_schedule_transition_count": int(full_schedule.sigma_transitions),
        "hires_effective_second_pass_transition_count": int(active_sigmas.numel() - 1),
        "hires_schedule_start_index": start_index,
        "hires_requested_denoising_strength": float(step_plan.requested_denoising_strength),
        "hires_denoising_strength": float(step_plan.normalized_denoising_strength),
        "hires_safe_denoising_strength": float(step_plan.safe_denoising_strength),
        "hires_denoising_strength_was_clamped": bool(step_plan.denoising_strength_was_clamped),
        "hires_starting_sigma": start_sigma,
        "hires_starting_timestep": start_timestep,
        "hires_schedule_counts_source": "recorded_schedule_replay_bytes",
        "image_conditioned_step_plan": step_plan.to_serializable_dict(),
    }
    active_schedule = SchedulerOutput(
        sigmas=active_sigmas,
        timesteps=active_timesteps,
        requested_steps=int(active_record.get("requested_steps", active_sigmas.numel() - 1)),
        effective_steps=int(active_record.get("effective_steps", active_sigmas.numel() - 1)),
        scheduler_step_override_applied=bool(active_record.get("scheduler_step_override_applied", False)),
        compatibility_mode=active_record.get("compatibility_mode"),
        metadata=active_metadata,
    )

    schedule = ImageConditionedSchedule(
        full_schedule=full_schedule,
        active_schedule=active_schedule,
        step_policy=step_plan.step_policy,
        requested_refinement_steps=int(record.get("requested_refinement_steps", step_plan.requested_refinement_steps)),
        internal_schedule_steps=int(record.get("internal_schedule_steps", full_schedule.sigma_transitions)),
        effective_refinement_steps=int(record.get("effective_refinement_steps", active_schedule.sigma_transitions)),
        denoising_strength=float(record.get("denoising_strength", step_plan.normalized_denoising_strength)),
        start_index=start_index,
        start_sigma=start_sigma,
        start_timestep=start_timestep,
        step_plan=step_plan,
    )

    if schedule.full_schedule.sigma_transitions != int(record.get("internal_schedule_steps", schedule.full_schedule.sigma_transitions)):
        raise ValueError("Recorded full schedule transition count does not match its replay contract.")
    if schedule.active_schedule.sigma_transitions != schedule.effective_refinement_steps:
        raise ValueError("Recorded active schedule transition count does not match its replay contract.")
    if abs(schedule.active_schedule.initial_sigma - schedule.start_sigma) > 1.0e-7:
        raise ValueError("Recorded active schedule start sigma does not match its replay contract.")
    if schedule.active_schedule.timesteps is not None and schedule.active_schedule.timesteps.numel() > 0:
        actual_start_timestep = float(schedule.active_schedule.timesteps[0].detach().cpu().item())
        if schedule.start_timestep is not None and abs(actual_start_timestep - schedule.start_timestep) > 1.0e-6:
            raise ValueError("Recorded active schedule start timestep does not match its replay contract.")

    actual_record = build_schedule_fingerprint_record(schedule, replay_record=record)
    if isinstance(expected_fingerprint, Mapping):
        expected_sha = str(expected_fingerprint.get("sha256") or "")
    else:
        expected_sha = str(expected_fingerprint or "")
    actual_sha = str(actual_record.get("sha256") or "")
    fingerprint_match = bool(expected_sha and expected_sha == actual_sha)
    if expected_sha and not fingerprint_match and strict_fingerprint:
        raise ValueError(
            "Recorded schedule fingerprint mismatch: expected "
            f"{expected_sha}, reconstructed {actual_sha}."
        )

    active_schedule.metadata["hires_schedule_replay"] = record
    active_schedule.metadata["hires_schedule_fingerprint"] = actual_record
    active_schedule.metadata["recorded_schedule_fingerprint_match"] = fingerprint_match
    return ScheduleRehydrationResult(
        schedule=schedule,
        expected_fingerprint=expected_sha,
        actual_fingerprint=actual_sha,
        fingerprint_match=fingerprint_match,
        replay_format=SCHEDULE_REPLAY_FORMAT,
    )

SCHEDULE_CONFORMANCE_SCHEMA_VERSION = 1
SCHEDULE_CONFORMANCE_FORMAT = "image-gen-schedule-conformance-v1"


def _tensor_conformance_difference(
    name: str,
    recorded: torch.Tensor | None,
    current: torch.Tensor | None,
    *,
    tolerance: float = 1.0e-7,
    max_examples: int = 12,
) -> dict[str, Any] | None:
    if recorded is None or current is None:
        if recorded is current:
            return None
        return {
            "category": "tensor_presence",
            "path": name,
            "recorded_present": recorded is not None,
            "current_present": current is not None,
        }
    recorded_cpu = recorded.detach().cpu().contiguous()
    current_cpu = current.detach().cpu().contiguous()
    if tuple(recorded_cpu.shape) != tuple(current_cpu.shape):
        return {
            "category": "tensor_shape",
            "path": name,
            "recorded_shape": [int(value) for value in recorded_cpu.shape],
            "current_shape": [int(value) for value in current_cpu.shape],
            "recorded_dtype": str(recorded_cpu.dtype),
            "current_dtype": str(current_cpu.dtype),
        }
    recorded_values = recorded_cpu.to(dtype=torch.float64).reshape(-1)
    current_values = current_cpu.to(dtype=torch.float64).reshape(-1)
    absolute = torch.abs(recorded_values - current_values)
    mismatch_indices = torch.nonzero(absolute > float(tolerance), as_tuple=False).reshape(-1)
    dtype_changed = recorded_cpu.dtype != current_cpu.dtype
    if int(mismatch_indices.numel()) == 0 and not dtype_changed:
        return None
    examples: list[dict[str, Any]] = []
    for index in mismatch_indices[:max_examples].tolist():
        examples.append(
            {
                "index": int(index),
                "recorded": float(recorded_values[index].item()),
                "current": float(current_values[index].item()),
                "absolute_difference": float(absolute[index].item()),
            }
        )
    return {
        "category": "tensor_values" if int(mismatch_indices.numel()) else "tensor_dtype",
        "path": name,
        "recorded_dtype": str(recorded_cpu.dtype),
        "current_dtype": str(current_cpu.dtype),
        "mismatch_count": int(mismatch_indices.numel()),
        "element_count": int(recorded_values.numel()),
        "max_absolute_difference": float(absolute.max().item()) if absolute.numel() else 0.0,
        "examples": examples,
    }


def _contract_conformance_differences(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fields = (
        "step_policy",
        "requested_refinement_steps",
        "planned_internal_schedule_steps",
        "internal_schedule_steps",
        "effective_refinement_steps",
        "requested_denoising_strength",
        "denoising_strength",
        "safe_denoising_strength",
        "start_index",
        "start_sigma",
        "start_timestep",
    )
    differences: list[dict[str, Any]] = []
    for field in fields:
        recorded_value = recorded.get(field)
        current_value = current.get(field)
        if recorded_value == current_value:
            continue
        if isinstance(recorded_value, (int, float)) and isinstance(current_value, (int, float)):
            if abs(float(recorded_value) - float(current_value)) <= 1.0e-9:
                continue
        differences.append(
            {
                "category": "contract",
                "path": f"contract.{field}",
                "recorded": recorded_value,
                "current": current_value,
            }
        )
    return differences


def compare_schedule_conformance(
    current_schedule: ImageConditionedSchedule,
    recorded_replay: Mapping[str, Any],
    *,
    recorded_fingerprint: Mapping[str, Any] | str | None = None,
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Compare a freshly reconstructed schedule with recorded replay tensors.

    This comparison performs no model sampling. It reports contract drift and
    precise full/active sigma or timestep differences so users can decide
    between exact recorded replay and current scheduler reconstruction.
    """

    rehydrated = rehydrate_schedule_replay_record(
        recorded_replay,
        expected_fingerprint=recorded_fingerprint,
        device="cpu",
        strict_fingerprint=False,
    )
    recorded_schedule = rehydrated.schedule
    current_record = build_schedule_replay_record(
        current_schedule,
        scheduler_identifier=str(recorded_replay.get("scheduler_identifier") or ""),
        scheduler_configuration=dict(recorded_replay.get("scheduler_configuration") or {}),
        sampler_name=str(recorded_replay.get("sampler_name") or ""),
        requires_terminal_zero=(dict(recorded_replay.get("terminal_zero") or {}).get("required")),
    )
    current_fingerprint = build_schedule_fingerprint_record(
        current_schedule,
        replay_record=current_record,
    )
    recorded_fingerprint_record = build_schedule_fingerprint_record(
        recorded_schedule,
        replay_record=recorded_replay,
    )

    differences = _contract_conformance_differences(
        dict(recorded_fingerprint_record.get("snapshot", {}).get("contract") or {}),
        dict(current_fingerprint.get("snapshot", {}).get("contract") or {}),
    )
    for path, recorded_tensor, current_tensor in (
        ("full_schedule.sigmas", recorded_schedule.full_schedule.sigmas, current_schedule.full_schedule.sigmas),
        ("full_schedule.timesteps", recorded_schedule.full_schedule.timesteps, current_schedule.full_schedule.timesteps),
        ("active_schedule.sigmas", recorded_schedule.active_schedule.sigmas, current_schedule.active_schedule.sigmas),
        ("active_schedule.timesteps", recorded_schedule.active_schedule.timesteps, current_schedule.active_schedule.timesteps),
    ):
        difference = _tensor_conformance_difference(
            path,
            recorded_tensor,
            current_tensor,
            tolerance=tolerance,
        )
        if difference is not None:
            differences.append(difference)

    categories = sorted({str(item.get("category") or "unknown") for item in differences})
    recorded_sha = str(recorded_fingerprint_record.get("sha256") or "")
    current_sha = str(current_fingerprint.get("sha256") or "")
    expected_sha = (
        str(recorded_fingerprint.get("sha256") or "")
        if isinstance(recorded_fingerprint, Mapping)
        else str(recorded_fingerprint or "")
    )
    expected_valid = bool(not expected_sha or expected_sha == recorded_sha)
    return {
        "schema_version": SCHEDULE_CONFORMANCE_SCHEMA_VERSION,
        "format": SCHEDULE_CONFORMANCE_FORMAT,
        "matches": not differences and recorded_sha == current_sha and expected_valid,
        "difference_count": len(differences),
        "difference_categories": categories,
        "recorded_fingerprint": recorded_sha,
        "expected_recorded_fingerprint": expected_sha,
        "expected_recorded_fingerprint_valid": expected_valid,
        "current_fingerprint": current_sha,
        "tolerance": float(tolerance),
        "differences": differences,
        "comparison_mode": "reconstructed_without_sampling",
    }
