from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

PROMPT_CFG_CONTRACT_VERSION = "image-gen-prompt-cfg-v2"
PROMPT_CFG_SOURCE = "superhybrid_prompt"
PROMPT_CFG_MAX = 30.0
PROMPT_CFG_MIN = 0.0
PROMPT_CFG_BEHAVIORS = {"replace_ui", "shape_ui", "disabled"}
PROMPT_CFG_INTERPOLATIONS = {"linear", "smoothstep", "cosine", "exp_decay"}

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CONTROL_POINT_RE = re.compile(rf"^\s*({_NUMBER})(?:\s*@\s*({_NUMBER}))?\s*$")


class PromptCFGScheduleError(ValueError):
    pass


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PromptCFGScheduleError(f"{label} must be a number, got {value!r}.") from exc
    if not math.isfinite(result):
        raise PromptCFGScheduleError(f"{label} must be finite, got {value!r}.")
    if result < PROMPT_CFG_MIN or result > PROMPT_CFG_MAX:
        raise PromptCFGScheduleError(
            f"{label} must be between {PROMPT_CFG_MIN:g} and {PROMPT_CFG_MAX:g}, got {result:g}."
        )
    return result


def _finite_fraction(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PromptCFGScheduleError(f"{label} must be a number, got {value!r}.") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise PromptCFGScheduleError(f"{label} must be between 0 and 1, got {value!r}.")
    return result


def normalize_prompt_cfg_behavior(value: Any) -> str:
    token = str(value or "replace_ui").strip().lower().replace("-", "_")
    aliases = {
        "replace": "replace_ui",
        "override": "replace_ui",
        "shape": "shape_ui",
        "shape_to_ui": "shape_ui",
        "use_shape": "shape_ui",
        "off": "disabled",
        "ignore": "disabled",
    }
    token = aliases.get(token, token)
    if token not in PROMPT_CFG_BEHAVIORS:
        supported = ", ".join(sorted(PROMPT_CFG_BEHAVIORS))
        raise PromptCFGScheduleError(f"prompt_cfg_behavior must be one of: {supported}.")
    return token


def normalize_prompt_cfg_interpolation(value: Any) -> str:
    token = str(value or "linear").strip().lower().replace("-", "_")
    aliases = {
        "piecewise_linear": "linear",
        "smooth": "smoothstep",
        "ease": "smoothstep",
        "ease_in_out": "smoothstep",
        "exponential": "exp_decay",
        "exp": "exp_decay",
    }
    token = aliases.get(token, token)
    if token not in PROMPT_CFG_INTERPOLATIONS:
        supported = ", ".join(sorted(PROMPT_CFG_INTERPOLATIONS))
        raise PromptCFGScheduleError(
            f"SuperHybrid CFG interpolation must be one of: {supported}."
        )
    return token


def _resolved_positions(explicit: Sequence[float | None]) -> list[float]:
    count = len(explicit)
    if count == 0:
        raise PromptCFGScheduleError("Prompt CFG schedule has no control points.")
    if count == 1:
        position = explicit[0]
        if position not in (None, 0.0, 1.0):
            raise PromptCFGScheduleError(
                "A single SuperHybrid CFG control point cannot use an intermediate @ position."
            )
        return [0.0]

    anchors: dict[int, float] = {0: 0.0, count - 1: 1.0}
    for index, raw_position in enumerate(explicit):
        if raw_position is None:
            continue
        position = _finite_fraction(raw_position, label=f"SuperHybrid CFG position {index + 1}")
        if index == 0 and position != 0.0:
            raise PromptCFGScheduleError("The first SuperHybrid CFG control point must be at @0.")
        if index == count - 1 and position != 1.0:
            raise PromptCFGScheduleError("The last SuperHybrid CFG control point must be at @1.")
        anchors[index] = position

    ordered = sorted(anchors.items())
    for (left_index, left_position), (right_index, right_position) in zip(ordered, ordered[1:]):
        if right_position <= left_position:
            raise PromptCFGScheduleError(
                "SuperHybrid CFG @ positions must increase from left to right."
            )
        if right_index <= left_index:
            raise PromptCFGScheduleError("SuperHybrid CFG control-point ordering is invalid.")

    positions = [0.0] * count
    for anchor_index, anchor_position in ordered:
        positions[anchor_index] = anchor_position
    for (left_index, left_position), (right_index, right_position) in zip(ordered, ordered[1:]):
        span = right_index - left_index
        for offset in range(1, span):
            positions[left_index + offset] = left_position + (
                (right_position - left_position) * (offset / float(span))
            )

    for left, right in zip(positions, positions[1:]):
        if right <= left:
            raise PromptCFGScheduleError(
                "SuperHybrid CFG control-point positions must be strictly increasing."
            )
    return positions


def parse_prompt_cfg_spec(raw_value: str) -> dict[str, Any]:
    text = str(raw_value or "").strip()
    if not text:
        raise PromptCFGScheduleError("SuperHybrid CFG directive is empty.")

    interpolation = "linear"
    body = text
    if ":" in text:
        candidate_body, candidate_interpolation = text.rsplit(":", 1)
        candidate = candidate_interpolation.strip().lower().replace("-", "_")
        if candidate in PROMPT_CFG_INTERPOLATIONS or candidate in {
            "piecewise_linear", "smooth", "ease", "ease_in_out", "exponential", "exp"
        }:
            interpolation = normalize_prompt_cfg_interpolation(candidate)
            body = candidate_body.strip()
        else:
            raise PromptCFGScheduleError(
                f"Unknown SuperHybrid CFG interpolation {candidate_interpolation.strip()!r}."
            )

    raw_parts = [part.strip() for part in body.split("->")]
    if any(not part for part in raw_parts):
        raise PromptCFGScheduleError(
            "SuperHybrid CFG schedule contains an empty control point."
        )
    if len(raw_parts) > 64:
        raise PromptCFGScheduleError(
            "SuperHybrid CFG schedule supports at most 64 control points."
        )

    values: list[float] = []
    explicit_positions: list[float | None] = []
    for index, part in enumerate(raw_parts):
        match = _CONTROL_POINT_RE.fullmatch(part)
        if match is None:
            raise PromptCFGScheduleError(
                "Invalid SuperHybrid CFG control point. Use values such as 8, 6@0.4, or 3."
            )
        values.append(
            _finite_float(match.group(1), label=f"SuperHybrid CFG control point {index + 1}")
        )
        explicit_positions.append(
            None
            if match.group(2) is None
            else _finite_fraction(
                match.group(2), label=f"SuperHybrid CFG position {index + 1}"
            )
        )

    positions = _resolved_positions(explicit_positions)
    return {
        "control_points": values,
        "control_point_positions": positions,
        "interpolation": interpolation,
    }


def parse_prompt_cfg_control_points(raw_value: str) -> list[float]:
    return list(parse_prompt_cfg_spec(raw_value)["control_points"])


def _curve_weight(value: float, interpolation: str) -> float:
    x = max(0.0, min(1.0, float(value)))
    curve = normalize_prompt_cfg_interpolation(interpolation)
    if curve == "linear":
        return x
    if curve == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return (1.0 - math.exp(-4.0 * x)) / (1.0 - math.exp(-4.0))


def _interpolate_control_points(
    control_points: Sequence[float],
    progress: float,
    *,
    positions: Sequence[float] | None = None,
    interpolation: str = "linear",
) -> float:
    values = [float(value) for value in control_points]
    if not values:
        raise PromptCFGScheduleError("Prompt CFG schedule has no control points.")
    if len(values) == 1:
        return values[0]

    if positions is None:
        point_positions = [index / float(len(values) - 1) for index in range(len(values))]
    else:
        point_positions = [float(value) for value in positions]
        if len(point_positions) != len(values):
            raise PromptCFGScheduleError(
                "Prompt CFG control-point positions must match the control-point count."
            )

    current = max(0.0, min(1.0, float(progress)))
    if current <= point_positions[0]:
        return values[0]
    if current >= point_positions[-1]:
        return values[-1]

    segment = 0
    for index in range(len(point_positions) - 1):
        if point_positions[index] <= current <= point_positions[index + 1]:
            segment = index
            break
    left_position = point_positions[segment]
    right_position = point_positions[segment + 1]
    span = right_position - left_position
    if span <= 0.0:
        raise PromptCFGScheduleError("Prompt CFG control-point positions are invalid.")
    local = _curve_weight((current - left_position) / span, interpolation)
    start = values[segment]
    end = values[segment + 1]
    return start + (end - start) * local


def materialize_prompt_cfg_schedule(
    control_points: Sequence[float],
    total_steps: int,
    *,
    positions: Sequence[float] | None = None,
    interpolation: str = "linear",
) -> list[float]:
    steps = max(1, int(total_steps))
    if steps == 1:
        return [
            _interpolate_control_points(
                control_points,
                0.0,
                positions=positions,
                interpolation=interpolation,
            )
        ]
    return [
        _interpolate_control_points(
            control_points,
            index / float(steps - 1),
            positions=positions,
            interpolation=interpolation,
        )
        for index in range(steps)
    ]


def _schedule_fingerprint_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": str(payload.get("contract_version") or PROMPT_CFG_CONTRACT_VERSION),
        "parser_id": str(payload.get("parser_id") or ""),
        "parser_version": str(payload.get("parser_version") or ""),
        "behavior": str(payload.get("behavior") or "disabled"),
        "interpolation": str(payload.get("interpolation") or "linear"),
        "control_points": [float(value) for value in payload.get("control_points") or []],
        "control_point_positions": [
            float(value) for value in payload.get("control_point_positions") or []
        ],
        "requested_steps": int(payload.get("requested_steps") or 0),
        "ui_cfg_scale": (
            None if payload.get("ui_cfg_scale") is None else float(payload.get("ui_cfg_scale"))
        ),
        "requested_schedule": [
            float(value) for value in payload.get("requested_schedule") or []
        ],
        "pass": str(payload.get("pass") or ""),
    }


def build_prompt_cfg_schedule_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint_source = _schedule_fingerprint_source(payload)
    digest = hashlib.sha256(
        json.dumps(
            fingerprint_source,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "digest": digest,
        "source": "materialized_prompt_cfg_schedule",
        "contract_version": PROMPT_CFG_CONTRACT_VERSION,
    }


def finalize_prompt_cfg_payload(
    payload: Mapping[str, Any],
    *,
    ui_cfg_scale: float,
    total_steps: int | None = None,
    pass_name: str = "",
) -> dict[str, Any]:
    output = dict(payload or {})
    behavior = normalize_prompt_cfg_behavior(output.get("behavior") or "replace_ui")
    ui_cfg = _finite_float(ui_cfg_scale, label="UI CFG scale")
    control_points = [
        _finite_float(value, label="Prompt CFG control point")
        for value in output.get("control_points") or []
    ]
    if not control_points:
        raise PromptCFGScheduleError("Prompt CFG schedule has no control points.")
    positions = output.get("control_point_positions")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        positions = None
    resolved_positions = (
        [index / float(len(control_points) - 1) for index in range(len(control_points))]
        if positions is None and len(control_points) > 1
        else ([0.0] if positions is None else [float(value) for value in positions])
    )
    interpolation = normalize_prompt_cfg_interpolation(output.get("interpolation") or "linear")
    steps = max(1, int(total_steps if total_steps is not None else output.get("requested_steps") or 1))
    prompt_shape = materialize_prompt_cfg_schedule(
        control_points,
        steps,
        positions=resolved_positions,
        interpolation=interpolation,
    )

    if behavior == "shape_ui":
        anchor = prompt_shape[0]
        if abs(anchor) <= 1e-12:
            raise PromptCFGScheduleError(
                "shape_ui requires a non-zero first SuperHybrid CFG control point."
            )
        requested_schedule = [ui_cfg * (value / anchor) for value in prompt_shape]
        requested_schedule = [
            _finite_float(value, label="Prompt CFG shape-adjusted value")
            for value in requested_schedule
        ]
    elif behavior == "disabled":
        requested_schedule = [ui_cfg for _ in range(steps)]
    else:
        requested_schedule = list(prompt_shape)

    output.update(
        {
            "contract_version": PROMPT_CFG_CONTRACT_VERSION,
            "source": str(output.get("source") or PROMPT_CFG_SOURCE),
            "behavior": behavior,
            "enabled": behavior != "disabled",
            "interpolation": interpolation,
            "control_points": control_points,
            "control_point_positions": resolved_positions,
            "requested_steps": steps,
            "prompt_shape_schedule": [float(value) for value in prompt_shape],
            "requested_schedule": [float(value) for value in requested_schedule],
            "ui_cfg_scale": float(ui_cfg),
            "minimum": min(requested_schedule),
            "maximum": max(requested_schedule),
            "start": requested_schedule[0],
            "end": requested_schedule[-1],
            "pass": str(pass_name or output.get("pass") or ""),
        }
    )
    output["schedule_fingerprint"] = build_prompt_cfg_schedule_fingerprint(output)
    return output


def build_prompt_cfg_payload(
    raw_value: str,
    *,
    total_steps: int,
    parser_id: str,
    parser_version: str,
    behavior: str = "replace_ui",
    raw_directive: str = "",
) -> dict[str, Any]:
    normalized_behavior = normalize_prompt_cfg_behavior(behavior)
    spec = parse_prompt_cfg_spec(raw_value)
    requested_steps = max(1, int(total_steps))
    prompt_shape = materialize_prompt_cfg_schedule(
        spec["control_points"],
        requested_steps,
        positions=spec["control_point_positions"],
        interpolation=spec["interpolation"],
    )
    payload = {
        "contract_version": PROMPT_CFG_CONTRACT_VERSION,
        "parser_id": str(parser_id),
        "parser_version": str(parser_version),
        "behavior": normalized_behavior,
        "interpolation": spec["interpolation"],
        "control_points": [float(value) for value in spec["control_points"]],
        "control_point_positions": [
            float(value) for value in spec["control_point_positions"]
        ],
        "requested_steps": requested_steps,
        "source": PROMPT_CFG_SOURCE,
        "enabled": normalized_behavior != "disabled",
        "raw_directive": str(raw_directive or ""),
        "raw_value": str(raw_value),
        "prompt_shape_schedule": [float(value) for value in prompt_shape],
        "requested_schedule": [float(value) for value in prompt_shape],
        "minimum": min(prompt_shape),
        "maximum": max(prompt_shape),
        "start": prompt_shape[0],
        "end": prompt_shape[-1],
        "ui_cfg_scale": None,
        "pass": "",
        "replay_locked": False,
    }
    payload["schedule_fingerprint"] = build_prompt_cfg_schedule_fingerprint(payload)
    return payload


def validate_recorded_prompt_cfg_payload(
    payload: Mapping[str, Any],
    *,
    total_steps: int,
    pass_name: str,
) -> dict[str, Any]:
    output = dict(payload or {})
    schedule = output.get("requested_schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise PromptCFGScheduleError("Recorded prompt CFG schedule is missing requested_schedule.")
    validated = [
        _finite_float(value, label="Recorded prompt CFG value") for value in schedule
    ]
    expected_steps = max(1, int(total_steps))
    if len(validated) != expected_steps:
        raise PromptCFGScheduleError(
            "Recorded prompt CFG schedule length does not match the active logical step count: "
            f"{len(validated)} != {expected_steps}."
        )
    fingerprint = output.get("schedule_fingerprint")
    if not isinstance(fingerprint, Mapping) or not fingerprint.get("digest"):
        raise PromptCFGScheduleError("Recorded prompt CFG schedule has no fingerprint.")
    expected = build_prompt_cfg_schedule_fingerprint(output)
    if str(fingerprint.get("digest")) != str(expected.get("digest")):
        raise PromptCFGScheduleError(
            "Recorded prompt CFG schedule fingerprint does not match its materialized values."
        )
    recorded_pass = str(output.get("pass") or pass_name or "")
    if pass_name and recorded_pass and recorded_pass != pass_name:
        raise PromptCFGScheduleError(
            f"Recorded prompt CFG pass {recorded_pass!r} does not match active pass {pass_name!r}."
        )
    output.update(
        {
            "contract_version": str(output.get("contract_version") or PROMPT_CFG_CONTRACT_VERSION),
            "requested_schedule": validated,
            "requested_steps": expected_steps,
            "pass": str(pass_name or recorded_pass),
            "replay_locked": True,
            "replay_source": "recorded_exact",
        }
    )
    return output


def prompt_cfg_payload_from_request(request: Any) -> dict[str, Any]:
    payload = getattr(request, "prompt_cfg_schedule", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = getattr(request, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        nested = diagnostics.get("prompt_cfg_schedule")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def requested_cfg_scale_for_step(
    request: Any,
    *,
    step_index: int,
    total_steps: int,
) -> tuple[float, dict[str, Any]]:
    ui_cfg = _finite_float(getattr(request, "cfg_scale", 1.0), label="UI CFG scale")
    payload = prompt_cfg_payload_from_request(request)
    enabled = bool(payload.get("enabled", False))
    behavior = normalize_prompt_cfg_behavior(payload.get("behavior") or "disabled")
    if not enabled or behavior == "disabled":
        return ui_cfg, {
            "cfg_source": "ui",
            "ui_cfg_scale": ui_cfg,
            "prompt_cfg_applied": False,
            "prompt_cfg_behavior": behavior,
            "prompt_cfg_contract_version": payload.get("contract_version"),
            "prompt_cfg_replay_locked": bool(payload.get("replay_locked", False)),
        }

    steps = max(1, int(total_steps))
    index = max(0, min(int(step_index), steps - 1))
    progress = 0.0 if steps <= 1 else index / float(steps - 1)

    recorded_schedule = payload.get("requested_schedule")
    replay_locked = bool(payload.get("replay_locked", False))
    if replay_locked:
        if not isinstance(recorded_schedule, Sequence) or isinstance(
            recorded_schedule, (str, bytes)
        ):
            raise PromptCFGScheduleError(
                "Replay-locked prompt CFG payload is missing requested_schedule."
            )
        if len(recorded_schedule) != steps:
            raise PromptCFGScheduleError(
                "Replay-locked prompt CFG schedule length does not match the active logical steps."
            )
        requested = _finite_float(
            recorded_schedule[index], label="Replay-locked prompt CFG value"
        )
    else:
        control_points = payload.get("control_points")
        if (
            not isinstance(control_points, Sequence)
            or isinstance(control_points, (str, bytes))
            or not control_points
        ):
            return ui_cfg, {
                "cfg_source": "ui",
                "ui_cfg_scale": ui_cfg,
                "prompt_cfg_applied": False,
                "prompt_cfg_behavior": behavior,
                "prompt_cfg_contract_version": payload.get("contract_version"),
                "prompt_cfg_replay_locked": False,
            }
        validated_control_points = [
            _finite_float(value, label="Prompt CFG control point")
            for value in control_points
        ]
        positions = payload.get("control_point_positions")
        validated_positions = (
            [float(value) for value in positions]
            if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes))
            else None
        )
        interpolation = normalize_prompt_cfg_interpolation(
            payload.get("interpolation") or "linear"
        )
        prompt_value = _interpolate_control_points(
            validated_control_points,
            progress,
            positions=validated_positions,
            interpolation=interpolation,
        )
        if behavior == "shape_ui":
            anchor = validated_control_points[0]
            if abs(anchor) <= 1e-12:
                raise PromptCFGScheduleError(
                    "shape_ui requires a non-zero first SuperHybrid CFG control point."
                )
            requested = _finite_float(
                ui_cfg * (prompt_value / anchor),
                label="Prompt CFG shape-adjusted value",
            )
        else:
            requested = prompt_value

    return requested, {
        "cfg_source": (
            "superhybrid_prompt_replay"
            if replay_locked
            else str(payload.get("source") or PROMPT_CFG_SOURCE)
        ),
        "ui_cfg_scale": ui_cfg,
        "prompt_cfg_applied": True,
        "prompt_cfg_behavior": behavior,
        "prompt_cfg_contract_version": str(
            payload.get("contract_version") or PROMPT_CFG_CONTRACT_VERSION
        ),
        "prompt_cfg_parser_id": str(payload.get("parser_id") or ""),
        "prompt_cfg_parser_version": str(payload.get("parser_version") or ""),
        "prompt_cfg_control_points": list(payload.get("control_points") or []),
        "prompt_cfg_control_point_positions": list(
            payload.get("control_point_positions") or []
        ),
        "prompt_cfg_interpolation": str(payload.get("interpolation") or "linear"),
        "prompt_cfg_progress_fraction": float(progress),
        "prompt_cfg_replay_locked": replay_locked,
        "prompt_cfg_schedule_fingerprint": dict(
            payload.get("schedule_fingerprint") or {}
        ),
        "requested_cfg_scale": float(requested),
    }
