from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from modules.contracts import SchedulerOutput
from modules.project_context import ProjectContext
from modules.sd3_runtime_assets import SD3RuntimeAssetResolver
from modules.sd3_runtime_profile import profile_from_id


SCHEDULE_DOMAIN = "flow_match"
DEFAULT_RUNTIME_PROFILE = "sd3-medium"
EXPECTED_SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read Flow Match scheduler config from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Flow Match scheduler config must contain a JSON object: {path}")
    return payload


def _validate_flow_match_config(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    config = dict(payload)
    scheduler_class = str(config.get("_class_name") or EXPECTED_SCHEDULER_CLASS).strip()
    if scheduler_class != EXPECTED_SCHEDULER_CLASS:
        raise ValueError(
            "Flow Match Euler requires a FlowMatchEulerDiscreteScheduler runtime config; "
            f"{source} declares {scheduler_class!r}."
        )
    num_train_timesteps = int(config.get("num_train_timesteps", 1000))
    shift = float(config.get("shift", 1.0))
    if num_train_timesteps < 2:
        raise ValueError("Flow Match Euler num_train_timesteps must be at least 2.")
    if not torch.isfinite(torch.tensor(shift)) or shift <= 0.0:
        raise ValueError("Flow Match Euler shift must be a finite value greater than zero.")
    config["num_train_timesteps"] = num_train_timesteps
    config["shift"] = shift
    return config


class FlowMatchEulerSchedulerAdapter:
    """IMAGE_GEN adapter over the installed Diffusers Flow Match Euler scheduler.

    Schedule construction is intentionally delegated to the installed Diffusers
    implementation. IMAGE_GEN owns only local runtime-config resolution, plugin
    capability metadata, canonical ``SchedulerOutput`` normalization, and state
    handoff for the later SD3 denoising phase.
    """

    def __init__(self, state: Any = None, default_name: str = "flow_match_euler") -> None:
        self.state = state
        self.default_name = default_name
        self._last_runtime_scheduler: Any | None = None

    @staticmethod
    def _device(request: Any, state: Any = None) -> torch.device:
        requested = getattr(request, "device", None)
        if requested is not None:
            return torch.device(requested)
        if state is not None and getattr(state, "d", None) is not None:
            candidate = getattr(state.d, "device", None)
            if candidate is not None:
                return torch.device(candidate)
        # Schedule construction is lightweight and does not need CUDA. Keeping it
        # on CPU also avoids turning this phase into a GPU-residency qualification.
        return torch.device("cpu")

    @staticmethod
    def _reference_scheduler_class():
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError as exc:
            raise RuntimeError(
                "Flow Match Euler requires the installed Diffusers package. "
                "Run the normal IMAGE_GEN dependency installer before using this scheduler."
            ) from exc
        return FlowMatchEulerDiscreteScheduler

    @staticmethod
    def _default_config_path() -> Path:
        context = ProjectContext.load()
        profile = profile_from_id(DEFAULT_RUNTIME_PROFILE)
        if profile is None:  # pragma: no cover - guarded by the built-in profile table.
            raise RuntimeError(f"Missing built-in SD3 runtime profile: {DEFAULT_RUNTIME_PROFILE}")
        return SD3RuntimeAssetResolver(context).resolve(profile).scheduler_config

    def _load_config(self, request: Any) -> tuple[dict[str, Any], str]:
        kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
        # Compatibility negotiation belongs to the sampler/scheduler registry and
        # is metadata, not a Diffusers FlowMatch constructor argument.
        kwargs.pop("pipeline_mode", None)
        kwargs.pop("compatibility", None)

        explicit_path = str(kwargs.pop("scheduler_config_path", "") or "").strip()
        config_path = Path(explicit_path).expanduser().resolve() if explicit_path else self._default_config_path()
        config = _validate_flow_match_config(_load_json_object(config_path), source=str(config_path))

        # Optional explicit overrides are experiments at the scheduler boundary;
        # model profiles never force or forbid them.
        if "num_train_timesteps" in kwargs:
            value = kwargs.pop("num_train_timesteps")
            if value is not None:
                config["num_train_timesteps"] = int(value)
        if "shift" in kwargs:
            value = kwargs.pop("shift")
            if value is not None:
                config["shift"] = float(value)
        config = _validate_flow_match_config(config, source=str(config_path))

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown flow_match_euler scheduler setting(s): {unknown}.")
        return config, str(config_path)

    @staticmethod
    def _construct_reference_scheduler(config: Mapping[str, Any]):
        scheduler_class = FlowMatchEulerSchedulerAdapter._reference_scheduler_class()
        if hasattr(scheduler_class, "from_config"):
            return scheduler_class.from_config(dict(config))
        constructor = {
            key: value
            for key, value in dict(config).items()
            if not str(key).startswith("_")
        }
        return scheduler_class(**constructor)

    @staticmethod
    def _normalise_schedule(runtime_scheduler: Any, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        sigmas = torch.as_tensor(runtime_scheduler.sigmas, dtype=torch.float32, device=device).flatten()
        timesteps = torch.as_tensor(runtime_scheduler.timesteps, dtype=torch.float32, device=device).flatten()
        if sigmas.numel() != timesteps.numel() + 1:
            raise RuntimeError(
                "Diffusers Flow Match Euler returned an unexpected schedule shape: "
                f"sigmas={sigmas.numel()}, timesteps={timesteps.numel()}."
            )
        return sigmas, timesteps

    def build_schedule(self, request: Any, state: Any = None) -> SchedulerOutput:
        active_state = state if state is not None else self.state
        steps = int(getattr(request, "steps", 0) or 0)
        if steps < 1:
            raise ValueError("Flow Match Euler requires steps >= 1.")

        device = self._device(request, active_state)
        config, config_path = self._load_config(request)
        runtime_scheduler = self._construct_reference_scheduler(config)
        runtime_scheduler.set_timesteps(steps, device=device)
        sigmas, timesteps = self._normalise_schedule(runtime_scheduler, device=device)

        output = SchedulerOutput(
            sigmas=sigmas,
            timesteps=timesteps,
            requested_steps=steps,
            effective_steps=steps,
            scheduler_step_override_applied=False,
            compatibility_mode="fixed_steps",
            metadata={
                "scheduler_name": "flow_match_euler",
                "scheduler_family": "flow_match_euler",
                "schedule_domain": SCHEDULE_DOMAIN,
                "scheduler_config_path": config_path,
                "reference_class": f"{runtime_scheduler.__class__.__module__}.{runtime_scheduler.__class__.__name__}",
                "num_train_timesteps": int(config["num_train_timesteps"]),
                "shift": float(config["shift"]),
                "terminal_sigma": float(sigmas[-1].detach().cpu().item()),
                "runtime_scheduler_available": True,
            },
        )
        self._last_runtime_scheduler = runtime_scheduler
        # Keep the qualified scheduler runtime attached to the in-memory schedule.
        # This is intentionally not part of serialized metadata; SD3-08 consumes
        # it directly for Flow Euler stepping, while replay may reconstruct it.
        setattr(output, "_runtime_scheduler", runtime_scheduler)

        if active_state is not None and hasattr(active_state, "sched"):
            active_state.sched.sigmas = output.sigmas
            active_state.sched.timesteps = output.timesteps
            active_state.sched.scheduler_name = "flow_match_euler"
            active_state.sched.selected_scheduler_name = "flow_match_euler"
            if hasattr(active_state.sched, "requested_steps"):
                active_state.sched.requested_steps = output.requested_steps
            if hasattr(active_state.sched, "effective_steps"):
                active_state.sched.effective_steps = output.effective_steps
            if hasattr(active_state.sched, "schedule_extra"):
                active_state.sched.schedule_extra = dict(output.extra)
            if hasattr(active_state.sched, "compatibility_mode"):
                active_state.sched.compatibility_mode = output.compatibility_mode
        if active_state is not None and hasattr(active_state, "extra") and isinstance(active_state.extra, dict):
            # Phase 8 can consume this exact local scheduler object for reference
            # Flow Euler stepping without reconstructing or downloading anything.
            active_state.extra["_scheduler_runtime_obj"] = runtime_scheduler

        return output


SCHEDULER_ADAPTER_CLASS = FlowMatchEulerSchedulerAdapter

meta = {
    "name": "flow_match_euler",
    "label": "Flow Match Euler",
    "summary_text": "Flow-matching Euler schedule backed by the installed Diffusers reference implementation.",
}

PLUGIN_DESCRIPTOR = {
    "plugin_id": "scheduler.flow_match_euler",
    "kind": "scheduler",
    "name": "flow_match_euler",
    "label": "Flow Match Euler",
    "description": meta["summary_text"],
    "version": "1",
    "module": __name__,
    "adapter_class": "FlowMatchEulerSchedulerAdapter",
    "aliases": ["flow match euler", "flowmatch euler", "sd3 flow euler"],
    "capabilities": {
        "pipeline_modes": ["fixed_steps", "compatible"],
        "supports_fixed_steps": True,
        "supports_step_expansion": False,
        "supports_tail_metadata": False,
        "supports_tail_steps": False,
        "supports_decay_tail": False,
        "supports_blended_tail": False,
        "supports_progressive_decay": False,
        "scheduler_family": "flow_match_euler",
        "schedule_domain": SCHEDULE_DOMAIN,
    },
    "config_schema": {
        "type": "object",
        "properties": {
            "scheduler_config_path": {"type": "string", "default": ""},
            "num_train_timesteps": {"type": ["integer", "null"], "default": None, "minimum": 2},
            "shift": {"type": ["number", "null"], "default": None, "exclusiveMinimum": 0},
        },
        "required": [],
        "additionalProperties": False,
    },
}


__all__ = [
    "FlowMatchEulerSchedulerAdapter",
    "SCHEDULER_ADAPTER_CLASS",
    "PLUGIN_DESCRIPTOR",
    "SCHEDULE_DOMAIN",
    "meta",
]
