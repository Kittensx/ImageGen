from __future__ import annotations

from typing import Any

import torch

from modules.contracts import SchedulerOutput
from modules.pipeline.schedule_compat import ensure_schedule_for_sampler
from modules.ss_registry.schedulers.scheduler_config_loader import prepare_scheduler_config
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes import SimpleKEScheduler
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import (
    KES_RUNTIME_DEFAULTS,
    validate_simple_kes_settings,
)


class SimpleKESSchedulerAdapter:
    """
    Pipeline-facing adapter for Simple KES.

    Responsibilities:
    - sync request into shared state
    - resolve scheduler settings before scheduler construction
    - instantiate SimpleKEScheduler with validated settings
    - call scheduler.build_schedule(...)
    - map KESScheduleResult into pipeline SchedulerOutput
    """

    def __init__(self, state=None, default_name: str = "simple_kes"):
        self.state = state
        self.default_name = default_name

    def build_schedule(self, request, state: Any = None) -> SchedulerOutput:
        active_state = state or self.state
        if active_state is None:
            raise ValueError("SimpleKESSchedulerAdapter requires a shared state object.")

        scheduler_name = request.scheduler_name or self.default_name
        scheduler_kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})

        # The generation request owns these canonical controls. Schema-driven
        # clients may still submit legacy profiles that contain them, so remove
        # them before forwarding plugin kwargs and avoid duplicate keyword
        # arguments such as build_schedule(steps=..., **{"steps": ...}).
        scheduler_kwargs.pop("steps", None)
        scheduler_kwargs.pop("device", None)

        self._sync_request_to_state(active_state, request)

        device = request.device or getattr(active_state.d, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        active_state.d.device = device

        config_path = scheduler_kwargs.pop("config_path", None)
        preset_name = scheduler_kwargs.pop("preset_name", None)
        pipeline_mode = scheduler_kwargs.pop("pipeline_mode", None)

        settings = prepare_scheduler_config(
            scheduler_name=scheduler_name,
            shared_state=active_state,
            config_path=config_path,
            preset_name=preset_name,
            overrides=scheduler_kwargs,
            extra_defaults=KES_RUNTIME_DEFAULTS,
            apply_to_state=True,
        )
        settings["steps"] = int(request.steps)
        settings = validate_simple_kes_settings(settings, pipeline_mode=pipeline_mode)
        settings["steps"] = int(request.steps)
        settings["device"] = str(device)

        scheduler = SimpleKEScheduler(
            shared_state=active_state,
            settings=settings,
            scheduler_name=scheduler_name,
            steps=request.steps,
            device=device,
            sigma_min=scheduler_kwargs.get("sigma_min"),
            sigma_max=scheduler_kwargs.get("sigma_max"),
            rho=scheduler_kwargs.get("rho"),
            decay_pattern=settings.get("decay_pattern"),
            decay_mode=settings.get("decay_mode"),
            tail_steps=settings.get("tail_steps"),
            verbose=scheduler_kwargs.get("verbose", False),
        )

        runtime_overrides = {
            k: v
            for k, v in scheduler_kwargs.items()
            if k
            not in {
                "sigma_min",
                "sigma_max",
                "rho",
                "decay_pattern",
                "decay_mode",
                "tail_steps",
                "verbose",
                "blend_methods",
                "blend_weights",
                "steps",
                "device",
            }
        }

        schedule_result = scheduler.build_schedule(
            steps=request.steps,
            device=device,
            sigma_min=scheduler_kwargs.get("sigma_min"),
            sigma_max=scheduler_kwargs.get("sigma_max"),
            rho=scheduler_kwargs.get("rho"),
            decay_pattern=scheduler_kwargs.get("decay_pattern"),
            # Blend configuration is already present in the validated settings
            # passed to the scheduler constructor. Do not forward the canonical
            # mapping through the legacy list-shaped runtime parameters.
            **runtime_overrides,
        )

        compatibility_mode = None
        if isinstance(schedule_result.extra, dict):
            compatibility_mode = schedule_result.extra.get("compatibility_mode")

        schedule_metadata = {
            "scheduler_name": scheduler_name,
            "schedule_mode": schedule_result.schedule_mode,
            "active_blend_methods": schedule_result.active_blend_methods,
            "active_blend_weights": schedule_result.active_blend_weights,
            "tail_features_used": schedule_result.tail_features_used,
            "prepass_used": schedule_result.prepass_used,
            "predicted_stop_step": schedule_result.predicted_stop_step,
            "validated_settings": dict(settings),
            **(schedule_result.extra or {}),
        }
        merged_schedule_extra = {
            **schedule_metadata,
            "requested_steps": schedule_result.requested_steps,
            "effective_steps": schedule_result.effective_steps,
            "scheduler_step_override_applied": schedule_result.step_count_changed,
            "compatibility_mode": compatibility_mode,
        }

        active_state.sched.sigmas = schedule_result.sigmas
        active_state.sched.scheduler_name = scheduler_name
        active_state.sched.selected_scheduler_name = scheduler_name
        active_state.sched.scheduler_fn = scheduler

        if hasattr(active_state.sched, "requested_steps"):
            active_state.sched.requested_steps = schedule_result.requested_steps
        if hasattr(active_state.sched, "effective_steps"):
            active_state.sched.effective_steps = schedule_result.effective_steps
        if hasattr(active_state.sched, "schedule_extra"):
            active_state.sched.schedule_extra = dict(merged_schedule_extra)
        if hasattr(active_state.sched, "compatibility_mode"):
            active_state.sched.compatibility_mode = compatibility_mode
        if hasattr(active_state.sched, "scheduler_settings"):
            active_state.sched.scheduler_settings = dict(settings)
        if hasattr(active_state.sched, "timesteps"):
            active_state.sched.timesteps = schedule_result.timesteps
        #Build Output
        active_state.extra["_scheduler_runtime_obj"] = scheduler
        schedule_output = SchedulerOutput(
            sigmas=schedule_result.sigmas,
            timesteps=schedule_result.timesteps,
            requested_steps=schedule_result.requested_steps,
            effective_steps=schedule_result.effective_steps,
            scheduler_step_override_applied=schedule_result.step_count_changed,
            compatibility_mode=compatibility_mode,
            metadata=schedule_metadata,
        )

        try:
            schedule_output = ensure_schedule_for_sampler(
                schedule=schedule_output,
                request=request,
                sampler_name=getattr(request, "sampler_name", None),
            )
        except Exception as e1:
            print(f"[SchedulerAdapter] strict compatibility failed: {e1}")

            schedule_output = ensure_schedule_for_sampler(
                schedule=schedule_output,
                request=request,
                sampler_name=getattr(request, "sampler_name", None),
                rebuild_schedule_fn=lambda enforced_request: self.build_schedule(
                    enforced_request,
                    state=active_state,
                ),
            )

        return schedule_output

    def _sync_request_to_state(self, state, request) -> None:
        if hasattr(state, "p"):
            if hasattr(state.p, "steps"):
                state.p.steps = request.steps
            if hasattr(state.p, "batch_size"):
                state.p.batch_size = request.batch_size
            if hasattr(state.p, "cfg_scale"):
                state.p.cfg_scale = request.cfg_scale
            if hasattr(state.p, "width"):
                state.p.width = int(getattr(request, "generation_width", request.width))
            if hasattr(state.p, "height"):
                state.p.height = int(getattr(request, "generation_height", request.height))
            if hasattr(state.p, "positive_prompt"):
                state.p.positive_prompt = request.positive_prompt
            if hasattr(state.p, "negative_prompt"):
                state.p.negative_prompt = request.negative_prompt
            if getattr(request, "seed", None) is not None and hasattr(state.p, "seed"):
                state.p.seed = request.seed
                
SCHEDULER_ADAPTER_CLASS=SimpleKESSchedulerAdapter