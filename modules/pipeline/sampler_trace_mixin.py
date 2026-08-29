# modules/pipeline/sampler_trace_mixin.py
from __future__ import annotations

from typing import Any, Optional
import torch

from modules.pipeline.live_preview import get_live_preview_sink


class SamplerTraceMixin:
    def _get_trace_recorder(self, request) -> Optional[Any]:
        sampler_kwargs = getattr(request, "sampler_kwargs", {}) or {}
        return sampler_kwargs.get("trace_recorder")

    def _apply_latent_step_hook(
        self,
        state: Any | None,
        *,
        request: Any,
        latent: torch.Tensor,
        step_index: int,
        sigma: Any = None,
        sigma_next: Any = None,
        timestep: Any = None,
    ) -> torch.Tensor:
        """Apply an optional pipeline-owned latent constraint after one transition.

        The hook is deliberately generic: samplers do not import outpaint code.
        Phase 14N-13P uses it for strict protected-latent restoration; future
        image-conditioned workflows may reuse the same narrow extension point.
        """
        extra = getattr(state, "extra", None) if state is not None else None
        hook = extra.get("sampling_latent_step_hook") if isinstance(extra, dict) else None
        if not callable(hook):
            return latent
        constrained = hook(
            latent,
            request=request,
            step_index=int(step_index),
            sigma=sigma,
            sigma_next=sigma_next,
            timestep=timestep,
        )
        if not torch.is_tensor(constrained):
            raise TypeError("sampling_latent_step_hook must return a torch.Tensor.")
        if constrained.shape != latent.shape:
            raise ValueError("sampling_latent_step_hook changed the latent tensor shape.")
        if not bool(torch.isfinite(constrained).all()):
            raise ValueError("sampling_latent_step_hook returned NaN or Inf values.")
        return constrained

    def _trace_step(
        self,
        request,
        *,
        step_index: int,
        sigma=None,
        sigma_next=None,
        timestep=None,
        latent_before=None,
        latent_after=None,
        noise_pred=None,
        guided_noise=None,
        cfg_scale=None,
        requested_cfg_scale=None,
        effective_cfg_scale=None,
        guidance_owner=None,
        unconditional_output=None,
        conditional_output=None,
        guidance_delta=None,
        guided_output=None,
        predicted_x0=None,
        stopping_candidate=None,
        extra=None,
        latent_snapshot=None,
        predicted_x0_snapshot=None,
    ) -> None:
        recorder = self._get_trace_recorder(request)
        if recorder is None or not getattr(recorder, "enabled", False):
            return

        recorder.record_step(
            step_index=step_index,
            sigma=sigma,
            sigma_next=sigma_next,
            timestep=timestep,
            latent_before=latent_before,
            latent_after=latent_after,
            noise_pred=noise_pred,
            guided_noise=guided_noise,
            cfg_scale=cfg_scale,
            requested_cfg_scale=requested_cfg_scale,
            effective_cfg_scale=effective_cfg_scale,
            guidance_owner=guidance_owner,
            unconditional_output=unconditional_output,
            conditional_output=conditional_output,
            guidance_delta=guidance_delta,
            guided_output=guided_output,
            predicted_x0=predicted_x0,
            stopping_candidate=stopping_candidate,
            extra=extra,
            latent_snapshot=latent_snapshot,
            predicted_x0_snapshot=predicted_x0_snapshot,
        )

    def _trace_sampler_summary(
        self,
        request,
        *,
        requested_steps=None,
        effective_steps=None,
        stopping_index=None,
    ) -> None:
        recorder = self._get_trace_recorder(request)
        if recorder is None or not getattr(recorder, "enabled", False):
            return

        recorder.set_runtime_summary(
            requested_steps=requested_steps,
            effective_steps=effective_steps,
            stopping_index=stopping_index,
        )
        

    def _emit_live_preview(
        self,
        state: Any | None,
        *,
        request: Any,
        step_index: int,
        total_steps: int,
        latent: torch.Tensor,
        predicted_x0: torch.Tensor | None = None,
        sigma: Any = None,
        model_timestep: Any = None,
        batch_index: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        sink = get_live_preview_sink(state)
        if sink is None:
            return

        payload = dict(metadata or {})
        progress_reporter = None
        if state is not None:
            state_extra = getattr(state, "extra", None)
            if isinstance(state_extra, dict):
                progress_reporter = state_extra.get("progress_reporter")
        phase_index = getattr(progress_reporter, "phase_index", None)
        if phase_index is not None:
            try:
                payload.setdefault("phase_index", max(0, int(phase_index)))
            except (TypeError, ValueError):
                pass
        payload.setdefault(
            "sampler_name",
            getattr(request, "sampler_name", None) or getattr(self, "SAMPLER_NAME", ""),
        )
        payload.setdefault("scheduler_name", getattr(request, "scheduler_name", None) or "")
        requested_cfg = getattr(request, "cfg_scale", None)
        sampler_kwargs = getattr(request, "sampler_kwargs", {}) or {}
        payload.setdefault("requested_cfg_scale", requested_cfg)
        payload.setdefault("effective_cfg_scale", requested_cfg)
        payload.setdefault(
            "guidance_mode",
            payload.get("cfg_guidance_mode")
            or sampler_kwargs.get("cfg_guidance_mode")
            or "flat",
        )
        payload.setdefault("cfg_rescale", getattr(request, "cfg_rescale", 0.0) or 0.0)
        payload.setdefault(
            "cfg_rescale_applied",
            bool(float(payload.get("cfg_rescale") or 0.0) > 0.0),
        )
        payload.setdefault("override_source", "base_request")
        payload.setdefault("transition_id", None)

        sink.on_step(
            step_index=int(step_index),
            total_steps=max(1, int(total_steps)),
            latent=latent,
            predicted_x0=predicted_x0,
            sigma=sigma,
            model_timestep=model_timestep,
            batch_index=int(batch_index),
            metadata=payload,
        )

    def _resolve_effective_steps(
        self,
        request: Any,
        schedule: Any,
        sigmas: torch.Tensor,
    ) -> tuple[int, int]:
        """
        Prefer scheduler-reported step counts when available.
        Fall back to request.steps and len(sigmas) - 1 otherwise.
        """
        schedule_extra = getattr(schedule, "extra", None)

        requested_steps = getattr(request, "steps", None)
        effective_steps = int(sigmas.numel() - 1)

        if isinstance(schedule_extra, dict):
            sched_requested = schedule_extra.get("requested_steps", None)
            sched_effective = schedule_extra.get("effective_steps", None)

            if sched_requested is not None:
                try:
                    requested_steps = int(sched_requested)
                except (TypeError, ValueError):
                    pass

            if sched_effective is not None:
                try:
                    effective_steps = int(sched_effective)
                except (TypeError, ValueError):
                    pass

        if requested_steps is None:
            requested_steps = int(sigmas.numel() - 1)
        else:
            requested_steps = int(requested_steps)

        return requested_steps, effective_steps
        
    def _materialize_sigmas(
        self,
        schedule: Any,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = getattr(schedule, "sigmas", None)
        if sigmas is None:
            raise ValueError("schedule must provide `sigmas`.")

        if callable(sigmas):
            sigmas = sigmas()

        if not torch.is_tensor(sigmas):
            sigmas = torch.tensor(
                sigmas,
                dtype=reference_tensor.dtype,
                device=reference_tensor.device,
            )
        else:
            sigmas = sigmas.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

        return sigmas.flatten()
    def _materialize_timesteps(
        self,
        schedule: Any,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = getattr(schedule, "timesteps", None)
        if timesteps is None:
            raise ValueError("schedule must provide `timesteps`.")

        if callable(timesteps):
            timesteps = timesteps()

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                timesteps,
                dtype=torch.float32,
                device=reference_tensor.device,
            )
        else:
            timesteps = timesteps.to(
                device=reference_tensor.device,
                dtype=torch.float32,
            )

        timesteps = timesteps.flatten()
        transition_count = int(self._materialize_sigmas(schedule, reference_tensor).numel() - 1)
        valid_lengths = {transition_count, transition_count + 1}
        if int(timesteps.numel()) not in valid_lengths:
            raise ValueError(
                "schedule.timesteps must provide one value per transition or one per sigma."
            )
        return timesteps

