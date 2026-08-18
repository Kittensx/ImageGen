from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import copy
import inspect
import random

import torch

from modules.ss_registry.schedulers.simple_kes_sched.utils.check_device import check_device

@dataclass
class KESScheduleResult:
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    requested_steps: int
    effective_steps: int
    step_count_changed: bool
    schedule_mode: str
    active_blend_methods: list[str]
    active_blend_weights: list[float]
    tail_features_used: dict[str, bool]
    prepass_used: bool
    predicted_stop_step: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

class ScheduleBuilderMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    def _sigmas_to_timesteps(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Map sigma values to the SD training timestep domain.

        Stable Diffusion 1.x uses a 1000-step scaled-linear beta schedule.
        The previous logarithmic approximation reversed the time direction for
        decreasing sigmas. This implementation reconstructs the training sigma
        table and interpolates in log-sigma space, matching the convention used
        by k-diffusion style samplers. Settings may override the SD1 defaults.
        """
        num_train_timesteps = int(self.settings.get("num_train_timesteps", 1000))
        beta_start = float(self.settings.get("beta_start", 0.00085))
        beta_end = float(self.settings.get("beta_end", 0.012))
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1.")

        device = sigmas.device
        betas = torch.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_train_timesteps,
            device=device,
            dtype=torch.float64,
        ).square()
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        training_sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
        log_training_sigmas = training_sigmas.log()

        requested = sigmas.to(device=device, dtype=torch.float64).clamp(min=0.0)
        zero_mask = requested <= 0.0
        training_sigma_min = float(training_sigmas[0].detach().cpu())
        training_sigma_max = float(training_sigmas[-1].detach().cpu())
        clipped_low = int(((requested > 0.0) & (requested < training_sigma_min)).sum().item())
        clipped_high = int((requested > training_sigma_max).sum().item())
        self._timestep_mapping_metadata = {
            "type": "sd_scaled_linear_beta_log_sigma_interpolation",
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "training_sigma_min": training_sigma_min,
            "training_sigma_max": training_sigma_max,
            "clipped_low_count": clipped_low,
            "clipped_high_count": clipped_high,
        }

        log_requested = requested.clamp(min=training_sigma_min).log()
        log_requested = log_requested.clamp(
            min=float(log_training_sigmas[0]),
            max=float(log_training_sigmas[-1]),
        )

        upper = torch.searchsorted(log_training_sigmas, log_requested)
        upper = upper.clamp(min=1, max=num_train_timesteps - 1)
        lower = upper - 1
        low_log = log_training_sigmas[lower]
        high_log = log_training_sigmas[upper]
        weight = (log_requested - low_log) / (high_log - low_log).clamp(min=1e-12)
        timesteps = lower.to(torch.float64) + weight
        timesteps = torch.where(zero_mask, torch.zeros_like(timesteps), timesteps)
        return timesteps.to(dtype=torch.float32)

    def build_schedule(
        self,
        steps: Optional[int] = None,
        device: torch.device | str | None = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: Optional[float] = None,
        decay_pattern: Optional[str] = None,
        blend_methods: Optional[list[str]] = None,
        blend_weights: Optional[list[float]] = None,
        **kwargs,
    ) -> KESScheduleResult:
        #this is the entry point for the schedule adapter.
        
        self._reset_runtime_flags()
        self.settings = copy.deepcopy(self._base_settings)
        self._apply_settings_to_self(self.settings)

        self._apply_runtime_request(
            steps=steps,
            device=device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            decay_pattern=decay_pattern,
            blend_methods=blend_methods,
            blend_weights=blend_weights,
            **kwargs,
        )
        if self.debug:
            self.log(
                "[Scheduler parameters] "
                f"steps={self.steps}, sigma_min={self.sigma_min}, "
                f"sigma_max={self.sigma_max}, rho={self.rho}, "
                f"decay_pattern={self.decay_pattern}, "
                f"blending_mode={getattr(self, 'blending_mode', None)}"
            )
        self._apply_compatibility_policy()

        if self.global_randomize:
            self.apply_global_randomization()
        self._apply_runtime_randomization()

        self.blend_method_dict = copy.deepcopy(self.settings.get("blend_methods", self.blend_method_dict))
        self.blend_methods = list(self.blend_method_dict.keys())
        self.blend_weights = [
            self.blend_method_dict[method].get("weight", 1.0)
            for method in self.blend_methods
        ]

        if steps is not None:
            requested_steps = int(steps)
        else:
            requested_steps = int(self.original_steps)

        if not self.skip_prepass:
            self.prepass_compute_sigmas(
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                device=self.device,
                schedule_type=getattr(self, "schedule_type", None),
                decay_pattern=self.decay_pattern,
                skip_prepass=self.skip_prepass,
            )

        if self.load_prepass_sigmas:
            self.generate_sigmas_schedule(mode="prepass")

        if self.load_sigma_cache:
            self.generate_sigmas_schedule(mode="final")
        else:
            self.config_values()
            self.generate_sigmas_schedule()

            if getattr(self, "blending_mode", "default") == "default":
                self.blend_sigma_sequence(
                    sigmas_karras=self.scheduler_registry.get("karras")(
                        steps=self.steps,
                        sigma_min=self.sigma_min,
                        sigma_max=self.sigma_max,
                        rho=self.rho,
                        device=self.device,
                        decay_pattern=self.decay_pattern,
                    )[2],
                    sigmas_exponential=self.scheduler_registry.get("exponential")(
                        steps=self.steps,
                        sigma_min=self.sigma_min,
                        sigma_max=self.sigma_max,
                        device=self.device,
                        decay_pattern=self.decay_pattern,
                    )[2],
                    pre_pass=False,
                    blend_methods=self.blend_methods,
                    blend_weights=self.blend_weights,
                )
            else:
                self.blend_sigma_sequence(
                    sigmas_karras=None,
                    sigmas_exponential=None,
                    pre_pass=False,
                    blend_methods=self.blend_methods,
                    blend_weights=self.blend_weights,
                )

        sigmas = self.compute_sigmas(
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=self.device,
        )
        policy = getattr(self, "_compatibility_policy", self._resolve_runtime_compatibility_policy())
        truncate_to_requested_steps = bool(policy.get("truncate_to_requested_steps", False))

        if truncate_to_requested_steps:
            sigmas = self._truncate_and_repair_sigmas(
                sigmas,
                requested_steps=requested_steps,
                history_window=int(getattr(self, "history_window", 10)),
                repair_steps=int(getattr(self, "repair_steps", 4)),
            )

        if torch.isnan(sigmas).any():
            raise ValueError("[SimpleKEScheduler] NaN detected in sigmas")
        if torch.isinf(sigmas).any():
            raise ValueError("[SimpleKEScheduler] Inf detected in sigmas")
        if (sigmas <= 0).all():
            raise ValueError("[SimpleKEScheduler] All sigma values are <= 0")
        if (sigmas > 1000).all():
            raise ValueError("[SimpleKEScheduler] Sigma values are extremely large — might explode the model")

        if self.debug:
            self.save_generation_settings()
            check_device(sigmas, enabled=self.debug)
        
        # A sigma schedule represents transitions and therefore must end at
        # sigma zero. The previous implementation padded a short schedule by
        # repeating the final positive sigma, creating a no-op final step.
        target_len = int(requested_steps) + 1
        sigmas = sigmas.flatten()
        terminal_epsilon = torch.finfo(sigmas.dtype).eps * 16
        has_terminal_zero = bool(torch.abs(sigmas[-1]).item() <= terminal_epsilon)
        body = sigmas[:-1] if has_terminal_zero else sigmas

        if truncate_to_requested_steps and body.numel() > requested_steps:
            body = body[:requested_steps]

        minimum_body_count = int(requested_steps) if truncate_to_requested_steps else max(
            int(requested_steps), int(body.numel())
        )
        if body.numel() < minimum_body_count:
            missing = minimum_body_count - int(body.numel())
            start = body[-1] if body.numel() else torch.as_tensor(
                self.sigma_max,
                device=sigmas.device,
                dtype=sigmas.dtype,
            )
            inserted = torch.linspace(
                start,
                torch.zeros((), device=sigmas.device, dtype=sigmas.dtype),
                missing + 2,
                device=sigmas.device,
                dtype=sigmas.dtype,
            )[1:-1]
            body = torch.cat([body, inserted])

        sigmas = torch.cat([
            body,
            torch.zeros(1, device=sigmas.device, dtype=sigmas.dtype),
        ])
        self._terminal_sigma_added = not has_terminal_zero

        if sigmas.numel() < target_len:
            raise ValueError(
                "[SimpleKEScheduler] Unable to construct one sigma per requested transition."
            )

        sigmas = self._repair_non_increasing_sigmas(sigmas)
        if torch.any(sigmas[1:] > sigmas[:-1]):
            raise ValueError("[SimpleKEScheduler] Sigma schedule must be non-increasing after safety repair.")

        effective_steps = max(int(sigmas.numel()) - 1, 0)
        step_count_changed = effective_steps != requested_steps

        final_active_blend_methods = list(getattr(self, "blend_methods", []) or [])
        final_active_blend_weights = list(getattr(self, "blend_weights", []) or [])

        tail_features_used = {
            "tail_steps_applied": self._tail_steps_applied,
            "decay_tail_applied": self._decay_tail_applied,
            "blended_tail_applied": self._blended_tail_applied,
            "progressive_decay_applied": self._progressive_decay_applied,
            "auto_stabilization_applied": self._auto_stabilization_applied,
            "step_expansion_applied": self._step_expansion_applied,
            "monotonic_repair_applied": self._monotonic_repair_applied,
        }


        predicted_stop_step = self.predicted_stop_step
        if predicted_stop_step is None:
            predicted_stop_step = effective_steps
        else:
            predicted_stop_step = int(predicted_stop_step)
            predicted_stop_step = max(0, min(predicted_stop_step, effective_steps))
        self.predicted_stop_step = predicted_stop_step
        if self.graph_save_enable:
            self.save_image_plot(sigmas, predicted_stop_step)
        timesteps = self._sigmas_to_timesteps(sigmas)
        result = KESScheduleResult(
            sigmas=sigmas,
            timesteps=timesteps,
            requested_steps=requested_steps,
            effective_steps=effective_steps,
            step_count_changed=step_count_changed,
            schedule_mode=str(getattr(self, "blending_mode", "default")),
            active_blend_methods=final_active_blend_methods,
            active_blend_weights=final_active_blend_weights,
            tail_features_used=tail_features_used,
            prepass_used=not bool(self.skip_prepass),
            predicted_stop_step=predicted_stop_step,
            extra={
                "compatibility_mode": self.settings.get("compatibility", {}).get("pipeline_mode"),
                "validation_warnings": self.settings.get("_validation_warnings", []),
                "scheduler_settings":  copy.deepcopy(self.settings),
                "skip_prepass": bool(self.skip_prepass),
                "load_prepass_sigmas": bool(self.load_prepass_sigmas),
                "load_sigma_cache": bool(self.load_sigma_cache),
                "terminal_sigma_added": bool(self._terminal_sigma_added),
                "monotonic_repair": {
                    "applied": bool(self._monotonic_repair_applied),
                    "violation_count": int(self._monotonic_repair_count),
                    "max_increase": float(self._monotonic_repair_max_increase),
                },
                "timestep_mapping": dict(getattr(self, "_timestep_mapping_metadata", {})),
            },
        )

        if hasattr(self.state, "sched"):
            self.state.sched.sigmas = sigmas
            if hasattr(self.state.sched, "requested_steps"):
                self.state.sched.requested_steps = requested_steps
            if hasattr(self.state.sched, "effective_steps"):
                self.state.sched.effective_steps = effective_steps
       
        
        return result

    def call_scheduler(self, method_name, *args, **kwargs):
        sigma_sequence = getattr(self, f"sigmas_{method_name}")
        if sigma_sequence is None:
            self.log(f"No sigma sequence found for method: {method_name}")
            return None
        return sigma_sequence

    def _generate_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern=None, decay_mode=None, tail_steps=None):
        scheduler_func = self.scheduler_registry.get(schedule_type)
        if scheduler_func is None:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        tails, decay, extras, sigmas = self.call_scheduler_function(
            scheduler_func,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
            decay_pattern=decay_pattern,
            decay_mode=decay_mode,
            tail_steps=tail_steps
        )

        return tails, decay, extras, sigmas

    def generate_sigmas_schedule(self, mode=None):
        if mode == 'prepass':
            if self.load_prepass_sigmas:
                self.cache_file = self.prepass_save_file
            self.mode = 'prepass'
        elif mode == 'final':
            if self.load_sigma_cache:
                self.cache_file = self.final_save_file
            self.mode = 'final'
        else:
            self.mode = None
            self.cache_file = None

        self.load_blend_method_sigmas(mode=self.mode)
        self.blend_pairs = []
        self.active_methods = [method for method in self.blend_methods if self.blend_method_dict[method].get('weight', 1.0) > 0]

        if self.blending_mode == 'default':
            self._call_legacy_mode(schedule_type='exponential')
            self._call_legacy_mode(schedule_type='karras')

            self.blend_pairs = [
                {'method_label': 'method_a', 'method': 'karras', 'sigmas': self.sigmas_karras},
                {'method_label': 'method_b', 'method': 'exponential', 'sigmas': self.sigmas_exponential},
            ]

            max_length = max(len(pair['sigmas']) for pair in self.blend_pairs)
            for pair in self.blend_pairs:
                if len(pair['sigmas']) < max_length:
                    padding = torch.full((max_length - len(pair['sigmas']),), pair['sigmas'][-1]).to(pair['sigmas'].device)
                    pair['sigmas'] = torch.cat([pair['sigmas'], padding])

            self.log(f"All sigma sequences aligned to length: {max_length}")

            if self.blend_pairs[0]['sigmas'] is None:
                raise ValueError("Sigmas karras failed to generate or assign correctly.")
            if self.blend_pairs[1]['sigmas'] is None:
                raise ValueError("Sigmas exponential failed to generate or assign correctly.")
        else:
            if len(self.active_methods) == 1:
                method = self.active_methods[0]
                self.blend_pairs = [{'method_label': 'method_a', 'method': method, 'sigmas': self.sigma_sequences[method]['sigmas']}]
            elif len(self.active_methods) >= 2:
                self.blend_pairs = []
                for idx, method in enumerate(self.active_methods):
                    self.blend_pairs.append({
                        'method_label': f'method_{chr(97 + idx)}',
                        'method': method,
                        'sigmas': self.sigma_sequences[method]['sigmas']
                    })
                for pair in self.blend_pairs:
                    if pair['sigmas'] is None:
                        raise ValueError(f"Sigmas {pair['method']} failed to generate or assign correctly.")

        if len(self.blend_pairs) > 1:
            target_length = min(len(pair['sigmas']) for pair in self.blend_pairs)
            for pair in self.blend_pairs:
                pair['sigmas'] = pair['sigmas'][:target_length]

            max_length = max(len(pair['sigmas']) for pair in self.blend_pairs)
            for pair in self.blend_pairs:
                if len(pair['sigmas']) < max_length:
                    padding = torch.full((max_length - len(pair['sigmas']),), pair['sigmas'][-1]).to(pair['sigmas'].device)
                    pair['sigmas'] = torch.cat([pair['sigmas'], padding])

            self.log(f"All sigma sequences aligned to length: {max_length}")
            self.sigs = torch.zeros(target_length, device=self.blend_pairs[0]['sigmas'].device)
        else:
            self.sigs = self.blend_pairs[0]['sigmas'].clone()

        '''
        #caused a bug!
        if not torch.any(self.sigs > 0):
            self.sigma_min = self.min_threshold
            self.sigma_max = self.min_threshold
            self.log(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
        else:
            self.sigma_min = self.sigs[self.sigs > 0].min()
            self.sigma_max = self.sigs.max()
        '''
        return {
            'blend_methods': self.blend_methods,
            'all_sigmas': self.all_sigmas,
            'sigs': self.sigs
        }

    def call_scheduler_function(self, scheduler_func, **kwargs):
        valid_params = inspect.signature(scheduler_func).parameters
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_params}

        result = scheduler_func(**filtered_args)

        if isinstance(result, dict):
            tails = result.get('tails', None)
            decay = result.get('decay', None)
            sigmas = result.get('sigmas')
            extras = result.get('extras', [])
            if sigmas is None:
                raise ValueError("Scheduler function must return a 'sigmas' key.")
            return tails, decay, extras, sigmas

        if not isinstance(result, tuple):
            return None, None, [], result

        if len(result) == 0:
            raise ValueError("Scheduler function returned an empty tuple. This is not allowed.")

        sigmas = result[-1]
        optional_returns = result[:-1]
        tails = optional_returns[0] if len(optional_returns) > 0 else None
        decay = optional_returns[1] if len(optional_returns) > 1 else None
        extras = optional_returns[2:] if len(optional_returns) > 2 else []

        return tails, decay, extras, sigmas

    def config_values(self):
        sigma_auto_enabled = getattr(self, "sigma_auto_enabled", False)
        sigma_auto_mode = getattr(self, "sigma_auto_mode", "sigma_min")
        sigma_scale_factor = float(getattr(self, "sigma_scale_factor", 3.0))

        if self.sigma_min >= self.sigma_max:
            correction_factor = random.uniform(0.01, 0.99)
            old_sigma_min = self.sigma_min
            self.sigma_min = self.sigma_max * correction_factor
            self.log(f"[Correction] sigma_min ({old_sigma_min}) was >= sigma_max ({self.sigma_max}). Adjusted sigma_min to {self.sigma_min} using correction factor {correction_factor}.")

        self.log(f"Final sigmas: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")

        if sigma_auto_enabled:
            if sigma_auto_mode not in ["sigma_min", "sigma_max"]:
                raise ValueError(f"[Config Error] Invalid sigma_auto_mode: {sigma_auto_mode}. Must be 'sigma_min' or 'sigma_max'.")
            if sigma_auto_mode == "sigma_min":
                self.sigma_min = self.sigma_max / sigma_scale_factor
                self.log(f"[Auto Sigma Min] sigma_min set to {self.sigma_min} using scale factor {sigma_scale_factor}")
            elif sigma_auto_mode == "sigma_max":
                self.sigma_max = self.sigma_min * self.sigma_scale_factor
                self.log(f"[Auto Sigma Max] sigma_max set to {self.sigma_max} using scale factor {sigma_scale_factor}")

        self.min_threshold = random.uniform(1e-5, 5e-5)

        if self.sigma_min < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_min was too low: {self.sigma_min} < min_threshold {self.min_threshold}")
            self.sigma_min = self.min_threshold

        if self.sigma_max < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_max was too low: {self.sigma_max} < min_threshold {self.min_threshold}")
            self.sigma_max = self.min_threshold

        valid_methods = ['mean', 'max', 'sum']
        if self.early_stopping_method not in valid_methods:
            self.log(f"[Config Correction] Invalid early_stopping_method: {self.early_stopping_method}. Defaulting to 'mean'.")
            self.early_stopping_method = 'mean'

    def prepass_compute_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type=None, decay_pattern=None, suffix=None, cache_key=None, skip_prepass=False) -> torch.Tensor:
        if self.steps is None:
            raise ValueError("Number of steps must be provided.")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)        
        self.generate_sigmas_schedule(mode='prepass')

        
        
      
        if self.sharpen_last_n_steps > len(self.sigs):
            self.sharpen_last_n_steps = len(self.sigs)
            self.log(f"[Sharpening Notice] Requested last {self.sharpen_last_n_steps} steps exceeds sequence length. Using entire sequence instead.")

        self.visual_sigma = max(0.8, self.sigma_min * self.min_visual_sigma)

        self.blend_sigma_sequence(
            sigmas_karras=None,
            sigmas_exponential=None,
            pre_pass=True,
            blend_methods=self.blend_methods,
            blend_weights=self.blend_weights
        )
        if torch.isnan(self.sigs).any() or torch.isinf(self.sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")
        final_steps = self.sigs[:self.predicted_stop_step +1 ].to(self.device)
        self.final_steps = final_steps
        if self.blending_mode == 'default':
            self.final_sigmas_karras = self.sigmas_karras
            self.final_sigmas_exponential = self.sigmas_exponential
            self.log(f" Final Steps = {self.final_steps}. Predicted_stop_step = {self.predicted_stop_step}. Original requested steps = {self.steps}")
            self.log(f"final sigmas karras: {self.final_sigmas_karras}")
        else:
            self.final_sigmas_blended = torch.tensor(self.blended_sigmas, device=self.device)
            self.log(f" Final Steps = {self.final_steps}. Predicted_stop_step = {self.predicted_stop_step}. Original requested steps = {self.steps}")
            self.log(f"final blended sigmas: {self.final_sigmas_blended}")
            for method, sigmas in zip(self.blend_methods, self.all_sigmas):
                self.log(f"Method: {method}, Sigma sequence: {sigmas}")

    def compute_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type=None, decay_pattern=None, cache_key=None) -> torch.Tensor:
        self.log(f"Using device: {self.device}")
        self.generate_sigmas_schedule(mode='final')

        # Initialize from the first active blend pair instead of assuming the
        # legacy Karras/Exponential pair exists. A valid weighted blend may omit
        # either or both of those methods entirely.
        if not self.blend_pairs:
            raise ValueError("[SimpleKEScheduler] No active sigma sequence is available for final blending.")
        base_sigmas = self.blend_pairs[0].get('sigmas')
        if base_sigmas is None:
            raise ValueError("[SimpleKEScheduler] The first active sigma sequence is unavailable.")
        if not isinstance(base_sigmas, torch.Tensor):
            base_sigmas = torch.tensor(base_sigmas, dtype=torch.float32, device=self.device)
        else:
            base_sigmas = base_sigmas.to(self.device, dtype=torch.float32).flatten()
        self.sigs = torch.zeros_like(base_sigmas, device=self.device)

        self.blend_sigma_sequence(
            sigmas_karras=getattr(self, 'final_sigmas_karras', getattr(self, 'sigmas_karras', None)),
            sigmas_exponential=getattr(self, 'final_sigmas_exponential', getattr(self, 'sigmas_exponential', None)),
            pre_pass=False,
            blend_methods=self.blend_methods,
            blend_weights=self.blend_weights
        )
        self.sigma_variance = torch.var(self.sigs).item()
        if self.sharpen_mode in ['last_n', 'both']:
            if self.sigma_variance < self.sharpen_variance_threshold:
                self.sharpen_mask = torch.where(self.sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                sharpen_indices = torch.where(self.sharpen_mask < 1.0)[0].tolist()
                self.sigs = self.sigs * self.sharpen_mask
                self.log(f"[Sharpen Mask] Full sharpening applied (low variance). Steps: {sharpen_indices}")
            else:
                recent_sigs = self.sigs[-self.sharpen_last_n_steps:]
                sharpen_mask = torch.where(recent_sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                self.sigs[-self.sharpen_last_n_steps:] = recent_sigs * sharpen_mask

                for j in range(len(self.sigs) - self.sharpen_last_n_steps, len(self.sigs)):
                    if self.sigs[j] < self.sigma_min * 1.5:
                        old_value = self.sigs[j].item()
                        self.sigs[j] = self.sigs[j] * self.sharpness
                        self.log(f"[Sharpening] Step {j+1}: Applied sharpening. Sigma changed from {old_value:.6f} to {self.sigs[j].item():.6f}")
                    else:
                        self.log(f"[Sharpening] Step {j+1}: No sharpening applied. Sigma: {self.sigs[j].item():.6f}")

        if self.sharpen_mode in ['full', 'both']:
            self.sharpen_mask = torch.where(self.sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
            sharpen_indices = torch.where(self.sharpen_mask < 1.0)[0].tolist()
            self.sigs = self.sigs * self.sharpen_mask
            self.log(f"[Sharpen Mask] Full sharpening applied at steps: {sharpen_indices}")

        return self.sigs.to(self.device)
