from __future__ import annotations

from typing import Any, Optional

import copy
import inspect
import os

import torch

from modules.ss_registry.schedulers.simple_kes_sched.get_sigmas import scheduler_registry
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import (
    KES_RUNTIME_DEFAULTS,
    resolve_simple_kes_pipeline_policy,
)
from modules.ss_registry.schedulers.simple_kes_sched.plugin_support import PluginSupport
from modules.ss_registry.schedulers.simple_kes_sched.schedule_builder import (
    KESScheduleResult,
    ScheduleBuilderMixin,
)
from modules.ss_registry.schedulers.simple_kes_sched.sigma_cache import SigmaCacheMixin
from modules.ss_registry.schedulers.simple_kes_sched.randomization import RandomizationMixin
from modules.ss_registry.schedulers.simple_kes_sched.blending import BlendingMixin
from modules.ss_registry.schedulers.simple_kes_sched.stabilization import StabilizationMixin
from modules.ss_registry.schedulers.simple_kes_sched.diagnostics import DiagnosticsMixin, SharedLogger


class SimpleKEScheduler(
    ScheduleBuilderMixin,
    SigmaCacheMixin,
    RandomizationMixin,
    BlendingMixin,
    StabilizationMixin,
    DiagnosticsMixin,
):
    """Public Simple KES scheduler facade.

    Runtime configuration and compatibility policy remain here. Internal schedule,
    cache, randomization, blending, stabilization, and diagnostic behaviors are
    composed from responsibility-specific mixins without changing caller-facing
    method names or registration.
    """

    _RUNTIME_DEFAULTS = dict(KES_RUNTIME_DEFAULTS)
    _ALLOWEDS = {
        "blending_mode": {"auto", "default", "smooth_blend", "weights"},
        "step_progress_mode": {"linear", "exponential", "logarithmic", "sigmoid"},
    }

    def __init__(
        self,
        shared_state: Any = None,
        steps: int = 25,
        device: torch.device | str | None = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: Optional[float] = None,
        decay_pattern: Optional[str] = None,
        decay_mode: Optional[str] = None,
        tail_steps: Optional[int] = None,
        verbose: Optional[bool] = None,
        settings: Optional[dict[str, Any]] = None,
        scheduler_name: str = "simple_kes",
        config_path: Optional[str] = None,
        preset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.state = shared_state
        if not self.state:
            caller = inspect.stack()[1]
            raise ValueError(
                f"SharedState was required but not provided. "
                f"Called by: {caller.function} in {caller.filename}:{caller.lineno}"
            )

        if settings is None:
            raise ValueError("SimpleKEScheduler requires pre-resolved settings")

        self.ps = PluginSupport
        self.update_state = self.ps.update_state

        self.p = self.state.p
        self.sched = self.state.sched
        self.d = self.state.d

        self.scheduler_registry = scheduler_registry
        self.scheduler_name = scheduler_name
        self.pipeline_mode = pipeline_mode
        self.kwargs = dict(kwargs)

        initial_debug = bool(
            kwargs.get("debug", False)
            or (settings or {}).get("debug", False)
            or getattr(getattr(self.state, "sched", None), "debug", False)
        )
        self.logger = SharedLogger(debug=initial_debug)
        self.log = self.logger.log
        self.prepass_log = self.logger.prepass_log

        self.RANDOMIZATION_TYPE_ALIASES = {
            "symmetric": "symmetric", "sym": "symmetric", "s": "symmetric",
            "asymmetric": "asymmetric", "assym": "asymmetric", "a": "asymmetric", "asym": "asymmetric",
            "logarithmic": "logarithmic", "log": "logarithmic", "l": "logarithmic",
            "exponential": "exponential", "exp": "exponential", "e": "exponential",
        }

        self._base_settings = copy.deepcopy(settings)
        self.settings = copy.deepcopy(settings)

        self.logger.debug = bool(self.settings.get("debug", False))
        self.log = self.logger.log
        self.prepass_log = self.logger.prepass_log

        self._apply_settings_to_self(self.settings)

        explicit_steps = steps if steps is not None else None
        settings_steps = self.settings.get("steps", None)
        state_steps = getattr(self.p, "steps", None)

        resolved_steps = (
            explicit_steps
            if explicit_steps is not None
            else settings_steps
            if settings_steps is not None
            else state_steps
            if state_steps is not None
            else 25
        )

        self.original_steps = int(resolved_steps)
        self.steps = self.original_steps    
        
        self.predicted_stop_step = max(int(self.original_steps) - 1, 0)        
        self.device = self._resolve_device(device)
        self.sigma_min = sigma_min if sigma_min is not None else self.settings.get("sigma_min")
        self.sigma_max = sigma_max if sigma_max is not None else self.settings.get("sigma_max")
        self.rho = rho if rho is not None else self.settings.get("rho")
        self.decay_pattern = decay_pattern if decay_pattern is not None else self.settings.get("decay_pattern", "extrapolate")
        self.decay_mode = decay_mode if decay_mode is not None else self.settings.get("decay_mode", "append")
        self.tail_steps = tail_steps if tail_steps is not None else self.settings.get("tail_steps", 1)
        self.verbose = bool(verbose if verbose is not None else self.settings.get("verbose", False))
        self.debug = bool(self.settings.get("debug", False))

        self.global_randomize = bool(self.settings.get("global_randomize", False))

        self.re_randomizable_keys = [
            "sigma_min", "sigma_max", "start_blend", "end_blend", "sharpness",
            "early_stopping_threshold", "initial_step_size", "final_step_size",
            "initial_noise_scale", "final_noise_scale", "smooth_blend_factor",
            "step_size_factor", "noise_scale_factor", "rho",
        ]
        for key in self.re_randomizable_keys:
            value = self.settings.get(key, getattr(self, key, None))
            setattr(self, key, value)

        self.auto_mode_enabled = bool(self.settings.get("auto_tail_smoothing", False))
        self.relative_converged = False
        self.max_converged = False
        self.delta_converged = False
        self.early_stop_triggered = False
        self.predicted_stop_step = None
        
       
        self.loaded_sigmas = None
        self.sigma_cache = {}
        self.sigma_sequences = {}
        self.schedule_type = None
        self.suffix = None
        self.ext = None

        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        self.sigma_save_subfolder = self.settings.get("sigma_save_subfolder", "saved_sigmas")
        self.sigma_save_folder = os.path.join(self.cache_dir, self.sigma_save_subfolder)
        self.prepass_save_file = os.path.join(self.sigma_save_folder, "simple_kes_prepass_sigmas.pt")
        self.final_save_file = os.path.join(self.sigma_save_folder, "simple_kes_final_sigmas.pt")
        self.cache_file = None

        self.blend_method_dict = copy.deepcopy(self.settings.get("blend_methods", {}))
        self.blend_methods = list(self.blend_method_dict.keys())
        self.blend_weights = [
            self.blend_method_dict[method].get("weight", 1.0)
            for method in self.blend_methods
        ]

        self._create_directories()
        self.initialize_generation_filename(folder=self.log_save_directory)
        log_stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        self.extras_log_filename = os.path.join(
            self.sigma_save_folder,
            f"{log_stem}_extras.txt"
        )
        self._reset_runtime_flags()

    def _apply_settings_to_self(self, settings: dict[str, Any]) -> None:
        for key, value in settings.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            setattr(self, key, value)

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is None:
            device = getattr(self.d, "device", None) or self.settings.get("device")
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _reset_runtime_flags(self) -> None:
        self._tail_steps_applied = False
        self._decay_tail_applied = False
        self._blended_tail_applied = False
        self._progressive_decay_applied = False
        self._auto_stabilization_applied = False
        self._step_expansion_applied = False
        self._terminal_sigma_added = False
        self._monotonic_repair_applied = False
        self._monotonic_repair_count = 0
        self._monotonic_repair_max_increase = 0.0

    def _resolve_runtime_compatibility_policy(self) -> dict[str, Any]:
        explicit_policy = self.settings.get("_policy")
        if isinstance(explicit_policy, dict):
            return copy.deepcopy(explicit_policy)

        return resolve_simple_kes_pipeline_policy(
            self.settings,
            pipeline_mode=self.pipeline_mode,
        )

    def _apply_compatibility_policy(self) -> list[str]:
        policy = self._resolve_runtime_compatibility_policy()
        warnings: list[str] = list(self.settings.get("_validation_warnings", []) or [])

        self.settings["_policy"] = copy.deepcopy(policy)

        compatibility = copy.deepcopy(self.settings.get("compatibility", {}) or {})
        compatibility["pipeline_mode"] = policy["compatibility_mode"]
        compatibility["truncate_to_requested_steps"] = policy["truncate_to_requested_steps"]
        compatibility["warn_on_feature_downgrade"] = policy["warn_on_feature_downgrade"]
        self.settings["compatibility"] = compatibility

        if not policy.get("allow_step_expansion", False):
            if getattr(self, "allow_step_expansion", False):
                warnings.append("allow_step_expansion requested but disabled by compatibility policy.")
            self.allow_step_expansion = False
            self.settings["allow_step_expansion"] = False

        if not policy.get("allow_tail_append", False):
            for key in ("apply_tail_steps", "apply_blended_tail"):
                if getattr(self, key, False):
                    warnings.append(f"{key} requested but disabled by compatibility policy.")
                setattr(self, key, False)
                self.settings[key] = False

        if not policy.get("allow_decay_append", False):
            for key in ("apply_decay_tail", "apply_progressive_decay"):
                if getattr(self, key, False):
                    warnings.append(f"{key} requested but disabled by compatibility policy.")
                setattr(self, key, False)
                self.settings[key] = False

        if not self.allow_step_expansion and getattr(self, "auto_tail_smoothing", False):
            self.auto_mode_enabled = False
        else:
            self.auto_mode_enabled = bool(getattr(self, "auto_tail_smoothing", False))

        self.settings["_validation_warnings"] = warnings
        self._compatibility_policy = policy
        return warnings

    def _apply_runtime_request(
        self,
        *,
        steps: Optional[int],
        device: torch.device | str | None,
        sigma_min: Optional[float],
        sigma_max: Optional[float],
        rho: Optional[float],
        decay_pattern: Optional[str],
        blend_methods: Optional[list[str]],
        blend_weights: Optional[list[float]],
        **kwargs,
    ) -> None:
        if steps is not None:
            self.steps = int(steps)
            self.settings["steps"] = self.steps
        else:
            self.steps = int(self.settings.get("steps", self.original_steps))

        self.device = self._resolve_device(device)
        self.settings["device"] = self.device

        if sigma_min is not None:
            self.sigma_min = sigma_min
            self.settings["sigma_min"] = sigma_min
        if sigma_max is not None:
            self.sigma_max = sigma_max
            self.settings["sigma_max"] = sigma_max
        if rho is not None:
            self.rho = rho
            self.settings["rho"] = rho
        if decay_pattern is not None:
            self.decay_pattern = decay_pattern
            self.settings["decay_pattern"] = decay_pattern
        if blend_methods is not None:
            if isinstance(blend_methods, dict):
                # Canonical Simple KES settings store blend methods as a
                # method -> configuration mapping. Phase 11D originally
                # treated every value as a list of method names and wrote a
                # list back into settings, which later caused .keys() to fail.
                self.blend_method_dict = copy.deepcopy(blend_methods)
            else:
                requested_methods = [str(item).strip() for item in blend_methods if str(item).strip()]
                current_methods = self.settings.get("blend_methods", {})
                if not isinstance(current_methods, dict):
                    current_methods = {}
                self.blend_method_dict = {
                    method: copy.deepcopy(current_methods.get(method, {"weight": 1.0}))
                    for method in requested_methods
                }
            self.settings["blend_methods"] = copy.deepcopy(self.blend_method_dict)
            self.blend_methods = list(self.blend_method_dict.keys())
        if blend_weights is not None:
            weights = list(blend_weights)
            self.blend_weights = weights
            for index, method in enumerate(getattr(self, "blend_methods", [])):
                if index >= len(weights):
                    break
                method_config = self.blend_method_dict.setdefault(method, {})
                method_config["weight"] = float(weights[index])
            self.settings["blend_methods"] = copy.deepcopy(self.blend_method_dict)

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
                self.settings[key] = value

        mode = str(getattr(self, "blending_mode", self.settings.get("blending_mode", "default"))).strip().lower()
        if mode == "auto":
            if not self.blend_methods:
                mode = "default"
            elif len(self.blend_methods) == 2:
                mode = "smooth_blend"
            elif len(self.blend_methods) > 2:
                mode = "weights"
            else:
                mode = "default"
        self.blending_mode = mode
        self.settings["blending_mode"] = mode

        step_progress_mode = str(self.settings.get("step_progress_mode", "linear")).strip().lower()
        if step_progress_mode not in self._ALLOWEDS["step_progress_mode"]:
            step_progress_mode = "linear"
        self.step_progress_mode = step_progress_mode
        self.settings["step_progress_mode"] = step_progress_mode

        self.exp_power = float(self.settings.get("exp_power", 2.0))

    def __call__(
        self,
        steps=None,
        device=None,
        sigma_min=None,
        sigma_max=None,
        rho=None,
        decay_pattern=None,
        blend_methods=None,
        blend_weights=None,
        **kwargs,
    ):
        return self.build_schedule(
            steps=steps,
            device=device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            decay_pattern=decay_pattern,
            blend_methods=blend_methods,
            blend_weights=blend_weights,
            **kwargs,
        ).sigmas


SCHEDULER_CLASS = SimpleKEScheduler
