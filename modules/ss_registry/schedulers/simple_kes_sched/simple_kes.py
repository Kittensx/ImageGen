from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

from copy import deepcopy

import copy
import glob
import hashlib
import inspect
import json
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from modules.ss_registry.schedulers.simple_kes_sched.get_sigmas import scheduler_registry
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import (
    KES_RANDOMIZATION_SAFE_BOUNDS,
    KES_RUNTIME_DEFAULTS,
    resolve_simple_kes_pipeline_policy,
)
from modules.ss_registry.schedulers.simple_kes_sched.utils.check_device import check_device
from modules.ss_registry.schedulers.simple_kes_sched.utils.plot_sigma_sequence import plot_sigma_sequence
from modules.ss_registry.schedulers.simple_kes_sched.plugin_support import PluginSupport
from modules.ss_registry.schedulers.simple_kes_sched.schedulers.shared import apply_decay_tail as apply_decay_tail_fn

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


class SharedLogger:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.log_buffer = []
        self.prepass_log_buffer = []

    def log(self, message):
        if self.debug:
            self.log_buffer.append(message)

    def prepass_log(self, message):
        if self.debug:
            self.prepass_log_buffer.append(message)


class SimpleKEScheduler:
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
        
    @staticmethod
    def _canonical_data_directory_name(value: str) -> str:
        """Correct known historical misspellings of image_generation_data."""

        replacements = {
            "image_gneraetion_data": "image_generation_data",
            "image_generaetion_data": "image_generation_data",
            "image_genaration_data": "image_generation_data",
        }
        path = Path(str(value or "image_generation_data").strip())
        corrected_name = replacements.get(path.name.casefold(), path.name)
        if corrected_name == path.name:
            return str(path)
        return str(path.with_name(corrected_name))

    def _resolve_data_directory(self, value: Any) -> str:
        raw = self._canonical_data_directory_name(str(value or "image_generation_data"))
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parents[4]
            candidate = project_root / candidate
        return str(candidate.resolve())

    def _create_directories(self):
        graph_directory = self._resolve_data_directory(
            self.settings.get("graph_save_directory", "image_generation_data")
        )
        log_directory = self._resolve_data_directory(
            self.settings.get("log_save_directory", "image_generation_data")
        )
        self.graph_save_directory = graph_directory
        self.log_save_directory = log_directory
        self.settings["graph_save_directory"] = graph_directory
        self.settings["log_save_directory"] = log_directory

        paths = [
            self.cache_dir,
            self.sigma_save_folder,
            graph_directory,
            log_directory,
        ]

        for directory in paths:
            try:
                if directory:
                    os.makedirs(directory, exist_ok=True)
            except Exception as exc:
                self.log(f"[Init Warning] Failed to create directory {directory}: {exc}")
            
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

    def _repair_non_increasing_sigmas(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Repair advanced tail/blend combinations that introduce upward sigma jumps.

        Simple KES exposes several experimental tail, decay, blending, and step-expansion
        features that may be combined freely. Some valid combinations can append a segment
        whose first value is above the preceding sigma. The sampler contract requires a
        non-increasing sequence, so normalize only the offending upward transitions rather
        than failing the entire generation before sampling starts.
        """
        values = sigmas.flatten()
        if values.numel() < 2:
            return values

        increases = values[1:] - values[:-1]
        violation_mask = increases > 0
        if not bool(torch.any(violation_mask).item()):
            return values

        violation_count = int(torch.count_nonzero(violation_mask).item())
        max_increase = float(torch.max(increases[violation_mask]).item())

        # Sigma values are non-negative. cummin preserves every already-valid decline and
        # clips only values that would rise above the lowest sigma reached so far.
        repaired = torch.cummin(torch.clamp(values, min=0), dim=0).values
        if repaired.numel():
            repaired[-1] = torch.zeros((), device=repaired.device, dtype=repaired.dtype)

        self._monotonic_repair_applied = True
        self._monotonic_repair_count = violation_count
        self._monotonic_repair_max_increase = max_increase
        warning = (
            "Simple KES repaired "
            f"{violation_count} upward sigma transition(s) produced by the selected "
            "advanced scheduler features so the schedule remains non-increasing."
        )
        warnings = list(self.settings.get("_validation_warnings", []) or [])
        if warning not in warnings:
            warnings.append(warning)
        self.settings["_validation_warnings"] = warnings
        self.log(
            "[Scheduler Safety] Repaired non-increasing sigma schedule: "
            f"violations={violation_count}, max_increase={max_increase:.8f}."
        )
        return repaired
    
    def _resolve_effective_history_window(
        self,
        full_sigmas: torch.Tensor,
        target_len: int,
        requested_history_window: int,
    ) -> int:
        max_history_window = max(3, min(int(full_sigmas.numel()) - 1, target_len - 1))
        requested_history_window = max(3, int(requested_history_window))
        requested_history_window = min(requested_history_window, max_history_window)

        if not bool(getattr(self, "auto_history", False)):
            return requested_history_window

        # Conservative first-pass heuristic:
        # start from a percentage of visible target length, capped by configured value.
        adaptive_history = max(4, int(target_len * 0.25))
        adaptive_history = min(adaptive_history, requested_history_window)

        # If the visible tail already looks flat, bias more local.
        visible = full_sigmas[:target_len]
        if visible.numel() >= 5:
            tail = visible[-5:]
            deltas = torch.abs(tail[:-1] - tail[1:])
            if deltas.numel() >= 2:
                delta_var = torch.var(deltas).item() if deltas.numel() > 1 else 0.0
                max_delta = torch.max(deltas).item()
                jaggedness_threshold = float(getattr(self, "jaggedness_threshold", 0.01))
                auto_tail_threshold = float(getattr(self, "auto_tail_threshold", 0.05))

                if max_delta <= auto_tail_threshold and delta_var <= jaggedness_threshold:
                    adaptive_history = max(3, min(adaptive_history, 4))

        self.log(
            f"[Compatibility] auto_history enabled: resolved history_window="
            f"{adaptive_history} (requested={requested_history_window}, target_len={target_len})"
        )
        return adaptive_history
        
    def _infer_tail_repair_pattern(
        self,
        history: torch.Tensor,
        fallback_pattern: str = "extrapolate",
    ) -> str:
        """
        Infer the best tail repair pattern from recent delta behavior.
        Returns one of the existing shared decay patterns.
        """
        if not isinstance(history, torch.Tensor) or history.numel() < 4:
            return fallback_pattern

        history = history.detach().flatten()
        deltas = torch.abs(history[:-1] - history[1:])

        if deltas.numel() < 2:
            return fallback_pattern

        delta_var = torch.var(deltas).item() if deltas.numel() > 1 else 0.0
        mean_delta = torch.mean(deltas).item()
        max_delta = torch.max(deltas).item()

        eps = 1e-8
        ratios = deltas[1:] / (deltas[:-1] + eps)
        ratio_var = torch.var(ratios).item() if ratios.numel() > 1 else 0.0
        mean_ratio = torch.mean(ratios).item() if ratios.numel() > 0 else 1.0

        jaggedness_threshold = float(getattr(self, "jaggedness_threshold", 0.01))

        # Very stable delta size -> linear-like tail
        if delta_var <= jaggedness_threshold * 0.25:
            return "linear"

        # Strongly multiplicative decay -> geometric/exponential-like
        if ratio_var <= jaggedness_threshold * 0.25:
            if mean_ratio < 0.85:
                return "exponential"
            if mean_ratio < 1.0:
                return "geometric"

        # Shrinking deltas but not multiplicatively stable -> harmonic/log-like
        if mean_ratio < 1.0:
            if max_delta > mean_delta * 1.5:
                return "harmonic"
            return "logarithmic"

        # Nearly constant last delta -> extrapolate current trend
        tail_delta_change = abs(deltas[-1].item() - deltas[-2].item()) if deltas.numel() >= 2 else 0.0
        if tail_delta_change <= jaggedness_threshold:
            return "extrapolate"

        return fallback_pattern
        
    def _truncate_and_repair_sigmas(
        self,
        sigmas: torch.Tensor,
        requested_steps: int,
        history_window: int = 10,
        repair_steps: int = 4,
    ) -> torch.Tensor:
        """
        Truncate a sigma sequence to requested_steps + 1 values and repair the
        exposed tail using an inferred shared decay pattern that best matches
        the local curve behavior.
        """
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"sigmas must be a torch.Tensor, got {type(sigmas)}")

        target_len = max(int(requested_steps), 0) + 1
        current_len = int(sigmas.numel())

        if current_len <= target_len:
            return sigmas

        full_sigmas = sigmas.detach().clone()
        truncated = full_sigmas[:target_len].clone()

        if truncated.numel() < 4:
            return truncated

        max_repair_steps = max(2, truncated.numel() - 2)
        repair_steps = max(2, int(repair_steps))
        repair_steps = min(repair_steps, max_repair_steps)

        history_window = self._resolve_effective_history_window(
            full_sigmas,
            target_len,
            history_window,
        )

        history_start = max(0, target_len - history_window - 1)
        history = full_sigmas[history_start:target_len]

        active_decay_pattern = str(getattr(self, "decay_pattern", "extrapolate") or "extrapolate").strip().lower()
        inferred_pattern = self._infer_tail_repair_pattern(
            history,
            fallback_pattern=active_decay_pattern,
        )

        

        # Keep prefix unchanged, rebuild only the last repair_steps values.
        prefix_len = truncated.numel() - repair_steps
        if prefix_len < 2:
            return truncated
        
        

        prefix = truncated[:prefix_len].clone()

        # Reuse the current shared tail generator behavior by extending from prefix.
        repaired_full = apply_decay_tail_fn(
            prefix,
            truncated.device,
            decay_pattern=inferred_pattern,
            tail_steps=repair_steps,
        )

        # apply_decay_tail_fn appends repair_steps values to prefix, giving us the target length.
        if repaired_full.numel() != truncated.numel():
            repaired_full = repaired_full[:truncated.numel()]

        # Safety clamp and monotonic cleanup
        repaired_full = torch.clamp(repaired_full, min=1e-5)

        if repaired_full.numel() >= 2:
            for i in range(1, repaired_full.numel()):
                if repaired_full[i] > repaired_full[i - 1]:
                    repaired_full[i] = repaired_full[i - 1]

        self.log(
            f"[Compatibility] Truncated sigma sequence from {current_len} to {target_len} "
            f"and repaired tail using pattern '{inferred_pattern}' over {repair_steps} steps."
        )

        return repaired_full
        
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

    def _log_extras_to_file(self, all_extras):
        if not self.extras_log_filename:
            return

        try:
            with open(self.extras_log_filename, "a", encoding="utf-8") as f:
                f.write("\n=== New Scheduler Extras ===\n")

                for method in self.blend_methods:
                    seq = self.sigma_sequences.get(method, {})
                    sigmas = seq.get("sigmas")
                    tails = seq.get("tails")
                    decay = seq.get("decay")
                    extras = seq.get("extras", [])

                    f.write(f"\nScheduler: {method}\n")

                    # method config
                    method_cfg = self.blend_method_dict.get(method, {})
                    f.write("Method settings:\n")
                    try:
                        f.write(json.dumps(method_cfg, indent=2))
                    except TypeError:
                        f.write(str(method_cfg))
                    f.write("\n")

                    # sigma preview
                    if isinstance(sigmas, torch.Tensor):
                        flat = sigmas.detach().flatten().cpu()
                        f.write(f"Sigma count: {flat.numel()}\n")
                        if flat.numel() > 0:
                            f.write(f"First sigma: {flat[0].item():.10f}\n")
                            f.write(f"Last sigma: {flat[-1].item():.10f}\n")
                            f.write(
                                f"First 5 sigmas: {[float(x) for x in flat[:5]]}\n"
                            )
                            f.write(
                                f"Last 5 sigmas: {[float(x) for x in flat[-5:]]}\n"
                            )
                    else:
                        f.write(f"Sigmas: {sigmas}\n")

                    if tails is not None:
                        if isinstance(tails, torch.Tensor):
                            f.write(f"Tail count: {tails.numel()}\n")
                        else:
                            f.write(f"Tails: {tails}\n")

                    if decay is not None:
                        if isinstance(decay, torch.Tensor):
                            f.write(f"Decay count: {decay.numel()}\n")
                        else:
                            f.write(f"Decay: {decay}\n")

                    # extras
                    if extras:
                        f.write("Extras:\n")
                        try:
                            f.write(json.dumps(extras, indent=2))
                        except TypeError:
                            f.write(str(extras))
                        f.write("\n")
                    else:
                        f.write("Extras: []\n")

                f.write("\n============================\n")
        except Exception as e:
            self.log(f"[Extras Log Warning] Failed to write extras log: {e}")

    def _safe_sigma_loader(self, cache_key):
        cache_folder = self.sigma_save_folder
        if not os.path.exists(cache_folder) or not os.listdir(cache_folder):
            self.log(f"[Cache Check] Cache folder {cache_folder} is empty or missing. Skipping load.")
            return None

        matching_files = [f for f in os.listdir(cache_folder) if cache_key in f and f.endswith('.pt')]
        if not matching_files:
            self.log(f"[Cache Check] No matching cache file found for key: {cache_key}. Skipping load.")
            return None

        filename = os.path.join(cache_folder, matching_files[0])
        self.log(f"[Cache Hit] Loading sigma cache from: {filename}")
        loaded_data = torch.load(filename, map_location=self.device)
        return loaded_data['sigma_values'].to(self.device)

    def call_scheduler(self, method_name, *args, **kwargs):
        sigma_sequence = getattr(self, f"sigmas_{method_name}")
        if sigma_sequence is None:
            self.log(f"No sigma sequence found for method: {method_name}")
            return None
        return sigma_sequence

    def is_sigma_randomized(self):
        return (
            self.settings.get('sigma_min_rand', False) or
            self.settings.get('sigma_max_rand', False) or
            self.settings.get('rho_rand', False) or
            self.settings.get('sigma_max_enable_randomization_type', False) or
            self.settings.get('sigma_min_enable_randomization_type', False) or
            self.settings.get('rho_enable_randomization_type', False)
        )

    def save_sigmas_as_csv(self, sigmas, filename):
        np.savetxt(filename, sigmas.cpu().numpy(), delimiter=",")

    def build_sigma_cache_filename(self, steps, sigma_min, sigma_max, rho=None, schedule_type='karras', decay_pattern='zero', cache_dir=None, suffix=None, ext=None):
        ext = ext or 'txt'
        if cache_dir is None:
            cache_dir = self.cache_dir or 'cache'
        if schedule_type == 'karras':
            base_filename = f'sigma_{schedule_type}_{steps}steps_rho{rho}_min{sigma_min}_max{sigma_max}_{decay_pattern}'
        else:
            base_filename = f'sigma_{schedule_type}_{steps}steps_min{sigma_min}_max{sigma_max}_{decay_pattern}'

        if suffix:
            base_filename += f'_{suffix}'
            version = self.get_next_version_number(cache_dir, base_filename, ext)
            filename = f'{version:03d}_{base_filename}.{ext}'
        else:
            filename = f'{base_filename}.{ext}'

        return os.path.join(cache_dir, filename)

    def get_next_version_number(self, cache_dir, base_filename, ext=None):
        pattern = os.path.join(cache_dir, f'*_{base_filename}')
        if ext:
            pattern = os.path.join(cache_dir, f'*_{base_filename}.{ext}')
        existing_files = glob.glob(pattern)

        version_numbers = []
        for file in existing_files:
            match = re.search(r'(\d{3})_' + re.escape(base_filename), os.path.basename(file))
            if match:
                version_numbers.append(int(match.group(1)))

        return max(version_numbers) + 1 if version_numbers else 1

    def get_sigma_with_cache(self, steps, sigma_min, sigma_max, rho=7.0, device='cpu',
                             schedule_type='karras', decay_pattern=None, cache_dir=None, cache_file=None,
                             suffix=None, ext=None, mode=None, cache_key=None):
        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.device = device
        self.schedule_type = schedule_type
        self.decay_pattern = decay_pattern
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.suffix = suffix
        self.ext = ext
        self.mode = mode
        self.cache_key = cache_key

        cached_sigmas = self.get_sigma_from_cache(cache_key)
        if cached_sigmas is not None:
            return cached_sigmas

        if self.is_sigma_randomized():
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        if self.loaded_sigmas is None:
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.loaded_sigmas = sigmas
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        if mode == 'prepass':
            self.cache_file = self.prepass_save_file
        elif mode == 'final':
            self.cache_file = self.final_save_file
        else:
            self.cache_file = self.build_sigma_cache_filename(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, cache_dir)

        if mode in ['prepass', 'final'] and self.load_prepass_sigmas:
            loaded_sigmas = self.load_sigmas_with_hash_validation(
                filename=self.cache_file,
                steps=steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                device=device,
                schedule_type=schedule_type,
                decay_pattern=decay_pattern,
                cache_key=cache_key
            )

            if loaded_sigmas is not None:
                self.loaded_sigmas = loaded_sigmas
                self.sigma_cache[cache_key] = loaded_sigmas
                return loaded_sigmas.to(device)

            self.log("[Cache Recovery] Cache load failed. Recalculating sigma schedule.")
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
        self.sigma_cache[cache_key] = sigmas
        return sigmas

    def load_sigmas_with_hash_validation(self, filename, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, cache_key=None, suffix=None):
        if self.load_prepass_sigmas and cache_key:
            try:
                loaded_data = torch.load(filename, map_location=self.device)
                self.loaded_sigmas = loaded_data['sigma_values'].to(self.device)
                loaded_hash = loaded_data['sigma_hash']
                expected_hash = self.generate_sigma_hash(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data, suffix)

                if loaded_hash != expected_hash:
                    self.log(f"[Sigma Validator] Hash mismatch. Expected: {expected_hash}, Found: {loaded_hash}. Recalculating.")
                    return None

                self.log(f"[Sigma Validator] Hash validated successfully for file: {filename}")
                return self.loaded_sigmas
            except Exception:
                self.log("[Cache Recovery] Sigma cache invalid or missing. Recalculating sigmas.")
                _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
                return sigmas
        return None

    def generate_sigma_hash(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, suffix=None):
        data_string = f'{steps}_{sigma_min}_{sigma_max}_{rho}_{device}_{schedule_type}_{decay_pattern}_{suffix}'
        hash_object = hashlib.sha256(data_string.encode())
        return hash_object.hexdigest()[:12]

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

    def initialize_generation_filename(self, folder=None, base_name="generation_log", ext="txt"):
        if folder is None:
            folder = self._resolve_data_directory(
                self.settings.get("log_save_directory", "image_generation_data")
            )
        else:
            folder = self._resolve_data_directory(folder)

        os.makedirs(folder, exist_ok=True)
        # Microseconds prevent two images prepared in the same second from
        # overwriting one another's final plot or generation log.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_filename = os.path.join(folder, f"{base_name}_{timestamp}.{ext}")

    def save_generation_settings(self):
        with open(self.log_filename, "w", encoding='utf-8') as f:
            for line in self.logger.log_buffer:
                f.write(f"{line}\n")
            for line in self.logger.prepass_log_buffer:
                f.write(f"{line}\n")
        self.log(f"[SimpleKEScheduler] Generation settings saved to {self.log_filename}")
        self.logger.log_buffer.clear()
        self.logger.prepass_log_buffer.clear()

    def save_image_plot(self, sigs, stopping_index):
        """Save the completed sigma schedule once without opening a window."""

        graph_plot = plot_sigma_sequence(
            sigs,
            stopping_index,
            self.log_filename,
            self.graph_save_directory,
            False,
        )
        self.log(f"Final sigma sequence plot saved to {graph_plot}")
        return graph_plot

    def apply_global_randomization(self):
        """Enable only the scheduler's declared randomization controls.

        The global switch never derives or stretches ranges. It only turns on
        the known min/max controls already present in the effective settings.
        """
        for base_key in KES_RANDOMIZATION_SAFE_BOUNDS:
            min_key = f"{base_key}_rand_min"
            max_key = f"{base_key}_rand_max"
            if min_key in self.settings and max_key in self.settings:
                self.settings[f"{base_key}_rand"] = True

    def _effective_randomization_bounds(self, key_prefix, default_value):
        safe_min, safe_max = KES_RANDOMIZATION_SAFE_BOUNDS.get(
            key_prefix,
            (float(default_value), float(default_value)),
        )
        try:
            rand_min = float(self.settings.get(f"{key_prefix}_rand_min", safe_min))
        except (TypeError, ValueError):
            rand_min = float(safe_min)
        try:
            rand_max = float(self.settings.get(f"{key_prefix}_rand_max", safe_max))
        except (TypeError, ValueError):
            rand_max = float(safe_max)

        if not math.isfinite(rand_min):
            rand_min = float(safe_min)
        if not math.isfinite(rand_max):
            rand_max = float(safe_max)
        if rand_min > rand_max:
            rand_min, rand_max = rand_max, rand_min

        override_enabled = bool(self.settings.get("allow_randomization_range_override", False))
        if not override_enabled:
            rand_min = max(rand_min, float(safe_min))
            rand_max = min(rand_max, float(safe_max))
            if rand_min > rand_max:
                rand_min, rand_max = float(safe_min), float(safe_max)

        return rand_min, rand_max

    def _clamp_randomized_value(self, key_prefix, value, default_value):
        rand_min, rand_max = self._effective_randomization_bounds(key_prefix, default_value)
        clamped = min(max(float(value), rand_min), rand_max)
        if clamped != float(value):
            self.log(
                f"[Randomization Safety] {key_prefix}: clamped {value} to {clamped} "
                f"inside configured range {rand_min} to {rand_max}."
            )
        return clamped

    def get_randomization_type(self, key_prefix):
        randomization_type_raw = self.settings.get(f'{key_prefix}_randomization_type', 'asymmetric')
        return self.RANDOMIZATION_TYPE_ALIASES.get(str(randomization_type_raw).lower(), 'asymmetric')

    def get_randomization_percent(self, key_prefix):
        try:
            value = float(self.settings.get(f'{key_prefix}_randomization_percent', 0.2))
        except (TypeError, ValueError):
            value = 0.2
        return max(0.0, value)

    def get_random_between_min_max(self, key_prefix, default_value):
        randomize_flag = self.settings.get(f'{key_prefix}_rand', False)
        if randomize_flag:
            rand_min, rand_max = self._effective_randomization_bounds(key_prefix, default_value)
            if rand_min == rand_max:
                self.log(f"[Random Range] {key_prefix}: min and max are equal ({rand_min}). Using single value.")
                return rand_min
            value = random.uniform(rand_min, rand_max)
            self.log(f"[Random Range] {key_prefix}: Picked random value {value} between {rand_min} and {rand_max}")
            return value
        self.log(f"[Random Range] {key_prefix}: Randomization is OFF. Using base value {default_value}")
        return default_value

    def get_random_by_type(self, key_prefix, default_value):
        randomization_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)
        if not randomization_enabled:
            self.log(f"[Randomization Type] {key_prefix}: Randomization type is OFF. Using base value {default_value}")
            return default_value

        randomization_type = self.get_randomization_type(key_prefix)
        randomization_percent = self.get_randomization_percent(key_prefix)
        base_value = float(default_value)

        if randomization_type == 'symmetric':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + randomization_percent)
            value = random.uniform(candidate_min, candidate_max)
            self.log(f"[Symmetric Randomization] {key_prefix}: Range {candidate_min} to {candidate_max}")
        elif randomization_type == 'asymmetric':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + (randomization_percent * 2))
            value = random.uniform(candidate_min, candidate_max)
            self.log(f"[Asymmetric Randomization] {key_prefix}: Range {candidate_min} to {candidate_max}")
        elif randomization_type == 'logarithmic':
            positive_floor = max(base_value * (1 - randomization_percent), 1e-12)
            positive_ceiling = max(base_value * (1 + randomization_percent), positive_floor)
            value = math.exp(random.uniform(math.log(positive_floor), math.log(positive_ceiling)))
            self.log(f"[Logarithmic Randomization] {key_prefix}: Log-space randomization resulted in {value}")
        elif randomization_type == 'exponential':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + randomization_percent)
            try:
                value = math.exp(random.uniform(candidate_min, candidate_max))
            except OverflowError:
                _range_min, range_max = self._effective_randomization_bounds(key_prefix, default_value)
                value = range_max
                self.log(
                    f"[Exponential Randomization] {key_prefix}: exponential candidate overflowed; "
                    f"using configured maximum {range_max}."
                )
            else:
                self.log(f"[Exponential Randomization] {key_prefix}: Randomized exponential value {value}")
        else:
            self.log(f"[Randomization Type] {key_prefix}: Invalid randomization type {randomization_type}. Using base value.")
            value = base_value

        return self._clamp_randomized_value(key_prefix, value, default_value)

    def _apply_runtime_randomization(self):
        for key in self.re_randomizable_keys:
            base_value = self.settings.get(key, getattr(self, key, None))
            if base_value is None:
                continue
            runtime_value = self.get_random_or_default(key, base_value)
            setattr(self, key, runtime_value)
            self.settings[key] = runtime_value

    def get_random_or_default(self, key_prefix, default_value):
        rand_type_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)
        min_max_enabled = self.settings.get(f'{key_prefix}_rand', False)

        if self.global_randomize:
            result_value = self.get_random_between_min_max(key_prefix, default_value)
            self.log(
                f"[Global Randomization] {key_prefix}: used the configured min/max range. "
                f"Final value: {result_value}"
            )
        elif rand_type_enabled and min_max_enabled:
            self.log(
                f"[Randomization Policy] Both min/max and randomization type enabled for {key_prefix}. "
                "The randomization type is applied and then clamped to the configured min/max range."
            )
            result_value = self.get_random_by_type(key_prefix, default_value)
        elif rand_type_enabled:
            result_value = self.get_random_by_type(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied randomization type. Final value: {result_value}")
        elif min_max_enabled:
            result_value = self.get_random_between_min_max(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied min/max randomization. Final value: {result_value}")
        else:
            result_value = default_value
            self.log(f"[Randomization] {key_prefix}: No randomization applied. Using default value: {result_value}")

        return result_value

    def resolve_blend_weights(self, blend_weights, blending_style):
        if blending_style == 'softmax':
            blend_weights = torch.tensor(blend_weights)
            normalized_weights = torch.softmax(blend_weights, dim=0)
            return normalized_weights.tolist()
        if blending_style == 'explicit':
            return blend_weights
        raise ValueError(f"Unknown blending_style: {blending_style}")

    def extract_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() > 1:
                return value.mean().item()
            return value.item()
        return value

    def _call_legacy_mode(self, schedule_type):
        if schedule_type not in ['karras', 'exponential']:
            self.log(f"[Legacy Mode] Unsupported schedule_type: {schedule_type}")
            return

        target_attr = f"sigmas_{schedule_type}"
        scheduler_func = self.scheduler_registry.get(schedule_type)

        tails, decay, extras, sigmas = self.call_scheduler_function(
            scheduler_func,
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=self.device,
            decay_pattern=self.decay_pattern
        )

        setattr(self, target_attr, sigmas)
        self.log(f"[Legacy Mode] Loaded sigma sequence for {schedule_type}. Assigned to self.{target_attr}")

    def blend_sigma_sequence(self, sigmas_karras=None, sigmas_exponential=None, pre_pass=False, blend_methods=None, blend_weights=None):
        active_methods = [
            method for method, config in self.blend_method_dict.items()
            if float(config.get('weight', 0.0)) > 0.0
        ]
        if not active_methods:
            raise ValueError(
                "[SimpleKEScheduler] No active schedulers selected. "
                "At least one blend method must have a weight greater than zero."
            )

        self.blend_methods = active_methods
        self.blend_method_dict = {
            method: self.blend_method_dict[method]
            for method in active_methods
        }
        self.settings['blend_methods'] = copy.deepcopy(self.blend_method_dict)
        self.blend_weights = [self.blend_method_dict[m]['weight'] for m in self.blend_methods]

        if len(self.blend_methods) == 1:
            self.log(f"[Blend] Only one active scheduler: {self.blend_methods[0]}. Skipping blending, using it directly.")
            self.sigs = self.sigma_sequences[self.blend_methods[0]]['sigmas']

        if self.blending_mode in {'default', 'auto'}:
            self.blending_mode = 'weights'
        elif self.blending_mode == 'smooth_blend' and len(self.blend_methods) != 2:
            self.log("[Blend] smooth_blend requires exactly two active methods; using weighted blending.")
            self.blending_mode = 'weights'
        self.settings['blending_mode'] = self.blending_mode

        if not self.allow_step_expansion and self.auto_mode_enabled:
            self.auto_mode_enabled = False
            self.log("[Auto Mode] Step expansion disallowed. Auto mode forcibly disabled.")

        self.progress = torch.linspace(0, 1, len(self.sigs), device=self.device)
        self.blended_sigmas = []
        self.change_log = []
        self.relative_converged = False
        self.max_converged = False
        self.delta_converged = False
        self.early_stop_triggered = False

        # Weighted and smooth blending consume the active method sequences from
        # ``self.sigma_sequences``. Do not manufacture legacy Karras/Exponential
        # sequences when either method has a zero weight; zero-weight methods must
        # remain excluded from both generation and blending.
        self.prepass_blended_sigmas = []
        self.blended_sigma = None
        self.blended_sigmas = []
        for i in range(len(self.sigs)):
            if self.step_progress_mode == "linear":
                progress_value = self.progress[i]
            elif self.step_progress_mode == "exponential":
                progress_value = self.progress[i] ** self.exp_power
            elif self.step_progress_mode == "logarithmic":
                progress_value = torch.log1p(self.progress[i] * (torch.exp(torch.tensor(1.0, device=self.device)) - 1))
            elif self.step_progress_mode == "sigmoid":
                progress_value = 1 / (1 + torch.exp(-12 * (self.progress[i] - 0.5)))
            else:
                progress_value = self.progress[i]

            self.dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
            self.smooth_blend = torch.sigmoid((self.dynamic_blend_factor - self.blend_midpoint) * self.smooth_blend_factor)
            self.noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor
            self.step_size = self.initial_step_size * (1 - progress_value) + self.final_step_size * progress_value * self.step_size_factor

            if self.blending_mode == 'default':
                self.blended_sigma = self.sigmas_karras[i] * (1 - self.smooth_blend) + self.sigmas_exponential[i] * self.smooth_blend

            if self.blending_mode == 'smooth_blend' or (self.blending_mode == 'auto' and len(self.blend_methods) == 2):
                sigma_seq_a = self.sigma_sequences[self.blend_methods[0]]['sigmas']
                sigma_seq_b = self.sigma_sequences[self.blend_methods[1]]['sigmas']
                self.blended_sigma = sigma_seq_a[i] * (1 - self.smooth_blend) + sigma_seq_b[i] * self.smooth_blend

            elif self.blending_mode == 'weights' or (self.blending_mode == 'auto' and len(self.blend_methods) > 2):
                if self.blend_weights is None:
                    self.blend_weights = [1.0] * len(self.all_sigmas)
                if self.blending_style is None:
                    self.blending_style = 'softmax'
                resolved_blend_weights = self.resolve_blend_weights(self.blend_weights, self.blending_style)
                weighted_sum = sum(w * self.extract_scalar(s[i]) for w, s in zip(resolved_blend_weights, self.all_sigmas))
                total_weight = sum(resolved_blend_weights)
                self.blended_sigma = weighted_sum / total_weight

            for s in self.all_sigmas:
                self.log(f"[DEBUG]sigma sequence shape: {s.shape}")

            self.sigs[i] = self.blended_sigma * self.step_size * self.noise_scale
            self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
            self.change_log.append(self.extract_scalar(self.change))
            relative_sigma_progress = (self.blended_sigma - self.sigs[-1].item()) / self.blended_sigma
            recent_changes = torch.abs(torch.tensor(self.change_log[-5:]))
            max_change = torch.max(recent_changes).item()
            mean_change = torch.mean(recent_changes).item()
            self.delta_change = abs(max_change - mean_change)
            self.blended_sigmas.append(self.extract_scalar(self.blended_sigma))

            self.relative_converged = relative_sigma_progress < 0.05
            self.max_converged = max_change < self.early_stopping_threshold
            self.delta_converged = self.delta_change < self.recent_change_convergence_delta

            if pre_pass:
                self.prepass_blended_sigmas = self.blended_sigmas.copy()
                self.prepass_blended_sigma = self.blended_sigma
                if i >= 2:
                    sigma_rate = abs(self.prepass_blended_sigmas[i] - self.prepass_blended_sigmas[i - 1])
                    previous_sigma_rate = abs(self.prepass_blended_sigmas[i - 1] - self.prepass_blended_sigmas[i - 2])
                    if sigma_rate > previous_sigma_rate:
                        self.prepass_log(f"Sigma decline is slowing down → possible plateau at step {i+1}.")

                if i == 0:
                    self.prepass_log("\n--- Starting Pre-Pass Blending ---\n")
                    step_label = "Prepass First Step"
                elif i == len(self.sigs) - 1:
                    step_label = "Prepass Last Step"
                else:
                    step_label = None

                if step_label:
                    self.prepass_log(f"[{step_label} - Step {i}/{len(self.sigs)}] Prepass Blended Sigma: {self.prepass_blended_sigma:.6f}, Final Sigma: {self.sigs[i]:.6f}")
                    self.prepass_log(f"{step_label} Delta Converged: {self.delta_converged} delta_change: {self.delta_change:.6f}, Target Default Settings:{self.recent_change_convergence_delta}")

                if i > self.safety_minimum_stop_step and len(self.change_log) > 10:
                    self.blended_tensor = torch.tensor(self.prepass_blended_sigmas)
                    if self.device == 'cpu':
                        self.sigma_variance = np.var(self.prepass_blended_sigmas)
                    else:
                        self.sigma_variance = torch.var(self.sigs).item()

                    self.min_sigma_threshold = self.sigma_variance * self.sigma_variance_scale
                    self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i} ---")
                    self.prepass_log(f"Current Blended Prepass Sigma: {self.prepass_blended_sigma:.6f}")
                    self.prepass_log(f"Sigma Variance: {self.sigma_variance:.6f}")
                    self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                    self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                    self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")

                    if self.prepass_blended_sigma > self.min_sigma_threshold:
                        self.prepass_log(f"Prepass Blended Sigma {self.prepass_blended_sigma:.6f} exceeds min sigma threshold {self.min_sigma_threshold:.6f} → Continuing.\n")

                    if self.early_stopping_method == "mean":
                        mean_change = sum(self.change_log) / len(self.change_log)
                        if mean_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {mean_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")
                    elif self.early_stopping_method == "max":
                        if max_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {max_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")
                    elif self.early_stopping_method == "sum":
                        stable_steps = sum(
                            1 for j in range(1, len(self.change_log))
                            if abs(self.change_log[j]) < self.early_stopping_threshold * abs(self.sigs[j])
                        )
                        if stable_steps >= 0.8 * len(self.change_log):
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by sum at step {i}. Stable steps: {stable_steps}/{len(self.change_log)}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")

                    if self.relative_converged and self.max_converged and self.delta_converged:
                        self.early_stop_triggered = True
                        self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i+1} ---")
                        self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                        self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                        self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")
                        self.prepass_log(f"Delta Change: {self.delta_change:.6f} (Target: {self.recent_change_convergence_delta})")
                        self.prepass_log(f"Early stopping criteria met at step {i+1} based on all convergence checks.")
                        self.predicted_stop_step = i
                        break

            if not pre_pass:
                if i == 0:
                    step_label = "First Step"
                    self.log("\n" + "=" * 10 + "\n[Start of Sigma Sequence Logging]\n" + "=" * 10)
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) // 2:
                    step_label = "Middle Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) - 1:
                    step_label = "Last Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                    self.log("\n" + "=" * 10 + "\n[End of Sigma Sequence Logging]\n" + "=" * 10)

                if i > 0:
                    self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
                    self.change_log.append(self.extract_scalar(self.change))

                if i > self.safety_minimum_stop_step and len(self.change_log) > 5:
                    final_target_sigma = self.sigs[-1].item()
                    if self.blended_sigma != 0:
                        relative_sigma_progress = (self.blended_sigma - final_target_sigma) / self.blended_sigma
                    else:
                        relative_sigma_progress = 0
                    self.sigma_variance = torch.var(self.sigs).item() if self.device != 'cpu' else np.var(self.blended_sigmas)
                    self.log(f"Sigma Variance: {self.sigma_variance:.6f}")

        num_methods = len(self.all_sigmas)
        if not self.blend_weights or len(self.blend_weights) != num_methods:
            self.blend_weights = [1.0] * num_methods

        if not getattr(self, "blending_style", None):
            self.blending_style = "softmax"

        resolved_blend_weights = self.resolve_blend_weights(self.blend_weights, self.blending_style)

        s = sum(resolved_blend_weights)
        if s <= 0:
            resolved_blend_weights = [1.0] * num_methods
            s = float(num_methods)
        resolved_blend_weights = [w / s for w in resolved_blend_weights]

        if not self.auto_mode_enabled:
            if not pre_pass:
                if self.apply_tail_steps:
                    for i, tail in enumerate(self.all_tails):
                        if tail is not None:
                            self.log(f"Appending tail from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, tail])
                            self._tail_steps_applied = True
                            self._step_expansion_applied = True

                if self.apply_decay_tail:
                    for i, decay in enumerate(self.all_decays):
                        if decay is not None:
                            self.log(f"Appending decay from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, decay])
                            self._decay_tail_applied = True
                            self._step_expansion_applied = True

                if self.apply_progressive_decay and any(self.all_decays):
                    progressive_decay = None
                    total_weight = 0.0
                    for w, decay in zip(resolved_blend_weights, self.all_decays):
                        if decay is not None:
                            decay = decay[:len(self.sigs)]
                            progressive_decay = (w * decay) if progressive_decay is None else (progressive_decay + w * decay)
                            total_weight += w
                    if progressive_decay is not None and total_weight > 0:
                        progressive_decay /= total_weight
                        self.sigs = self.sigs * progressive_decay
                        self._progressive_decay_applied = True

                if self.apply_blended_tail and any(self.all_tails):
                    blended_tail = None
                    total_weight = 0.0
                    for w, tail in zip(resolved_blend_weights, self.all_tails):
                        if tail is not None:
                            blended_tail = (w * tail) if blended_tail is None else (blended_tail + w * tail)
                            total_weight += w
                    if blended_tail is not None and total_weight > 0:
                        blended_tail /= total_weight
                        self.sigs = torch.cat([self.sigs, blended_tail])
                        self._blended_tail_applied = True
                        self._step_expansion_applied = True
        else:
            if len(self.sigs) > self.steps:
                self.auto_stabilization_sequence = []
                self.log(f"[Auto Mode] Sigma sequence length {len(self.sigs)} exceeds requested steps {self.steps}. Disabling auto stabilization.")
                self.auto_mode_enabled = False
                self.sigs = self.sigs[:self.steps]
                return self.sigs
            self.run_auto_stabilization(self.sigs)

        if pre_pass and self.early_stop_triggered:
            return self.sigs[:self.predicted_stop_step]
        return self.sigs

    def run_auto_stabilization(self, *_args):
        if not self.allow_step_expansion:
            self.log("[Auto Mode] Step expansion is disabled by configuration. Skipping auto stabilization.")
            return self.sigs
        unstable = self.detect_sequence_instability()

        if not unstable:
            self.log("[Auto Mode] Sigma sequence is already stable.")
            return

        self.log("[Auto Mode] Detected instability in sigma sequence. Starting stabilization sequence.")

        for method in self.auto_stabilization_sequence:
            if not unstable:
                self.log(f"[Auto Mode] Sequence stabilized after {method}. Stopping further corrections.")
                break

            if method == 'smooth_interpolation':
                unstable = self.smooth_interpolation()
            elif method == 'append_tail':
                unstable = self.append_tail()
            elif method == 'blend_tail':
                unstable = self.blend_tail()
            elif method == 'apply_decay':
                unstable = self.apply_decay()
            elif method == 'progressive_decay':
                unstable = self.progressive_decay()
            else:
                self.log(f"[Auto Mode] Unknown stabilization method: {method}")

    def detect_sequence_instability(self):
        delta_sigmas = self.sigs[:-1] - self.sigs[1:]
        second_deltas = torch.diff(delta_sigmas)

        steep_drop_detected = torch.any(delta_sigmas > self.auto_tail_threshold)
        jaggedness_score = torch.var(second_deltas[-5:]) if len(second_deltas) >= 5 else 0
        jagged_transition_detected = jaggedness_score > self.jaggedness_threshold

        if steep_drop_detected:
            self.log(f"[Auto Mode] Steep drop detected. Max drop: {torch.max(delta_sigmas).item():.6f}")
        if jagged_transition_detected:
            self.log(f"[Auto Mode] Jagged transition detected. Jaggedness score: {jaggedness_score:.6f}")

        return steep_drop_detected or jagged_transition_detected

    def smooth_interpolation(self):
        self.log("[Auto Mode] Applying smooth interpolation to last 5 steps.")
        if len(self.sigs) >= 5:
            start = self.sigs[-6].item()
            end = self.sigs[-1].item()
            interpolated = torch.linspace(start, end, steps=6, device=self.device)[1:]
            self.sigs[-5:] = interpolated

        return self.detect_sequence_instability()

    def append_tail(self):
        self.log("[Auto Mode] Attempting to append available tail.")
        if hasattr(self, 'all_tails') and self.all_tails:
            for tail in self.all_tails:
                if tail is not None:
                    tail = tail.to(self.device)
                    if tail.shape[0] > self.sigs.shape[0]:
                        tail = tail[:len(self.sigs)]
                    self.sigs = torch.cat([self.sigs, tail])
                    self.log("[Auto Mode] Appended tail to sigma sequence.")
                    self._auto_stabilization_applied = True
                    self._tail_steps_applied = True
                    self._step_expansion_applied = True
                    break

        return self.detect_sequence_instability()

    def blend_tail(self):
        if not hasattr(self, 'all_tails') or not self.all_tails:
            self.log("[Auto Mode] No available tails to blend.")
            return self.detect_sequence_instability()

        self.log("[Auto Mode] Attempting to blend multiple tails.")
        blended_tail = None
        total_weight = 0

        for w, tail in zip(self.blend_weights, self.all_tails):
            if tail is not None:
                tail = tail.to(self.device)
                if tail.shape[0] > self.sigs.shape[0]:
                    tail = tail[:len(self.sigs)]

                if blended_tail is None:
                    blended_tail = w * tail
                else:
                    blended_tail += w * tail
                total_weight += w

        if blended_tail is not None and total_weight > 0:
            blended_tail /= total_weight
            self.sigs = torch.cat([self.sigs, blended_tail])
            self.log("[Auto Mode] Appended blended tail to sigma sequence.")
            self._auto_stabilization_applied = True
            self._blended_tail_applied = True
            self._step_expansion_applied = True

        return self.detect_sequence_instability()

    def apply_decay(self):
        self.log("[Auto Mode] Attempting to append decay tails.")
        if hasattr(self, 'all_decays') and self.all_decays:
            for decay in self.all_decays:
                if decay is not None:
                    decay = decay.to(self.device)
                    if decay.shape[0] > self.sigs.shape[0]:
                        decay = decay[:len(self.sigs)]

                    self.sigs = torch.cat([self.sigs, decay])
                    self.log("[Auto Mode] Appended decay tail to sigma sequence.")
                    self._auto_stabilization_applied = True
                    self._decay_tail_applied = True
                    self._step_expansion_applied = True
                    break

        return self.detect_sequence_instability()

    def progressive_decay(self):
        self.log("[Auto Mode] Applying progressive decay to sigma sequence.")
        progressive_decay = None
        total_weight = 0

        for w, decay in zip(self.blend_weights, self.all_decays):
            if decay is not None:
                decay = decay.to(self.device)
                if decay.shape[0] != self.sigs.shape[0]:
                    decay = decay.view(1, 1, -1)
                    decay = F.interpolate(decay, size=self.sigs.shape[0], mode='linear', align_corners=False)
                    decay = decay.view(-1)

                if progressive_decay is None:
                    progressive_decay = w * decay
                else:
                    progressive_decay += w * decay

                total_weight += w

        if progressive_decay is not None and total_weight > 0:
            progressive_decay /= total_weight
            self.sigs = self.sigs * progressive_decay
            self.log("[Auto Mode] Applied progressive decay to sigma sequence.")
            self._auto_stabilization_applied = True
            self._progressive_decay_applied = True

        return self.detect_sequence_instability()

    def load_blend_method_sigmas(self, mode=None):
        self.all_sigmas = []
        shared_tail_config = {
            'decay_pattern': self.decay_pattern if self.decay_pattern is not None else 'zero',
            'decay_mode': self.decay_mode if self.decay_mode is not None else 'blend',
            'tail_steps': self.tail_steps if self.tail_steps is not None else 1,
        }

        for method in self.blend_methods:
            self.method_config = dict(self.blend_method_dict.get(method) or {})
            self.current_config = dict(shared_tail_config)

            tails, decay, extras, sigmas = self.call_scheduler_function(
                self.scheduler_registry.get(method),
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                device=self.device,
                decay_pattern=self.current_config['decay_pattern'],
                decay_mode=self.current_config['decay_mode'],
                tail_steps=self.current_config['tail_steps']
            )
            if self.debug:
                if isinstance(sigmas, torch.Tensor):
                    flat = sigmas.detach().flatten().cpu()
                    if flat.numel() > 0:
                        self.log(
                            f"[Raw scheduler {method}] count={flat.numel()}, "
                            f"first={flat[0].item():.10f}, last={flat[-1].item():.10f}"
                        )
                else:
                    self.log(
                        f"[Raw scheduler {method}] non-tensor result: {type(sigmas)}"
                    )
            self.sigma_sequences[method] = {
                'sigmas': sigmas,
                'tails': tails,
                'decay': decay,
                'extras': extras
            }
            setattr(self, f"sigmas_{method}", sigmas)

        self.all_sigmas = [self.sigma_sequences[method]['sigmas'] for method in self.blend_methods]
        self.all_tails = [self.sigma_sequences[method]['tails'] for method in self.blend_methods]
        self.all_decays = [self.sigma_sequences[method]['decay'] for method in self.blend_methods]
        self.all_extras = [self.sigma_sequences[method].get('extras', []) for method in self.blend_methods]
        self._log_extras_to_file(self.all_extras)

        norm = []
        for i, s in enumerate(self.all_sigmas):
            if isinstance(s, list):
                s = torch.tensor(s, dtype=torch.float32, device=self.device)
            elif isinstance(s, torch.Tensor):
                s = s.detach().to(self.device, dtype=torch.float32).flatten()
            else:
                raise TypeError(f"Sigma sequence for {self.blend_methods[i]} must be list or Tensor, got {type(s)}")
            norm.append(s)

        maxlen = max(len(s) for s in norm)
        for i, s in enumerate(norm):
            if len(s) < maxlen:
                pad = s[-1].repeat(maxlen - len(s))
                norm[i] = torch.cat([s, pad])

        self.all_sigmas = norm
        self.log(f"Loaded sigma schedules for blend methods: {self.blend_methods} using mode: {mode}")

    def validate_and_align_sigmas(self):
        if not self.all_sigmas or len(self.all_sigmas) == 0:
            raise ValueError("No sigma sequences were loaded for blending.")

        target_length = max(len(s) for s in self.all_sigmas)

        for idx, sigmas in enumerate(self.all_sigmas):
            if sigmas is None or len(sigmas) == 0:
                raise ValueError(f"Sigma sequence at index {idx} is invalid or empty: {sigmas}")

            if len(sigmas) < target_length:
                padding = torch.full((target_length - len(sigmas),), sigmas[-1]).to(sigmas.device)
                self.all_sigmas[idx] = torch.cat([sigmas, padding])

        self.log(f"Validated and aligned all sigma sequences to length {target_length}.")

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

    def load_or_regenerate_sigmas(self, cache_key):
        if self.load_sigma_cache and cache_key:
            try:
                loaded_data = torch.load(self.cache_file, map_location=self.device)
                sigmas = loaded_data['sigma_values'].to(self.device)
                return sigmas
            except FileNotFoundError:
                self.log(f"[Cache Warning] Cache file not found: {self.cache_file}")
                self.log(f"[Cache Recovery] Automatically recomputing sigma schedule.")

        _, _, _, sigmas = self._generate_sigmas(
            self.steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            self.device,
            self.schedule_type,
            self.decay_pattern
        )
        return sigmas

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

    def get_sigma_from_cache(self, cache_key):
        if cache_key in self.sigma_cache:
            cached_sigmas = self.sigma_cache[cache_key]
            self.log(f"[Cache Hit] Returning cached sigma sequence for key: {cache_key}")

            if isinstance(cached_sigmas, torch.Tensor):
                return cached_sigmas.clone().detach().to(self.device)
            elif isinstance(cached_sigmas, list):
                return copy.deepcopy(cached_sigmas)
            return cached_sigmas

        self.log(f"[Cache Miss] Cache key not found: {cache_key}")
        return None

SCHEDULER_CLASS = SimpleKEScheduler