# modules.ss_registry.sampleres.kes_sampler.simple_kes_sampler.cfg_strategy.py

from __future__ import annotations

import math
from typing import Optional

import torch


class CFGStrategy:
    def __init__(self, shared_state=None, sampler_state=None, verbose: Optional[bool] = False):
        self.state = shared_state
        self.sampler_state = sampler_state
        self.verbose = bool(verbose)

        # ---- shared_state shortcuts (optional)
        self.cfgm_state = getattr(shared_state, "cfgm", None)
        self.p_state = getattr(shared_state, "p", None)
        self.cond_state = getattr(shared_state, "conditioning", None)

        # ---- sampler_state shortcuts (optional)
        self.cfg_samp = getattr(sampler_state, "cfg", None)
        self.gen_samp = getattr(sampler_state, "gen", None)

        # normalize naming drift: tensor_data vs sigdata
        self.tensor_samp = getattr(sampler_state, "tensor_data", None)
        if self.tensor_samp is None:
            self.tensor_samp = getattr(sampler_state, "sigdata", None)

        self.lsu_samp = getattr(sampler_state, "lsu", None)

        # adaptive tracking
        self.last_x = None
        self.last_denoised = None
        self.last_step = -1

    # ------------------------------------------------------------
    # safe readers
    # ------------------------------------------------------------

    def _cfg(self, name: str, default=None):
        if self.cfg_samp is not None and hasattr(self.cfg_samp, name):
            value = getattr(self.cfg_samp, name)
            if value is not None:
                return value
        return default

    def _gen(self, name: str, default=None):
        if self.gen_samp is not None and hasattr(self.gen_samp, name):
            value = getattr(self.gen_samp, name)
            if value is not None:
                return value
        return default

    # ------------------------------------------------------------
    # guidance helpers
    # ------------------------------------------------------------

    def apply_cfg_rescale(
        self,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        *,
        cfg_scale_override: float | None = None,
    ) -> torch.Tensor:
        """
        Returns guidance tensor, optionally rescaled/clamped.

        ``cfg_scale_override`` is used by Phase 11F effective-guidance shaping so
        the sampler can keep a stable requested CFG while applying an explicit
        per-step effective multiplier.
        """
        cfg_scale = float(
            cfg_scale_override if cfg_scale_override is not None else self._gen("cfg_scale", 1.0)
        )
        rescale_factor = float(self._cfg("rescale_cfg_factor", 1.0))
        rescale_cfg = bool(self._cfg("rescale_cfg", False))

        delta = cond - uncond
        guided = uncond + (cfg_scale * rescale_factor * delta)

        if rescale_cfg:
            clamp_range = self._cfg("clamp_range", [-1.0, 1.0])
            if not isinstance(clamp_range, (list, tuple)) or len(clamp_range) != 2:
                clamp_range = [-1.0, 1.0]
            clamp_min, clamp_max = float(clamp_range[0]), float(clamp_range[1])
            guided = torch.clamp(guided, clamp_min, clamp_max)

        return guided

    def apply_initial_noise(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Add deterministic optional initial noise."""
        strength = float(self._cfg("initial_noise_strength", 0.0))
        if strength > 0.0:
            noise = torch.randn(
                x.shape,
                generator=generator,
                device=x.device,
                dtype=x.dtype,
            )
            x = x + noise * strength
        return x

    # ------------------------------------------------------------
    # eta / noise schedule helpers
    # ------------------------------------------------------------

    def get_noise_schedule_scale(self, step: int, total_steps: int, mode: Optional[str] = None) -> float:
        """
        Returns a scaling factor for eta-based noise per step.
        """
        t = step / max(total_steps - 1, 1)
        mode = (mode or self._cfg("noise_schedule_scaling", "none") or "none").lower()

        if mode == "linear":
            return t
        if mode == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * t))
        if mode == "exp_decay":
            return math.exp(-5.0 * t)
        if mode == "ease_out":
            return 1.0 - ((1.0 - t) ** 3)

        # "none", unknown, or disabled
        return 1.0

    def get_eta_noise_gamma(
        self,
        step: int,
        sigma,
        sigma_next,
        total_steps: int,
        mode: Optional[str] = None,
    ) -> float:
        """
        Returns actual stochastic noise scale for this step.
        """
        eta = float(self._cfg("eta", 0.0))
        if eta <= 0.0:
            return 0.0

        eta_scale_factor = float(self._cfg("eta_scale_factor", 1.0))
        eta_schedule_mode = (self._cfg("eta_schedule_mode", "none") or "none").lower()

        # Prefer explicit mode override, otherwise use eta_schedule_mode, otherwise fall back
        schedule_mode = mode or eta_schedule_mode

        if schedule_mode == "auto":
            if total_steps <= 10:
                scale = 0.2
            elif total_steps <= 25:
                scale = 0.5
            else:
                scale = 1.0
        else:
            scale = self.get_noise_schedule_scale(step, total_steps, mode=schedule_mode)

        sigma_val = float(sigma.item() if torch.is_tensor(sigma) else sigma)
        sigma_next_val = float(sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next)

        sigma_diff_sq = max((sigma_val ** 2) - (sigma_next_val ** 2), 0.0)
        gamma = eta * eta_scale_factor * scale * math.sqrt(sigma_diff_sq)

        return float(max(gamma, 0.0))

    def get_eta_noise_gamma_adaptive(
        self,
        step: int,
        sigma,
        sigma_next,
        total_steps: int,
        x: Optional[torch.Tensor] = None,
        denoised: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Safe adaptive wrapper.

        Keeps the adaptive hook alive without depending on unfinished / unstable
        variables. It starts from the normal eta gamma, then optionally nudges it
        based on delta analysis.
        """
        base_gamma = self.get_eta_noise_gamma(
            step=step,
            sigma=sigma,
            sigma_next=sigma_next,
            total_steps=total_steps,
        )
        if base_gamma <= 0.0:
            self._save_adaptive_state(step=step, x=x, denoised=denoised)
            return 0.0

        adjustment = 1.0

        # Compare current latent movement to previous step.
        if torch.is_tensor(x) and torch.is_tensor(self.last_x):
            delta_x = torch.norm(x - self.last_x).item()

            delta_low_floor = float(self._cfg("adaptive_delta_low_floor", 0.1))
            delta_high_floor = float(self._cfg("adaptive_delta_high_floor", 1.0))
            low_adjustment_multiplier = float(self._cfg("adaptive_low_adjustment_multiplier", 1.5))
            high_adjustment_multiplier = float(self._cfg("adaptive_high_adjustment_multiplier", 0.5))

            if delta_x < delta_low_floor:
                adjustment *= low_adjustment_multiplier
            elif delta_x > delta_high_floor:
                adjustment *= high_adjustment_multiplier

        # Compare denoised output stability.
        if torch.is_tensor(denoised) and torch.is_tensor(self.last_denoised):
            delta_d = torch.norm(denoised - self.last_denoised).item()
            denoised_floor = float(self._cfg("adaptive_denoised_floor", 0.05))
            denoised_adjustment_multiplier = float(self._cfg("adaptive_denoised_adjustment_multiplier", 1.25))

            if delta_d < denoised_floor:
                adjustment *= denoised_adjustment_multiplier

        # Optional time shaping
        time_mode = (self._cfg("adaptive_time_mode", "none") or "none").lower()
        t = step / max(total_steps - 1, 1)

        if time_mode == "time_boost":
            adjustment *= (1.0 + (1.0 - t))
        elif time_mode == "time_curve":
            adjustment *= 0.5 * (1.0 + math.cos(math.pi * t))
        elif time_mode == "sigma":
            sigma_val = float(sigma.item() if torch.is_tensor(sigma) else sigma)
            if sigma_val < 0.1:
                adjustment *= 0.5
        elif time_mode == "manual":
            low_adjustment = float(self._cfg("adaptive_manual_low_adjustment", 1.0))
            high_adjustment = float(self._cfg("adaptive_manual_high_adjustment", 1.0))
            adjustment *= (low_adjustment * high_adjustment)

        gamma = float(max(base_gamma * adjustment, 0.0))

        self._save_adaptive_state(step=step, x=x, denoised=denoised)
        return gamma

    # ------------------------------------------------------------
    # adaptive tracking
    # ------------------------------------------------------------

    def _save_adaptive_state(self, step: int, x=None, denoised=None) -> None:
        self.last_x = x.detach() if torch.is_tensor(x) else None
        self.last_denoised = denoised.detach() if torch.is_tensor(denoised) else None
        self.last_step = step