"""Archived advanced KES sampler helpers.

The original research file mixed tabs and spaces and could not be compiled.
It is still not part of the active sampler path, but the preserved helpers are
now valid, importable Python so repository-wide static checks remain useful.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any


class SamplerAdvanced:
    """Optional helper methods for future advanced KES sampler experiments."""

    def __init__(self, shared_state: Any = None, sampler_state: Any = None) -> None:
        self.state: Any = None
        self.sampler_state: Any = None
        self.cfg: Any = None
        self.torchp: Any = None
        self.gen: Any = None
        if shared_state is not None or sampler_state is not None:
            self.init(shared_state=shared_state, sampler_state=sampler_state)

    def init(self, shared_state: Any = None, sampler_state: Any = None) -> "SamplerAdvanced":
        self.state = shared_state
        self.sampler_state = sampler_state
        self.cfg = getattr(sampler_state, "cfg", None)
        self.torchp = getattr(sampler_state, "torchp", None)
        self.gen = getattr(sampler_state, "gen", None)
        return self

    def _config_value(self, name: str, default: Any = None) -> Any:
        if self.cfg is None:
            return default
        value = getattr(self.cfg, name, None)
        if value is not None:
            return value
        settings = getattr(self.cfg, "settings", None)
        if isinstance(settings, Mapping):
            return settings.get(name, default)
        return default

    @staticmethod
    def _as_float(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def resolve_heun_blend_weight(
        self,
        i: int,
        steps: int,
        sigma: Any = None,
        shape: Any = None,
    ) -> float:
        """Resolve the experimental Heun blend weight for the current step."""

        default_weight = float(self._config_value("heun_blend_weight", 0.5))
        style = str(self._config_value("heun_blend_style", "fixed"))

        if style == "fixed":
            return default_weight

        if style == "adaptive_step":
            denominator = max(int(steps), 1)
            t = min(max(float(i) / denominator, 0.0), 1.0)
            curve = str(self._config_value("heun_adaptive_curve", "linear"))
            if curve == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * t))
            if curve == "linear":
                return 1.0 - t
            if curve == "exp_decay":
                return math.exp(-5.0 * t)
            return default_weight

        if style == "adaptive_noise" and sigma is not None:
            sigma_value = self._as_float(sigma)
            curve = str(self._config_value("heun_noise_curve", "inverse_sigma"))
            if curve == "inverse_sigma":
                return min(1.0, 1.0 / (max(sigma_value, 0.0) + 1e-5))
            if curve == "thresholded":
                threshold = float(self._config_value("heun_noise_threshold", 1.0))
                high = float(self._config_value("heun_blend_weight_high", default_weight))
                low = float(self._config_value("heun_blend_weight_low", default_weight))
                return high if sigma_value > threshold else low
            return default_weight

        if style == "adaptive_resolution" and shape:
            if len(shape) < 2:
                return default_weight
            resolution = int(shape[-1]) * int(shape[-2])
            scale = float(self._config_value("heun_resolution_scale", 1.0))
            return min(1.0, max(0.0, scale * resolution))

        return default_weight

    @staticmethod
    def resolve_randomized_value(
        base: float,
        min_: float,
        max_: float,
        stretch: float,
        *,
        rng: random.Random | None = None,
    ) -> float:
        """Return a randomized value across an optionally stretched range."""

        del base  # Retained for compatibility with the original experiment.
        if min_ > max_:
            raise ValueError("min_ cannot be greater than max_")
        if stretch < 0:
            raise ValueError("stretch cannot be negative")
        range_span = max_ - min_
        low_stretched = min_ - (range_span * stretch)
        high_stretched = max_ + (range_span * stretch)
        generator = rng or random
        return float(generator.uniform(low_stretched, high_stretched))

    def resolve_method_param(
        self,
        param_base: str,
        method: str | None,
        fallback: Any = None,
    ) -> Any:
        """Resolve a method-specific setting before its generic fallback."""

        default_method = self._config_value("sampler_type", "")
        selected_method = method or str(default_method or "")
        if selected_method:
            method_value = self._config_value(f"{selected_method}_{param_base}", None)
            if method_value is not None:
                return method_value
        return self._config_value(param_base, fallback)
