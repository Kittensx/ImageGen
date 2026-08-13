from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict


_DEFAULT_MODE = "legacy_flat"
_VALID_MODES = {"legacy_flat", "auto_low_cfg", "sigma_shaped", "step_shaped"}
_VALID_CURVES = {"smoothstep", "cosine", "linear", "exp_decay"}


@dataclass(slots=True)
class EffectiveGuidanceProfile:
    requested_cfg_scale: float
    cfg_guidance_mode: str = _DEFAULT_MODE
    cfg_curve_type: str = "smoothstep"
    cfg_curve_strength: float = 1.0
    cfg_high_sigma_boost: float = 1.2
    cfg_low_sigma_taper: float = 0.3
    cfg_auto_low_cfg_threshold: float = 6.5
    cfg_early_floor_enabled: bool = False
    cfg_early_floor_value: float = 6.2
    cfg_early_floor_until_fraction: float = 0.3
    sampler_local_rescale_cfg: bool = False
    sampler_local_rescale_factor: float = 1.0
    cfg_effective_min_lock: float | None = None
    cfg_effective_max_lock: float | None = None

    @property
    def mode_active(self) -> bool:
        return self.cfg_guidance_mode != _DEFAULT_MODE

    @classmethod
    def from_settings(
        cls,
        *,
        requested_cfg_scale: float,
        get_setting: Callable[[str, Any], Any],
        sampler_local_rescale_cfg: bool,
        sampler_local_rescale_factor: float,
    ) -> "EffectiveGuidanceProfile":
        mode = _normalize_mode(get_setting("cfg_guidance_mode", _DEFAULT_MODE))
        curve = _normalize_curve(get_setting("cfg_curve_type", "smoothstep"))
        return cls(
            requested_cfg_scale=float(requested_cfg_scale),
            cfg_guidance_mode=mode,
            cfg_curve_type=curve,
            cfg_curve_strength=max(0.0, _coerce_float(get_setting("cfg_curve_strength", 1.0), 1.0)),
            cfg_high_sigma_boost=max(0.0, _coerce_float(get_setting("cfg_high_sigma_boost", 1.2), 1.2)),
            cfg_low_sigma_taper=max(0.0, _coerce_float(get_setting("cfg_low_sigma_taper", 0.3), 0.3)),
            cfg_auto_low_cfg_threshold=max(0.0, _coerce_float(get_setting("cfg_auto_low_cfg_threshold", 6.5), 6.5)),
            cfg_early_floor_enabled=bool(get_setting("cfg_early_floor_enabled", False)),
            cfg_early_floor_value=max(0.0, _coerce_float(get_setting("cfg_early_floor_value", 6.2), 6.2)),
            cfg_early_floor_until_fraction=_clamp01(
                _coerce_float(get_setting("cfg_early_floor_until_fraction", 0.3), 0.3)
            ),
            sampler_local_rescale_cfg=bool(sampler_local_rescale_cfg),
            sampler_local_rescale_factor=float(sampler_local_rescale_factor),
            cfg_effective_min_lock=_coerce_optional_float(get_setting("cfg_effective_min_lock", None)),
            cfg_effective_max_lock=_coerce_optional_float(get_setting("cfg_effective_max_lock", None)),
        )


class EffectiveGuidanceController:
    """Shared per-step effective CFG shaping used by all samplers.

    The shaping logic is sampler-agnostic. KES may still opt into sampler-local
    rescaling, but the core effective CFG curve is now standard across the
    pipeline samplers as well.
    """

    def __init__(self, profile: EffectiveGuidanceProfile) -> None:
        self.profile = profile

    def compute(
        self,
        *,
        step_index: int,
        total_steps: int,
        sigma: Any,
        sigma_max: Any,
        sigma_min: Any,
        requested_cfg_scale: float | None = None,
    ) -> tuple[float, Dict[str, Any]]:
        requested = (
            float(self.profile.requested_cfg_scale)
            if requested_cfg_scale is None
            else max(0.0, _coerce_float(requested_cfg_scale, self.profile.requested_cfg_scale))
        )
        progress = _normalized_progress(step_index, total_steps)
        sigma_fraction = _normalized_sigma_fraction(sigma, sigma_max, sigma_min)
        mode = self.profile.cfg_guidance_mode
        shaped_active = False
        auto_low_cfg_applied = False

        if mode == "legacy_flat":
            base_effective = requested
            boost = 0.0
            taper = 0.0
            early_weight = 0.0
            late_weight = 0.0
        else:
            if mode == "step_shaped":
                early_weight = _curve_weight(1.0 - progress, self.profile.cfg_curve_type)
                late_weight = _curve_weight(progress, self.profile.cfg_curve_type)
                shaped_active = True
            elif mode == "sigma_shaped":
                early_weight = _curve_weight(sigma_fraction, self.profile.cfg_curve_type)
                late_weight = _curve_weight(1.0 - sigma_fraction, self.profile.cfg_curve_type)
                shaped_active = True
            elif mode == "auto_low_cfg":
                if requested < self.profile.cfg_auto_low_cfg_threshold:
                    early_weight = _curve_weight(sigma_fraction, self.profile.cfg_curve_type)
                    late_weight = _curve_weight(1.0 - sigma_fraction, self.profile.cfg_curve_type)
                    shaped_active = True
                    auto_low_cfg_applied = True
                else:
                    early_weight = 0.0
                    late_weight = 0.0
            else:
                early_weight = 0.0
                late_weight = 0.0

            boost = self.profile.cfg_high_sigma_boost * self.profile.cfg_curve_strength * early_weight
            taper = self.profile.cfg_low_sigma_taper * self.profile.cfg_curve_strength * late_weight
            base_effective = requested + boost - taper

        floor_applied = False
        if self.profile.cfg_early_floor_enabled and progress <= self.profile.cfg_early_floor_until_fraction:
            floor_value = float(self.profile.cfg_early_floor_value)
            if base_effective < floor_value:
                base_effective = floor_value
                floor_applied = True
        else:
            floor_value = float(self.profile.cfg_early_floor_value)

        base_effective = max(0.0, float(base_effective))
        if self.profile.sampler_local_rescale_cfg:
            effective_cfg_scale = base_effective * float(self.profile.sampler_local_rescale_factor)
        else:
            effective_cfg_scale = base_effective

        unclamped_effective_cfg_scale = float(effective_cfg_scale)
        if self.profile.cfg_effective_min_lock is not None:
            effective_cfg_scale = max(float(self.profile.cfg_effective_min_lock), float(effective_cfg_scale))
        if self.profile.cfg_effective_max_lock is not None:
            effective_cfg_scale = min(float(self.profile.cfg_effective_max_lock), float(effective_cfg_scale))
        cfg_range_lock_applied = abs(float(effective_cfg_scale) - unclamped_effective_cfg_scale) > 1e-12

        details = {
            "cfg_guidance_mode": mode,
            "cfg_curve_type": self.profile.cfg_curve_type,
            "cfg_curve_strength": float(self.profile.cfg_curve_strength),
            "cfg_high_sigma_boost": float(self.profile.cfg_high_sigma_boost),
            "cfg_low_sigma_taper": float(self.profile.cfg_low_sigma_taper),
            "cfg_auto_low_cfg_threshold": float(self.profile.cfg_auto_low_cfg_threshold),
            "cfg_early_floor_enabled": bool(self.profile.cfg_early_floor_enabled),
            "cfg_early_floor_value": floor_value,
            "cfg_early_floor_until_fraction": float(self.profile.cfg_early_floor_until_fraction),
            "requested_cfg_scale": requested,
            "effective_cfg_scale": float(effective_cfg_scale),
            "effective_cfg_scale_pre_rescale": float(base_effective),
            "sampler_local_rescale_cfg": bool(self.profile.sampler_local_rescale_cfg),
            "sampler_local_rescale_factor": float(self.profile.sampler_local_rescale_factor),
            "cfg_effective_min_lock": self.profile.cfg_effective_min_lock,
            "cfg_effective_max_lock": self.profile.cfg_effective_max_lock,
            "cfg_range_lock_applied": bool(cfg_range_lock_applied),
            "effective_cfg_scale_before_range_lock": unclamped_effective_cfg_scale,
            "guidance_shaping_active": bool(shaped_active),
            "guidance_shaping_auto_applied": bool(auto_low_cfg_applied),
            "cfg_early_floor_applied": bool(floor_applied),
            "progress_fraction": float(progress),
            "sigma_fraction": float(sigma_fraction),
            "curve_weight_early": float(early_weight),
            "curve_weight_late": float(late_weight),
            "boost_applied": float(boost),
            "taper_applied": float(taper),
            "step_index": int(step_index),
            "total_steps": int(max(total_steps, 0)),
        }
        return float(effective_cfg_scale), details

    def summary(self) -> Dict[str, Any]:
        return {
            "requested_cfg_scale": float(self.profile.requested_cfg_scale),
            "cfg_guidance_mode": self.profile.cfg_guidance_mode,
            "cfg_curve_type": self.profile.cfg_curve_type,
            "cfg_curve_strength": float(self.profile.cfg_curve_strength),
            "cfg_high_sigma_boost": float(self.profile.cfg_high_sigma_boost),
            "cfg_low_sigma_taper": float(self.profile.cfg_low_sigma_taper),
            "cfg_auto_low_cfg_threshold": float(self.profile.cfg_auto_low_cfg_threshold),
            "cfg_early_floor_enabled": bool(self.profile.cfg_early_floor_enabled),
            "cfg_early_floor_value": float(self.profile.cfg_early_floor_value),
            "cfg_early_floor_until_fraction": float(self.profile.cfg_early_floor_until_fraction),
            "sampler_local_rescale_cfg": bool(self.profile.sampler_local_rescale_cfg),
            "sampler_local_rescale_factor": float(self.profile.sampler_local_rescale_factor),
            "cfg_effective_min_lock": self.profile.cfg_effective_min_lock,
            "cfg_effective_max_lock": self.profile.cfg_effective_max_lock,
        }


def _normalize_mode(value: Any) -> str:
    text = str(value or _DEFAULT_MODE).strip().lower()
    return text if text in _VALID_MODES else _DEFAULT_MODE


def _normalize_curve(value: Any) -> str:
    text = str(value or "smoothstep").strip().lower()
    return text if text in _VALID_CURVES else "smoothstep"


def _curve_weight(value: float, curve_type: str) -> float:
    x = _clamp01(value)
    curve = _normalize_curve(curve_type)
    if curve == "linear":
        return x
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * x)
    if curve == "exp_decay":
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        numerator = 1.0 - math.exp(-4.0 * x)
        denominator = 1.0 - math.exp(-4.0)
        return numerator / denominator
    return x * x * (3.0 - 2.0 * x)


def _normalized_progress(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    return _clamp01(float(step_index) / float(max(total_steps - 1, 1)))


def _normalized_sigma_fraction(sigma: Any, sigma_max: Any, sigma_min: Any) -> float:
    sigma_value = _coerce_float(sigma, 0.0)
    sigma_max_value = _coerce_float(sigma_max, sigma_value)
    sigma_min_value = _coerce_float(sigma_min, 0.0)
    denominator = sigma_max_value - sigma_min_value
    if abs(denominator) <= 1e-12:
        return 1.0
    return _clamp01((sigma_value - sigma_min_value) / denominator)


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return float(default)
        return result
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
