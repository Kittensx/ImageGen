from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class SDXLRuntimeProfile:
    profile_id: str
    family: str
    prediction_type: str
    native_width: int = 1024
    native_height: int = 1024
    latent_scale_factor: int = 8
    vae_scaling_factor: float = 0.13025
    required_steps: int | None = None
    recommended_steps: tuple[int, ...] = ()
    image_gen_cfg_scale: float | None = None
    source_guidance_scale: float | None = None
    sampler_name: str = ""
    scheduler_name: str = ""
    recommended_cfg_preset: str = ""
    timestep_spacing: str = ""
    enforce_steps: bool = False
    enforce_cfg: bool = False
    enforce_sampler_scheduler: bool = False
    source_note: str = ""
    generation_qualified: bool = True
    qualification_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "architecture": "sdxl",
            "family": self.family,
            "prediction_type": self.prediction_type,
            "native_width": self.native_width,
            "native_height": self.native_height,
            "latent_scale_factor": self.latent_scale_factor,
            "vae_scaling_factor": self.vae_scaling_factor,
            "required_steps": self.required_steps,
            "recommended_steps": list(self.recommended_steps),
            "image_gen_cfg_scale": self.image_gen_cfg_scale,
            "source_guidance_scale": self.source_guidance_scale,
            "sampler_name": self.sampler_name,
            "scheduler_name": self.scheduler_name,
            "recommended_cfg_preset": self.recommended_cfg_preset,
            "timestep_spacing": self.timestep_spacing,
            "enforce_steps": self.enforce_steps,
            "enforce_cfg": self.enforce_cfg,
            "enforce_sampler_scheduler": self.enforce_sampler_scheduler,
            "source_note": self.source_note,
            "generation_qualified": self.generation_qualified,
            "qualification_note": self.qualification_note,
        }


_BASE = SDXLRuntimeProfile(
    profile_id="sdxl-base",
    family="base",
    prediction_type="epsilon",
    recommended_steps=(20, 30, 40, 50),
    source_note="Generic SDXL 1.0 architecture profile; execution settings are not forced.",
)


def _lightning(steps: int, *, x0: bool = False) -> SDXLRuntimeProfile:
    return SDXLRuntimeProfile(
        profile_id=f"sdxl-lightning-{steps}step" + ("-x0" if x0 else ""),
        family="lightning",
        prediction_type="sample" if x0 else "epsilon",
        required_steps=steps,
        recommended_steps=(steps,),
        # Diffusers guidance_scale=0 disables CFG. IMAGE_GEN owns CFG explicitly as
        # uncond + scale * (cond - uncond), so scale=1 is the semantic equivalent:
        # the positive conditional model output with no guidance amplification.
        image_gen_cfg_scale=1.0,
        source_guidance_scale=0.0,
        sampler_name="simple_euler",
        scheduler_name="sdxl_euler_trailing",
        recommended_cfg_preset="sdxl_lightning_recommended",
        timestep_spacing="trailing",
        enforce_steps=False,
        enforce_cfg=False,
        enforce_sampler_scheduler=False,
        source_note=(
            "SDXL-Lightning recommendation: exact checkpoint step count, Euler with trailing "
            "timesteps, no CFG amplification; 1-step x0 uses sample prediction. "
            "Sampler and scheduler selections remain user-controlled."
        ),
    )


_PROFILES: dict[str, SDXLRuntimeProfile] = {
    _BASE.profile_id: _BASE,
    "sdxl-lightning-1step-x0": _lightning(1, x0=True),
    "sdxl-lightning-2step": _lightning(2),
    "sdxl-lightning-4step": _lightning(4),
    "sdxl-lightning-8step": _lightning(8),
    "sdxl-turbo": SDXLRuntimeProfile(
        profile_id="sdxl-turbo",
        family="turbo",
        prediction_type="epsilon",
        native_width=512,
        native_height=512,
        required_steps=1,
        recommended_steps=(1, 2, 3, 4),
        image_gen_cfg_scale=1.0,
        source_guidance_scale=0.0,
        enforce_steps=False,
        enforce_cfg=False,
        enforce_sampler_scheduler=False,
        timestep_spacing="trailing",
        source_note=(
            "SDXL-Turbo uses the generic SDXL architecture with 512x512 native generation, "
            "1-4 steps, disabled guidance, and a trailing Euler-Ancestral scheduler contract."
        ),
        qualification_note=(
            "Runtime profile settings are recommendations only; generation is not gated by profile qualification."
        ),
    ),
    "sdxl-refiner": SDXLRuntimeProfile(
        profile_id="sdxl-refiner",
        family="refiner",
        prediction_type="epsilon",
        native_width=1024,
        native_height=1024,
        recommended_steps=(20, 30, 40, 50),
        source_note=(
            "SDXL Refiner is detected as a distinct second-stage profile with its own conditioning and UNet contract. "
            "The current normal txt2img runtime intentionally does not execute that second-stage pipeline."
        ),
        generation_qualified=False,
        qualification_note=(
            "SDXL Refiner is structurally recognized and available for qualification, but normal txt2img execution "
            "is not yet implemented for the distinct second-stage Refiner pipeline."
        ),
    ),
}

_ALIASES = {
    "sdxl": "sdxl-base",
    "base": "sdxl-base",
    "sdxl-base-1.0": "sdxl-base",
    "lightning-1": "sdxl-lightning-1step-x0",
    "lightning-1step": "sdxl-lightning-1step-x0",
    "lightning-1step-x0": "sdxl-lightning-1step-x0",
    "lightning-2": "sdxl-lightning-2step",
    "lightning-4": "sdxl-lightning-4step",
    "lightning-8": "sdxl-lightning-8step",
    "turbo": "sdxl-turbo",
    "refiner": "sdxl-refiner",
    "sdxl-refiner-1.0": "sdxl-refiner",
}


def normalize_sdxl_profile_id(value: str | None) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    return _ALIASES.get(token, token)


def get_sdxl_runtime_profile(profile_id: str) -> SDXLRuntimeProfile:
    normalized = normalize_sdxl_profile_id(profile_id)
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown SDXL runtime profile {profile_id!r}. Available: {choices}") from exc


def sdxl_runtime_profiles() -> tuple[SDXLRuntimeProfile, ...]:
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def profile_for_sdxl_filename(file_name: str) -> SDXLRuntimeProfile:
    name = Path(str(file_name or "")).name.lower()
    if "refiner" in name:
        return get_sdxl_runtime_profile("sdxl-refiner")
    exact = {
        "sdxl_lightning_1step_x0.safetensors": "sdxl-lightning-1step-x0",
        "sdxl_lightning_2step.safetensors": "sdxl-lightning-2step",
        "sdxl_lightning_4step.safetensors": "sdxl-lightning-4step",
        "sdxl_lightning_8step.safetensors": "sdxl-lightning-8step",
        "sdxl_lightning_1step_unet_x0.safetensors": "sdxl-lightning-1step-x0",
        "sdxl_lightning_2step_unet.safetensors": "sdxl-lightning-2step",
        "sdxl_lightning_4step_unet.safetensors": "sdxl-lightning-4step",
        "sdxl_lightning_8step_unet.safetensors": "sdxl-lightning-8step",
    }
    if name in exact:
        return get_sdxl_runtime_profile(exact[name])
    match = re.search(r"sdxl[-_]lightning[-_](1|2|4|8)step(?P<x0>[-_]x0)?", name)
    if match:
        steps = int(match.group(1))
        if steps == 1 and match.group("x0"):
            return get_sdxl_runtime_profile("sdxl-lightning-1step-x0")
        if steps in {2, 4, 8}:
            return get_sdxl_runtime_profile(f"sdxl-lightning-{steps}step")
    if "sdxl" in name and "turbo" in name:
        return get_sdxl_runtime_profile("sdxl-turbo")
    return _BASE


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(name, default)
    return getattr(request, name, default)


def sdxl_profile_recommendation_warnings(
    request: Any,
    profile: SDXLRuntimeProfile,
) -> tuple[str, ...]:
    """Return advisory model-profile warnings without blocking or mutating the request."""

    if profile.family not in {"lightning", "turbo"}:
        return ()

    warnings: list[str] = []
    steps = int(_request_value(request, "steps", 0) or 0)
    cfg_scale = float(_request_value(request, "cfg_scale", 0.0) or 0.0)
    sampler_name = str(_request_value(request, "sampler_name", "") or "").strip()
    scheduler_name = str(_request_value(request, "scheduler_name", "") or "").strip()

    recommended_steps = tuple(int(value) for value in profile.recommended_steps if int(value) > 0)
    if recommended_steps and steps > 0 and steps not in recommended_steps:
        if len(recommended_steps) == 1:
            recommended_label = str(recommended_steps[0])
        else:
            recommended_label = ", ".join(str(value) for value in recommended_steps)
        direction = "above" if steps > max(recommended_steps) else "outside"
        warnings.append(
            f"{profile.profile_id} is designed around {recommended_label} step"
            f"{'s' if len(recommended_steps) != 1 or recommended_steps[0] != 1 else ''}; "
            f"the requested {steps} steps are {direction} the recommended setting. "
            "Generation is allowed."
        )

    if profile.image_gen_cfg_scale is not None and abs(cfg_scale - float(profile.image_gen_cfg_scale)) > 1e-6:
        warnings.append(
            f"{profile.profile_id} recommends CFG {profile.image_gen_cfg_scale:g}; "
            f"the requested CFG is {cfg_scale:g}. Generation is allowed."
        )

    if profile.sampler_name and sampler_name and sampler_name != profile.sampler_name:
        warnings.append(
            f"{profile.profile_id} recommends sampler {profile.sampler_name!r}; "
            f"the selected sampler is {sampler_name!r}. Generation is allowed."
        )

    if profile.scheduler_name and scheduler_name and scheduler_name != profile.scheduler_name:
        warnings.append(
            f"{profile.profile_id} recommends scheduler {profile.scheduler_name!r}; "
            f"the selected scheduler is {scheduler_name!r}. Generation is allowed."
        )

    return tuple(dict.fromkeys(warnings))


def _set_request_value(request: Any, name: str, value: Any) -> None:
    if isinstance(request, dict):
        request[name] = value
    else:
        setattr(request, name, value)


def apply_sdxl_profile_to_request(
    request: Any,
    profile: SDXLRuntimeProfile,
    *,
    enforce_steps: bool | None = None,
    enforce_cfg: bool | None = None,
) -> dict[str, Any]:
    """Apply opt-in SDXL recommendations without gating experimentation.

    The WebUI recommendation checkboxes are explicit opt-ins. When enabled, the
    recommended step count and/or CFG become the effective generation values, but
    sampler and scheduler selections are never rewritten. The visible fields may
    still contain experimental values so the UI can warn and the user can opt out
    simply by clearing the relevant checkbox.
    """

    requested_defaults = {
        "steps": bool(enforce_steps),
        "cfg": bool(enforce_cfg),
    }
    before = {
        "steps": int(_request_value(request, "steps", 0) or 0),
        "cfg_scale": float(_request_value(request, "cfg_scale", 0.0) or 0.0),
        "sampler_name": str(_request_value(request, "sampler_name", "") or ""),
        "scheduler_name": str(_request_value(request, "scheduler_name", "") or ""),
    }
    warnings = list(sdxl_profile_recommendation_warnings(request, profile))
    effective_enforcement = {
        "steps": False,
        "cfg": False,
        "sampler_scheduler": False,
    }

    recommended_steps = tuple(int(value) for value in profile.recommended_steps if int(value) > 0)
    preferred_steps = int(profile.required_steps or (recommended_steps[0] if recommended_steps else 0) or 0)
    if bool(enforce_steps) and preferred_steps > 0:
        _set_request_value(request, "steps", preferred_steps)
        effective_enforcement["steps"] = True
        if before["steps"] != preferred_steps:
            warnings.append(
                f"Use recommended steps is enabled for {profile.profile_id}; "
                f"the requested {before['steps']} steps were replaced with {preferred_steps}. "
                "Uncheck the recommendation to run a custom step count."
            )

    if bool(enforce_cfg) and profile.image_gen_cfg_scale is not None:
        recommended_cfg = float(profile.image_gen_cfg_scale)
        _set_request_value(request, "cfg_scale", recommended_cfg)
        effective_enforcement["cfg"] = True
        if abs(before["cfg_scale"] - recommended_cfg) > 1e-6:
            warnings.append(
                f"Use recommended CFG is enabled for {profile.profile_id}; "
                f"the requested CFG {before['cfg_scale']:g} was replaced with {recommended_cfg:g}. "
                "Uncheck the recommendation to run a custom CFG."
            )

    after = {
        "steps": int(_request_value(request, "steps", 0) or 0),
        "cfg_scale": float(_request_value(request, "cfg_scale", 0.0) or 0.0),
        "sampler_name": str(_request_value(request, "sampler_name", "") or ""),
        "scheduler_name": str(_request_value(request, "scheduler_name", "") or ""),
    }
    warnings = list(dict.fromkeys(warnings))
    diagnostics = getattr(request, "diagnostics", None)
    if isinstance(diagnostics, dict):
        diagnostics["sdxl_runtime_profile"] = profile.to_dict()
        diagnostics["sdxl_profile_recommendation_warnings"] = list(warnings)

    return {
        "profile": profile.to_dict(),
        "before": before,
        "after": after,
        "recommendation_defaults_requested": requested_defaults,
        "effective_enforcement": effective_enforcement,
        "user_override": {
            "steps": bool(recommended_steps and before["steps"] not in recommended_steps),
            "cfg": bool(
                profile.image_gen_cfg_scale is not None
                and abs(before["cfg_scale"] - float(profile.image_gen_cfg_scale)) > 1e-6
            ),
        },
        "warnings": warnings,
    }

