from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SD3RuntimeProfile:
    profile_id: str
    display_name: str
    checkpoint_variant: str
    runtime_assets_subdir: str
    transformer_class: str = "SD3Transformer2DModel"
    transformer_num_layers: int = 24
    transformer_num_attention_heads: int = 24
    transformer_attention_head_dim: int = 64
    transformer_in_channels: int = 16
    transformer_out_channels: int = 16
    transformer_joint_attention_dim: int = 4096
    transformer_pooled_projection_dim: int = 2048
    transformer_caption_projection_dim: int = 1536
    transformer_patch_size: int = 2
    transformer_sample_size: int = 128
    transformer_pos_embed_max_size: int = 192
    transformer_qk_norm: str = ""
    transformer_dual_attention_layers: tuple[int, ...] = ()
    scheduler_class: str = "FlowMatchEulerDiscreteScheduler"
    scheduler_num_train_timesteps: int = 1000
    scheduler_shift: float = 3.0
    vae_class: str = "AutoencoderKL"
    vae_latent_channels: int = 16
    vae_scaling_factor: float = 1.5305
    vae_shift_factor: float = 0.0609
    vae_sample_size: int = 1024
    vae_force_upcast: bool = True
    vae_mid_block_add_attention: bool | None = None
    recommended_steps: tuple[int, ...] = (20,)
    image_gen_cfg_scale: float | None = 5.0
    sampler_name: str = "flow_euler"
    scheduler_name: str = "flow_match_euler"
    recommendation_ui_enabled: bool = True
    enforce_steps: bool = False
    enforce_cfg: bool = False
    enforce_sampler_scheduler: bool = False
    generation_qualified: bool = True
    source_note: str = (
        "SD3-11 qualified reference settings. These are recommendations only; "
        "sampler, scheduler, steps, and CFG remain user-controlled."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "checkpoint_variant": self.checkpoint_variant,
            "runtime_assets_subdir": self.runtime_assets_subdir,
            "transformer_class": self.transformer_class,
            "transformer_num_layers": self.transformer_num_layers,
            "transformer_num_attention_heads": self.transformer_num_attention_heads,
            "transformer_attention_head_dim": self.transformer_attention_head_dim,
            "transformer_in_channels": self.transformer_in_channels,
            "transformer_out_channels": self.transformer_out_channels,
            "transformer_joint_attention_dim": self.transformer_joint_attention_dim,
            "transformer_pooled_projection_dim": self.transformer_pooled_projection_dim,
            "transformer_caption_projection_dim": self.transformer_caption_projection_dim,
            "transformer_patch_size": self.transformer_patch_size,
            "transformer_sample_size": self.transformer_sample_size,
            "transformer_pos_embed_max_size": self.transformer_pos_embed_max_size,
            "transformer_qk_norm": self.transformer_qk_norm,
            "transformer_dual_attention_layers": list(self.transformer_dual_attention_layers),
            "scheduler_class": self.scheduler_class,
            "scheduler_num_train_timesteps": self.scheduler_num_train_timesteps,
            "scheduler_shift": self.scheduler_shift,
            "vae_class": self.vae_class,
            "vae_latent_channels": self.vae_latent_channels,
            "vae_scaling_factor": self.vae_scaling_factor,
            "vae_shift_factor": self.vae_shift_factor,
            "vae_sample_size": self.vae_sample_size,
            "vae_force_upcast": self.vae_force_upcast,
            "vae_mid_block_add_attention": self.vae_mid_block_add_attention,
            "architecture": "sd3.x",
            "family": "sd3.5-medium" if self.checkpoint_variant == "sd3_5_medium" else "sd3-medium",
            "recommended_steps": list(self.recommended_steps),
            "image_gen_cfg_scale": self.image_gen_cfg_scale,
            "sampler_name": self.sampler_name,
            "scheduler_name": self.scheduler_name,
            "recommendation_ui_enabled": self.recommendation_ui_enabled,
            "enforce_steps": self.enforce_steps,
            "enforce_cfg": self.enforce_cfg,
            "enforce_sampler_scheduler": self.enforce_sampler_scheduler,
            "generation_qualified": self.generation_qualified,
            "source_note": self.source_note,
        }


SD3_MEDIUM = SD3RuntimeProfile(
    profile_id="sd3-medium",
    display_name="Stable Diffusion 3 Medium",
    checkpoint_variant="sd3_medium",
    runtime_assets_subdir=str(Path("stable_diffusion") / "sd3_medium_diffusers"),
    transformer_pos_embed_max_size=192,
    transformer_qk_norm="",
    transformer_dual_attention_layers=(),
    vae_mid_block_add_attention=None,
)

SD3_5_MEDIUM = SD3RuntimeProfile(
    profile_id="sd3.5-medium",
    display_name="Stable Diffusion 3.5 Medium",
    checkpoint_variant="sd3_5_medium",
    runtime_assets_subdir=str(Path("stable_diffusion") / "sd3.5_medium"),
    transformer_pos_embed_max_size=384,
    transformer_qk_norm="rms_norm",
    transformer_dual_attention_layers=tuple(range(13)),
    vae_mid_block_add_attention=True,
)


_PROFILES_BY_ID = {
    profile.profile_id: profile
    for profile in (SD3_MEDIUM, SD3_5_MEDIUM)
}

_PROFILES_BY_VARIANT = {
    profile.checkpoint_variant: profile
    for profile in (SD3_MEDIUM, SD3_5_MEDIUM)
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def profile_from_id(profile_id: str | None) -> SD3RuntimeProfile | None:
    key = _normalize_text(profile_id)
    return _PROFILES_BY_ID.get(key)


def profile_from_checkpoint_variant(checkpoint_variant: str | None) -> SD3RuntimeProfile | None:
    text = str(checkpoint_variant or "").strip().lower()
    aliases = {
        "sd3_medium": "sd3_medium",
        "sd3-medium": "sd3_medium",
        "sd3 medium": "sd3_medium",
        "sd3.5_medium": "sd3_5_medium",
        "sd3_5_medium": "sd3_5_medium",
        "sd3.5-medium": "sd3_5_medium",
        "sd3-5-medium": "sd3_5_medium",
        "sd3.5 medium": "sd3_5_medium",
    }
    return _PROFILES_BY_VARIANT.get(aliases.get(text, text))


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(name, default)
    return getattr(request, name, default)


def sd3_profile_recommendation_warnings(
    request: Any,
    profile: SD3RuntimeProfile,
) -> tuple[str, ...]:
    """Return advisory SD3 profile warnings without blocking generation."""
    warnings: list[str] = []
    steps = int(_request_value(request, "steps", 0) or 0)
    cfg_scale = float(_request_value(request, "cfg_scale", 0.0) or 0.0)
    sampler_name = str(_request_value(request, "sampler_name", "") or "").strip()
    scheduler_name = str(_request_value(request, "scheduler_name", "") or "").strip()
    recommended_steps = tuple(int(value) for value in profile.recommended_steps if int(value) > 0)
    if recommended_steps and steps > 0 and steps not in recommended_steps:
        warnings.append(
            f"{profile.profile_id} recommends {', '.join(str(v) for v in recommended_steps)} steps; "
            f"the requested value is {steps}. Generation is still allowed."
        )
    if profile.image_gen_cfg_scale is not None and abs(cfg_scale - float(profile.image_gen_cfg_scale)) > 1e-6:
        warnings.append(
            f"{profile.profile_id} recommends CFG {profile.image_gen_cfg_scale:g}; "
            f"the requested CFG is {cfg_scale:g}. Generation is still allowed."
        )
    if profile.sampler_name and sampler_name and sampler_name != profile.sampler_name:
        warnings.append(
            f"{profile.profile_id} recommends sampler {profile.sampler_name!r}; "
            f"the selected sampler is {sampler_name!r}. Generation is still allowed if the selected sampler supports the active mathematical domain."
        )
    if profile.scheduler_name and scheduler_name and scheduler_name != profile.scheduler_name:
        warnings.append(
            f"{profile.profile_id} recommends scheduler {profile.scheduler_name!r}; "
            f"the selected scheduler is {scheduler_name!r}. Generation is still allowed if the selected scheduler is compatible with the sampler/model domain."
        )
    return tuple(warnings)


def apply_sd3_profile_to_request(
    request: Any,
    profile: SD3RuntimeProfile,
    *,
    enforce_steps: bool | None = None,
    enforce_cfg: bool | None = None,
) -> dict[str, Any]:
    """Optionally apply only explicitly enabled step/CFG recommendations.

    Sampler and scheduler are never rewritten by a model profile. Compatibility
    remains the responsibility of the sampler/scheduler capability boundary.
    """
    before = {
        "steps": int(_request_value(request, "steps", 0) or 0),
        "cfg_scale": float(_request_value(request, "cfg_scale", 0.0) or 0.0),
        "sampler_name": str(_request_value(request, "sampler_name", "") or ""),
        "scheduler_name": str(_request_value(request, "scheduler_name", "") or ""),
    }
    use_steps = profile.enforce_steps if enforce_steps is None else bool(enforce_steps)
    use_cfg = profile.enforce_cfg if enforce_cfg is None else bool(enforce_cfg)
    if use_steps and profile.recommended_steps:
        value = int(profile.recommended_steps[0])
        if isinstance(request, dict):
            request["steps"] = value
        else:
            request.steps = value
    if use_cfg and profile.image_gen_cfg_scale is not None:
        value = float(profile.image_gen_cfg_scale)
        if isinstance(request, dict):
            request["cfg_scale"] = value
        else:
            request.cfg_scale = value
    after = {
        "steps": int(_request_value(request, "steps", 0) or 0),
        "cfg_scale": float(_request_value(request, "cfg_scale", 0.0) or 0.0),
        "sampler_name": str(_request_value(request, "sampler_name", "") or ""),
        "scheduler_name": str(_request_value(request, "scheduler_name", "") or ""),
    }
    return {
        "profile_id": profile.profile_id,
        "before": before,
        "after": after,
        "enforce_steps": use_steps,
        "enforce_cfg": use_cfg,
        "sampler_scheduler_forced": False,
        "warnings": list(sd3_profile_recommendation_warnings(request, profile)),
    }
