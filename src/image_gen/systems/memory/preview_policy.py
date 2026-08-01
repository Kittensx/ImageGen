from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VALID_PREVIEW_POLICIES = {
    "normal",
    "suspend_on_pressure",
    "disable_during_hires",
    "disabled",
}
_VAE_PREVIEW_MODES = {"balanced", "accurate"}
_HIRES_STAGES = {"hires", "hires_second_pass", "second_pass"}


def normalize_preview_policy(value: str | None) -> str:
    token = str(value or "normal").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "off": "disabled",
        "disable": "disabled",
        "no_preview": "disabled",
        "pressure": "suspend_on_pressure",
        "hires_only": "disable_during_hires",
    }
    token = aliases.get(token, token)
    if token not in VALID_PREVIEW_POLICIES:
        raise ValueError(
            "preview policy must be one of: normal, suspend_on_pressure, "
            "disable_during_hires, disabled."
        )
    return token


@dataclass(frozen=True)
class PreviewStagePolicy:
    requested_policy: str
    stage: str
    configured_preview_mode: str
    effective_preview_mode: str
    image_decode_enabled: bool
    vae_preview_requested: bool
    vae_residency: str
    suspend_on_pressure: bool
    suspension_reason: str
    suspension_source: str
    cfg_telemetry_continues: bool = True
    suspension_is_one_way: bool = True

    @property
    def requires_vae(self) -> bool:
        return self.vae_residency == "required"

    @property
    def optional_vae(self) -> bool:
        return self.vae_residency == "optional"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_preview_stage_policy(
    *,
    requested_policy: str | None,
    stage: str,
    preview_mode: str | None,
    force_disabled: bool = False,
    force_disabled_reason: str = "",
    already_suspended: bool = False,
    existing_suspension_reason: str = "",
    existing_suspension_source: str = "",
) -> PreviewStagePolicy:
    policy = normalize_preview_policy(requested_policy)
    normalized_stage = str(stage or "sampling").strip().lower().replace("-", "_")
    mode = str(preview_mode or "disabled").strip().lower()
    if mode not in {"fast", "balanced", "accurate", "disabled"}:
        mode = "fast"

    reason = ""
    source = ""
    disabled = mode == "disabled"

    if already_suspended:
        disabled = True
        reason = str(existing_suspension_reason or "Image preview decoding was suspended earlier in this job.")
        source = str(existing_suspension_source or "previous_stage")
    elif policy == "disabled":
        disabled = True
        reason = "Preview policy disabled image decoding for this job."
        source = "policy_disabled"
    elif force_disabled:
        disabled = True
        reason = str(force_disabled_reason or "The active stage memory profile disabled image preview decoding.")
        source = "stage_memory_profile"
    elif policy == "disable_during_hires" and normalized_stage in _HIRES_STAGES:
        disabled = True
        reason = "Preview policy disabled image decoding during the hires second pass."
        source = "policy_disable_during_hires"

    image_decode_enabled = not disabled
    vae_preview_requested = bool(image_decode_enabled and mode in _VAE_PREVIEW_MODES)
    pressure_enabled = bool(policy == "suspend_on_pressure" and vae_preview_requested)
    if not vae_preview_requested:
        vae_residency = "not_needed"
    elif pressure_enabled:
        vae_residency = "optional"
    else:
        vae_residency = "required"

    return PreviewStagePolicy(
        requested_policy=policy,
        stage=normalized_stage,
        configured_preview_mode=mode,
        effective_preview_mode="disabled" if disabled else mode,
        image_decode_enabled=image_decode_enabled,
        vae_preview_requested=vae_preview_requested,
        vae_residency=vae_residency,
        suspend_on_pressure=pressure_enabled,
        suspension_reason=reason,
        suspension_source=source,
    )
