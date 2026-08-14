from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_POLICIES = {"auto", "high_vram", "balanced", "low_vram", "cpu_fallback"}


@dataclass(frozen=True)
class MemoryPolicyDecision:
    requested_profile: str
    effective_profile: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "reason": self.reason,
        }


def normalize_policy(value: str | None) -> str:
    token = str(value or "auto").strip().lower().replace(" ", "_")
    aliases = {
        "high": "high_vram",
        "low": "low_vram",
        "cpu": "cpu_fallback",
    }
    token = aliases.get(token, token)
    return token if token in VALID_POLICIES else "auto"


def resolve_policy(
    requested_profile: str,
    *,
    cuda_payload: dict[str, Any] | None,
) -> MemoryPolicyDecision:
    requested = normalize_policy(requested_profile)
    if requested != "auto":
        return MemoryPolicyDecision(requested, requested, "explicit user profile")

    cuda = dict(cuda_payload or {})
    if not cuda.get("available"):
        return MemoryPolicyDecision("auto", "cpu_fallback", "CUDA telemetry unavailable")
    total = int(cuda.get("total_vram_bytes") or 0)
    free = int(cuda.get("free_vram_bytes") or 0)
    gib = 1024 ** 3
    if total >= 16 * gib and free >= 8 * gib:
        return MemoryPolicyDecision("auto", "high_vram", "high VRAM capacity and headroom")
    # SD 1.x checkpoints can remain fully resident on common 8 GiB cards.
    # Treat those devices as balanced while at least 1 GiB of physical headroom
    # remains; stage planning and bounded OOM recovery can still downgrade when
    # a specific request does not fit.
    if total >= 6 * gib and free >= 1 * gib:
        return MemoryPolicyDecision("auto", "balanced", "resident checkpoint fits with usable VRAM headroom")
    return MemoryPolicyDecision("auto", "low_vram", "limited reported VRAM headroom")


def post_stage_offload_candidates(profile: str, stage: str) -> set[str]:
    effective = normalize_policy(profile)
    if effective == "high_vram":
        return set()
    if effective == "balanced":
        if stage == "conditioning":
            return {"text_encoder", "text_encoder_2"}
        if stage == "sampling":
            return {"text_encoder", "text_encoder_2", "unet"}
        if stage == "final_decode":
            return {"text_encoder", "text_encoder_2"}
        return set()
    if effective in {"low_vram", "cpu_fallback"}:
        return {"text_encoder", "text_encoder_2", "unet", "vae", "preview_decoder", "upscaler"}
    return set()
