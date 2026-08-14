from __future__ import annotations

import math
from typing import Any, Iterable

from .contracts import MemoryEstimate, ResidencyPlan
from .policy import resolve_policy


def _dtype_bytes(dtype_value: Any) -> int:
    token = str(dtype_value or "").lower()
    if "64" in token:
        return 8
    if "16" in token or "bfloat16" in token:
        return 2
    return 4


class MemoryEstimator:
    """Conservative, inspectable estimator for current and future stage planning."""

    def estimate_txt2img(
        self,
        *,
        request: Any,
        dimension_plan: Any,
        sampler_name: str,
        dtype: Any,
        preview_mode: str,
        component_bytes: dict[str, int],
        available_bytes: int | None,
        safety_margin_bytes: int,
        stage: str = "sampling",
    ) -> MemoryEstimate:
        batch = max(1, int(getattr(request, "batch_size", 1) or 1))
        width = int(getattr(dimension_plan, "generation_width", getattr(request, "width", 512)))
        height = int(getattr(dimension_plan, "generation_height", getattr(request, "height", 512)))
        latent_scale = max(1, int(getattr(dimension_plan, "latent_scale_factor", 8) or 8))
        latent_width = int(math.ceil(width / latent_scale))
        latent_height = int(math.ceil(height / latent_scale))
        scalar_bytes = _dtype_bytes(dtype)
        latent_bytes = batch * 4 * latent_width * latent_height * scalar_bytes
        sampler = str(sampler_name or "").lower()
        history_multiplier = 3 if "dpmpp" in sampler or "2m" in sampler else 2
        sampler_history = latent_bytes * history_multiplier
        is_sdxl = "text_encoder_2" in component_bytes
        conditioning_width = 2048 if is_sdxl else 1024
        pooled_width = 1280 if is_sdxl else 0
        conditioning_bytes = batch * 2 * 77 * conditioning_width * scalar_bytes
        pooled_conditioning_bytes = batch * 2 * pooled_width * scalar_bytes
        vae_decode_output = batch * 3 * width * height * 4
        preview_long_edge = max(64, int(getattr(request, "live_preview_width", 384) or 384))
        preview_frame = batch * 3 * preview_long_edge * preview_long_edge * 4
        preview_runtime = preview_frame if str(preview_mode).lower() != "disabled" else 0

        contributors: dict[str, int] = {
            "latent_tensor": int(latent_bytes),
            "sampler_history": int(sampler_history),
            "conditioning": int(conditioning_bytes),
            "pooled_conditioning": int(pooled_conditioning_bytes),
            "vae_decode_output": int(vae_decode_output),
            "preview_frame": int(preview_runtime),
        }
        if stage == "conditioning":
            contributors["text_encoder_parameters"] = int(component_bytes.get("text_encoder", 0))
            if is_sdxl:
                contributors["text_encoder_2_parameters"] = int(component_bytes.get("text_encoder_2", 0))
        elif stage == "final_decode":
            contributors["vae_parameters"] = int(component_bytes.get("vae", 0))
            contributors["decode_workspace"] = int(max(vae_decode_output, latent_bytes * 4))
        else:
            contributors["unet_parameters"] = int(component_bytes.get("unet", 0))
            if str(preview_mode).lower() in {"balanced", "accurate"}:
                contributors["vae_preview_parameters"] = int(component_bytes.get("vae", 0))

        expected = int(sum(contributors.values()))
        minimum = int(
            latent_bytes
            + sampler_history
            + conditioning_bytes
            + pooled_conditioning_bytes
            + (
                component_bytes.get("text_encoder", 0)
                + component_bytes.get("text_encoder_2", 0)
                if stage == "conditioning"
                else component_bytes.get("vae", 0)
                if stage == "final_decode"
                else component_bytes.get("unet", 0)
            )
        )
        safety_adjusted = int(expected + max(0, safety_margin_bytes))
        headroom = None if available_bytes is None else int(available_bytes - safety_adjusted)
        feasible = None if available_bytes is None else bool(headroom >= 0)
        confidence = "medium" if available_bytes is not None else "low"
        return MemoryEstimate(
            stage=stage,
            estimated_minimum_bytes=minimum,
            estimated_expected_bytes=expected,
            safety_adjusted_required_bytes=safety_adjusted,
            available_bytes=available_bytes,
            headroom_bytes=headroom,
            confidence=confidence,
            major_contributors=contributors,
            feasible=feasible,
        )


class MemoryPlanner:
    def plan(
        self,
        *,
        stage: str,
        requested_profile: str,
        target_device: str,
        required: Iterable[str],
        preferred: Iterable[str],
        optional: Iterable[str],
        component_bytes: dict[str, int],
        available_bytes: int | None,
        safety_margin_bytes: int,
        estimated_stage_bytes: int,
        preview_requires_vae: bool = False,
        resident_on_target: Iterable[str] = (),
    ) -> ResidencyPlan:
        required_ids = tuple(dict.fromkeys(str(value) for value in required))
        preferred_ids = tuple(dict.fromkeys(str(value) for value in preferred))
        optional_ids = tuple(dict.fromkeys(str(value) for value in optional))
        resident_ids = set(str(value) for value in resident_on_target)
        decision = resolve_policy(
            requested_profile,
            cuda_payload={
                "available": target_device.startswith("cuda") and available_bytes is not None,
                "free_vram_bytes": available_bytes,
                "total_vram_bytes": available_bytes,
            },
        )
        effective = decision.effective_profile
        if requested_profile == "auto" and target_device.startswith("cuda"):
            # Auto starts from the resident-friendly balanced policy. It only
            # drops to sequential low-VRAM residency when the incremental bytes
            # needed by this stage cannot fit inside current physical headroom.
            effective = "balanced"
        if not target_device.startswith("cuda"):
            effective = "cpu_fallback"

        selected = list(required_ids)
        reasons = [decision.reason]
        remaining = None
        if available_bytes is not None:
            stage_component_ids = tuple(dict.fromkeys(
                (*required_ids, *preferred_ids, *optional_ids)
            ))
            represented_parameter_bytes = sum(
                component_bytes.get(item, 0) for item in stage_component_ids
            )
            # MemoryEstimator includes stage component parameters in its expected
            # total. Subtract them before adding only the components that still
            # need to move onto the target device; otherwise resident modules are
            # counted twice and Auto incorrectly falls into low-VRAM mode.
            workspace_bytes = max(
                0,
                int(estimated_stage_bytes) - represented_parameter_bytes,
            )
            missing_required_bytes = sum(
                component_bytes.get(item, 0)
                for item in required_ids
                if item not in resident_ids
            )
            incremental_required_bytes = workspace_bytes + missing_required_bytes
            remaining = available_bytes - safety_margin_bytes - incremental_required_bytes
            if requested_profile == "auto" and remaining < 0:
                effective = "low_vram"
                reasons.append("stage incremental requirement exceeded current VRAM headroom")

        if effective == "high_vram":
            selected.extend(preferred_ids)
            selected.extend(optional_ids)
        elif effective == "balanced":
            for component_id in preferred_ids:
                component_size = component_bytes.get(component_id, 0)
                if component_id in resident_ids or remaining is None or remaining >= component_size:
                    selected.append(component_id)
                    if remaining is not None and component_id not in resident_ids:
                        remaining -= component_size
            for component_id in optional_ids:
                component_size = component_bytes.get(component_id, 0)
                if component_id in resident_ids:
                    if remaining is None or remaining >= 0:
                        selected.append(component_id)
                elif remaining is not None and remaining >= component_size * 2:
                    selected.append(component_id)
                    remaining -= component_size
        elif effective == "low_vram":
            reasons.append("sequential whole-component residency selected")
        elif effective == "cpu_fallback":
            selected = list(required_ids)

        selected = list(dict.fromkeys(selected))
        if (
            preview_requires_vae
            and "vae" in optional_ids
            and "vae" in selected
            and remaining is not None
        ):
            incremental_nonrequired_bytes = sum(
                component_bytes.get(component_id, 0)
                for component_id in selected
                if component_id not in required_ids and component_id not in resident_ids
            )
            if incremental_nonrequired_bytes > remaining:
                selected.remove("vae")
                reasons.append(
                    "optional VAE preview residency would violate the VRAM safety margin"
                )
        preview_suspended = bool(preview_requires_vae and "vae" not in selected)
        if preview_suspended:
            reasons.append("image preview decode suspended to preserve VRAM; CFG telemetry remains enabled")
        all_ids = set(required_ids) | set(preferred_ids) | set(optional_ids)
        cpu_ids = tuple(sorted(all_ids - set(selected)))
        return ResidencyPlan(
            stage=str(stage),
            requested_profile=str(requested_profile),
            effective_profile=effective,
            target_device=str(target_device),
            required=required_ids,
            preferred=preferred_ids,
            optional=optional_ids,
            selected_for_target=tuple(selected),
            selected_for_cpu=cpu_ids,
            estimated_stage_bytes=max(0, int(estimated_stage_bytes)),
            available_bytes=available_bytes,
            safety_margin_bytes=max(0, int(safety_margin_bytes)),
            preview_image_decode_suspended=preview_suspended,
            reasons=tuple(reasons),
        )
