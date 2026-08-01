from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from typing import Any, Iterable

import torch

from .policy import normalize_policy


VALID_HIRES_MEMORY_PROFILES = {"inherit", "balanced", "low_vram", "maximum"}


@dataclass(frozen=True)
class HiresMemoryBehavior:
    requested_profile: str
    effective_profile: str
    planner_profile: str
    pre_cleanup_requested: bool
    pre_cleanup_required: bool
    disable_preview_during_hires: bool
    safety_margin_mb: int
    sequential_component_residency: bool
    release_base_temporaries: bool
    offload_text_encoder_before_sampling: bool
    offload_vae_before_sampling: bool
    attention_slicing_requested: bool
    vae_tiling_requested: bool
    vae_slicing_requested: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HiresCleanupReport:
    performed: bool
    profile: str
    reason: str
    preserved_tensors: tuple[dict[str, Any], ...]
    released_reference_names: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    before: dict[str, Any]
    after: dict[str, Any]
    reclaimed_allocated_bytes: int
    reclaimed_reserved_bytes: int
    garbage_collected_objects: int
    cuda_synchronized: bool
    cuda_cache_emptied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_hires_memory_profile(value: str | None) -> str:
    token = str(value or "inherit").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "low": "low_vram",
        "lowvram": "low_vram",
        "memory_saver": "low_vram",
        "maximum_memory_savings": "maximum",
        "max": "maximum",
    }
    token = aliases.get(token, token)
    if token not in VALID_HIRES_MEMORY_PROFILES:
        raise ValueError(
            "hires memory profile must be one of: inherit, balanced, low_vram, maximum."
        )
    return token


def resolve_hires_memory_behavior(
    *,
    requested_profile: str | None,
    base_memory_policy: str,
    pre_hires_cleanup: bool,
    preview_policy: str,
    base_safety_margin_mb: int,
) -> HiresMemoryBehavior:
    requested = normalize_hires_memory_profile(requested_profile)
    base = normalize_policy(base_memory_policy)
    if requested == "inherit":
        effective = base
        if effective == "auto":
            # The manager will still resolve auto from live telemetry for its base
            # stages. Hires needs deterministic behavior before that stage lease,
            # so inherit-auto uses balanced transition semantics and lets the
            # planner strengthen them if measured headroom is insufficient.
            effective = "balanced"
    else:
        effective = requested

    maximum = effective == "maximum"
    low_vram = effective == "low_vram"
    balanced = effective == "balanced"
    cleanup_required = bool(pre_hires_cleanup or low_vram or maximum)
    normalized_preview = str(preview_policy or "normal").strip().lower().replace("-", "_")
    disable_preview = bool(
        normalized_preview in {"disable_during_hires", "disabled"}
        or low_vram
        or maximum
    )
    margin = max(0, int(base_safety_margin_mb))
    if maximum:
        margin = max(1536, margin + 512)

    return HiresMemoryBehavior(
        requested_profile=requested,
        effective_profile=effective,
        planner_profile="low_vram" if maximum else effective,
        pre_cleanup_requested=bool(pre_hires_cleanup),
        pre_cleanup_required=cleanup_required,
        disable_preview_during_hires=disable_preview,
        safety_margin_mb=margin,
        sequential_component_residency=bool(low_vram or maximum),
        release_base_temporaries=cleanup_required,
        offload_text_encoder_before_sampling=bool(low_vram or maximum or balanced),
        offload_vae_before_sampling=bool(low_vram or maximum or balanced),
        attention_slicing_requested=maximum,
        vae_tiling_requested=maximum,
        vae_slicing_requested=maximum,
    )


def _tensor_descriptor(name: str, value: Any) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {"name": str(name), "tensor": False}
    return {
        "name": str(name),
        "tensor": True,
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


def _cuda_value(snapshot: dict[str, Any], key: str) -> int:
    try:
        return int(dict(snapshot.get("cuda") or {}).get(key) or 0)
    except (TypeError, ValueError):
        return 0


def perform_pre_hires_cleanup(
    manager: Any,
    *,
    behavior: HiresMemoryBehavior,
    preserved_tensors: Iterable[tuple[str, Any]] = (),
    released_reference_names: Iterable[str] = (),
) -> HiresCleanupReport:
    """Execute the measurable cleanup boundary before hires latent allocation.

    Callers must remove their no-longer-needed Python references before invoking
    this function. The function then drains preview work, offloads inactive
    components, runs garbage collection, and finally releases allocator cache.
    """

    preserved = tuple(_tensor_descriptor(name, value) for name, value in preserved_tensors)
    released_names = tuple(str(value) for value in released_reference_names)
    actions: list[dict[str, Any]] = []
    before = manager.capture("before_pre_hires_cleanup")

    if behavior.disable_preview_during_hires:
        suspension_source = (
            "preview_policy"
            if str(behavior.requested_profile) == "inherit"
            and str(behavior.effective_profile) not in {"low_vram", "maximum"}
            else "hires_memory_profile"
        )
        try:
            suspended_result = manager.suspend_preview_image_decode(
                "Image preview decoding was disabled before the hires second pass.",
                source=suspension_source,
            )
        except TypeError:
            # Compatibility with pre-14K-7 memory-manager fakes and extensions.
            suspended_result = manager.suspend_preview_image_decode(
                "Image preview decoding was disabled before the hires second pass."
            )
        suspended = bool(suspended_result)
        actions.append({"action": "suspend_hires_preview_decode", "applied": suspended})

    if behavior.pre_cleanup_required or behavior.disable_preview_during_hires:
        preview_report = manager.release_preview_work_for_hires()
        if preview_report:
            actions.extend(preview_report)

    component_ids = []
    if behavior.offload_text_encoder_before_sampling:
        component_ids.append("text_encoder")
    if behavior.offload_vae_before_sampling:
        component_ids.extend(("vae", "preview_decoder", "upscaler"))
    actions.extend(
        manager.offload_inactive_components(
            component_ids,
            stage="pre_hires_cleanup",
            reason=f"hires profile {behavior.effective_profile}",
        )
    )

    collected = 0
    synchronized = False
    cache_emptied = False
    if behavior.pre_cleanup_required:
        collected = int(gc.collect())
        actions.append({"action": "python_gc", "collected_objects": collected})
        if manager.target_device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(manager.target_device)
                synchronized = True
                actions.append({"action": "cuda_synchronize", "applied": True})
            except Exception as exc:  # pragma: no cover - hardware dependent
                actions.append(
                    {
                        "action": "cuda_synchronize",
                        "applied": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            try:
                torch.cuda.empty_cache()
                cache_emptied = True
                actions.append({"action": "empty_released_allocator_cache", "applied": True})
            except Exception as exc:  # pragma: no cover - hardware dependent
                actions.append(
                    {
                        "action": "empty_released_allocator_cache",
                        "applied": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    after = manager.capture("after_pre_hires_cleanup")
    reclaimed_allocated = max(
        0,
        _cuda_value(before, "allocated_vram_bytes")
        - _cuda_value(after, "allocated_vram_bytes"),
    )
    reclaimed_reserved = max(
        0,
        _cuda_value(before, "reserved_vram_bytes")
        - _cuda_value(after, "reserved_vram_bytes"),
    )
    report = HiresCleanupReport(
        performed=True,
        profile=behavior.effective_profile,
        reason=(
            "profile-required cleanup"
            if behavior.pre_cleanup_required
            else "component residency transition without allocator cleanup"
        ),
        preserved_tensors=preserved,
        released_reference_names=released_names,
        actions=tuple(actions),
        before=before,
        after=after,
        reclaimed_allocated_bytes=reclaimed_allocated,
        reclaimed_reserved_bytes=reclaimed_reserved,
        garbage_collected_objects=collected,
        cuda_synchronized=synchronized,
        cuda_cache_emptied=cache_emptied,
    )
    manager.record_hires_cleanup(report.to_dict())
    return report


__all__ = [
    "HiresCleanupReport",
    "HiresMemoryBehavior",
    "VALID_HIRES_MEMORY_PROFILES",
    "normalize_hires_memory_profile",
    "perform_pre_hires_cleanup",
    "resolve_hires_memory_behavior",
]
