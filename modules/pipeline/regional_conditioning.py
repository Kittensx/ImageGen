# -----------------------------------------------------------------------------
# REGION ATTRIBUTION
#
# Original REGION syntax and core regional-conditioning design are based on
# work by GitHub user Konpr:
#   https://github.com/Konpr/whats-/tree/main/new_version3
#
# The original author granted permission to use/adapt the code with credit.
# IMAGE_GEN retains that attribution for the REGION language and semantics,
# including REGION{...}, spatial coordinates/tiling, *base=, overlay/common
# modes, latent backend semantics, branch weights/curves, start/stop windows,
# blur, canvas masks, and base_ratio.
#
# This file's IMAGE_GEN implementation is a substantial native adaptation:
# parser-independent conditioning, native model-output execution, canonical CFG
# integration, batching, hires/resolution transforms, replay/fingerprinting,
# validation, telemetry, caching, and low-VRAM execution are IMAGE_GEN work.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch

from image_gen.systems.regional_prompting import (
    RuntimeRegionCondition,
    RuntimeRegionSpec,
    build_region_mask_cached,
    region_mask_cache_stats,
    region_strength_for_step,
)


@dataclass
class RegionConditioningEntry:
    spec: RuntimeRegionSpec
    positive_multicond: Any
    semantic_digest: str = ""


@dataclass
class _RegionTelemetryEntry:
    slot_index: int
    region_index: int
    prompt: str
    active_steps: set[int] = field(default_factory=set)
    unet_calls: int = 0
    duration_ms: float = 0.0
    mask_local_hits: int = 0
    mask_shared_hits: int = 0
    mask_misses: int = 0


class RegionalRuntimeTelemetry:
    """Request-scoped REGION counters safe to expose in sampler metadata."""

    def __init__(self, *, pass_name: str, entries: list[RegionConditioningEntry]) -> None:
        self.pass_name = str(pass_name or "base")
        self._lock = RLock()
        self.resolver_calls = 0
        self.steps_with_regions: set[int] = set()
        self.active_region_instances = 0
        self._entries: dict[tuple[int, int], _RegionTelemetryEntry] = {
            (int(item.spec.slot_index), int(item.spec.region_index)): _RegionTelemetryEntry(
                slot_index=int(item.spec.slot_index),
                region_index=int(item.spec.region_index),
                prompt=str(item.spec.prompt or ""),
            )
            for item in entries
        }

    def _entry(self, slot_index: int, region_index: int, prompt: str = "") -> _RegionTelemetryEntry:
        key = (int(slot_index), int(region_index))
        value = self._entries.get(key)
        if value is None:
            value = _RegionTelemetryEntry(key[0], key[1], str(prompt or ""))
            self._entries[key] = value
        return value

    def record_resolve(self, *, step_index: int, active_regions: list[RuntimeRegionCondition]) -> None:
        with self._lock:
            self.resolver_calls += 1
            if active_regions:
                self.steps_with_regions.add(int(step_index))
            self.active_region_instances += len(active_regions)
            for region in active_regions:
                self._entry(region.slot_index, region.region_index).active_steps.add(int(step_index))

    def record_mask(self, *, spec: RuntimeRegionSpec, local_hit: bool, shared_hit: bool) -> None:
        with self._lock:
            entry = self._entry(spec.slot_index, spec.region_index, spec.prompt)
            if local_hit:
                entry.mask_local_hits += 1
            elif shared_hit:
                entry.mask_shared_hits += 1
            else:
                entry.mask_misses += 1

    def record_unet_call(self, *, slot_index: int, region_index: int, duration_ms: float) -> None:
        with self._lock:
            entry = self._entry(slot_index, region_index)
            entry.unet_calls += 1
            entry.duration_ms += max(0.0, float(duration_ms))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = [
                {
                    "slot_index": value.slot_index,
                    "region_index": value.region_index,
                    "prompt": value.prompt,
                    "active_steps": sorted(value.active_steps),
                    "active_step_count": len(value.active_steps),
                    "unet_calls": value.unet_calls,
                    "host_elapsed_ms": round(value.duration_ms, 4),
                    "duration_ms": round(value.duration_ms, 4),
                    "mask_local_hits": value.mask_local_hits,
                    "mask_shared_hits": value.mask_shared_hits,
                    "mask_misses": value.mask_misses,
                }
                for _key, value in sorted(self._entries.items())
            ]
            return {
                "contract_version": "image-gen-region-runtime-v1",
                "pass": self.pass_name,
                "evaluation_mode": "sequential_low_vram",
                "resolver_calls": self.resolver_calls,
                "steps_with_regions": sorted(self.steps_with_regions),
                "active_region_instances": self.active_region_instances,
                "regional_unet_calls": sum(item["unet_calls"] for item in entries),
                "regional_host_elapsed_ms": round(sum(item["host_elapsed_ms"] for item in entries), 4),
                "regional_unet_duration_ms": round(sum(item["host_elapsed_ms"] for item in entries), 4),
                "timing_semantics": "host_elapsed_unsynchronized",
                "regions": entries,
                "shared_mask_cache": region_mask_cache_stats(),
            }


class RegionalConditioningResolver:
    """Resolve base conditioning and slot-specific regional conditioning per logical step."""

    def __init__(
        self,
        *,
        base_resolver: Any,
        entries: list[RegionConditioningEntry],
        total_steps: int,
        generation_width: int,
        generation_height: int,
        overlap_policy: str,
        region_counts_by_slot: dict[int, int],
        pass_name: str = "base",
    ) -> None:
        self.base_resolver = base_resolver
        self.entries = list(entries)
        self.total_steps = int(total_steps)
        self.generation_width = int(generation_width)
        self.generation_height = int(generation_height)
        self.overlap_policy = str(overlap_policy or "additive").strip().lower()
        self.region_counts_by_slot = {
            int(key): int(value) for key, value in dict(region_counts_by_slot or {}).items()
        }
        self._mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self.telemetry = RegionalRuntimeTelemetry(pass_name=pass_name, entries=self.entries)

    def resolve(self, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_resolver.resolve(step_index)

    def resolve_regions(
        self,
        *,
        step_index: int,
        latents: torch.Tensor,
    ) -> list[RuntimeRegionCondition]:
        if not self.entries:
            return []
        output: list[RuntimeRegionCondition] = []
        for entry in self.entries:
            spec = entry.spec
            strength = region_strength_for_step(
                spec,
                step_index=int(step_index),
                total_steps=self.total_steps,
            )
            if strength <= 0.0:
                continue
            cond = self.base_resolver._resolve_multicond_for_step(
                entry.positive_multicond,
                int(step_index),
            )
            if int(cond.shape[0]) != 1:
                raise ValueError("Regional conditioning must contain exactly one batch item.")
            count = self.region_counts_by_slot.get(int(spec.slot_index), 1)
            cache_key = (
                int(spec.slot_index),
                int(spec.region_index),
                int(latents.shape[-2]),
                int(latents.shape[-1]),
                str(latents.device),
                str(latents.dtype),
                int(count),
            )
            mask = self._mask_cache.get(cache_key)
            local_hit = mask is not None
            shared_hit = False
            if mask is None:
                mask, shared_hit = build_region_mask_cached(
                    spec,
                    latent_height=int(latents.shape[-2]),
                    latent_width=int(latents.shape[-1]),
                    generation_width=self.generation_width,
                    generation_height=self.generation_height,
                    region_count=count,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                self._mask_cache[cache_key] = mask
            self.telemetry.record_mask(spec=spec, local_hit=local_hit, shared_hit=shared_hit)
            output.append(
                RuntimeRegionCondition(
                    slot_index=int(spec.slot_index),
                    region_index=int(spec.region_index),
                    conditioning=cond.to(device=latents.device, dtype=latents.dtype),
                    mask=mask,
                    strength=float(strength),
                    mode=str(spec.mode),
                    metadata={
                        "semantic_digest": str(entry.semantic_digest or ""),
                        "curve": str(spec.curve),
                        "start": float(spec.start),
                        "stop": float(spec.stop),
                        "weight": float(spec.weight),
                        "base_ratio": float(spec.base_ratio),
                        "_regional_telemetry": self.telemetry,
                    },
                )
            )
        self.telemetry.record_resolve(step_index=int(step_index), active_regions=output)
        return output

    def runtime_snapshot(self) -> dict[str, Any]:
        return self.telemetry.snapshot()


def get_regional_conditioning_resolver(conditioning: Any) -> RegionalConditioningResolver | None:
    extra = getattr(conditioning, "extra", None)
    if isinstance(extra, dict):
        value = extra.get("regional_resolver")
        if isinstance(value, RegionalConditioningResolver):
            return value
    return None


def has_regional_conditioning(conditioning: Any) -> bool:
    resolver = get_regional_conditioning_resolver(conditioning)
    return resolver is not None and bool(resolver.entries)
