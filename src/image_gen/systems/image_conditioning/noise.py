from __future__ import annotations

from typing import Any

HIRES_NOISE_POLICY_ID = "offset_seed_plus_1000003_v1"
HIRES_NOISE_SEED_OFFSET = 1_000_003


def noise_policy_metadata(policy: str | None = None) -> dict[str, Any]:
    selected = str(policy or HIRES_NOISE_POLICY_ID)
    return {
        "policy_id": selected,
        "deterministic": True,
        "seed_offset": HIRES_NOISE_SEED_OFFSET if selected == HIRES_NOISE_POLICY_ID else None,
    }


def noise_stream_metadata(seeds: list[int] | tuple[int, ...], policy: str | None = None) -> dict[str, Any]:
    selected = str(policy or HIRES_NOISE_POLICY_ID)
    base_seeds = [int(seed) for seed in seeds]
    derived = [int(seed) + HIRES_NOISE_SEED_OFFSET for seed in base_seeds]
    return {
        **noise_policy_metadata(selected),
        "stream_identifier": selected,
        "base_seeds": base_seeds,
        "derived_seeds": derived,
        "batch_size": len(base_seeds),
    }
