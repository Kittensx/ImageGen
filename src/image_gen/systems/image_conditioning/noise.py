from __future__ import annotations

from typing import Any

import torch

from image_gen.contracts import SchedulerOutput

HIRES_NOISE_POLICY_ID = "offset_seed_plus_1000003_v1"
HIRES_NOISE_SEED_OFFSET = 1_000_003

SIGMA_ADDITIVE_FORWARD_PROCESS_ID = "sigma_additive_v1"
FLOW_MATCH_LINEAR_FORWARD_PROCESS_ID = "flow_match_linear_interpolation_v1"


def normalize_image_conditioned_scheduler_domain(value: str | None) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"flow", "flowmatch", "flow_matching", "flow_match"}:
        return "flow_match"
    if token in {"", "sigma", "sigma_additive", "additive", "epsilon", "diffusion"}:
        return "sigma_additive"
    raise ValueError(f"Unsupported image-conditioned scheduler domain: {value!r}.")


def image_conditioned_forward_process_metadata(
    schedule: SchedulerOutput,
    *,
    scheduler_domain: str | None,
) -> dict[str, Any]:
    domain = normalize_image_conditioned_scheduler_domain(scheduler_domain)
    sigma = float(schedule.initial_sigma)
    if domain == "flow_match":
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(
                "Flow-match image conditioning requires the selected starting sigma "
                f"to be within [0, 1]; received {sigma}."
            )
        policy = FLOW_MATCH_LINEAR_FORWARD_PROCESS_ID
        signal_coefficient = 1.0 - sigma
        noise_coefficient = sigma
        formula = "sample * (1 - sigma) + noise * sigma"
    else:
        policy = SIGMA_ADDITIVE_FORWARD_PROCESS_ID
        signal_coefficient = 1.0
        noise_coefficient = sigma
        formula = "sample + noise * sigma"
    return {
        "schema_version": "image-gen-image-conditioned-forward-process-v1",
        "scheduler_domain": domain,
        "policy": policy,
        "start_sigma": sigma,
        "signal_coefficient": float(signal_coefficient),
        "noise_coefficient": float(noise_coefficient),
        "formula": formula,
    }


def apply_image_conditioned_forward_noise(
    sample: torch.Tensor,
    noise: torch.Tensor,
    *,
    schedule: SchedulerOutput,
    scheduler_domain: str | None,
) -> torch.Tensor:
    if not torch.is_tensor(sample) or not torch.is_tensor(noise):
        raise TypeError("Image-conditioned forward noise requires tensor sample and noise inputs.")
    if sample.shape != noise.shape:
        raise ValueError(
            "Image-conditioned forward noise requires noise to match the sample shape; "
            f"sample={tuple(sample.shape)}, noise={tuple(noise.shape)}."
        )
    metadata = image_conditioned_forward_process_metadata(
        schedule,
        scheduler_domain=scheduler_domain,
    )
    signal = float(metadata["signal_coefficient"])
    noise_scale = float(metadata["noise_coefficient"])
    # Keep the operation in the source tensor dtype/device. The scalar
    # coefficients are intentionally Python floats so PyTorch follows the
    # tensor's normal type-promotion rules without creating CPU tensors.
    return sample * signal + noise * noise_scale


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
