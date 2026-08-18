from __future__ import annotations

from image_gen.program_metadata import PRODUCT_NAME

import json
from pathlib import Path
from typing import Any

import torch


OUTPUT_QUALITY_CONTRACT_VERSION = "image-gen-output-quality-v1"


def summarize_tensor(value: torch.Tensor | None) -> dict[str, Any]:
    """Return bounded, JSON-safe tensor statistics without retaining the tensor."""
    if value is None or not torch.is_tensor(value):
        return {"available": False}
    detached = value.detach()
    report: dict[str, Any] = {
        "available": True,
        "shape": [int(item) for item in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        report.update({"finite": True, "finite_fraction": 1.0})
        return report

    sample = detached.to(device="cpu", dtype=torch.float32)
    finite = torch.isfinite(sample)
    finite_count = int(finite.sum().item())
    report["finite"] = finite_count == sample.numel()
    report["finite_fraction"] = round(finite_count / max(1, sample.numel()), 8)
    if finite_count == 0:
        return report

    values = sample[finite]
    report.update(
        {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "abs_mean": float(values.abs().mean().item()),
            "dynamic_range": float((values.max() - values.min()).item()),
        }
    )
    if values.numel() > 1:
        report["p01"] = float(torch.quantile(values, 0.01).item())
        report["p05"] = float(torch.quantile(values, 0.05).item())
        report["p95"] = float(torch.quantile(values, 0.95).item())
        report["p99"] = float(torch.quantile(values, 0.99).item())

    if sample.ndim == 4:
        channel_reports: list[dict[str, Any]] = []
        channel_means: list[float] = []
        for channel_index in range(int(sample.shape[1])):
            channel = sample[:, channel_index, ...]
            channel_finite = channel[torch.isfinite(channel)]
            if channel_finite.numel() == 0:
                channel_reports.append({"channel": channel_index, "finite": False})
                continue
            channel_mean = float(channel_finite.mean().item())
            channel_means.append(channel_mean)
            channel_reports.append(
                {
                    "channel": channel_index,
                    "finite": True,
                    "min": float(channel_finite.min().item()),
                    "max": float(channel_finite.max().item()),
                    "mean": channel_mean,
                    "std": float(channel_finite.std(unbiased=False).item()),
                    "dynamic_range": float(
                        (channel_finite.max() - channel_finite.min()).item()
                    ),
                }
            )
        if sample.shape[-1] > 1:
            horizontal = float((sample[..., 1:] - sample[..., :-1]).abs().mean().item())
        else:
            horizontal = 0.0
        if sample.shape[-2] > 1:
            vertical = float((sample[..., 1:, :] - sample[..., :-1, :]).abs().mean().item())
        else:
            vertical = 0.0
        report["channels"] = channel_reports
        report["spatial_delta_mean"] = (horizontal + vertical) / 2.0
        report["channel_mean_range"] = (max(channel_means) - min(channel_means)) if channel_means else 0.0
    return report


def classify_normalized_images(images: torch.Tensor) -> dict[str, Any]:
    """Classify suspicious near-uniform decoded images conservatively.

    This does not block generation. It creates a diagnostic warning so deliberately
    bright or dark images remain valid outputs while accidental collapse is visible.
    """
    summary = summarize_tensor(images)
    if not summary.get("available") or not summary.get("finite", False):
        return {
            "classification": "invalid",
            "suspect": True,
            "reasons": ["Decoded output is unavailable or contains non-finite values."],
            "summary": summary,
        }

    sample = images.detach().to(device="cpu", dtype=torch.float32)
    high_fraction = float((sample >= 0.90).to(torch.float32).mean().item())
    low_fraction = float((sample <= 0.10).to(torch.float32).mean().item())
    clipped_high_fraction = float((sample >= 0.999).to(torch.float32).mean().item())
    clipped_low_fraction = float((sample <= 0.001).to(torch.float32).mean().item())
    mean = float(summary.get("mean", 0.0))
    std = float(summary.get("std", 0.0))
    dynamic_range = float(summary.get("dynamic_range", 0.0))
    spatial_delta = float(summary.get("spatial_delta_mean", 0.0))
    channel_mean_range = float(summary.get("channel_mean_range", 0.0))

    near_uniform = std <= 0.02 and dynamic_range <= 0.15
    near_white = mean >= 0.90 and high_fraction >= 0.95
    near_black = mean <= 0.10 and low_fraction >= 0.95
    collapsed = std <= 0.003 and dynamic_range <= 0.03
    low_detail_monotone = (
        std <= 0.08
        and dynamic_range <= 0.45
        and spatial_delta <= 0.045
        and channel_mean_range <= 0.35
    )

    reasons: list[str] = []
    classification = "normal"
    if near_uniform and near_white:
        classification = "near_uniform_white"
        reasons.append(
            "Decoded image is nearly uniform and more than 95% of values are at or above 0.90."
        )
    elif near_uniform and near_black:
        classification = "near_uniform_black"
        reasons.append(
            "Decoded image is nearly uniform and more than 95% of values are at or below 0.10."
        )
    elif collapsed:
        classification = "near_uniform_collapsed"
        reasons.append("Decoded image has extremely low variance and dynamic range.")
    elif low_detail_monotone:
        classification = "low_detail_monotone"
        reasons.append(
            "Decoded image stayed within a narrow tonal/color band and showed very little local contrast."
        )

    return {
        "classification": classification,
        "suspect": classification != "normal",
        "reasons": reasons,
        "thresholds": {
            "near_uniform_std_max": 0.02,
            "near_uniform_dynamic_range_max": 0.15,
            "near_white_mean_min": 0.90,
            "near_black_mean_max": 0.10,
            "dominant_fraction_min": 0.95,
            "low_detail_std_max": 0.08,
            "low_detail_dynamic_range_max": 0.45,
            "low_detail_spatial_delta_max": 0.045,
            "low_detail_channel_mean_range_max": 0.35,
        },
        "fractions": {
            "at_or_above_0_90": high_fraction,
            "at_or_below_0_10": low_fraction,
            "clipped_high": clipped_high_fraction,
            "clipped_low": clipped_low_fraction,
        },
        "texture_metrics": {
            "spatial_delta_mean": spatial_delta,
            "channel_mean_range": channel_mean_range,
        },
        "summary": summary,
    }


def build_output_quality_report(
    *,
    final_latents: torch.Tensor,
    raw_vae_output: torch.Tensor,
    normalized_images: torch.Tensor,
    vae_scaling_factor: float,
    vae_shift_factor: float = 0.0,
) -> dict[str, Any]:
    classification = classify_normalized_images(normalized_images)
    return {
        "contract_version": OUTPUT_QUALITY_CONTRACT_VERSION,
        "suspect": bool(classification.get("suspect")),
        "classification": str(classification.get("classification") or "unknown"),
        "reasons": list(classification.get("reasons") or []),
        "vae_scaling_factor": float(vae_scaling_factor),
        "vae_shift_factor": float(vae_shift_factor),
        "final_latents": summarize_tensor(final_latents),
        "scaled_latents_entering_vae": summarize_tensor(
            final_latents / float(vae_scaling_factor) + float(vae_shift_factor)
        ),
        "raw_vae_output": summarize_tensor(raw_vae_output),
        "normalized_images": dict(classification.get("summary") or {}),
        "classification_details": classification,
    }


def output_quality_text(report: dict[str, Any]) -> str:
    lines = [
        f"{PRODUCT_NAME} Output Quality Diagnostic",
        "=" * 35,
        f"Classification: {report.get('classification', 'unknown')}",
        f"Suspect output: {bool(report.get('suspect'))}",
    ]
    reasons = list(report.get("reasons") or [])
    if reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in reasons)
    for label, key in (
        ("Final latents", "final_latents"),
        ("Scaled latents entering VAE", "scaled_latents_entering_vae"),
        ("Raw VAE output", "raw_vae_output"),
        ("Normalized image", "normalized_images"),
    ):
        value = dict(report.get(key) or {})
        lines.extend(
            [
                "",
                label + ":",
                f"  shape: {value.get('shape')}",
                f"  dtype/device: {value.get('dtype')} / {value.get('device')}",
                f"  finite: {value.get('finite')} ({value.get('finite_fraction')})",
                f"  min/max: {value.get('min')} / {value.get('max')}",
                f"  mean/std: {value.get('mean')} / {value.get('std')}",
                f"  dynamic range: {value.get('dynamic_range')}",
            ]
        )
    artifact_path = report.get("artifact_path")
    if artifact_path:
        lines.extend(["", f"Artifact directory: {artifact_path}"])
    return "\n".join(lines) + "\n"


def write_output_quality_bundle(
    *,
    root: Path,
    run_id: str,
    report: dict[str, Any],
    images: torch.Tensor,
) -> Path:
    """Persist a small self-contained bundle for suspect decoded output."""
    from PIL import Image

    bundle = root / "output-quality" / str(run_id)
    bundle.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["artifact_path"] = str(bundle)
    (bundle / "output_quality.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "output_quality.txt").write_text(
        output_quality_text(payload),
        encoding="utf-8",
    )

    cpu = images.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    for index, image in enumerate(cpu):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        array = (
            image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        )
        Image.fromarray(array, mode="RGB").save(bundle / f"decoded_{index:02d}.png")
    return bundle
