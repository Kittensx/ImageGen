from __future__ import annotations

from image_gen.program_metadata import PRODUCT_NAME

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from modules.checkpoint_inspector import CheckpointInspector, CheckpointReport

from .capabilities import capability_for
from .metrics import compare_tensors, tensor_digest, tensor_statistics
from .models import (
    RealCheckpointValidationReport,
    ValidationCheck,
    ValidationProfile,
    ValidationRunRecord,
)


class RealCheckpointValidationSystem:
    """Collect and evaluate Phase 07 real-checkpoint evidence."""

    def __init__(self, project_context: Any) -> None:
        self.project_context = project_context
        self.inspector = CheckpointInspector()

    def inspect_checkpoint(self, model_path: str | Path) -> CheckpointReport:
        return self.inspector.inspect(str(model_path))

    def require_supported_architecture(self, report: CheckpointReport):
        capability = capability_for(report.architecture)
        if not capability.validation_supported:
            requirements = "; ".join(capability.requirements)
            raise ValueError(
                f"Architecture {report.architecture!r} is not enabled for Phase 07 "
                f"validation: {capability.reason} Required work: {requirements}"
            )
        if report.checkpoint_kind != "full":
            raise ValueError(
                f"Phase 07 requires a full checkpoint, got {report.checkpoint_kind!r}."
            )
        return capability

    def validate_tokenizer(self, tokenizer: Any, prompts: list[str]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                padding="max_length",
                max_length=int(getattr(tokenizer, "model_max_length", 77)),
                truncation=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"]
            repeated = tokenizer(
                prompt,
                padding="max_length",
                max_length=int(getattr(tokenizer, "model_max_length", 77)),
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            rows.append(
                {
                    "prompt": prompt,
                    "shape": list(ids.shape),
                    "non_padding_tokens": int(
                        (ids != int(getattr(tokenizer, "pad_token_id", 1))).sum().item()
                    ),
                    "deterministic": bool(torch.equal(ids, repeated)),
                    "token_ids_sha256": tensor_digest(ids),
                    "first_token_ids": [int(value) for value in ids[0, :12].tolist()],
                }
            )
        return {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "prompts": rows,
            "passed": all(row["deterministic"] and row["shape"][-1] == 77 for row in rows),
        }

    @staticmethod
    def component_coverage(built_components: Any) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name in ("unet_result", "text_encoder_result", "text_encoder_2_result", "vae_result"):
            result = getattr(built_components, name, None)
            if result is None:
                continue
            converter = getattr(result, "to_validation_dict", None)
            if callable(converter):
                data = converter()
            else:
                expected = int(getattr(result, "expected_keys", 0) or 0)
                matched = int(getattr(result, "matched_keys", 0) or 0)
                data = {
                    "name": getattr(result, "name", name.replace("_result", "")),
                    "success": bool(getattr(result, "success", False)),
                    "provided_keys": int(getattr(result, "loaded_keys", 0) or 0),
                    "expected_keys": expected,
                    "matched_keys": matched,
                    "coverage_ratio": matched / expected if expected else 0.0,
                    "missing_key_count": len(getattr(result, "missing_keys", []) or []),
                    "unexpected_key_count": len(
                        getattr(result, "unexpected_keys", []) or []
                    ),
                    "error": getattr(result, "error", None),
                }
            report[data["name"]] = data
        return report

    @staticmethod
    def run_record(label: str, result: Any) -> ValidationRunRecord:
        pipeline = result.pipeline_result
        diagnostics = dict(result.diagnostics or {})
        summaries = diagnostics.get("tensor_summaries") or {}
        conditioning = pipeline.conditioning
        schedule = pipeline.schedule
        sampler = pipeline.sampler
        return ValidationRunRecord(
            label=label,
            sampler_name=str(result.request.sampler_name or ""),
            scheduler_name=str(result.request.scheduler_name or ""),
            run_id=result.run_id,
            generation_time_sec=result.generation_time_sec,
            seed=int(result.request.seed),
            output_paths=[record.image_path for record in result.saved_records],
            conditioning={
                "cond": tensor_statistics(conditioning.cond if conditioning else None),
                "uncond": tensor_statistics(conditioning.uncond if conditioning else None),
            },
            schedule=(schedule.to_serializable_dict() if schedule is not None else {}),
            initial_latents=dict(summaries.get("latent_preparation.latents") or {}),
            final_latents=tensor_statistics(pipeline.latents),
            decoded_images=tensor_statistics(pipeline.images),
            sampler=(sampler.to_serializable_dict() if sampler is not None else {}),
            diagnostics={
                "timings": diagnostics.get("timings", []),
                "warnings": diagnostics.get("warnings", []),
            },
            image_digest=tensor_digest(pipeline.images),
            latent_digest=tensor_digest(pipeline.latents),
        )

    @staticmethod
    def environment() -> dict[str, Any]:
        cuda_name = None
        if torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device": cuda_name,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        }

    def build_report(
        self,
        *,
        profile: ValidationProfile,
        checkpoint: CheckpointReport,
        tokenizer_report: dict[str, Any],
        component_coverage: dict[str, Any],
        baseline_first: Any,
        baseline_repeat: Any,
        comparison_run: Any,
    ) -> RealCheckpointValidationReport:
        capability = capability_for(checkpoint.architecture)
        first_record = self.run_record("baseline_first", baseline_first)
        repeat_record = self.run_record("baseline_repeat", baseline_repeat)
        comparison_record = self.run_record("comparison_sampler", comparison_run)

        image_repeat = compare_tensors(
            baseline_first.pipeline_result.images,
            baseline_repeat.pipeline_result.images,
            atol=profile.absolute_tolerance,
            rtol=profile.relative_tolerance,
        )
        latent_repeat = compare_tensors(
            baseline_first.pipeline_result.latents,
            baseline_repeat.pipeline_result.latents,
            atol=profile.absolute_tolerance,
            rtol=profile.relative_tolerance,
        )
        sampler_comparison = {
            "images": compare_tensors(
                baseline_first.pipeline_result.images,
                comparison_run.pipeline_result.images,
                atol=profile.absolute_tolerance,
                rtol=profile.relative_tolerance,
            ),
            "latents": compare_tensors(
                baseline_first.pipeline_result.latents,
                comparison_run.pipeline_result.latents,
                atol=profile.absolute_tolerance,
                rtol=profile.relative_tolerance,
            ),
            "note": "Baseline and KES outputs are recorded for comparison; equality is not expected.",
        }
        comparisons = {
            "fixed_seed_repeat": {
                "images": image_repeat,
                "latents": latent_repeat,
                "metadata_equal": (
                    first_record.schedule == repeat_record.schedule
                    and first_record.conditioning == repeat_record.conditioning
                ),
            },
            "baseline_vs_comparison_sampler": sampler_comparison,
        }

        component_pass = bool(component_coverage) and all(
            item.get("success") and item.get("coverage_ratio", 0.0) >= 0.95
            for item in component_coverage.values()
        )
        schedule = baseline_first.pipeline_result.schedule
        schedule_pass = bool(
            schedule is not None
            and schedule.sigmas.numel() == schedule.effective_steps + 1
            and schedule.timesteps is not None
            and torch.isfinite(schedule.sigmas).all()
            and torch.isfinite(schedule.timesteps).all()
        )
        conditioning = baseline_first.pipeline_result.conditioning
        conditioning_pass = bool(
            conditioning is not None
            and conditioning.cond.ndim == 3
            and conditioning.uncond.shape == conditioning.cond.shape
            and conditioning.cond.shape[-1] == 768
            and torch.isfinite(conditioning.cond).all()
            and torch.isfinite(conditioning.uncond).all()
        )
        images = baseline_first.pipeline_result.images
        decode_pass = bool(
            torch.is_tensor(images)
            and tuple(images.shape)
            == (profile.batch_size, 3, profile.height, profile.width)
            and torch.isfinite(images).all()
            and float(images.min().item()) >= 0.0
            and float(images.max().item()) <= 1.0
        )
        checks = [
            ValidationCheck(
                "checkpoint.full_sd1",
                "pass" if checkpoint.checkpoint_kind == "full" and capability.validation_supported else "fail",
                "Checkpoint is a full, explicitly supported SD1 checkpoint.",
                {"architecture": checkpoint.architecture, "kind": checkpoint.checkpoint_kind},
            ),
            ValidationCheck(
                "tokenizer.local_deterministic",
                "pass" if tokenizer_report.get("passed") else "fail",
                "Local tokenizer produces deterministic 77-token batches.",
                tokenizer_report,
            ),
            ValidationCheck(
                "components.coverage",
                "pass" if component_pass else "fail",
                "UNet, VAE, and text-encoder component coverage is at least 95%.",
                component_coverage,
            ),
            ValidationCheck(
                "conditioning.sd1_contract",
                "pass" if conditioning_pass else "fail",
                "Conditioning is finite BCH-style data with SD1 width 768.",
                first_record.conditioning,
            ),
            ValidationCheck(
                "schedule.valid",
                "pass" if schedule_pass else "fail",
                "Sigma and timestep sequences are finite and aligned.",
                first_record.schedule,
            ),
            ValidationCheck(
                "decode.valid",
                "pass" if decode_pass else "fail",
                "VAE output has the requested RGB shape and normalized finite range.",
                first_record.decoded_images,
            ),
            ValidationCheck(
                "reproducibility.fixed_seed",
                "pass"
                if image_repeat.get("materially_equivalent")
                and latent_repeat.get("materially_equivalent")
                else "fail",
                "Repeated baseline request is materially equivalent under the fixed seed.",
                comparisons["fixed_seed_repeat"],
            ),
            ValidationCheck(
                "samplers.executed",
                "pass"
                if first_record.sampler_name == profile.baseline_sampler
                and comparison_record.sampler_name == profile.comparison_sampler
                else "fail",
                "Baseline and KES sampler paths both completed under the fixed profile.",
                {
                    "baseline": first_record.sampler_name,
                    "comparison": comparison_record.sampler_name,
                },
            ),
        ]
        return RealCheckpointValidationReport(
            profile=profile,
            checkpoint=checkpoint.to_dict(),
            architecture_capability=capability,
            tokenizer=tokenizer_report,
            component_coverage=component_coverage,
            runs=[first_record, repeat_record, comparison_record],
            comparisons=comparisons,
            checks=checks,
            environment=self.environment(),
        )

    @staticmethod
    def write_report(
        report: RealCheckpointValidationReport,
        output_dir: str | Path,
    ) -> tuple[Path, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "phase07_validation_report.json"
        markdown_path = root / "phase07_validation_report.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        lines = [
            f"# {PRODUCT_NAME} Phase 07 Real Checkpoint Validation",
            "",
            f"**Result:** {'PASS' if report.passed else 'FAIL'}",
            f"**Created:** {report.created_utc}",
            f"**Checkpoint:** `{report.checkpoint.get('file_name', '')}`",
            f"**SHA-256:** `{report.checkpoint.get('sha256', '')}`",
            f"**Architecture:** `{report.checkpoint.get('architecture', '')}`",
            "",
            "## Checks",
            "",
            "| Check | Result | Summary |",
            "|---|---:|---|",
        ]
        for check in report.checks:
            lines.append(
                f"| `{check.check_id}` | **{check.status.upper()}** | {check.summary} |"
            )
        lines.extend(
            [
                "",
                "## Runs",
                "",
                "| Label | Sampler | Scheduler | Seed | Seconds |",
                "|---|---|---|---:|---:|",
            ]
        )
        for run in report.runs:
            seconds = "" if run.generation_time_sec is None else f"{run.generation_time_sec:.3f}"
            lines.append(
                f"| {run.label} | `{run.sampler_name}` | `{run.scheduler_name}` | {run.seed} | {seconds} |"
            )
        lines.extend(
            [
                "",
                "## Architecture Capability",
                "",
                report.architecture_capability.reason,
                "",
                "The JSON report contains complete checkpoint inventory, component coverage, tokenizer evidence, schedules, tensor statistics, sampler comparison, and reproducibility metrics.",
                "",
            ]
        )
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, markdown_path
