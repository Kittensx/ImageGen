from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ArchitectureCapability:
    architecture: str
    status: str
    generation_supported: bool
    validation_supported: bool
    reason: str
    requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requirements"] = list(self.requirements)
        return data


@dataclass(frozen=True)
class ValidationProfile:
    prompt: str = "a small red ceramic teapot on a plain wooden table, studio light"
    negative_prompt: str = "blurry, distorted, low resolution"
    seed: int = 707
    steps: int = 12
    width: int = 256
    height: int = 256
    cfg_scale: float = 7.0
    scheduler_name: str = "simple_kes"
    baseline_sampler: str = "simple_euler"
    comparison_sampler: str = "kes"
    batch_size: int = 1
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationCheck:
    check_id: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationRunRecord:
    label: str
    sampler_name: str
    scheduler_name: str
    run_id: str | None
    generation_time_sec: float | None
    seed: int
    output_paths: list[str] = field(default_factory=list)
    conditioning: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    initial_latents: dict[str, Any] = field(default_factory=dict)
    final_latents: dict[str, Any] = field(default_factory=dict)
    decoded_images: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    image_digest: str | None = None
    latent_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealCheckpointValidationReport:
    profile: ValidationProfile
    checkpoint: dict[str, Any]
    architecture_capability: ArchitectureCapability
    tokenizer: dict[str, Any] = field(default_factory=dict)
    component_coverage: dict[str, Any] = field(default_factory=dict)
    runs: list[ValidationRunRecord] = field(default_factory=list)
    comparisons: dict[str, Any] = field(default_factory=dict)
    checks: list[ValidationCheck] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    format: str = "image-gen-phase07-validation-v1"

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "created_utc": self.created_utc,
            "passed": self.passed,
            "profile": self.profile.to_dict(),
            "checkpoint": self.checkpoint,
            "architecture_capability": self.architecture_capability.to_dict(),
            "tokenizer": self.tokenizer,
            "component_coverage": self.component_coverage,
            "runs": [run.to_dict() for run in self.runs],
            "comparisons": self.comparisons,
            "checks": [check.to_dict() for check in self.checks],
            "environment": self.environment,
        }
