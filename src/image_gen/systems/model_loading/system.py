from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, TYPE_CHECKING
from pathlib import Path

import torch

from image_gen.contracts import PipelineComponents
from image_gen.systems.validation.capabilities import capability_for
from image_gen.systems.memory.telemetry import MemoryTelemetry
from image_gen.contracts.vae_provenance import attach_vae_provenance

if TYPE_CHECKING:
    from modules.load_safetensors_model import LoadModel


@dataclass
class LoadedModel:
    components: PipelineComponents
    load_plan: Any
    built_components: Any
    memory_telemetry: dict[str, Any] | None = None


class ModelLoadingSystem:
    """Checkpoint loading boundary preserving the existing loader implementation."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    @staticmethod
    def _prediction_contract(plan: Any) -> tuple[str, str]:
        report = getattr(plan, "report", None)
        explicit = getattr(report, "prediction_type", None)
        if explicit:
            return str(explicit), "checkpoint_report"
        architecture = str(getattr(report, "architecture", "")).strip().lower()
        if architecture.startswith("sd1") or "stable-diffusion-1" in architecture:
            return "epsilon", "sd1_architecture_contract"
        if architecture.startswith("sd2") or "stable-diffusion-2" in architecture:
            return "v_prediction", "sd2_architecture_contract"
        return "epsilon", "legacy_supported_checkpoint_default"

    @property
    def default_model_path(self) -> str | None:
        value = getattr(self.loader, "MODEL_PATH", None)
        return str(value) if value else None

    def load(
        self,
        model_path: str,
        *,
        tokenizer: Any,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> LoadedModel:
        load_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        telemetry = MemoryTelemetry(device=load_device)
        before_memory = telemetry.capture("before_checkpoint_component_load").to_dict()
        plan = self.loader.prepare_load_plan(model_path)
        checkpoint_report = getattr(plan, "report", None)
        if checkpoint_report is not None:
            capability = capability_for(getattr(checkpoint_report, "architecture", "unknown"))
            if not capability.generation_supported:
                raise RuntimeError(
                    f"Checkpoint architecture {capability.architecture!r} is not enabled: "
                    f"{capability.reason}"
                )
        build_method = self.loader.build_components_from_plan
        try:
            parameters = inspect.signature(build_method).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_device = "device" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_device:
            built = build_method(plan, dtype=dtype, device=load_device)
        else:
            built = build_method(plan, dtype=dtype)
        failures = []
        for label, result in (
            ("UNet", built.unet_result),
            ("Text encoder", built.text_encoder_result),
            ("VAE", built.vae_result),
        ):
            if not result.success:
                failures.append(f"{label} failed: {result.error}")
        if failures:
            raise RuntimeError("; ".join(failures))
        prediction_type, prediction_type_source = self._prediction_contract(plan)
        model_identity = str(
            getattr(checkpoint_report, "sha256", "")
            or getattr(checkpoint_report, "model_path", "")
            or getattr(checkpoint_report, "file_name", "")
            or model_path
        )
        checkpoint_hash = str(getattr(checkpoint_report, "sha256", "") or "")
        checkpoint_path = str(
            getattr(checkpoint_report, "model_path", "")
            or getattr(checkpoint_report, "path", "")
            or model_path
        )
        vae_provenance = attach_vae_provenance(
            built.vae,
            {
                "source_kind": "embedded_checkpoint",
                "source_path": checkpoint_path,
                "sha256": checkpoint_hash,
                "identity": f"embedded_checkpoint:{checkpoint_hash}" if checkpoint_hash else f"embedded_checkpoint:{checkpoint_path}",
                "display_name": f"Embedded VAE ({getattr(checkpoint_report, 'file_name', '') or Path(checkpoint_path).name})",
                "embedded_in_checkpoint": True,
            },
        )
        return LoadedModel(
            components=PipelineComponents(
                unet=built.unet,
                vae=built.vae,
                text_encoder=built.text_encoder,
                tokenizer=tokenizer,
                prediction_type=prediction_type,
                prediction_type_source=prediction_type_source,
                model_identity=model_identity,
                model_hash=checkpoint_hash,
                vae_provenance=vae_provenance,
            ),
            load_plan=plan,
            built_components=built,
            memory_telemetry={
                "before_checkpoint_component_load": before_memory,
                "after_checkpoint_component_load": telemetry.capture(
                    "after_checkpoint_component_load",
                    component_residency=[],
                ).to_dict(),
            },
        )
