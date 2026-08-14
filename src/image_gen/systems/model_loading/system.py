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
from image_gen.systems.model_loading.vae_override import apply_external_vae_override

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
            sd2_contract = getattr(plan, "sd2_contract", None)
            resolved = str(getattr(sd2_contract, "prediction_type", "") or "")
            if resolved:
                return resolved, str(
                    getattr(sd2_contract, "prediction_type_source", "runtime_scheduler_config")
                    or "runtime_scheduler_config"
                )
            raise RuntimeError(
                "SD2 prediction type is unresolved. IMAGE_GEN no longer guesses v_prediction for all SD2 checkpoints; "
                "the checkpoint must provide metadata or resolve through a qualified SD2 runtime profile."
            )
        if architecture == "sdxl":
            sdxl_contract = getattr(plan, "sdxl_contract", None)
            resolved = str(getattr(sdxl_contract, "prediction_type", "") or "")
            if resolved:
                return resolved, str(
                    getattr(sdxl_contract, "prediction_type_source", "sdxl_runtime_profile")
                    or "sdxl_runtime_profile"
                )
            raise RuntimeError("SDXL prediction type is unresolved; a qualified SDXL runtime profile is required.")
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
        request_extras: dict[str, Any] | None = None,
    ) -> LoadedModel:
        load_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        telemetry = MemoryTelemetry(device=load_device)
        before_memory = telemetry.capture("before_checkpoint_component_load").to_dict()
        extras = dict(request_extras or {})
        allow_sd2_validation = bool(extras.get("sd2_dedicated_generation"))
        allow_sdxl_validation = bool(
            extras.get("sdxl_phase08_validation")
            or extras.get("sdxl_dedicated_generation")
        )
        allow_validation_generation = bool(allow_sd2_validation or allow_sdxl_validation)
        explicit_sd2_profile = str(extras.get("sd2_runtime_profile_override") or "").strip() or None
        explicit_sdxl_profile = str(extras.get("sdxl_runtime_profile_override") or "").strip() or None
        plan = self.loader.prepare_load_plan(
            model_path,
            require_generation_support=not allow_validation_generation,
            explicit_sd2_runtime_profile=explicit_sd2_profile,
            explicit_sdxl_runtime_profile=explicit_sdxl_profile,
        )
        checkpoint_report = getattr(plan, "report", None)
        if checkpoint_report is not None:
            capability = capability_for(getattr(checkpoint_report, "architecture", "unknown"))
            if allow_validation_generation:
                architecture = str(getattr(checkpoint_report, "architecture", "")).strip().lower()
                allowed = (
                    architecture == "sd2.x" and allow_sd2_validation
                ) or (
                    architecture == "sdxl" and allow_sdxl_validation
                )
                if not allowed:
                    raise RuntimeError(
                        "Dedicated validation generation override does not match the detected checkpoint architecture."
                    )
                if not (capability.validation_supported or capability.generation_supported):
                    raise RuntimeError(
                        f"Checkpoint architecture {capability.architecture!r} is not enabled for dedicated validation generation: "
                        f"{capability.reason}"
                    )
            elif not capability.generation_supported:
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
        component_results = [
            ("UNet", built.unet_result),
            ("Text encoder", built.text_encoder_result),
            ("VAE", built.vae_result),
        ]
        text_encoder_2_result = getattr(built, "text_encoder_2_result", None)
        if text_encoder_2_result is not None:
            component_results.append(("Text encoder 2", text_encoder_2_result))
        for label, result in component_results:
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
        override_path = str(extras.get("vae_path") or "").strip()
        active_vae = built.vae
        if override_path:
            override = apply_external_vae_override(built.vae, override_path)
            active_vae = override.vae
            vae_provenance = override.provenance
            # Keep the compatibility view aligned with the effective runtime
            # component. This also releases the superseded embedded VAE once no
            # other references remain, instead of retaining two VAEs for the
            # lifetime of the loaded model.
            built.vae = active_vae
            placement_reports = getattr(built, "placement_reports", None)
            if isinstance(placement_reports, dict):
                placement_reports["vae"] = dict(
                    getattr(active_vae, "_image_gen_placement_report", {}) or {}
                )
        built_tokenizer = getattr(built, "tokenizer", None)
        built_tokenizer_2 = getattr(built, "tokenizer_2", None)
        sdxl_contract = getattr(plan, "sdxl_contract", None)
        model_runtime_profile = (
            dict(sdxl_contract.profile.to_dict()) if sdxl_contract is not None else {}
        )
        vae_scaling_factor = float(
            getattr(sdxl_contract, "vae_scaling_factor", 0.18215)
            if sdxl_contract is not None else 0.18215
        )
        vae_force_upcast = bool(
            getattr(sdxl_contract, "vae_force_upcast", False)
            if sdxl_contract is not None else getattr(built, "vae_force_upcast", False)
        )
        vae_execution_dtype = str(
            getattr(built, "vae_execution_dtype", "")
            or ("torch.float32" if vae_force_upcast else "")
        )
        if vae_force_upcast:
            active_vae.to(dtype=torch.float32)
            setattr(active_vae, "_image_gen_vae_force_upcast", True)
            setattr(active_vae, "_image_gen_vae_execution_dtype", "torch.float32")
        architecture = str(getattr(checkpoint_report, "architecture", "") or "")
        return LoadedModel(
            components=PipelineComponents(
                unet=built.unet,
                vae=active_vae,
                text_encoder=built.text_encoder,
                tokenizer=built_tokenizer if built_tokenizer is not None else tokenizer,
                text_encoder_2=getattr(built, "text_encoder_2", None),
                tokenizer_2=built_tokenizer_2,
                prediction_type=prediction_type,
                prediction_type_source=prediction_type_source,
                architecture=architecture,
                model_runtime_profile=model_runtime_profile,
                vae_scaling_factor=vae_scaling_factor,
                vae_force_upcast=vae_force_upcast,
                vae_execution_dtype=vae_execution_dtype,
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
                    component_residency=[
                        {"component_id": name, **dict(report)}
                        for name, report in dict(getattr(built, "placement_reports", {}) or {}).items()
                    ],
                ).to_dict(),
                "cpu_first_hydration": bool(getattr(built, "cpu_first_hydration", False)),
                "runtime_target_device": str(getattr(built, "runtime_target_device", load_device)),
            },
        )
