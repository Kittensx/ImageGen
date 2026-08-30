from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, TYPE_CHECKING
from types import SimpleNamespace
from pathlib import Path

import torch

from image_gen.contracts import PipelineComponents, resolve_latent_vae_contract
from image_gen.runtime.model_load_variant import sanitize_model_load_runtime_settings
from image_gen.systems.validation.capabilities import capability_for
from image_gen.systems.memory.telemetry import MemoryTelemetry
from image_gen.contracts.vae_provenance import attach_vae_provenance
from image_gen.systems.model_loading.vae_override import apply_external_vae_override
from modules.advanced_model_composition import apply_advanced_component_composition
from modules.registry.composition_projection import project_runtime_composition

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
        if architecture == "sd3.x":
            return "not_applicable_flow_match", "sd3_flow_match_contract"
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
        extras = sanitize_model_load_runtime_settings(request_extras)
        allow_sd2_validation = bool(extras.get("sd2_dedicated_generation"))
        allow_sdxl_validation = bool(
            extras.get("sdxl_phase08_validation")
            or extras.get("sdxl_dedicated_generation")
        )
        allow_sd3_validation = bool(extras.get("sd3_phase04_validation"))
        allow_validation_generation = bool(allow_sd2_validation or allow_sdxl_validation or allow_sd3_validation)
        explicit_sd2_profile = str(extras.get("sd2_runtime_profile_override") or "").strip() or None
        explicit_sdxl_profile = str(extras.get("sdxl_runtime_profile_override") or "").strip() or None
        explicit_sd3_profile = str(extras.get("sd3_runtime_profile_override") or "").strip() or None
        prepare_method = self.loader.prepare_load_plan
        prepare_kwargs: dict[str, Any] = {
            "require_generation_support": not allow_validation_generation,
            "explicit_sd2_runtime_profile": explicit_sd2_profile,
            "explicit_sdxl_runtime_profile": explicit_sdxl_profile,
            "explicit_sd3_runtime_profile": explicit_sd3_profile,
        }
        try:
            prepare_parameters = inspect.signature(prepare_method).parameters
        except (TypeError, ValueError):
            prepare_parameters = {}
        if (
            "request_extras" in prepare_parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in prepare_parameters.values())
        ):
            prepare_kwargs["request_extras"] = extras
        plan = prepare_method(model_path, **prepare_kwargs)
        advanced_composition = apply_advanced_component_composition(
            plan,
            extras.get("_advanced_model_resolved"),
            inspector=getattr(self.loader, "inspector", None),
            mapper=getattr(self.loader, "mapper", None),
            runtime_source_plan=extras.get("_runtime_component_source_plan"),
        )
        extras["_advanced_model_applied_composition"] = dict(advanced_composition)
        # Preserve SD3-12 whole-checkpoint behavior: embedded T5 is optional and
        # remains disabled unless the user explicitly selects a T5 component in
        # Advanced Models. The builder can hydrate T5, but selection is authoritative.
        if (
            not advanced_composition
            and str(getattr(getattr(plan, "report", None), "architecture", "") or "").strip().lower() == "sd3.x"
            and getattr(plan, "mapped_state", None) is not None
        ):
            plan.mapped_state.text_encoder_3 = {}
        checkpoint_report = getattr(plan, "report", None)
        if checkpoint_report is not None:
            capability = capability_for(getattr(checkpoint_report, "architecture", "unknown"))
            if allow_validation_generation:
                architecture = str(getattr(checkpoint_report, "architecture", "")).strip().lower()
                allowed = (
                    architecture == "sd2.x" and allow_sd2_validation
                ) or (
                    architecture == "sdxl" and allow_sdxl_validation
                ) or (
                    architecture == "sd3.x" and allow_sd3_validation
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
        accepts_request_extras = "request_extras" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        build_kwargs: dict[str, Any] = {"dtype": dtype}
        if accepts_device:
            build_kwargs["device"] = load_device
        if accepts_request_extras:
            build_kwargs["request_extras"] = extras
        built = build_method(plan, **build_kwargs)
        failures = []
        denoiser_kind = str(getattr(built, "denoiser_kind", "unet") or "unet").strip().lower()
        denoiser_result = getattr(built, "denoiser_result", None) or getattr(built, "unet_result", None)
        component_results = [
            ("Transformer" if denoiser_kind == "transformer" else "UNet", denoiser_result),
            ("Text encoder", built.text_encoder_result),
            ("VAE", built.vae_result),
        ]
        text_encoder_2_result = getattr(built, "text_encoder_2_result", None)
        if text_encoder_2_result is not None:
            component_results.append(("Text encoder 2", text_encoder_2_result))
        text_encoder_3_result = getattr(built, "text_encoder_3_result", None)
        if text_encoder_3_result is not None:
            component_results.append(("Text encoder 3 / T5", text_encoder_3_result))
        for label, result in component_results:
            if result is None:
                failures.append(f"{label} failed: no component load result was produced")
                continue
            if not result.success:
                failures.append(f"{label} failed: {result.error}")
        if failures:
            raise RuntimeError("; ".join(failures))
        prediction_type, prediction_type_source = self._prediction_contract(plan)
        composition_sha256 = str(advanced_composition.get("composition_sha256") or "")
        model_identity = str(
            (f"advanced:{composition_sha256}" if composition_sha256 else "")
            or getattr(checkpoint_report, "sha256", "")
            or getattr(checkpoint_report, "model_path", "")
            or getattr(checkpoint_report, "file_name", "")
            or model_path
        )
        checkpoint_hash = composition_sha256 or str(getattr(checkpoint_report, "sha256", "") or "")
        checkpoint_path = str(
            getattr(checkpoint_report, "model_path", "")
            or getattr(checkpoint_report, "path", "")
            or model_path
        )
        advanced_vae = dict((advanced_composition.get("components") or {}).get("vae") or {})
        if advanced_vae:
            advanced_vae_path = str(advanced_vae.get("source_path") or "")
            advanced_vae_hash = str(advanced_vae.get("component_sha256") or "")
            advanced_vae_embedded = str(advanced_vae.get("source_asset_type") or "") == "checkpoint"
            vae_provenance = attach_vae_provenance(
                built.vae,
                {
                    "source_kind": "advanced_component",
                    "source_path": advanced_vae_path,
                    "sha256": advanced_vae_hash,
                    "identity": f"component_sha256:{advanced_vae_hash}",
                    "display_name": f"Advanced VAE ({advanced_vae_hash[:8]})",
                    "embedded_in_checkpoint": advanced_vae_embedded,
                },
            )
        else:
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
            override = apply_external_vae_override(
                built.vae,
                override_path,
                project_context=getattr(self.loader, "context", None),
            )
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
        sd2_contract = getattr(plan, "sd2_contract", None)
        sdxl_contract = getattr(plan, "sdxl_contract", None)
        sd3_contract = getattr(plan, "sd3_contract", None)
        if sd3_contract is not None:
            model_runtime_profile = dict(sd3_contract.profile.to_dict())
            model_runtime_profile["text_encoder_sources"] = dict(
                getattr(built, "sd3_text_encoder_sources", {}) or {}
            )
            if advanced_composition:
                model_runtime_profile["advanced_model_composition"] = dict(advanced_composition)
                model_runtime_profile["t5_device"] = str(advanced_composition.get("t5_device") or "off")
        elif sdxl_contract is not None:
            model_runtime_profile = dict(sdxl_contract.profile.to_dict())
        elif sd2_contract is not None:
            model_runtime_profile = dict(sd2_contract.profile.to_dict())
        else:
            model_runtime_profile = {}
        vae_scaling_factor = float(
            sd3_contract.assets.vae_payload().get("scaling_factor", 1.5305)
            if sd3_contract is not None
            else (
                getattr(sdxl_contract, "vae_scaling_factor", 0.18215)
                if sdxl_contract is not None else 0.18215
            )
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

        latent_vae_contract = resolve_latent_vae_contract(
            SimpleNamespace(
                vae=active_vae,
                latent_channels=(
                    int(getattr(sd3_contract, "latent_channels", 16))
                    if sd3_contract is not None
                    else 4
                ),
                latent_scale_factor=8,
                vae_scaling_factor=vae_scaling_factor,
                vae_shift_factor=(
                    float(sd3_contract.assets.vae_payload().get("shift_factor", 0.0609))
                    if sd3_contract is not None
                    else 0.0
                ),
                vae_force_upcast=vae_force_upcast,
            )
        )
        setattr(active_vae, "_image_gen_latent_vae_contract", latent_vae_contract.to_serializable_dict())

        architecture = str(getattr(checkpoint_report, "architecture", "") or "")
        composition_projection = project_runtime_composition(
            plan,
            registry=getattr(self.loader, "asset_registry", None),
            advanced_composition=advanced_composition,
            sd3_text_encoder_sources=dict(getattr(built, "sd3_text_encoder_sources", {}) or {}),
            vae_provenance=vae_provenance,
            text_encoder_3_device=extras.get("text_encoder_3_device") or advanced_composition.get("t5_device") or "auto",
        )
        composition_projection_payload = composition_projection.to_dict()
        composition_contract = dict(composition_projection_payload.get("composition_contract") or {})
        composition_sha256_unified = str(composition_projection_payload.get("composition_sha256") or "")
        advanced_model_composition_sha256 = str(advanced_composition.get("composition_sha256") or "")
        component_transition_report = dict(extras.get("component_transition_report") or {})
        if component_transition_report:
            component_transition_report["requested_composition_sha256"] = composition_sha256_unified
            component_transition_report["loaded_component_count"] = sum(
                1
                for item in dict(component_transition_report.get("role_diff") or {}).values()
                if str(item.get("action") or "") in {"replace", "add", "cannot_reuse"}
            )
            component_transition_report["released_component_count"] = sum(
                1
                for item in dict(component_transition_report.get("role_diff") or {}).values()
                if str(item.get("action") or "") in {"replace", "remove", "cannot_reuse"}
            )
            extras["component_transition_report"] = component_transition_report
        return LoadedModel(
            components=PipelineComponents(
                unet=built.unet,
                vae=active_vae,
                text_encoder=built.text_encoder,
                tokenizer=built_tokenizer if built_tokenizer is not None else tokenizer,
                text_encoder_2=getattr(built, "text_encoder_2", None),
                tokenizer_2=built_tokenizer_2,
                text_encoder_3=getattr(built, "text_encoder_3", None),
                tokenizer_3=getattr(built, "tokenizer_3", None),
                prediction_type=prediction_type,
                prediction_type_source=prediction_type_source,
                architecture=architecture,
                model_runtime_profile=model_runtime_profile,
                latent_channels=latent_vae_contract.latent_channels,
                latent_scale_factor=latent_vae_contract.latent_scale_factor,
                vae_scaling_factor=latent_vae_contract.scaling_factor,
                vae_shift_factor=latent_vae_contract.shift_factor,
                vae_force_upcast=latent_vae_contract.force_upcast,
                vae_use_quant_conv=latent_vae_contract.use_quant_conv,
                vae_use_post_quant_conv=latent_vae_contract.use_post_quant_conv,
                vae_execution_dtype=vae_execution_dtype,
                model_identity=model_identity,
                model_hash=checkpoint_hash,
                vae_provenance=vae_provenance,
                composition_sha256=composition_sha256_unified,
                composition_identity_version=str(composition_contract.get("identity_version") or ""),
                composition_contract=composition_contract,
                component_sources={
                    str(role): dict(source)
                    for role, source in dict(composition_projection_payload.get("component_sources") or {}).items()
                },
                composition_projection=composition_projection_payload,
                advanced_model_composition_sha256=advanced_model_composition_sha256,
                component_transition_report=component_transition_report,
                runtime_component_source_plan=dict(extras.get("_runtime_component_source_plan") or getattr(plan, "runtime_source_plan", {}) or {}),
                denoiser=getattr(built, "denoiser", None),
                denoiser_kind=str(getattr(built, "denoiser_kind", "unet") or "unet"),
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
