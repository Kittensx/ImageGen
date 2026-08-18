from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import gc
import json
import os
import time

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, SD3Transformer2DModel
from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5Config,
    T5EncoderModel,
)

from modules.frozenclip.modules import FrozenCLIPEmbedder

from modules.config_resolver import ResolvedConfigs
from modules.state_dict_mapper import MappedStateDict
from modules.state_dict_converter import StateDictConverter
from modules.sd3_state_dict_converter import SD3StateDictConverter
from modules.config_deriver import ConfigDeriver
from modules.ldm_vae_builder import LDMVAEBuilder, LDMVAEBuildResult
from modules.component_placement import place_component
from modules.attention_backend import (
    attention_backend_report,
    configure_unet_attention,
    configure_unet_attention_slicing,
)
from modules.attention_runtime import build_model_attention_signature, install_attention_layout_capture

@dataclass
class ComponentLoadResult:
    name: str
    loaded_keys: int
    expected_keys: int = 0
    matched_keys: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def coverage_ratio(self) -> float:
        if self.expected_keys <= 0:
            return 0.0
        return self.matched_keys / self.expected_keys

    def to_validation_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "provided_keys": self.loaded_keys,
            "expected_keys": self.expected_keys,
            "matched_keys": self.matched_keys,
            "coverage_ratio": self.coverage_ratio,
            "missing_key_count": len(self.missing_keys),
            "unexpected_key_count": len(self.unexpected_keys),
            "missing_key_samples": self.missing_keys[:25],
            "unexpected_key_samples": self.unexpected_keys[:25],
            "error": self.error,
        }


@dataclass
class BuiltComponents:
    unet: UNet2DConditionModel | None
    vae: Any
    text_encoder: Any

    unet_result: ComponentLoadResult | None
    vae_result: Any
    text_encoder_result: ComponentLoadResult
    placement_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    attention_backend_report: dict[str, Any] = field(default_factory=dict)
    text_encoder_2: Any = None
    text_encoder_2_result: ComponentLoadResult | None = None
    text_encoder_3: Any = None
    text_encoder_3_result: ComponentLoadResult | None = None
    tokenizer: Any = None
    tokenizer_2: Any = None
    tokenizer_3: Any = None
    cpu_first_hydration: bool = False
    runtime_target_device: str = ""
    vae_force_upcast: bool = False
    vae_execution_dtype: str = ""
    denoiser: Any = None
    denoiser_kind: str = "unet"
    denoiser_result: ComponentLoadResult | None = None
    conversion_reports: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.denoiser_kind = str(self.denoiser_kind or "unet").strip().lower() or "unet"
        if self.denoiser is None and self.denoiser_kind == "unet":
            self.denoiser = self.unet
        if self.denoiser_result is None and self.denoiser_kind == "unet":
            self.denoiser_result = self.unet_result
        if self.denoiser_kind == "unet" and self.unet is None:
            self.unet = self.denoiser
        if self.denoiser is self.unet and self.unet is not None and "denoiser" not in self.placement_reports:
            report = dict(self.placement_reports.get("unet", {}) or {})
            if report:
                self.placement_reports["denoiser"] = report


class ComponentBuilder:
    """
    Builds local model components from local config files and mapped checkpoint state.

    Responsibilities:
    - load JSON configs from disk
    - instantiate component classes locally
    - load split checkpoint weights into those components
    - return structured results
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        *,
        defer_attention_configuration: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.defer_attention_configuration = bool(defer_attention_configuration)
        self.converter = StateDictConverter()
        self.sd3_converter = SD3StateDictConverter()
        self.deriver = ConfigDeriver()
        self.ldm_vae_builder = LDMVAEBuilder(device=self.device, dtype=self.dtype)

    @staticmethod
    def _release_state_payloads(*payloads: Any) -> None:
        """Drop hydrated tensor dictionaries as soon as a component owns its weights.

        Advanced Models can source several components from different checkpoints. The
        runtime must not keep the raw donor tensors alive after ``load_state_dict`` has
        copied them into the component module, especially on low-memory Windows hosts
        where stale CPU mappings can push the process into pagefile exhaustion.
        """
        released = False
        for payload in payloads:
            if isinstance(payload, dict) and payload:
                payload.clear()
                released = True
        if released:
            gc.collect()

    def build_components(
        self,
        configs: ResolvedConfigs,
        mapped_state: MappedStateDict,
    ) -> BuiltComponents:
        if self._is_sd3_architecture(configs.architecture):
            return self._build_sd3_components(configs=configs, mapped_state=mapped_state)

        unet_config = self._load_json(configs.unet_config_path)
        converted = self.converter.convert_all(
            unet_state=mapped_state.unet,
            vae_state=mapped_state.vae,
            text_state=mapped_state.text_encoder,
            text_state_2=mapped_state.text_encoder_2,
            architecture=configs.architecture,
            unet_config=unet_config,
        )

        unet = UNet2DConditionModel.from_config(unet_config)

        unet_result = self._load_component_state(
            component=unet,
            state_dict=converted.unet,
            name="unet",
        )
        self._release_state_payloads(converted.unet, mapped_state.unet)

        # SDXL Phase 08 supports CPU-first hydration so an 8 GB card never has
        # UNet + VAE + both text encoders resident merely because the checkpoint
        # is being constructed. Attention backend qualification must observe the
        # UNet on its actual execution device, so CPU-first builds defer backend
        # activation until the memory manager leases the UNet for sampling.
        unet_placement = self._move_component(unet, name="unet")
        model_attention_signature = build_model_attention_signature(unet)
        install_attention_layout_capture(
            unet,
            model_signature=model_attention_signature,
        )
        if self.defer_attention_configuration:
            attention_report = attention_backend_report(unet)
            attention_report.update({
                "deferred": True,
                "deferred_reason": "cpu_first_sdxl_hydration",
                "runtime_configuration_required": True,
                "initialization_metrics": {
                    "duration_ms": 0.0,
                    "includes_model_signature": True,
                    "includes_layout_validation": False,
                    "includes_processor_attachment": False,
                },
            })
            setattr(unet, "_image_gen_attention_configuration_deferred", True)
            setattr(unet, "_image_gen_attention_backend_report", dict(attention_report))
        else:
            attention_initialization_started = time.perf_counter()
            attention_report = configure_unet_attention(
                unet,
                model_signature=model_attention_signature,
            )
            attention_report = configure_unet_attention_slicing(
                unet,
                backend_report=attention_report,
            )
            attention_initialization_duration_ms = (
                time.perf_counter() - attention_initialization_started
            ) * 1000.0
            attention_report = attention_backend_report(unet)
            attention_report["initialization_metrics"] = {
                "duration_ms": round(attention_initialization_duration_ms, 3),
                "includes_model_signature": True,
                "includes_layout_validation": True,
                "includes_processor_attachment": True,
            }
            setattr(unet, "_image_gen_attention_backend_report", dict(attention_report))
            setattr(unet, "_image_gen_attention_configuration_deferred", False)
            print(
                "UNet attention backend: "
                f"requested={attention_report['requested_backend']} "
                f"effective={attention_report['effective_backend']} "
                f"device={attention_report['validation_device']} "
                f"dtype={attention_report['validation_dtype']} "
                f"head_dims={model_attention_signature['unique_head_dimensions']} "
                f"processors={attention_report['processor_types_after']}"
            )
        architecture = str(configs.architecture or "").strip().lower()
        is_sdxl = architecture in {
            "sdxl",
            "stable-diffusion-xl",
            "stable-diffusion-xl-base",
        }
        vae_config = self._load_json(configs.vae_config_path)
        vae_force_upcast = bool(is_sdxl and vae_config.get("force_upcast", False))
        vae_dtype = torch.float32 if vae_force_upcast else self.dtype
        vae_builder = LDMVAEBuilder(device=self.device, dtype=vae_dtype)
        vae_result = vae_builder.build_and_load(mapped_state.vae)
        self._release_state_payloads(converted.vae, mapped_state.vae)
        vae = vae_result.model
        setattr(vae, "_image_gen_vae_force_upcast", vae_force_upcast)
        setattr(vae, "_image_gen_vae_execution_dtype", str(vae_dtype or "model_dtype"))

        text_encoder = self._build_text_encoder(configs.text_encoder_config_path)
        setattr(text_encoder, "_image_gen_architecture", str(configs.architecture or ""))

        text_encoder_result = self._load_component_state(
            component=text_encoder,
            state_dict=converted.text_encoder,
            name="text_encoder",
        )
        self._release_state_payloads(converted.text_encoder, mapped_state.text_encoder)

        text_encoder_placement = self._move_component(text_encoder, name="text_encoder")

        text_encoder_2 = None
        text_encoder_2_result = None
        text_encoder_2_placement: dict[str, Any] | None = None
        tokenizer = None
        tokenizer_2 = None
        if is_sdxl:
            if not configs.text_encoder_2_config_path:
                raise FileNotFoundError(
                    "SDXL requires text_encoder_2_config_path from the canonical SDXL runtime assets."
                )
            text_encoder_2 = self._build_text_encoder_2(configs.text_encoder_2_config_path)
            setattr(text_encoder_2, "_image_gen_architecture", str(configs.architecture or ""))
            setattr(text_encoder_2, "_image_gen_text_encoder_role", "text_encoder_2")
            text_encoder_2_result = self._load_component_state(
                component=text_encoder_2,
                state_dict=converted.text_encoder_2,
                name="text_encoder_2",
            )
            self._release_state_payloads(converted.text_encoder_2, mapped_state.text_encoder_2)
            text_encoder_2_placement = self._move_component(
                text_encoder_2, name="text_encoder_2"
            )
            if not configs.tokenizer_dir or not configs.tokenizer_2_dir:
                raise FileNotFoundError(
                    "SDXL requires tokenizer_dir and tokenizer_2_dir from the canonical SDXL runtime assets."
                )
            tokenizer = self._build_tokenizer(configs.tokenizer_dir)
            tokenizer_2 = self._build_tokenizer(configs.tokenizer_2_dir)

        self._release_state_payloads(mapped_state.extras)

        placement_reports = {
            "unet": unet_placement,
            "denoiser": dict(unet_placement),
            "vae": dict(getattr(vae, "_image_gen_placement_report", {}) or {}),
            "text_encoder": text_encoder_placement,
        }
        if text_encoder_2_placement is not None:
            placement_reports["text_encoder_2"] = text_encoder_2_placement

        return BuiltComponents(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            unet_result=unet_result,
            vae_result=vae_result,
            text_encoder_result=text_encoder_result,
            placement_reports=placement_reports,
            attention_backend_report=dict(attention_report),
            text_encoder_2=text_encoder_2,
            text_encoder_2_result=text_encoder_2_result,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae_force_upcast=vae_force_upcast,
            vae_execution_dtype=str(vae_dtype or "model_dtype"),
            denoiser=unet,
            denoiser_kind="unet",
            denoiser_result=unet_result,
        )

    @staticmethod
    def _is_sd3_architecture(architecture: str | None) -> bool:
        return str(architecture or "").strip().lower() in {
            "sd3",
            "sd3.x",
            "sd3.5",
            "stable-diffusion-3.x",
        }

    def _build_sd3_components(
        self,
        *,
        configs: ResolvedConfigs,
        mapped_state: MappedStateDict,
    ) -> BuiltComponents:
        if not configs.transformer_config_path:
            raise FileNotFoundError("SD3 requires transformer_config_path from local runtime assets.")
        if not configs.text_encoder_2_config_path:
            raise FileNotFoundError("SD3 requires text_encoder_2_config_path from local runtime assets.")

        transformer_config = self._load_json(configs.transformer_config_path)
        vae_config = self._load_json(configs.vae_config_path)
        clip_l_config = self._load_json(configs.text_encoder_config_path)
        clip_g_config = self._load_json(configs.text_encoder_2_config_path)

        converted_transformer = self.sd3_converter.convert_transformer(
            mapped_state.transformer,
            transformer_config,
        )
        transformer = SD3Transformer2DModel.from_config(transformer_config)
        transformer_result = self._load_component_state(
            component=transformer,
            state_dict=converted_transformer.state_dict,
            name="transformer",
        )
        self._release_state_payloads(converted_transformer.state_dict, mapped_state.transformer)
        transformer_placement = self._move_component(transformer, name="denoiser")
        setattr(transformer, "_image_gen_denoiser_kind", "transformer")
        setattr(transformer, "_image_gen_attention_configuration_deferred", True)

        # SD3's transformer attention is intentionally left on the native
        # Diffusers processor in this phase. Existing UNet xFormers/MSLK setup
        # is not assumed compatible with MMDiT.
        attention_report = {
            "deferred": True,
            "deferred_reason": "sd3_transformer_attention_backend_not_qualified",
            "runtime_configuration_required": False,
            "effective_backend": "diffusers_native",
            "component_kind": "transformer",
        }
        setattr(transformer, "_image_gen_attention_backend_report", dict(attention_report))

        vae = AutoencoderKL.from_config(vae_config)
        converted_vae = self.sd3_converter.convert_vae(mapped_state.vae, vae.config)
        vae_result = self._load_component_state(
            component=vae,
            state_dict=converted_vae.state_dict,
            name="vae",
        )
        self._release_state_payloads(converted_vae.state_dict, mapped_state.vae)
        vae_force_upcast = bool(vae_config.get("force_upcast", False))
        vae_dtype = torch.float32 if vae_force_upcast else self.dtype
        vae_placement = place_component(
            vae,
            device=self.device,
            dtype=vae_dtype,
            owner="ComponentBuilder",
            component_name="vae",
        ).to_dict()
        setattr(vae, "_image_gen_vae_force_upcast", vae_force_upcast)
        setattr(vae, "_image_gen_vae_execution_dtype", str(vae_dtype or "model_dtype"))

        clip_l = self._build_sd3_clip(configs.text_encoder_config_path)
        setattr(clip_l, "_image_gen_architecture", str(configs.architecture or ""))
        setattr(clip_l, "_image_gen_text_encoder_role", "clip_l")
        converted_clip_l = self.sd3_converter.convert_clip_l(mapped_state.text_encoder, clip_l_config)
        clip_l_result = self._load_component_state(
            component=clip_l,
            state_dict=converted_clip_l.state_dict,
            name="clip_l",
        )
        self._release_state_payloads(converted_clip_l.state_dict, mapped_state.text_encoder)
        clip_l_placement = self._move_component(clip_l, name="text_encoder")

        clip_g = self._build_sd3_clip(configs.text_encoder_2_config_path)
        setattr(clip_g, "_image_gen_architecture", str(configs.architecture or ""))
        setattr(clip_g, "_image_gen_text_encoder_role", "clip_g")
        converted_clip_g = self.sd3_converter.convert_clip_g(mapped_state.text_encoder_2)
        clip_g_result = self._load_component_state(
            component=clip_g,
            state_dict=converted_clip_g.state_dict,
            name="clip_g",
        )
        self._release_state_payloads(converted_clip_g.state_dict, mapped_state.text_encoder_2)
        clip_g_placement = self._move_component(clip_g, name="text_encoder_2")

        tokenizer = self._build_tokenizer(configs.tokenizer_dir) if configs.tokenizer_dir else None
        tokenizer_2 = self._build_tokenizer(configs.tokenizer_2_dir) if configs.tokenizer_2_dir else None

        text_encoder_3 = None
        text_encoder_3_result = None
        tokenizer_3 = None
        text_encoder_3_placement = None
        if mapped_state.text_encoder_3:
            if not configs.text_encoder_3_config_path:
                raise FileNotFoundError("SD3 T5 selection requires text_encoder_3_config_path from local runtime assets.")
            if not configs.tokenizer_3_dir:
                raise FileNotFoundError("SD3 T5 selection requires tokenizer_3_dir from local runtime assets.")
            t5_config = T5Config.from_dict(self._load_json(configs.text_encoder_3_config_path))
            previous_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                text_encoder_3 = T5EncoderModel(t5_config).eval()
            finally:
                torch.set_default_dtype(previous_default_dtype)
            text_encoder_3 = text_encoder_3.to(device="cpu", dtype=torch.bfloat16)
            setattr(text_encoder_3, "_image_gen_architecture", str(configs.architecture or ""))
            setattr(text_encoder_3, "_image_gen_text_encoder_role", "t5xxl")
            text_encoder_3_result = self._load_component_state(
                component=text_encoder_3,
                state_dict=mapped_state.text_encoder_3,
                name="t5xxl",
            )
            self._release_state_payloads(mapped_state.text_encoder_3)
            text_encoder_3_placement = place_component(
                text_encoder_3,
                device="cpu",
                dtype=torch.bfloat16,
                owner="ComponentBuilder.sd3_t5_cpu_first",
                component_name="text_encoder_3",
            ).to_dict()
            tokenizer_3 = AutoTokenizer.from_pretrained(
                str(configs.tokenizer_3_dir),
                local_files_only=True,
            )

        self._release_state_payloads(mapped_state.extras)

        placement_reports = {
            "denoiser": transformer_placement,
            "transformer": dict(transformer_placement),
            "vae": vae_placement,
            "text_encoder": clip_l_placement,
            "text_encoder_2": clip_g_placement,
        }
        if text_encoder_3_placement is not None:
            placement_reports["text_encoder_3"] = text_encoder_3_placement

        conversion_reports = {
            "transformer": converted_transformer.report.to_dict(),
            "vae": converted_vae.report.to_dict(),
            "clip_l": converted_clip_l.report.to_dict(),
            "clip_g": converted_clip_g.report.to_dict(),
        }

        return BuiltComponents(
            unet=None,
            vae=vae,
            text_encoder=clip_l,
            unet_result=None,
            vae_result=vae_result,
            text_encoder_result=clip_l_result,
            placement_reports=placement_reports,
            attention_backend_report=attention_report,
            text_encoder_2=clip_g,
            text_encoder_2_result=clip_g_result,
            text_encoder_3=text_encoder_3,
            text_encoder_3_result=text_encoder_3_result,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            vae_force_upcast=vae_force_upcast,
            vae_execution_dtype=str(vae_dtype or "model_dtype"),
            denoiser=transformer,
            denoiser_kind="transformer",
            denoiser_result=transformer_result,
            conversion_reports=conversion_reports,
        )

    def _build_sd3_clip(self, config_path: str) -> CLIPTextModelWithProjection:
        config_dict = self._load_json(config_path)
        text_config = CLIPTextConfig(**config_dict)
        return CLIPTextModelWithProjection(text_config)

    def _build_tokenizer(self, tokenizer_dir: str) -> CLIPTokenizer:
        return CLIPTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )

    def _build_text_encoder(self, config_path: str) -> CLIPTextModel:
        config_dict = self._load_json(config_path)
        text_config = CLIPTextConfig(**config_dict)
        return CLIPTextModel(text_config)

    def _build_text_encoder_2(self, config_path: str) -> CLIPTextModelWithProjection:
        config_dict = self._load_json(config_path)
        text_config = CLIPTextConfig(**config_dict)
        return CLIPTextModelWithProjection(text_config)
        
    def _build_unet(self, config_path: str) -> UNet2DConditionModel:
        config = self._load_json(config_path)
        return UNet2DConditionModel.from_config(config)

    def _build_vae(self, config_path: str, mapped_vae_state: dict[str, Any]) -> AutoencoderKL:
        derived = self.deriver.derive_vae_config(mapped_vae_state)
        print(f"Derived VAE block_out_channels: {derived.block_out_channels}")
        config = derived.to_diffusers_config()
        return AutoencoderKL.from_config(config)
    
    def _load_component_state(
        self,
        component: torch.nn.Module,
        state_dict: dict[str, Any],
        name: str,
    ) -> ComponentLoadResult:
        expected = set(component.state_dict().keys())
        provided = set(state_dict.keys())
        matched = expected.intersection(provided)
        if not state_dict:
            return ComponentLoadResult(
                name=name,
                loaded_keys=0,
                expected_keys=len(expected),
                matched_keys=0,
                error=f"No state_dict entries found for component '{name}'.",
            )

        try:
            incompatible = component.load_state_dict(state_dict, strict=False)

            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

            return ComponentLoadResult(
                name=name,
                loaded_keys=len(state_dict),
                expected_keys=len(expected),
                matched_keys=len(matched),
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
                error=None,
            )
        except Exception as e:
            return ComponentLoadResult(
                name=name,
                loaded_keys=len(state_dict),
                expected_keys=len(expected),
                matched_keys=len(matched),
                error=str(e),
            )

    def _move_component(
        self,
        component: torch.nn.Module,
        *,
        name: str,
    ) -> dict[str, Any]:
        return place_component(
            component,
            device=self.device,
            dtype=self.dtype,
            owner="ComponentBuilder",
            component_name=name,
        ).to_dict()

    def _load_json(self, path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)