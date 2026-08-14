from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import os
import time

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from modules.frozenclip.modules import FrozenCLIPEmbedder

from modules.config_resolver import ResolvedConfigs
from modules.state_dict_mapper import MappedStateDict
from modules.state_dict_converter import StateDictConverter
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
    unet: UNet2DConditionModel
    vae: Any
    text_encoder: Any

    unet_result: ComponentLoadResult
    vae_result: Any
    text_encoder_result: ComponentLoadResult
    placement_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    attention_backend_report: dict[str, Any] = field(default_factory=dict)
    text_encoder_2: Any = None
    text_encoder_2_result: ComponentLoadResult | None = None
    tokenizer: Any = None
    tokenizer_2: Any = None
    cpu_first_hydration: bool = False
    runtime_target_device: str = ""
    vae_force_upcast: bool = False
    vae_execution_dtype: str = ""


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
        self.deriver = ConfigDeriver()
        self.ldm_vae_builder = LDMVAEBuilder(device=self.device, dtype=self.dtype)

    def build_components(
        self,
        configs: ResolvedConfigs,
        mapped_state: MappedStateDict,
    ) -> BuiltComponents:
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
            text_encoder_2_placement = self._move_component(
                text_encoder_2, name="text_encoder_2"
            )
            if not configs.tokenizer_dir or not configs.tokenizer_2_dir:
                raise FileNotFoundError(
                    "SDXL requires tokenizer_dir and tokenizer_2_dir from the canonical SDXL runtime assets."
                )
            tokenizer = self._build_tokenizer(configs.tokenizer_dir)
            tokenizer_2 = self._build_tokenizer(configs.tokenizer_2_dir)

        placement_reports = {
            "unet": unet_placement,
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
        )

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