from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import os
import time

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextConfig, CLIPTextModel

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


class ComponentBuilder:
    """
    Builds local model components from local config files and mapped checkpoint state.

    Responsibilities:
    - load JSON configs from disk
    - instantiate component classes locally
    - load split checkpoint weights into those components
    - return structured results
    """

    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.converter = StateDictConverter()
        self.deriver = ConfigDeriver()
        self.ldm_vae_builder = LDMVAEBuilder(device=self.device, dtype=self.dtype)

    def build_components(
        self,
        configs: ResolvedConfigs,
        mapped_state: MappedStateDict,
    ) -> BuiltComponents:
        converted = self.converter.convert_all(
            unet_state=mapped_state.unet,
            vae_state=mapped_state.vae,
            text_state=mapped_state.text_encoder,
        )

        unet = self._build_unet(configs.unet_config_path)

        unet_result = self._load_component_state(
            component=unet,
            state_dict=converted.unet,
            name="unet",
        )

        # Phase 14K-2.2: xFormers/MSLK validation must observe the loaded UNet
        # on its final runtime device and dtype. The former order configured
        # attention while the component was still CPU/float32, which triggered
        # Diffusers' synthetic float32 probe and a guessed head dimension.
        unet_placement = self._move_component(unet, name="unet")
        attention_initialization_started = time.perf_counter()
        model_attention_signature = build_model_attention_signature(unet)
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
        install_attention_layout_capture(
            unet,
            model_signature=model_attention_signature,
        )
        attention_report = attention_backend_report(unet)
        attention_report["initialization_metrics"] = {
            "duration_ms": round(attention_initialization_duration_ms, 3),
            "includes_model_signature": True,
            "includes_layout_validation": True,
            "includes_processor_attachment": True,
        }
        setattr(unet, "_image_gen_attention_backend_report", dict(attention_report))
        print(
            "UNet attention backend: "
            f"requested={attention_report['requested_backend']} "
            f"effective={attention_report['effective_backend']} "
            f"device={attention_report['validation_device']} "
            f"dtype={attention_report['validation_dtype']} "
            f"head_dims={model_attention_signature['unique_head_dimensions']} "
            f"processors={attention_report['processor_types_after']}"
        )
        vae_result = self.ldm_vae_builder.build_and_load(mapped_state.vae)
        vae = vae_result.model
        
        text_encoder = self._build_text_encoder(configs.text_encoder_config_path)

        text_encoder_result = self._load_component_state(
            component=text_encoder,
            state_dict=converted.text_encoder,
            name="text_encoder",
        )

        text_encoder_placement = self._move_component(text_encoder, name="text_encoder")
       

        return BuiltComponents(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            unet_result=unet_result,
            vae_result=vae_result,
            text_encoder_result=text_encoder_result,
            placement_reports={
                "unet": unet_placement,
                "vae": dict(getattr(vae, "_image_gen_placement_report", {}) or {}),
                "text_encoder": text_encoder_placement,
            },
            attention_backend_report=dict(attention_report),
           
        )

    def _build_text_encoder(self, config_path: str) -> CLIPTextModel:
        config_dict = self._load_json(config_path)
        text_config = CLIPTextConfig(**config_dict)
        return CLIPTextModel(text_config)
        
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