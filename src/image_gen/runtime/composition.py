from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from image_gen.contracts import (
    PipelineComponents,
    PromptAdapterProtocol,
    SamplerAdapterProtocol,
    SchedulerAdapterProtocol,
)
from image_gen.systems.conditioning import ConditioningSystem
from image_gen.systems.decoding import DecodingSystem
from image_gen.systems.denoising import DenoisingSystem
from image_gen.systems.diagnostics import DiagnosticsSystem
from image_gen.systems.sampling import LatentPreparationSystem, SamplingSystem
from image_gen.systems.scheduling import SchedulingSystem
from image_gen.systems.memory import AdaptiveComponentMemoryManager
from modules.component_placement import component_matches_placement, place_component


@dataclass(frozen=True)
class GenerationSystems:
    conditioning: Any
    scheduling: Any
    latent_preparation: Any
    denoising: Any
    sampling: Any
    decoding: Any
    diagnostics: Any

    def with_overrides(self, **overrides: Any) -> "GenerationSystems":
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown generation system override(s): {sorted(unknown)}")
        return replace(self, **overrides)


class PipelineCompositionRoot:
    """Construct a generation runtime from independent replaceable systems."""

    def __init__(
        self,
        *,
        components: PipelineComponents,
        prompt_adapter: PromptAdapterProtocol,
        scheduler_adapter: SchedulerAdapterProtocol,
        sampler_adapter: SamplerAdapterProtocol,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float = 0.18215,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        system_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.components = components
        self.prompt_adapter = prompt_adapter
        self.scheduler_adapter = scheduler_adapter
        self.sampler_adapter = sampler_adapter
        self.device = device or self._infer_device(components)
        self.dtype = dtype or self._infer_dtype(components, self.device)
        self.latent_scale_factor = int(latent_scale_factor)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.system_overrides = dict(system_overrides or {})

    @staticmethod
    def _infer_device(components: PipelineComponents) -> torch.device:
        for module in (components.unet, components.vae, components.text_encoder):
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _infer_dtype(components: PipelineComponents, device: torch.device) -> torch.dtype:
        for module in (components.unet, components.vae, components.text_encoder):
            try:
                return next(module.parameters()).dtype
            except StopIteration:
                continue
        return torch.float16 if device.type == "cuda" else torch.float32

    def prepare_components(self) -> None:
        component_map = {
            "unet": self.components.unet,
            "vae": self.components.vae,
            "text_encoder": self.components.text_encoder,
        }
        for name, module in component_map.items():
            if not component_matches_placement(
                module,
                device=self.device,
                dtype=self.dtype,
            ):
                # Real checkpoint components are already placed by ComponentBuilder.
                # This compatibility repair is only for injected/legacy components.
                place_component(
                    module,
                    device=self.device,
                    dtype=self.dtype,
                    owner="PipelineCompositionRoot.compatibility_repair",
                    component_name=name,
                )
            module.eval()

    def create_systems(self, *, prepare_components: bool = True) -> GenerationSystems:
        if prepare_components:
            self.prepare_components()
        else:
            for module in (self.components.unet, self.components.vae, self.components.text_encoder):
                module.eval()
        systems = GenerationSystems(
            conditioning=ConditioningSystem(self.prompt_adapter),
            scheduling=SchedulingSystem(self.scheduler_adapter),
            latent_preparation=LatentPreparationSystem(
                latent_scale_factor=self.latent_scale_factor,
                device=self.device,
                dtype=self.dtype,
            ),
            denoising=DenoisingSystem(
                self.components.unet,
                prediction_type=self.components.prediction_type,
                prediction_type_source=self.components.prediction_type_source,
            ),
            sampling=SamplingSystem(self.sampler_adapter),
            decoding=DecodingSystem(
                self.components.vae,
                vae_scaling_factor=self.vae_scaling_factor,
            ),
            diagnostics=DiagnosticsSystem(),
        )
        return systems.with_overrides(**self.system_overrides)

    def build(self, *, state: Any | None = None):
        from image_gen.runtime.generation_pipeline import GenerationPipeline

        state_extra = getattr(state, "extra", None)
        memory_policy = (
            str(state_extra.get("memory_policy") or "auto").strip().lower().replace(" ", "_")
            if isinstance(state_extra, dict)
            else "auto"
        )
        cpu_fallback = memory_policy == "cpu_fallback"
        if cpu_fallback:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        systems = self.create_systems(prepare_components=cpu_fallback)
        memory_manager = AdaptiveComponentMemoryManager.from_state(
            target_device=self.device,
            state=state,
        )
        memory_manager.register_core_components(self.components)
        return GenerationPipeline(
            components=self.components,
            systems=systems,
            state=state,
            device=self.device,
            dtype=self.dtype,
            latent_scale_factor=self.latent_scale_factor,
            vae_scaling_factor=self.vae_scaling_factor,
            memory_manager=memory_manager,
        )


class PipelineBuilder:
    """Compatibility-friendly builder backed by the canonical composition root."""

    def __init__(
        self,
        prompt_adapter: PromptAdapterProtocol,
        scheduler_adapter: SchedulerAdapterProtocol,
        sampler_adapter: SamplerAdapterProtocol,
        *,
        system_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.prompt_adapter = prompt_adapter
        self.scheduler_adapter = scheduler_adapter
        self.sampler_adapter = sampler_adapter
        self.system_overrides = dict(system_overrides or {})

    def build(
        self,
        built_components: Any,
        tokenizer: Any = None,
        state: Any | None = None,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float = 0.18215,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if isinstance(built_components, PipelineComponents):
            components = built_components
            if tokenizer is not None:
                components.tokenizer = tokenizer
        else:
            components = PipelineComponents(
                unet=built_components.unet,
                vae=built_components.vae,
                text_encoder=built_components.text_encoder,
                tokenizer=tokenizer,
                prediction_type=getattr(built_components, "prediction_type", "epsilon"),
                prediction_type_source=getattr(
                    built_components, "prediction_type_source", "legacy_built_components"
                ),
            )
        return PipelineCompositionRoot(
            components=components,
            prompt_adapter=self.prompt_adapter,
            scheduler_adapter=self.scheduler_adapter,
            sampler_adapter=self.sampler_adapter,
            latent_scale_factor=latent_scale_factor,
            vae_scaling_factor=vae_scaling_factor,
            device=device,
            dtype=dtype,
            system_overrides=self.system_overrides,
        ).build(state=state)
