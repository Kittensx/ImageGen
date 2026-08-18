from __future__ import annotations

from typing import Any, Callable

import torch

from image_gen.runtime.lora_runtime import LoRARuntimeManager
from image_gen.runtime.model_preflight import (
    ModelPreflightMixin,
    _advanced_model_family,
    _optional_bool,
)
from image_gen.runtime.pipeline_factory import (
    PipelineFactoryMixin,
    _build_memory_event_callback,
    _compact_memory_event_payload,
)
from image_gen.runtime.request_execution import (
    RequestExecutionMixin,
    Txt2ImgRunResult,
    _prepare_output_directory,
    _verify_saved_records,
)
from image_gen.runtime.residency import ResidencyMixin
from image_gen.systems.diagnostics import DiagnosticsSystem
from image_gen.systems.model_loading import ModelLoadingSystem
from image_gen.systems.output import OutputSystem
from image_gen.systems.registry import RuntimeRegistrySystem
from modules.project_context import ProjectContext
from modules.shared_state import SharedState


class Txt2ImgRunner(
    ModelPreflightMixin,
    PipelineFactoryMixin,
    ResidencyMixin,
    RequestExecutionMixin,
):
    """Public txt2img use-case coordinator.

    Runtime responsibilities live in focused internal services while this class
    preserves the historical public API and coordinates shared dependencies.
    """

    def __init__(
        self,
        *,
        prompt_adapter: Any | None = None,
        scheduler_adapter: Any | None = None,
        sampler_adapter: Any | None = None,
        prompt_adapter_factory: Callable[..., Any] | None = None,
        scheduler_adapter_factory: Callable[..., Any] | None = None,
        sampler_adapter_factory: Callable[..., Any] | None = None,
        model_loader: Any | None = None,
        project_context: ProjectContext | None = None,
        state: Any | None = None,
        tokenizer: Any = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float | None = None,
        registry_system: RuntimeRegistrySystem | None = None,
        output_system: OutputSystem | None = None,
        model_loading_system: ModelLoadingSystem | None = None,
        diagnostics_system: DiagnosticsSystem | None = None,
        system_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.prompt_adapter = prompt_adapter
        self.scheduler_adapter = scheduler_adapter
        self.sampler_adapter = sampler_adapter
        self.prompt_adapter_factory = prompt_adapter_factory
        self.scheduler_adapter_factory = scheduler_adapter_factory
        self.sampler_adapter_factory = sampler_adapter_factory

        inherited_context = getattr(model_loader, "context", None)
        self.project_context = project_context or inherited_context or ProjectContext.load()
        if model_loader is None:
            from modules.load_safetensors_model import LoadModel

            model_loader = LoadModel(project_context=self.project_context)
        self.model_loader = model_loader
        self.state = state or SharedState()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.latent_scale_factor = int(latent_scale_factor)
        self.vae_scaling_factor = (None if vae_scaling_factor is None else float(vae_scaling_factor))
        self.system_overrides = dict(system_overrides or {})

        override_diagnostics = self.system_overrides.get("diagnostics")
        self.diagnostics_system = (
            diagnostics_system
            or override_diagnostics
            or DiagnosticsSystem.from_project_context(self.project_context)
        )
        self.system_overrides["diagnostics"] = self.diagnostics_system

        self.registry_system = registry_system or RuntimeRegistrySystem(
            self.state, project_context=self.project_context
        )
        self.registry_system.bind_state(self.state)
        self.output_system = output_system or OutputSystem()
        self.model_loading_system = model_loading_system or ModelLoadingSystem(self.model_loader)
        self.tokenizer = tokenizer
        self._tokenizer_identity = "injected" if tokenizer is not None else ""
        self._loaded_model_cache: dict[tuple[str, str, str, int, int], Any] = {}
        self.last_loaded_model: Any | None = None
        self.lora_runtime_manager = LoRARuntimeManager(self.project_context)

    def reset_runtime_state(self) -> None:
        """Reset request-scoped mutable state while retaining loaded components."""
        self.state = SharedState()
        self.registry_system.bind_state(self.state)
