from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from image_gen.contracts import GenerationRequest
from image_gen.runtime.performance_metrics import GenerationPerformanceRecorder
from image_gen.systems.diagnostics import DiagnosticSession, DiagnosticsSystem


@dataclass
class GenerationContext:
    """Mutable request-scoped state passed between generation stages.

    The context deliberately contains only values that cross stage boundaries.
    Stage-local temporaries stay local to their owning stage so the orchestration
    layer does not become a second copy of the old monolithic ``generate`` frame.
    """

    request: GenerationRequest
    diagnostics: DiagnosticsSystem
    session: DiagnosticSession
    owns_session: bool
    performance: GenerationPerformanceRecorder
    provider_execution_before_generation: dict[str, Any]
    provider_execution_after_base: dict[str, Any]
    provider_execution_after_hires: dict[str, Any]
    auxiliary_images: dict[str, Any]
    outpaint_enabled: bool
    outpaint_source: torch.Tensor | None
    outpaint_canvas: torch.Tensor | None
    outpaint_masks: dict[str, torch.Tensor]
    outpaint_latent_mask: torch.Tensor | None
    outpaint_plan: Any
    outpaint_hook: Any
    outpaint_prompt_contract: dict[str, Any]
    outpaint_metadata: dict[str, Any]
    outpaint_stage: Callable[[str, Any], Any]
    configure_vae_memory: Any
    base_vae_memory_controls: dict[str, Any]
    pixel_hires_job: bool
    pixel_hires_preflight: Any
    pixel_hires_stage_timings: dict[str, float]
    pixel_hires_cancelled_stage: str

    dimension_plan: Any = None
    base_dimension_plan: Any = None
    final_output_width: int = 0
    final_output_height: int = 0
    hires_execution_plan: Any = None
    hires_metadata: dict[str, Any] = field(default_factory=dict)
    state_extra: dict[str, Any] = field(default_factory=dict)
    preview_mode: str = "disabled"
    base_preview_policy: Any = None
    preview_policy_report: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    conditioning: Any = None
    latents: Any = None
    schedule: Any = None
    raw_model_fn: Any = None
    guided_model_fn: Any = None
    sample_output: Any = None
    diagnostic_decode_enabled: bool = False
    diagnostic_decode_source: str = ""
    images: Any = None
    output_quality: dict[str, Any] = field(default_factory=dict)
    trace_exports: list[Any] = field(default_factory=list)
