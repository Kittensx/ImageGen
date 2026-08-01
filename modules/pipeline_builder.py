"""Compatibility facade for the Phase 04 system-composed generation runtime."""

from image_gen.contracts import (  # historical re-exports
    ConditioningOutput,
    GenerationRequest,
    GenerationResult,
    GuidedModelFnProtocol,
    PipelineComponents,
    PromptAdapterProtocol,
    RawModelFnProtocol,
    SamplerAdapterProtocol,
    SamplerOutput,
    SchedulerAdapterProtocol,
    SchedulerOutput,
)
from image_gen.runtime import (
    CustomSDPipeline,
    GenerationPipeline,
    GenerationSystems,
    PipelineBuilder,
    PipelineCompositionRoot,
)

PromptAdapter = PromptAdapterProtocol
SchedulerAdapter = SchedulerAdapterProtocol
SamplerAdapter = SamplerAdapterProtocol

__all__ = [
    "ConditioningOutput",
    "CustomSDPipeline",
    "GenerationPipeline",
    "GenerationRequest",
    "GenerationResult",
    "GenerationSystems",
    "GuidedModelFnProtocol",
    "PipelineBuilder",
    "PipelineComponents",
    "PipelineCompositionRoot",
    "PromptAdapter",
    "PromptAdapterProtocol",
    "RawModelFnProtocol",
    "SamplerAdapter",
    "SamplerAdapterProtocol",
    "SamplerOutput",
    "SchedulerAdapter",
    "SchedulerAdapterProtocol",
    "SchedulerOutput",
]
