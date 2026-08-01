"""Canonical generation contracts used by every IMAGE_GEN system."""

from image_gen.contracts.compatibility import (
    SamplerCapabilities,
    SchedulerCompatibilityResult,
)
from image_gen.contracts.protocols import (
    AdapterConformanceError,
    AdapterConformanceResult,
    DenoisedModelFnProtocol,
    GuidedModelFnProtocol,
    PromptAdapterProtocol,
    RawModelFnProtocol,
    SamplerAdapterProtocol,
    SchedulerAdapterProtocol,
    check_adapter_conformance,
    require_adapter_conformance,
)
from image_gen.contracts.runtime import (
    ConditioningOutput,
    GenerationRequest,
    GenerationResult,
    PipelineComponents,
    SamplerOutput,
    SchedulerOutput,
)

__all__ = [
    "AdapterConformanceError",
    "AdapterConformanceResult",
    "ConditioningOutput",
    "GenerationRequest",
    "GenerationResult",
    "DenoisedModelFnProtocol",
    "GuidedModelFnProtocol",
    "PipelineComponents",
    "PromptAdapterProtocol",
    "RawModelFnProtocol",
    "SamplerAdapterProtocol",
    "SamplerCapabilities",
    "SamplerOutput",
    "SchedulerAdapterProtocol",
    "SchedulerCompatibilityResult",
    "SchedulerOutput",
    "check_adapter_conformance",
    "require_adapter_conformance",
]
