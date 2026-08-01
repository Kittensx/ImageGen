from __future__ import annotations

from typing import Any

from image_gen.contracts import ConditioningOutput, GenerationRequest, PipelineComponents, PromptAdapterProtocol


class ConditioningSystem:
    """Own prompt parsing/tokenization/encoding through one adapter boundary."""

    def __init__(self, adapter: PromptAdapterProtocol) -> None:
        self.adapter = adapter

    def encode(
        self,
        components: PipelineComponents,
        request: GenerationRequest,
        state: Any | None = None,
    ) -> ConditioningOutput:
        output = self.adapter.encode(components=components, request=request, state=state)
        if not isinstance(output, ConditioningOutput):
            raise TypeError("Conditioning adapter must return ConditioningOutput.")
        if output.cond.shape[0] != output.uncond.shape[0]:
            raise ValueError("Conditional and unconditional batches must have equal size.")
        return output
