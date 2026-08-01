from __future__ import annotations

from typing import Any

from image_gen.contracts import GenerationRequest, SchedulerAdapterProtocol, SchedulerOutput


class SchedulingSystem:
    """Own full-schedule construction and canonical schedule validation.

    Image-conditioned policies select an active region after this system has
    returned a validated full schedule; scheduler adapters do not interpret
    hires denoising strength.
    """

    def __init__(self, adapter: SchedulerAdapterProtocol) -> None:
        self.adapter = adapter

    def build(self, request: GenerationRequest, state: Any | None = None) -> SchedulerOutput:
        output = self.adapter.build_schedule(request=request, state=state)
        if not isinstance(output, SchedulerOutput):
            raise TypeError("Scheduler adapter must return SchedulerOutput.")
        output.validate()
        return output
