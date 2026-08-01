from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from image_gen.contracts.runtime import (
    ConditioningOutput,
    GenerationRequest,
    PipelineComponents,
    SamplerOutput,
    SchedulerOutput,
)


@runtime_checkable
class RawModelFnProtocol(Protocol):
    def __call__(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class GuidedModelFnProtocol(Protocol):
    def __call__(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class DenoisedModelFnProtocol(Protocol):
    def __call__(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return the canonical predicted clean latent in float32."""
        ...


@runtime_checkable
class PromptAdapterProtocol(Protocol):
    def encode(
        self,
        components: PipelineComponents,
        request: GenerationRequest,
        state: Any | None = None,
    ) -> ConditioningOutput:
        ...


@runtime_checkable
class SchedulerAdapterProtocol(Protocol):
    def build_schedule(
        self,
        request: GenerationRequest,
        state: Any | None = None,
    ) -> SchedulerOutput:
        ...


@runtime_checkable
class SamplerAdapterProtocol(Protocol):
    def sample(
        self,
        raw_model_fn: RawModelFnProtocol,
        guided_model_fn: GuidedModelFnProtocol,
        latents: torch.Tensor,
        schedule: SchedulerOutput,
        conditioning: ConditioningOutput,
        request: GenerationRequest,
        state: Any | None = None,
    ) -> SamplerOutput:
        ...


class AdapterConformanceError(TypeError):
    pass


@dataclass(frozen=True)
class AdapterConformanceResult:
    adapter_kind: str
    adapter_name: str
    method_name: str
    expected_parameters: tuple[str, ...]
    actual_parameters: tuple[str, ...]
    is_conformant: bool
    reason: str = ""

    def raise_if_invalid(self) -> None:
        if not self.is_conformant:
            raise AdapterConformanceError(self.reason)


_EXPECTED_METHODS = {
    "prompt": ("encode", ("components", "request", "state")),
    "scheduler": ("build_schedule", ("request", "state")),
    "sampler": (
        "sample",
        (
            "raw_model_fn",
            "guided_model_fn",
            "latents",
            "schedule",
            "conditioning",
            "request",
            "state",
        ),
    ),
}


def check_adapter_conformance(adapter: type[Any] | Any, adapter_kind: str) -> AdapterConformanceResult:
    kind = str(adapter_kind).strip().lower()
    if kind not in _EXPECTED_METHODS:
        raise ValueError(f"Unknown adapter kind: {adapter_kind!r}")

    method_name, expected = _EXPECTED_METHODS[kind]
    adapter_name = getattr(adapter, "__name__", type(adapter).__name__)
    method = getattr(adapter, method_name, None)
    if method is None or not callable(method):
        return AdapterConformanceResult(
            adapter_kind=kind,
            adapter_name=adapter_name,
            method_name=method_name,
            expected_parameters=expected,
            actual_parameters=(),
            is_conformant=False,
            reason=f"{adapter_name} must define callable {method_name}(...).",
        )

    parameters = list(inspect.signature(method).parameters.values())
    if inspect.isclass(adapter) and parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    actual = tuple(parameter.name for parameter in parameters)

    is_conformant = actual == expected
    reason = ""
    if not is_conformant:
        reason = (
            f"{adapter_name}.{method_name} has parameters {actual}; "
            f"expected exactly {expected}."
        )
    elif parameters[-1].default is inspect.Parameter.empty:
        is_conformant = False
        reason = f"{adapter_name}.{method_name} must make state optional with a default value."

    return AdapterConformanceResult(
        adapter_kind=kind,
        adapter_name=adapter_name,
        method_name=method_name,
        expected_parameters=expected,
        actual_parameters=actual,
        is_conformant=is_conformant,
        reason=reason,
    )


def require_adapter_conformance(adapter: type[Any] | Any, adapter_kind: str) -> AdapterConformanceResult:
    result = check_adapter_conformance(adapter, adapter_kind)
    result.raise_if_invalid()
    return result
