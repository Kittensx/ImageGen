from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

STANDARD_DIFFUSERS_LOADER_ID = "image_gen.adapter_loader.standard_diffusers.v1"


@dataclass(frozen=True)
class AdapterLoaderCapability:
    loader_id: str
    supported_formats: frozenset[str]
    supported_families: frozenset[str]
    supported_targets: frozenset[str]
    allow_unknown_family: bool = True
    family_target_overrides: Mapping[str, frozenset[str]] = field(default_factory=dict)
    supported_extensions: frozenset[str] = frozenset()

    def supports_family(self, family: str) -> bool:
        normalized = str(family or "").strip().lower()
        if not normalized:
            return self.allow_unknown_family
        return normalized in self.supported_families

    def supported_targets_for_family(self, family: str) -> frozenset[str]:
        normalized = str(family or "").strip().lower()
        override = self.family_target_overrides.get(normalized)
        if override is not None:
            return frozenset(override)
        return self.supported_targets


class AdapterLoaderRegistry:
    def __init__(self, capabilities: Iterable[AdapterLoaderCapability] | None = None) -> None:
        self._capabilities: dict[str, AdapterLoaderCapability] = {}
        self._implementations: dict[str, Any] = {}
        for capability in capabilities or ():
            self.register(capability)

    def register(self, capability: AdapterLoaderCapability, implementation: Any | None = None) -> None:
        if not capability.loader_id:
            raise ValueError("Adapter loader capability requires a stable loader_id.")
        self._capabilities[capability.loader_id] = capability
        if implementation is not None:
            self.register_implementation(capability.loader_id, implementation)

    def register_implementation(self, loader_id: str, implementation: Any) -> None:
        token = str(loader_id or "").strip()
        if not token:
            raise ValueError("Adapter loader implementation requires a stable loader_id.")
        if token not in self._capabilities:
            raise ValueError(f"Adapter loader implementation '{token}' has no registered capability.")
        implementation_id = str(getattr(implementation, "loader_id", token) or token)
        if implementation_id != token:
            raise ValueError(
                f"Adapter loader implementation ID '{implementation_id}' does not match registered ID '{token}'."
            )
        self._implementations[token] = implementation

    def capability(self, loader_id: str) -> AdapterLoaderCapability | None:
        return self._capabilities.get(str(loader_id or ""))

    def implementation(self, loader_id: str) -> Any | None:
        return self._implementations.get(str(loader_id or ""))

    def implementations(self) -> tuple[Any, ...]:
        return tuple(self._implementations[key] for key in sorted(self._implementations))

    def loader_for_format(self, adapter_format: str, *, family: str = "") -> AdapterLoaderCapability | None:
        format_id = str(adapter_format or "").strip().lower()
        for capability in self._capabilities.values():
            if format_id in capability.supported_formats and capability.supports_family(family):
                return capability
        return None

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            loader_id: {
                "loader_id": capability.loader_id,
                "supported_formats": sorted(capability.supported_formats),
                "supported_families": sorted(capability.supported_families),
                "supported_targets": sorted(capability.supported_targets),
                "family_target_overrides": {
                    family: sorted(targets)
                    for family, targets in sorted(capability.family_target_overrides.items())
                },
                "supported_extensions": sorted(capability.supported_extensions),
                "allow_unknown_family": capability.allow_unknown_family,
                "implementation_registered": loader_id in self._implementations,
            }
            for loader_id, capability in sorted(self._capabilities.items())
        }


def default_adapter_loader_registry() -> AdapterLoaderRegistry:
    capability = AdapterLoaderCapability(
        loader_id=STANDARD_DIFFUSERS_LOADER_ID,
        supported_formats=frozenset({
            "standard_kohya_lora",
            "standard_diffusers_peft_lora",
            "standard_lora_up_down",
        }),
        supported_families=frozenset({"sd1", "sd2", "sdxl"}),
        supported_targets=frozenset({"unet", "text_encoder", "text_encoder_2", "linear", "convolution"}),
        family_target_overrides={
            "sd1": frozenset({"unet", "text_encoder", "linear", "convolution"}),
            "sd2": frozenset({"unet", "text_encoder", "linear", "convolution"}),
            "sdxl": frozenset({"unet", "text_encoder", "text_encoder_2", "linear", "convolution"}),
        },
        supported_extensions=frozenset(),
        allow_unknown_family=True,
    )
    registry = AdapterLoaderRegistry([capability])
    from image_gen.runtime.adapters.standard_loader import StandardDiffusersAdapterLoader

    registry.register_implementation(STANDARD_DIFFUSERS_LOADER_ID, StandardDiffusersAdapterLoader())
    return registry
