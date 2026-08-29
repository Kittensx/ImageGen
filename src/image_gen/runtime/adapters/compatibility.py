from __future__ import annotations

from typing import Any

from image_gen.runtime.adapters.contracts import AdapterCompatibilityResult, AdapterInspectionRecord
from image_gen.runtime.adapters.registry import AdapterLoaderRegistry, default_adapter_loader_registry
from image_gen.runtime.lora_inspector import canonical_model_family


class AdapterCompatibilityService:
    def __init__(self, registry: AdapterLoaderRegistry | None = None) -> None:
        self.registry = registry or default_adapter_loader_registry()

    def evaluate(
        self,
        inspection: AdapterInspectionRecord | dict[str, Any],
        *,
        active_checkpoint_family: str = "",
    ) -> AdapterCompatibilityResult:
        record = inspection if isinstance(inspection, AdapterInspectionRecord) else AdapterInspectionRecord.from_mapping(inspection)
        adapter_family = canonical_model_family(record.model_family)
        checkpoint_family = canonical_model_family(active_checkpoint_family)
        warnings = list(record.inspection_warnings)

        if record.adapter_format == "inspection_restricted":
            return AdapterCompatibilityResult(
                family_status="unknown" if not adapter_family or not checkpoint_family else ("compatible" if adapter_family == checkpoint_family else "incompatible"),
                format_status="restricted",
                target_status="unknown",
                overall_support_state="restricted",
                runtime_loadable=False,
                blocking_reason="Adapter inspection is restricted because this legacy format may contain executable pickle payloads.",
                warnings=tuple(warnings),
            )

        if record.inspection_errors or record.adapter_format == "invalid":
            return AdapterCompatibilityResult(
                family_status="unknown" if not adapter_family or not checkpoint_family else ("compatible" if adapter_family == checkpoint_family else "incompatible"),
                format_status="invalid",
                target_status="unknown",
                overall_support_state="invalid",
                runtime_loadable=False,
                blocking_reason="Adapter file could not be inspected as a valid Safetensors adapter.",
                warnings=tuple(warnings),
            )

        if record.adapter_format == "non_adapter_full_model":
            return AdapterCompatibilityResult(
                family_status="unknown" if not adapter_family or not checkpoint_family else ("compatible" if adapter_family == checkpoint_family else "incompatible"),
                format_status="misclassified",
                target_status="unsupported",
                overall_support_state="misclassified",
                runtime_loadable=False,
                blocking_reason="LoRA asset appears to be a full checkpoint, not an adapter.",
                warnings=tuple(warnings),
            )

        if adapter_family and checkpoint_family and adapter_family != checkpoint_family:
            return AdapterCompatibilityResult(
                family_status="incompatible",
                format_status="supported" if self.registry.loader_for_format(record.adapter_format, family=adapter_family) else "unsupported",
                target_status="unknown",
                overall_support_state="unsupported",
                runtime_loadable=False,
                blocking_reason=(
                    f"Adapter targets {self._family_label(adapter_family)} but the active checkpoint is "
                    f"{self._family_label(checkpoint_family)}."
                ),
                warnings=tuple(warnings),
            )

        family_status = "compatible" if adapter_family and checkpoint_family else "unknown"
        if not adapter_family:
            warnings.append("Adapter model family could not be proven from technical metadata or tensor shapes.")
        elif not checkpoint_family:
            warnings.append("Active checkpoint family was not available during adapter preflight.")

        loader = self.registry.loader_for_format(record.adapter_format, family=adapter_family)
        if loader is None:
            return AdapterCompatibilityResult(
                family_status=family_status,
                format_status="unsupported",
                target_status="unknown",
                overall_support_state="unsupported",
                runtime_loadable=False,
                blocking_reason=self._unsupported_format_reason(record),
                warnings=tuple(warnings),
            )

        target_scopes = set(record.target_scopes)
        supported_targets = set(loader.supported_targets_for_family(adapter_family))
        unsupported_targets = sorted(target_scopes - supported_targets)
        if unsupported_targets:
            return AdapterCompatibilityResult(
                family_status=family_status,
                format_status="supported",
                target_status="partial",
                overall_support_state="partial",
                runtime_loadable=False,
                blocking_reason=(
                    "Adapter contains targets that the selected loader cannot safely apply: "
                    + ", ".join(unsupported_targets)
                    + "."
                ),
                warnings=tuple(warnings),
                loader_id=loader.loader_id,
            )

        unsupported_extensions = sorted(set(record.adapter_extensions) - set(loader.supported_extensions))
        if unsupported_extensions:
            return AdapterCompatibilityResult(
                family_status=family_status,
                format_status="supported",
                target_status="partial",
                overall_support_state="partial",
                runtime_loadable=False,
                blocking_reason=(
                    "Adapter uses a standard-LoRA extension that is not yet runtime-qualified: "
                    + ", ".join(unsupported_extensions)
                    + "."
                ),
                warnings=tuple(warnings),
                loader_id=loader.loader_id,
            )

        if not target_scopes:
            warnings.append("Adapter target scope could not be proven from tensor keys.")
            target_status = "unknown"
        else:
            target_status = "supported"

        overall = "supported" if family_status == "compatible" and target_status == "supported" and not warnings else "supported_with_warning"
        return AdapterCompatibilityResult(
            family_status=family_status,
            format_status="supported",
            target_status=target_status,
            overall_support_state=overall,
            runtime_loadable=True,
            blocking_reason="",
            warnings=tuple(warnings),
            loader_id=loader.loader_id,
        )

    @staticmethod
    def _unsupported_format_reason(record: AdapterInspectionRecord) -> str:
        mapping = {
            "lycoris_loha": "Adapter is a valid LyCORIS/LoHa adapter, but this build does not yet provide the LoHa loader.",
            "lycoris_lokr": "Adapter is a valid LyCORIS/LoKr adapter, but this build does not yet provide the LoKr loader.",
            "lycoris_locon": "Adapter uses a LyCORIS/LoCon representation that is not yet qualified by this build.",
            "lycoris_other": "Adapter uses a LyCORIS algorithm that is not yet supported by this build.",
            "unknown_adapter": "Adapter tensor format is not recognized by this build.",
            "inspection_restricted": "Adapter inspection is restricted because this legacy format may contain executable pickle payloads.",
        }
        return mapping.get(record.adapter_format, f"Adapter format '{record.adapter_format}' is not supported by this build.")

    @staticmethod
    def _family_label(value: str) -> str:
        return {"sd1": "SD1.x", "sd2": "SD2.x", "sdxl": "SDXL"}.get(value, value or "unknown")
