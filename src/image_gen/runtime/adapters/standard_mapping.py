from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

STANDARD_MAPPING_CONTRACT_VERSION = "image-gen-standard-lora-mapping-v1"

_COMPONENT_TARGETS = ("unet", "text_encoder", "text_encoder_2")


@dataclass(frozen=True)
class StandardAdapterKeyMapping:
    source_key: str
    normalized_key: str
    adapter_format: str
    model_family: str
    component_target: str
    module_path: str
    parameter_role: str
    recognized: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": STANDARD_MAPPING_CONTRACT_VERSION,
            "source_key": self.source_key,
            "normalized_key": self.normalized_key,
            "recognized_format": self.adapter_format,
            "model_family": self.model_family,
            "component_target": self.component_target,
            "module_path": self.module_path,
            "parameter_role": self.parameter_role,
            "recognized": bool(self.recognized),
            "reason": self.reason,
        }


def _canonical_family(value: Any) -> str:
    token = str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if token.startswith(("sdxl", "stablediffusionxl")):
        return "sdxl"
    if token.startswith(("sd2", "stablediffusion2")):
        return "sd2"
    if token.startswith(("sd1", "stablediffusion1")):
        return "sd1"
    return ""


def _parameter_role(key: str) -> str:
    lowered = key.lower()
    if lowered.endswith((".lora_a.weight", ".lora_down.weight")) or "lora_down.weight" in lowered:
        return "down"
    if lowered.endswith((".lora_b.weight", ".lora_up.weight")) or "lora_up.weight" in lowered:
        return "up"
    if lowered.endswith(".alpha") or lowered.endswith("_alpha"):
        return "alpha"
    if "lora_magnitude_vector" in lowered or "magnitude_vector" in lowered:
        return "magnitude"
    return "unknown"


def _is_adapter_parameter_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        marker in lowered
        for marker in (
            ".lora_a.",
            ".lora_b.",
            ".lora_down.",
            ".lora_up.",
            "lora_down.weight",
            "lora_up.weight",
            "lora_a.weight",
            "lora_b.weight",
            "lora_magnitude_vector",
            "magnitude_vector",
        )
    ) or lowered.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"))


def _direct_component_from_module(module_path: str, expected_targets: set[str]) -> tuple[str, str]:
    lowered = module_path.lower()
    if lowered.startswith(("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm.")):
        if expected_targets == {"text_encoder_2"}:
            return "text_encoder_2", "component_native_text_encoder_2"
        return "text_encoder", "component_native_text_encoder"
    if lowered.startswith(
        (
            "down_blocks.",
            "up_blocks.",
            "mid_block.",
            "conv_in.",
            "conv_out.",
            "time_embedding.",
            "time_embed.",
            "add_embedding.",
            "transformer_in.",
            "proj_in.",
            "proj_out.",
        )
    ):
        return "unet", "component_native_unet"
    component_expectations = expected_targets.intersection(_COMPONENT_TARGETS)
    if len(component_expectations) == 1:
        target = next(iter(component_expectations))
        return target, f"single_expected_component:{target}"
    return "", "ambiguous_component_native_key"


def map_standard_adapter_key(
    key: Any,
    *,
    adapter_format: str,
    model_family: str = "",
    expected_targets: Iterable[str] = (),
) -> StandardAdapterKeyMapping:
    source = str(key or "")
    family = _canonical_family(model_family)
    expected = {str(item) for item in expected_targets if str(item)}
    lowered = source.lower()
    role = _parameter_role(source)

    for prefix, target in (
        ("text_encoder_2.", "text_encoder_2"),
        ("text_encoder.", "text_encoder"),
        ("unet.", "unet"),
    ):
        if lowered.startswith(prefix):
            module_path = source[len(prefix):]
            return StandardAdapterKeyMapping(
                source_key=source,
                normalized_key=source,
                adapter_format=adapter_format,
                model_family=family,
                component_target=target,
                module_path=module_path,
                parameter_role=role,
                recognized=True,
                reason="pipeline_component_prefix",
            )

    for prefix, target in (
        ("lora_te2_", "text_encoder_2"),
        ("lora_te1_", "text_encoder"),
        ("lora_te_", "text_encoder"),
        ("lora_unet_", "unet"),
    ):
        if lowered.startswith(prefix):
            return StandardAdapterKeyMapping(
                source_key=source,
                normalized_key=source,
                adapter_format=adapter_format,
                model_family=family,
                component_target=target,
                module_path=source[len(prefix):],
                parameter_role=role,
                recognized=True,
                reason="kohya_component_prefix",
            )

    if not _is_adapter_parameter_key(source) and role != "alpha":
        return StandardAdapterKeyMapping(
            source_key=source,
            normalized_key=source,
            adapter_format=adapter_format,
            model_family=family,
            component_target="",
            module_path=source,
            parameter_role=role,
            recognized=False,
            reason="not_a_recognized_standard_lora_parameter_key",
        )

    module_path = source
    for marker in (
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_magnitude_vector",
        ".alpha",
    ):
        position = module_path.lower().find(marker.lower())
        if position >= 0:
            module_path = module_path[:position]
            break
    component, reason = _direct_component_from_module(module_path, expected)
    if not component:
        return StandardAdapterKeyMapping(
            source_key=source,
            normalized_key=source,
            adapter_format=adapter_format,
            model_family=family,
            component_target="",
            module_path=module_path,
            parameter_role=role,
            recognized=False,
            reason=reason,
        )
    return StandardAdapterKeyMapping(
        source_key=source,
        normalized_key=f"{component}.{source}",
        adapter_format=adapter_format,
        model_family=family,
        component_target=component,
        module_path=module_path,
        parameter_role=role,
        recognized=True,
        reason=reason,
    )


def normalize_standard_state_dict(
    state_dict: Mapping[str, Any],
    network_alphas: Mapping[str, Any] | None = None,
    *,
    adapter_format: str,
    model_family: str = "",
    expected_targets: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalized_state: dict[str, Any] = {}
    mappings: list[StandardAdapterKeyMapping] = []
    unmapped_parameter_keys: list[str] = []
    changed = False
    for key, value in state_dict.items():
        mapping = map_standard_adapter_key(
            key,
            adapter_format=adapter_format,
            model_family=model_family,
            expected_targets=expected_targets,
        )
        mappings.append(mapping)
        normalized_state[mapping.normalized_key] = value
        changed = changed or mapping.normalized_key != str(key)
        if not mapping.recognized and _is_adapter_parameter_key(str(key)):
            unmapped_parameter_keys.append(str(key))

    normalized_alphas: dict[str, Any] = {}
    alpha_unmapped: list[str] = []
    for key, value in dict(network_alphas or {}).items():
        mapping = map_standard_adapter_key(
            key,
            adapter_format=adapter_format,
            model_family=model_family,
            expected_targets=expected_targets,
        )
        normalized_alphas[mapping.normalized_key] = value
        changed = changed or mapping.normalized_key != str(key)
        if not mapping.recognized:
            alpha_unmapped.append(str(key))

    component_counts = {target: 0 for target in _COMPONENT_TARGETS}
    role_counts: dict[str, int] = {}
    for mapping in mappings:
        if mapping.component_target in component_counts:
            component_counts[mapping.component_target] += 1
        role_counts[mapping.parameter_role] = role_counts.get(mapping.parameter_role, 0) + 1

    if changed:
        prefix_mode = "deterministic_component_normalization"
    elif mappings and all(mapping.recognized for mapping in mappings):
        prefix_mode = "pipeline_prefixed_or_kohya_converted"
    else:
        prefix_mode = "unrecognized"

    report = {
        "contract_version": STANDARD_MAPPING_CONTRACT_VERSION,
        "prefix_mode": prefix_mode,
        "mapping_count": len(mappings),
        "recognized_mapping_count": sum(1 for item in mappings if item.recognized),
        "unmapped_parameter_count": len(unmapped_parameter_keys),
        "unmapped_parameter_examples": unmapped_parameter_keys[:8],
        "unmapped_alpha_count": len(alpha_unmapped),
        "unmapped_alpha_examples": alpha_unmapped[:8],
        "component_key_counts": component_counts,
        "parameter_role_counts": role_counts,
        "mapping_examples": [item.to_dict() for item in mappings[:8]],
    }
    return normalized_state, normalized_alphas, report


def supported_component_targets_for_family(model_family: str) -> frozenset[str]:
    family = _canonical_family(model_family)
    if family == "sdxl":
        return frozenset({"unet", "text_encoder", "text_encoder_2"})
    if family in {"sd1", "sd2"}:
        return frozenset({"unet", "text_encoder"})
    return frozenset({"unet", "text_encoder"})
