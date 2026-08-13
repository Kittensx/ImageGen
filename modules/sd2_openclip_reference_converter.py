from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class TensorComparison:
    key: str
    source_keys: list[str]
    matched: bool
    reason: str = ""
    expected_shape: list[int] | None = None
    actual_shape: list[int] | None = None
    max_abs_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source_keys": list(self.source_keys),
            "matched": self.matched,
            "reason": self.reason,
            "expected_shape": list(self.expected_shape) if self.expected_shape is not None else None,
            "actual_shape": list(self.actual_shape) if self.actual_shape is not None else None,
            "max_abs_diff": self.max_abs_diff,
        }


@dataclass
class LayerComparison:
    label: str
    comparisons: list[TensorComparison] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return all(item.matched for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "matched": self.matched,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


class SD2OpenCLIPReferenceConverter:
    """Convert SD2/OpenCLIP text-encoder weights into HF CLIPTextModel layout."""

    SOURCE_BLOCK_COUNT = 24
    RUNTIME_LAYER_COUNT = 23

    def normalize_source_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = str(key)
            if new_key.startswith("cond_stage_model.model."):
                new_key = new_key[len("cond_stage_model.model."):]
            elif new_key.startswith("model."):
                new_key = new_key[len("model."):]
            normalized[new_key] = value
        return normalized

    def convert(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        source = self.normalize_source_state_dict(state_dict)
        converted: Dict[str, Any] = {}

        self._copy_required_global(source, converted)
        for index in range(self.RUNTIME_LAYER_COUNT):
            self._convert_block(source, converted, index)
        return converted

    def compare_against_reference(
        self,
        source_state_dict: Dict[str, Any],
        reference_state_dict: Dict[str, Any],
    ) -> dict[str, Any]:
        converted = self.convert(source_state_dict)
        reference = dict(reference_state_dict)
        reference.pop("text_model.embeddings.position_ids", None)

        global_layers = [
            self._compare_tensor(
                key="text_model.embeddings.token_embedding.weight",
                converted=converted,
                reference=reference,
                source_keys=["token_embedding.weight"],
            ),
            self._compare_tensor(
                key="text_model.embeddings.position_embedding.weight",
                converted=converted,
                reference=reference,
                source_keys=["positional_embedding"],
            ),
            self._compare_tensor(
                key="text_model.final_layer_norm.weight",
                converted=converted,
                reference=reference,
                source_keys=["ln_final.weight"],
            ),
            self._compare_tensor(
                key="text_model.final_layer_norm.bias",
                converted=converted,
                reference=reference,
                source_keys=["ln_final.bias"],
            ),
        ]

        blocks: list[LayerComparison] = []
        for index in range(self.RUNTIME_LAYER_COUNT):
            blocks.append(self._compare_block(converted, reference, index))

        converted_keys = set(converted.keys())
        reference_keys = set(reference.keys())
        missing_in_converted = sorted(reference_keys - converted_keys)
        unexpected_in_converted = sorted(converted_keys - reference_keys)
        normalized_source = self.normalize_source_state_dict(source_state_dict)
        ignored_source_keys = sorted(
            key for key in normalized_source
            if key.startswith("transformer.resblocks.23.")
        )

        return {
            "converted_key_count": len(converted),
            "reference_key_count": len(reference),
            "global": LayerComparison(label="global", comparisons=global_layers).to_dict(),
            "blocks": [block.to_dict() for block in blocks],
            "missing_in_converted": missing_in_converted,
            "unexpected_in_converted": unexpected_in_converted,
            "ignored_source_keys": ignored_source_keys,
            "ignored_source_reason": "SD2 OpenCLIP uses penultimate conditioning; source block 23 is intentionally omitted from the 23-layer Hugging Face CLIPTextModel runtime representation.",
            "matched": (
                not missing_in_converted
                and not unexpected_in_converted
                and all(item.matched for item in global_layers)
                and all(block.matched for block in blocks)
            ),
        }

    def _copy_required_global(self, source: Dict[str, Any], converted: Dict[str, Any]) -> None:
        mapping = {
            "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
            "positional_embedding": "text_model.embeddings.position_embedding.weight",
            "ln_final.weight": "text_model.final_layer_norm.weight",
            "ln_final.bias": "text_model.final_layer_norm.bias",
        }
        for source_key, target_key in mapping.items():
            if source_key not in source:
                raise KeyError(f"Missing required SD2/OpenCLIP source key: {source_key}")
            converted[target_key] = source[source_key]

    def _convert_block(self, source: Dict[str, Any], converted: Dict[str, Any], index: int) -> None:
        src = f"transformer.resblocks.{index}."
        dst = f"text_model.encoder.layers.{index}."
        simple_map = {
            src + "ln_1.weight": dst + "layer_norm1.weight",
            src + "ln_1.bias": dst + "layer_norm1.bias",
            src + "attn.out_proj.weight": dst + "self_attn.out_proj.weight",
            src + "attn.out_proj.bias": dst + "self_attn.out_proj.bias",
            src + "ln_2.weight": dst + "layer_norm2.weight",
            src + "ln_2.bias": dst + "layer_norm2.bias",
            src + "mlp.c_fc.weight": dst + "mlp.fc1.weight",
            src + "mlp.c_fc.bias": dst + "mlp.fc1.bias",
            src + "mlp.c_proj.weight": dst + "mlp.fc2.weight",
            src + "mlp.c_proj.bias": dst + "mlp.fc2.bias",
        }
        for source_key, target_key in simple_map.items():
            if source_key not in source:
                raise KeyError(f"Missing required SD2/OpenCLIP source key: {source_key}")
            converted[target_key] = source[source_key]

        in_proj_weight_key = src + "attn.in_proj_weight"
        in_proj_bias_key = src + "attn.in_proj_bias"
        if in_proj_weight_key not in source:
            raise KeyError(f"Missing required SD2/OpenCLIP source key: {in_proj_weight_key}")
        if in_proj_bias_key not in source:
            raise KeyError(f"Missing required SD2/OpenCLIP source key: {in_proj_bias_key}")

        q_weight, k_weight, v_weight = torch.chunk(source[in_proj_weight_key], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(source[in_proj_bias_key], 3, dim=0)
        converted[dst + "self_attn.q_proj.weight"] = q_weight
        converted[dst + "self_attn.q_proj.bias"] = q_bias
        converted[dst + "self_attn.k_proj.weight"] = k_weight
        converted[dst + "self_attn.k_proj.bias"] = k_bias
        converted[dst + "self_attn.v_proj.weight"] = v_weight
        converted[dst + "self_attn.v_proj.bias"] = v_bias

    def _compare_block(
        self,
        converted: Dict[str, Any],
        reference: Dict[str, Any],
        index: int,
    ) -> LayerComparison:
        src = f"transformer.resblocks.{index}."
        dst = f"text_model.encoder.layers.{index}."
        comparisons = [
            self._compare_tensor(dst + "layer_norm1.weight", converted, reference, [src + "ln_1.weight"]),
            self._compare_tensor(dst + "layer_norm1.bias", converted, reference, [src + "ln_1.bias"]),
            self._compare_tensor(dst + "self_attn.q_proj.weight", converted, reference, [src + "attn.in_proj_weight"]),
            self._compare_tensor(dst + "self_attn.q_proj.bias", converted, reference, [src + "attn.in_proj_bias"]),
            self._compare_tensor(dst + "self_attn.k_proj.weight", converted, reference, [src + "attn.in_proj_weight"]),
            self._compare_tensor(dst + "self_attn.k_proj.bias", converted, reference, [src + "attn.in_proj_bias"]),
            self._compare_tensor(dst + "self_attn.v_proj.weight", converted, reference, [src + "attn.in_proj_weight"]),
            self._compare_tensor(dst + "self_attn.v_proj.bias", converted, reference, [src + "attn.in_proj_bias"]),
            self._compare_tensor(dst + "self_attn.out_proj.weight", converted, reference, [src + "attn.out_proj.weight"]),
            self._compare_tensor(dst + "self_attn.out_proj.bias", converted, reference, [src + "attn.out_proj.bias"]),
            self._compare_tensor(dst + "layer_norm2.weight", converted, reference, [src + "ln_2.weight"]),
            self._compare_tensor(dst + "layer_norm2.bias", converted, reference, [src + "ln_2.bias"]),
            self._compare_tensor(dst + "mlp.fc1.weight", converted, reference, [src + "mlp.c_fc.weight"]),
            self._compare_tensor(dst + "mlp.fc1.bias", converted, reference, [src + "mlp.c_fc.bias"]),
            self._compare_tensor(dst + "mlp.fc2.weight", converted, reference, [src + "mlp.c_proj.weight"]),
            self._compare_tensor(dst + "mlp.fc2.bias", converted, reference, [src + "mlp.c_proj.bias"]),
        ]
        return LayerComparison(label=f"transformer_block_{index}", comparisons=comparisons)

    def _compare_tensor(
        self,
        key: str,
        converted: Dict[str, Any],
        reference: Dict[str, Any],
        source_keys: list[str],
    ) -> TensorComparison:
        if key not in converted:
            expected_shape = list(reference[key].shape) if key in reference and hasattr(reference[key], "shape") else None
            return TensorComparison(
                key=key,
                source_keys=source_keys,
                matched=False,
                reason="missing_in_converted",
                expected_shape=expected_shape,
            )
        if key not in reference:
            actual_shape = list(converted[key].shape) if hasattr(converted[key], "shape") else None
            return TensorComparison(
                key=key,
                source_keys=source_keys,
                matched=False,
                reason="missing_in_reference",
                actual_shape=actual_shape,
            )

        actual = converted[key]
        expected = reference[key]
        actual_shape = list(actual.shape) if hasattr(actual, "shape") else None
        expected_shape = list(expected.shape) if hasattr(expected, "shape") else None
        if actual_shape != expected_shape:
            return TensorComparison(
                key=key,
                source_keys=source_keys,
                matched=False,
                reason="shape_mismatch",
                expected_shape=expected_shape,
                actual_shape=actual_shape,
            )

        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        max_abs_diff = float(torch.max(torch.abs(actual_f - expected_f)).item()) if actual_f.numel() else 0.0
        matched = bool(torch.equal(actual_f, expected_f))
        reason = "" if matched else "value_mismatch"
        return TensorComparison(
            key=key,
            source_keys=source_keys,
            matched=matched,
            reason=reason,
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            max_abs_diff=max_abs_diff,
        )
