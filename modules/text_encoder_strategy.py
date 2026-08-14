from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict



@dataclass
class LayerCoverage:
    label: str
    expected_keys: int
    present_keys: int
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)

    @property
    def coverage_ratio(self) -> float:
        if self.expected_keys <= 0:
            return 0.0
        return self.present_keys / self.expected_keys

    @property
    def complete(self) -> bool:
        return self.present_keys >= self.expected_keys and not self.missing_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "expected_keys": self.expected_keys,
            "present_keys": self.present_keys,
            "coverage_ratio": self.coverage_ratio,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "complete": self.complete,
        }


@dataclass
class TextEncoderAuditReport:
    architecture: str
    handler: str
    source_key_count: int
    normalized_key_count: int
    recognized_key_count: int
    required_global_keys: LayerCoverage
    transformer_blocks: list[LayerCoverage] = field(default_factory=list)
    optional_keys_present: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    normalized_key_samples: list[str] = field(default_factory=list)

    @property
    def coverage_ratio(self) -> float:
        expected = self.required_global_keys.expected_keys + sum(
            block.expected_keys for block in self.transformer_blocks
        )
        present = self.required_global_keys.present_keys + sum(
            block.present_keys for block in self.transformer_blocks
        )
        if expected <= 0:
            return 0.0
        return present / expected

    @property
    def missing_block_indices(self) -> list[int]:
        missing: list[int] = []
        for block in self.transformer_blocks:
            match = re.search(r"(\d+)$", block.label)
            if match and block.present_keys == 0:
                missing.append(int(match.group(1)))
        return missing

    @property
    def passed(self) -> bool:
        return (
            self.required_global_keys.complete
            and all(block.complete for block in self.transformer_blocks)
            and not self.unexpected_keys
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "handler": self.handler,
            "source_key_count": self.source_key_count,
            "normalized_key_count": self.normalized_key_count,
            "recognized_key_count": self.recognized_key_count,
            "coverage_ratio": self.coverage_ratio,
            "passed": self.passed,
            "required_global_keys": self.required_global_keys.to_dict(),
            "transformer_blocks": [block.to_dict() for block in self.transformer_blocks],
            "missing_block_indices": self.missing_block_indices,
            "optional_keys_present": list(self.optional_keys_present),
            "unexpected_key_count": len(self.unexpected_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "normalized_key_samples": list(self.normalized_key_samples),
        }


class BaseTextEncoderStrategy:
    architecture = "unknown"
    handler_name = "base"

    def convert_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return dict(state_dict)

    def audit_state_dict(self, state_dict: Dict[str, Any]) -> TextEncoderAuditReport:
        normalized = self.convert_state_dict(state_dict)
        coverage = LayerCoverage(
            label="text_encoder",
            expected_keys=len(normalized),
            present_keys=len(normalized),
        )
        return TextEncoderAuditReport(
            architecture=self.architecture,
            handler=self.handler_name,
            source_key_count=len(state_dict),
            normalized_key_count=len(normalized),
            recognized_key_count=len(normalized),
            required_global_keys=coverage,
            transformer_blocks=[],
            optional_keys_present=[],
            unexpected_keys=[],
            normalized_key_samples=sorted(normalized.keys())[:50],
        )


class SD1TextEncoderStrategy(BaseTextEncoderStrategy):
    architecture = "sd1.x"
    handler_name = "sd1_clip_hf"

    def convert_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            str(key).replace("transformer.", ""): value
            for key, value in state_dict.items()
        }


class SDXLTextEncoder1Strategy(SD1TextEncoderStrategy):
    architecture = "sdxl"
    handler_name = "sdxl_clip_l_hf"

    def convert_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = str(key)
            if new_key.startswith("transformer."):
                new_key = new_key[len("transformer.") :]
            # Transformers registers CLIP position_ids as a non-persistent
            # buffer, so a monolithic SDXL checkpoint copy must not be sent
            # to load_state_dict as a model parameter/buffer key.
            if new_key == "text_model.embeddings.position_ids":
                continue
            converted[new_key] = value
        return converted


class SDXLTextEncoder2Strategy(BaseTextEncoderStrategy):
    architecture = "sdxl"
    handler_name = "sdxl_openclip_big_g_hf"

    GLOBAL_KEY_MAP = {
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "text_projection": "text_projection.weight",
    }

    TRANSFORM_REPLACEMENTS = (
        ("resblocks.", "text_model.encoder.layers."),
        ("ln_1", "layer_norm1"),
        ("ln_2", "layer_norm2"),
        (".c_fc.", ".fc1."),
        (".c_proj.", ".fc2."),
        (".attn", ".self_attn"),
    )

    def _strip_model_prefix(self, key: str) -> str:
        key = str(key)
        if key.startswith("model."):
            return key[len("model.") :]
        return key

    def _transform_key(self, key: str) -> str:
        new_key = str(key)
        for old, new in self.TRANSFORM_REPLACEMENTS:
            new_key = new_key.replace(old, new)
        return new_key

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", ())
        return tuple(int(dim) for dim in shape)

    def _infer_hidden_size(self, normalized: Dict[str, Any]) -> int:
        projection = normalized.get("text_projection")
        if projection is not None:
            shape = self._shape(projection)
            if len(shape) == 2 and shape[0] > 0:
                return int(shape[0])

        token_embedding = normalized.get("token_embedding.weight")
        if token_embedding is not None:
            shape = self._shape(token_embedding)
            if len(shape) == 2 and shape[1] > 0:
                return int(shape[1])

        for key, value in normalized.items():
            if key.endswith(".attn.in_proj_weight"):
                shape = self._shape(value)
                if len(shape) == 2 and shape[0] % 3 == 0:
                    return int(shape[0] // 3)

        raise ValueError(
            "Unable to infer SDXL Text Encoder 2 hidden size from text_projection, "
            "token_embedding.weight, or an attention in_proj_weight tensor."
        )

    @staticmethod
    def _transpose_projection(value: Any) -> Any:
        transposed = value.T
        contiguous = getattr(transposed, "contiguous", None)
        return contiguous() if callable(contiguous) else transposed

    @staticmethod
    def _validate_qkv_tensor(value: Any, *, hidden_size: int, key: str, bias: bool) -> None:
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        expected = (hidden_size * 3,) if bias else (hidden_size * 3, hidden_size)
        if shape != expected:
            raise ValueError(
                f"Unexpected SDXL Text Encoder 2 {key} shape {shape}; expected {expected}."
            )

    def convert_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            self._strip_model_prefix(key): value for key, value in state_dict.items()
        }
        if not normalized:
            return {}

        hidden_size = self._infer_hidden_size(normalized)
        converted: Dict[str, Any] = {}

        for key, value in normalized.items():
            if key == "logit_scale":
                # OpenCLIP carries this contrastive-training parameter, but
                # CLIPTextModelWithProjection does not own it.
                continue

            target = self.GLOBAL_KEY_MAP.get(key)
            if target is not None:
                converted[target] = (
                    self._transpose_projection(value) if key == "text_projection" else value
                )
                continue

            if not key.startswith("transformer."):
                # Preserve genuinely unknown keys so strict=False loading reports
                # them as unexpected rather than silently discarding future layout changes.
                converted[key] = value
                continue

            transformer_key = key[len("transformer.") :]
            if transformer_key.endswith(".in_proj_weight"):
                self._validate_qkv_tensor(
                    value, hidden_size=hidden_size, key=key, bias=False
                )
                base = transformer_key[: -len(".in_proj_weight")]
                base = self._transform_key(base)
                converted[base + ".q_proj.weight"] = value[:hidden_size, :]
                converted[base + ".k_proj.weight"] = value[hidden_size : hidden_size * 2, :]
                converted[base + ".v_proj.weight"] = value[hidden_size * 2 :, :]
                continue

            if transformer_key.endswith(".in_proj_bias"):
                self._validate_qkv_tensor(
                    value, hidden_size=hidden_size, key=key, bias=True
                )
                base = transformer_key[: -len(".in_proj_bias")]
                base = self._transform_key(base)
                converted[base + ".q_proj.bias"] = value[:hidden_size]
                converted[base + ".k_proj.bias"] = value[hidden_size : hidden_size * 2]
                converted[base + ".v_proj.bias"] = value[hidden_size * 2 :]
                continue

            converted[self._transform_key(transformer_key)] = value

        return converted


class SD2TextEncoderStrategy(BaseTextEncoderStrategy):
    architecture = "sd2.x"
    handler_name = "sd2_openclip_audit"

    REQUIRED_GLOBAL_KEYS = (
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "ln_final.bias",
    )
    OPTIONAL_GLOBAL_KEYS = (
        "text_projection",
        "logit_scale",
        "attn_mask",
    )
    BLOCK_REQUIRED_SUFFIXES = (
        "ln_1.weight",
        "ln_1.bias",
        "attn.in_proj_weight",
        "attn.in_proj_bias",
        "attn.out_proj.weight",
        "attn.out_proj.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    _BLOCK_INDEX_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.")

    def convert_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_state: Dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = str(key)
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            new_state[new_key] = value
        return new_state

    def audit_state_dict(
        self,
        state_dict: Dict[str, Any],
        *,
        expected_layer_count: int = 24,
    ) -> TextEncoderAuditReport:
        normalized = self.convert_state_dict(state_dict)
        normalized_keys = set(normalized.keys())
        recognized_keys: set[str] = set()

        required_globals_present = [
            key for key in self.REQUIRED_GLOBAL_KEYS if key in normalized_keys
        ]
        required_globals_missing = [
            key for key in self.REQUIRED_GLOBAL_KEYS if key not in normalized_keys
        ]
        recognized_keys.update(required_globals_present)

        optional_keys_present = [
            key for key in self.OPTIONAL_GLOBAL_KEYS if key in normalized_keys
        ]
        recognized_keys.update(optional_keys_present)

        block_reports: list[LayerCoverage] = []
        for index in range(expected_layer_count):
            prefix = f"transformer.resblocks.{index}."
            present_suffixes = []
            missing_suffixes = []
            unexpected_keys = []
            block_keys = sorted(key for key in normalized_keys if key.startswith(prefix))
            for suffix in self.BLOCK_REQUIRED_SUFFIXES:
                full_key = prefix + suffix
                if full_key in normalized_keys:
                    present_suffixes.append(full_key)
                    recognized_keys.add(full_key)
                else:
                    missing_suffixes.append(full_key)
            for block_key in block_keys:
                if block_key not in present_suffixes:
                    unexpected_keys.append(block_key)
            block_reports.append(
                LayerCoverage(
                    label=f"transformer_block_{index}",
                    expected_keys=len(self.BLOCK_REQUIRED_SUFFIXES),
                    present_keys=len(present_suffixes),
                    missing_keys=missing_suffixes,
                    unexpected_keys=unexpected_keys,
                )
            )

        unexpected_global_keys = []
        for key in sorted(normalized_keys - recognized_keys):
            if self._BLOCK_INDEX_RE.match(key):
                continue
            unexpected_global_keys.append(key)

        required_global_report = LayerCoverage(
            label="required_global_keys",
            expected_keys=len(self.REQUIRED_GLOBAL_KEYS),
            present_keys=len(required_globals_present),
            missing_keys=required_globals_missing,
        )

        seen_block_indices = sorted({
            int(match.group(1))
            for key in normalized_keys
            for match in [self._BLOCK_INDEX_RE.match(key)]
            if match is not None
        })
        for extra_index in seen_block_indices:
            if extra_index < expected_layer_count:
                continue
            prefix = f"transformer.resblocks.{extra_index}."
            extra_keys = sorted(key for key in normalized_keys if key.startswith(prefix))
            unexpected_global_keys.extend(extra_keys)

        return TextEncoderAuditReport(
            architecture=self.architecture,
            handler=self.handler_name,
            source_key_count=len(state_dict),
            normalized_key_count=len(normalized),
            recognized_key_count=len(recognized_keys),
            required_global_keys=required_global_report,
            transformer_blocks=block_reports,
            optional_keys_present=optional_keys_present,
            unexpected_keys=unexpected_global_keys,
            normalized_key_samples=sorted(normalized.keys())[:80],
        )


def text_encoder_strategy_for(
    architecture: str | None,
    *,
    encoder_index: int = 1,
) -> BaseTextEncoderStrategy:
    normalized = str(architecture or "").strip().lower()
    if normalized in {"sdxl", "stable-diffusion-xl", "stable-diffusion-xl-base"}:
        if int(encoder_index) == 1:
            return SDXLTextEncoder1Strategy()
        if int(encoder_index) == 2:
            return SDXLTextEncoder2Strategy()
        raise ValueError(f"SDXL text encoder index must be 1 or 2, got {encoder_index!r}.")
    if normalized in {"sd2", "sd2.1", "sd2.x", "stable-diffusion-2.x"}:
        return SD2TextEncoderStrategy()
    if int(encoder_index) != 1:
        raise ValueError(
            f"Architecture {architecture!r} does not define text encoder {encoder_index}."
        )
    return SD1TextEncoderStrategy()
