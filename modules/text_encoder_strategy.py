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


def text_encoder_strategy_for(architecture: str | None) -> BaseTextEncoderStrategy:
    normalized = str(architecture or "").strip().lower()
    if normalized in {"sd2", "sd2.1", "sd2.x", "stable-diffusion-2.x"}:
        return SD2TextEncoderStrategy()
    return SD1TextEncoderStrategy()
