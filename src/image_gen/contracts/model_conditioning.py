from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BranchModelConditioningKwargs:
    """Branch-aware kwargs supplied to the denoising model boundary.

    ``conditional`` and ``unconditional`` contain branch-specific keyword
    arguments. ``shared`` is merged into either branch before the UNet call.
    The denoising system owns branch selection for pipeline-guided CFG, while
    raw-model samplers may explicitly select a branch before invoking the model.
    """

    conditional: Mapping[str, Any] = field(default_factory=dict)
    unconditional: Mapping[str, Any] = field(default_factory=dict)
    shared: Mapping[str, Any] = field(default_factory=dict)

    def for_branch(self, branch: str) -> dict[str, Any]:
        normalized = str(branch or "").strip().lower().replace("-", "_")
        aliases = {
            "cond": "conditional",
            "positive": "conditional",
            "uncond": "unconditional",
            "negative": "unconditional",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"conditional", "unconditional"}:
            raise ValueError(
                "Model-conditioning branch must be 'conditional' or 'unconditional'."
            )
        branch_values = self.conditional if normalized == "conditional" else self.unconditional
        merged = dict(self.shared or {})
        merged.update(dict(branch_values or {}))
        return merged


ModelConditioningKwargs = dict[str, Any] | BranchModelConditioningKwargs | None


def select_model_conditioning_branch(
    value: ModelConditioningKwargs,
    branch: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BranchModelConditioningKwargs):
        return value.for_branch(branch)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        "Model conditioning kwargs must be a dict, BranchModelConditioningKwargs, or None."
    )
