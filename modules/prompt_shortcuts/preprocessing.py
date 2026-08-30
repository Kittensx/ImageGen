from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.prompt_shortcuts.contracts import PromptShortcutProfileDescriptor


@dataclass(frozen=True)
class PromptPreprocessResult:
    raw_prompt: str
    resolved_prompt: str
    pipeline: str
    stages: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_prompt": self.raw_prompt,
            "resolved_prompt": self.resolved_prompt,
            "pipeline": self.pipeline,
            "stages": [dict(item) for item in self.stages],
        }


def preprocess_prompt_for_profile(
    raw_prompt: str,
    *,
    profile: PromptShortcutProfileDescriptor,
    style_template: str = "",
) -> PromptPreprocessResult:
    """Apply profile-owned preprocessing before alias/semantic translation.

    PPSR-10B implements the A1111 ``{prompt}`` style placeholder ordering.  The
    caller must explicitly provide a style template; ImageGen does not invent a
    style append rule when no template is selected.  Extra-network tags remain
    owned by the existing ImageGen asset/runtime adapter and are intentionally
    not consumed here.
    """

    raw = str(raw_prompt or "")
    resolved = raw
    pipeline = str(profile.preprocessing.get("pipeline") or "none")
    stages: list[dict[str, Any]] = []
    if pipeline == "a1111_compat_preprocess_v1" and str(style_template or ""):
        template = str(style_template)
        if "{prompt}" in template:
            resolved = template.replace("{prompt}", resolved)
            stages.append({
                "stage": "style_template",
                "algorithm": "a1111_prompt_placeholder_v1",
                "template": template,
            })
        else:
            # The local PPSR compatibility contract only locks {prompt}
            # substitution.  Do not silently guess A1111's style-append UI rule.
            stages.append({
                "stage": "style_template",
                "algorithm": "a1111_prompt_placeholder_v1",
                "template": template,
                "applied": False,
                "reason": "missing_{prompt}_placeholder",
            })
    return PromptPreprocessResult(
        raw_prompt=raw,
        resolved_prompt=resolved,
        pipeline=pipeline,
        stages=tuple(stages),
    )
