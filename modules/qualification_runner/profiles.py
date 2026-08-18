from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.sdxl_runtime_profile import profile_for_sdxl_filename

from .artifact_io import _load_yaml_mapping, _slug, _write_yaml
from .contracts import BlueprintSnapshot, QUALIFICATION_RUNNER_SCHEMA_VERSION


DEFAULT_TEST_PROMPT = (
    "cinematic portrait photograph, detailed face, natural skin texture, detailed eyes, "
    "soft directional light, realistic fabric, coherent background, high detail"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, duplicate features, malformed hands, low detail, oversaturated, text, watermark"
)


class QualificationProfilesMixin:
    """Cohesive qualification-runner responsibility mixin used by the public facade."""

    def default_generation_profile(self, blueprint: BlueprintSnapshot) -> dict[str, Any]:
        request = dict(self.context.generation_defaults())
        runtime_profile_hint: dict[str, Any] = {}
        if blueprint.family == "sdxl":
            # Reuse the existing SDXL model-profile knowledge as an editable starting
            # point. This is recommendation provenance, not new compatibility evidence.
            profile = profile_for_sdxl_filename(blueprint.model_filename)
            runtime_profile_hint = profile.to_dict()
            recommended_steps = [int(value) for value in profile.recommended_steps if int(value) > 0]
            preferred_steps = int(profile.required_steps or (recommended_steps[0] if recommended_steps else 0) or 0)
            if preferred_steps > 0:
                request["steps"] = preferred_steps
            request["width"] = int(profile.native_width)
            request["height"] = int(profile.native_height)
            if profile.image_gen_cfg_scale is not None:
                request["cfg_scale"] = float(profile.image_gen_cfg_scale)
            if profile.sampler_name:
                request["sampler_name"] = str(profile.sampler_name)
            if profile.scheduler_name:
                request["scheduler_name"] = str(profile.scheduler_name)

        request.update(
            {
                "model_path": blueprint.model_path,
                "positive_prompt": str(request.get("positive_prompt") or DEFAULT_TEST_PROMPT),
                "negative_prompt": str(request.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
                "seed": int(request.get("seed") if request.get("seed") is not None else 123456789),
                "batch_size": 1,
                "batch_count": 1,
                "unlimited": False,
                "save_images": True,
                "hires_enabled": False,
                "loras": [],
                "lora_paths": [],
                "textual_inversions": [],
                "advanced_models_enabled": True,
                "advanced_model_family": blueprint.family,
                "advanced_model_components": dict(blueprint.components),
                "advanced_model_allow_digital_components": True,
                "advanced_model_t5_device": "cpu",
                # Existing architecture profiles may apply qualified step/CFG
                # recommendations. Sampler/scheduler remain explicit test inputs.
                "model_enforce_recommended_steps": True,
                "model_enforce_recommended_cfg": True,
            }
        )
        return {
            "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
            "profile_id": f"qualification-{_slug(Path(blueprint.model_filename).stem)}",
            "model": blueprint.to_dict(),
            "profile_state": "editable_unvalidated",
            "runtime_profile_hint": runtime_profile_hint,
            "notes": (
                "Edit generation values before qualification when the checkpoint has documented "
                "recommended settings. Existing SDXL runtime-profile recommendations are used as "
                "editable starting values when detected. The runner records the effective request "
                "and never treats technical success as quality acceptance."
            ),
            "request": request,
        }

    def write_profile_template(self, model_path: str | Path, output_path: str | Path) -> Path:
        blueprint = self.blueprint_for_model(model_path)
        destination = Path(output_path).expanduser().resolve()
        _write_yaml(destination, self.default_generation_profile(blueprint))
        return destination

    def load_generation_profile(
        self,
        blueprint: BlueprintSnapshot,
        profile_path: str | Path | None,
    ) -> dict[str, Any]:
        base = self.default_generation_profile(blueprint)
        if profile_path is None:
            return base
        supplied = _load_yaml_mapping(Path(profile_path).expanduser().resolve())
        request = dict(base["request"])
        supplied_request = supplied.get("request") or {}
        if not isinstance(supplied_request, dict):
            raise ValueError("Generation profile 'request' must be a mapping.")
        request.update(dict(supplied_request))
        # The selected checkpoint blueprint is authoritative for the control.
        request["advanced_models_enabled"] = True
        request["advanced_model_family"] = blueprint.family
        request["advanced_model_components"] = dict(blueprint.components)
        request["model_path"] = blueprint.model_path
        output = dict(base)
        output.update({key: value for key, value in supplied.items() if key != "request"})
        output["request"] = request
        return output
