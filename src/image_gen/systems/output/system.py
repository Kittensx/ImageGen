from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    GenerationRequest,
    GenerationResult,
    normalize_prompt_asset_list,
)
from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import AssetReference, GenerationManifest
from modules.txt2img.manifest_builder import build_generation_manifest
from modules.txt2img.output_saver import (
    GenerationOutputSaver,
    SavedImageRecord,
    save_generation_batch,
)


def _build_prompt_asset_manifest_entries(
    entries: Any,
    *,
    asset_type: str,
) -> list[AssetReference]:
    normalized = normalize_prompt_asset_list(
        entries or [],
        asset_type=asset_type,
        default_source="api_request",
    )
    output: list[AssetReference] = []
    for index, selection in enumerate(normalized):
        payload = selection.to_serializable_dict()
        requested_path = str(selection.requested_path or selection.path or "")
        resolved_path = str(selection.resolved_path or selection.path or "")
        requested_name = str(selection.name or Path(resolved_path or requested_path).stem or "")
        resolved_exists = bool(resolved_path and Path(resolved_path).expanduser().is_file())
        asset = AssetReference.create_asset(asset_type)
        asset.provider = "local"
        asset.requested_display_name = requested_name
        asset.requested_filename = Path(requested_path or resolved_path).name if (requested_path or resolved_path) else ""
        asset.requested_path = requested_path
        asset.requested_identifier = selection.catalog_asset_id or selection.asset_id or requested_name
        asset.requested_hash = selection.requested_hash
        asset.requested_hash_type = "sha256" if selection.requested_hash else ""
        asset.resolved_display_name = requested_name
        asset.resolved_filename = Path(resolved_path).name if resolved_path else ""
        asset.resolved_path = resolved_path
        asset.resolved_identifier = selection.catalog_asset_id or selection.asset_id or requested_name
        asset.resolved_hash = selection.resolved_hash
        asset.resolved_hash_type = "sha256" if selection.resolved_hash else ""
        asset.resolution_status = "resolved" if resolved_path else "unresolved"
        asset.resolution_method = "runtime_catalog" if resolved_path else "request_only"
        asset.action_taken = "applied" if selection.enabled and asset_type == "lora" and resolved_path else ("staged" if selection.enabled else "disabled")
        asset.was_found = bool(resolved_path)
        asset.was_used_for_generation = bool(
            selection.enabled
            and resolved_path
            and (asset_type == "lora" or selection.metadata.get("runtime_applied") is True)
        )
        asset.is_required_for_rerun = bool(selection.enabled)
        asset.should_autoload = bool(selection.enabled)
        asset.source_url = selection.source_url
        asset.warning_messages = [] if resolved_exists or not resolved_path else ["Resolved prompt asset path was not present when the manifest was written."]
        asset.extra = {
            "contract_version": PROMPT_ASSET_CONTRACT_VERSION,
            "asset_id": selection.asset_id,
            "catalog_asset_id": selection.catalog_asset_id,
            "weight": float(selection.weight),
            "enabled": bool(selection.enabled),
            "polarity": selection.polarity,
            "activation_text": selection.activation_text,
            "model_family": selection.model_family,
            "source": selection.source,
            "original_source": selection.original_source,
            "order": int(selection.order if selection.order is not None else index),
            "metadata": dict(selection.metadata or {}),
        }
        output.append(asset)
    return output


@dataclass
class PreparedOutputSaveRequest:
    output_dir: str
    final_prefix: str
    images: list[Image.Image] = field(default_factory=list)
    manifest: Any | None = None
    save_txt: bool = True
    save_json: bool = True
    lowres_prefix: str | None = None
    lowres_images: list[Image.Image] = field(default_factory=list)
    lowres_manifest: Any | None = None
    expected_count: int = 0


class OutputSystem:
    """Own manifest construction and all normal txt2img file persistence."""

    def build_manifest(
        self,
        *,
        request: GenerationRequest,
        extras: dict[str, Any] | None,
        pipeline_result: GenerationResult,
        generation_time_sec: float | None = None,
        device_name: str | None = None,
    ):
        extras = dict(extras or {})
        schedule_extra = dict(pipeline_result.schedule_extra or {})
        sampler_extra = dict(pipeline_result.sampler_extra or {})
        seed = request.seed
        if seed is None:
            resolved = list(request.resolved_seeds or [])
            seed = resolved[0] if resolved else -1
        hires_dimension_plan = dict(getattr(request, "hires_dimension_plan", {}) or {})
        hires_enabled = bool(getattr(request, "hires_enabled", False))
        manifest_width = int(
            hires_dimension_plan.get("effective_width") or request.width
            if hires_enabled
            else request.width
        )
        manifest_height = int(
            hires_dimension_plan.get("effective_height") or request.height
            if hires_enabled
            else request.height
        )

        manifest = build_generation_manifest(
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            seed=int(seed),
            width=manifest_width,
            height=manifest_height,
            steps=int(request.steps),
            cfg_scale=float(request.cfg_scale),
            batch_size=int(request.batch_size),
            batch_count=int(extras.get("batch_count", 1) or 1),
            sampler_name=str(request.sampler_name or ""),
            scheduler_name=str(request.scheduler_name or ""),
            model_path=str(extras.get("model_path") or ""),
            request=request,
            compatibility_mode=schedule_extra.get("compatibility_mode"),
            effective_steps=sampler_extra.get(
                "effective_steps", schedule_extra.get("effective_steps")
            ),
            scheduler_step_override_applied=sampler_extra.get(
                "scheduler_step_override_applied",
                schedule_extra.get("scheduler_step_override_applied"),
            ),
            active_blend_methods=schedule_extra.get("active_blend_methods"),
            active_blend_weights=schedule_extra.get("active_blend_weights"),
            tail_features_used=schedule_extra.get("tail_features_used"),
            predicted_stop_step=schedule_extra.get("predicted_stop_step"),
            device_name=device_name,
            generation_time_sec=generation_time_sec,
        )
        # The VAE is carried as a request extra today. Persist an explicit
        # selection (including ``None`` for the checkpoint/default VAE) so a
        # replay preflight can distinguish "not recorded" from "recorded as
        # default" without guessing.
        manifest.optional_for_rerun.extra["vae_path"] = extras.get("vae_path")
        scheduler_resolution = dict(
            (getattr(request, "diagnostics", {}) or {}).get("scheduler_settings")
            or extras.get("scheduler_settings_resolution")
            or {}
        )
        if scheduler_resolution:
            effective_scheduler_settings = dict(
                scheduler_resolution.get("effective_settings") or {}
            )
            if effective_scheduler_settings:
                # Replay is value-driven: embed the complete effective object,
                # including the resolved step count, rather than relying on a
                # mutable preset reference.
                manifest.optional_for_rerun.scheduler_kwargs = json_safe(
                    effective_scheduler_settings
                )
            manifest.optional_for_rerun.extra["scheduler_preset_reference"] = json_safe(
                scheduler_resolution.get("preset_reference") or {}
            )
            manifest.extra["scheduler_settings"] = json_safe(scheduler_resolution)
        model_provenance = dict(extras.get("model_provenance") or {})
        if model_provenance:
            manifest.base_model.requested_path = str(
                model_provenance.get("requested_path") or ""
            )
            manifest.base_model.requested_filename = str(
                model_provenance.get("file_name") or ""
            )
            manifest.base_model.resolved_path = str(
                model_provenance.get("loaded_path")
                or model_provenance.get("resolved_path")
                or ""
            )
            manifest.base_model.resolved_filename = str(
                model_provenance.get("file_name") or ""
            )
            manifest.base_model.resolved_hash = str(
                model_provenance.get("sha256") or ""
            )
            manifest.base_model.resolved_hash_type = (
                "sha256" if manifest.base_model.resolved_hash else ""
            )
            manifest.base_model.resolution_status = "loaded"
            manifest.base_model.resolution_method = "runtime_model_loader"
            manifest.base_model.action_taken = "loaded_for_generation"
            manifest.base_model.was_found = True
            manifest.base_model.was_used_for_generation = True
            manifest.base_model.is_required_for_rerun = True
        manifest.extra.update(json_safe(extras))
        if hires_enabled:
            manifest.extra["base_dimensions"] = {
                "width": int(request.width),
                "height": int(request.height),
            }
            manifest.extra["output_dimensions"] = {
                "width": manifest_width,
                "height": manifest_height,
            }
        pipeline_metadata = json_safe(pipeline_result.metadata or {})
        denoising_contract = dict(
            (pipeline_result.metadata or {}).get("denoising_contract") or {}
        )
        manifest.extra["pipeline_metadata"] = pipeline_metadata
        manifest.extra["denoising_contract"] = json_safe(denoising_contract)
        memory_management = dict((pipeline_result.metadata or {}).get("memory_management") or {})
        if memory_management:
            manifest.extra["memory_management"] = json_safe(memory_management)
        runtime_execution = dict((pipeline_result.metadata or {}).get("runtime_execution") or {})
        if runtime_execution:
            manifest.extra["runtime_execution"] = json_safe(runtime_execution)
            manifest.optional_for_rerun.extra["runtime_execution_schema_version"] = int(
                runtime_execution.get("schema_version", 1) or 1
            )
        prompt_parser = dict((pipeline_result.metadata or {}).get("prompt_parser") or {})
        prompt_shortcut_profile = dict((pipeline_result.metadata or {}).get("prompt_shortcut_profile") or {})
        prompt_translation = dict((pipeline_result.metadata or {}).get("prompt_translation") or {})
        prompt_contract = dict((pipeline_result.metadata or {}).get("prompt_contract") or {})
        prompt_processing = dict((pipeline_result.metadata or {}).get("prompt_processing") or {})
        prompt_preflight = dict((pipeline_result.metadata or {}).get("prompt_preflight") or getattr(request, "prompt_preflight", {}) or {})
        if prompt_parser:
            manifest.extra["prompt_parser"] = json_safe(prompt_parser)
        if prompt_shortcut_profile:
            manifest.extra["prompt_shortcut_profile"] = json_safe(prompt_shortcut_profile)
            manifest.optional_for_rerun.extra["prompt_shortcut_profile_name"] = str(prompt_shortcut_profile.get("name") or getattr(request, "prompt_shortcut_profile_name", "legacy_default"))
            manifest.optional_for_rerun.extra["prompt_shortcut_profile_snapshot"] = json_safe(prompt_shortcut_profile.get("effective_mapping") or getattr(request, "prompt_shortcut_profile_snapshot", {}) or {})
            manifest.optional_for_rerun.extra["prompt_parser_preset_name"] = str(prompt_shortcut_profile.get("parser_preset_name") or getattr(request, "prompt_parser_preset_name", "") or "")
        if prompt_translation:
            manifest.extra["prompt_translation"] = json_safe(prompt_translation)
        if prompt_contract:
            manifest.extra["prompt_contract"] = json_safe(prompt_contract)
            manifest.optional_for_rerun.extra["canonical_prompt_contract"] = json_safe(prompt_contract)
        if prompt_processing:
            manifest.extra["prompt_processing"] = json_safe(prompt_processing)
        if prompt_preflight:
            manifest.extra["prompt_preflight"] = json_safe(prompt_preflight)
            manifest.optional_for_rerun.extra["prompt_preflight"] = json_safe(prompt_preflight)
        for key in (
            "prediction_type",
            "prediction_conversion",
            "model_input_preconditioning",
            "cfg_rescale",
            "cfg_rescale_applied",
            "guidance_owner",
            "guidance_mode",
            "guidance_math_version",
            "legacy_clamp_guidance",
            "model_dtype",
            "solver_dtype",
        ):
            if key in denoising_contract:
                manifest.extra[key] = json_safe(denoising_contract[key])
        for key in (
            "guidance_owner",
            "guidance_mode",
            "guidance_math_version",
            "cfg_rescale",
            "cfg_rescale_applied",
            "legacy_clamp_guidance",
        ):
            if key in sampler_extra:
                manifest.extra[key] = json_safe(sampler_extra[key])
        manifest.extra["cfg_scale"] = float(request.cfg_scale)
        effective_steps = int(
            sampler_extra.get("effective_steps")
            or schedule_extra.get("effective_steps")
            or getattr(request, "steps", 0)
            or 0
        )
        cfg_step_series = sampler_extra.get("cfg_step_series")
        if not isinstance(cfg_step_series, dict) or not isinstance(cfg_step_series.get("points"), list):
            cfg_step_series = {
                "schema_version": 1,
                "coordinate": "completed_denoising_step",
                "source": "flat_request_fallback",
                "supports_future_step_overrides": True,
                "points": [
                    {
                        "step_index": int(index),
                        "requested_cfg_scale": float(request.cfg_scale),
                        "effective_cfg_scale": float(request.cfg_scale),
                        "guidance_mode": sampler_extra.get("guidance_mode", "flat"),
                        "cfg_rescale": float(getattr(request, "cfg_rescale", 0.0) or 0.0),
                        "cfg_rescale_applied": bool(float(getattr(request, "cfg_rescale", 0.0) or 0.0) > 0.0),
                        "override_source": "base_request",
                        "transition_id": None,
                    }
                    for index in range(max(effective_steps, 0))
                ],
            }
        manifest.extra["cfg_step_series"] = json_safe(cfg_step_series)
        if "cfg_effective_guidance_summary" in sampler_extra:
            manifest.extra["cfg_effective_guidance_summary"] = json_safe(
                sampler_extra.get("cfg_effective_guidance_summary")
            )
        if "cfg_effective_range" in sampler_extra:
            manifest.extra["cfg_effective_range"] = json_safe(
                sampler_extra.get("cfg_effective_range")
            )
        manifest.extra["sigma_used_for_prediction"] = json_safe(
            sampler_extra.get("sigma_used_for_prediction")
        )
        manifest.extra["model_timestep"] = json_safe(
            sampler_extra.get("model_timestep")
        )
        manifest.extra["schedule"] = json_safe(schedule_extra)
        manifest.extra["sampler"] = json_safe(sampler_extra)
        lora_stack = getattr(request, "loras", None) or extras.get("resolved_lora_stack") or extras.get("loras") or []
        textual_inversion_stack = getattr(request, "textual_inversions", None) or extras.get("textual_inversions") or []
        normalized_loras = normalize_prompt_asset_list(lora_stack, asset_type="lora", default_source="api_request")
        normalized_textual_inversions = normalize_prompt_asset_list(
            textual_inversion_stack,
            asset_type="textual_inversion",
            default_source="api_request",
        )
        manifest.loras = _build_prompt_asset_manifest_entries(normalized_loras, asset_type="lora")
        manifest.embeddings = _build_prompt_asset_manifest_entries(
            normalized_textual_inversions,
            asset_type="textual_inversion",
        )
        prompt_asset_contract = {
            "contract_version": PROMPT_ASSET_CONTRACT_VERSION,
            "loras": [asset.to_serializable_dict() for asset in normalized_loras],
            "textual_inversions": [asset.to_serializable_dict() for asset in normalized_textual_inversions],
            "exact_order": [
                {"asset_type": asset.asset_type, "identity": asset.identity_key(), "order": asset.order}
                for asset in [*normalized_loras, *normalized_textual_inversions]
            ],
        }
        manifest.optional_for_rerun.extra["prompt_asset_contract_version"] = PROMPT_ASSET_CONTRACT_VERSION
        manifest.optional_for_rerun.extra["loras"] = prompt_asset_contract["loras"]
        manifest.optional_for_rerun.extra["lora_paths"] = [
            asset.resolved_path or asset.path or asset.requested_path
            for asset in normalized_loras
            if asset.resolved_path or asset.path or asset.requested_path
        ]
        manifest.optional_for_rerun.extra["textual_inversions"] = prompt_asset_contract["textual_inversions"]
        manifest.optional_for_rerun.extra["prompt_assets"] = prompt_asset_contract
        manifest.extra["prompt_assets"] = prompt_asset_contract
        active_assets = extras.get("_webui_active_prompt_assets")
        if isinstance(active_assets, list):
            manifest.optional_for_rerun.extra["_webui_active_prompt_assets"] = json_safe(active_assets)
        return manifest

    @staticmethod
    def _build_lowres_manifest(
        manifest: Any | None,
        request: GenerationRequest,
    ) -> GenerationManifest | None:
        if manifest is None:
            return None
        source = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
        lowres = GenerationManifest.from_dict(json_safe(source))
        lowres.required_for_rerun.width = int(request.width)
        lowres.required_for_rerun.height = int(request.height)
        lowres.optional_for_rerun.extra["hires_enabled"] = False
        lowres.optional_for_rerun.extra["hires_save_lowres"] = False
        lowres.optional_for_rerun.extra["hires_dimension_plan"] = {}
        lowres.extra["artifact_role"] = "hires_base_lowres"
        lowres.extra["artifact_source"] = "exact_base_pass_latents"
        lowres.extra["output_dimensions"] = {
            "width": int(request.width),
            "height": int(request.height),
        }
        lowres.extra["hires_parent"] = {
            "enabled": True,
            "scale": float(getattr(request, "hires_scale", 1.5) or 1.5),
            "upscaler": str(getattr(request, "hires_upscaler", "latent_bicubic") or "latent_bicubic"),
        }
        return lowres


    @staticmethod
    def _copy_manifest(manifest: Any | None) -> Any | None:
        if manifest is None:
            return None
        source = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
        return GenerationManifest.from_dict(json_safe(source))

    def prepare_save_request(
        self,
        *,
        pipeline_result: GenerationResult,
        request: GenerationRequest,
        manifest: Any | None,
        output_dir: str,
        save_txt: bool = True,
        save_json: bool = True,
    ) -> PreparedOutputSaveRequest:
        if pipeline_result.images is None:
            raise ValueError("Cannot save images because GenerationResult.images is None.")
        final_images = GenerationOutputSaver._coerce_pil_images(pipeline_result.images)
        prepared = PreparedOutputSaveRequest(
            output_dir=str(output_dir),
            final_prefix=str(request.output_prefix),
            images=final_images,
            manifest=self._copy_manifest(manifest),
            save_txt=bool(save_txt),
            save_json=bool(save_json),
            expected_count=len(final_images),
        )
        lowres_images = pipeline_result.auxiliary_images.get("hires_base_lowres")
        if (
            bool(getattr(request, "hires_enabled", False))
            and bool(getattr(request, "hires_save_lowres", False))
            and lowres_images is not None
        ):
            prepared.lowres_prefix = f"lowres-{request.output_prefix}"
            prepared.lowres_images = GenerationOutputSaver._coerce_pil_images(lowres_images)
            prepared.lowres_manifest = self._build_lowres_manifest(manifest, request)
            prepared.expected_count += len(prepared.lowres_images)
        return prepared

    def save_prepared(self, prepared: PreparedOutputSaveRequest) -> list[SavedImageRecord]:
        records: list[SavedImageRecord] = []
        if prepared.lowres_images:
            lowres_records = save_generation_batch(
                images=prepared.lowres_images,
                output_dir=prepared.output_dir,
                prefix=str(prepared.lowres_prefix or "lowres"),
                manifest=self._copy_manifest(prepared.lowres_manifest),
                save_txt=prepared.save_txt,
                save_json=prepared.save_json,
            )
            records.extend(lowres_records)

        final_records = save_generation_batch(
            images=prepared.images,
            output_dir=prepared.output_dir,
            prefix=prepared.final_prefix,
            manifest=self._copy_manifest(prepared.manifest),
            save_txt=prepared.save_txt,
            save_json=prepared.save_json,
        )
        records.extend(final_records)
        return records

    def save(
        self,
        *,
        pipeline_result: GenerationResult,
        request: GenerationRequest,
        manifest: Any | None,
        output_dir: str,
        save_txt: bool = True,
        save_json: bool = True,
    ) -> list[SavedImageRecord]:
        prepared = self.prepare_save_request(
            pipeline_result=pipeline_result,
            request=request,
            manifest=manifest,
            output_dir=output_dir,
            save_txt=save_txt,
            save_json=save_json,
        )
        records = self.save_prepared(prepared)
        if prepared.lowres_images:
            hires_metadata = pipeline_result.metadata.setdefault("hires_fix", {})
            lowres_metadata = hires_metadata.setdefault("lowres_artifact", {})
            lowres_metadata["saved_paths"] = [
                record.image_path for record in records[: len(prepared.lowres_images)]
            ]

        pipeline_result.saved_paths = [record.image_path for record in records]
        return records
