from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from image_gen.contracts import GenerationRequest, GenerationResult
from image_gen.systems.diagnostics import DiagnosticSession, PipelineStageError
from image_gen.systems.output import PreparedOutputSaveRequest
from image_gen.systems.outpainting import (
    OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
    build_post_generation_shape_action,
    format_outpaint_failure,
    normalize_outpaint_source_handoff_mode,
    resolve_outpaint_shape_target,
)
from modules.project_context import ProjectContext
from modules.txt2img.output_saver import SavedImageRecord
from modules.txt2img.request_loader import load_request_payload, payload_to_generation_request


@dataclass
class Txt2ImgRunResult:
    request: GenerationRequest
    request_extras: dict[str, Any] = field(default_factory=dict)
    pipeline_result: GenerationResult = field(default_factory=GenerationResult)
    manifest: Any | None = None
    saved_records: list[SavedImageRecord] = field(default_factory=list)
    generation_time_sec: float | None = None
    run_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    prepared_save_request: PreparedOutputSaveRequest | None = None
    expected_saved_count: int = 0


def _prepare_output_directory(
    project_context: ProjectContext,
    request: GenerationRequest,
    *,
    should_save: bool,
) -> Path | None:
    """Create the runtime-owned output directory before path validation."""
    request.save_images = bool(should_save)
    if not should_save:
        return None

    configured_output = request.output_dir or project_context.txt2img_output_root
    output_path = Path(str(configured_output)).expanduser()
    if not output_path.is_absolute():
        output_path = project_context.resolve_project_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    request.output_dir = str(output_path)
    return output_path

def _verify_saved_records(records: list[SavedImageRecord]) -> list[SavedImageRecord]:
    """Require at least one real image file when persistence was requested."""
    if not records:
        raise RuntimeError(
            "Image saving was requested, but the output system returned no saved records."
        )
    missing = [record.image_path for record in records if not Path(record.image_path).is_file()]
    if missing:
        raise RuntimeError(
            "The output system reported image paths that do not exist: "
            + ", ".join(missing)
        )
    return records


class RequestExecutionMixin:
    def _generate_with_optional_shape_expansion(
        self,
        *,
        pipeline: Any,
        request: GenerationRequest,
        session: DiagnosticSession,
    ) -> GenerationResult:
        if not bool(getattr(request, "outpaint_shape_expansion_enabled", False)):
            return pipeline.generate(request, diagnostic_session=session)

        base_width = int(getattr(request, "outpaint_shape_base_width", 0) or request.width)
        base_height = int(getattr(request, "outpaint_shape_base_height", 0) or request.height)
        request.outpaint_shape_base_width = base_width
        request.outpaint_shape_base_height = base_height
        latent_preparation = getattr(getattr(pipeline, "systems", None), "latent_preparation", None)
        pixel_alignment_multiple = int(
            getattr(latent_preparation, "pixel_alignment_multiple", 0)
            or getattr(pipeline, "latent_scale_factor", 0)
            or 8
        )
        try:
            target = resolve_outpaint_shape_target(
                base_width=base_width,
                base_height=base_height,
                target_mode=str(getattr(request, "outpaint_shape_target_mode", "square") or "square"),
                target_width=int(getattr(request, "outpaint_shape_target_width", 0) or 0),
                target_height=int(getattr(request, "outpaint_shape_target_height", 0) or 0),
                dimension_multiple=pixel_alignment_multiple,
            )
        except Exception as exc:
            raise RuntimeError(format_outpaint_failure("outpaint_source_handoff", str(exc))) from exc

        base_request = replace(
            request,
            width=base_width,
            height=base_height,
            outpaint_prototype_enabled=False,
        )
        base_request.outpaint_shape_base_width = base_width
        base_request.outpaint_shape_base_height = base_height
        base_result = pipeline.generate(base_request, diagnostic_session=session)
        if not torch.is_tensor(base_result.images):
            raise RuntimeError(format_outpaint_failure(
                "outpaint_live_source_capture",
                "Fresh txt2img base generation did not return an image tensor for P-3 shape expansion.",
            ))
        if not torch.is_tensor(base_result.latents):
            raise RuntimeError(format_outpaint_failure(
                "outpaint_live_source_capture",
                "Fresh txt2img base generation did not return a sampled latent for P-3 shape expansion.",
            ))

        expansion_request = replace(
            base_request,
            width=int(target["target_width"]),
            height=int(target["target_height"]),
            outpaint_target_width=int(target["target_width"]),
            outpaint_target_height=int(target["target_height"]),
            outpaint_prototype_enabled=True,
            outpaint_source_image="",
            outpaint_anchor=str(getattr(request, "outpaint_shape_anchor", "center") or "center"),
            outpaint_source_x=-1,
            outpaint_source_y=-1,
            outpaint_context_seed_mode=str(
                getattr(request, "outpaint_shape_context_seed_mode", "edge_pad_v1") or "edge_pad_v1"
            ),
            outpaint_denoising_strength=float(
                getattr(request, "outpaint_shape_denoising_strength", 0.40) or 0.40
            ),
            # P-2 demonstrated that a provisional context seed is useful only
            # when the encoded provisional canvas participates in the new area.
            outpaint_latent_strategy="canvas_regional_noise_v1",
            outpaint_prompt_mode=str(
                getattr(request, "outpaint_shape_prompt_mode", "overlay_only_v1") or "overlay_only_v1"
            ),
            outpaint_overlay_positive_prompt=str(
                getattr(request, "outpaint_shape_overlay_positive_prompt", "") or ""
            ),
            outpaint_overlay_negative_prompt=str(
                getattr(request, "outpaint_shape_overlay_negative_prompt", "") or ""
            ),
        )
        expansion_request.outpaint_shape_expansion_enabled = True
        expansion_request.outpaint_shape_target_width = int(target["target_width"])
        expansion_request.outpaint_shape_target_height = int(target["target_height"])
        expansion_request.outpaint_shape_base_width = base_width
        expansion_request.outpaint_shape_base_height = base_height
        setattr(expansion_request, "_outpaint_runtime_source_tensor", base_result.images)
        setattr(expansion_request, "_outpaint_runtime_source_latent", base_result.latents)
        setattr(
            expansion_request,
            "_outpaint_runtime_source_handoff_requested",
            str(getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"),
        )

        expanded_result = pipeline.generate(expansion_request, diagnostic_session=session)
        if bool(getattr(request, "outpaint_shape_save_base", False)):
            expanded_result.auxiliary_images["outpaint_pre_expansion_base"] = (
                base_result.images.detach().clone()
            )

        outpaint_record = dict(expanded_result.metadata.get("outpaint_prototype") or {})
        source_handoff = dict(outpaint_record.get("source_handoff") or {})
        source_handoff_contract = dict(outpaint_record.get("source_handoff_contract") or {})
        runtime_record = {
            "contract_version": OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
            "enabled": True,
            "source_kind": "fresh_txt2img_generation",
            "source_origin": str(source_handoff_contract.get("source_origin") or "fresh_generation"),
            "disk_round_trip": False,
            "base_generation_width": base_width,
            "base_generation_height": base_height,
            "base_latent_shape": list(base_result.latents.shape),
            "base_latent_dtype": str(base_result.latents.dtype),
            "base_latent_device_at_handoff": str(base_result.latents.device),
            "target_mode": str(target["target_mode"]),
            "target_width": int(target["target_width"]),
            "target_height": int(target["target_height"]),
            "anchor": str(expansion_request.outpaint_anchor),
            "context_seed_mode": str(expansion_request.outpaint_context_seed_mode),
            "source_handoff_requested": str(
                getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"
            ),
            "source_handoff_requested_stable": normalize_outpaint_source_handoff_mode(
                getattr(request, "outpaint_shape_source_handoff", "auto") or "auto",
                default="auto",
            ),
            "source_handoff_actual": str(source_handoff.get("actual") or ""),
            "source_handoff_actual_stable": str(source_handoff_contract.get("actual_source_handoff") or ""),
            "source_handoff_fallback_reason": str(source_handoff.get("fallback_reason") or source_handoff_contract.get("source_handoff_fallback_reason") or ""),
            "latent_grid_alignment": dict(source_handoff_contract.get("latent_grid_alignment") or source_handoff.get("alignment") or {}),
            "preservation_reference_source": str(
                source_handoff_contract.get("preservation_reference_source") or source_handoff.get("preservation_reference_source") or ""
            ),
            "source_was_vae_reencoded_for_protected_latent": bool(
                source_handoff_contract.get("source_was_vae_reencoded_for_protected_latent", source_handoff.get("source_was_vae_reencoded_for_protected_latent", True))
            ),
            "live_source_latent_reused": bool(source_handoff_contract.get("live_source_latent_reused", source_handoff.get("live_source_latent_reused", False))),
            "outpaint_prompt_mode": str(expansion_request.outpaint_prompt_mode),
            "outpaint_overlay_positive_prompt": str(expansion_request.outpaint_overlay_positive_prompt),
            "outpaint_overlay_negative_prompt": str(expansion_request.outpaint_overlay_negative_prompt),
            "outpaint_denoising_strength": float(expansion_request.outpaint_denoising_strength),
            "provisional_base_saved": bool(getattr(request, "outpaint_shape_save_base", False)),
            "expanded_result_is_primary": True,
            "post_generation_shape_action": build_post_generation_shape_action(
                base_width=base_width,
                base_height=base_height,
                target_width=int(target["target_width"]),
                target_height=int(target["target_height"]),
                anchor=str(expansion_request.outpaint_anchor),
                context_seed_mode=str(expansion_request.outpaint_context_seed_mode),
                source_handoff_policy=str(getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"),
                overlay_positive_prompt=str(expansion_request.outpaint_overlay_positive_prompt),
                overlay_negative_prompt=str(expansion_request.outpaint_overlay_negative_prompt),
                denoise_strength=float(expansion_request.outpaint_denoising_strength),
                save_pre_expansion_base=bool(getattr(request, "outpaint_shape_save_base", False)),
            ),
            "geometry_fingerprint": dict(outpaint_record.get("geometry_fingerprint") or {}),
            "inference_fingerprint": dict(outpaint_record.get("inference_fingerprint") or {}),
            "audit": dict(outpaint_record.get("audit") or {}),
        }
        runtime_record["runtime_handoff_tensors_released_after_expansion"] = True
        expansion_request.outpaint_shape_runtime_record = dict(runtime_record)
        for transient_name in (
            "_outpaint_runtime_source_tensor",
            "_outpaint_runtime_source_latent",
            "_outpaint_runtime_source_handoff_requested",
        ):
            if hasattr(expansion_request, transient_name):
                delattr(expansion_request, transient_name)
        # The prototype flag is an internal implementation detail of the second
        # in-job pass. Persist only the P-3 shape-expansion contract so replay
        # regenerates the base first instead of trying to load an uploaded source.
        expansion_request.outpaint_prototype_enabled = False
        expansion_request.outpaint_source_image = ""
        expanded_result.request = expansion_request
        expanded_result.metadata["outpaint_shape_expansion"] = dict(runtime_record)
        expanded_result.metadata["base_generation"] = {
            "width": base_width,
            "height": base_height,
            "latent_shape": list(base_result.latents.shape),
            "output_dimensions": dict(base_result.metadata.get("output_dimensions") or {}),
        }
        return expanded_result

    def run_request(
        self,
        request: GenerationRequest,
        extras: dict[str, Any] | None = None,
        *,
        save_images: bool | None = None,
        save_txt: bool = True,
        save_json: bool = True,
        save_diagnostics_json: bool = True,
        defer_output_save: bool = False,
    ) -> Txt2ImgRunResult:
        should_save = request.save_images if save_images is None else bool(save_images)
        _prepare_output_directory(
            self.project_context,
            request,
            should_save=should_save,
        )

        extras = dict(extras or {})
        effective_config_fn = getattr(self.project_context, "effective_config", None)
        effective_config = (
            effective_config_fn()
            if callable(effective_config_fn)
            else {
                "project_root": str(
                    getattr(self.project_context, "project_root", ".")
                )
            }
        )
        session = self.diagnostics_system.start(
            request,
            effective_config=effective_config,
            request_extras=extras,
        )
        try:
            model_path = extras.get("model_path") or self.model_loading_system.default_model_path
            self.diagnostics_system.run_stage(
                session,
                "configuration",
                "validate_generation_paths",
                lambda: self.project_context.require_generation_ready(
                    model_path=model_path,
                    output_dir=request.output_dir,
                    require_output=should_save,
                ),
            )
            request.device = str(self.device)
            self._configure_runtime_state(extras, session)
            self.diagnostics_system.run_stage(
                session,
                "model_profile",
                "sdxl_preflight",
                lambda: self._apply_sdxl_runtime_preflight(request, extras),
            )
            self.diagnostics_system.run_stage(
                session,
                "model_profile",
                "sd3_preflight",
                lambda: self._apply_sd3_runtime_preflight(request, extras),
            )
            request, extras = self.diagnostics_system.run_stage(
                session,
                "registry",
                "resolve_plugins",
                lambda: self.registry_system.apply_resolution(request, extras),
            )
            session.request_extras.update(
                {
                    key: value
                    for key, value in extras.items()
                    if key not in {
                        "progress_reporter",
                        "live_preview_callback",
                        "live_preview_warning_callback",
                        "live_preview_sink_factory",
                        "live_preview_sink",
                        "live_preview_frame_writer",
                        "live_preview_event_callback",
                        "live_preview_memory_event_callback",
                        "memory_event_callback",
                        "model_runtime_event_callback",
                        "model_runtime_first_step_callback",
                    }
                }
            )
            pipeline = self.diagnostics_system.run_stage(
                session,
                "runtime",
                "compose_pipeline",
                lambda: self._build_pipeline(request, extras, session),
            )

            # CNRR-06: once the active composition is fully built, a generation
            # profile may declare future same-family compositions. Establish one
            # process-local lease and start a single ordered CPU warm worker before
            # sampling so target-only component construction can overlap the current
            # GPU generation. Shared live components remain placement-preserved.
            planned_schedule = extras.get("model_runtime_composition_schedule")
            if planned_schedule:
                ensure_lease = getattr(self, "ensure_composition_execution_lease", None)
                prime_prefetch = getattr(self, "prime_composition_prefetch", None)
                if callable(ensure_lease):
                    lease_result = ensure_lease(planned_schedule, extras)
                    extras["composition_execution_lease"] = dict(lease_result or {})
                    session.request_extras["composition_execution_lease"] = dict(lease_result or {})
                    if (lease_result or {}).get("state") == "active" and callable(prime_prefetch):
                        prefetch_result = prime_prefetch(extras)
                        extras["composition_prefetch"] = dict(prefetch_result or {})
                        session.request_extras["composition_prefetch"] = dict(prefetch_result or {})

            performance_matrix_enabled = os.environ.get(
                "IMAGE_GEN_PERF_MATRIX", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            phase_started_unix = time.time()
            if performance_matrix_enabled:
                print(
                    "PERFORMANCE_PHASE_JSON: "
                    + json.dumps(
                        {
                            "event": "generation_start",
                            "timestamp_unix": phase_started_unix,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            started = time.perf_counter()
            generation_time_sec = 0.0
            try:
                pipeline_result = self._generate_with_optional_shape_expansion(
                    pipeline=pipeline, request=request, session=session
                )
            finally:
                generation_time_sec = time.perf_counter() - started
                if performance_matrix_enabled:
                    print(
                        "PERFORMANCE_PHASE_JSON: "
                        + json.dumps(
                            {
                                "event": "generation_end",
                                "timestamp_unix": time.time(),
                                "elapsed_sec": generation_time_sec,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            request = pipeline_result.request or request
            pipeline_result.metadata["model_provenance"] = dict(
                extras.get("model_provenance") or {}
            )
            pipeline_result.metadata["vae_provenance"] = dict(
                extras.get("vae_provenance") or {}
            )

            manifest = self.diagnostics_system.run_stage(
                session,
                "output",
                "build_manifest",
                lambda: self.output_system.build_manifest(
                    request=request,
                    extras=extras,
                    pipeline_result=pipeline_result,
                    generation_time_sec=generation_time_sec,
                    device_name=str(request.device or self.device),
                ),
            )

            saved_records: list[SavedImageRecord] = []
            prepared_save_request: PreparedOutputSaveRequest | None = None
            expected_saved_count = 0
            if should_save:
                output_dir = request.output_dir or extras.get("output_dir")
                if not output_dir:
                    raise ValueError("An output directory is required when saving images.")
                pipeline.memory_manager.capture("before_output_save")
                if hasattr(manifest, "extra"):
                    manifest.extra["memory_management"] = pipeline.memory_manager.summary()
                if defer_output_save:
                    prepared_save_request = self.diagnostics_system.run_stage(
                        session,
                        "output",
                        "prepare_save_request",
                        lambda: self.output_system.prepare_save_request(
                            pipeline_result=pipeline_result,
                            request=request,
                            manifest=manifest,
                            output_dir=str(output_dir),
                            save_txt=save_txt,
                            save_json=save_json,
                            save_diagnostics_json=save_diagnostics_json,
                        ),
                    )
                    expected_saved_count = int(prepared_save_request.expected_count or 0)
                    pipeline_result.metadata["memory_management"] = pipeline.memory_manager.summary()
                else:
                    def save_and_verify_outputs() -> list[SavedImageRecord]:
                        records = self.output_system.save(
                            pipeline_result=pipeline_result,
                            request=request,
                            manifest=manifest,
                            output_dir=str(output_dir),
                            save_txt=save_txt,
                            save_json=save_json,
                            save_diagnostics_json=save_diagnostics_json,
                        )
                        return _verify_saved_records(records)

                    saved_records = self.diagnostics_system.run_stage(
                        session,
                        "output",
                        "save",
                        lambda: pipeline.memory_manager.observe_stage(
                            "output_save",
                            save_and_verify_outputs,
                        ),
                    )
                    expected_saved_count = len(saved_records)
                    pipeline_result.metadata["memory_management"] = pipeline.memory_manager.summary()

            diagnostics_summary = self.diagnostics_system.complete(
                session, result=pipeline_result
            )
            pipeline_result.metadata["diagnostics"] = diagnostics_summary
            return Txt2ImgRunResult(
                request=request,
                request_extras=extras,
                pipeline_result=pipeline_result,
                manifest=manifest,
                saved_records=saved_records,
                generation_time_sec=generation_time_sec,
                run_id=session.run_id,
                diagnostics=diagnostics_summary,
                prepared_save_request=prepared_save_request,
                expected_saved_count=expected_saved_count,
            )
        except PipelineStageError:
            raise
        except Exception as exc:
            raise self.diagnostics_system.fail_unassigned(
                session, exc, system="runtime", operation="run_request"
            ) from exc

    def run_from_sources(
        self,
        *,
        config_path: str | None = None,
        manifest_path: str | None = None,
        infotext_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
        base_payload: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        save_txt: bool = True,
        save_json: bool = True,
        save_diagnostics_json: bool = True,
    ) -> Txt2ImgRunResult:
        payload = load_request_payload(
            config_path=config_path,
            manifest_path=manifest_path,
            infotext_path=infotext_path,
            cli_overrides=cli_overrides,
            base_payload=base_payload,
        )
        request, payload_extras = payload_to_generation_request(payload)
        merged_extras = dict(extras or {})
        merged_extras.update(payload_extras)
        return self.run_request(
            request,
            merged_extras,
            save_txt=save_txt,
            save_json=save_json,
            save_diagnostics_json=save_diagnostics_json,
        )
