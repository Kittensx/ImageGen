from __future__ import annotations

from typing import Any

from modules.attention_backend import attention_backend_report

from image_gen.contracts import GenerationResult
from image_gen.runtime_options import (
    build_runtime_execution_record,
    compare_runtime_execution_records,
    runtime_execution_fingerprint,
)
from image_gen.systems.memory import compare_preflight_to_actual

from .context import GenerationContext

def _execution_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "successful_call_delta": max(
            0,
            int(after.get("successful_call_count", 0) or 0)
            - int(before.get("successful_call_count", 0) or 0),
        ),
        "failed_call_delta": max(
            0,
            int(after.get("failed_call_count", 0) or 0)
            - int(before.get("failed_call_count", 0) or 0),
        ),
    }


class FinalizeStageMixin:
    def _run_finalize_stage(self, ctx: GenerationContext) -> GenerationResult:
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        owns_session = ctx.owns_session
        performance = ctx.performance
        provider_execution_before_generation = ctx.provider_execution_before_generation
        provider_execution_after_base = ctx.provider_execution_after_base
        provider_execution_after_hires = ctx.provider_execution_after_hires
        auxiliary_images = ctx.auxiliary_images
        base_dimension_plan = ctx.base_dimension_plan
        base_vae_memory_controls = ctx.base_vae_memory_controls
        conditioning = ctx.conditioning
        diagnostic_decode_enabled = ctx.diagnostic_decode_enabled
        diagnostic_decode_source = ctx.diagnostic_decode_source
        dimension_plan = ctx.dimension_plan
        final_output_width = ctx.final_output_width
        final_output_height = ctx.final_output_height
        hires_execution_plan = ctx.hires_execution_plan
        hires_metadata = ctx.hires_metadata
        images = ctx.images
        outpaint_metadata = ctx.outpaint_metadata
        output_quality = ctx.output_quality
        pixel_hires_job = ctx.pixel_hires_job
        pixel_hires_preflight = ctx.pixel_hires_preflight
        pixel_hires_stage_timings = ctx.pixel_hires_stage_timings
        preview_policy_report = ctx.preview_policy_report
        sample_output = ctx.sample_output
        schedule = ctx.schedule
        state_extra = ctx.state_extra
        trace_exports = ctx.trace_exports

        effective_denoising_contract = (
            self.systems.denoising.guided_epsilon_contract_metadata()
            if sample_output.extra.get("integration_mode_used") == "dpmpp_2m"
            else self.systems.denoising.contract_metadata()
        )
        for key in (
            "model_prediction_type",
            "prediction_conversion",
            "model_input_preconditioning",
            "cfg_rescale",
            "model_dtype",
            "solver_dtype",
        ):
            if key in sample_output.extra:
                target = "prediction_type" if key == "model_prediction_type" else key
                effective_denoising_contract[target] = sample_output.extra[key]

        preview_writer = getattr(self.state, "extra", {}).get(
            "live_preview_frame_writer"
        ) if self.state is not None else None
        if preview_writer is not None and callable(getattr(preview_writer, "close", None)):
            try:
                preview_writer.close()
            except Exception:
                pass
        live_preview_summary = (
            preview_writer.summary()
            if preview_writer is not None and callable(getattr(preview_writer, "summary", None))
            else {"enabled": False}
        )

        self.memory_manager.capture("generation_complete")
        attention_report = attention_backend_report(self.components.unet)
        attention_execution_by_pass = {
            "before_generation": provider_execution_before_generation,
            "after_base_pass": provider_execution_after_base,
            "after_hires_pass": provider_execution_after_hires,
            "base_pass": _execution_delta(
                provider_execution_before_generation, provider_execution_after_base
            ),
            "hires_pass": _execution_delta(
                provider_execution_after_base, provider_execution_after_hires
            ),
        }
        hires_metadata["attention_execution"] = dict(
            attention_execution_by_pass["hires_pass"]
        )
        memory_summary = self.memory_manager.summary()
        if pixel_hires_preflight is not None:
            hires_metadata["memory_forecast_vs_actual"] = compare_preflight_to_actual(
                pixel_hires_preflight, memory_summary
            )
        if pixel_hires_job:
            hires_metadata["phase14n6_stage_timings_ms"] = dict(
                pixel_hires_stage_timings
            )
            hires_metadata["phase14n6_transfer_count"] = len(
                list(memory_summary.get("transfers") or [])
            )
        vae_memory_report = (
            self.systems.decoding.memory_control_report()
            if callable(
                getattr(
                    self.systems.decoding,
                    "memory_control_report",
                    None,
                )
            )
            else dict(base_vae_memory_controls)
        )
        runtime_execution = build_runtime_execution_record(
            startup_options=(
                state_extra.get("runtime_startup_options")
                if isinstance(state_extra, dict)
                else None
            ),
            attention_report=attention_report,
            memory_summary=memory_summary,
            preview_policy=preview_policy_report,
            vae_report=vae_memory_report,
            hires_metadata=hires_metadata,
            runtime_job_settings=(
                state_extra if isinstance(state_extra, dict) else None
            ),
            execution_device=str(self.device),
            external_fallback_reasons=(
                [str(state_extra.get("cpu_fallback_reason"))]
                if isinstance(state_extra, dict)
                and state_extra.get("cpu_fallback_reason")
                else None
            ),
        )
        conformance_source = (
            state_extra.get("runtime_replay_conformance_source")
            if isinstance(state_extra, dict)
            else None
        )
        runtime_execution["conformance_fingerprint"] = runtime_execution_fingerprint(
            runtime_execution
        )
        if isinstance(conformance_source, dict):
            conformance_report = compare_runtime_execution_records(
                conformance_source,
                runtime_execution,
            )
            runtime_execution.setdefault("replay", {})["conformance"] = conformance_report

        performance_summary = performance.finish(
            attention_backend=attention_report,
            memory_management=memory_summary,
            hires_enabled=hires_execution_plan.enabled,
        )
        if pixel_hires_job:
            performance_index = dict(performance_summary.get("stage_index") or {})
            final_decode_metric = dict(performance_index.get("final_decode") or {})
            if final_decode_metric.get("duration_ms") is not None:
                pixel_hires_stage_timings["final_decode_ms"] = float(
                    final_decode_metric["duration_ms"]
                )
            hires_metadata["phase14n6_stage_timings_ms"] = dict(
                pixel_hires_stage_timings
            )
        result = GenerationResult(
            request=request,
            images=images,
            latents=sample_output.latents,
            conditioning=conditioning,
            schedule=schedule,
            sampler=sample_output,
            trace_exports=trace_exports,
            auxiliary_images=auxiliary_images,
            metadata={
                "model_prediction_type": self.systems.denoising.prediction_type,
                "model_architecture": str(getattr(self.components, "architecture", "") or ""),
                "model_runtime_profile": dict(getattr(self.components, "model_runtime_profile", {}) or {}),
                "vae_scaling_factor": float(
                    getattr(
                        self.systems.decoding,
                        "vae_scaling_factor",
                        getattr(self.components, "vae_scaling_factor", 0.18215),
                    )
                ),
                "component_placement": self.components.placement_metadata(),
                "attention_backend": attention_report,
                "attention_performance": performance_summary,
                "attention_execution_by_pass": attention_execution_by_pass,
                "denoising_contract": effective_denoising_contract,
                "initial_noise_sigma": schedule.initial_sigma,
                "output_owner": "txt2img.output_saver",
                "output_system": "image_gen.systems.output.OutputSystem",
                "runtime": "image_gen.runtime.GenerationPipeline",
                "live_preview": live_preview_summary,
                "preview_policy": preview_policy_report,
                "memory_management": memory_summary,
                "runtime_execution": runtime_execution,
                "oom_recovery": dict(memory_summary.get("oom_recovery") or {}),
                "attention_and_vae_memory_controls": {
                    "attention_slicing": str(
                        self.memory_manager.settings.attention_slicing
                    ),
                    "vae": vae_memory_report,
                },
                "prompt_parser": dict((conditioning.extra or {}).get("prompt_parser") or {}),
                "prompt_shortcut_profile": dict((conditioning.extra or {}).get("prompt_shortcut_profile") or {}),
                "prompt_translation": dict((conditioning.extra or {}).get("prompt_translation") or {}),
                "prompt_contract": dict((conditioning.extra or {}).get("prompt_contract") or {}),
                "regional_runtime": dict(
                    (getattr(request, "diagnostics", {}) or {}).get("regional_runtime")
                    or sample_output.extra.get("regional_runtime")
                    or {}
                ),
                "regional_runtime_passes": dict(
                    (getattr(request, "diagnostics", {}) or {}).get("regional_runtime_passes")
                    or {}
                ),
                "dimension_plan": dimension_plan.to_serializable_dict(),
                "base_dimension_plan": base_dimension_plan.to_serializable_dict(),
                "hires_fix": hires_metadata,
                "outpaint_prototype": dict(outpaint_metadata),
                "diagnostic_decode": {
                    "enabled": diagnostic_decode_enabled,
                    "source": diagnostic_decode_source,
                    "final_output_decode_unaffected": True,
                },
                "output_dimensions": {
                    "width": final_output_width,
                    "height": final_output_height,
                },
                "output_quality": dict(output_quality),
            },
        )
        result = diagnostics.run_stage(
            session,
            "runtime",
            "result_validation",
            lambda: self._validate_result(request, result),
        )
        result.metadata["diagnostics"] = diagnostics.summary(session)
        if owns_session:
            result.metadata["diagnostics"] = diagnostics.complete(
                session, result=result
            )
        return result
