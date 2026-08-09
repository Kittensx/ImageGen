import { api } from "../api.js";
import { state } from "../state.js";
import { $, notify } from "../utils.js";
import { renderCfgGraph } from "./cfg-lab.js?v=0.1.45";

let collectValues = () => ({});
let applyValues = async () => ({});
let onJobQueued = async () => {};

const REPLAY_STATUS_LABELS = {
  exact: "Exact",
  normalized: "Normalized",
  preserved_backend_only: "Preserved",
  replaced: "Replaced",
  missing: "Missing",
  unsupported: "Unsupported",
};

const SECTION_DEFINITIONS = [
  {
    title: "Image",
    fields: [
      ["Filename", "image.name"],
      ["Output path", "image.relative_path"],
      ["Generated", "image.timestamp"],
      ["Width", "replay.width", "width"],
      ["Height", "replay.height", "height"],
    ],
  },
  {
    title: "Prompt",
    fields: [
      ["Positive prompt", "replay.positive_prompt", "positive_prompt", "prompt"],
      ["Negative prompt", "replay.negative_prompt", "negative_prompt", "prompt"],
    ],
  },
  {
    title: "Prompt Processing",
    fields: [
      ["Base prompt parser", "replay.prompt_parser_name", "prompt_parser_name"],
      ["Base parser options", "replay.prompt_parser_kwargs", "prompt_parser_kwargs", "json"],
      ["Base shortcut profile", "replay.prompt_shortcut_profile_name", "prompt_shortcut_profile_name"],
      ["Parser preset", "replay.prompt_parser_preset_name", "prompt_parser_preset_name"],
      ["Parser shadow comparison enabled", "replay.prompt_shadow_compare", "prompt_shadow_compare"],
      ["Base prompt route plan", "replay.prompt_route_plan", "prompt_route_plan", "json"],
      ["Hires prompt route plan", "replay.hires_prompt_route_plan", "hires_prompt_route_plan", "json"],
      ["Semantic replay mode", "replay.prompt_semantic_replay_mode", "prompt_semantic_replay_mode"],
      ["Base semantic replay contract", "replay.prompt_semantic_pass_records.base", "prompt_semantic_pass_records.base", "json"],
      ["Hires semantic replay contract", "replay.prompt_semantic_pass_records.hires", "prompt_semantic_pass_records.hires", "json"],
      ["Original batch semantic contracts", "replay.batch_prompt_semantic_pass_records", "batch_prompt_semantic_pass_records", "json"],
      ["REGION replay mode", "replay.region_replay_mode", "region_replay_mode"],
      ["Base REGION plan", "replay.region_pass_records.base", "region_pass_records.base", "json"],
      ["Hires REGION plan", "replay.region_pass_records.hires", "region_pass_records.hires", "json"],
      ["Original batch REGION plans", "replay.batch_region_pass_records", "batch_region_pass_records", "json"],
      ["Base REGION runtime estimate", "replay.region_pass_records.base.runtime_estimate", "region_pass_records.base.runtime_estimate", "json"],
      ["Hires REGION runtime estimate", "replay.region_pass_records.hires.runtime_estimate", "region_pass_records.hires.runtime_estimate", "json"],
      ["REGION runtime telemetry", "replay.regional_runtime", "regional_runtime", "json"],
      ["REGION runtime telemetry by pass", "replay.regional_runtime_passes", "regional_runtime_passes", "json"],
      ["Original batch REGION telemetry", "replay.batch_regional_runtime_passes", "batch_regional_runtime_passes", "json"],
      ["Base positive shadow comparison", "manifest.extra.prompt_processing.base.positive.shadow_comparison", "", "json"],
      ["Base negative shadow comparison", "manifest.extra.prompt_processing.base.negative.shadow_comparison", "", "json"],
      ["Shortcut mapping hash", "manifest.extra.prompt_shortcut_profile.mapping_hash"],
      ["Raw positive prompt", "manifest.extra.prompt_contract.raw_positive"],
      ["Translated positive parser input", "manifest.extra.prompt_contract.translated_positive"],
      ["Canonical positive prompt", "manifest.extra.prompt_contract.canonical_positive"],
      ["Raw negative prompt", "manifest.extra.prompt_contract.raw_negative"],
      ["Translated negative parser input", "manifest.extra.prompt_contract.translated_negative"],
      ["Canonical negative prompt", "manifest.extra.prompt_contract.canonical_negative"],
      ["Hires parser mode", "replay.hires_prompt_parser_mode", "hires_prompt_parser_mode"],
      ["Hires parser", "replay.hires_prompt_parser_name", "hires_prompt_parser_name"],
      ["Hires parser options", "replay.hires_prompt_parser_kwargs", "hires_prompt_parser_kwargs", "json"],
      ["Hires shortcut mode", "replay.hires_shortcut_profile_mode", "hires_shortcut_profile_mode"],
      ["Hires shortcut profile", "replay.hires_shortcut_profile_name", "hires_shortcut_profile_name"],
      ["Hires positive prompt", "replay.hires_positive_prompt", "hires_positive_prompt", "prompt"],
      ["Hires negative prompt", "replay.hires_negative_prompt", "hires_negative_prompt", "prompt"],
      ["Hires size mode", "replay.hires_size_mode", "hires_size_mode"],
      ["Uniform hires scale", "replay.hires_uniform_scale", "hires_uniform_scale"],
      ["Hires target width", "replay.hires_width", "hires_width"],
      ["Hires target height", "replay.hires_height", "hires_height"],
      ["Hires width scale", "replay.hires_axis_scale_width", "hires_axis_scale_width"],
      ["Hires height scale", "replay.hires_axis_scale_height", "hires_axis_scale_height"],
      ["Hires aspect ratio changed", "replay.hires_aspect_ratio_changed", "hires_aspect_ratio_changed"],
      ["Hires dimension plan version", "replay.hires_dimension_plan_version", "hires_dimension_plan_version"],
      ["Hires dimension plan", "replay.hires_dimension_plan", "hires_dimension_plan", "json"],
      ["Hires prompt processing", "manifest.extra.prompt_processing.hires", "", "json"],
      ["Prompt substitutions", "manifest.extra.prompt_translation", "", "json"],
      ["Prompt preflight", "manifest.extra.prompt_preflight", "", "json"],
      ["Replay compatibility", "manifest.extra.prompt_processing", "", "json"],
    ],
  },
  {
    title: "Core generation",
    fields: [
      ["Seed", "replay.seed", "seed"],
      ["Steps", "replay.steps", "steps"],
      ["CFG scale", "replay.cfg_scale", "cfg_scale"],
      ["Batch size", "replay.batch_size", "batch_size"],
      ["Batch count", "replay.batch_count", "batch_count"],
    ],
  },
  {
    title: "Existing image expansion",
    fields: [
      ["Expansion enabled", "replay.outpaint_prototype_enabled", "outpaint_prototype_enabled"],
      ["Source image", "replay.outpaint_source_image", "outpaint_source_image"],
      ["Image placement", "replay.outpaint_anchor", "outpaint_anchor"],
      ["Blend width", "replay.outpaint_feather_px", "outpaint_feather_px"],
      ["Edge initialization", "replay.outpaint_context_seed_mode", "outpaint_context_seed_mode"],
      ["Denoise strength", "replay.outpaint_denoising_strength", "outpaint_denoising_strength"],
      ["New-area initialization", "replay.outpaint_latent_strategy", "outpaint_latent_strategy"],
      ["Extension prompt source", "replay.outpaint_prompt_mode", "outpaint_prompt_mode"],
      ["Extension prompt", "replay.outpaint_overlay_positive_prompt", "outpaint_overlay_positive_prompt"],
      ["Extension negative prompt", "replay.outpaint_overlay_negative_prompt", "outpaint_overlay_negative_prompt"],
      ["Effective conditioning", "manifest.extra.pipeline_metadata.outpaint_prototype.conditioning", "", "json"],
      ["Expansion diagnostics", "replay.outpaint_diagnostic_artifacts", "outpaint_diagnostic_artifacts"],
      ["Canvas plan", "manifest.extra.pipeline_metadata.outpaint_prototype.canvas_plan", "", "json"],
      ["Mask contract", "manifest.extra.pipeline_metadata.outpaint_prototype.mask", "", "json"],
      ["Schedule start sigma", "manifest.extra.pipeline_metadata.outpaint_prototype.schedule.start_sigma"],
      ["Schedule start timestep", "manifest.extra.pipeline_metadata.outpaint_prototype.schedule.start_timestep"],
      ["Noise policy", "manifest.extra.pipeline_metadata.outpaint_prototype.noise.policy_id"],
      ["Preservation strategy", "manifest.extra.pipeline_metadata.outpaint_prototype.preservation.strategy"],
      ["Sampler-step preservation calls", "manifest.extra.pipeline_metadata.outpaint_prototype.preservation.sampler_step_restore_calls"],
      ["Diagnostic artifact path", "manifest.extra.pipeline_metadata.outpaint_prototype.diagnostic_artifact_path"],
      ["Expansion runtime record", "replay.outpaint_prototype_record", "outpaint_prototype_record", "json"],
    ],
  },
  {
    title: "Post-generation expansion",
    fields: [
      ["Enabled", "replay.outpaint_shape_expansion_enabled", "outpaint_shape_expansion_enabled"],
      ["Base width", "replay.outpaint_shape_base_width", "outpaint_shape_base_width"],
      ["Base height", "replay.outpaint_shape_base_height", "outpaint_shape_base_height"],
      ["Target mode", "replay.outpaint_shape_target_mode", "outpaint_shape_target_mode"],
      ["Target width", "replay.outpaint_shape_target_width", "outpaint_shape_target_width"],
      ["Target height", "replay.outpaint_shape_target_height", "outpaint_shape_target_height"],
      ["Image placement", "replay.outpaint_shape_anchor", "outpaint_shape_anchor"],
      ["Edge initialization", "replay.outpaint_shape_context_seed_mode", "outpaint_shape_context_seed_mode"],
      ["Requested source reuse", "replay.outpaint_shape_source_handoff", "outpaint_shape_source_handoff"],
      ["Actual source reuse", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.source_handoff_actual"],
      ["Source reuse fallback reason", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.source_handoff_fallback_reason"],
      ["Latent-grid alignment", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.latent_grid_alignment", "", "json"],
      ["Live source latent reused", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.live_source_latent_reused"],
      ["Protected source re-encoded", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.source_was_vae_reencoded_for_protected_latent"],
      ["Extension prompt source", "replay.outpaint_shape_prompt_mode", "outpaint_shape_prompt_mode"],
      ["Extension prompt", "replay.outpaint_shape_overlay_positive_prompt", "outpaint_shape_overlay_positive_prompt"],
      ["Extension negative prompt", "replay.outpaint_shape_overlay_negative_prompt", "outpaint_shape_overlay_negative_prompt"],
      ["Denoise", "replay.outpaint_shape_denoising_strength", "outpaint_shape_denoising_strength"],
      ["Saved pre-expansion base", "replay.outpaint_shape_save_base", "outpaint_shape_save_base"],
      ["Geometry fingerprint", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.geometry_fingerprint.sha256"],
      ["Inference fingerprint", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.inference_fingerprint.sha256"],
      ["Outpaint audit", "manifest.extra.pipeline_metadata.outpaint_shape_expansion.audit.summary_lines", "", "json"],
      ["Runtime record", "replay.outpaint_shape_runtime_record", "outpaint_shape_runtime_record", "json"],
    ],
  },
  {
    title: "Model and assets",
    fields: [
      ["Model display name", "image.model.display_name"],
      ["Model path", "replay.model_path", "model_path"],
      ["Model hash", "image.model.hash"],
      ["Model architecture summary", "image.model.architecture_summary"],
      ["Model family", "image.model.architecture_contract.family"],
      ["Prediction type", "image.model.architecture_contract.prediction_type"],
      ["Conditioning dimension", "image.model.architecture_contract.conditioning_dimension"],
      ["Model architecture", "image.model.architecture"],
      ["VAE display name", "image.vae.display_name"],
      ["VAE mode", "image.vae.mode"],
      ["VAE source", "image.vae.effective_source"],
      ["VAE path", "replay.vae_path", "vae_path"],
      ["VAE hash", "image.vae.hash"],
      ["LoRAs", "image.loras"],
      ["Embeddings", "image.embeddings"],
      ["Other assets", "image.other_assets"],
    ],
  },
  {
    title: "Sampling",
    fields: [
      ["Sampler", "replay.sampler_name", "sampler_name"],
      ["Scheduler", "replay.scheduler_name", "scheduler_name"],
      ["Sampler advanced settings", "replay.sampler_kwargs", "sampler_kwargs", "json"],
      ["Scheduler advanced settings", "replay.scheduler_kwargs", "scheduler_kwargs", "json"],
    ],
  },
  {
    title: "Hires refinement schedule",
    fields: [
      ["Algorithm version", "manifest.extra.pipeline_metadata.hires_fix.algorithm_version"],
      ["Step policy", "replay.hires_step_policy", "hires_step_policy"],
      ["Requested refinement steps", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.requested_refinement_steps"],
      ["Internal schedule steps", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.internal_schedule_steps"],
      ["Effective refinement steps", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.effective_refinement_steps"],
      ["Denoising strength", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.denoising_strength"],
      ["Schedule start index", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.start_index"],
      ["Starting sigma", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.start_sigma"],
      ["Starting timestep", "manifest.extra.pipeline_metadata.hires_fix.schedule_contract.start_timestep"],
      ["Hires sampler", "replay.hires_sampler_name", "hires_sampler_name"],
      ["Hires scheduler", "replay.hires_scheduler_name", "hires_scheduler_name"],
      ["Hires CFG scale", "replay.hires_cfg_scale", "hires_cfg_scale"],
      ["Hires CFG rescale", "replay.hires_cfg_rescale", "hires_cfg_rescale"],
      ["Hires strategy", "replay.hires_strategy", "hires_strategy"],
      ["Upscaler ID", "replay.hires_upscaler_id", "hires_upscaler_id"],
      ["Upscaler display name", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.display_name"],
      ["Upscaler architecture", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.architecture"],
      ["Upscaler native scale", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.native_scale"],
      ["Upscaler SHA-256", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.sha256"],
      ["Upscaler device", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.device"],
      ["Upscaler dtype", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.upscaler.dtype"],
      ["Tile size", "replay.hires_tile_size", "hires_tile_size"],
      ["Tile overlap", "replay.hires_tile_overlap", "hires_tile_overlap"],
      ["Tile count", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.tiling.tile_count"],
      ["Aspect handling", "replay.hires_aspect_policy", "hires_aspect_policy"],
      ["Padding mode", "replay.hires_padding_mode", "hires_padding_mode"],
      ["Blurred-edge method", "replay.hires_blurred_edge_method", "hires_blurred_edge_method"],
      ["Blurred-edge comparison diagnostics", "replay.hires_blurred_edge_compare_diagnostics", "hires_blurred_edge_compare_diagnostics"],
      ["Final size correction filter", "replay.hires_final_size_correction_filter", "hires_final_size_correction_filter"],
      ["Predicted native width", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.predicted_native_width"],
      ["Predicted native height", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.predicted_native_height"],
      ["Actual native width", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.actual_native_width"],
      ["Actual native height", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.actual_native_height"],
      ["Native dimension verified", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.native_dimension_match"],
      ["Native dimension discrepancy", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.native_dimension_discrepancy"],
      ["Target correction geometry", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.target_correction", "", "json"],
      ["Blurred-edge runtime", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.target_correction.blurred_edge_runtime", "", "json"],
      ["Correction audit summary", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.correction_audit.summary"],
      ["Correction severity", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.correction_audit.severity"],
      ["Correction fingerprint SHA-256", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.correction_fingerprint.sha256"],
      ["Correction fingerprint contract", "manifest.extra.pipeline_metadata.hires_fix.pixel_source_preparation.upscale_metadata.correction_fingerprint.contract", "", "json"],
      ["Save pre-denoise", "replay.hires_save_upscaled_pre_denoise", "hires_save_upscaled_pre_denoise"],
      ["Save VAE round-trip", "replay.hires_save_vae_roundtrip", "hires_save_vae_roundtrip"],
      ["VAE SHA-256", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.vae.sha256"],
      ["Intermediate hashes", "manifest.extra.pipeline_metadata.hires_fix.phase14n7_diagnostics.intermediate_artifacts.hashes", "", "json"],
      ["Noise policy", "manifest.extra.pipeline_metadata.hires_fix.noise_stream.policy_id"],
      ["Noise stream identifier", "manifest.extra.pipeline_metadata.hires_fix.noise_stream.stream_identifier"],
      ["Noise base seeds", "manifest.extra.pipeline_metadata.hires_fix.noise_stream.base_seeds", "", "json"],
      ["Noise derived seeds", "manifest.extra.pipeline_metadata.hires_fix.noise_stream.derived_seeds", "", "json"],
      ["Schedule fingerprint", "manifest.extra.pipeline_metadata.hires_fix.schedule_fingerprint.sha256"],
      ["Schedule conformance match", "manifest.extra.pipeline_metadata.hires_fix.schedule_conformance.matches"],
      ["Schedule conformance difference count", "manifest.extra.pipeline_metadata.hires_fix.schedule_conformance.difference_count"],
      ["Schedule conformance categories", "manifest.extra.pipeline_metadata.hires_fix.schedule_conformance.difference_categories", "", "json"],
      ["Schedule conformance differences", "manifest.extra.pipeline_metadata.hires_fix.schedule_conformance.differences", "", "json"],
    ],
  },
  {
    title: "Guidance / CFG diagnostics",
    fields: [
      ["Requested CFG", "replay.cfg_scale", "cfg_scale"],
      ["Prompt CFG pass schedules", "replay.prompt_cfg_pass_schedules", "", "json"],
      ["Base prompt CFG source", "replay.prompt_cfg_pass_schedules.base.source"],
      ["Base prompt CFG behavior", "replay.prompt_cfg_pass_schedules.base.behavior"],
      ["Base prompt CFG interpolation", "replay.prompt_cfg_pass_schedules.base.interpolation"],
      ["Base prompt CFG requested schedule", "replay.prompt_cfg_pass_schedules.base.requested_schedule", "", "json"],
      ["Base prompt CFG fingerprint", "replay.prompt_cfg_pass_schedules.base.schedule_fingerprint.digest"],
      ["Hires prompt CFG requested schedule", "replay.prompt_cfg_pass_schedules.hires.requested_schedule", "", "json"],
      ["Hires prompt CFG fingerprint", "replay.prompt_cfg_pass_schedules.hires.schedule_fingerprint.digest"],
      ["Prompt expansion pass records", "replay.prompt_expansion_pass_records", "", "json"],
      ["Base prompt expansion scope", "replay.prompt_expansion_pass_records.base.scope"],
      ["Base expanded prompts by slot", "replay.prompt_expansion_pass_records.base.expanded_positive_by_slot", "", "json"],
      ["Base expanded negatives by slot", "replay.prompt_expansion_pass_records.base.expanded_negative_by_slot", "", "json"],
      ["Base prompt expansion slot records", "replay.prompt_expansion_pass_records.base.slot_records", "", "json"],
      ["Hires prompt expansion scope", "replay.prompt_expansion_pass_records.hires.scope"],
      ["Hires expanded prompts by slot", "replay.prompt_expansion_pass_records.hires.expanded_positive_by_slot", "", "json"],
      ["Hires expanded negatives by slot", "replay.prompt_expansion_pass_records.hires.expanded_negative_by_slot", "", "json"],
      ["Hires prompt expansion slot records", "replay.prompt_expansion_pass_records.hires.slot_records", "", "json"],
      ["Batch prompt expansion records", "manifest.optional_for_rerun.extra.batch_prompt_expansion_pass_records", "", "json"],
      ["Base expanded positive prompt", "replay.prompt_expansion_pass_records.base.expanded_positive"],
      ["Base expanded negative prompt", "replay.prompt_expansion_pass_records.base.expanded_negative"],
      ["Base TONEG additions", "replay.prompt_expansion_pass_records.base.toneg_additions", "", "json"],
      ["Base wildcard selections", "replay.prompt_expansion_pass_records.base.wildcard_selections", "", "json"],
      ["Base random selections", "replay.prompt_expansion_pass_records.base.random_selections", "", "json"],
      ["Base variables", "replay.prompt_expansion_pass_records.base.variables", "", "json"],
      ["Base macro expansions", "replay.prompt_expansion_pass_records.base.macro_expansions", "", "json"],
      ["Base prompt expansion fingerprint", "replay.prompt_expansion_pass_records.base.fingerprint.digest"],
      ["Hires expanded positive prompt", "replay.prompt_expansion_pass_records.hires.expanded_positive"],
      ["Hires expanded negative prompt", "replay.prompt_expansion_pass_records.hires.expanded_negative"],
      ["Hires TONEG additions", "replay.prompt_expansion_pass_records.hires.toneg_additions", "", "json"],
      ["Hires wildcard selections", "replay.prompt_expansion_pass_records.hires.wildcard_selections", "", "json"],
      ["Hires random selections", "replay.prompt_expansion_pass_records.hires.random_selections", "", "json"],
      ["Hires variables", "replay.prompt_expansion_pass_records.hires.variables", "", "json"],
      ["Hires macro expansions", "replay.prompt_expansion_pass_records.hires.macro_expansions", "", "json"],
      ["Hires prompt expansion fingerprint", "replay.prompt_expansion_pass_records.hires.fingerprint.digest"],
      ["Guidance mode", "manifest.extra.guidance_mode"],
      ["Guidance math version", "manifest.extra.guidance_math_version"],
      ["Canonical CFG rescale", "replay.cfg_rescale", "cfg_rescale"],
      ["Canonical rescale applied", "manifest.extra.cfg_rescale_applied"],
      ["Legacy KES clamp guidance", "manifest.extra.legacy_clamp_guidance"],
      ["Effective CFG range", "manifest.extra.cfg_effective_range", "", "json"],
      ["Effective-guidance summary", "manifest.extra.cfg_effective_guidance_summary", "", "json"],
    ],
  },
  {
    title: "Memory management",
    fields: [
      ["Requested policy", "manifest.extra.memory_management.requested_policy"],
      ["Effective policy", "manifest.extra.memory_management.effective_policy"],
      ["Active GPU components", "manifest.extra.memory_management.active_gpu_components"],
      ["Offloaded components", "manifest.extra.memory_management.offloaded_components"],
      ["Requested preview policy", "manifest.extra.memory_management.preview_policy"],
      ["Image preview suspended", "manifest.extra.memory_management.preview_image_decode_suspended"],
      ["Preview suspension source", "manifest.extra.memory_management.preview_image_decode_suspension_source"],
      ["Preview suspension reason", "manifest.extra.memory_management.preview_image_decode_suspension_reason"],
      ["Preview decoder released", "manifest.extra.memory_management.preview_decoder_released"],
      ["Preview policy by stage", "manifest.extra.pipeline_metadata.preview_policy", "", "json"],
      ["CFG telemetry continued", "manifest.extra.memory_management.cfg_telemetry_continues_during_preview_suspension"],
      ["OOM retries", "manifest.extra.memory_management.oom_retry_count_by_stage", "", "json"],
      ["Automatic actions", "manifest.extra.memory_management.automatic_actions", "", "json"],
    ],
  },
  {
    title: "Runtime execution and replay conformance",
    fields: [
      ["Runtime profile", "manifest.extra.runtime_execution.runtime_profile.profile_id"],
      ["Attention requested", "manifest.extra.runtime_execution.attention.requested_backend"],
      ["Attention effective", "manifest.extra.runtime_execution.attention.effective_backend"],
      ["Verified kernel provider", "manifest.extra.runtime_execution.attention.verified_kernel_provider"],
      ["Runtime fingerprint", "manifest.extra.runtime_execution.conformance_fingerprint.sha256"],
      ["Replay runtime match", "manifest.extra.runtime_execution.replay.conformance.matches"],
      ["Replay difference count", "manifest.extra.runtime_execution.replay.conformance.difference_count"],
      ["Replay difference categories", "manifest.extra.runtime_execution.replay.conformance.difference_categories", "", "json"],
      ["Replay runtime differences", "manifest.extra.runtime_execution.replay.conformance.differences", "", "json"],
    ],
  },
  {
    title: "Provenance and diagnostics",
    fields: [
      ["Metadata source", "metadata_source"],
      ["JSON sidecar", "provenance.json_sidecar"],
      ["TXT sidecar", "provenance.txt_sidecar"],
      ["PNG IMAGE_GEN manifest", "provenance.png_manifest_available"],
      ["PNG parameters", "provenance.png_parameters_available"],
      ["Runtime diagnostics", "provenance.runtime_diagnostics"],
    ],
  },
];

function valueAt(source, path) {
  return String(path || "").split(".").reduce((value, part) => value?.[part], source);
}

function hasPath(source, path) {
  const parts = String(path || "").split(".");
  let value = source;
  for (const part of parts) {
    if (value === null || value === undefined || !Object.prototype.hasOwnProperty.call(value, part)) {
      return false;
    }
    value = value[part];
  }
  return true;
}

function setValueAtPath(target, path, value) {
  const parts = String(path || "").split(".");
  let cursor = target;
  parts.forEach((part, index) => {
    if (index === parts.length - 1) {
      cursor[part] = value;
    } else {
      if (!cursor[part] || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) cursor[part] = {};
      cursor = cursor[part];
    }
  });
}

function cloneValue(value) {
  if (value && typeof value === "object") {
    return JSON.parse(JSON.stringify(value));
  }
  return value;
}

function humanizeToken(token) {
  return String(token || "")
    .replaceAll("_", " ")
    .replaceAll(".", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function flattenLeafPaths(source, prefix = "") {
  if (source === null || source === undefined) return [];
  if (typeof source !== "object" || Array.isArray(source)) return [[prefix, source]];
  const entries = Object.entries(source);
  if (!entries.length) return [[prefix, {}]];
  return entries.flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return flattenLeafPaths(value, path);
  });
}

function valueIsMissing(value) {
  return value === undefined || value === null || value === "";
}

function displayValue(value, recorded = !valueIsMissing(value)) {
  if (!recorded || value === undefined || value === null) return "Not recorded";
  if (value === "") return "(empty value recorded)";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function comparable(value) {
  if (value && typeof value === "object") {
    const sorted = Object.keys(value).sort().reduce((output, key) => {
      output[key] = comparable(value[key]);
      return output;
    }, {});
    return sorted;
  }
  return value;
}

function valuesEqual(left, right) {
  return JSON.stringify(comparable(left)) === JSON.stringify(comparable(right));
}

function currentValueIsEmpty(value) {
  if (value === undefined || value === null || value === "") return true;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === "object") return Object.keys(value).length === 0;
  return false;
}

function unsupportedEntry(fieldName) {
  return state.outputDetails.data?.unsupported?.[fieldName] || null;
}

function supportedAdvancedValue(fieldName, value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const output = { ...value };
  Object.keys(state.outputDetails.data?.unsupported || {}).forEach((path) => {
    const prefix = `${fieldName}.`;
    if (path.startsWith(prefix)) {
      delete output[path.slice(prefix.length)];
    }
  });
  return output;
}

function replayFieldSupported(fieldName) {
  if (!fieldName || unsupportedEntry(fieldName)) return false;
  const replay = state.outputDetails.data?.replay || {};
  if (!hasPath(replay, fieldName)) return false;
  const value = valueAt(replay, fieldName);
  if (value === null || value === undefined) return false;
  if (value === "" && !["positive_prompt", "negative_prompt"].includes(fieldName)) return false;
  if (fieldName.endsWith("_kwargs")) {
    return Object.keys(supportedAdvancedValue(fieldName, value) || {}).length > 0;
  }
  return true;
}

function selectedReplayPayload(fieldNames) {
  const replay = state.outputDetails.data?.replay || {};
  const payload = {};
  fieldNames.forEach((fieldName) => {
    if (!replayFieldSupported(fieldName)) return;
    const value = fieldName.endsWith("_kwargs")
      ? supportedAdvancedValue(fieldName, valueAt(replay, fieldName))
      : cloneValue(valueAt(replay, fieldName));
    setValueAtPath(payload, fieldName, value);
  });
  return payload;
}

function announce(message, kind = "info") {
  const live = $("#outputDetailsLiveRegion");
  live.textContent = message;
  live.className = `output-details-live ${kind}`;
}

async function copyText(value, label = "Value") {
  const text = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value ?? "");
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      if (!document.execCommand("copy")) throw new Error("Copy command was rejected.");
      textarea.remove();
    }
    announce(`${label} copied.`);
  } catch (error) {
    announce(`Unable to copy ${label.toLowerCase()}: ${error.message}`, "error");
  }
}

function renderObjectFieldGroup(labelText, path, replayField) {
  const group = document.createElement("div");
  group.className = "metadata-subsection";
  const title = document.createElement("h4");
  title.textContent = labelText;
  const values = flattenLeafPaths(valueAt(state.outputDetails.data, path));
  if (!values.length || (values.length === 1 && values[0][0] === "")) {
    group.replaceChildren(renderField([labelText, path, replayField, "json"]));
    return group;
  }
  const body = document.createElement("div");
  body.className = "metadata-fields";
  values
    .sort(([left], [right]) => String(left).localeCompare(String(right)))
    .forEach(([leafPath]) => {
      const prettyPath = leafPath.split(".").filter(Boolean).map(humanizeToken).join(" · ");
      body.append(renderField([
        `${labelText.replace(/ settings$/i, "")} · ${prettyPath || "value"}`,
        `${path}.${leafPath}`,
        `${replayField}.${leafPath}`,
      ]));
    });
  group.append(title, body);
  return group;
}

function renderField([labelText, path, replayField = "", displayKind = ""]) {
  const data = state.outputDetails.data;
  const value = valueAt(data, path);
  const recorded = replayField
    ? hasPath(data?.replay || {}, replayField)
    : hasPath(data, path) && !valueIsMissing(value);
  const missing = !recorded;
  const row = document.createElement("article");
  row.className = `metadata-field${missing ? " is-missing" : ""}`;

  const heading = document.createElement("header");
  const label = document.createElement("strong");
  label.textContent = labelText;
  const actions = document.createElement("div");
  actions.className = "metadata-field-actions";

  if (replayField) {
    const includeLabel = document.createElement("label");
    includeLabel.className = "metadata-include";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.replayField = replayField;
    checkbox.checked = replayFieldSupported(replayField);
    checkbox.disabled = !replayFieldSupported(replayField);
    checkbox.setAttribute("aria-label", `Include ${labelText}`);
    checkbox.addEventListener("change", () => {
      state.outputDetails.selectedFields = [...document.querySelectorAll("[data-replay-field]:checked")]
        .map((item) => item.dataset.replayField);
      const noSelectedFields = state.outputDetails.selectedFields.length === 0;
      $("#outputDetailsSendSelectedButton").disabled = noSelectedFields;
      $("#outputDetailsPreviewMergedButton").disabled = noSelectedFields;
    });
    const text = document.createElement("span");
    text.textContent = "Include";
    includeLabel.append(checkbox, text);
    actions.append(includeLabel);
  }

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "small-button metadata-copy-button";
  copyButton.textContent = "Copy";
  copyButton.disabled = missing;
  copyButton.addEventListener("click", () => copyText(value, labelText));
  actions.append(copyButton);
  heading.append(label, actions);

  let valueElement;
  if (displayKind === "prompt") {
    valueElement = document.createElement("textarea");
    valueElement.rows = 4;
    valueElement.readOnly = true;
    valueElement.value = displayValue(value, recorded);
  } else {
    valueElement = document.createElement("pre");
    valueElement.textContent = displayValue(value, recorded);
  }
  valueElement.className = "metadata-field-value";
  row.append(heading, valueElement);

  const directUnsupported = replayField ? unsupportedEntry(replayField) : null;
  if (directUnsupported) {
    const status = document.createElement("small");
    status.className = "metadata-field-status unsupported";
    status.textContent = directUnsupported.reason;
    row.append(status);
  } else if (missing) {
    const status = document.createElement("small");
    status.className = "metadata-field-status";
    status.textContent = "This value was not present in the available metadata source.";
    row.append(status);
  } else if (value === "") {
    const status = document.createElement("small");
    status.className = "metadata-field-status";
    status.textContent = "This field was explicitly recorded as empty and can be replayed.";
    row.append(status);
  }
  return row;
}

function renderUnsupported() {
  const entries = Object.entries(state.outputDetails.data?.unsupported || {});
  const section = $("#outputDetailsUnsupportedSection");
  const list = $("#outputDetailsUnsupportedList");
  list.replaceChildren();
  section.hidden = entries.length === 0;
  entries.forEach(([path, details]) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = path;
    const reason = document.createElement("span");
    reason.textContent = details.reason || details.status || "Unsupported by the current form.";
    const value = document.createElement("pre");
    value.textContent = displayValue(details.value);
    item.append(title, reason, value);
    list.append(item);
  });
}

function renderWarnings() {
  const warnings = state.outputDetails.data?.warnings || [];
  const box = $("#outputDetailsWarnings");
  box.replaceChildren();
  box.hidden = warnings.length === 0;
  warnings.forEach((message) => {
    const item = document.createElement("p");
    item.textContent = message;
    box.append(item);
  });
}

function cfgStepPoints(data) {
  const direct = valueAt(data, "manifest.extra.cfg_step_series.points");
  if (Array.isArray(direct) && direct.length) return direct;
  const nested = valueAt(data, "manifest.extra.sampler.cfg_step_series.points");
  if (Array.isArray(nested) && nested.length) return nested;
  const legacy = valueAt(data, "manifest.extra.sampler.cfg_effective_per_step");
  if (Array.isArray(legacy) && legacy.length) return legacy;
  return [];
}

function renderCfgStepDiagnostics(data) {
  const points = cfgStepPoints(data);
  if (!points.length) return null;
  const section = document.createElement("section");
  section.className = "metadata-section cfg-output-diagnostics";
  const heading = document.createElement("div");
  heading.className = "cfg-output-heading";
  const title = document.createElement("h3");
  title.textContent = "Actual CFG by denoising step";
  const source = document.createElement("span");
  source.textContent = valueAt(data, "manifest.extra.cfg_step_series.source") || "recorded series";
  heading.append(title, source);

  const requestedValues = points.map((point) => Number(point.requested_cfg_scale)).filter(Number.isFinite);
  const effectiveValues = points.map((point) => Number(point.effective_cfg_scale)).filter(Number.isFinite);
  const summary = document.createElement("div");
  summary.className = "cfg-output-summary-grid";
  const cards = [
    ["Steps recorded", points.length],
    ["Requested CFG", requestedValues.length ? requestedValues[0].toFixed(2) : "Not recorded"],
    ["Effective minimum", effectiveValues.length ? Math.min(...effectiveValues).toFixed(2) : "Not recorded"],
    ["Effective maximum", effectiveValues.length ? Math.max(...effectiveValues).toFixed(2) : "Not recorded"],
  ];
  cards.forEach(([labelText, value]) => {
    const card = document.createElement("article");
    const label = document.createElement("strong");
    label.textContent = labelText;
    const body = document.createElement("span");
    body.textContent = String(value);
    card.append(label, body);
    summary.append(card);
  });

  const graph = document.createElement("div");
  graph.className = "cfg-curve-graph cfg-output-graph";
  renderCfgGraph(graph, points);

  const details = document.createElement("details");
  details.className = "cfg-step-data-details";
  const detailsSummary = document.createElement("summary");
  detailsSummary.textContent = "Show recorded per-step CFG data";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(points, null, 2);
  details.append(detailsSummary, pre);

  const note = document.createElement("small");
  note.textContent = "This series uses explicit step/value records so future Dream-State branches can record CFG replacements or ramps beginning at a selected step without changing the graph contract.";
  section.append(heading, summary, graph, details, note);
  return section;
}

function renderDetails() {
  const data = state.outputDetails.data;
  $("#outputDetailsTitle").textContent = data?.image?.name || "Image Details";
  $("#outputDetailsImage").src = data?.image?.url || "";
  $("#outputDetailsImage").alt = data?.image?.name ? `Selected output ${data.image.name}` : "Selected output";
  $("#outputDetailsSourceBadge").textContent = String(data?.metadata_source || "unknown").replaceAll("_", " ");

  const content = $("#outputDetailsContent");
  content.replaceChildren();
  SECTION_DEFINITIONS.forEach((definition) => {
    const section = document.createElement("section");
    section.className = "metadata-section";
    const title = document.createElement("h3");
    title.textContent = definition.title;
    const fields = document.createElement("div");
    fields.className = "metadata-fields";
    definition.fields.forEach((field) => {
      if (field[3] === "json") {
        fields.append(renderObjectFieldGroup(field[0], field[1], field[2]));
      } else {
        fields.append(renderField(field));
      }
    });
    section.append(title, fields);
    content.append(section);
  });
  const cfgDiagnostics = renderCfgStepDiagnostics(data);
  if (cfgDiagnostics) content.append(cfgDiagnostics);

  state.outputDetails.selectedFields = [...document.querySelectorAll("[data-replay-field]:checked")]
    .map((item) => item.dataset.replayField);
  renderWarnings();
  renderUnsupported();
  $("#outputDetailsLoading").hidden = true;
  $("#outputDetailsBody").hidden = false;
  $("#outputDetailsCopyFullButton").disabled = false;
  const noSelectedFields = state.outputDetails.selectedFields.length === 0;
  const noReplayFields = Object.keys(selectedReplayPayload(Object.keys(data.replay || {}))).length === 0;
  $("#outputDetailsSendSelectedButton").disabled = noSelectedFields;
  $("#outputDetailsPreviewMergedButton").disabled = noSelectedFields;
  $("#outputDetailsDuplicateButton").disabled = noReplayFields;
  $("#outputDetailsExactReplayButton").disabled = noReplayFields;
  $("#outputDetailsNewSeedButton").disabled = noReplayFields;
  $("#outputDetailsCurrentModelButton").disabled = noReplayFields;
}

function renderError(error) {
  state.outputDetails.error = error.message || String(error);
  $("#outputDetailsLoading").hidden = true;
  $("#outputDetailsBody").hidden = true;
  $("#outputDetailsError").hidden = false;
  $("#outputDetailsError").textContent = state.outputDetails.error;
  announce(state.outputDetails.error, "error");
}

function showOverwriteConfirmation(values) {
  const current = collectValues();
  const changes = Object.entries(values).filter(([fieldName, value]) => (
    !valuesEqual(current[fieldName], value) && !currentValueIsEmpty(current[fieldName])
  ));
  if (!changes.length) return false;

  state.outputDetails.pendingValues = values;
  $("#outputDetailsConfirmationCount").textContent = String(changes.length);
  $("#outputDetailsConfirmationFields").textContent = changes.map(([name]) => name.replaceAll("_", " ")).join(", ");
  $("#outputDetailsConfirmation").hidden = false;
  $("#outputDetailsConfirmationApplyButton").focus();
  return true;
}

async function applyPayload(values) {
  if (!Object.keys(values).length) {
    announce("No supported replay fields are selected.", "error");
    return;
  }
  try {
    const result = await applyValues(values);
    const ignored = result?.unsupported || [];
    if (ignored.length) {
      announce(`Applied supported fields. ${ignored.length} advanced setting(s) remain unavailable.`, "error");
    } else {
      announce("Generation fields applied to the editable form.");
    }
    notify("Output metadata copied to the generation form.");
    closeOutputDetails();
  } catch (error) {
    announce(`Unable to apply metadata: ${error.message}`, "error");
  }
}

async function requestApply(values) {
  if (!showOverwriteConfirmation(values)) {
    await applyPayload(values);
  }
}

function closeOutputDetails() {
  const dialog = $("#outputDetailsDialog");
  if (dialog.open) dialog.close();
}

function prepareOutputDetailsDialog(opener = null, outputId = "") {
  const dialog = $("#outputDetailsDialog");
  state.outputDetails = {
    open: true,
    loading: true,
    outputId,
    data: null,
    selectedFields: [],
    error: "",
    opener: opener || document.activeElement,
    pendingValues: null,
  };
  $("#outputDetailsLoading").hidden = false;
  $("#outputDetailsBody").hidden = true;
  $("#outputDetailsError").hidden = true;
  $("#outputDetailsConfirmation").hidden = true;
  $("#outputDetailsCopyFullButton").disabled = true;
  $("#outputDetailsSendSelectedButton").disabled = true;
  $("#outputDetailsDuplicateButton").disabled = true;
  $("#outputDetailsPreviewMergedButton").disabled = true;
  $("#outputDetailsExactReplayButton").disabled = true;
  $("#outputDetailsNewSeedButton").disabled = true;
  $("#outputDetailsCurrentModelButton").disabled = true;
  $("#outputDetailsLiveRegion").textContent = "Loading image metadata.";
  if (!dialog.open) dialog.showModal();
  $("#outputDetailsCloseButton").focus();
  return dialog;
}

export async function openOutputDetailsData(data, { opener = null } = {}) {
  if (!data || typeof data !== "object") {
    notify("Unable to load image details.", "error");
    return;
  }
  prepareOutputDetailsDialog(opener, data.output_id || data.image?.relative_path || data.image?.name || "uploaded-image");
  state.outputDetails.data = data;
  state.outputDetails.loading = false;
  renderDetails();
  announce(`Image details loaded from ${(data.metadata_source || "image").replaceAll("_", " ")}.`);
}

export async function openOutputDetails(item, { opener = null } = {}) {
  const outputId = item?.output_id;
  if (!outputId) {
    notify("This output does not have a safe metadata identifier.", "error");
    return;
  }
  prepareOutputDetailsDialog(opener, outputId);
  try {
    state.outputDetails.data = await api.outputDetails(outputId, item?.details_url || "");
    state.outputDetails.loading = false;
    renderDetails();
    announce(`Image details loaded from ${state.outputDetails.data.metadata_source.replaceAll("_", " ")}.`);
  } catch (error) {
    state.outputDetails.loading = false;
    renderError(error);
  }
}


function selectedFieldNames() {
  return [...document.querySelectorAll("[data-replay-field]:checked")]
    .map((item) => item.dataset.replayField);
}

function replaySpecification({ mode = "exact", seedMode = "original", modelMode = "original", promptMode = "" } = {}) {
  const selectedPromptMode = promptMode || $("#outputDetailsReplayPromptMode")?.value || "raw_original";
  return {
    output_id: state.outputDetails.outputId,
    mode,
    selected_fields: mode === "selected" ? selectedFieldNames() : [],
    current_values: collectValues(),
    seed_mode: seedMode,
    model_mode: modelMode,
    prompt_mode: selectedPromptMode,
    remap: {},
  };
}

function announceReplay(message, kind = "info") {
  const live = $("#replayPreflightLiveRegion");
  live.textContent = message;
  live.className = `output-details-live ${kind}`;
}

function summaryCard(label, value, { multiline = false } = {}) {
  const card = document.createElement("article");
  card.className = `replay-summary-card${multiline ? " is-multiline" : ""}`;
  const heading = document.createElement("strong");
  heading.textContent = label;
  const body = multiline ? document.createElement("pre") : document.createElement("span");
  body.textContent = displayValue(value, value !== undefined && value !== null);
  card.append(heading, body);
  return card;
}

function remapOptions(kind) {
  if (kind === "model") return state.models.map((item) => [item.path, item.name || item.path]);
  if (kind === "vae") return state.vaes.map((item) => [item.path, item.name || item.path]);
  if (kind === "sampler") return state.samplers.map((item) => [item.name || item.plugin_id, item.label || item.name || item.plugin_id]);
  if (kind === "scheduler") return state.schedulers.map((item) => [item.name || item.plugin_id, item.label || item.name || item.plugin_id]);
  return [];
}

function renderMissingAssets(items = []) {
  const section = $("#replayPreflightMissingSection");
  const list = $("#replayPreflightMissingAssets");
  list.replaceChildren();
  const unique = new Map();
  items.forEach((item) => unique.set(item.field, item));
  section.hidden = unique.size === 0;
  unique.forEach((item) => {
    const row = document.createElement("label");
    row.className = "replay-remap-row";
    const title = document.createElement("strong");
    title.textContent = `Replace ${item.kind}`;
    const reason = document.createElement("small");
    reason.textContent = item.reason || "The recorded asset is unavailable.";
    const select = document.createElement("select");
    select.dataset.replayRemapField = item.field;
    select.setAttribute("aria-label", `Replacement for ${item.kind}`);
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = `Choose a replacement ${item.kind}`;
    select.append(placeholder);
    remapOptions(item.kind).forEach(([value, label]) => {
      if (!value) return;
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      if (state.replayPreflight.specification?.remap?.[item.field] === value) option.selected = true;
      select.append(option);
    });
    row.append(title, reason, select);
    list.append(row);
  });
}

function renderMessages(sectionSelector, listSelector, messages = []) {
  const section = $(sectionSelector);
  const list = $(listSelector);
  list.replaceChildren();
  section.hidden = messages.length === 0;
  messages.forEach((message) => {
    const item = document.createElement("li");
    item.textContent = message;
    list.append(item);
  });
}

function renderFieldResults(results = []) {
  const list = $("#replayPreflightFieldResults");
  list.replaceChildren();
  const counts = {};
  results.forEach((result) => {
    counts[result.status] = (counts[result.status] || 0) + 1;
    const row = document.createElement("article");
    row.className = "replay-field-result";
    const field = document.createElement("strong");
    field.textContent = result.field;
    const badge = document.createElement("span");
    badge.className = `replay-status ${result.status}`;
    badge.textContent = REPLAY_STATUS_LABELS[result.status]
      || String(result.status || "unknown").replaceAll("_", " ");
    const comparison = document.createElement("div");
    comparison.className = "replay-field-comparison";
    const original = document.createElement("pre");
    original.dataset.label = "Original";
    original.textContent = displayValue(result.original, result.original !== undefined);
    const outgoing = document.createElement("pre");
    outgoing.dataset.label = "Outgoing";
    outgoing.textContent = displayValue(result.outgoing, result.outgoing !== undefined);
    comparison.append(original, outgoing);
    row.append(field, badge, comparison);
    if (result.reason) {
      const reason = document.createElement("small");
      reason.textContent = result.reason;
      row.append(reason);
    }
    list.append(row);
  });
  const countText = Object.entries(counts)
    .map(([status, count]) => `${count} ${status.replaceAll("_", " ")}`)
    .join(" · ");
  $("#replayPreflightCounts").textContent = countText || "No field results";
}

function renderReplayPreflight() {
  const data = state.replayPreflight.data;
  const summary = data.summary || {};
  $("#replayPreflightTitle").textContent = summary.replay_label || "Replay Preflight";
  const quality = $("#replayPreflightQuality");
  quality.className = `replay-quality-banner ${data.completeness?.quality || "best_available"}`;
  quality.textContent = data.completeness?.quality === "exact_request"
    ? "Exact Request Replay: the complete Phase 10B manifest contract is present."
    : "Best Available Replay: one or more exactness fields were not recorded by this older output.";

  const grid = $("#replayPreflightSummary");
  grid.replaceChildren(
    summaryCard("Positive prompt", summary.prompt, { multiline: true }),
    summaryCard("Negative prompt", summary.negative_prompt, { multiline: true }),
    summaryCard("Seed", summary.seed),
    summaryCard("Model", summary.model_path),
    summaryCard("VAE", summary.vae_path || "Automatic / checkpoint embedded"),
    summaryCard("VAE mode", data.image?.vae?.mode || "checkpoint_embedded_auto"),
    summaryCard("Sampler", summary.sampler_name),
    summaryCard("Scheduler", summary.scheduler_name),
    summaryCard("Dimensions", `${summary.width ?? "?"} × ${summary.height ?? "?"}`),
    summaryCard("Steps / CFG", `${summary.steps ?? "?"} / ${summary.cfg_scale ?? "?"}`),
    summaryCard("Batch", `${summary.batch_size ?? 1} × ${summary.batch_count ?? 1}`),
    summaryCard("Advanced settings", summary.advanced_setting_count ?? 0),
    summaryCard("Preserved backend-only", summary.preserved_setting_count ?? 0),
    summaryCard("Unsupported", summary.unsupported_setting_count ?? 0),
  );

  renderMissingAssets(data.missing_assets || []);
  renderMessages("#replayPreflightWarningsSection", "#replayPreflightWarnings", data.warnings || []);
  renderMessages("#replayPreflightErrorsSection", "#replayPreflightErrors", data.errors || []);
  renderFieldResults(data.field_results || []);
  $("#replayPreflightAdvancedJson").textContent = JSON.stringify({
    canonical_request: data.request,
    preserved_backend_only: data.preserved_settings,
    unsupported: data.unsupported_settings,
    completeness: data.completeness,
  }, null, 2);
  $("#replayPreflightLoading").hidden = true;
  $("#replayPreflightError").hidden = true;
  $("#replayPreflightBody").hidden = false;
  $("#replayPreflightQueueButton").disabled = !data.valid || !data.preflight_token;
  announceReplay(data.valid ? "Replay preflight passed. Review the request before queueing." : "Replay preflight found blocking issues.", data.valid ? "info" : "error");
}

async function runReplayPreflight(specification) {
  state.replayPreflight.loading = true;
  state.replayPreflight.error = "";
  state.replayPreflight.specification = specification;
  $("#replayPreflightLoading").hidden = false;
  $("#replayPreflightBody").hidden = true;
  $("#replayPreflightError").hidden = true;
  $("#replayPreflightQueueButton").disabled = true;
  announceReplay("Validating replay request.");
  try {
    state.replayPreflight.data = await api.replayPreflight(specification);
    state.replayPreflight.loading = false;
    renderReplayPreflight();
  } catch (error) {
    state.replayPreflight.loading = false;
    state.replayPreflight.error = error.message || String(error);
    $("#replayPreflightLoading").hidden = true;
    $("#replayPreflightBody").hidden = true;
    $("#replayPreflightError").hidden = false;
    $("#replayPreflightError").textContent = state.replayPreflight.error;
    announceReplay(state.replayPreflight.error, "error");
  }
}

async function openReplayPreflight(specification) {
  state.replayPreflight = {
    open: true,
    loading: true,
    data: null,
    specification,
    error: "",
    returnToDetails: true,
  };
  const detailsDialog = $("#outputDetailsDialog");
  if (detailsDialog.open) detailsDialog.close();
  const dialog = $("#replayPreflightDialog");
  if (!dialog.open) dialog.showModal();
  $("#replayPreflightCloseButton").focus();
  await runReplayPreflight(specification);
}

function closeReplayPreflight({ returnToDetails = false } = {}) {
  const dialog = $("#replayPreflightDialog");
  state.replayPreflight.returnToDetails = returnToDetails;
  if (dialog.open) dialog.close();
}

async function submitReplay() {
  const data = state.replayPreflight.data;
  if (!data?.valid || !data.preflight_token) return;
  const button = $("#replayPreflightQueueButton");
  button.disabled = true;
  announceReplay("Submitting the validated replay to the normal generation queue.");
  try {
    const response = await api.submitReplay(data.preflight_token);
    await onJobQueued(response.job, { message: "Replay added to the normal generation queue." });
    state.replayPreflight.returnToDetails = false;
    closeReplayPreflight({ returnToDetails: false });
  } catch (error) {
    button.disabled = false;
    announceReplay(`Replay submission failed: ${error.message}`, "error");
  }
}

export function bindOutputDetails({ collect = () => ({}), apply = async () => ({}), onJobQueued: queued = async () => {} } = {}) {
  collectValues = collect;
  applyValues = apply;
  onJobQueued = queued;
  const dialog = $("#outputDetailsDialog");
  const replayDialog = $("#replayPreflightDialog");

  $("#outputDetailsCloseButton").addEventListener("click", closeOutputDetails);
  $("#outputDetailsCloseFooterButton").addEventListener("click", closeOutputDetails);
  $("#outputDetailsCopyFullButton").addEventListener("click", () => {
    if (state.outputDetails.data) copyText(state.outputDetails.data, "Full metadata");
  });
  $("#outputDetailsSendSelectedButton").addEventListener("click", () => {
    const fields = [...document.querySelectorAll("[data-replay-field]:checked")]
      .map((item) => item.dataset.replayField);
    requestApply(selectedReplayPayload(fields));
  });
  $("#outputDetailsDuplicateButton").addEventListener("click", async () => {
    const payload = selectedReplayPayload(Object.keys(state.outputDetails.data?.replay || {}));
    if (state.settings.metadata_import_auto_apply_full_run) {
      await applyPayload(payload);
      return;
    }
    requestApply(payload);
  });
  $("#outputDetailsExactReplayButton").addEventListener("click", () => {
    openReplayPreflight(replaySpecification());
  });
  $("#outputDetailsNewSeedButton").addEventListener("click", () => {
    openReplayPreflight(replaySpecification({ seedMode: "random" }));
  });
  $("#outputDetailsCurrentModelButton").addEventListener("click", () => {
    openReplayPreflight(replaySpecification({ modelMode: "current" }));
  });
  $("#outputDetailsPreviewMergedButton").addEventListener("click", () => {
    openReplayPreflight(replaySpecification({ mode: "selected" }));
  });
  $("#outputDetailsConfirmationApplyButton").addEventListener("click", async () => {
    const pending = state.outputDetails.pendingValues || {};
    state.outputDetails.pendingValues = null;
    $("#outputDetailsConfirmation").hidden = true;
    await applyPayload(pending);
  });
  $("#outputDetailsConfirmationCancelButton").addEventListener("click", () => {
    state.outputDetails.pendingValues = null;
    $("#outputDetailsConfirmation").hidden = true;
    announce("Form changes cancelled.");
    $("#outputDetailsSendSelectedButton").focus();
  });

  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeOutputDetails();
  });
  dialog.addEventListener("close", () => {
    const restore = state.outputDetails.opener;
    state.outputDetails.open = false;
    state.outputDetails.pendingValues = null;
    if (!state.replayPreflight.open && restore?.isConnected) restore.focus();
  });

  $("#replayPreflightCloseButton").addEventListener("click", () => closeReplayPreflight());
  $("#replayPreflightCancelButton").addEventListener("click", () => closeReplayPreflight());
  $("#replayPreflightReturnButton").addEventListener("click", () => closeReplayPreflight({ returnToDetails: true }));
  $("#replayPreflightQueueButton").addEventListener("click", submitReplay);
  $("#replayPreflightRerunButton").addEventListener("click", () => {
    const remap = {};
    document.querySelectorAll("[data-replay-remap-field]").forEach((select) => {
      if (select.value) remap[select.dataset.replayRemapField] = select.value;
    });
    runReplayPreflight({ ...state.replayPreflight.specification, remap });
  });

  replayDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeReplayPreflight();
  });
  replayDialog.addEventListener("close", () => {
    const returnToDetails = state.replayPreflight.returnToDetails;
    state.replayPreflight.open = false;
    state.replayPreflight.returnToDetails = false;
    if (returnToDetails && state.outputDetails.data) {
      state.outputDetails.open = true;
      if (!dialog.open) dialog.showModal();
      $("#outputDetailsExactReplayButton").focus();
      return;
    }
    const restore = state.outputDetails.opener;
    if (restore?.isConnected) restore.focus();
  });
}
