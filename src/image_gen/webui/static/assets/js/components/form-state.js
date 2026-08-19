import { $, clampedNumberValue, numberValue } from "../utils.js";
import { readAdvancedValues } from "./advanced-editor.js";
import { applyCfgLabValues, readCfgLabValues } from "../features/cfg-lab.js?v=0.1.47-lightning-recommendation";
import { normalizeHiresSizeMode, planHiresDimensions } from "./hires-dimensions.js";
import { applyParameterRanges, cfgEffectiveRangeLocks, collectParameterRanges } from "../features/parameter-ranges.js?v=qol2";
import { collectSeedValues, syncSeedControlsFromGenerationValues } from "../features/seed-controls.js?v=qol-seed-ui5";

function normalizedHiresSizeMode(value, enabled = true) {
  return normalizeHiresSizeMode(value, enabled);
}

function jsonValue(selector, fallback = {}) {
  const raw = String($(selector)?.value || "").trim();
  if (!raw) return fallback;
  try {
    const value = JSON.parse(raw);
    return value && typeof value === "object" && !Array.isArray(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

export function collectGenerationValues(selectionMetadata = {}) {
  const seedValues = collectSeedValues();
  const cfgLab = readCfgLabValues();
  const advancedSampler = readAdvancedValues($("#samplerAdvancedContent"));
  const parameterRanges = collectParameterRanges();
  const cfgLocks = cfgEffectiveRangeLocks();
  const samplerKwargs = { ...advancedSampler, ...(cfgLab.sampler_kwargs || {}) };
  if (cfgLocks.minimum !== null) samplerKwargs.cfg_effective_min_lock = cfgLocks.minimum;
  if (cfgLocks.maximum !== null) samplerKwargs.cfg_effective_max_lock = cfgLocks.maximum;
  const hiresEnabled = Boolean($("#hiresEnabled")?.checked);
  const hiresPlan = planHiresDimensions({
    baseWidth: numberValue($("#width"), 640),
    baseHeight: numberValue($("#height"), 960),
    mode: normalizedHiresSizeMode($("#hiresSizeMode")?.value, hiresEnabled),
    scale: numberValue($("#hiresScale"), 1.5),
    targetWidth: numberValue($("#hiresWidth"), 0),
    targetHeight: numberValue($("#hiresHeight"), 0),
    enabled: hiresEnabled,
  });
  return {
    positive_prompt: $("#positivePrompt").value,
    negative_prompt: $("#negativePrompt").value,
    model_path: $("#modelPath").value,
    vae_path: $("#vaePath")?.value || null,
    advanced_models_enabled: Boolean($("#advancedModelsEnabled")?.checked),
    advanced_model_family: $("#advancedModelFamily")?.value || "",
    advanced_model_components: Object.fromEntries(
      Array.from(document.querySelectorAll("[data-advanced-component-role]")).map((node) => [node.dataset.advancedComponentRole, node.value]),
    ),
    advanced_model_allow_digital_components: Boolean($("#advancedModelAllowDigitalComponents")?.checked ?? true),
    advanced_model_t5_device: $("#advancedModelT5Device")?.value || "cpu",
    sd2_runtime_profile_override: $("#sd2RuntimeProfileOverride")?.value || null,
    sd2_dedicated_generation: Boolean($("#sd2DedicatedGeneration")?.checked),
    width: numberValue($("#width"), 640),
    height: numberValue($("#height"), 960),
    steps: numberValue($("#steps"), 20),
    cfg_scale: numberValue($("#cfgScale"), 7),
    sdxl_enforce_recommended_steps: Boolean($("#sdxlEnforceRecommendedSteps")?.checked),
    sdxl_enforce_recommended_cfg: Boolean($("#sdxlEnforceRecommendedCfg")?.checked),
    model_enforce_recommended_steps: Boolean($("#sdxlEnforceRecommendedSteps")?.checked),
    model_enforce_recommended_cfg: Boolean($("#sdxlEnforceRecommendedCfg")?.checked),
    cfg_rescale: cfgLab.cfg_rescale,
    seed: seedValues.seed,
    batch_size: numberValue($("#batchSize"), 1),
    batch_count: numberValue($("#batchCount"), 1),
    batch_seed_mode: seedValues.batch_seed_mode,
    seed_range_min: seedValues.seed_range_min,
    seed_range_max: seedValues.seed_range_max,
    seed_no_duplicates: seedValues.seed_no_duplicates,
    _random_ranges: parameterRanges,
    sampler_name: $("#samplerName").value,
    scheduler_name: $("#schedulerName").value,
    sampler_kwargs: samplerKwargs,
    scheduler_kwargs: readAdvancedValues($("#schedulerAdvancedContent")),
    prompt_parser_name: $("#promptParserName")?.value || "legacy",
    prompt_parser_kwargs: jsonValue("#promptParserKwargs"),
    prompt_shortcut_profile_name: $("#promptShortcutProfileName")?.value || "legacy_default",
    prompt_shortcut_profile_snapshot: jsonValue("#promptShortcutProfileSnapshot"),
    prompt_parser_preset_name: $("#promptParserPresetName")?.value || "",
    prompt_shadow_compare: Boolean($("#promptShadowCompare")?.checked),
    base_prompt_parser_name: $("#promptParserName")?.value || "legacy",
    base_shortcut_profile_name: $("#promptShortcutProfileName")?.value || "legacy_default",
    hires_prompt_parser_mode: $("#hiresPromptParserMode")?.value || "same_as_base",
    hires_prompt_parser_name: $("#hiresPromptParserName")?.value || $("#promptParserName")?.value || "legacy",
    hires_prompt_parser_kwargs: jsonValue("#hiresPromptParserKwargs"),
    hires_shortcut_profile_mode: $("#hiresShortcutProfileMode")?.value || "same_as_base",
    hires_shortcut_profile_name: $("#hiresShortcutProfileName")?.value || $("#promptShortcutProfileName")?.value || "legacy_default",
    hires_shortcut_profile_snapshot: jsonValue("#hiresShortcutProfileSnapshot"),
    hires_enabled: hiresEnabled,
    hires_configuration_mode: $("#hiresConfigurationMode")?.value || "custom",
    hires_auto_resolution_record: jsonValue("#hiresAutoResolutionRecord"),
    hires_lifecycle_state: jsonValue("#hiresLifecycleState"),
    hires_positive_prompt: $("#hiresPositivePrompt")?.value || "",
    hires_negative_prompt: $("#hiresNegativePrompt")?.value || "",
    hires_size_mode: hiresPlan.mode,
    hires_scale: hiresPlan.requested_scale,
    hires_width: hiresPlan.requested_width,
    hires_height: hiresPlan.requested_height,
    hires_axis_scale_width: hiresPlan.axis_scale_width,
    hires_axis_scale_height: hiresPlan.axis_scale_height,
    hires_uniform_scale: hiresPlan.uniform_scale,
    hires_aspect_ratio_changed: hiresPlan.aspect_ratio_changed,
    hires_dimension_plan_version: hiresPlan.contract_version || "",
    hires_dimension_plan: hiresPlan,
    hires_steps: numberValue($("#hiresSteps"), 20),
    hires_denoising_strength: numberValue($("#hiresDenoisingStrength"), 0.4),
    hires_step_policy: $("#hiresStepPolicy")?.value || "a1111_fixed_steps_v1",
    hires_sampler_name: $("#hiresSamplerName")?.value || "",
    hires_scheduler_name: $("#hiresSchedulerName")?.value || "",
    hires_cfg_scale: numberValue($("#hiresCfgScale"), null),
    hires_cfg_rescale: clampedNumberValue($("#hiresCfgRescale"), 0, 1, null),
    hires_strategy: $("#hiresStrategy")?.value || "pixel_neural",
    hires_upscaler: $("#hiresUpscaler")?.value || "",
    hires_upscaler_id: $("#hiresUpscaler")?.value || "",
    hires_tile_size: numberValue($("#hiresTileSize"), 0),
    hires_tile_overlap: numberValue($("#hiresTileOverlap"), 16),
    hires_tile_batch_size: numberValue($("#hiresTileBatchSize"), 1),
    hires_exact_resize_filter: "bicubic",
    hires_final_size_correction_filter: $("#hiresFinalSizeCorrectionFilter")?.value || "auto",
    hires_aspect_policy: $("#hiresAspectPolicy")?.value || "stretch",
    hires_padding_mode: $("#hiresPaddingMode")?.value || "reflect",
    hires_correction_fingerprint_enabled: Boolean($("#hiresCorrectionFingerprintDiagnostics")?.checked),
    hires_save_upscaled_pre_denoise: Boolean($("#hiresSaveUpscaledPreDenoise")?.checked),
    hires_save_vae_roundtrip: Boolean($("#hiresSaveVaeRoundtrip")?.checked),
    hires_save_lowres: Boolean($("#hiresSaveLowres")?.checked),
    outpaint_prototype_enabled: Boolean($("#outpaintPrototypeEnabled")?.checked),
    outpaint_source_image: $("#outpaintSourceImage")?.value || "",
    outpaint_anchor: $("#outpaintAnchor")?.value || "center",
    outpaint_source_x: -1,
    outpaint_source_y: -1,
    outpaint_feather_px: numberValue($("#outpaintFeatherPx"), 24),
    outpaint_context_seed_mode: $("#outpaintContextSeedMode")?.value || "edge_pad_v1",
    outpaint_denoising_strength: numberValue($("#outpaintDenoisingStrength"), 0.70),
    outpaint_latent_strategy: $("#outpaintLatentStrategy")?.value || "noise_only_new_regions_v1",
    outpaint_prompt_mode: $("#outpaintPromptMode")?.value || "overlay_only_v1",
    outpaint_overlay_positive_prompt: $("#outpaintOverlayPositivePrompt")?.value || "",
    outpaint_overlay_negative_prompt: $("#outpaintOverlayNegativePrompt")?.value || "",
    outpaint_diagnostic_artifacts: Boolean($("#outpaintDiagnosticArtifacts")?.checked),
    outpaint_shape_expansion_enabled: Boolean($("#outpaintShapeExpansionEnabled")?.checked),
    outpaint_shape_target_mode: $("#outpaintShapeTargetMode")?.value || "square",
    outpaint_shape_target_width: numberValue($("#outpaintShapeTargetWidth"), 0),
    outpaint_shape_target_height: numberValue($("#outpaintShapeTargetHeight"), 0),
    outpaint_shape_base_width: 0,
    outpaint_shape_base_height: 0,
    outpaint_shape_anchor: $("#outpaintShapeAnchor")?.value || "center",
    outpaint_shape_context_seed_mode: $("#outpaintShapeContextSeedMode")?.value || "edge_pad_v1",
    outpaint_shape_source_handoff: $("#outpaintShapeSourceHandoff")?.value || "auto",
    outpaint_shape_prompt_mode: $("#outpaintShapePromptMode")?.value || "overlay_only_v1",
    outpaint_shape_overlay_positive_prompt: $("#outpaintShapeOverlayPositivePrompt")?.value || "",
    outpaint_shape_overlay_negative_prompt: $("#outpaintShapeOverlayNegativePrompt")?.value || "",
    outpaint_shape_denoising_strength: numberValue($("#outpaintShapeDenoisingStrength"), 0.40),
    outpaint_shape_save_base: Boolean($("#outpaintShapeSaveBase")?.checked),
    output_dir: $("#outputDir").value,
    output_prefix: $("#outputPrefix").value || "{index:05d}-{seed}",
    save_images: true,
    save_txt: Boolean($("#saveTxt")?.checked),
    save_json: Boolean($("#saveJson")?.checked),
    save_diagnostics_json: Boolean($("#saveDiagnosticsJson")?.checked),
    ...selectionMetadata,
  };
}

export function applyGenerationValues(values = {}) {
  const normalizedValues = {
    ...values,
    hires_upscaler: values.hires_upscaler_id || values.hires_upscaler,
    hires_size_mode: normalizedHiresSizeMode(
      values.hires_size_mode,
      Boolean(values.hires_enabled),
    ),
    hires_final_size_correction_filter: values.hires_final_size_correction_filter || values.hires_exact_resize_filter || "auto",
    hires_aspect_policy: values.hires_aspect_policy || "stretch",
    hires_padding_mode: values.hires_padding_mode || "reflect",
    hires_correction_fingerprint_enabled: Boolean(values.hires_correction_fingerprint_enabled),
  };
  const mapping = {
    positive_prompt: "#positivePrompt",
    negative_prompt: "#negativePrompt",
    model_path: "#modelPath",
    vae_path: "#vaePath",
    sd2_runtime_profile_override: "#sd2RuntimeProfileOverride",
    width: "#width",
    height: "#height",
    steps: "#steps",
    cfg_scale: "#cfgScale",
    seed: "#seed",
    batch_size: "#batchSize",
    batch_count: "#batchCount",
    sampler_name: "#samplerName",
    scheduler_name: "#schedulerName",
    prompt_parser_name: "#promptParserName",
    prompt_shortcut_profile_name: "#promptShortcutProfileName",
    prompt_parser_preset_name: "#promptParserPresetName",
    base_prompt_parser_name: "#promptParserName",
    base_shortcut_profile_name: "#promptShortcutProfileName",
    hires_prompt_parser_mode: "#hiresPromptParserMode",
    hires_prompt_parser_name: "#hiresPromptParserName",
    hires_shortcut_profile_mode: "#hiresShortcutProfileMode",
    hires_shortcut_profile_name: "#hiresShortcutProfileName",
    hires_positive_prompt: "#hiresPositivePrompt",
    hires_negative_prompt: "#hiresNegativePrompt",
    hires_configuration_mode: "#hiresConfigurationMode",
    hires_lifecycle_state: "#hiresLifecycleState",
    hires_size_mode: "#hiresSizeMode",
    hires_scale: "#hiresScale",
    hires_width: "#hiresWidth",
    hires_height: "#hiresHeight",
    hires_steps: "#hiresSteps",
    hires_denoising_strength: "#hiresDenoisingStrength",
    hires_step_policy: "#hiresStepPolicy",
    hires_sampler_name: "#hiresSamplerName",
    hires_scheduler_name: "#hiresSchedulerName",
    hires_cfg_scale: "#hiresCfgScale",
    hires_cfg_rescale: "#hiresCfgRescale",
    hires_strategy: "#hiresStrategy",
    hires_upscaler: "#hiresUpscaler",
    hires_tile_size: "#hiresTileSize",
    hires_tile_overlap: "#hiresTileOverlap",
    hires_tile_batch_size: "#hiresTileBatchSize",
    hires_final_size_correction_filter: "#hiresFinalSizeCorrectionFilter",
    hires_aspect_policy: "#hiresAspectPolicy",
    hires_padding_mode: "#hiresPaddingMode",
    outpaint_source_image: "#outpaintSourceImage",
    outpaint_anchor: "#outpaintAnchor",
    outpaint_feather_px: "#outpaintFeatherPx",
    outpaint_context_seed_mode: "#outpaintContextSeedMode",
    outpaint_denoising_strength: "#outpaintDenoisingStrength",
    outpaint_latent_strategy: "#outpaintLatentStrategy",
    outpaint_prompt_mode: "#outpaintPromptMode",
    outpaint_overlay_positive_prompt: "#outpaintOverlayPositivePrompt",
    outpaint_overlay_negative_prompt: "#outpaintOverlayNegativePrompt",
    outpaint_shape_target_mode: "#outpaintShapeTargetMode",
    outpaint_shape_target_width: "#outpaintShapeTargetWidth",
    outpaint_shape_target_height: "#outpaintShapeTargetHeight",
    outpaint_shape_anchor: "#outpaintShapeAnchor",
    outpaint_shape_context_seed_mode: "#outpaintShapeContextSeedMode",
    outpaint_shape_source_handoff: "#outpaintShapeSourceHandoff",
    outpaint_shape_prompt_mode: "#outpaintShapePromptMode",
    outpaint_shape_overlay_positive_prompt: "#outpaintShapeOverlayPositivePrompt",
    outpaint_shape_overlay_negative_prompt: "#outpaintShapeOverlayNegativePrompt",
    outpaint_shape_denoising_strength: "#outpaintShapeDenoisingStrength",
    output_dir: "#outputDir",
    output_prefix: "#outputPrefix",
  };

  Object.entries(mapping).forEach(([name, selector]) => {
    if (!(name in normalizedValues)) return;
    const input = $(selector);
    if (!input) return;
    if (name === "hires_upscaler" && (!normalizedValues.hires_enabled || !String(normalizedValues[name] || "").trim())) return;
    input.value = normalizedValues[name] ?? "";
    if (name === "output_prefix") {
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  if ($("#sdxlEnforceRecommendedSteps")) {
    if ("model_enforce_recommended_steps" in values) {
      $("#sdxlEnforceRecommendedSteps").checked = Boolean(values.model_enforce_recommended_steps);
    } else if ("sdxl_enforce_recommended_steps" in values) {
      $("#sdxlEnforceRecommendedSteps").checked = Boolean(values.sdxl_enforce_recommended_steps);
    }
  }
  if ($("#sdxlEnforceRecommendedCfg")) {
    if ("model_enforce_recommended_cfg" in values) {
      $("#sdxlEnforceRecommendedCfg").checked = Boolean(values.model_enforce_recommended_cfg);
    } else if ("sdxl_enforce_recommended_cfg" in values) {
      $("#sdxlEnforceRecommendedCfg").checked = Boolean(values.sdxl_enforce_recommended_cfg);
    }
  }
  if ($("#outpaintPrototypeEnabled")) {
    $("#outpaintPrototypeEnabled").checked = Boolean(values.outpaint_prototype_enabled);
    $("#outpaintPrototypeEnabled").dispatchEvent(new Event("change", { bubbles: true }));
  }
  if ($("#outpaintDiagnosticArtifacts")) $("#outpaintDiagnosticArtifacts").checked = Boolean(values.outpaint_diagnostic_artifacts);
  if ($("#outpaintShapeExpansionEnabled")) {
    $("#outpaintShapeExpansionEnabled").checked = Boolean(values.outpaint_shape_expansion_enabled);
    if (values.outpaint_shape_expansion_enabled) {
      const replayBaseWidth = Number(values.outpaint_shape_base_width || 0);
      const replayBaseHeight = Number(values.outpaint_shape_base_height || 0);
      if (replayBaseWidth > 0 && $("#width")) $("#width").value = String(replayBaseWidth);
      if (replayBaseHeight > 0 && $("#height")) $("#height").value = String(replayBaseHeight);
    }
    $("#outpaintShapeExpansionEnabled").dispatchEvent(new Event("change", { bubbles: true }));
  }
  if ($("#outpaintShapeSaveBase")) $("#outpaintShapeSaveBase").checked = Boolean(values.outpaint_shape_save_base);
  if ($("#hiresEnabled")) {
    $("#hiresEnabled").checked = Boolean(values.hires_enabled);
    $("#hiresEnabled").dispatchEvent(new Event("change", { bubbles: true }));
  }
  if ($("#hiresAutoResolutionRecord")) $("#hiresAutoResolutionRecord").value = JSON.stringify(values.hires_auto_resolution_record || {});
  if ($("#hiresLifecycleState")) $("#hiresLifecycleState").value = JSON.stringify(values.hires_lifecycle_state || {});
  if ($("#hiresSaveLowres")) $("#hiresSaveLowres").checked = Boolean(values.hires_save_lowres);
  if ($("#hiresCorrectionFingerprintDiagnostics")) $("#hiresCorrectionFingerprintDiagnostics").checked = Boolean(values.hires_correction_fingerprint_enabled);
  if ($("#hiresSaveUpscaledPreDenoise")) $("#hiresSaveUpscaledPreDenoise").checked = Boolean(values.hires_save_upscaled_pre_denoise);
  if ($("#hiresSaveVaeRoundtrip")) $("#hiresSaveVaeRoundtrip").checked = Boolean(values.hires_save_vae_roundtrip);
  if ($("#advancedModelsEnabled")) $("#advancedModelsEnabled").checked = Boolean(values.advanced_models_enabled);
  if ($("#advancedModelAllowDigitalComponents")) {
    $("#advancedModelAllowDigitalComponents").checked = values.advanced_model_allow_digital_components !== false;
  }
  if ($("#advancedModelT5Device")) $("#advancedModelT5Device").value = String(values.advanced_model_t5_device || "cpu");
  if ($("#sd2DedicatedGeneration")) $("#sd2DedicatedGeneration").checked = Boolean(values.sd2_dedicated_generation);
  if ($("#saveTxt")) $("#saveTxt").checked = Boolean(values.save_txt);
  if ($("#saveJson")) $("#saveJson").checked = values.save_json !== false;
  if ($("#saveDiagnosticsJson")) $("#saveDiagnosticsJson").checked = Boolean(values.save_diagnostics_json);
  syncSeedControlsFromGenerationValues(values);
  applyParameterRanges(values._random_ranges || {});
  if ($("#promptParserKwargs")) $("#promptParserKwargs").value = JSON.stringify(values.prompt_parser_kwargs || {});
  if ($("#promptShadowCompare")) $("#promptShadowCompare").checked = Boolean(values.prompt_shadow_compare);
  if ($("#promptShortcutProfileSnapshot")) $("#promptShortcutProfileSnapshot").value = JSON.stringify(values.prompt_shortcut_profile_snapshot || {});
  if ($("#hiresPromptParserKwargs")) $("#hiresPromptParserKwargs").value = JSON.stringify(values.hires_prompt_parser_kwargs || values.prompt_parser_kwargs || {});
  if ($("#hiresShortcutProfileSnapshot")) $("#hiresShortcutProfileSnapshot").value = JSON.stringify(values.hires_shortcut_profile_snapshot || {});
  applyCfgLabValues(values);
  window.dispatchEvent(new CustomEvent("image-gen-generation-values-applied", { detail: { values } }));
}
