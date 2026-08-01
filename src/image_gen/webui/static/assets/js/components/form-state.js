import { $, numberValue } from "../utils.js";
import { readAdvancedValues } from "./advanced-editor.js";
import { applyCfgLabValues, readCfgLabValues } from "../features/cfg-lab.js";

function normalizedHiresSizeMode(value, enabled = true) {
  const mode = String(value || "scale_from_base").trim().toLowerCase();
  if (enabled && mode === "same_as_base") return "scale_from_base";
  return ["same_as_base", "scale_from_base", "explicit_dimensions"].includes(mode)
    ? mode
    : "scale_from_base";
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
  const seed = numberValue($("#seed"), null);
  const cfgLab = readCfgLabValues();
  const advancedSampler = readAdvancedValues($("#samplerAdvancedContent"));
  return {
    positive_prompt: $("#positivePrompt").value,
    negative_prompt: $("#negativePrompt").value,
    model_path: $("#modelPath").value,
    vae_path: $("#vaePath")?.value || null,
    width: numberValue($("#width"), 640),
    height: numberValue($("#height"), 960),
    steps: numberValue($("#steps"), 20),
    cfg_scale: numberValue($("#cfgScale"), 7),
    cfg_rescale: cfgLab.cfg_rescale,
    seed,
    batch_size: numberValue($("#batchSize"), 1),
    batch_count: numberValue($("#batchCount"), 1),
    sampler_name: $("#samplerName").value,
    scheduler_name: $("#schedulerName").value,
    sampler_kwargs: { ...advancedSampler, ...(cfgLab.sampler_kwargs || {}) },
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
    hires_enabled: Boolean($("#hiresEnabled")?.checked),
    hires_positive_prompt: $("#hiresPositivePrompt")?.value || "",
    hires_negative_prompt: $("#hiresNegativePrompt")?.value || "",
    hires_size_mode: normalizedHiresSizeMode(
      $("#hiresSizeMode")?.value,
      Boolean($("#hiresEnabled")?.checked),
    ),
    hires_scale: numberValue($("#hiresScale"), 1.5),
    hires_width: numberValue($("#hiresWidth"), 0),
    hires_height: numberValue($("#hiresHeight"), 0),
    hires_steps: numberValue($("#hiresSteps"), 20),
    hires_denoising_strength: numberValue($("#hiresDenoisingStrength"), 0.4),
    hires_step_policy: $("#hiresStepPolicy")?.value || "a1111_fixed_steps_v1",
    hires_sampler_name: $("#hiresSamplerName")?.value || "",
    hires_scheduler_name: $("#hiresSchedulerName")?.value || "",
    hires_cfg_scale: numberValue($("#hiresCfgScale"), null),
    hires_cfg_rescale: numberValue($("#hiresCfgRescale"), null),
    hires_upscaler: $("#hiresUpscaler")?.value || "latent_bicubic",
    hires_save_lowres: Boolean($("#hiresSaveLowres")?.checked),
    output_dir: $("#outputDir").value,
    output_prefix: $("#outputPrefix").value || "{index:05d}-{seed}",
    save_images: true,
    ...selectionMetadata,
  };
}

export function applyGenerationValues(values = {}) {
  const normalizedValues = {
    ...values,
    hires_size_mode: normalizedHiresSizeMode(
      values.hires_size_mode,
      Boolean(values.hires_enabled),
    ),
  };
  const mapping = {
    positive_prompt: "#positivePrompt",
    negative_prompt: "#negativePrompt",
    model_path: "#modelPath",
    vae_path: "#vaePath",
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
    hires_upscaler: "#hiresUpscaler",
    output_dir: "#outputDir",
    output_prefix: "#outputPrefix",
  };

  Object.entries(mapping).forEach(([name, selector]) => {
    if (!(name in normalizedValues)) return;
    const input = $(selector);
    if (!input) return;
    input.value = normalizedValues[name] ?? "";
    if (name === "output_prefix") {
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  if ($("#hiresEnabled")) {
    $("#hiresEnabled").checked = Boolean(values.hires_enabled);
    $("#hiresEnabled").dispatchEvent(new Event("change", { bubbles: true }));
  }
  if ($("#hiresSaveLowres")) $("#hiresSaveLowres").checked = values.hires_save_lowres !== false;
  if ($("#promptParserKwargs")) $("#promptParserKwargs").value = JSON.stringify(values.prompt_parser_kwargs || {});
  if ($("#promptShadowCompare")) $("#promptShadowCompare").checked = Boolean(values.prompt_shadow_compare);
  if ($("#promptShortcutProfileSnapshot")) $("#promptShortcutProfileSnapshot").value = JSON.stringify(values.prompt_shortcut_profile_snapshot || {});
  if ($("#hiresPromptParserKwargs")) $("#hiresPromptParserKwargs").value = JSON.stringify(values.hires_prompt_parser_kwargs || values.prompt_parser_kwargs || {});
  if ($("#hiresShortcutProfileSnapshot")) $("#hiresShortcutProfileSnapshot").value = JSON.stringify(values.hires_shortcut_profile_snapshot || {});
  applyCfgLabValues(values);
}
