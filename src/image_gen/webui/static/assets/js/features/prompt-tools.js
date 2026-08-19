import { state } from "../state.js";
import { $, clampedNumberValue } from "../utils.js";
import { configurePromptRuntime } from "./prompt/runtime.js";
import {
  currentParserId,
  profileById,
  safeParseJson,
  setParserKwargs,
} from "./prompt/shared.js";
import { renderBaseParserSettings } from "./prompt/parser-settings.js";
import {
  bindParserProfiles,
  populateParsers,
  populatePresets,
  populateProfiles,
} from "./prompt/parser-profiles.js";
import { bindPromptFocus, renderPalette } from "./prompt/symbol-palette.js";
import {
  bindHiresPrompt,
  initializeHiresPrompt,
  normalizedHiresSizeMode,
} from "./prompt/hires-prompt.js";
import { bindRegionTools } from "./prompt/region-tools.js";
import { bindPromptPreflight, preflightCurrentPrompt } from "./prompt/preflight.js";
import { bindProfileEditor } from "./prompt/profile-editor.js";

export function initializePromptTools(current = {}) {
  populateParsers(current.prompt_parser_name || current.base_prompt_parser_name || "legacy");
  populateProfiles(current.prompt_shortcut_profile_name || current.base_shortcut_profile_name || "");
  populatePresets(current.prompt_parser_preset_name || "");
  setParserKwargs(current.prompt_parser_kwargs || {});
  const embedded = current.prompt_shortcut_profile_snapshot || {};
  if (embedded.profile_id) {
    const existing = profileById(embedded.profile_id);
    if (!existing) state.promptShortcutProfiles.push({ ...embedded, valid: true, palettes: {} });
    populateProfiles(embedded.profile_id);
    $("#promptShortcutProfileSnapshot").value = JSON.stringify(embedded);
  }
  if ($("#promptShadowCompare")) $("#promptShadowCompare").checked = Boolean(current.prompt_shadow_compare);
  initializeHiresPrompt(current);
  renderBaseParserSettings();
  renderPalette();
}

export { preflightCurrentPrompt };

export function bindPromptTools(options = {}) {
  configurePromptRuntime(options);
  bindPromptPreflight();
  bindPromptFocus();
  bindRegionTools();
  bindParserProfiles();
  bindHiresPrompt();
  bindProfileEditor();
}

export function refreshPromptConfigurationCatalogs(payload = {}) {
  if (payload.prompt_parsers) state.promptParsers = payload.prompt_parsers;
  if (payload.prompt_shortcut_profiles) state.promptShortcutProfiles = payload.prompt_shortcut_profiles;
  if (payload.prompt_parser_presets) state.promptParserPresets = payload.prompt_parser_presets;
  initializePromptTools({
    prompt_parser_name: $("#promptParserName")?.value || "legacy",
    prompt_shortcut_profile_name: $("#promptShortcutProfileName")?.value || "legacy_default",
    prompt_parser_preset_name: $("#promptParserPresetName")?.value || "",
    prompt_shadow_compare: Boolean($("#promptShadowCompare")?.checked),
    prompt_parser_kwargs: safeParseJson($("#promptParserKwargs")?.value, {}),
    prompt_shortcut_profile_snapshot: safeParseJson($("#promptShortcutProfileSnapshot")?.value, {}),
    hires_prompt_parser_mode: $("#hiresPromptParserMode")?.value || "same_as_base",
    hires_prompt_parser_name: $("#hiresPromptParserName")?.value || currentParserId(),
    hires_prompt_parser_kwargs: safeParseJson($("#hiresPromptParserKwargs")?.value, {}),
    hires_shortcut_profile_mode: $("#hiresShortcutProfileMode")?.value || "same_as_base",
    hires_shortcut_profile_name: $("#hiresShortcutProfileName")?.value || $("#promptShortcutProfileName")?.value || "legacy_default",
    hires_shortcut_profile_snapshot: safeParseJson($("#hiresShortcutProfileSnapshot")?.value, {}),
    hires_positive_prompt: $("#hiresPositivePrompt")?.value,
    hires_negative_prompt: $("#hiresNegativePrompt")?.value,
    hires_enabled: Boolean($("#hiresEnabled")?.checked),
    hires_size_mode: normalizedHiresSizeMode(
      $("#hiresSizeMode")?.value,
      Boolean($("#hiresEnabled")?.checked),
    ),
    hires_scale: Number($("#hiresScale")?.value || 1.5),
    hires_width: Number($("#hiresWidth")?.value || 0),
    hires_height: Number($("#hiresHeight")?.value || 0),
    hires_steps: Number($("#hiresSteps")?.value || 20),
    hires_denoising_strength: Number($("#hiresDenoisingStrength")?.value || 0.4),
    hires_step_policy: $("#hiresStepPolicy")?.value || "a1111_fixed_steps_v1",
    hires_sampler_name: $("#hiresSamplerName")?.value || "",
    hires_scheduler_name: $("#hiresSchedulerName")?.value || "",
    hires_cfg_scale: String($("#hiresCfgScale")?.value || "").trim() === "" ? null : Number($("#hiresCfgScale")?.value),
    hires_cfg_rescale: clampedNumberValue($("#hiresCfgRescale"), 0, 1, null),
    hires_upscaler: $("#hiresUpscaler")?.value || "",
    hires_save_lowres: Boolean($("#hiresSaveLowres")?.checked),
  });
}
