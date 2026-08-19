import { $ } from "../../utils.js";
import { currentParserId, defaultProfileId, safeParseJson } from "./shared.js";

export function translationPayload(values = {}) {
  const parserName = values.prompt_parser_name || values.base_prompt_parser_name || currentParserId();
  const profileName = values.prompt_shortcut_profile_name || values.base_shortcut_profile_name || $("#promptShortcutProfileName")?.value || defaultProfileId(parserName);
  return {
    positive_prompt: values.positive_prompt ?? $("#positivePrompt")?.value ?? "",
    negative_prompt: values.negative_prompt ?? $("#negativePrompt")?.value ?? "",
    steps: values.steps ?? Number($("#steps")?.value || 20),
    width: Number(values.generation_width ?? values.width ?? $("#width")?.value ?? 512),
    height: Number(values.generation_height ?? values.height ?? $("#height")?.value ?? 512),
    generation_width: Number(values.generation_width ?? values.width ?? $("#width")?.value ?? 512),
    generation_height: Number(values.generation_height ?? values.height ?? $("#height")?.value ?? 512),
    batch_size: values.batch_size ?? Number($("#batchSize")?.value || 1),
    seed: values.seed ?? ($("#seed")?.value === "" ? null : String($("#seed")?.value || "").trim()),
    prompt_parser_name: parserName,
    base_prompt_parser_name: parserName,
    prompt_parser_kwargs: values.prompt_parser_kwargs || safeParseJson($("#promptParserKwargs")?.value, {}),
    prompt_shortcut_profile_name: profileName,
    base_shortcut_profile_name: profileName,
    prompt_shortcut_profile_snapshot: values.prompt_shortcut_profile_snapshot || safeParseJson($("#promptShortcutProfileSnapshot")?.value, {}),
    prompt_parser_preset_name: values.prompt_parser_preset_name ?? $("#promptParserPresetName")?.value ?? "",
    prompt_shadow_compare: values.prompt_shadow_compare ?? Boolean($("#promptShadowCompare")?.checked),
    hires_prompt_parser_mode: values.hires_prompt_parser_mode || $("#hiresPromptParserMode")?.value || "same_as_base",
    hires_prompt_parser_name: values.hires_prompt_parser_name || $("#hiresPromptParserName")?.value || parserName,
    hires_prompt_parser_kwargs: values.hires_prompt_parser_kwargs || safeParseJson($("#hiresPromptParserKwargs")?.value, {}),
    hires_shortcut_profile_mode: values.hires_shortcut_profile_mode || $("#hiresShortcutProfileMode")?.value || "same_as_base",
    hires_shortcut_profile_name: values.hires_shortcut_profile_name || $("#hiresShortcutProfileName")?.value || profileName,
    hires_shortcut_profile_snapshot: values.hires_shortcut_profile_snapshot || safeParseJson($("#hiresShortcutProfileSnapshot")?.value, {}),
    hires_positive_prompt: values.hires_positive_prompt ?? $("#hiresPositivePrompt")?.value ?? "",
    hires_negative_prompt: values.hires_negative_prompt ?? $("#hiresNegativePrompt")?.value ?? "",
    hires_size_mode: values.hires_size_mode || $("#hiresSizeMode")?.value || "scale_from_base",
    hires_scale: Number(values.hires_scale ?? $("#hiresScale")?.value ?? 1.5),
    hires_width: Number(values.hires_width ?? $("#hiresWidth")?.value ?? 0),
    hires_height: Number(values.hires_height ?? $("#hiresHeight")?.value ?? 0),
  };
}

export function conciseRoute(route) {
  if (!route || !Object.keys(route).length) return "No combined route recorded.";
  const features = route.analysis?.features || [];
  const ambiguities = route.ambiguities || [];
  const lines = [
    `Strategy: ${route.strategy || "unknown"}`,
    `Selected parser: ${route.selected_parser || "none"}`,
    `Features: ${features.length ? features.join(", ") : "none"}`,
    `Fallback: ${route.fallback_policy || "fail"}`,
    `Ambiguities: ${ambiguities.length}`,
    `Exact replay: ${route.exact_replay_supported === false ? "blocked" : "supported"}`,
    `Route hash: ${route.route_hash || "unavailable"}`,
  ];
  return `${lines.join("\n")}\n\n${JSON.stringify(route, null, 2)}`;
}

export function conciseShadow(shadow) {
  if (!shadow || !Object.keys(shadow).length) return "Shadow comparison not requested.";
  const lines = [
    `Classification: ${shadow.classification || "unknown"}`,
    `Legacy valid: ${shadow.results?.legacy?.valid ?? "unknown"}`,
    `Parser 21 valid: ${shadow.results?.parser21?.valid ?? "unknown"}`,
    `Differences: ${Object.keys(shadow.differences || {}).join(", ") || "none"}`,
  ];
  return `${lines.join("\n")}\n\n${JSON.stringify(shadow, null, 2)}`;
}
