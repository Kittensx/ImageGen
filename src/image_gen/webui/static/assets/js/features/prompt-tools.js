import { api } from "../api.js";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

let saveSessionSoon = () => {};

function option(value, label, selected = false, disabled = false) {
  const node = document.createElement("option");
  node.value = String(value ?? "");
  node.textContent = label;
  node.selected = selected;
  node.disabled = disabled;
  return node;
}

function slug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "") || "custom_profile";
}

function safeParseJson(value, fallback = {}) {
  try {
    const parsed = JSON.parse(String(value || ""));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function parserById(value) {
  const token = String(value || "legacy").toLowerCase();
  return state.promptParsers.find((item) => String(item.parser_id || "").toLowerCase() === token) || null;
}

function profileById(value) {
  const token = String(value || "").toLowerCase();
  return state.promptShortcutProfiles.find((item) => String(item.profile_id || "").toLowerCase() === token) || null;
}

function presetById(value) {
  const token = String(value || "").toLowerCase();
  return state.promptParserPresets.find((item) => [item.preset_id, item.name].some((candidate) => String(candidate || "").toLowerCase() === token)) || null;
}

function profileSnapshot(profile) {
  if (!profile) return {};
  const keys = [
    "contract_version", "profile_id", "label", "version", "aliases", "parser_emitters",
    "compatible_parsers", "escape_character", "builtin", "credit", "description", "source",
    "palette", "mapping_hash",
  ];
  return Object.fromEntries(keys.filter((key) => key in profile).map((key) => [key, profile[key]]));
}

function setSnapshot(profile) {
  const snapshot = profileSnapshot(profile);
  state.promptConfiguration.shortcutProfileSnapshot = snapshot;
  const input = $("#promptShortcutProfileSnapshot");
  if (input) input.value = JSON.stringify(snapshot);
}

function setParserKwargs(values = {}) {
  state.promptConfiguration.parserKwargs = { ...(values || {}) };
  const input = $("#promptParserKwargs");
  if (input) input.value = JSON.stringify(state.promptConfiguration.parserKwargs);
  document.dispatchEvent(new CustomEvent("prompt-parser-options-changed", {
    detail: { parserId: currentParserId(), options: { ...state.promptConfiguration.parserKwargs } },
  }));
}

function parserSettingsSchema(parserId) {
  return parserById(parserId)?.settings_schema || { properties: {} };
}

function parserDefaultOptions(parserId) {
  const properties = parserSettingsSchema(parserId).properties || {};
  return Object.fromEntries(Object.entries(properties)
    .filter(([, spec]) => Object.prototype.hasOwnProperty.call(spec || {}, "default"))
    .map(([key, spec]) => [key, structuredClone(spec.default)]));
}

function renderParserSettings(containerSelector, parserId, values = {}, onChange = () => {}) {
  const container = $(containerSelector);
  if (!container) return;
  container.replaceChildren();
  const parser = parserById(parserId);
  const properties = parser?.settings_schema?.properties || {};
  const effective = { ...parserDefaultOptions(parserId), ...(values || {}) };
  if (!Object.keys(properties).length) {
    const message = document.createElement("small");
    message.textContent = "This parser has no request-scoped advanced settings.";
    container.append(message);
    return;
  }
  Object.entries(properties).forEach(([key, specValue]) => {
    const spec = specValue || {};
    const label = document.createElement("label");
    label.className = `prompt-parser-setting${spec.type === "boolean" ? " is-boolean" : ""}`;
    const title = document.createElement("span");
    title.textContent = spec.title || key;
    let input;
    if (Array.isArray(spec.enum)) {
      input = document.createElement("select");
      spec.enum.forEach((value) => input.append(option(value, String(value), effective[key] === value)));
    } else {
      input = document.createElement("input");
      if (spec.type === "boolean") {
        input.type = "checkbox";
        input.checked = Boolean(effective[key]);
      } else if (spec.type === "integer" || spec.type === "number") {
        input.type = "number";
        if (spec.minimum !== undefined) input.min = String(spec.minimum);
        if (spec.maximum !== undefined) input.max = String(spec.maximum);
        input.step = spec.type === "integer" ? "1" : String(spec.multipleOf || "any");
        input.value = effective[key] ?? "";
        if (spec.x_nullable) input.placeholder = "Inherit generation value";
      } else {
        input.type = "text";
        input.value = effective[key] ?? "";
      }
    }
    input.dataset.parserSetting = key;
    input.setAttribute("aria-label", spec.title || key);
    const commit = () => {
      const next = { ...effective };
      if (spec.type === "boolean") next[key] = input.checked;
      else if (input.value === "" && spec.x_nullable) delete next[key];
      else if (spec.type === "integer") next[key] = Number.parseInt(input.value, 10);
      else if (spec.type === "number") next[key] = Number.parseFloat(input.value);
      else next[key] = input.value;
      onChange(next);
    };
    input.addEventListener("change", commit);
    input.addEventListener("input", commit);
    label.append(title, input);
    if (spec.description) {
      const description = document.createElement("small");
      description.textContent = spec.description;
      label.append(description);
    }
    container.append(label);
  });
}

function renderBaseParserSettings() {
  const parser = parserById(currentParserId());
  const warning = $("#promptParserExperimentalWarning");
  if (warning) {
    warning.hidden = !parser?.experimental;
    warning.textContent = parser?.experimental
      ? `${parser.label || parser.parser_id} is experimental. Validate the prompt before queueing and preserve parser metadata for replay.`
      : "";
  }
  renderParserSettings(
    "#promptParserAdvancedContent",
    currentParserId(),
    safeParseJson($("#promptParserKwargs")?.value, {}),
    (next) => { setParserKwargs(next); saveSessionSoon(); },
  );
  const status = $("#promptParserSettingsStatus");
  if (status) {
    const processScoped = parser?.process_scoped_settings || [];
    status.textContent = processScoped.length
      ? `${processScoped.length} additional parser settings are process-scoped and are reported for diagnostics rather than changed per queue item.`
      : "Settings are request-scoped and saved with replay metadata.";
  }
}

function populateHiresParsers(selected = "") {
  const select = $("#hiresPromptParserName");
  if (!select) return;
  const available = state.promptParsers.filter((item) => item.available !== false);
  const fallback = available.some((item) => item.parser_id === currentParserId())
    ? currentParserId()
    : (available[0]?.parser_id || "legacy");
  const preferred = available.some((item) => item.parser_id === selected) ? selected : fallback;
  select.replaceChildren(...state.promptParsers.map((item) => option(
    item.parser_id,
    `${item.label || item.parser_id}${item.experimental ? " - Experimental" : ""}${item.available === false ? " - Unavailable" : ""}`,
    item.parser_id === preferred,
    item.available === false,
  )));
  select.value = preferred;
}

function populateHiresProfiles(selected = "") {
  const select = $("#hiresShortcutProfileName");
  if (!select) return;
  const parserId = $("#hiresPromptParserName")?.value || currentParserId();
  const profiles = compatibleProfiles(parserId);
  const defaultId = defaultProfileId(parserId);
  const preferred = profiles.some((item) => item.profile_id === selected)
    ? selected
    : (profiles.some((item) => item.profile_id === defaultId) ? defaultId : (profiles[0]?.profile_id || ""));
  select.replaceChildren(...profiles.map((item) => option(
    item.profile_id,
    `${item.label || item.profile_id}${item.builtin ? "" : " - User"}`,
    item.profile_id === preferred,
  )));
  select.value = preferred;
  const profile = profileById(select.value);
  const snapshot = $("#hiresShortcutProfileSnapshot");
  if (snapshot) snapshot.value = JSON.stringify(profileSnapshot(profile));
}

function renderHiresParserSettings() {
  const parserId = $("#hiresPromptParserName")?.value || currentParserId();
  renderParserSettings(
    "#hiresPromptParserAdvancedContent",
    parserId,
    safeParseJson($("#hiresPromptParserKwargs")?.value, {}),
    (next) => {
      const input = $("#hiresPromptParserKwargs");
      if (input) input.value = JSON.stringify(next);
      saveSessionSoon();
    },
  );
}

function updateHiresRouting() {
  const parserMode = $("#hiresPromptParserMode")?.value || "same_as_base";
  const profileMode = $("#hiresShortcutProfileMode")?.value || "same_as_base";
  const parserSelect = $("#hiresPromptParserName");
  const profileSelect = $("#hiresShortcutProfileName");
  if (parserMode === "same_as_base" || parserMode === "canonical_only") {
    populateHiresParsers(currentParserId());
    if (parserSelect) parserSelect.disabled = true;
    const kwargs = $("#hiresPromptParserKwargs");
    if (kwargs) kwargs.value = $("#promptParserKwargs")?.value || "{}";
  } else {
    if (parserSelect) parserSelect.disabled = parserMode === "canonical_only";
  }
  if (profileMode === "same_as_base") {
    populateHiresProfiles($("#promptShortcutProfileName")?.value || "");
    if (profileSelect) profileSelect.disabled = true;
  } else if (profileMode === "canonical_only" || parserMode === "canonical_only") {
    populateHiresProfiles("canonical");
    if (profileSelect) {
      profileSelect.value = "canonical";
      profileSelect.disabled = true;
    }
  } else if (profileSelect) {
    profileSelect.disabled = false;
  }
  renderHiresParserSettings();
  const status = $("#hiresPromptRoutingStatus");
  if (status) {
    status.textContent = parserMode === "same_as_base" && profileMode === "same_as_base"
      ? "The second pass inherits the base parser, shortcut profile, and parser options."
      : `Second pass: ${parserMode.replaceAll("_", " ")} parser · ${profileMode.replaceAll("_", " ")} shortcut profile.`;
  }
}

function renderMessageList(sectionSelector, listSelector, messages = []) {
  const section = $(sectionSelector);
  const list = $(listSelector);
  if (!section || !list) return;
  section.hidden = !messages.length;
  list.replaceChildren(...messages.map((item) => {
    const node = document.createElement("li");
    node.textContent = item.message || String(item);
    return node;
  }));
}

function currentParserId() {
  return $("#promptParserName")?.value || "legacy";
}

function currentProfile() {
  return profileById($("#promptShortcutProfileName")?.value);
}

function compatibleProfiles(parserId) {
  const normalizedParser = String(parserId || "legacy").toLowerCase();
  return state.promptShortcutProfiles.filter((profile) => {
    const compatible = (profile.compatible_parsers || []).map((item) => String(item).toLowerCase());
    const combinedCompatible = normalizedParser === "combined" && compatible.some((item) => ["combined", "legacy", "parser21"].includes(item));
    return profile.valid !== false && (compatible.includes(normalizedParser) || combinedCompatible);
  });
}

function defaultProfileId(parserId) {
  const normalized = String(parserId || "legacy").toLowerCase();
  if (normalized === "parser21") return "parser21_native";
  if (normalized === "superhybrid") return "superhybrid_native";
  if (normalized === "combined") return "canonical";
  return "legacy_default";
}

function populateParsers(selected = "legacy") {
  const select = $("#promptParserName");
  if (!select) return;
  const available = state.promptParsers.filter((item) => item.available !== false);
  const fallback = available.some((item) => item.parser_id === "legacy")
    ? "legacy"
    : (available[0]?.parser_id || "legacy");
  const preferred = available.some((item) => item.parser_id === selected) ? selected : fallback;
  select.replaceChildren(...state.promptParsers.map((item) => option(
    item.parser_id,
    `${item.label || item.parser_id}${item.experimental ? " - Experimental" : ""}${item.available === false ? " - Unavailable" : ""}`,
    item.parser_id === preferred,
    item.available === false,
  )));
  select.value = preferred;
  select.disabled = available.length < 2;
}

function populateProfiles(selected = "") {
  const parserId = currentParserId();
  const select = $("#promptShortcutProfileName");
  if (!select) return;
  const profiles = compatibleProfiles(parserId);
  const preferred = profiles.some((item) => item.profile_id === selected) ? selected : defaultProfileId(parserId);
  select.replaceChildren(...profiles.map((item) => option(
    item.profile_id,
    `${item.label || item.profile_id}${item.builtin ? "" : " — User"}`,
    item.profile_id === preferred,
  )));
  select.value = profiles.some((item) => item.profile_id === preferred) ? preferred : (profiles[0]?.profile_id || "");
  select.disabled = profiles.length < 2;
  setSnapshot(currentProfile());
}

function populatePresets(selected = "") {
  const select = $("#promptParserPresetName");
  if (!select) return;
  const preferred = presetById(selected)?.preset_id || "";
  select.replaceChildren(
    option("", "No parser preset", !preferred),
    ...state.promptParserPresets.map((item) => option(
      item.preset_id,
      `${item.name || item.preset_id}${item.builtin ? " - Built in" : " - User"}`,
      item.preset_id === preferred,
    )),
  );
  select.value = preferred;
  select.disabled = false;
  updatePresetDeleteState();
}

function updatePresetDeleteState() {
  const preset = presetById($("#promptParserPresetName")?.value);
  const button = $("#deletePromptParserPresetButton");
  if (button) button.disabled = !preset || Boolean(preset.builtin);
}

function aliasFor(profile, operator, fallback = "") {
  const aliases = profile?.aliases?.[operator] || [];
  return String(aliases[0] || fallback || operator);
}

function resolvedPaletteItem(item, profile) {
  const operator = String(item.operator || "").toUpperCase();
  const alias = aliasFor(profile, operator, item.alias || item.label || operator);
  const value = { ...item, alias };
  value.label = String(item.label || alias);
  if (["AND", "SEQUENCE", "DEEP_SEQUENCE", "CLOSE", "TOP_CLOSE"].includes(operator)) {
    value.label = alias;
    value.insert = operator === "AND" ? ` ${alias} ` : alias;
  }
  if (operator === "GROUP_OPEN") {
    value.label = `${aliasFor(profile, "GROUP_OPEN", "{")} ${aliasFor(profile, "GROUP_CLOSE", "}")}`;
    value.prefix = aliasFor(profile, "GROUP_OPEN", "{");
    value.suffix = aliasFor(profile, "GROUP_CLOSE", "}");
  }
  if (value.template && ["BIND", "CHUNK", "BLEND", "POOL", "MORPH", "ASSEMBLE", "COMPOUND"].includes(operator)) {
    value.template = String(value.template).replaceAll(operator, alias);
    value.label = alias;
  }
  return value;
}

function renderPalette() {
  const palette = $("#promptSymbolPalette");
  if (!palette) return;
  palette.replaceChildren();
  const parserId = currentParserId();
  const profile = currentProfile();
  const items = profile?.palettes?.[parserId] || profile?.palette || [];
  items.map((item) => resolvedPaletteItem(item, profile)).forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "prompt-symbol-button";
    button.textContent = item.label || item.alias || item.operator;
    button.title = `${item.description || item.operator}${item.example ? `\nExample: ${item.example}` : ""}`;
    button.setAttribute("aria-label", `${item.description || "Insert prompt syntax"}: ${item.label || item.operator}`);
    button.addEventListener("click", () => insertPaletteItem(item));
    palette.append(button);
  });
  if (!palette.children.length) {
    const message = document.createElement("small");
    message.textContent = "No symbol helpers are defined for this parser/profile combination.";
    palette.append(message);
  }
  const parser = parserById(parserId);
  const status = $("#promptParserSelectionStatus");
  if (status) {
    status.textContent = `${parser?.label || parserId} · ${profile?.label || "No shortcut profile"} · ${profile?.mapping_hash ? `mapping ${profile.mapping_hash.slice(0, 12)}` : "mapping unavailable"}`;
  }
}

function targetTextarea() {
  const mode = $("#promptSymbolTarget")?.value || "auto";
  const targets = {
    positive: "#positivePrompt",
    negative: "#negativePrompt",
    hires_positive: "#hiresPositivePrompt",
    hires_negative: "#hiresNegativePrompt",
  };
  if (mode !== "auto" && targets[mode]) return $(targets[mode]);
  const active = document.activeElement;
  if (["positivePrompt", "negativePrompt", "hiresPositivePrompt", "hiresNegativePrompt"].includes(active?.id)) return active;
  return $(targets[state.promptConfiguration.lastPromptTarget] || targets.positive);
}

function insertPaletteItem(item) {
  const textarea = targetTextarea();
  if (!textarea) return;
  const start = Number.isInteger(textarea.selectionStart) ? textarea.selectionStart : textarea.value.length;
  const end = Number.isInteger(textarea.selectionEnd) ? textarea.selectionEnd : start;
  const selected = textarea.value.slice(start, end);
  const placeholder = selected || item.placeholder || "";
  let inserted = "";
  let selectionOffset = 0;
  let selectionLength = 0;
  if (item.kind === "wrap") {
    inserted = `${item.prefix || ""}${placeholder}${item.suffix || ""}`;
    selectionOffset = String(item.prefix || "").length;
    selectionLength = placeholder.length;
  } else if (item.kind === "template") {
    inserted = String(item.template || item.alias || "").replaceAll("{{selection}}", placeholder);
    const marker = inserted.indexOf("{{caret}}");
    if (marker >= 0) inserted = inserted.replace("{{caret}}", "");
    const first = placeholder ? inserted.indexOf(placeholder) : -1;
    selectionOffset = marker >= 0 ? marker : Math.max(0, first);
    selectionLength = placeholder.length;
  } else {
    inserted = String(item.insert ?? item.alias ?? item.label ?? "");
    selectionOffset = inserted.length;
  }
  textarea.setRangeText(inserted, start, end, "end");
  textarea.focus();
  if (selectionLength > 0) textarea.setSelectionRange(start + selectionOffset, start + selectionOffset + selectionLength);
  else textarea.setSelectionRange(start + selectionOffset, start + selectionOffset);
  const targetById = {
    positivePrompt: "positive",
    negativePrompt: "negative",
    hiresPositivePrompt: "hires_positive",
    hiresNegativePrompt: "hires_negative",
  };
  state.promptConfiguration.lastPromptTarget = targetById[textarea.id] || "positive";
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
  textarea.dispatchEvent(new Event("change", { bubbles: true }));
  saveSessionSoon();
}

function translationPayload(values = {}) {
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
    seed: values.seed ?? ($("#seed")?.value === "" ? null : Number($("#seed")?.value)),
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

function conciseRoute(route) {
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

function conciseShadow(shadow) {
  if (!shadow || !Object.keys(shadow).length) return "Shadow comparison not requested.";
  const lines = [
    `Classification: ${shadow.classification || "unknown"}`,
    `Legacy valid: ${shadow.results?.legacy?.valid ?? "unknown"}`,
    `Parser 21 valid: ${shadow.results?.parser21?.valid ?? "unknown"}`,
    `Differences: ${Object.keys(shadow.differences || {}).join(", ") || "none"}`,
  ];
  return `${lines.join("\n")}\n\n${JSON.stringify(shadow, null, 2)}`;
}

function formatRegionBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = bytes;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount.toFixed(index ? 2 : 0)} ${units[index]}`;
}

function renderRegionTimeline(passData, timelineSelector, estimateSelector) {
  const timeline = $(timelineSelector);
  const estimateNode = $(estimateSelector);
  const regional = passData?.regional_prompting || {};
  const slots = regional.slots || [];
  const regions = slots.flatMap((slot, slotIndex) => (slot.regions || []).map((region, regionIndex) => ({
    slotIndex: Number(slot.slot_index ?? slotIndex),
    regionIndex: Number(region.region_index ?? regionIndex),
    prompt: String(region.prompt || "region"),
    start: Math.max(0, Math.min(1, Number(region.start ?? 0))),
    stop: Math.max(0, Math.min(1, Number(region.stop ?? 1))),
    curve: String(region.curve || "linear"),
  })));
  if (timeline) {
    timeline.replaceChildren();
    if (!regions.length) {
      const empty = document.createElement("div");
      empty.className = "region-timeline-empty";
      empty.textContent = "No native REGION branches detected.";
      timeline.append(empty);
    } else {
      regions.forEach((region) => {
        const row = document.createElement("div");
        row.className = "region-timeline-row";
        const label = document.createElement("div");
        label.className = "region-timeline-label";
        label.textContent = `S${region.slotIndex + 1} R${region.regionIndex + 1} · ${region.prompt}`;
        label.title = `${region.prompt} · ${region.start.toFixed(2)}–${region.stop.toFixed(2)} · ${region.curve}`;
        const track = document.createElement("div");
        track.className = "region-timeline-track";
        const windowNode = document.createElement("div");
        windowNode.className = "region-timeline-window";
        windowNode.style.left = `${region.start * 100}%`;
        windowNode.style.width = `${Math.max(0.5, (region.stop - region.start) * 100)}%`;
        windowNode.title = label.title;
        track.append(windowNode);
        row.append(label, track);
        timeline.append(row);
      });
    }
  }
  if (estimateNode) {
    const estimate = regional.runtime_estimate || {};
    const peak = estimate.estimated_incremental_peak_bytes || {};
    const masks = estimate.estimated_mask_cache_bytes || {};
    estimateNode.textContent = regions.length ? [
      `Backend: ${regional.backend || "image_gen_model_output"}`,
      `Overlap: ${regional.overlap_policy || "additive"}`,
      `Regions: ${estimate.region_count ?? regions.length}`,
      `Estimated extra UNet calls: ${estimate.extra_unet_calls ?? "unknown"}`,
      `Maximum active branches per step: ${estimate.max_active_regions_per_step ?? "unknown"}`,
      `Estimated FP16 mask cache: ${formatRegionBytes(masks.fp16)}`,
      `Estimated FP16 incremental peak: ${formatRegionBytes(peak.fp16)}`,
      "Estimate excludes model residency and allocator overhead.",
    ].join("\n") : "No REGION runtime overhead estimated.";
  }
}

const REGION_BUILDER_TARGETS = {
  positive: "#positivePrompt",
  hires_positive: "#hiresPositivePrompt",
};
let regionBuilderDialog = null;
let regionBuilderFrame = null;
let regionBuilderTarget = "positive";
let regionBuilderBound = false;
let regionBuilderReady = false;

function normalizedRegionBuilderTarget() {
  let target = String($("#promptSymbolTarget")?.value || "auto");
  // The REGION Builder defaults to the base positive prompt. The symbol
  // palette's auto target may remain on a hires field after focus moves back
  // to the base prompt, which can send the wrong pass dimensions.
  if (target === "auto") target = "positive";
  if (target === "negative") target = "positive";
  if (target === "hires_negative") target = "hires_positive";
  return REGION_BUILDER_TARGETS[target] ? target : "positive";
}

function regionBuilderDimensions(target = regionBuilderTarget) {
  const baseWidth = Math.max(64, Math.round(Number($("#width")?.value || 512)));
  const baseHeight = Math.max(64, Math.round(Number($("#height")?.value || 512)));
  if (target !== "hires_positive") {
    return { width: baseWidth, height: baseHeight, pass: "base" };
  }

  const hiresEnabled = Boolean($("#hiresEnabled")?.checked);
  if (!hiresEnabled) {
    return { width: baseWidth, height: baseHeight, pass: "base" };
  }

  const mode = String($("#hiresSizeMode")?.value || "scale_from_base");
  if (mode === "explicit_dimensions") {
    return {
      width: Math.max(64, Math.round(Number($("#hiresWidth")?.value || baseWidth))),
      height: Math.max(64, Math.round(Number($("#hiresHeight")?.value || baseHeight))),
      pass: "hires",
    };
  }

  const scale = Math.max(1.01, Number($("#hiresScale")?.value || 1.5));
  return {
    width: Math.max(64, Math.round(baseWidth * scale)),
    height: Math.max(64, Math.round(baseHeight * scale)),
    pass: "hires",
  };
}

function findRegionBlockRange(text) {
  const source = String(text || "");
  const start = source.indexOf("REGION{");
  if (start < 0) return null;
  let depth = 1;
  let index = start + "REGION{".length;
  while (index < source.length && depth > 0) {
    if (source[index] === "\\" && index + 1 < source.length) {
      index += 2;
      continue;
    }
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    index += 1;
  }
  if (depth !== 0) return null;
  let end = index;
  let cursor = end;
  while (cursor < source.length && /\s/.test(source[cursor])) cursor += 1;
  if (source[cursor] === "[") {
    let bracketDepth = 1;
    cursor += 1;
    while (cursor < source.length && bracketDepth > 0) {
      if (source[cursor] === "\\" && cursor + 1 < source.length) {
        cursor += 2;
        continue;
      }
      if (source[cursor] === "[") bracketDepth += 1;
      if (source[cursor] === "]") bracketDepth -= 1;
      cursor += 1;
    }
    if (bracketDepth === 0) end = cursor;
  } else if (source[cursor] === ":") {
    cursor += 1;
    while (cursor < source.length && !/\s/.test(source[cursor])) cursor += 1;
    end = cursor;
  }
  return { start, end };
}

function applyRegionBuilderPrompt(prompt) {
  const selector = REGION_BUILDER_TARGETS[regionBuilderTarget] || "#positivePrompt";
  const field = $(selector);
  if (!field) return;
  const replacement = String(prompt || "").trim();
  if (!replacement) return;
  const existing = String(field.value || "");
  const range = findRegionBlockRange(existing);
  if (range) {
    field.value = `${existing.slice(0, range.start)}${replacement}${existing.slice(range.end)}`.replace(/\s{2,}/g, " ").trim();
  } else {
    const start = Number.isInteger(field.selectionStart) ? field.selectionStart : existing.length;
    const end = Number.isInteger(field.selectionEnd) ? field.selectionEnd : start;
    const before = existing.slice(0, start).trimEnd();
    const after = existing.slice(end).trimStart();
    field.value = [before, replacement, after].filter(Boolean).join(" ");
  }
  field.dispatchEvent(new Event("input", { bubbles: true }));
  field.dispatchEvent(new Event("change", { bubbles: true }));
  field.focus();
  saveSessionSoon();
  notify("Applied the REGION plan to the prompt.");
}

function regionBuilderView() {
  if (!regionBuilderDialog) regionBuilderDialog = $("#regionBuilderDialog");
  if (!regionBuilderFrame) regionBuilderFrame = $("#regionBuilderFrame");
  return {
    dialog: regionBuilderDialog,
    frame: regionBuilderFrame,
    closeButton: $("#regionBuilderCloseButton"),
    closeToolbar: $("#regionBuilderCloseToolbarButton"),
  };
}

function closeRegionBuilder() {
  const view = regionBuilderView();
  if (view.dialog?.open) view.dialog.close();
}

function sendRegionBuilderInit({ reason = "open" } = {}) {
  const view = regionBuilderView();
  const hostWindow = view.frame?.contentWindow;
  if (!hostWindow) return;
  const selector = REGION_BUILDER_TARGETS[regionBuilderTarget] || "#positivePrompt";
  const dimensions = regionBuilderDimensions(regionBuilderTarget);
  hostWindow.postMessage({
    type: "imagegen-region-builder-init",
    target: regionBuilderTarget,
    target_pass: dimensions.pass,
    prompt: $(selector)?.value || "",
    width: dimensions.width,
    height: dimensions.height,
    reason,
  }, window.location.origin);
}

function openRegionBuilder() {
  const view = regionBuilderView();
  if (!view.dialog || !view.frame) {
    notify("REGION Builder UI is unavailable in this build.", "error");
    return;
  }
  regionBuilderTarget = normalizedRegionBuilderTarget();
  const dimensions = regionBuilderDimensions(regionBuilderTarget);
  const builderUrl = new URL("/region-builder.html", window.location.origin);
  builderUrl.searchParams.set("v", "0.1.66");
  builderUrl.searchParams.set("target", regionBuilderTarget);
  builderUrl.searchParams.set("pass", dimensions.pass);
  builderUrl.searchParams.set("width", String(dimensions.width));
  builderUrl.searchParams.set("height", String(dimensions.height));
  const nextUrl = builderUrl.toString();
  if (view.frame.dataset.loadedSrc !== nextUrl) {
    regionBuilderReady = false;
    view.frame.dataset.loadedSrc = nextUrl;
    view.frame.src = nextUrl;
  } else {
    window.setTimeout(() => sendRegionBuilderInit({ reason: "reopen" }), 80);
  }
  if (!view.dialog.open) view.dialog.showModal();
  window.setTimeout(() => {
    if (view.frame.contentWindow) sendRegionBuilderInit({ reason: regionBuilderReady ? "refresh" : "open" });
  }, 140);
}

function bindRegionBuilderBridge() {
  if (regionBuilderBound) return;
  regionBuilderBound = true;
  const view = regionBuilderView();
  const close = () => closeRegionBuilder();
  view.closeButton?.addEventListener("click", close);
  view.closeToolbar?.addEventListener("click", close);
  view.dialog?.addEventListener("click", (event) => {
    if (event.target === view.dialog) closeRegionBuilder();
  });
  view.dialog?.addEventListener("close", () => {
    regionBuilderReady = false;
  });
  view.frame?.addEventListener("load", () => {
    regionBuilderReady = false;
    window.setTimeout(() => sendRegionBuilderInit({ reason: "frame-load" }), 180);
  });
  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const payload = event.data || {};
    if (payload.type === "imagegen-region-builder-ready") {
      regionBuilderReady = true;
      sendRegionBuilderInit({ reason: "ready" });
    } else if (payload.type === "imagegen-region-builder-apply") {
      if (REGION_BUILDER_TARGETS[payload.target]) regionBuilderTarget = payload.target;
      const dimensions = regionBuilderDimensions(regionBuilderTarget);
      const builderWidth = Math.round(Number(payload.width || 0));
      const builderHeight = Math.round(Number(payload.height || 0));
      const usesPixels = Boolean(payload.pixel_coordinates);
      if (usesPixels && (builderWidth !== dimensions.width || builderHeight !== dimensions.height)) {
        view.frame?.contentWindow?.postMessage({
          type: "imagegen-region-builder-resync",
          target: regionBuilderTarget,
          target_pass: dimensions.pass,
          width: dimensions.width,
          height: dimensions.height,
        }, window.location.origin);
        notify(
          `REGION Builder resolution ${builderWidth}x${builderHeight} did not match ${dimensions.pass} generation ${dimensions.width}x${dimensions.height}. The builder was resynchronized; review the layout and click Apply again.`,
          "warning",
        );
        return;
      }
      applyRegionBuilderPrompt(payload.prompt);
    }
  });
}

function renderTranslation(data, { revealPreview = false } = {}) {
  state.promptConfiguration.translationPreview = data;
  const set = (selector, value) => { const node = $(selector); if (node) node.textContent = value ?? ""; };
  const base = data.base || data;
  const hires = data.hires || {};
  set("#promptTranslationPositiveRaw", base.positive?.raw_prompt);
  set("#promptTranslationPositiveExpanded", base.positive?.parser_input);
  set("#promptTranslationPositiveCanonical", base.positive?.parser_canonical_prompt || base.positive?.canonical_prompt);
  set("#promptTranslationNegativeRaw", base.negative?.raw_prompt);
  set("#promptTranslationNegativeExpanded", base.negative?.parser_input);
  set("#promptTranslationNegativeCanonical", base.negative?.parser_canonical_prompt || base.negative?.canonical_prompt);
  set("#promptTranslationHiresPositiveRaw", hires.positive?.raw_prompt);
  set("#promptTranslationHiresPositiveExpanded", hires.positive?.parser_input);
  set("#promptTranslationHiresPositiveCanonical", hires.positive?.parser_canonical_prompt || hires.positive?.canonical_prompt);
  set("#promptTranslationHiresNegativeRaw", hires.negative?.raw_prompt);
  set("#promptTranslationHiresNegativeExpanded", hires.negative?.parser_input);
  set("#promptTranslationHiresNegativeCanonical", hires.negative?.parser_canonical_prompt || hires.negative?.canonical_prompt);
  const renderSlots = (passData) => JSON.stringify({
    scope: passData?.prompt_expansion_scope || passData?.expansion_scope || "per_batch",
    positive: passData?.expanded_prompts_by_slot?.positive || [],
    negative: passData?.expanded_prompts_by_slot?.negative || [],
    semantic_fingerprints: passData?.semantic_fingerprints_by_slot || {},
  }, null, 2);
  set("#promptTranslationBaseSlots", renderSlots(base));
  set("#promptTranslationHiresSlots", renderSlots(hires));
  renderRegionTimeline(base, "#promptRegionBaseTimeline", "#promptRegionBaseEstimate");
  renderRegionTimeline(hires, "#promptRegionHiresTimeline", "#promptRegionHiresEstimate");
  const routes = [base.positive?.route_plan, base.negative?.route_plan, hires.positive?.route_plan, hires.negative?.route_plan];
  const shadows = [base.positive?.shadow_comparison, base.negative?.shadow_comparison, hires.positive?.shadow_comparison, hires.negative?.shadow_comparison];
  set("#promptRouteBasePositive", conciseRoute(routes[0]));
  set("#promptRouteBaseNegative", conciseRoute(routes[1]));
  set("#promptRouteHiresPositive", conciseRoute(routes[2]));
  set("#promptRouteHiresNegative", conciseRoute(routes[3]));
  set("#promptShadowBasePositive", conciseShadow(shadows[0]));
  set("#promptShadowBaseNegative", conciseShadow(shadows[1]));
  set("#promptShadowHiresPositive", conciseShadow(shadows[2]));
  set("#promptShadowHiresNegative", conciseShadow(shadows[3]));
  const routeSection = $("#promptRouteSummarySection");
  if (routeSection) routeSection.hidden = !routes.some((item) => item && Object.keys(item).length) && !shadows.some((item) => item && Object.keys(item).length);
  renderMessageList("#promptPreflightBlockingSection", "#promptPreflightBlocking", data.blocking_errors || []);
  renderMessageList("#promptPreflightWarningSection", "#promptPreflightWarnings", data.behavior_warnings || []);
  renderMessageList("#promptPreflightNoticeSection", "#promptPreflightNotices", data.informational_notices || []);
  const summary = data.valid
    ? `Prompt preflight valid · base ${base.parser?.parser_id || currentParserId()} / ${base.shortcut_profile?.profile_id || "profile"} · hires ${hires.parser?.parser_id || "inherit"} / ${hires.shortcut_profile?.profile_id || "inherit"}`
    : "Prompt preflight contains blocking errors.";
  set("#promptTranslationWarnings", summary);
  const differs = Object.values(hires.interpretation_diff || {}).some((item) => item?.different);
  const diffWarning = $("#promptHiresInterpretationWarning");
  if (diffWarning) {
    diffWarning.hidden = !differs;
    diffWarning.textContent = differs
      ? "The base and hires passes do not resolve to the same parser input or canonical prompt. Review both columns before queueing."
      : "";
  }
  const details = $("#promptTranslationPreview");
  if (details && revealPreview) details.open = true;
}

async function validateCurrentPrompt() {
  const button = $("#validateCurrentPromptButton");
  try {
    if (button) button.disabled = true;
    const report = await api.preflightPrompts(translationPayload());
    renderTranslation(report, { revealPreview: true });
    notify(report.valid ? "Prompt preflight completed successfully." : "Prompt preflight found blocking errors.", report.valid ? "info" : "error");
  } catch (error) {
    const warning = $("#promptTranslationWarnings");
    if (warning) warning.textContent = error.message;
    const details = $("#promptTranslationPreview");
    if (details) details.open = true;
    notify(error.message, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

function applyPreset(preset) {
  if (!preset) return;
  populateParsers(preset.prompt_parser_name || "legacy");
  populateProfiles(preset.shortcut_profile_name || defaultProfileId(currentParserId()));
  setParserKwargs(preset.prompt_parser_kwargs || {});
  setSnapshot(currentProfile());
  const inheritance = preset.hires_inheritance || "same_as_base";
  if ($("#hiresPromptParserMode")) $("#hiresPromptParserMode").value = inheritance === "same_as_base" ? "same_as_base" : "explicit";
  if ($("#hiresShortcutProfileMode")) $("#hiresShortcutProfileMode").value = inheritance === "same_as_base" ? "same_as_base" : "explicit";
  renderBaseParserSettings();
  updateHiresRouting();
  renderPalette();
  updatePresetDeleteState();
  saveSessionSoon();
}

async function saveParserPreset() {
  const name = window.prompt("Name this prompt parser preset:");
  if (!name?.trim()) return;
  try {
    const response = await api.savePromptParserPreset({
      preset_id: slug(name),
      name: name.trim(),
      prompt_parser_name: currentParserId(),
      shortcut_profile_name: $("#promptShortcutProfileName")?.value,
      prompt_parser_kwargs: safeParseJson($("#promptParserKwargs")?.value, {}),
      fallback_policy: "fail",
      hires_inheritance: "same_as_base",
    });
    state.promptParserPresets = response.presets || state.promptParserPresets;
    populatePresets(response.preset?.preset_id || slug(name));
    saveSessionSoon();
    notify(`Saved prompt parser preset: ${name.trim()}`);
  } catch (error) {
    notify(error.message, "error");
  }
}

async function deleteParserPreset() {
  const preset = presetById($("#promptParserPresetName")?.value);
  if (!preset || preset.builtin) return;
  if (!window.confirm(`Delete prompt parser preset “${preset.name}”?`)) return;
  try {
    const response = await api.deletePromptParserPreset(preset.preset_id);
    state.promptParserPresets = response.presets || state.promptParserPresets;
    populatePresets("");
    saveSessionSoon();
    notify(`Deleted prompt parser preset: ${preset.name}`);
  } catch (error) {
    notify(error.message, "error");
  }
}

function editorPayload() {
  const body = safeParseJson($("#promptShortcutProfileEditorJson")?.value, {});
  return {
    ...body,
    profile_id: $("#promptShortcutProfileEditorId")?.value.trim(),
    label: $("#promptShortcutProfileEditorLabel")?.value.trim(),
    version: $("#promptShortcutProfileEditorVersion")?.value.trim() || "1",
    compatible_parsers: String($("#promptShortcutProfileEditorParsers")?.value || "")
      .split(",").map((item) => item.trim()).filter(Boolean),
    builtin: false,
    source: "user",
  };
}

function editableProfileBody(profile) {
  const keys = ["aliases", "parser_emitters", "escape_character", "description", "credit", "palette"];
  return Object.fromEntries(keys.filter((key) => key in profile).map((key) => [key, profile[key]]));
}

function loadProfileEditor(profile) {
  const selected = profile || {
    profile_id: "custom_profile",
    label: "Custom Prompt Shortcuts",
    version: "1",
    compatible_parsers: ["legacy", "parser21", "superhybrid"],
    builtin: false,
    aliases: {},
    parser_emitters: { legacy: {}, parser21: {}, superhybrid: {} },
    escape_character: "\\",
    palette: [],
  };
  $("#promptShortcutProfileEditorId").value = selected.profile_id || "";
  $("#promptShortcutProfileEditorLabel").value = selected.label || "";
  $("#promptShortcutProfileEditorVersion").value = selected.version || "1";
  $("#promptShortcutProfileEditorParsers").value = (selected.compatible_parsers || []).join(", ");
  $("#promptShortcutProfileEditorJson").value = JSON.stringify(editableProfileBody(selected), null, 2);
  const readonly = Boolean(selected.builtin);
  ["#promptShortcutProfileEditorId", "#promptShortcutProfileEditorLabel", "#promptShortcutProfileEditorVersion", "#promptShortcutProfileEditorParsers", "#promptShortcutProfileEditorJson"].forEach((selector) => {
    const node = $(selector); if (node) node.disabled = readonly;
  });
  $("#savePromptShortcutProfileButton").disabled = readonly;
  $("#deletePromptShortcutProfileButton").disabled = readonly;
  $("#promptShortcutProfileValidation").textContent = readonly ? "Built-in profile: duplicate it to edit." : "Validate before saving.";
}

function populateProfileEditorSelect(selectedId = "") {
  const select = $("#promptShortcutProfileEditorSelect");
  if (!select) return;
  select.replaceChildren(...state.promptShortcutProfiles.map((profile) => option(
    profile.profile_id,
    `${profile.label}${profile.builtin ? " — Built-in" : " — User"}`,
    profile.profile_id === selectedId,
  )));
  select.value = state.promptShortcutProfiles.some((profile) => profile.profile_id === selectedId)
    ? selectedId
    : (state.promptShortcutProfiles[0]?.profile_id || "");
  loadProfileEditor(profileById(select.value));
}

function openProfileEditor() {
  populateProfileEditorSelect($("#promptShortcutProfileName")?.value || "");
  $("#promptShortcutProfileDialog")?.showModal();
}

async function validateProfileEditor({ quiet = false } = {}) {
  try {
    const result = await api.validatePromptShortcutProfile(editorPayload());
    const messages = (result.issues || []).map((item) => `[${item.severity}] ${item.message}`);
    $("#promptShortcutProfileValidation").textContent = result.valid
      ? `Valid · mapping ${String(result.mapping_hash || "").slice(0, 16)}${messages.length ? `\n${messages.join("\n")}` : ""}`
      : messages.join("\n") || "Profile is invalid.";
    if (!quiet) notify(result.valid ? "Shortcut profile is valid." : "Shortcut profile has validation errors.", result.valid ? "info" : "error");
    return result;
  } catch (error) {
    $("#promptShortcutProfileValidation").textContent = error.message;
    if (!quiet) notify(error.message, "error");
    return { valid: false };
  }
}

async function saveProfileEditor() {
  const validation = await validateProfileEditor({ quiet: true });
  if (!validation.valid) {
    notify("Fix shortcut profile validation errors before saving.", "error");
    return;
  }
  try {
    const response = await api.savePromptShortcutProfile(editorPayload());
    state.promptShortcutProfiles = response.profiles || state.promptShortcutProfiles;
    const profileId = response.profile?.profile_id || editorPayload().profile_id;
    populateProfileEditorSelect(profileId);
    populateProfiles(profileId);
    renderPalette();
    saveSessionSoon();
    notify(`Saved shortcut profile: ${profileId}`);
  } catch (error) {
    notify(error.message, "error");
  }
}

async function deleteProfileEditor() {
  const profile = profileById($("#promptShortcutProfileEditorSelect")?.value);
  if (!profile || profile.builtin) return;
  if (!window.confirm(`Delete shortcut profile “${profile.label}”?`)) return;
  try {
    const response = await api.deletePromptShortcutProfile(profile.profile_id);
    state.promptShortcutProfiles = response.profiles || state.promptShortcutProfiles;
    populateProfileEditorSelect();
    populateProfiles(defaultProfileId(currentParserId()));
    renderPalette();
    saveSessionSoon();
    notify(`Deleted shortcut profile: ${profile.label}`);
  } catch (error) {
    notify(error.message, "error");
  }
}

function duplicateProfileEditor() {
  const profile = profileById($("#promptShortcutProfileEditorSelect")?.value);
  if (!profile) return;
  const copy = structuredClone(profile);
  copy.profile_id = `${profile.profile_id}_copy`;
  copy.label = `${profile.label} Copy`;
  copy.builtin = false;
  copy.source = "user";
  loadProfileEditor(copy);
}

function newProfileEditor() {
  loadProfileEditor(null);
}

function exportProfileEditor() {
  const payload = editorPayload();
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${slug(payload.profile_id)}.prompt-shortcuts.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function importProfileFile(file) {
  try {
    const payload = JSON.parse(await file.text());
    loadProfileEditor({ ...payload, builtin: false, source: "user" });
    await validateProfileEditor({ quiet: true });
  } catch (error) {
    notify(`Unable to import shortcut profile: ${error.message}`, "error");
  }
}

function bindPromptFocus() {
  const targets = {
    positive: "#positivePrompt",
    negative: "#negativePrompt",
    hires_positive: "#hiresPositivePrompt",
    hires_negative: "#hiresNegativePrompt",
  };
  Object.entries(targets).forEach(([target, selector]) => {
    const node = $(selector);
    ["focus", "click", "keyup", "select"].forEach((eventName) => {
      node?.addEventListener(eventName, () => { state.promptConfiguration.lastPromptTarget = target; });
    });
  });
}

function normalizedHiresSizeMode(value, enabled = true) {
  const mode = String(value || "scale_from_base").trim().toLowerCase();
  if (enabled && mode === "same_as_base") return "scale_from_base";
  return ["same_as_base", "scale_from_base", "explicit_dimensions"].includes(mode)
    ? mode
    : "scale_from_base";
}

function normalizedHiresDimension(value, fallback) {
  const parsed = Number(value);
  const candidate = Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  return Math.max(64, Math.min(16384, Math.round(candidate / 8) * 8));
}

function updateHiresScheduleSummary() {
  const status = $("#hiresStepSummary");
  if (!status) return;
  const enabled = $("#hiresEnabled")?.checked === true;
  const requested = Math.max(1, Math.min(200, Math.round(Number($("#hiresSteps")?.value || 20))));
  const requestedStrength = Number($("#hiresDenoisingStrength")?.value ?? 0.4);
  const strength = Math.max(0.01, Math.min(1.0, Number.isFinite(requestedStrength) ? requestedStrength : 0.4));
  const policy = String($("#hiresStepPolicy")?.value || "a1111_fixed_steps_v1");
  if (!enabled) {
    status.textContent = "Hires schedule: disabled.";
    status.className = "field-status subtle hires-schedule-summary";
    return;
  }
  if (policy === "proportional_tail_v1") {
    const executed = Math.max(1, Math.min(requested, Math.round(requested * strength)));
    status.textContent = `Legacy proportional tail: ${requested} requested · ${requested} internal schedule transitions · ${executed} executed.`;
    status.className = "field-status warning hires-schedule-summary";
    return;
  }
  const safeStrength = Math.min(Math.max(strength, 0.01), 0.999);
  const internal = Math.max(requested, Math.floor(requested / safeStrength));
  status.textContent = `Fixed executed steps: ${requested} requested · ${internal} internal schedule transitions · ${requested} executed.`;
  status.className = "field-status ready hires-schedule-summary";
}

function updateHiresPairStatus() {
  const status = $("#hiresPairStatus");
  if (!status) return;
  const enabled = $("#hiresEnabled")?.checked === true;
  if (!enabled) {
    status.textContent = "Hires generation is disabled.";
    status.className = "field-status subtle";
    return;
  }
  const explicitSampler = String($("#hiresSamplerName")?.value || "");
  const explicitScheduler = String($("#hiresSchedulerName")?.value || "");
  const baseSampler = String($("#samplerName")?.value || "");
  const baseScheduler = String($("#schedulerName")?.value || "");
  const sampler = explicitSampler || baseSampler;
  const scheduler = explicitScheduler || baseScheduler;
  const inheritSamplerOption = $("#hiresSamplerName")?.querySelector('option[value=""]');
  const inheritSchedulerOption = $("#hiresSchedulerName")?.querySelector('option[value=""]');
  if (inheritSamplerOption) inheritSamplerOption.textContent = `Inherit base sampler (${baseSampler || "current"})`;
  if (inheritSchedulerOption) inheritSchedulerOption.textContent = `Inherit base scheduler (${baseScheduler || "current"})`;
  const cfg = String($("#hiresCfgScale")?.value || "").trim();
  const cfgRescale = String($("#hiresCfgRescale")?.value || "").trim();
  const inherited = [
    explicitSampler ? "sampler explicit" : "sampler inherited",
    explicitScheduler ? "scheduler explicit" : "scheduler inherited",
    cfg ? `CFG ${cfg}` : "CFG inherited",
    cfgRescale ? `CFG rescale ${cfgRescale}` : "CFG rescale inherited",
  ].join(" · ");
  const experimental = sampler === "kes" || scheduler === "simple_kes";
  status.textContent = `Second pass: ${sampler || "sampler"} + ${scheduler || "scheduler"} · ${inherited}${experimental ? " · Experimental KES comparison path" : ""}.`;
  status.className = experimental ? "field-status warning" : "field-status subtle";
}

function updateHiresSizeControls() {
  const enabled = $("#hiresEnabled")?.checked === true;
  const sizeMode = $("#hiresSizeMode");
  const mode = normalizedHiresSizeMode(sizeMode?.value, enabled);
  if (sizeMode && sizeMode.value !== mode) sizeMode.value = mode;
  const scaleField = $("#hiresScaleField");
  const widthField = $("#hiresWidthField");
  const heightField = $("#hiresHeightField");
  const scaleInput = $("#hiresScale");
  const widthInput = $("#hiresWidth");
  const heightInput = $("#hiresHeight");
  if (scaleInput) scaleInput.disabled = !enabled || mode !== "scale_from_base";
  if (widthInput) widthInput.disabled = !enabled || mode !== "explicit_dimensions";
  if (heightInput) heightInput.disabled = !enabled || mode !== "explicit_dimensions";
  scaleField?.classList.toggle("is-disabled", !enabled || mode !== "scale_from_base");
  widthField?.classList.toggle("is-disabled", !enabled || mode !== "explicit_dimensions");
  heightField?.classList.toggle("is-disabled", !enabled || mode !== "explicit_dimensions");
  [
    "#hiresPromptParserMode", "#hiresPromptParserName", "#hiresShortcutProfileMode",
    "#hiresShortcutProfileName", "#hiresPositivePrompt", "#hiresNegativePrompt",
    "#hiresSizeMode", "#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy",
    "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale",
    "#hiresUpscaler", "#hiresSaveLowres",
  ].forEach((selector) => {
    const node = $(selector);
    if (node) node.disabled = !enabled;
  });
  const baseWidth = normalizedHiresDimension($("#width")?.value, 512);
  const baseHeight = normalizedHiresDimension($("#height")?.value, 512);
  let width = baseWidth;
  let height = baseHeight;
  if (mode === "scale_from_base") {
    const scale = Math.max(1.01, Math.min(8, Number(scaleInput?.value || 1.5)));
    width = normalizedHiresDimension(baseWidth * scale, baseWidth * 1.5);
    height = normalizedHiresDimension(baseHeight * scale, baseHeight * 1.5);
  } else if (mode === "explicit_dimensions") {
    width = normalizedHiresDimension(widthInput?.value, baseWidth * 2);
    height = normalizedHiresDimension(heightInput?.value, baseHeight * 2);
  }
  const status = $("#hiresSizeStatus");
  if (status) status.textContent = enabled
    ? `Second-pass dimensions: ${width} × ${height} · ${mode.replaceAll("_", " ")}.`
    : "Hires generation is disabled.";
  updateHiresScheduleSummary();
  updateHiresPairStatus();
}

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
  if ($("#hiresEnabled")) $("#hiresEnabled").checked = Boolean(current.hires_enabled);
  if ($("#hiresPromptParserMode")) $("#hiresPromptParserMode").value = current.hires_prompt_parser_mode || "same_as_base";
  populateHiresParsers(current.hires_prompt_parser_name || current.prompt_parser_name || "legacy");
  if ($("#hiresPromptParserKwargs")) $("#hiresPromptParserKwargs").value = JSON.stringify(current.hires_prompt_parser_kwargs || current.prompt_parser_kwargs || {});
  if ($("#hiresShortcutProfileMode")) $("#hiresShortcutProfileMode").value = current.hires_shortcut_profile_mode || "same_as_base";
  populateHiresProfiles(current.hires_shortcut_profile_name || current.prompt_shortcut_profile_name || "");
  if ($("#hiresShortcutProfileSnapshot")) $("#hiresShortcutProfileSnapshot").value = JSON.stringify(current.hires_shortcut_profile_snapshot || {});
  if ($("#hiresPositivePrompt")) $("#hiresPositivePrompt").value = current.hires_positive_prompt || "";
  if ($("#hiresNegativePrompt")) $("#hiresNegativePrompt").value = current.hires_negative_prompt || "";
  if ($("#hiresSizeMode")) {
    $("#hiresSizeMode").value = normalizedHiresSizeMode(
      current.hires_size_mode,
      Boolean(current.hires_enabled),
    );
  }
  if ($("#hiresScale")) $("#hiresScale").value = String(current.hires_scale || 1.5);
  if ($("#hiresWidth")) $("#hiresWidth").value = String(current.hires_width || Math.max(64, Number($("#width")?.value || 640) * 1.5));
  if ($("#hiresHeight")) $("#hiresHeight").value = String(current.hires_height || Math.max(64, Number($("#height")?.value || 960) * 1.5));
  if ($("#hiresSteps")) $("#hiresSteps").value = String(current.hires_steps || 20);
  if ($("#hiresDenoisingStrength")) $("#hiresDenoisingStrength").value = String(current.hires_denoising_strength ?? 0.4);
  if ($("#hiresStepPolicy")) $("#hiresStepPolicy").value = current.hires_step_policy || "a1111_fixed_steps_v1";
  if ($("#hiresSamplerName")) $("#hiresSamplerName").value = current.hires_sampler_name || "";
  if ($("#hiresSchedulerName")) $("#hiresSchedulerName").value = current.hires_scheduler_name || "";
  if ($("#hiresCfgScale")) $("#hiresCfgScale").value = current.hires_cfg_scale ?? "";
  if ($("#hiresCfgRescale")) $("#hiresCfgRescale").value = current.hires_cfg_rescale ?? "";
  if ($("#hiresUpscaler")) $("#hiresUpscaler").value = current.hires_upscaler || "latent_bicubic";
  if ($("#hiresSaveLowres")) $("#hiresSaveLowres").checked = current.hires_save_lowres !== false;
  updateHiresSizeControls();
  renderBaseParserSettings();
  updateHiresRouting();
  renderPalette();
}

export async function preflightCurrentPrompt(values = {}) {
  const report = await api.preflightPrompts(translationPayload(values));
  // Generation preflight updates the preview contents without changing the
  // user's collapsed/expanded state. Only the explicit Validate action reveals it.
  renderTranslation(report, { revealPreview: false });
  return report;
}

export function bindPromptTools(options = {}) {
  saveSessionSoon = options.saveSessionSoon || saveSessionSoon;
  bindPromptFocus();
  bindRegionBuilderBridge();
  $("#openRegionBuilderButton")?.addEventListener("click", openRegionBuilder);
  $("#promptParserName")?.addEventListener("change", () => {
    populateProfiles(defaultProfileId(currentParserId()));
    $("#promptParserPresetName").value = "";
    setParserKwargs(parserDefaultOptions(currentParserId()));
    renderBaseParserSettings();
    updateHiresRouting();
    renderPalette();
    saveSessionSoon();
  });
  $("#promptShadowCompare")?.addEventListener("change", saveSessionSoon);
  $("#promptShortcutProfileName")?.addEventListener("change", () => {
    setSnapshot(currentProfile());
    $("#promptParserPresetName").value = "";
    updateHiresRouting();
    renderPalette();
    saveSessionSoon();
  });
  $("#promptParserPresetName")?.addEventListener("change", (event) => {
    applyPreset(presetById(event.target.value));
    updatePresetDeleteState();
  });
  $("#hiresPromptParserMode")?.addEventListener("change", () => { updateHiresRouting(); saveSessionSoon(); });
  $("#hiresPromptParserName")?.addEventListener("change", () => {
    const kwargs = $("#hiresPromptParserKwargs");
    if (kwargs) kwargs.value = JSON.stringify(parserDefaultOptions($("#hiresPromptParserName")?.value));
    populateHiresProfiles(defaultProfileId($("#hiresPromptParserName")?.value));
    renderHiresParserSettings();
    saveSessionSoon();
  });
  $("#hiresShortcutProfileMode")?.addEventListener("change", () => { updateHiresRouting(); saveSessionSoon(); });
  $("#hiresShortcutProfileName")?.addEventListener("change", () => {
    const snapshot = $("#hiresShortcutProfileSnapshot");
    if (snapshot) snapshot.value = JSON.stringify(profileSnapshot(profileById($("#hiresShortcutProfileName")?.value)));
    saveSessionSoon();
  });
  ["#hiresPositivePrompt", "#hiresNegativePrompt"].forEach((selector) => {
    $(selector)?.addEventListener("input", saveSessionSoon);
  });
  ["#hiresEnabled", "#hiresSizeMode", "#hiresScale", "#hiresWidth", "#hiresHeight", "#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy", "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale", "#hiresUpscaler", "#hiresSaveLowres", "#samplerName", "#schedulerName", "#cfgScale", "#width", "#height"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => { updateHiresSizeControls(); saveSessionSoon(); });
    $(selector)?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  });
  $("#savePromptParserPresetButton")?.addEventListener("click", saveParserPreset);
  $("#deletePromptParserPresetButton")?.addEventListener("click", deleteParserPreset);
  $("#validateCurrentPromptButton")?.addEventListener("click", validateCurrentPrompt);
  $("#editPromptShortcutProfilesButton")?.addEventListener("click", openProfileEditor);
  $("#promptShortcutProfileEditorSelect")?.addEventListener("change", (event) => loadProfileEditor(profileById(event.target.value)));
  $("#duplicatePromptShortcutProfileButton")?.addEventListener("click", duplicateProfileEditor);
  $("#newPromptShortcutProfileButton")?.addEventListener("click", newProfileEditor);
  $("#validatePromptShortcutProfileButton")?.addEventListener("click", () => validateProfileEditor());
  $("#savePromptShortcutProfileButton")?.addEventListener("click", saveProfileEditor);
  $("#deletePromptShortcutProfileButton")?.addEventListener("click", deleteProfileEditor);
  $("#exportPromptShortcutProfileButton")?.addEventListener("click", exportProfileEditor);
  $("#importPromptShortcutProfileButton")?.addEventListener("click", () => $("#importPromptShortcutProfileInput")?.click());
  $("#importPromptShortcutProfileInput")?.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    event.target.value = "";
    if (file) await importProfileFile(file);
  });
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
    hires_cfg_rescale: String($("#hiresCfgRescale")?.value || "").trim() === "" ? null : Number($("#hiresCfgRescale")?.value),
    hires_upscaler: $("#hiresUpscaler")?.value || "latent_bicubic",
    hires_save_lowres: $("#hiresSaveLowres")?.checked !== false,
  });
}
