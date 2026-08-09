import { api } from "../api.js";
import { state } from "../state.js";
import { $, notify } from "../utils.js";
import { clampHiresDimension, normalizeHiresSizeMode, planHiresDimensions } from "../components/hires-dimensions.js?v=0.1.79";
import { setActionIcon } from "../components/action-icons.js?v=2";
import { updateHiresUpscalerPlanUI } from "./hires-upscalers.js?v=0.1.79";

let saveSessionSoon = () => {};
let hiresPlannerParityTimer = null;
let hiresPlannerParitySequence = 0;

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
  const grouped = new Map();
  (messages || []).forEach((item) => {
    const message = item?.message || String(item);
    const key = `${item?.code || "message"}::${message}`;
    if (!grouped.has(key)) grouped.set(key, { message, contexts: [], count: 0 });
    const entry = grouped.get(key);
    entry.count += 1;
    const context = [item?.pass_name, item?.prompt_role].filter(Boolean).join(" · ");
    if (context && !entry.contexts.includes(context)) entry.contexts.push(context);
  });
  const entries = [...grouped.values()];
  section.hidden = !entries.length;
  list.replaceChildren(...entries.map((entry) => {
    const node = document.createElement("li");
    node.className = "prompt-message-group";
    const copy = document.createElement("span");
    copy.textContent = entry.message;
    node.append(copy);
    if (entry.count > 1) {
      const count = document.createElement("strong");
      count.className = "prompt-message-count";
      count.textContent = `×${entry.count}`;
      count.title = `${entry.count} matching occurrences`;
      node.append(count);
    }
    if (entry.contexts.length) {
      const context = document.createElement("small");
      context.textContent = `Affected: ${entry.contexts.join(", ")}`;
      node.append(context);
    }
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

  const plan = planHiresDimensions({
    baseWidth,
    baseHeight,
    mode: $("#hiresSizeMode")?.value || "scale_from_base",
    scale: $("#hiresScale")?.value || 1.5,
    targetWidth: $("#hiresWidth")?.value,
    targetHeight: $("#hiresHeight")?.value,
    enabled: true,
  });
  return {
    width: plan.final_width,
    height: plan.final_height,
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

function promptStageText(role = {}, stage = "raw") {
  if (stage === "raw") return String(role.raw_prompt || "");
  if (stage === "parser") return String(role.parser_input || "");
  return String(role.parser_canonical_prompt || role.canonical_prompt || "");
}

function normalizePromptSource(value) {
  return String(value || "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .join("\n")
    .trim();
}

function parseCanonicalValue(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(String(value || ""));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function canonicalStructureForRole(role = {}) {
  const direct = role.parser_canonical_structure || role.canonical_structure;
  if (direct && typeof direct === "object" && !Array.isArray(direct)) return direct;
  return parseCanonicalValue(promptStageText(role, "canonical"));
}

function canonicalSourceForRole(role = {}) {
  const structure = canonicalStructureForRole(role);
  if (typeof structure.lossless_source === "string") return structure.lossless_source;
  return promptStageText(role, "parser");
}

function canonicalTypeLabel(value) {
  const labels = {
    text: "text node",
    conjunction: "AND conjunction",
    scheduled_text: "scheduled text",
    alternate_text: "alternate text",
    weighted_text: "weighted text",
    attention_group: "attention group",
    deep_sequence: "deep sequence",
    sequence: "sequence",
    extension: "extension operator",
  };
  const token = String(value || "node");
  return labels[token] || token.replaceAll("_", " ");
}

function canonicalStructureSummary(structure = {}) {
  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const counts = new Map();
  nodes.forEach((node) => {
    const label = canonicalTypeLabel(node?.type);
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  return {
    contract: String(structure.contract || "canonical prompt"),
    parserNamespace: String(structure.parser_namespace || "unknown"),
    nodeCount: nodes.length,
    nodeLabels: [...counts.entries()].map(([label, count]) => `${count} ${label}${count === 1 ? "" : "s"}`),
  };
}

function promptDiffTokens(value) {
  return String(value || "").split(/(\s+|[,;:{}()[\]|])/g).filter((item) => item !== "");
}

function promptTokenDiff(before, after) {
  const left = promptDiffTokens(before);
  const right = promptDiffTokens(after);
  if (left.join("") === right.join("")) return [{ type: "equal", text: left.join("") }];
  if (left.length > 220 || right.length > 220) {
    return [
      ...(left.length ? [{ type: "remove", text: left.join("") }] : []),
      ...(right.length ? [{ type: "add", text: right.join("") }] : []),
    ];
  }
  const table = Array.from({ length: left.length + 1 }, () => new Uint16Array(right.length + 1));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      table[i][j] = left[i] === right[j]
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const ops = [];
  const push = (type, text) => {
    if (!text) return;
    const last = ops[ops.length - 1];
    if (last?.type === type) last.text += text;
    else ops.push({ type, text });
  };
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      push("equal", left[i]);
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      push("remove", left[i]);
      i += 1;
    } else {
      push("add", right[j]);
      j += 1;
    }
  }
  while (i < left.length) { push("remove", left[i]); i += 1; }
  while (j < right.length) { push("add", right[j]); j += 1; }
  return ops;
}

function appendPromptDiff(target, before, after) {
  const diff = document.createElement("div");
  diff.className = "prompt-inline-diff";
  promptTokenDiff(before, after).forEach((part) => {
    const node = document.createElement("span");
    node.className = `prompt-diff-${part.type}`;
    node.textContent = part.text;
    diff.append(node);
  });
  target.append(diff);
}

function makeCopyableTransformationBlock(label, text, copyLabel = "Copy recognized text") {
  const wrapper = document.createElement("div");
  wrapper.className = "prompt-transformation-copy-block";
  const header = document.createElement("div");
  header.className = "prompt-transformation-copy-header";
  const heading = document.createElement("span");
  heading.textContent = label;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "ui-action-button ui-icon-control ui-action-button--compact";
  setActionIcon(copy, "copy", { label: copyLabel, replace: true });
  const pre = document.createElement("pre");
  pre.textContent = String(text || "");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent || "");
      notify(`${label} copied.`);
    } catch (error) {
      notify(`Unable to copy ${label.toLowerCase()}: ${error.message}`, "error");
    }
  });
  header.append(heading, copy);
  wrapper.append(header, pre);
  return wrapper;
}

function promptTransformationDetails(roleData = {}, before = "", after = "") {
  const container = document.createElement("div");
  container.className = "prompt-transformation-details";
  const substitutions = Array.isArray(roleData.substitutions) ? roleData.substitutions : [];
  if (substitutions.length) {
    substitutions.forEach((item, index) => {
      const card = document.createElement("section");
      card.className = "prompt-transformation-item";
      const title = document.createElement("strong");
      title.textContent = `Transformation ${index + 1}: ${item.canonical_operator || "shortcut"}`;
      const meta = document.createElement("small");
      meta.textContent = `Shortcut ${JSON.stringify(item.source || "")} → parser output ${JSON.stringify(item.parser_emission || "")}`;
      const start = Number(item.start);
      const end = Number(item.end);
      const exactSource = Number.isFinite(start) && Number.isFinite(end) && end >= start
        ? String(before || "").slice(start, end)
        : String(item.source || "");
      card.append(
        title,
        meta,
        makeCopyableTransformationBlock("Recognized source", exactSource, `Copy source for transformation ${index + 1}`),
        makeCopyableTransformationBlock("Parser output", item.parser_emission || "", `Copy parser output for transformation ${index + 1}`),
      );
      container.append(card);
    });
    return container;
  }

  const parts = promptTokenDiff(before, after);
  const removed = parts.filter((part) => part.type === "remove").map((part) => part.text).join("").trim();
  const added = parts.filter((part) => part.type === "add").map((part) => part.text).join("").trim();
  if (removed || added) {
    const card = document.createElement("section");
    card.className = "prompt-transformation-item";
    const title = document.createElement("strong");
    title.textContent = "Detected text transformation";
    card.append(title);
    if (removed) card.append(makeCopyableTransformationBlock("Recognized source", removed));
    if (added) card.append(makeCopyableTransformationBlock("Result", added, "Copy transformed result"));
    container.append(card);
  }
  return container;
}

function promptTransitionRow(label, before, after, { roleData = null, showTransformationBlocks = false } = {}) {
  const row = document.createElement("div");
  row.className = "prompt-change-row";
  const heading = document.createElement("strong");
  heading.textContent = label;
  row.append(heading);
  if (String(before || "") === String(after || "")) {
    const same = document.createElement("span");
    same.className = "prompt-change-none";
    same.textContent = "No changes";
    row.append(same);
  } else {
    appendPromptDiff(row, before, after);
    if (showTransformationBlocks) row.append(promptTransformationDetails(roleData || {}, before, after));
  }
  return row;
}

function promptCanonicalStructureRow(roleData = {}, inspectorTarget = "") {
  const row = document.createElement("div");
  row.className = "prompt-change-row prompt-canonical-structure-row";
  const heading = document.createElement("strong");
  heading.textContent = "Parser input → Canonical structure";
  row.append(heading);

  const parserInput = promptStageText(roleData, "parser");
  const structure = canonicalStructureForRole(roleData);
  const canonicalSource = canonicalSourceForRole(roleData);
  const normalizedParser = normalizePromptSource(parserInput);
  const sourceChanged = normalizedParser !== String(canonicalSource || "");

  const sourceStatus = document.createElement("p");
  sourceStatus.className = sourceChanged ? "prompt-canonical-source-warning" : "prompt-change-none";
  sourceStatus.textContent = sourceChanged
    ? "Canonicalization changed the normalized source text. Review the source diff below."
    : "No source-text changes. Canonicalization only describes the prompt in a machine-readable structure.";
  row.append(sourceStatus);
  if (sourceChanged) appendPromptDiff(row, normalizedParser, canonicalSource);

  const summary = canonicalStructureSummary(structure);
  const facts = document.createElement("ul");
  facts.className = "prompt-canonical-facts";
  [
    `Contract: ${summary.contract}`,
    `Parser namespace: ${summary.parserNamespace}`,
    `${summary.nodeCount} canonical node${summary.nodeCount === 1 ? "" : "s"}`,
    ...summary.nodeLabels,
  ].forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    facts.append(item);
  });
  row.append(facts);

  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const compactSources = [];
  nodes.forEach((node, index) => {
    const source = typeof node?.source === "string" ? node.source : "";
    const start = Number(node?.start);
    const end = Number(node?.end);
    const exact = Number.isFinite(start) && Number.isFinite(end) && end > start
      ? canonicalSource.slice(start, end)
      : source;
    if (!exact || exact === canonicalSource) return;
    const key = `${node?.type || "node"}:${exact}`;
    if (compactSources.some((item) => item.key === key)) return;
    compactSources.push({ key, label: `${canonicalTypeLabel(node?.type)} ${index + 1}`, text: exact });
  });
  if (compactSources.length) {
    const details = document.createElement("details");
    details.className = "prompt-canonical-recognized";
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = `${compactSources.length} recognized canonical block${compactSources.length === 1 ? "" : "s"}`;
    details.append(detailsSummary);
    compactSources.forEach((item) => details.append(
      makeCopyableTransformationBlock(item.label, item.text, `Copy ${item.label}`),
    ));
    row.append(details);
  }

  if (inspectorTarget) {
    const inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "ui-action-button ui-icon-control prompt-canonical-inspect-button";
    inspect.dataset.promptInspectorTarget = inspectorTarget;
    inspect.dataset.promptInspectorTitle = "Canonical representation";
    setActionIcon(inspect, "maximize", { label: "Inspect canonical representation", replace: true });
    inspect.addEventListener("click", () => openPromptInspector(inspect));
    row.append(inspect);
  }
  return { row, sourceChanged };
}

function semanticTransformationCount(roleData = {}) {
  const substitutions = Array.isArray(roleData.substitutions) ? roleData.substitutions : [];
  if (substitutions.length) return substitutions.length;
  return promptStageText(roleData, "raw") === promptStageText(roleData, "parser") ? 0 : 1;
}

function renderRoleChanges(roleData, listSelector, countSelector, inspectorTarget = "") {
  const list = $(listSelector);
  const count = $(countSelector);
  if (!list || !count) return 0;
  const raw = promptStageText(roleData, "raw");
  const parser = promptStageText(roleData, "parser");
  const rows = [
    promptTransitionRow("Raw → Parser input", raw, parser, { roleData, showTransformationBlocks: true }),
  ];
  const canonical = promptCanonicalStructureRow(roleData, inspectorTarget);
  rows.push(canonical.row);
  list.replaceChildren(...rows);
  const semanticCount = semanticTransformationCount(roleData) + (canonical.sourceChanged ? 1 : 0);
  count.textContent = semanticCount
    ? `${semanticCount} semantic change${semanticCount === 1 ? "" : "s"}`
    : "No semantic changes";
  count.classList.toggle("has-changes", Boolean(semanticCount));
  return semanticCount;
}

function regionBranchCount(passData = {}) {
  return (passData.regional_prompting?.slots || []).reduce((total, slot) => total + (slot.regions || []).length, 0);
}

function canonicalSourceFromSerialized(value) {
  const structure = parseCanonicalValue(value);
  return typeof structure.lossless_source === "string" ? structure.lossless_source : String(value || "");
}

function canonicalStructureSignature(value) {
  const structure = parseCanonicalValue(value);
  if (!Object.keys(structure).length) return String(value || "");
  return JSON.stringify({
    contract: structure.contract || "",
    parser_namespace: structure.parser_namespace || "",
    nodes: Array.isArray(structure.nodes) ? structure.nodes : [],
  });
}

function compactCanonicalDifferenceRow(label, beforeValue, afterValue) {
  const beforeSource = canonicalSourceFromSerialized(beforeValue);
  const afterSource = canonicalSourceFromSerialized(afterValue);
  if (beforeSource !== afterSource) {
    return promptTransitionRow(`${label} source`, beforeSource, afterSource);
  }
  const row = document.createElement("div");
  row.className = "prompt-change-row";
  const heading = document.createElement("strong");
  heading.textContent = `${label} structure`;
  const message = document.createElement("span");
  message.className = "prompt-change-none";
  message.textContent = canonicalStructureSignature(beforeValue) === canonicalStructureSignature(afterValue)
    ? "Canonical serialization differs, but source text and structural interpretation are equivalent."
    : "Source text is unchanged; only the canonical structural interpretation differs between passes.";
  row.append(heading, message);
  return row;
}

function renderHiresChangeSummary(base, hires) {
  const list = $("#promptHiresChanges");
  const count = $("#promptHiresChangeCount");
  const summary = $("#promptHiresInterpretationSummary");
  if (!list || !count) return 0;
  const rows = [];
  if ((base.parser?.parser_id || "") !== (hires.parser?.parser_id || "")) {
    rows.push(promptTransitionRow("Parser", base.parser?.parser_id || "base", hires.parser?.parser_id || "hires"));
  }
  if ((base.shortcut_profile?.profile_id || "") !== (hires.shortcut_profile?.profile_id || "")) {
    rows.push(promptTransitionRow("Shortcut profile", base.shortcut_profile?.profile_id || "base", hires.shortcut_profile?.profile_id || "hires"));
  }
  ["positive", "negative"].forEach((role) => {
    const diff = hires.interpretation_diff?.[role] || {};
    if (!diff.different) return;
    const label = `${role[0].toUpperCase()}${role.slice(1)}`;
    if (String(diff.base_parser_input || "") !== String(diff.hires_parser_input || "")) {
      rows.push(promptTransitionRow(`${label} parser input`, diff.base_parser_input || "", diff.hires_parser_input || ""));
    }
    if (String(diff.base_canonical_prompt || "") !== String(diff.hires_canonical_prompt || "")) {
      rows.push(compactCanonicalDifferenceRow(`${label} canonical`, diff.base_canonical_prompt || "", diff.hires_canonical_prompt || ""));
    }
  });
  if (!rows.length) {
    const same = document.createElement("p");
    same.className = "prompt-change-none";
    same.textContent = "The hires pass uses the same prompt interpretation as the base pass.";
    list.replaceChildren(same);
    count.textContent = "Same as base";
    if (summary) summary.textContent = "Same as base · parser, shortcut profile, parser input, and canonical prompt are unchanged.";
    return 0;
  }
  list.replaceChildren(...rows);
  count.textContent = `${rows.length} difference${rows.length === 1 ? "" : "s"}`;
  count.classList.add("has-changes");
  if (summary) summary.textContent = `${rows.length} hires interpretation difference${rows.length === 1 ? "" : "s"} detected. Review Changes before queueing.`;
  return rows.length;
}

function renderRegionChangeSummary(base, hires) {
  const baseCount = regionBranchCount(base);
  const hiresCount = regionBranchCount(hires);
  const list = $("#promptRegionChanges");
  const count = $("#promptRegionChangeCount");
  const overview = $("#promptPreflightRegionSummary");
  if (list && count) {
    if (!baseCount && !hiresCount) {
      const inactive = document.createElement("p");
      inactive.className = "prompt-change-none";
      inactive.textContent = "No native REGION branches detected; no REGION runtime overhead is estimated.";
      list.replaceChildren(inactive);
      count.textContent = "Inactive";
    } else {
      const active = document.createElement("p");
      active.textContent = `Base: ${baseCount} branch${baseCount === 1 ? "" : "es"} · Hires: ${hiresCount} branch${hiresCount === 1 ? "" : "es"}`;
      list.replaceChildren(active);
      count.textContent = `${baseCount + hiresCount} active`;
      count.classList.add("has-changes");
    }
  }
  if (overview) overview.textContent = baseCount || hiresCount ? `${baseCount + hiresCount} active` : "Inactive";
  const baseCard = $("#promptRegionBaseCard");
  const hiresCard = $("#promptRegionHiresCard");
  if (baseCard) baseCard.hidden = !baseCount;
  if (hiresCard) hiresCard.hidden = !hiresCount;
  return { baseCount, hiresCount };
}

function renderPromptPreflightSummary(data, base, hires) {
  const positiveChanges = renderRoleChanges(base.positive || {}, "#promptPositiveChanges", "#promptPositiveChangeCount", "promptTranslationPositiveCanonical");
  const negativeChanges = renderRoleChanges(base.negative || {}, "#promptNegativeChanges", "#promptNegativeChangeCount", "promptTranslationNegativeCanonical");
  const hiresChanges = renderHiresChangeSummary(base, hires);
  renderRegionChangeSummary(base, hires);
  const setText = (selector, text) => { const node = $(selector); if (node) node.textContent = text; };
  setText("#promptPreflightValidity", data.valid ? "Valid" : "Blocked");
  setText("#promptPreflightPositiveSummary", positiveChanges ? `${positiveChanges} change${positiveChanges === 1 ? "" : "s"}` : "Unchanged");
  setText("#promptPreflightNegativeSummary", negativeChanges ? `${negativeChanges} change${negativeChanges === 1 ? "" : "s"}` : "Unchanged");
  setText("#promptPreflightHiresSummary", hiresChanges ? `${hiresChanges} difference${hiresChanges === 1 ? "" : "s"}` : "Same as base");
}

function setPromptPreflightView(mode = "changes") {
  const changes = mode !== "pipeline";
  const changesView = $("#promptPreflightChangesView");
  const pipelineView = $("#promptPreflightPipelineView");
  const changesButton = $("#promptPreflightChangesTab");
  const pipelineButton = $("#promptPreflightPipelineTab");
  if (changesView) changesView.hidden = !changes;
  if (pipelineView) pipelineView.hidden = changes;
  if (changesButton) {
    changesButton.classList.toggle("is-active", changes);
    changesButton.setAttribute("aria-pressed", String(changes));
  }
  if (pipelineButton) {
    pipelineButton.classList.toggle("is-active", !changes);
    pipelineButton.setAttribute("aria-pressed", String(!changes));
  }
}

let promptInspectorDialog = null;

function ensurePromptInspectorDialog() {
  if (promptInspectorDialog) return promptInspectorDialog;
  const dialog = document.createElement("dialog");
  dialog.className = "prompt-inspector-dialog";
  dialog.innerHTML = `
    <section class="prompt-inspector-dialog-shell">
      <header class="prompt-inspector-dialog-header">
        <div><small>Prompt pipeline inspector</small><h3 data-prompt-inspector-dialog-title>Prompt stage</h3></div>
        <div class="prompt-inspector-dialog-actions">
          <button type="button" data-prompt-inspector-dialog-compare></button>
          <button type="button" data-prompt-inspector-dialog-copy></button>
          <button type="button" data-prompt-inspector-dialog-close></button>
        </div>
      </header>
      <div class="prompt-inspector-dialog-body">
        <section><h4>Selected stage</h4><pre data-prompt-inspector-dialog-primary></pre></section>
        <section data-prompt-inspector-dialog-comparison hidden><h4>Next stage</h4><pre data-prompt-inspector-dialog-secondary></pre></section>
        <section data-prompt-inspector-dialog-diff hidden><h4>Highlighted changes</h4><div class="prompt-inspector-dialog-diff-content"></div></section>
      </div>
    </section>`;
  document.body.append(dialog);
  const compare = dialog.querySelector("[data-prompt-inspector-dialog-compare]");
  const copy = dialog.querySelector("[data-prompt-inspector-dialog-copy]");
  const close = dialog.querySelector("[data-prompt-inspector-dialog-close]");
  setActionIcon(compare, "compare", { label: "Compare with next prompt stage", replace: true });
  setActionIcon(copy, "copy", { label: "Copy prompt stage", replace: true });
  setActionIcon(close, "remove", { label: "Close prompt inspector", replace: true });
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  close.addEventListener("click", () => dialog.close());
  copy.addEventListener("click", async () => {
    const text = dialog.querySelector("[data-prompt-inspector-dialog-primary]")?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      notify("Prompt stage copied.");
    } catch (error) {
      notify(`Unable to copy prompt stage: ${error.message}`, "error");
    }
  });
  compare.addEventListener("click", () => {
    const comparison = dialog.querySelector("[data-prompt-inspector-dialog-comparison]");
    const diff = dialog.querySelector("[data-prompt-inspector-dialog-diff]");
    const visible = comparison?.hidden !== false;
    if (comparison) comparison.hidden = !visible;
    if (diff) diff.hidden = !visible;
    compare.setAttribute("aria-pressed", String(visible));
  });
  promptInspectorDialog = dialog;
  return dialog;
}

function openPromptInspector(button) {
  const target = document.getElementById(button.dataset.promptInspectorTarget || "");
  if (!target) return;
  const compareTarget = document.getElementById(button.dataset.promptInspectorCompareTarget || "");
  const dialog = ensurePromptInspectorDialog();
  const title = button.dataset.promptInspectorTitle || "Prompt stage";
  dialog.querySelector("[data-prompt-inspector-dialog-title]").textContent = title;
  const primary = dialog.querySelector("[data-prompt-inspector-dialog-primary]");
  const secondary = dialog.querySelector("[data-prompt-inspector-dialog-secondary]");
  const comparison = dialog.querySelector("[data-prompt-inspector-dialog-comparison]");
  const diffSection = dialog.querySelector("[data-prompt-inspector-dialog-diff]");
  const diffContent = dialog.querySelector(".prompt-inspector-dialog-diff-content");
  const compareButton = dialog.querySelector("[data-prompt-inspector-dialog-compare]");
  primary.textContent = target.textContent || "";
  secondary.textContent = compareTarget?.textContent || "";
  if (comparison) comparison.hidden = true;
  if (diffSection) diffSection.hidden = true;
  if (compareButton) {
    compareButton.hidden = !compareTarget;
    compareButton.setAttribute("aria-pressed", "false");
  }
  diffContent.replaceChildren();
  if (compareTarget) appendPromptDiff(diffContent, primary.textContent, secondary.textContent);
  if (!dialog.open) dialog.showModal();
}

function bindPromptPreflightInspectors() {
  $("#promptPreflightChangesTab")?.addEventListener("click", () => setPromptPreflightView("changes"));
  $("#promptPreflightPipelineTab")?.addEventListener("click", () => setPromptPreflightView("pipeline"));
  document.querySelectorAll("[data-prompt-inspector-target]").forEach((button) => {
    const title = button.dataset.promptInspectorTitle || "prompt stage";
    button.setAttribute("aria-label", `Open ${title} in a large inspector`);
    button.title = `Open ${title} in a large inspector`;
    button.addEventListener("click", () => openPromptInspector(button));
  });
  setPromptPreflightView("changes");
}

function formatCanonicalForDisplay(role = {}) {
  const structure = canonicalStructureForRole(role);
  if (Object.keys(structure).length) return JSON.stringify(structure, null, 2);
  return promptStageText(role, "canonical");
}

function renderTranslation(data, { revealPreview = false } = {}) {
  state.promptConfiguration.translationPreview = data;
  const set = (selector, value) => { const node = $(selector); if (node) node.textContent = value ?? ""; };
  const base = data.base || data;
  const hires = data.hires || {};
  set("#promptTranslationPositiveRaw", base.positive?.raw_prompt);
  set("#promptTranslationPositiveExpanded", base.positive?.parser_input);
  set("#promptTranslationPositiveCanonical", formatCanonicalForDisplay(base.positive || {}));
  set("#promptTranslationNegativeRaw", base.negative?.raw_prompt);
  set("#promptTranslationNegativeExpanded", base.negative?.parser_input);
  set("#promptTranslationNegativeCanonical", formatCanonicalForDisplay(base.negative || {}));
  set("#promptTranslationHiresPositiveRaw", hires.positive?.raw_prompt);
  set("#promptTranslationHiresPositiveExpanded", hires.positive?.parser_input);
  set("#promptTranslationHiresPositiveCanonical", formatCanonicalForDisplay(hires.positive || {}));
  set("#promptTranslationHiresNegativeRaw", hires.negative?.raw_prompt);
  set("#promptTranslationHiresNegativeExpanded", hires.negative?.parser_input);
  set("#promptTranslationHiresNegativeCanonical", formatCanonicalForDisplay(hires.negative || {}));
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
  const hiresRouteSection = $("#hiresPromptRouteSummarySection");
  if (hiresRouteSection) hiresRouteSection.hidden = !routes.slice(2).some((item) => item && Object.keys(item).length) && !shadows.slice(2).some((item) => item && Object.keys(item).length);
  renderMessageList("#promptPreflightBlockingSection", "#promptPreflightBlocking", data.blocking_errors || []);
  renderMessageList("#promptPreflightWarningSection", "#promptPreflightWarnings", data.behavior_warnings || []);
  renderMessageList("#promptPreflightNoticeSection", "#promptPreflightNotices", data.informational_notices || []);
  const summary = data.valid
    ? `Prompt preflight valid · base ${base.parser?.parser_id || currentParserId()} / ${base.shortcut_profile?.profile_id || "profile"} · hires ${hires.parser?.parser_id || "inherit"} / ${hires.shortcut_profile?.profile_id || "inherit"}`
    : "Prompt preflight contains blocking errors.";
  set("#promptTranslationWarnings", summary);
  renderPromptPreflightSummary(data, base, hires);
  const differs = Object.values(hires.interpretation_diff || {}).some((item) => item?.different);
  const diffWarning = $("#promptHiresInterpretationWarning");
  if (diffWarning) {
    diffWarning.hidden = !differs;
    diffWarning.textContent = differs
      ? "The hires pass resolves at least one prompt differently from the base pass. The differences are summarized above; expand the full hires pipeline only when you need the exact representations."
      : "";
  }
  const hiresPipeline = $("#promptHiresFullPipeline");
  if (hiresPipeline && !differs) hiresPipeline.open = false;
  const details = $("#promptTranslationPreview");
  if (details && revealPreview) {
    details.open = true;
    setPromptPreflightView("changes");
  }
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

const SHORTCUT_OPERATOR_DEFINITIONS = [
  ["AND", "AND", "Composable conditioning branch"],
  ["GROUP_OPEN", "Group open", "Opening group delimiter"],
  ["GROUP_CLOSE", "Group close", "Closing group delimiter"],
  ["SEQUENCE", "Sequence", "Sequence separator"],
  ["DEEP_SEQUENCE", "Deep sequence", "Top-level sequence separator"],
  ["CLOSE", "Close", "Close the current sequence"],
  ["TOP_CLOSE", "Top close", "Close the top-level sequence"],
  ["CHUNK", "Chunk", "Parser 21 chunk operator"],
  ["BLEND", "Blend", "Weighted branch blend"],
  ["BIND", "Bind", "Bind details to a prompt branch"],
  ["POOL", "Pool", "Prompt option pool"],
  ["MORPH", "Morph", "Prompt transition operator"],
  ["ASSEMBLE", "Assemble", "Structured prompt assembly"],
  ["COMPOUND", "Compound", "Compound Parser 21 operation"],
];

let profileEditorBody = {};
let profileEditorReadonly = false;

function editableProfileBody(profile = {}) {
  const keys = ["aliases", "parser_emitters", "escape_character", "description", "credit", "palette"];
  return Object.fromEntries(keys.filter((key) => key in profile).map((key) => [key, structuredClone(profile[key])]));
}

function normalizeProfileEditorBody(profile = {}) {
  const body = editableProfileBody(profile);
  body.aliases = body.aliases && typeof body.aliases === "object" ? body.aliases : {};
  body.parser_emitters = body.parser_emitters && typeof body.parser_emitters === "object" ? body.parser_emitters : {};
  body.escape_character = String(body.escape_character || "\\");
  body.description = String(body.description || "");
  body.credit = String(body.credit || "");
  body.palette = Array.isArray(body.palette) ? body.palette : [];
  return body;
}

function syncProfileEditorBody() {
  const hidden = $("#promptShortcutProfileEditorJson");
  if (hidden) hidden.value = JSON.stringify(profileEditorBody);
}

function operatorDefinitions() {
  const known = new Map(SHORTCUT_OPERATOR_DEFINITIONS.map((item) => [item[0], item]));
  const extras = new Set([
    ...Object.keys(profileEditorBody.aliases || {}),
    ...Object.values(profileEditorBody.parser_emitters || {}).flatMap((values) => Object.keys(values || {})),
  ]);
  for (const key of extras) {
    const operator = String(key || "").trim().toUpperCase();
    if (operator && !known.has(operator)) known.set(operator, [operator, operator, "Custom operator"]);
  }
  return [...known.values()];
}

function aliasValuesFromRow(row) {
  return [...row.querySelectorAll("input[data-shortcut-alias]")]
    .map((input) => String(input.value || "").trim())
    .filter(Boolean);
}

function syncAliasRow(row, operator) {
  const values = aliasValuesFromRow(row);
  if (values.length) profileEditorBody.aliases[operator] = values;
  else delete profileEditorBody.aliases[operator];
  syncProfileEditorBody();
}

function createAliasInput(row, operator, label, value = "", readonly = false) {
  const item = document.createElement("span");
  item.className = "prompt-shortcut-alias-input";
  const input = document.createElement("input");
  input.type = "text";
  input.value = String(value || "");
  input.placeholder = "Shortcut";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.dataset.shortcutAlias = operator;
  input.setAttribute("aria-label", `${label} shortcut`);
  input.disabled = readonly;
  input.addEventListener("input", () => syncAliasRow(row, operator));
  item.append(input);
  if (!readonly) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ui-action-button ui-icon-control ui-action-button--compact";
    setActionIcon(remove, "remove", { label: `Remove ${label} shortcut`, title: `Remove ${label} shortcut`, replace: true });
    remove.addEventListener("click", () => {
      item.remove();
      if (!row.querySelector("input[data-shortcut-alias]")) createAliasInput(row, operator, label, "", false);
      syncAliasRow(row, operator);
    });
    item.append(remove);
  }
  row.querySelector(".prompt-shortcut-alias-inputs")?.append(item);
  return input;
}

function renderShortcutAliasEditor(readonly = false) {
  const host = $("#promptShortcutAliasEditor");
  if (!host) return;
  host.replaceChildren();
  for (const [operator, label, description] of operatorDefinitions()) {
    const row = document.createElement("div");
    row.className = "prompt-shortcut-operator-row";
    row.dataset.operator = operator;
    const heading = document.createElement("div");
    heading.className = "prompt-shortcut-operator-label";
    const strong = document.createElement("strong");
    strong.textContent = label;
    const code = document.createElement("code");
    code.textContent = operator;
    const hint = document.createElement("small");
    hint.textContent = description;
    heading.append(strong, code, hint);
    const inputs = document.createElement("div");
    inputs.className = "prompt-shortcut-alias-inputs";
    row.append(heading, inputs);
    host.append(row);
    const values = Array.isArray(profileEditorBody.aliases?.[operator]) ? profileEditorBody.aliases[operator] : [];
    (values.length ? values : [""]).forEach((value) => createAliasInput(row, operator, label, value, readonly));
    if (!readonly) {
      const add = document.createElement("button");
      add.type = "button";
      add.className = "ui-action-button ui-icon-control ui-action-button--compact prompt-shortcut-add-alias";
      setActionIcon(add, "new", { label: `Add ${label} shortcut`, title: `Add another ${label} shortcut`, replace: true });
      add.addEventListener("click", () => {
        const input = createAliasInput(row, operator, label, "", false);
        input.focus();
      });
      row.append(add);
    }
  }
}

function paletteShortcutFields(item = {}) {
  const kind = String(item.kind || "insert").toLowerCase();
  if (kind === "wrap") return [["prefix", "Opening text"], ["suffix", "Closing text"]];
  if (kind === "template") return [["template", "Template"]];
  return [["insert", "Inserted text"]];
}

function renderShortcutPaletteEditor(readonly = false) {
  const host = $("#promptShortcutPaletteEditor");
  if (!host) return;
  host.replaceChildren();
  const palette = Array.isArray(profileEditorBody.palette) ? profileEditorBody.palette : [];
  if (!palette.length) {
    const empty = document.createElement("p");
    empty.className = "field-status subtle";
    empty.textContent = "This profile does not define symbol palette helpers.";
    host.append(empty);
    return;
  }
  palette.forEach((item, index) => {
    const row = document.createElement("section");
    row.className = "prompt-shortcut-palette-row";
    const heading = document.createElement("div");
    heading.className = "prompt-shortcut-palette-label";
    const strong = document.createElement("strong");
    strong.textContent = String(item.label || item.id || item.operator || `Helper ${index + 1}`);
    const code = document.createElement("code");
    code.textContent = String(item.operator || item.id || "palette");
    heading.append(strong, code);
    row.append(heading);

    const fields = document.createElement("div");
    fields.className = "prompt-shortcut-palette-fields";
    for (const [key, label] of paletteShortcutFields(item)) {
      const field = document.createElement("label");
      field.className = "field-block";
      const caption = document.createElement("span");
      caption.textContent = label;
      const input = document.createElement("input");
      input.type = "text";
      input.value = String(item[key] || "");
      input.autocomplete = "off";
      input.spellcheck = false;
      input.disabled = readonly;
      input.setAttribute("aria-label", `${strong.textContent} ${label.toLowerCase()}`);
      input.addEventListener("input", () => {
        profileEditorBody.palette[index][key] = String(input.value || "");
        syncProfileEditorBody();
      });
      field.append(caption, input);
      fields.append(field);
    }

    const labelField = document.createElement("label");
    labelField.className = "field-block";
    const labelCaption = document.createElement("span");
    labelCaption.textContent = "Palette label";
    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.value = String(item.label || "");
    labelInput.disabled = readonly;
    labelInput.addEventListener("input", () => {
      profileEditorBody.palette[index].label = String(labelInput.value || "");
      strong.textContent = String(labelInput.value || item.id || item.operator || `Helper ${index + 1}`);
      syncProfileEditorBody();
    });
    labelField.append(labelCaption, labelInput);
    fields.append(labelField);

    row.append(fields);
    host.append(row);
  });
}

function compatibleEditorParsers() {
  const requested = String($("#promptShortcutProfileEditorParsers")?.value || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return [...new Set([...requested, ...Object.keys(profileEditorBody.parser_emitters || {})])];
}

function renderShortcutEmitterEditor(readonly = false) {
  const host = $("#promptShortcutEmitterEditor");
  if (!host) return;
  host.replaceChildren();
  const parsers = compatibleEditorParsers();
  for (const parser of parsers) {
    if (!profileEditorBody.parser_emitters[parser]) profileEditorBody.parser_emitters[parser] = {};
    const group = document.createElement("section");
    group.className = "prompt-shortcut-emitter-group";
    const title = document.createElement("h4");
    title.textContent = parser;
    group.append(title);
    const grid = document.createElement("div");
    grid.className = "prompt-shortcut-emitter-grid";
    for (const [operator, label] of operatorDefinitions()) {
      const field = document.createElement("label");
      field.className = "field-block prompt-shortcut-emitter-field";
      const caption = document.createElement("span");
      caption.textContent = label;
      const input = document.createElement("input");
      input.type = "text";
      input.value = String(profileEditorBody.parser_emitters?.[parser]?.[operator] || "");
      input.placeholder = operator;
      input.autocomplete = "off";
      input.spellcheck = false;
      input.disabled = readonly;
      input.setAttribute("aria-label", `${parser} output mapping for ${label}`);
      input.addEventListener("input", () => {
        const value = String(input.value || "").trim();
        if (value) profileEditorBody.parser_emitters[parser][operator] = value;
        else delete profileEditorBody.parser_emitters[parser][operator];
        syncProfileEditorBody();
      });
      field.append(caption, input);
      grid.append(field);
    }
    group.append(grid);
    host.append(group);
  }
  syncProfileEditorBody();
}

function editorPayload() {
  syncProfileEditorBody();
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
    description: "",
    credit: "",
    palette: [],
  };
  profileEditorBody = normalizeProfileEditorBody(selected);
  profileEditorReadonly = Boolean(selected.builtin);
  $("#promptShortcutProfileEditorId").value = selected.profile_id || "";
  $("#promptShortcutProfileEditorLabel").value = selected.label || "";
  $("#promptShortcutProfileEditorVersion").value = selected.version || "1";
  $("#promptShortcutProfileEditorParsers").value = (selected.compatible_parsers || []).join(", ");
  $("#promptShortcutProfileEditorEscape").value = profileEditorBody.escape_character;
  $("#promptShortcutProfileEditorDescription").value = profileEditorBody.description;
  $("#promptShortcutProfileEditorCredit").value = profileEditorBody.credit;
  syncProfileEditorBody();
  [
    "#promptShortcutProfileEditorId",
    "#promptShortcutProfileEditorLabel",
    "#promptShortcutProfileEditorVersion",
    "#promptShortcutProfileEditorParsers",
    "#promptShortcutProfileEditorEscape",
    "#promptShortcutProfileEditorDescription",
    "#promptShortcutProfileEditorCredit",
  ].forEach((selector) => {
    const node = $(selector); if (node) node.disabled = profileEditorReadonly;
  });
  renderShortcutAliasEditor(profileEditorReadonly);
  renderShortcutPaletteEditor(profileEditorReadonly);
  renderShortcutEmitterEditor(profileEditorReadonly);
  $("#savePromptShortcutProfileButton").disabled = profileEditorReadonly;
  $("#deletePromptShortcutProfileButton").disabled = profileEditorReadonly;
  $("#promptShortcutProfileValidation").textContent = profileEditorReadonly
    ? "Built-in profile: duplicate it to edit."
    : "Edit shortcuts inline, then validate before saving.";
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
  ["#promptShortcutProfileTopActionBar", "#promptShortcutProfileBottomActionBar"].forEach((selector) => {
    $(selector)?.dispatchEvent(new CustomEvent("ui-action-bar-refresh"));
  });
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
  return normalizeHiresSizeMode(value, enabled);
}

function normalizedHiresDimension(value, fallback) {
  return clampHiresDimension(value, fallback);
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

function scheduleHiresPlannerParity(plan, enabled) {
  const toggle = $("#hiresPlannerParityDiagnostics");
  const status = $("#hiresPlannerParityStatus");
  if (hiresPlannerParityTimer) window.clearTimeout(hiresPlannerParityTimer);
  if (!toggle?.checked || !enabled) {
    if (status) {
      status.textContent = toggle?.checked ? "Planner parity diagnostic is waiting for hires to be enabled." : "Planner parity diagnostic is off.";
      status.className = "field-status subtle";
    }
    return;
  }
  const sequence = ++hiresPlannerParitySequence;
  if (status) {
    status.textContent = `Planner parity diagnostic: checking browser plan ${plan.contract_version || "unknown"} against server…`;
    status.className = "field-status subtle";
  }
  hiresPlannerParityTimer = window.setTimeout(async () => {
    try {
      const response = await api.hiresDimensionPlan({
        width: plan.base_width,
        height: plan.base_height,
        hires_size_mode: plan.mode,
        hires_scale: plan.requested_scale,
        hires_width: plan.requested_width,
        hires_height: plan.requested_height,
      });
      if (sequence !== hiresPlannerParitySequence) return;
      const server = response?.plan || {};
      const keys = [
        "contract_version", "mode", "base_width", "base_height", "requested_width",
        "requested_height", "internal_width", "internal_height", "final_width",
        "final_height", "axis_scale_width", "axis_scale_height", "uniform_scale",
        "aspect_ratio_changed", "alignment_applied", "dimension_multiple",
      ];
      const mismatches = keys.filter((key) => JSON.stringify(server[key]) !== JSON.stringify(plan[key]));
      if (status) {
        status.textContent = mismatches.length
          ? `Planner parity mismatch: ${mismatches.join(", ")} · browser ${plan.contract_version || "unknown"} / server ${response?.contract_version || server.contract_version || "unknown"}.`
          : `Planner parity: Match · ${plan.contract_version || "unknown"} · alignment multiple ${plan.dimension_multiple}.`;
        status.className = mismatches.length ? "field-status error" : "field-status ready";
      }
    } catch (error) {
      if (sequence !== hiresPlannerParitySequence) return;
      if (status) {
        status.textContent = `Planner parity diagnostic failed: ${error.message}`;
        status.className = "field-status error";
      }
    }
  }, 300);
}

function updateHiresSizeControls({ source = "" } = {}) {
  const enabled = $("#hiresEnabled")?.checked === true;
  const sizeMode = $("#hiresSizeMode");
  let mode = normalizedHiresSizeMode(sizeMode?.value, enabled);
  const scaleField = $("#hiresScaleField");
  const widthField = $("#hiresWidthField");
  const heightField = $("#hiresHeightField");
  const scaleInput = $("#hiresScale");
  const widthInput = $("#hiresWidth");
  const heightInput = $("#hiresHeight");

  if (enabled && mode === "scale_from_base" && (source === "width" || source === "height")) {
    mode = "explicit_dimensions";
    if (sizeMode) sizeMode.value = mode;
  } else if (sizeMode && sizeMode.value !== mode) {
    sizeMode.value = mode;
  }

  [
    "#hiresPromptParserMode", "#hiresPromptParserName", "#hiresShortcutProfileMode",
    "#hiresShortcutProfileName", "#hiresPositivePrompt", "#hiresNegativePrompt",
    "#hiresSizeMode", "#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy",
    "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale",
    "#hiresUpscaler", "#hiresSaveLowres", "#hiresAspectPolicy", "#hiresPaddingMode",
    "#hiresBlurredEdgeMethod", "#hiresFinalSizeCorrectionFilter", "#hiresCorrectionFingerprintDiagnostics",
    "#hiresBlurredEdgeCompareDiagnostics",
  ].forEach((selector) => {
    const node = $(selector);
    if (node) node.disabled = !enabled;
  });

  if (scaleInput) scaleInput.disabled = !enabled || mode !== "scale_from_base";
  if (widthInput) widthInput.disabled = !enabled;
  if (heightInput) heightInput.disabled = !enabled;
  scaleField?.classList.toggle("is-disabled", !enabled || mode !== "scale_from_base");
  widthField?.classList.toggle("is-disabled", !enabled);
  heightField?.classList.toggle("is-disabled", !enabled);
  widthField?.classList.toggle("is-calculated", enabled && mode === "scale_from_base");
  heightField?.classList.toggle("is-calculated", enabled && mode === "scale_from_base");

  const plan = planHiresDimensions({
    baseWidth: $("#width")?.value,
    baseHeight: $("#height")?.value,
    mode,
    scale: scaleInput?.value || 1.5,
    targetWidth: widthInput?.value,
    targetHeight: heightInput?.value,
    enabled,
  });

  if (enabled && mode === "scale_from_base") {
    if (widthInput) widthInput.value = String(plan.requested_width);
    if (heightInput) heightInput.value = String(plan.requested_height);
  } else if (enabled && mode === "explicit_dimensions") {
    if (plan.uniform_scale !== null && scaleInput) {
      scaleInput.value = String(plan.uniform_scale);
    } else if (scaleInput) {
      scaleInput.value = "";
    }
  }

  const status = $("#hiresSizeStatus");
  if (status) status.textContent = enabled
    ? `Requested target: ${plan.requested_width} × ${plan.requested_height} · ${mode === "scale_from_base" ? "scale from base" : "exact target dimensions"}.`
    : "Hires generation is disabled.";

  const effectiveStatus = $("#hiresEffectiveScaleStatus");
  if (effectiveStatus) {
    if (!enabled) {
      effectiveStatus.textContent = "Effective scaling: disabled.";
      effectiveStatus.className = "field-status subtle";
    } else if (plan.is_uniform_scale) {
      effectiveStatus.textContent = `Effective scaling: Uniform ${Number(plan.uniform_scale).toFixed(6)}x.`;
      effectiveStatus.className = "field-status subtle";
    } else {
      effectiveStatus.textContent = `Effective scaling: Width ${plan.axis_scale_width.toFixed(6)}x / Height ${plan.axis_scale_height.toFixed(6)}x · Aspect ratio changed.`;
      effectiveStatus.className = "field-status warning";
    }
  }

  const alignmentStatus = $("#hiresAlignmentStatus");
  if (alignmentStatus) {
    alignmentStatus.textContent = enabled
      ? `Dimension plan: Requested ${plan.requested_width} × ${plan.requested_height} · Internal aligned ${plan.internal_width} × ${plan.internal_height} · Final target ${plan.final_width} × ${plan.final_height}${plan.alignment_applied ? " · alignment correction required" : " · no alignment correction"}.`
      : "Dimension plan: disabled.";
    alignmentStatus.className = plan.alignment_applied && enabled ? "field-status warning" : "field-status subtle";
  }

  updateHiresUpscalerPlanUI(plan, enabled);
  scheduleHiresPlannerParity(plan, enabled);
  const correctionFingerprintStatus = $("#hiresCorrectionFingerprintStatus");
  if (correctionFingerprintStatus) {
    const fingerprintEnabled = Boolean($("#hiresCorrectionFingerprintDiagnostics")?.checked);
    correctionFingerprintStatus.textContent = fingerprintEnabled
      ? "Correction fingerprint enabled: a deterministic metadata-only SHA-256 will be recorded for this hires correction contract."
      : "Correction fingerprint is off. Enabling it hashes only the deterministic correction contract; it does not add an image-processing pass.";
    correctionFingerprintStatus.className = fingerprintEnabled ? "field-status ready" : "field-status subtle";
  }
  const blurredEdgeCompareStatus = $("#hiresBlurredEdgeCompareStatus");
  if (blurredEdgeCompareStatus) {
    const compareEnabled = Boolean($("#hiresBlurredEdgeCompareDiagnostics")?.checked);
    blurredEdgeCompareStatus.textContent = compareEnabled
      ? "Blurred-edge comparison diagnostic enabled: the selected method and the alternate method will both run so diagnostics can record timing and quality-proxy deltas."
      : "Blurred-edge comparison diagnostic is off. Enable it only when you want side-by-side timing and quality-proxy metadata.";
    blurredEdgeCompareStatus.className = compareEnabled ? "field-status ready" : "field-status subtle";
  }
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
  if ($("#hiresUpscaler")) $("#hiresUpscaler").value = current.hires_upscaler_id || current.hires_upscaler || "";
  if ($("#hiresAspectPolicy")) $("#hiresAspectPolicy").value = current.hires_aspect_policy || "stretch";
  if ($("#hiresPaddingMode")) $("#hiresPaddingMode").value = current.hires_padding_mode || "reflect";
  if ($("#hiresBlurredEdgeMethod")) $("#hiresBlurredEdgeMethod").value = current.hires_blurred_edge_method || "box";
  if ($("#hiresFinalSizeCorrectionFilter")) $("#hiresFinalSizeCorrectionFilter").value = current.hires_final_size_correction_filter || current.hires_exact_resize_filter || "auto";
  if ($("#hiresPlannerParityDiagnostics")) $("#hiresPlannerParityDiagnostics").checked = false;
  if ($("#hiresCorrectionFingerprintDiagnostics")) $("#hiresCorrectionFingerprintDiagnostics").checked = Boolean(current.hires_correction_fingerprint_enabled);
  if ($("#hiresBlurredEdgeCompareDiagnostics")) $("#hiresBlurredEdgeCompareDiagnostics").checked = Boolean(current.hires_blurred_edge_compare_diagnostics);
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
  bindPromptPreflightInspectors();
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
  const bindHiresDimensionEvent = (selector, source) => {
    $(selector)?.addEventListener("input", () => { updateHiresSizeControls({ source }); saveSessionSoon(); });
    $(selector)?.addEventListener("change", () => { updateHiresSizeControls({ source }); saveSessionSoon(); });
  };
  bindHiresDimensionEvent("#hiresScale", "scale");
  bindHiresDimensionEvent("#hiresWidth", "width");
  bindHiresDimensionEvent("#hiresHeight", "height");
  bindHiresDimensionEvent("#hiresSizeMode", "mode");
  bindHiresDimensionEvent("#hiresEnabled", "enabled");
  bindHiresDimensionEvent("#width", "base");
  bindHiresDimensionEvent("#height", "base");
  ["#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy", "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale", "#hiresUpscaler", "#hiresAspectPolicy", "#hiresPaddingMode", "#hiresBlurredEdgeMethod", "#hiresFinalSizeCorrectionFilter", "#hiresSaveLowres", "#samplerName", "#schedulerName", "#cfgScale"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => { updateHiresSizeControls(); saveSessionSoon(); });
    $(selector)?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  });
  $("#hiresPlannerParityDiagnostics")?.addEventListener("change", () => updateHiresSizeControls());
  $("#hiresCorrectionFingerprintDiagnostics")?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  $("#hiresBlurredEdgeCompareDiagnostics")?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  window.addEventListener("image-gen-hires-upscaler-change", () => updateHiresSizeControls());
  $("#savePromptParserPresetButton")?.addEventListener("click", saveParserPreset);
  $("#deletePromptParserPresetButton")?.addEventListener("click", deleteParserPreset);
  $("#validateCurrentPromptButton")?.addEventListener("click", validateCurrentPrompt);
  $("#editPromptShortcutProfilesButton")?.addEventListener("click", openProfileEditor);
  $("#promptShortcutProfileEditorSelect")?.addEventListener("change", (event) => loadProfileEditor(profileById(event.target.value)));
  $("#promptShortcutProfileEditorParsers")?.addEventListener("change", () => {
    if (!profileEditorReadonly) renderShortcutEmitterEditor(false);
  });
  $("#promptShortcutProfileEditorEscape")?.addEventListener("input", (event) => {
    profileEditorBody.escape_character = String(event.target.value || "\\");
    syncProfileEditorBody();
  });
  $("#promptShortcutProfileEditorDescription")?.addEventListener("input", (event) => {
    profileEditorBody.description = String(event.target.value || "");
    syncProfileEditorBody();
  });
  $("#promptShortcutProfileEditorCredit")?.addEventListener("input", (event) => {
    profileEditorBody.credit = String(event.target.value || "");
    syncProfileEditorBody();
  });
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
    hires_upscaler: $("#hiresUpscaler")?.value || "",
    hires_save_lowres: $("#hiresSaveLowres")?.checked !== false,
  });
}
