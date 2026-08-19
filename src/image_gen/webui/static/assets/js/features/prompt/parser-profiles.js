import { api } from "../../api.js";
import { state } from "../../state.js";
import { $, notify } from "../../utils.js";
import { saveSessionSoon } from "./runtime.js";
import { compatibleProfiles, currentParserId, currentProfile, defaultProfileId, option, presetById, safeParseJson, setParserKwargs, setSnapshot, slug } from "./shared.js";
import { parserDefaultOptions, renderBaseParserSettings } from "./parser-settings.js";
import { updateHiresRouting } from "./hires-prompt.js";
import { renderPalette } from "./symbol-palette.js";

export function populateParsers(selected = "legacy") {
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

export function populateProfiles(selected = "") {
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

export function populatePresets(selected = "") {
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

export function updatePresetDeleteState() {
  const preset = presetById($("#promptParserPresetName")?.value);
  const button = $("#deletePromptParserPresetButton");
  if (button) button.disabled = !preset || Boolean(preset.builtin);
}

export function applyPreset(preset) {
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

export async function saveParserPreset() {
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

export async function deleteParserPreset() {
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

export function bindParserProfiles() {
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
  $("#savePromptParserPresetButton")?.addEventListener("click", saveParserPreset);
  $("#deletePromptParserPresetButton")?.addEventListener("click", deleteParserPreset);
}
