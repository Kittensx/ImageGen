import { api } from "../../api.js";
import { state } from "../../state.js";
import { $, notify } from "../../utils.js";
import { saveSessionSoon } from "./runtime.js";
import { currentParserId, defaultProfileId, option, profileById, safeParseJson, slug } from "./shared.js";
import { populateProfiles } from "./parser-profiles.js";
import { renderPalette } from "./symbol-palette.js";

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

export function editableProfileBody(profile = {}) {
  const keys = ["aliases", "parser_emitters", "escape_character", "description", "credit", "palette"];
  return Object.fromEntries(keys.filter((key) => key in profile).map((key) => [key, structuredClone(profile[key])]));
}

export function normalizeProfileEditorBody(profile = {}) {
  const body = editableProfileBody(profile);
  body.aliases = body.aliases && typeof body.aliases === "object" ? body.aliases : {};
  body.parser_emitters = body.parser_emitters && typeof body.parser_emitters === "object" ? body.parser_emitters : {};
  body.escape_character = String(body.escape_character || "\\");
  body.description = String(body.description || "");
  body.credit = String(body.credit || "");
  body.palette = Array.isArray(body.palette) ? body.palette : [];
  return body;
}

export function syncProfileEditorBody() {
  const hidden = $("#promptShortcutProfileEditorJson");
  if (hidden) hidden.value = JSON.stringify(profileEditorBody);
}

export function operatorDefinitions() {
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

export function aliasValuesFromRow(row) {
  return [...row.querySelectorAll("input[data-shortcut-alias]")]
    .map((input) => String(input.value || "").trim())
    .filter(Boolean);
}

export function syncAliasRow(row, operator) {
  const values = aliasValuesFromRow(row);
  if (values.length) profileEditorBody.aliases[operator] = values;
  else delete profileEditorBody.aliases[operator];
  syncProfileEditorBody();
}

export function createAliasInput(row, operator, label, value = "", readonly = false) {
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

export function renderShortcutAliasEditor(readonly = false) {
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

export function paletteShortcutFields(item = {}) {
  const kind = String(item.kind || "insert").toLowerCase();
  if (kind === "wrap") return [["prefix", "Opening text"], ["suffix", "Closing text"]];
  if (kind === "template") return [["template", "Template"]];
  return [["insert", "Inserted text"]];
}

export function renderShortcutPaletteEditor(readonly = false) {
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

export function compatibleEditorParsers() {
  const requested = String($("#promptShortcutProfileEditorParsers")?.value || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return [...new Set([...requested, ...Object.keys(profileEditorBody.parser_emitters || {})])];
}

export function renderShortcutEmitterEditor(readonly = false) {
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

export function editorPayload() {
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

export function loadProfileEditor(profile) {
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

export function populateProfileEditorSelect(selectedId = "") {
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

export function openProfileEditor() {
  populateProfileEditorSelect($("#promptShortcutProfileName")?.value || "");
  $("#promptShortcutProfileDialog")?.showModal();
  ["#promptShortcutProfileTopActionBar", "#promptShortcutProfileBottomActionBar"].forEach((selector) => {
    $(selector)?.dispatchEvent(new CustomEvent("ui-action-bar-refresh"));
  });
}

export async function validateProfileEditor({ quiet = false } = {}) {
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

export async function saveProfileEditor() {
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

export async function deleteProfileEditor() {
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

export function duplicateProfileEditor() {
  const profile = profileById($("#promptShortcutProfileEditorSelect")?.value);
  if (!profile) return;
  const copy = structuredClone(profile);
  copy.profile_id = `${profile.profile_id}_copy`;
  copy.label = `${profile.label} Copy`;
  copy.builtin = false;
  copy.source = "user";
  loadProfileEditor(copy);
}

export function newProfileEditor() {
  loadProfileEditor(null);
}

export function exportProfileEditor() {
  const payload = editorPayload();
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${slug(payload.profile_id)}.prompt-shortcuts.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

export async function importProfileFile(file) {
  try {
    const payload = JSON.parse(await file.text());
    loadProfileEditor({ ...payload, builtin: false, source: "user" });
    await validateProfileEditor({ quiet: true });
  } catch (error) {
    notify(`Unable to import shortcut profile: ${error.message}`, "error");
  }
}

export function bindProfileEditor() {
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
