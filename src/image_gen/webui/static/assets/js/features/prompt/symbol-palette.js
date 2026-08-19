import { state } from "../../state.js";
import { $ } from "../../utils.js";
import { saveSessionSoon } from "./runtime.js";
import { currentParserId, currentProfile, parserById } from "./shared.js";

export function aliasFor(profile, operator, fallback = "") {
  const aliases = profile?.aliases?.[operator] || [];
  return String(aliases[0] || fallback || operator);
}

export function resolvedPaletteItem(item, profile) {
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

export function renderPalette() {
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

export function targetTextarea() {
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

export function insertPaletteItem(item) {
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

export function bindPromptFocus() {
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
