import { $, $$, notify } from "../utils.js";

const TOKEN_DEFINITIONS = [
  { key: "index", label: "Index", value: "{index:05d}" },
  { key: "seed", label: "Seed", value: "{seed}" },
  { key: "date", label: "Date", value: "{date}" },
  { key: "time", label: "Time", value: "{time}" },
  { key: "datetime", label: "Datetime", value: "{datetime}" },
  { key: "model", label: "Model", value: "{model}" },
  { key: "vae", label: "VAE", value: "{vae}" },
  { key: "lora", label: "LoRA", value: "{lora}" },
  { key: "sampler", label: "Sampler", value: "{sampler}" },
  { key: "scheduler", label: "Scheduler", value: "{scheduler}" },
  { key: "width", label: "Width", value: "{width}" },
  { key: "height", label: "Height", value: "{height}" },
  { key: "steps", label: "Steps", value: "{steps}" },
  { key: "cfg_scale", label: "CFG", value: "{cfg_scale}" },
  { key: "prompt", label: "Prompt", value: "{prompt}" },
  { key: "prefix", label: "Prefix", value: "{prefix}" },
];

const TOKEN_BY_VALUE = new Map(TOKEN_DEFINITIONS.map((item) => [item.value, item]));
const TOKEN_BY_KEY = new Map(TOKEN_DEFINITIONS.map((item) => [item.key, item]));
const DEFAULT_PATTERN = "{index:05d}-{seed}";

let state = {
  tokens: [],
  separators: [""],
};

let nodes = {};
let dragIndex = null;
let syncingFromInput = false;
let syncingFromBuilder = false;

function cloneState() {
  return {
    tokens: [...state.tokens],
    separators: [...state.separators],
  };
}

function normalizeState(nextState) {
  const tokens = Array.isArray(nextState?.tokens) ? [...nextState.tokens] : [];
  const separators = Array.isArray(nextState?.separators) ? [...nextState.separators] : [""];
  while (separators.length < tokens.length + 1) separators.push("");
  while (separators.length > tokens.length + 1) separators.pop();
  if (!separators.length) separators.push("");
  return { tokens, separators };
}

function parsePattern(pattern) {
  const text = String(pattern || "");
  const regex = /\{[^{}]+\}/g;
  const tokens = [];
  const separators = [""];
  let lastIndex = 0;
  for (const match of text.matchAll(regex)) {
    const matchText = match[0];
    const start = match.index || 0;
    separators[separators.length - 1] += text.slice(lastIndex, start);
    const definition = TOKEN_BY_VALUE.get(matchText);
    if (definition) {
      tokens.push(definition.key);
      separators.push("");
    } else {
      separators[separators.length - 1] += matchText;
    }
    lastIndex = start + matchText.length;
  }
  separators[separators.length - 1] += text.slice(lastIndex);
  return normalizeState({ tokens, separators });
}

function buildPattern() {
  const parts = [];
  for (let index = 0; index < state.tokens.length; index += 1) {
    parts.push(state.separators[index] || "");
    const token = TOKEN_BY_KEY.get(state.tokens[index]);
    parts.push(token?.value || "");
  }
  parts.push(state.separators[state.tokens.length] || "");
  return parts.join("") || DEFAULT_PATTERN;
}

function dispatchPatternChanged() {
  const input = nodes.patternInput;
  if (!input) return;
  syncingFromBuilder = true;
  input.value = buildPattern();
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
  syncingFromBuilder = false;
}

function applyState(nextState, { renderOnly = false } = {}) {
  state = normalizeState(nextState);
  renderBuilder();
  if (!renderOnly) dispatchPatternChanged();
}

function insertToken(tokenKey) {
  const token = TOKEN_BY_KEY.get(tokenKey);
  if (!token) return;
  const next = cloneState();
  const insertionIndex = next.tokens.length;
  next.tokens.push(token.key);
  if (next.tokens.length > 1) {
    next.separators.splice(insertionIndex, 0, nodes.defaultSpacer?.value ?? "_");
  }
  next.separators = normalizeState(next).separators;
  applyState(next);
}

function removeToken(index) {
  if (index < 0 || index >= state.tokens.length) return;
  const next = cloneState();
  const left = next.separators[index] || "";
  const right = next.separators[index + 1] || "";
  next.tokens.splice(index, 1);
  next.separators.splice(index, 2, `${left}${right}`);
  applyState(next);
}

function moveToken(fromIndex, toIndex) {
  if (fromIndex === toIndex || fromIndex == null || toIndex == null) return;
  const next = cloneState();
  const [token] = next.tokens.splice(fromIndex, 1);
  next.tokens.splice(toIndex, 0, token);
  applyState(next);
}

function clearPattern() {
  const resetPattern = parsePattern(DEFAULT_PATTERN);
  if (nodes.defaultSpacer) nodes.defaultSpacer.value = "_";
  applyState(resetPattern);
}

function refreshFromInput() {
  if (!nodes.patternInput) return;
  syncingFromInput = true;
  applyState(parsePattern(nodes.patternInput.value || DEFAULT_PATTERN), { renderOnly: true });
  syncingFromInput = false;
}

function updateSeparator(index, value) {
  const next = cloneState();
  next.separators[index] = value;
  applyState(next);
}

function renderSeparator(index, label) {
  const wrapper = document.createElement("label");
  wrapper.className = "output-pattern-gap";

  const caption = document.createElement("span");
  caption.textContent = label;
  wrapper.append(caption);

  const input = document.createElement("input");
  input.type = "text";
  input.className = "output-pattern-gap-input";
  input.value = state.separators[index] || "";
  input.placeholder = index === 0 ? "" : (nodes.defaultSpacer?.value || "_");
  input.addEventListener("input", (event) => updateSeparator(index, event.target.value));
  wrapper.append(input);
  return wrapper;
}

function renderToken(index, tokenKey) {
  const definition = TOKEN_BY_KEY.get(tokenKey);
  const chip = document.createElement("div");
  chip.className = "output-pattern-chip";
  chip.draggable = true;
  chip.dataset.tokenIndex = String(index);

  const grip = document.createElement("button");
  grip.type = "button";
  grip.className = "output-pattern-chip-grip";
  grip.textContent = "↕";
  grip.title = `Drag to reorder ${definition?.label || tokenKey}`;
  chip.append(grip);

  const label = document.createElement("button");
  label.type = "button";
  label.className = "output-pattern-chip-label";
  label.textContent = definition?.label || tokenKey;
  label.title = definition?.value || tokenKey;
  chip.append(label);

  const template = document.createElement("small");
  template.textContent = definition?.value || "";
  chip.append(template);

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "output-pattern-chip-remove";
  remove.textContent = "×";
  remove.title = `Remove ${definition?.label || tokenKey}`;
  remove.addEventListener("click", () => removeToken(index));
  chip.append(remove);

  chip.addEventListener("dragstart", () => {
    dragIndex = index;
    chip.classList.add("is-dragging");
  });
  chip.addEventListener("dragend", () => {
    dragIndex = null;
    chip.classList.remove("is-dragging");
    $$(".output-pattern-chip").forEach((item) => item.classList.remove("is-drag-target"));
  });
  chip.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (dragIndex == null || dragIndex === index) return;
    chip.classList.add("is-drag-target");
  });
  chip.addEventListener("dragleave", () => chip.classList.remove("is-drag-target"));
  chip.addEventListener("drop", (event) => {
    event.preventDefault();
    chip.classList.remove("is-drag-target");
    if (dragIndex == null) return;
    moveToken(dragIndex, index);
  });
  return chip;
}

function renderBuilder() {
  const canvas = nodes.builderCanvas;
  if (!canvas) return;
  canvas.replaceChildren();

  const hasTokens = state.tokens.length > 0;
  if (!hasTokens) {
    const empty = document.createElement("p");
    empty.className = "output-pattern-empty";
    empty.textContent = "Add filename tokens below. You can drag them to reorder, remove them, and edit the separators before, between, and after each token.";
    canvas.append(empty);
    canvas.append(renderSeparator(0, "Whole pattern / literal text"));
    return;
  }

  canvas.append(renderSeparator(0, "Prefix"));
  state.tokens.forEach((tokenKey, index) => {
    canvas.append(renderToken(index, tokenKey));
    const label = index === state.tokens.length - 1 ? "Suffix" : `Spacer ${index + 1}`;
    canvas.append(renderSeparator(index + 1, label));
  });
}

function bindPalette() {
  const palette = nodes.palette;
  if (!palette) return;
  palette.replaceChildren(...TOKEN_DEFINITIONS.map((token) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary-button compact-button output-pattern-token-button";
    button.textContent = token.label;
    button.title = `Insert ${token.value}`;
    button.addEventListener("click", () => insertToken(token.key));
    return button;
  }));
}

function bindDefaultSpacer() {
  nodes.defaultSpacer?.addEventListener("input", () => {
    $$(".output-pattern-gap-input", nodes.builderCanvas || document).forEach((input) => {
      if (!input.value && input.placeholder !== undefined) {
        input.placeholder = nodes.defaultSpacer?.value || "_";
      }
    });
  });
}

export function bindOutputPatternBuilder() {
  nodes = {
    patternInput: $("#outputPrefix"),
    palette: $("#outputPatternPalette"),
    builderCanvas: $("#outputPatternBuilder"),
    defaultSpacer: $("#outputPatternDefaultSpacer"),
    clearButton: $("#outputPatternClearButton"),
    syncButton: $("#outputPatternSyncButton"),
  };
  if (!nodes.patternInput || !nodes.palette || !nodes.builderCanvas) return;

  bindPalette();
  bindDefaultSpacer();

  const initialPattern = nodes.patternInput.value || DEFAULT_PATTERN;
  if (!nodes.defaultSpacer?.value) nodes.defaultSpacer.value = "_";
  state = parsePattern(initialPattern);
  renderBuilder();

  nodes.patternInput.addEventListener("input", () => {
    if (syncingFromBuilder) return;
    refreshFromInput();
  });
  nodes.patternInput.addEventListener("change", () => {
    if (syncingFromBuilder || syncingFromInput) return;
    refreshFromInput();
  });

  nodes.syncButton?.addEventListener("click", () => {
    refreshFromInput();
    notify("Filename pattern builder synced from the text field.");
  });
  nodes.clearButton?.addEventListener("click", () => {
    clearPattern();
    notify("Filename pattern reset to the default builder layout.");
  });
}
