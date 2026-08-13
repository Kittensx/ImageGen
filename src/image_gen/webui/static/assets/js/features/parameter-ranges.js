import { $, notify } from "../utils.js";

const CFG_FIELD_MAP = {
  cfgRescaleNumber: "cfg_rescale",
  cfgCurveStrengthNumber: "sampler_kwargs.cfg_curve_strength",
  cfgHighSigmaBoostNumber: "sampler_kwargs.cfg_high_sigma_boost",
  cfgLateTaperNumber: "sampler_kwargs.cfg_low_sigma_taper",
  cfgAutoThresholdNumber: "sampler_kwargs.cfg_auto_low_cfg_threshold",
  cfgEarlyFloorValueNumber: "sampler_kwargs.cfg_early_floor_value",
  cfgEarlyFloorDurationNumber: "sampler_kwargs.cfg_early_floor_until_fraction",
};

const EXCLUDED_FIELDS = new Set([
  "batch_size",
  "batch_count",
  "seed_range_min",
  "seed_range_max",
  "live_preview_batch_index",
]);

let specs = {};
let activePath = "";
let activeInput = null;

const FIELD_METADATA = {
  cfg_scale: {
    placeholder: "[5.0, 7.5]",
    example: "[5.0, 7.5]",
    recommendation: "Recommended CFG range: 5.0–7.5",
    defaultMin: 5.0,
    defaultMax: 7.5,
  },
};

function pathForInput(input) {
  if (!input) return "";
  const mapped = CFG_FIELD_MAP[input.id];
  if (mapped) return mapped;
  const name = String(input.name || "").trim();
  if (!name || EXCLUDED_FIELDS.has(name) || name === "seed") return "";
  return name;
}

function integerInput(input) {
  const step = String(input?.step || "").trim();
  return step === "1" || (input?.inputMode === "numeric" && !step.includes("."));
}

function buttonFor(path) {
  return document.querySelector(`[data-parameter-range-button="${CSS.escape(path)}"]`);
}

function updateButton(path) {
  const button = buttonFor(path);
  if (!button) return;
  const spec = specs[path] || {};
  const active = Boolean(spec.enabled || spec.lock_min !== null && spec.lock_min !== undefined || spec.lock_max !== null && spec.lock_max !== undefined);
  button.classList.toggle("is-active", active);
  const pieces = [];
  if (spec.enabled) pieces.push(`random ${spec.min}–${spec.max}`);
  if (spec.lock_min !== null && spec.lock_min !== undefined) pieces.push(`min ≥ ${spec.lock_min}`);
  if (spec.lock_max !== null && spec.lock_max !== undefined) pieces.push(`max ≤ ${spec.lock_max}`);
  button.title = pieces.length ? `Advanced range: ${pieces.join(" · ")}` : "Configure random range and min/max locks";
}

function metadataFor(path, input) {
  const mapped = FIELD_METADATA[path] || {};
  const inputMin = Number(input?.min);
  const inputMax = Number(input?.max);
  const current = Number(input?.value);
  return {
    placeholder: mapped.placeholder || "[x, y]",
    example: mapped.example || (Number.isFinite(inputMin) && Number.isFinite(inputMax) ? `[${inputMin}, ${inputMax}]` : "[x, y]"),
    recommendation: mapped.recommendation || "",
    defaultMin: mapped.defaultMin ?? (Number.isFinite(inputMin) ? inputMin : (Number.isFinite(current) ? current : 0)),
    defaultMax: mapped.defaultMax ?? (Number.isFinite(inputMax) ? inputMax : (Number.isFinite(current) ? current : 1)),
  };
}

function labelForInput(input) {
  const label = input?.closest("label");
  const span = label?.querySelector(":scope > span:first-child");
  return String(span?.textContent || input?.name || input?.id || "Setting").trim();
}

function expressionForSpec(spec = {}) {
  if (!spec.enabled || spec.min === null || spec.min === undefined || spec.max === null || spec.max === undefined) return "";
  return `[${spec.min}, ${spec.max}]`;
}

function parseRangeExpression(raw) {
  const text = String(raw || "").trim();
  if (!text) return null;
  const match = text.match(/^(?:-1(?:\s*,\s*|\s+))?\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]$/);
  if (!match) throw new Error("Use [minimum,maximum] or optional -1 [minimum,maximum]. Punctuation after -1 is optional.");
  const min = Number(match[1]);
  const max = Number(match[2]);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) throw new Error("Range minimum must be a finite number no greater than the maximum.");
  return { min, max };
}

function setRangeSyntaxHint(path, input) {
  const meta = metadataFor(path, input);
  const expression = $("#parameterRangeExpression");
  if (expression) expression.placeholder = meta.placeholder;
  const hint = $("#parameterRangeSyntaxHint");
  if (hint) {
    const example = meta.example ? ` Example: ${meta.example}.` : "";
    const recommendation = meta.recommendation ? ` ${meta.recommendation}.` : "";
    hint.textContent = `Advanced syntax accepts [x,y] and optionally -1 [x,y] or -1, [x,y].${example}${recommendation}`;
  }
}

function syncExpressionFromControls() {
  const enabled = Boolean($("#parameterRangeEnabled")?.checked);
  const min = Number($("#parameterRangeMin")?.value);
  const max = Number($("#parameterRangeMax")?.value);
  const expression = $("#parameterRangeExpression");
  if (!expression) return;
  if (!enabled) {
    expression.value = "";
    return;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return;
  expression.value = expressionForSpec({ enabled: true, min, max });
}

function syncControlsFromExpression() {
  const expression = String($("#parameterRangeExpression")?.value || "").trim();
  if (!expression) return false;
  const parsed = parseRangeExpression(expression);
  $("#parameterRangeEnabled").checked = true;
  $("#parameterRangeMin").value = parsed.min;
  $("#parameterRangeMax").value = parsed.max;
  $("#parameterRangeExpression").value = expressionForSpec({ enabled: true, min: parsed.min, max: parsed.max });
  return true;
}

function openRangeDialog(path, input) {
  activePath = path;
  activeInput = input;
  const spec = specs[path] || {};
  const meta = metadataFor(path, input);
  $("#parameterRangeTitle").textContent = `Advanced Range · ${labelForInput(input)}`;
  $("#parameterRangePath").textContent = path;
  setRangeSyntaxHint(path, input);
  $("#parameterRangeExpression").value = expressionForSpec(spec);
  $("#parameterRangeEnabled").checked = Boolean(spec.enabled);
  $("#parameterRangeMin").value = spec.min ?? meta.defaultMin ?? input?.min ?? input?.value ?? "";
  $("#parameterRangeMax").value = spec.max ?? meta.defaultMax ?? input?.max ?? input?.value ?? "";
  $("#parameterRangeLockMinEnabled").checked = spec.lock_min !== null && spec.lock_min !== undefined;
  $("#parameterRangeLockMin").value = spec.lock_min ?? spec.min ?? meta.defaultMin ?? input?.min ?? "";
  $("#parameterRangeLockMaxEnabled").checked = spec.lock_max !== null && spec.lock_max !== undefined;
  $("#parameterRangeLockMax").value = spec.lock_max ?? spec.max ?? meta.defaultMax ?? input?.max ?? "";
  $("#parameterRangeInteger").checked = spec.integer ?? integerInput(input);
  syncExpressionFromControls();
  const dialog = $("#parameterRangeDialog");
  if (dialog && !dialog.open) dialog.showModal();
}

function installRangeButton(input) {
  const path = pathForInput(input);
  if (!path || buttonFor(path)) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-action-button ui-icon-control parameter-range-button";
  button.dataset.parameterRangeButton = path;
  button.textContent = "↔";
  button.setAttribute("aria-label", `Configure range for ${labelForInput(input)}`);
  button.title = "Configure random range and min/max locks";
  button.addEventListener("click", () => openRangeDialog(path, input));

  const rangePair = input.closest(".cfg-range-control");
  if (rangePair) rangePair.append(button);
  else input.insertAdjacentElement("afterend", button);
  updateButton(path);
}

export function collectParameterRanges() {
  return structuredClone(specs);
}

export function cfgEffectiveRangeLocks() {
  const spec = specs.cfg_scale || {};
  return {
    minimum: spec.lock_min ?? null,
    maximum: spec.lock_max ?? null,
  };
}

export function applyParameterRanges(value = {}) {
  specs = value && typeof value === "object" && !Array.isArray(value) ? structuredClone(value) : {};
  document.querySelectorAll("[data-parameter-range-button]").forEach((button) => updateButton(button.dataset.parameterRangeButton));
}

export function bindParameterRanges() {
  document.querySelectorAll('#generationForm input[type="number"]').forEach(installRangeButton);
  window.addEventListener("image-gen-apply-parameter-ranges", (event) => {
    const incoming = event.detail && typeof event.detail === "object" ? event.detail : {};
    specs = { ...specs, ...structuredClone(incoming) };
    document.querySelectorAll("[data-parameter-range-button]").forEach((button) => updateButton(button.dataset.parameterRangeButton));
    window.dispatchEvent(new CustomEvent("image-gen-parameter-ranges-changed", { detail: collectParameterRanges() }));
  });

  $("#parameterRangeExpression")?.addEventListener("change", () => {
    try {
      syncControlsFromExpression();
    } catch (error) {
      notify(error.message, "error");
    }
  });

  ["#parameterRangeEnabled", "#parameterRangeMin", "#parameterRangeMax"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => {
      syncExpressionFromControls();
    });
    $(selector)?.addEventListener("change", () => {
      syncExpressionFromControls();
    });
  });

  $("#parameterRangeSaveButton")?.addEventListener("click", () => {
    if (!activePath) return;
    let enabled = Boolean($("#parameterRangeEnabled")?.checked);
    let min = Number($("#parameterRangeMin")?.value);
    let max = Number($("#parameterRangeMax")?.value);
    const expression = String($("#parameterRangeExpression")?.value || "").trim();
    const controlsAreValid = enabled && Number.isFinite(min) && Number.isFinite(max) && min <= max;
    if (!controlsAreValid && expression) {
      try {
        const parsed = parseRangeExpression(expression);
        enabled = true;
        min = parsed.min;
        max = parsed.max;
      } catch (error) {
        notify(error.message, "error");
        return;
      }
    }
    const lockMinEnabled = Boolean($("#parameterRangeLockMinEnabled")?.checked);
    const lockMaxEnabled = Boolean($("#parameterRangeLockMaxEnabled")?.checked);
    const lockMin = Number($("#parameterRangeLockMin")?.value);
    const lockMax = Number($("#parameterRangeLockMax")?.value);
    if (enabled && (!Number.isFinite(min) || !Number.isFinite(max))) {
      notify("Random ranges require both a finite minimum and maximum.", "error");
      return;
    }
    if (enabled && min > max) {
      notify("Random range minimum cannot exceed its maximum.", "error");
      return;
    }
    if (lockMinEnabled && !Number.isFinite(lockMin)) {
      notify("Minimum lock requires a finite value.", "error");
      return;
    }
    if (lockMaxEnabled && !Number.isFinite(lockMax)) {
      notify("Maximum lock requires a finite value.", "error");
      return;
    }
    if (lockMinEnabled && lockMaxEnabled && lockMin > lockMax) {
      notify("Minimum lock cannot exceed maximum lock.", "error");
      return;
    }
    specs[activePath] = {
      enabled,
      min: enabled ? min : null,
      max: enabled ? max : null,
      integer: Boolean($("#parameterRangeInteger")?.checked),
      lock_min: lockMinEnabled ? lockMin : null,
      lock_max: lockMaxEnabled ? lockMax : null,
    };
    $("#parameterRangeExpression").value = expressionForSpec({ enabled, min, max });
    updateButton(activePath);
    $("#parameterRangeDialog")?.close();
    window.dispatchEvent(new CustomEvent("image-gen-parameter-ranges-changed", { detail: collectParameterRanges() }));
  });

  $("#parameterRangeClearButton")?.addEventListener("click", () => {
    if (!activePath) return;
    delete specs[activePath];
    $("#parameterRangeExpression").value = "";
    updateButton(activePath);
    $("#parameterRangeDialog")?.close();
    window.dispatchEvent(new CustomEvent("image-gen-parameter-ranges-changed", { detail: collectParameterRanges() }));
  });
}
