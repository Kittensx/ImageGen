import { api } from "../api.js?v=asset-card-latency1";
import { state } from "../state.js";
import { $, notify } from "../utils.js";
import { cfgEffectiveRangeLocks } from "./parameter-ranges.js?v=qol2";

const GUIDANCE_FIELDS = {
  cfg_guidance_mode: "#cfgGuidanceMode",
  cfg_curve_type: "#cfgCurveType",
  cfg_curve_strength: "#cfgCurveStrengthNumber",
  cfg_high_sigma_boost: "#cfgHighSigmaBoostNumber",
  cfg_low_sigma_taper: "#cfgLateTaperNumber",
  cfg_auto_low_cfg_threshold: "#cfgAutoThresholdNumber",
  cfg_early_floor_enabled: "#cfgEarlyFloorEnabled",
  cfg_early_floor_value: "#cfgEarlyFloorValueNumber",
  cfg_early_floor_until_fraction: "#cfgEarlyFloorDurationNumber",
};

let rememberedUnlockedCfgRescale = 0;
let cfgRescaleArchitectureLocked = false;

const RANGE_PAIRS = [
  ["#cfgRescaleRange", "#cfgRescaleNumber"],
  ["#cfgCurveStrengthRange", "#cfgCurveStrengthNumber"],
  ["#cfgHighSigmaBoostRange", "#cfgHighSigmaBoostNumber"],
  ["#cfgLateTaperRange", "#cfgLateTaperNumber"],
  ["#cfgAutoThresholdRange", "#cfgAutoThresholdNumber"],
  ["#cfgEarlyFloorValueRange", "#cfgEarlyFloorValueNumber"],
  ["#cfgEarlyFloorDurationRange", "#cfgEarlyFloorDurationNumber"],
];

function normalizedModelPath(value) {
  return String(value || "").trim().replaceAll("\\", "/").toLowerCase();
}

function selectedModelRecord() {
  const requestedPath = $("#modelPath")?.value || "";
  const normalizedRequested = normalizedModelPath(requestedPath);
  return state.models.find((item) => normalizedModelPath(item.path) === normalizedRequested) || state.activeModel || null;
}

function modelFamilyForCfgRescaleGuard(model = null) {
  const source = model || selectedModelRecord();
  const raw = String(
    source?.runtime_profile?.architecture
      || source?.architecture
      || source?.architecture_contract?.family
      || source?.model_family
      || source?.architecture_summary
      || "",
  ).trim().toLowerCase();
  if (!raw) return "";
  if (raw.includes("sd3") || raw.includes("stable diffusion 3") || raw.includes("stable-diffusion-3") || raw.includes("flowmatch") || raw.includes("rectified flow")) {
    return "sd3.x";
  }
  return raw;
}

function cfgRescaleLockedForCurrentArchitecture() {
  return modelFamilyForCfgRescaleGuard() === "sd3.x";
}

function currentCfgRescaleValue() {
  return Number.parseFloat($("#cfgRescaleNumber")?.value || $("#cfgRescaleRange")?.value || "0") || 0;
}

function setCfgRescaleInputs(value) {
  const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
  const asText = String(normalized);
  if ($("#cfgRescaleNumber")) $("#cfgRescaleNumber").value = asText;
  if ($("#cfgRescaleRange")) $("#cfgRescaleRange").value = asText;
}

function updateCfgRescaleArchitectureState() {
  const range = $("#cfgRescaleRange");
  const number = $("#cfgRescaleNumber");
  const status = $("#cfgRescaleStatus");
  if (!range || !number || !status) return;
  const locked = cfgRescaleLockedForCurrentArchitecture();

  if (locked) {
    const currentValue = currentCfgRescaleValue();
    if (currentValue !== 0) rememberedUnlockedCfgRescale = currentValue;
    setCfgRescaleInputs(0);
    range.disabled = true;
    number.disabled = true;
    const message = "CFG rescale is fixed to 0 for SD3.x / FlowMatchEuler checkpoints because this flow path does not support guidance rescale.";
    range.title = message;
    number.title = message;
    status.textContent = message;
    status.className = "field-status warning";
  } else {
    range.disabled = false;
    number.disabled = false;
    range.title = "CFG rescale adjusts guidance saturation near the denoised prediction.";
    number.title = range.title;
    if (cfgRescaleArchitectureLocked && currentCfgRescaleValue() === 0 && rememberedUnlockedCfgRescale > 0) {
      setCfgRescaleInputs(rememberedUnlockedCfgRescale);
    }
  }
  cfgRescaleArchitectureLocked = locked;
}

export function enforceCfgRescaleRequestGuardrails(values = {}) {
  const next = { ...values };
  if (cfgRescaleLockedForCurrentArchitecture()) {
    next.cfg_rescale = 0;
    if ("hires_cfg_rescale" in next || next.hires_enabled) next.hires_cfg_rescale = 0;
  }
  return next;
}

export const CFG_PRESETS = {
  sdxl_lightning_recommended: {
    label: "SDXL Lightning Recommended",
    preserve_cfg_scale: true,
    cfg_rescale: 0.0,
    sampler_kwargs: {
      cfg_guidance_mode: "legacy_flat",
      cfg_curve_type: "smoothstep",
      cfg_curve_strength: 0.0,
      cfg_high_sigma_boost: 0.0,
      cfg_low_sigma_taper: 0.0,
      cfg_auto_low_cfg_threshold: 1.0,
      cfg_early_floor_enabled: false,
      cfg_early_floor_value: 1.0,
      cfg_early_floor_until_fraction: 0.0,
    },
  },
  classic_flat: {
    label: "Classic / Flat",
    preserve_cfg_scale: true,
    cfg_rescale: 0.0,
    sampler_kwargs: {
      cfg_guidance_mode: "legacy_flat",
      cfg_curve_type: "smoothstep",
      cfg_curve_strength: 0.0,
      cfg_high_sigma_boost: 0.0,
      cfg_low_sigma_taper: 0.0,
      cfg_auto_low_cfg_threshold: 6.5,
      cfg_early_floor_enabled: false,
      cfg_early_floor_value: 0.0,
      cfg_early_floor_until_fraction: 0.0,
    },
  },
  low_cfg_safe: {
    label: "Low-CFG Safe",
    preserve_cfg_scale: true,
    cfg_rescale: 0.0,
    sampler_kwargs: {
      cfg_guidance_mode: "auto_low_cfg",
      cfg_curve_type: "smoothstep",
      cfg_curve_strength: 0.35,
      cfg_high_sigma_boost: 0.35,
      cfg_low_sigma_taper: 0.0,
      cfg_auto_low_cfg_threshold: 6.5,
      cfg_early_floor_enabled: false,
      cfg_early_floor_value: 0.0,
      cfg_early_floor_until_fraction: 0.0,
    },
  },
  low_cfg_strong: {
    label: "Low-CFG Strong Composition",
    preserve_cfg_scale: true,
    cfg_rescale: 0.0,
    sampler_kwargs: {
      cfg_guidance_mode: "auto_low_cfg",
      cfg_curve_type: "smoothstep",
      cfg_curve_strength: 0.6,
      cfg_high_sigma_boost: 0.6,
      cfg_low_sigma_taper: 0.0,
      cfg_auto_low_cfg_threshold: 6.8,
      cfg_early_floor_enabled: false,
      cfg_early_floor_value: 0.0,
      cfg_early_floor_until_fraction: 0.0,
    },
  },
  soft_detail_taper: {
    label: "Soft Detail Taper",
    preserve_cfg_scale: true,
    cfg_rescale: 0.15,
    sampler_kwargs: {
      cfg_guidance_mode: "sigma_shaped",
      cfg_curve_type: "cosine",
      cfg_curve_strength: 0.5,
      cfg_high_sigma_boost: 0.2,
      cfg_low_sigma_taper: 0.35,
      cfg_auto_low_cfg_threshold: 6.5,
      cfg_early_floor_enabled: false,
      cfg_early_floor_value: 0.0,
      cfg_early_floor_until_fraction: 0.0,
    },
  },
};

function userPresets() {
  const value = state.settings?.cfg_lab_user_presets;
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function presetRecord(key) {
  if (String(key).startsWith("user:")) return userPresets()[String(key).slice(5)] || null;
  return CFG_PRESETS[key] || null;
}

function renderCfgPresetOptions(preferred = "") {
  const select = $("#cfgGuidancePreset");
  if (!select) return;
  const previous = preferred || select.value || "classic_flat";
  select.replaceChildren();
  const builtins = document.createElement("optgroup");
  builtins.label = "Built-in";
  Object.entries(CFG_PRESETS).forEach(([key, preset]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = preset.label || key;
    builtins.append(option);
  });
  select.append(builtins);
  const entries = Object.entries(userPresets()).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length) {
    const group = document.createElement("optgroup");
    group.label = "Saved";
    entries.forEach(([name, preset]) => {
      const option = document.createElement("option");
      option.value = `user:${name}`;
      option.textContent = preset.label || name;
      group.append(option);
    });
    select.append(group);
  }
  const exists = [...select.options].some((item) => item.value === previous);
  select.value = exists ? previous : "classic_flat";
  if ($("#deleteCfgPresetButton")) $("#deleteCfgPresetButton").disabled = !select.value.startsWith("user:");
}

function cfgRangeSubset(values = {}) {
  const ranges = values._random_ranges || {};
  return Object.fromEntries(Object.entries(ranges).filter(([path]) => path === "cfg_scale" || path === "cfg_rescale" || path.startsWith("sampler_kwargs.cfg_")));
}

function snapshotCfgPreset(collect) {
  const values = collect();
  const sampler = {};
  Object.keys(GUIDANCE_FIELDS).forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(values.sampler_kwargs || {}, key)) sampler[key] = values.sampler_kwargs[key];
  });
  ["cfg_effective_min_lock", "cfg_effective_max_lock"].forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(values.sampler_kwargs || {}, key)) sampler[key] = values.sampler_kwargs[key];
  });
  return {
    schema_version: 1,
    label: "",
    cfg_scale: Number(values.cfg_scale),
    cfg_rescale: Number(values.cfg_rescale || 0),
    sampler_kwargs: sampler,
    random_ranges: cfgRangeSubset(values),
  };
}

async function saveCfgPreset(collect) {
  const proposed = window.prompt("CFG Lab preset name:", "My CFG Preset");
  const name = String(proposed || "").trim();
  if (!name) return;
  const snapshot = snapshotCfgPreset(collect);
  snapshot.label = name;
  const next = { ...userPresets(), [name]: snapshot };
  const saved = await api.saveSettings({ cfg_lab_user_presets: next });
  state.settings = { ...state.settings, ...saved };
  renderCfgPresetOptions(`user:${name}`);
  notify(`Saved CFG Lab preset: ${name}`);
}

function downloadCfgPreset() {
  const key = $("#cfgGuidancePreset")?.value || "classic_flat";
  const preset = presetRecord(key);
  if (!preset) return;
  const payload = { kind: "image_gen_cfg_lab_preset", schema_version: 1, preset };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${String(preset.label || key).replace(/[^A-Za-z0-9._-]+/g, "_")}.cfg-lab.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function importCfgPreset(file) {
  const raw = JSON.parse(await file.text());
  const preset = raw?.preset && typeof raw.preset === "object" ? raw.preset : raw;
  if (!preset || typeof preset !== "object" || (preset.preserve_cfg_scale !== true && !Number.isFinite(Number(preset.cfg_scale)))) throw new Error("The selected file is not a valid CFG Lab preset.");
  const fallback = String(preset.label || file.name.replace(/\.cfg-lab\.json$/i, "").replace(/\.json$/i, "")).trim() || "Imported CFG Preset";
  const proposed = window.prompt("Imported preset name:", fallback);
  const name = String(proposed || "").trim();
  if (!name) return;
  const next = { ...userPresets(), [name]: { ...preset, label: name, schema_version: 1 } };
  const saved = await api.saveSettings({ cfg_lab_user_presets: next });
  state.settings = { ...state.settings, ...saved };
  renderCfgPresetOptions(`user:${name}`);
  notify(`Imported CFG Lab preset: ${name}`);
}

async function deleteCfgPreset() {
  const key = $("#cfgGuidancePreset")?.value || "";
  if (!key.startsWith("user:")) return;
  const name = key.slice(5);
  if (!window.confirm(`Delete CFG Lab preset "${name}"?`)) return;
  const next = { ...userPresets() };
  delete next[name];
  const saved = await api.saveSettings({ cfg_lab_user_presets: next });
  state.settings = { ...state.settings, ...saved };
  renderCfgPresetOptions("classic_flat");
  notify(`Deleted CFG Lab preset: ${name}`);
}

function isKesSampler() {
  return ["kes", "kes_sampler", "simple_kes_sampler"].includes(String($("#samplerName")?.value || "").toLowerCase());
}

function numberValue(selector, fallback) {
  const value = Number($(selector)?.value);
  return Number.isFinite(value) ? value : fallback;
}

const PROMPT_CFG_DIRECTIVE = /<param\s*\[\s*cfg\s*\]\s*:\s*([^<]+?)\s*(?<!-)>/i;
const PROMPT_CFG_CURVES = new Set(["linear", "smoothstep", "cosine", "exp_decay"]);

function safeJsonInput(selector) {
  try {
    const value = JSON.parse(String($(selector)?.value || "{}"));
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

function isSuperHybridParser() {
  return String($("#promptParserName")?.value || "").toLowerCase() === "superhybrid";
}

function syncPromptCfgBehaviorFromOptions() {
  const select = $("#promptCfgBehavior");
  if (!select) return;
  const options = safeJsonInput("#promptParserKwargs");
  const behavior = ["replace_ui", "shape_ui", "disabled"].includes(String(options.prompt_cfg_behavior || ""))
    ? String(options.prompt_cfg_behavior)
    : "replace_ui";
  select.value = behavior;
}

function writePromptCfgBehavior() {
  const select = $("#promptCfgBehavior");
  const hidden = $("#promptParserKwargs");
  if (!select || !hidden) return;
  const options = safeJsonInput("#promptParserKwargs");
  options.prompt_cfg_behavior = select.value || "replace_ui";
  hidden.value = JSON.stringify(options);
  const mirrored = document.querySelector('[data-parser-setting="prompt_cfg_behavior"]');
  if (mirrored) mirrored.value = options.prompt_cfg_behavior;
  if (($("#hiresPromptParserMode")?.value || "same_as_base") === "same_as_base") {
    const hiresHidden = $("#hiresPromptParserKwargs");
    if (hiresHidden) hiresHidden.value = hidden.value;
  }
  hidden.dispatchEvent(new Event("input", { bubbles: true }));
}

function resolvePromptCfgPositions(explicit) {
  const count = explicit.length;
  if (count === 1) return [0];
  const anchors = new Map([[0, 0], [count - 1, 1]]);
  explicit.forEach((position, index) => {
    if (position === null) return;
    if (!Number.isFinite(position) || position < 0 || position > 1) throw new Error("@ positions must be between 0 and 1");
    if (index === 0 && position !== 0) throw new Error("the first control point must be at @0");
    if (index === count - 1 && position !== 1) throw new Error("the last control point must be at @1");
    anchors.set(index, position);
  });
  const ordered = [...anchors.entries()].sort((a, b) => a[0] - b[0]);
  const positions = Array(count).fill(0);
  ordered.forEach(([index, position]) => { positions[index] = position; });
  for (let pair = 0; pair < ordered.length - 1; pair += 1) {
    const [leftIndex, leftPosition] = ordered[pair];
    const [rightIndex, rightPosition] = ordered[pair + 1];
    if (rightPosition <= leftPosition) throw new Error("@ positions must increase from left to right");
    const span = rightIndex - leftIndex;
    for (let offset = 1; offset < span; offset += 1) {
      positions[leftIndex + offset] = leftPosition + (rightPosition - leftPosition) * (offset / span);
    }
  }
  return positions;
}

function parsePromptCfgDirective(raw) {
  let text = String(raw || "").trim();
  if (!text) throw new Error("the CFG directive is empty");
  let interpolation = "linear";
  if (text.includes(":")) {
    const index = text.lastIndexOf(":");
    const candidate = text.slice(index + 1).trim().toLowerCase().replaceAll("-", "_");
    const aliases = { smooth: "smoothstep", ease: "smoothstep", ease_in_out: "smoothstep", exp: "exp_decay", exponential: "exp_decay", piecewise_linear: "linear" };
    interpolation = aliases[candidate] || candidate;
    if (!PROMPT_CFG_CURVES.has(interpolation)) throw new Error(`unknown interpolation ${candidate}`);
    text = text.slice(0, index).trim();
  }
  const parts = text.split("->").map((part) => part.trim());
  if (!parts.length || parts.some((part) => !part)) throw new Error("the CFG curve contains an empty control point");
  const values = [];
  const explicit = [];
  parts.forEach((part) => {
    const match = part.match(/^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\s*@\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))?$/);
    if (!match) throw new Error(`invalid control point ${part}`);
    const value = Number(match[1]);
    if (!Number.isFinite(value) || value < 0 || value > 30) throw new Error("CFG values must be between 0 and 30");
    values.push(value);
    explicit.push(match[2] === undefined ? null : Number(match[2]));
  });
  return { values, positions: resolvePromptCfgPositions(explicit), interpolation };
}

function promptCurveWeight(value, curve) {
  const x = clamp01(value);
  if (curve === "smoothstep") return x * x * (3 - 2 * x);
  if (curve === "cosine") return 0.5 - 0.5 * Math.cos(Math.PI * x);
  if (curve === "exp_decay") return x <= 0 ? 0 : (x >= 1 ? 1 : (1 - Math.exp(-4 * x)) / (1 - Math.exp(-4)));
  return x;
}

function materializePromptCfg(spec, steps) {
  if (spec.values.length === 1) return Array(steps).fill(spec.values[0]);
  return Array.from({ length: steps }, (_, index) => {
    const progress = steps <= 1 ? 0 : index / (steps - 1);
    let segment = 0;
    for (let cursor = 0; cursor < spec.positions.length - 1; cursor += 1) {
      if (progress >= spec.positions[cursor] && progress <= spec.positions[cursor + 1]) { segment = cursor; break; }
    }
    if (progress <= spec.positions[0]) return spec.values[0];
    if (progress >= spec.positions.at(-1)) return spec.values.at(-1);
    const left = spec.positions[segment];
    const right = spec.positions[segment + 1];
    const local = promptCurveWeight((progress - left) / (right - left), spec.interpolation);
    return spec.values[segment] + (spec.values[segment + 1] - spec.values[segment]) * local;
  });
}

function requestedPromptCfgSeries(steps) {
  const cfgInput = $("#cfgScale");
  const recommendedEffective = Number(cfgInput?.dataset?.recommendedEffectiveCfg);
  const uiCfg = Number.isFinite(recommendedEffective) ? recommendedEffective : numberValue("#cfgScale", 7);
  const flat = Array(steps).fill(uiCfg);
  const parserActive = isSuperHybridParser();
  const behavior = $("#promptCfgBehavior")?.value || "replace_ui";
  const match = String($("#positivePrompt")?.value || "").match(PROMPT_CFG_DIRECTIVE);
  if (!parserActive) return { values: flat, source: "UI CFG", detail: "SuperHybrid parser is not active.", behavior, directive: null, interpolation: "linear" };
  if (!match) return { values: flat, source: "UI CFG", detail: "No SuperHybrid CFG directive detected.", behavior, directive: null, interpolation: "linear" };
  try {
    const spec = parsePromptCfgDirective(match[1]);
    const shape = materializePromptCfg(spec, steps);
    if (behavior === "disabled") return { values: flat, source: "UI CFG", detail: "SuperHybrid CFG directive detected but disabled by policy.", behavior, directive: match[0], interpolation: spec.interpolation };
    if (behavior === "shape_ui") {
      if (Math.abs(shape[0]) < 1e-12) throw new Error("shape-with-UI-start requires a non-zero first value");
      return {
        values: shape.map((value) => uiCfg * value / shape[0]),
        source: "SuperHybrid curve shape + UI start",
        detail: `${match[0]} is normalized to the UI CFG start value.`,
        behavior,
        directive: match[0],
        interpolation: spec.interpolation,
        controlCount: spec.values.length,
      };
    }
    return {
      values: shape,
      source: "SuperHybrid prompt",
      detail: `${match[0]} replaces the UI CFG schedule before CFG Lab shaping.`,
      behavior,
      directive: match[0],
      interpolation: spec.interpolation,
      controlCount: spec.values.length,
    };
  } catch (error) {
    return { values: flat, source: "Invalid SuperHybrid CFG", detail: String(error?.message || error), behavior, directive: match[0], interpolation: "linear", error: true };
  }
}

export function readCfgLabValues() {
  const sampler_kwargs = {};
  Object.entries(GUIDANCE_FIELDS).forEach(([name, selector]) => {
    const input = $(selector);
    if (!input) return;
    if (input.type === "checkbox") sampler_kwargs[name] = Boolean(input.checked);
    else if (input.type === "number" || input.type === "range") sampler_kwargs[name] = Number(input.value);
    else sampler_kwargs[name] = input.value;
  });
  return {
    cfg_rescale: numberValue("#cfgRescaleNumber", 0),
    sampler_kwargs,
  };
}

function assignControl(selector, value) {
  const input = $(selector);
  if (!input || value === undefined || value === null) return;
  if (input.type === "checkbox") input.checked = Boolean(value);
  else input.value = String(value);
}

export function applyCfgLabValues(values = {}) {
  if (Object.prototype.hasOwnProperty.call(values, "cfg_rescale")) {
    assignControl("#cfgRescaleNumber", values.cfg_rescale);
    assignControl("#cfgRescaleRange", values.cfg_rescale);
  }
  const kwargs = values.sampler_kwargs || {};
  Object.entries(GUIDANCE_FIELDS).forEach(([name, selector]) => {
    if (!Object.prototype.hasOwnProperty.call(kwargs, name)) return;
    assignControl(selector, kwargs[name]);
  });
  RANGE_PAIRS.forEach(([rangeSelector, numberSelector]) => {
    const number = $(numberSelector);
    if (number) assignControl(rangeSelector, number.value);
  });
  updateCfgRescaleArchitectureState();
  renderCfgCurvePreview();
  updateSamplerAvailability();
}

function clamp01(value) { return Math.max(0, Math.min(1, Number(value))); }
function curveWeight(value, curveType) {
  const x = clamp01(value);
  if (curveType === "linear") return x;
  if (curveType === "cosine") return 0.5 - 0.5 * Math.cos(Math.PI * x);
  if (curveType === "exp_decay") return x <= 0 ? 0 : (x >= 1 ? 1 : (1 - Math.exp(-4 * x)) / (1 - Math.exp(-4)));
  return x * x * (3 - 2 * x);
}

function configuredModel() {
  const steps = Math.max(1, Math.round(numberValue("#steps", 20)));
  const promptInput = requestedPromptCfgSeries(steps);
  const mode = $("#cfgGuidanceMode")?.value || "legacy_flat";
  const curveType = $("#cfgCurveType")?.value || "smoothstep";
  const strength = numberValue("#cfgCurveStrengthNumber", 1);
  const boostAmount = numberValue("#cfgHighSigmaBoostNumber", 1.2);
  const taperAmount = numberValue("#cfgLateTaperNumber", 0.3);
  const threshold = numberValue("#cfgAutoThresholdNumber", 6.5);
  const floorEnabled = Boolean($("#cfgEarlyFloorEnabled")?.checked);
  const floorValue = numberValue("#cfgEarlyFloorValueNumber", 6.2);
  const floorDuration = numberValue("#cfgEarlyFloorDurationNumber", 0.3);
  const points = promptInput.values.map((requested, stepIndex) => {
    const progress = steps <= 1 ? 0 : stepIndex / (steps - 1);
    const sigmaFraction = 1 - progress;
    let early = 0;
    let late = 0;
    let active = mode !== "legacy_flat";
    if (active && mode === "step_shaped") {
      early = curveWeight(1 - progress, curveType);
      late = curveWeight(progress, curveType);
    } else if (active && mode === "sigma_shaped") {
      early = curveWeight(sigmaFraction, curveType);
      late = curveWeight(1 - sigmaFraction, curveType);
    } else if (active && mode === "auto_low_cfg" && requested < threshold) {
      early = curveWeight(sigmaFraction, curveType);
      late = curveWeight(1 - sigmaFraction, curveType);
    } else if (mode === "auto_low_cfg") {
      active = false;
    }
    let effective = active ? requested + boostAmount * strength * early - taperAmount * strength * late : requested;
    if (floorEnabled && progress <= floorDuration) effective = Math.max(effective, floorValue);
    const locks = cfgEffectiveRangeLocks();
    if (locks.minimum !== null) effective = Math.max(effective, Number(locks.minimum));
    if (locks.maximum !== null) effective = Math.min(effective, Number(locks.maximum));
    return { step_index: stepIndex, requested_cfg_scale: requested, effective_cfg_scale: Math.max(0, effective), cfg_source: promptInput.source };
  });
  const requestedMin = Math.min(...promptInput.values);
  const requestedMax = Math.max(...promptInput.values);
  const effective = points.map((point) => point.effective_cfg_scale);
  const rescale = numberValue("#cfgRescaleNumber", 0);
  const transform = `${mode.replaceAll("_", " ")}${mode === "legacy_flat" ? "" : ` · ${curveType}`}${rescale > 0 ? ` · rescale ${rescale.toFixed(2)}` : ""}`;
  return {
    points,
    source: promptInput.source,
    detail: promptInput.detail,
    error: Boolean(promptInput.error),
    requestedSummary: requestedMin === requestedMax
      ? `${requestedMin.toFixed(2)} flat`
      : `${promptInput.values[0].toFixed(2)} → ${promptInput.values.at(-1).toFixed(2)} · ${promptInput.interpolation}`,
    transform,
    owner: "Shared CFG Lab guidance",
    replay: "Reconstruct from prompt · fingerprint on save",
    effectiveMin: Math.min(...effective),
    effectiveMax: Math.max(...effective),
  };
}

function configuredSeries() {
  return configuredModel().points;
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

export function renderCfgGraph(container, points, { compact = false, currentStepIndex = null } = {}) {
  if (!container) return;
  const clean = (points || []).filter((point) => Number.isFinite(Number(point.effective_cfg_scale)));
  if (!clean.length) {
    container.textContent = "No per-step CFG data is available.";
    return;
  }
  const width = compact ? 560 : 720;
  const height = compact ? 210 : 240;
  const pad = { left: 42, right: 18, top: 18, bottom: 34 };
  const values = clean.flatMap((point) => [Number(point.requested_cfg_scale), Number(point.effective_cfg_scale)]).filter(Number.isFinite);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 1);
  const span = Math.max(maxValue - minValue, 1);
  const x = (index) => pad.left + (clean.length <= 1 ? 0 : index / (clean.length - 1)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - (Number(value) - minValue) / span) * (height - pad.top - pad.bottom);
  const pathFor = (field) => clean.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(point[field]).toFixed(2)}`).join(" ");

  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Requested and effective CFG by denoising step" });
  svg.classList.add("cfg-series-svg");
  for (let grid = 0; grid <= 4; grid += 1) {
    const value = minValue + (span * grid / 4);
    const lineY = y(value);
    svg.append(svgElement("line", { x1: pad.left, y1: lineY, x2: width - pad.right, y2: lineY, class: "cfg-grid-line" }));
    const label = svgElement("text", { x: pad.left - 7, y: lineY + 4, class: "cfg-axis-label", "text-anchor": "end" });
    label.textContent = value.toFixed(1);
    svg.append(label);
  }
  svg.append(svgElement("path", { d: pathFor("requested_cfg_scale"), class: "cfg-requested-line" }));
  svg.append(svgElement("path", { d: pathFor("effective_cfg_scale"), class: "cfg-effective-line" }));
  clean.forEach((point, index) => {
    const marker = svgElement("circle", { cx: x(index), cy: y(point.effective_cfg_scale), r: 2.4, class: "cfg-effective-marker" });
    const title = svgElement("title");
    title.textContent = `Step ${Number(point.step_index) + 1}: requested ${Number(point.requested_cfg_scale).toFixed(2)}, effective ${Number(point.effective_cfg_scale).toFixed(2)}`;
    marker.append(title);
    svg.append(marker);

    const overrideSource = String(point.override_source || "base_request");
    if (point.transition_id || overrideSource !== "base_request") {
      const transition = svgElement("rect", {
        x: x(index) - 3.5,
        y: y(point.effective_cfg_scale) - 3.5,
        width: 7,
        height: 7,
        transform: `rotate(45 ${x(index)} ${y(point.effective_cfg_scale)})`,
        class: "cfg-transition-marker",
      });
      const transitionTitle = svgElement("title");
      transitionTitle.textContent = `CFG transition at step ${Number(point.step_index) + 1}: ${overrideSource}${point.transition_id ? ` · ${point.transition_id}` : ""}`;
      transition.append(transitionTitle);
      svg.append(transition);
    }
  });

  if (currentStepIndex !== null && currentStepIndex !== undefined) {
    const currentArrayIndex = clean.findIndex((point) => Number(point.step_index) === Number(currentStepIndex));
    if (currentArrayIndex >= 0) {
      const currentPoint = clean[currentArrayIndex];
      svg.append(svgElement("line", {
        x1: x(currentArrayIndex),
        y1: pad.top,
        x2: x(currentArrayIndex),
        y2: height - pad.bottom,
        class: "cfg-live-cursor-line",
      }));
      const cursor = svgElement("circle", {
        cx: x(currentArrayIndex),
        cy: y(currentPoint.effective_cfg_scale),
        r: 5.5,
        class: "cfg-live-cursor-marker",
      });
      const cursorTitle = svgElement("title");
      cursorTitle.textContent = `Current live step ${Number(currentPoint.step_index) + 1}: effective CFG ${Number(currentPoint.effective_cfg_scale).toFixed(2)}`;
      cursor.append(cursorTitle);
      svg.append(cursor);
    }
  }
  const start = svgElement("text", { x: pad.left, y: height - 9, class: "cfg-axis-label" });
  start.textContent = "Step 1";
  const end = svgElement("text", { x: width - pad.right, y: height - 9, class: "cfg-axis-label", "text-anchor": "end" });
  end.textContent = `Step ${clean.length}`;
  svg.append(start, end);
  container.replaceChildren(svg);
}

export function renderCfgCurvePreview() {
  updateCfgRescaleArchitectureState();
  syncPromptCfgBehaviorFromOptions();
  const model = configuredModel();
  renderCfgGraph($("#cfgCurvePreviewGraph"), model.points, { compact: true });
  const summary = $("#cfgCurvePreviewSummary");
  if (summary) summary.textContent = `Requested ${model.requestedSummary} · effective ${model.effectiveMin.toFixed(2)}–${model.effectiveMax.toFixed(2)} · ${model.points.length} steps`;
  if ($("#cfgSourceValue")) $("#cfgSourceValue").textContent = model.source;
  if ($("#cfgRequestedCurveValue")) $("#cfgRequestedCurveValue").textContent = model.requestedSummary;
  if ($("#cfgTransformValue")) $("#cfgTransformValue").textContent = model.transform;
  if ($("#cfgGuidanceOwnerValue")) $("#cfgGuidanceOwnerValue").textContent = model.owner;
  if ($("#cfgReplayValue")) $("#cfgReplayValue").textContent = model.replay;
  const status = $("#cfgPromptDirectiveStatus");
  if (status) {
    status.textContent = model.detail;
    status.classList.toggle("error", model.error);
  }
}

function updateSamplerAvailability() {
  document.querySelectorAll(".cfg-kes-only").forEach((node) => {
    node.classList.remove("is-disabled");
    node.querySelectorAll("input, select").forEach((input) => { input.disabled = false; });
  });
  const policyField = $("#promptCfgBehavior")?.closest(".cfg-prompt-policy-field");
  const promptPolicyEnabled = isSuperHybridParser();
  if ($("#promptCfgBehavior")) $("#promptCfgBehavior").disabled = !promptPolicyEnabled;
  policyField?.classList.toggle("is-disabled", !promptPolicyEnabled);
  const status = $("#cfgLabSamplerStatus");
  if (status) status.textContent = "CFG Lab shaping is now standard across samplers. Completed outputs record the actual requested and effective CFG used at every step.";
}

export function applyCfgPreset(name, { saveSession = null, notifyUser = true } = {}) {
  const preset = presetRecord(name);
  if (!preset) return false;
  const select = $("#cfgGuidancePreset");
  if (select && [...select.options].some((option) => option.value === name)) select.value = name;
  if (preset.preserve_cfg_scale !== true && Number.isFinite(Number(preset.cfg_scale))) {
    assignControl("#cfgScale", preset.cfg_scale);
  }
  applyCfgLabValues(preset);
  if (preset.random_ranges) {
    window.dispatchEvent(new CustomEvent("image-gen-apply-parameter-ranges", { detail: preset.random_ranges }));
  }
  saveSession?.();
  if (notifyUser) notify(`Applied CFG preset: ${preset.label || name}`);
  return true;
}

function seedLockedBases(current) {
  let seed = Number(current.seed);
  if (!Number.isInteger(seed) || seed < 0) {
    const buffer = new Uint32Array(1);
    crypto.getRandomValues(buffer);
    seed = Number(buffer[0]);
  }
  const lanes = [
    [5.0, "legacy_flat", "CFG 5.0 flat"],
    [5.0, "auto_low_cfg", "CFG 5.0 auto low-CFG"],
    [6.0, "legacy_flat", "CFG 6.0 flat"],
    [6.0, "auto_low_cfg", "CFG 6.0 auto low-CFG"],
    [7.0, "legacy_flat", "CFG 7.0 flat baseline"],
  ];
  return {
    seed,
    bases: lanes.map(([cfgScale, mode]) => ({
      ...structuredClone(current),
      seed,
      cfg_scale: cfgScale,
      batch_size: 1,
      batch_count: 1,
      sampler_kwargs: {
        ...(current.sampler_kwargs || {}),
        cfg_guidance_mode: mode,
      },
    })),
    lineage: lanes.map(([, , label]) => ({ source: "cfg_lab", source_id: null, source_label: label })),
  };
}

export function bindCfgLab({ collect = () => ({}), saveSession = () => {}, openVariationMatrix = () => {} } = {}) {
  RANGE_PAIRS.forEach(([rangeSelector, numberSelector]) => {
    const range = $(rangeSelector);
    const number = $(numberSelector);
    if (!range || !number) return;
    const sync = (source, target) => {
      target.value = source.value;
      renderCfgCurvePreview();
      saveSession();
    };
    range.addEventListener("input", () => sync(range, number));
    number.addEventListener("input", () => sync(number, range));
  });
  ["#cfgScale", "#steps", "#cfgGuidanceMode", "#cfgCurveType", "#cfgEarlyFloorEnabled", "#positivePrompt"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => { renderCfgCurvePreview(); saveSession(); });
    $(selector)?.addEventListener("change", () => { renderCfgCurvePreview(); saveSession(); });
  });
  $("#modelPath")?.addEventListener("change", renderCfgCurvePreview);
  window.addEventListener("image-gen-model-activated", renderCfgCurvePreview);
  window.addEventListener("image-gen-model-unloaded", renderCfgCurvePreview);
  window.addEventListener("image-gen-generation-values-applied", renderCfgCurvePreview);
  $("#samplerName")?.addEventListener("change", () => { updateSamplerAvailability(); renderCfgCurvePreview(); });
  $("#promptParserName")?.addEventListener("change", () => { syncPromptCfgBehaviorFromOptions(); updateSamplerAvailability(); renderCfgCurvePreview(); });
  $("#promptCfgBehavior")?.addEventListener("change", () => { writePromptCfgBehavior(); renderCfgCurvePreview(); saveSession(); });
  document.addEventListener("prompt-parser-options-changed", () => { syncPromptCfgBehaviorFromOptions(); renderCfgCurvePreview(); });
  renderCfgPresetOptions();
  $("#cfgGuidancePreset")?.addEventListener("change", () => {
    if ($("#deleteCfgPresetButton")) $("#deleteCfgPresetButton").disabled = !$("#cfgGuidancePreset").value.startsWith("user:");
  });
  $("#applyCfgPresetButton")?.addEventListener("click", () => applyCfgPreset($("#cfgGuidancePreset").value, { saveSession }));
  $("#saveCfgPresetButton")?.addEventListener("click", () => saveCfgPreset(collect).catch((error) => notify(error.message, "error")));
  $("#exportCfgPresetButton")?.addEventListener("click", downloadCfgPreset);
  $("#deleteCfgPresetButton")?.addEventListener("click", () => deleteCfgPreset().catch((error) => notify(error.message, "error")));
  $("#importCfgPresetButton")?.addEventListener("click", () => $("#importCfgPresetInput")?.click());
  $("#importCfgPresetInput")?.addEventListener("change", async (event) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;
    try { await importCfgPreset(file); } catch (error) { notify(error.message, "error"); }
  });
  $("#openCfgSweepButton")?.addEventListener("click", () => {
    const current = collect();
    const { seed, bases, lineage } = seedLockedBases(current);
    openVariationMatrix({
      baseRequests: bases,
      baseLineage: lineage,
      initialDimensions: [],
      recipeName: `CFG Lab seed ${seed}`,
      title: "Seed-Locked CFG Lab",
    });
  });
  syncPromptCfgBehaviorFromOptions();
  updateSamplerAvailability();
  renderCfgCurvePreview();
}
