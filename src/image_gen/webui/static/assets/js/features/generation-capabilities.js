import { api } from "../api.js";
import { state } from "../state.js";

const PREFERENCE_SCHEMA_VERSION = 1;
const STANDARD_MODE = "standard";
const FIELD_IDS = {
  cfg_rescale: ["cfgRescaleRange", "cfgRescaleNumber"],
  hires_cfg_rescale: ["hiresCfgRescale"],
  sampler_name: ["samplerName"],
  scheduler_name: ["schedulerName"],
  hires_upscaler_id: ["hiresUpscaler"],
  text_encoder_3: ["sd3T5Enabled", "sd3T5Source", "sd3T5Device"],
  model_path: ["modelPath"],
};
const FIELD_STATUS_IDS = {
  cfg_rescale: "cfgRescaleStatus",
  hires_cfg_rescale: "hiresCfgRescaleStatus",
  sampler_name: "samplerCapabilityStatus",
  scheduler_name: "schedulerCapabilityStatus",
  hires_upscaler_id: "hiresUpscalerStatus",
  text_encoder_3: "sd3T5Status",
  model_path: "generationCapabilityStatus",
};

function setStatus(id, text, severity = "info") {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = text || "";
  element.dataset.status = severity;
  element.classList.toggle("error", severity === "error");
  element.classList.toggle("warning", severity === "warning");
  element.classList.toggle("ready", severity === "info" || severity === "ready");
}

function currentValue(id, fallback = "") {
  const element = document.getElementById(id);
  if (!element) return fallback;
  if (element.type === "checkbox") return Boolean(element.checked);
  return element.value ?? fallback;
}

function selectedUpscalerId() {
  return String(currentValue("hiresUpscaler", "") || currentValue("hiresUpscalerId", "")).trim();
}

function selectElement(id) {
  const element = document.getElementById(id);
  return element instanceof HTMLSelectElement ? element : null;
}

function controlByName(name) {
  return state.generationCapabilities?.controls?.[name] || null;
}

function preferenceStore() {
  const existing = state.generationCapabilityPreferences;
  if (existing && existing.schema_version === PREFERENCE_SCHEMA_VERSION && existing.contexts && typeof existing.contexts === "object") {
    return existing;
  }
  const created = { schema_version: PREFERENCE_SCHEMA_VERSION, contexts: {} };
  state.generationCapabilityPreferences = created;
  return created;
}

function capabilityContextKey(capabilities = state.generationCapabilities) {
  const model = capabilities?.model || {};
  const architecture = String(model.architecture || "unresolved").trim().toLowerCase() || "unresolved";
  const profile = String(model.profile || "default").trim().toLowerCase() || "default";
  return `${architecture}::${profile}::base::${STANDARD_MODE}`;
}

function contextPreferences(capabilities = state.generationCapabilities) {
  const store = preferenceStore();
  const key = capabilityContextKey(capabilities);
  if (!store.contexts[key]) store.contexts[key] = {};
  return store.contexts[key];
}

function numericInputValue(id) {
  const value = Number(document.getElementById(id)?.value);
  return Number.isFinite(value) ? value : null;
}

function rememberCurrentContext(capabilities = state.generationCapabilities) {
  if (!capabilities || capabilities.binding?.status !== "bound") return;
  const prefs = contextPreferences(capabilities);
  const cfg = capabilities.controls?.cfg_rescale || {};
  if (cfg.state === "supported" || cfg.state === "constrained") {
    const value = numericInputValue("cfgRescaleNumber");
    if (value !== null) prefs.cfg_rescale = value;
    const hiresValue = numericInputValue("hiresCfgRescale");
    if (hiresValue !== null) prefs.hires_cfg_rescale = hiresValue;
  }
  const sampler = capabilities.controls?.sampler_name || {};
  const samplerValue = String(currentValue("samplerName", "")).trim();
  if (samplerValue && (sampler.allowed_names || []).includes(samplerValue)) prefs.sampler_name = samplerValue;
  const scheduler = capabilities.controls?.scheduler_name || {};
  const schedulerValue = String(currentValue("schedulerName", "")).trim();
  if (schedulerValue && (scheduler.allowed_names || []).includes(schedulerValue)) prefs.scheduler_name = schedulerValue;
  const t5 = capabilities.controls?.text_encoder_3 || {};
  if (t5.available) {
    prefs.sd3_t5_enabled = Boolean(currentValue("sd3T5Enabled", false));
    prefs.sd3_t5_source = String(currentValue("sd3T5Source", t5.effective_source || t5.default_source || "") || "");
    prefs.text_encoder_3_device = String(currentValue("sd3T5Device", "auto") || "auto");
  }
}

export function loadGenerationCapabilityPreferences(value = {}) {
  if (!value || typeof value !== "object") return preferenceStore();
  const contexts = value.contexts && typeof value.contexts === "object" ? value.contexts : {};
  state.generationCapabilityPreferences = {
    schema_version: PREFERENCE_SCHEMA_VERSION,
    contexts: structuredClone(contexts),
  };
  return state.generationCapabilityPreferences;
}

export function generationCapabilityPreferencesSnapshot({ captureCurrent = true } = {}) {
  if (captureCurrent) rememberCurrentContext();
  return structuredClone(preferenceStore());
}

function setControlDisabled(ids, disabled, explanation = "") {
  ids.forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.disabled = Boolean(disabled);
    node.setAttribute("aria-disabled", disabled ? "true" : "false");
    if (explanation) node.title = explanation;
  });
}

function setCfgRescaleValue(value) {
  const text = String(Number(value));
  const range = document.getElementById("cfgRescaleRange");
  const number = document.getElementById("cfgRescaleNumber");
  if (range) range.value = text;
  if (number) number.value = text;
}

function setHiresCfgRescaleValue(value) {
  const input = document.getElementById("hiresCfgRescale");
  if (input) input.value = String(Number(value));
}

function clearFieldError(fieldName) {
  (FIELD_IDS[fieldName] || []).forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.classList.remove("capability-invalid", "field-error-focus");
    node.removeAttribute("aria-invalid");
    node.removeAttribute("data-capability-issue-code");
    node.removeAttribute("aria-errormessage");
  });
}

function markFieldError(fieldName, code, message) {
  (FIELD_IDS[fieldName] || []).forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.classList.add("capability-invalid");
    node.setAttribute("aria-invalid", "true");
    node.dataset.capabilityIssueCode = String(code || "capability_invalid");
    const errorStatusId = FIELD_STATUS_IDS[fieldName];
    if (errorStatusId) node.setAttribute("aria-errormessage", errorStatusId);
    if (message) node.title = message;
  });
}

function reconcileFixedCfg(control, prefs) {
  const explanation = control.explanation || "CFG rescale is fixed by the active model contract.";
  const fixedValue = Number(control.effective_value ?? 0);
  setCfgRescaleValue(fixedValue);
  setHiresCfgRescaleValue(fixedValue);
  setControlDisabled(FIELD_IDS.cfg_rescale, true, explanation);
  setControlDisabled(FIELD_IDS.hires_cfg_rescale, true, explanation);
  [...FIELD_IDS.cfg_rescale, ...FIELD_IDS.hires_cfg_rescale].forEach((id) => document.getElementById(id)?.classList.add("capability-fixed"));
  clearFieldError("cfg_rescale");
  clearFieldError("hires_cfg_rescale");
  setStatus("cfgRescaleStatus", `CFG rescale fixed at ${fixedValue.toFixed(2)} for this model contract.`, "info");
  setStatus("hiresCfgRescaleStatus", `Hires CFG rescale is fixed at ${fixedValue.toFixed(2)} by the same active model contract.`, "info");
  return prefs;
}

function reconcileEditableCfg(control, prefs) {
  setControlDisabled(FIELD_IDS.cfg_rescale, false, control.explanation || "");
  setControlDisabled(FIELD_IDS.hires_cfg_rescale, false, control.explanation || "");
  [...FIELD_IDS.cfg_rescale, ...FIELD_IDS.hires_cfg_rescale].forEach((id) => document.getElementById(id)?.classList.remove("capability-fixed"));
  if (Number.isFinite(Number(prefs.cfg_rescale))) setCfgRescaleValue(Number(prefs.cfg_rescale));
  if (Number.isFinite(Number(prefs.hires_cfg_rescale))) setHiresCfgRescaleValue(Number(prefs.hires_cfg_rescale));
  clearFieldError("cfg_rescale");
  clearFieldError("hires_cfg_rescale");
  setStatus("cfgRescaleStatus", "CFG rescale is available for this model contract.", "info");
  setStatus("hiresCfgRescaleStatus", "Hires CFG rescale may inherit or use its saved value for this capability context.", "info");
}

function reconcileSelect(selectId, control, prefs, preferenceKey, statusId, label) {
  const select = selectElement(selectId);
  if (!select) return { changed: false, blocking: false };
  const allowedNames = Array.isArray(control?.allowed_names) ? control.allowed_names : [];
  const allowed = new Set(allowedNames);
  const restrict = control?.state === "constrained" || control?.state === "unsupported";
  Array.from(select.options).forEach((option) => {
    option.disabled = Boolean(option.value && restrict && !allowed.has(option.value));
  });

  if (control?.state === "unsupported" || !allowedNames.length) {
    markFieldError(preferenceKey, control?.reason_code || "model_capability_unresolved", `${label} compatibility is unresolved.`);
    setStatus(statusId, `${label}: no qualified option is available for the active capability contract.`, "error");
    return { changed: false, blocking: true };
  }

  const current = String(select.value || "").trim();
  if (current && allowed.has(current)) {
    clearFieldError(preferenceKey);
    const pairedWith = String(control?.paired_with || "").trim();
    setStatus(
      statusId,
      pairedWith
        ? `${label}: ${allowedNames.length} compatible option(s) with ${pairedWith}.`
        : `${label}: ${allowedNames.length} compatible option(s).`,
      "info",
    );
    return { changed: false, blocking: false };
  }

  const stored = String(prefs[preferenceKey] || "").trim();
  const replacement = [stored, control?.replacement_name, control?.preferred_name]
    .map((value) => String(value || "").trim())
    .find((value) => value && allowed.has(value)) || "";
  if (!replacement) {
    markFieldError(preferenceKey, control?.reason_code || "capability_no_replacement", `${label} needs a compatible value.`);
    setStatus(statusId, `${label}: choose one of the qualified values for this model.`, "error");
    return { changed: false, blocking: true };
  }

  const previous = current;
  select.value = replacement;
  prefs[preferenceKey] = replacement;
  clearFieldError(preferenceKey);
  setStatus(
    statusId,
    previous
      ? `${label}: ${replacement} selected because ${previous} is incompatible with the active model contract.`
      : `${label}: ${replacement} selected from the active model/profile policy.`,
    "info",
  );
  select.dispatchEvent(new CustomEvent("image-gen-capability-auto-repaired", {
    bubbles: true,
    detail: { field: preferenceKey, previous, replacement, reason_code: control?.reason_code || "" },
  }));
  return { changed: true, blocking: false };
}

function reconcileHiresRequirement(control) {
  clearFieldError("hires_upscaler_id");
  const recovery = document.getElementById("hiresUpscalerRecovery");
  if (recovery) recovery.hidden = true;
  const enabled = Boolean(currentValue("hiresEnabled", false));
  const strategy = String(currentValue("hiresStrategy", "pixel_neural")).trim().toLowerCase();
  if (!enabled || strategy !== "pixel_neural") return { blocking: false, reasons: [] };

  const allowedIds = new Set(control?.strategies?.pixel_neural?.allowed_upscaler_ids || []);
  const selected = selectedUpscalerId();
  if (selected && allowedIds.has(selected) && control?.blocking !== true) return { blocking: false, reasons: [] };

  const code = control?.reason_code || "required_asset_missing";
  const message = control?.explanation || "Pixel-neural hires requires an installed qualified upscaler with a stable ID.";
  markFieldError("hires_upscaler_id", code, message);
  setStatus("hiresUpscalerStatus", `${message} Choose a supported value from the Upscaler list. [${code}]`, "error");
  if (recovery) recovery.hidden = false;
  const recoveryMessage = document.getElementById("hiresUpscalerRecoveryMessage");
  if (recoveryMessage) recoveryMessage.textContent = `${message} Select a qualified upscaler, refresh discovery, or update the configured asset path.`;
  return { blocking: true, reasons: [{ code, field: "hires_upscaler_id", message }] };
}

function reconcileT5Control(control, prefs) {
  const row = document.getElementById("modelT5Control");
  const enabled = document.getElementById("sd3T5Enabled");
  const sourceBlock = document.getElementById("sd3T5SourceBlock");
  const source = document.getElementById("sd3T5Source");
  const deviceBlock = document.getElementById("sd3T5DeviceBlock");
  const device = document.getElementById("sd3T5Device");
  if (!row || !enabled || !source || !device) return { blocking: false, changed: false };

  const available = Boolean(control?.available) && control?.state !== "hidden";
  row.hidden = !available;
  row.classList.toggle("is-hidden", !available);
  if (!available) {
    enabled.checked = false;
    enabled.disabled = true;
    source.replaceChildren();
    source.disabled = true;
    device.value = "auto";
    device.disabled = true;
    if (sourceBlock) sourceBlock.hidden = true;
    if (deviceBlock) deviceBlock.hidden = true;
    clearFieldError("text_encoder_3");
    return { blocking: false, changed: false };
  }

  enabled.disabled = false;
  enabled.setAttribute("aria-disabled", "false");
  const hasStored = Object.prototype.hasOwnProperty.call(prefs, "sd3_t5_enabled");
  const nextEnabled = hasStored ? Boolean(prefs.sd3_t5_enabled) : Boolean(control?.enabled ?? control?.default_enabled);
  let changed = enabled.checked !== nextEnabled;
  enabled.checked = nextEnabled;

  const options = Array.isArray(control?.source_options) ? control.source_options : [];
  const previousOptions = Array.from(source.options).map((item) => `${item.value}::${item.textContent}`).join("|");
  const nextOptions = options.map((item) => `${String(item?.value || "")}::${String(item?.label || item?.value || "T5")}`).join("|");
  if (previousOptions !== nextOptions) {
    source.replaceChildren(...options.map((item) => {
      const option = document.createElement("option");
      option.value = String(item?.value || "");
      option.textContent = String(item?.label || item?.value || "T5");
      return option;
    }));
  }
  const allowedSources = new Set(options.map((item) => String(item?.value || "")).filter(Boolean));
  const storedSource = String(prefs.sd3_t5_source || source.value || control?.effective_source || control?.default_source || "").trim().toLowerCase();
  const fallbackSource = String(control?.effective_source || control?.default_source || options[0]?.value || "").trim().toLowerCase();
  const nextSource = allowedSources.has(storedSource) ? storedSource : fallbackSource;
  if (source.value !== nextSource) changed = true;
  source.value = nextSource;
  source.disabled = options.length === 0;
  source.setAttribute("aria-disabled", source.disabled ? "true" : "false");
  if (sourceBlock) sourceBlock.hidden = false;

  const allowedDevices = new Set(control?.allowed_devices || ["auto", "cpu", "cuda"]);
  const storedDevice = String(prefs.text_encoder_3_device || device.value || "auto").trim().toLowerCase();
  device.value = allowedDevices.has(storedDevice) ? storedDevice : "auto";
  device.disabled = !nextEnabled;
  device.setAttribute("aria-disabled", nextEnabled ? "false" : "true");
  if (deviceBlock) deviceBlock.hidden = !nextEnabled;

  const selectedOption = options.find((item) => String(item?.value || "") === nextSource);
  const selectedLabel = String(selectedOption?.label || "qualified T5/T5XXL source");
  setStatus(
    "sd3T5Status",
    nextEnabled
      ? `T5/T5XXL conditioning enabled using ${selectedLabel}.`
      : `${options.length} qualified T5 source${options.length === 1 ? " is" : "s are"} available; enable T5 to use the selected model.`,
    "info",
  );
  clearFieldError("text_encoder_3");
  return { blocking: false, changed };
}

function renderModelCapabilityStatus(capabilities) {
  const model = capabilities?.model || {};
  const binding = capabilities?.binding || {};
  if (binding.status !== "bound") {
    markFieldError("model_path", binding.reason_code || "no_active_model", "An authoritative model capability contract is required before generation.");
    setStatus("generationCapabilityStatus", "Capability contract is unresolved. Generation is disabled until a model is authoritatively bound.", "error");
    return;
  }
  clearFieldError("model_path");
  const domain = String(model.prediction_domain || "unknown").replaceAll("_", "-");
  const alignment = model.pixel_alignment_multiple ? `${model.pixel_alignment_multiple}px alignment` : "alignment unresolved";
  const identity = model.fingerprint_verified
    ? `${model.identity_source || "component registry"} / fingerprint-verified`
    : String(model.identity_source || "structural capability evidence").replaceAll("_", " ");
  setStatus("generationCapabilityStatus", `Capability contract · ${domain} · ${alignment} · ${identity}`, "info");
}

function updateSubmissionGate(blockingReasons) {
  state.generationCapabilityBlockingReasons = Array.isArray(blockingReasons) ? blockingReasons : [];
  const blocked = state.generationCapabilityBlockingReasons.length > 0;
  ["generateButton", "infinityButton", "generateMenuButton"].forEach((id) => {
    const button = document.getElementById(id);
    if (!button) return;
    button.dataset.capabilityBlocked = blocked ? "true" : "false";
    if (!button.getAttribute("aria-busy") || button.getAttribute("aria-busy") === "false") {
      button.disabled = blocked;
    }
    if (blocked) button.title = state.generationCapabilityBlockingReasons.map((item) => item.message).join(" | ");
    else if (button.dataset.capabilityBlocked === "false") button.removeAttribute("title");
  });
}

export function generationCapabilityBlocksSubmission() {
  return Array.isArray(state.generationCapabilityBlockingReasons) && state.generationCapabilityBlockingReasons.length > 0;
}

export function getGenerationCapabilities() {
  return state.generationCapabilities || null;
}

export function generationSpatialRequirements() {
  const spatial = state.generationCapabilities?.spatial || {};
  const latentScale = Math.max(1, Math.round(Number(spatial.latent_scale_factor) || 8));
  const pixelAlignment = Math.max(1, Math.round(Number(spatial.pixel_alignment_multiple) || latentScale));
  return {
    latentScaleFactor: latentScale,
    pixelAlignmentMultiple: pixelAlignment,
    authoritative: Number(spatial.pixel_alignment_multiple) > 0 && Number(spatial.latent_scale_factor) > 0,
    source: String(spatial.source || ""),
  };
}

export function cfgRescaleCapability() {
  return controlByName("cfg_rescale");
}

export function collectGenerationCapabilityRequest({ changedField = "" } = {}) {
  return {
    _capability_changed_field: String(changedField || ""),
    model_path: currentValue("modelPath", ""),
    sampler_name: currentValue("samplerName", ""),
    scheduler_name: currentValue("schedulerName", ""),
    cfg_guidance_mode: currentValue("cfgGuidanceMode", ""),
    cfg_rescale: Number(currentValue("cfgRescaleNumber", 0) || currentValue("cfgRescaleRange", 0) || 0),
    hires_enabled: Boolean(currentValue("hiresEnabled", false)),
    hires_strategy: currentValue("hiresStrategy", "pixel_neural"),
    hires_upscaler_id: selectedUpscalerId(),
    advanced_models_enabled: Boolean(currentValue("advancedModelsEnabled", false)),
    advanced_model_family: currentValue("advancedModelFamily", ""),
    advanced_model_components: Object.fromEntries(
      Array.from(document.querySelectorAll("[data-advanced-component-role]")).map((node) => [node.dataset.advancedComponentRole, node.value]),
    ),
    advanced_model_allow_digital_components: Boolean(document.getElementById("advancedModelAllowDigitalComponents")?.checked ?? true),
    advanced_model_t5_device: currentValue("advancedModelT5Device", "cpu"),
    sd3_t5_enabled: Boolean(currentValue("sd3T5Enabled", false)),
    sd3_t5_source: currentValue("sd3T5Source", "auto"),
    text_encoder_3_device: currentValue("sd3T5Device", "auto"),
    vae_path: currentValue("vaePath", ""),
  };
}

export function applyGenerationCapabilities(payload) {
  const previous = state.generationCapabilities;
  if (previous && previous !== payload) rememberCurrentContext(previous);
  state.generationCapabilities = payload || null;
  const capabilities = state.generationCapabilities;
  if (!capabilities) {
    updateSubmissionGate([{ code: "capability_contract_unavailable", field: "model_path", message: "Capability contract unavailable." }]);
    setStatus("generationCapabilityStatus", "Capability contract unavailable. Generation is disabled.", "error");
    return null;
  }

  renderModelCapabilityStatus(capabilities);
  const bindingBlocked = capabilities.binding?.status !== "bound";
  const prefs = contextPreferences(capabilities);
  const controls = capabilities.controls || {};

  const cfg = controls.cfg_rescale || {};
  if (cfg.state === "fixed") reconcileFixedCfg(cfg, prefs);
  else if (["supported", "constrained"].includes(cfg.state)) reconcileEditableCfg(cfg, prefs);
  else {
    setControlDisabled([...FIELD_IDS.cfg_rescale, ...FIELD_IDS.hires_cfg_rescale], true, cfg.explanation || "CFG rescale is unavailable.");
    markFieldError("cfg_rescale", cfg.reason_code || "model_capability_unresolved", cfg.explanation || "CFG rescale is unresolved.");
    setStatus("cfgRescaleStatus", cfg.explanation || "CFG rescale is unresolved.", "error");
  }

  const samplerResult = reconcileSelect("samplerName", controls.sampler_name || {}, prefs, "sampler_name", "samplerCapabilityStatus", "Sampler");
  const schedulerResult = reconcileSelect("schedulerName", controls.scheduler_name || {}, prefs, "scheduler_name", "schedulerCapabilityStatus", "Scheduler");
  const hiresResult = reconcileHiresRequirement(controls.hires_strategy || {});
  const t5Result = reconcileT5Control(controls.text_encoder_3 || {}, prefs);

  const reasons = [];
  if (bindingBlocked) reasons.push({ code: capabilities.binding?.reason_code || "no_active_model", field: "model_path", message: "An authoritative model capability contract is required." });
  if (samplerResult.blocking) reasons.push({ code: controls.sampler_name?.reason_code || "sampler_unresolved", field: "sampler_name", message: "Choose a compatible sampler." });
  if (schedulerResult.blocking) reasons.push({ code: controls.scheduler_name?.reason_code || "scheduler_unresolved", field: "scheduler_name", message: "Choose a compatible scheduler." });
  if (!["supported", "constrained", "fixed"].includes(cfg.state)) reasons.push({ code: cfg.reason_code || "cfg_rescale_unresolved", field: "cfg_rescale", message: cfg.explanation || "CFG rescale is unresolved." });
  reasons.push(...hiresResult.reasons);
  updateSubmissionGate(reasons);

  window.dispatchEvent(new CustomEvent("image-gen-capabilities-changed", {
    detail: {
      capabilities,
      context_key: capabilityContextKey(capabilities),
      auto_repaired: Boolean(samplerResult.changed || schedulerResult.changed || t5Result.changed),
      blocking_reasons: reasons,
    },
  }));
  return capabilities;
}

export async function refreshGenerationCapabilities({ changedField = "" } = {}) {
  const response = await api.generationCapabilities(collectGenerationCapabilityRequest({ changedField }));
  return applyGenerationCapabilities(response);
}
