import { api } from "../api.js?v=ha3";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function valuesEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

const HIRES_LIFECYCLE_VERSION = 1;
const HIRES_LIFECYCLE_METADATA_KEYS = new Set([
  "hires_configuration_mode",
  "hires_auto_resolution_record",
  "hires_lifecycle_state",
  "hires_dimension_plan_version",
  "hires_dimension_plan",
  "hires_axis_scale_width",
  "hires_axis_scale_height",
  "hires_uniform_scale",
  "hires_aspect_ratio_changed",
  "hires_recorded_schedule_replay",
  "hires_recorded_schedule_fingerprint",
  "hires_schedule_conformance_source_replay",
  "hires_schedule_conformance_source_fingerprint",
  "hires_expected_upscaler_sha256",
  "hires_expected_native_scale",
  "hires_diagnostic_vae_execution_fingerprint",
  "hires_recorded_target_correction",
  "hires_recorded_correction_fingerprint",
]);
const HIRES_CONTEXT_SELECTORS = ["#modelPath", "#width", "#height", "#samplerName", "#schedulerName", "#cfgScale", "#cfgRescale"];

function uniqueValues(values = []) {
  return [...new Set((values || []).filter(Boolean).map((item) => String(item)))];
}

function profileDisplayName(profile = {}) {
  return String(profile.name || profile.label || profile.profile_id || "Saved hires preset");
}

function profileCompatibility(profile = {}) {
  const compatibility = clone(profile.compatibility || {});
  compatibility.model_families = uniqueValues(compatibility.model_families || []);
  compatibility.upscaler_sha256 = uniqueValues(compatibility.upscaler_sha256 || []);
  return compatibility;
}

export function trackedHiresValues(values = {}) {
  const tracked = {};
  Object.entries(values || {}).forEach(([key, value]) => {
    if (!String(key).startsWith("hires_")) return;
    if (HIRES_LIFECYCLE_METADATA_KEYS.has(key)) return;
    tracked[key] = clone(value);
  });
  return tracked;
}

export function modifiedHiresFields(baselineValues = {}, currentValues = {}) {
  const baseline = baselineValues || {};
  const current = currentValues || {};
  const keys = new Set([...Object.keys(baseline), ...Object.keys(current)]);
  return [...keys].filter((key) => !valuesEqual(baseline[key], current[key])).sort();
}

function modeDisplayLabel(mode = "custom") {
  if (mode === "auto") return "Auto";
  if (mode === "profile") return "Profile";
  return "Custom";
}

function autoBaselineLabel(resolutionRecord = {}, fallbackFamily = "") {
  const profileNames = uniqueValues((resolutionRecord.applied_profiles || []).map((item) => item?.name || item?.profile_name || item?.label));
  if (profileNames.length) return `Auto (${profileNames.join(" + ")})`;
  const family = normalizeFamily(resolutionRecord?.diagnostics?.family || fallbackFamily || "");
  return family ? `Auto (${family})` : "Auto";
}

export function lifecycleStateFromAuto({ values = {}, resolutionRecord = {}, context = {} } = {}) {
  const currentValues = { ...(values || {}), hires_configuration_mode: "auto", hires_auto_resolution_record: clone(resolutionRecord || {}) };
  return {
    version: HIRES_LIFECYCLE_VERSION,
    active_mode: "auto",
    baseline_mode: "auto",
    baseline_label: autoBaselineLabel(resolutionRecord, context.model_family || ""),
    baseline_profile_id: "",
    baseline_resolution_fingerprint: String(resolutionRecord?.resolution_fingerprint || ""),
    baseline_values: trackedHiresValues(currentValues),
    modified_fields: [],
    baseline_context: {
      model_family: normalizeFamily(context.model_family || resolutionRecord?.diagnostics?.family || ""),
      checkpoint_sha256: String(context.checkpoint_sha256 || "").toLowerCase(),
      upscaler_id: String(resolutionRecord?.selected_upscaler?.upscaler_id || resolutionRecord?.selected_upscaler?.id || currentValues.hires_upscaler_id || currentValues.hires_upscaler || ""),
      upscaler_sha256: String(resolutionRecord?.selected_upscaler?.sha256 || "").toLowerCase(),
    },
    source_profile_id: "",
    source_profile_name: "",
    source_profile_compatibility: {},
  };
}

export function lifecycleStateFromProfile({ values = {}, profile = {}, context = {} } = {}) {
  const currentValues = { ...(values || {}), hires_configuration_mode: "profile", hires_auto_resolution_record: {} };
  return {
    version: HIRES_LIFECYCLE_VERSION,
    active_mode: "profile",
    baseline_mode: "profile",
    baseline_label: profileDisplayName(profile),
    baseline_profile_id: String(profile.profile_id || ""),
    baseline_resolution_fingerprint: "",
    baseline_values: trackedHiresValues(currentValues),
    modified_fields: [],
    baseline_context: {
      model_family: normalizeFamily(context.model_family || ""),
      checkpoint_sha256: String(context.checkpoint_sha256 || "").toLowerCase(),
      upscaler_id: String(context.upscaler_id || currentValues.hires_upscaler_id || currentValues.hires_upscaler || ""),
      upscaler_sha256: String(context.upscaler_sha256 || "").toLowerCase(),
    },
    source_profile_id: String(profile.profile_id || ""),
    source_profile_name: profileDisplayName(profile),
    source_profile_compatibility: profileCompatibility(profile),
  };
}

export function normalizeLifecycleState(value = {}, currentValues = {}) {
  const stateValue = value && typeof value === "object" ? clone(value) : {};
  const baselineValues = trackedHiresValues(stateValue.baseline_values || {});
  const activeMode = ["auto", "profile", "custom"].includes(stateValue.active_mode)
    ? stateValue.active_mode
    : (["auto", "profile", "custom"].includes(currentValues?.hires_configuration_mode) ? currentValues.hires_configuration_mode : "custom");
  const baselineMode = ["auto", "profile"].includes(stateValue.baseline_mode)
    ? stateValue.baseline_mode
    : (activeMode === "custom" ? "custom" : activeMode);
  const currentTracked = trackedHiresValues(currentValues || {});
  const modified = modifiedHiresFields(baselineValues, currentTracked);
  const normalized = {
    version: HIRES_LIFECYCLE_VERSION,
    active_mode: activeMode,
    baseline_mode: baselineMode,
    baseline_label: String(stateValue.baseline_label || ""),
    baseline_profile_id: String(stateValue.baseline_profile_id || ""),
    baseline_resolution_fingerprint: String(stateValue.baseline_resolution_fingerprint || ""),
    baseline_values: baselineValues,
    modified_fields: modified,
    baseline_context: clone(stateValue.baseline_context || {}),
    source_profile_id: String(stateValue.source_profile_id || stateValue.baseline_profile_id || ""),
    source_profile_name: String(stateValue.source_profile_name || stateValue.baseline_label || ""),
    source_profile_compatibility: clone(stateValue.source_profile_compatibility || {}),
  };
  if (!Object.keys(normalized.baseline_values).length && Object.keys(currentTracked).length && activeMode !== "custom") {
    normalized.baseline_values = currentTracked;
    normalized.modified_fields = [];
  }
  if (!normalized.baseline_label) {
    normalized.baseline_label = normalized.baseline_mode === "profile"
      ? (normalized.source_profile_name || "Saved hires preset")
      : (normalized.baseline_mode === "auto" ? "Auto" : "Current settings");
  }
  return normalized;
}

export function transitionLifecycleState(lifecycleState = {}, currentValues = {}) {
  const currentTracked = trackedHiresValues(currentValues || {});
  const next = normalizeLifecycleState(lifecycleState, currentValues);
  if (!Object.keys(next.baseline_values || {}).length) {
    next.baseline_values = currentTracked;
    next.modified_fields = [];
    return next;
  }
  const modified = modifiedHiresFields(next.baseline_values, currentTracked);
  next.modified_fields = modified;
  if (!modified.length) {
    next.active_mode = next.baseline_mode === "custom" ? "custom" : next.baseline_mode;
    return next;
  }
  next.active_mode = "custom";
  return next;
}

export function lifecycleCompatibilityNotice(lifecycleState = {}, context = {}, profiles = []) {
  const lifecycle = normalizeLifecycleState(lifecycleState, {});
  const family = normalizeFamily(context.model_family || "");
  const checkpointSha = String(context.checkpoint_sha256 || "").toLowerCase();
  if (!family && !checkpointSha) return "";
  if (lifecycle.active_mode === "custom" && lifecycle.baseline_mode === "auto") {
    const baselineFamily = normalizeFamily(lifecycle.baseline_context?.model_family || "");
    if (baselineFamily && family && baselineFamily !== family) {
      return `Custom hires settings are still based on ${lifecycle.baseline_label} from ${baselineFamily}, but the active checkpoint family is ${family}. Reapply Auto if you want the recommended settings for this model.`;
    }
    return "";
  }
  const profileId = lifecycle.source_profile_id || lifecycle.baseline_profile_id;
  const profile = (profiles || []).find((item) => String(item.profile_id || "") === String(profileId || "")) || {};
  const compatibility = profileCompatibility(profileId ? profile : (lifecycle.source_profile_compatibility ? { compatibility: lifecycle.source_profile_compatibility } : {}));
  if (compatibility.model_families?.length && family && !compatibility.model_families.includes(family)) {
    return `${lifecycle.source_profile_name || lifecycle.baseline_label || "This preset"} targets ${compatibility.model_families.join(", ")}, but the active model family is ${family}. The profile remains loaded; review the settings if that mismatch was not intentional.`;
  }
  return "";
}

export function editorModelForDescriptor(descriptor = {}) {
  const allowed = Array.isArray(descriptor.allowed_values) ? descriptor.allowed_values : [];
  if (descriptor.editable === false || descriptor.editor_kind === "read_only" || descriptor.editor_kind === "dynamic_object") {
    return { kind: "read_only" };
  }
  if (descriptor.value_type === "boolean" || descriptor.editor_kind === "boolean") return { kind: "boolean" };
  if (descriptor.value_type === "array") return allowed.length ? { kind: "multi_select", options: allowed } : { kind: "read_only" };
  if (allowed.length || ["select", "asset_select"].includes(descriptor.editor_kind)) return { kind: "select", options: allowed };
  if (["integer", "number"].includes(descriptor.value_type) || ["integer", "number"].includes(descriptor.editor_kind)) {
    return {
      kind: descriptor.current_value == null ? "nullable_number" : "number",
      minimum: descriptor.minimum,
      maximum: descriptor.maximum,
      step: descriptor.step,
    };
  }
  return { kind: "read_only" };
}

export function descriptorsForView(descriptors = [], view = "all") {
  const list = Array.isArray(descriptors) ? descriptors : [];
  return view === "modified" ? list.filter((item) => Boolean(item.modified)) : list;
}

export function groupDescriptors(descriptors = []) {
  const groups = new Map();
  descriptors.forEach((descriptor) => {
    const group = String(descriptor.group || "Other");
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(descriptor);
  });
  return [...groups.entries()].map(([name, items]) => ({ name, items }));
}

export function createDraftFromSchema(schemaPayload = {}, currentValues = {}, index = 1, options = {}) {
  const manifest = clone(schemaPayload.manifest || {});
  const baselineValues = trackedHiresValues(options.baseline_values || {});
  const baselineLabel = String(options.baseline_label || "");
  const descriptors = (manifest.descriptors || []).map((descriptor) => {
    const item = clone(descriptor);
    if (Object.prototype.hasOwnProperty.call(currentValues, item.key)) {
      item.current_value = clone(currentValues[item.key]);
    }
    if (Object.prototype.hasOwnProperty.call(baselineValues, item.key)) {
      item.baseline_value = clone(baselineValues[item.key]);
    }
    item.included = item.persistence_eligibility === "eligible";
    item.modified = !valuesEqual(item.current_value, item.baseline_value);
    return item;
  });
  return {
    profile: {
      profile_id: "",
      name: `Hires Preset ${Math.max(1, Number(index) || 1)}`,
      description: "",
      read_only: false,
      source: "draft",
      baseline_profile_id: "",
      compatibility: {},
      values: {},
      included_fields: [],
      baseline_label: baselineLabel,
    },
    manifest: { ...manifest, descriptors },
    assignments: [],
  };
}

function normalizeFamily(value = "") {
  const text = String(value || "").trim().toLowerCase();
  if (text.includes("sdxl")) return "sdxl";
  if (text.startsWith("sd3") || text.includes("stable diffusion 3")) return "sd3.x";
  if (text.startsWith("sd2") || text.includes("stable diffusion 2")) return "sd2.x";
  if (text.startsWith("sd1") || text.includes("stable diffusion 1")) return "sd1.x";
  return text;
}

function activeCheckpointContext() {
  const active = state.activeModel || {};
  const selectedPath = $("#modelPath")?.value || "";
  const catalog = (state.models || []).find((item) => String(item.path || "") === String(selectedPath)) || {};
  const sha256 = String(active.sha256 || active.model_hash || active.resolved_hash || catalog.sha256 || catalog.model_hash || "").toLowerCase();
  const family = normalizeFamily(active.architecture || active.runtime_profile?.architecture || catalog.architecture || catalog.architecture_summary || "");
  return {
    sha256: /^[0-9a-f]{64}$/.test(sha256) ? sha256 : "",
    family,
    label: active.model_name || active.file_name || catalog.name || catalog.file_name || selectedPath.split(/[\\/]/).pop() || "Current checkpoint",
  };
}

function activeUpscalerContext() {
  const id = String($("#hiresUpscaler")?.value || "");
  const item = (state.upscalers?.neural || []).find((candidate) => String(candidate.upscaler_id || "") === id) || {};
  const sha256 = String(item.sha256 || "").toLowerCase();
  return {
    id,
    sha256: /^[0-9a-f]{64}$/.test(sha256) ? sha256 : "",
    label: item.display_name || item.file_name || id || "Current upscaler",
  };
}

function textForValue(value) {
  if (value == null) return "Inherit / automatic";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None selected";
  if (typeof value === "object") return "Structured settings managed by the selected component";
  return String(value);
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function optionNode(value, label, disabled = false) {
  const node = document.createElement("option");
  node.value = String(value ?? "");
  node.textContent = String(label ?? value ?? "");
  node.disabled = Boolean(disabled);
  return node;
}

let model = {
  profiles: [],
  allAssignments: [],
  schema: null,
  familyChoices: [],
  compatibilityChoices: { model_families: [], upscalers: [] },
  selectedProfileId: "",
  draft: null,
  view: "all",
  collect: null,
  apply: null,
  saveSessionSoon: null,
  lastResolution: null,
  lifecycleTrackingSuspended: false,
  lastContextSignature: "",
};

function autoResolutionContext() {
  const values = model.collect?.() || {};
  const checkpoint = activeCheckpointContext();
  const active = state.activeModel || {};
  const runtimeProfile = clone(active.runtime_profile || active.model_runtime_profile || {});
  const vaeContract = clone(active.latent_vae_contract || runtimeProfile?.latent_vae_contract || {});
  return {
    model_family: checkpoint.family,
    checkpoint_sha256: checkpoint.sha256,
    current_dimensions: { width: Number(values.width || 512), height: Number(values.height || 512) },
    requested_scale: Number(values.hires_scale || 2),
    requested_target: { width: Number(values.hires_width || 0), height: Number(values.hires_height || 0) },
    explicit_user_upscaler: "",
    preferred_user_upscaler: String(state.settings?.preferred_hires_upscaler_id || ""),
    base_values: {
      sampler_name: values.sampler_name || "",
      scheduler_name: values.scheduler_name || "",
      cfg_scale: values.cfg_scale,
      cfg_rescale: values.cfg_rescale,
      prompt_parser_name: values.prompt_parser_name || "legacy",
      prompt_shortcut_profile_name: values.prompt_shortcut_profile_name || "legacy_default",
    },
    runtime_profile: runtimeProfile,
    vae_contract: vaeContract,
  };
}

function currentLifecycleState(values = null) {
  const currentValues = values || model.collect?.() || {};
  return normalizeLifecycleState(currentValues.hires_lifecycle_state || jsonValueSafe("#hiresLifecycleState"), currentValues);
}

function jsonValueSafe(selector) {
  try {
    const raw = $(selector)?.value || "{}";
    const parsed = JSON.parse(raw || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeLifecycleState(stateValue = {}) {
  const normalized = normalizeLifecycleState(stateValue, model.collect?.() || {});
  const hidden = $("#hiresLifecycleState");
  if (hidden) hidden.value = JSON.stringify(normalized);
  const mode = $("#hiresConfigurationMode");
  if (mode) mode.value = normalized.active_mode || "custom";
  return normalized;
}

function activeContextSummary() {
  const checkpoint = activeCheckpointContext();
  const upscaler = activeUpscalerContext();
  return {
    model_family: checkpoint.family,
    checkpoint_sha256: checkpoint.sha256,
    model_label: checkpoint.label,
    upscaler_id: upscaler.id,
    upscaler_sha256: upscaler.sha256,
    upscaler_label: upscaler.label,
  };
}

function contextSignature() {
  const context = autoResolutionContext();
  return JSON.stringify({
    model_family: context.model_family,
    checkpoint_sha256: context.checkpoint_sha256,
    width: context.current_dimensions?.width,
    height: context.current_dimensions?.height,
    requested_scale: context.requested_scale,
    requested_target: context.requested_target,
    sampler_name: context.base_values?.sampler_name,
    scheduler_name: context.base_values?.scheduler_name,
    cfg_scale: context.base_values?.cfg_scale,
    cfg_rescale: context.base_values?.cfg_rescale,
    prompt_parser_name: context.base_values?.prompt_parser_name,
    prompt_shortcut_profile_name: context.base_values?.prompt_shortcut_profile_name,
  });
}

async function applyRuntimeValues(values = {}) {
  model.lifecycleTrackingSuspended = true;
  try {
    await model.apply?.(values);
  } finally {
    model.lifecycleTrackingSuspended = false;
  }
  const lifecycle = writeLifecycleState(values.hires_lifecycle_state || currentLifecycleState({ ...(model.collect?.() || {}), ...(values || {}) }));
  renderLifecycleStatus(lifecycle);
  return lifecycle;
}

function renderLifecycleStatus(stateValue = null) {
  const lifecycle = normalizeLifecycleState(stateValue || currentLifecycleState(), model.collect?.() || {});
  const status = $("#hiresLifecycleStatus");
  const notice = $("#hiresLifecycleNotice");
  const reapplyAuto = $("#hiresLifecycleReapplyAutoButton");
  const resetProfile = $("#hiresLifecycleResetProfileButton");
  if (status) {
    const modifiedCount = lifecycle.modified_fields?.length || 0;
    if (lifecycle.active_mode === "auto") {
      status.textContent = `Active hires mode: Auto${lifecycle.baseline_label ? ` · ${lifecycle.baseline_label}` : ""}.`;
    } else if (lifecycle.active_mode === "profile") {
      status.textContent = `Active hires mode: Profile · ${lifecycle.baseline_label || "Saved hires preset"}.`;
    } else if (lifecycle.baseline_mode === "auto") {
      status.textContent = `Active hires mode: Custom · Based on ${lifecycle.baseline_label || "Auto"}${modifiedCount ? ` · ${modifiedCount} setting${modifiedCount === 1 ? "" : "s"} changed` : ""}.`;
    } else if (lifecycle.baseline_mode === "profile") {
      status.textContent = `Active hires mode: Custom · Based on ${lifecycle.baseline_label || "Saved hires preset"}${modifiedCount ? ` · ${modifiedCount} setting${modifiedCount === 1 ? "" : "s"} changed` : ""}.`;
    } else {
      status.textContent = "Active hires settings are currently treated as custom.";
    }
    status.className = lifecycle.active_mode === "custom" ? "field-status ready" : "field-status subtle";
  }
  if (reapplyAuto) reapplyAuto.hidden = !(lifecycle.active_mode === "auto" || lifecycle.baseline_mode === "auto");
  if (resetProfile) {
    const showReset = lifecycle.active_mode === "custom" && lifecycle.baseline_mode === "profile" && Boolean(lifecycle.baseline_profile_id || lifecycle.source_profile_id);
    resetProfile.hidden = !showReset;
    resetProfile.textContent = lifecycle.baseline_label ? `Reset to ${lifecycle.baseline_label}` : "Reset to Profile";
  }
  if (notice) {
    const message = lifecycleCompatibilityNotice(lifecycle, activeContextSummary(), model.profiles);
    notice.hidden = !message;
    notice.textContent = message;
    notice.className = message ? "field-status warning" : "field-status subtle";
  }
}

function syncLifecycleFromCurrentValues() {
  const values = model.collect?.() || {};
  const lifecycle = writeLifecycleState(currentLifecycleState(values));
  renderLifecycleStatus(lifecycle);
  model.lastContextSignature = contextSignature();
  return lifecycle;
}

function handleManualLifecycleEdit() {
  if (model.lifecycleTrackingSuspended) return;
  const values = model.collect?.() || {};
  const current = currentLifecycleState(values);
  const next = transitionLifecycleState(current, values);
  writeLifecycleState(next);
  renderLifecycleStatus(next);
  model.saveSessionSoon?.();
}

function scheduleAutoContextRefresh() {
  if (model.lifecycleTrackingSuspended) return;
  const signature = contextSignature();
  if (signature === model.lastContextSignature) return;
  model.lastContextSignature = signature;
  const lifecycle = currentLifecycleState();
  if (lifecycle.active_mode === "auto") {
    void resolveAuto({ silent: true });
    return;
  }
  renderLifecycleStatus(lifecycle);
}

function renderAutoResolution() {
  const details = $("#hiresAutoResolutionInspector");
  const host = $("#hiresAutoResolutionSummary");
  if (!details || !host) return;
  const result = model.lastResolution;
  details.hidden = !result;
  host.replaceChildren();
  if (!result) return;
  const selected = result.selected_upscaler || {};
  const summary = element("div", "hires-auto-resolution-overview");
  const diagnostics = result.diagnostics || {};
  const refinement = diagnostics.refinement_policy || {};
  const correction = diagnostics.quality_correction_policy || {};
  summary.append(
    element("strong", "hires-auto-resolution-title", `Resolved ${result.validation?.valid ? "successfully" : "with errors"}`),
    element("div", "hires-auto-resolution-upscaler", `Upscaler: ${selected.label || selected.display_name || selected.upscaler_id || "Unresolved"}`),
    element("small", "field-status subtle hires-auto-resolution-fingerprint", `Fingerprint: ${String(result.resolution_fingerprint || "").slice(0, 12)}`),
  );
  if (refinement.effective_refinement_steps) {
    summary.append(
      element(
        "small",
        "field-status subtle hires-auto-resolution-refinement",
        `Refinement: ${refinement.effective_refinement_steps} active evaluations from an internal ${refinement.internal_schedule_steps || refinement.effective_refinement_steps}-step schedule (${refinement.step_policy || "policy"})`
      )
    );
  }
  if (correction.selected_filter) {
    summary.append(
      element(
        "small",
        "field-status subtle hires-auto-resolution-correction",
        `Correction filter: ${correction.selected_filter}${correction.reason ? ` · ${correction.reason}` : ""}`
      )
    );
  }
  host.appendChild(summary);
  Object.entries(result.field_sources || {}).forEach(([key, source]) => {
    const row = element("div", "hires-auto-resolution-setting");
    row.append(
      element("strong", "hires-auto-resolution-label", key.replace(/^hires_/, "").replaceAll("_", " ")),
      element("div", "hires-profile-readonly hires-auto-resolution-value", textForValue(result.values?.[key])),
      element("small", "field-status subtle hires-auto-resolution-source", `Source: ${source.profile_name || source.source || source.scope || "resolver"}${source.reason ? ` · ${source.reason}` : ""}`),
    );
    host.appendChild(row);
  });
  (result.warnings || []).forEach((warning) => host.appendChild(element("small", "field-status warning", warning)));
}

async function resolveAuto({ silent = false } = {}) {
  const button = $("#hiresAutoResolveButton");
  if (button) button.disabled = true;
  try {
    const context = autoResolutionContext();
    const result = await api.resolveHiresAuto(context);
    model.lastResolution = clone(result);
    const autoValues = { ...(result.values || {}), hires_configuration_mode: "auto", hires_auto_resolution_record: clone(result) };
    autoValues.hires_lifecycle_state = lifecycleStateFromAuto({ values: autoValues, resolutionRecord: result, context });
    await applyRuntimeValues(autoValues);
    model.saveSessionSoon?.();
    const upscaler = result.selected_upscaler || {};
    $("#hiresAutoStatus").textContent = `Auto resolved ${upscaler.label || upscaler.display_name || upscaler.upscaler_id || "hires settings"}.`;
    $("#hiresAutoStatus").className = "field-status ready";
    renderAutoResolution();
    if (!silent) notify("Auto hires settings resolved for the current model and target.");
  } catch (error) {
    const detail = error?.detail || error?.message || String(error);
    $("#hiresAutoStatus").textContent = `Auto resolution failed: ${typeof detail === "string" ? detail : "See resolution details."}`;
    $("#hiresAutoStatus").className = "field-status error";
    if (!silent) notify(`Unable to resolve Auto hires: ${error.message}`, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

function renderProfileSelector() {
  const select = $("#hiresProfileSelect");
  if (!select) return;
  const current = model.selectedProfileId;
  select.replaceChildren(optionNode("", "Current settings"));
  model.profiles.forEach((profile) => select.appendChild(optionNode(profile.profile_id, profile.name)));
  select.value = model.profiles.some((item) => item.profile_id === current) ? current : "";
  const quick = $("#hiresProfileApplyQuickButton");
  if (quick) quick.disabled = !select.value;
}

function updateDraftModified(descriptor) {
  descriptor.modified = !valuesEqual(descriptor.current_value, descriptor.baseline_value);
}

function renderSettingEditor(descriptor, readOnly) {
  const editor = editorModelForDescriptor(descriptor);
  const disabled = readOnly || descriptor.editable === false;
  if (editor.kind === "read_only") {
    return element("div", "hires-profile-readonly", textForValue(descriptor.current_value));
  }
  if (editor.kind === "boolean") {
    const label = element("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(descriptor.current_value);
    input.disabled = disabled;
    input.addEventListener("change", () => {
      descriptor.current_value = input.checked;
      updateDraftModified(descriptor);
      renderSettings();
    });
    label.append(input, document.createTextNode(input.checked ? " On" : " Off"));
    return label;
  }
  if (editor.kind === "select") {
    const select = document.createElement("select");
    select.disabled = disabled;
    const values = new Set();
    (editor.options || []).forEach((item) => {
      values.add(String(item.value ?? ""));
      select.appendChild(optionNode(item.value, item.label, item.available === false));
    });
    const current = String(descriptor.current_value ?? "");
    if (!values.has(current)) select.appendChild(optionNode(current, `${textForValue(descriptor.current_value)} (saved value)`));
    select.value = current;
    select.addEventListener("change", () => {
      descriptor.current_value = select.value;
      updateDraftModified(descriptor);
      renderSettings();
    });
    return select;
  }
  if (editor.kind === "multi_select") {
    const wrap = element("div", "hires-profile-multi-select");
    const selected = new Set(Array.isArray(descriptor.current_value) ? descriptor.current_value.map(String) : []);
    (editor.options || []).forEach((item) => {
      const label = element("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = String(item.value ?? "");
      input.checked = selected.has(input.value);
      input.disabled = disabled || item.available === false;
      input.addEventListener("change", () => {
        const values = [...wrap.querySelectorAll('input[type="checkbox"]:checked')].map((node) => node.value);
        descriptor.current_value = values;
        updateDraftModified(descriptor);
        renderSettings();
      });
      label.append(input, document.createTextNode(` ${item.label ?? item.value}`));
      wrap.appendChild(label);
    });
    return wrap;
  }
  if (editor.kind === "nullable_number") {
    const wrap = element("div", "hires-profile-nullable-number");
    const mode = document.createElement("select");
    mode.append(optionNode("inherit", "Inherit / automatic"), optionNode("custom", "Custom value"));
    const input = document.createElement("input");
    input.type = "number";
    if (editor.minimum != null) input.min = String(editor.minimum);
    if (editor.maximum != null) input.max = String(editor.maximum);
    if (editor.step != null) input.step = String(editor.step);
    input.value = String(editor.minimum ?? 0);
    mode.value = descriptor.current_value == null ? "inherit" : "custom";
    if (descriptor.current_value != null) input.value = String(descriptor.current_value);
    input.disabled = disabled || mode.value === "inherit";
    mode.disabled = disabled;
    mode.addEventListener("change", () => {
      input.disabled = disabled || mode.value === "inherit";
      descriptor.current_value = mode.value === "inherit" ? null : Number(input.value);
      updateDraftModified(descriptor);
      renderSettings();
    });
    input.addEventListener("change", () => {
      descriptor.current_value = Number(input.value);
      updateDraftModified(descriptor);
      renderSettings();
    });
    wrap.append(mode, input);
    return wrap;
  }
  const input = document.createElement("input");
  input.type = "number";
  if (editor.minimum != null) input.min = String(editor.minimum);
  if (editor.maximum != null) input.max = String(editor.maximum);
  if (editor.step != null) input.step = String(editor.step);
  input.value = String(descriptor.current_value ?? editor.minimum ?? 0);
  input.disabled = disabled;
  input.addEventListener("change", () => {
    descriptor.current_value = descriptor.value_type === "integer" ? Number.parseInt(input.value, 10) : Number(input.value);
    updateDraftModified(descriptor);
    renderSettings();
  });
  return input;
}

function renderSettings() {
  const host = $("#hiresProfileSettings");
  if (!host || !model.draft) return;
  host.replaceChildren();
  const readOnly = Boolean(model.draft.profile?.read_only);
  const descriptors = descriptorsForView(model.draft.manifest?.descriptors || [], model.view);
  if (!descriptors.length) {
    host.appendChild(element("p", "field-status subtle", "No settings differ from this preset's baseline."));
    return;
  }
  groupDescriptors(descriptors).forEach(({ name, items }) => {
    const group = element("section", "hires-profile-group");
    group.appendChild(element("h4", "", name));
    items.forEach((descriptor) => {
      const rejected = String(descriptor.persistence_eligibility || "") !== "eligible";
      const row = element("div", `hires-profile-setting${descriptor.modified ? " is-modified" : ""}${rejected ? " is-rejected" : ""}`);
      const include = document.createElement("input");
      include.type = "checkbox";
      include.checked = Boolean(descriptor.included);
      include.disabled = readOnly || rejected;
      include.title = "Include this setting in the saved preset";
      include.addEventListener("change", () => { descriptor.included = include.checked; });
      const copy = element("div", "hires-profile-setting-copy");
      copy.appendChild(element("div", "hires-profile-setting-label", descriptor.label || descriptor.key));
      if (descriptor.description) copy.appendChild(element("small", "", descriptor.description));
      copy.appendChild(renderSettingEditor(descriptor, readOnly || rejected));
      const baseline = element("small", "", `Baseline: ${textForValue(descriptor.baseline_value)} · ${descriptor.included ? "Included" : "Not included"}`);
      copy.appendChild(baseline);
      if (rejected) copy.appendChild(element("small", "", `Persistence audit: ${descriptor.persistence_eligibility}`));
      row.append(include, copy);
      group.appendChild(row);
    });
    host.appendChild(group);
  });
}

function checkedValues(hostSelector) {
  return [...document.querySelectorAll(`${hostSelector} input[type="checkbox"]:checked`)].map((node) => node.value);
}

function renderCompatibility() {
  if (!model.draft) return;
  const readOnly = Boolean(model.draft.profile?.read_only);
  const compatibility = model.draft.profile.compatibility || {};
  const familyHost = $("#hiresProfileCompatibilityFamilies");
  const upscalerHost = $("#hiresProfileCompatibilityUpscalers");
  familyHost?.replaceChildren();
  upscalerHost?.replaceChildren();
  if (familyHost) {
    familyHost.appendChild(element("strong", "", "Compatible model families"));
    const selected = new Set((compatibility.model_families || []).map(String));
    (model.compatibilityChoices.model_families || model.familyChoices || []).forEach((item) => {
      const label = element("label");
      const input = document.createElement("input");
      input.type = "checkbox"; input.value = String(item.value); input.checked = selected.has(input.value); input.disabled = readOnly;
      label.append(input, document.createTextNode(` ${item.label}`)); familyHost.appendChild(label);
    });
  }
  if (upscalerHost) {
    upscalerHost.appendChild(element("strong", "", "Compatible upscalers"));
    const selected = new Set((compatibility.upscaler_sha256 || []).map(String));
    (model.compatibilityChoices.upscalers || []).filter((item) => item.sha256).forEach((item) => {
      const label = element("label");
      const input = document.createElement("input");
      input.type = "checkbox"; input.value = String(item.sha256); input.checked = selected.has(input.value); input.disabled = readOnly || item.available === false;
      label.append(input, document.createTextNode(` ${item.label}`)); upscalerHost.appendChild(label);
    });
  }
}

function assignmentOwned(scope, extras = {}) {
  if (!model.draft?.profile?.profile_id) return false;
  return model.allAssignments.some((item) => {
    if (item.profile_id !== model.draft.profile.profile_id || item.scope !== scope) return false;
    return Object.entries(extras).every(([key, value]) => String(item[key] || "").toLowerCase() === String(value || "").toLowerCase());
  });
}

function assignmentCheckbox({ scope, label, disabled = false, extras = {}, detail = "" }) {
  const wrapper = element("label");
  const input = document.createElement("input");
  input.type = "checkbox"; input.dataset.assignmentScope = scope; input.disabled = disabled || Boolean(model.draft?.profile?.read_only);
  Object.entries(extras).forEach(([key, value]) => { input.dataset[key] = value; });
  input.checked = assignmentOwned(scope, extras);
  const content = element("span"); content.append(document.createTextNode(label));
  if (detail) content.appendChild(element("small", "", detail));
  wrapper.append(input, content); return wrapper;
}

function renderAssignments() {
  const host = $("#hiresProfileDefaultAssignments");
  if (!host || !model.draft) return;
  host.replaceChildren();
  const checkpoint = activeCheckpointContext();
  const upscaler = activeUpscalerContext();
  host.appendChild(assignmentCheckbox({ scope: "global", label: "Global hires default" }));
  host.appendChild(element("strong", "", "Model family defaults"));
  (model.familyChoices || []).forEach((item) => host.appendChild(assignmentCheckbox({
    scope: "model_family", label: item.label, extras: { model_family: item.value },
  })));
  host.appendChild(element("strong", "", "Current context defaults"));
  host.appendChild(assignmentCheckbox({
    scope: "checkpoint", label: "Current checkpoint", disabled: !checkpoint.sha256,
    extras: { checkpoint_sha256: checkpoint.sha256 }, detail: checkpoint.sha256 ? checkpoint.label : "No stable checkpoint SHA-256 is available.",
  }));
  host.appendChild(assignmentCheckbox({
    scope: "upscaler", label: "Current upscaler", disabled: !upscaler.sha256,
    extras: { upscaler_sha256: upscaler.sha256 }, detail: upscaler.sha256 ? upscaler.label : "Select a fingerprinted upscaler first.",
  }));
  host.appendChild(assignmentCheckbox({
    scope: "model_family_upscaler", label: "Current family + upscaler", disabled: !checkpoint.family || !upscaler.sha256,
    extras: { model_family: checkpoint.family, upscaler_sha256: upscaler.sha256 }, detail: checkpoint.family && upscaler.sha256 ? `${checkpoint.family} + ${upscaler.label}` : "Requires both a model family and upscaler.",
  }));
  host.appendChild(assignmentCheckbox({
    scope: "checkpoint_upscaler", label: "Current checkpoint + upscaler", disabled: !checkpoint.sha256 || !upscaler.sha256,
    extras: { checkpoint_sha256: checkpoint.sha256, upscaler_sha256: upscaler.sha256 }, detail: checkpoint.sha256 && upscaler.sha256 ? `${checkpoint.label} + ${upscaler.label}` : "Requires both stable identities.",
  }));
}

function renderPanel() {
  if (!model.draft) return;
  const profile = model.draft.profile || {};
  $("#hiresProfilePanelTitle").textContent = profile.name || "Preset Configuration";
  $("#hiresProfileName").value = profile.name || "New Hires Preset";
  $("#hiresProfileName").disabled = Boolean(profile.read_only);
  $("#hiresProfileReadOnlyStatus").textContent = profile.read_only
    ? "Built-in IMAGE_GEN presets are read-only. Duplicate this preset to customize it."
    : "Editing this panel changes the saved preset draft only. Active generation settings change only when Apply Preset is used.";
  $("#hiresProfileSaveButton").disabled = Boolean(profile.read_only);
  $("#hiresProfileDuplicateButton").hidden = !profile.read_only;
  $("#hiresProfileDeleteButton").hidden = Boolean(profile.read_only) || !profile.profile_id;
  $("#hiresProfileApplyButton").disabled = !profile.profile_id && !model.draft;
  renderSettings(); renderCompatibility(); renderAssignments();
}

async function refreshCatalog() {
  const [catalog, schema] = await Promise.all([api.hiresProfiles(), api.hiresProfileSchema()]);
  model.profiles = catalog.profiles || [];
  model.allAssignments = catalog.assignments || [];
  model.familyChoices = catalog.model_family_choices || [];
  model.schema = schema;
  model.compatibilityChoices = schema.compatibility_choices || { model_families: model.familyChoices, upscalers: [] };
  renderProfileSelector();
  renderLifecycleStatus();
}

async function loadProfile(profileId) {
  if (!profileId) return null;
  const bundle = await api.hiresProfile(profileId);
  model.selectedProfileId = profileId;
  model.draft = clone(bundle);
  model.allAssignments = [...model.allAssignments.filter((item) => item.profile_id !== profileId), ...(bundle.assignments || [])];
  renderProfileSelector(); renderPanel();
  return bundle;
}

function openPanel() {
  const panel = $("#hiresProfilePanel");
  if (panel) panel.hidden = false;
}

function closePanel() { const panel = $("#hiresProfilePanel"); if (panel) panel.hidden = true; }

function ensureDraft() {
  if (model.draft) return;
  const currentValues = model.collect?.() || {};
  const lifecycle = currentLifecycleState(currentValues);
  const baselineOptions = lifecycle.active_mode === "custom" && Object.keys(lifecycle.baseline_values || {}).length
    ? { baseline_values: lifecycle.baseline_values, baseline_label: lifecycle.baseline_label }
    : {};
  model.draft = createDraftFromSchema(
    model.schema || {},
    currentValues,
    model.profiles.filter((item) => item.source === "user").length + 1,
    baselineOptions,
  );
  model.selectedProfileId = "";
}

function profilePayload() {
  const descriptors = model.draft?.manifest?.descriptors || [];
  const values = {};
  const includedFields = [];
  descriptors.forEach((descriptor) => {
    if (!descriptor.included || descriptor.persistence_eligibility !== "eligible") return;
    values[descriptor.key] = clone(descriptor.current_value);
    includedFields.push(descriptor.key);
  });
  return {
    profile_id: model.draft?.profile?.profile_id || "",
    name: String($("#hiresProfileName")?.value || model.draft?.profile?.name || "New Hires Preset").trim() || "New Hires Preset",
    description: model.draft?.profile?.description || "",
    baseline_profile_id: model.draft?.profile?.baseline_profile_id || "",
    values,
    included_fields: includedFields,
    compatibility: {
      model_families: checkedValues("#hiresProfileCompatibilityFamilies"),
      upscaler_sha256: checkedValues("#hiresProfileCompatibilityUpscalers"),
    },
  };
}

function desiredAssignments(profileId) {
  return [...document.querySelectorAll('#hiresProfileDefaultAssignments input[data-assignment-scope]:checked')].map((input) => {
    const payload = { scope: input.dataset.assignmentScope, profile_id: profileId };
    for (const key of ["model_family", "checkpoint_sha256", "upscaler_sha256"]) {
      if (input.dataset[key]) payload[key] = input.dataset[key];
    }
    return payload;
  });
}

async function syncAssignments(profileId) {
  const desired = desiredAssignments(profileId);
  const existing = model.allAssignments.filter((item) => item.profile_id === profileId);
  const desiredKeys = new Set(desired.map((item) => assignmentKey(item)));
  for (const item of existing) {
    if (!desiredKeys.has(item.assignment_key || assignmentKey(item))) await api.deleteHiresDefaultAssignment(item.assignment_key || assignmentKey(item));
  }
  for (const item of desired) await api.saveHiresDefaultAssignment(item);
}

function assignmentKey(item = {}) {
  const parts = [item.scope];
  if (["model_family", "model_family_upscaler"].includes(item.scope)) parts.push(item.model_family || "");
  if (["checkpoint", "checkpoint_upscaler"].includes(item.scope)) parts.push(item.checkpoint_sha256 || "");
  if (["upscaler", "model_family_upscaler", "checkpoint_upscaler"].includes(item.scope)) parts.push(item.upscaler_sha256 || "");
  return parts.join(":").toLowerCase();
}

async function saveDraft() {
  if (!model.draft || model.draft.profile?.read_only) return;
  const saved = await api.saveHiresProfile(profilePayload());
  const profileId = saved.profile.profile_id;
  await syncAssignments(profileId);
  await refreshCatalog();
  await loadProfile(profileId);
  notify(`Hires preset "${saved.profile.name}" saved.`);
}

async function applyDraft() {
  if (!model.draft) return;
  const values = {};
  (model.draft.manifest?.descriptors || []).forEach((descriptor) => {
    if (descriptor.included && descriptor.persistence_eligibility === "eligible") values[descriptor.key] = clone(descriptor.current_value);
  });
  const context = activeContextSummary();
  const appliedValues = {
    ...values,
    hires_configuration_mode: "profile",
    hires_auto_resolution_record: {},
  };
  appliedValues.hires_lifecycle_state = lifecycleStateFromProfile({ values: appliedValues, profile: model.draft.profile || {}, context });
  await applyRuntimeValues(appliedValues);
  model.saveSessionSoon?.();
  notify(`Applied hires preset ${model.draft.profile?.name || "settings"}.`);
}

export async function bindHiresProfiles({ collect, apply, saveSessionSoon } = {}) {
  model.collect = collect;
  model.apply = apply;
  model.saveSessionSoon = saveSessionSoon;
  await refreshCatalog();
  syncLifecycleFromCurrentValues();

  $("#hiresAutoResolveButton")?.addEventListener("click", () => void resolveAuto());
  $("#hiresLifecycleReapplyAutoButton")?.addEventListener("click", () => void resolveAuto());
  $("#hiresLifecycleResetProfileButton")?.addEventListener("click", async () => {
    const lifecycle = currentLifecycleState();
    const profileId = lifecycle.baseline_profile_id || lifecycle.source_profile_id;
    if (!profileId) return;
    try {
      await loadProfile(profileId);
      await applyDraft();
    } catch (error) {
      notify(`Unable to restore the baseline hires preset: ${error.message}`, "error");
    }
  });

  $("#hiresProfileSelect")?.addEventListener("change", async (event) => {
    model.selectedProfileId = event.target.value;
    model.draft = null;
    $("#hiresProfileApplyQuickButton").disabled = !event.target.value;
    $("#hiresProfileStatus").textContent = event.target.value
      ? "Preset selected. Active hires settings are unchanged until Apply is used."
      : "Current settings are active; no saved hires preset is selected.";
  });
  $("#hiresProfileConfigureButton")?.addEventListener("click", async () => {
    try {
      const selected = $("#hiresProfileSelect")?.value || "";
      if (selected) await loadProfile(selected); else { ensureDraft(); renderPanel(); }
      openPanel();
    } catch (error) { notify(`Unable to open hires presets: ${error.message}`, "error"); }
  });
  $("#hiresProfileApplyQuickButton")?.addEventListener("click", async () => {
    try { const selected = $("#hiresProfileSelect")?.value || ""; if (selected) await loadProfile(selected); await applyDraft(); }
    catch (error) { notify(`Unable to apply hires preset: ${error.message}`, "error"); }
  });
  $("#hiresProfileCloseButton")?.addEventListener("click", closePanel);
  document.querySelectorAll('input[name="hiresProfileViewMode"]').forEach((input) => input.addEventListener("change", () => {
    model.view = input.value; renderSettings();
  }));
  $("#hiresProfileSaveButton")?.addEventListener("click", () => void saveDraft().catch((error) => notify(`Unable to save hires preset: ${error.message}`, "error")));
  $("#hiresProfileApplyButton")?.addEventListener("click", () => void applyDraft().catch((error) => notify(`Unable to apply hires preset: ${error.message}`, "error")));
  $("#hiresProfileDuplicateButton")?.addEventListener("click", async () => {
    try {
      const source = model.draft?.profile;
      if (!source?.profile_id) return;
      const result = await api.duplicateHiresProfile(source.profile_id, `${source.name} Copy`);
      await refreshCatalog(); await loadProfile(result.profile.profile_id); renderPanel();
      notify(`Created editable copy "${result.profile.name}".`);
    } catch (error) { notify(`Unable to duplicate hires preset: ${error.message}`, "error"); }
  });
  $("#hiresProfileDeleteButton")?.addEventListener("click", async () => {
    const profile = model.draft?.profile;
    if (!profile?.profile_id || profile.read_only) return;
    if (!window.confirm(`Delete hires preset "${profile.name}"?`)) return;
    try {
      await api.deleteHiresProfile(profile.profile_id);
      model.selectedProfileId = ""; model.draft = null; await refreshCatalog(); closePanel();
      notify(`Deleted hires preset "${profile.name}".`);
    } catch (error) { notify(`Unable to delete hires preset: ${error.message}`, "error"); }
  });
  $("#hiresProfileName")?.addEventListener("input", (event) => {
    if (!model.draft?.profile?.read_only) {
      model.draft.profile.name = event.target.value || "New Hires Preset";
      $("#hiresProfilePanelTitle").textContent = model.draft.profile.name;
    }
  });
  window.addEventListener("image-gen-hires-upscaler-change", () => { if (!$("#hiresProfilePanel")?.hidden) renderAssignments(); });
  $("#promptHiresRoutingSection")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (["hiresAutoResolveButton", "hiresProfileSelect", "hiresProfileApplyQuickButton", "hiresProfileConfigureButton", "hiresLifecycleReapplyAutoButton", "hiresLifecycleResetProfileButton"].includes(target.id)) return;
    if (!target.id?.startsWith("hires") && !target.getAttribute("name")?.startsWith("hires_")) return;
    handleManualLifecycleEdit();
  });
  $("#promptHiresRoutingSection")?.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!["number", "text", "range"].includes(target.type)) return;
    if (!target.id?.startsWith("hires") && !target.getAttribute("name")?.startsWith("hires_")) return;
    handleManualLifecycleEdit();
  });
  HIRES_CONTEXT_SELECTORS.forEach((selector) => $(selector)?.addEventListener("change", scheduleAutoContextRefresh));
  window.addEventListener("image-gen-generation-values-applied", () => {
    if (!model.lifecycleTrackingSuspended) syncLifecycleFromCurrentValues();
  });
  window.addEventListener("image-gen-model-loaded", scheduleAutoContextRefresh);
  window.addEventListener("image-gen-model-unloaded", scheduleAutoContextRefresh);
}
