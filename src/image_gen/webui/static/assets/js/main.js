import { loadFragments } from "./fragments.js";

import { api } from "./api.js?v=0.1.82";
import { state, setCatalogs, samplerDescriptor, schedulerDescriptor } from "./state.js";
import { $, $$, debounce, option, replaceOptions, notify } from "./utils.js";
import { renderAdvancedEditor } from "./components/advanced-editor.js";
import { collectGenerationValues, applyGenerationValues } from "./components/form-state.js?v=0.1.83";
import { acceptQueuedJob, bindGeneration } from "./features/generation.js?v=0.1.82";
import { bindGallery, initializeRecentOutputBrowser, recentOutputApiFilters, renderGallery } from "./features/gallery.js?v=0.1.45";
import { bindPromptPresets, renderPromptPresets } from "./features/presets.js";
import { bindGenerationProfiles, renderGenerationProfiles } from "./features/profiles.js";
import { bindSettings } from "./features/settings.js?v=0.1.62";
import { bindRuntimeCommandCopy, renderRuntimeStartupStatus } from "./features/memory-status.js?v=0.1.62";
import { bindWorkspaceLayout } from "./features/layout.js?v=0.1.74";
import { bindDefaultAssets } from "./features/default-assets.js?v=0.1.77";
import { bindCheckpointWorkspace } from "./features/checkpoints.js?v=0.1.74";
import { bindLoraWorkspace } from "./features/loras.js?v=0.1.77";
import { bindWorkspaceTabs } from "./features/workspace-tabs.js?v=0.1.74";
import { bindLightbox } from "./features/lightbox.js?v=0.1.40";
import { enforceExactDimensionInputs } from "./features/exact-dimensions.js";
import { bindOutputDetails } from "./features/output-details.js?v=0.1.84";
import { bindQueueComposer } from "./features/queue-composer.js";
import { bindBatchIO } from "./features/batch-io.js";
import { bindVariationMatrix, openVariationMatrix } from "./features/variation-matrix.js";
import { bindCfgLab } from "./features/cfg-lab.js?v=0.1.45";
import { bindOutputPatternBuilder } from "./features/output-pattern-builder.js";
import { bindPromptTools, initializePromptTools, refreshPromptConfigurationCatalogs } from "./features/prompt-tools.js?v=0.1.80";
import { bindPromptLoraSync } from "./features/prompt-lora-sync.js?v=0.1.77";
import { bindHiresUpscalers, initializeHiresUpscalers } from "./features/hires-upscalers.js?v=0.1.79";
import { bindOutpaintPrototype } from "./features/outpaint-prototype.js?v=0.1.84";

const PROMPT_ASSET_CONTRACT_VERSION = "image-gen-prompt-assets-v1";

function promptAssetSource(value, fallback = "visual_selection") {
  const token = String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  const aliases = { visual: "visual_selection", inline: "inline_syntax", model: "model_default", global: "global_default" };
  return aliases[token] || token || fallback;
}

window.__IMAGE_GEN_BOOT_MODULE_LOADED__ = true;

const SCHEDULER_PRESET_KIND = "scheduler";
const BUILTIN_SCHEDULER_PRESETS = {
  simple_kes: [
    { name: "Balanced Default", source: "builtin", values: {} },
    { name: "Wide Sigma Exploration", source: "builtin", values: { sigma_max: 60, rho: 8, tail_steps: 2 } },
    { name: "Gentle Tail Blend", source: "builtin", values: { sigma_min: 0.25, sigma_max: 42, rho: 6.5, tail_steps: 2, decay_mode: "blend", decay_pattern: "harmonic" } },
  ],
};

let samplerValues = {};
let schedulerValues = {};
let schedulerUserSelected = false;
let modelActivationPromise = Promise.resolve(null);
let modelRuntimeReadyPath = "";
let schedulerPresetPluginId = "";
let schedulerPresetName = "";
let schedulerPresetSource = "";
let schedulerPresetOptions = [];
let refreshOutputsPromise = null;

function pluginValue(descriptors, requested, fallback = "") {
  const resolve = (value) => {
    const needle = String(value || "").trim().toLowerCase();
    if (!needle) return null;
    return descriptors.find((item) => [
      item.plugin_id,
      item.name,
      item.label,
      ...(item.aliases || []),
    ].some((candidate) => String(candidate || "").trim().toLowerCase() === needle)) || null;
  };
  return resolve(requested)?.name || resolve(fallback)?.name || "";
}

function preferredSchedulerForSampler(samplerName) {
  const descriptor = samplerDescriptor(samplerName);
  const preferred = descriptor?.capabilities?.preferred_scheduler;
  return pluginValue(state.schedulers, preferred, samplerName === "kes" ? "simple_kes" : "standard_karras");
}

function currentSchedulerDescriptor() {
  return schedulerDescriptor($("#schedulerName").value);
}

function currentSchedulerPluginId() {
  const descriptor = currentSchedulerDescriptor();
  return String(descriptor?.plugin_id || descriptor?.name || $("#schedulerName").value || "").trim();
}

function schedulerPresetSupportEnabled() {
  return currentSchedulerPluginId() === "simple_kes";
}

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") {
    return Object.keys(value).sort().reduce((acc, key) => {
      acc[key] = sortKeys(value[key]);
      return acc;
    }, {});
  }
  return value;
}

function stableJson(value) {
  return JSON.stringify(sortKeys(value || {}));
}

function valuesEqual(left, right) {
  return stableJson(left || {}) === stableJson(right || {});
}

function replayPromptAssets(values = {}) {
  if (Array.isArray(values._webui_active_prompt_assets) && values._webui_active_prompt_assets.length) {
    return values._webui_active_prompt_assets.map((item) => ({ ...item }));
  }
  return [
    ...(Array.isArray(values.loras) ? values.loras.map((item) => ({
      ...item,
      asset_type: "lora",
      source: promptAssetSource(item.source || "replay", "replay"),
      source_scope: item.source_scope || "replay",
    })) : []),
    ...(Array.isArray(values.textual_inversions) ? values.textual_inversions.map((item) => ({
      ...item,
      asset_type: "textual_inversion",
      source: promptAssetSource(item.source || "replay", "replay"),
      source_scope: item.source_scope || "replay",
    })) : []),
  ];
}

function collectCurrentValues() {
  const activeAssets = Array.isArray(state.activePromptAssets) ? state.activePromptAssets : [];
  const activeLoras = activeAssets.filter((item) => item.asset_type === "lora" && item.enabled !== false).map((item) => ({
    asset_id: item.asset_id || item.catalog_asset_id || "",
    catalog_asset_id: item.catalog_asset_id || item.asset_id || "",
    name: item.name || "",
    path: item.path || "",
    weight: Number(item.weight ?? 1),
    enabled: item.enabled !== false,
    polarity: item.polarity || "positive",
    activation_text: item.activation_text || "",
    model_family: item.model_family || "",
    source_url: item.source_url || "",
    source: promptAssetSource(item.source || item.source_scope || "visual_selection"),
    original_source: item.original_source || "",
    source_scope: item.source_scope || "visual",
  }));
  const activeTextualInversions = activeAssets.filter((item) => item.asset_type === "textual_inversion" && item.enabled !== false).map((item) => ({
    asset_id: item.asset_id || item.catalog_asset_id || "",
    catalog_asset_id: item.catalog_asset_id || item.asset_id || "",
    name: item.name || "",
    path: item.path || "",
    polarity: item.polarity || "positive",
    activation_text: item.activation_text || "",
    model_family: item.model_family || "",
    source: promptAssetSource(item.source || item.source_scope || "visual_selection"),
    original_source: item.original_source || "",
    source_scope: item.source_scope || "visual",
    enabled: item.enabled !== false,
  }));
  return collectGenerationValues({
    _webui_selection_version: 2,
    _webui_scheduler_user_selected: schedulerUserSelected,
    _webui_scheduler_browser_name: $("#schedulerName").value,
    _webui_scheduler_requires_warning_acknowledgement: true,
    _webui_model_selection_id: state.activeModel?.selection_id || "",
    _webui_model_requested_path: $("#modelPath").value,
    _webui_scheduler_preset_name: schedulerPresetName,
    _webui_scheduler_preset_plugin_id: schedulerPresetPluginId,
    _webui_scheduler_preset_source: schedulerPresetSource,
    _webui_active_prompt_assets: activeAssets,
    prompt_asset_contract_version: PROMPT_ASSET_CONTRACT_VERSION,
    loras: activeLoras,
    lora_paths: activeLoras.map((item) => item.path).filter(Boolean),
    textual_inversions: activeTextualInversions,
  });
}


function normalizedModelPath(value) {
  return String(value || "").trim().replaceAll("/", "\\").toLowerCase();
}

function modelFileName(value) {
  return String(value || "").trim().split(/[\\/]/).pop()?.toLowerCase() || "";
}

function resolveCatalogModelPath(value) {
  const requested = String(value || "").trim();
  if (!requested) return "";
  const exact = state.models.find((item) => normalizedModelPath(item.path) === normalizedModelPath(requested));
  if (exact) return exact.path;
  const fileName = modelFileName(requested);
  if (!fileName) return "";
  const byFilename = state.models.find((item) => modelFileName(item.path) === fileName);
  return byFilename?.path || "";
}

function syncModelDropdownSelection(requestedPath, { label = "" } = {}) {
  const select = $("#modelPath");
  if (!select) return "";

  const rawRequested = String(requestedPath || "").trim();
  if (!rawRequested) return "";

  const resolved = resolveCatalogModelPath(rawRequested) || rawRequested;
  const optionLabel = label || (resolved === rawRequested
    ? undefined
    : `${resolved} (resolved from ${rawRequested})`);
  ensureSelectValue(select, resolved, optionLabel);
  select.value = resolved;

  if (resolved !== rawRequested) {
    const staleOption = [...select.options].find((item) => item.value === rawRequested);
    const requestedIsCatalogEntry = state.models.some(
      (item) => normalizedModelPath(item.path) === normalizedModelPath(rawRequested),
    );
    if (staleOption && !requestedIsCatalogEntry) staleOption.remove();
  }
  return resolved;
}

function modelRuntimeIsReady(status, requestedPath) {
  const currentPath = normalizedModelPath(status?.current_model_path);
  const requested = normalizedModelPath(requestedPath);
  if (!currentPath || !requested || currentPath !== requested) return false;
  if (status?.cuda_available !== false) return status?.gpu_loaded === true;
  return status?.cpu_loaded === true;
}

function rememberModelRuntime(status, requestedPath = "") {
  modelRuntimeReadyPath = modelRuntimeIsReady(status, requestedPath || status?.current_model_path)
    ? String(status.current_model_path || requestedPath)
    : "";
  return Boolean(modelRuntimeReadyPath);
}

function setModelReadyState(ready, message, kind = "") {
  const status = $("#modelLoadStatus");
  status.textContent = message;
  status.className = `field-status ${kind}`.trim();
  ["#topGenerateButton", "#topInfinityButton", "#generateButton", "#generateMenuButton", "#infinityButton"].forEach((selector) => {
    const button = $(selector);
    if (button) button.disabled = !ready;
  });
}

function renderModelArchitectureStatus(activeModel = null) {
  const status = $("#modelArchitectureStatus");
  if (!status) return;
  const summary = String(activeModel?.architecture_summary || activeModel?.architecture_contract?.summary || "").trim();
  if (!activeModel) {
    status.textContent = "Architecture: waiting for activation.";
    status.className = "field-status subtle";
    return;
  }
  if (summary) {
    status.textContent = `Architecture: ${summary}`;
    status.className = "field-status ready subtle";
    return;
  }
  status.textContent = "Architecture: unknown for the current checkpoint.";
  status.className = "field-status subtle";
}

async function activateSelectedModel({ quiet = false } = {}) {
  const requestedPath = $("#modelPath").value;
  if (!requestedPath) {
    state.activeModel = null;
    setModelReadyState(false, "Choose a checkpoint model.", "error");
    renderModelArchitectureStatus(null);
    throw new Error("Choose a checkpoint model first.");
  }

  setModelReadyState(false, "Activating selected checkpoint…", "loading");
  const activation = api.activateModel(requestedPath).then((payload) => {
    const active = payload.active_model;
    if (!active?.resolved_path) {
      throw new Error("The backend did not return an activated checkpoint.");
    }
    const effectivePath = syncModelDropdownSelection(active.resolved_path) || active.resolved_path;
    if (!rememberModelRuntime(payload.model_runtime, effectivePath)) {
      throw new Error("The checkpoint selection completed, but the model is not resident on the execution device.");
    }
    state.activeModel = active;
    if (state.bootstrap) state.bootstrap.model_runtime = payload.model_runtime || state.bootstrap.model_runtime;
    renderModelArchitectureStatus(active);
    window.dispatchEvent(new CustomEvent("image-gen-model-activated", {
      detail: { activeModel: active, defaultAssets: payload.default_assets || null },
    }));
    const summary = active.architecture_summary ? ` · ${active.architecture_summary}` : "";
    setModelReadyState(
      true,
      `Selected: ${active.model_name}${summary} · canonical loader will use this checkpoint · ${active.selection_id}`,
      "ready",
    );
    api.runtimeStartupStatus()
      .then((status) => renderRuntimeStartupStatus(status))
      .catch((error) => console.warn("Unable to refresh runtime status after model activation", error));
    if (!quiet) {
      const resolutionNote = normalizedModelPath(effectivePath) !== normalizedModelPath(requestedPath)
        ? " · resolved to the current checkpoint library path"
        : "";
      notify(`Activated checkpoint: ${active.model_name}${summary ? ` (${active.architecture_summary})` : ""}${resolutionNote}`);
    }
    return active;
  }).catch((error) => {
    state.activeModel = null;
    modelRuntimeReadyPath = "";
    renderModelArchitectureStatus(null);
    setModelReadyState(false, `Model activation failed: ${error.message}`, "error");
    throw error;
  });
  modelActivationPromise = activation;
  return activation;
}

async function ensureSelectedModelReady() {
  const requestedPath = $("#modelPath").value;
  if (
    state.activeModel?.resolved_path === requestedPath
    && normalizedModelPath(modelRuntimeReadyPath) === normalizedModelPath(requestedPath)
  ) {
    return state.activeModel;
  }
  try {
    const runtimeStatus = await api.modelRuntimeStatus();
    if (state.activeModel?.resolved_path === requestedPath && rememberModelRuntime(runtimeStatus, requestedPath)) {
      setModelReadyState(true, `Selected: ${state.activeModel.model_name} · model resident and ready.`, "ready");
      return state.activeModel;
    }
  } catch (error) {
    console.warn("Unable to verify resident model status before generation", error);
  }
  return activateSelectedModel({ quiet: true });
}

async function applyStartupModelBehavior(bootstrap, current = {}) {
  const mode = String(state.settings.checkpoint_startup_mode || "last_used").trim().toLowerCase();
  const preload = state.settings.checkpoint_preload_on_startup !== false;
  const lastUsed = resolveCatalogModelPath(current.model_path || "") || current.model_path || "";
  const pinned = resolveCatalogModelPath(state.settings.checkpoint_startup_path || "") || state.settings.checkpoint_startup_path || "";
  const configuredDefaultSource = bootstrap.defaults?.model_path || bootstrap.effective_generation?.model_path || "";
  const configuredDefault = resolveCatalogModelPath(configuredDefaultSource) || configuredDefaultSource;
  const active = bootstrap.active_model || null;

  let selectedPath = "";
  if (mode === "pinned_default") selectedPath = pinned || configuredDefault;
  else if (mode === "last_used") selectedPath = lastUsed || pinned || configuredDefault;

  if (selectedPath) {
    syncModelDropdownSelection(selectedPath);
  }

  if (mode === "none" || !selectedPath) {
    state.activeModel = active || null;
    modelRuntimeReadyPath = "";
    renderModelArchitectureStatus(state.activeModel);
    setModelReadyState(false, "Startup behavior is set to start with no model. Choose a checkpoint before generating.", "subtle");
    return;
  }

  if (active?.resolved_path && normalizedModelPath(active.resolved_path) === normalizedModelPath(selectedPath)
      && rememberModelRuntime(bootstrap.model_runtime, selectedPath)) {
    state.activeModel = active;
    renderModelArchitectureStatus(active);
    setModelReadyState(true, `Selected: ${active.model_name} · model resident and ready.`, "ready");
    return;
  }

  state.activeModel = active?.resolved_path === selectedPath ? active : null;
  modelRuntimeReadyPath = "";
  renderModelArchitectureStatus(state.activeModel);
  if (preload) {
    await activateSelectedModel({ quiet: true });
    return;
  }
  setModelReadyState(true, "Selected model will load on demand when generation starts.", "subtle");
}

function applyVaeSelectionPolicy() {
  const select = $("#vaePath");
  const status = $("#vaeSelectionStatus");
  if (!select || !status) return;
  select.disabled = false;
  if (!select.value) {
    status.textContent = "Automatic / checkpoint embedded.";
  } else {
    const label = select.selectedOptions?.[0]?.textContent || select.value;
    status.textContent = `Manual external VAE selected: ${label}`;
  }
  status.className = "field-status subtle";
}

function populateModels(current = {}) {
  const modelOptions = state.models.map((item) => option(
    item.path,
    `${item.name} · ${item.size_mb} MB`,
  ));
  const requestedModelRaw = current.model_path || "";
  const requestedModel = resolveCatalogModelPath(requestedModelRaw) || requestedModelRaw;
  if (requestedModel && !state.models.some((item) => normalizedModelPath(item.path) === normalizedModelPath(requestedModel))) {
    const configuredLabel = requestedModelRaw && normalizedModelPath(requestedModelRaw) !== normalizedModelPath(requestedModel)
      ? `${requestedModel} (resolved from ${requestedModelRaw})`
      : `${requestedModel} (configured)`;
    modelOptions.unshift(option(requestedModel, configuredLabel));
  }
  replaceOptions($("#modelPath"), modelOptions, requestedModel);

  const vaeOptions = [option("", "Automatic / checkpoint embedded")];
  state.vaes.forEach((item) => vaeOptions.push(option(item.path, `${item.name} · ${item.size_mb} MB`)));
  replaceOptions($("#vaePath"), vaeOptions, current.vae_path || "");
  applyVaeSelectionPolicy();
  $("#modelCount").textContent = state.models.length;
  $("#vaeCount").textContent = state.vaes.length;
}

function populatePlugins(current = {}) {
  const samplerName = pluginValue(state.samplers, current.sampler_name, "kes");
  const schedulerName = pluginValue(
    state.schedulers,
    current.scheduler_name,
    preferredSchedulerForSampler(samplerName),
  );
  replaceOptions(
    $("#samplerName"),
    state.samplers.map((item) => option(item.name, item.label)),
    samplerName,
  );
  replaceOptions(
    $("#schedulerName"),
    state.schedulers.map((item) => option(item.name, item.label)),
    schedulerName,
  );
  const requestedHiresSampler = String(current.hires_sampler_name || "");
  const requestedHiresScheduler = String(current.hires_scheduler_name || "");
  replaceOptions(
    $("#hiresSamplerName"),
    [
      option("", `Inherit base sampler (${samplerName || "current"})`),
      ...state.samplers.map((item) => option(
        item.name,
        item.name === "kes" ? `${item.label} · experimental hires comparison` : item.label,
      )),
    ],
    requestedHiresSampler,
  );
  replaceOptions(
    $("#hiresSchedulerName"),
    [
      option("", `Inherit base scheduler (${schedulerName || "current"})`),
      ...state.schedulers.map((item) => option(
        item.name,
        item.name === "simple_kes" ? `${item.label} · experimental hires comparison` : item.label,
      )),
    ],
    requestedHiresScheduler,
  );
}

function mergedSchedulerPresets(pluginId) {
  const builtin = (BUILTIN_SCHEDULER_PRESETS[pluginId] || []).map((item) => ({ ...item }));
  const user = (schedulerPresetOptions || []).filter((item) => item.source === "user" && item.plugin_id === pluginId);
  return [...builtin, ...user];
}

function schedulerPresetSelection() {
  const pluginId = currentSchedulerPluginId();
  return mergedSchedulerPresets(pluginId).find((item) => item.name === schedulerPresetName) || null;
}

function updateSchedulerPresetStatus() {
  const toolbar = $("#schedulerPresetToolbar");
  const input = $("#schedulerPresetSelect");
  const status = $("#schedulerPresetStatus");
  const loadButton = $("#schedulerPresetLoadButton");
  const saveAsButton = $("#schedulerPresetSaveButton");
  const saveButton = $("#schedulerPresetUpdateButton");
  const deleteButton = $("#schedulerPresetDeleteButton");
  if (!toolbar || !input || !status) return;

  if (toolbar.hidden) {
    status.textContent = "Preset management is available for supported schedulers.";
    return;
  }

  const preset = schedulerPresetSelection();
  const currentValues = collectCurrentValues().scheduler_kwargs || {};
  const dirty = preset ? !valuesEqual(currentValues, preset.values) : Object.keys(currentValues || {}).length > 0;

  if (preset) {
    status.textContent = dirty
      ? `Preset “${preset.name}” is selected, but the current scheduler settings have unsaved manual changes.`
      : `${preset.source === "builtin" ? "Built-in" : "Saved"} preset “${preset.name}” is active.`;
  } else if (Object.keys(currentValues || {}).length) {
    status.textContent = "Manual scheduler configuration is active and has not been saved as a preset.";
  } else {
    status.textContent = "Default scheduler settings are active.";
  }

  loadButton.disabled = !preset;
  saveAsButton.disabled = false;
  saveButton.disabled = !(preset && preset.source === "user");
  deleteButton.disabled = !(preset && preset.source === "user");
}

async function refreshSchedulerPresetToolbar({ restoreSelection = true } = {}) {
  const toolbar = $("#schedulerPresetToolbar");
  const input = $("#schedulerPresetSelect");
  const list = $("#schedulerPresetList");
  const status = $("#schedulerPresetStatus");
  if (!toolbar || !input || !list || !status) return;

  const pluginId = currentSchedulerPluginId();
  if (!schedulerPresetSupportEnabled()) {
    toolbar.hidden = true;
    schedulerPresetPluginId = "";
    schedulerPresetName = "";
    schedulerPresetSource = "";
    schedulerPresetOptions = [];
    return;
  }

  toolbar.hidden = false;
  schedulerPresetPluginId = pluginId;
  try {
    const profiles = await api.profiles(SCHEDULER_PRESET_KIND, pluginId);
    schedulerPresetOptions = (profiles || []).map((item) => ({
      name: item.name,
      values: item.values || {},
      plugin_id: item.plugin_id || pluginId,
      source: "user",
    }));
  } catch (error) {
    schedulerPresetOptions = [];
    status.textContent = `Unable to load scheduler presets: ${error.message}`;
  }

  const presets = mergedSchedulerPresets(pluginId);
  list.replaceChildren(...presets.map((item) => option(item.name, item.source === "builtin" ? `${item.name} (built-in)` : item.name)));
  if (!restoreSelection) schedulerPresetName = "";
  input.value = restoreSelection ? schedulerPresetName : "";

  const selected = presets.find((item) => item.name === input.value) || null;
  schedulerPresetName = selected?.name || "";
  schedulerPresetSource = selected?.source || "";
  updateSchedulerPresetStatus();
}

async function loadSchedulerPreset(name) {
  const pluginId = currentSchedulerPluginId();
  const preset = mergedSchedulerPresets(pluginId).find((item) => item.name === name);
  if (!preset) {
    throw new Error(`Scheduler preset not found: ${name}`);
  }
  schedulerValues = { ...(preset.values || {}) };
  schedulerPresetName = preset.name;
  schedulerPresetPluginId = pluginId;
  schedulerPresetSource = preset.source || "user";
  await refreshAdvancedEditors({ preservePresetSelection: true });
  updateSchedulerPresetStatus();
  saveSessionSoon();
  notify(`Loaded scheduler preset: ${preset.name}`);
}

async function saveSchedulerPreset({ overwrite = false } = {}) {
  const pluginId = currentSchedulerPluginId();
  if (!pluginId) return;
  const selected = schedulerPresetSelection();
  let name = "";
  if (overwrite && selected?.source === "user") {
    name = selected.name;
  } else {
    name = $("#schedulerPresetSelect")?.value?.trim() || "";
  }
  if (!name) return;
  const values = collectCurrentValues().scheduler_kwargs || {};
  await api.saveProfile(SCHEDULER_PRESET_KIND, {
    name,
    plugin_id: pluginId,
    values,
    overwrite,
  });
  schedulerPresetName = name;
  schedulerPresetPluginId = pluginId;
  schedulerPresetSource = "user";
  await refreshSchedulerPresetToolbar({ restoreSelection: true });
  updateSchedulerPresetStatus();
  saveSessionSoon();
  notify(`Saved scheduler preset: ${name}`);
}

async function deleteSchedulerPreset() {
  const selected = schedulerPresetSelection();
  if (!selected || selected.source !== "user") return;
  if (!window.confirm(`Delete scheduler preset “${selected.name}”?`)) return;
  await api.deleteProfile(SCHEDULER_PRESET_KIND, selected.name, currentSchedulerPluginId());
  schedulerPresetName = "";
  schedulerPresetSource = "";
  await refreshSchedulerPresetToolbar({ restoreSelection: false });
  updateSchedulerPresetStatus();
  saveSessionSoon();
  notify(`Deleted scheduler preset: ${selected.name}`);
}

async function refreshAdvancedEditors({ preservePresetSelection = false } = {}) {
  await renderAdvancedEditor({
    container: $("#samplerAdvancedContent"),
    descriptor: samplerDescriptor($("#samplerName").value),
    kind: "sampler",
    currentValues: samplerValues,
    onChange: () => {
      saveSessionSoon();
    },
  });
  await renderAdvancedEditor({
    container: $("#schedulerAdvancedContent"),
    descriptor: schedulerDescriptor($("#schedulerName").value),
    kind: "scheduler",
    currentValues: schedulerValues,
    onChange: () => {
      updateSchedulerPresetStatus();
      saveSessionSoon();
    },
  });
  await refreshSchedulerPresetToolbar({ restoreSelection: preservePresetSelection });
  updateSchedulerPresetStatus();
}

function ensureSelectValue(select, value, label) {
  if (!select || value === undefined || value === null || value === "") return;
  const normalized = String(value);
  if (![...select.options].some((item) => item.value === normalized)) {
    select.prepend(option(normalized, label || `${normalized} (from output metadata)`));
  }
}

function missingAdvancedKeys(container, values = {}) {
  const available = new Set(
    [...container.querySelectorAll("[data-schema-path]")]
      .map((input) => String(input.dataset.schemaPath || "").split(".")[0])
      .filter(Boolean),
  );
  container.querySelectorAll("[data-schema-object-path]").forEach((item) => {
    const root = String(item.dataset.schemaObjectPath || "").split(".")[0];
    if (root) available.add(root);
  });
  return Object.keys(values || {}).filter((key) => !available.has(key));
}

async function applyReplayValues(values = {}) {
  const replayValues = { ...values };
  if (replayValues.model_path) {
    replayValues.model_path = resolveCatalogModelPath(replayValues.model_path) || replayValues.model_path;
  }

  if ("sampler_kwargs" in replayValues) samplerValues = replayValues.sampler_kwargs || {};
  if ("scheduler_kwargs" in replayValues) schedulerValues = replayValues.scheduler_kwargs || {};
  if (replayValues.scheduler_name) schedulerUserSelected = true;
  schedulerPresetName = replayValues._webui_scheduler_preset_name || "";
  schedulerPresetPluginId = replayValues._webui_scheduler_preset_plugin_id || "";
  schedulerPresetSource = replayValues._webui_scheduler_preset_source || "";

  ensureSelectValue($("#modelPath"), replayValues.model_path, `${replayValues.model_path} (from output metadata)`);
  ensureSelectValue($("#vaePath"), replayValues.vae_path, `${replayValues.vae_path} (from output metadata)`);
  ensureSelectValue($("#samplerName"), replayValues.sampler_name, `${replayValues.sampler_name} (unavailable plugin)`);
  ensureSelectValue($("#schedulerName"), replayValues.scheduler_name, `${replayValues.scheduler_name} (unavailable plugin)`);

  const restoredPromptAssets = replayPromptAssets(replayValues);
  state.activePromptAssets = restoredPromptAssets;
  window.dispatchEvent(new CustomEvent("image-gen-active-prompt-assets-updated", {
    detail: {
      active_assets: [...restoredPromptAssets],
      loras: restoredPromptAssets.filter((item) => item.asset_type === "lora"),
      textual_inversions: restoredPromptAssets.filter((item) => item.asset_type === "textual_inversion"),
    },
  }));

  applyGenerationValues(replayValues);
  window.imageGenPromptLoraSync?.syncFromPrompts?.();
  applyVaeSelectionPolicy();
  initializePromptTools(replayValues);
  await refreshAdvancedEditors({ preservePresetSelection: true });

  const unsupported = [
    ...missingAdvancedKeys($("#samplerAdvancedContent"), replayValues.sampler_kwargs),
    ...missingAdvancedKeys($("#schedulerAdvancedContent"), replayValues.scheduler_kwargs),
  ];

  if (replayValues.model_path) {
    try {
      await activateSelectedModel({ quiet: true });
    } catch (error) {
      notify(`Metadata was applied, but the recorded model could not be activated: ${error.message}`, "error");
    }
  }
  saveSessionSoon();
  return { unsupported };
}

async function refreshOutputs(options = {}) {
  if (refreshOutputsPromise) return refreshOutputsPromise;
  refreshOutputsPromise = (async () => {
    try {
      renderGallery(await api.recentOutputs(recentOutputApiFilters()), options);
    } catch (error) {
      notify(error.message, "error");
    } finally {
      refreshOutputsPromise = null;
    }
  })();
  return refreshOutputsPromise;
}

const saveSessionSoon = debounce(async () => {
  try {
    await api.saveSession(collectCurrentValues());
  } catch (error) {
    console.error("Unable to save session", error);
  }
}, 650);

function bindFormPersistence() {
  $("#generationForm").addEventListener("input", saveSessionSoon);
  $("#generationForm").addEventListener("change", saveSessionSoon);
  $("#randomSeedButton").addEventListener("click", () => {
    $("#seed").value = "-1";
    saveSessionSoon();
  });
}

function bindModelSelection() {
  $("#modelPath").addEventListener("change", async () => {
    try {
      await activateSelectedModel();
      saveSessionSoon();
    } catch (error) {
      notify(error.message, "error");
    }
  });
}

function bindAdvancedButtons() {
  $("#samplerSettingsButton").addEventListener("click", () => {
    $("#samplerAdvanced").open = true;
    $("#samplerAdvanced").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("#schedulerSettingsButton").addEventListener("click", () => {
    $("#schedulerAdvanced").open = true;
    $("#schedulerAdvanced").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("#samplerName").addEventListener("change", async () => {
    samplerValues = {};
    schedulerValues = {};
    schedulerUserSelected = false;
    schedulerPresetName = "";
    schedulerPresetPluginId = "";
    schedulerPresetSource = "";
    const preferred = preferredSchedulerForSampler($("#samplerName").value);
    if (preferred) $("#schedulerName").value = preferred;
    await refreshAdvancedEditors();
    saveSessionSoon();
  });
  $("#schedulerName").addEventListener("change", async () => {
    schedulerValues = {};
    schedulerUserSelected = true;
    schedulerPresetName = "";
    schedulerPresetPluginId = currentSchedulerPluginId();
    schedulerPresetSource = "";
    await refreshAdvancedEditors();
    saveSessionSoon();
  });

  $("#schedulerPresetPickerButton")?.addEventListener("click", () => {
    try {
      $("#schedulerPresetSelect")?.focus();
      $("#schedulerPresetSelect")?.showPicker?.();
    } catch {
      $("#schedulerPresetSelect")?.focus();
    }
  });
  $("#schedulerPresetLoadButton")?.addEventListener("click", async () => {
    const name = $("#schedulerPresetSelect").value.trim();
    if (!name) return;
    try {
      await loadSchedulerPreset(name);
    } catch (error) {
      notify(error.message, "error");
    }
  });
  $("#schedulerPresetSaveButton")?.addEventListener("click", async () => {
    try {
      await saveSchedulerPreset({ overwrite: false });
    } catch (error) {
      notify(error.message, "error");
    }
  });
  $("#schedulerPresetUpdateButton")?.addEventListener("click", async () => {
    try {
      await saveSchedulerPreset({ overwrite: true });
    } catch (error) {
      notify(error.message, "error");
    }
  });
  $("#schedulerPresetDeleteButton")?.addEventListener("click", async () => {
    try {
      await deleteSchedulerPreset();
    } catch (error) {
      notify(error.message, "error");
    }
  });
  $("#schedulerPresetSelect")?.addEventListener("input", () => {
    updateSchedulerPresetStatus();
  });
  $("#schedulerPresetSelect")?.addEventListener("change", async (event) => {
    const pluginId = currentSchedulerPluginId();
    const preset = mergedSchedulerPresets(pluginId).find((item) => item.name === event.target.value) || null;
    schedulerPresetName = preset?.name || event.target.value.trim() || "";
    schedulerPresetSource = preset?.source || "";
    if (preset) {
      try {
        await loadSchedulerPreset(preset.name);
      } catch (error) {
        notify(error.message, "error");
      }
    }
    updateSchedulerPresetStatus();
    saveSessionSoon();
  });
  $("#schedulerPresetExportButton")?.addEventListener("click", () => {
    const payload = {
      export_version: 1,
      kind: SCHEDULER_PRESET_KIND,
      plugin_id: currentSchedulerPluginId(),
      name: $("#schedulerPresetSelect")?.value?.trim() || "scheduler_preset",
      values: collectCurrentValues().scheduler_kwargs || {},
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${payload.name.replace(/[^a-z0-9._-]+/gi, "_") || "scheduler_preset"}.image_gen_scheduler_preset.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  });
  $("#schedulerPresetImportButton")?.addEventListener("click", () => $("#schedulerPresetImportInput")?.click());
  $("#schedulerPresetImportInput")?.addEventListener("change", async () => {
    const [file] = $("#schedulerPresetImportInput").files || [];
    $("#schedulerPresetImportInput").value = "";
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      schedulerValues = payload.values || {};
      schedulerPresetName = String(payload.name || file.name.replace(/\.json$/i, "")).trim();
      schedulerPresetSource = "";
      $("#schedulerPresetSelect").value = schedulerPresetName;
      await refreshAdvancedEditors({ preservePresetSelection: true });
      updateSchedulerPresetStatus();
      notify("Scheduler preset loaded from local file.");
    } catch (error) {
      notify(`Unable to load the local scheduler preset: ${error.message}`, "error");
    }
  });
}

function bindPanels() {
  const syncPanelToggle = (button, target) => {
    if (!button || !target) return;
    const collapsed = target.classList.contains("is-collapsed");
    button.textContent = collapsed ? "⌄" : "⌃";
    button.setAttribute("aria-expanded", String(!collapsed));
    const label = button.getAttribute("aria-label") || "Toggle panel";
    const normalized = label.replace(/^(Collapse|Expand)\s+/i, "");
    button.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${normalized}`);
    if (target.id === "recentOutputsPanel") {
      $("#centerSplitter")?.classList.toggle("is-hidden", collapsed);
    }
  };

  $$(".panel-toggle").forEach((button) => {
    if (button.dataset.layoutBound === "true") return;
    const target = document.getElementById(button.dataset.target);
    syncPanelToggle(button, target);
    button.addEventListener("click", () => {
      target?.classList.toggle("is-collapsed");
      syncPanelToggle(button, target);
    });
  });
  const outputsButton = $("#openOutputsButton");
  outputsButton?.addEventListener("click", async () => {
    outputsButton.disabled = true;
    try {
      await api.openOutputFolder();
      notify("Output folder opened.");
    } catch (error) {
      notify(`Unable to open the output folder: ${error.message}`, "error");
    } finally {
      outputsButton.disabled = false;
    }
  });
}

async function applyProfile(values) {
  await applyReplayValues(values);
}

async function clearJobCache() {
  const accepted = window.confirm(
    "Clear cached WebUI generation jobs?\n\n"
      + "This removes previous job requests, prompts, logs, diagnostics, and live-preview frames from data/webui/jobs. "
      + "Final generated images in the output folder are not deleted. Active or queued jobs are preserved.",
  );
  if (!accepted) return;
  const button = $("#clearJobCacheButton");
  const status = $("#jobCacheStatus");
  if (button) button.disabled = true;
  if (status) status.textContent = "Clearing inactive job cache…";
  try {
    const result = await api.clearJobCache();
    const removed = Number(result.removed_count || 0);
    const preserved = Array.isArray(result.preserved_active) ? result.preserved_active.length : 0;
    if (status) {
      status.textContent = `${removed} cached item${removed === 1 ? "" : "s"} removed${preserved ? ` · ${preserved} active/queued preserved` : ""}. Final outputs were not touched.`;
      status.className = "field-status ready";
    }
    window.dispatchEvent(new CustomEvent("job-cache-cleared", { detail: result }));
    notify(`${removed} cached job item${removed === 1 ? "" : "s"} removed. Final output images were preserved.`);
  } catch (error) {
    if (status) {
      status.textContent = `Unable to clear job cache: ${error.message}`;
      status.className = "field-status error";
    }
    notify(error.message, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

async function refreshModels() {
  try {
    const current = collectCurrentValues();
    const models = await api.refreshModels();
    state.models = models.models || [];
    state.vaes = models.vaes || [];
    state.loras = models.loras || [];
    state.textualInversions = models.textual_inversions || [];
    state.checkpointCatalog = [...state.models];
    populateModels(current);
    window.dispatchEvent(new CustomEvent("image-gen-asset-catalog-refreshed", { detail: models }));
    notify("Model and prompt-asset catalogs refreshed. The resident checkpoint was not reloaded.");
  } catch (error) {
    notify(error.message, "error");
  }
}

function waitForBackendRestart(previousInstanceId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const check = async () => {
      try {
        const response = await fetch(`/api/health?restart_probe=${Date.now()}`, {
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" },
        });
        if (response.ok) {
          const payload = await response.json();
          if (payload.instance_id && payload.instance_id !== previousInstanceId) {
            resolve(payload);
            return;
          }
        }
      } catch {
        // The connection normally drops while the backend process restarts.
      }
      if (Date.now() >= deadline) {
        reject(new Error("The backend did not return within two minutes. Check the WebUI launcher window for the startup error."));
        return;
      }
      window.setTimeout(check, 500);
    };
    check();
  });
}

async function reloadWorkspace() {
  const accepted = window.confirm(
    "Restart the IMAGE_GEN WebUI backend?\n\n"
      + "This stops active and queued generation work, recreates the Python backend process, and reloads all modules and startup-only settings. "
      + "The browser tab will remain open and reconnect automatically.",
  );
  if (!accepted) return;

  const button = $("#reloadWorkspaceButton");
  const label = button?.querySelector(".nav-button-label");
  const previousLabel = label?.textContent || button?.textContent || "Reload UI";
  if (button) {
    button.disabled = true;
    if (label) label.textContent = "Restarting…";
    else button.textContent = "Restarting…";
  }
  try {
    const response = await api.restartBackend();
    notify("Backend restart requested. Waiting for the new process…");
    await waitForBackendRestart(response.previous_instance_id);
    window.location.reload();
  } catch (error) {
    notify(`Unable to restart the WebUI backend: ${error.message}`, "error");
    if (button) {
      button.disabled = false;
      const currentLabel = button.querySelector(".nav-button-label");
      if (currentLabel) currentLabel.textContent = previousLabel;
      else button.textContent = previousLabel;
    }
  }
}

async function start() {
  try {
    await loadFragments();
    enforceExactDimensionInputs();
    const bootstrap = await api.bootstrap();
    state.bootstrap = bootstrap;
    state.settings = bootstrap.settings || {};
    state.activeModel = bootstrap.active_model || null;
    setCatalogs(bootstrap);
    state.recentOutputs = bootstrap.recent_outputs || [];

    const current = bootstrap.effective_generation || {
      ...(bootstrap.defaults || {}),
      ...(bootstrap.session || {}),
    };
    samplerValues = current.sampler_kwargs || {};
    schedulerValues = current.scheduler_kwargs || {};
    schedulerUserSelected = Boolean(current._webui_scheduler_user_selected);
    schedulerPresetName = current._webui_scheduler_preset_name || "";
    schedulerPresetPluginId = current._webui_scheduler_preset_plugin_id || "";
    schedulerPresetSource = current._webui_scheduler_preset_source || "";
    const restoredPromptAssets = replayPromptAssets(current);
    state.activePromptAssets = restoredPromptAssets;

    const applicationMetadata = bootstrap.application || {};
    const applicationVersion = applicationMetadata.version || bootstrap.version;
    const buildDisplay = applicationMetadata.build?.display || "";
    $("#appVersion").textContent = `v${applicationVersion}${buildDisplay ? ` · ${buildDisplay}` : ""}`;
    $("#projectPath").textContent = `Project: ${bootstrap.project_root}`;
    populateModels(current);
    populatePlugins(current);
    initializeHiresUpscalers(bootstrap.upscalers || {}, current);
    applyGenerationValues(current);
    applyVaeSelectionPolicy();
    initializePromptTools(current);
    await refreshAdvancedEditors({ preservePresetSelection: true });
    renderModelArchitectureStatus(state.activeModel);
    const workspaceLayout = bindWorkspaceLayout(bootstrap.settings || {});
    const defaultAssetsController = bindDefaultAssets(bootstrap.default_assets || {});
    const promptLoraSync = bindPromptLoraSync({ defaultAssetsController });
    promptLoraSync.syncFromPrompts();
    let workspaceTabs = null;
    const checkpointWorkspace = bindCheckpointWorkspace({
      activateModelPath: async (modelPath) => {
        if (!state.models.some((item) => normalizedModelPath(item.path) === normalizedModelPath(modelPath))) {
          state.models.push({ name: modelPath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || modelPath, path: modelPath, size_mb: 0 });
          populateModels(collectCurrentValues());
        }
        $("#modelPath").value = modelPath;
        const active = await activateSelectedModel();
        saveSessionSoon();
        return active;
      },
      unloadModel: async () => {
        const result = await api.unloadModel();
        state.activeModel = null;
        if (state.bootstrap) state.bootstrap.model_runtime = result.model_runtime || {};
        modelRuntimeReadyPath = "";
        renderModelArchitectureStatus(null);
        setModelReadyState(false, "No checkpoint is resident. Choose or load a model before generating.", "subtle");
        window.dispatchEvent(new CustomEvent("image-gen-model-unloaded", { detail: result }));
        return result;
      },
      showGenerationWorkspace: () => workspaceTabs?.showGeneration(),
      refreshGenerationModelSelect: () => populateModels(collectCurrentValues()),
    });
    const loraWorkspace = bindLoraWorkspace({
      defaultAssetsController,
      showGenerationWorkspace: () => workspaceTabs?.showGeneration(),
    });
    workspaceTabs = bindWorkspaceTabs({ checkpointWorkspace, loraWorkspace });
    window.addEventListener("image-gen-active-prompt-assets-updated", saveSessionSoon);
    bindLightbox();
    bindOutputDetails({ collect: collectCurrentValues, apply: applyReplayValues, onJobQueued: acceptQueuedJob });
    initializeRecentOutputBrowser(state.settings);
    bindGallery({ refreshOutputs });
    bindQueueComposer({ onQueued: acceptQueuedJob });
    bindBatchIO({ collect: collectCurrentValues, onQueued: acceptQueuedJob });
    bindVariationMatrix({ collect: collectCurrentValues, onQueued: acceptQueuedJob });
    bindCfgLab({ collect: collectCurrentValues, saveSession: saveSessionSoon, openVariationMatrix });
    bindOutputPatternBuilder();
    bindPromptTools({ saveSessionSoon });
    bindHiresUpscalers(saveSessionSoon);
    bindOutpaintPrototype(saveSessionSoon);
    $("#vaePath")?.addEventListener("change", applyVaeSelectionPolicy);
    renderGallery(state.recentOutputs);
    renderPromptPresets(bootstrap.prompt_presets || []);
    renderGenerationProfiles(bootstrap.generation_profiles || []);
    renderRuntimeStartupStatus(bootstrap.runtime_startup_status || null);
    bindRuntimeCommandCopy({ api, notify });
    bindSettings(bootstrap.settings || {}, {
      resetLayout: workspaceLayout.reset,
      saveLayoutDefault: workspaceLayout.saveCurrentScaleDefault,
      runtimeStartupStatus: bootstrap.runtime_startup_status || null,
    });
    bindPromptPresets(saveSessionSoon);
    bindGenerationProfiles({ collect: collectCurrentValues, apply: applyProfile });
    bindGeneration({
      collectValues: collectCurrentValues,
      refreshOutputs,
      ensureModelReady: ensureSelectedModelReady,
    });
    bindFormPersistence();
    bindModelSelection();
    bindAdvancedButtons();
    bindPanels();

    try {
      await applyStartupModelBehavior(bootstrap, current);
    } catch (error) {
      console.error("Initial model activation failed", error);
    }

    if ((bootstrap.selection_notes || []).length) {
      notify(bootstrap.selection_notes[0]);
    }

    $("#refreshModelsButton").addEventListener("click", refreshModels);
    $("#clearJobCacheButton")?.addEventListener("click", clearJobCache);
    $("#refreshOutputsButton").addEventListener("click", () => refreshOutputs());
    $("#reloadWorkspaceButton").addEventListener("click", reloadWorkspace);
    $("#restoreLastSession").addEventListener("change", async (event) => {
      const saved = await api.saveSettings({ restore_last_session: event.target.checked });
      $("#dialogRestoreLastSession").checked = saved.restore_last_session;
    });
    await refreshOutputs();
    window.__IMAGE_GEN_BOOT_READY__ = true;
    document.getElementById("webuiBootFailure")?.remove();
  } catch (error) {
    window.__IMAGE_GEN_BOOT_ERROR__ = String(error?.message || error || "Unknown startup error");
    window.dispatchEvent(new CustomEvent("image-gen-boot-failed", {
      detail: { message: window.__IMAGE_GEN_BOOT_ERROR__ },
    }));
    notify(`WebUI startup failed: ${error.message}`, "error");
    console.error(error);
  }
}

start();
