import { loadFragments } from "./fragments.js";
import { configureBranding, productName } from "./branding.js?v=brand1";

import { api } from "./api.js?v=cancel-controls1-ha2";
import { state, setCatalogs, samplerDescriptor, schedulerDescriptor } from "./state.js";
import { $, $$, debounce, option, replaceOptions, notify } from "./utils.js";
import { renderAdvancedEditor } from "./components/advanced-editor.js?v=scheduler-profile-scope2";
import { setSubsystemStatus } from "./components/status-indicators.js?v=1";
import { initResponsiveActionBars } from "./components/action-bar.js?v=responsive-action-bar1";
import { collectGenerationValues, applyGenerationValues } from "./components/form-state.js?v=sdxl-cfg-recommendations4";
import { acceptQueuedJob, bindGeneration } from "./features/generation.js?v=queue-active-pin1";
import { bindGallery, initializeRecentOutputBrowser, recentOutputApiFilters, renderGallery } from "./features/gallery.js?v=responsive-action-bar1";
import { bindPromptPresets, renderPromptPresets } from "./features/presets.js";
import { bindGenerationProfiles, renderGenerationProfiles } from "./features/profiles.js";
import { bindSettings } from "./features/settings.js?v=theme-manager-tm02a";
import { bindCivitaiConnection } from "./features/civitai-connection.js?v=civitai-connect1";
import { bindRuntimeCommandCopy, renderRuntimeStartupStatus } from "./features/memory-status.js?v=0.1.62";
import { bindWorkspaceLayout } from "./features/layout.js?v=responsive-action-bar1";
import { bindDefaultAssets } from "./features/default-assets.js?v=0.1.77";
import { bindCheckpointWorkspace } from "./features/checkpoints.js?v=asset-grid-qol1";
import { bindLoraWorkspace } from "./features/loras.js?v=asset-grid-qol1";
import { bindWorkspaceTabs } from "./features/workspace-tabs.js?v=0.1.74";
import { bindHomeWorkspace } from "./features/home.js?v=home-shell1";
import { bindHomeComponents } from "./features/home-components.js?v=content-capabilities2";
import { bindWorkspaceManager } from "./features/workspace-manager.js?v=workspace-responsive2";
import { bindChangelog } from "./features/changelog.js?v=content-capabilities2";
import { bindHelpCenter } from "./features/help-center.js?v=help-center1";
import { bindBugReporter } from "./features/bug-reports.js?v=bug-manager1";
import { bindImageGenProfile } from "./features/profile.js?v=clean-install1";
import { bindLightbox } from "./features/lightbox.js?v=0.1.40";
import { enforceExactDimensionInputs } from "./features/exact-dimensions.js";
import { bindOutputDetails } from "./features/output-details.js?v=sdxl-cfg-recommendations4";
import { bindQueueComposer } from "./features/queue-composer.js";
import { bindBatchIO } from "./features/batch-io.js";
import { bindVariationMatrix, openVariationMatrix } from "./features/variation-matrix.js?v=responsive-action-bar1";
import { applyCfgPreset, bindCfgLab } from "./features/cfg-lab.js?v=0.1.47-lightning-recommendation";
import { bindOutputPatternBuilder } from "./features/output-pattern-builder.js";
import { bindPromptTools, initializePromptTools, refreshPromptConfigurationCatalogs } from "./features/prompt-tools.js?v=r10.3";
import { bindPromptLoraSync } from "./features/prompt-lora-sync.js?v=0.1.77";
import { bindHiresUpscalers, initializeHiresUpscalers } from "./features/hires-upscalers.js?v=ha3";
import { bindHiresProfiles } from "./features/hires-profiles.js?v=ha3";
import { bindUserConfigEditor } from "./features/user-config-editor.js?v=qol1";
import { bindParameterRanges } from "./features/parameter-ranges.js?v=qol2";
import { bindSeedControls } from "./features/seed-controls.js?v=qol-seed-ui5";
import { bindOutpaintPrototype } from "./features/outpaint-prototype.js?v=0.1.84";
import { bindAdvancedModels } from "./features/advanced-models.js?v=component-phase05";

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
let refreshOutputsPromise = null;
let startupActivationScheduled = false;

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

function currentSchedulerPresetCatalogKey() {
  const descriptor = currentSchedulerDescriptor();
  return String(descriptor?.name || descriptor?.plugin_id || $("#schedulerName").value || "").trim();
}

function schedulerPresetSupportEnabled() {
  return currentSchedulerPresetCatalogKey() === "simple_kes";
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
  if (status?.generation_ready === true) return true;
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
  const messageText = String(message || "");
  const lowerMessage = messageText.toLowerCase();
  const isTransitioning = kind === "loading" || lowerMessage.includes("activating") || lowerMessage.includes("loading");
  const isFailure = kind === "error" && (lowerMessage.includes("failed") || lowerMessage.includes("did not return") || lowerMessage.includes("could not"));
  const isUnselected = lowerMessage.includes("choose") || lowerMessage.includes("no checkpoint") || lowerMessage.includes("start with no model");
  setSubsystemStatus({
    id: "modelActivationStatusLight",
    host: "#modelStatusLightHost",
    label: "Checkpoint model",
    status: ready ? "healthy" : isTransitioning ? "transitioning" : isFailure ? "critical" : isUnselected ? "inactive" : "warning",
    stateLabel: ready ? "Ready" : isTransitioning ? "Activating" : isFailure ? "Activation failed" : isUnselected ? "Not loaded" : "Attention",
    summary: messageText || "Checkpoint model status is unavailable.",
    detail: state.activeModel?.resolved_path ? `Resolved model: ${state.activeModel.resolved_path}` : "No resident checkpoint path is currently reported.",
    facts: {
      selected_model: $("#modelPath")?.value || "none",
      runtime_ready: ready ? "yes" : "no",
      active_model: state.activeModel?.model_name || "none",
    },
    diagnosticTarget: "#modelLoadStatus",
  });
  const allowQueueWhileLoading = !ready && isTransitioning;
  const shouldDisableGeneration = !ready && !allowQueueWhileLoading;
  ["#generateButton", "#generateMenuButton", "#infinityButton"].forEach((selector) => {
    const button = $(selector);
    if (button) button.disabled = shouldDisableGeneration;
  });
}

function renderModelRecommendationControls(activeModel = null, { applyDefaults = false, applyField = "all" } = {}) {
  const panel = $("#sdxlRecommendationControls");
  const stepsToggle = $("#sdxlEnforceRecommendedSteps");
  const cfgToggle = $("#sdxlEnforceRecommendedCfg");
  const stepsInput = $("#steps");
  const cfgInput = $("#cfgScale");
  const samplerInput = $("#samplerName");
  const schedulerInput = $("#schedulerName");
  if (!panel || !stepsToggle || !cfgToggle || !stepsInput || !cfgInput) return;

  const profile = activeModel?.runtime_profile || {};
  const profileId = String(profile.profile_id || "").trim();
  const family = String(profile.family || "").trim().toLowerCase();
  const recommendationProfile = Boolean(profile.recommendation_ui_enabled) || family === "lightning" || family === "turbo";
  const previousProfileId = String(panel.dataset.profileId || "");
  const profileChanged = Boolean(profileId && previousProfileId !== profileId);

  stepsInput.disabled = false;
  cfgInput.disabled = false;

  if (!recommendationProfile) {
    panel.classList.add("is-hidden");
    panel.classList.remove("is-experimental");
    panel.dataset.profileId = profileId || "non-accelerated-sdxl";
    delete cfgInput.dataset.recommendedEffectiveCfg;
    return;
  }

  panel.dataset.profileId = profileId;
  panel.classList.remove("is-hidden");

  const recommendedSteps = Array.isArray(profile.recommended_steps)
    ? profile.recommended_steps.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0)
    : [];
  const preferredSteps = Number(profile.required_steps || recommendedSteps[0] || 0);
  const recommendedCfg = Number(profile.image_gen_cfg_scale);
  const recommendedCfgPreset = String(profile.recommended_cfg_preset || "").trim();
  const applyStepsNow = (profileChanged || (applyDefaults && ["all", "steps"].includes(applyField))) && stepsToggle.checked;
  const applyCfgNow = (profileChanged || (applyDefaults && ["all", "cfg"].includes(applyField))) && cfgToggle.checked;

  if (applyStepsNow && Number.isFinite(preferredSteps) && preferredSteps > 0) {
    stepsInput.value = String(preferredSteps);
  }
  if (applyCfgNow && Number.isFinite(recommendedCfg)) {
    cfgInput.value = String(recommendedCfg);
    if (recommendedCfgPreset) applyCfgPreset(recommendedCfgPreset, { notifyUser: false });
  }

  if (cfgToggle.checked && Number.isFinite(recommendedCfg)) cfgInput.dataset.recommendedEffectiveCfg = String(recommendedCfg);
  else delete cfgInput.dataset.recommendedEffectiveCfg;

  const stepsLabel = $("#sdxlRecommendedStepsLabel");
  const cfgLabel = $("#sdxlRecommendedCfgLabel");
  const profileLabel = $("#sdxlRecommendationProfile");
  const status = $("#sdxlRecommendationStatus");
  const samplerSchedulerStatus = $("#sdxlRecommendedSamplerScheduler");
  if (stepsLabel) {
    const label = recommendedSteps.length ? recommendedSteps.join(", ") : (preferredSteps > 0 ? String(preferredSteps) : "model default");
    stepsLabel.textContent = `Use recommended steps (${label})`;
  }
  if (cfgLabel) {
    const cfgText = Number.isFinite(recommendedCfg) ? `Use recommended CFG (${recommendedCfg})` : "Use the model's recommended CFG";
    cfgLabel.textContent = recommendedCfgPreset ? `${cfgText} + recommended CFG Lab preset` : cfgText;
  }
  if (profileLabel) profileLabel.textContent = profileId;
  if (samplerSchedulerStatus) {
    const recommendedSampler = String(profile.sampler_name || "").trim();
    const recommendedScheduler = String(profile.scheduler_name || "").trim();
    if (recommendedSampler || recommendedScheduler) {
      const pair = [recommendedSampler || "model choice", recommendedScheduler || "model choice"].join(" / ");
      samplerSchedulerStatus.textContent = `Recommended sampler / scheduler: ${pair}. These are advisory and are never selected or enforced by the model profile.`;
    } else {
      samplerSchedulerStatus.textContent = "This model profile does not declare a sampler/scheduler recommendation.";
    }
  }

  const warnings = [];
  const requestedSteps = Number(stepsInput.value);
  const requestedCfg = Number(cfgInput.value);
  const samplerName = String(samplerInput?.value || "").trim();
  const schedulerName = String(schedulerInput?.value || "").trim();
  if (recommendedSteps.length && Number.isFinite(requestedSteps) && !recommendedSteps.includes(requestedSteps)) {
    const maxRecommended = Math.max(...recommendedSteps);
    const relation = requestedSteps > maxRecommended ? "exceeds" : "is outside";
    warnings.push(stepsToggle.checked
      ? `Steps ${requestedSteps} ${relation} the recommended ${recommendedSteps.join("/")} setting; the recommended value will be used because its checkbox is checked.`
      : `Steps ${requestedSteps} ${relation} the recommended ${recommendedSteps.join("/")} setting; generation is still allowed.`);
  }
  if (Number.isFinite(recommendedCfg) && Number.isFinite(requestedCfg) && Math.abs(requestedCfg - recommendedCfg) > 1e-6) {
    warnings.push(cfgToggle.checked
      ? `CFG ${requestedCfg} differs from the recommended ${recommendedCfg}; CFG ${recommendedCfg} will be used because its checkbox is checked.`
      : `CFG ${requestedCfg} differs from the recommended ${recommendedCfg}; generation is still allowed.`);
  }
  if (profile.sampler_name && samplerName && samplerName !== profile.sampler_name) {
    warnings.push(`Sampler ${samplerName} differs from the recommended ${profile.sampler_name}; generation is still allowed.`);
  }
  if (profile.scheduler_name && schedulerName && schedulerName !== profile.scheduler_name) {
    warnings.push(`Scheduler ${schedulerName} differs from the recommended ${profile.scheduler_name}; generation is still allowed.`);
  }

  panel.classList.toggle("is-experimental", warnings.length > 0);
  if (status) {
    const activeRules = [];
    if (stepsToggle.checked) activeRules.push(`recommended steps${Number.isFinite(preferredSteps) && preferredSteps > 0 ? ` ${preferredSteps}` : ""}`);
    if (cfgToggle.checked) activeRules.push(`recommended CFG${Number.isFinite(recommendedCfg) ? ` ${recommendedCfg}` : ""}`);
    if (cfgToggle.checked && recommendedCfgPreset) activeRules.push("the recommended CFG Lab preset on activation/enable");
    const ruleText = activeRules.length
      ? ` Enabled recommendation${activeRules.length === 1 ? "" : "s"}: ${activeRules.join(", ")}. Fields remain editable; uncheck a recommendation to make that field value authoritative for generation.`
      : " All model recommendations are advisory; the values you enter will be used.";
    status.textContent = warnings.length
      ? `${warnings.join(" ")}${ruleText}`
      : `The current settings match this model's recommendations.${ruleText}`;
    status.className = `field-status subtle${warnings.length ? " warning" : ""}`;
  }

 }

// Backwards-compatible internal alias for SDXL-focused tests/extensions.
// The implementation is now model-profile generic and also serves SD3.x.
const renderSdxlRecommendationControls = renderModelRecommendationControls;

function renderModelArchitectureStatus(activeModel = null) {
  const status = $("#modelArchitectureStatus");
  if (!status) return;
  const summary = String(activeModel?.architecture_summary || activeModel?.architecture_contract?.summary || "").trim();
  if (!activeModel) {
    status.textContent = "Architecture: waiting for activation.";
    status.className = "field-status subtle";
    applySd2RuntimePolicy({ activeModel: selectedModelRecord(), autoEnable: true });
    renderModelRecommendationControls(null);
    return;
  }
  if (summary) {
    status.textContent = `Architecture: ${summary}`;
    status.className = "field-status ready subtle";
    applySd2RuntimePolicy({ activeModel, autoEnable: true });
    renderModelRecommendationControls(activeModel);
    return;
  }
  status.textContent = "Architecture: unknown for the current checkpoint.";
  status.className = "field-status subtle";
  applySd2RuntimePolicy({ activeModel, autoEnable: true });
  renderModelRecommendationControls(activeModel);
}

async function activateSelectedModel({ quiet = false } = {}) {
  if ($("#advancedModelsEnabled")?.checked) {
    throw new Error("Advanced Models is enabled. Disable it before activating a whole checkpoint.");
  }
  const requestedPath = $("#modelPath").value;
  if (!requestedPath) {
    state.activeModel = null;
    setModelReadyState(false, "No checkpoint is installed or selected. Add a checkpoint before generating.", "subtle");
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
  const lastUsed = resolveCatalogModelPath(current.model_path || "");
  const pinned = resolveCatalogModelPath(state.settings.checkpoint_startup_path || "");
  const configuredDefaultSource = bootstrap.defaults?.model_path || bootstrap.effective_generation?.model_path || "";
  const configuredDefault = resolveCatalogModelPath(configuredDefaultSource);
  const active = bootstrap.active_model || null;

  let selectedPath = "";
  if (mode === "pinned_default") selectedPath = pinned || configuredDefault;
  else if (mode === "last_used") selectedPath = lastUsed || pinned || configuredDefault;

  if (selectedPath) {
    syncModelDropdownSelection(selectedPath);
  }

  if (!state.models.length) {
    state.activeModel = null;
    modelRuntimeReadyPath = "";
    renderModelArchitectureStatus(null);
    setModelReadyState(false, "No checkpoints are installed. Add a checkpoint to the model library before generating.", "subtle");
    return;
  }

  if (mode === "none" || !selectedPath) {
    state.activeModel = active || null;
    modelRuntimeReadyPath = "";
    renderModelArchitectureStatus(state.activeModel);
    setModelReadyState(false, "Start with no model is active. Choose a checkpoint before generating.", "subtle");
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
    setModelReadyState(true, "Loading remembered startup checkpoint in the background…", "subtle");
    if (!startupActivationScheduled) {
      startupActivationScheduled = true;
      window.setTimeout(() => {
        void activateSelectedModel({ quiet: true }).finally(() => {
          startupActivationScheduled = false;
        });
      }, 0);
    }
    return;
  }
  setModelReadyState(true, "Selected model will load on demand when generation starts.", "subtle");
}

function applyVaeSelectionPolicy() {
  const select = $("#vaePath");
  const status = $("#vaeSelectionStatus");
  if (!select || !status) return;
  const advancedModelsEnabled = Boolean($("#advancedModelsEnabled")?.checked);
  select.disabled = advancedModelsEnabled;
  const civitaiButton = $("#vaeFetchCivitaiButton");
  if (civitaiButton) civitaiButton.disabled = advancedModelsEnabled || !select.value;
  if (advancedModelsEnabled) {
    status.textContent = "Advanced Models owns VAE selection; the whole-model VAE override is ignored.";
    status.className = "field-status subtle";
    return;
  }
  if (!select.value) {
    status.textContent = "Automatic / checkpoint embedded.";
  } else {
    const label = select.selectedOptions?.[0]?.textContent || select.value;
    const record = state.vaes.find((item) => normalizedModelPath(item.path) === normalizedModelPath(select.value));
    const lookup = record?.civitai_lookup || record?.metadata?._civitai_lookup || {};
    const civitaiLabel = lookup?.model_name
      ? ` · CivitAI: ${lookup.model_name}${lookup.creator ? ` by ${lookup.creator}` : ""}`
      : "";
    status.textContent = `Manual external VAE selected: ${label}${civitaiLabel}`;
  }
  status.className = "field-status subtle";
}

function selectedModelRecord() {
  const requestedPath = $("#modelPath")?.value || "";
  const normalizedRequested = normalizedModelPath(requestedPath);
  return state.models.find((item) => normalizedModelPath(item.path) === normalizedRequested) || state.activeModel || null;
}

function modelFamilyForSd2Controls(model = null) {
  const source = model || selectedModelRecord();
  const raw = String(
    source?.architecture
      || source?.architecture_contract?.family
      || source?.model_family
      || source?.architecture_summary
      || "",
  ).trim().toLowerCase();
  if (!raw) return "";
  if (raw.includes("sd2") || raw.includes("stable diffusion 2") || raw.includes("2.0") || raw.includes("2.1")) {
    return "sd2.x";
  }
  return raw;
}

function applySd2RuntimePolicy({ activeModel = null, autoEnable = true } = {}) {
  const select = $("#sd2RuntimeProfileOverride");
  const toggle = $("#sd2DedicatedGeneration");
  const status = $("#sd2RuntimeStatus");
  if (!select || !toggle || !status) return;
  const family = modelFamilyForSd2Controls(activeModel);
  const isSd2 = family === "sd2.x";
  if (!isSd2) {
    select.disabled = true;
    toggle.disabled = true;
    toggle.checked = false;
    select.value = "";
    delete toggle.dataset.userOverridden;
    status.textContent = "SD2.x runtime controls are disabled until an SD2.x checkpoint is selected.";
    status.className = "field-status subtle";
    return;
  }
  select.disabled = false;
  toggle.disabled = false;
  if (autoEnable && !toggle.dataset.userOverridden) {
    toggle.checked = true;
  }
  const mode = toggle.checked ? "Dedicated SD2.x runtime enabled." : "Dedicated SD2.x runtime is currently disabled.";
  const profileLabel = select.value
    ? ` Runtime profile override: ${select.selectedOptions?.[0]?.textContent || select.value}.`
    : " Runtime profile: automatic checkpoint-qualified inference.";
  status.textContent = `${mode}${profileLabel}`;
  status.className = `field-status subtle${toggle.checked ? " ready" : ""}`;
}

async function enrichSelectedVaeFromCivitai() {
  const select = $("#vaePath");
  const button = $("#vaeFetchCivitaiButton");
  if (!select?.value) {
    notify("Select an external VAE before fetching CivitAI metadata.", "warning");
    return;
  }
  const record = state.vaes.find((item) => normalizedModelPath(item.path) === normalizedModelPath(select.value));
  if (!record?.asset_id) {
    notify(`The selected VAE is not registered in the ${productName()} asset catalog. Refresh models and try again.`, "warning");
    return;
  }
  const previousTitle = button?.title || "Refresh selected VAE metadata from CivitAI";
  if (button) {
    button.disabled = true;
    button.classList.add("is-working");
    button.title = "Fetching VAE metadata from CivitAI…";
  }
  try {
    const updated = await api.enrichAssetFromCivitai("vae", record.asset_id, false);
    state.vaes = state.vaes.map((item) => item.asset_id === updated.asset_id ? { ...item, ...updated } : item);
    const currentModel = $("#modelPath")?.value || "";
    const currentVae = select.value;
    populateModels({ model_path: currentModel, vae_path: currentVae });
    applyVaeSelectionPolicy();
    const lookup = updated.civitai_lookup || updated.metadata?._civitai_lookup || {};
    notify(lookup.preview_image_downloaded
      ? "CivitAI metadata and a preview image were added to the VAE sidecar."
      : "CivitAI metadata was added to the VAE sidecar.");
  } catch (error) {
    notify(`Unable to fetch VAE metadata from CivitAI: ${error.message}`, "error");
  } finally {
    if (button) {
      button.disabled = !select.value;
      button.classList.remove("is-working");
      button.title = previousTitle;
    }
  }
}

function populateModels(current = {}) {
  const modelOptions = state.models.map((item) => option(
    item.path,
    `${item.display_name || item.embedded_name || item.name} · ${item.size_mb} MB`,
  ));
  const requestedModelRaw = current.model_path || "";
  const requestedModel = resolveCatalogModelPath(requestedModelRaw);
  if (!modelOptions.length) {
    modelOptions.push(option("", "No checkpoints installed"));
  }
  replaceOptions($("#modelPath"), modelOptions, requestedModel);

  const vaeOptions = [option("", "Automatic / checkpoint embedded")];
  state.vaes.forEach((item) => vaeOptions.push(option(item.path, `${item.name} · ${item.size_mb} MB`)));
  replaceOptions($("#vaePath"), vaeOptions, current.vae_path || "");
  applyVaeSelectionPolicy();
  applySd2RuntimePolicy({ autoEnable: !("sd2_dedicated_generation" in current), activeModel: selectedModelRecord() });
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

async function refreshAdvancedEditors({ preservePresetSelection = false } = {}) {
  await renderAdvancedEditor({
    container: $("#samplerAdvancedContent"),
    descriptor: samplerDescriptor($("#samplerName").value),
    kind: "sampler",
    currentValues: samplerValues,
    onProfileStateChange: ({ values = {} } = {}) => {
      samplerValues = { ...(values || {}) };
    },
    onChange: () => {
      saveSessionSoon();
    },
  });

  const schedulerPluginId = currentSchedulerPluginId();
  const schedulerPresetCatalogKey = currentSchedulerPresetCatalogKey();
  const builtinProfiles = schedulerPresetSupportEnabled()
    ? (BUILTIN_SCHEDULER_PRESETS[schedulerPresetCatalogKey] || [])
    : [];
  await renderAdvancedEditor({
    container: $("#schedulerAdvancedContent"),
    descriptor: schedulerDescriptor($("#schedulerName").value),
    kind: "scheduler",
    currentValues: schedulerValues,
    builtinProfiles,
    selectedProfileName: preservePresetSelection ? schedulerPresetName : "",
    onProfileStateChange: ({ name = "", source = "", values = {} } = {}) => {
      schedulerValues = { ...(values || {}) };
      schedulerPresetName = name;
      schedulerPresetPluginId = schedulerPluginId;
      schedulerPresetSource = source;
    },
    onChange: () => {
      saveSessionSoon();
    },
  });
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
    $("#seed").dispatchEvent(new Event("input", { bubbles: true }));
    saveSessionSoon();
  });
}

function bindModelSelection() {
  $("#modelPath").addEventListener("change", async () => {
    applySd2RuntimePolicy({ activeModel: selectedModelRecord(), autoEnable: true });
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


}

function bindPanels() {
  const syncPanelToggle = (button, target) => {
    if (!button || !target) return;
    const collapsed = target.classList.contains("is-collapsed");
    button.setAttribute("aria-expanded", String(!collapsed));
    const label = button.getAttribute("aria-label") || "Toggle panel";
    const normalized = label.replace(/^(Collapse|Expand)\s+/i, "");
    const nextLabel = `${collapsed ? "Expand" : "Collapse"} ${normalized}`;
    button.setAttribute("aria-label", nextLabel);
    button.title = nextLabel;
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
    `Restart the ${productName()} WebUI backend?\n\n`
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
    bindParameterRanges();
    bindSeedControls({ onChange: saveSessionSoon });
    const bootstrap = await api.bootstrap();
    configureBranding(bootstrap.application || {});
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
    [
      ["#sdxlEnforceRecommendedSteps", "steps"],
      ["#sdxlEnforceRecommendedCfg", "cfg"],
    ].forEach(([selector, field]) => {
      const control = $(selector);
      if (!control) return;
      control.addEventListener("change", () => {
        renderModelRecommendationControls(state.activeModel, { applyDefaults: control.checked, applyField: field });
        window.dispatchEvent(new CustomEvent("image-gen-cfg-recommendation-changed"));
        saveSessionSoon();
      });
    });
    window.addEventListener("image-gen-cfg-recommendation-changed", () => {
      $("#cfgScale")?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    ["#steps", "#cfgScale"].forEach((selector) => {
      const control = $(selector);
      if (!control) return;
      control.addEventListener("input", () => {
        // Recommendation toggles are startup/default preferences, not locks.
        // Keep them exactly as the user set them while allowing the current
        // Steps/CFG value to diverge. The recommendation panel will warn and
        // the checkbox remains available if the user wants to disable future
        // auto-application for that field.
        renderModelRecommendationControls(state.activeModel);
        saveSessionSoon();
      });
    });
    ["#samplerName", "#schedulerName"].forEach((selector) => {
      const control = $(selector);
      if (!control) return;
      control.addEventListener("change", () => renderModelRecommendationControls(state.activeModel));
    });
    applyVaeSelectionPolicy();
    initializePromptTools(current);
    await refreshAdvancedEditors({ preservePresetSelection: true });
    renderModelArchitectureStatus(state.activeModel);
    const workspaceLayout = bindWorkspaceLayout(bootstrap.settings || {});
    bindHomeComponents(document);
    initResponsiveActionBars(document);
    const defaultAssetsController = bindDefaultAssets(bootstrap.default_assets || {});
    const promptLoraSync = bindPromptLoraSync({ defaultAssetsController });
    promptLoraSync.syncFromPrompts();
    let workspaceTabs = null;
    const checkpointWorkspace = bindCheckpointWorkspace({
      activateModelPath: async (modelPath) => {
        if ($("#advancedModelsEnabled")?.checked) {
          notify("Advanced Models is active. Disable it before activating a whole checkpoint.", "warning");
          return null;
        }
        if (!state.models.some((item) => normalizedModelPath(item.path) === normalizedModelPath(modelPath))) {
          state.models.push({ name: modelPath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || modelPath, path: modelPath, size_mb: 0 });
          populateModels(collectCurrentValues());
        }
        $("#modelPath").value = modelPath;
    applySd2RuntimePolicy({ activeModel: selectedModelRecord(), autoEnable: true });
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
    bindWorkspaceManager();
    bindHomeWorkspace({
      models: state.models,
      vaes: state.vaes,
      loras: state.loras,
    });
    bindChangelog();
    await bindHelpCenter();
    bindBugReporter();
    bindImageGenProfile();
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
    await bindHiresProfiles({
      collect: collectCurrentValues,
      apply: async (values) => applyGenerationValues({ ...collectCurrentValues(), ...(values || {}) }),
      saveSessionSoon,
    });
    bindUserConfigEditor();
    window.addEventListener("image-gen-parameter-ranges-changed", saveSessionSoon);
    bindOutpaintPrototype(saveSessionSoon);
    await bindAdvancedModels({ values: current, saveSessionSoon });
    $("#vaePath")?.addEventListener("change", applyVaeSelectionPolicy);
    $("#sd2DedicatedGeneration")?.addEventListener("change", (event) => {
      event.currentTarget.dataset.userOverridden = "true";
      applySd2RuntimePolicy({ activeModel: selectedModelRecord(), autoEnable: false });
    });
    $("#sd2RuntimeProfileOverride")?.addEventListener("change", () => {
      applySd2RuntimePolicy({ activeModel: selectedModelRecord(), autoEnable: false });
    });
    $("#vaeFetchCivitaiButton")?.addEventListener("click", enrichSelectedVaeFromCivitai);
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
    bindCivitaiConnection();
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
      if (!current.advanced_models_enabled) {
        await applyStartupModelBehavior(bootstrap, current);
      }
    } catch (error) {
      console.error("Initial model activation failed", error);
    }

    if ((bootstrap.selection_notes || []).length) {
      notify(bootstrap.selection_notes[0]);
    }

    $("#refreshModelsButton").addEventListener("click", refreshModels);
    $("#clearJobCacheButton")?.addEventListener("click", clearJobCache);
    $("#refreshOutputsButton").addEventListener("click", async () => {
      const button = $("#refreshOutputsButton");
      button?.classList.add("is-working");
      if (button) button.disabled = true;
      try {
        await refreshOutputs();
      } finally {
        button?.classList.remove("is-working");
        if (button) button.disabled = false;
      }
    });
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
