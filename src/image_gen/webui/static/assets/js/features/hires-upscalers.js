import { api } from "../api.js?v=civitai-connect1";
import { state } from "../state.js";
import { $, notify } from "../utils.js";
import { setSubsystemStatus } from "../components/status-indicators.js?v=1";

function catalog() {
  return state.upscalers || {};
}

function selectedDescriptor(id = $("#hiresUpscaler")?.value || "") {
  return (catalog().neural || []).find((item) => item.upscaler_id === id) || null;
}

function addOption(select, item, { disabled = false } = {}) {
  const option = document.createElement("option");
  option.value = item.upscaler_id;
  option.textContent = item.display_name || item.upscaler_id;
  option.disabled = disabled;
  select.appendChild(option);
}

function friendlyAvailability(item = {}) {
  const loadStatus = String(item.load_status || "").toLowerCase();
  const runtimeStatus = String(item.runtime_qualification?.status || "unqualified").toLowerCase();
  if (loadStatus === "deferred_architecture") return "Detected but not currently supported in this build.";
  if (loadStatus === "deferred_scale") return "Detected, but this model's native scale is not currently supported.";
  if (item.selectable && ["unqualified", "backend_unqualified"].includes(runtimeStatus)) {
    return "Detected and usable, but not yet performance-qualified on this system.";
  }
  if (item.selectable) return "Detected and available for neural hires.";
  return item.bounded_error || "Detected but unavailable for neural hires.";
}

function renderDiagnostics() {
  const node = $("#hiresUpscalerDiagnosticsText");
  if (!node) return;
  const values = [];
  (catalog().unavailable_neural || []).forEach((item) => {
    values.push(`${item.file_name || item.display_name}: ${friendlyAvailability(item)}`);
  });
  (catalog().diagnostics || []).forEach((item) => {
    values.push(`${item.severity || "info"}: ${item.code || "discovery"} - ${item.message || ""}${item.path ? ` (${item.path})` : ""}`);
  });
  node.textContent = values.length ? values.join("\n") : "No unsupported upscaler files were reported.";
}


function clearHiresUpscalerError() {
  const select = $("#hiresUpscaler");
  select?.classList.remove("field-error-focus");
  const recovery = $("#hiresUpscalerRecovery");
  if (recovery) recovery.hidden = true;
}

function focusHiresUpscalerError(message = "") {
  const select = $("#hiresUpscaler");
  if (!select) return;
  select.classList.add("field-error-focus");
  select.scrollIntoView({ behavior: "smooth", block: "center" });
  select.focus({ preventScroll: true });
  const recovery = $("#hiresUpscalerRecovery");
  if (recovery) recovery.hidden = false;
  const copy = $("#hiresUpscalerRecoveryMessage");
  if (copy) {
    copy.textContent = message || "Install or move a supported upscaler into a configured asset folder, then refresh discovery. If your asset path is custom, update user-config.yml.";
  }
}

function renderStatus() {
  const status = $("#hiresUpscalerStatus");
  if (!status) return;
  const item = selectedDescriptor();
  if (!item) {
    status.textContent = "No supported neural .pth upscaler is selected. Refresh or install a recognized model.";
    status.className = "field-status error";
    setSubsystemStatus({
      id: "hiresUpscalerSubsystemStatusLight",
      host: "#hiresUpscalerStatusLightHost",
      label: "Neural hires upscaler",
      status: "inactive",
      stateLabel: "Not selected",
      summary: "No supported neural upscaler is currently selected.",
      detail: "Refresh discovery or select a recognized .pth upscaler to resolve runtime qualification.",
      diagnosticTarget: "#hiresUpscalerDiagnostics",
    });
    const civitaiButton = $("#upscalerFetchCivitaiButton");
    if (civitaiButton) civitaiButton.disabled = true;
    return;
  }
  const lookup = item.civitai_lookup || item.metadata?._civitai_lookup || {};
  const civitai = lookup?.model_name
    ? ` · CivitAI: ${lookup.model_name}${lookup.creator ? ` by ${lookup.creator}` : ""}`
    : "";
  clearHiresUpscalerError();
  status.textContent = `${item.display_name} · ${item.architecture} · native x${item.native_scale} · ${friendlyAvailability(item)}${civitai}`;
  status.className = item.selectable ? "field-status ready" : "field-status error";
  const loadStatus = String(item.load_status || "").toLowerCase();
  const runtimeStatus = String(item.runtime_qualification?.status || "unqualified").toLowerCase();
  const caution = item.selectable && ["unqualified", "backend_unqualified"].includes(runtimeStatus);
  const deferred = loadStatus.startsWith("deferred_");
  setSubsystemStatus({
    id: "hiresUpscalerSubsystemStatusLight",
    host: "#hiresUpscalerStatusLightHost",
    label: "Neural hires upscaler",
    status: item.selectable ? (caution ? "warning" : "healthy") : (deferred ? "warning" : "critical"),
    stateLabel: item.selectable ? (caution ? "Usable, unqualified" : "Available") : (deferred ? "Deferred" : "Unavailable"),
    summary: friendlyAvailability(item),
    detail: item.bounded_error || `Architecture: ${item.architecture}. Native scale: x${item.native_scale}. Runtime qualification: ${runtimeStatus}.`,
    facts: {
      upscaler: item.display_name || item.upscaler_id,
      architecture: item.architecture || "unknown",
      native_scale: item.native_scale || "unknown",
      load_status: loadStatus || "unknown",
      runtime_qualification: runtimeStatus,
    },
    diagnosticTarget: "#hiresUpscalerDiagnostics",
  });
  const civitaiButton = $("#upscalerFetchCivitaiButton");
  if (civitaiButton) civitaiButton.disabled = !item?.upscaler_id;
}

function setTileControls(enabled, item) {
  const capability = String(item?.tile_capability || (item?.tile_supported ? "supported" : item?.selectable ? "unsupported" : "unqualified"));
  const tileEnabled = Boolean(enabled && item && capability === "supported");
  ["#hiresTileSize", "#hiresTileOverlap", "#hiresTileBatchSize"].forEach((selector) => {
    const node = $(selector);
    if (node) node.disabled = !tileEnabled;
  });
  const status = $("#hiresTilingStatus");
  if (!status) return;
  if (!enabled) status.textContent = "Tiling controls are disabled while hires generation is off.";
  else if (!item) status.textContent = "Select an upscaler to resolve tiling capability.";
  else if (capability === "supported") status.textContent = "Tiling is supported by this upscaler descriptor.";
  else if (capability === "unsupported") status.textContent = "This upscaler runs untiled.";
  else status.textContent = "Tiling has not been qualified for this model architecture.";
  status.className = capability === "supported" ? "field-status subtle" : "field-status warning";
}

function correctionPreview(nativeWidth, nativeHeight, targetWidth, targetHeight, policy) {
  if (!(nativeWidth > 0 && nativeHeight > 0 && targetWidth > 0 && targetHeight > 0)) return null;
  const sourceAspect = nativeWidth / nativeHeight;
  const targetAspect = targetWidth / targetHeight;
  if (policy === "crop_to_fill") {
    const scale = Math.max(targetWidth / nativeWidth, targetHeight / nativeHeight);
    const preWidth = Math.max(targetWidth, Math.ceil(nativeWidth * scale - 1e-12));
    const preHeight = Math.max(targetHeight, Math.ceil(nativeHeight * scale - 1e-12));
    const fraction = 1 - ((targetWidth * targetHeight) / Math.max(1, preWidth * preHeight));
    return { scale, geometryFraction: Math.max(0, fraction), geometryLabel: "cropped area" };
  }
  if (policy === "pad_to_fit") {
    const scale = Math.min(targetWidth / nativeWidth, targetHeight / nativeHeight);
    const fitWidth = Math.min(targetWidth, Math.max(1, Math.floor(nativeWidth * scale + 1e-12)));
    const fitHeight = Math.min(targetHeight, Math.max(1, Math.floor(nativeHeight * scale + 1e-12)));
    const fraction = 1 - ((fitWidth * fitHeight) / Math.max(1, targetWidth * targetHeight));
    return { scale, geometryFraction: Math.max(0, fraction), geometryLabel: "padded area" };
  }
  const scaleX = targetWidth / nativeWidth;
  const scaleY = targetHeight / nativeHeight;
  const geometryFraction = Math.abs(targetAspect - sourceAspect) / Math.max(sourceAspect, 1e-12);
  return {
    scale: Math.max(Math.abs(scaleX - 1), Math.abs(scaleY - 1)) + 1,
    resizeRatio: Math.max(Math.abs(scaleX - 1), Math.abs(scaleY - 1)),
    geometryFraction,
    geometryLabel: "aspect deformation",
  };
}

function automaticFilter(nativeWidth, nativeHeight, targetWidth, targetHeight, policy) {
  if (nativeWidth === targetWidth && nativeHeight === targetHeight) return "No correction";
  if (policy === "crop_to_fill") {
    const scale = Math.max(targetWidth / nativeWidth, targetHeight / nativeHeight);
    return Math.abs(scale - 1) < 1e-12 ? "No resize" : scale < 1 ? "Area" : "Bicubic";
  }
  if (policy === "pad_to_fit") {
    const scale = Math.min(targetWidth / nativeWidth, targetHeight / nativeHeight);
    return Math.abs(scale - 1) < 1e-12 ? "No resize" : scale < 1 ? "Area" : "Bicubic";
  }
  const shrinking = nativeWidth >= targetWidth && nativeHeight >= targetHeight;
  const enlarging = nativeWidth <= targetWidth && nativeHeight <= targetHeight;
  return shrinking && !enlarging ? "Area" : "Bicubic";
}

export function updateHiresUpscalerPlanUI(plan = {}, enabled = false) {
  const item = selectedDescriptor();
  setTileControls(enabled, item);
  const policyControl = $("#hiresAspectPolicy");
  const paddingControl = $("#hiresPaddingMode");
  const filterControl = $("#hiresFinalSizeCorrectionFilter");
  const aspectChanged = Boolean(plan.aspect_ratio_changed);
  if (policyControl) policyControl.disabled = !enabled || !item || !aspectChanged;
  const policy = String(policyControl?.value || "stretch");
  if (paddingControl) paddingControl.disabled = !enabled || !item || !aspectChanged || policy !== "pad_to_fit";

  const nativeWidth = item ? Number(plan.base_width || 0) * Number(item.native_scale || 0) : 0;
  const nativeHeight = item ? Number(plan.base_height || 0) * Number(item.native_scale || 0) : 0;
  const targetWidth = Number(plan.internal_width || 0);
  const targetHeight = Number(plan.internal_height || 0);
  const correctionNeeded = nativeWidth !== targetWidth || nativeHeight !== targetHeight;
  if (filterControl) filterControl.disabled = !enabled || !item || !correctionNeeded;

  const nativeStatus = $("#hiresNativePlanStatus");
  if (nativeStatus) {
    if (!enabled) nativeStatus.textContent = "Native upscaler target plan is disabled.";
    else if (!item) nativeStatus.textContent = "Native upscaler target plan is unavailable until a supported model is selected.";
    else {
      const requested = `${plan.final_width} × ${plan.final_height}`;
      const correctionCanvas = `${targetWidth} × ${targetHeight}`;
      const filter = String(filterControl?.value || "auto");
      const resolvedFilter = filter === "auto" ? automaticFilter(nativeWidth, nativeHeight, targetWidth, targetHeight, policy) : filter;
      nativeStatus.textContent = `${item.display_name}: native x${item.native_scale} predicts ${nativeWidth} × ${nativeHeight} · correction canvas ${correctionCanvas} · final saved target ${requested} · ${policy.replaceAll("_", " ")} · filter ${filter === "auto" ? `Auto → ${resolvedFilter}` : resolvedFilter}.`;
    }
    nativeStatus.className = "field-status subtle";
  }

  const severityStatus = $("#hiresCorrectionSeverityStatus");
  if (severityStatus) {
    const preview = item ? correctionPreview(nativeWidth, nativeHeight, targetWidth, targetHeight, policy) : null;
    if (!enabled || !preview) {
      severityStatus.textContent = "Correction severity: unavailable.";
      severityStatus.className = "field-status subtle";
    } else {
      const resizeRatio = preview.resizeRatio ?? Math.abs(Number(preview.scale || 1) - 1);
      const score = Math.max(resizeRatio, Number(preview.geometryFraction || 0));
      severityStatus.textContent = `Correction severity: ${(score * 100).toFixed(1)}% · resize change ${(resizeRatio * 100).toFixed(1)}% · ${preview.geometryLabel} ${(Number(preview.geometryFraction || 0) * 100).toFixed(1)}%.`;
      severityStatus.className = score >= 0.5 ? "field-status warning" : "field-status subtle";
    }
  }
}

function renderOptions(preferred = "") {
  const select = $("#hiresUpscaler");
  if (!select) return;
  const previous = preferred || select.value;
  select.replaceChildren();
  const supported = catalog().supported_neural || [];
  if (!supported.length) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "No supported neural upscalers discovered";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);
  } else {
    supported.forEach((item) => addOption(select, item));
    select.value = supported.some((item) => item.upscaler_id === previous) ? previous : supported[0].upscaler_id;
  }
  const unavailable = catalog().unavailable_neural || [];
  if (unavailable.length) {
    const group = document.createElement("optgroup");
    group.label = "Unavailable (diagnostics only)";
    unavailable.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.upscaler_id;
      option.textContent = `${item.display_name} - ${friendlyAvailability(item)}`;
      option.disabled = true;
      group.appendChild(option);
    });
    select.appendChild(group);
  }
  renderStatus();
  renderDiagnostics();
}

export function initializeHiresUpscalers(payload = {}, current = {}) {
  state.upscalers = payload || {};
  const replayed = current.hires_enabled ? (current.hires_upscaler_id || current.hires_upscaler || "") : "";
  const preferred = replayed || state.settings?.preferred_hires_upscaler_id || $("#hiresUpscaler")?.value || "";
  if ($("#hiresStrategy")) $("#hiresStrategy").value = "pixel_neural";
  renderOptions(preferred);
}

export function bindHiresUpscalers(saveSessionSoon = null) {
  window.addEventListener("image-gen-hires-upscaler-error", (event) => {
    focusHiresUpscalerError(String(event.detail?.message || ""));
  });
  $("#hiresRecoveryRefreshButton")?.addEventListener("click", () => $("#refreshUpscalersButton")?.click());
  $("#hiresRecoveryConfigButton")?.addEventListener("click", () => {
    window.dispatchEvent(new CustomEvent("image-gen-open-user-config"));
  });
  $("#hiresUpscaler")?.addEventListener("change", async () => {
    clearHiresUpscalerError();
    renderStatus();
    const preferred = String($("#hiresUpscaler")?.value || "").trim();
    if (preferred && preferred !== String(state.settings?.preferred_hires_upscaler_id || "")) {
      try {
        const saved = await api.saveSettings({ preferred_hires_upscaler_id: preferred });
        state.settings = { ...state.settings, ...saved };
      } catch (error) {
        notify(`Could not save preferred hires upscaler: ${error.message}`, "warning");
      }
    }
    window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-change"));
    saveSessionSoon?.();
  });
  $("#upscalerFetchCivitaiButton")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const item = selectedDescriptor();
    if (!item?.upscaler_id) {
      notify("Select a supported neural upscaler before fetching CivitAI metadata.", "warning");
      return;
    }
    const previousTitle = button.title || "Refresh selected upscaler metadata from CivitAI";
    button.disabled = true;
    button.classList.add("is-working");
    button.title = "Fetching upscaler metadata from CivitAI…";
    try {
      const result = await api.enrichAssetFromCivitai("upscaler", item.upscaler_id, false);
      state.upscalers = result.catalog || await api.upscalers();
      renderOptions(item.upscaler_id);
      window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-change"));
      const lookup = result.asset?.civitai_lookup || {};
      notify(lookup.preview_image_downloaded
        ? "CivitAI metadata and a preview image were added to the upscaler sidecar."
        : "CivitAI metadata was added to the upscaler sidecar.");
    } catch (error) {
      notify(`Unable to fetch upscaler metadata from CivitAI: ${error.message}`, "error");
    } finally {
      button.classList.remove("is-working");
      button.title = previousTitle;
      button.disabled = !selectedDescriptor();
    }
  });
  $("#refreshUpscalersButton")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.classList.add("is-working");
    try {
      state.upscalers = await api.refreshUpscalers("all");
      renderOptions($("#hiresUpscaler")?.value || "");
      window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-change"));
      notify(`Upscaler catalog revision ${state.upscalers.catalog_revision} loaded.`);
    } catch (error) {
      notify(`Unable to refresh upscalers: ${error.message}`, "error");
    } finally {
      button.classList.remove("is-working");
      button.disabled = false;
    }
  });
}
