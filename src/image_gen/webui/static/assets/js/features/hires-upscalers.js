import { api } from "../api.js?v=0.1.79";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

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

function renderStatus() {
  const status = $("#hiresUpscalerStatus");
  if (!status) return;
  const item = selectedDescriptor();
  if (!item) {
    status.textContent = "No supported neural .pth upscaler is selected. Refresh or install a recognized model.";
    status.className = "field-status error";
    return;
  }
  status.textContent = `${item.display_name} · ${item.architecture} · native x${item.native_scale} · ${friendlyAvailability(item)}`;
  status.className = item.selectable ? "field-status ready" : "field-status error";
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
  const recorded = current.hires_upscaler_id || current.hires_upscaler || "";
  if ($("#hiresStrategy")) $("#hiresStrategy").value = "pixel_neural";
  renderOptions(recorded);
}

export function bindHiresUpscalers(saveSessionSoon = null) {
  $("#hiresUpscaler")?.addEventListener("change", () => {
    renderStatus();
    window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-change"));
    saveSessionSoon?.();
  });
  $("#refreshUpscalersButton")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      state.upscalers = await api.refreshUpscalers("all");
      renderOptions($("#hiresUpscaler")?.value || "");
      window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-change"));
      notify(`Upscaler catalog revision ${state.upscalers.catalog_revision} loaded.`);
    } catch (error) {
      notify(`Unable to refresh upscalers: ${error.message}`, "error");
    } finally {
      button.disabled = false;
    }
  });
}
