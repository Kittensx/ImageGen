import { api } from "../api.js?v=0.1.77";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

function catalog() {
  return state.upscalers || {};
}

function selectedDescriptor(id) {
  return (catalog().neural || []).find((item) => item.upscaler_id === id) || null;
}

function addOption(select, item, { disabled = false } = {}) {
  const option = document.createElement("option");
  option.value = item.upscaler_id;
  option.textContent = item.display_name || item.upscaler_id;
  option.disabled = disabled;
  select.appendChild(option);
}

function renderDiagnostics() {
  const node = $("#hiresUpscalerDiagnosticsText");
  if (!node) return;
  const values = [];
  (catalog().unavailable_neural || []).forEach((item) => {
    values.push(`${item.file_name || item.display_name}: ${item.load_status || "unavailable"}${item.bounded_error ? ` - ${item.bounded_error}` : ""}`);
  });
  (catalog().diagnostics || []).forEach((item) => {
    values.push(`${item.severity || "info"}: ${item.code || "discovery"} - ${item.message || ""}${item.path ? ` (${item.path})` : ""}`);
  });
  node.textContent = values.length ? values.join("\n") : "No unsupported upscaler files were reported.";
}

function renderStatus() {
  const status = $("#hiresUpscalerStatus");
  const selected = $("#hiresUpscaler")?.value || "";
  if (!status) return;
  const item = selectedDescriptor(selected);
  if (!item) {
    status.textContent = "No supported neural .pth upscaler is selected. Refresh or install a recognized model.";
    status.className = "field-status error";
    return;
  }
  const qualification = item.runtime_qualification || {};
  status.textContent = `${item.display_name} · ${item.architecture} · x${item.native_scale} · discovery ${item.load_status} · runtime ${qualification.status || "unqualified"}.`;
  status.className = item.selectable ? "field-status ready" : "field-status error";
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
    select.value = supported.some((item) => item.upscaler_id === previous)
      ? previous
      : supported[0].upscaler_id;
  }
  const unavailable = catalog().unavailable_neural || [];
  if (unavailable.length) {
    const group = document.createElement("optgroup");
    group.label = "Unavailable (diagnostics only)";
    unavailable.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.upscaler_id;
      option.textContent = `${item.display_name} - ${item.load_status}`;
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
    saveSessionSoon?.();
  });
  $("#refreshUpscalersButton")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      state.upscalers = await api.refreshUpscalers("all");
      renderOptions($("#hiresUpscaler")?.value || "");
      notify(`Upscaler catalog revision ${state.upscalers.catalog_revision} loaded.`);
    } catch (error) {
      notify(`Unable to refresh upscalers: ${error.message}`, "error");
    } finally {
      button.disabled = false;
    }
  });
}
