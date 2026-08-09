import { api } from "../api.js?v=civitai-connect1";
import { productName } from "../branding.js?v=brand1";
import { $, notify } from "../utils.js";

const STATUS_ICON = Object.freeze({
  healthy: "/assets/icons/status/green.svg",
  transitioning: "/assets/icons/status/green.svg",
  warning: "/assets/icons/status/amber.svg",
  critical: "/assets/icons/status/red.svg",
  inactive: "/assets/icons/status/gray.svg",
});

let lastStatus = null;
let dialogReason = "";

function stateFromStatus(status = {}) {
  if (status.usable === true) return "healthy";
  if (status.configured === true) return "warning";
  return "inactive";
}

function statusLabel(status = {}) {
  if (status.usable === true) return "Credential configured";
  if (status.configured === true) return "Credential needs attention";
  return "Not connected";
}

function applyLight(element, state, label) {
  if (!element) return;
  const normalized = STATUS_ICON[state] ? state : "inactive";
  element.src = STATUS_ICON[normalized];
  element.dataset.status = normalized;
  element.alt = "";
  const owner = element.closest("button") || element;
  owner.title = `CivitAI: ${label}`;
  owner.setAttribute?.("aria-label", `CivitAI: ${label}`);
}

function renderStatus(status = {}, { overrideState = "", overrideMessage = "" } = {}) {
  lastStatus = { ...status };
  const state = overrideState || stateFromStatus(status);
  const label = overrideMessage || statusLabel(status);
  applyLight($("#civitaiSettingsStatusLight"), state, label);
  applyLight($("#civitaiDialogStatusLight"), state, label);

  const settingsStatus = $("#civitaiSettingsStatusText");
  if (settingsStatus) settingsStatus.textContent = label;
  const dialogStatus = $("#civitaiConnectionStatusText");
  if (dialogStatus) dialogStatus.textContent = label;
  const detail = $("#civitaiConnectionStatusDetail");
  if (detail) detail.textContent = overrideMessage || status.message || label;
  const path = $("#civitaiCredentialPath");
  if (path) path.textContent = status.key_file || "secrets/civitai_api_key.txt";

  const remove = $("#removeCivitaiCredentialButton");
  if (remove) remove.disabled = status.configured !== true || status.managed_by_ui === false;
  const test = $("#testCivitaiConnectionButton");
  if (test) test.disabled = status.usable !== true;
  const input = $("#civitaiApiKeyInput");
  const save = $("#saveCivitaiCredentialButton");
  const externallyManaged = status.managed_by_ui === false;
  if (input) {
    input.disabled = externallyManaged;
    input.placeholder = externallyManaged
      ? `${productName()} credential file is managed outside the project`
      : "Paste your CivitAI API key";
  }
  if (save) save.disabled = externallyManaged;
  const managedNote = $("#civitaiCredentialManagedNote");
  if (managedNote) {
    managedNote.hidden = !externallyManaged;
    managedNote.textContent = externallyManaged
      ? `${productName()} is configured to use a credential file outside the project. For safety, the WebUI will not overwrite that external file.`
      : "";
  }
}

async function refreshStatus() {
  try {
    const status = await api.civitaiConnectionStatus();
    renderStatus(status);
    return status;
  } catch (error) {
    renderStatus(lastStatus || {}, { overrideState: "critical", overrideMessage: "Unable to read CivitAI connection status." });
    console.warn("Unable to read CivitAI connection status", error);
    return null;
  }
}

export async function openCivitaiConnection({ reason = "" } = {}) {
  dialogReason = String(reason || "").trim();
  const dialog = $("#civitaiConnectionDialog");
  if (!dialog) return;
  const reasonNode = $("#civitaiConnectionReason");
  if (reasonNode) {
    reasonNode.hidden = !dialogReason;
    reasonNode.textContent = dialogReason || "";
  }
  await refreshStatus();
  if (!dialog.open) dialog.showModal();
  window.setTimeout(() => $("#civitaiApiKeyInput")?.focus(), 0);
}

async function saveCredential() {
  const input = $("#civitaiApiKeyInput");
  const button = $("#saveCivitaiCredentialButton");
  const apiKey = String(input?.value || "").trim();
  if (!apiKey) {
    notify("Paste your CivitAI API key before saving.", "warning");
    input?.focus();
    return;
  }
  renderStatus(lastStatus || {}, { overrideState: "transitioning", overrideMessage: "Saving CivitAI credential…" });
  if (button) button.disabled = true;
  try {
    const status = await api.saveCivitaiCredential(apiKey);
    if (input) input.value = "";
    renderStatus(status, { overrideState: "transitioning", overrideMessage: "Credential saved. Verifying CivitAI connection…" });
    try {
      await api.testCivitaiConnection();
      renderStatus(status, { overrideState: "healthy", overrideMessage: "Connected and verified." });
      notify(`CivitAI connected. The API key was saved to the local ${productName()} secrets folder.`, "success");
    } catch (error) {
      renderStatus(status, { overrideState: "warning", overrideMessage: "Credential saved, but CivitAI could not be verified right now." });
      notify(`CivitAI key saved, but connection verification failed: ${error.message}`, "warning");
    }
  } catch (error) {
    await refreshStatus();
    notify(error.message, "error");
  } finally {
    if (button) button.disabled = lastStatus?.managed_by_ui === false;
  }
}

async function removeCredential() {
  const button = $("#removeCivitaiCredentialButton");
  if (button) button.disabled = true;
  try {
    const status = await api.removeCivitaiCredential();
    renderStatus(status);
    notify(`CivitAI credential removed from this ${productName()} project.`);
  } catch (error) {
    notify(error.message, "error");
  } finally {
    await refreshStatus();
  }
}

async function testConnection() {
  const button = $("#testCivitaiConnectionButton");
  renderStatus(lastStatus || {}, { overrideState: "transitioning", overrideMessage: "Testing CivitAI connection…" });
  if (button) button.disabled = true;
  try {
    await api.testCivitaiConnection();
    renderStatus(lastStatus || {}, { overrideState: "healthy", overrideMessage: "Connected and verified." });
    notify("CivitAI connection verified.", "success");
  } catch (error) {
    renderStatus(lastStatus || {}, { overrideState: "warning", overrideMessage: "CivitAI connection test failed." });
    notify(error.message, "warning");
  } finally {
    if (button) button.disabled = lastStatus?.usable !== true;
  }
}

export function bindCivitaiConnection() {
  $("#openCivitaiConnectionButton")?.addEventListener("click", () => openCivitaiConnection());
  $("#civitaiSettingsStatusButton")?.addEventListener("click", () => openCivitaiConnection());
  $("#saveCivitaiCredentialButton")?.addEventListener("click", saveCredential);
  $("#removeCivitaiCredentialButton")?.addEventListener("click", removeCredential);
  $("#testCivitaiConnectionButton")?.addEventListener("click", testConnection);
  $("#civitaiApiKeyInput")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      saveCredential();
    }
  });
  window.addEventListener("image-gen-civitai-credential-required", (event) => {
    openCivitaiConnection({ reason: event.detail?.message || "This action requires a CivitAI API key." });
  });
  window.addEventListener("image-gen-settings-opened", refreshStatus);
  refreshStatus();
}
