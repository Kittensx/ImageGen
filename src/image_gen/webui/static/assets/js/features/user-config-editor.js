import { api } from "../api.js?v=asset-card-latency1";
import { $, notify } from "../utils.js";

function setStatus(message, kind = "subtle") {
  const node = $("#userConfigStatus");
  if (!node) return;
  node.textContent = message;
  node.className = `field-status ${kind}`;
}

async function loadUserConfig() {
  setStatus("Loading user-config.yml…");
  const data = await api.userConfig();
  if ($("#userConfigPath")) $("#userConfigPath").textContent = data.path || "user_config/user-config.yml";
  if ($("#userConfigText")) $("#userConfigText").value = data.text || "";
  setStatus(data.exists ? "Loaded. Changes require a full ImageGen restart when they affect startup configuration." : "The file does not exist yet. Saving will create it.", "ready");
  return data;
}

export function openUserConfigEditor() {
  const dialog = $("#userConfigDialog");
  if (!dialog) return;
  if (!dialog.open) dialog.showModal();
  loadUserConfig().catch((error) => setStatus(error.message, "error"));
}

export function bindUserConfigEditor() {
  $("#openUserConfigEditorButton")?.addEventListener("click", openUserConfigEditor);
  $("#reloadUserConfigButton")?.addEventListener("click", () => {
    loadUserConfig().catch((error) => setStatus(error.message, "error"));
  });
  $("#saveUserConfigButton")?.addEventListener("click", async () => {
    const button = $("#saveUserConfigButton");
    if (button) button.disabled = true;
    setStatus("Validating YAML…");
    try {
      const data = await api.saveUserConfig($("#userConfigText")?.value || "");
      const backup = data.backup_path ? " A backup of the previous file was created." : "";
      setStatus(`Saved successfully.${backup} Restart ImageGen to apply startup-only changes.`, "ready");
      notify("user-config.yml saved. Restart ImageGen if the edited setting is startup-only.");
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  });
  window.addEventListener("image-gen-open-user-config", openUserConfigEditor);
}
