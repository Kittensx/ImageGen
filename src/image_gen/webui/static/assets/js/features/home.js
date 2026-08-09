import { $ } from "../utils.js";

function requestWorkspace(workspace) {
  const target = String(workspace || "").trim().toLowerCase();
  if (!target) return;
  window.history.pushState(null, "", target === "home" ? "/" : `/#${target}`);
  window.dispatchEvent(new CustomEvent("image-gen-workspace-request", {
    detail: { workspace: target, source: "home" },
  }));
}

function requestSettings() {
  window.history.pushState(null, "", "/#settings");
  window.dispatchEvent(new CustomEvent("image-gen-open-settings", {
    detail: { source: "home" },
  }));
}

function count(value) {
  return Array.isArray(value) ? value.length : 0;
}

function renderReadiness(catalogs = {}) {
  const checkpointCount = count(catalogs.models);
  const vaeCount = count(catalogs.vaes);
  const loraCount = count(catalogs.loras);
  const hasCheckpoint = checkpointCount > 0;

  if ($("#homeCheckpointCount")) $("#homeCheckpointCount").textContent = String(checkpointCount);
  if ($("#homeVaeCount")) $("#homeVaeCount").textContent = String(vaeCount);
  if ($("#homeLoraCount")) $("#homeLoraCount").textContent = String(loraCount);

  const badge = $("#homeReadinessBadge");
  const title = $("#homeReadinessTitle");
  const message = $("#homeReadinessMessage");
  const primary = $("#homePrimaryAction");

  badge?.classList.toggle("is-ready", hasCheckpoint);
  badge?.classList.toggle("is-attention", !hasCheckpoint);

  if (hasCheckpoint) {
    if (badge) badge.textContent = "Ready";
    if (title) title.textContent = "Checkpoint assets are available";
    if (message) message.textContent = "IMAGE_GEN found at least one checkpoint. VAEs and LoRAs remain optional and can be added whenever your workflow needs them.";
    if (primary) {
      primary.textContent = "Open Generation";
      primary.dataset.homeTarget = "generation";
    }
  } else {
    if (badge) badge.textContent = "Needs checkpoint";
    if (title) title.textContent = "No checkpoint models found";
    if (message) message.textContent = "Start by adding or linking checkpoint assets. The Checkpoints workspace is the current place to review the local model catalog; guided starter-asset installation can be added here later.";
    if (primary) {
      primary.textContent = "Open Checkpoints";
      primary.dataset.homeTarget = "checkpoints";
    }
  }
}

export function bindHomeWorkspace(catalogs = {}) {
  renderReadiness(catalogs);

  $("#homePrimaryAction")?.addEventListener("click", (event) => {
    requestWorkspace(event.currentTarget.dataset.homeTarget || "generation");
  });
  $("#homeQuickGeneration")?.addEventListener("click", () => requestWorkspace("generation"));
  $("#homeQuickCheckpoints")?.addEventListener("click", () => requestWorkspace("checkpoints"));
  $("#homeQuickLoras")?.addEventListener("click", () => requestWorkspace("loras"));
  $("#homeQuickSettings")?.addEventListener("click", requestSettings);

  window.addEventListener("image-gen-asset-catalog-refreshed", (event) => {
    renderReadiness(event.detail || {});
  });

  return {
    refresh: renderReadiness,
  };
}
