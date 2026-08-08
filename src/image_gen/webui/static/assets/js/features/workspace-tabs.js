import { $ } from "../utils.js";

const WORKSPACES = new Set(["home", "generation", "checkpoints", "loras"]);

/*
 * Historical filename retained for compatibility. Workspace switching is no
 * longer owned by the removed top tabs; the shared sidebar and Home quick
 * actions request workspace changes through image-gen-workspace-request.
 */
export function bindWorkspaceTabs({ checkpointWorkspace, loraWorkspace } = {}) {
  const home = $("#homeWorkspace");
  const generation = $("#workspace");
  const checkpoints = $("#checkpointWorkspace");
  const loras = $("#loraWorkspace");

  const legacyTabs = {
    generation: $("#generationWorkspaceTab"),
    checkpoints: $("#checkpointsWorkspaceTab"),
    loras: $("#lorasWorkspaceTab"),
  };

  const activate = (name) => {
    const target = WORKSPACES.has(name) ? name : "home";
    home?.classList.toggle("is-hidden", target !== "home");
    generation?.classList.toggle("is-hidden", target !== "generation");
    checkpoints?.classList.toggle("is-hidden", target !== "checkpoints");
    loras?.classList.toggle("is-hidden", target !== "loras");

    Object.entries(legacyTabs).forEach(([key, tab]) => {
      const active = key === target;
      tab?.classList.toggle("is-active", active);
      tab?.toggleAttribute("aria-current", active);
    });

    if (target === "checkpoints") checkpointWorkspace?.show?.();
    else checkpointWorkspace?.hide?.();
    if (target === "loras") loraWorkspace?.show?.();
    else loraWorkspace?.hide?.();

    document.body.dataset.activeWorkspace = target;
    window.dispatchEvent(new CustomEvent("image-gen-workspace-changed", { detail: { workspace: target } }));
  };

  legacyTabs.generation?.addEventListener("click", () => activate("generation"));
  legacyTabs.checkpoints?.addEventListener("click", () => activate("checkpoints"));
  legacyTabs.loras?.addEventListener("click", () => activate("loras"));

  window.addEventListener("image-gen-workspace-request", (event) => {
    const requested = String(event.detail?.workspace || "").trim().toLowerCase();
    if (WORKSPACES.has(requested)) activate(requested);
  });

  return {
    activate,
    showHome: () => activate("home"),
    showGeneration: () => activate("generation"),
    showCheckpoints: () => activate("checkpoints"),
    showLoras: () => activate("loras"),
  };
}
