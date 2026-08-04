import { $ } from "../utils.js";

export function bindWorkspaceTabs({ checkpointWorkspace, loraWorkspace } = {}) {
  const generation = $("#workspace");
  const checkpoints = $("#checkpointWorkspace");
  const loras = $("#loraWorkspace");
  const tabs = {
    generation: $("#generationWorkspaceTab"),
    checkpoints: $("#checkpointsWorkspaceTab"),
    loras: $("#lorasWorkspaceTab"),
  };

  const activate = (name) => {
    const target = ["generation", "checkpoints", "loras"].includes(name) ? name : "generation";
    generation?.classList.toggle("is-hidden", target !== "generation");
    checkpoints?.classList.toggle("is-hidden", target !== "checkpoints");
    loras?.classList.toggle("is-hidden", target !== "loras");
    Object.entries(tabs).forEach(([key, tab]) => {
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

  tabs.generation?.addEventListener("click", () => activate("generation"));
  tabs.checkpoints?.addEventListener("click", () => activate("checkpoints"));
  tabs.loras?.addEventListener("click", () => activate("loras"));

  return {
    activate,
    showGeneration: () => activate("generation"),
    showCheckpoints: () => activate("checkpoints"),
    showLoras: () => activate("loras"),
  };
}
