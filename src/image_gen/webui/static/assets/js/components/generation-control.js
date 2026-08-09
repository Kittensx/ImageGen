import { $ } from "../utils.js";

const DEFAULT_LABEL = "Generate";

function elements() {
  return {
    button: $("#generateButton"),
    label: $("#generateButtonLabel"),
    status: $("#generateButtonStatus"),
  };
}

function visualStateForPhase(phase, busy = false) {
  const token = String(phase || "idle").trim().toLowerCase();
  if (["running", "preparing_model", "warming_model", "paused", "finalizing", "cancelling"].includes(token)) {
    return "generating";
  }
  if (busy || ["validating", "validating_prompt", "validating_scheduler", "activating_model", "queueing", "submitting"].includes(token)) {
    return "submitting";
  }
  return "idle";
}

export function setGenerateControlState({ phase = "idle", label = DEFAULT_LABEL, status = "", busy = false } = {}) {
  const { button, label: labelNode, status: statusNode } = elements();
  if (!button || !labelNode || !statusNode) return;
  const visual = visualStateForPhase(phase, busy);
  button.dataset.generationState = String(phase || "idle");
  button.classList.toggle("is-generation-submitting", visual === "submitting");
  button.classList.toggle("is-generating", visual === "generating");
  button.setAttribute("aria-busy", busy ? "true" : "false");
  labelNode.textContent = label || DEFAULT_LABEL;
  const statusText = String(status || "").trim();
  statusNode.textContent = statusText;
  statusNode.hidden = !statusText;
  statusNode.setAttribute("aria-hidden", statusText ? "false" : "true");
}

export function resetGenerateControl() {
  setGenerateControlState({ phase: "idle", label: DEFAULT_LABEL, status: "", busy: false });
}
