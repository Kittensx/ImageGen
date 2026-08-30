import { api } from "../api.js?v=cancel-controls1";
import { state } from "../state.js";
import { $, shortText, formatTime, notify } from "../utils.js";
import {
  bindLivePreview,
  renderLivePreviewJob,
  setLivePreviewTransport,
  stopLivePreview,
} from "./live-preview.js?v=sdxl-cfg-recommendations4";
import { showOutput, upsertRecentOutput } from "./gallery.js?v=0.1.46";
import { openOutputDetailsData } from "./output-details.js";
import { preflightCurrentPrompt } from "./prompt-tools.js?v=r10.3";
import { resetGenerateControl, setGenerateControlState } from "../components/generation-control.js?v=0.1.0";
import { setSubsystemStatus } from "../components/status-indicators.js?v=1";
import { setActionIcon } from "../components/action-icons.js?v=0.1.0";
import { generationCapabilityBlocksSubmission, refreshGenerationCapabilities } from "./generation-capabilities.js";
import { enforceCfgRescaleRequestGuardrails } from "./cfg-lab.js?v=0.1.47-lightning-recommendation";

let refreshOutputs = async () => {};
let collectValues = () => ({});
let ensureModelReady = async () => null;
let pollTimer = null;
let recentOutputsPollTimer = null;
let postCompletionRefreshTimer = null;
let pollJobsPromise = null;
let submitInFlight = false;
let eventStream = null;
let eventStreamJobId = null;
const shownOutputMetadataWarnings = new Set();
let eventStreamFailures = 0;
let eventStreamDisabledUntil = 0;
let workerState = {};

function setSubmissionBusy(active, phase = "submitting", stage = "") {
  const capabilityBlocked = generationCapabilityBlocksSubmission();
  const button = $("#generateButton");
  if (button) button.disabled = active || capabilityBlocked;
  if (active) {
    setGenerateControlState({ phase, status: stage || "Queueing…", busy: true });
  } else {
    resetGenerateControl();
  }
  ["#generateMenuButton", "#infinityButton"].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = active || capabilityBlocked;
  });
}

function recentOutputsRefreshDelay() {
  const active = Boolean(activeJob());
  const enabled = state.settings.recent_outputs_background_refresh_enabled !== false;
  if (!enabled) return null;
  const base = active
    ? Number(state.settings.recent_outputs_refresh_ms_active || 4000)
    : Number(state.settings.recent_outputs_refresh_ms_idle || 12000);
  if (document.hidden) return Math.max(base, 20000);
  return base;
}

function scheduleRecentOutputsPolling() {
  if (recentOutputsPollTimer) window.clearTimeout(recentOutputsPollTimer);
  const delay = recentOutputsRefreshDelay();
  if (!delay) return;
  recentOutputsPollTimer = window.setTimeout(async () => {
    await refreshOutputs({ preserveSelection: true });
    scheduleRecentOutputsPolling();
  }, delay);
}

const ACTIVE_JOB_STATUSES = new Set(["preparing_model", "warming_model", "running", "paused", "finalizing", "cancelling"]);
const CANCELLABLE_JOB_STATUSES = new Set(["preparing_model", "warming_model", "running", "paused"]);

function isTerminalStatus(status) {
  return ["completed", "cancelled", "failed"].includes(String(status || ""));
}

function queueFilterStatus(job) {
  const status = String(job?.status || "");
  return ["preparing_model", "warming_model", "paused", "finalizing"].includes(status) ? "running" : status;
}

function humanizeStage(value) {
  const token = String(value || "idle").replaceAll("_", " ");
  return token.charAt(0).toUpperCase() + token.slice(1);
}

function canRemoveQueueJob(job) {
  return isTerminalStatus(job?.status);
}

function visibleQueueJobs() {
  const allowed = new Set(state.queueFilters || []);
  return (state.jobs || []).filter((job) => !state.hiddenQueueJobIds.has(job.job_id) && allowed.has(queueFilterStatus(job)));
}

function autoHideCompletedJobs() {
  if (!state.settings.queue_auto_remove_completed) return;
  (state.jobs || []).forEach((job) => {
    if (job?.status === "completed") state.hiddenQueueJobIds.add(job.job_id);
  });
}

function dismissQueueJob(jobId) {
  if (!jobId) return;
  state.hiddenQueueJobIds.add(jobId);
}

function refreshQueueToolbar() {
  const filters = new Set(state.queueFilters || []);
  document.querySelectorAll("[data-queue-filter]").forEach((input) => {
    input.checked = filters.has(input.dataset.queueFilter);
  });
  const visible = visibleQueueJobs();
  const completed = visible.filter((job) => job.status === "completed").length;
  const removable = (state.jobs || []).filter((job) => !state.hiddenQueueJobIds.has(job.job_id) && canRemoveQueueJob(job)).length;
  const active = visible.filter((job) => job.status === "queued" || ACTIVE_JOB_STATUSES.has(String(job.status || ""))).length;
  const summary = [];
  if (active) summary.push(`${active} active`);
  if (completed) summary.push(`${completed} completed`);
  if (removable - completed) summary.push(`${removable - completed} removable`);
  const summaryNode = $("#queueFilterSummary");
  if (summaryNode) summaryNode.textContent = summary.join(" · ") || "No queue items visible.";
  const clearCompletedButton = $("#clearCompletedQueueButton");
  if (clearCompletedButton) clearCompletedButton.disabled = removable === 0;
}

async function viewQueueJob(job) {
  if (!job?.job_id) return;
  const candidateId = (state.recentOutputs || []).find((item) =>
    (job.output_paths || []).some((path) => String(path).endsWith(String(item.name || ""))),
  )?.output_id;
  if (candidateId) {
    const button = document.querySelector(`[data-gallery-output-id="${CSS.escape(candidateId)}"] .thumbnail-button`);
    if (button) {
      button.click();
      return;
    }
  }
  try {
    const output = await api.jobPrimaryOutput(job.job_id);
    if (output) showOutput(output);
    return;
  } catch (error) {
    const message = String(error?.message || "");
    if (message.includes("No generated output is available") || message.includes("404")) {
      await refreshOutputs({ preserveSelection: true });
      notify("This queue item no longer has an image in the output folder.", "warning");
      return;
    }
    notify(message || "Unable to open the queued output.", "error");
  }
}


function decorateQueueAction(control, { icon, label }) {
  if (!control) return control;
  control.classList.add("queue-action-button");
  setActionIcon(control, icon, { label, title: label, replace: true });
  return control;
}

function concreteQueueSeed(job) {
  const candidates = [
    job?.resolved_seed,
    ...(Array.isArray(job?.resolved_seeds) ? job.resolved_seeds : []),
    job?.request?.seed,
  ];
  for (const value of candidates) {
    if (value === null || value === undefined || value === "") continue;
    const numeric = Number(value);
    if (Number.isSafeInteger(numeric) && numeric >= 0) return numeric;
  }
  return null;
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(String(text));
    return;
  }
  const fallback = document.createElement("textarea");
  fallback.value = String(text);
  fallback.setAttribute("readonly", "");
  fallback.style.position = "fixed";
  fallback.style.opacity = "0";
  document.body.append(fallback);
  fallback.select();
  const copied = document.execCommand("copy");
  fallback.remove();
  if (!copied) throw new Error("Clipboard copy was not available.");
}

async function copyQueueSeed(job) {
  const seed = concreteQueueSeed(job);
  if (seed === null) {
    notify("A concrete seed is not available for this queue item yet.", "warning");
    return;
  }
  try {
    await copyTextToClipboard(seed);
    notify(`Seed ${seed} copied.`);
  } catch (error) {
    notify(error.message || "Unable to copy the seed.", "error");
  }
}

function activeJob() {
  const workerActiveId = String(workerState?.active_job_id || "");
  if (workerActiveId) {
    const workerActive = state.jobs.find((job) => String(job.job_id || "") === workerActiveId);
    if (workerActive) return workerActive;
  }
  return state.jobs.find((job) => {
    const status = String(job.status || "");
    return ACTIVE_JOB_STATUSES.has(status) && status !== "paused";
  }) || null;
}

function monitoredJob() {
  const running = activeJob();
  if (running) return running;
  if (!state.activeJobId) return null;
  return state.jobs.find((job) => job.job_id === state.activeJobId) || null;
}

function renderQueueSubsystemStatus() {
  const online = Boolean(workerState?.online);
  const queued = (state.jobs || []).filter((job) => job.status === "queued").length;
  const active = (state.jobs || []).filter((job) => {
    const status = String(job.status || "");
    return ACTIVE_JOB_STATUSES.has(status) && status !== "paused";
  }).length;
  const failed = (state.jobs || []).filter((job) => job.status === "failed").length;
  const paused = Boolean(workerState?.queue_pause_requested || workerState?.queue_paused);
  let status = "healthy";
  let stateLabel = "Ready";
  let summary = "Queue is accepting generation work.";
  if (!online) {
    status = "inactive";
    stateLabel = "Offline";
    summary = "Queue worker is offline.";
  } else if (paused) {
    status = "warning";
    stateLabel = workerState?.queue_paused ? "Paused" : "Pause pending";
    summary = "Queue processing is intentionally paused or waiting to pause.";
  } else if (active || queued) {
    status = "transitioning";
    stateLabel = active ? "Processing" : "Queued";
    summary = active ? "Queue is actively processing generation work." : "Generation work is waiting in the queue.";
  } else if (failed) {
    status = "warning";
    stateLabel = "Attention";
    summary = `${failed} failed queue entr${failed === 1 ? "y" : "ies"} remain in this session.`;
  }
  setSubsystemStatus({
    id: "queueSubsystemStatusLight",
    host: "#queueStatusLightHost",
    label: "Generation queue",
    status,
    stateLabel,
    summary,
    detail: `Worker online: ${online ? "yes" : "no"}. Active: ${active}. Queued: ${queued}. Failed: ${failed}.`,
    facts: { active, queued, failed, pause_requested: paused ? "yes" : "no" },
    diagnosticTarget: "#queuePanel",
  });
}

function formatResidencyBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 GiB";
  return `${(bytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function modelResidencyDisplay(runtime = {}) {
  const stateValue = String(runtime.residency_state_effective || "empty").trim().toLowerCase();
  const stage = String(runtime.stage || "").trim().toLowerCase();
  if (["preparing_model", "loading_tokenizer", "loading_checkpoint", "applying_retention_policy"].includes(stage)) return "LOADING";
  if (["unloading", "superseded"].includes(stage) || stateValue === "switching") return "SWITCHING";
  if (stage === "recovering" || stateValue === "recovering") return "RECOVERING";
  if (stateValue === "hot_gpu") return "HOT - GPU";
  if (stateValue === "hot_staged") return "HOT - STAGED";
  if (stateValue === "managed_resident") return "MANAGED - RESIDENT";
  return "UNLOADED";
}

function formatResidencyMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : "—";
}

function residencyPerformanceReport(runtime = {}) {
  const reports = Array.isArray(runtime.recent_job_reports) ? runtime.recent_job_reports.slice(-5) : [];
  const transitions = Array.isArray(runtime.residency_transition_history) ? runtime.residency_transition_history.slice(-8) : [];
  const lines = [];
  lines.push(`Requested policy: ${runtime.residency_mode_requested || "managed"}`);
  lines.push(`Effective state: ${modelResidencyDisplay(runtime)}`);
  lines.push(`Current reason: ${runtime.last_residency_reason || "none"}`);
  lines.push(`Hot reuse count: ${Number(runtime.hot_reuse_count || 0)}`);
  lines.push("");
  if (!reports.length) {
    lines.push("No completed resident-runtime generation report has been recorded yet.");
  } else {
    lines.push("Recent generations (newest last):");
    reports.forEach((report, index) => {
      const timings = report.timings || {};
      const modelName = String(report.model_path || "").split(/[\\/]/).pop() || "unknown";
      lines.push(`${index + 1}. ${modelName} · ${report.generation_residency_classification || "unknown"} · ${report.residency_state_effective || "unknown"}`);
      lines.push(`   total=${formatResidencyMs(timings.last_job_total_ms)} setup=${formatResidencyMs(timings.request_setup_time_ms)} generation=${formatResidencyMs(timings.generation_execution_time_ms)} residency=${formatResidencyMs(timings.post_generation_residency_time_ms)} save-wait=${formatResidencyMs(timings.output_save_wait_time_ms)} finalize=${formatResidencyMs(timings.post_generation_finalize_time_ms)}`);
      lines.push(`   checkpoint-hydration=${formatResidencyMs(timings.checkpoint_hydration_time_ms)} cpu->gpu=${formatResidencyMs(timings.cpu_to_gpu_promotion_time_ms)} first-step=${formatResidencyMs(timings.first_step_latency_ms)}`);
      lines.push(`   policy=${report.memory_policy || "unknown"} retention-device=${report.retention_device_policy || "unknown"} execution-device=${report.execution_device_policy || "unknown"}`);
      lines.push(`   change=${report.resident_change_classification || "unknown"} retention=${report.post_job_residency_action || "unknown"} reason=${report.degradation_reason || report.staging_reason || report.residency_reason || "none"}`);
    });
  }
  if (transitions.length) {
    lines.push("");
    lines.push("Recent residency transitions:");
    transitions.forEach((item) => {
      lines.push(`- ${item.from || "?"} -> ${item.to || "?"}: ${item.reason || "unspecified"}`);
    });
  }
  return lines.join("\n");
}

function renderModelResidencyStatus(runtime = {}) {
  if (state.bootstrap) state.bootstrap.model_runtime = runtime;
  const label = modelResidencyDisplay(runtime);
  const stage = String(runtime.stage || (runtime.online ? "idle" : "offline"));
  const modelName = String(runtime.current_model_path || runtime.selected_model_path || "").split(/[\\/]/).pop() || "none";
  const isTransition = ["LOADING", "SWITCHING", "RECOVERING"].includes(label);
  const active = ["HOT - GPU", "HOT - STAGED", "MANAGED - RESIDENT"].includes(label);
  const indicatorStatus = runtime.last_error ? "critical" : isTransition ? "transitioning" : active ? "healthy" : "inactive";
  setSubsystemStatus({
    id: "modelResidencyStatusLight",
    host: "#workerStatusLights",
    label: "Model residency",
    status: indicatorStatus,
    stateLabel: runtime.last_error ? "Error" : label,
    summary: runtime.last_error ? "Resident model runtime reported an error." : `${label}${modelName !== "none" ? ` · ${modelName}` : ""}`,
    detail: runtime.last_error || `Requested policy: ${runtime.residency_mode_requested || "managed"}. Stage: ${stage}.`,
    facts: {
      policy: runtime.residency_mode_requested || "managed",
      residency: label,
      model: modelName,
      hot_reuse_count: Number(runtime.hot_reuse_count || 0),
      last_generation: runtime.last_generation_residency_classification || "none",
    },
    diagnosticTarget: "#runtimeStatusPanel",
  });

  const memory = runtime.memory || {};
  const allocated = formatResidencyBytes(memory.allocated_bytes);
  const free = memory.free_bytes == null ? "unknown" : formatResidencyBytes(memory.free_bytes);
  const devices = Object.entries(runtime.component_devices || {}).map(([name, device]) => `${name}: ${device}`).join(" · ");
  const activationMs = Number(runtime.timings?.activate_time_ms ?? runtime.timings?.initial_activation_time_ms);
  const activationText = Number.isFinite(activationMs) ? `${activationMs.toFixed(0)} ms` : "—";
  const reuseCount = Number(runtime.hot_reuse_count || 0);
  const lastReuse = String(runtime.last_generation_residency_classification || "none").replaceAll("_", " ");

  if ($("#runtimeModelResidencyStatus")) $("#runtimeModelResidencyStatus").textContent = label;
  if ($("#runtimeResidentModelStatus")) $("#runtimeResidentModelStatus").textContent = modelName === "none" ? "None" : `${modelName}${devices ? ` · ${devices}` : ""}`;
  if ($("#runtimeModelMemoryStatus")) $("#runtimeModelMemoryStatus").textContent = `allocated ${allocated} · free ${free}`;
  if ($("#runtimeModelActivationStatus")) $("#runtimeModelActivationStatus").textContent = activationText;
  if ($("#runtimeHotReuseStatus")) $("#runtimeHotReuseStatus").textContent = `${reuseCount} reuse${reuseCount === 1 ? "" : "s"} · last: ${lastReuse}`;
  if ($("#runtimeResidencyReport")) $("#runtimeResidencyReport").textContent = residencyPerformanceReport(runtime);

  window.dispatchEvent(new CustomEvent("image-gen-model-runtime-status", { detail: runtime }));
}

function renderWorker(worker) {
  workerState = worker || {};
  const online = Boolean(worker?.online);
  const pauseLabel = worker?.queue_paused
    ? "Paused"
    : worker?.queue_pause_requested
      ? "Pause pending"
      : "";
  const pausedCount = Number(worker?.paused || 0);
  const queueSummary = `${Number(worker?.queued || 0)} queued${pausedCount ? ` · ${pausedCount} paused` : ""}`;
  $("#workerText").textContent = online
    ? pauseLabel
      ? `${pauseLabel} · ${queueSummary}`
      : worker.active_job_id ? `Active · ${queueSummary}` : `Online · ${queueSummary}`
    : "Offline";
  $("#footerWorkerStatus").textContent = `Worker: ${online ? (pauseLabel || "online").toLowerCase() : "offline"}`;
  $("#workerPill").classList.toggle("is-offline", !online);
  setSubsystemStatus({
    id: "workerConnectionStatusLight",
    host: "#workerStatusLights",
    label: "Worker connection",
    status: online ? "healthy" : "inactive",
    stateLabel: online ? "Online" : "Offline",
    summary: online ? "Generation worker is online." : "Generation worker is offline.",
    detail: online
      ? `${worker.active_job_id ? "An active job is running." : "No active job."} ${Number(worker.queued || 0)} queued.`
      : "The WebUI is not currently connected to an active generation worker.",
    facts: {
      online: online ? "yes" : "no",
      active_job: worker.active_job_id || "none",
      queued: Number(worker.queued || 0),
      queue_paused: worker.queue_paused ? "yes" : "no",
    },
    diagnosticTarget: "#runtimeStatusPanel",
    placement: "prepend",
  });
  renderModelResidencyStatus(worker?.model_runtime || {});
  renderQueueSubsystemStatus();
}

function jobCanBeCancelled(job) {
  const status = String(job?.status || "");
  return status === "queued" || CANCELLABLE_JOB_STATUSES.has(status);
}

async function cancelQueueJob(job, { confirmUser = true } = {}) {
  if (!job?.job_id || !jobCanBeCancelled(job) || job.status === "cancelling") return;
  const target = job.status === "queued" ? "queued generation" : "active generation";
  if (confirmUser && !window.confirm(`Cancel this ${target}?`)) return;
  try {
    await api.cancelJob(job.job_id);
    notify(job.status === "queued" ? "Queued generation cancelled." : "Cancellation requested.");
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

function renderQueue(jobs) {
  state.jobs = jobs || [];
  autoHideCompletedJobs();
  const list = $("#queueList");
  list.replaceChildren();
  const visibleJobs = visibleQueueJobs();
  refreshQueueToolbar();
  renderQueueSubsystemStatus();
  if (!visibleJobs.length) {
    list.className = "stack-list empty-state";
    list.textContent = state.jobs.length ? "No queue items match the current filter." : "No queued generations.";
    return;
  }

  list.className = "stack-list";
  visibleJobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "queue-card";
    card.dataset.jobId = job.job_id;
    card.tabIndex = 0;
    const header = document.createElement("header");
    const title = document.createElement("strong");
    const queueTitle = shortText(job.request?.positive_prompt || "Untitled generation", 38);
    title.textContent = job.status === "paused" ? `|| ${queueTitle}` : queueTitle;
    const badge = document.createElement("span");
    badge.className = `status-badge ${job.status}`;
    badge.textContent = humanizeStage(job.worker_stage || job.status);
    header.append(title, badge);

    const settings = document.createElement("p");
    const runtimeModel = job.model_diagnostics?.runtime?.file_name;
    const activeModel = job.model_selection?.model_name ? `${job.model_selection.model_name}${job.model_selection.extension || ""}` : "";
    const requestModel = String(job.request?.model_path || "model").split(/[\\/]/).pop();
    const selectedModel = runtimeModel || activeModel || requestModel;
    const schedulerLabel = job.scheduler_name || job.request?.scheduler_name || "scheduler";
    const presetLabel = job.scheduler_preset_name ? ` · preset ${job.scheduler_preset_name}` : "";
    const warningCount = Number(job.scheduler_validation_warning_count || 0);
    const warningLabel = warningCount ? ` · ${warningCount} setting warning${warningCount === 1 ? "" : "s"}` : "";
    const executionLabel = job.execution_mode ? ` · ${String(job.execution_mode).replaceAll("_", " ")}` : "";
    const parserLabel = job.prompt_preflight?.base?.parser?.label || job.request?.prompt_parser_name || "legacy";
    const profileLabel = job.prompt_preflight?.base?.shortcut_profile?.label || job.request?.prompt_shortcut_profile_name || "profile";
    const promptValidationLabel = job.prompt_preflight?.valid === false ? " · prompt invalid" : " · prompt validated";
    settings.textContent = `${job.request?.width || "?"}×${job.request?.height || "?"} · ${job.request?.steps || "?"} steps · ${job.request?.sampler_name || "sampler"} · ${schedulerLabel}${presetLabel}${warningLabel} · ${parserLabel} / ${profileLabel}${promptValidationLabel} · ${selectedModel}${executionLabel}`;
    const timing = document.createElement("small");
    timing.textContent = job.completed_at ? `Finished ${formatTime(job.completed_at)}` : `Started ${formatTime(job.started_at || job.created_at)}`;
    card.append(header, settings, timing);
    const orchestration = job.model_runtime_diagnostics?.batch_orchestration || {};
    const completedProgress = Math.max(0, Number(orchestration.completed_images ?? job.resume_completed_images ?? 0));
    const attemptedProgress = Math.max(completedProgress, Number(orchestration.attempted_images ?? job.resume_image_index ?? 0));
    const unlimitedProgress = Boolean(job.request?.unlimited || orchestration.mode === "unlimited");
    const finiteProgressTotal = Math.max(
      1,
      Number(orchestration.requested_image_count || 0)
        || (Math.max(1, Number(job.request?.batch_count || 1)) * Math.max(1, Number(job.request?.batch_size || 1))),
    );
    if (unlimitedProgress || finiteProgressTotal > 1) {
      const progress = document.createElement("small");
      progress.className = "queue-batch-progress";
      progress.textContent = `${attemptedProgress} of ${unlimitedProgress ? "∞" : finiteProgressTotal}`;
      card.append(progress);
    }
    const pendingSaves = Number(job.pending_save_batches || 0);
    const completedSaves = Number(job.completed_save_batches || 0);
    const failedSaves = Number(job.failed_save_batches || 0);
    if (pendingSaves > 0 || completedSaves > 0 || failedSaves > 0) {
      const saveStatus = document.createElement("small");
      saveStatus.className = "queue-diagnostic-path";
      const saveParts = [];
      if (pendingSaves > 0) saveParts.push(`${pendingSaves} pending save${pendingSaves === 1 ? "" : "s"}`);
      if (completedSaves > 0) saveParts.push(`${completedSaves} completed`);
      if (failedSaves > 0) saveParts.push(`${failedSaves} failed`);
      saveStatus.textContent = `Output save queue: ${saveParts.join(" · ")}`;
      card.append(saveStatus);
    }
    if (job.pause_after_current_requested || job.status === "paused") {
      const pauseStatus = document.createElement("small");
      pauseStatus.className = "queue-diagnostic-path";
      pauseStatus.textContent = job.status === "paused"
        ? "Paused. Other queued batches may continue."
        : "This batch will pause after the current image finishes.";
      card.append(pauseStatus);
    }
    if (Number(job.skipped_images || 0) > 0) {
      const skipStatus = document.createElement("small");
      skipStatus.className = "queue-diagnostic-path";
      skipStatus.textContent = `${job.skipped_images} image${Number(job.skipped_images) === 1 ? "" : "s"} skipped.`;
      card.append(skipStatus);
    }
    const outputQuality = job.output_quality_diagnostics || {};
    if (outputQuality.suspect) {
      const qualityWarning = document.createElement("p");
      qualityWarning.className = "queue-output-quality-warning";
      const classification = String(outputQuality.classification || "near-uniform output").replaceAll("_", " ");
      const artifact = outputQuality.artifact_path ? ` Diagnostic: ${outputQuality.artifact_path}` : "";
      qualityWarning.textContent = `Output warning: ${classification}.${artifact}`;
      card.append(qualityWarning);
      card.classList.add("has-output-quality-warning");
    } else if (outputQuality.artifact_path) {
      const diagnosticNote = document.createElement("small");
      diagnosticNote.className = "queue-diagnostic-path";
      diagnosticNote.textContent = `Diagnostics: ${outputQuality.artifact_path}`;
      diagnosticNote.title = outputQuality.artifact_path;
      card.append(diagnosticNote);
    }

    const actions = document.createElement("div");
    actions.className = "queue-actions";
    if (["queued", "running", "preparing_model", "warming_model", "paused"].includes(String(job.status || ""))) {
      const pauseResumeButton = document.createElement("button");
      pauseResumeButton.type = "button";
      pauseResumeButton.className = "small-button queue-pause-resume-button";
      const isPaused = job.status === "paused";
      const pauseRequested = Boolean(job.pause_after_current_requested);
      decorateQueueAction(pauseResumeButton, {
        icon: isPaused ? "play" : "pause",
        label: isPaused ? "Play / requeue batch" : pauseRequested ? "Pause pending" : "Pause batch",
      });
      pauseResumeButton.disabled = pauseRequested;
      pauseResumeButton.addEventListener("click", async () => {
        try {
          const response = isPaused ? await api.resumeJob(job.job_id) : await api.pauseJob(job.job_id);
          renderWorker(response.worker || workerState);
          renderQueue(response.jobs || state.jobs);
          updateActiveState();
        } catch (error) {
          notify(error.message, "error");
        }
      });
      actions.append(pauseResumeButton);
    }
    if (job.status === "queued") {
      for (const [direction, icon, label] of [["up", "chevron-up", "Move higher"], ["down", "chevron-down", "Move lower"]]) {
        const moveButton = document.createElement("button");
        moveButton.type = "button";
        moveButton.className = "small-button queue-reorder-button";
        decorateQueueAction(moveButton, { icon, label });
        moveButton.addEventListener("click", async () => {
          try {
            const response = await api.reorderJob(job.job_id, direction);
            renderWorker(response.worker || workerState);
            renderQueue(response.jobs || state.jobs);
          } catch (error) {
            notify(error.message, "error");
          }
        });
        actions.append(moveButton);
      }
    }
    if (jobCanBeCancelled(job)) {
      const cancelButton = document.createElement("button");
      cancelButton.type = "button";
      cancelButton.className = "small-button queue-cancel-button";
      cancelButton.disabled = job.status === "cancelling";
      const cancelLabel = job.status === "queued"
        ? "Cancel queued item"
        : job.status === "cancelling" ? "Cancelling…" : "Cancel generation";
      decorateQueueAction(cancelButton, { icon: "stop", label: cancelLabel });
      cancelButton.classList.toggle("is-working", job.status === "cancelling");
      cancelButton.addEventListener("click", () => cancelQueueJob(job));
      actions.append(cancelButton);
      card.classList.add("is-cancellable");
      card.title = "Right-click this queue item to cancel it.";
      card.addEventListener("contextmenu", (event) => {
        if (event.target.closest("a, button, input, select, textarea")) return;
        event.preventDefault();
        cancelQueueJob(job);
      });
    }
    if ((job.output_paths || []).length) {
      const viewButton = document.createElement("button");
      viewButton.type = "button";
      viewButton.className = "small-button";
      decorateQueueAction(viewButton, { icon: "image-view", label: "View output" });
      viewButton.addEventListener("click", () => viewQueueJob(job));
      actions.append(viewButton);
    }
    const seedButton = document.createElement("button");
    seedButton.type = "button";
    seedButton.className = "small-button";
    const copySeedValue = concreteQueueSeed(job);
    decorateQueueAction(seedButton, {
      icon: "seed",
      label: copySeedValue === null ? "Seed unavailable" : "Copy seed",
    });
    seedButton.disabled = copySeedValue === null;
    seedButton.title = copySeedValue === null
      ? "A concrete seed will be available after the job resolves its seed."
      : `Copy seed ${copySeedValue}`;
    seedButton.addEventListener("click", () => copyQueueSeed(job));
    actions.append(seedButton);
    const diagnosticsLink = document.createElement("a");
    diagnosticsLink.className = "small-button queue-log-link";
    diagnosticsLink.href = `/api/jobs/${encodeURIComponent(job.job_id)}/diagnostics`;
    diagnosticsLink.target = "_blank";
    diagnosticsLink.rel = "noopener";
    decorateQueueAction(diagnosticsLink, { icon: "diagnostics", label: "Run diagnostics" });
    actions.append(diagnosticsLink);
    if (job.console_log_path) {
      const logLink = document.createElement("a");
      logLink.className = "small-button queue-log-link";
      logLink.href = `/api/jobs/${encodeURIComponent(job.job_id)}/log`;
      logLink.target = "_blank";
      logLink.rel = "noopener";
      decorateQueueAction(logLink, { icon: "console-log", label: "Console log" });
      actions.append(logLink);
    }
    if (canRemoveQueueJob(job)) {
      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.className = "small-button";
      decorateQueueAction(removeButton, { icon: "remove", label: "Remove" });
      removeButton.addEventListener("click", () => {
        dismissQueueJob(job.job_id);
        renderQueue(state.jobs);
      });
      actions.append(removeButton);
    }
    card.append(actions);

    if (job.error) {
      const error = document.createElement("small");
      error.className = "queue-error";
      const failurePrefix = job.failure_stage_label
        ? (job.failure_stage_domain === "outpaint"
          ? `Outpaint failed during ${job.failure_stage_label}: `
          : `Hires failed during ${job.failure_stage_label}: `)
        : "";
      error.textContent = shortText(`${failurePrefix}${job.error}`, 140);
      error.title = `${failurePrefix}${job.error}`;
      card.append(error);

      if (job.failure_bundle_path) {
        const bundle = document.createElement("span");
        bundle.className = "failure-bundle-hint";
        bundle.textContent = "Failure bundle saved";
        bundle.title = job.failure_bundle_path;
        actions.append(bundle);
      }
    }

    if ((job.output_paths || []).length) {
      card.addEventListener("click", (event) => {
        if (event.target.closest("a, button, input, select, textarea")) return;
        viewQueueJob(job);
      });
      card.addEventListener("keydown", (event) => {
        if ((event.key === "Enter" || event.key === " ") && !event.target.closest("a, button, input, select, textarea")) {
          event.preventDefault();
          viewQueueJob(job);
        }
      });
    }
    list.append(card);
  });
}

function updateActiveState() {
  const running = activeJob();
  if (running) state.activeJobId = running.job_id;
  if (!running && !state.activeJobId) {
    const mostRecentTerminal = state.jobs.find((job) => ["completed", "cancelled", "failed"].includes(job.status));
    if (mostRecentTerminal) state.activeJobId = mostRecentTerminal.job_id;
  }
  const job = monitoredJob();
  const forever = Boolean(running?.request?.unlimited);
  const queuePauseRequested = Boolean(workerState?.queue_pause_requested);
  const canRequestPause = Boolean(
    running
    && !["finalizing", "cancelling"].includes(String(running.status || "")),
  );
  const pauseControlAvailable = queuePauseRequested || canRequestPause || Number(workerState?.queued || 0) > 0;
  const resumeQueue = queuePauseRequested || running?.status === "paused";
  const pauseLabel = resumeQueue ? "Resume queue" : "Pause after current image";
  const canSkip = Boolean(
    running
    && running.status === "running"
    && !running.skip_current_requested
    && String(workerState?.model_runtime?.current_job_id || "") === String(running.job_id || ""),
  );
  const generationMotionActive = Boolean(running && running.status !== "paused");
  document.querySelectorAll('.site-nav-icon[data-icon="generate"]').forEach((icon) => {
    icon.classList.toggle("is-generating", generationMotionActive);
  });
  const cancelAvailable = jobCanBeCancelled(running);
  const cancelGroup = $("#cancelGenerationGroup");
  cancelGroup?.classList.toggle("is-hidden", !cancelAvailable);
  const cancelButton = $("#cancelButton");
  if (cancelButton) {
    cancelButton.disabled = !cancelAvailable;
    cancelButton.setAttribute("aria-label", "Stop current batch");
    cancelButton.title = "Stop current batch";
  }
  const cancelMenuButton = $("#cancelMenuButton");
  if (cancelMenuButton) cancelMenuButton.disabled = !cancelAvailable;
  const pauseButton = $("#pauseGenerationButton");
  if (pauseButton) {
    pauseButton.classList.toggle("is-active", queuePauseRequested);
    pauseButton.disabled = !pauseControlAvailable;
    pauseButton.classList.toggle("is-hidden", !pauseControlAvailable);
    setActionIcon(pauseButton, resumeQueue ? "play" : "pause", { label: pauseLabel, title: pauseLabel });
  }
  const skipButton = $("#skipGenerationButton");
  if (skipButton) {
    const skipLabel = running?.skip_current_requested ? "Skipping current image" : "Skip current image";
    skipButton.disabled = !canSkip;
    skipButton.classList.toggle("is-hidden", !running);
    skipButton.classList.toggle("is-working", Boolean(running?.skip_current_requested));
    setActionIcon(skipButton, "skip-next", { label: skipLabel, title: skipLabel });
  }
  const cancelMenuSkipButton = $("#cancelMenuSkipImage");
  if (cancelMenuSkipButton) cancelMenuSkipButton.disabled = !canSkip;
  const infinityButton = $("#infinityButton");
  infinityButton?.classList.toggle("is-active", forever);
  const infinityLabel = $("#infinityButtonLabel");
  if (infinityLabel) infinityLabel.textContent = forever ? "Generating Forever" : "Generate Forever";
  if (infinityButton) {
    const label = forever ? "Cancel generate forever" : "Generate forever";
    infinityButton.setAttribute("aria-label", label);
    infinityButton.title = label;
    infinityButton.querySelector(":scope > .ui-icon")?.classList.toggle("is-generating", forever && generationMotionActive);
  }
  if (!submitInFlight) {
    if (running) {
      const stageLabel = running?.skip_current_requested
        ? "Skipping current image…"
        : queuePauseRequested && !canRequestPause
          ? "Queue paused"
          : humanizeStage(running.worker_stage || running.status);
      setGenerateControlState({
        phase: running.worker_stage || running.status || "running",
        status: forever ? `${stageLabel} · ∞` : stageLabel,
        busy: true,
      });
    } else if (queuePauseRequested) {
      setGenerateControlState({ phase: "paused", status: "Queue paused", busy: true });
    } else {
      resetGenerateControl();
    }
  }
  renderLivePreviewJob(job);
}

function queueDisplayRank(job) {
  const status = String(job?.status || "");
  if (ACTIVE_JOB_STATUSES.has(status) && status !== "paused") return 0;
  if (status === "queued") return 1;
  if (status === "paused") return 2;
  return 3;
}

export function mergeQueueJobState(currentJobs, updatedJob) {
  if (!updatedJob?.job_id) return Array.isArray(currentJobs) ? currentJobs : [];
  const existing = new Map((currentJobs || []).map((job) => [job.job_id, job]));
  existing.set(updatedJob.job_id, { ...(existing.get(updatedJob.job_id) || {}), ...updatedJob });
  return [...existing.values()]
    .map((job, stableIndex) => ({ job, stableIndex }))
    .sort((left, right) => {
      const rankDifference = queueDisplayRank(left.job) - queueDisplayRank(right.job);
      return rankDifference || left.stableIndex - right.stableIndex;
    })
    .map(({ job }) => job);
}

function upsertJob(updatedJob) {
  if (!updatedJob?.job_id) return;
  state.jobs = mergeQueueJobState(state.jobs, updatedJob);
}

async function refreshOutputsForLatest() {
  await refreshOutputs({ selectNewest: state.layout.followNewestOutput });
}

function clearPostCompletionRefreshTimer() {
  if (!postCompletionRefreshTimer) return;
  window.clearTimeout(postCompletionRefreshTimer);
  postCompletionRefreshTimer = null;
}

function outputNamesForJob(job) {
  return (job?.output_paths || [])
    .map((path) => String(path || "").split(/[\/]/).pop())
    .filter(Boolean);
}

function recentOutputsContainJobOutput(job) {
  const expected = outputNamesForJob(job);
  if (!expected.length) return false;
  const known = new Set((state.recentOutputs || []).map((item) => String(item?.name || item?.output_id || "")));
  return expected.some((name) => known.has(name));
}

function schedulePostCompletionRefresh(job, attempt = 0) {
  const delays = [0, 350, 1200, 3000];
  const index = Math.max(0, Math.min(attempt, delays.length - 1));
  clearPostCompletionRefreshTimer();
  const run = async () => {
    try {
      await refreshOutputsForLatest();
    } finally {
      postCompletionRefreshTimer = null;
      if (!recentOutputsContainJobOutput(job) && index < delays.length - 1) {
        schedulePostCompletionRefresh(job, index + 1);
      }
    }
  };
  const delay = delays[index];
  if (!delay) {
    run();
    return;
  }
  postCompletionRefreshTimer = window.setTimeout(run, delay);
}

async function handleTerminalEvent(payload) {
  const job = payload?.job || payload;
  if (!job?.job_id) return;
  state.activeJobId = job.job_id;
  if (payload.type === "job-completed") {
    schedulePostCompletionRefresh(job);
  }
  await pollJobs();
}

function notifyOutputMetadataWarnings(job) {
  const warnings = Array.isArray(job?.output_save_status?.warnings)
    ? job.output_save_status.warnings
    : [];
  warnings.forEach((warning) => {
    const message = String(warning || "").trim();
    if (!message) return;
    const key = `${job.job_id || "job"}:${message}`;
    if (shownOutputMetadataWarnings.has(key)) return;
    shownOutputMetadataWarnings.add(key);
    notify(message, "warning");
  });
}

function applyEventPayload(payload) {
  const job = payload?.job || payload;
  if (!job?.job_id) return;
  upsertJob(job);
  state.activeJobId = job.job_id;
  renderQueue(state.jobs);
  renderLivePreviewJob(job);
  notifyOutputMetadataWarnings(job);
  updateActiveState();
}

function closeEventStream({ terminal = false } = {}) {
  if (eventStream) {
    eventStream.close();
    eventStream = null;
  }
  eventStreamJobId = null;
  setLivePreviewTransport(terminal ? "closed" : "fallback", eventStreamFailures);
}

function connectEventStream(job) {
  if (!job?.job_id || !(job.status === "queued" || ACTIVE_JOB_STATUSES.has(String(job.status || "")))) {
    if (eventStream) closeEventStream({ terminal: Boolean(job) });
    return;
  }
  if (Date.now() < eventStreamDisabledUntil) {
    setLivePreviewTransport("fallback", eventStreamFailures);
    return;
  }
  if (eventStream && eventStreamJobId === job.job_id) return;
  closeEventStream();
  eventStreamJobId = job.job_id;
  setLivePreviewTransport("connecting", eventStreamFailures);
  eventStream = new EventSource(`/api/jobs/${encodeURIComponent(job.job_id)}/events`);
  eventStream.onopen = () => {
    eventStreamFailures = 0;
    setLivePreviewTransport("live", 0);
  };

  const bindEvent = (name, handler) => {
    eventStream.addEventListener(name, async (event) => {
      eventStreamFailures = 0;
      setLivePreviewTransport("live", 0);
      try {
        const payload = JSON.parse(event.data);
        payload.type = name;
        await handler(payload);
      } catch (error) {
        console.error(error);
      }
    });
  };

  bindEvent("job-started", applyEventPayload);
  bindEvent("job-progress", applyEventPayload);
  bindEvent("job-paused", applyEventPayload);
  bindEvent("job-image-skipped", applyEventPayload);
  bindEvent("step-preview", applyEventPayload);
  bindEvent("job-output-produced", async (payload) => {
    applyEventPayload(payload);
    window.dispatchEvent(new CustomEvent("image-gen-profile-refresh"));
    const recentOutput = payload?.recent_output || null;
    if (recentOutput) {
      upsertRecentOutput(recentOutput, {
        selectNewest: Boolean(state.layout.followNewestOutput || !state.selectedOutput),
      });
      return;
    }
    schedulePostCompletionRefresh(payload?.job || payload);
  });
  bindEvent("job-output-timing-updated", async (payload) => {
    applyEventPayload(payload);
    const recentOutput = payload?.recent_output || null;
    if (recentOutput) {
      upsertRecentOutput(recentOutput, {
        selectNewest: Boolean(!state.selectedOutput),
      });
    }
  });
  bindEvent("job-cancelled", async (payload) => {
    applyEventPayload(payload);
    closeEventStream({ terminal: true });
    await handleTerminalEvent(payload);
  });
  bindEvent("job-completed", async (payload) => {
    applyEventPayload(payload);
    closeEventStream({ terminal: true });
    await handleTerminalEvent(payload);
  });
  bindEvent("job-failed", async (payload) => {
    applyEventPayload(payload);
    closeEventStream({ terminal: true });
    await handleTerminalEvent(payload);
  });

  eventStream.onerror = () => {
    eventStreamFailures += 1;
    setLivePreviewTransport("reconnecting", eventStreamFailures);
    if (eventStreamFailures >= 3) {
      closeEventStream();
      eventStreamDisabledUntil = Date.now() + 15000;
      setLivePreviewTransport("fallback", eventStreamFailures);
    }
  };
}

async function pollJobs() {
  if (pollJobsPromise) return pollJobsPromise;
  pollJobsPromise = (async () => {
    try {
      const payload = await api.jobs();
      const previousStatuses = new Map(state.jobs.map((job) => [job.job_id, job.status]));
      const previousOutputCounts = new Map(state.jobs.map((job) => [job.job_id, (job.output_paths || []).length]));
      renderWorker(payload.worker);
      renderQueue(payload.jobs);

      const newlyCompleted = payload.jobs.some((job) => {
        const before = previousStatuses.get(job.job_id);
        return before && before !== job.status && job.status === "completed";
      });
      const outputsChanged = payload.jobs.some((job) => {
        const before = previousOutputCounts.get(job.job_id) || 0;
        const after = (job.output_paths || []).length;
        return after > before;
      });

      updateActiveState();
      const job = activeJob();
      connectEventStream(job);
      if (!job && !eventStream) {
        const monitored = monitoredJob();
        setLivePreviewTransport(monitored ? "closed" : "idle", eventStreamFailures);
      }
      if (newlyCompleted || outputsChanged) {
        const completedJob = payload.jobs.find((job) => {
          const before = previousStatuses.get(job.job_id);
          return before && before !== job.status && job.status === "completed";
        });
        const outputJob = payload.jobs.find((job) => {
          const before = previousOutputCounts.get(job.job_id) || 0;
          const after = (job.output_paths || []).length;
          return after > before;
        });
        schedulePostCompletionRefresh(completedJob || outputJob || monitoredJob());
      }
    } catch (error) {
      renderWorker({ online: false, queued: 0 });
      setSubsystemStatus({
        id: "workerConnectionStatusLight",
        host: "#workerStatusLights",
        label: "Worker connection",
        status: "critical",
        stateLabel: "Connection error",
        summary: "The WebUI could not read generation worker status.",
        detail: error?.message || String(error || "Unknown worker status error"),
        diagnosticTarget: "#runtimeStatusPanel",
        placement: "prepend",
      });
      setSubsystemStatus({
        id: "queueSubsystemStatusLight",
        host: "#queueStatusLightHost",
        label: "Generation queue",
        status: "critical",
        stateLabel: "Status unavailable",
        summary: "Queue status could not be refreshed because the worker status request failed.",
        detail: error?.message || String(error || "Unknown queue status error"),
        diagnosticTarget: "#queuePanel",
      });
      setLivePreviewTransport("fallback", eventStreamFailures);
      console.error(error);
    } finally {
      pollJobsPromise = null;
    }
  })();
  return pollJobsPromise;
}

export async function acceptQueuedJob(job, { message = "Generation added to the queue." } = {}) {
  if (!job?.job_id) throw new Error("The server did not return a queued job identifier.");
  const previouslyMonitored = monitoredJob();
  const shouldMonitorNewJob = !activeJob() && (!previouslyMonitored || isTerminalStatus(previouslyMonitored.status));
  upsertJob({ ...job, request: job.request || {}, status: job.status || "queued" });
  if (shouldMonitorNewJob) state.activeJobId = job.job_id;
  renderQueue(state.jobs);
  renderLivePreviewJob(monitoredJob(), { forceLatest: true });
  connectEventStream(monitoredJob());
  notify(message);
  await pollJobs();
}

async function submit(unlimited) {
  if (submitInFlight) return;
  submitInFlight = true;
  setSubmissionBusy(true, "validating", "Validating…");
  try {
    const values = { ...collectValues(), unlimited: Boolean(unlimited) };
    setSubmissionBusy(true, "validating_prompt", "Validating prompt…");
    const promptPreflight = await preflightCurrentPrompt(values);
    const promptErrors = promptPreflight.blocking_errors || [];
    if (promptErrors.length) {
      throw new Error(promptErrors.map((item) => item.message || String(item)).join(" | "));
    }
    const promptWarnings = promptPreflight.behavior_warnings || [];
    if (promptWarnings.length) {
      values._webui_prompt_warnings_acknowledged = true;
      values._webui_prompt_warnings_recorded = promptWarnings.length;
    }
    Object.assign(values, promptPreflight.normalized_fields || {});
    values.prompt_preflight = promptPreflight;
    setSubmissionBusy(true, "validating_scheduler", "Validating scheduler…");
    const schedulerPreflight = await api.validateScheduler(values);
    const warnings = schedulerPreflight.validation_warnings || [];
    if (warnings.length) {
      const accepted = window.confirm(
        `The generation settings produced ${warnings.length} advisory warning${warnings.length === 1 ? "" : "s"}:\n\n${warnings.join("\n")}\n\nThese warnings do not block generation. Queue this generation with the selected/effective settings?`,
      );
      if (!accepted) return;
      values._webui_scheduler_warnings_acknowledged = true;
    }
    if (values.advanced_models_enabled) {
      setSubmissionBusy(true, "resolving_components", "Resolving model components…");
      values._webui_model_selection_id = "";
    } else {
      setSubmissionBusy(true, "authorizing_model", "Authorizing checkpoint selection…");
      if (!String(values.model_path || "").trim()) {
        throw new Error("Choose a checkpoint model first.");
      }
      values._webui_model_selection_id = state.activeModel?.resolved_path === values.model_path
        ? (state.activeModel.selection_id || values._webui_model_selection_id || "")
        : "";
    }
    setSubmissionBusy(true, "resolving_capabilities", "Resolving capabilities…");
    await refreshGenerationCapabilities();
    if (generationCapabilityBlocksSubmission()) {
      const reason = state.generationCapabilityBlockingReasons?.[0]?.message || "Generation settings require attention before queueing.";
      throw new Error(reason);
    }
    const guardedValues = enforceCfgRescaleRequestGuardrails({ ...values, ...collectValues() });
    setSubmissionBusy(true, "queueing", "Queueing…");
    const job = await api.submitJob(guardedValues);
    const modelWillActivateOnWorker = guardedValues.advanced_models_enabled || !state.activeModel || state.activeModel.resolved_path !== guardedValues.model_path;
    const message = unlimited
      ? "Generate forever started."
      : guardedValues.advanced_models_enabled
        ? "Generation added to the queue. The worker will assemble the selected component composition."
        : modelWillActivateOnWorker
          ? "Generation added to the queue. The worker will activate the selected checkpoint and then load it for the run."
          : "Generation added to the queue.";
    await acceptQueuedJob(
      { ...job, request: job.request || guardedValues },
      { message },
    );
    scheduleRecentOutputsPolling();
  } catch (error) {
    const message = String(error?.message || error || "Generation could not be queued.");
    if (/upscaler|hires[^.]*model/i.test(message)) {
      window.dispatchEvent(new CustomEvent("image-gen-hires-upscaler-error", { detail: { message } }));
    } else {
      notify(message, "error");
    }
  } finally {
    submitInFlight = false;
    setSubmissionBusy(false);
  }
}

async function stopCurrentBatch() {
  const active = activeJob();
  if (String(active?.status || "") === "finalizing") {
    notify("Generation is complete and the output is being saved. Saving will continue.", "warning");
    return;
  }
  if (!active || !jobCanBeCancelled(active)) {
    notify("There is no active batch to stop.", "warning");
    return;
  }
  try {
    await api.cancelJob(active.job_id);
    notify("Stop requested for the current batch.");
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

async function forceStopGeneration() {
  if (!window.confirm("Force stop the active generation and clear all queued jobs?")) return;
  try {
    const response = await api.forceStopGeneration();
    state.jobs = response.jobs || state.jobs;
    renderWorker(response.worker || workerState || {});
    renderQueue(state.jobs);
    updateActiveState();
    notify("Generation worker stopped and queued jobs cleared.", "warning");
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

async function togglePauseQueue() {
  const active = activeJob();
  try {
    if (workerState?.queue_pause_requested) {
      const response = await api.resumeQueue();
      renderWorker(response.worker || {});
      renderQueue(response.jobs || state.jobs);
      notify("Generation queue resumed.");
    } else {
      const response = await api.pauseQueueAfterCurrent(active?.job_id || "");
      renderWorker(response.worker || {});
      renderQueue(response.jobs || state.jobs);
      notify(active ? "Queue will pause after the current image finishes." : "Generation queue paused.");
    }
    updateActiveState();
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

async function skipCurrentImage() {
  const active = activeJob();
  if (!active || active.status !== "running") {
    notify("There is no actively sampling image to skip.", "warning");
    return;
  }
  try {
    await api.skipJobImage(active.job_id);
    notify("Skipping the current image. The remaining queue will continue.");
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

export function bindGeneration(options) {
  collectValues = options.collectValues;
  refreshOutputs = options.refreshOutputs;
  ensureModelReady = options.ensureModelReady || ensureModelReady;
  bindLivePreview({ cancelActive: stopCurrentBatch });
  setSubsystemStatus({
    id: "workerConnectionStatusLight",
    host: "#workerStatusLights",
    label: "Worker connection",
    status: "transitioning",
    stateLabel: "Connecting",
    summary: "Connecting to the generation worker.",
    detail: "The WebUI is waiting for the first worker status response.",
    diagnosticTarget: "#runtimeStatusPanel",
    placement: "prepend",
  });
  setSubsystemStatus({
    id: "modelResidencyStatusLight",
    host: "#workerStatusLights",
    label: "Model residency",
    status: "inactive",
    stateLabel: "Checking",
    summary: "Resident model state has not been received yet.",
    detail: "This indicator updates with the first worker status response.",
    diagnosticTarget: "#runtimeStatusPanel",
  });

  $("#generationForm").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(false);
  });
  const submitNow = (event) => {
    event?.preventDefault?.();
    submit(false);
  };
  $("#copyResidencyReportButton")?.addEventListener("click", async () => {
    const report = $("#runtimeResidencyReport")?.textContent || "No residency report is available.";
    try {
      await navigator.clipboard.writeText(report);
      notify("Residency report copied.");
    } catch (error) {
      notify(`Unable to copy residency report: ${error.message}`, "error");
    }
  });
  $("#generateButton")?.addEventListener("click", submitNow);
  $("#infinityButton").addEventListener("click", () => {
    const job = activeJob();
    if (job?.request?.unlimited) stopCurrentBatch();
    else submit(true);
  });
  $("#cancelButton")?.addEventListener("click", stopCurrentBatch);
  $("#pauseGenerationButton")?.addEventListener("click", togglePauseQueue);
  $("#skipGenerationButton")?.addEventListener("click", skipCurrentImage);

  const menu = $("#generateMenu");
  $("#generateMenuButton").addEventListener("click", (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    menu.style.left = `${Math.max(8, rect.right - 190)}px`;
    menu.style.top = `${rect.bottom + 6}px`;
    menu.classList.toggle("is-hidden");
  });
  menu.addEventListener("click", (event) => {
    const mode = event.target.dataset.generationMode;
    if (mode === "once") submit(false);
    if (mode === "forever") submit(true);
    if (mode === "cancel") stopCurrentBatch();
    menu.classList.add("is-hidden");
  });
  document.addEventListener("click", (event) => {
    if (!menu.contains(event.target) && event.target !== $("#generateMenuButton")) {
      menu.classList.add("is-hidden");
    }
  });

  const cancelMenu = $("#cancelGenerationMenu");
  const cancelMenuButton = $("#cancelMenuButton");
  cancelMenuButton?.addEventListener("click", (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    cancelMenu.style.left = `${Math.max(8, rect.right - 220)}px`;
    cancelMenu.style.top = `${rect.bottom + 6}px`;
    const opening = cancelMenu.classList.contains("is-hidden");
    cancelMenu.classList.toggle("is-hidden");
    cancelMenuButton.setAttribute("aria-expanded", opening ? "true" : "false");
  });
  cancelMenu?.addEventListener("click", async (event) => {
    const action = event.target?.dataset?.cancelAction;
    if (!action) return;
    cancelMenu.classList.add("is-hidden");
    cancelMenuButton?.setAttribute("aria-expanded", "false");
    if (action === "batch") await stopCurrentBatch();
    if (action === "skip") await skipCurrentImage();
    if (action === "force") await forceStopGeneration();
  });
  document.addEventListener("click", (event) => {
    if (!cancelMenu?.contains(event.target) && event.target !== cancelMenuButton) {
      cancelMenu?.classList.add("is-hidden");
      cancelMenuButton?.setAttribute("aria-expanded", "false");
    }
  });

  $("#clearCompletedQueueButton")?.addEventListener("click", async () => {
    const button = $("#clearCompletedQueueButton");
    try {
      if (button) button.disabled = true;
      const response = await api.dismissTerminalJobs();
      state.jobs = response.jobs || state.jobs.filter((job) => !isTerminalStatus(job.status));
      const count = Number(response.removed_count || 0);
      renderQueue(state.jobs);
      notify(count ? `Cleared ${count} completed, cancelled, or failed queue entr${count === 1 ? "y" : "ies"}.` : "There were no removable queue entries.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      await pollJobs();
    }
  });
  $("#queueFilterToggleButton")?.addEventListener("click", (event) => {
    const panel = $("#queueFilterPanel");
    if (!panel) return;
    panel.classList.toggle("is-hidden");
    event.currentTarget.setAttribute("aria-expanded", String(!panel.classList.contains("is-hidden")));
  });
  document.querySelectorAll("[data-queue-filter]").forEach((input) => {
    input.addEventListener("change", () => {
      const selected = [...document.querySelectorAll("[data-queue-filter]:checked")].map((item) => item.dataset.queueFilter);
      state.queueFilters = selected.length ? selected : ["queued"];
      renderQueue(state.jobs);
    });
  });

  const loadImageInput = $("#loadImageSettingsInput");
  $("#loadImageSettingsButton")?.addEventListener("click", () => loadImageInput?.click());
  loadImageInput?.addEventListener("change", async () => {
    const [file] = loadImageInput.files || [];
    loadImageInput.value = "";
    if (!file) return;
    try {
      const formData = new FormData();
      formData.append("file", file, file.name);
      const details = await api.inspectOutputUpload(formData);
      if (state.settings.metadata_import_auto_apply_full_run) {
        const result = await options.applyValues(details.replay || {});
        const ignored = result?.unsupported || [];
        notify(ignored.length ? `Image settings loaded. ${ignored.length} field(s) remain unsupported.` : "Image settings loaded into the form.");
      } else {
        await openOutputDetailsData(details, { opener: $("#loadImageSettingsButton") });
      }
    } catch (error) {
      notify(error.message, "error");
    }
  });

  $("#warmModelNowButton")?.addEventListener("click", async () => {
    const modelPath = String(collectValues().model_path || "").trim();
    if (!modelPath) {
      notify("Choose a checkpoint model before warming it.", "error");
      return;
    }
    const button = $("#warmModelNowButton");
    try {
      if (button) {
        button.disabled = true;
        button.textContent = "Warming…";
      }
      const response = await api.preloadWarmModel(modelPath);
      if (response.active_model) state.activeModel = response.active_model;
      renderModelResidencyStatus(response.model_runtime || response.result?.status || {});
      if (response.result?.disabled || response.result?.ok === false) {
        notify(response.result?.reason || "Persistent warmup is disabled by the selected policy.", "warning");
      } else {
        notify("Selected checkpoint is warm and ready for queued generations.");
      }
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Warm Selected Model";
      }
      await pollJobs();
    }
  });
  $("#unloadWarmModelButton")?.addEventListener("click", async () => {
    const button = $("#unloadWarmModelButton");
    try {
      if (button) button.disabled = true;
      const response = await api.unloadWarmModel();
      renderModelResidencyStatus(response.model_runtime || {});
      notify("Warm checkpoint unloaded.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      await pollJobs();
    }
  });
  $("#recoverWarmWorkerButton")?.addEventListener("click", async () => {
    const button = $("#recoverWarmWorkerButton");
    try {
      if (button) {
        button.disabled = true;
        button.textContent = "Recovering…";
      }
      const response = await api.recoverWarmWorker({
        clear_active: true,
        clear_queue: false,
        reason: "Manual recovery requested from the WebUI.",
      });
      renderModelResidencyStatus(response.worker?.model_runtime || {});
      if (response.active_job_id) notify(`Recovered the persistent worker and released stuck job ${response.active_job_id}.`, "warning");
      else notify("Persistent worker recovered and ready for the next generation.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Recover Worker";
      }
      await pollJobs();
    }
  });
  $("#clearQueuedJobsButton")?.addEventListener("click", async () => {
    const button = $("#clearQueuedJobsButton");
    try {
      if (button) {
        button.disabled = true;
        button.textContent = "Clearing…";
      }
      const response = await api.clearQueuedJobs({
        reason: "Queued jobs were cleared from the WebUI.",
      });
      const count = Number(response.cleared_count || 0);
      notify(count ? `Cleared ${count} queued job${count === 1 ? "" : "s"}.` : "There were no queued jobs to clear.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Clear Queue";
      }
      await pollJobs();
    }
  });
  window.addEventListener("warm-worker-settings-changed", () => pollJobs());

  window.addEventListener("job-cache-cleared", () => pollJobs());
  window.addEventListener("recent-outputs-refresh-settings-changed", () => scheduleRecentOutputsPolling());
  document.addEventListener("visibilitychange", () => scheduleRecentOutputsPolling());
  refreshQueueToolbar();
  pollJobs();
  refreshOutputs();
  pollTimer = window.setInterval(() => {
    pollJobs();
    if (!activeJob() && !submitInFlight) scheduleRecentOutputsPolling();
  }, 1200);
  scheduleRecentOutputsPolling();
}

export function stopGenerationPolling() {
  closeEventStream();
  stopLivePreview();
  if (pollTimer) window.clearInterval(pollTimer);
  if (recentOutputsPollTimer) window.clearTimeout(recentOutputsPollTimer);
  clearPostCompletionRefreshTimer();
}
