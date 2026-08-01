import { api } from "../api.js";
import { state } from "../state.js";
import { $, shortText, formatTime, notify } from "../utils.js";
import {
  bindLivePreview,
  renderLivePreviewJob,
  setLivePreviewTransport,
  stopLivePreview,
} from "./live-preview.js?v=0.1.56";
import { showOutput } from "./gallery.js";
import { openOutputDetailsData } from "./output-details.js";
import { preflightCurrentPrompt } from "./prompt-tools.js?v=0.1.57";

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
let eventStreamFailures = 0;
let eventStreamDisabledUntil = 0;

function setSubmissionBusy(active, stage = "") {
  ["#generateButton", "#topGenerateButton"].forEach((selector) => {
    const button = $(selector);
    if (!button) return;
    button.disabled = active;
    if (active) {
      button.dataset.originalText = button.dataset.originalText || button.textContent || "Generate";
      button.textContent = stage || "Queueing…";
    } else if (button.dataset.originalText) {
      button.textContent = button.dataset.originalText;
    }
  });
  ["#generateMenuButton", "#topInfinityButton", "#infinityButton"].forEach((selector) => {
    const button = $(selector);
    if (button) button.disabled = active;
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

const ACTIVE_JOB_STATUSES = new Set(["preparing_model", "warming_model", "running", "finalizing", "cancelling"]);

function isTerminalStatus(status) {
  return ["completed", "cancelled", "failed"].includes(String(status || ""));
}

function queueFilterStatus(job) {
  const status = String(job?.status || "");
  return ["preparing_model", "warming_model", "finalizing"].includes(status) ? "running" : status;
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
  if (label) {
    control.title = label;
    if (!control.getAttribute("aria-label")) control.setAttribute("aria-label", label);
  }
  const iconNode = document.createElement("span");
  iconNode.className = "queue-action-icon";
  iconNode.setAttribute("aria-hidden", "true");
  iconNode.textContent = icon || "•";
  const labelNode = document.createElement("span");
  labelNode.className = "queue-action-label";
  labelNode.textContent = label || "";
  control.replaceChildren(iconNode, labelNode);
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
  return state.jobs.find((job) => ACTIVE_JOB_STATUSES.has(String(job.status || ""))) || null;
}

function monitoredJob() {
  const running = activeJob();
  if (running) return running;
  if (!state.activeJobId) return null;
  return state.jobs.find((job) => job.job_id === state.activeJobId) || null;
}

function renderWarmWorkerStatus(warm = {}) {
  const persistentMode = state.settings.generation_worker_mode === "persistent_experimental" && state.settings.warm_worker_enabled === true;
  $("#modelWarmStatusPanel")?.classList.toggle("is-hidden", !persistentMode);
  const stage = String(warm.stage || (warm.online ? "idle" : "offline"));
  const warmState = String(warm.warm_state || "cold");
  const badge = $("#modelWarmStatusBadge");
  const text = $("#modelWarmStatusText");
  const memory = $("#modelWarmMemoryText");
  const unloadButton = $("#unloadWarmModelButton");
  const recoverButton = $("#recoverWarmWorkerButton");
  const clearQueueButton = $("#clearQueuedJobsButton");
  if (badge) {
    badge.textContent = stage === "ready" ? "Ready" : humanizeStage(stage);
    badge.className = `status-badge ${stage === "ready" ? "warm" : stage}`;
  }
  const modelName = String(warm.current_model_path || warm.selected_model_path || "").split(/[\\/]/).pop();
  if (text) {
    if (warm.last_error) text.textContent = `Warm worker error: ${warm.last_error}`;
    else if (warmState === "warm") text.textContent = `${modelName || "Selected model"} is retained by the persistent worker.`;
    else if (["preparing_model", "loading_tokenizer", "loading_checkpoint", "reusing_checkpoint", "model_ready", "applying_retention_policy"].includes(stage)) {
      text.textContent = `${humanizeStage(stage)}${modelName ? ` · ${modelName}` : ""}`;
    } else if (!warm.online) text.textContent = "The persistent worker is offline and will start when needed.";
    else text.textContent = "The persistent worker is online without a loaded checkpoint.";
  }
  if (memory) {
    const devices = Object.entries(warm.component_devices || {})
      .map(([name, device]) => `${name}: ${device}`)
      .join(" · ");
    const allocated = Number(warm.memory?.allocated_bytes || 0) / (1024 ** 3);
    const cudaState = warm.cuda_available === false ? "CUDA unavailable" : warm.cuda_available === true ? "CUDA available" : "CUDA status unknown";
    const policy = warm.execution_device_policy ? ` · policy: ${String(warm.execution_device_policy).replaceAll("_", " ")}` : "";
    const fallback = warm.cpu_fallback_reason ? ` · CPU fallback: ${warm.cpu_fallback_reason}` : "";
    memory.textContent = `${devices || `GPU residency: ${warm.gpu_loaded ? "active" : "none"}`}${allocated ? ` · ${allocated.toFixed(2)} GiB allocated` : ""} · ${cudaState}${policy}${fallback}`;
  }
  if (unloadButton) unloadButton.disabled = warmState !== "warm" && !warm.current_model_path;
  if (recoverButton) recoverButton.disabled = false;
  if (clearQueueButton) clearQueueButton.disabled = false;
}

function renderWorker(worker) {
  const online = Boolean(worker?.online);
  $("#workerText").textContent = online
    ? worker.active_job_id ? `Active · ${worker.queued} queued` : `Online · ${worker.queued} queued`
    : "Offline";
  $("#footerWorkerStatus").textContent = `Worker: ${online ? "online" : "offline"}`;
  $("#workerPill").classList.toggle("is-offline", !online);
  renderWarmWorkerStatus(worker?.warm_worker || {});
}

function jobCanBeCancelled(job) {
  const status = String(job?.status || "");
  return status === "queued" || ACTIVE_JOB_STATUSES.has(status);
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
  if (!visibleJobs.length) {
    list.className = "stack-list empty-state";
    list.textContent = state.jobs.length ? "No queue items match the current filter." : "No queued generations.";
    return;
  }

  list.className = "stack-list";
  visibleJobs.slice(0, 12).forEach((job) => {
    const card = document.createElement("article");
    card.className = "queue-card";
    card.dataset.jobId = job.job_id;
    card.tabIndex = 0;
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = shortText(job.request?.positive_prompt || "Untitled generation", 38);
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
    const warningLabel = warningCount ? ` · ${warningCount} scheduler warning${warningCount === 1 ? "" : "s"}` : "";
    const executionLabel = job.execution_mode ? ` · ${String(job.execution_mode).replaceAll("_", " ")}` : "";
    const parserLabel = job.prompt_preflight?.base?.parser?.label || job.request?.prompt_parser_name || "legacy";
    const profileLabel = job.prompt_preflight?.base?.shortcut_profile?.label || job.request?.prompt_shortcut_profile_name || "profile";
    const promptValidationLabel = job.prompt_preflight?.valid === false ? " · prompt invalid" : " · prompt validated";
    settings.textContent = `${job.request?.width || "?"}×${job.request?.height || "?"} · ${job.request?.steps || "?"} steps · ${job.request?.sampler_name || "sampler"} · ${schedulerLabel}${presetLabel}${warningLabel} · ${parserLabel} / ${profileLabel}${promptValidationLabel} · ${selectedModel}${executionLabel}`;
    const timing = document.createElement("small");
    timing.textContent = job.completed_at ? `Finished ${formatTime(job.completed_at)}` : `Started ${formatTime(job.started_at || job.created_at)}`;
    card.append(header, settings, timing);
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
    if (jobCanBeCancelled(job)) {
      const cancelButton = document.createElement("button");
      cancelButton.type = "button";
      cancelButton.className = "small-button queue-cancel-button";
      cancelButton.disabled = job.status === "cancelling";
      const cancelLabel = job.status === "queued"
        ? "Cancel queued item"
        : job.status === "cancelling" ? "Cancelling…" : "Cancel generation";
      decorateQueueAction(cancelButton, { icon: job.status === "cancelling" ? "…" : "⏹", label: cancelLabel });
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
      decorateQueueAction(viewButton, { icon: "🖼", label: "View output" });
      viewButton.addEventListener("click", () => viewQueueJob(job));
      actions.append(viewButton);
    }
    const seedButton = document.createElement("button");
    seedButton.type = "button";
    seedButton.className = "small-button";
    const copySeedValue = concreteQueueSeed(job);
    decorateQueueAction(seedButton, {
      icon: "🌱",
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
    decorateQueueAction(diagnosticsLink, { icon: "🩺", label: "Run diagnostics" });
    actions.append(diagnosticsLink);
    if (job.console_log_path) {
      const logLink = document.createElement("a");
      logLink.className = "small-button queue-log-link";
      logLink.href = `/api/jobs/${encodeURIComponent(job.job_id)}/log`;
      logLink.target = "_blank";
      logLink.rel = "noopener";
      decorateQueueAction(logLink, { icon: "🧾", label: "Console log" });
      actions.append(logLink);
    }
    if (canRemoveQueueJob(job)) {
      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.className = "small-button";
      decorateQueueAction(removeButton, { icon: "✕", label: "Remove" });
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
      error.textContent = shortText(job.error, 100);
      error.title = job.error;
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
  $("#cancelButton").classList.toggle("is-hidden", !running);
  $("#infinityButton").classList.toggle("is-active", forever);
  $("#infinityButton").textContent = forever ? "[ ∞ Generating ]" : "∞ Generate Forever";
  renderLivePreviewJob(job);
}

function upsertJob(updatedJob) {
  if (!updatedJob?.job_id) return;
  const existing = new Map(state.jobs.map((job) => [job.job_id, job]));
  existing.set(updatedJob.job_id, { ...(existing.get(updatedJob.job_id) || {}), ...updatedJob });
  const values = [...existing.values()];
  values.sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")));
  state.jobs = values;
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

function applyEventPayload(payload) {
  const job = payload?.job || payload;
  if (!job?.job_id) return;
  upsertJob(job);
  state.activeJobId = job.job_id;
  renderQueue(state.jobs);
  renderLivePreviewJob(job);
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
  bindEvent("step-preview", applyEventPayload);
  bindEvent("job-output-produced", async (payload) => {
    applyEventPayload(payload);
    schedulePostCompletionRefresh(payload?.job || payload);
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
  state.activeJobId = job.job_id;
  upsertJob({ ...job, request: job.request || {}, status: job.status || "queued" });
  renderQueue(state.jobs);
  renderLivePreviewJob(monitoredJob(), { forceLatest: true });
  connectEventStream(monitoredJob());
  notify(message);
  await pollJobs();
}

async function submit(unlimited) {
  if (submitInFlight) return;
  submitInFlight = true;
  setSubmissionBusy(true, "Validating…");
  try {
    const values = { ...collectValues(), unlimited: Boolean(unlimited) };
    setSubmissionBusy(true, "Validating prompt…");
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
    setSubmissionBusy(true, "Validating scheduler…");
    const schedulerPreflight = await api.validateScheduler(values);
    const warnings = schedulerPreflight.validation_warnings || [];
    if (warnings.length) {
      const accepted = window.confirm(
        `The scheduler settings produced ${warnings.length} warning${warnings.length === 1 ? "" : "s"}:\n\n${warnings.join("\n")}\n\nQueue this generation with the normalized effective settings?`,
      );
      if (!accepted) return;
      values._webui_scheduler_warnings_acknowledged = true;
    }
    setSubmissionBusy(true, "Activating model selection…");
    const activeModel = await ensureModelReady();
    if (!activeModel || activeModel.resolved_path !== values.model_path) {
      throw new Error("The selected checkpoint was not activated by the backend. Select the model again before generating.");
    }
    values._webui_model_selection_id = activeModel.selection_id || values._webui_model_selection_id || "";
    setSubmissionBusy(true, "Queueing…");
    const job = await api.submitJob(values);
    const modelWillActivateOnWorker = !state.activeModel || state.activeModel.resolved_path !== values.model_path;
    const message = unlimited
      ? "Generate forever started."
      : modelWillActivateOnWorker
        ? "Generation added to the queue. The worker will activate the selected checkpoint and then load it for the run."
        : "Generation added to the queue.";
    await acceptQueuedJob(
      { ...job, request: job.request || values },
      { message },
    );
    scheduleRecentOutputsPolling();
  } catch (error) {
    notify(error.message, "error");
  } finally {
    submitInFlight = false;
    setSubmissionBusy(false);
  }
}

async function cancelActive() {
  const job = activeJob() || state.jobs.find((item) => item.status === "queued") || null;
  if (!job) {
    notify("There is no active or queued generation to cancel.", "warning");
    return;
  }
  try {
    const mode = state.settings.cancel_generation_mode || "immediate";
    if (mode === "after_current_run") {
      const queued = state.jobs.filter((item) => item.status === "queued");
      await Promise.all(queued.map((item) => api.cancelJob(item.job_id).catch(() => null)));
      notify("Current run will continue. Remaining queued items were cancelled.");
    } else {
      await api.cancelJob(job.job_id);
      notify("Cancellation requested.");
    }
    await pollJobs();
  } catch (error) {
    notify(error.message, "error");
  }
}

export function bindGeneration(options) {
  collectValues = options.collectValues;
  refreshOutputs = options.refreshOutputs;
  ensureModelReady = options.ensureModelReady || ensureModelReady;
  bindLivePreview({ cancelActive });

  $("#generationForm").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(false);
  });
  const submitNow = (event) => {
    event?.preventDefault?.();
    submit(false);
  };
  $("#topGenerateButton")?.addEventListener("click", submitNow);
  $("#generateButton")?.addEventListener("click", submitNow);
  $("#topInfinityButton")?.addEventListener("click", () => {
    const job = activeJob();
    if (job?.request?.unlimited) cancelActive();
    else submit(true);
  });
  $("#topCancelButton")?.addEventListener("click", cancelActive);
  $("#infinityButton").addEventListener("click", () => {
    const job = activeJob();
    if (job?.request?.unlimited) cancelActive();
    else submit(true);
  });
  $("#cancelButton").addEventListener("click", cancelActive);

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
    if (mode === "cancel") cancelActive();
    menu.classList.add("is-hidden");
  });
  document.addEventListener("click", (event) => {
    if (!menu.contains(event.target) && event.target !== $("#generateMenuButton")) {
      menu.classList.add("is-hidden");
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
      renderWarmWorkerStatus(response.warm_worker || response.result?.status || {});
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
      renderWarmWorkerStatus(response.warm_worker || {});
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
      renderWarmWorkerStatus(response.worker?.warm_worker || {});
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
