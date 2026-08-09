import { state } from "../state.js";
import { $, shortText } from "../utils.js";
import { openLiveLightbox, syncLiveLightbox } from "./lightbox.js";
import { renderCfgGraph } from "./cfg-lab.js?v=0.1.45";
import { renderMemoryStatus } from "./memory-status.js?v=status-lights1";
import { setSubsystemStatus } from "../components/status-indicators.js?v=1";
import { setActionIcon } from "../components/action-icons.js?v=0.1.0";

let currentJob = null;
let cancelGeneration = async () => {};
let elapsedTimer = null;

const TERMINAL_STATUSES = new Set(["completed", "cancelled", "failed"]);

function isTerminal(status) {
  return TERMINAL_STATUSES.has(String(status || "").toLowerCase());
}

function isActive(status) {
  return ["queued", "preparing_model", "warming_model", "running", "finalizing", "cancelling"].includes(String(status || "").toLowerCase());
}

function modelName(job) {
  return job?.model_diagnostics?.runtime?.file_name
    || (job?.model_selection?.model_name ? `${job.model_selection.model_name}${job.model_selection.extension || ""}` : "")
    || String(job?.request?.model_path || "model").split(/[\\/]/).pop();
}

function numericProgress(job) {
  const total = Number(job?.progress?.total_steps ?? job?.total_steps ?? job?.request?.steps ?? 0);
  const step = Number(job?.progress?.current_step ?? job?.current_step ?? job?.step ?? 0);
  const explicit = Number(job?.progress?.percent ?? job?.progress_percent);
  if (Number.isFinite(explicit)) {
    return { step, total, percent: Math.max(0, Math.min(100, explicit)) };
  }
  if (Number.isFinite(step) && Number.isFinite(total) && total > 0) {
    return { step, total, percent: Math.max(0, Math.min(100, (step / total) * 100)) };
  }
  return { step: 0, total, percent: null };
}

function normalizedHistory(job) {
  const history = Array.isArray(job?.live_preview_history)
    ? job.live_preview_history.filter((item) => item?.preview_url)
    : [];
  history.sort((left, right) => Number(left.step || 0) - Number(right.step || 0));
  if (job?.request?.live_preview_keep_history === "latest_only") {
    return history.length ? [history[history.length - 1]] : [];
  }
  return history;
}

function latestHistoryRecord(job) {
  const history = normalizedHistory(job);
  return history.length ? history[history.length - 1] : null;
}

function displayFrame(record, { follow = false } = {}) {
  if (!record?.preview_url) return;
  state.livePreview.frameUrl = record.preview_url;
  state.livePreview.frameName = record.is_final
    ? "Final output"
    : `Live frame · step ${record.step || "?"}`;
  state.livePreview.selectedStep = Number(record.step || 0) || null;
  state.livePreview.decodeMode = record.decode_mode || state.livePreview.decodeMode || "";
  state.livePreview.frameWidth = Number(record.image_width || state.livePreview.frameWidth || 0);
  state.livePreview.frameHeight = Number(record.image_height || state.livePreview.frameHeight || 0);
  if (follow) {
    state.livePreview.followLatest = true;
    state.livePreview.updatesPaused = false;
  }
}

function jumpToLatest() {
  if (!state.livePreview.latestFrameUrl) return;
  const latest = state.livePreview.history[state.livePreview.history.length - 1];
  if (latest) displayFrame(latest, { follow: true });
  else {
    state.livePreview.frameUrl = state.livePreview.latestFrameUrl;
    state.livePreview.selectedStep = state.livePreview.latestStep || null;
    state.livePreview.followLatest = true;
    state.livePreview.updatesPaused = false;
  }
  state.lightbox.followLatest = true;
  renderCurrent();
  syncLiveLightbox();
}

function elapsedSeconds() {
  const start = Date.parse(state.livePreview.startedAt || "");
  if (!Number.isFinite(start)) return null;
  const completed = Date.parse(state.livePreview.completedAt || "");
  const end = Number.isFinite(completed) ? completed : Date.now();
  return Math.max(0, (end - start) / 1000);
}

function formatElapsed(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} sec`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

function formatTimingMilliseconds(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(milliseconds < 10000 ? 2 : 1)} sec`;
  const totalSeconds = Math.round(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function renderSamplingTiming() {
  const label = $("#livePreviewStepTiming");
  if (!label) return;
  const timing = state.livePreview.samplingTiming || {};
  const last = formatTimingMilliseconds(timing.step_duration_ms);
  const average = formatTimingMilliseconds(
    timing.rolling_average_step_ms ?? timing.average_step_ms
  );
  const eta = formatTimingMilliseconds(timing.estimated_remaining_ms);
  label.textContent = `Last step: ${last} · Rolling avg: ${average} · Sampling ETA: ${eta}`;
}

function updateElapsed() {
  const seconds = elapsedSeconds();
  state.livePreview.elapsedSeconds = Number.isFinite(seconds) ? seconds : 0;
  const label = $("#livePreviewElapsed");
  if (label) label.textContent = `Elapsed: ${formatElapsed(seconds)}`;
}

function transportLabel() {
  const labels = {
    idle: "Idle",
    connecting: "Connecting",
    live: "Live stream",
    reconnecting: "Reconnecting",
    fallback: "Polling fallback",
    closed: "Stream closed",
  };
  const base = labels[state.livePreview.streamStatus] || "Polling fallback";
  return state.livePreview.reconnectCount > 0 && state.livePreview.streamStatus !== "live"
    ? `${base} · ${state.livePreview.reconnectCount}`
    : base;
}

function renderTransport() {
  const element = $("#livePreviewStreamStatus");
  if (!element) return;
  element.textContent = transportLabel();
  element.className = `live-stream-status ${state.livePreview.streamStatus || "idle"}`;
  const streamStatus = String(state.livePreview.streamStatus || "idle");
  const mapping = {
    live: ["healthy", "Live"],
    connecting: ["transitioning", "Connecting"],
    reconnecting: ["transitioning", "Reconnecting"],
    fallback: ["warning", "Polling fallback"],
    closed: ["inactive", "Closed"],
    idle: ["inactive", "Idle"],
  };
  const [indicatorStatus, stateLabel] = mapping[streamStatus] || ["warning", transportLabel()];
  setSubsystemStatus({
    id: "livePreviewTransportStatusLight",
    host: "#livePreviewStatusLightHost",
    label: "Live Preview transport",
    status: indicatorStatus,
    stateLabel,
    summary: `Live Preview transport: ${transportLabel()}.`,
    detail: state.livePreview.reconnectCount > 0
      ? `${state.livePreview.reconnectCount} reconnect attempt${state.livePreview.reconnectCount === 1 ? "" : "s"} recorded.`
      : "No reconnect attempts are currently recorded.",
    facts: {
      transport: streamStatus,
      reconnects: state.livePreview.reconnectCount || 0,
      job_status: state.livePreview.status || "idle",
    },
    diagnosticTarget: "#livePreviewPanel",
  });
}

function renderLiveCfgVisual() {
  const panel = $("#livePreviewCfgVisual");
  const graph = $("#livePreviewCfgGraph");
  const stepLabel = $("#livePreviewCfgStepLabel");
  const summary = $("#livePreviewCfgSummary");
  if (!panel || !graph || !stepLabel || !summary) return;

  const enabled = state.settings.cfg_lab_enabled === true && state.settings.live_preview_cfg_visual_enabled === true;
  panel.hidden = !enabled || !currentJob;
  if (panel.hidden) return;

  const points = Array.isArray(state.livePreview.cfgPoints)
    ? state.livePreview.cfgPoints.filter((point) => Number.isFinite(Number(point?.effective_cfg_scale)))
    : [];
  if (!points.length) {
    graph.textContent = "Waiting for per-step CFG telemetry.";
    stepLabel.textContent = "Waiting for CFG telemetry";
    summary.replaceChildren();
    ["Requested: —", "Effective: —", "Mode: —"].forEach((value) => {
      const item = document.createElement("span");
      item.textContent = value;
      summary.append(item);
    });
    return;
  }

  const latest = points[points.length - 1];
  renderCfgGraph(graph, points, {
    compact: true,
    currentStepIndex: Number(latest.step_index),
  });
  const stepNumber = Number(latest.step_index) + 1;
  stepLabel.textContent = `Step ${stepNumber} / ${state.livePreview.totalSteps || points.length}`;
  const values = [
    `Requested: ${Number(latest.requested_cfg_scale).toFixed(2)}`,
    `Effective: ${Number(latest.effective_cfg_scale).toFixed(2)}`,
    `Mode: ${latest.guidance_mode || state.livePreview.guidanceMode || "flat"}`,
  ];
  summary.replaceChildren();
  values.forEach((value) => {
    const item = document.createElement("span");
    item.textContent = value;
    summary.append(item);
  });
}

function renderHistory() {
  const strip = $("#livePreviewHistory");
  const status = $("#livePreviewHistoryStatus");
  if (!strip || !status) return;
  strip.replaceChildren();
  const history = state.livePreview.history;
  const metrics = state.livePreview.metrics || {};
  const processed = Number(metrics.frames_processed);
  const replaced = Number(metrics.frames_replaced);
  const metricText = Number.isFinite(processed)
    ? ` · processed ${processed}${Number.isFinite(replaced) ? ` · coalesced ${replaced}` : ""}`
    : "";
  status.textContent = history.length
    ? `${history.length} preview frame${history.length === 1 ? "" : "s"} retained${state.livePreview.latestStep ? ` · latest preview ${state.livePreview.latestStep}` : ""}${metricText}`
    : `No frames${metricText}`;
  if (!history.length) {
    strip.className = "live-preview-history empty-state";
    strip.textContent = "No preview frames yet.";
    return;
  }
  strip.className = "live-preview-history";
  history.forEach((record) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "live-history-frame";
    button.classList.toggle("is-selected", Number(record.step) === Number(state.livePreview.selectedStep));
    button.title = record.is_final
      ? "Final decoded output"
      : `Inspect preview step ${record.step} · ${record.decode_mode || "preview"}`;
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = record.preview_url;
    image.alt = record.is_final ? "Final output thumbnail" : `Preview step ${record.step}`;
    const label = document.createElement("span");
    label.textContent = record.is_final ? "Final" : String(record.step || "?");
    button.append(image, label);
    button.addEventListener("click", () => {
      state.livePreview.followLatest = false;
      state.livePreview.updatesPaused = false;
      state.lightbox.followLatest = false;
      displayFrame(record);
      renderCurrent();
    });
    strip.append(button);
  });
}

function statusText(progress) {
  const status = state.livePreview.status;
  const saveStatus = currentJob?.output_save_status || {};
  const pendingSaves = Number(currentJob?.pending_save_batches || saveStatus.pending_batches || 0);
  const completedSaves = Number(currentJob?.completed_save_batches || saveStatus.completed_batches || 0);
  if (
    !isTerminal(status)
    && state.livePreview.selectedStep
    && state.livePreview.selectedStep !== progress.step
  ) {
    return `Viewing preview step ${state.livePreview.selectedStep} · sampler ${progress.step || "?"} / ${progress.total || "?"}`;
  }
  if (status === "cancelled") {
    return progress.step && progress.total
      ? `Cancelled at step ${progress.step} / ${progress.total}`
      : "Cancelled";
  }
  if (status === "failed") {
    return progress.step && progress.total
      ? `Failed at step ${progress.step} / ${progress.total}`
      : "Failed";
  }
  if (status === "completed") return "Completed";
  if (status === "finalizing" && pendingSaves > 0) {
    return completedSaves > 0
      ? `Saving output · ${completedSaves} completed · ${pendingSaves} pending`
      : `Saving output · ${pendingSaves} pending`;
  }
  if (status === "queued") return "Queued";
  if (status === "cancelling") return "Cancelling";
  if (progress.step > 0 && progress.total > 0) return `Step ${progress.step} / ${progress.total}`;
  return currentJob?.request?.unlimited ? "Generating forever" : "Generating";
}

function currentPreviewSource() {
  return state.livePreview.frameUrl || state.livePreview.latestFrameUrl || state.livePreview.finalOutputUrl || "";
}

function applyPreviewGeometry(stage, image) {
  const width = Number(state.livePreview.frameWidth || image?.naturalWidth || 0);
  const height = Number(state.livePreview.frameHeight || image?.naturalHeight || 0);
  const known = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0;
  stage.classList.toggle("has-known-aspect", known);
  if (!known) {
    stage.style.removeProperty("--live-preview-aspect-ratio");
    stage.style.removeProperty("--live-preview-aspect-value");
    return;
  }
  state.livePreview.frameWidth = width;
  state.livePreview.frameHeight = height;
  stage.style.setProperty("--live-preview-aspect-ratio", `${width} / ${height}`);
  stage.style.setProperty("--live-preview-aspect-value", String(width / height));
}

function openCurrentPreviewLightbox(opener) {
  const frameUrl = currentPreviewSource();
  if (!frameUrl) return;
  openLiveLightbox({
    ...state.livePreview,
    frameUrl,
    frameName: state.livePreview.frameName || (state.livePreview.finalOutputUrl === frameUrl ? "Final output" : "Live preview"),
    decodeMode: state.livePreview.decodeMode || (state.livePreview.finalOutputUrl === frameUrl ? "final" : "preview"),
  }, { opener });
}

function renderCurrent() {
  const panel = $("#livePreviewPanel");
  const stage = $("#livePreviewStage");
  const image = $("#livePreviewImage");
  const placeholder = $("#livePreviewPlaceholder");
  const track = $("#liveProgressTrack");
  const fill = $("#liveProgressFill");
  const openLarge = $("#livePreviewOpenLargeButton");
  if (!panel) return;

  const idle = !currentJob;
  panel.classList.toggle("is-idle", idle);
  $("#livePreviewIdleLine").textContent = idle
    ? "No active or recent generation."
    : `${shortText(state.livePreview.prompt || "Untitled generation", 52)}`;
  renderTransport();
  renderLiveCfgVisual();

  if (idle) {
    stage.classList.remove("has-image");
    stage.disabled = true;
    openLarge.disabled = true;
    image.removeAttribute("src");
    state.livePreview.frameWidth = 0;
    state.livePreview.frameHeight = 0;
    applyPreviewGeometry(stage, image);
    $("#livePreviewStep").textContent = "Idle";
    $("#livePreviewPercent").textContent = "—";
    track.classList.remove("is-running", "is-indeterminate");
    fill.style.width = "0%";
    return;
  }

  const frameUrl = currentPreviewSource();
  stage.classList.toggle("has-image", Boolean(frameUrl));
  stage.disabled = !frameUrl;
  openLarge.disabled = !frameUrl;
  if (frameUrl) {
    if (image.getAttribute("src") !== frameUrl) image.src = frameUrl;
  } else {
    image.removeAttribute("src");
    const previewSuspended = state.livePreview.imageDecodeSuspended === true;
    placeholder.querySelector("strong").textContent = previewSuspended
      ? "Image preview suspended"
      : state.livePreview.status === "queued"
        ? "Generation queued"
        : "Waiting for first frame";
    placeholder.querySelector("span").textContent = previewSuspended
      ? `${state.livePreview.imageDecodeSuspensionReason || "Preview decoding was suspended to preserve VRAM."} CFG telemetry and generation progress continue.`
      : state.livePreview.status === "queued"
        ? "The preview panel will start updating when sampling begins."
        : "The first decoded frame will appear shortly.";
  }

  applyPreviewGeometry(stage, image);

  const progress = {
    step: state.livePreview.step,
    total: state.livePreview.totalSteps,
    percent: state.livePreview.progress,
  };
  $("#livePreviewStep").textContent = statusText(progress);
  $("#livePreviewPercent").textContent = progress.percent == null ? "—" : `${Math.round(progress.percent)}%`;
  renderSamplingTiming();
  $("#livePreviewJobStatus").textContent = `Status: ${state.livePreview.status || "idle"}`;
  $("#livePreviewModel").textContent = `Model: ${state.livePreview.modelName || "—"}`;
  $("#livePreviewSampler").textContent = `Sampler: ${state.livePreview.samplerName || "—"}`;
  $("#livePreviewScheduler").textContent = `Scheduler: ${state.livePreview.schedulerName || "—"}`;
  $("#livePreviewSeed").textContent = `Seed: ${state.livePreview.seed ?? "—"}`;
  const finalOutputDisplayed = Boolean(
    state.livePreview.status === "completed"
    && state.livePreview.finalOutputUrl
    && currentPreviewSource() === state.livePreview.finalOutputUrl
  );
  const suspensionSource = String(state.livePreview.imageDecodeSuspensionSource || "")
    .trim()
    .replaceAll("_", " ");
  const suspensionSuffix = suspensionSource ? ` (${suspensionSource})` : "";
  $("#livePreviewDecodeMode").textContent = state.livePreview.imageDecodeSuspended
    ? finalOutputDisplayed
      ? `Decode: Final · step image previews were suspended${suspensionSuffix}`
      : `Decode: Suspended${suspensionSuffix} · CFG telemetry continues${state.livePreview.previewDecoderReleased ? " · decoder released" : ""}`
    : `Decode: ${state.livePreview.decodeMode || "—"}${state.livePreview.selectedStep ? ` · preview step ${state.livePreview.selectedStep}` : ""}`;
  $("#livePreviewDecodeMode").title = state.livePreview.imageDecodeSuspended
    ? String(state.livePreview.imageDecodeSuspensionReason || "Image-preview decoding was suspended for this job.")
    : "";
  updateElapsed();

  track.classList.toggle("is-running", isActive(state.livePreview.status));
  track.classList.toggle("is-indeterminate", progress.percent == null && isActive(state.livePreview.status));
  fill.style.width = progress.percent == null ? (isActive(state.livePreview.status) ? "36%" : "0%") : `${progress.percent}%`;

  const follow = $("#livePreviewFollowButton");
  const pause = $("#livePreviewPauseButton");
  const jump = $("#livePreviewJumpLatestButton");
  const viewFinal = $("#livePreviewViewFinalButton");
  follow.setAttribute("aria-pressed", String(state.livePreview.followLatest));
  const followLabel = state.livePreview.followLatest ? "Following latest preview" : "Follow latest preview";
  setActionIcon(follow, "follow-latest", { label: followLabel, title: followLabel });
  pause.setAttribute("aria-pressed", String(state.livePreview.updatesPaused));
  const pauseLabel = state.livePreview.updatesPaused ? "Resume live preview" : "Pause live preview";
  setActionIcon(pause, state.livePreview.updatesPaused ? "play" : "pause", { label: pauseLabel, title: pauseLabel });
  pause.disabled = !state.livePreview.latestFrameUrl || isTerminal(state.livePreview.status);
  const behindLatest = Boolean(
    state.livePreview.latestFrameUrl
    && (state.livePreview.frameUrl !== state.livePreview.latestFrameUrl || !state.livePreview.followLatest || state.livePreview.updatesPaused)
  );
  jump.classList.toggle("is-hidden", !behindLatest);
  viewFinal.classList.toggle("is-hidden", state.livePreview.status !== "completed" || !state.livePreview.finalOutputUrl);

  const diagnostics = $("#livePreviewDiagnosticsLink");
  const consoleLink = $("#livePreviewConsoleLink");
  const cancel = $("#livePreviewCancelButton");
  const showDiagnostics = ["failed", "cancelled"].includes(state.livePreview.status);
  diagnostics.classList.toggle("is-hidden", !showDiagnostics);
  diagnostics.href = currentJob ? `/api/jobs/${encodeURIComponent(currentJob.job_id)}/diagnostics` : "";
  consoleLink.classList.toggle("is-hidden", !showDiagnostics || !currentJob?.console_log_path);
  consoleLink.href = currentJob ? `/api/jobs/${encodeURIComponent(currentJob.job_id)}/log` : "";
  cancel.classList.toggle("is-hidden", !isActive(state.livePreview.status));
  cancel.disabled = state.livePreview.status === "cancelling";

  renderMemoryStatus(currentJob);
  renderHistory();
  syncLiveLightbox();
}

export function renderLivePreviewJob(job, { forceLatest = false } = {}) {
  if (!job) {
    currentJob = null;
    state.livePreview.jobId = null;
    state.livePreview.status = "idle";
    state.livePreview.streamStatus = "idle";
    state.livePreview.cfgPoints = [];
    state.livePreview.requestedCfgScale = null;
    state.livePreview.effectiveCfgScale = null;
    state.livePreview.guidanceMode = "";
    state.livePreview.imageDecodeSuspended = false;
    state.livePreview.imageDecodeSuspensionReason = "";
    state.livePreview.imageDecodeSuspensionSource = "";
    state.livePreview.previewDecoderReleased = false;
    state.livePreview.samplingTiming = {};
    renderCurrent();
    return;
  }

  const newJob = state.livePreview.jobId !== job.job_id;
  currentJob = job;
  if (newJob) {
    state.livePreview.followLatest = true;
    state.livePreview.updatesPaused = false;
    state.livePreview.selectedStep = null;
    state.livePreview.frameUrl = "";
    state.livePreview.frameWidth = 0;
    state.livePreview.frameHeight = 0;
    state.livePreview.history = [];
    state.livePreview.cfgPoints = [];
    state.livePreview.requestedCfgScale = null;
    state.livePreview.effectiveCfgScale = null;
    state.livePreview.guidanceMode = "";
    state.livePreview.samplingTiming = {};
    state.livePreview.imageDecodeSuspended = false;
    state.livePreview.imageDecodeSuspensionReason = "";
    state.livePreview.imageDecodeSuspensionSource = "";
    state.livePreview.previewDecoderReleased = false;
  }

  const progress = numericProgress(job);
  const history = normalizedHistory(job);
  const latest = latestHistoryRecord(job);
  const latestUrl = job.live_preview_url || latest?.preview_url || "";
  const resolvedSeed = job.resolved_seed ?? (
    Number.isFinite(Number(job.request?.seed)) && Number(job.request.seed) >= 0
      ? Number(job.request.seed)
      : null
  );

  state.livePreview.jobId = job.job_id;
  state.livePreview.prompt = job.request?.positive_prompt || "";
  state.livePreview.seed = resolvedSeed;
  state.livePreview.step = progress.step;
  state.livePreview.totalSteps = progress.total;
  state.livePreview.progress = progress.percent;
  state.livePreview.status = job.status || "queued";
  state.livePreview.startedAt = job.started_at || job.created_at || null;
  state.livePreview.completedAt = job.completed_at || null;
  state.livePreview.modelName = modelName(job) || "";
  state.livePreview.samplerName = job.request?.sampler_name || "";
  state.livePreview.schedulerName = job.request?.scheduler_name || "";
  state.livePreview.decodeMode = job.live_preview_decode_mode || latest?.decode_mode || state.livePreview.decodeMode || "";
  state.livePreview.history = history;
  state.livePreview.metrics = { ...(job.live_preview_metrics || {}) };
  state.livePreview.samplingTiming = { ...(job.sampling_timing || {}) };
  const previewMemoryStatus = job.memory_status || {};
  state.livePreview.imageDecodeSuspended = Boolean(
    state.livePreview.metrics.image_decode_suspended
    || previewMemoryStatus.preview_image_decode_suspended
    || latest?.preview_image_suspended
  );
  state.livePreview.imageDecodeSuspensionReason = String(
    state.livePreview.metrics.image_decode_suspension_reason
    || previewMemoryStatus.preview_image_decode_suspension_reason
    || latest?.preview_image_suspension_reason
    || ""
  );
  state.livePreview.imageDecodeSuspensionSource = String(
    state.livePreview.metrics.image_decode_suspension_source
    || previewMemoryStatus.preview_image_decode_suspension_source
    || latest?.preview_image_suspension_source
    || ""
  );
  state.livePreview.previewDecoderReleased = Boolean(
    state.livePreview.metrics.preview_decoder_released
    || previewMemoryStatus.preview_decoder_released
    || latest?.preview_decoder_released
  );
  state.livePreview.cfgPoints = Array.isArray(job.live_cfg_step_series?.points)
    ? job.live_cfg_step_series.points.map((point) => ({ ...point }))
    : [];
  const latestCfg = state.livePreview.cfgPoints[state.livePreview.cfgPoints.length - 1] || latest || null;
  state.livePreview.requestedCfgScale = Number.isFinite(Number(latestCfg?.requested_cfg_scale))
    ? Number(latestCfg.requested_cfg_scale)
    : null;
  state.livePreview.effectiveCfgScale = Number.isFinite(Number(latestCfg?.effective_cfg_scale))
    ? Number(latestCfg.effective_cfg_scale)
    : null;
  state.livePreview.guidanceMode = latestCfg?.guidance_mode || "";
  state.livePreview.cfgRescale = Number(latestCfg?.cfg_rescale || job.request?.cfg_rescale || 0);
  state.livePreview.latestFrameUrl = latestUrl;
  state.livePreview.latestStep = Number(latest?.step ?? progress.step ?? 0);
  if (latest) {
    state.livePreview.frameWidth = Number(latest.image_width || state.livePreview.frameWidth || 0);
    state.livePreview.frameHeight = Number(latest.image_height || state.livePreview.frameHeight || 0);
  }
  state.livePreview.finalOutputUrl = job.final_output_url || (latest?.is_final ? latestUrl : "");

  const forceTerminalFinal = state.livePreview.status === "completed" && Boolean(state.livePreview.finalOutputUrl);
  if (forceTerminalFinal) {
    state.livePreview.frameUrl = state.livePreview.finalOutputUrl;
    state.livePreview.frameName = "Final output";
    state.livePreview.selectedStep = state.livePreview.latestStep || progress.step || null;
    state.livePreview.decodeMode = "final";
  } else if (
    latest
    && (newJob || forceLatest || (state.livePreview.followLatest && !state.livePreview.updatesPaused))
  ) {
    displayFrame(latest);
  } else if (!state.livePreview.frameUrl && latestUrl) {
    state.livePreview.frameUrl = latestUrl;
    state.livePreview.selectedStep = state.livePreview.latestStep || null;
  }
  renderCurrent();
}

export function setLivePreviewTransport(status, reconnectCount = state.livePreview.reconnectCount) {
  state.livePreview.streamStatus = status || "fallback";
  state.livePreview.reconnectCount = Math.max(0, Number(reconnectCount) || 0);
  renderTransport();
}

export function bindLivePreview(options = {}) {
  cancelGeneration = options.cancelActive || cancelGeneration;
  const previewImage = $("#livePreviewImage");
  previewImage?.addEventListener("load", () => {
    if (previewImage.naturalWidth > 0 && previewImage.naturalHeight > 0) {
      state.livePreview.frameWidth = previewImage.naturalWidth;
      state.livePreview.frameHeight = previewImage.naturalHeight;
      applyPreviewGeometry($("#livePreviewStage"), previewImage);
    }
  });
  previewImage?.addEventListener("error", () => {
    $("#livePreviewStage")?.classList.remove("has-image");
  });
  $("#livePreviewStage").addEventListener("click", (event) => {
    openCurrentPreviewLightbox(event.currentTarget);
  });
  $("#livePreviewOpenLargeButton").addEventListener("click", (event) => {
    openCurrentPreviewLightbox(event.currentTarget);
  });
  $("#livePreviewFollowButton").addEventListener("click", () => {
    state.livePreview.followLatest = !state.livePreview.followLatest;
    state.lightbox.followLatest = state.livePreview.followLatest;
    if (state.livePreview.followLatest) jumpToLatest();
    else renderCurrent();
  });
  $("#livePreviewPauseButton").addEventListener("click", () => {
    state.livePreview.updatesPaused = !state.livePreview.updatesPaused;
    if (!state.livePreview.updatesPaused && state.livePreview.followLatest) jumpToLatest();
    else renderCurrent();
  });
  $("#livePreviewJumpLatestButton").addEventListener("click", jumpToLatest);
  $("#livePreviewViewFinalButton").addEventListener("click", (event) => {
    if (!state.livePreview.finalOutputUrl) return;
    openLiveLightbox({
      ...state.livePreview,
      frameUrl: state.livePreview.finalOutputUrl,
      frameName: "Final output",
      decodeMode: "final",
    }, { opener: event.currentTarget });
  });
  $("#livePreviewCancelButton").addEventListener("click", () => cancelGeneration());
  window.addEventListener("live-preview-cfg-visual-setting-changed", () => {
    renderCurrent();
  });
  window.addEventListener("live-preview-follow-changed", (event) => {
    state.livePreview.followLatest = event.detail?.followLatest !== false;
    if (state.livePreview.followLatest && !state.livePreview.updatesPaused) jumpToLatest();
    else renderCurrent();
  });
  if (!elapsedTimer) elapsedTimer = window.setInterval(updateElapsed, 1000);
  renderCurrent();
}

export function stopLivePreview() {
  if (elapsedTimer) window.clearInterval(elapsedTimer);
  elapsedTimer = null;
}
