import { api } from "../api.js";
import { productName } from "../branding.js?v=brand1";
import { state } from "../state.js";
import { $, shortText, notify } from "../utils.js";

let collectValues = () => ({});
let onJobQueued = () => {};

const EDITABLE_FIELDS = [
  "positive_prompt", "negative_prompt", "seed", "width", "height", "steps",
  "cfg_scale", "batch_size", "batch_count", "model_path", "vae_path", "sd2_runtime_profile_override", "sd3_runtime_profile_override", "sd3_text_encoder_source", "sd3_t5_enabled", "sd3_t5_source", "advanced_models_enabled", "advanced_model_family", "advanced_model_components", "advanced_model_allow_digital_components", "advanced_model_composition_sha256", "advanced_model_t5_device", "text_encoder_3_device", "model_enforce_recommended_steps", "model_enforce_recommended_cfg", "sd2_dedicated_generation", "sampler_name", "scheduler_name",
];

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = value ?? "";
  node.textContent = label ?? value ?? "";
  return node;
}

function fillSelect(select, values, { path = false, empty = "No common replacement" } = {}) {
  if (!select) return;
  select.replaceChildren(option("", empty));
  for (const item of values || []) {
    const value = path ? String(item.path || item.value || item.name || "") : String(item.name || item.plugin_id || "");
    if (!value) continue;
    select.append(option(value, item.label || item.name || value));
  }
}

function downloadBlob(result) {
  const url = URL.createObjectURL(result.blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = result.filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  if (result.warnings?.length) notify(result.warnings[0]);
  else notify(`Exported ${result.jobCount} job${result.jobCount === 1 ? "" : "s"} to ${result.filename}.`);
}

export async function exportJobs(jobs, format = "native", filenameStem = "image_gen_queue") {
  if (!jobs?.length) throw new Error("There are no jobs to export.");
  const result = await api.exportBatch({
    format,
    filename_stem: filenameStem,
    source: `${productName()} WebUI`,
    jobs,
  });
  downloadBlob(result);
  return result;
}

function selectedFormat() {
  return $("#batchExportFormat")?.value || "native";
}

function resetState() {
  state.batchIO.open = true;
  state.batchIO.selectedFile = null;
  state.batchIO.detectedFormat = "";
  state.batchIO.defaultsPolicy = "file_only";
  state.batchIO.parseResult = null;
  state.batchIO.editedJobs = {};
  state.batchIO.selectedJobIds = [];
  state.batchIO.order = [];
  state.batchIO.commonRemap = {};
  state.batchIO.itemRemaps = {};
  state.batchIO.preflight = null;
  state.batchIO.loading = false;
  state.batchIO.error = "";
}

function setError(message = "") {
  state.batchIO.error = message;
  const node = $("#batchImportError");
  if (!node) return;
  node.textContent = message;
  node.hidden = !message;
}

function setStage(stage) {
  $("#batchImportFileStage").hidden = stage !== "file";
  $("#batchImportJobsStage").hidden = stage !== "jobs";
  $("#batchImportValidationStage").hidden = stage !== "validation";
  $("#batchImportParseButton").hidden = stage !== "file";
  $("#batchImportValidateButton").hidden = stage !== "jobs";
  $("#batchImportBackButton").hidden = stage === "file";
  $("#batchImportQueueAllButton").hidden = stage !== "validation";
  $("#batchImportQueueValidButton").hidden = stage !== "validation";
  $("#batchImportVariationButton").hidden = stage !== "validation";
}

function jobRequest(job) {
  return state.batchIO.editedJobs[job.job_id] || structuredClone(job.normalized || job.request || {});
}

function inputFor(job, field) {
  const request = jobRequest(job);
  const input = field.includes("prompt") ? document.createElement("textarea") : document.createElement("input");
  input.dataset.jobId = job.job_id;
  input.dataset.importField = field;
  if (field.includes("prompt")) input.rows = field === "positive_prompt" ? 2 : 1;
  else if (["seed", "width", "height", "steps", "batch_size", "batch_count"].includes(field)) {
    input.type = "number";
    input.step = "1";
  } else if (field === "cfg_scale") {
    input.type = "number";
    input.step = "0.1";
  } else input.type = "text";
  input.value = request[field] ?? "";
  input.addEventListener("input", () => {
    const edited = state.batchIO.editedJobs[job.job_id] || structuredClone(job.normalized || {});
    let value = input.value;
    if (["seed", "width", "height", "steps", "batch_size", "batch_count"].includes(field) && value !== "") value = Number.parseInt(value, 10);
    if (field === "cfg_scale" && value !== "") value = Number.parseFloat(value);
    edited[field] = value;
    state.batchIO.editedJobs[job.job_id] = edited;
    state.batchIO.preflight = null;
  });
  return input;
}

function renderParsedJobs() {
  const result = state.batchIO.parseResult;
  const body = $("#batchImportJobsBody");
  body.replaceChildren();
  for (const job of result?.jobs || []) {
    const row = document.createElement("tr");
    row.dataset.jobId = job.job_id;
    if (job.errors?.length) row.className = "is-invalid";
    const selected = document.createElement("input");
    selected.type = "checkbox";
    selected.checked = state.batchIO.selectedJobIds.includes(job.job_id);
    selected.setAttribute("aria-label", `Select ${job.source_label}`);
    selected.addEventListener("change", () => {
      state.batchIO.selectedJobIds = selected.checked
        ? [...new Set([...state.batchIO.selectedJobIds, job.job_id])]
        : state.batchIO.selectedJobIds.filter((id) => id !== job.job_id);
    });
    const selectCell = document.createElement("td");
    selectCell.append(selected);
    const sourceCell = document.createElement("td");
    sourceCell.textContent = job.source_label;
    const fieldCells = EDITABLE_FIELDS.map((field) => {
      const cell = document.createElement("td");
      cell.append(inputFor(job, field));
      return cell;
    });
    const statusCell = document.createElement("td");
    statusCell.textContent = [...(job.errors || []), ...(job.warnings || [])].join("; ") || "Parsed";
    const actionsCell = document.createElement("td");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text-button danger-text";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.batchIO.parseResult.jobs = state.batchIO.parseResult.jobs.filter((item) => item.job_id !== job.job_id);
      state.batchIO.selectedJobIds = state.batchIO.selectedJobIds.filter((id) => id !== job.job_id);
      state.batchIO.order = state.batchIO.order.filter((id) => id !== job.job_id);
      delete state.batchIO.editedJobs[job.job_id];
      renderParsedJobs();
    });
    actionsCell.append(remove);
    row.append(selectCell, sourceCell, ...fieldCells, statusCell, actionsCell);
    body.append(row);
  }
  $("#batchImportParsedSummary").textContent = `${result?.job_count || 0} parsed · ${result?.valid_parse_count || 0} without parse errors · ${result?.invalid_parse_count || 0} with parse errors`;
  $("#batchImportValidateButton").disabled = !state.batchIO.selectedJobIds.length;
}

function preflightPayload() {
  const jobs = (state.batchIO.parseResult?.jobs || []).map((job) => ({
    ...job,
    edited: state.batchIO.editedJobs[job.job_id] || job.normalized,
  }));
  const commonRemap = {
    model_path: $("#batchImportCommonModel").value,
    vae_path: $("#batchImportCommonVae").value,
    sampler_name: $("#batchImportCommonSampler").value,
    scheduler_name: $("#batchImportCommonScheduler").value,
  };
  Object.keys(commonRemap).forEach((key) => { if (!commonRemap[key]) delete commonRemap[key]; });
  state.batchIO.commonRemap = commonRemap;
  return {
    jobs,
    order: state.batchIO.order,
    selected_job_ids: state.batchIO.selectedJobIds,
    common_remap: commonRemap,
    item_remaps: state.batchIO.itemRemaps,
  };
}

function renderValidation(data) {
  state.batchIO.preflight = data;
  $("#batchImportValidationSummary").textContent = `${data.summary?.valid_selected_count || 0} selected valid · ${data.summary?.invalid_selected_count || 0} selected invalid`;
  const body = $("#batchImportValidationBody");
  body.replaceChildren();
  for (const job of data.jobs || []) {
    const row = document.createElement("tr");
    row.className = job.valid ? "is-valid" : "is-invalid";
    const values = [
      job.source_label,
      shortText(job.summary?.prompt || job.job_id, 64),
      job.summary?.seed ?? "—",
      shortText(job.summary?.model_path || "—", 36),
      job.summary?.sampler_name || "—",
      job.summary?.scheduler_name || "—",
      job.valid ? "Valid" : (job.errors || []).join("; "),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
  $("#batchImportQueueAllButton").disabled = !data.valid || !data.preflight_token;
  $("#batchImportQueueValidButton").disabled = !(data.summary?.valid_selected_count > 0 && data.preflight_token);
  $("#batchImportVariationButton").disabled = !(data.summary?.valid_selected_count > 0 && data.preflight_token);
  setStage("validation");
}

async function parseSelectedFile() {
  const file = $("#batchImportFile").files?.[0];
  if (!file) {
    setError("Choose a native JSON, JSONL, or CSV file first.");
    return;
  }
  setError("");
  state.batchIO.loading = true;
  $("#batchImportParseButton").disabled = true;
  try {
    const result = await api.parseBatchImport(file, {
      formatHint: $("#batchImportFormat").value,
      defaultsPolicy: $("#batchImportDefaultsPolicy").value,
      currentValues: collectValues(),
    });
    state.batchIO.selectedFile = file;
    state.batchIO.detectedFormat = result.format;
    state.batchIO.defaultsPolicy = $("#batchImportDefaultsPolicy").value;
    state.batchIO.parseResult = result;
    if (!(result.jobs || []).length && (result.errors || []).length) {
      setError(result.errors.join(" "));
      return;
    }
    state.batchIO.selectedJobIds = (result.jobs || []).map((job) => job.job_id);
    state.batchIO.order = [...state.batchIO.selectedJobIds];
    state.batchIO.editedJobs = {};
    renderParsedJobs();
    setStage("jobs");
    if ((result.errors || []).length) setError(result.errors.join(" "));
    $("#batchImportLiveRegion").textContent = `Parsed ${result.job_count} jobs from ${file.name}.`;
  } catch (error) {
    setError(error.message);
  } finally {
    state.batchIO.loading = false;
    $("#batchImportParseButton").disabled = false;
  }
}

async function runValidation() {
  setError("");
  state.batchIO.loading = true;
  $("#batchImportValidateButton").disabled = true;
  try {
    renderValidation(await api.preflightBatchImport(preflightPayload()));
  } catch (error) {
    setError(error.message);
  } finally {
    state.batchIO.loading = false;
    $("#batchImportValidateButton").disabled = false;
  }
}

async function submitImport(queueValidOnly) {
  const token = state.batchIO.preflight?.preflight_token;
  if (!token) return;
  setError("");
  try {
    const response = await api.submitBatchImport(token, queueValidOnly);
    for (const job of response.submitted || []) onJobQueued(job);
    notify(`${response.submitted_count} imported job${response.submitted_count === 1 ? "" : "s"} queued${response.rejected_count ? `; ${response.rejected_count} rejected` : ""}.`);
    closeImportDialog();
  } catch (error) {
    setError(error.message);
  }
}

function openImportDialog() {
  resetState();
  fillSelect($("#batchImportCommonModel"), state.models, { path: true });
  fillSelect($("#batchImportCommonVae"), state.vaes, { path: true });
  fillSelect($("#batchImportCommonSampler"), state.samplers);
  fillSelect($("#batchImportCommonScheduler"), state.schedulers);
  $("#batchImportFile").value = "";
  $("#batchImportFormat").value = "";
  $("#batchImportDefaultsPolicy").value = "file_only";
  setError("");
  setStage("file");
  $("#batchImportDialog").showModal();
}

function closeImportDialog() {
  state.batchIO.open = false;
  if ($("#batchImportDialog").open) $("#batchImportDialog").close();
}

async function exportCurrentForm() {
  await exportJobs([{ request: collectValues() }], selectedFormat(), "current_generation_request");
}

async function exportSelectedGallery() {
  const outputIds = state.gallerySelection.outputIds || [];
  if (!outputIds.length) throw new Error("Select at least one gallery output first.");
  const preflight = await api.batchReplayPreflight({
    output_ids: outputIds,
    order: outputIds,
    mode: "exact_or_override",
    overrides: {},
    override_fields: [],
    seed_policy: { mode: "keep_original" },
    common_remap: {},
    item_remaps: {},
  });
  const jobs = (preflight.jobs || []).map((job) => ({
    job_id: job.output_id,
    request: job.request,
    provenance: {
      replay_quality: job.completeness?.quality,
      replay_label: job.completeness?.label,
      metadata_source: job.summary?.metadata_source,
      manifest_version: job.completeness?.contract_version,
      applied_remaps: {},
    },
  }));
  await exportJobs(jobs, selectedFormat(), "selected_gallery_outputs");
}

async function exportInspectedOutput() {
  const details = state.outputDetails.data;
  if (!details?.replay) throw new Error("Open Image Details before exporting an inspected output.");
  await exportJobs([{
    job_id: details.output_id,
    request: details.replay,
    provenance: {
      replay_quality: details.completeness?.quality,
      replay_label: details.completeness?.label,
      metadata_source: details.metadata_source,
      manifest_version: details.manifest?.manifest_version,
      applied_remaps: {},
    },
  }], selectedFormat(), "inspected_output_request");
}

async function exportComposedQueue() {
  const jobs = state.queueComposer.preflight?.jobs || [];
  if (!jobs.length) throw new Error("Preview the composed queue before exporting it.");
  await exportJobs(jobs.map((job) => ({
    job_id: job.output_id,
    request: job.request,
    provenance: {
      replay_quality: job.completeness?.quality,
      replay_label: job.completeness?.label,
      metadata_source: job.summary?.metadata_source,
      manifest_version: job.completeness?.contract_version,
      applied_remaps: {
        ...(state.queueComposer.commonRemap || {}),
        ...(state.queueComposer.itemRemaps?.[job.output_id] || {}),
      },
    },
  })), selectedFormat(), "composed_queue");
}

function guarded(callback) {
  return async () => {
    try { await callback(); }
    catch (error) { notify(error.message, "error"); }
  };
}

export function bindBatchIO({ collect = () => ({}), onQueued = () => {} } = {}) {
  collectValues = collect;
  onJobQueued = onQueued;
  $("#importQueueButton")?.addEventListener("click", openImportDialog);
  $("#batchImportCloseButton")?.addEventListener("click", closeImportDialog);
  $("#batchImportCancelButton")?.addEventListener("click", closeImportDialog);
  $("#batchImportParseButton")?.addEventListener("click", parseSelectedFile);
  $("#batchImportValidateButton")?.addEventListener("click", runValidation);
  $("#batchImportBackButton")?.addEventListener("click", () => {
    if (!$("#batchImportValidationStage").hidden) setStage("jobs");
    else setStage("file");
  });
  $("#batchImportQueueAllButton")?.addEventListener("click", () => submitImport(false));
  $("#batchImportQueueValidButton")?.addEventListener("click", () => submitImport(true));
  $("#exportCurrentButton")?.addEventListener("click", guarded(exportCurrentForm));
  $("#exportSelectedOutputsButton")?.addEventListener("click", guarded(exportSelectedGallery));
  $("#outputDetailsExportButton")?.addEventListener("click", guarded(exportInspectedOutput));
  $("#queueComposerExportButton")?.addEventListener("click", guarded(exportComposedQueue));
}

export { EDITABLE_FIELDS };
