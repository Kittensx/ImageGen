import { api } from "../api.js";
import { state } from "../state.js";
import { $, notify, shortText } from "../utils.js";
import { setActionIcon } from "../components/action-icons.js?v=0.1.1";

let collectValues = () => ({});
let onJobQueued = () => {};

const CORE_FIELDS = [
  ["seed", "Seed"],
  ["model_path", "Model"],
  ["vae_path", "VAE"],
  ["width", "Width"],
  ["height", "Height"],
  ["steps", "Steps"],
  ["cfg_scale", "CFG scale"],
  ["sampler_name", "Sampler"],
  ["scheduler_name", "Scheduler"],
  ["positive_prompt", "Positive prompt"],
  ["negative_prompt", "Negative prompt"],
  ["batch_size", "Batch size"],
  ["batch_count", "Batch count"],
  ["cfg_rescale", "CFG rescale"],
  ["clip_skip", "CLIP skip"],
];
const INTEGER_FIELDS = new Set(["seed", "width", "height", "steps", "batch_size", "batch_count", "clip_skip"]);
const NUMBER_FIELDS = new Set(["cfg_scale", "cfg_rescale"]);

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function availableFields() {
  const fields = [...CORE_FIELDS];
  for (const descriptor of state.samplers || []) {
    for (const [name] of Object.entries(descriptor.config_schema?.properties || {})) {
      fields.push([`sampler_kwargs.${name}`, `Sampler advanced · ${name}`]);
    }
  }
  for (const descriptor of state.schedulers || []) {
    for (const [name] of Object.entries(descriptor.config_schema?.properties || {})) {
      fields.push([`scheduler_kwargs.${name}`, `Scheduler advanced · ${name}`]);
    }
  }
  return [...new Map(fields.map((item) => [item[0], item])).values()];
}

function resetVariationState({
  baseRequests = null,
  baseLineage = null,
  importPreflightToken = "",
  initialDimensions = null,
  recipeName = "Variation Matrix",
  mode = "cartesian",
  title = "Advanced Variation Matrix",
} = {}) {
  const current = collectValues();
  state.variationMatrix.open = true;
  state.variationMatrix.baseRequests = baseRequests?.length ? structuredClone(baseRequests) : [structuredClone(current)];
  state.variationMatrix.baseLineage = baseLineage?.length ? structuredClone(baseLineage) : [{
    source: "current_form",
    source_id: null,
    source_label: "Current generation form",
  }];
  state.variationMatrix.importPreflightToken = importPreflightToken || "";
  state.variationMatrix.dimensions = Array.isArray(initialDimensions) ? structuredClone(initialDimensions) : null;
  state.variationMatrix.mode = mode || "cartesian";
  state.variationMatrix.zipLengthPolicy = "reject";
  state.variationMatrix.baseMode = "apply_to_each";
  state.variationMatrix.deduplicate = true;
  state.variationMatrix.recipeName = recipeName || "Variation Matrix";
  state.variationMatrix.title = title || "Advanced Variation Matrix";
  state.variationMatrix.jobLimit = 250;
  state.variationMatrix.confirmLargePlan = false;
  state.variationMatrix.preflight = null;
  state.variationMatrix.error = "";
}

function setError(message = "") {
  state.variationMatrix.error = message;
  const node = $("#variationMatrixError");
  node.textContent = message;
  node.hidden = !message;
}

function setLoading(loading) {
  state.variationMatrix.loading = loading;
  ["#variationPreflightButton", "#variationSubmitButton", "#variationExportButton"].forEach((selector) => {
    const button = $(selector);
    if (!button) return;
    if (selector === "#variationExportButton") button.disabled = loading || !state.variationMatrix.preflight?.valid;
    else button.disabled = loading;
  });
}

function closeVariationMatrix() {
  state.variationMatrix.open = false;
  const dialog = $("#variationMatrixDialog");
  if (dialog.open) dialog.close();
}

function parseManualValues(field, text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) {
    const value = JSON.parse(trimmed);
    if (!Array.isArray(value)) throw new Error("JSON variation input must be an array.");
    return value;
  }
  const values = trimmed.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  if (INTEGER_FIELDS.has(field)) return values.map((item) => {
    const value = Number(item);
    if (!Number.isInteger(value)) throw new Error(`${field} values must be integers.`);
    return value;
  });
  if (NUMBER_FIELDS.has(field)) return values.map((item) => {
    const value = Number(item);
    if (!Number.isFinite(value)) throw new Error(`${field} values must be numeric.`);
    return value;
  });
  return values;
}

function readDimensionRow(row) {
  const field = row.querySelector("[data-variation-field]").value;
  const source = row.querySelector("[data-variation-source]").value;
  if (source === "range") {
    return {
      field,
      source,
      range: {
        start: row.querySelector("[data-variation-range-start]").value,
        stop: row.querySelector("[data-variation-range-stop]").value,
        step: row.querySelector("[data-variation-range-step]").value,
      },
    };
  }
  return {
    field,
    source: "manual",
    values: parseManualValues(field, row.querySelector("[data-variation-values]").value),
  };
}

function currentDimensions() {
  return [...document.querySelectorAll(".variation-dimension-row")].map(readDimensionRow);
}

function estimateCount() {
  try {
    const dimensions = currentDimensions();
    const mode = $("#variationMode").value;
    const baseCount = state.variationMatrix.importPreflightToken
      ? Number(state.batchIO.preflight?.summary?.valid_selected_count || 0)
      : state.variationMatrix.baseRequests.length;
    const counts = dimensions.map((item) => item.source === "range" ? null : item.values.length);
    let combinations = 1;
    if (mode === "cartesian") combinations = counts.reduce((total, count) => total * Math.max(1, count || 1), 1);
    else if (mode === "paired") combinations = Math.max(0, ...counts.filter((item) => item !== null));
    else combinations = 1 + counts.reduce((total, count) => total + (count || 0), 0);
    const text = counts.includes(null)
      ? `${baseCount} base request(s) · range count calculated by server`
      : `${baseCount} base request(s) × ${combinations} combination(s) = ${baseCount * combinations} job(s)`;
    $("#variationCountEstimate").textContent = text;
  } catch (error) {
    $("#variationCountEstimate").textContent = error.message;
  }
}

function renderDimensionRow(initial = {}) {
  const row = document.createElement("div");
  row.className = "variation-dimension-row";
  const field = document.createElement("select");
  field.dataset.variationField = "";
  for (const [value, label] of availableFields()) field.append(option(value, label));
  field.value = initial.field || "seed";

  const source = document.createElement("select");
  source.dataset.variationSource = "";
  source.append(option("manual", "Manual list"), option("range", "Numeric range"));
  source.value = initial.source || "manual";

  const manual = document.createElement("textarea");
  manual.dataset.variationValues = "";
  manual.rows = 3;
  manual.placeholder = "One complete value per line, or a JSON array";
  manual.value = (initial.values || []).map((item) => typeof item === "object" ? JSON.stringify(item) : item).join("\n");

  const range = document.createElement("div");
  range.className = "variation-range-grid";
  for (const [key, label, value] of [
    ["start", "Start", initial.range?.start ?? ""],
    ["stop", "Stop", initial.range?.stop ?? ""],
    ["step", "Step", initial.range?.step ?? "1"],
  ]) {
    const wrapper = document.createElement("label");
    const caption = document.createElement("span");
    caption.textContent = label;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = value;
    input.dataset[`variationRange${key[0].toUpperCase()}${key.slice(1)}`] = "";
    wrapper.append(caption, input);
    range.append(wrapper);
  }

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "ui-action-button ui-icon-control";
  setActionIcon(remove, "remove", { label: "Remove variation dimension", title: "Remove variation dimension", replace: true });
  remove.addEventListener("click", () => { row.remove(); estimateCount(); });

  const toggle = () => {
    const isRange = source.value === "range";
    manual.hidden = isRange;
    range.hidden = !isRange;
    estimateCount();
  };
  source.addEventListener("change", toggle);
  field.addEventListener("change", estimateCount);
  manual.addEventListener("input", estimateCount);
  range.addEventListener("input", estimateCount);
  row.append(field, source, manual, range, remove);
  $("#variationDimensions").append(row);
  toggle();
}

function renderBaseSummary() {
  const count = state.variationMatrix.importPreflightToken
    ? Number(state.batchIO.preflight?.summary?.valid_selected_count || 0)
    : state.variationMatrix.baseRequests.length;
  $("#variationBaseSummary").textContent = state.variationMatrix.importPreflightToken
    ? `${count} validated imported request(s) from Phase 10D preflight`
    : `${count} validated base request(s)`;
}

function renderPreflight(data) {
  state.variationMatrix.preflight = data;
  const banner = $("#variationPreflightSummary");
  banner.textContent = `${data.base_count} base request(s) × ${data.combination_count} combination(s) = ${data.total_job_count} final job(s)${data.removed_duplicate_count ? ` · ${data.removed_duplicate_count} duplicate(s) removed` : ""}`;
  banner.className = `variation-summary ${data.valid ? "is-valid" : "is-invalid"}`;
  const messages = [...(data.errors || []), ...(data.warnings || [])];
  $("#variationMessages").textContent = messages.join("\n");
  $("#variationMessages").hidden = messages.length === 0;

  const body = $("#variationPreviewBody");
  body.replaceChildren();
  for (const job of data.jobs || []) {
    const row = document.createElement("tr");
    if (!job.valid) row.classList.add("is-invalid");
    const values = Object.entries(job.variation_values || {}).map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · ") || "Baseline";
    const cells = [
      job.job_index,
      job.base_source_label || `Base ${job.base_index + 1}`,
      values,
      shortText(job.summary?.prompt || "", 80),
      job.summary?.seed ?? "—",
      (job.summary?.model_path || "").split(/[\\/]/).pop() || "—",
      job.summary?.sampler_name || "—",
      job.summary?.scheduler_name || "—",
      job.valid ? "Valid" : (job.errors || []).join("; "),
    ];
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    body.append(row);
  }
  $("#variationPreviewSection").hidden = false;
  $("#variationSubmitButton").disabled = false;
  $("#variationSubmitButton").textContent = data.total_job_count > 0 ? `Queue ${data.total_job_count} Job${data.total_job_count === 1 ? "" : "s"}` : "Queue Jobs";
  $("#variationExportButton").disabled = !(data.jobs || []).some((item) => item.valid);
}

function collectPlan() {
  return {
    base_requests: state.variationMatrix.importPreflightToken ? undefined : state.variationMatrix.baseRequests,
    base_lineage: state.variationMatrix.importPreflightToken ? undefined : state.variationMatrix.baseLineage,
    import_preflight_token: state.variationMatrix.importPreflightToken || undefined,
    recipe_name: $("#variationRecipeName").value,
    mode: $("#variationMode").value,
    zip_length_policy: $("#variationZipPolicy").value,
    base_mode: $("#variationBaseMode").value,
    deduplicate: $("#variationDeduplicate").checked,
    job_limit: Number($("#variationJobLimit").value || 250),
    confirm_large_plan: $("#variationConfirmLarge").checked,
    dimensions: currentDimensions(),
  };
}

async function runPreflight() {
  setError("");
  setLoading(true);
  try {
    const data = await api.variationPreflight(collectPlan());
    renderPreflight(data);
    if (data.valid) notify(`Variation matrix validated ${data.total_job_count} job(s).`);
  } catch (error) {
    state.variationMatrix.preflight = null;
    setError(error.message);
  } finally {
    setLoading(false);
  }
}

async function submitVariations() {
  setLoading(true);
  try {
    // Queue is intentionally one-click. Backend preflight remains authoritative,
    // but the user does not have to manufacture a separate "validated job" first.
    const preflight = await api.variationPreflight(collectPlan());
    renderPreflight(preflight);
    if (!preflight.valid || !preflight.preflight_token) {
      setError((preflight.errors || []).join("\n") || "The variation plan contains invalid jobs.");
      return;
    }
    const response = await api.submitVariations(preflight.preflight_token);
    for (const job of response.submitted || []) onJobQueued(job);
    notify(`${response.submitted_count} variation job${response.submitted_count === 1 ? "" : "s"} queued${response.rejected_count ? `; ${response.rejected_count} rejected` : ""}.`);
    closeVariationMatrix();
  } catch (error) {
    setError(error.message);
  } finally {
    setLoading(false);
  }
}

async function exportVariations() {
  const token = state.variationMatrix.preflight?.preflight_token;
  if (!token) return;
  try {
    const result = await api.exportVariations(token, $("#variationExportFormat").value);
    const url = URL.createObjectURL(result.blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = result.filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    notify(`Exported ${result.jobCount} variation job(s).`);
  } catch (error) {
    setError(error.message);
  }
}

export function openVariationMatrix(options = {}) {
  const dialog = $("#variationMatrixDialog");
  if (!dialog) {
    notify("Variation Matrix is unavailable because its dialog was not loaded.", "error");
    return false;
  }
  resetVariationState(options);
  setError("");
  $("#variationMatrixTitle").textContent = state.variationMatrix.title || "Advanced Variation Matrix";
  $("#variationRecipeName").value = state.variationMatrix.recipeName;
  $("#variationMode").value = state.variationMatrix.mode || "cartesian";
  $("#variationZipPolicy").value = "reject";
  $("#variationBaseMode").value = "apply_to_each";
  $("#variationDeduplicate").checked = true;
  $("#variationJobLimit").value = "250";
  $("#variationConfirmLarge").checked = false;
  $("#variationDimensions").replaceChildren();
  const initialDimensions = state.variationMatrix.dimensions;
  if (Array.isArray(initialDimensions)) {
    initialDimensions.forEach((dimension) => renderDimensionRow(dimension));
  } else {
    renderDimensionRow({ field: "seed", values: [] });
  }
  renderBaseSummary();
  $("#variationPreviewSection").hidden = true;
  $("#variationMessages").hidden = true;
  $("#variationPreflightSummary").textContent = "Preview checks every expanded request. Queue also validates automatically before submission.";
  $("#variationSubmitButton").disabled = false;
  $("#variationSubmitButton").textContent = "Queue Jobs";
  $("#variationExportButton").disabled = true;
  estimateCount();
  if (!dialog.open) dialog.showModal();
  return true;
}

function openFromOutputDetails() {
  const details = state.outputDetails.data;
  if (!details?.replay) return;
  $("#outputDetailsDialog")?.close();
  openVariationMatrix({
    baseRequests: [details.replay],
    baseLineage: [{ source: "output_details", source_id: details.output_id, source_label: details.image?.name || details.output_id }],
  });
}

function openFromQueueComposer() {
  const jobs = (state.queueComposer.preflight?.jobs || []).filter((item) => item.valid);
  if (!jobs.length) return;
  $("#queueComposerDialog")?.close();
  openVariationMatrix({
    baseRequests: jobs.map((item) => item.request),
    baseLineage: jobs.map((item) => ({ source: "queue_composer", source_id: item.output_id, source_label: item.output_id })),
  });
}

function openFromImport() {
  const token = state.batchIO.preflight?.preflight_token;
  if (!token) return;
  $("#batchImportDialog")?.close();
  openVariationMatrix({ importPreflightToken: token });
}

export function bindVariationMatrix({ collect = () => ({}), onQueued = () => {} } = {}) {
  collectValues = collect;
  onJobQueued = onQueued;
  const toolbarButton = $("#openVariationMatrixButton");
  if (toolbarButton && toolbarButton.dataset.variationMatrixBound !== "true") {
    toolbarButton.dataset.variationMatrixBound = "true";
    toolbarButton.addEventListener("click", () => openVariationMatrix());
  }
  $("#outputDetailsVariationButton")?.addEventListener("click", openFromOutputDetails);
  $("#queueComposerVariationButton")?.addEventListener("click", openFromQueueComposer);
  $("#batchImportVariationButton")?.addEventListener("click", openFromImport);
  $("#variationCloseButton")?.addEventListener("click", closeVariationMatrix);
  $("#variationCancelButton")?.addEventListener("click", closeVariationMatrix);
  $("#variationAddDimensionButton")?.addEventListener("click", () => { renderDimensionRow(); estimateCount(); });
  $("#variationPreflightButton")?.addEventListener("click", runPreflight);
  $("#variationSubmitButton")?.addEventListener("click", submitVariations);
  $("#variationExportButton")?.addEventListener("click", exportVariations);
  ["#variationMode", "#variationBaseMode", "#variationZipPolicy"].forEach((selector) => $(selector)?.addEventListener("change", estimateCount));
}
