import { api } from "../api.js";
import { state } from "../state.js";
import { $, shortText, notify } from "../utils.js";

const CORE_OVERRIDE_FIELDS = [
  "model_path", "vae_path", "sd2_runtime_profile_override", "sd2_dedicated_generation", "width", "height", "steps", "cfg_scale",
  "sampler_name", "scheduler_name", "batch_count",
];

let onJobQueued = () => {};

function outputById(outputId) {
  return state.recentOutputs.find((item) => String(item.output_id || item.name) === outputId) || null;
}

function resetComposerFromSelection() {
  state.queueComposer.open = true;
  state.queueComposer.order = [...state.gallerySelection.outputIds];
  state.queueComposer.overrides = {};
  state.queueComposer.overrideFields = [];
  state.queueComposer.seedPolicy = { mode: "keep_original", start: 1 };
  state.queueComposer.commonRemap = {};
  state.queueComposer.itemRemaps = {};
  state.queueComposer.preflight = null;
  state.queueComposer.error = "";
}

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = value ?? "";
  node.textContent = label ?? value ?? "";
  return node;
}

function descriptorValue(item) {
  return String(item?.name || item?.plugin_id || "");
}

function fillSelect(select, values, { emptyLabel = "Choose replacement…", path = false } = {}) {
  select.replaceChildren(option("", emptyLabel));
  for (const item of values || []) {
    const value = path ? String(item.path || item.value || item.name || "") : descriptorValue(item);
    if (!value) continue;
    select.append(option(value, item.label || item.name || value));
  }
}

function populateOverrideCatalogs() {
  fillSelect($("#queueOverrideModelValue"), state.models, { emptyLabel: "Choose model…", path: true });
  fillSelect($("#queueOverrideVaeValue"), state.vaes, { emptyLabel: "Use original / none…", path: true });
  fillSelect($("#queueOverrideSamplerValue"), state.samplers, { emptyLabel: "Choose sampler…" });
  fillSelect($("#queueOverrideSchedulerValue"), state.schedulers, { emptyLabel: "Choose scheduler…" });
}

function setStage(stage) {
  const setup = $("#queueComposerSetup");
  const preview = $("#queueComposerPreview");
  setup.hidden = stage !== "setup";
  preview.hidden = stage !== "preview";
  $("#queueComposerPreflightButton").hidden = stage !== "setup";
  $("#queueComposerBackButton").hidden = stage !== "preview";
  $("#queueComposerSubmitButton").hidden = stage !== "preview";
  $("#queueComposerQueueValidButton").hidden = stage !== "preview";
}

function renderSelectedRows() {
  const list = $("#queueComposerSelectedList");
  list.replaceChildren();
  const order = state.queueComposer.order;
  $("#queueComposerSelectedCount").textContent = `${order.length} selected output${order.length === 1 ? "" : "s"}`;
  order.forEach((outputId, index) => {
    const item = outputById(outputId);
    const row = document.createElement("article");
    row.className = "queue-composer-selected-row";
    row.dataset.outputId = outputId;
    const image = document.createElement("img");
    image.src = item?.url || "";
    image.alt = "";
    const copy = document.createElement("div");
    copy.innerHTML = `<strong>${index + 1}. ${shortText(item?.prompt || outputId, 72)}</strong><span>${outputId}</span>`;
    const actions = document.createElement("div");
    actions.className = "queue-row-actions";
    for (const [label, direction] of [["↑", -1], ["↓", 1]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "icon-button";
      button.textContent = label;
      button.disabled = index + direction < 0 || index + direction >= order.length;
      button.setAttribute("aria-label", direction < 0 ? "Move job up" : "Move job down");
      button.addEventListener("click", () => {
        const next = [...state.queueComposer.order];
        [next[index], next[index + direction]] = [next[index + direction], next[index]];
        state.queueComposer.order = next;
        state.queueComposer.preflight = null;
        renderSelectedRows();
      });
      actions.append(button);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text-button danger-text";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.queueComposer.order = state.queueComposer.order.filter((id) => id !== outputId);
      delete state.queueComposer.itemRemaps[outputId];
      state.queueComposer.preflight = null;
      renderSelectedRows();
    });
    actions.append(remove);
    row.append(image, copy, actions);
    list.append(row);
  });
  $("#queueComposerPreflightButton").disabled = order.length === 0;
}

function collectOverrideValues() {
  const values = {
    model_path: $("#queueOverrideModelValue").value,
    vae_path: $("#queueOverrideVaeValue").value,
    width: $("#queueOverrideWidthValue").value,
    height: $("#queueOverrideHeightValue").value,
    steps: $("#queueOverrideStepsValue").value,
    cfg_scale: $("#queueOverrideCfgValue").value,
    sampler_name: $("#queueOverrideSamplerValue").value,
    scheduler_name: $("#queueOverrideSchedulerValue").value,
    batch_count: $("#queueOverrideBatchCountValue").value,
  };
  const numeric = new Set(["width", "height", "steps", "batch_count"]);
  const floats = new Set(["cfg_scale"]);
  const overrides = {};
  const fields = [];
  document.querySelectorAll("[data-queue-override-field]").forEach((checkbox) => {
    if (!checkbox.checked) return;
    const field = checkbox.dataset.queueOverrideField;
    let value = values[field];
    if (numeric.has(field)) value = Number.parseInt(value, 10);
    if (floats.has(field)) value = Number.parseFloat(value);
    overrides[field] = value;
    fields.push(field);
  });
  state.queueComposer.overrides = overrides;
  state.queueComposer.overrideFields = fields;
  const seedMode = $("#queueComposerSeedPolicy").value;
  state.queueComposer.seedPolicy = {
    mode: seedMode,
    start: Number.parseInt($("#queueComposerSequentialSeed").value, 10) || 1,
  };
}

function composerPayload() {
  collectOverrideValues();
  return {
    output_ids: [...state.gallerySelection.outputIds],
    mode: state.queueComposer.mode,
    overrides: state.queueComposer.overrides,
    override_fields: state.queueComposer.overrideFields,
    seed_policy: state.queueComposer.seedPolicy,
    order: [...state.queueComposer.order],
    common_remap: state.queueComposer.commonRemap,
    item_remaps: state.queueComposer.itemRemaps,
  };
}

function qualityBadge(item) {
  const quality = item.completeness?.quality || "best_available";
  const badge = document.createElement("span");
  badge.className = `queue-quality-badge ${quality}`;
  badge.textContent = item.completeness?.label || (quality === "exact_request" ? "Exact Request" : "Best Available");
  return badge;
}

function remapChoices(kind) {
  if (kind === "model") return state.models.map((item) => ({ value: item.path || item.value || item.name, label: item.label || item.name || item.path }));
  if (kind === "vae") return state.vaes.map((item) => ({ value: item.path || item.value || item.name, label: item.label || item.name || item.path }));
  if (kind === "sampler") return state.samplers.map((item) => ({ value: descriptorValue(item), label: item.label || item.name }));
  if (kind === "scheduler") return state.schedulers.map((item) => ({ value: descriptorValue(item), label: item.label || item.name }));
  return [];
}

function renderRemapControls(data) {
  const section = $("#queueComposerRemapSection");
  const list = $("#queueComposerRemapList");
  list.replaceChildren();
  const missing = data.jobs.flatMap((job) => (job.missing_assets || []).map((item) => ({ ...item, outputId: job.output_id })));
  section.hidden = missing.length === 0;
  if (!missing.length) return;

  const byField = new Map();
  for (const item of missing) {
    if (!byField.has(item.field)) byField.set(item.field, []);
    byField.get(item.field).push(item);
  }
  for (const [field, items] of byField) {
    const group = document.createElement("section");
    group.className = "queue-remap-group";
    const heading = document.createElement("div");
    heading.innerHTML = `<strong>Common ${items[0].kind} replacement</strong><span>${items.length} unresolved job${items.length === 1 ? "" : "s"}</span>`;
    const select = document.createElement("select");
    select.append(option("", "No common replacement"));
    remapChoices(items[0].kind).forEach((choice) => select.append(option(choice.value, choice.label)));
    select.value = state.queueComposer.commonRemap[field] || "";
    select.addEventListener("change", () => {
      if (select.value) state.queueComposer.commonRemap[field] = select.value;
      else delete state.queueComposer.commonRemap[field];
    });
    group.append(heading, select);

    for (const item of items) {
      const row = document.createElement("label");
      row.className = "queue-item-remap-row";
      const label = document.createElement("span");
      label.textContent = `${shortText(item.outputId, 42)}: ${item.reason}`;
      const itemSelect = document.createElement("select");
      itemSelect.append(option("", "Use common replacement"));
      remapChoices(item.kind).forEach((choice) => itemSelect.append(option(choice.value, choice.label)));
      itemSelect.value = state.queueComposer.itemRemaps[item.outputId]?.[field] || "";
      itemSelect.addEventListener("change", () => {
        const remaps = state.queueComposer.itemRemaps[item.outputId] || {};
        if (itemSelect.value) remaps[field] = itemSelect.value;
        else delete remaps[field];
        state.queueComposer.itemRemaps[item.outputId] = remaps;
      });
      row.append(label, itemSelect);
      group.append(row);
    }
    list.append(group);
  }
}

function renderPreflight(data) {
  state.queueComposer.preflight = data;
  const summary = data.summary || {};
  $("#queueComposerQualitySummary").textContent = `${summary.exact_request_count || 0} Exact Request · ${summary.best_available_count || 0} Best Available`;
  $("#queueComposerValidationSummary").textContent = `${summary.valid_count || 0} valid · ${summary.invalid_count || 0} invalid`;
  const body = $("#queueComposerPreviewBody");
  body.replaceChildren();
  for (const item of data.jobs || []) {
    const row = document.createElement("tr");
    row.className = item.valid ? "is-valid" : "is-invalid";
    const request = item.request || {};
    const values = [
      item.order,
      item.summary?.prompt_summary || item.output_id,
      request.seed ?? "—",
      shortText(request.model_path || "—", 28),
      request.width && request.height ? `${request.width}×${request.height}` : "—",
      request.steps ?? "—",
      request.cfg_scale ?? "—",
      request.sampler_name || "—",
      request.scheduler_name || "—",
      item.summary?.advanced_warning_count || 0,
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    const status = document.createElement("td");
    status.append(qualityBadge(item));
    const stateLabel = document.createElement("strong");
    stateLabel.textContent = item.valid ? "Valid" : `Invalid: ${(item.errors || []).join("; ")}`;
    status.append(stateLabel);
    const previewActions = document.createElement("div");
    previewActions.className = "queue-preview-actions";
    for (const [label, direction] of [["Move up", -1], ["Move down", 1]]) {
      const action = document.createElement("button");
      action.type = "button";
      action.className = "text-button";
      action.textContent = direction < 0 ? "↑" : "↓";
      action.title = label;
      action.setAttribute("aria-label", `${label}: ${item.output_id}`);
      action.disabled = item.order - 1 + direction < 0 || item.order - 1 + direction >= state.queueComposer.order.length;
      action.addEventListener("click", async () => {
        const index = state.queueComposer.order.indexOf(item.output_id);
        const target = index + direction;
        if (index < 0 || target < 0 || target >= state.queueComposer.order.length) return;
        const next = [...state.queueComposer.order];
        [next[index], next[target]] = [next[target], next[index]];
        state.queueComposer.order = next;
        await runPreflight();
      });
      previewActions.append(action);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text-button danger-text";
    remove.textContent = "Remove job";
    remove.addEventListener("click", async () => {
      state.queueComposer.order = state.queueComposer.order.filter((id) => id !== item.output_id);
      delete state.queueComposer.itemRemaps[item.output_id];
      if (state.queueComposer.order.length) await runPreflight();
      else setStage("setup");
    });
    previewActions.append(remove);
    status.append(previewActions);
    row.append(status);
    body.append(row);
  }
  renderRemapControls(data);
  $("#queueComposerSubmitButton").disabled = !data.valid || !data.preflight_token;
  $("#queueComposerQueueValidButton").disabled = !(summary.valid_count > 0 && summary.invalid_count > 0 && data.preflight_token);
  const exportButton = $("#queueComposerExportButton");
  if (exportButton) {
    exportButton.disabled = !(data.jobs || []).length;
    exportButton.textContent = `Export Composed (${(data.jobs || []).length})`;
  }
  const variationButton = $("#queueComposerVariationButton");
  if (variationButton) variationButton.disabled = !(data.jobs || []).some((item) => item.valid);
  setStage("preview");
}

async function runPreflight() {
  state.queueComposer.loading = true;
  $("#queueComposerError").hidden = true;
  $("#queueComposerPreflightButton").disabled = true;
  try {
    const data = await api.batchReplayPreflight(composerPayload());
    renderPreflight(data);
  } catch (error) {
    $("#queueComposerError").textContent = error.message;
    $("#queueComposerError").hidden = false;
  } finally {
    state.queueComposer.loading = false;
    $("#queueComposerPreflightButton").disabled = state.queueComposer.order.length === 0;
  }
}

async function submitPreflight(queueValidOnly) {
  const token = state.queueComposer.preflight?.preflight_token;
  if (!token) return;
  try {
    const response = await api.submitBatchReplay(token, queueValidOnly);
    for (const job of response.submitted || []) onJobQueued(job);
    notify(`${response.submitted_count} replay job${response.submitted_count === 1 ? "" : "s"} queued${response.rejected_count ? `; ${response.rejected_count} rejected` : ""}.`);
    closeComposer();
  } catch (error) {
    $("#queueComposerError").textContent = error.message;
    $("#queueComposerError").hidden = false;
  }
}

function closeComposer() {
  state.queueComposer.open = false;
  const dialog = $("#queueComposerDialog");
  if (dialog.open) dialog.close();
}

export function openQueueComposer() {
  if (!state.gallerySelection.outputIds.length) return;
  resetComposerFromSelection();
  populateOverrideCatalogs();
  document.querySelectorAll("[data-queue-override-field]").forEach((checkbox) => { checkbox.checked = false; });
  $("#queueComposerSeedPolicy").value = "keep_original";
  $("#queueComposerSequentialSeed").disabled = true;
  $("#queueComposerError").hidden = true;
  const exportButton = $("#queueComposerExportButton");
  if (exportButton) {
    exportButton.disabled = true;
    exportButton.textContent = "Export Composed (0)";
  }
  renderSelectedRows();
  setStage("setup");
  $("#queueComposerDialog").showModal();
}

export function bindQueueComposer({ onQueued = () => {} } = {}) {
  onJobQueued = onQueued;
  $("#composeQueueButton").addEventListener("click", openQueueComposer);
  $("#queueComposerCloseButton").addEventListener("click", closeComposer);
  $("#queueComposerCancelButton").addEventListener("click", closeComposer);
  $("#queueComposerPreflightButton").addEventListener("click", runPreflight);
  $("#queueComposerBackButton").addEventListener("click", () => setStage("setup"));
  $("#queueComposerSubmitButton").addEventListener("click", () => submitPreflight(false));
  $("#queueComposerQueueValidButton").addEventListener("click", () => submitPreflight(true));
  $("#queueComposerRerunButton").addEventListener("click", runPreflight);
  $("#queueComposerSeedPolicy").addEventListener("change", (event) => {
    $("#queueComposerSequentialSeed").disabled = event.target.value !== "sequential";
  });
  document.querySelectorAll("[data-queue-override-field]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const input = document.querySelector(`[data-queue-override-value="${checkbox.dataset.queueOverrideField}"]`);
      if (input) input.disabled = !checkbox.checked;
    });
    const input = document.querySelector(`[data-queue-override-value="${checkbox.dataset.queueOverrideField}"]`);
    if (input) input.disabled = true;
  });
}

export { CORE_OVERRIDE_FIELDS };
