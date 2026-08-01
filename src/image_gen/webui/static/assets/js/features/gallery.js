import { api } from "../api.js";
import { $, shortText, formatTime, notify } from "../utils.js";
import { state } from "../state.js";
import { openCompletedLightbox } from "./lightbox.js";
import { openOutputDetails } from "./output-details.js";

let reconcileTimer = null;
let settingsSaveTimer = null;
let refreshRecentOutputs = async () => {};
const DEFAULT_THUMBNAIL_PAGE_SIZE = 12;
const THUMBNAIL_CELL_MIN_SIZE = 72;
const THUMBNAIL_CELL_MAX_SIZE = 72;
const THUMBNAIL_GRID_GAP = 8;
let carouselWheelLock = false;
let thumbnailWindowStart = 0;
let galleryResizeObserver = null;

function outputId(item) {
  return String(item?.output_id || item?.name || "");
}

function outputKey(item) {
  return String(item?.output_id || item?.absolute_path || item?.path || item?.url || item?.name || "");
}

function sameOutput(left, right) {
  if (!left || !right) return false;
  const leftId = outputId(left);
  const rightId = outputId(right);
  if (leftId && rightId) return leftId === rightId;
  const leftPath = String(left?.absolute_path || left?.path || "");
  const rightPath = String(right?.absolute_path || right?.path || "");
  if (leftPath && rightPath) return leftPath === rightPath;
  return outputKey(left) === outputKey(right);
}

function sortRecentOutputCatalog(items) {
  return [...(items || [])].sort((left, right) => {
    const modifiedDelta = Number(right?.modified_ns || 0) - Number(left?.modified_ns || 0);
    if (modifiedDelta) return modifiedDelta;
    return String(right?.timestamp || "").localeCompare(String(left?.timestamp || ""));
  });
}

function selectedSet() {
  return new Set(state.gallerySelection.outputIds || []);
}

function selectedIndex() {
  if (!state.selectedOutput) return -1;
  return state.recentOutputs.findIndex((item) => outputId(item) === outputId(state.selectedOutput));
}

function createOption(value, label, selected = false) {
  const element = document.createElement("option");
  element.value = String(value ?? "");
  element.textContent = label;
  element.selected = selected;
  return element;
}

function normalizeString(value) {
  return String(value || "").trim().toLowerCase();
}

function uniqueValues(items) {
  return [...new Set(items.filter(Boolean).map((item) => String(item).trim()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right, undefined, { sensitivity: "base" }));
}

function copyText(text, successMessage = "Value copied.") {
  const value = String(text ?? "").trim();
  if (!value) {
    notify("Nothing to copy.", "error");
    return;
  }
  const copy = async () => {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    if (!document.execCommand("copy")) throw new Error("Copy command was rejected.");
    textarea.remove();
  };
  copy()
    .then(() => notify(successMessage))
    .catch((error) => notify(`Unable to copy the value: ${error.message}`, "error"));
}

function concreteSeed(item) {
  if (item?.seed === 0) return 0;
  return item?.seed ?? null;
}


function thumbnailViewport() {
  return $("#recentOutputCarouselViewport") || $("#recentOutputs");
}

function thumbnailPageSize() {
  const viewport = thumbnailViewport();
  if (!viewport) return state.recentOutputs.length || state.recentOutputBrowser?.thumbnailsPerPage || DEFAULT_THUMBNAIL_PAGE_SIZE;
  const width = Math.max(0, viewport.clientWidth || 0);
  const height = Math.max(0, viewport.clientHeight || 0);
  if (!width || !height) return state.recentOutputs.length || state.recentOutputBrowser?.thumbnailsPerPage || DEFAULT_THUMBNAIL_PAGE_SIZE;
  const columns = Math.max(1, Math.floor((width + THUMBNAIL_GRID_GAP) / (THUMBNAIL_CELL_MIN_SIZE + THUMBNAIL_GRID_GAP)));
  const rows = Math.max(1, Math.floor((height + THUMBNAIL_GRID_GAP) / (THUMBNAIL_CELL_MIN_SIZE + THUMBNAIL_GRID_GAP)));
  return Math.max(1, columns * rows);
}

function applyThumbnailGridSizing(visibleCount) {
  const strip = $("#recentOutputs");
  const viewport = thumbnailViewport() || strip;
  if (!strip || !viewport) return;
  const width = Math.max(0, viewport.clientWidth || 0);
  const availableColumns = Math.max(1, Math.floor((width + THUMBNAIL_GRID_GAP) / (THUMBNAIL_CELL_MIN_SIZE + THUMBNAIL_GRID_GAP)));
  const columns = visibleCount > 0 ? Math.max(1, Math.min(availableColumns, visibleCount)) : 1;
  const cellSize = Math.max(THUMBNAIL_CELL_MIN_SIZE, Math.min(THUMBNAIL_CELL_MAX_SIZE, THUMBNAIL_CELL_MIN_SIZE));
  strip.style.setProperty("--thumbnail-cell-size", `${cellSize}px`);
  strip.style.gridTemplateColumns = `repeat(${columns}, minmax(${cellSize}px, ${cellSize}px))`;
  strip.style.gridAutoRows = `${cellSize}px`;
}

function maxThumbnailWindowStart() {
  return 0;
}

function clampThumbnailWindowStart(value) {
  return 0;
}

function syncThumbnailWindow(targetIndex = -1) {
  thumbnailWindowStart = 0;
}

function recentOutputViewportHasOverflow() {
  const viewport = thumbnailViewport();
  if (!viewport) return false;
  return (viewport.scrollHeight - viewport.clientHeight) > 2;
}

function updateThumbnailCarouselControls() {
  const total = state.recentOutputs.length;
  const viewport = thumbnailViewport();
  const hasOverflow = recentOutputViewportHasOverflow();
  const start = total ? 1 : 0;
  const end = total;
  const status = $("#recentOutputCarouselStatus");
  if (status) {
    status.textContent = total
      ? `Showing ${start}–${end} of ${total} recent outputs`
      : "Showing 0–0 of 0 recent outputs";
  }
  const previous = $("#recentOutputCarouselPrevButton");
  const next = $("#recentOutputCarouselNextButton");
  if (previous) previous.disabled = !hasOverflow || !viewport || viewport.scrollTop <= 0;
  if (next) next.disabled = !hasOverflow || !viewport || (viewport.scrollTop + viewport.clientHeight) >= (viewport.scrollHeight - 2);
}

function moveThumbnailWindow(direction) {
  const viewport = thumbnailViewport();
  if (!viewport) return;
  const maxScrollTop = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
  if (maxScrollTop <= 0) {
    updateThumbnailCarouselControls();
    return;
  }
  const pageDistance = Math.max(96, viewport.clientHeight - 24);
  const movingForward = direction > 0;
  const atTop = viewport.scrollTop <= 2;
  const atBottom = viewport.scrollTop >= (maxScrollTop - 2);
  if (movingForward && atBottom) {
    viewport.scrollTo({ top: 0, behavior: "smooth" });
  } else if (!movingForward && atTop) {
    viewport.scrollTo({ top: maxScrollTop, behavior: "smooth" });
  } else {
    viewport.scrollBy({ top: direction * pageDistance, behavior: "smooth" });
  }
  window.setTimeout(updateThumbnailCarouselControls, 170);
}

function closeRecentOutputFilterPanel() {
  if (!state.recentOutputFilters.panelOpen) return;
  state.recentOutputFilters.panelOpen = false;
  updateRecentOutputFilterControls();
}

function populateSelectOptions(select, values, currentValue = "", allLabel = "All") {
  if (!select) return;
  const selectedValue = String(currentValue ?? "");
  const options = [createOption("", allLabel, selectedValue === "")];
  values.forEach((value) => {
    options.push(createOption(value, value, String(value) === selectedValue));
  });
  select.replaceChildren(...options);
  select.value = values.includes(selectedValue) ? selectedValue : "";
}

function promptTokens(prompt, filters = state.recentOutputFilters) {
  const text = String(prompt || "");
  if (!text.trim()) return [];
  const start = String(filters.parserStart || "");
  const midpoint = String(filters.parserMidpoint || "");
  const end = String(filters.parserEnd || "");
  const segments = [];

  const pushMatches = (first, second) => {
    if (!first || !second) return;
    let searchIndex = 0;
    while (searchIndex < text.length) {
      const startIndex = text.indexOf(first, searchIndex);
      if (startIndex < 0) break;
      const valueStart = startIndex + first.length;
      const endIndex = text.indexOf(second, valueStart);
      if (endIndex < 0) break;
      const candidate = text.slice(valueStart, endIndex).trim();
      if (candidate) segments.push(candidate);
      searchIndex = endIndex + second.length;
    }
  };

  if (start && end) pushMatches(start, end);
  if (!segments.length && start && midpoint) pushMatches(start, midpoint);
  if (!segments.length && midpoint && end) pushMatches(midpoint, end);
  if (segments.length) return segments;

  return text.split(",").map((item) => item.trim()).filter(Boolean);
}

function promptQueryTerms() {
  return String(state.recentOutputFilters.promptQuery || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function matchesPrompt(item) {
  const terms = promptQueryTerms();
  if (!terms.length) return true;
  const tokens = promptTokens(item.prompt || "").map((value) => value.toLowerCase());
  const rawPrompt = String(item.prompt || "").toLowerCase();
  return terms.every((term) => tokens.some((token) => token.includes(term)) || rawPrompt.includes(term));
}

function matchesSelect(value, expected) {
  return !expected || normalizeString(value) === normalizeString(expected);
}

function filterRecentOutputs(outputs) {
  return (outputs || []).filter((item) => {
    if (!matchesPrompt(item)) return false;
    if (!matchesSelect(item.sampler_name, state.recentOutputFilters.sampler)) return false;
    if (!matchesSelect(item.scheduler_name, state.recentOutputFilters.scheduler)) return false;
    if (!matchesSelect(item.model_name || item.model_path, state.recentOutputFilters.model)) return false;
    if (!matchesSelect(item.vae_name || item.vae_path, state.recentOutputFilters.vae)) return false;
    if (state.recentOutputFilters.lora) {
      const loras = (item.loras || []).map(normalizeString);
      if (!loras.includes(normalizeString(state.recentOutputFilters.lora))) return false;
    }
    if (state.recentOutputFilters.resolution) {
      const resolution = `${item.width || ""}×${item.height || ""}`;
      if (resolution !== state.recentOutputFilters.resolution) return false;
    }
    if (!matchesSelect(item.generation_mode, state.recentOutputFilters.generationMode)) return false;
    if (!matchesSelect(item.metadata_source, state.recentOutputFilters.metadataSource)) return false;
    if (state.recentOutputFilters.hires === "yes" && item.hires !== true) return false;
    if (state.recentOutputFilters.hires === "no" && item.hires !== false) return false;
    if (state.recentOutputFilters.hires === "unknown" && item.hires !== null) return false;
    return true;
  });
}

function filterSummaryText(filteredCount, totalCount) {
  const timeLabelMap = {
    "24": "24 hours",
    "48": "48 hours",
    "72": "72 hours",
    custom: `${state.recentOutputFilters.customHours} hours`,
    all: "all time",
  };
  const timeLabel = timeLabelMap[state.recentOutputFilters.timeWindow] || `${state.recentOutputFilters.timeWindow} hours`;
  const extraLocations = (state.recentOutputFilters.sourcePaths || []).length;
  const folderLabel = extraLocations === 0 ? "output folder" : `output folder + ${extraLocations} custom location${extraLocations === 1 ? "" : "s"}`;
  return `${filteredCount} shown of ${totalCount} scanned · ${timeLabel} · ${folderLabel}${state.recentOutputFilters.includeSubfolders ? " · subfolders on" : ""}`;
}

function deriveFilterOptions(outputs) {
  return {
    samplers: uniqueValues(outputs.map((item) => item.sampler_name)),
    schedulers: uniqueValues(outputs.map((item) => item.scheduler_name)),
    models: uniqueValues(outputs.map((item) => item.model_name || item.model_path)),
    vaes: uniqueValues(outputs.map((item) => item.vae_name || item.vae_path)),
    loras: uniqueValues(outputs.flatMap((item) => item.loras || [])),
    resolutions: uniqueValues(outputs.map((item) => item.width && item.height ? `${item.width}×${item.height}` : "")),
    modes: uniqueValues(outputs.map((item) => item.generation_mode || "unknown")),
    metadataSources: uniqueValues(outputs.map((item) => item.metadata_source || "")),
  };
}

function renderSourcePathList() {
  const list = $("#recentOutputSourceList");
  if (!list) return;
  list.replaceChildren();
  const outputRootNode = $("#recentOutputDefaultFolderPath");
  if (outputRootNode) {
    outputRootNode.textContent = state.bootstrap?.output_root || "Current output folder";
  }
  if (!(state.recentOutputFilters.sourcePaths || []).length) {
    const empty = document.createElement("div");
    empty.className = "recent-source-empty";
    empty.textContent = "No extra folders added.";
    list.append(empty);
    return;
  }
  state.recentOutputFilters.sourcePaths.forEach((path, index) => {
    const row = document.createElement("div");
    row.className = "recent-source-row";
    const label = document.createElement("code");
    label.textContent = path;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "ghost compact";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", async () => {
      state.recentOutputFilters.sourcePaths.splice(index, 1);
      persistRecentOutputBrowserSettings();
      renderSourcePathList();
      await refreshRecentOutputs();
    });
    row.append(label, removeButton);
    list.append(row);
  });
}

function updateRecentOutputFilterControls() {
  const filters = state.recentOutputFilters;
  const panel = $("#recentOutputFilterPanel");
  if (panel) panel.hidden = !filters.panelOpen;
  const toggle = $("#recentOutputFilterToggleButton");
  if (toggle) {
    toggle.classList.toggle("is-active", filters.panelOpen);
    toggle.setAttribute("aria-expanded", String(filters.panelOpen));
  }
  const timeWindow = $("#recentOutputTimeWindow");
  if (timeWindow) timeWindow.value = filters.timeWindow;
  const customHours = $("#recentOutputCustomHours");
  if (customHours) customHours.value = String(filters.customHours || 24);
  const customWrap = $("#recentOutputCustomHoursWrap");
  if (customWrap) customWrap.hidden = filters.timeWindow !== "custom";
  const includeSubfolders = $("#recentOutputIncludeSubfolders");
  if (includeSubfolders) includeSubfolders.checked = Boolean(filters.includeSubfolders);
  const metadataGate = $("#recentOutputRequireMetadata");
  if (metadataGate) metadataGate.checked = Boolean(filters.requireMetadataForExternal);
  [
    ["#recentOutputPromptQuery", filters.promptQuery],
    ["#recentOutputParserStart", filters.parserStart],
    ["#recentOutputParserMidpoint", filters.parserMidpoint],
    ["#recentOutputParserEnd", filters.parserEnd],
  ].forEach(([selector, value]) => {
    const input = $(selector);
    if (input) input.value = value || "";
  });
  renderSourcePathList();
}

function updateFilterSelectsFromCatalog() {
  const options = deriveFilterOptions(state.recentOutputCatalog || []);
  populateSelectOptions($("#recentOutputSamplerFilter"), options.samplers, state.recentOutputFilters.sampler);
  populateSelectOptions($("#recentOutputSchedulerFilter"), options.schedulers, state.recentOutputFilters.scheduler);
  populateSelectOptions($("#recentOutputModelFilter"), options.models, state.recentOutputFilters.model);
  populateSelectOptions($("#recentOutputVaeFilter"), options.vaes, state.recentOutputFilters.vae);
  populateSelectOptions($("#recentOutputLoraFilter"), options.loras, state.recentOutputFilters.lora);
  populateSelectOptions($("#recentOutputResolutionFilter"), options.resolutions, state.recentOutputFilters.resolution);
  populateSelectOptions($("#recentOutputGenerationModeFilter"), options.modes, state.recentOutputFilters.generationMode);
  populateSelectOptions($("#recentOutputMetadataSourceFilter"), options.metadataSources, state.recentOutputFilters.metadataSource);
  populateSelectOptions($("#recentOutputHiresFilter"), ["yes", "no", "unknown"], state.recentOutputFilters.hires);
}

function persistRecentOutputBrowserSettings() {
  window.clearTimeout(settingsSaveTimer);
  settingsSaveTimer = window.setTimeout(async () => {
    try {
      const saved = await api.saveSettings({
        recent_outputs_browser: {
          time_window: state.recentOutputFilters.timeWindow,
          custom_hours: state.recentOutputFilters.customHours,
          include_subfolders: state.recentOutputFilters.includeSubfolders,
          source_paths: [...state.recentOutputFilters.sourcePaths],
          require_metadata_for_external: state.recentOutputFilters.requireMetadataForExternal,
        },
      });
      state.settings = saved;
    } catch (error) {
      console.error("Unable to save recent-output browser settings", error);
    }
  }, 250);
}

function serverQueryFromFilters() {
  return {
    hours: state.recentOutputFilters.timeWindow === "all"
      ? "0"
      : state.recentOutputFilters.timeWindow === "custom"
        ? String(Math.max(1, Number(state.recentOutputFilters.customHours) || 24))
        : state.recentOutputFilters.timeWindow,
    include_subfolders: state.recentOutputFilters.includeSubfolders,
    source_paths: (state.recentOutputFilters.sourcePaths || []).join("|"),
    require_metadata_for_external: state.recentOutputFilters.requireMetadataForExternal,
  };
}

export function recentOutputApiFilters() {
  return serverQueryFromFilters();
}

function applyAndRenderGallery({ selectNewest = false, focusOutputId = "" } = {}) {
  updateRecentOutputFilterControls();
  updateFilterSelectsFromCatalog();
  const previousSelection = selectedSet();
  const previousName = state.selectedOutput?.name || "";
  const previousOutputId = outputId(state.selectedOutput);
  const filteredOutputs = filterRecentOutputs(state.recentOutputCatalog || []);
  state.recentOutputs = filteredOutputs;

  const visibleIds = new Set(state.recentOutputs.map(outputId));
  const retained = [...previousSelection].filter((id) => visibleIds.has(id));
  state.gallerySelection.outputIds = retained;
  if (!visibleIds.has(state.gallerySelection.anchorOutputId)) {
    state.gallerySelection.anchorOutputId = retained.at(-1) || null;
  }

  const strip = $("#recentOutputs");
  const runs = $("#recentRuns");
  strip.replaceChildren();
  runs?.replaceChildren();

  const summaryNode = $("#recentOutputFilterSummary");
  if (summaryNode) summaryNode.textContent = filterSummaryText(filteredOutputs.length, (state.recentOutputCatalog || []).length);

  if (!state.recentOutputs.length) {
    thumbnailWindowStart = 0;
    state.selectedOutput = null;
    state.gallerySelection.outputIds = [];
    strip.className = "thumbnail-strip empty-state";
    strip.textContent = (state.recentOutputCatalog || []).length
      ? "No images matched the current Recent Output filters."
      : "Generated images will appear here.";
    if (runs) {
      runs.className = "compact-list empty-state";
      runs.textContent = (state.recentOutputCatalog || []).length ? "No recent runs matched the current filters." : "No recent runs.";
    }
    $("#outputStage").classList.remove("has-image");
    $("#outputImage").removeAttribute("src");
    $("#outputDimensions").textContent = "—";
    $("#outputStatus").textContent = "No selection";
    updateThumbnailCarouselControls();
    updateNavigationState();
    publishSelection();
    return;
  }

  const preferred = state.recentOutputs.find((item) => outputId(item) === String(focusOutputId || ""))
    || (selectNewest ? state.recentOutputs[0] : null)
    || state.recentOutputs.find((item) => outputId(item) === previousOutputId)
    || state.recentOutputs.find((item) => item.name === previousName)
    || state.recentOutputs[0];
  const preferredIndex = state.recentOutputs.findIndex((item) => outputId(item) === outputId(preferred));
  syncThumbnailWindow(preferredIndex);

  strip.className = "thumbnail-strip";
  if (runs) runs.className = "compact-list";

  const visibleOutputs = state.recentOutputs;
  applyThumbnailGridSizing(visibleOutputs.length);
  const preferredId = outputId(preferred);

  visibleOutputs.forEach((item, index) => {
    const itemId = outputId(item);
    const wrapper = document.createElement("div");
    wrapper.className = "thumbnail-selectable";
    wrapper.dataset.galleryOutputId = itemId;
    wrapper.setAttribute("role", "listitem");

    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = "thumbnail-button";
    thumb.dataset.name = item.name;
    thumb.setAttribute("aria-label", item.prompt || item.name || `Recent output ${index + 1}`);
    thumb.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight Home End Control+Space Shift+Space");
    thumb.setAttribute("aria-pressed", selectedSet().has(itemId) ? "true" : "false");
    thumb.tabIndex = itemId === preferredId ? 0 : -1;
    thumb.addEventListener("click", (event) => {
      if (event.shiftKey) {
        toggleSelection(itemId, { range: true });
      } else if (event.ctrlKey || event.metaKey) {
        toggleSelection(itemId);
      } else {
        showOutput(item);
      }
    });

    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.prompt || item.name || `Recent output ${index + 1}`;
    image.loading = index < 12 ? "eager" : "lazy";
    image.fetchPriority = index < 12 ? "high" : "low";
    image.addEventListener("error", () => {
      thumb.classList.add("is-missing-output");
      reconcileRecentOutputs();
    }, { once: true });
    const label = document.createElement("div");
    label.className = "thumbnail-label";
    label.textContent = shortText(item.prompt || item.name || "Output", 60);
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.className = "thumbnail-select-toggle";
    toggle.checked = selectedSet().has(itemId);
    toggle.tabIndex = -1;
    toggle.setAttribute("aria-hidden", "true");
    thumb.append(image, label);
    wrapper.append(thumb, toggle);
    strip.append(wrapper);
  });

  state.recentOutputs.slice(0, 5).forEach((item) => {
    const row = document.createElement("div");
    row.className = "compact-row with-actions";

    const runButton = document.createElement("button");
    runButton.type = "button";
    runButton.className = "compact-row-button";
    runButton.dataset.name = item.name;
    runButton.dataset.galleryOutputId = outputId(item);
    runButton.addEventListener("click", () => showOutput(item));

    const prompt = document.createElement("span");
    prompt.textContent = shortText(item.prompt || item.name, 48);
    const meta = document.createElement("small");
    meta.textContent = item.width && item.height ? `${item.width}×${item.height}` : formatTime(item.timestamp);
    runButton.append(prompt, meta);

    const actions = document.createElement("div");
    actions.className = "compact-row-actions";
    const seedButton = document.createElement("button");
    seedButton.type = "button";
    seedButton.className = "ghost compact icon-button seed-copy-button";
    seedButton.title = concreteSeed(item) === null ? "Seed unavailable" : `Copy seed ${concreteSeed(item)}`;
    seedButton.setAttribute("aria-label", concreteSeed(item) === null ? "Seed unavailable" : `Copy seed ${concreteSeed(item)}`);
    seedButton.textContent = "🌱";
    seedButton.disabled = concreteSeed(item) === null;
    seedButton.addEventListener("click", (event) => {
      event.stopPropagation();
      const seed = concreteSeed(item);
      if (seed === null) return;
      copyText(String(seed), `Seed ${seed} copied.`);
    });
    actions.append(seedButton);
    row.append(runButton, actions);
    runs?.append(row);
  });

  updateThumbnailCarouselControls();
  showOutput(preferred, { skipWindowSync: true });
  if (focusOutputId) focusThumbnailButton(focusOutputId);
  if (previousSelection.size) publishSelection();
}

function updateNavigationState() {
  const hasOutputs = state.recentOutputs.length > 0;
  $("#previousOutputButton").disabled = !hasOutputs;
  $("#nextOutputButton").disabled = !hasOutputs;
  $("#openLargeOutputButton").disabled = !state.selectedOutput?.url;
  $("#imageDetailsButton").disabled = !state.selectedOutput?.output_id;
  updateThumbnailCarouselControls();
  const clearRecentButton = $("#clearRecentOutputsButton");
  if (clearRecentButton) clearRecentButton.disabled = !(state.recentOutputCatalog || []).length;
}

function publishSelection() {
  const count = state.gallerySelection.outputIds.length;
  const countNode = $("#gallerySelectionCount");
  const clearButton = $("#clearGallerySelectionButton");
  const composeButton = $("#composeQueueButton");
  const exportButton = $("#exportSelectedOutputsButton");
  if (countNode) countNode.textContent = `${count} selected`;
  if (clearButton) clearButton.disabled = count === 0;
  if (exportButton) {
    exportButton.disabled = count === 0;
    exportButton.textContent = `Export Selected (${count})`;
  }
  if (composeButton) {
    composeButton.disabled = count === 0;
    composeButton.textContent = `Compose Queue (${count})`;
  }
  const selected = selectedSet();
  document.querySelectorAll("[data-gallery-output-id]").forEach((node) => {
    const id = node.dataset.galleryOutputId;
    const active = selected.has(id);
    node.classList.toggle("is-multi-selected", active);
    const checkbox = node.querySelector(".thumbnail-select-toggle");
    if (checkbox) checkbox.checked = active;
    const button = node.querySelector(".thumbnail-button");
    if (button) button.setAttribute("aria-pressed", String(active));
  });
  document.dispatchEvent(new CustomEvent("gallery-selection-changed", {
    detail: { outputIds: [...state.gallerySelection.outputIds] },
  }));
}

export function selectedGalleryOutputIds() {
  return [...state.gallerySelection.outputIds];
}

export function clearGallerySelection() {
  state.gallerySelection.outputIds = [];
  state.gallerySelection.anchorOutputId = null;
  publishSelection();
}

function setSelection(ids, anchor = null) {
  const visible = new Set(state.recentOutputs.map(outputId));
  state.gallerySelection.outputIds = [...new Set(ids)].filter((id) => visible.has(id));
  state.gallerySelection.anchorOutputId = anchor;
  publishSelection();
}

function toggleSelection(id, { range = false } = {}) {
  const ids = state.recentOutputs.map(outputId);
  const current = selectedSet();
  if (range && state.gallerySelection.anchorOutputId) {
    const anchorIndex = ids.indexOf(state.gallerySelection.anchorOutputId);
    const targetIndex = ids.indexOf(id);
    if (anchorIndex >= 0 && targetIndex >= 0) {
      const [start, end] = anchorIndex <= targetIndex ? [anchorIndex, targetIndex] : [targetIndex, anchorIndex];
      ids.slice(start, end + 1).forEach((item) => current.add(item));
    }
  } else if (current.has(id)) {
    current.delete(id);
  } else {
    current.add(id);
  }
  setSelection([...current], id);
}

function selectAllVisible() {
  setSelection(state.recentOutputs.map(outputId), state.recentOutputs.at(-1) ? outputId(state.recentOutputs.at(-1)) : null);
}

function focusThumbnailButton(id) {
  const value = String(id || "");
  if (!value) return false;
  const button = document.querySelector(
    `[data-gallery-output-id="${CSS.escape(value)}"] .thumbnail-button`,
  );
  if (!button) return false;
  button.focus({ preventScroll: true });
  button.scrollIntoView({ block: "nearest", inline: "nearest" });
  return true;
}

export function showOutput(item, { skipWindowSync = false, focusThumbnail = false } = {}) {
  if (!item) return;
  state.selectedOutput = item;
  const stage = $("#outputStage");
  const image = $("#outputImage");
  image.onerror = () => {
    stage.classList.remove("has-image");
    reconcileRecentOutputs();
  };
  image.src = item.url;
  image.alt = item.prompt ? `Generated image: ${shortText(item.prompt, 100)}` : "Generated image";
  stage.classList.add("has-image");
  $("#outputDimensions").textContent = item.width && item.height ? `${item.width} × ${item.height}` : "—";
  $("#outputSampler").textContent = item.sampler_name || "—";
  $("#outputScheduler").textContent = item.scheduler_name || "—";
  $("#outputSeed").textContent = item.seed ?? "—";
  $("#outputStatus").textContent = item.timestamp ? formatTime(item.timestamp) : "Completed";
  document.querySelectorAll(".thumbnail-button").forEach((button) => {
    const id = button.closest("[data-gallery-output-id]")?.dataset.galleryOutputId;
    const active = id === outputId(item);
    button.classList.toggle("is-selected", active);
    button.tabIndex = active ? 0 : -1;
    if (active) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  });
  updateNavigationState();
  if (focusThumbnail) focusThumbnailButton(outputId(item));
}

function navigate(direction, { focusThumbnail = false, fromOutputId = "" } = {}) {
  if (!state.recentOutputs.length) return;
  const requestedIndex = fromOutputId
    ? state.recentOutputs.findIndex((item) => outputId(item) === fromOutputId)
    : -1;
  const current = requestedIndex >= 0 ? requestedIndex : selectedIndex();
  const next = current < 0 ? 0 : (current + direction + state.recentOutputs.length) % state.recentOutputs.length;
  showOutput(state.recentOutputs[next], { focusThumbnail });
}

function navigateToEdge(edge, { focusThumbnail = false } = {}) {
  if (!state.recentOutputs.length) return;
  const index = edge === "end" ? state.recentOutputs.length - 1 : 0;
  showOutput(state.recentOutputs[index], { focusThumbnail });
}

function toggleFitMode() {
  state.outputFitMode = state.outputFitMode === "fit" ? "actual" : "fit";
  const fit = state.outputFitMode === "fit";
  const stage = $("#outputStage");
  stage.classList.toggle("is-fit", fit);
  stage.classList.toggle("is-actual", !fit);
  $("#fitOutputButton").textContent = fit ? "Fit to Panel" : "Actual Size";
  $("#fitOutputButton").setAttribute("aria-pressed", String(fit));
}

function openSelectedOutput(opener) {
  if (state.selectedOutput?.url) {
    openCompletedLightbox(state.selectedOutput, { opener });
  }
}

async function clearRecentOutputs() {
  if (!(state.recentOutputCatalog || []).length) return;
  const count = state.recentOutputCatalog.length;
  const accepted = window.confirm(
    `Clear ${count} recent output${count === 1 ? "" : "s"} from this list?\n\nThe image and metadata files in the output folder will not be deleted.`,
  );
  if (!accepted) return;

  const button = $("#clearRecentOutputsButton");
  if (button) button.disabled = true;
  try {
    const response = await api.clearRecentOutputs();
    renderGallery(response.recent_outputs || []);
    const cleared = Number(response.cleared_count || count);
    notify(`${cleared} recent output${cleared === 1 ? " was" : "s were"} cleared from the WebUI list. Files were not deleted.`);
  } catch (error) {
    notify(error.message, "error");
    updateNavigationState();
  }
}

async function onRefreshNeeded() {
  persistRecentOutputBrowserSettings();
  await refreshRecentOutputs();
}

async function reloadRecentOutputFolder() {
  const button = $("#recentOutputReloadButton");
  if (button) button.disabled = true;
  try {
    const response = await api.reloadRecentOutputs();
    thumbnailWindowStart = 0;
    if (response.time_window) {
      state.recentOutputFilters.timeWindow = String(response.time_window);
    }
    renderGallery(response.recent_outputs || []);
    updateRecentOutputFilterControls();
    notify(
      response.full_rescan
        ? `Output folder rescanned. ${Number(response.count || response.recent_outputs?.length || 0)} image${Number(response.count || response.recent_outputs?.length || 0) === 1 ? "" : "s"} loaded in All time view.`
        : "Recent output folder reloaded from disk.",
    );
  } catch (error) {
    notify(error.message, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

function bindRecentOutputFilterControls() {
  $("#recentOutputFilterToggleButton")?.addEventListener("click", (event) => {
    event.stopPropagation();
    state.recentOutputFilters.panelOpen = !state.recentOutputFilters.panelOpen;
    updateRecentOutputFilterControls();
  });
  $("#recentOutputFilterCloseButton")?.addEventListener("click", () => {
    closeRecentOutputFilterPanel();
  });
  $("#recentOutputTimeWindow")?.addEventListener("change", async (event) => {
    state.recentOutputFilters.timeWindow = event.target.value || "72";
    updateRecentOutputFilterControls();
    await onRefreshNeeded();
  });
  $("#recentOutputCustomHours")?.addEventListener("change", async (event) => {
    state.recentOutputFilters.customHours = Math.max(1, Number(event.target.value) || 24);
    event.target.value = String(state.recentOutputFilters.customHours);
    if (state.recentOutputFilters.timeWindow === "custom") await onRefreshNeeded();
    else persistRecentOutputBrowserSettings();
  });
  $("#recentOutputIncludeSubfolders")?.addEventListener("change", async (event) => {
    state.recentOutputFilters.includeSubfolders = Boolean(event.target.checked);
    await onRefreshNeeded();
  });
  $("#recentOutputRequireMetadata")?.addEventListener("change", async (event) => {
    state.recentOutputFilters.requireMetadataForExternal = Boolean(event.target.checked);
    await onRefreshNeeded();
  });
  $("#addRecentOutputSourceButton")?.addEventListener("click", async () => {
    const input = $("#recentOutputSourceInput");
    const value = String(input?.value || "").trim();
    if (!value) {
      notify("Enter a folder path to add an external image location.", "error");
      return;
    }
    const exists = (state.recentOutputFilters.sourcePaths || []).some((item) => normalizeString(item) === normalizeString(value));
    if (!exists) state.recentOutputFilters.sourcePaths.push(value);
    if (input) input.value = "";
    renderSourcePathList();
    await onRefreshNeeded();
  });
  $("#recentOutputPromptQuery")?.addEventListener("input", (event) => {
    state.recentOutputFilters.promptQuery = event.target.value;
    applyAndRenderGallery();
  });
  ["#recentOutputParserStart", "#recentOutputParserMidpoint", "#recentOutputParserEnd"].forEach((selector) => {
    $(selector)?.addEventListener("input", (event) => {
      const key = selector.endsWith("Start") ? "parserStart" : selector.endsWith("Midpoint") ? "parserMidpoint" : "parserEnd";
      state.recentOutputFilters[key] = event.target.value;
      applyAndRenderGallery();
    });
  });
  [
    ["#recentOutputSamplerFilter", "sampler"],
    ["#recentOutputSchedulerFilter", "scheduler"],
    ["#recentOutputModelFilter", "model"],
    ["#recentOutputVaeFilter", "vae"],
    ["#recentOutputLoraFilter", "lora"],
    ["#recentOutputResolutionFilter", "resolution"],
    ["#recentOutputGenerationModeFilter", "generationMode"],
    ["#recentOutputHiresFilter", "hires"],
    ["#recentOutputMetadataSourceFilter", "metadataSource"],
  ].forEach(([selector, key]) => {
    $(selector)?.addEventListener("change", (event) => {
      state.recentOutputFilters[key] = event.target.value || "";
      applyAndRenderGallery();
    });
  });
  $("#resetRecentOutputFiltersButton")?.addEventListener("click", async () => {
    Object.assign(state.recentOutputFilters, {
      timeWindow: "72",
      customHours: 24,
      includeSubfolders: true,
      sourcePaths: [],
      requireMetadataForExternal: true,
      promptQuery: "",
      parserStart: "",
      parserMidpoint: "",
      parserEnd: "",
      sampler: "",
      scheduler: "",
      model: "",
      vae: "",
      lora: "",
      resolution: "",
      generationMode: "",
      hires: "",
      metadataSource: "",
    });
    updateRecentOutputFilterControls();
    await onRefreshNeeded();
  });
}

export function initializeRecentOutputBrowser(settings = {}) {
  const browser = settings.recent_outputs_browser || {};
  Object.assign(state.recentOutputFilters, {
    timeWindow: String(browser.time_window || "72"),
    customHours: Number(browser.custom_hours || 24),
    includeSubfolders: browser.include_subfolders !== false,
    sourcePaths: [...(browser.source_paths || [])],
    requireMetadataForExternal: browser.require_metadata_for_external !== false,
  });
  updateRecentOutputFilterControls();
}

function reconcileRecentOutputs({ delay = 150 } = {}) {
  if (reconcileTimer) window.clearTimeout(reconcileTimer);
  reconcileTimer = window.setTimeout(async () => {
    reconcileTimer = null;
    try {
      renderGallery(await api.recentOutputs(serverQueryFromFilters()));
    } catch (error) {
      console.error("Unable to reconcile recent outputs", error);
    }
  }, delay);
}

export function bindGallery(options = {}) {
  refreshRecentOutputs = options.refreshOutputs || refreshRecentOutputs;
  $("#previousOutputButton").addEventListener("click", () => navigate(-1));
  $("#nextOutputButton").addEventListener("click", () => navigate(1));
  $("#fitOutputButton").addEventListener("click", toggleFitMode);
  $("#selectAllVisibleButton").addEventListener("click", selectAllVisible);
  $("#clearGallerySelectionButton").addEventListener("click", clearGallerySelection);
  $("#clearRecentOutputsButton").addEventListener("click", clearRecentOutputs);
  $("#openLargeOutputButton").addEventListener("click", (event) => {
    openSelectedOutput(event.currentTarget);
  });
  $("#imageDetailsButton").addEventListener("click", (event) => {
    if (state.selectedOutput?.output_id) {
      openOutputDetails(state.selectedOutput, { opener: event.currentTarget });
    }
  });
  $("#outputStage").addEventListener("dblclick", (event) => {
    event.preventDefault();
    openSelectedOutput(event.currentTarget);
  });
  $("#outputStage").addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigate(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      navigate(1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      openSelectedOutput(event.currentTarget);
    }
  });
  $("#recentOutputCarouselPrevButton")?.addEventListener("click", () => moveThumbnailWindow(-1));
  $("#recentOutputCarouselNextButton")?.addEventListener("click", () => moveThumbnailWindow(1));
  $("#recentOutputReloadButton")?.addEventListener("click", reloadRecentOutputFolder);
  const recentOutputs = $("#recentOutputs");
  recentOutputs.addEventListener("keydown", (event) => {
    const button = event.target.closest(".thumbnail-button");
    if (!button || !recentOutputs.contains(button)) return;
    const id = button.closest("[data-gallery-output-id]")?.dataset.galleryOutputId || "";
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
      event.preventDefault();
      selectAllVisible();
      return;
    }
    if ((event.ctrlKey || event.metaKey || event.shiftKey) && event.key === " ") {
      event.preventDefault();
      toggleSelection(id, { range: event.shiftKey });
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigate(-1, { focusThumbnail: true, fromOutputId: id });
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      navigate(1, { focusThumbnail: true, fromOutputId: id });
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      navigateToEdge("start", { focusThumbnail: true });
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      navigateToEdge("end", { focusThumbnail: true });
    }
  });
  $("#recentOutputCarousel")?.addEventListener("wheel", (event) => {
    if (Math.abs(event.deltaY) < 18 && Math.abs(event.deltaX) < 18) return;
    event.preventDefault();
    if (carouselWheelLock) return;
    carouselWheelLock = true;
    window.setTimeout(() => { carouselWheelLock = false; }, 120);
    moveThumbnailWindow((event.deltaY > 0 || event.deltaX > 0) ? 1 : -1);
  }, { passive: false });
  thumbnailViewport()?.addEventListener("scroll", () => {
    updateThumbnailCarouselControls();
  }, { passive: true });
  document.addEventListener("pointerdown", (event) => {
    if (!state.recentOutputFilters.panelOpen) return;
    const panel = $("#recentOutputFilterPanel");
    const toggle = $("#recentOutputFilterToggleButton");
    const target = event.target;
    if (panel?.contains(target) || toggle?.contains(target)) return;
    closeRecentOutputFilterPanel();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeRecentOutputFilterPanel();
  });
  if (typeof ResizeObserver !== "undefined") {
    galleryResizeObserver?.disconnect?.();
    const viewport = thumbnailViewport();
    if (viewport) {
      let previousSignature = "";
      galleryResizeObserver = new ResizeObserver(() => {
        const width = viewport.clientWidth || 0;
        const height = viewport.clientHeight || 0;
        const nextSignature = `${width}x${height}`;
        if (nextSignature === previousSignature && previousSignature) return;
        previousSignature = nextSignature;
        applyAndRenderGallery({ focusOutputId: outputId(state.selectedOutput) });
      });
      galleryResizeObserver.observe(viewport);
    }
  }
  bindRecentOutputFilterControls();
  updateNavigationState();
  publishSelection();
}

export function upsertRecentOutput(item, { selectNewest = false, focusThumbnail = false } = {}) {
  if (!item || !outputKey(item)) return;
  const merged = [item, ...(state.recentOutputCatalog || []).filter((existing) => !sameOutput(item, existing))];
  state.recentOutputCatalog = sortRecentOutputCatalog(merged);
  const visible = filterRecentOutputs([item]).length > 0;
  const focusOutputId = (selectNewest || focusThumbnail) && visible ? outputId(item) : "";
  applyAndRenderGallery({
    selectNewest: selectNewest && visible,
    focusOutputId,
  });
}

export function renderGallery(outputs, { selectNewest = false } = {}) {
  state.recentOutputCatalog = sortRecentOutputCatalog(outputs || []);
  applyAndRenderGallery({ selectNewest });
}
