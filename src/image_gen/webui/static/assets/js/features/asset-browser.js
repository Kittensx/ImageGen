import { api } from "../api.js";
import { $, notify } from "../utils.js";
import { setComponentShellState } from "../components/component-shell.js?v=content-capabilities2";
import { bindAssetBrowserComponents } from "./asset-browser-components.js?v=asset-browser-dsv2-02-chip-sync2";

const DETAIL_FETCH_TIMEOUT_MS = 15000;
const PROVIDER_PAGE_SPACING_MS = 500;
const PREVIEW_BATCH_SIZE = 25;
const PREVIEW_FETCH_CONCURRENCY = 6;
const PREVIEW_BATCH_FLUSH_DELAY_MS = 450;
const PREVIEW_FETCH_TIMEOUT_MS = 12000;
const PREVIEW_FETCH_MAX_ATTEMPTS = 3;
const PREVIEW_RETRY_BASE_DELAY_MS = 500;
const SAVED_ASSETS_STORAGE_KEY = "imagegen.asset-browser.saved-assets.v1";
const SEARCH_PREFERENCES_STORAGE_KEY = "imagegen.asset-browser.search-preferences.v1";
const DEFAULT_SEARCH_PREFERENCES = Object.freeze({ pausePreviousOnNewSearch: true, pagingMode: "continuous" });
const LOCAL_INDEX_PAGE_SIZE = 100;
const LOCAL_FILTER_DEBOUNCE_MS = 320;
const AUTO_LOCAL_LOAD_THRESHOLD_PX = 900;
// Compatibility note: the legacy assetHubIndexSearch endpoint remains available
// for older callers; DSV2-02 uses assetHubIndexQuery for the faceted local view.
// assetHubIndexSearch
const DOWNLOAD_ACTIVE_STATUSES = new Set(["queued", "resolving", "downloading", "verifying", "pausing", "cancelling"]);
const state = {
  items: [],
  selectedModelId: "",
  model: null,
  initialized: false,
  preparingDownloads: new Set(),
  mode: "browse",
  searchSessions: [],
  activeSearchSessionId: "",
  sessionRuntime: new Map(),
  searchQueue: [],
  activeProviderSessionIds: new Set(),
  searchControllers: new Map(),
  searchPreferences: { ...DEFAULT_SEARCH_PREFERENCES },
  detailRequestSerial: 0,
  activeDetailController: null,
  newSearchPromise: null,
  downloadPollTimer: 0,
  downloadSettingsLoaded: false,
  downloadJobs: [],
  sessionDownloadJobIds: new Set(),
  savedAssets: [],
  detailOverlay: null,
  previewQueue: [],
  previewActiveCount: 0,
  previewSequence: 0,
  previewLoadCache: new Map(),
  previewActiveJobs: new Set(),
  previewUrlStates: new Map(),
  visiblePreviewModelIds: new Set(),
  previewIntersectionObserver: null,
  detailGalleryIndexByVersion: new Map(),
  localFilterTimer: 0,
  keywordCommitTimer: 0,
  localQuerySerial: 0,
  localQueryControllers: new Map(),
  gallerySettings: { detailFetchMode: "current_only", libraryGalleryMode: "hero_only", retentionMode: "days", retentionDays: 7, maxCacheGiB: 10 },
  gallerySettingsLoaded: false,
  galleryImageUrls: new Map(),
  detailGalleryRenderSerial: 0,
  autoLocalLoadPending: false,
};

function sleep(ms) { return new Promise((resolve) => window.setTimeout(resolve, ms)); }

function normalizeSearchPreferences(value = {}) {
  return {
    pausePreviousOnNewSearch: value?.pausePreviousOnNewSearch !== false,
    pagingMode: String(value?.pagingMode || "continuous") === "manual" ? "manual" : "continuous",
  };
}

function loadSearchPreferences() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(SEARCH_PREFERENCES_STORAGE_KEY) || "{}");
    state.searchPreferences = normalizeSearchPreferences(parsed);
  } catch {
    state.searchPreferences = { ...DEFAULT_SEARCH_PREFERENCES };
  }
  syncSearchPreferenceControls();
}

function saveSearchPreferences() {
  state.searchPreferences = normalizeSearchPreferences({
    pausePreviousOnNewSearch: $("#assetBrowserPausePreviousSearch")?.checked ?? true,
    pagingMode: $("#assetBrowserPagingMode")?.value || "continuous",
  });
  try { window.localStorage.setItem(SEARCH_PREFERENCES_STORAGE_KEY, JSON.stringify(state.searchPreferences)); } catch { /* best effort */ }
  renderResults();
  if (state.searchPreferences.pagingMode === "continuous") window.setTimeout(() => { void maybeAutoLoadLocalResults(); }, 0);
}

function syncSearchPreferenceControls() {
  const pause = $("#assetBrowserPausePreviousSearch");
  const paging = $("#assetBrowserPagingMode");
  if (pause) pause.checked = state.searchPreferences.pausePreviousOnNewSearch !== false;
  if (paging) paging.value = state.searchPreferences.pagingMode;
}

function syncDetailModeControls() {
  const current = state.detailOverlay?.getState?.().mode || "drawer";
  const select = $("#assetBrowserDetailMode");
  if (select) select.value = current;
}

function openDetailSurface() {
  state.detailOverlay?.setAvailable?.(Boolean(state.selectedModelId));
  state.detailOverlay?.open?.();
  syncDetailModeControls();
}

function collapseDetailSurface() {
  state.detailOverlay?.collapse?.();
}

function bindDetailSurfaceInteractions() {
  $("#assetBrowserDetailPanel")?.addEventListener("workspace-overlay-state-change", syncDetailModeControls);
  syncDetailModeControls();
}

function isTransientProviderError(error) {
  const code = String(error?.code || "").toLowerCase();
  const status = Number(error?.status || 0);
  return ["provider_unavailable", "provider_timeout", "provider_rate_limited", "download_network_error"].includes(code)
    || [429, 502, 503, 504].includes(status);
}

function providerRetryDelayMs(error, attempt) {
  const retryAfter = Number(error?.detail?.retryAfterSeconds || 0);
  if (retryAfter > 0) return Math.min(5000, Math.max(500, retryAfter * 1000));
  return Math.min(4000, 700 * (2 ** Math.max(0, attempt)));
}

async function providerSearchWithRetry(filters, options, sessionId, pageNumber) {
  let lastError = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      return await api.assetHubSearch(filters, options);
    } catch (error) {
      if (error?.name === "AbortError" || !isTransientProviderError(error) || attempt >= 2) throw error;
      lastError = error;
      const delay = providerRetryDelayMs(error, attempt);
      if (String(state.activeSearchSessionId) === String(sessionId)) {
        renderActiveSessionStatus(`CivitAI is temporarily unavailable while refreshing page ${pageNumber}. Retrying in ${(delay / 1000).toFixed(1)}s; cached results remain available.`);
      }
      await sleep(delay);
    }
  }
  throw lastError || new Error("Provider refresh failed.");
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "Unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let current = bytes;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) { current /= 1024; index += 1; }
  return `${current >= 100 || index === 0 ? current.toFixed(0) : current.toFixed(1)} ${units[index]}`;
}

function formatSpeed(value) {
  const speed = Number(value || 0);
  return speed > 0 ? `${formatBytes(speed)}/s` : "—";
}

function setProviderLoading(active, text = "Gathering CivitAI assets…") {
  const panel = $("#assetBrowserProviderProgress");
  const label = $("#assetBrowserProviderProgressText");
  if (panel) panel.hidden = !active;
  if (label) label.textContent = text;
  // Search/Browse/Refresh intentionally remain active. Search tabs own their
  // provider requests independently; the search preference decides whether a
  // newly created tab pauses only the previously active fetch.
}

function displayKind(value) {
  const token = String(value || "unknown").toLowerCase();
  const labels = {
    lora: "LoRA",
    checkpoint: "Checkpoint",
    vae: "VAE",
    textual_inversion: "Textual Inversion",
    upscaler: "Upscaler",
    controlnet: "ControlNet",
    workflow: "Workflow",
    other: "Other",
    unknown: "Unknown",
  };
  return labels[token] || String(value || "Asset");
}

function supportLabel(value) {
  const token = String(value || "unknown").toLowerCase();
  if (token === "supported") return "Supported";
  if (token === "unsupported") return "Unsupported";
  return "Unknown";
}

function supportClass(value) {
  const token = String(value || "unknown").toLowerCase();
  return token === "supported" ? "is-supported" : token === "unsupported" ? "is-unsupported" : "is-unknown";
}

function firstPreview(model) {
  for (const version of model?.versions || []) {
    const preview = (version?.previews || []).find((item) => item?.kind === "image" && item?.url);
    if (preview) return preview.url;
  }
  return "";
}

function providerPreview(model) {
  const remote = String(model?.providerPreviewUrl || firstPreview(model) || "").trim();
  if (remote) return remote;
  const source = String(model?.localPreviewSource || "").trim().toLowerCase();
  return source === "civitai_cache" ? String(model?.localPreviewUrl || "").trim() : "";
}

function localPreview(model) {
  const source = String(model?.localPreviewSource || "").trim().toLowerCase();
  if (source === "civitai_cache") return "";
  return String(model?.localPreviewUrl || "").trim();
}

function localPreviewLabel(model) {
  const source = String(model?.localPreviewSource || "").trim().toLowerCase();
  if (source === "generated_output") return "User generated";
  if (source === "user_upload") return "User image";
  return "Local replacement";
}

function previewMode() {
  return $("#assetBrowserPreviewMode")?.value || "any";
}

function hasPreviewForMode(model, mode = previewMode()) {
  const provider = Boolean(providerPreview(model));
  const local = Boolean(localPreview(model));
  if (mode === "provider") return provider;
  if (mode === "local") return local;
  if (mode === "both") return provider && local;
  return true;
}

function providerPreviewRating(model, url = providerPreview(model)) {
  const selectedUrl = String(url || "");
  for (const version of model?.versions || []) {
    for (const preview of version?.previews || []) {
      if (selectedUrl && String(preview?.url || "") !== selectedUrl) continue;
      const maturity = preview?.maturity || {};
      const levels = Array.isArray(maturity?.levels) ? maturity.levels.map(String) : [];
      const order = { PG: 0, PG13: 1, R: 2, X: 3, XXX: 4, Blocked: 5 };
      const known = levels.filter((level) => Object.hasOwn(order, level));
      if (String(maturity?.state || "") === "known" && known.length) {
        return known.sort((left, right) => order[right] - order[left])[0];
      }
      return "Unknown";
    }
  }
  return "Unknown";
}

function isMatureProviderPreview(model, url = providerPreview(model)) {
  return ["R", "X", "XXX", "Blocked"].includes(providerPreviewRating(model, url));
}

function maturePreviewModeForSession(sessionId = state.activeSearchSessionId) {
  const session = sessionById(sessionId);
  return canonicalResultFilters(session?.resultFilters || {}).maturePreviewMode || "show";
}

function previewEntries(model, mode = previewMode(), matureMode = maturePreviewModeForSession()) {
  const provider = providerPreview(model);
  const local = localPreview(model);
  const entries = [];
  const providerMature = provider ? isMatureProviderPreview(model, provider) : false;
  const providerEntry = provider && !(providerMature && matureMode === "hide")
    ? { url: provider, label: "CivitAI", source: "provider", mature: providerMature, presentation: providerMature ? matureMode : "show" }
    : null;
  if (mode === "provider") {
    if (providerEntry) entries.push(providerEntry);
    return entries;
  }
  if (mode === "local") {
    if (local) entries.push({ url: local, label: localPreviewLabel(model), source: "local", mature: false, presentation: "show" });
    return entries;
  }
  if (providerEntry) entries.push(providerEntry);
  if (local) entries.push({ url: local, label: localPreviewLabel(model), source: "local", mature: false, presentation: "show" });
  return entries;
}

// P3A policy: search/browse cards own exactly one hero preview. Gallery images
// remain metadata until the user opens Asset Details, where they are fetched
// one-at-a-time on explicit previous/next navigation. This prevents a broad
// search from turning thousands of discovered records into thousands of extra
// gallery-image downloads.
function searchCardPreviewEntries(model, mode = previewMode(), matureMode = maturePreviewModeForSession()) {
  const entries = previewEntries(model, mode, matureMode);
  if (!entries.length) return [];
  // Prefer the requested source; for Any/Both the provider hero remains the
  // stable search-card identity, falling back to a local preview only when the
  // provider hero is unavailable/hidden by policy.
  const preferred = mode === "local"
    ? entries.find((entry) => entry.source === "local")
    : entries.find((entry) => entry.source === "provider") || entries.find((entry) => entry.source === "local");
  return preferred ? [preferred] : [];
}

function versionLabels(model) {
  const labels = [];
  for (const version of model?.versions || []) {
    const token = version?.architecture || version?.baseModel || "";
    if (token && !labels.includes(token)) labels.push(token);
  }
  return labels.slice(0, 3);
}

function badge(text, className = "") {
  const span = document.createElement("span");
  span.className = `asset-browser-badge${className ? ` ${className}` : ""}`;
  span.textContent = text;
  return span;
}

function preloadImage(url) {
  const selected = String(url || "").trim();
  if (!selected) return Promise.reject(new Error("Preview URL is empty."));
  const cached = state.previewLoadCache.get(selected);
  if (cached) return cached;

  const promise = new Promise((resolve, reject) => {
    const image = new window.Image();
    image.alt = "";
    image.decoding = "async";
    image.referrerPolicy = "no-referrer";
    let settled = false;
    const timeout = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      image.src = "";
      reject(new Error(`Timed out loading preview: ${selected}`));
    }, PREVIEW_FETCH_TIMEOUT_MS);

    image.onload = async () => {
      if (settled) return;
      try {
        if (typeof image.decode === "function") await image.decode();
      } catch {
        // A decoded browser-cache hit is ideal, but a cross-origin decode()
        // rejection does not make an otherwise loaded preview unusable.
      }
      settled = true;
      window.clearTimeout(timeout);
      resolve(selected);
    };
    image.onerror = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      reject(new Error(`Unable to load preview: ${selected}`));
    };
    image.src = selected;
  });
  state.previewLoadCache.set(selected, promise);
  void promise.catch(() => {
    if (state.previewLoadCache.get(selected) === promise) state.previewLoadCache.delete(selected);
  });
  return promise;
}

function previewEntryState(modelId, url, runtime = activeRuntime()) {
  const record = runtime?.previewStates?.get(String(modelId || ""));
  const urlState = record?.urlStates?.get?.(String(url || ""));
  if (urlState?.status === "loaded") return "loaded";
  if (urlState?.status === "failed") return "failed";
  return "pending";
}

function createPreviewPane(entry, modelId) {
  const pane = document.createElement("div");
  const visibility = String(entry?.visibility || (entry?.presentation === "blur" ? "blur" : entry?.presentation === "hide" ? "hide" : "show"));
  const status = visibility === "hide" ? "hidden" : previewEntryState(modelId, entry.url);
  pane.className = `asset-browser-card-preview-pane is-${entry.source} is-${visibility} is-${status === "loaded" ? "loaded" : status === "failed" ? "error" : status === "hidden" ? "hidden" : "loading"}`;
  if (entry?.mature && entry?.presentation === "blur") pane.classList.add("is-mature-blurred");

  const label = document.createElement("span");
  label.className = "asset-browser-preview-source-label";
  label.textContent = entry.label;

  if (visibility === "hide") {
    const fallback = document.createElement("div");
    fallback.className = "asset-browser-card-fallback asset-browser-preview-policy-placeholder";
    fallback.textContent = "Mature preview hidden";
    pane.append(fallback, label);
    return pane;
  }

  if (status === "loaded") {
    // Keep the waiting surface until this exact DOM image is completely loaded
    // and decoded. Search cards must change waiting -> complete image atomically;
    // they never reveal progressive paints, fades, or gallery cycling.
    const image = document.createElement("img");
    image.className = "asset-browser-card-preview";
    image.alt = `${entry.label} preview`;
    image.loading = "eager";
    image.decoding = "async";
    image.referrerPolicy = "no-referrer";
    image.hidden = true;

    const placeholder = document.createElement("div");
    placeholder.className = "asset-browser-card-preview-placeholder";
    placeholder.setAttribute("aria-hidden", "true");
    const shimmer = document.createElement("div");
    shimmer.className = "asset-browser-card-preview-shimmer";
    placeholder.append(shimmer);

    let revealed = false;
    const revealCompleteImage = async () => {
      if (revealed) return;
      try {
        if (typeof image.decode === "function") await image.decode();
      } catch {
        // load() already proved the resource is usable; cross-origin cache paths
        // may still reject decode() even though the image is complete.
      }
      if (!image.complete || !image.naturalWidth) return;
      revealed = true;
      image.hidden = false;
      placeholder.remove();
      pane.classList.add("is-dom-ready");
    };
    image.addEventListener("load", () => { void revealCompleteImage(); }, { once: true });
    image.addEventListener("error", () => {
      placeholder.classList.add("is-error");
      placeholder.replaceChildren();
      placeholder.textContent = entry.label;
    }, { once: true });
    image.src = entry.url;
    if (image.complete && image.naturalWidth) void revealCompleteImage();
    pane.append(image, placeholder, label);
    return pane;
  }

  if (status === "failed") {
    const fallback = document.createElement("div");
    fallback.className = "asset-browser-card-fallback";
    fallback.textContent = entry.label;
    pane.append(fallback, label);
    return pane;
  }

  const placeholder = document.createElement("div");
  placeholder.className = "asset-browser-card-preview-placeholder";
  placeholder.setAttribute("aria-hidden", "true");
  const shimmer = document.createElement("div");
  shimmer.className = "asset-browser-card-preview-shimmer";
  placeholder.append(shimmer);
  pane.append(placeholder, label);
  return pane;
}

function renderCardMedia(media, model) {
  const previews = searchCardPreviewEntries(model);
  media.className = "asset-browser-card-media";
  const nodes = [];
  if (previews.length) {
    previews.forEach((entry) => nodes.push(createPreviewPane(entry, model?.remoteModelId)));
  } else {
    const fallback = document.createElement("div");
    fallback.className = "asset-browser-card-fallback";
    fallback.textContent = displayKind(model.assetKind).toUpperCase();
    nodes.push(fallback);
  }
  media.replaceChildren(...nodes);
}

function renderCardCopy(copy, model) {
  copy.replaceChildren();
  const title = document.createElement("strong");
  title.className = "asset-browser-card-title";
  title.textContent = model.name || `Model ${model.remoteModelId}`;
  const meta = document.createElement("div");
  meta.className = "asset-browser-card-meta";
  meta.textContent = model.creator ? `by ${model.creator}` : `${model.providerId || "provider"} #${model.remoteModelId}`;
  const badges = document.createElement("div");
  badges.className = "asset-browser-card-badges";
  badges.append(badge(displayKind(model.assetKind)));
  versionLabels(model).forEach((label) => badges.append(badge(label)));
  badges.append(badge(supportLabel(model.supportState), supportClass(model.supportState)));
  if (model.libraryStatus === "installed") badges.append(badge("In library", "is-installed"));
  if (localPreview(model)) badges.append(badge(localPreviewLabel(model), "is-local-preview"));
  (model.searchMatches || []).slice(0, 1).forEach((label) => badges.append(badge(label)));
  copy.append(title, meta, badges);
}

function resultPreviewSignature(model) {
  return searchCardPreviewEntries(model).map((entry) => `${entry.source}:${entry.url}`).join("|");
}

function resultPreviewStateSignature(model) {
  const modelId = String(model?.remoteModelId || "");
  return searchCardPreviewEntries(model).map((entry) => `${entry.source}:${entry.url}:${previewEntryState(modelId, entry.url)}`).join("|");
}

function card(model) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `asset-browser-card${state.selectedModelId === model.remoteModelId ? " is-selected" : ""}`;
  button.dataset.modelId = model.remoteModelId;
  button.dataset.previewSignature = resultPreviewSignature(model);
  button.dataset.previewStateSignature = resultPreviewStateSignature(model);

  const media = document.createElement("div");
  renderCardMedia(media, model);

  const copy = document.createElement("div");
  copy.className = "asset-browser-card-copy";
  renderCardCopy(copy, model);
  button.append(media, copy);
  button.addEventListener("click", () => selectModel(model.remoteModelId));
  return button;
}

function sessionById(sessionId) {
  return state.searchSessions.find((item) => String(item.sessionId) === String(sessionId)) || null;
}

function activeSession() {
  return sessionById(state.activeSearchSessionId);
}

function runtimeFor(sessionId) {
  const key = String(sessionId || "");
  if (!state.sessionRuntime.has(key)) {
    state.sessionRuntime.set(key, {
      items: [],
      cursor: "",
      providerCount: 0,
      cachedCount: 0,
      stopRequested: false,
      pauseRequested: false,
      forceRefresh: false,
      pagesThisRun: 0,
      initialized: false,
      stagedOrder: [],
      previewStates: new Map(),
      publishedIds: new Set(),
      previewFlushTimer: 0,
      discoveryTerminal: false,
      previewGeneration: 0,
      candidateCount: 0,
      matchCount: 0,
      localNextOffset: null,
      facets: {},
      resultFilters: {},
      keywordDraft: "",
      localQuerySerial: 0,
      localPageLoading: false,
    });
  }
  return state.sessionRuntime.get(key);
}

function activeRuntime() {
  return runtimeFor(state.activeSearchSessionId);
}

function previewModelId(model) {
  return String(model?.remoteModelId || "");
}

function previewModeForSession(sessionId) {
  const session = sessionById(sessionId);
  return String(normalizeSessionFilters(session).previewFilter || "any");
}

function resetPreviewStaging(sessionId) {
  const selected = String(sessionId || "");
  const runtime = runtimeFor(selected);
  runtime.previewGeneration += 1;
  runtime.stagedOrder = [];
  runtime.previewStates = new Map();
  runtime.publishedIds = new Set();
  runtime.discoveryTerminal = false;
  if (runtime.previewFlushTimer) {
    window.clearTimeout(runtime.previewFlushTimer);
    runtime.previewFlushTimer = 0;
  }
  state.previewQueue = state.previewQueue.filter((job) => String(job.sessionId) !== selected);
  if (String(state.activeSearchSessionId || "") === selected) state.visiblePreviewModelIds.clear();
}

function summarizePreviewRecord(record) {
  if (!record?.urlStates || record.urlStates.size === 0) {
    return { status: "unavailable", total: 0, loaded: 0, pending: 0, failed: 0 };
  }
  let loaded = 0;
  let pending = 0;
  let failed = 0;
  record.urlStates.forEach((entry) => {
    const status = String(entry?.status || "pending");
    if (status === "loaded") loaded += 1;
    else if (status === "failed") failed += 1;
    else pending += 1;
  });
  const status = pending > 0 ? "pending" : loaded > 0 ? "ready" : "failed";
  return { status, total: record.urlStates.size, loaded, pending, failed };
}

function syncPreviewRecordStatus(record) {
  const summary = summarizePreviewRecord(record);
  record.status = summary.status;
  record.pending = summary.pending;
  record.loaded = summary.loaded;
  record.failed = summary.failed;
  return summary;
}

function previewCounts(runtime = activeRuntime()) {
  const modelIds = new Set((runtime?.items || []).map(previewModelId).filter(Boolean));
  let ready = 0;
  let pending = 0;
  let failed = 0;
  let unavailable = 0;
  modelIds.forEach((modelId) => {
    const record = runtime?.previewStates?.get(modelId);
    const status = record ? syncPreviewRecordStatus(record).status : "pending";
    if (status === "ready") ready += 1;
    else if (status === "failed") failed += 1;
    else if (status === "unavailable") unavailable += 1;
    else pending += 1;
  });
  let published = 0;
  runtime?.publishedIds?.forEach((modelId) => { if (modelIds.has(String(modelId))) published += 1; });
  const resolved = ready + failed + unavailable;
  return {
    discovered: modelIds.size,
    ready,
    pending,
    failed,
    unavailable,
    resolved,
    waiting: pending,
    published,
    readyWaiting: Math.max(0, resolved - published),
  };
}

function previewCountSummary(counts) {
  return `${counts.ready} previews ready · ${counts.pending} pending · ${counts.failed} failed · ${counts.unavailable} unavailable`;
}

function publishedItems(runtime = activeRuntime()) {
  const published = runtime?.publishedIds || new Set();
  return (runtime?.items || []).filter((item) => published.has(previewModelId(item)));
}

function previewJobPriority(job) {
  const sessionId = String(job?.sessionId || "");
  const modelId = String(job?.modelId || "");
  if (sessionId === String(state.activeSearchSessionId || "") && modelId === String(state.selectedModelId || "")) return -1000;
  if (job?.urgent) return -500;
  if (sessionId === String(state.activeSearchSessionId || "") && state.visiblePreviewModelIds.has(modelId)) return -250;
  if (sessionId === String(state.activeSearchSessionId || "")) return 0;
  return 100;
}

function previewJobKey(sessionId, modelId, url, generation) {
  return `${String(sessionId || "")}::${String(modelId || "")}::${Number(generation || 0)}::${String(url || "")}`;
}

function enqueuePreviewJob(sessionId, modelId, url, signature, generation, attempt = 0) {
  const selectedUrl = String(url || "").trim();
  if (!selectedUrl) return;
  const key = previewJobKey(sessionId, modelId, selectedUrl, generation);
  if (state.previewActiveJobs.has(key)) return;
  if (state.previewQueue.some((job) => previewJobKey(job.sessionId, job.modelId, job.url, job.generation) === key)) return;
  state.previewQueue.push({
    sessionId: String(sessionId || ""),
    modelId: String(modelId || ""),
    url: selectedUrl,
    signature: String(signature || ""),
    generation: Number(generation || 0),
    attempt: Number(attempt || 0),
    sequence: state.previewSequence += 1,
    urgent: false,
  });
}

function prioritizeModelPreview(sessionId, modelId) {
  const selectedSession = String(sessionId || "");
  const selectedModel = String(modelId || "");
  const runtime = state.sessionRuntime.get(selectedSession);
  const record = runtime?.previewStates?.get(selectedModel);
  if (record) ensurePreviewJobsForRecord(selectedSession, selectedModel, record);
  state.previewQueue.forEach((job) => {
    if (String(job.sessionId) === selectedSession && String(job.modelId) === selectedModel) job.urgent = true;
  });
  pumpPreviewQueue();
}

function ensurePreviewJobsForRecord(sessionId, modelId, record) {
  const selected = String(sessionId || "");
  if (!record?.urlStates) return;
  record.urlStates.forEach((urlState, url) => {
    if (["loaded", "failed"].includes(String(urlState?.status || ""))) return;
    const attempt = Math.max(0, Number(urlState?.attempts || 0));
    enqueuePreviewJob(selected, modelId, url, record.signature, record.generation, Math.min(attempt, PREVIEW_FETCH_MAX_ATTEMPTS - 1));
  });
  syncPreviewRecordStatus(record);
}

function rebuildMissingPreviewWork(sessionId) {
  const selected = String(sessionId || "");
  const runtime = state.sessionRuntime.get(selected);
  if (!runtime) return;
  (runtime.items || []).forEach((model) => {
    const modelId = previewModelId(model);
    const record = runtime.previewStates.get(modelId);
    if (!record) {
      stageSearchResults(selected, [model], { publishImmediately: runtime.publishedIds.has(modelId) });
      return;
    }
    ensurePreviewJobsForRecord(selected, modelId, record);
  });
  pumpPreviewQueue();
}

function stageSearchResults(sessionId, models = [], { publishImmediately = false } = {}) {
  const selected = String(sessionId || "");
  const runtime = runtimeFor(selected);
  const mode = previewModeForSession(selected);
  const generation = runtime.previewGeneration;

  (Array.isArray(models) ? models : []).forEach((model) => {
    const modelId = previewModelId(model);
    if (!modelId) return;
    if (!runtime.stagedOrder.includes(modelId)) runtime.stagedOrder.push(modelId);
    // Persistent/local results use the replay fast lane: card metadata is known
    // already, so never make the user wait for preview networking before the
    // prior search becomes visible again. Preview fetching still runs below.
    if (publishImmediately) runtime.publishedIds.add(modelId);

    const entries = searchCardPreviewEntries(model, mode, maturePreviewModeForSession(selected));
    const urls = [...new Set(entries.map((entry) => String(entry?.url || "").trim()).filter(Boolean))];
    const signature = entries.map((entry) => `${entry.source}:${entry.url}`).join("|");
    const prior = runtime.previewStates.get(modelId);
    if (prior && prior.signature === signature && prior.generation === generation) {
      ensurePreviewJobsForRecord(selected, modelId, prior);
      return;
    }

    if (!urls.length) {
      runtime.previewStates.set(modelId, { signature, generation, status: "unavailable", pending: 0, loaded: 0, failed: 0, urlStates: new Map() });
      if (runtime.publishedIds.has(modelId) && String(state.activeSearchSessionId || "") === selected) refreshResultCardPreview(modelId);
      else schedulePreviewBatchFlush(selected);
      return;
    }

    const urlStates = new Map();
    urls.forEach((url) => {
      const globalState = state.previewUrlStates.get(url);
      urlStates.set(url, {
        status: globalState?.status === "loaded" ? "loaded" : "pending",
        attempts: 0,
        lastError: "",
      });
    });
    const record = {
      signature,
      generation,
      status: "pending",
      pending: 0,
      loaded: 0,
      failed: 0,
      urlStates,
    };
    syncPreviewRecordStatus(record);
    runtime.previewStates.set(modelId, record);
    ensurePreviewJobsForRecord(selected, modelId, record);
  });
  pumpPreviewQueue();
}

function updatePreviewUiForJob(job) {
  const selected = String(job?.sessionId || "");
  const runtime = state.sessionRuntime.get(selected);
  if (!runtime) return;
  const record = runtime.previewStates.get(String(job?.modelId || ""));
  const summary = record ? syncPreviewRecordStatus(record) : null;
  if (runtime.publishedIds.has(String(job?.modelId || "")) && String(state.activeSearchSessionId || "") === selected) {
    refreshResultCardPreview(String(job.modelId || ""));
    renderActiveSessionStatus();
  }
  if (summary && summary.status !== "pending") schedulePreviewBatchFlush(selected);
}

function resolvePreviewJob(job, ok, error = null) {
  const runtime = state.sessionRuntime.get(String(job.sessionId || ""));
  if (!runtime || Number(runtime.previewGeneration || 0) !== Number(job.generation || 0)) return;
  const record = runtime.previewStates.get(String(job.modelId || ""));
  if (!record || record.signature !== String(job.signature || "") || record.generation !== Number(job.generation || 0)) return;
  const urlState = record.urlStates?.get?.(String(job.url || ""));
  if (!urlState) return;

  if (ok) {
    urlState.status = "loaded";
    urlState.lastError = "";
    state.previewUrlStates.set(String(job.url || ""), { status: "loaded", updatedAt: Date.now() });
    syncPreviewRecordStatus(record);
    updatePreviewUiForJob(job);
    return;
  }

  const completedAttempts = Number(job.attempt || 0) + 1;
  urlState.attempts = completedAttempts;
  urlState.lastError = String(error?.message || "Preview load failed.");
  if (completedAttempts < PREVIEW_FETCH_MAX_ATTEMPTS) {
    urlState.status = "retrying";
    syncPreviewRecordStatus(record);
    updatePreviewUiForJob(job);
    const delay = PREVIEW_RETRY_BASE_DELAY_MS * (2 ** Math.max(0, completedAttempts - 1));
    window.setTimeout(() => {
      const currentRuntime = state.sessionRuntime.get(String(job.sessionId || ""));
      const currentRecord = currentRuntime?.previewStates?.get(String(job.modelId || ""));
      const currentUrlState = currentRecord?.urlStates?.get?.(String(job.url || ""));
      if (!currentRuntime || !currentRecord || !currentUrlState) return;
      if (Number(currentRuntime.previewGeneration || 0) !== Number(job.generation || 0) || currentRecord.signature !== String(job.signature || "")) return;
      if (currentUrlState.status === "loaded" || currentUrlState.status === "failed") return;
      currentUrlState.status = "pending";
      enqueuePreviewJob(job.sessionId, job.modelId, job.url, job.signature, job.generation, completedAttempts);
      pumpPreviewQueue();
    }, delay);
    return;
  }

  urlState.status = "failed";
  state.previewUrlStates.set(String(job.url || ""), { status: "failed", updatedAt: Date.now(), error: urlState.lastError });
  syncPreviewRecordStatus(record);
  updatePreviewUiForJob(job);
}

function pumpPreviewQueue() {
  if (!state.previewQueue.length || state.previewActiveCount >= PREVIEW_FETCH_CONCURRENCY) return;
  state.previewQueue.sort((left, right) => {
    const priority = previewJobPriority(left) - previewJobPriority(right);
    return priority || Number(left.sequence || 0) - Number(right.sequence || 0);
  });

  while (state.previewActiveCount < PREVIEW_FETCH_CONCURRENCY && state.previewQueue.length) {
    const job = state.previewQueue.shift();
    const runtime = state.sessionRuntime.get(String(job?.sessionId || ""));
    const record = runtime?.previewStates?.get(String(job?.modelId || ""));
    if (!runtime || !record || Number(runtime.previewGeneration || 0) !== Number(job?.generation || 0) || record.signature !== String(job?.signature || "")) continue;
    const urlState = record.urlStates?.get?.(String(job?.url || ""));
    if (!urlState || ["loaded", "failed"].includes(String(urlState.status || ""))) continue;
    urlState.status = "loading";
    urlState.attempts = Math.max(Number(urlState.attempts || 0), Number(job.attempt || 0));
    syncPreviewRecordStatus(record);
    const jobKey = previewJobKey(job.sessionId, job.modelId, job.url, job.generation);
    state.previewActiveJobs.add(jobKey);
    state.previewActiveCount += 1;
    void preloadImage(job.url).then(() => {
      resolvePreviewJob(job, true, null);
    }).catch((error) => {
      resolvePreviewJob(job, false, error);
    }).finally(() => {
      state.previewActiveJobs.delete(jobKey);
      state.previewActiveCount = Math.max(0, state.previewActiveCount - 1);
      pumpPreviewQueue();
    });
  }
}

function readyUnpublishedModelIds(runtime) {
  const available = new Set((runtime?.items || []).map(previewModelId).filter(Boolean));
  return (runtime?.stagedOrder || []).filter((modelId) => {
    if (!available.has(String(modelId)) || runtime.publishedIds.has(String(modelId))) return false;
    return ["ready", "failed", "unavailable"].includes(String(runtime.previewStates.get(String(modelId))?.status || ""));
  });
}

function flushReadyResultBatch(sessionId) {
  const selected = String(sessionId || "");
  const runtime = state.sessionRuntime.get(selected);
  if (!runtime) return;
  if (runtime.previewFlushTimer) {
    window.clearTimeout(runtime.previewFlushTimer);
    runtime.previewFlushTimer = 0;
  }
  const ready = readyUnpublishedModelIds(runtime);
  if (!ready.length) return;
  ready.slice(0, PREVIEW_BATCH_SIZE).forEach((modelId) => runtime.publishedIds.add(String(modelId)));
  if (String(state.activeSearchSessionId || "") === selected) {
    syncActiveItems();
    renderResults();
    renderActiveSessionStatus();
  }
  if (ready.length > PREVIEW_BATCH_SIZE) {
    runtime.previewFlushTimer = window.setTimeout(() => flushReadyResultBatch(selected), 60);
  }
}

function schedulePreviewBatchFlush(sessionId) {
  const selected = String(sessionId || "");
  const runtime = state.sessionRuntime.get(selected);
  if (!runtime) return;
  const ready = readyUnpublishedModelIds(runtime);
  if (ready.length >= PREVIEW_BATCH_SIZE) {
    flushReadyResultBatch(selected);
    return;
  }
  const counts = previewCounts(runtime);
  if (counts.waiting === 0 && ready.length) {
    flushReadyResultBatch(selected);
    return;
  }
  if (!ready.length || runtime.previewFlushTimer) return;
  runtime.previewFlushTimer = window.setTimeout(() => flushReadyResultBatch(selected), PREVIEW_BATCH_FLUSH_DELAY_MS);
}

function syncActiveItems() {
  const runtime = activeRuntime();
  state.items = runtime?.items || [];
}

function formatSessionTime(value) {
  const timestamp = Number(value || 0);
  if (!timestamp) return "never";
  try { return new Date(timestamp * 1000).toLocaleString(); } catch { return "unknown"; }
}

function sessionStatusLabel(session) {
  const status = String(session?.status || "idle");
  const waiting = session?.sessionId ? previewCounts(runtimeFor(session.sessionId)).waiting : 0;
  if (status === "running") return "refreshing";
  if (status === "queued") return "queued";
  if (status === "stopping") return "stopping";
  if (status === "paused") return waiting > 0 ? "paused · images" : "paused";
  if (status === "completed") return waiting > 0 ? "finishing previews" : "ready";
  if (status === "failed") return waiting > 0 ? "failed · images" : "failed";
  if (status === "stopped") return waiting > 0 ? "stopped · images" : (session?.partial ? "stopped" : "ready");
  if (status === "creating") return "creating";
  return "idle";
}

function renderSearchTabs() {
  const root = $("#assetBrowserSearchTabs");
  if (!root) return;
  const nodes = [];
  state.searchSessions.forEach((session) => {
    if (session.closed) return;
    const wrap = document.createElement("div");
    wrap.className = `asset-browser-search-tab-wrap${String(session.sessionId) === String(state.activeSearchSessionId) ? " is-active" : ""}`;
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "asset-browser-search-tab";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(session.sessionId) === String(state.activeSearchSessionId) ? "true" : "false");
    const title = document.createElement("span");
    title.textContent = session.title || "Search";
    const status = document.createElement("span");
    const label = sessionStatusLabel(session);
    status.className = `asset-browser-search-tab-status is-${String(session.status || "idle")}`;
    status.textContent = label;
    tab.append(title, status);
    tab.addEventListener("click", () => {
      if (String(session.status || "") === "creating") {
        state.activeSearchSessionId = String(session.sessionId);
        applyFiltersToControls(session);
        syncKeywordComposerForSession(session.sessionId);
        resetDetailPane();
        syncActiveItems();
        renderSearchTabs();
        renderResults();
        renderActiveSessionStatus("Creating new search tab…");
        return;
      }
      void activateSearchSession(session.sessionId);
    });

    const close = document.createElement("button");
    close.type = "button";
    close.className = "asset-browser-search-tab-close";
    close.title = "Close search tab";
    close.setAttribute("aria-label", `Close ${session.title || "search"}`);
    close.textContent = "×";
    close.disabled = String(session.status || "") === "creating";
    close.addEventListener("click", (event) => {
      event.stopPropagation();
      void closeSearchSession(session.sessionId);
    });
    wrap.append(tab, close);
    nodes.push(wrap);
  });
  root.replaceChildren(...nodes);

  const toggle = $("#assetBrowserStopSearch");
  const session = activeSession();
  if (toggle) {
    const status = String(session?.status || "");
    const canPause = ["queued", "running"].includes(status);
    const canResume = ["paused", "stopped", "failed"].includes(status) && Boolean(runtimeFor(session?.sessionId).cursor || session?.nextCursor || Number(session?.providerResultCount || 0) === 0);
    toggle.disabled = !(canPause || canResume);
    toggle.dataset.action = canResume ? "resume" : "pause";
    toggle.textContent = canResume ? "Resume" : (status === "stopping" ? "Stopping…" : "Pause");
    toggle.title = canResume ? "Resume fetching this search. Other resumed search tabs may continue at the same time." : "Pause provider fetching without losing this tab, its results, or its continuation cursor.";
  }
}

function renderActiveSessionStatus(message = "") {
  const status = $("#assetBrowserStatus");
  if (!status) return;
  if (message) {
    status.textContent = message;
    return;
  }
  const session = activeSession();
  const runtime = activeRuntime();
  if (!session) {
    status.textContent = "Create a search tab, then Search or Browse.";
    return;
  }
  const counts = previewCounts(runtime);
  const publishedVisible = publishedItems(runtime).filter((item) => hasPreviewForMode(item)).length;
  const refreshed = session.lastProviderRefreshAtUnix
    ? ` Provider refreshed ${formatSessionTime(session.lastProviderRefreshAtUnix)}.`
    : " Provider has not been refreshed for this tab yet.";
  const partial = session.partial ? " Partial results are retained." : "";
  const batching = counts.readyWaiting > 0 ? ` · ${counts.readyWaiting} resolved card(s) waiting for the next display batch` : "";
  const candidates = Number(runtime.candidateCount || session.cachedResultCount || 0);
  const matches = Number(runtime.matchCount || session.resultCount || 0);
  const previews = ` ${matches.toLocaleString()} matching of ${candidates.toLocaleString()} candidates · ${publishedVisible.toLocaleString()} loaded cards · ${previewCountSummary(counts)}${batching}.`;
  status.textContent = `${sessionStatusLabel(session)}.${previews}${refreshed}${partial}`;
}

function syncProviderProgress() {
  const activeIds = [...state.activeProviderSessionIds];
  if (!activeIds.length) {
    setProviderLoading(false);
    return;
  }
  const preferredId = activeIds.includes(String(state.activeSearchSessionId)) ? String(state.activeSearchSessionId) : activeIds[0];
  const session = sessionById(preferredId);
  if (!session) {
    setProviderLoading(false);
    return;
  }
  const runtime = runtimeFor(session.sessionId);
  const page = Math.max(1, Number(runtime.pagesThisRun || 0) + 1);
  const count = Number(runtime.candidateCount || session.cachedResultCount || 0);
  const concurrent = activeIds.length > 1 ? ` · ${activeIds.length} searches fetching` : "";
  setProviderLoading(true, `Refreshing “${session.title || "search"}” · page ${page} · ${count.toLocaleString()} known candidates retained${concurrent}…`);
}

function syncSelectedResultCard() {
  const root = $("#assetBrowserResults");
  if (!root) return;
  root.querySelectorAll(".asset-browser-card").forEach((element) => {
    element.classList.toggle("is-selected", String(element.dataset.modelId || "") === String(state.selectedModelId || ""));
  });
}

function removeResultCard(modelId) {
  const root = $("#assetBrowserResults");
  if (!root) return false;
  const existing = root.querySelector(`.asset-browser-card[data-model-id="${CSS.escape(String(modelId || ""))}"]`);
  if (!existing) return false;
  existing.remove();
  return true;
}

function dispatchInstalledAsset(install) {
  if (!install || String(install.status || "") !== "installed") return;
  window.dispatchEvent(new CustomEvent("image-gen-asset-installed", {
    detail: {
      installId: String(install.installId || ""),
      downloadJobId: String(install.downloadJobId || ""),
      providerId: String(install.providerId || ""),
      remoteModelId: String(install.remoteModelId || ""),
      remoteVersionId: String(install.remoteVersionId || ""),
      remoteFileId: String(install.remoteFileId || ""),
      assetKind: String(install.assetKind || ""),
      installedPath: String(install.installedPath || ""),
      registryAssetId: String(install.registryAssetId || ""),
    },
  }));
}

async function syncInstalledAssetFromLocalIndex(install, selectedModelId) {
  const session = activeSession();
  if (!session) return false;
  const modelId = String(install?.remoteModelId || selectedModelId || "");
  if (!modelId) return false;
  const filters = normalizeSessionFilters(session);
  const payload = await api.assetHubIndexModel(String(install?.providerId || "civitai"), modelId);
  const indexed = payload?.model || null;
  const runtime = activeRuntime();

  if (!indexed || filters.libraryFilter === "not_installed") {
    if (filters.libraryFilter === "not_installed") {
      runtime.items = (runtime.items || []).filter((item) => String(item?.remoteModelId || "") !== modelId);
      syncActiveItems();
      removeResultCard(modelId);
      if (String(state.selectedModelId || "") === modelId) {
        resetDetailPane();
        renderActiveSessionStatus("The downloaded asset is now in your library and no longer matches the current Not in library filter.");
      }
      renderActiveSessionStatus();
      return true;
    }
    return false;
  }

  const prior = (runtime.items || []).find((item) => String(item?.remoteModelId || "") === modelId) || {};
  const merged = mergeModelRecord(prior, indexed);
  runtime.items = mergeItems(runtime.items || [], [merged]);
  stageSearchResults(session.sessionId, [merged]);
  syncActiveItems();
  renderResults();
  if (String(state.selectedModelId || "") === modelId) {
    const current = state.model && String(state.model.remoteModelId || "") === modelId ? state.model : {};
    renderModelDetail(mergeModelRecord(current, merged), "Library state synchronized from the local discovery index.");
  }
  renderActiveSessionStatus();
  return true;
}

function reconcileResultCards(root, visibleItems) {
  const localLoadMore = $("#assetBrowserLoadMoreLocal");
  const loadMore = $("#assetBrowserLoadMore");
  const existing = new Map([...root.querySelectorAll(".asset-browser-card")].map((node) => [String(node.dataset.modelId || ""), node]));
  const desiredIds = new Set(visibleItems.map((item) => String(item?.remoteModelId || "")));

  existing.forEach((node, modelId) => {
    if (!desiredIds.has(modelId)) node.remove();
  });

  visibleItems.forEach((model) => {
    const modelId = String(model?.remoteModelId || "");
    if (!modelId) return;
    let node = existing.get(modelId);
    const previewSignature = resultPreviewSignature(model);
    if (node && String(node.dataset.previewSignature || "") !== previewSignature) {
      const replacement = card(model);
      node.replaceWith(replacement);
      node = replacement;
    } else if (node) {
      const previewStateSignature = resultPreviewStateSignature(model);
      if (String(node.dataset.previewStateSignature || "") !== previewStateSignature) {
        const media = node.querySelector(".asset-browser-card-media");
        if (media) renderCardMedia(media, model);
        node.dataset.previewStateSignature = previewStateSignature;
      }
      const copy = node.querySelector(".asset-browser-card-copy");
      if (copy) renderCardCopy(copy, model);
      node.classList.toggle("is-selected", String(state.selectedModelId || "") === modelId);
    } else {
      node = card(model);
    }
    root.insertBefore(node, localLoadMore || loadMore || null);
  });
}

function refreshResultCardPreview(modelId) {
  const selected = String(modelId || "");
  if (!selected) return;
  const root = $("#assetBrowserResults");
  const escaped = globalThis.CSS?.escape ? globalThis.CSS.escape(selected) : selected.replace(/["\\]/g, "\\$&");
  const node = root?.querySelector?.(`.asset-browser-card[data-model-id="${escaped}"]`);
  if (!node) return;
  const model = (activeRuntime().items || []).find((item) => String(item?.remoteModelId || "") === selected);
  if (!model) return;
  const previewSignature = resultPreviewSignature(model);
  const previewStateSignature = resultPreviewStateSignature(model);
  if (String(node.dataset.previewSignature || "") !== previewSignature) {
    const replacement = card(model);
    node.replaceWith(replacement);
    observeVisibleResultCards(root);
    return;
  }
  if (String(node.dataset.previewStateSignature || "") === previewStateSignature) return;
  const media = node.querySelector(".asset-browser-card-media");
  if (media) renderCardMedia(media, model);
  node.dataset.previewStateSignature = previewStateSignature;
}

function observeVisibleResultCards(root) {
  state.visiblePreviewModelIds.clear();
  state.previewIntersectionObserver?.disconnect?.();
  state.previewIntersectionObserver = null;
  if (!root || typeof window.IntersectionObserver !== "function") return;
  state.previewIntersectionObserver = new window.IntersectionObserver((entries) => {
    let changed = false;
    entries.forEach((entry) => {
      const modelId = String(entry.target?.dataset?.modelId || "");
      if (!modelId) return;
      if (entry.isIntersecting) {
        if (!state.visiblePreviewModelIds.has(modelId)) {
          state.visiblePreviewModelIds.add(modelId);
          changed = true;
        }
      } else if (state.visiblePreviewModelIds.delete(modelId)) {
        changed = true;
      }
    });
    if (changed) pumpPreviewQueue();
  }, { root, rootMargin: "240px 0px", threshold: 0.01 });
  root.querySelectorAll(".asset-browser-card").forEach((node) => state.previewIntersectionObserver.observe(node));
}

function resultsNearViewportEnd() {
  const workspace = $("#assetBrowserWorkspace");
  if (!workspace) return false;
  const distance = workspace.scrollHeight - workspace.scrollTop - workspace.clientHeight;
  return distance <= AUTO_LOCAL_LOAD_THRESHOLD_PX;
}

async function maybeAutoLoadLocalResults() {
  if (state.searchPreferences.pagingMode !== "continuous" || state.autoLocalLoadPending) return false;
  const session = activeSession();
  const runtime = activeRuntime();
  if (!session || runtime.localPageLoading || runtime.localNextOffset === null || runtime.localNextOffset === undefined) return false;
  if (!resultsNearViewportEnd()) return false;
  state.autoLocalLoadPending = true;
  try {
    const loaded = await refreshLocalResults(session.sessionId, { append: true, persistFilters: false, quiet: true });
    if (loaded && runtime.localNextOffset !== null && runtime.localNextOffset !== undefined && resultsNearViewportEnd()) {
      window.setTimeout(() => { void maybeAutoLoadLocalResults(); }, 0);
    }
    return loaded;
  } finally {
    state.autoLocalLoadPending = false;
  }
}

function renderResults() {
  syncActiveItems();
  const root = $("#assetBrowserResults");
  if (!root) return;
  const session = activeSession();
  const runtime = activeRuntime();
  const visibleItems = publishedItems(runtime).filter((item) => hasPreviewForMode(item));
  reconcileResultCards(root, visibleItems);
  observeVisibleResultCards(root);
  const localLoadMore = $("#assetBrowserLoadMoreLocal");
  if (localLoadMore) {
    const hasMoreLocal = runtime.localNextOffset !== null && runtime.localNextOffset !== undefined;
    const canLoadLocal = state.searchPreferences.pagingMode === "manual" && hasMoreLocal;
    localLoadMore.hidden = !canLoadLocal;
    localLoadMore.disabled = !canLoadLocal;
    localLoadMore.textContent = `Load more matching results${canLoadLocal ? ` (${Number(runtime.items?.length || 0).toLocaleString()} / ${Number(runtime.matchCount || 0).toLocaleString()})` : ""}`;
  }
  const loadMore = $("#assetBrowserLoadMore");
  if (loadMore) {
    const canContinue = state.searchPreferences.pagingMode === "manual"
      && Boolean(runtime.cursor || session?.nextCursor)
      && ["paused", "stopped", "failed"].includes(String(session?.status || ""));
    loadMore.hidden = !canContinue;
    loadMore.disabled = !canContinue;
    loadMore.textContent = "Load more provider results";
  }
  if (state.searchPreferences.pagingMode === "continuous") window.setTimeout(() => { void maybeAutoLoadLocalResults(); }, 0);
}

function selectedFacetValues(containerId) {
  const root = $(`#${containerId}`);
  if (!root) return [];
  return [...root.querySelectorAll("[data-facet-value][aria-pressed='true']")]
    .map((node) => String(node.dataset.facetValue || "").trim())
    .filter(Boolean);
}

function maturePreviewModeValue() {
  const selected = $("#assetBrowserMaturePreviewMode")?.querySelector("button[aria-pressed='true']");
  return String(selected?.dataset?.value || "show");
}

function canonicalResultFilters(raw = {}) {
  const array = (value) => Array.isArray(value) ? [...new Set(value.map((item) => String(item || "").trim()).filter(Boolean))] : [];
  const legacyArchitecture = array(raw.baseModels);
  const supportStates = array(raw.supportStates);
  if (!supportStates.length && raw.supportFilter && raw.supportFilter !== "any") supportStates.push(String(raw.supportFilter));
  const libraryStates = array(raw.libraryStates);
  if (!libraryStates.length && raw.libraryFilter && raw.libraryFilter !== "any") libraryStates.push(String(raw.libraryFilter));
  const previewSources = array(raw.previewSources);
  if (!previewSources.length && raw.previewFilter && raw.previewFilter !== "any") previewSources.push(String(raw.previewFilter));
  const creators = array(raw.creators);
  if (!creators.length && raw.creator) creators.push(String(raw.creator));
  const keywordTerms = array(raw.keywordTerms).length
    ? array(raw.keywordTerms)
    : (String(raw.keywords || raw.keyword || "").trim() ? [String(raw.keywords || raw.keyword || "").trim()] : []);
  return {
    keywordTerms,
    // Legacy aggregate retained so older callers/tests can still inspect a string.
    keywords: keywordTerms.join(" "),
    keywordMode: ["all_words", "any_word", "exact_phrase"].includes(String(raw.keywordMode || "")) ? String(raw.keywordMode) : "all_words",
    architectures: array(raw.architectures).length ? array(raw.architectures) : legacyArchitecture,
    assetKinds: array(raw.assetKinds),
    ratings: array(raw.ratings).map((value) => String(value).replace("PG-13", "PG13")),
    ratingBasis: ["model", "author_previews", "strictest"].includes(String(raw.ratingBasis || "")) ? String(raw.ratingBasis) : "strictest",
    supportStates,
    libraryStates,
    previewSources,
    creators,
    categories: array(raw.categories),
    maturePreviewMode: ["show", "blur", "hide"].includes(String(raw.maturePreviewMode || "")) ? String(raw.maturePreviewMode) : "show",
    localSort: ["candidate_order", "safest", "most_mature", "newest", "title"].includes(String(raw.localSort || "")) ? String(raw.localSort) : "candidate_order",
  };
}

function resultFiltersFromControls() {
  const support = String($("#assetBrowserSupport")?.value || "any");
  const library = String($("#assetBrowserLibrary")?.value || "any");
  const preview = String($("#assetBrowserPreviewMode")?.value || "any");
  const creator = String($("#assetBrowserFilterCreator")?.value || "").trim();
  const committed = canonicalResultFilters(activeSession()?.resultFilters || activeRuntime()?.resultFilters || {});
  return canonicalResultFilters({
    keywordTerms: committed.keywordTerms,
    keywordMode: $("#assetBrowserKeywordMode")?.value || committed.keywordMode || "all_words",
    architectures: selectedFacetValues("assetBrowserFacetArchitecture"),
    assetKinds: selectedFacetValues("assetBrowserFacetAssetKind"),
    ratings: selectedFacetValues("assetBrowserFacetRating"),
    ratingBasis: $("#assetBrowserRatingBasis")?.value || "strictest",
    supportStates: support === "any" ? [] : [support],
    libraryStates: library === "any" ? [] : [library],
    previewSources: preview === "any" ? [] : [preview],
    creators: creator ? [creator] : [],
    categories: selectedFacetValues("assetBrowserFacetCategory"),
    maturePreviewMode: maturePreviewModeValue(),
    localSort: $("#assetBrowserLocalSort")?.value || "candidate_order",
  });
}

function controlsToFilters({ mode = state.mode, clearQuery = false } = {}) {
  const selectedMode = mode === "search" ? "search" : "browse";
  const discoveryCriteria = {
    providerId: "civitai",
    query: clearQuery || selectedMode === "browse" ? "" : ($("#assetBrowserQuery")?.value?.trim() || ""),
    assetType: $("#assetBrowserType")?.value || "any",
    providerSort: $("#assetBrowserSort")?.value || "",
    period: $("#assetBrowserPeriod")?.value || "",
    safeContent: $("#assetBrowserSafeContent")?.checked ?? true,
    mode: selectedMode,
    limit: PREVIEW_BATCH_SIZE,
  };
  const resultFilters = resultFiltersFromControls();
  return {
    provider: discoveryCriteria.providerId,
    query: discoveryCriteria.query,
    type: discoveryCriteria.assetType,
    creator: "",
    sort: discoveryCriteria.providerSort,
    period: discoveryCriteria.period,
    safeContent: discoveryCriteria.safeContent,
    mode: selectedMode,
    limit: PREVIEW_BATCH_SIZE,
    discoveryCriteria,
    resultFilters,
  };
}

function normalizeSessionFilters(session) {
  const raw = session?.filters || {};
  const discovery = session?.discoveryCriteria || {};
  const resultFilters = canonicalResultFilters(session?.resultFilters || {});
  const mode = String(discovery.mode || raw.mode || session?.mode || "browse") === "search" ? "search" : "browse";
  const previewFilter = resultFilters.previewSources.length === 1 ? resultFilters.previewSources[0] : "any";
  const supportFilter = resultFilters.supportStates.length === 1 ? resultFilters.supportStates[0] : "any";
  const libraryFilter = resultFilters.libraryStates.length === 1 ? resultFilters.libraryStates[0] : "any";
  const combined = {
    provider: String(discovery.providerId || raw.provider || session?.providerId || "civitai"),
    query: mode === "search" ? String(discovery.query || raw.query || "") : "",
    type: String(discovery.assetType || raw.type || "any"),
    creator: String(discovery.creator || ""),
    sort: String(discovery.providerSort || raw.sort || ""),
    period: String(discovery.period || raw.period || ""),
    safeContent: (discovery.safeContent ?? raw.safeContent) !== false,
    supportFilter,
    libraryFilter,
    previewFilter,
    mode,
    limit: Number(discovery.limit || raw.limit || PREVIEW_BATCH_SIZE),
    resultFilters,
  };
  combined.discoveryCriteria = {
    providerId: combined.provider,
    query: combined.query,
    assetType: combined.type,
    creator: combined.creator,
    providerSort: combined.sort,
    period: combined.period,
    safeContent: combined.safeContent,
    mode: combined.mode,
    limit: combined.limit,
  };
  return combined;
}

function setMaturePreviewMode(value) {
  const selected = ["show", "blur", "hide"].includes(String(value || "")) ? String(value) : "show";
  $("#assetBrowserMaturePreviewMode")?.querySelectorAll("button[data-value]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.value || "") === selected ? "true" : "false");
  });
}

function applyFiltersToControls(session) {
  const filters = normalizeSessionFilters(session);
  const resultFilters = filters.resultFilters;
  state.mode = filters.mode;
  const assignments = [
    ["assetBrowserQuery", filters.query],
    ["assetBrowserType", filters.type],
    ["assetBrowserSort", filters.sort],
    ["assetBrowserPeriod", filters.period],
    ["assetBrowserKeywordMode", resultFilters.keywordMode],
    ["assetBrowserRatingBasis", resultFilters.ratingBasis],
    ["assetBrowserLocalSort", resultFilters.localSort],
    ["assetBrowserSupport", filters.supportFilter],
    ["assetBrowserLibrary", filters.libraryFilter],
    ["assetBrowserPreviewMode", filters.previewFilter],
    ["assetBrowserFilterCreator", resultFilters.creators[0] || ""],
  ];
  assignments.forEach(([id, value]) => {
    const element = $(`#${id}`);
    if (element) element.value = value;
  });
  const safe = $("#assetBrowserSafeContent");
  if (safe) safe.checked = filters.safeContent;
  setMaturePreviewMode(resultFilters.maturePreviewMode);
}

function autoSessionTitle(filters, mode) {
  const query = String(filters?.query || "").trim();
  if (mode === "search" && query) return query.slice(0, 42);
  const type = String(filters?.type || "any");
  const kind = type === "any" ? "Assets" : displayKind(type);
  return `${mode === "browse" ? "Browse" : "Search"} · ${kind}`.slice(0, 120);
}

// Merge provider/index refreshes without dropping local-library state or either
// preview source. Search-session refactoring previously removed this helper
// while leaving callers in provider refresh and post-install reconciliation.
function mergeByIdentity(existing = [], incoming = [], identityKey) {
  const map = new Map((Array.isArray(existing) ? existing : []).map((item) => [String(identityKey(item) || ""), item]));
  (Array.isArray(incoming) ? incoming : []).forEach((item) => {
    const key = String(identityKey(item) || "");
    if (!key) return;
    map.set(key, { ...(map.get(key) || {}), ...item });
  });
  return [...map.values()];
}

function mergeModelRecord(prior = {}, item = {}) {
  const mergedVersions = mergeByIdentity(prior?.versions, item?.versions, (version) => version?.remoteVersionId).map((version) => {
    const priorVersion = (prior?.versions || []).find((candidate) => String(candidate?.remoteVersionId || "") === String(version?.remoteVersionId || "")) || {};
    return {
      ...priorVersion,
      ...version,
      files: mergeByIdentity(priorVersion?.files, version?.files, (file) => file?.remoteFileId),
      previews: (version?.previews?.length ? version.previews : priorVersion?.previews) || [],
      libraryStatus: version?.libraryStatus === "installed" || priorVersion?.libraryStatus === "installed"
        ? "installed"
        : (version?.libraryStatus || priorVersion?.libraryStatus || "not_installed"),
    };
  });
  return {
    ...prior,
    ...item,
    versions: mergedVersions,
    providerPreviewUrl: item?.providerPreviewUrl || prior?.providerPreviewUrl || "",
    localPreviewUrl: item?.localPreviewUrl || prior?.localPreviewUrl || "",
    localPreviewSource: item?.localPreviewSource || prior?.localPreviewSource || "",
    localAssetId: item?.localAssetId || prior?.localAssetId || null,
    localAssetType: item?.localAssetType || prior?.localAssetType || null,
    libraryStatus: item?.libraryStatus === "installed" || prior?.libraryStatus === "installed"
      ? "installed"
      : (item?.libraryStatus || prior?.libraryStatus || "not_installed"),
    firstSeenAtUnix: prior?.firstSeenAtUnix || item?.firstSeenAtUnix || null,
    lastRefreshedAtUnix: item?.lastRefreshedAtUnix || prior?.lastRefreshedAtUnix || null,
  };
}

function mergeItems(existing = [], incoming = []) {
  const map = new Map((Array.isArray(existing) ? existing : []).map((item) => [String(item?.remoteModelId || ""), item]));
  (Array.isArray(incoming) ? incoming : []).forEach((item) => {
    const key = String(item?.remoteModelId || "");
    if (!key) return;
    map.set(key, mergeModelRecord(map.get(key) || {}, item));
  });
  return [...map.values()];
}

function mergeSession(session) {
  const index = state.searchSessions.findIndex((item) => String(item.sessionId) === String(session?.sessionId));
  if (index >= 0) state.searchSessions[index] = session;
  else if (session) state.searchSessions.push(session);
  renderSearchTabs();
  return session;
}

async function patchSearchSession(sessionId, values = {}) {
  const payload = await api.assetHubUpdateSearchSession(sessionId, values);
  return mergeSession(payload?.session || sessionById(sessionId));
}

function resetDetailPane() {
  if (state.activeDetailController) {
    state.activeDetailController.abort();
    state.activeDetailController = null;
  }
  state.detailRequestSerial += 1;
  state.selectedModelId = "";
  state.model = null;
  state.detailOverlay?.setAvailable?.(false);
  const detail = $("#assetBrowserDetail");
  if (detail) detail.hidden = true;
}

function facetDisplayValue(facet, value) {
  const token = String(value || "");
  if (facet === "architecture") {
    return { "sd1.x": "SD 1.x", "sd2.x": "SD 2.x", sdxl: "SDXL", "sd3.x": "SD3 / SD3.5", flux: "Flux" }[token] || token;
  }
  if (facet === "asset_kind") return displayKind(token);
  if (facet === "rating") return token === "PG13" ? "PG-13" : token;
  if (facet === "library") return token === "installed" ? "In library" : token === "not_installed" ? "Not in library" : token;
  if (facet === "preview") return { provider: "Provider only", local: "Local only", both: "Provider + local", none: "No preview" }[token] || token;
  if (facet === "support") return supportLabel(token);
  return token;
}

function renderFacetButtons(containerId, facet, options = [], selectedValues = []) {
  const root = $(`#${containerId}`);
  if (!root) return;
  const selected = new Set((selectedValues || []).map(String));
  const nodes = [];
  (Array.isArray(options) ? options : []).forEach((option) => {
    const value = String(option?.value || "").trim();
    const count = Number(option?.count || 0);
    if (!value || count <= 0) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "asset-browser-facet-option";
    button.dataset.facetValue = value;
    button.dataset.facetName = facet;
    button.setAttribute("aria-pressed", selected.has(value) ? "true" : "false");
    const label = document.createElement("span");
    label.textContent = facetDisplayValue(facet, value);
    const badgeCount = document.createElement("span");
    badgeCount.className = "asset-browser-facet-count";
    badgeCount.textContent = count.toLocaleString();
    button.append(label, badgeCount);
    button.addEventListener("click", () => {
      button.setAttribute("aria-pressed", button.getAttribute("aria-pressed") === "true" ? "false" : "true");
      scheduleLocalFilterRefresh({ immediate: true });
    });
    nodes.push(button);
  });
  if (!nodes.length) {
    const empty = document.createElement("span");
    empty.className = "subtle";
    empty.textContent = "No values in the current candidate scope.";
    nodes.push(empty);
  }
  root.replaceChildren(...nodes);
}

function updateSelectFacetCounts(selectId, facet, options = []) {
  const select = $(`#${selectId}`);
  if (!select) return;
  const counts = new Map((Array.isArray(options) ? options : []).map((item) => [String(item?.value || ""), Number(item?.count || 0)]));
  [...select.options].forEach((option) => {
    const original = option.dataset.baseLabel || option.textContent.replace(/\s+\([\d,]+\)$/, "");
    option.dataset.baseLabel = original;
    if (option.value === "any") {
      option.textContent = original;
      option.disabled = false;
      return;
    }
    const count = counts.get(String(option.value));
    option.textContent = count === undefined ? original : `${original} (${count.toLocaleString()})`;
    option.disabled = count === 0 && option.value !== select.value;
  });
}

function renderCreatorFacet(options = []) {
  const list = $("#assetBrowserCreatorFacetOptions");
  if (!list) return;
  const nodes = (Array.isArray(options) ? options : []).slice(0, 200).map((option) => {
    const node = document.createElement("option");
    node.value = String(option?.value || "");
    node.label = `${String(option?.value || "")} (${Number(option?.count || 0).toLocaleString()})`;
    return node;
  });
  list.replaceChildren(...nodes);
}

function renderFacetControls(runtime = activeRuntime(), session = activeSession()) {
  const filters = canonicalResultFilters(session?.resultFilters || runtime?.resultFilters || {});
  const facets = runtime?.facets || {};
  renderFacetButtons("assetBrowserFacetArchitecture", "architecture", facets.architecture, filters.architectures);
  renderFacetButtons("assetBrowserFacetAssetKind", "asset_kind", facets.asset_kind, filters.assetKinds);
  renderFacetButtons("assetBrowserFacetRating", "rating", facets.rating, filters.ratings);
  renderFacetButtons("assetBrowserFacetCategory", "category", facets.category, filters.categories);
  const categoryGroup = $("#assetBrowserCategoryFacetGroup");
  if (categoryGroup) categoryGroup.hidden = !(facets.category || []).length;
  updateSelectFacetCounts("assetBrowserSupport", "support", facets.support);
  updateSelectFacetCounts("assetBrowserLibrary", "library", facets.library);
  updateSelectFacetCounts("assetBrowserPreviewMode", "preview", facets.preview);
  renderCreatorFacet(facets.creator);
}

function clearFacetSelection(facet, value = "") {
  if (facet === "keyword") {
    const session = activeSession();
    if (session) {
      const next = canonicalResultFilters(session.resultFilters || {});
      next.keywordTerms = next.keywordTerms.filter((term) => String(term) !== String(value));
      next.keywords = next.keywordTerms.join(" ");
      session.resultFilters = next;
      activeRuntime().resultFilters = next;
    }
  } else if (facet === "keywords") {
    const session = activeSession();
    if (session) {
      const next = canonicalResultFilters(session.resultFilters || {});
      next.keywordTerms = [];
      next.keywords = "";
      session.resultFilters = next;
      activeRuntime().resultFilters = next;
    }
  } else if (facet === "creator") {
    const input = $("#assetBrowserFilterCreator");
    if (input) input.value = "";
  } else if (facet === "support") {
    const select = $("#assetBrowserSupport");
    if (select) select.value = "any";
  } else if (facet === "library") {
    const select = $("#assetBrowserLibrary");
    if (select) select.value = "any";
  } else if (facet === "preview") {
    const select = $("#assetBrowserPreviewMode");
    if (select) select.value = "any";
  } else if (facet === "mature") {
    setMaturePreviewMode("show");
  } else if (facet === "sort") {
    const select = $("#assetBrowserLocalSort");
    if (select) select.value = "candidate_order";
  } else if (facet === "rating_basis") {
    const select = $("#assetBrowserRatingBasis");
    if (select) select.value = "strictest";
  } else {
    const map = {
      architecture: "assetBrowserFacetArchitecture",
      asset_kind: "assetBrowserFacetAssetKind",
      rating: "assetBrowserFacetRating",
      category: "assetBrowserFacetCategory",
    };
    const root = $(`#${map[facet] || ""}`);
    root?.querySelectorAll("[data-facet-value]").forEach((button) => {
      if (!value || String(button.dataset.facetValue || "") === String(value)) button.setAttribute("aria-pressed", "false");
    });
  }
}

function renderActiveFilterChips(session = activeSession()) {
  const root = $("#assetBrowserActiveFilters");
  if (!root) return;
  const filters = canonicalResultFilters(session?.resultFilters || {});
  const specs = [];
  filters.keywordTerms.forEach((value) => specs.push(["keyword", value, `Keyword: ${value}`]));
  filters.architectures.forEach((value) => specs.push(["architecture", value, facetDisplayValue("architecture", value)]));
  filters.assetKinds.forEach((value) => specs.push(["asset_kind", value, facetDisplayValue("asset_kind", value)]));
  filters.ratings.forEach((value) => specs.push(["rating", value, facetDisplayValue("rating", value)]));
  filters.supportStates.forEach((value) => specs.push(["support", value, facetDisplayValue("support", value)]));
  filters.libraryStates.forEach((value) => specs.push(["library", value, facetDisplayValue("library", value)]));
  filters.previewSources.forEach((value) => specs.push(["preview", value, facetDisplayValue("preview", value)]));
  filters.creators.forEach((value) => specs.push(["creator", value, `Creator: ${value}`]));
  filters.categories.forEach((value) => specs.push(["category", value, `Category: ${value}`]));
  if (filters.maturePreviewMode !== "show") specs.push(["mature", filters.maturePreviewMode, `Mature previews: ${filters.maturePreviewMode}`]);
  if (filters.localSort !== "candidate_order") specs.push(["sort", filters.localSort, `Sort: ${filters.localSort.replaceAll("_", " ")}`]);
  if (filters.ratingBasis !== "strictest" && filters.ratings.length) specs.push(["rating_basis", filters.ratingBasis, `Rating basis: ${filters.ratingBasis.replaceAll("_", " ")}`]);
  if (!specs.length) {
    const empty = document.createElement("span");
    empty.className = "subtle";
    empty.textContent = "No local filters active.";
    root.replaceChildren(empty);
    return;
  }
  root.replaceChildren(...specs.map(([facet, value, label]) => {
    const chip = document.createElement("span");
    chip.className = "asset-browser-filter-chip";
    chip.append(document.createTextNode(label));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${label}`);
    remove.addEventListener("click", () => {
      clearFacetSelection(facet, value);
      scheduleLocalFilterRefresh({ immediate: true });
    });
    chip.append(remove);
    return chip;
  }));
}

function renderLocalFilterStatus(runtime = activeRuntime()) {
  const status = $("#assetBrowserFilterStatus");
  const summary = $("#assetBrowserFacetMatchSummary");
  if (!status || !runtime) return;
  const candidateCount = Number(runtime.candidateCount || 0);
  const matchCount = Number(runtime.matchCount || 0);
  const loaded = Number(runtime.items?.length || 0);
  status.textContent = `Showing ${matchCount.toLocaleString()} matching of ${candidateCount.toLocaleString()} candidates · ${loaded.toLocaleString()} loaded in grid. Local refinement does not contact the provider.`;
  if (summary) summary.textContent = `${matchCount.toLocaleString()} / ${candidateCount.toLocaleString()}`;
}

async function refreshLocalResults(sessionId, { append = false, persistFilters = false, quiet = false, filtersOverride = null, preserveLoadedCount = 0 } = {}) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session || String(session.status || "") === "creating") return false;
  const runtime = runtimeFor(selected);
  if (runtime.localPageLoading && append) return false;
  runtime.localPageLoading = true;
  const controller = new AbortController();
  state.localQueryControllers.get(selected)?.abort?.();
  state.localQueryControllers.set(selected, controller);
  const serial = state.localQuerySerial += 1;
  runtime.localQuerySerial = serial;
  const filters = filtersOverride ? canonicalResultFilters(filtersOverride) : (persistFilters ? resultFiltersFromControls() : canonicalResultFilters(session.resultFilters || {}));
  const wantedCount = append ? Number(runtime.items?.length || 0) + LOCAL_INDEX_PAGE_SIZE : Math.max(LOCAL_INDEX_PAGE_SIZE, Number(preserveLoadedCount || 0));
  const priorLength = Number(runtime.items?.length || 0);
  let offset = append ? priorLength : 0;
  let mergedItems = append ? [...(runtime.items || [])] : [];
  let firstPage = true;
  let lastPage = null;
  let facetSnapshot = null;
  try {
    while (true) {
      const payload = await api.assetHubIndexQuery({
        sessionId: selected,
        ...(persistFilters && firstPage ? { filters } : {}),
        sort: filters.localSort,
        offset,
        limit: LOCAL_INDEX_PAGE_SIZE,
        facets: firstPage ? ["architecture", "asset_kind", "rating", "support", "library", "preview", "creator", "category"] : [],
      }, { signal: controller.signal });
      if (runtime.localQuerySerial !== serial) return false;
      const page = payload?.page || {};
      const incoming = Array.isArray(page.items) ? page.items : [];
      if (!append && firstPage) resetPreviewStaging(selected);
      mergedItems = mergeItems(mergedItems, incoming);
      if (firstPage && page.facets) facetSnapshot = page.facets;
      lastPage = { payload, page, incoming };
      const nextOffset = page.nextOffset === null || page.nextOffset === undefined ? null : Number(page.nextOffset);
      offset = nextOffset === null ? offset : nextOffset;
      if (nextOffset === null || !incoming.length || mergedItems.length >= wantedCount) break;
      firstPage = false;
    }
    if (!lastPage) return false;
    const { payload, page } = lastPage;
    // DSV2-02 compatibility invariant: the first local page previously used
    // runtime.items = incoming; P3B preserves that filtered ownership while
    // stageSearchResults(selected, incoming, { publishImmediately: true }); remains
    // allowing already-loaded pages to survive provider refreshes.
    runtime.items = mergedItems;
    runtime.candidateCount = Number(page.candidateCount || 0);
    runtime.matchCount = Number(page.matchCount || 0);
    runtime.localNextOffset = page.nextOffset === null || page.nextOffset === undefined ? null : Number(page.nextOffset);
    runtime.facets = facetSnapshot || runtime.facets || {};
    runtime.resultFilters = canonicalResultFilters(page.filters || filters);
    runtime.cachedCount = runtime.candidateCount;
    runtime.initialized = true;
    mergeSession(payload?.session || session);
    const publishItems = append ? mergedItems.slice(priorLength) : mergedItems;
    if (String(state.activeSearchSessionId) === selected) {
      applyFiltersToControls(sessionById(selected));
      renderFacetControls(runtime, sessionById(selected));
      renderActiveFilterChips(sessionById(selected));
      stageSearchResults(selected, publishItems, { publishImmediately: true });
      syncActiveItems();
      renderResults();
      renderLocalFilterStatus(runtime);
      renderActiveSessionStatus();
      window.setTimeout(() => maybeAutoLoadLocalResults(), 0);
    } else {
      stageSearchResults(selected, publishItems, { publishImmediately: true });
    }
    return true;
  } catch (error) {
    if (error?.name === "AbortError") return false;
    if (!quiet && String(state.activeSearchSessionId) === selected) {
      const status = $("#assetBrowserFilterStatus");
      if (status) status.textContent = `Local facet query failed: ${error.message}`;
    }
    return false;
  } finally {
    runtime.localPageLoading = false;
    if (state.localQueryControllers.get(selected) === controller) state.localQueryControllers.delete(selected);
  }
}

function syncKeywordComposerForSession(sessionId = state.activeSearchSessionId) {
  const input = $("#assetBrowserFilterKeywords");
  if (!input) return;
  input.value = String(runtimeFor(sessionId).keywordDraft || "");
}

function commitKeywordDraft() {
  if (state.keywordCommitTimer) {
    window.clearTimeout(state.keywordCommitTimer);
    state.keywordCommitTimer = 0;
  }
  const session = activeSession();
  if (!session || String(session.status || "") === "creating") return;
  const runtime = activeRuntime();
  const input = $("#assetBrowserFilterKeywords");
  const term = String(input?.value || runtime.keywordDraft || "").trim();
  if (!term) {
    runtime.keywordDraft = "";
    if (input) input.value = "";
    return;
  }
  const next = canonicalResultFilters(session.resultFilters || runtime.resultFilters || {});
  if (!next.keywordTerms.some((value) => String(value).toLocaleLowerCase() === String(term).toLocaleLowerCase())) {
    next.keywordTerms.push(term);
  }
  next.keywords = next.keywordTerms.join(" ");
  session.resultFilters = next;
  runtime.resultFilters = next;
  runtime.keywordDraft = "";
  if (input) input.value = "";
  renderActiveFilterChips(session);
  void refreshLocalResults(session.sessionId, { persistFilters: true, filtersOverride: next });
}

function scheduleKeywordCommit({ immediate = false } = {}) {
  if (state.keywordCommitTimer) {
    window.clearTimeout(state.keywordCommitTimer);
    state.keywordCommitTimer = 0;
  }
  const session = activeSession();
  if (!session || String(session.status || "") === "creating") return;
  const runtime = activeRuntime();
  const input = $("#assetBrowserFilterKeywords");
  runtime.keywordDraft = String(input?.value || "");
  if (!runtime.keywordDraft.trim()) return;
  if (immediate) commitKeywordDraft();
  else state.keywordCommitTimer = window.setTimeout(commitKeywordDraft, LOCAL_FILTER_DEBOUNCE_MS);
}

function commitLocalFilterControls() {
  const session = activeSession();
  if (!session || String(session.status || "") === "creating") return null;
  const runtime = activeRuntime();
  const next = resultFiltersFromControls();
  session.resultFilters = next;
  runtime.resultFilters = next;
  // Active chips are optimistic UI: adding/removing a local facet is visible
  // immediately while the SQLite facet query recalculates counts/results.
  renderActiveFilterChips(session);
  return next;
}

function scheduleLocalFilterRefresh({ immediate = false } = {}) {
  if (state.localFilterTimer) {
    window.clearTimeout(state.localFilterTimer);
    state.localFilterTimer = 0;
  }
  const session = activeSession();
  if (!session || String(session.status || "") === "creating") return;
  const run = () => {
    state.localFilterTimer = 0;
    const next = commitLocalFilterControls();
    if (!next) return;
    void refreshLocalResults(session.sessionId, { persistFilters: true, filtersOverride: next });
  };
  if (immediate) run();
  else state.localFilterTimer = window.setTimeout(run, LOCAL_FILTER_DEBOUNCE_MS);
}

async function loadIndexedForSession(sessionId, { quiet = false } = {}) {
  // Legacy fast-replay acceptance anchor: stageSearchResults(sessionId, incoming, { publishImmediately: true });
  return refreshLocalResults(sessionId, { append: false, persistFilters: false, quiet });
}

function clearLocalFilterControls() {
  const session = activeSession();
  if (session) {
    const next = canonicalResultFilters(session.resultFilters || {});
    next.keywordTerms = [];
    next.keywords = "";
    session.resultFilters = next;
    activeRuntime().resultFilters = next;
    activeRuntime().keywordDraft = "";
  }
  const assignments = [
    ["assetBrowserFilterKeywords", ""],
    ["assetBrowserKeywordMode", "all_words"],
    ["assetBrowserRatingBasis", "strictest"],
    ["assetBrowserLocalSort", "candidate_order"],
    ["assetBrowserSupport", "any"],
    ["assetBrowserLibrary", "any"],
    ["assetBrowserPreviewMode", "any"],
    ["assetBrowserFilterCreator", ""],
  ];
  assignments.forEach(([id, value]) => {
    const element = $(`#${id}`);
    if (element) element.value = value;
  });
  ["assetBrowserFacetArchitecture", "assetBrowserFacetAssetKind", "assetBrowserFacetRating", "assetBrowserFacetCategory"].forEach((id) => {
    $(`#${id}`)?.querySelectorAll("[data-facet-value]").forEach((button) => button.setAttribute("aria-pressed", "false"));
  });
  setMaturePreviewMode("show");
  scheduleLocalFilterRefresh({ immediate: true });
}

async function persistActiveDraft() {
  const session = activeSession();
  if (!session) return;
  const filters = controlsToFilters({ mode: state.mode });
  try {
    await patchSearchSession(session.sessionId, { filters, discoveryCriteria: filters.discoveryCriteria, resultFilters: filters.resultFilters, mode: state.mode });
  } catch {
    // Draft persistence is best effort; explicit Search/Browse will retry.
  }
}

async function activateSearchSession(sessionId) {
  if (!sessionById(sessionId)) return;
  if (state.keywordCommitTimer) {
    window.clearTimeout(state.keywordCommitTimer);
    state.keywordCommitTimer = 0;
  }
  if (state.activeSearchSessionId && String(state.activeSearchSessionId) !== String(sessionId)) {
    void persistActiveDraft();
  }
  state.activeSearchSessionId = String(sessionId);
  const session = activeSession();
  applyFiltersToControls(session);
  syncKeywordComposerForSession(sessionId);
  resetDetailPane();
  const runtime = activeRuntime();
  syncActiveItems();
  renderSearchTabs();
  renderResults();
  renderActiveSessionStatus();
  if (runtime.initialized) rebuildMissingPreviewWork(sessionId);
  else await loadIndexedForSession(sessionId, { quiet: false });
  pumpPreviewQueue();
}

function pendingSearchSession(filters) {
  const randomId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const now = Date.now() / 1000;
  return {
    sessionId: `pending:${randomId}`,
    title: "New search",
    providerId: "civitai",
    mode: "browse",
    filters,
    discoveryCriteria: filters.discoveryCriteria,
    resultFilters: filters.resultFilters,
    status: "creating",
    closed: false,
    partial: false,
    nextCursor: null,
    candidateCount: 0,
    resultCount: 0,
    cachedResultCount: 0,
    providerResultCount: 0,
    createdAtUnix: now,
    updatedAtUnix: now,
  };
}

function setNewSearchBusy(active) {
  const button = $("#assetBrowserNewSearch");
  if (!button) return;
  button.disabled = Boolean(active);
  button.textContent = active ? "Creating…" : "New Search";
}

async function createSearchSession({ clearQuery = true } = {}) {
  // Creating a tab is intentionally single-flight. Previously several clicks
  // could issue concurrent POSTs while SQLite/provider work delayed the first
  // response; all of those tabs then appeared in a burst later.
  if (state.newSearchPromise) return state.newSearchPromise;

  const filters = controlsToFilters({ mode: "browse", clearQuery });
  const previousSessionId = String(state.activeSearchSessionId || "");
  const pending = pendingSearchSession(filters);
  const pendingId = String(pending.sessionId);

  // Render the tab before waiting for the backend so New Search always has
  // immediate visible feedback, even if the persistent store is momentarily busy.
  state.searchSessions.push(pending);
  state.activeSearchSessionId = pendingId;
  runtimeFor(pendingId);
  applyFiltersToControls(pending);
  syncKeywordComposerForSession(pendingId);
  resetDetailPane();
  syncActiveItems();
  renderSearchTabs();
  renderResults();
  renderActiveSessionStatus("Creating new search tab…");
  setNewSearchBusy(true);

  const task = (async () => {
    try {
      const previous = sessionById(previousSessionId);
      if (state.searchPreferences.pausePreviousOnNewSearch !== false && previous && ["queued", "running"].includes(String(previous.status || ""))) {
        void pauseSearchSession(previousSessionId, { quiet: true });
      }
      const payload = await api.assetHubCreateSearchSession({
        title: "New search",
        providerId: "civitai",
        mode: "browse",
        filters,
        discoveryCriteria: filters.discoveryCriteria,
        resultFilters: filters.resultFilters,
      });
      const session = payload?.session;
      if (!session) throw new Error("Asset Hub did not create a search session.");

      const index = state.searchSessions.findIndex((item) => String(item.sessionId) === pendingId);
      if (index >= 0) state.searchSessions[index] = session;
      else state.searchSessions.push(session);

      const pendingRuntime = state.sessionRuntime.get(pendingId);
      if (pendingRuntime) state.sessionRuntime.set(String(session.sessionId), pendingRuntime);
      else runtimeFor(session.sessionId);
      state.sessionRuntime.delete(pendingId);

      if (String(state.activeSearchSessionId) === pendingId) {
        state.activeSearchSessionId = String(session.sessionId);
        // Keep the live controls untouched. New Search is editable while the
        // persistent session ID is being created; user-entered query/filter changes
        // are authoritative and must not be overwritten by the original blank draft.
        syncActiveItems();
        renderSearchTabs();
        renderResults();
        renderActiveSessionStatus("New search ready. Choose filters, then Search or Browse.");
        // A large cached candidate pool must not make New Search feel hung.
        // Hydrate in the background after the tab is already usable.
        void loadIndexedForSession(session.sessionId, { quiet: true }).then(() => {
          if (String(state.activeSearchSessionId) !== String(session.sessionId)) return;
          syncActiveItems();
          renderResults();
          renderActiveSessionStatus();
        });
      } else {
        renderSearchTabs();
      }
      return session;
    } catch (error) {
      state.searchSessions = state.searchSessions.filter((item) => String(item.sessionId) !== pendingId);
      state.sessionRuntime.delete(pendingId);
      if (String(state.activeSearchSessionId) === pendingId) {
        const fallback = sessionById(previousSessionId) || state.searchSessions[state.searchSessions.length - 1] || null;
        state.activeSearchSessionId = fallback ? String(fallback.sessionId) : "";
        if (fallback) {
          applyFiltersToControls(fallback);
          syncKeywordComposerForSession(fallback.sessionId);
        }
        syncActiveItems();
        renderResults();
        renderActiveSessionStatus(fallback ? "Unable to create the new tab; returned to the previous search." : "Unable to create a search tab.");
      }
      renderSearchTabs();
      notify(`Unable to create search tab: ${error.message}`, "error");
      throw error;
    } finally {
      state.newSearchPromise = null;
      setNewSearchBusy(false);
    }
  })();

  state.newSearchPromise = task;
  return task;
}

async function restoreSearchSessions() {
  try {
    const payload = await api.assetHubSearchSessions(false, 40);
    const restored = Array.isArray(payload?.sessions) ? payload.sessions : [];
    state.searchSessions = restored.slice().reverse();
    state.searchSessions.forEach((session) => runtimeFor(session.sessionId));
    if (!state.searchSessions.length) {
      await createSearchSession({ clearQuery: true });
      return;
    }
    const newest = state.searchSessions[state.searchSessions.length - 1];
    await activateSearchSession(newest.sessionId);
  } catch (error) {
    notify(`Unable to restore Asset Browser search tabs: ${error.message}`, "warning");
    state.searchSessions = [];
    await createSearchSession({ clearQuery: true });
  }
}

function removeFromSearchQueue(sessionId) {
  const selected = String(sessionId || "");
  state.searchQueue = state.searchQueue.filter((value) => String(value) !== selected);
}

async function pauseSearchSession(sessionId, { quiet = false } = {}) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session) return false;
  removeFromSearchQueue(selected);
  const runtime = runtimeFor(selected);
  // Pause is discovery-only. Already-discovered hero preview work may keep
  // draining, but no new provider pages are requested.
  runtime.pauseRequested = true;
  runtime.stopRequested = false;
  const controller = state.searchControllers.get(selected);
  if (controller) controller.abort();
  session.status = "paused";
  session.partial = true;
  session.nextCursor = runtime.cursor || session.nextCursor || "";
  session.updatedAtUnix = Date.now() / 1000;
  renderSearchTabs();
  syncProviderProgress();
  if (!quiet && String(state.activeSearchSessionId) === selected) {
    const counts = previewCounts(runtime);
    renderActiveSessionStatus(`${counts.discovered} result(s) retained. Provider fetching paused; ${previewCountSummary(counts)}. Preview recovery continues independently. Resume continues from this search's saved cursor.`);
  }
  try {
    const payload = await api.assetHubPauseSearchSession(selected);
    mergeSession(payload?.session || session);
  } catch (error) {
    if (!quiet) notify(`Unable to pause search: ${error.message}`, "error");
    return false;
  }
  renderSearchTabs();
  syncProviderProgress();
  return true;
}

async function resumeSearchSession(sessionId, { quiet = false } = {}) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session) return false;
  const runtime = runtimeFor(selected);
  runtime.pauseRequested = false;
  runtime.stopRequested = false;
  runtime.pagesThisRun = 0;
  runtime.cursor = String(runtime.cursor || session.nextCursor || "");
  session.status = "queued";
  session.partial = true;
  session.updatedAtUnix = Date.now() / 1000;
  renderSearchTabs();
  if (!quiet && String(state.activeSearchSessionId) === selected) {
    const counts = previewCounts(runtime);
    renderActiveSessionStatus(`${counts.discovered} result(s) retained. Provider resume queued; ${previewCountSummary(counts)}.`);
  }
  try {
    const payload = await api.assetHubResumeSearchSession(selected);
    mergeSession(payload?.session || session);
  } catch (error) {
    if (!quiet) notify(`Unable to resume search: ${error.message}`, "error");
    return false;
  }
  if (!state.searchQueue.includes(selected)) state.searchQueue.push(selected);
  renderSearchTabs();
  if (!quiet && String(state.activeSearchSessionId) === selected) {
    const counts = previewCounts(runtime);
    renderActiveSessionStatus(`${counts.discovered} result(s) retained. Provider fetching resumed${state.activeProviderSessionIds.size ? " alongside other active searches" : ""}; ${previewCountSummary(counts)}.`);
  }
  void pumpSearchQueue();
  return true;
}

async function stopSearchSession(sessionId, { quiet = false } = {}) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session) return;
  removeFromSearchQueue(selected);
  const runtime = runtimeFor(selected);
  runtime.stopRequested = true;
  runtime.pauseRequested = false;
  const controller = state.searchControllers.get(selected);
  if (controller) controller.abort();
  try {
    const payload = await api.assetHubStopSearchSession(selected);
    mergeSession(payload?.session || session);
  } catch (error) {
    if (!quiet) notify(`Unable to stop search: ${error.message}`, "error");
  }
  if (String(state.activeSearchSessionId) === selected) {
    const counts = previewCounts(runtime);
    renderActiveSessionStatus(`${counts.discovered} result(s) retained. Provider fetching stopped; ${previewCountSummary(counts)}. Existing preview work continues draining before staged results finish publishing.`);
  }
  renderSearchTabs();
  syncProviderProgress();
}

function activateSearchSessionImmediately(session) {
  if (!session) return;
  const selected = String(session.sessionId || "");
  if (!selected) return;
  state.activeSearchSessionId = selected;
  applyFiltersToControls(session);
  syncKeywordComposerForSession(selected);
  resetDetailPane();
  const runtime = runtimeFor(selected);
  syncActiveItems();
  renderSearchTabs();
  renderResults();
  renderActiveSessionStatus();
  if (runtime.initialized) {
    rebuildMissingPreviewWork(selected);
    pumpPreviewQueue();
    return;
  }
  void loadIndexedForSession(selected, { quiet: true }).then(() => {
    if (String(state.activeSearchSessionId) !== selected) return;
    syncActiveItems();
    renderResults();
    renderActiveSessionStatus();
    pumpPreviewQueue();
  });
}

function closeSearchSession(sessionId) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session) return;

  // Closing is local-first. Abort provider work and remove the tab immediately;
  // persistence cleanup follows asynchronously and must never strand the UI.
  removeFromSearchQueue(selected);
  const runtime = runtimeFor(selected);
  runtime.stopRequested = true;
  runtime.pauseRequested = false;
  const controller = state.searchControllers.get(selected);
  if (controller) controller.abort();

  const closingRuntime = state.sessionRuntime.get(selected);
  if (closingRuntime?.previewFlushTimer) window.clearTimeout(closingRuntime.previewFlushTimer);
  state.previewQueue = state.previewQueue.filter((job) => String(job.sessionId) !== selected);
  state.searchSessions = state.searchSessions.filter((item) => String(item.sessionId) !== selected);
  state.sessionRuntime.delete(selected);

  if (String(state.activeSearchSessionId) === selected) {
    const next = state.searchSessions[state.searchSessions.length - 1] || null;
    if (next) activateSearchSessionImmediately(next);
    else {
      state.activeSearchSessionId = "";
      resetDetailPane();
      syncActiveItems();
      renderSearchTabs();
      renderResults();
      renderActiveSessionStatus("No search tabs open. Creating a new search…");
      void createSearchSession({ clearQuery: true });
    }
  } else {
    renderSearchTabs();
  }

  void api.assetHubCloseSearchSession(selected).catch((error) => {
    notify(`Search tab closed locally, but persistent cleanup failed: ${error.message}`, "warning");
  });
}

// DSV2-02 removed the legacy `filters.libraryFilter === "installed"` provider
// short-circuit (which used to complete "without contacting CivitAI"). Library
// membership is now a local secondary facet and never changes the first-pass
// provider discovery contract.
async function enqueueActiveSearch({ mode = state.mode, refresh = false, continueFromCursor = false } = {}) {
  // If Search/Browse is pressed immediately after New Search, wait for that
  // single pending tab to receive its persistent ID, then queue this request on
  // that tab. Never create a second session just because persistence is slow.
  if (state.newSearchPromise && String(activeSession()?.status || "") === "creating") {
    await state.newSearchPromise;
  }
  let session = activeSession();
  if (!session) session = await createSearchSession({ clearQuery: mode === "browse" });
  const sessionId = String(session.sessionId);
  if (["queued", "running", "stopping"].includes(String(session.status || ""))) {
    await stopSearchSession(sessionId, { quiet: true });
    session = sessionById(sessionId) || session;
  }

  const filters = controlsToFilters({ mode });
  state.mode = filters.mode;
  const runtime = runtimeFor(sessionId);
  runtime.stopRequested = false;
  runtime.pauseRequested = false;
  runtime.forceRefresh = Boolean(refresh);
  runtime.pagesThisRun = 0;
  const resetDiscovery = !continueFromCursor && !refresh;
  if (!continueFromCursor) {
    runtime.cursor = "";
    runtime.providerCount = 0;
    if (resetDiscovery) {
      runtime.cachedCount = 0;
      runtime.candidateCount = 0;
      runtime.matchCount = 0;
      runtime.localNextOffset = null;
      runtime.items = [];
      resetPreviewStaging(sessionId);
      if (String(state.activeSearchSessionId) === sessionId) {
        syncActiveItems();
        renderResults();
      }
    }
  }
  const title = autoSessionTitle(filters, mode);
  session = await patchSearchSession(sessionId, {
    title,
    mode,
    filters,
    discoveryCriteria: filters.discoveryCriteria,
    resultFilters: filters.resultFilters,
    status: "queued",
    partial: Boolean(continueFromCursor),
    nextCursor: continueFromCursor ? (runtime.cursor || session.nextCursor || "") : "",
    providerResultCount: continueFromCursor ? Number(session.providerResultCount || 0) : 0,
    errorMessage: "",
    resetCandidates: resetDiscovery,
  });
  await loadIndexedForSession(sessionId, { quiet: false });

  if (!state.searchQueue.includes(sessionId)) state.searchQueue.push(sessionId);
  renderSearchTabs();
  const queuedCounts = previewCounts(runtime);
  renderActiveSessionStatus(`${queuedCounts.discovered} persistent/indexed result(s) restored immediately. Their previews are resolving in the background; current-provider refresh is queued in the background without blocking the cached cards.`);
  void pumpSearchQueue();
  return true;
}

async function runProviderSession(sessionId) {
  const selected = String(sessionId || "");
  const session = sessionById(selected);
  if (!session || session.closed || state.activeProviderSessionIds.has(selected)) return;
  const runtime = runtimeFor(selected);
  runtime.stopRequested = false;
  runtime.pauseRequested = false;
  runtime.pagesThisRun = 0;
  const controller = new AbortController();
  state.searchControllers.set(selected, controller);
  state.activeProviderSessionIds.add(selected);
  let currentSession = await patchSearchSession(selected, { status: "running", errorMessage: "" });
  let cursor = String(runtime.cursor || currentSession.nextCursor || "");
  let providerCount = Number(currentSession.providerResultCount || runtime.providerCount || 0);
  syncProviderProgress();

  try {
    while (!runtime.stopRequested && !runtime.pauseRequested) {
      const filters = normalizeSessionFilters(currentSession);
      const requestCursor = cursor;
      const payload = await providerSearchWithRetry({
        provider: filters.provider,
        query: filters.query,
        type: filters.type,
        creator: filters.creator,
        sort: filters.sort,
        period: filters.period,
        safeContent: filters.safeContent,
        mode: filters.mode,
        limit: filters.limit,
        cursor: requestCursor,
        sessionId: selected,
        refresh: runtime.forceRefresh && runtime.pagesThisRun === 0,
      }, { signal: controller.signal }, selected, runtime.pagesThisRun + 1);
      const page = payload?.page || {};
      const incoming = Array.isArray(page.items) ? page.items : [];
      // DSV2-02 intentionally replaces the old raw-provider publication calls:
      // runtime.items = mergeItems(runtime.items, incoming);
      // stageSearchResults(selected, incoming);
      // Provider rows first enter the persistent candidate pool, then the local
      // faceted query decides which records are eligible for visible staging.
      runtime.providerCount = providerCount += incoming.length;
      runtime.pagesThisRun += 1;
      cursor = String(page.nextCursor || "");
      if (cursor && cursor === requestCursor) {
        throw new Error("Provider returned the same continuation cursor twice; search paused to avoid an endless fetch loop.");
      }
      runtime.cursor = cursor;
      runtime.initialized = true;
      const done = !cursor;
      runtime.discoveryTerminal = done;
      currentSession = await patchSearchSession(selected, {
        status: done ? "completed" : "running",
        partial: !done,
        nextCursor: cursor,
        touchProvider: true,
        providerResultCount: providerCount,
        errorMessage: "",
      });
      await refreshLocalResults(selected, {
        append: false,
        persistFilters: false,
        quiet: true,
        preserveLoadedCount: Number(runtime.items?.length || 0),
      });
      currentSession = sessionById(selected) || currentSession;
      if (String(state.activeSearchSessionId) === selected) {
        const counts = previewCounts(runtime);
        renderActiveSessionStatus(done
          ? `${runtime.matchCount.toLocaleString()} matching of ${runtime.candidateCount.toLocaleString()} candidates. Provider discovery is complete; ${previewCountSummary(counts)}.`
          : `${runtime.matchCount.toLocaleString()} matching of ${runtime.candidateCount.toLocaleString()} candidates through CivitAI page ${runtime.pagesThisRun}. ${previewCountSummary(counts)} while provider discovery continues…`);
      }
      renderSearchTabs();
      syncProviderProgress();
      schedulePreviewBatchFlush(selected);
      if (done) break;

      if (state.searchPreferences.pagingMode === "manual") {
        runtime.pauseRequested = true;
        currentSession = await patchSearchSession(selected, {
          status: "paused",
          partial: true,
          nextCursor: runtime.cursor,
          errorMessage: "",
        });
        if (String(state.activeSearchSessionId) === selected) {
          renderResults();
          const counts = previewCounts(runtime);
          renderActiveSessionStatus(`${counts.discovered} result(s) retained. Manual paging paused provider discovery; ${previewCountSummary(counts)}. Use Load more or Resume for the next provider page.`);
        }
        renderSearchTabs();
        break;
      }

      await sleep(PROVIDER_PAGE_SPACING_MS);
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      const latestStatus = String(sessionById(selected)?.status || "");
      // Resume can be clicked while an aborted provider request is still unwinding.
      // Do not let the old run overwrite the newer queued state.
      if (latestStatus === "queued") {
        // The finally block will release this run and pump the queued resume.
      } else if (runtime.pauseRequested) {
        await patchSearchSession(selected, { status: "paused", partial: true, nextCursor: runtime.cursor });
      } else if (!runtime.stopRequested) {
        await patchSearchSession(selected, { status: "stopped", partial: true, nextCursor: runtime.cursor });
      }
    } else if (isTransientProviderError(error)) {
      await patchSearchSession(selected, {
        status: "paused",
        partial: true,
        nextCursor: runtime.cursor,
        errorMessage: error.message,
      });
      if (String(state.activeSearchSessionId) === selected) {
        const counts = previewCounts(runtime);
        renderActiveSessionStatus(`CivitAI is temporarily unavailable. ${counts.discovered} fetched/indexed result(s) were retained; ${previewCountSummary(counts)}. Use Resume to retry provider discovery.`);
      }
      notify("CivitAI is temporarily unavailable. Cached results were kept; resume the search to retry.", "warning");
    } else {
      await patchSearchSession(selected, {
        status: "failed",
        partial: true,
        nextCursor: runtime.cursor,
        errorMessage: error.message,
      });
      if (String(state.activeSearchSessionId) === selected) {
        const counts = previewCounts(runtime);
        renderActiveSessionStatus(`Provider refresh failed: ${error.message}. ${counts.discovered} fetched/indexed result(s) were retained; ${previewCountSummary(counts)}. Existing preview work continues independently.`);
      }
      notify(`Asset Browser provider refresh failed: ${error.message}`, "error");
    }
  } finally {
    state.searchControllers.delete(selected);
    state.activeProviderSessionIds.delete(selected);
    renderSearchTabs();
    syncProviderProgress();
    schedulePreviewBatchFlush(selected);
    pumpPreviewQueue();
    void pumpSearchQueue();
  }
}

async function pumpSearchQueue() {
  const pending = state.searchQueue.splice(0);
  pending.forEach((value) => {
    const sessionId = String(value || "");
    const session = sessionById(sessionId);
    if (!session || session.closed || String(session.status) !== "queued") return;
    if (state.activeProviderSessionIds.has(sessionId)) {
      if (!state.searchQueue.includes(sessionId)) state.searchQueue.push(sessionId);
      return;
    }
    void runProviderSession(sessionId);
  });
  syncProviderProgress();
}

async function continueActiveSearch() {
  const session = activeSession();
  if (!session) return;
  const runtime = activeRuntime();
  runtime.cursor = String(runtime.cursor || session.nextCursor || "");
  await resumeSearchSession(session.sessionId);
}

// Backward-compatible internal name used after an install refresh. Search now
// means "queue the active persistent search session" rather than owning one
// page-global request/controller.
async function runSearch({ refresh = false, mode = state.mode } = {}) {
  return enqueueActiveSearch({ mode, refresh });
}

function galleryPolicyFromControls() {
  return {
    detailFetchMode: String($("#assetGalleryDetailFetchMode")?.value || "current_only"),
    libraryGalleryMode: String($("#assetGalleryLibraryMode")?.value || "hero_only"),
    retentionMode: String($("#assetGalleryRetentionMode")?.value || "days"),
    retentionDays: Number($("#assetGalleryRetentionDays")?.value || 7),
    maxCacheGiB: Number($("#assetGalleryMaxCacheGiB")?.value || 10),
  };
}

function applyGalleryPolicy(settings = state.gallerySettings, cache = null) {
  state.gallerySettings = { ...state.gallerySettings, ...(settings || {}) };
  const assignments = [
    ["assetGalleryDetailFetchMode", state.gallerySettings.detailFetchMode],
    ["assetGalleryLibraryMode", state.gallerySettings.libraryGalleryMode],
    ["assetGalleryRetentionMode", state.gallerySettings.retentionMode],
    ["assetGalleryRetentionDays", state.gallerySettings.retentionDays],
    ["assetGalleryMaxCacheGiB", state.gallerySettings.maxCacheGiB],
  ];
  assignments.forEach(([id, value]) => { const node = $(`#${id}`); if (node) node.value = String(value ?? ""); });
  const summary = $("#assetGalleryCacheSummary");
  if (summary && cache) {
    summary.textContent = `${formatBytes(cache.temporaryBytes || 0)} temporary (${Number(cache.temporaryImages || 0).toLocaleString()} images) · ${formatBytes(cache.libraryBytes || 0)} library (${Number(cache.libraryImages || 0).toLocaleString()} images)`;
  }
}

async function refreshGalleryPolicy() {
  try {
    const payload = await api.assetHubGallerySettings();
    applyGalleryPolicy(payload?.settings || {}, payload?.cache || {});
    state.gallerySettingsLoaded = true;
    return payload;
  } catch (error) {
    const summary = $("#assetGalleryCacheSummary");
    if (summary) summary.textContent = `Gallery policy unavailable: ${error.message}`;
    return null;
  }
}

async function saveGalleryPolicy() {
  try {
    const payload = await api.saveAssetHubGallerySettings(galleryPolicyFromControls());
    applyGalleryPolicy(payload?.settings || {}, payload?.cache || {});
    state.gallerySettingsLoaded = true;
    notify("Gallery and preview storage policy saved.", "success");
  } catch (error) {
    notify(`Unable to save gallery policy: ${error.message}`, "error");
  }
}

async function cleanTemporaryGalleryCache() {
  try {
    const payload = await api.assetHubCleanupGalleryCache();
    applyGalleryPolicy(state.gallerySettings, payload?.cache || {});
    state.galleryImageUrls.clear();
    notify("Temporary provider gallery cache cleared.", "success");
    renderDetailPreviews();
  } catch (error) {
    notify(`Unable to clean temporary gallery cache: ${error.message}`, "error");
  }
}

function selectedVersion() {
  const id = $("#assetBrowserVersion")?.value || "";
  return (state.model?.versions || []).find((item) => String(item.remoteVersionId) === String(id)) || null;
}

function detailGalleryKey(model = state.model, version = selectedVersion()) {
  return `${String(model?.remoteModelId || "")}::${String(version?.remoteVersionId || "")}`;
}

function providerGalleryEntries(model = state.model, version = selectedVersion()) {
  if (!model) return [];
  const previews = (version?.previews || [])
    .filter((item) => String(item?.kind || "image").toLowerCase() === "image" && String(item?.url || "").trim())
    .map((preview) => {
      const url = String(preview.url || "").trim();
      const mature = isMatureProviderPreview(model, url);
      const presentation = mature ? maturePreviewModeForSession() : "show";
      return {
        url,
        providerImageId: String(preview?.providerImageId || ""),
        mature,
        presentation,
      };
    });
  if (previews.length) return previews;

  // A version without its own gallery should not borrow another version's
  // gallery. The model-level hero is safe only when there is a single version.
  if ((model?.versions || []).length <= 1) {
    const fallback = providerPreview(model);
    if (fallback) {
      const mature = isMatureProviderPreview(model, fallback);
      return [{ url: fallback, providerImageId: "", mature, presentation: mature ? maturePreviewModeForSession() : "show" }];
    }
  }
  return [];
}

function normalizedDetailGalleryIndex(entries, model = state.model, version = selectedVersion()) {
  if (!entries.length) return 0;
  const key = detailGalleryKey(model, version);
  const current = Number(state.detailGalleryIndexByVersion.get(key) || 0);
  const normalized = ((current % entries.length) + entries.length) % entries.length;
  state.detailGalleryIndexByVersion.set(key, normalized);
  return normalized;
}

function stepDetailGallery(delta) {
  const version = selectedVersion();
  const entries = providerGalleryEntries(state.model, version);
  if (entries.length <= 1) return;
  const key = detailGalleryKey(state.model, version);
  const current = normalizedDetailGalleryIndex(entries, state.model, version);
  const next = (current + Number(delta || 0) + entries.length) % entries.length;
  state.detailGalleryIndexByVersion.set(key, next);
  renderDetailPreviews();
}

function selectedFile() {
  const version = selectedVersion();
  const id = $("#assetBrowserFile")?.value || "";
  return (version?.files || []).find((item) => String(item.remoteFileId) === String(id)) || null;
}

function populateFiles() {
  const version = selectedVersion();
  const select = $("#assetBrowserFile");
  if (!select) return;
  select.replaceChildren();
  (version?.files || []).forEach((file) => {
    const option = document.createElement("option");
    option.value = file.remoteFileId;
    const library = file.libraryStatus === "installed" ? " · In library" : "";
    option.textContent = `${file.primary ? "Primary · " : ""}${file.fileName || `File ${file.remoteFileId}`} · ${formatBytes(file.sizeBytes)}${library}`;
    select.append(option);
  });
  const primary = (version?.files || []).find((file) => file.primary);
  if (primary) select.value = primary.remoteFileId;
  renderDetailSelection();
}

async function managedDetailGalleryUrl(entry, version) {
  const sourceUrl = String(entry?.url || "").trim();
  if (!sourceUrl) return "";
  const cached = state.galleryImageUrls.get(sourceUrl);
  if (cached) return cached;
  const payload = await api.assetHubFetchGalleryImage({
    providerId: "civitai",
    remoteModelId: String(state.model?.remoteModelId || ""),
    remoteVersionId: String(version?.remoteVersionId || ""),
    providerImageId: String(entry?.providerImageId || ""),
    imageUrl: sourceUrl,
  });
  const managed = String(payload?.image?.imageUrl || "").trim();
  if (managed) state.galleryImageUrls.set(sourceUrl, managed);
  return managed;
}

function prefetchAdjacentLibraryGallery(entries, index, version) {
  if (String(state.gallerySettings.detailFetchMode || "current_only") !== "current_and_adjacent") return;
  if (String(state.model?.libraryStatus || "") !== "installed" || entries.length <= 1) return;
  const positions = [...new Set([(index - 1 + entries.length) % entries.length, (index + 1) % entries.length])];
  positions.forEach((position) => {
    const entry = entries[position];
    if (!entry || entry.presentation === "hide" || state.galleryImageUrls.has(String(entry.url || ""))) return;
    void managedDetailGalleryUrl(entry, version).catch(() => {});
  });
}

async function resolveDetailProviderImage(providerEntry, providerEntries, providerIndex, version, renderSerial) {
  const providerImage = $("#assetBrowserDetailPreview");
  const providerPlaceholder = $("#assetBrowserDetailProviderPreviewPlaceholder");
  if (!providerImage || !providerEntry || providerEntry.presentation === "hide") return;
  let renderUrl = "";
  try {
    renderUrl = await managedDetailGalleryUrl(providerEntry, version);
  } catch {
    // Preserve detail usability if the managed cache is temporarily unavailable.
    // This remains a single user-requested image; no gallery bulk fetch occurs.
    renderUrl = String(providerEntry.url || "");
  }
  if (!renderUrl || renderSerial !== state.detailGalleryRenderSerial) return;
  const currentEntries = providerGalleryEntries(state.model, selectedVersion());
  const currentIndex = normalizedDetailGalleryIndex(currentEntries, state.model, selectedVersion());
  if (String(currentEntries[currentIndex]?.url || "") !== String(providerEntry.url || "")) return;
  let settled = false;
  const reveal = async () => {
    if (settled || renderSerial !== state.detailGalleryRenderSerial || String(providerImage.dataset.requestedUrl || "") !== renderUrl) return;
    try { if (typeof providerImage.decode === "function") await providerImage.decode(); } catch { /* load is sufficient */ }
    if (!providerImage.complete || !providerImage.naturalWidth || String(providerImage.dataset.requestedUrl || "") !== renderUrl) return;
    settled = true;
    providerImage.hidden = false;
    if (providerPlaceholder) providerPlaceholder.hidden = true;
    prefetchAdjacentLibraryGallery(providerEntries, providerIndex, version);
  };
  providerImage.dataset.requestedUrl = renderUrl;
  providerImage.addEventListener("load", () => { void reveal(); }, { once: true });
  providerImage.addEventListener("error", () => {
    if (String(providerImage.dataset.requestedUrl || "") !== renderUrl) return;
    settled = true;
    providerImage.hidden = true;
    if (providerPlaceholder) {
      providerPlaceholder.hidden = false;
      providerPlaceholder.textContent = "Preview unavailable";
    }
  }, { once: true });
  providerImage.src = renderUrl;
  if (providerImage.complete && providerImage.naturalWidth) void reveal();
}

function renderDetailPreviews() {
  const version = selectedVersion();
  const providerEntries = providerGalleryEntries(state.model, version);
  const providerIndex = normalizedDetailGalleryIndex(providerEntries, state.model, version);
  const providerEntry = providerEntries[providerIndex] || null;
  const providerUrl = String(providerEntry?.url || "");
  const localUrl = localPreview(state.model);
  const mode = previewMode();
  const providerFigure = $("#assetBrowserDetailProviderPreviewFigure");
  const localFigure = $("#assetBrowserDetailLocalPreviewFigure");
  const providerImage = $("#assetBrowserDetailPreview");
  const providerPlaceholder = $("#assetBrowserDetailProviderPreviewPlaceholder");
  const providerCounter = $("#assetBrowserDetailProviderPreviewCounter");
  const providerCaption = $("#assetBrowserDetailProviderPreviewLabel");
  const previous = $("#assetBrowserDetailPreviewPrevious");
  const next = $("#assetBrowserDetailPreviewNext");
  const localImage = $("#assetBrowserDetailLocalPreview");
  const localLabel = $("#assetBrowserDetailLocalPreviewLabel");
  const providerHidden = Boolean(providerEntry?.mature && providerEntry?.presentation === "hide");
  const showProvider = Boolean(providerEntry) && mode !== "local";
  const showLocal = Boolean(localUrl) && mode !== "provider";

  if (providerFigure) providerFigure.hidden = !showProvider;
  if (localFigure) localFigure.hidden = !showLocal;
  if (providerCaption && showProvider) {
    providerCaption.textContent = version?.name ? `CivitAI · ${version.name}` : "CivitAI";
  }
  if (providerCounter) {
    providerCounter.hidden = !showProvider || providerEntries.length <= 1;
    providerCounter.textContent = showProvider ? `${providerIndex + 1} / ${providerEntries.length}` : "";
  }
  [previous, next].forEach((button) => {
    if (!button) return;
    button.hidden = !showProvider || providerEntries.length <= 1;
    button.disabled = !showProvider || providerEntries.length <= 1;
  });

  if (providerImage) {
    providerImage.classList.toggle("is-mature-blurred", showProvider && !providerHidden && Boolean(providerEntry?.mature) && providerEntry?.presentation === "blur");
    providerImage.hidden = true;
    providerImage.removeAttribute("src");
  }
  if (providerPlaceholder) {
    providerPlaceholder.hidden = !showProvider;
    providerPlaceholder.classList.toggle("is-policy-hidden", providerHidden);
    providerPlaceholder.textContent = providerHidden ? "Mature preview hidden" : "Loading preview…";
  }

  if (showProvider && providerImage && !providerHidden && providerUrl) {
    // P3A invariant remains: providerImage.src = renderUrl; is performed by
    // resolveDetailProviderImage only after the managed-cache URL is resolved.
    state.detailGalleryRenderSerial += 1;
    const renderSerial = state.detailGalleryRenderSerial;
    void resolveDetailProviderImage(providerEntry, providerEntries, providerIndex, version, renderSerial);
  } else {
    state.detailGalleryRenderSerial += 1;
  }

  if (localImage) {
    if (showLocal) localImage.src = localUrl;
    else localImage.removeAttribute("src");
  }
  if (localLabel && showLocal) localLabel.textContent = localPreviewLabel(state.model);
  const root = $("#assetBrowserDetailPreviews");
  if (root) {
    root.hidden = !showProvider && !showLocal;
    root.classList.toggle("has-multiple-previews", showProvider && showLocal);
  }
}

function downloadIdentityKey(modelId, versionId, fileId, providerId = "civitai") {
  return [String(providerId || "civitai").toLowerCase(), String(modelId || ""), String(versionId || ""), String(fileId || "")].join(":");
}

function normalizeAssetIdentity(value = {}) {
  return {
    providerId: String(value.providerId || "civitai").toLowerCase(),
    remoteModelId: String(value.remoteModelId || ""),
    remoteVersionId: String(value.remoteVersionId || ""),
    remoteFileId: String(value.remoteFileId || ""),
    fileName: String(value.fileName || ""),
    modelName: String(value.modelName || ""),
    creator: String(value.creator || ""),
    savedAt: Number(value.savedAt || Date.now()),
  };
}

function assetIdentityKey(value = {}) {
  const item = normalizeAssetIdentity(value);
  return downloadIdentityKey(item.remoteModelId, item.remoteVersionId, item.remoteFileId, item.providerId);
}

function loadSavedAssets() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(SAVED_ASSETS_STORAGE_KEY) || "[]");
    state.savedAssets = Array.isArray(parsed) ? parsed.map(normalizeAssetIdentity).filter((item) => item.remoteModelId) : [];
  } catch {
    state.savedAssets = [];
  }
}

function persistSavedAssets() {
  try {
    window.localStorage.setItem(SAVED_ASSETS_STORAGE_KEY, JSON.stringify(state.savedAssets.slice(0, 200)));
  } catch (error) {
    notify(`Unable to save Asset Browser bookmarks: ${error.message}`, "warning");
  }
}

function isAssetSaved(identity) {
  const key = assetIdentityKey(identity);
  return state.savedAssets.some((item) => assetIdentityKey(item) === key);
}

function saveAssetForLater(identity) {
  const item = normalizeAssetIdentity(identity);
  if (!item.remoteModelId) return;
  const key = assetIdentityKey(item);
  state.savedAssets = [item, ...state.savedAssets.filter((saved) => assetIdentityKey(saved) !== key)].slice(0, 200);
  persistSavedAssets();
  renderSavedAssets();
  renderDetailSelection();
}

function removeSavedAsset(identity) {
  const key = assetIdentityKey(identity);
  state.savedAssets = state.savedAssets.filter((item) => assetIdentityKey(item) !== key);
  persistSavedAssets();
  renderSavedAssets();
  renderDetailSelection();
}

function selectedAssetIdentity() {
  const version = selectedVersion();
  const file = selectedFile();
  if (!state.model) return null;
  return normalizeAssetIdentity({
    providerId: "civitai",
    remoteModelId: state.model.remoteModelId,
    remoteVersionId: version?.remoteVersionId,
    remoteFileId: file?.remoteFileId,
    fileName: file?.fileName,
    modelName: state.model.name,
    creator: state.model.creator,
  });
}

async function ensureAssetBrowserSession() {
  if (!state.initialized) {
    state.initialized = true;
    await restoreSearchSessions();
  } else if (!activeSession()) {
    await createSearchSession({ clearQuery: true });
  }
}

async function openAssetIdentity(identity) {
  const item = normalizeAssetIdentity(identity);
  if (!item.remoteModelId) return;
  await ensureAssetBrowserSession();
  window.dispatchEvent(new CustomEvent("image-gen-workspace-request", { detail: { workspace: "asset-browser" } }));
  await selectModel(item.remoteModelId, { refresh: false });

  const versionSelect = $("#assetBrowserVersion");
  if (versionSelect && item.remoteVersionId && [...versionSelect.options].some((option) => String(option.value) === item.remoteVersionId)) {
    versionSelect.value = item.remoteVersionId;
    populateFiles();
  }
  const fileSelect = $("#assetBrowserFile");
  if (fileSelect && item.remoteFileId && [...fileSelect.options].some((option) => String(option.value) === item.remoteFileId)) {
    fileSelect.value = item.remoteFileId;
    renderDetailSelection();
  }

  const cardElement = document.querySelector(`.asset-browser-card[data-model-id="${CSS.escape(item.remoteModelId)}"]`);
  if (cardElement) {
    cardElement.scrollIntoView({ behavior: "smooth", block: "center" });
    cardElement.focus({ preventScroll: true });
  } else {
    openDetailSurface();
  }
}

function renderSavedAssets() {
  const root = $("#assetBrowserSavedItems");
  const count = $("#assetBrowserSavedCount");
  if (count) count.textContent = state.savedAssets.length ? `${state.savedAssets.length} saved` : "Empty";
  if (!root) return;
  if (!state.savedAssets.length) {
    root.innerHTML = '<div class="subtle">No assets saved for later.</div>';
    return;
  }
  root.replaceChildren(...state.savedAssets.slice(0, 50).map((item) => {
    const row = document.createElement("div");
    row.className = "asset-saved-item";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "asset-saved-item-open";
    const title = document.createElement("strong");
    title.textContent = item.modelName || item.fileName || `Model ${item.remoteModelId}`;
    const meta = document.createElement("span");
    meta.textContent = [item.creator, item.fileName, `${item.providerId} #${item.remoteModelId}`].filter(Boolean).join(" · ");
    copy.append(title, meta);
    copy.addEventListener("click", () => void openAssetIdentity(item));
    const remove = actionButton("Remove", (event) => {
      event?.stopPropagation?.();
      removeSavedAsset(item);
    });
    row.append(copy, remove);
    return row;
  }));
}

function selectedDownloadKey() {
  const version = selectedVersion();
  const file = selectedFile();
  if (!state.model || !version || !file) return "";
  return downloadIdentityKey(state.model.remoteModelId, version.remoteVersionId, file.remoteFileId);
}

function setSelectedDownloadStatus(key, text) {
  if (String(key || "") !== selectedDownloadKey()) return;
  const status = $("#assetBrowserDownloadStatus");
  if (status) status.textContent = text;
}

function selectedDownloadJob() {
  const version = selectedVersion();
  const file = selectedFile();
  if (!state.model || !version || !file) return null;
  const modelId = String(state.model.remoteModelId || "");
  const versionId = String(version.remoteVersionId || "");
  const fileId = String(file.remoteFileId || "");
  return state.downloadJobs.find((job) => (
    String(job.providerId || "").toLowerCase() === "civitai"
    && String(job.remoteModelId || "") === modelId
    && String(job.remoteVersionId || "") === versionId
    && String(job.remoteFileId || "") === fileId
    && !["completed", "cancelled"].includes(String(job.status || ""))
  )) || null;
}

async function cancelSelectedDownload() {
  const job = selectedDownloadJob();
  if (!job?.jobId || !job.canCancel) return;
  const status = $("#assetBrowserDownloadStatus");
  try {
    if (status) status.textContent = `Cancelling ${job.fileName || "download"}…`;
    await api.assetHubCancelDownload(job.jobId);
    await refreshDownloadManager();
    if (status) status.textContent = "Download cancelled.";
  } catch (error) {
    if (status) status.textContent = `Unable to cancel download: ${error.message}`;
    notify(`Unable to cancel download: ${error.message}`, "error");
  }
}

function renderDetailSelection() {
  const version = selectedVersion();
  const file = selectedFile();
  const metadata = $("#assetBrowserMetadata");
  if (metadata) {
    metadata.replaceChildren();
    const rows = [
      ["Support", supportLabel(version?.supportState || state.model?.supportState)],
      ["Architecture", version?.architecture || file?.architecture || "Unknown"],
      ["Base model", version?.baseModel || file?.baseModel || "Unknown"],
      ["Library", file?.libraryStatus === "installed" ? "Already in library" : "Not in library"],
      ["Version ID", version?.remoteVersionId || "—"],
      ["File ID", file?.remoteFileId || "—"],
      ["File", file?.fileName || "—"],
      ["Size", formatBytes(file?.sizeBytes)],
      ["SHA-256", file?.hashes?.SHA256 || "Provider did not expose SHA-256"],
      ["First seen", state.model?.firstSeenAtUnix ? formatSessionTime(state.model.firstSeenAtUnix) : "Not recorded"],
      ["Provider refreshed", state.model?.lastRefreshedAtUnix ? formatSessionTime(state.model.lastRefreshedAtUnix) : "Not yet recorded"],
    ];
    rows.forEach(([key, value]) => {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      wrap.append(dt, dd);
      metadata.append(wrap);
    });
  }

  const support = $("#assetBrowserDetailSupport");
  if (support) {
    const supportState = version?.supportState || state.model?.supportState || "unknown";
    support.textContent = supportLabel(supportState);
    support.className = `asset-browser-badge ${supportClass(supportState)}`;
    support.title = version?.supportReason || state.model?.supportReason || "Compatibility is based on current provider metadata and IMAGE_GEN capability policy.";
  }

  renderDetailPreviews();
  const source = file?.sourcePageUrl || state.model?.sourcePageUrl || "https://civitai.com";
  const link = $("#assetBrowserOpenCivitai");
  if (link) link.href = source;

  const download = $("#assetBrowserDownload");
  const cancelDownload = $("#assetBrowserCancelDownload");
  const installed = file?.libraryStatus === "installed";
  const activeDownload = selectedDownloadJob();
  const downloadKey = selectedDownloadKey();
  const preparing = Boolean(downloadKey && state.preparingDownloads.has(downloadKey));
  if (download) {
    const resumable = Boolean(activeDownload?.canResume);
    download.hidden = !file || installed || Boolean(activeDownload && !resumable);
    download.disabled = preparing || !file;
    download.textContent = preparing ? "Preparing…" : resumable ? "Resume download" : "Download";
  }
  const saveLater = $("#assetBrowserSaveLater");
  const identity = selectedAssetIdentity();
  if (saveLater) {
    const saved = Boolean(identity && isAssetSaved(identity));
    saveLater.hidden = !state.model;
    saveLater.textContent = saved ? "Saved for later" : "Save for later";
    saveLater.setAttribute("aria-pressed", saved ? "true" : "false");
  }
  if (cancelDownload) {
    cancelDownload.hidden = !activeDownload?.canCancel;
    cancelDownload.disabled = !activeDownload?.canCancel || String(activeDownload?.status || "") === "cancelling";
    cancelDownload.textContent = String(activeDownload?.status || "") === "cancelling" ? "Cancelling…" : "Cancel download";
  }

}

function renderModelDetail(model, providerStatusText = "") {
  if (!model) return;
  state.model = model;
  const detail = $("#assetBrowserDetail");
  if (detail) detail.hidden = false;
  const kind = $("#assetBrowserDetailKind");
  const name = $("#assetBrowserDetailName");
  const creator = $("#assetBrowserDetailCreator");
  const description = $("#assetBrowserDetailDescription");
  const providerStatus = $("#assetBrowserDetailProviderStatus");
  if (kind) kind.textContent = model.providerType || displayKind(model.assetKind);
  if (name) name.textContent = model.name || `Model ${model.remoteModelId}`;
  if (creator) creator.textContent = model.creator ? `by ${model.creator}` : `CivitAI #${model.remoteModelId}`;
  if (description) description.textContent = model.description || "No provider description cached.";
  if (providerStatus) providerStatus.textContent = providerStatusText;
  const versionSelect = $("#assetBrowserVersion");
  if (versionSelect) {
    const priorVersion = versionSelect.value;
    versionSelect.replaceChildren();
    (model.versions || []).forEach((version) => {
      const option = document.createElement("option");
      option.value = version.remoteVersionId;
      option.textContent = `${version.name || `Version ${version.remoteVersionId}`} · ${version.baseModel || version.architecture || "Unknown model"} · ${supportLabel(version.supportState)}`;
      versionSelect.append(option);
    });
    if ((model.versions || []).some((version) => String(version.remoteVersionId) === String(priorVersion))) {
      versionSelect.value = priorVersion;
    }
  }
  populateFiles();
}

async function selectModel(modelId, { refresh = false } = {}) {
  const selectedId = String(modelId || "");
  const sessionId = String(state.activeSearchSessionId || "");
  if (String(state.selectedModelId || "") !== selectedId) state.detailGalleryIndexByVersion.clear();
  state.selectedModelId = selectedId;
  prioritizeModelPreview(sessionId, selectedId);
  openDetailSurface();
  const downloadStatus = $("#assetBrowserDownloadStatus");
  if (downloadStatus) downloadStatus.textContent = "";
  // Selection must not rebuild the result grid. Recreating every card briefly
  // exposes placeholders/backgrounds and makes provider thumbnails visibly flash.
  syncSelectedResultCard();

  if (state.activeDetailController) state.activeDetailController.abort();
  const requestSerial = ++state.detailRequestSerial;
  const controller = new AbortController();
  state.activeDetailController = controller;
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, DETAIL_FETCH_TIMEOUT_MS);

  const cached = (activeRuntime().items || []).find((item) => String(item.remoteModelId) === selectedId) || null;
  const detail = $("#assetBrowserDetail");
  if (cached) {
    renderModelDetail(cached, "Showing indexed details now; refreshing current provider details in the background…");
  } else {
    if (detail) detail.hidden = true;
  }

  try {
    const payload = await api.assetHubModel("civitai", selectedId, refresh, { browser: true, signal: controller.signal });
    if (requestSerial !== state.detailRequestSerial) return;
    if (sessionId !== String(state.activeSearchSessionId || "") || selectedId !== String(state.selectedModelId || "")) return;
    const providerModel = payload?.model || null;
    if (!providerModel) throw new Error("Provider returned no model details.");
    providerModel.lastRefreshedAtUnix = Date.now() / 1000;
    providerModel.firstSeenAtUnix = cached?.firstSeenAtUnix || providerModel.lastRefreshedAtUnix;
    const merged = cached ? mergeItems([cached], [providerModel])[0] : providerModel;
    const runtime = activeRuntime();
    runtime.items = mergeItems(runtime.items || [], [merged]);
    stageSearchResults(sessionId, [merged]);
    syncActiveItems();
    renderModelDetail(merged, `Provider details refreshed ${new Date().toLocaleTimeString()}.`);
    syncSelectedResultCard();
  } catch (error) {
    if (requestSerial !== state.detailRequestSerial) return;
    if (sessionId !== String(state.activeSearchSessionId || "") || selectedId !== String(state.selectedModelId || "")) return;
    if (error?.name === "AbortError") {
      if (cached) {
        const providerStatus = $("#assetBrowserDetailProviderStatus");
        if (providerStatus) providerStatus.textContent = timedOut
          ? "Provider refresh timed out; indexed details remain available."
          : "Provider refresh was superseded; indexed details remain available.";
      } else if (timedOut) {
        if (detail) detail.hidden = true;
        renderActiveSessionStatus("Provider details timed out. Try selecting the asset again.");
      }
      return;
    }
    if (cached) {
      const providerStatus = $("#assetBrowserDetailProviderStatus");
      if (providerStatus) providerStatus.textContent = `Provider refresh failed: ${error.message}. Showing indexed details.`;
    } else {
      if (detail) detail.hidden = true;
      renderActiveSessionStatus(`Unable to load asset: ${error.message}`);
    }
  } finally {
    window.clearTimeout(timeout);
    if (requestSerial === state.detailRequestSerial) state.activeDetailController = null;
  }
}


function downloadStatusLabel(job) {
  const status = String(job?.status || "unknown");
  if (status === "queued") return job.queuePosition ? `Queued #${job.queuePosition}` : "Queued";
  if (status === "resolving") return "Resolving provider file";
  if (status === "downloading") return "Downloading";
  if (status === "verifying") return "Verifying SHA-256";
  if (status === "pausing") return "Pausing";
  if (status === "paused") return "Paused";
  if (status === "cancelling") return "Cancelling";
  if (status === "completed") {
    if (job.install?.status === "installed") return "Installed";
    if (job.install?.status === "quarantined") return "Quarantined for review";
    return "Verified · finalizing library";
  }
  if (status === "failed") return "Failed";
  if (status === "cancelled") return "Cancelled";
  return status;
}

function actionButton(label, handler, { disabled = false } = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    handler(event);
  });
  return button;
}

function downloadMatchesFilter(job) {
  const statusFilter = $("#assetDownloadFilterStatus")?.value || "all";
  const textFilter = String($("#assetDownloadFilterText")?.value || "").trim().toLowerCase();
  const status = String(job?.status || "");
  if (statusFilter === "active" && !DOWNLOAD_ACTIVE_STATUSES.has(status)) return false;
  if (!["all", "active"].includes(statusFilter) && status !== statusFilter) return false;
  if (!textFilter) return true;
  const haystack = [job.fileName, job.providerId, job.remoteModelId, job.remoteVersionId, job.remoteFileId, job.install?.installedPath]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(textFilter);
}

function clearHistoryStatusForFilter() {
  const statusFilter = $("#assetDownloadFilterStatus")?.value || "all";
  if (statusFilter === "all") return "inactive";
  if (["completed", "failed", "cancelled"].includes(statusFilter)) return statusFilter;
  return "";
}

async function runBulkDownloadAction(action) {
  const status = $("#assetDownloadMaintenanceStatus");
  try {
    const payload = await api.assetHubBulkDownloadAction(action);
    if (status) status.textContent = `${payload?.affected || 0} download(s) updated; ${payload?.skipped || 0} unchanged.`;
    await refreshDownloadManager();
  } catch (error) {
    if (status) status.textContent = `Bulk action failed: ${error.message}`;
    notify(`Download bulk action failed: ${error.message}`, "error");
  }
}

async function clearFilteredDownloadHistory() {
  const status = $("#assetDownloadMaintenanceStatus");
  const filter = clearHistoryStatusForFilter();
  if (!filter) {
    if (status) status.textContent = "Active, queued, and paused downloads cannot be cleared from history.";
    return;
  }
  try {
    const payload = await api.assetHubClearDownloadHistory(filter);
    if (status) status.textContent = `${payload?.removed || 0} history row(s) cleared; ${payload?.skippedRecoverable || 0} recoverable item(s) preserved.`;
    await refreshDownloadManager();
  } catch (error) {
    if (status) status.textContent = `Unable to clear download history: ${error.message}`;
    notify(`Unable to clear download history: ${error.message}`, "error");
  }
}

async function cleanOldDownloadPartials() {
  const status = $("#assetDownloadMaintenanceStatus");
  try {
    const payload = await api.assetHubCleanupStaleDownloads({ maxAgeHours: 24, includeRecentUnrecoverable: true });
    if (status) status.textContent = `Cleaned ${payload?.removedFiles || 0} partial file(s) and ${payload?.removedOrphanDirectories || 0} orphan folder(s), freeing ${formatBytes(payload?.removedBytes || 0)}. ${payload?.preservedResumable || 0} resumable item(s) preserved.`;
    await refreshDownloadManager();
  } catch (error) {
    if (status) status.textContent = `Partial cleanup failed: ${error.message}`;
    notify(`Unable to clean old download partials: ${error.message}`, "error");
  }
}

async function waitForLibraryFinalization(jobId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const payload = await api.assetHubDownloadJob(jobId);
    const job = payload?.job || {};
    if (job.install) return job.install;
    if (String(job.status || "") !== "completed") return null;
    if (String(job.resumeNote || "").includes("Automatic library finalization is pending")) return null;
    await sleep(750);
  }
  return null;
}

function captureSearchScrollAnchor() {
  const workspace = $("#assetBrowserWorkspace");
  const results = $("#assetBrowserResults");
  if (!workspace || !results || workspace.hidden) return null;
  const workspaceRect = workspace.getBoundingClientRect();
  const cards = [...results.querySelectorAll(".asset-browser-card")];
  const anchor = cards.find((card) => card.getBoundingClientRect().bottom >= workspaceRect.top + 4) || null;
  return {
    workspace,
    scrollTop: workspace.scrollTop,
    modelId: String(anchor?.dataset?.modelId || ""),
    viewportOffset: anchor ? anchor.getBoundingClientRect().top - workspaceRect.top : null,
  };
}

function restoreSearchScrollAnchor(anchor) {
  if (!anchor?.workspace?.isConnected) return;
  window.requestAnimationFrame(() => {
    const workspace = anchor.workspace;
    if (anchor.modelId) {
      const escaped = globalThis.CSS?.escape ? globalThis.CSS.escape(anchor.modelId) : anchor.modelId.replace(/["\\]/g, "\$&");
      const card = $("#assetBrowserResults")?.querySelector?.(`.asset-browser-card[data-model-id="${escaped}"]`);
      if (card && anchor.viewportOffset !== null) {
        const currentOffset = card.getBoundingClientRect().top - workspace.getBoundingClientRect().top;
        workspace.scrollTop += currentOffset - anchor.viewportOffset;
        return;
      }
    }
    workspace.scrollTop = anchor.scrollTop;
  });
}

function renderDownloadManager(payload) {
  const scrollAnchor = captureSearchScrollAnchor();
  const allJobs = Array.isArray(payload?.jobs) ? payload.jobs : [];
  // Keep terminal jobs as filterable history. The filter is presentation-only;
  // persistent recoverability remains owned by the backend download manager.
  const jobs = allJobs.filter(downloadMatchesFilter);
  state.downloadJobs = allJobs;
  const root = $("#assetDownloadJobs");
  const summary = $("#assetDownloadSummary");
  const settings = payload?.settings || {};
  if (!state.downloadSettingsLoaded) {
    const maxActive = $("#assetDownloadMaxActive");
    const maxQueued = $("#assetDownloadMaxQueued");
    const bandwidth = $("#assetDownloadBandwidth");
    const spacing = $("#assetDownloadRequestSpacing");
    const retries = $("#assetDownloadRetries");
    if (maxActive) maxActive.value = String(settings.maxActiveDownloads ?? 2);
    if (maxQueued) maxQueued.value = String(settings.maxQueuedDownloads ?? 64);
    if (bandwidth) bandwidth.value = String(settings.bandwidthLimitMiBPerSecond ?? 0);
    if (spacing) spacing.value = String(settings.providerMinRequestIntervalSeconds ?? 0.25);
    if (retries) retries.value = String(settings.retryAttempts ?? 3);
    state.downloadSettingsLoaded = true;
  }

  const active = allJobs.filter((job) => ["resolving", "downloading", "verifying", "pausing", "cancelling"].includes(String(job.status))).length;
  const queued = allJobs.filter((job) => String(job.status) === "queued").length;
  const paused = allJobs.filter((job) => String(job.status) === "paused").length;
  if (summary) summary.textContent = active || queued || paused ? `${active} active · ${queued} queued · ${paused} paused` : "Idle";
  const clearButton = $("#assetDownloadClearFiltered");
  if (clearButton) clearButton.disabled = !clearHistoryStatusForFilter();
  if (!root) return;
  if (!jobs.length) {
    root.innerHTML = allJobs.length
      ? '<div class="subtle">No downloads match the current filter.</div>'
      : '<div class="subtle">No download jobs yet.</div>';
    renderDetailSelection();
    restoreSearchScrollAnchor(scrollAnchor);
    return;
  }

  root.replaceChildren(...jobs.slice(0, 100).map((job) => {
    const row = document.createElement("div");
    row.className = "asset-download-job";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.title = "Open this asset in Asset Browser";
    const identity = normalizeAssetIdentity(job);
    row.addEventListener("click", () => void openAssetIdentity(identity));
    row.addEventListener("keydown", (event) => {
      if (event.target !== row) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void openAssetIdentity(identity);
      }
    });
    const copy = document.createElement("div");
    copy.className = "asset-download-job-copy";
    const name = document.createElement("div");
    name.className = "asset-download-job-name";
    name.textContent = job.fileName || `Download ${job.jobId}`;
    const meta = document.createElement("div");
    meta.className = "asset-download-job-meta";
    meta.textContent = `${job.providerId || "provider"} · ${downloadStatusLabel(job)}`;
    copy.append(name, meta);

    const progressWrap = document.createElement("div");
    progressWrap.className = "asset-download-job-progress";
    const progress = document.createElement("progress");
    progress.max = 1;
    if (typeof job.progress === "number") progress.value = Math.max(0, Math.min(1, job.progress));
    else progress.removeAttribute("value");
    const progressText = document.createElement("div");
    progressText.className = "asset-download-job-progress-text";
    const byteText = document.createElement("span");
    byteText.textContent = job.expectedBytes ? `${formatBytes(job.receivedBytes)} / ${formatBytes(job.expectedBytes)}` : formatBytes(job.receivedBytes);
    const speedText = document.createElement("span");
    speedText.textContent = formatSpeed(job.bytesPerSecond);
    progressText.append(byteText, speedText);
    progressWrap.append(progress, progressText);

    const actions = document.createElement("div");
    actions.className = "asset-download-job-actions";
    if (job.canPause) actions.append(actionButton("Pause", async () => { await api.assetHubPauseDownload(job.jobId); await refreshDownloadManager(); }));
    if (job.canResume) actions.append(actionButton("Resume", async () => { await api.assetHubResumeDownload(job.jobId); await refreshDownloadManager(); }));
    if (job.canCancel) actions.append(actionButton("Cancel", async () => { await api.assetHubCancelDownload(job.jobId); await refreshDownloadManager(); }));
    actions.append(actionButton(isAssetSaved(identity) ? "Saved" : "Save", () => {
      if (isAssetSaved(identity)) removeSavedAsset(identity);
      else saveAssetForLater(identity);
      refreshDownloadManager();
    }));
    if (job.install?.status === "installed" && job.install?.installId) {
      actions.append(actionButton("Open folder", async () => {
        try { await api.assetHubOpenInstallFolder(job.install.installId); }
        catch (error) { notify(`Unable to open install folder: ${error.message}`, "error"); }
      }));
    }

    row.append(copy, progressWrap, actions);
    const path = job.install?.installedPath;
    if (path) {
      const pathRow = document.createElement("div");
      pathRow.className = "asset-download-job-path";
      pathRow.textContent = `Installed to: ${path}`;
      row.append(pathRow);
    } else if (job.error?.message || job.resumeNote) {
      const pathRow = document.createElement("div");
      pathRow.className = "asset-download-job-path";
      pathRow.textContent = [job.error?.message, job.resumeNote].filter(Boolean).join(" · ");
      row.append(pathRow);
    }
    return row;
  }));
  renderDetailSelection();
  restoreSearchScrollAnchor(scrollAnchor);
}

async function refreshDownloadManager() {
  try {
    const payload = await api.assetHubDownloadJobs(100);
    renderDownloadManager(payload);
    return payload;
  } catch (error) {
    const summary = $("#assetDownloadSummary");
    if (summary) summary.textContent = `Unavailable: ${error.message}`;
    return null;
  }
}

function scheduleDownloadPolling() {
  if (state.downloadPollTimer) window.clearInterval(state.downloadPollTimer);
  state.downloadPollTimer = window.setInterval(() => refreshDownloadManager(), 900);
}

async function saveDownloadSettings() {
  const values = {
    maxActiveDownloads: Number($("#assetDownloadMaxActive")?.value || 2),
    maxQueuedDownloads: Number($("#assetDownloadMaxQueued")?.value || 64),
    bandwidthLimitMiBPerSecond: Number($("#assetDownloadBandwidth")?.value || 0),
    providerMinRequestIntervalSeconds: Number($("#assetDownloadRequestSpacing")?.value || 0.25),
    retryAttempts: Number($("#assetDownloadRetries")?.value || 3),
  };
  try {
    const payload = await api.saveAssetHubDownloadSettings(values);
    state.downloadSettingsLoaded = false;
    renderDownloadManager({ jobs: (await api.assetHubDownloadJobs(100))?.jobs || [], settings: payload?.settings || {} });
    notify("Downloader settings applied and saved to user-config.yml.", "success");
  } catch (error) {
    notify(`Unable to save downloader settings: ${error.message}`, "error");
  }
}

async function waitForDownload(jobId) {
  const status = $("#assetBrowserDownloadStatus");
  const deadline = Date.now() + (2 * 60 * 60 * 1000);
  while (Date.now() < deadline) {
    const payload = await api.assetHubDownloadJob(jobId);
    const job = payload?.job || {};
    const current = String(job.status || "unknown");
    if (status) {
      const progress = job.expectedBytes
        ? `${formatBytes(job.receivedBytes)} / ${formatBytes(job.expectedBytes)}`
        : formatBytes(job.receivedBytes);
      status.textContent = `Download: ${current} · ${progress}`;
    }
    if (current === "completed") return job;
    if (current === "paused") {
      if (status) status.textContent = "Download paused. Resume it from the Downloads panel to continue.";
      await sleep(900);
      continue;
    }
    if (current === "cancelled") return job;
    if (current === "failed") {
      throw new Error(job.error?.message || "Download failed.");
    }
    await sleep(650);
  }
  throw new Error("Download did not finish before the browser wait timeout.");
}

async function maybeCacheInstalledGallery(modelId, versionId) {
  const mode = String(state.gallerySettings.libraryGalleryMode || "hero_only");
  if (mode === "hero_only" || !modelId) return;
  try {
    const payload = await api.assetHubCacheLibraryGallery({
      providerId: "civitai",
      remoteModelId: String(modelId || ""),
      remoteVersionId: String(versionId || ""),
    });
    applyGalleryPolicy(state.gallerySettings, payload?.cache || null);
    if (Number(payload?.cached || 0) > 0) notify(`${Number(payload.cached).toLocaleString()} gallery image(s) cached for the installed asset.`, "success");
  } catch (error) {
    notify(`Asset installed, but its optional gallery could not be cached: ${error.message}`, "warning");
  }
}

async function completeDownloadLifecycle(jobId, fileName, selectedModelId, selectedVersionId = "") {
  const status = $("#assetBrowserDownloadStatus");
  try {
    const completedDownload = await waitForDownload(jobId);
    if (String(completedDownload?.status || "") === "cancelled") {
      if (status) status.textContent = "Download cancelled.";
      await refreshDownloadManager();
      return;
    }
    if (status) status.textContent = "Download verified. Finalizing into the IMAGE_GEN library…";
    const installedJob = await waitForLibraryFinalization(jobId);
    if (installedJob?.status === "installed") {
      if (status) status.textContent = `Installed in library: ${installedJob.installedPath || fileName}`;
      notify(`${fileName} downloaded and added to the configured IMAGE_GEN library.`, "success");
    } else if (installedJob?.status === "quarantined") {
      if (status) status.textContent = "Download completed, but IMAGE_GEN placed the asset in quarantine because it could not be safely routed into a live library folder.";
      notify(`${fileName} downloaded but requires review before library use.`, "warning");
    } else if (status) {
      status.textContent = "Download verified. Automatic library finalization is pending and will retry after restart if necessary.";
    }

    await refreshDownloadManager();
    if (installedJob) {
      // B1 has already synchronized the persistent catalog/discovery authorities.
      // B2 consumes that local state directly instead of refreshing the provider or
      // rebuilding the complete search grid after every successful install.
      if (installedJob.status === "installed") dispatchInstalledAsset(installedJob);
      const installScrollAnchor = captureSearchScrollAnchor();
      try {
        await syncInstalledAssetFromLocalIndex(installedJob, selectedModelId);
        if (installedJob.status === "installed") void maybeCacheInstalledGallery(selectedModelId, selectedVersionId || installedJob.remoteVersionId || "");
      } catch (refreshError) {
        if (status && installedJob?.status === "installed") {
          status.textContent = `Installed in library: ${installedJob.installedPath || fileName}. The open browser view can be refreshed if needed.`;
        }
        notify(`Asset installed, but the open Asset Browser view could not synchronize: ${refreshError.message}`, "warning");
      } finally {
        restoreSearchScrollAnchor(installScrollAnchor);
      }
    }
  } catch (error) {
    if (status) status.textContent = `Download failed: ${error.message}`;
    notify(`Asset download failed: ${error.message}`, "error");
    await refreshDownloadManager();
  }
}

async function downloadOrResumeSelected() {
  const existing = selectedDownloadJob();
  if (existing?.canResume && existing?.jobId) {
    const status = $("#assetBrowserDownloadStatus");
    try {
      if (status) status.textContent = `Resuming ${existing.fileName || "download"} from its preserved partial file…`;
      await api.assetHubResumeDownload(existing.jobId);
      state.sessionDownloadJobIds.add(String(existing.jobId));
      await refreshDownloadManager();
      void completeDownloadLifecycle(existing.jobId, existing.fileName || "asset", existing.remoteModelId, existing.remoteVersionId || "");
      return;
    } catch (error) {
      if (status) status.textContent = `Unable to resume download: ${error.message}`;
      notify(`Unable to resume download: ${error.message}`, "error");
      return;
    }
  }
  await downloadSelected();
}

function toggleSelectedSavedAsset() {
  const identity = selectedAssetIdentity();
  if (!identity) return;
  if (isAssetSaved(identity)) removeSavedAsset(identity);
  else saveAssetForLater(identity);
}

async function downloadSelected() {
  const version = selectedVersion();
  const file = selectedFile();
  const model = state.model;
  if (!model || !version || !file || file.libraryStatus === "installed") return;

  // Capture immutable provider identity before any await. Selecting another card
  // while this request is preparing must not redirect its UI state or queue the
  // newly selected asset by accident.
  const modelId = String(model.remoteModelId || "");
  const versionId = String(version.remoteVersionId || "");
  const fileId = String(file.remoteFileId || "");
  const fileName = String(file.fileName || `asset-${fileId}.bin`);
  const downloadKey = downloadIdentityKey(modelId, versionId, fileId);
  if (state.preparingDownloads.has(downloadKey)) return;

  state.preparingDownloads.add(downloadKey);
  renderDetailSelection();
  setSelectedDownloadStatus(downloadKey, "Preparing provider download…");
  try {
    const planPayload = await api.assetHubCreateDownloadPlan({
      providerId: "civitai",
      remoteModelId: modelId,
      remoteVersionId: versionId,
      remoteFileId: fileId,
      fileName,
      expectedBytes: Number(file.sizeBytes || 0),
      expectedSha256: String(file.hashes?.SHA256 || ""),
    });
    const downloadPlan = planPayload?.plan;
    if (!downloadPlan?.planId) throw new Error("Asset Hub did not return a download plan.");

    const jobPayload = await api.assetHubCreateDownloadJob(downloadPlan.planId);
    const jobId = jobPayload?.job?.jobId;
    if (!jobId) throw new Error("Asset Hub did not create a download job.");
    state.sessionDownloadJobIds.add(String(jobId));
    const manager = $("#assetDownloadManager");
    if (manager) setComponentShellState(manager, "expanded");
    setSelectedDownloadStatus(downloadKey, `Queued ${fileName}. Track it in Downloads.`);
    await refreshDownloadManager();
    void completeDownloadLifecycle(jobId, fileName, modelId, versionId);
  } catch (error) {
    setSelectedDownloadStatus(downloadKey, `Download failed: ${error.message}`);
    notify(`Asset download failed: ${error.message}`, "error");
    await refreshDownloadManager();
  } finally {
    state.preparingDownloads.delete(downloadKey);
    renderDetailSelection();
  }
}

export function bindAssetBrowser() {
  const workspace = $("#assetBrowserWorkspace");
  if (!workspace) return null;
  const componentWorkspace = bindAssetBrowserComponents(workspace);
  state.detailOverlay = componentWorkspace?.detailsOverlay || null;
  bindDetailSurfaceInteractions();
  loadSearchPreferences();
  $("#assetBrowserSearchForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    void enqueueActiveSearch({ mode: "search" });
  });
  $("#assetBrowserBrowseButton")?.addEventListener("click", () => void enqueueActiveSearch({ mode: "browse" }));
  $("#assetBrowserRefreshButton")?.addEventListener("click", () => void enqueueActiveSearch({ refresh: true, mode: state.mode }));
  $("#assetBrowserNewSearch")?.addEventListener("click", () => void createSearchSession({ clearQuery: true }));
  $("#assetBrowserStopSearch")?.addEventListener("click", () => {
    const session = activeSession();
    const button = $("#assetBrowserStopSearch");
    if (!session || !button) return;
    if (button.dataset.action === "resume") void resumeSearchSession(session.sessionId);
    else void pauseSearchSession(session.sessionId);
  });
  $("#assetBrowserPausePreviousSearch")?.addEventListener("change", saveSearchPreferences);
  $("#assetBrowserPagingMode")?.addEventListener("change", saveSearchPreferences);
  $("#assetBrowserLoadMore")?.addEventListener("click", () => void continueActiveSearch());
  $("#assetBrowserLoadMoreLocal")?.addEventListener("click", () => {
    const session = activeSession();
    if (session) void refreshLocalResults(session.sessionId, { append: true, persistFilters: false });
  });
  $("#assetBrowserVersion")?.addEventListener("change", populateFiles);
  $("#assetBrowserDetailPreviewPrevious")?.addEventListener("click", () => stepDetailGallery(-1));
  $("#assetBrowserDetailPreviewNext")?.addEventListener("click", () => stepDetailGallery(1));
  $("#assetBrowserFile")?.addEventListener("change", renderDetailSelection);
  ["assetBrowserKeywordMode", "assetBrowserRatingBasis", "assetBrowserLocalSort", "assetBrowserSupport", "assetBrowserLibrary", "assetBrowserPreviewMode"].forEach((id) => {
    $(`#${id}`)?.addEventListener("change", () => {
      scheduleLocalFilterRefresh({ immediate: true });
      if (id === "assetBrowserPreviewMode") renderDetailPreviews();
    });
  });
  $("#assetBrowserFilterKeywords")?.addEventListener("input", () => scheduleKeywordCommit({ immediate: false }));
  $("#assetBrowserFilterKeywords")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      scheduleKeywordCommit({ immediate: true });
    }
  });
  $("#assetBrowserFilterCreator")?.addEventListener("input", () => scheduleLocalFilterRefresh({ immediate: false }));
  $("#assetBrowserMaturePreviewMode")?.querySelectorAll("button[data-value]").forEach((button) => {
    button.addEventListener("click", () => {
      setMaturePreviewMode(button.dataset.value || "show");
      scheduleLocalFilterRefresh({ immediate: true });
      renderDetailPreviews();
    });
  });
  $("#assetBrowserClearFilters")?.addEventListener("click", clearLocalFilterControls);
  $("#assetBrowserDownload")?.addEventListener("click", downloadOrResumeSelected);
  $("#assetBrowserCancelDownload")?.addEventListener("click", cancelSelectedDownload);
  $("#assetBrowserSaveLater")?.addEventListener("click", toggleSelectedSavedAsset);
  $("#assetDownloadFilterStatus")?.addEventListener("change", () => renderDownloadManager({ jobs: state.downloadJobs, settings: {} }));
  $("#assetDownloadFilterText")?.addEventListener("input", () => renderDownloadManager({ jobs: state.downloadJobs, settings: {} }));
  $("#assetDownloadPauseAll")?.addEventListener("click", () => void runBulkDownloadAction("pause"));
  $("#assetDownloadResumeAll")?.addEventListener("click", () => void runBulkDownloadAction("resume"));
  $("#assetDownloadCancelAll")?.addEventListener("click", () => void runBulkDownloadAction("cancel"));
  $("#assetDownloadClearFiltered")?.addEventListener("click", () => void clearFilteredDownloadHistory());
  $("#assetDownloadCleanOldPartials")?.addEventListener("click", () => void cleanOldDownloadPartials());
  $("#assetDownloadSaveSettings")?.addEventListener("click", saveDownloadSettings);
  $("#assetGallerySaveSettings")?.addEventListener("click", () => void saveGalleryPolicy());
  $("#assetGalleryCleanTemporary")?.addEventListener("click", () => void cleanTemporaryGalleryCache());
  let localScrollFrame = 0;
  workspace.addEventListener("scroll", () => {
    if (localScrollFrame) return;
    localScrollFrame = window.requestAnimationFrame(() => {
      localScrollFrame = 0;
      void maybeAutoLoadLocalResults();
    });
  }, { passive: true });
  loadSavedAssets();
  renderSavedAssets();
  refreshDownloadManager();
  void refreshGalleryPolicy();
  scheduleDownloadPolling();
  return {
    show: () => {
      if (!state.initialized) {
        state.initialized = true;
        // Restore tabs and cached results only. Interrupted queued/running sessions
        // are recovered as paused by the backend on startup and never auto-resume.
        void restoreSearchSessions();
      }
      componentWorkspace?.responsive?.refresh?.();
      state.detailOverlay?.refresh?.();
      refreshDownloadManager();
      void refreshGalleryPolicy();
      window.setTimeout(() => { void maybeAutoLoadLocalResults(); }, 0);
    },
    hide: () => {},
    refresh: () => enqueueActiveSearch({ refresh: true, mode: state.mode }),
    componentWorkspace,
  };
}
