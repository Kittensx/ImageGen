import { api } from "../api.js?v=asset-card-latency1";
import { createProgressiveAssetGrid } from "../components/progressive-asset-grid.js?v=asset-grid-qol1";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

function assetLabel(asset = {}) {
  const nickname = String(asset?.nickname || "").trim();
  if (nickname) return nickname;
  const embeddedName = String(asset?.embedded_name || "").trim();
  if (embeddedName) return embeddedName;
  const canonicalName = String(asset?.name || "").trim();
  if (canonicalName) return canonicalName;
  const displayName = String(asset?.display_name || "").trim();
  if (displayName) return displayName;
  const filename = String(asset?.filename || "").trim();
  if (filename) return filename.replace(/\.[^.]+$/, "");
  return "Unnamed asset";
}

function normalizedPath(value) {
  return String(value || "").trim().replaceAll("/", "\\").toLowerCase();
}

function formatSize(bytes, sizeMb = 0) {
  const value = Number(bytes || 0) || (Number(sizeMb || 0) * 1024 * 1024);
  if (!value) return "Unknown";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  return `${(value / 1024 ** 2).toFixed(0)} MB`;
}


function residencyLabel(runtime = {}) {
  const effective = String(runtime.residency_state_effective || "empty").trim().toLowerCase();
  const stage = String(runtime.stage || "").trim().toLowerCase();
  if (["preparing_model", "loading_tokenizer", "loading_checkpoint", "applying_retention_policy"].includes(stage)) return "LOADING";
  if (["unloading", "superseded"].includes(stage) || effective === "switching") return "SWITCHING";
  if (stage === "recovering" || effective === "recovering") return "RECOVERING";
  if (effective === "hot_gpu") return "HOT - GPU";
  if (effective === "hot_staged") return "HOT - STAGED";
  if (effective === "managed_resident") return "MANAGED - RESIDENT";
  return "UNLOADED";
}

function formatGiB(value) {
  const bytes = Number(value || 0);
  return Number.isFinite(bytes) && bytes > 0 ? `${(bytes / (1024 ** 3)).toFixed(2)} GiB` : "0 GiB";
}
function familyLabel(value) {
  const token = String(value || "").trim().toLowerCase();
  if (token.includes("sd1") || token.includes("1.x") || token.includes("1.5")) return "SD 1.x";
  if (token.includes("sd2") || token.includes("2.x") || token.includes("2.1")) return "SD 2.x";
  if (token.includes("xl")) return "SDXL";
  return value || "Unknown";
}

function tagsFor(model) {
  const values = [model.model_family || model.architecture, ...(model.tags || [])]
    .map((value) => String(value || "").trim())
    .filter(Boolean);
  return [...new Map(values.map((value) => [value.casefold?.() || value.toLowerCase(), value])).values()];
}

function addTag(container, text, className = "") {
  if (!container || !text) return;
  const tag = document.createElement("span");
  tag.className = `checkpoint-tag ${className}`.trim();
  tag.textContent = text;
  container.append(tag);
}

function civitaiLookupFor(model = {}) {
  const metadata = model?.metadata || {};
  const lookup = model?.civitai_lookup || metadata?._civitai_lookup || {};
  return lookup && typeof lookup === "object" ? lookup : {};
}

function sourceUrlFor(model = {}) {
  const metadata = model?.metadata || {};
  const lookup = civitaiLookupFor(model);
  return String(metadata.source_url || model.source_url || lookup.source_url || "").trim();
}



async function chooseRecentPreview({ assetId, title, loadCandidates, replaceFromOutput, onApplied }) {
  const rows = await loadCandidates(assetId, 48);
  if (!Array.isArray(rows) || rows.length === 0) {
    notify("No recent outputs are available.", "warning");
    return;
  }
  const options = rows.map((item, index) => {
    const reasons = Array.isArray(item.match_reasons) && item.match_reasons.length ? ` — ${item.match_reasons.join(", ")}` : "";
    return `${index + 1}. ${item.name || item.relative_name || item.output_id}${reasons}`;
  }).join("\n");
  const answer = window.prompt(`${title}\n\nChoose a recent output number:\n\n${options}`);
  if (answer == null) return;
  const position = Number.parseInt(answer, 10);
  if (!Number.isFinite(position) || position < 1 || position > rows.length) {
    notify("Enter one of the listed output numbers.", "warning");
    return;
  }
  const chosen = rows[position - 1];
  const updated = await replaceFromOutput(assetId, chosen.output_id);
  onApplied(updated);
}
function buildPreviewUrl(model = {}) {
  const baseUrl = String(model?.preview_url || "").trim();
  if (!baseUrl) return "";
  const token = String(
    model?.preview_revision
    || model?.preview_modified_ns
    || model?.catalog_revision
    || model?.modified_ns
    || Date.now()
  ).trim();
  if (!token) return baseUrl;
  const sanitized = baseUrl.replace(/([?&])igcb=[^&]*(&|$)/, (_match, prefix, suffix) => (prefix === "?" && suffix ? "?" : prefix === "&" && suffix ? "&" : ""));
  return `${sanitized}${sanitized.includes("?") ? "&" : "?"}igcb=${encodeURIComponent(token)}`;
}

function setPreview(image, fallback, model) {
  if (!image || !fallback) return;
  const url = buildPreviewUrl(model);
  const requestToken = `${model?.asset_id || "detail"}|${url}`;
  image.dataset.previewRequest = requestToken;
  image.classList.remove("has-image");
  image.removeAttribute("src");
  fallback.classList.remove("is-hidden");
  fallback.textContent = String(model?.name || "CKPT").slice(0, 4).toUpperCase();
  if (!url) return;

  // The visible <img> is intentionally hidden until the preview is ready. Native
  // lazy loading can defer a display:none image forever, so use a detached eager
  // probe to prove the file can load, then reveal the cached image atomically.
  const probe = new Image();
  probe.decoding = "async";
  probe.onload = () => {
    if (image.dataset.previewRequest !== requestToken) return;
    image.src = url;
    image.classList.add("has-image");
    fallback.classList.add("is-hidden");
  };
  probe.onerror = () => {
    if (image.dataset.previewRequest !== requestToken) return;
    image.classList.remove("has-image");
    image.removeAttribute("src");
    fallback.classList.remove("is-hidden");
  };
  probe.src = url;
}

function modelMatchesFilter(model, filter) {
  if (filter === "all") return true;
  const family = String(model.model_family || model.architecture || "").toLowerCase();
  const tags = (model.tags || []).map((value) => String(value).toLowerCase());
  if (filter === "sd1.x") return family.includes("sd1") || family.includes("1.x") || family.includes("1.5");
  if (filter === "sd2.x") return family.includes("sd2") || family.includes("2.x") || family.includes("2.1");
  if (filter === "favorite") return model.favorite === true;
  if (filter === "recent") return true;
  return tags.includes(filter) || String(model.category || "").toLowerCase() === filter;
}

function searchableText(model) {
  return [
    model.name,
    model.nickname,
    model.embedded_name,
    model.display_name,
    model.filename,
    model.path,
    model.description,
    model.source_url,
    model.model_family,
    model.architecture,
    model.architecture_summary,
    model.prediction_type,
    model.conditioning_dimension,
    model.category,
    ...(model.tags || []),
  ].join(" ").toLowerCase();
}

function sortModels(models, mode) {
  const output = [...models];
  const byName = (a, b) => assetLabel(a).localeCompare(assetLabel(b), undefined, { sensitivity: "base" });
  if (mode === "name") return output.sort(byName);
  if (mode === "size_desc") return output.sort((a, b) => Number(b.size_bytes || 0) - Number(a.size_bytes || 0) || byName(a, b));
  if (mode === "size_asc") return output.sort((a, b) => Number(a.size_bytes || 0) - Number(b.size_bytes || 0) || byName(a, b));
  if (mode === "favorites") return output.sort((a, b) => Number(Boolean(b.favorite)) - Number(Boolean(a.favorite)) || byName(a, b));
  return output.sort((a, b) => Number(b.modified_ns || 0) - Number(a.modified_ns || 0) || byName(a, b));
}

export function bindCheckpointWorkspace({
  activateModelPath,
  unloadModel,
  showGenerationWorkspace,
  refreshGenerationModelSelect,
} = {}) {
  let models = Array.isArray(state.models) ? [...state.models] : [];
  let selectedId = "";
  let selectedDetails = null;
  let detailsRequestSerial = 0;
  const detailCache = new Map();
  let search = "";
  let activeFilter = "all";
  let sortMode = "recent";

  const workspace = $("#checkpointWorkspace");
  const detailsPanel = $(".checkpoint-details-panel");

  const startupPath = () => String(state.settings.checkpoint_startup_path || "").trim();
  const isPinnedDefault = (model) => normalizedPath(startupPath()) === normalizedPath(model?.path);
  const isCurrent = (model) => normalizedPath(state.activeModel?.resolved_path) === normalizedPath(model?.path);

  const effectiveModel = (model) => {
    if (!model) return model;
    if (!isCurrent(model)) return model;
    const active = state.activeModel || {};
    const mergedArchitecture = String(active.architecture || model.architecture || "").trim();
    const mergedModelFamily = String(model.model_family || mergedArchitecture || "").trim();
    return {
      ...model,
      architecture: mergedArchitecture,
      architecture_summary: String(active.architecture_summary || model.architecture_summary || mergedArchitecture || "").trim(),
      prediction_type: String(active.prediction_type || model.prediction_type || "").trim(),
      conditioning_dimension: active.conditioning_dimension ?? model.conditioning_dimension ?? null,
      checkpoint_kind: String(active.checkpoint_kind || model.checkpoint_kind || "").trim(),
      model_family: mergedModelFamily,
    };
  };

  const mergeCatalogModel = (payload) => {
    if (!payload?.asset_id) return null;
    let merged = null;
    models = models.map((item) => {
      if (item.asset_id !== payload.asset_id) return item;
      merged = {
        ...item,
        ...payload,
        metadata: payload.metadata || item.metadata,
        model_family: String(payload.model_family || item.model_family || payload.architecture || item.architecture || "").trim(),
        architecture: String(payload.architecture || item.architecture || "").trim(),
        architecture_summary: String(payload.architecture_summary || item.architecture_summary || payload.architecture || item.architecture || "").trim(),
        prediction_type: String(payload.prediction_type || item.prediction_type || "").trim(),
        conditioning_dimension: payload.conditioning_dimension ?? item.conditioning_dimension ?? null,
        checkpoint_kind: String(payload.checkpoint_kind || item.checkpoint_kind || "").trim(),
      };
      return merged;
    });
    if (!merged) return null;
    state.models = [...models];
    state.checkpointCatalog = [...models];
    return merged;
  };

  const saveSettings = async (updates) => {
    const saved = await api.saveSettings(updates);
    state.settings = { ...state.settings, ...saved };
    renderStartupSettings();
    renderCards();
    renderCurrentModel();
    return saved;
  };

  const saveDefaultAssetBehavior = async (enabled) => {
    const profiles = JSON.parse(JSON.stringify(state.defaultAssets?.profiles || {}));
    profiles.auto_apply_on_model_load = Boolean(enabled);
    const saved = await api.saveDefaultAssets(profiles);
    state.defaultAssets = saved;
    window.dispatchEvent(new CustomEvent("image-gen-default-assets-updated", { detail: saved }));
    return saved;
  };

  const renderStartupSettings = () => {
    const mode = String(state.settings.checkpoint_startup_mode || "last_used");
    document.querySelectorAll('input[name="checkpointStartupMode"]').forEach((input) => {
      input.checked = input.value === mode;
    });
    $("#checkpointPreloadOnStartup").checked = state.settings.checkpoint_preload_on_startup !== false;
    $("#checkpointKeepResident").checked = state.settings.memory_retain_checkpoint_between_jobs !== false;
    $("#checkpointAutoApplyDefaults").checked = state.defaultAssets?.auto_apply_on_model_load !== false;
    $("#checkpointStartupStatus").textContent = mode === "pinned_default" && !startupPath()
      ? "Choose a checkpoint and set it as the startup default."
      : "Startup behavior is saved in application settings.";
  };

  const matchedCatalogModel = (path) => {
    const model = models.find((item) => normalizedPath(item.path) === normalizedPath(path)) || null;
    return effectiveModel(model);
  };

  const renderCurrentModel = () => {
    const active = state.activeModel;
    const catalogModel = matchedCatalogModel(active?.resolved_path);
    const runtime = state.bootstrap?.model_runtime || {};
    const runtimeResidency = residencyLabel(runtime);
    $("#currentCheckpointBadge").textContent = runtimeResidency === "UNLOADED" ? (active ? "Selected" : "None") : runtimeResidency;
    $("#currentCheckpointBadge").classList.toggle("ready", runtimeResidency.startsWith("HOT") || runtimeResidency === "MANAGED - RESIDENT");
    $("#currentCheckpointName").textContent = active?.model_name || "No model loaded";
    $("#currentCheckpointArchitecture").textContent = active?.architecture_summary || active?.architecture || catalogModel?.architecture || "—";
    $("#currentCheckpointSize").textContent = active ? formatSize(active.size_bytes, catalogModel?.size_mb) : "—";
    const devices = runtime.component_devices || {};
    $("#currentCheckpointDevice").textContent = devices.unet || devices.transformer || (runtime.gpu_loaded ? "cuda" : (runtime.cpu_loaded ? "cpu" : "—"));
    const residency = residencyLabel(runtime);
    $("#currentCheckpointResidency").textContent = residency;
    const memory = runtime.memory || {};
    const memoryText = memory.free_bytes == null
      ? `${formatGiB(memory.allocated_bytes)} allocated`
      : `${formatGiB(memory.allocated_bytes)} allocated · ${formatGiB(memory.free_bytes)} free`;
    $("#currentCheckpointMemory").textContent = runtime.current_model_path ? memoryText : "—";
    $("#currentCheckpointReuse").textContent = String(runtime.last_generation_residency_classification || "—").replaceAll("_", " ");
    const activationMs = Number(runtime.timings?.activate_time_ms ?? runtime.timings?.initial_activation_time_ms);
    $("#currentCheckpointActivation").textContent = Number.isFinite(activationMs) ? `${activationMs.toFixed(0)} ms` : "—";
    const tags = $("#currentCheckpointTags");
    tags.replaceChildren();
    if (active) addTag(tags, familyLabel(active.architecture || catalogModel?.model_family), "is-family");
    (catalogModel?.tags || []).slice(0, 2).forEach((tag) => addTag(tags, tag));
    setPreview($("#currentCheckpointPreview"), $("#currentCheckpointPreviewFallback"), catalogModel || active);
    $("#checkpointUnloadButton").disabled = !active;
    $("#checkpointUseCurrentButton").disabled = !active;
    $("#checkpointSetStartupButton").disabled = !active;
  };

  const updateSelectedCardState = () => {
    document.querySelectorAll("#checkpointCardGrid .checkpoint-card").forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.assetId === selectedId);
    });
  };

  const openDetails = async (model) => {
    const requestId = ++detailsRequestSerial;
    selectedId = model.asset_id;
    state.selectedCheckpointAssetId = selectedId;
    selectedDetails = effectiveModel(detailCache.get(model.asset_id) || model);
    updateSelectedCardState();
    $("#checkpointDetailEmpty").classList.add("is-hidden");
    $("#checkpointDetailContent").classList.remove("is-hidden");
    detailsPanel?.classList.add("is-open");
    renderDetails();
    $("#checkpointDetailStatus").textContent = detailCache.has(model.asset_id)
      ? "Showing cached checkpoint metadata."
      : "Loading saved checkpoint metadata…";
    try {
      const response = await api.checkpointDetails(model.asset_id, { inspect: false });
      const merged = mergeCatalogModel(response) || response;
      detailCache.set(model.asset_id, merged);
      checkpointGrid.refreshItem(model.asset_id, effectiveModel(merged));
      if (requestId !== detailsRequestSerial || selectedId !== model.asset_id) return;
      selectedDetails = effectiveModel(merged);
      renderCurrentModel();
      renderDetails();
    } catch (error) {
      if (requestId !== detailsRequestSerial || selectedId !== model.asset_id) return;
      selectedDetails = effectiveModel({ ...selectedDetails, inspection_error: error.message });
      renderDetails();
      $("#checkpointDetailStatus").textContent = `Unable to load checkpoint metadata: ${error.message}`;
    }
  };

  const activate = async (model) => {
    if (!activateModelPath) return;
    try {
      await activateModelPath(model.path);
      renderCurrentModel();
      renderCards();
      if (selectedDetails?.asset_id === model.asset_id) renderDetails();
    } catch (error) {
      notify(error.message, "error");
    }
  };

  const setPinnedDefault = async (model, enabled = true) => {
    await saveSettings({
      checkpoint_startup_mode: enabled ? "pinned_default" : "last_used",
      checkpoint_startup_path: enabled ? model.path : "",
    });
    notify(enabled ? `${model.name} is now the pinned startup checkpoint.` : "Pinned startup checkpoint cleared.");
  };

  const toggleFavorite = async (model) => {
    try {
      const updated = await api.saveCheckpointMetadata(model.asset_id, { favorite: !model.favorite });
      models = models.map((item) => item.asset_id === model.asset_id ? { ...item, ...updated } : item);
      state.models = [...models];
      state.checkpointCatalog = [...models];
      if (selectedDetails?.asset_id === model.asset_id) {
        selectedDetails = effectiveModel(updated);
        detailCache.set(model.asset_id, selectedDetails);
      }
      renderCards();
      renderCurrentModel();
      renderDetails();
    } catch (error) {
      notify(`Unable to update favorite: ${error.message}`, "error");
    }
  };

  const createCard = (inputModel) => {
    const model = effectiveModel(inputModel);
    const card = document.createElement("article");
    card.className = "checkpoint-card";
    card.dataset.assetId = model.asset_id;
    card.classList.toggle("is-current", isCurrent(model));
    card.classList.toggle("is-selected", selectedId === model.asset_id);

    const previewWrap = document.createElement("div");
    previewWrap.className = "checkpoint-card-preview-wrap";
    const fallback = document.createElement("div");
    fallback.className = "checkpoint-preview-fallback";
    const image = document.createElement("img");
    image.className = "checkpoint-card-preview";
    image.alt = `${model.name} preview`;
    image.loading = "lazy";
    image.decoding = "async";
    setPreview(image, fallback, model);
    const badges = document.createElement("div");
    badges.className = "checkpoint-card-badges";
    if (isCurrent(model)) {
      const badge = document.createElement("span");
      badge.className = "checkpoint-state-badge is-current";
      badge.textContent = "Currently loaded";
      badges.append(badge);
    } else if (isPinnedDefault(model)) {
      const badge = document.createElement("span");
      badge.className = "checkpoint-state-badge is-default";
      badge.textContent = "Pinned default";
      badges.append(badge);
    }
    previewWrap.append(fallback, image, badges);

    const body = document.createElement("div");
    body.className = "checkpoint-card-body";
    const titleRow = document.createElement("div");
    titleRow.className = "checkpoint-card-title-row";
    const title = document.createElement("strong");
    title.textContent = assetLabel(model);
    const favorite = document.createElement("button");
    favorite.className = "favorite-button";
    favorite.type = "button";
    favorite.textContent = model.favorite ? "★" : "☆";
    favorite.title = model.favorite ? "Remove favorite" : "Add favorite";
    favorite.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleFavorite(model);
    });
    titleRow.append(title, favorite);
    const tags = document.createElement("div");
    tags.className = "checkpoint-tag-row";
    const family = familyLabel(model.model_family || model.architecture);
    if (family !== "Unknown") addTag(tags, family, "is-family");
    (model.tags || []).slice(0, 2).forEach((tag) => addTag(tags, tag));
    const meta = document.createElement("div");
    meta.className = "checkpoint-card-meta";
    meta.textContent = `${model.extension || "file"} · ${formatSize(model.size_bytes, model.size_mb)}`;
    const source = document.createElement("div");
    source.className = "checkpoint-card-source";
    if (model.source_url) {
      const link = document.createElement("a");
      link.href = model.source_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Source ↗";
      source.append(link);
    } else {
      source.textContent = "Local checkpoint";
    }
    body.append(titleRow, tags, meta, source);

    const actions = document.createElement("div");
    actions.className = "checkpoint-card-actions";
    const load = document.createElement("button");
    load.className = "primary-button compact-button";
    load.type = "button";
    load.textContent = isCurrent(model) ? "Loaded" : "Load Model";
    load.disabled = isCurrent(model);
    load.addEventListener("click", () => activate(model));
    const startup = document.createElement("button");
    startup.className = "secondary-button compact-button";
    startup.type = "button";
    startup.textContent = isPinnedDefault(model) ? "★ Startup Default" : "☆ Set as Startup";
    startup.addEventListener("click", () => setPinnedDefault(model, !isPinnedDefault(model)));
    const details = document.createElement("button");
    details.className = "secondary-button compact-button";
    details.type = "button";
    details.textContent = "ⓘ View Details";
    details.addEventListener("click", () => openDetails(model));
    actions.append(load, startup, details);
    card.append(previewWrap, body, actions);
    card.addEventListener("click", (event) => {
      if (event.target.closest("button,a,input,textarea,select,label")) return;
      openDetails(model);
    });
    return card;
  };

  const filteredModels = () => {
    const needle = search.trim().toLowerCase();
    return sortModels(models.map((model) => effectiveModel(model)).filter((model) => (
      (!needle || searchableText(model).includes(needle))
      && modelMatchesFilter(model, activeFilter)
    )), sortMode);
  };

  const checkpointGrid = createProgressiveAssetGrid({
    grid: $("#checkpointCardGrid"),
    createCard,
    batchSize: 50,
    onProgress: ({ shown, total, hasMore }) => {
      const status = $("#checkpointLoadStatus");
      if (!status) return;
      status.textContent = total === 0
        ? "No checkpoint cards to display."
        : hasMore
          ? `Showing ${shown} of ${total} matching checkpoints. Scroll to load more.`
          : `Showing all ${shown} matching checkpoints.`;
    },
  });

  const renderCards = ({ resetWindow = false } = {}) => {
    const visible = filteredModels();
    checkpointGrid.setItems(visible, { reset: resetWindow });
    $("#checkpointResultCount").textContent = String(visible.length);
    $("#checkpointLibrarySummary").textContent = `${visible.length} of ${models.length} installed checkpoints`;
    $("#checkpointEmptyState").classList.toggle("is-hidden", visible.length > 0);
  };

  const broadcastCatalogRefresh = () => {
    window.dispatchEvent(new CustomEvent("image-gen-asset-catalog-refreshed", {
      detail: {
        models: [...models],
        checkpoints: [...models],
        vaes: state.vaes,
        loras: state.loras,
        textual_inversions: state.textualInversions,
      },
    }));
  };

  const renderDetails = () => {
    if (!selectedDetails) return;
    const model = effectiveModel(selectedDetails);
    selectedDetails = model;
    const metadata = model.metadata || {};
    $("#checkpointDetailName").textContent = assetLabel(model);
    $("#checkpointDetailFilename").textContent = model.filename || "—";
    $("#checkpointDetailSize").textContent = formatSize(model.size_bytes, model.size_mb);
    $("#checkpointDetailHash").textContent = model.sha256 || (model.inspection_error ? "Inspection unavailable" : "Not inspected");
    $("#checkpointDetailArchitecture").textContent = model.architecture_summary || model.architecture || metadata.architecture || "Unknown";
    $("#checkpointDetailPrediction").textContent = model.prediction_type || metadata.prediction_type || "Unknown";
    $("#checkpointDetailConditioning").textContent = model.conditioning_dimension || metadata.conditioning_dimension || "Unknown";
    $("#checkpointDetailModified").textContent = model.modified_iso || "—";
    $("#checkpointMetadataNickname").value = metadata.nickname || model.nickname || "";
    $("#checkpointMetadataSourceUrl").value = metadata.source_url || model.source_url || "";
    $("#checkpointMetadataTags").value = (metadata.tags || model.tags || []).join(", ");
    $("#checkpointMetadataNotes").value = metadata.notes || model.notes || "";
    const civitai = civitaiLookupFor(model);
    const civitaiText = $("#checkpointCivitaiMetadataText");
    if (civitaiText) {
      civitaiText.textContent = Object.keys(civitai).length
        ? JSON.stringify(civitai, null, 2)
        : "No CivitAI metadata has been retrieved for this checkpoint.";
    }
    const sourceButton = $("#checkpointOpenSourceButton");
    if (sourceButton) sourceButton.disabled = !sourceUrlFor(model);
    $("#checkpointDetailStartupToggle").checked = isPinnedDefault(model);
    $("#checkpointFavoriteButton").textContent = model.favorite ? "★" : "☆";
    $("#checkpointFavoriteButton").setAttribute("aria-pressed", String(Boolean(model.favorite)));
    const tags = $("#checkpointDetailTags");
    tags.replaceChildren();
    addTag(tags, familyLabel(model.model_family || model.architecture), "is-family");
    (metadata.tags || model.tags || []).slice(0, 4).forEach((tag) => addTag(tags, tag));
    setPreview($("#checkpointDetailPreview"), $("#checkpointDetailPreviewFallback"), model);
    $("#checkpointLoadNowButton").textContent = isCurrent(model) ? "Currently Loaded" : "Load Now";
    $("#checkpointLoadNowButton").disabled = isCurrent(model);
    $("#checkpointDetailStatus").textContent = model.inspection_error
      ? `Header inspection was unavailable: ${model.inspection_error}`
      : "Editable metadata is stored beside the checkpoint in an .imagegen.json sidecar.";
  };

  const refreshCatalog = async ({ announce = true } = {}) => {
    const payload = await api.refreshCheckpointAssets();
    models = Array.isArray(payload.checkpoints) ? payload.checkpoints : [];
    state.models = [...models];
    state.checkpointCatalog = [...models];
    refreshGenerationModelSelect?.();
    renderCards({ resetWindow: true });
    renderCurrentModel();
    if (selectedId) {
      const selected = models.find((model) => model.asset_id === selectedId);
      if (selected) await openDetails(selected);
      else clearDetails();
    }
    if (announce) notify(`Checkpoint library refreshed: ${models.length} model(s).`);
  };

  const clearDetails = () => {
    detailsRequestSerial += 1;
    selectedId = "";
    selectedDetails = null;
    state.selectedCheckpointAssetId = "";
    $("#checkpointDetailEmpty").classList.remove("is-hidden");
    $("#checkpointDetailContent").classList.add("is-hidden");
    detailsPanel?.classList.remove("is-open");
    renderCards();
  };

  const runCivitaiMetadataFetch = async (mode = "missing") => {
    const button = $("#checkpointFetchCivitaiButton");
    const previousLabel = button?.textContent || "Fetch CivitAI Metadata";
    if (button) {
      button.disabled = true;
      button.textContent = "Fetching…";
    }
    try {
      const payload = await api.enrichAssetsFromCivitai("checkpoint", mode);
      models = Array.isArray(payload.checkpoints) ? payload.checkpoints : models;
      state.models = [...models];
      state.checkpointCatalog = [...models];
      detailCache.clear();
      refreshGenerationModelSelect?.();
      renderCards();
      renderCurrentModel();
      broadcastCatalogRefresh();
      if (selectedId) {
        const selected = models.find((model) => model.asset_id === selectedId);
        if (selected) await openDetails(selected);
      }
      const summary = payload?.civitai || {};
      const errors = Array.isArray(summary.errors) ? summary.errors.length : 0;
      notify(`CivitAI metadata fetch complete: ${summary.matched || 0} matched, ${summary.previews_downloaded || 0} preview image(s) downloaded, ${summary.skipped || 0} already current, ${errors} error(s).`, errors ? "warning" : "success");
    } catch (error) {
      notify(`Unable to fetch CivitAI metadata: ${error.message}`, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousLabel;
      }
    }
  };

  const bindSearch = (selector, counterpart) => {
    $(selector)?.addEventListener("input", (event) => {
      search = event.target.value;
      if ($(counterpart) && $(counterpart).value !== search) $(counterpart).value = search;
      renderCards({ resetWindow: true });
    });
  };
  bindSearch("#checkpointSidebarSearch", "#checkpointLibrarySearch");
  bindSearch("#checkpointLibrarySearch", "#checkpointSidebarSearch");

  $("#checkpointFilterChips")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-checkpoint-filter]");
    if (!button) return;
    activeFilter = button.dataset.checkpointFilter || "all";
    document.querySelectorAll("[data-checkpoint-filter]").forEach((candidate) => candidate.classList.toggle("is-active", candidate === button));
    renderCards({ resetWindow: true });
  });
  $("#checkpointClearFiltersButton")?.addEventListener("click", () => {
    activeFilter = "all";
    search = "";
    $("#checkpointSidebarSearch").value = "";
    $("#checkpointLibrarySearch").value = "";
    document.querySelectorAll("[data-checkpoint-filter]").forEach((button) => button.classList.toggle("is-active", button.dataset.checkpointFilter === "all"));
    renderCards({ resetWindow: true });
  });
  $("#checkpointSortSelect")?.addEventListener("change", (event) => {
    sortMode = event.target.value;
    renderCards({ resetWindow: true });
  });
  document.querySelectorAll('input[name="checkpointStartupMode"]').forEach((input) => {
    input.addEventListener("change", async () => {
      if (!input.checked) return;
      try {
        await saveSettings({ checkpoint_startup_mode: input.value });
      } catch (error) {
        notify(`Unable to save startup mode: ${error.message}`, "error");
      }
    });
  });
  $("#checkpointPreloadOnStartup")?.addEventListener("change", async (event) => {
    try { await saveSettings({ checkpoint_preload_on_startup: event.target.checked }); }
    catch (error) { notify(`Unable to save preload behavior: ${error.message}`, "error"); }
  });
  $("#checkpointKeepResident")?.addEventListener("change", async (event) => {
    try {
      await saveSettings({
        memory_retain_checkpoint_between_jobs: event.target.checked,
        memory_retain_vae_between_jobs: event.target.checked,
        model_runtime_retain_text_encoder_between_jobs: event.target.checked,
      });
    } catch (error) { notify(`Unable to save residency behavior: ${error.message}`, "error"); }
  });
  $("#checkpointAutoApplyDefaults")?.addEventListener("change", async (event) => {
    try { await saveDefaultAssetBehavior(event.target.checked); }
    catch (error) { notify(`Unable to save default-asset behavior: ${error.message}`, "error"); }
  });

  window.addEventListener("image-gen-model-runtime-status", (event) => {
    const runtime = event.detail || {};
    if (state.bootstrap) state.bootstrap.model_runtime = runtime;
    renderCurrentModel();
  });
  $("#checkpointRefreshButton")?.addEventListener("click", () => refreshCatalog().catch((error) => notify(error.message, "error")));
  $("#checkpointFetchCivitaiButton")?.addEventListener("click", () => runCivitaiMetadataFetch("missing"));
  $("#closeCheckpointDetailsButton")?.addEventListener("click", clearDetails);
  $("#checkpointUseCurrentButton")?.addEventListener("click", () => showGenerationWorkspace?.());
  $("#checkpointUnloadButton")?.addEventListener("click", async () => {
    if (!window.confirm("Unload the resident checkpoint from CPU/GPU memory?")) return;
    try {
      await unloadModel?.();
      renderCurrentModel();
      renderCards();
      notify("Resident checkpoint unloaded.");
    } catch (error) {
      notify(`Unable to unload checkpoint: ${error.message}`, "error");
    }
  });
  $("#checkpointSetStartupButton")?.addEventListener("click", async () => {
    const model = matchedCatalogModel(state.activeModel?.resolved_path);
    if (model) await setPinnedDefault(model, true);
  });
  $("#checkpointLoadNowButton")?.addEventListener("click", () => selectedDetails && activate(selectedDetails));
  $("#checkpointInspectTechnicalButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    const assetId = selectedDetails.asset_id;
    const requestId = ++detailsRequestSerial;
    const button = $("#checkpointInspectTechnicalButton");
    if (button) {
      button.disabled = true;
      button.textContent = "Inspecting…";
    }
    $("#checkpointDetailStatus").textContent = "Inspecting technical metadata and hashing the checkpoint. Large files may take a moment…";
    try {
      const response = await api.checkpointDetails(assetId, { inspect: true });
      const merged = mergeCatalogModel(response) || response;
      detailCache.set(assetId, merged);
      if (requestId !== detailsRequestSerial || selectedId !== assetId) return;
      selectedDetails = effectiveModel(merged);
      renderDetails();
    } catch (error) {
      if (requestId === detailsRequestSerial && selectedId === assetId) {
        $("#checkpointDetailStatus").textContent = `Unable to inspect checkpoint: ${error.message}`;
      }
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Inspect Technical Metadata";
      }
    }
  });
  $("#checkpointFetchCivitaiDetailButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    const button = $("#checkpointFetchCivitaiDetailButton");
    const previousLabel = button?.textContent || "Refresh from CivitAI";
    if (button) {
      button.disabled = true;
      button.textContent = "Fetching…";
    }
    $("#checkpointDetailStatus").textContent = "Calculating hash and fetching CivitAI metadata…";
    try {
      const updated = await api.enrichAssetFromCivitai("checkpoint", selectedDetails.asset_id, false);
      selectedDetails = mergeCatalogModel(updated) || updated;
      detailCache.set(selectedDetails.asset_id, selectedDetails);
      refreshGenerationModelSelect?.();
      renderCards();
      checkpointGrid.refreshItem(selectedDetails.asset_id, effectiveModel(selectedDetails));
      renderCurrentModel();
      renderDetails();
      broadcastCatalogRefresh();
      const lookup = civitaiLookupFor(selectedDetails);
      notify(lookup.preview_image_downloaded
        ? "CivitAI metadata and a preview image were added to the checkpoint sidecar."
        : "CivitAI metadata was added to the checkpoint sidecar.");
    } catch (error) {
      renderDetails();
      notify(`Unable to fetch CivitAI metadata: ${error.message}`, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousLabel;
      }
    }
  });
  $("#checkpointOpenSourceButton")?.addEventListener("click", () => {
    const url = sourceUrlFor(selectedDetails || {});
    if (url) window.open(url, "_blank", "noopener");
  });
  $("#checkpointOpenFolderButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try { await api.openCheckpointFolder(selectedDetails.asset_id); }
    catch (error) { notify(`Unable to open checkpoint folder: ${error.message}`, "error"); }
  });
  $("#checkpointFavoriteButton")?.addEventListener("click", () => selectedDetails && toggleFavorite(selectedDetails));
  $("#checkpointDetailStartupToggle")?.addEventListener("change", async (event) => {
    if (selectedDetails) await setPinnedDefault(selectedDetails, event.target.checked);
  });
  $("#checkpointSaveMetadataButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try {
      const updated = await api.saveCheckpointMetadata(selectedDetails.asset_id, {
        nickname: $("#checkpointMetadataNickname").value,
        source_url: $("#checkpointMetadataSourceUrl").value,
        tags: $("#checkpointMetadataTags").value,
        notes: $("#checkpointMetadataNotes").value,
        favorite: selectedDetails.favorite === true,
      });
      selectedDetails = mergeCatalogModel(updated) || updated;
      detailCache.set(selectedDetails.asset_id, selectedDetails);
      refreshGenerationModelSelect?.();
      renderCards();
      renderDetails();
      notify("Checkpoint metadata saved.");
    } catch (error) {
      notify(`Unable to save checkpoint metadata: ${error.message}`, "error");
    }
  });
  $("#checkpointReplacePreviewButton")?.addEventListener("click", () => $("#checkpointPreviewFileInput")?.click());
  $("#checkpointChooseRecentPreviewButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try {
      await chooseRecentPreview({
        assetId: selectedDetails.asset_id,
        title: "Checkpoint preview picker",
        loadCandidates: api.loadCheckpointPreviewCandidates,
        replaceFromOutput: api.replaceCheckpointPreviewFromOutput,
        onApplied: (updated) => {
          selectedDetails = mergeCatalogModel({ ...selectedDetails, ...updated }) || { ...selectedDetails, ...updated };
          renderCards();
          renderDetails();
          renderCurrentModel();
          notify("Checkpoint preview replaced from recent output.");
        },
      });
    } catch (error) {
      notify(`Unable to load checkpoint preview candidates: ${error.message}`, "error");
    }
  });
  $("#checkpointPreviewFileInput")?.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    event.target.value = "";
    if (!file || !selectedDetails) return;
    try {
      const updated = await api.replaceCheckpointPreview(selectedDetails.asset_id, file);
      selectedDetails = mergeCatalogModel({
        ...selectedDetails,
        ...updated,
      }) || { ...selectedDetails, ...updated };
      renderCards();
      renderDetails();
      renderCurrentModel();
      notify("Checkpoint preview replaced.");
    } catch (error) {
      notify(`Unable to replace checkpoint preview: ${error.message}`, "error");
    }
  });

  window.addEventListener("image-gen-model-activated", (event) => {
    if (event.detail?.activeModel) {
      state.activeModel = event.detail.activeModel;
      const activeCatalogModel = matchedCatalogModel(event.detail.activeModel?.resolved_path);
      if (activeCatalogModel?.asset_id) mergeCatalogModel({ ...activeCatalogModel, ...event.detail.activeModel });
    }
    if (event.detail?.defaultAssets) state.defaultAssets = event.detail.defaultAssets;
    renderCurrentModel();
    renderCards();
    renderStartupSettings();
    if (selectedDetails) renderDetails();
  });
  window.addEventListener("image-gen-model-unloaded", () => {
    state.activeModel = null;
    renderCurrentModel();
    renderCards();
    if (selectedDetails) renderDetails();
  });
  window.addEventListener("image-gen-asset-installed", (event) => {
    if (String(event.detail?.assetKind || "") !== "checkpoint") return;
    void refreshCatalog({ announce: false }).catch((error) => {
      console.warn("Unable to synchronize the checkpoint workspace after an Asset Hub install", error);
    });
  });

  const show = () => {
    workspace?.classList.remove("is-hidden");
    renderCards();
    renderCurrentModel();
    renderStartupSettings();
  };

  renderStartupSettings();
  renderCards();
  renderCurrentModel();

  return {
    show,
    hide: () => workspace?.classList.add("is-hidden"),
    refresh: refreshCatalog,
    render: () => { renderCards(); renderCurrentModel(); renderStartupSettings(); },
  };
}
