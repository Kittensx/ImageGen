import { api } from "../api.js?v=civitai-connect1";
import { productName } from "../branding.js?v=brand1";
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

function clone(value) {
  return JSON.parse(JSON.stringify(value ?? null));
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
function normalizedPath(value) {
  return String(value || "").trim().replaceAll("\\", "/").toLowerCase();
}

function normalizeFamily(value) {
  const token = String(value || "").trim().toLowerCase().replaceAll("_", " ").replaceAll("-", " ");
  if (!token || token === "unknown") return "";
  if (token.includes("sdxl") || token.includes("sd xl")) return "sdxl";
  if (token.includes("sd2") || token.includes("sd 2") || token.includes("2.0") || token.includes("2.1") || token.includes("1024")) return "sd2.x";
  if (token.includes("sd1") || token.includes("sd 1") || token.includes("1.4") || token.includes("1.5") || token.includes("768")) return "sd1.x";
  if (token === "any" || token === "all") return "any";
  return token.replaceAll(" ", "");
}

function familyLabel(value) {
  const family = normalizeFamily(value);
  if (family === "sd1.x") return "SD 1.x";
  if (family === "sd2.x") return "SD 2.x";
  if (family === "sdxl") return "SDXL";
  if (family === "any") return "Any model";
  return value || "Unknown";
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

function previewUrlFor(model = {}) {
  const url = String(model?.preview_url || "").trim();
  if (!url) return "";
  if (/[?&](?:v|r)=/.test(url)) return url;
  const revision = String(
    model?.preview_revision
    || model?.preview_modified_ns
    || model?.catalog_revision
    || "",
  ).trim();
  if (!revision) return url;
  return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(revision)}`;
}

function selectedCheckpointPath() {
  const field = $("#modelPath");
  return normalizedPath(field?.value || state.activeModel?.resolved_path || "");
}

function selectedCheckpointRecord() {
  const selectedPath = selectedCheckpointPath();
  if (!selectedPath) return null;
  const pools = [state.models, state.checkpointCatalog];
  for (const pool of pools) {
    if (!Array.isArray(pool)) continue;
    const match = pool.find((item) => normalizedPath(item?.path || item?.resolved_path) === selectedPath);
    if (match) return match;
  }
  return null;
}

function selectedCheckpointName() {
  const record = selectedCheckpointRecord();
  if (record) return assetLabel(record);
  return String(
    state.activeModel?.model_name
    || state.defaultAssets?.active_model?.model_name
    || "",
  ).trim();
}

function activeModelFamily() {
  const record = selectedCheckpointRecord();
  return normalizeFamily(
    record?.model_family
    || record?.architecture
    || state.activeModel?.architecture
    || state.activeModel?.architecture_contract?.family
    || state.defaultAssets?.active_model?.model_family
    || "",
  );
}

function compatibilityFor(model) {
  const loraFamily = normalizeFamily(model?.model_family || model?.detected_model_family || model?.architecture);
  const modelFamily = activeModelFamily();
  if (!loraFamily || loraFamily === "any") return { compatible: true, pending: !loraFamily, message: "LoRA model family is not restricted." };
  if (!modelFamily) return { compatible: true, pending: true, message: `Compatibility with ${familyLabel(loraFamily)} will be checked after a checkpoint is selected.` };
  if (loraFamily === modelFamily) return { compatible: true, pending: false, message: `Compatible with the selected ${familyLabel(modelFamily)} checkpoint.` };
  return { compatible: false, pending: false, message: `This LoRA targets ${familyLabel(loraFamily)}, but the selected checkpoint is ${familyLabel(modelFamily)}.` };
}

function formatSize(bytes, sizeMb = 0) {
  const value = Number(bytes || 0) || Number(sizeMb || 0) * 1024 * 1024;
  if (!value) return "Unknown";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  return `${(value / 1024 ** 2).toFixed(value < 100 * 1024 ** 2 ? 1 : 0)} MB`;
}

function assetIdentity(asset = {}) {
  const path = normalizedPath(asset.path);
  const basis = path || String(asset.name || asset.activation_text || "").trim().toLowerCase();
  return `${asset.asset_type || "lora"}|${asset.polarity || "positive"}|${basis}`;
}

function sameAsset(left, right) {
  if (!left || !right) return false;
  if (left.catalog_asset_id && right.asset_id && left.catalog_asset_id === right.asset_id) return true;
  if (left.asset_id && right.asset_id && left.asset_id === right.asset_id) return true;
  return Boolean(normalizedPath(left.path) && normalizedPath(left.path) === normalizedPath(right.path));
}

function addTag(container, text, className = "") {
  if (!container || !text) return;
  const tag = document.createElement("span");
  tag.className = `lora-tag ${className}`.trim();
  tag.textContent = text;
  container.append(tag);
}

function setPreview(image, fallback, model) {
  if (!image || !fallback) return;
  const url = previewUrlFor(model);
  const requestToken = `${model?.asset_id || "detail"}|${url}`;
  image.dataset.previewRequest = requestToken;
  image.classList.remove("has-image");
  image.removeAttribute("src");
  fallback.classList.remove("is-hidden");
  fallback.textContent = String(model?.name || "LORA").slice(0, 4).toUpperCase();
  if (!url) return;
  image.onload = () => {
    if (image.dataset.previewRequest !== requestToken) return;
    image.classList.add("has-image");
    fallback.classList.add("is-hidden");
  };
  image.onerror = () => {
    if (image.dataset.previewRequest !== requestToken) return;
    image.classList.remove("has-image");
    fallback.classList.remove("is-hidden");
  };
  image.src = url;
}

async function copyText(value, label = "Text") {
  const text = String(value || "");
  if (!text) throw new Error(`${label} is unavailable.`);
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
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
    model.notes,
    model.activation_text,
    model.source_url,
    model.model_family,
    model.category,
    ...(model.tags || []),
  ].join(" ").toLowerCase();
}

function loraScanSummary(items) {
  const summary = { total: items.length, scanned: 0, unknown: 0, errors: 0, unsupported: 0 };
  items.forEach((item) => {
    const status = String(item?.scan_status || "").trim().toLowerCase();
    if (status === "error") summary.errors += 1;
    else if (status === "unsupported") summary.unsupported += 1;
    else if (status === "scanned") summary.scanned += 1;
    else if (status === "unscanned" || !status) summary.unknown += 1;
    else summary.scanned += 1;
  });
  return summary;
}

function renderScanSummary(items) {
  const summary = loraScanSummary(items);
  const badge = $("#loraScanStatusBadge");
  if (badge) badge.textContent = String(summary.unknown);
  const label = $("#loraScanSummary");
  if (label) {
    const parts = [];
    parts.push(`${summary.scanned} cached/scanned`);
    if (summary.unknown) parts.push(`${summary.unknown} unidentified`);
    if (summary.errors) parts.push(`${summary.errors} scan error`);
    if (summary.unsupported) parts.push(`${summary.unsupported} unsupported`);
    label.textContent = parts.join(" • ") || "No LoRAs discovered yet.";
  }
}

function sortLoras(items, mode) {
  const output = [...items];
  const byName = (a, b) => assetLabel(a).localeCompare(assetLabel(b), undefined, { sensitivity: "base" });
  if (mode === "name") return output.sort(byName);
  if (mode === "favorites") return output.sort((a, b) => Number(Boolean(b.favorite)) - Number(Boolean(a.favorite)) || byName(a, b));
  if (mode === "weight_desc") return output.sort((a, b) => Number(b.preferred_weight ?? 1) - Number(a.preferred_weight ?? 1) || byName(a, b));
  if (mode === "size_desc") return output.sort((a, b) => Number(b.size_bytes || 0) - Number(a.size_bytes || 0) || byName(a, b));
  return output.sort((a, b) => Number(b.modified_ns || 0) - Number(a.modified_ns || 0) || byName(a, b));
}

function modelMatchesFilter(model, filter) {
  if (filter === "all" || filter === "recent") return true;
  if (filter === "favorite") return model.favorite === true;
  if (filter === "compatible") return compatibilityFor(model).compatible;
  const category = String(model.category || "").trim().toLowerCase();
  const tags = (model.tags || []).map((item) => String(item || "").trim().toLowerCase());
  if (filter === "background") return category === "background" || tags.includes("background") || tags.includes("backgrounds");
  if (filter === "character") return category === "character" || tags.includes("character") || tags.includes("characters");
  if (filter === "pose") return category === "pose" || tags.includes("pose") || tags.includes("poses");
  return category === filter || tags.includes(filter);
}

function insertAtPromptCaret(text) {
  const prompt = $("#positivePrompt");
  if (!prompt) return;
  const value = String(prompt.value || "");
  const start = Number.isInteger(prompt.selectionStart) ? prompt.selectionStart : value.length;
  const end = Number.isInteger(prompt.selectionEnd) ? prompt.selectionEnd : start;
  const before = value.slice(0, start);
  const after = value.slice(end);
  const prefix = before && !/[\s,]$/.test(before) ? ", " : "";
  const suffix = after && !/^[\s,]/.test(after) ? ", " : "";
  prompt.value = `${before}${prefix}${text}${suffix}${after}`;
  const caret = before.length + prefix.length + text.length;
  prompt.setSelectionRange(caret, caret);
  prompt.dispatchEvent(new Event("input", { bubbles: true }));
  prompt.dispatchEvent(new Event("change", { bubbles: true }));
}

function miniDefaultChip(asset) {
  const row = document.createElement("article");
  row.className = `lora-default-chip is-${asset.polarity || "positive"}`;
  const avatar = document.createElement("span");
  avatar.className = "lora-default-avatar";
  const preview = previewUrlFor(asset);
  if (preview) {
    const image = document.createElement("img");
    image.src = preview;
    image.alt = "";
    image.loading = "eager";
    image.decoding = "async";
    avatar.append(image);
  } else {
    avatar.textContent = asset.asset_type === "textual_inversion" ? "TI" : String(asset.name || "L").slice(0, 2).toUpperCase();
  }
  const name = document.createElement("span");
  name.textContent = asset.name || asset.activation_text || "Unnamed asset";
  const weight = document.createElement("span");
  weight.className = "lora-default-weight";
  weight.textContent = asset.asset_type === "lora" ? Number(asset.weight ?? 1).toFixed(2) : "TI";
  row.append(avatar, name, weight);
  return row;
}

export function bindLoraWorkspace({ defaultAssetsController, showGenerationWorkspace } = {}) {
  let loras = Array.isArray(state.loras) ? [...state.loras] : [];
  let selectedId = "";
  let selectedDetails = null;
  let search = "";
  let activeFilter = "all";
  let sortMode = "recent";
  let page = 1;
  let pageSize = 16;
  let detailsRequestSerial = 0;

  const workspace = $("#loraWorkspace");
  const detailsPanel = $(".lora-details-panel");

  const controllerPayload = () => defaultAssetsController?.current?.() || state.defaultAssets || {};
  const activeAssets = () => defaultAssetsController?.activeAssets?.() || clone(state.activePromptAssets || []);
  const integrationMode = () => String(state.settings.lora_prompt_integration_mode || "visual").toLowerCase() === "inline" ? "inline" : "visual";

  const defaultRecordFor = (model) => {
    const payload = controllerPayload();
    const records = [
      ...(payload.effective_assets || []),
      ...(payload.incompatible_assets || []),
      ...(payload.disabled_assets || []),
    ];
    return records.find((item) => item.asset_type === "lora" && sameAsset(item, model)) || null;
  };

  const activeRecordFor = (model) => activeAssets().find((item) => item.asset_type === "lora" && sameAsset(item, model)) || null;

  const mergeCatalogRecord = (payload) => {
    if (!payload?.asset_id) return null;
    let merged = null;
    loras = loras.map((item) => {
      if (item.asset_id !== payload.asset_id) return item;
      merged = { ...item, ...payload, metadata: payload.metadata || item.metadata };
      return merged;
    });
    if (!merged) {
      merged = { ...payload };
      loras.push(merged);
    }
    state.loras = [...loras];
    return merged;
  };

  const applyCatalogPayload = (payload) => {
    loras = Array.isArray(payload?.loras) ? payload.loras.map((item) => ({ ...item })) : [];
    state.loras = [...loras];
    renderScanSummary(loras);
  };

  const setFilterState = (nextFilter) => {
    activeFilter = nextFilter || "all";
    page = 1;
    document.querySelectorAll("[data-lora-filter]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.loraFilter === activeFilter);
    });
  };

  const autoCompatibleFilter = () => {
    if (activeFilter !== "all") return;
    if (!selectedCheckpointPath()) return;
    setFilterState("compatible");
  };

  const renderCompatibilityContext = () => {
    const label = $("#loraCompatibilityContext");
    if (!label) return;
    const name = selectedCheckpointName();
    const family = activeModelFamily();
    const counts = { compatible: 0, pending: 0, incompatible: 0 };
    loras.forEach((model) => {
      const compatibility = compatibilityFor(model);
      if (!compatibility.compatible) counts.incompatible += 1;
      else if (compatibility.pending) counts.pending += 1;
      else counts.compatible += 1;
    });
    if (!selectedCheckpointPath()) {
      label.textContent = "No checkpoint selected yet. Choose a checkpoint to focus the library on compatible and unclassified LoRAs.";
      return;
    }
    const modelLabel = name || "Selected checkpoint";
    if (!family) {
      label.textContent = `${modelLabel} is selected, but its family is still unknown. ${counts.pending} LoRA(s) have unknown family metadata.`;
      return;
    }
    label.textContent = `${modelLabel} • ${familyLabel(family)} • ${counts.compatible} compatible • ${counts.pending} family unknown • ${counts.incompatible} incompatible`;
  };

  const ensureActivationText = async (model) => {
    const existing = String(model?.activation_text || "").trim();
    if (existing) return model;
    const answer = window.prompt(
      `No activation text is saved for "${model?.name || "this LoRA"}".

If this LoRA needs trigger words, enter them now so ${productName()} can append them automatically when the LoRA is active.
Leave the field blank to continue without activation text. Press Cancel to stop adding the LoRA.`,
      "",
    );
    if (answer == null) return null;
    const activationText = String(answer || "").trim();
    if (!activationText) return model;
    try {
      const updated = await api.saveLoraMetadata(model.asset_id, { activation_text: activationText });
      const merged = mergeCatalogRecord(updated) || { ...model, activation_text: activationText };
      if (selectedDetails?.asset_id === merged.asset_id) selectedDetails = merged;
      renderCards();
      if (selectedDetails) renderDetails();
      notify(`Saved activation text for ${merged.name || model.name}.`);
      return merged;
    } catch (error) {
      notify(`Unable to save activation text sidecar: ${error.message}. Continuing with the current session value only.`, "warning");
      return { ...model, activation_text: activationText };
    }
  };

  const renderPromptIntegration = () => {
    const mode = integrationMode();
    document.querySelectorAll('input[name="loraPromptIntegrationMode"]').forEach((input) => {
      input.checked = input.value === mode;
    });
    const status = $("#loraPromptIntegrationStatus");
    if (status) {
      status.textContent = "Prompt boxes are now the source of truth for LoRAs. Adding a LoRA writes A1111-compatible <lora:name:weight> syntax into the positive prompt, appends saved activation text when available, and the visual indicators update automatically from the prompt.";
    }
  };

  const renderDefaults = () => {
    const payload = controllerPayload();
    const assets = payload.effective_assets || [];
    const positive = assets.filter((item) => item.polarity !== "negative");
    const negative = assets.filter((item) => item.polarity === "negative");
    const positiveContainer = $("#loraPositiveDefaults");
    const negativeContainer = $("#loraNegativeDefaults");
    positiveContainer?.replaceChildren();
    negativeContainer?.replaceChildren();
    const enrichDefault = (item) => {
      const catalog = loras.find((model) => sameAsset(item, model));
      return { ...clone(catalog || {}), ...clone(item), preview_url: item.preview_url || catalog?.preview_url || "" };
    };
    if (!positive.length) {
      const empty = document.createElement("div");
      empty.className = "lora-default-empty";
      empty.textContent = "No compatible positive defaults.";
      positiveContainer?.append(empty);
    } else positive.forEach((item) => positiveContainer?.append(miniDefaultChip(enrichDefault(item))));
    if (!negative.length) {
      const empty = document.createElement("div");
      empty.className = "lora-default-empty";
      empty.textContent = "No compatible negative defaults.";
      negativeContainer?.append(empty);
    } else negative.forEach((item) => negativeContainer?.append(miniDefaultChip(enrichDefault(item))));
    if ($("#loraDefaultAssetCount")) $("#loraDefaultAssetCount").textContent = String(assets.length);
    const modelScopeOption = [...($("#loraSaveDefaultScope")?.options || [])].find((option) => option.value === "model");
    if (modelScopeOption) modelScopeOption.disabled = !controllerPayload().active_model?.model_key;
    if ($("#loraSaveDefaultScope") && modelScopeOption?.disabled && $("#loraSaveDefaultScope").value === "model") $("#loraSaveDefaultScope").value = "global";
    if ($("#loraApplySavedDefaults")) $("#loraApplySavedDefaults").checked = Boolean(payload.apply_saved_defaults);
    if ($("#loraAutoApplyDefaults")) $("#loraAutoApplyDefaults").checked = payload.auto_apply_on_model_load !== false;
  };

  const activeRow = (asset) => {
    const row = document.createElement("article");
    row.className = "lora-workspace-active-row";
    row.dataset.assetIdentity = assetIdentity(asset);

    const previewWrap = document.createElement("span");
    previewWrap.className = "lora-active-preview";
    const catalog = loras.find((model) => sameAsset(asset, model));
    const previewUrl = previewUrlFor({ ...catalog, ...asset, preview_url: asset.preview_url || catalog?.preview_url || "" });
    if (previewUrl) {
      const image = document.createElement("img");
      image.src = previewUrl;
      image.alt = "";
      image.loading = "lazy";
      previewWrap.append(image);
    } else previewWrap.textContent = String(asset.name || "L").slice(0, 2).toUpperCase();

    const identity = document.createElement("div");
    identity.className = "lora-active-identity";
    const name = document.createElement("strong");
    name.textContent = asset.name || "Unnamed LoRA";
    const badges = document.createElement("div");
    badges.className = "lora-active-badges";
    if (asset.source_scope === "global" || asset.source_scope === "model") addTag(badges, `Default ${asset.polarity === "negative" ? "Negative" : "Positive"}`, "is-default");
    if (asset.activation_text) addTag(badges, asset.activation_text, "is-trigger");
    identity.append(name, badges);

    const enabled = document.createElement("label");
    enabled.className = "asset-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = asset.enabled !== false;
    const toggleVisual = document.createElement("span");
    enabled.append(checkbox, toggleVisual);

    const weightWrap = document.createElement("div");
    weightWrap.className = "lora-active-weight";
    const range = document.createElement("input");
    range.type = "range";
    range.min = "-2";
    range.max = "2";
    range.step = "0.05";
    range.value = String(Math.max(-2, Math.min(2, Number(asset.weight ?? 1))));
    const number = document.createElement("input");
    number.type = "number";
    number.min = "-4";
    number.max = "4";
    number.step = "0.05";
    number.value = Number(asset.weight ?? 1).toFixed(2);
    weightWrap.append(range, number);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.title = `Remove ${asset.name || "LoRA"}`;
    remove.textContent = "×";

    const identityKey = assetIdentity(asset);
    const promptSync = window.imageGenPromptLoraSync;
    const promptManaged = ["inline", "inline_syntax", "visual", "visual_selection"].includes(String(asset.source || asset.source_scope || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_"));
    const commitWeight = (value) => {
      const parsed = Math.max(-4, Math.min(4, Number(value) || 0));
      range.value = String(Math.max(-2, Math.min(2, parsed)));
      number.value = parsed.toFixed(2);
      if (promptManaged && promptSync?.updateLoraWeight) {
        promptSync.updateLoraWeight(asset, parsed);
        return;
      }
      defaultAssetsController?.updateActiveAsset?.(identityKey, { weight: parsed });
    };
    range.addEventListener("input", () => commitWeight(range.value));
    number.addEventListener("change", () => commitWeight(number.value));
    checkbox.addEventListener("change", () => {
      if (promptManaged && promptSync?.setEnabled) {
        if (!checkbox.checked) promptSync.setEnabled(asset, false);
        else promptSync.syncFromPrompts?.();
        return;
      }
      defaultAssetsController?.updateActiveAsset?.(identityKey, { enabled: checkbox.checked });
    });
    remove.addEventListener("click", () => {
      if (promptManaged && promptSync?.removeLora) {
        promptSync.removeLora(asset);
        return;
      }
      defaultAssetsController?.removeActiveAsset?.(identityKey);
    });
    row.addEventListener("dblclick", () => catalog && openDetails(catalog));

    row.append(previewWrap, identity, enabled, weightWrap, remove);
    return row;
  };

  const renderActive = () => {
    const assets = activeAssets();
    const loraItems = assets.filter((item) => item.asset_type === "lora");
    const tiItems = assets.filter((item) => item.asset_type === "textual_inversion");
    const loraList = $("#loraWorkspaceActiveList");
    const tiList = $("#loraWorkspaceTiList");
    loraList?.replaceChildren();
    tiList?.replaceChildren();
    if (!loraItems.length) {
      const empty = document.createElement("div");
      empty.className = "lora-active-empty";
      empty.textContent = "No LoRAs are active for the current prompt.";
      loraList?.append(empty);
    } else loraItems.forEach((asset) => loraList?.append(activeRow(asset)));
    if (!tiItems.length) {
      const empty = document.createElement("div");
      empty.className = "lora-active-empty";
      empty.textContent = "No textual inversions are active.";
      tiList?.append(empty);
    } else {
      tiItems.forEach((asset) => {
        const chip = document.createElement("span");
        chip.className = `lora-ti-chip is-${asset.polarity || "positive"}`;
        const label = document.createElement("span");
        label.textContent = `${asset.polarity === "negative" ? "Negative TI" : "TI"}: ${asset.name || asset.activation_text}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.addEventListener("click", () => defaultAssetsController?.removeActiveAsset?.(assetIdentity(asset)));
        chip.append(label, remove);
        tiList?.append(chip);
      });
    }
    if ($("#loraWorkspaceActiveCount")) $("#loraWorkspaceActiveCount").textContent = String(loraItems.length);
    if ($("#loraWorkspaceTiCount")) $("#loraWorkspaceTiCount").textContent = String(tiItems.length);
    renderCards();
    if (selectedDetails) renderDetails();
  };

  const stageLora = async (model) => {
    try {
      const compatibility = compatibilityFor(model);
      if (!compatibility.compatible) {
        notify(compatibility.message, "error");
        return;
      }
      const prepared = await ensureActivationText(model);
      if (!prepared) return;
      const existing = activeRecordFor(prepared);
      const weight = Number(existing?.weight ?? prepared.preferred_weight ?? prepared.metadata?.preferred_weight ?? 1);
      const promptSync = window.imageGenPromptLoraSync;
      if (promptSync?.insertLora) {
        promptSync.insertLora({ ...prepared, weight }, { fieldId: "positivePrompt", includeActivationText: true });
        notify(`${prepared.name} added to the positive prompt. Weight edits now sync both ways between the prompt and the active LoRA rows.`);
        return;
      }
      const syntax = `<lora:${prepared.name}:${weight.toFixed(2)}>`;
      const insertion = prepared.activation_text ? `${syntax} ${prepared.activation_text}` : syntax;
      insertAtPromptCaret(insertion);
      defaultAssetsController?.addActiveAsset?.({
        asset_id: prepared.asset_id,
        catalog_asset_id: prepared.asset_id,
        asset_type: "lora",
        polarity: "positive",
        name: prepared.name,
        path: prepared.path,
        weight,
        enabled: true,
        activation_text: prepared.activation_text || "",
        model_family: prepared.model_family || prepared.detected_model_family || "",
        source_url: prepared.source_url || "",
        preview_path: prepared.preview_path || "",
        preview_url: prepared.preview_url || "",
        notes: prepared.notes || "",
      }, { sourceScope: "inline" });
      notify(`${prepared.name} added with inline syntax to the positive prompt.`);
    } catch (error) {
      notify(`Unable to stage LoRA: ${error.message}`, "error");
    }
  };

  const toggleFavorite = async (model) => {
    try {
      const updated = await api.saveLoraMetadata(model.asset_id, { favorite: !model.favorite });
      const merged = mergeCatalogRecord(updated) || updated;
      if (selectedDetails?.asset_id === model.asset_id) selectedDetails = merged;
      renderCards();
      renderDetails();
    } catch (error) {
      notify(`Unable to update favorite: ${error.message}`, "error");
    }
  };

  const createActionButton = (label, title, handler, className = "") => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `lora-card-action ${className}`.trim();
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      handler();
    });
    return button;
  };

  const createCard = (model) => {
    const card = document.createElement("article");
    card.className = "lora-card";
    card.dataset.assetId = model.asset_id;
    card.classList.toggle("is-selected", selectedId === model.asset_id);
    card.classList.toggle("is-active", Boolean(activeRecordFor(model)));
    const compatibility = compatibilityFor(model);
    card.classList.toggle("is-incompatible", !compatibility.compatible);

    const previewWrap = document.createElement("div");
    previewWrap.className = "lora-card-preview-wrap";
    const fallback = document.createElement("div");
    fallback.className = "lora-preview-fallback";
    const image = document.createElement("img");
    image.className = "lora-card-preview";
    image.alt = `${model.name} preview`;
    image.loading = "eager";
    image.decoding = "async";
    const badgeWrap = document.createElement("div");
    badgeWrap.className = "lora-card-badges";
    if (activeRecordFor(model)) {
      const badge = document.createElement("span");
      badge.className = "lora-state-badge is-active";
      badge.textContent = "Currently active";
      badgeWrap.append(badge);
    } else {
      const defaultRecord = defaultRecordFor(model);
      if (defaultRecord) {
        const badge = document.createElement("span");
        badge.className = `lora-state-badge is-default is-${defaultRecord.polarity || "positive"}`;
        badge.textContent = `Default ${defaultRecord.polarity === "negative" ? "Negative" : "Positive"}`;
        badgeWrap.append(badge);
      } else if (!compatibility.compatible) {
        const badge = document.createElement("span");
        badge.className = "lora-state-badge is-incompatible";
        badge.textContent = "Incompatible";
        badgeWrap.append(badge);
      } else if (compatibility.pending) {
        const badge = document.createElement("span");
        badge.className = "lora-state-badge is-pending";
        badge.textContent = "Family unknown";
        badgeWrap.append(badge);
      } else {
        const badge = document.createElement("span");
        badge.className = "lora-state-badge is-compatible";
        badge.textContent = "Compatible";
        badgeWrap.append(badge);
      }
    }
    const favorite = document.createElement("button");
    favorite.className = "favorite-button lora-card-favorite";
    favorite.type = "button";
    favorite.textContent = model.favorite ? "★" : "☆";
    favorite.title = model.favorite ? "Remove favorite" : "Add favorite";
    favorite.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleFavorite(model);
    });
    previewWrap.append(fallback, image, badgeWrap, favorite);

    const body = document.createElement("div");
    body.className = "lora-card-body";
    const title = document.createElement("strong");
    title.textContent = assetLabel(model);
    title.title = assetLabel(model);
    const triggerRow = document.createElement("div");
    triggerRow.className = "lora-card-trigger";
    const triggerLabel = document.createElement("span");
    triggerLabel.textContent = "Trigger:";
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "text-button";
    trigger.textContent = model.activation_text || "not set";
    trigger.title = model.activation_text ? "Copy activation text" : "No activation text saved";
    trigger.disabled = !model.activation_text;
    trigger.addEventListener("click", async (event) => {
      event.stopPropagation();
      try { await copyText(model.activation_text, "Activation text"); notify("Activation text copied."); }
      catch (error) { notify(error.message, "error"); }
    });
    triggerRow.append(triggerLabel, trigger);
    const tagRow = document.createElement("div");
    tagRow.className = "lora-tag-row";
    if (model.category) addTag(tagRow, model.category);
    const family = familyLabel(model.model_family);
    if (family !== "Unknown") addTag(tagRow, family, "is-family");
    (model.tags || []).filter((tag) => String(tag).toLowerCase() !== String(model.category || "").toLowerCase()).slice(0, 2).forEach((tag) => addTag(tagRow, tag));
    const info = document.createElement("div");
    info.className = "lora-card-info";
    const source = document.createElement("span");
    source.textContent = model.source_url ? "Source available" : formatSize(model.size_bytes, model.size_mb);
    const weight = document.createElement("span");
    weight.className = "lora-weight-badge";
    weight.textContent = Number(model.preferred_weight ?? 1).toFixed(2);
    info.append(source, weight);
    body.append(title, triggerRow, tagRow, info);

    const actions = document.createElement("div");
    actions.className = "lora-card-actions";
    actions.append(
      createActionButton("+", `Add ${assetLabel(model)} to the current prompt`, () => stageLora(model), "is-primary"),
      createActionButton("◉", `View ${assetLabel(model)} details`, () => openDetails(model)),
      createActionButton("✎", `Edit ${assetLabel(model)} metadata`, () => openDetails(model, { focusEditor: true })),
      createActionButton("⧉", `Copy ${assetLabel(model)} file path`, async () => {
        try { await copyText(model.path, "LoRA path"); notify("LoRA path copied."); }
        catch (error) { notify(error.message, "error"); }
      }),
      createActionButton("↗", `Open ${assetLabel(model)} source website`, () => {
        if (model.source_url) window.open(model.source_url, "_blank", "noopener");
        else notify("No source URL is saved for this LoRA.", "error");
      }),
    );
    card.append(previewWrap, body, actions);
    card.addEventListener("dblclick", () => openDetails(model));
    queueMicrotask(() => {
      if (image.isConnected) setPreview(image, fallback, model);
    });
    return card;
  };

  const filteredLoras = () => {
    const needle = search.trim().toLowerCase();
    return sortLoras(loras.filter((model) => (
      (!needle || searchableText(model).includes(needle))
      && modelMatchesFilter(model, activeFilter)
    )), sortMode);
  };

  const renderCards = () => {
    autoCompatibleFilter();
    const visible = filteredLoras();
    const pageCount = Math.max(1, Math.ceil(visible.length / pageSize));
    page = Math.max(1, Math.min(page, pageCount));
    const offset = (page - 1) * pageSize;
    const pageItems = visible.slice(offset, offset + pageSize);
    const grid = $("#loraCardGrid");
    grid?.replaceChildren(...pageItems.map(createCard));
    if ($("#loraResultCount")) $("#loraResultCount").textContent = String(visible.length);
    if ($("#loraLibrarySummary")) {
      const summary = loraScanSummary(loras);
      const suffix = summary.unknown ? ` • ${summary.unknown} unidentified` : "";
      const filterLabel = activeFilter === "compatible" ? " compatible/unclassified" : "";
      $("#loraLibrarySummary").textContent = `${visible.length} of ${loras.length} installed LoRAs${filterLabel}${suffix}`;
    }
    renderScanSummary(loras);
    renderCompatibilityContext();
    if ($("#loraEmptyState")) $("#loraEmptyState").classList.toggle("is-hidden", visible.length > 0);
    if ($("#loraPageStatus")) $("#loraPageStatus").textContent = `Page ${page} of ${pageCount}`;
    if ($("#loraPreviousPageButton")) $("#loraPreviousPageButton").disabled = page <= 1;
    if ($("#loraNextPageButton")) $("#loraNextPageButton").disabled = page >= pageCount;
    if ($("#loraPagination")) $("#loraPagination").classList.toggle("is-hidden", visible.length === 0);
  };

  const renderDetails = () => {
    if (!selectedDetails) return;
    const model = selectedDetails;
    const metadata = model.metadata || {};
    $("#loraDetailName").textContent = assetLabel(model);
    $("#loraDetailFilename").textContent = model.filename || "—";
    $("#loraDetailSize").textContent = formatSize(model.size_bytes, model.size_mb);
    $("#loraDetailHash").textContent = model.sha256 || (model.inspection_error ? "Inspection unavailable" : "Not inspected");
    $("#loraDetailNetworkType").textContent = model.network_type || "Unknown";
    $("#loraDetailTensorFormat").textContent = model.tensor_key_format || "Unknown";
    $("#loraDetailModelFamily").textContent = familyLabel(model.model_family || model.detected_model_family);
    $("#loraDetailModified").textContent = model.modified_iso || "—";
    $("#loraMetadataNickname").value = metadata.nickname || model.nickname || "";
    $("#loraMetadataActivationText").value = metadata.activation_text || model.activation_text || "";
    $("#loraMetadataPreferredWeight").value = Number(metadata.preferred_weight ?? model.preferred_weight ?? 1).toFixed(2);
    $("#loraMetadataModelFamily").value = normalizeFamily(metadata.model_family || model.model_family || model.detected_model_family);
    $("#loraMetadataSourceUrl").value = metadata.source_url || model.source_url || civitaiLookupFor(model).source_url || "";
    $("#loraMetadataCategory").value = metadata.category || model.category || "";
    $("#loraMetadataTags").value = (metadata.tags || model.tags || []).join(", ");
    $("#loraMetadataDescription").value = metadata.description || model.description || "";
    $("#loraMetadataNotes").value = metadata.notes || model.notes || "";
    $("#loraFavoriteButton").textContent = model.favorite ? "★" : "☆";
    $("#loraFavoriteButton").setAttribute("aria-pressed", String(Boolean(model.favorite)));
    const tags = $("#loraDetailTags");
    tags.replaceChildren();
    if (model.category) addTag(tags, model.category);
    const family = familyLabel(model.model_family || model.detected_model_family);
    if (family !== "Unknown") addTag(tags, family, "is-family");
    (metadata.tags || model.tags || []).slice(0, 4).forEach((tag) => addTag(tags, tag));
    setPreview($("#loraDetailPreview"), $("#loraDetailPreviewFallback"), model);
    const active = activeRecordFor(model);
    $("#loraDetailAddButton").textContent = active ? "Update Active LoRA" : "+ Add to Prompt";
    $("#loraOpenSourceButton").disabled = !sourceUrlFor(model);
    const compatibility = compatibilityFor(model);
    const civitai = civitaiLookupFor(model);
    const civitaiText = $("#loraCivitaiMetadataText");
    if (civitaiText) {
      civitaiText.textContent = Object.keys(civitai).length
        ? JSON.stringify(civitai, null, 2)
        : "No CivitAI metadata has been retrieved for this LoRA.";
    }
    let civitaiStatus = "";
    if (civitai.status === "matched" && civitai.manual_activation_text_search_required) {
      civitaiStatus = " Civitai matched the file but returned no trainedWords; open the Civitai page to check the description or comments for activation text.";
    } else if (civitai.status === "matched" && civitai.activation_text) {
      civitaiStatus = " Civitai metadata includes activation text from trainedWords.";
    }
    $("#loraDetailStatus").textContent = model.inspection_error
      ? `Technical inspection was unavailable: ${model.inspection_error}${civitaiStatus}`
      : `${compatibility.message} Editable metadata is stored in an .imagegen.json sidecar.${civitaiStatus}`;
    $("#loraDetailStatus").classList.toggle("warning", !compatibility.compatible || Boolean(civitai.manual_activation_text_search_required));
  };

  const openDetails = async (model, { focusEditor = false } = {}) => {
    const requestId = ++detailsRequestSerial;
    selectedId = model.asset_id;
    state.selectedLoraAssetId = selectedId;
    renderCards();
    $("#loraDetailEmpty")?.classList.add("is-hidden");
    $("#loraDetailContent")?.classList.remove("is-hidden");
    detailsPanel?.classList.add("is-open");
    $("#loraDetailStatus").textContent = "Inspecting LoRA metadata…";
    try {
      const response = await api.loraDetails(model.asset_id);
      const merged = mergeCatalogRecord(response) || response;
      renderCards();
      if (requestId !== detailsRequestSerial || selectedId !== model.asset_id) return;
      selectedDetails = merged;
      renderDetails();
      if (focusEditor) $("#loraMetadataNickname")?.focus();
    } catch (error) {
      if (requestId !== detailsRequestSerial || selectedId !== model.asset_id) return;
      selectedDetails = { ...model, metadata: {}, inspection_error: error.message };
      renderDetails();
    }
  };

  const clearDetails = () => {
    detailsRequestSerial += 1;
    selectedId = "";
    selectedDetails = null;
    state.selectedLoraAssetId = "";
    $("#loraDetailEmpty")?.classList.remove("is-hidden");
    $("#loraDetailContent")?.classList.add("is-hidden");
    detailsPanel?.classList.remove("is-open");
    renderCards();
  };

  const refreshCatalog = async ({ announce = true } = {}) => {
    const payload = await api.refreshLoraAssets();
    applyCatalogPayload(payload);
    window.dispatchEvent(new CustomEvent("image-gen-asset-catalog-refreshed", {
      detail: { loras: [...loras], textual_inversions: state.textualInversions },
    }));
    renderCards();
    renderDefaults();
    renderActive();
    if (selectedId) {
      const selected = loras.find((item) => item.asset_id === selectedId);
      if (selected) await openDetails(selected);
      else clearDetails();
    }
    const scan = payload?.scan || null;
    if (announce) {
      const scanSuffix = scan ? ` ${scan.scanned || 0} scanned, ${Math.max(0, loraScanSummary(loras).unknown)} unidentified remaining.` : "";
      notify(`LoRA library refreshed: ${loras.length} installed LoRA(s).${scanSuffix}`);
    }
  };

  const runCompatibilityScan = async (mode = "missing") => {
    const button = $("#loraScanUnknownButton");
    if (button) button.disabled = true;
    try {
      const payload = await api.scanLoras(mode);
      applyCatalogPayload(payload);
      renderCards();
      renderDefaults();
      renderActive();
      if (selectedId) {
        const selected = loras.find((item) => item.asset_id === selectedId);
        if (selected) await openDetails(selected);
        else clearDetails();
      }
      const scan = payload?.scan || {};
      notify(`LoRA compatibility scan complete: ${scan.scanned || 0} scanned, ${scan.errors || 0} error(s), ${scan.unsupported || 0} unsupported.`);
    } catch (error) {
      notify(`Unable to scan LoRAs: ${error.message}`, "error");
    } finally {
      if (button) button.disabled = false;
    }
  };

  const runCivitaiMetadataFetch = async (mode = "missing") => {
    const button = $("#loraFetchCivitaiButton");
    const previousLabel = button?.textContent || "Fetch Civitai Metadata";
    if (button) {
      button.disabled = true;
      button.textContent = "Fetching…";
    }
    try {
      const payload = await api.enrichAssetsFromCivitai("lora", mode);
      applyCatalogPayload(payload);
      renderCards();
      renderDefaults();
      renderActive();
      if (selectedId) {
        const selected = loras.find((item) => item.asset_id === selectedId);
        if (selected) await openDetails(selected);
        else clearDetails();
      }
      const summary = payload?.civitai || {};
      const errors = Array.isArray(summary.errors) ? summary.errors.length : 0;
      const previewErrors = Number(summary.preview_download_errors || 0);
      notify(`Civitai metadata fetch complete: ${summary.matched || 0} matched, ${summary.activation_text_found || 0} with activation text, ${summary.previews_downloaded || 0} preview image(s) downloaded, ${summary.manual_search_required || 0} requiring manual page review, ${errors + previewErrors} error(s).`, errors || previewErrors ? "warning" : "success");
    } catch (error) {
      notify(`Unable to fetch Civitai metadata: ${error.message}`, "error");
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
      page = 1;
      if ($(counterpart) && $(counterpart).value !== search) $(counterpart).value = search;
      renderCards();
    });
  };
  bindSearch("#loraSidebarSearch", "#loraLibrarySearch");
  bindSearch("#loraLibrarySearch", "#loraSidebarSearch");

  $("#loraFilterChips")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-lora-filter]");
    if (!button) return;
    setFilterState(button.dataset.loraFilter || "all");
    renderCards();
  });
  $("#loraClearFiltersButton")?.addEventListener("click", () => {
    search = "";
    if ($("#loraSidebarSearch")) $("#loraSidebarSearch").value = "";
    if ($("#loraLibrarySearch")) $("#loraLibrarySearch").value = "";
    setFilterState(selectedCheckpointPath() ? "compatible" : "all");
    renderCards();
  });
  $("#loraSortSelect")?.addEventListener("change", (event) => {
    sortMode = event.target.value;
    page = 1;
    renderCards();
  });
  $("#loraPageSizeSelect")?.addEventListener("change", (event) => {
    pageSize = Math.max(4, Number(event.target.value) || 16);
    page = 1;
    renderCards();
  });
  $("#loraPreviousPageButton")?.addEventListener("click", () => { page -= 1; renderCards(); });
  $("#loraNextPageButton")?.addEventListener("click", () => { page += 1; renderCards(); });
  $("#loraRefreshButton")?.addEventListener("click", () => refreshCatalog().catch((error) => notify(`Unable to refresh LoRAs: ${error.message}`, "error")));
  $("#loraScanUnknownButton")?.addEventListener("click", () => runCompatibilityScan("missing"));
  $("#loraFetchCivitaiButton")?.addEventListener("click", () => runCivitaiMetadataFetch("missing"));
  if ($("#loraAutoScanUnknown")) $("#loraAutoScanUnknown").checked = state.settings.lora_auto_scan_unknown_on_startup !== false;
  $("#loraAutoScanUnknown")?.addEventListener("change", async (event) => {
    try {
      const saved = await api.saveSettings({ lora_auto_scan_unknown_on_startup: event.target.checked });
      state.settings = { ...state.settings, ...saved };
      renderScanSummary(loras);
    } catch (error) {
      event.target.checked = !event.target.checked;
      notify(`Unable to save LoRA auto-scan setting: ${error.message}`, "error");
    }
  });

  document.querySelectorAll('input[name="loraPromptIntegrationMode"]').forEach((input) => {
    input.addEventListener("change", async () => {
      if (!input.checked) return;
      try {
        const saved = await api.saveSettings({ lora_prompt_integration_mode: input.value });
        state.settings = { ...state.settings, ...saved };
        renderPromptIntegration();
      } catch (error) {
        notify(`Unable to save prompt integration mode: ${error.message}`, "error");
      }
    });
  });

  $("#loraApplySavedDefaults")?.addEventListener("change", async (event) => {
    try {
      await defaultAssetsController?.setApplySavedDefaults?.(event.target.checked);
      renderDefaults();
      renderActive();
    } catch (error) {
      event.target.checked = !event.target.checked;
      notify(`Unable to save default behavior: ${error.message}`, "error");
    }
  });
  $("#loraAutoApplyDefaults")?.addEventListener("change", async (event) => {
    try {
      await defaultAssetsController?.setAutoApplyOnModelLoad?.(event.target.checked);
      renderDefaults();
    } catch (error) {
      event.target.checked = !event.target.checked;
      notify(`Unable to save model-load behavior: ${error.message}`, "error");
    }
  });
  $("#loraLoadDefaultsButton")?.addEventListener("click", () => {
    defaultAssetsController?.loadEffectiveDefaults?.();
    renderActive();
  });
  $("#loraEditDefaultsButton")?.addEventListener("click", () => defaultAssetsController?.openEditor?.({ type: "lora" }));
  $("#loraManageDefaultsButton")?.addEventListener("click", () => defaultAssetsController?.openEditor?.({ type: "lora" }));
  $("#loraSaveActiveDefaultsButton")?.addEventListener("click", async () => {
    const scope = $("#loraSaveDefaultScope")?.value || "global";
    const label = scope === "model" ? "the current model profile" : "the global profile";
    if (!window.confirm(`Replace ${label} with the currently active prompt assets?`)) return;
    try {
      await defaultAssetsController?.saveActiveAssetsAsDefaults?.(scope);
      renderDefaults();
      renderCards();
      notify(`Current prompt assets saved to ${label}.`);
    } catch (error) {
      notify(`Unable to save current defaults: ${error.message}`, "error");
    }
  });
  $("#loraClearActiveButton")?.addEventListener("click", () => defaultAssetsController?.clearActiveAssets?.());
  $("#loraApplyToPromptButton")?.addEventListener("click", () => {
    const count = activeAssets().filter((item) => item.asset_type === "lora" && item.enabled !== false).length;
    showGenerationWorkspace?.();
    notify(`${count} active LoRA selection(s) are already being driven by the prompt boxes.`);
  });

  $("#closeLoraDetailsButton")?.addEventListener("click", clearDetails);
  $("#loraFavoriteButton")?.addEventListener("click", () => selectedDetails && toggleFavorite(selectedDetails));
  $("#loraDetailAddButton")?.addEventListener("click", () => selectedDetails && stageLora(selectedDetails));
  $("#loraCopyPathButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try { await copyText(selectedDetails.path, "LoRA path"); notify("LoRA path copied."); }
    catch (error) { notify(error.message, "error"); }
  });
  $("#loraCopySyntaxButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    const syntax = `<lora:${selectedDetails.name}:${Number(selectedDetails.preferred_weight ?? 1).toFixed(2)}>`;
    try { await copyText(syntax, "LoRA syntax"); notify("A1111 LoRA syntax copied."); }
    catch (error) { notify(error.message, "error"); }
  });
  $("#loraFetchCivitaiDetailButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    const button = $("#loraFetchCivitaiDetailButton");
    const previousLabel = button?.textContent || "Refresh from Civitai";
    if (button) {
      button.disabled = true;
      button.textContent = "Fetching…";
    }
    $("#loraDetailStatus").textContent = "Calculating hashes and fetching Civitai metadata…";
    try {
      const updated = await api.enrichAssetFromCivitai("lora", selectedDetails.asset_id, false);
      selectedDetails = mergeCatalogRecord(updated) || updated;
      renderCards();
      renderDetails();
      renderDefaults();
      renderActive();
      const lookup = civitaiLookupFor(selectedDetails);
      if (lookup.manual_activation_text_search_required) {
        const previewNote = lookup.preview_image_downloaded ? " A preview image was downloaded." : "";
        notify(`Civitai matched this LoRA but returned no activation text. Use Open Civitai / Source Page to check the description or comments.${previewNote}`, "warning");
      } else if (lookup.preview_image_downloaded) {
        notify("Civitai metadata and a preview image were added to the LoRA sidecar.");
      } else {
        notify("Civitai metadata was added to the LoRA sidecar.");
      }
    } catch (error) {
      renderDetails();
      notify(`Unable to fetch Civitai metadata: ${error.message}`, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousLabel;
      }
    }
  });
  $("#loraOpenSourceButton")?.addEventListener("click", () => {
    const url = sourceUrlFor(selectedDetails || {});
    if (url) window.open(url, "_blank", "noopener");
  });
  $("#loraOpenFolderButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try { await api.openLoraFolder(selectedDetails.asset_id); }
    catch (error) { notify(`Unable to open LoRA folder: ${error.message}`, "error"); }
  });
  $("#loraReplacePreviewButton")?.addEventListener("click", () => $("#loraPreviewFileInput")?.click());
  $("#loraChooseRecentPreviewButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try {
      await chooseRecentPreview({
        assetId: selectedDetails.asset_id,
        title: "LoRA preview picker",
        loadCandidates: api.loadLoraPreviewCandidates,
        replaceFromOutput: api.replaceLoraPreviewFromOutput,
        onApplied: (updated) => {
          selectedDetails = mergeCatalogRecord({ ...selectedDetails, ...updated }) || updated;
          renderCards();
          renderDetails();
          renderActive();
          notify("LoRA preview replaced from recent output.");
        },
      });
    } catch (error) {
      notify(`Unable to load LoRA preview candidates: ${error.message}`, "error");
    }
  });
  $("#loraPreviewFileInput")?.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    event.target.value = "";
    if (!file || !selectedDetails) return;
    try {
      const updated = await api.replaceLoraPreview(selectedDetails.asset_id, file);
      selectedDetails = mergeCatalogRecord({ ...selectedDetails, ...updated }) || updated;
      renderCards();
      renderDetails();
      renderActive();
      notify("LoRA preview replaced.");
    } catch (error) {
      notify(`Unable to replace LoRA preview: ${error.message}`, "error");
    }
  });
  $("#loraSaveMetadataButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    try {
      const updated = await api.saveLoraMetadata(selectedDetails.asset_id, {
        nickname: $("#loraMetadataNickname").value,
        activation_text: $("#loraMetadataActivationText").value,
        preferred_weight: Number($("#loraMetadataPreferredWeight").value || 1),
        model_family: $("#loraMetadataModelFamily").value,
        source_url: $("#loraMetadataSourceUrl").value,
        category: $("#loraMetadataCategory").value,
        tags: $("#loraMetadataTags").value,
        description: $("#loraMetadataDescription").value,
        notes: $("#loraMetadataNotes").value,
        favorite: selectedDetails.favorite === true,
      });
      selectedDetails = mergeCatalogRecord(updated) || updated;
      renderCards();
      renderDetails();
      renderDefaults();
      renderActive();
      window.dispatchEvent(new CustomEvent("image-gen-asset-catalog-refreshed", {
        detail: { loras: [...loras], textual_inversions: state.textualInversions },
      }));
      notify("LoRA metadata saved.");
    } catch (error) {
      notify(`Unable to save LoRA metadata: ${error.message}`, "error");
    }
  });
  $("#loraDeleteButton")?.addEventListener("click", async () => {
    if (!selectedDetails) return;
    const filename = selectedDetails.filename || selectedDetails.name;
    const confirmation = window.prompt(`This permanently deletes the LoRA file, its ${productName()} sidecar, and its preview.\n\nType ${filename} to confirm.`);
    if (confirmation !== filename) return;
    try {
      await api.deleteLora(selectedDetails.asset_id);
      defaultAssetsController?.removeActiveAsset?.(selectedDetails.asset_id);
      loras = loras.filter((item) => item.asset_id !== selectedDetails.asset_id);
      state.loras = [...loras];
      clearDetails();
      renderCards();
      renderScanSummary(loras);
      notify(`${filename} was deleted.`);
    } catch (error) {
      notify(`Unable to delete LoRA: ${error.message}`, "error");
    }
  });

  window.addEventListener("image-gen-active-prompt-assets-updated", renderActive);
  window.addEventListener("image-gen-default-assets-updated", () => {
    renderDefaults();
    renderCards();
    renderActive();
  });
  window.addEventListener("image-gen-model-activated", () => {
    renderCards();
    renderDefaults();
    if (selectedDetails) renderDetails();
  });
  window.addEventListener("image-gen-model-unloaded", () => {
    renderCards();
    if (selectedDetails) renderDetails();
  });

  const show = () => {
    workspace?.classList.remove("is-hidden");
    renderPromptIntegration();
    if ($("#loraAutoScanUnknown")) $("#loraAutoScanUnknown").checked = state.settings.lora_auto_scan_unknown_on_startup !== false;
    renderDefaults();
    renderActive();
    renderCards();
    renderScanSummary(loras);
  };

  renderPromptIntegration();
  if ($("#loraAutoScanUnknown")) $("#loraAutoScanUnknown").checked = state.settings.lora_auto_scan_unknown_on_startup !== false;
  renderDefaults();
  renderActive();
  renderCards();
  renderScanSummary(loras);

  return {
    show,
    hide: () => workspace?.classList.add("is-hidden"),
    refresh: refreshCatalog,
    render: () => {
      renderPromptIntegration();
      if ($("#loraAutoScanUnknown")) $("#loraAutoScanUnknown").checked = state.settings.lora_auto_scan_unknown_on_startup !== false;
      renderDefaults();
      renderActive();
      renderCards();
      renderScanSummary(loras);
    },
  };
}
