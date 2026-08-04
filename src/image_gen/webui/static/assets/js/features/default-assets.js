import { api } from "../api.js";
import { state } from "../state.js";
import { $, notify } from "../utils.js";

function clone(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function contractSource(value, fallback = "visual_selection") {
  const token = String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  const aliases = {
    visual: "visual_selection",
    visual_selection: "visual_selection",
    inline: "inline_syntax",
    inline_syntax: "inline_syntax",
    model: "model_default",
    model_default: "model_default",
    global: "global_default",
    global_default: "global_default",
    replay: "replay",
  };
  return aliases[token] || token || fallback;
}

function emptyProfiles() {
  return {
    contract_version: "image-gen-default-assets-v1",
    apply_saved_defaults: false,
    auto_apply_on_model_load: true,
    global_profile: {
      profile_id: "global",
      display_name: "Global defaults",
      model_key: "",
      model_path: "",
      model_name: "",
      model_family: "",
      assets: [],
    },
    model_profiles: {},
  };
}

function normalizedPayload(value = {}) {
  const profiles = value.profiles && typeof value.profiles === "object"
    ? value.profiles
    : emptyProfiles();
  return {
    ...value,
    profiles: {
      ...emptyProfiles(),
      ...profiles,
      global_profile: { ...emptyProfiles().global_profile, ...(profiles.global_profile || {}) },
      model_profiles: { ...(profiles.model_profiles || {}) },
    },
    active_model: { ...(value.active_model || {}) },
    effective_assets: Array.isArray(value.effective_assets) ? value.effective_assets : [],
    incompatible_assets: Array.isArray(value.incompatible_assets) ? value.incompatible_assets : [],
    disabled_assets: Array.isArray(value.disabled_assets) ? value.disabled_assets : [],
    counts: { total: 0, loras: 0, textual_inversions: 0, positive: 0, negative: 0, incompatible: 0, disabled: 0, ...(value.counts || {}) },
    apply_saved_defaults: Boolean(value.apply_saved_defaults ?? profiles.apply_saved_defaults),
    auto_apply_on_model_load: Boolean(value.auto_apply_on_model_load ?? profiles.auto_apply_on_model_load ?? true),
  };
}

function assetIdentity(asset = {}) {
  const path = String(asset.path || "").trim().replaceAll("\\", "/").toLowerCase();
  const basis = path || String(asset.name || asset.activation_text || "").trim().toLowerCase();
  return `${asset.asset_type || "lora"}|${asset.polarity || "positive"}|${basis}`;
}

function createEmptyState(message) {
  const element = document.createElement("div");
  element.className = "startup-defaults-empty";
  element.textContent = message;
  return element;
}

function createScopeBadge(scope) {
  const source = contractSource(scope || "global_default");
  const labels = {
    visual_selection: "Visual",
    inline_syntax: "Inline",
    model_default: "Model Default",
    global_default: "Global Default",
    replay: "Replay",
    imported: "Imported",
    api_request: "API",
  };
  const badge = document.createElement("span");
  badge.className = `default-asset-scope-badge is-${source}`;
  badge.textContent = labels[source] || source.replaceAll("_", " ");
  return badge;
}

function createDefaultAssetChip(asset, { compact = false } = {}) {
  const row = document.createElement("article");
  row.className = compact ? "default-asset-chip is-compact" : "default-asset-chip";
  row.dataset.assetId = String(asset.asset_id || "");

  const identity = document.createElement("div");
  identity.className = "default-asset-chip-identity";
  const icon = document.createElement("span");
  icon.className = `default-asset-type-icon is-${asset.asset_type}`;
  icon.textContent = asset.asset_type === "textual_inversion" ? "TI" : "L";
  const label = document.createElement("span");
  label.className = "default-asset-chip-label";
  label.textContent = asset.name || "Unnamed asset";
  identity.append(icon, label);

  const meta = document.createElement("div");
  meta.className = "default-asset-chip-meta";
  meta.append(createScopeBadge(asset.source || asset.source_scope || "global_default"));
  const polarity = document.createElement("span");
  polarity.className = `default-asset-polarity is-${asset.polarity || "positive"}`;
  polarity.textContent = asset.polarity === "negative" ? "Negative" : "Positive";
  meta.append(polarity);
  if (asset.asset_type === "lora") {
    const weight = document.createElement("span");
    weight.className = "default-asset-weight";
    weight.textContent = Number(asset.weight ?? 1).toFixed(2);
    meta.append(weight);
  }

  row.append(identity, meta);
  return row;
}

function renderList(container, assets, emptyMessage) {
  if (!container) return;
  container.replaceChildren();
  if (!assets.length) {
    container.append(createEmptyState(emptyMessage));
    return;
  }
  assets.forEach((asset) => container.append(createDefaultAssetChip(asset, { compact: true })));
}

function activeAssetRow(asset, onChange, onRemove) {
  const row = document.createElement("article");
  row.className = "active-lora-row";
  row.dataset.assetIdentity = assetIdentity(asset);

  const avatar = document.createElement("span");
  avatar.className = "active-asset-avatar";
  const previewUrl = String(asset.preview_url || "").trim();
  if (previewUrl) {
    const image = document.createElement("img");
    image.src = previewUrl;
    image.alt = "";
    image.loading = "lazy";
    image.addEventListener("error", () => {
      avatar.replaceChildren(document.createTextNode(String(asset.name || "L").slice(0, 2).toUpperCase()));
    });
    avatar.append(image);
  } else {
    avatar.textContent = String(asset.name || "L").slice(0, 2).toUpperCase();
  }

  const name = document.createElement("div");
  name.className = "active-asset-name";
  const strong = document.createElement("strong");
  strong.textContent = asset.name || "Unnamed LoRA";
  const scope = createScopeBadge(asset.source || asset.source_scope || "global_default");
  name.append(strong, scope);

  const enabled = document.createElement("label");
  enabled.className = "asset-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = asset.enabled !== false;
  const sliderVisual = document.createElement("span");
  enabled.append(checkbox, sliderVisual);

  const weightWrap = document.createElement("label");
  weightWrap.className = "active-asset-weight-control";
  const weightLabel = document.createElement("span");
  weightLabel.textContent = "Weight";
  const range = document.createElement("input");
  range.type = "range";
  range.min = "-2";
  range.max = "2";
  range.step = "0.05";
  range.value = String(asset.weight ?? 1);
  const number = document.createElement("input");
  number.type = "number";
  number.min = "-4";
  number.max = "4";
  number.step = "0.05";
  number.value = Number(asset.weight ?? 1).toFixed(2);
  weightWrap.append(weightLabel, range, number);

  const remove = document.createElement("button");
  remove.className = "icon-button active-asset-remove";
  remove.type = "button";
  remove.title = `Remove ${asset.name || "LoRA"}`;
  remove.textContent = "×";

  const commitWeight = (value) => {
    const parsed = Math.max(-4, Math.min(4, Number(value) || 0));
    asset.weight = parsed;
    range.value = String(Math.max(-2, Math.min(2, parsed)));
    number.value = parsed.toFixed(2);
    onChange();
  };
  range.addEventListener("input", () => commitWeight(range.value));
  number.addEventListener("change", () => commitWeight(number.value));
  checkbox.addEventListener("change", () => {
    asset.enabled = checkbox.checked;
    onChange();
  });
  remove.addEventListener("click", () => onRemove());

  row.append(avatar, name, enabled, weightWrap, remove);
  return row;
}

function textualInversionChip(asset, onRemove) {
  const chip = document.createElement("span");
  chip.className = `active-ti-chip is-${asset.polarity || "positive"}`;
  const label = document.createElement("span");
  label.textContent = `${asset.polarity === "negative" ? "Negative TI" : "TI"}: ${asset.name || asset.activation_text || "Unnamed"}`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.title = "Remove textual inversion";
  remove.textContent = "×";
  remove.addEventListener("click", onRemove);
  chip.append(label, remove);
  return chip;
}

function profileForScope(profiles, scope, activeModel) {
  if (scope === "global") return profiles.global_profile;
  const key = String(activeModel?.model_key || "");
  if (!key) return null;
  if (!profiles.model_profiles[key]) {
    profiles.model_profiles[key] = {
      profile_id: key,
      display_name: activeModel.model_name || "Current model",
      model_key: key,
      model_path: activeModel.model_path || "",
      model_name: activeModel.model_name || "",
      model_family: activeModel.model_family || "",
      assets: [],
    };
  }
  return profiles.model_profiles[key];
}

export function bindDefaultAssets(initialPayload = {}) {
  let payload = normalizedPayload(initialPayload);
  let workingProfiles = clone(payload.profiles);
  let activeAssets = Array.isArray(state.activePromptAssets) ? clone(state.activePromptAssets) : [];
  let editorScope = "global";
  let selectedCatalogAsset = null;

  state.defaultAssets = payload;
  state.activePromptAssets = activeAssets;

  const normalizedCatalogPath = (value) => String(value || "").trim().replaceAll("\\", "/").toLowerCase();
  const catalogAssetFor = (asset = {}) => {
    const catalog = asset.asset_type === "textual_inversion" ? state.textualInversions : state.loras;
    const catalogId = String(asset.catalog_asset_id || "");
    const path = normalizedCatalogPath(asset.path);
    return (catalog || []).find((item) => (
      (catalogId && String(item.asset_id || "") === catalogId)
      || (path && normalizedCatalogPath(item.path) === path)
    )) || null;
  };
  const enrichActiveAsset = (asset = {}) => {
    const catalog = catalogAssetFor(asset);
    return {
      ...clone(catalog || {}),
      ...clone(asset),
      catalog_asset_id: asset.catalog_asset_id || catalog?.asset_id || "",
      preview_url: asset.preview_url || catalog?.preview_url || "",
      preview_path: asset.preview_path || catalog?.preview_path || "",
      source_url: asset.source_url || catalog?.source_url || "",
      activation_text: asset.activation_text || catalog?.activation_text || "",
      model_family: asset.model_family || catalog?.model_family || "",
      weight: Number(asset.weight ?? catalog?.preferred_weight ?? 1),
      source: contractSource(asset.source || asset.source_scope || "visual_selection"),
      original_source: asset.original_source || "",
      enabled: asset.enabled !== false,
    };
  };
  const emitActiveAssets = () => {
    activeAssets = activeAssets.map(enrichActiveAsset);
    state.activePromptAssets = activeAssets;
    window.dispatchEvent(new CustomEvent("image-gen-active-prompt-assets-updated", {
      detail: { assets: clone(activeAssets) },
    }));
  };

  const refreshDrawer = () => {
    const effective = payload.effective_assets || [];
    const positiveLoras = effective.filter((item) => item.asset_type === "lora" && item.polarity !== "negative");
    const negativeLoras = effective.filter((item) => item.asset_type === "lora" && item.polarity === "negative");
    const textual = effective.filter((item) => item.asset_type === "textual_inversion");
    renderList($("#startupDefaultPositiveLoras"), positiveLoras, "No positive LoRA defaults configured.");
    renderList($("#startupDefaultNegativeLoras"), negativeLoras, "No negative LoRA defaults configured.");
    renderList($("#startupDefaultTextualInversions"), textual, "No textual-inversion defaults configured.");
    if ($("#startupDefaultPositiveLorasCount")) $("#startupDefaultPositiveLorasCount").textContent = String(positiveLoras.length);
    if ($("#startupDefaultNegativeLorasCount")) $("#startupDefaultNegativeLorasCount").textContent = String(negativeLoras.length);
    if ($("#startupDefaultTiCount")) $("#startupDefaultTiCount").textContent = String(textual.length);
    if ($("#startupDefaultsCount")) $("#startupDefaultsCount").textContent = String(payload.counts.total || 0);
    if ($("#applySavedDefaults")) $("#applySavedDefaults").checked = payload.apply_saved_defaults;
    if ($("#autoApplyDefaultsOnModelLoad")) $("#autoApplyDefaultsOnModelLoad").checked = payload.auto_apply_on_model_load;

    const modelName = payload.active_model?.model_name || "No active model";
    const modelProfile = payload.model_profile ? ` + ${modelName}` : "";
    if ($("#startupDefaultsSource")) $("#startupDefaultsSource").textContent = `Global profile${modelProfile} / user config`;
    if ($("#startupDefaultCompatibilityCount")) $("#startupDefaultCompatibilityCount").textContent = String(payload.counts.incompatible || 0);
    if ($("#startupDefaultsCompatibility")) {
      $("#startupDefaultsCompatibility").textContent = payload.incompatible_assets.length
        ? `${payload.incompatible_assets.length} saved asset(s) were skipped because they target a different model family.`
        : `No compatibility conflicts detected for ${modelName}.`;
      $("#startupDefaultsCompatibility").classList.toggle("is-warning", payload.incompatible_assets.length > 0);
    }
  };

  const renderActiveAssets = () => {
    const loras = activeAssets.filter((item) => item.asset_type === "lora");
    const textual = activeAssets.filter((item) => item.asset_type === "textual_inversion");
    const loraList = $("#activeLoraList");
    const tiList = $("#activeTextualInversionList");
    loraList?.replaceChildren();
    tiList?.replaceChildren();

    if (!loras.length) loraList?.append(createEmptyState("No LoRAs are staged for the current prompt."));
    loras.forEach((asset) => {
      loraList?.append(activeAssetRow(asset, () => {
        emitActiveAssets();
        renderActiveAssets();
      }, () => {
        activeAssets = activeAssets.filter((candidate) => candidate !== asset);
        emitActiveAssets();
        renderActiveAssets();
      }));
    });

    if (!textual.length) tiList?.append(createEmptyState("No textual inversions are staged for the current prompt."));
    textual.forEach((asset) => {
      tiList?.append(textualInversionChip(asset, () => {
        activeAssets = activeAssets.filter((candidate) => candidate !== asset);
        emitActiveAssets();
        renderActiveAssets();
      }));
    });
    if ($("#activeLorasCount")) $("#activeLorasCount").textContent = String(loras.length);
    if ($("#activeTextualInversionsCount")) $("#activeTextualInversionsCount").textContent = String(textual.length);
    if ($("#activePromptAssetsSource")) {
      $("#activePromptAssetsSource").textContent = activeAssets.length
        ? `${activeAssets.length} structured asset(s) staged`
        : "Visual prompt assets";
    }
  };

  const loadEffectiveDefaults = ({ announce = true } = {}) => {
    const explicit = activeAssets.filter((item) => ["visual_selection", "inline_syntax", "api_request", "imported"].includes(
      contractSource(item.source || item.source_scope),
    ));
    const merged = new Map(explicit.map((item) => [assetIdentity(item), item]));
    payload.effective_assets.forEach((item) => {
      const key = assetIdentity(item);
      if (!merged.has(key)) merged.set(key, enrichActiveAsset({
        ...clone(item),
        source: contractSource(item.source || item.source_scope || "global"),
        enabled: item.enabled !== false,
      }));
    });
    activeAssets = [...merged.values()].map(enrichActiveAsset);
    emitActiveAssets();
    renderActiveAssets();
    if ($("#activePromptAssetsStatus")) {
      $("#activePromptAssetsStatus").textContent = payload.effective_assets.length
        ? `${payload.effective_assets.length} compatible saved default(s) staged in the structured generation request.`
        : "No compatible saved defaults were available for the active checkpoint.";
    }
    if (announce) notify(`${payload.effective_assets.length} saved default asset(s) staged for the current prompt.`);
  };

  const removeDefaultSourcedAssets = () => {
    activeAssets = activeAssets.filter((item) => ["visual_selection", "inline_syntax", "api_request", "imported"].includes(
      contractSource(item.source || item.source_scope),
    ));
    emitActiveAssets();
    renderActiveAssets();
  };

  const updatePayload = (next) => {
    payload = normalizedPayload(next);
    state.defaultAssets = payload;
    workingProfiles = clone(payload.profiles);
    refreshDrawer();
  };

  const saveProfiles = async (profiles = workingProfiles) => {
    const saved = await api.saveDefaultAssets(profiles);
    updatePayload(saved);
    window.dispatchEvent(new CustomEvent("image-gen-default-assets-updated", { detail: clone(saved) }));
    return saved;
  };

  const currentEditorProfile = () => profileForScope(workingProfiles, editorScope, payload.active_model);

  const catalogForType = (assetType = $("#defaultAssetType")?.value || "lora") => (
    assetType === "textual_inversion" ? state.textualInversions : state.loras
  );

  const selectCatalogAsset = (asset) => {
    selectedCatalogAsset = asset || null;
    $("#defaultAssetCatalogId").value = asset?.asset_id || "";
    if (!asset) {
      $("#defaultAssetPickerStatus").textContent = "Select an installed asset card to populate the default entry.";
      renderAssetPicker();
      return;
    }
    $("#defaultAssetName").value = asset.name || asset.display_name || "";
    $("#defaultAssetPath").value = asset.path || "";
    $("#defaultAssetWeight").value = Number(asset.preferred_weight ?? 1).toFixed(2);
    $("#defaultAssetActivationText").value = asset.activation_text || "";
    $("#defaultAssetModelFamily").value = asset.model_family || payload.active_model?.model_family || "";
    $("#defaultAssetSourceUrl").value = asset.source_url || "";
    $("#defaultAssetNotes").value = asset.notes || "";
    $("#defaultAssetPickerStatus").textContent = `Selected installed asset: ${asset.name || asset.filename || "asset"}`;
    renderAssetPicker();
  };

  const renderAssetPicker = () => {
    const grid = $("#defaultAssetPickerGrid");
    if (!grid) return;
    const query = String($("#defaultAssetCatalogSearch")?.value || "").trim().toLowerCase();
    const assets = (catalogForType() || []).filter((asset) => {
      if (!query) return true;
      return [asset.name, asset.filename, asset.path, asset.activation_text, ...(asset.tags || [])]
        .join(" ").toLowerCase().includes(query);
    });
    grid.replaceChildren();
    $("#defaultAssetCatalogCount").textContent = String(assets.length);
    if (!assets.length) {
      grid.append(createEmptyState(
        $("#defaultAssetType")?.value === "lora"
          ? "No installed LoRAs match this search. Place LoRA files in the configured LoRA folder and refresh."
          : "No installed textual inversions match this search."
      ));
      return;
    }
    assets.forEach((asset) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "default-asset-picker-card";
      card.classList.toggle("is-selected", selectedCatalogAsset?.asset_id === asset.asset_id);
      if (asset.preview_url) {
        const image = document.createElement("img");
        image.src = asset.preview_url;
        image.alt = `${asset.name || "Asset"} preview`;
        image.loading = "lazy";
        image.addEventListener("error", () => {
          const fallback = document.createElement("div");
          fallback.className = "default-asset-picker-fallback";
          fallback.textContent = String(asset.name || "Asset").slice(0, 2).toUpperCase();
          image.replaceWith(fallback);
        });
        card.append(image);
      } else {
        const fallback = document.createElement("div");
        fallback.className = "default-asset-picker-fallback";
        fallback.textContent = String(asset.name || "Asset").slice(0, 2).toUpperCase();
        card.append(fallback);
      }
      const label = document.createElement("span");
      label.textContent = asset.name || asset.filename || "Installed asset";
      card.append(label);
      card.addEventListener("click", () => selectCatalogAsset(asset));
      grid.append(card);
    });
  };

  const resetEntryForm = () => {
    selectedCatalogAsset = null;
    $("#defaultAssetEditingId").value = "";
    $("#defaultAssetCatalogId").value = "";
    $("#defaultAssetType").value = "lora";
    $("#defaultAssetPolarity").value = "positive";
    $("#defaultAssetName").value = "";
    $("#defaultAssetWeight").value = "1.00";
    $("#defaultAssetPath").value = "";
    $("#defaultAssetActivationText").value = "";
    $("#defaultAssetModelFamily").value = payload.active_model?.model_family || "";
    $("#defaultAssetSourceUrl").value = "";
    $("#defaultAssetNotes").value = "";
    $("#defaultAssetEnabled").checked = true;
    $("#saveDefaultAssetEntryButton").textContent = "Add Asset";
    if ($("#defaultAssetCatalogSearch")) $("#defaultAssetCatalogSearch").value = "";
    renderAssetPicker();
  };

  const readEntryForm = () => ({
    asset_id: String($("#defaultAssetEditingId").value || "").trim(),
    asset_type: $("#defaultAssetType").value,
    polarity: $("#defaultAssetPolarity").value,
    name: String($("#defaultAssetName").value || "").trim(),
    weight: Number($("#defaultAssetWeight").value || 1),
    path: String($("#defaultAssetPath").value || "").trim(),
    activation_text: String($("#defaultAssetActivationText").value || "").trim(),
    model_family: $("#defaultAssetModelFamily").value,
    source_url: String($("#defaultAssetSourceUrl").value || "").trim(),
    notes: String($("#defaultAssetNotes").value || "").trim(),
    preview_path: selectedCatalogAsset?.preview_path || "",
    catalog_asset_id: String($("#defaultAssetCatalogId").value || "").trim(),
    enabled: $("#defaultAssetEnabled").checked,
  });

  const editEntry = (asset) => {
    const catalog = asset.asset_type === "textual_inversion" ? state.textualInversions : state.loras;
    selectedCatalogAsset = (catalog || []).find((item) => (
      String(item.path || "").replaceAll("\\", "/").toLowerCase()
      === String(asset.path || "").replaceAll("\\", "/").toLowerCase()
    )) || null;
    $("#defaultAssetCatalogId").value = selectedCatalogAsset?.asset_id || "";
    $("#defaultAssetEditingId").value = asset.asset_id || "";
    $("#defaultAssetType").value = asset.asset_type || "lora";
    $("#defaultAssetPolarity").value = asset.polarity || "positive";
    $("#defaultAssetName").value = asset.name || "";
    $("#defaultAssetWeight").value = Number(asset.weight ?? 1).toFixed(2);
    $("#defaultAssetPath").value = asset.path || "";
    $("#defaultAssetActivationText").value = asset.activation_text || "";
    $("#defaultAssetModelFamily").value = asset.model_family || "";
    $("#defaultAssetSourceUrl").value = asset.source_url || "";
    $("#defaultAssetNotes").value = asset.notes || "";
    $("#defaultAssetEnabled").checked = asset.enabled !== false;
    $("#saveDefaultAssetEntryButton").textContent = "Update Asset";
    renderAssetPicker();
    $("#defaultAssetName").focus();
  };

  const renderEditorList = () => {
    const profile = currentEditorProfile();
    const list = $("#defaultAssetEditorList");
    list?.replaceChildren();
    const assets = profile?.assets || [];
    if ($("#defaultAssetEditorCount")) $("#defaultAssetEditorCount").textContent = String(assets.length);
    if ($("#defaultAssetListTitle")) {
      $("#defaultAssetListTitle").textContent = editorScope === "model"
        ? `${payload.active_model?.model_name || "Current model"} defaults`
        : "Global defaults";
    }
    if (!assets.length) {
      list?.append(createEmptyState("No assets have been saved in this profile."));
      return;
    }
    assets.forEach((asset) => {
      const row = createDefaultAssetChip({ ...asset, source_scope: editorScope }, { compact: false });
      row.classList.toggle("is-disabled", asset.enabled === false);
      const actions = document.createElement("div");
      actions.className = "default-asset-editor-actions";
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "secondary-button compact-button";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => editEntry(asset));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary-button compact-button danger-button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        profile.assets = profile.assets.filter((candidate) => candidate !== asset);
        renderEditorList();
      });
      actions.append(edit, remove);
      row.append(actions);
      list?.append(row);
    });
  };

  const openEditor = ({ type = "" } = {}) => {
    workingProfiles = clone(payload.profiles);
    editorScope = "global";
    $("#defaultAssetScope").value = editorScope;
    const modelOption = [...$("#defaultAssetScope").options].find((item) => item.value === "model");
    if (modelOption) modelOption.disabled = !payload.active_model?.model_key;
    if ($("#defaultAssetsModelSummary")) {
      $("#defaultAssetsModelSummary").textContent = payload.active_model?.model_key
        ? `${payload.active_model.model_name || "Current model"} · ${payload.active_model.model_family || "family unknown"}`
        : "No current checkpoint selected; model-specific scope is unavailable.";
    }
    resetEntryForm();
    if (type) $("#defaultAssetType").value = type;
    renderAssetPicker();
    renderEditorList();
    const dialog = $("#defaultAssetsDialog");
    if (!dialog.open) dialog.showModal();
  };

  const addActiveAsset = (asset, { sourceScope = "visual" } = {}) => {
    const candidate = enrichActiveAsset({
      ...clone(asset),
      asset_type: asset.asset_type || "lora",
      polarity: asset.polarity || "positive",
      source_scope: sourceScope,
      source: contractSource(asset.source || sourceScope),
      original_source: asset.original_source || "",
      enabled: asset.enabled !== false,
    });
    const key = assetIdentity(candidate);
    const index = activeAssets.findIndex((item) => assetIdentity(item) === key);
    if (index >= 0) activeAssets[index] = { ...activeAssets[index], ...candidate };
    else activeAssets.push(candidate);
    emitActiveAssets();
    renderActiveAssets();
    return clone(candidate);
  };

  const updateActiveAsset = (identity, updates = {}) => {
    const index = activeAssets.findIndex((item) => assetIdentity(item) === identity || item.asset_id === identity || item.catalog_asset_id === identity);
    if (index < 0) return null;
    activeAssets[index] = enrichActiveAsset({ ...activeAssets[index], ...clone(updates) });
    emitActiveAssets();
    renderActiveAssets();
    return clone(activeAssets[index]);
  };

  const removeActiveAsset = (identity) => {
    const before = activeAssets.length;
    activeAssets = activeAssets.filter((item) => assetIdentity(item) !== identity && item.asset_id !== identity && item.catalog_asset_id !== identity);
    if (activeAssets.length === before) return false;
    emitActiveAssets();
    renderActiveAssets();
    return true;
  };

  const clearActiveAssets = () => {
    activeAssets = [];
    emitActiveAssets();
    renderActiveAssets();
  };

  const setApplySavedDefaults = async (enabled) => {
    const next = clone(payload.profiles);
    next.apply_saved_defaults = Boolean(enabled);
    const saved = await saveProfiles(next);
    if (saved.apply_saved_defaults) loadEffectiveDefaults({ announce: false });
    else removeDefaultSourcedAssets();
    return clone(payload);
  };

  const setAutoApplyOnModelLoad = async (enabled) => {
    const next = clone(payload.profiles);
    next.auto_apply_on_model_load = Boolean(enabled);
    await saveProfiles(next);
    return clone(payload);
  };

  const saveActiveAssetsAsDefaults = async (scope = "model") => {
    const next = clone(payload.profiles);
    const resolvedScope = scope === "global" ? "global" : "model";
    const profile = profileForScope(next, resolvedScope, payload.active_model);
    if (!profile) throw new Error("Activate a checkpoint before saving model-specific defaults.");
    profile.assets = activeAssets.map((item) => ({
      asset_id: item.asset_id || "",
      asset_type: item.asset_type || "lora",
      polarity: item.polarity || "positive",
      name: item.name || "",
      path: item.path || "",
      weight: Number(item.weight ?? 1),
      enabled: item.enabled !== false,
      activation_text: item.activation_text || "",
      model_family: item.model_family || "",
      source_url: item.source_url || "",
      preview_path: item.preview_path || "",
      catalog_asset_id: item.catalog_asset_id || "",
      notes: item.notes || "",
    }));
    await saveProfiles(next);
    return clone(payload);
  };

  $("#applySavedDefaults")?.addEventListener("change", async (event) => {
    try {
      await setApplySavedDefaults(event.target.checked);
    } catch (error) {
      event.target.checked = !event.target.checked;
      notify(`Unable to save default behavior: ${error.message}`, "error");
    }
  });
  $("#autoApplyDefaultsOnModelLoad")?.addEventListener("change", async (event) => {
    try {
      await setAutoApplyOnModelLoad(event.target.checked);
    } catch (error) {
      event.target.checked = !event.target.checked;
      notify(`Unable to save model-load behavior: ${error.message}`, "error");
    }
  });
  $("#loadStartupDefaultsButton")?.addEventListener("click", () => loadEffectiveDefaults());
  $("#reloadActiveDefaultsButton")?.addEventListener("click", () => loadEffectiveDefaults());
  $("#clearActivePromptAssetsButton")?.addEventListener("click", () => {
    activeAssets = [];
    emitActiveAssets();
    renderActiveAssets();
  });
  $("#editStartupDefaultsButton")?.addEventListener("click", () => openEditor());
  $("#addPromptLoraButton")?.addEventListener("click", () => openEditor({ type: "lora" }));

  $("#defaultAssetType")?.addEventListener("change", () => {
    selectedCatalogAsset = null;
    $("#defaultAssetCatalogId").value = "";
    $("#defaultAssetPath").value = "";
    renderAssetPicker();
  });
  $("#defaultAssetCatalogSearch")?.addEventListener("input", renderAssetPicker);
  $("#refreshDefaultAssetCatalogButton")?.addEventListener("click", async () => {
    try {
      const catalogs = await api.refreshModels();
      state.loras = catalogs.loras || [];
      state.textualInversions = catalogs.textual_inversions || [];
      renderAssetPicker();
      notify(`Installed asset catalog refreshed: ${state.loras.length} LoRA(s), ${state.textualInversions.length} textual inversion(s).`);
    } catch (error) {
      notify(`Unable to refresh installed assets: ${error.message}`, "error");
    }
  });

  $("#defaultAssetScope")?.addEventListener("change", (event) => {
    editorScope = event.target.value;
    resetEntryForm();
    renderEditorList();
  });
  $("#saveDefaultAssetEntryButton")?.addEventListener("click", () => {
    const profile = currentEditorProfile();
    if (!profile) {
      notify("Select and activate a checkpoint before editing model-specific defaults.", "error");
      return;
    }
    const entry = readEntryForm();
    if (!entry.path) {
      notify("Choose an installed asset from the image list before adding it to defaults.", "error");
      return;
    }
    const editingId = entry.asset_id;
    if (editingId) {
      const index = profile.assets.findIndex((item) => item.asset_id === editingId);
      if (index >= 0) profile.assets[index] = entry;
      else profile.assets.push(entry);
    } else {
      profile.assets.push(entry);
    }
    resetEntryForm();
    renderEditorList();
  });
  $("#clearDefaultAssetEntryButton")?.addEventListener("click", resetEntryForm);
  $("#closeDefaultAssetsDialogButton")?.addEventListener("click", () => $("#defaultAssetsDialog").close());
  $("#cancelDefaultAssetsDialogButton")?.addEventListener("click", () => $("#defaultAssetsDialog").close());
  $("#saveDefaultAssetsDialogButton")?.addEventListener("click", async () => {
    try {
      workingProfiles.apply_saved_defaults = payload.apply_saved_defaults;
      workingProfiles.auto_apply_on_model_load = payload.auto_apply_on_model_load;
      await saveProfiles(workingProfiles);
      $("#defaultAssetsDialog").close();
      notify("Saved default asset profiles.");
      if (payload.apply_saved_defaults) loadEffectiveDefaults({ announce: false });
    } catch (error) {
      notify(`Unable to save default assets: ${error.message}`, "error");
    }
  });

  window.addEventListener("image-gen-asset-catalog-refreshed", (event) => {
    state.loras = event.detail?.loras || state.loras;
    state.textualInversions = event.detail?.textual_inversions || state.textualInversions;
    renderAssetPicker();
  });
  window.addEventListener("image-gen-default-assets-updated", (event) => {
    if (event.detail) updatePayload(event.detail);
  });

  window.addEventListener("image-gen-model-activated", async (event) => {
    try {
      const next = event.detail?.defaultAssets || await api.defaultAssets();
      updatePayload(next);
      window.dispatchEvent(new CustomEvent("image-gen-default-assets-updated", { detail: clone(next) }));
      if (payload.apply_saved_defaults && payload.auto_apply_on_model_load) loadEffectiveDefaults({ announce: false });
    } catch (error) {
      notify(`Unable to refresh model-specific defaults: ${error.message}`, "error");
    }
  });

  refreshDrawer();
  renderActiveAssets();
  if (payload.apply_saved_defaults && payload.effective_assets.length) loadEffectiveDefaults({ announce: false });

  emitActiveAssets();

  return {
    current: () => clone(payload),
    activeAssets: () => clone(activeAssets),
    profiles: () => clone(payload.profiles),
    refresh: async () => updatePayload(await api.defaultAssets()),
    loadEffectiveDefaults,
    openEditor,
    addActiveAsset,
    updateActiveAsset,
    removeActiveAsset,
    clearActiveAssets,
    setApplySavedDefaults,
    setAutoApplyOnModelLoad,
    saveActiveAssetsAsDefaults,
    renderActiveAssets,
  };
}
