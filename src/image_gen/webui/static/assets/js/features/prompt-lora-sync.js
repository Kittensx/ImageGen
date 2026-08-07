import { state } from "../state.js";
import { $, notify } from "../utils.js";

const LORA_TOKEN_RE = /<lora:([^:>]+?)(?::([-+]?\d*\.?\d+))?>/gi;
const PROMPT_FIELDS = [
  { id: "positivePrompt", polarity: "positive" },
  { id: "negativePrompt", polarity: "negative" },
  { id: "hiresPositivePrompt", polarity: "positive" },
  { id: "hiresNegativePrompt", polarity: "negative" },
];

function clone(value) {
  return JSON.parse(JSON.stringify(value ?? null));
}

function sourceKey(value) {
  return String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
}

function normalizedText(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizedPath(value) {
  return String(value || "").trim().replaceAll("\\", "/").toLowerCase();
}

function modelKey(model = {}) {
  return {
    name: normalizedText(model.name || model.display_name || ""),
    stem: normalizedText(String(model.filename || model.path || model.name || "").split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || ""),
    path: normalizedPath(model.path || ""),
    assetId: normalizedText(model.asset_id || ""),
  };
}

function assetMatchesToken(asset = {}, token = {}) {
  const assetPath = normalizedPath(asset.path || asset.resolved_path || "");
  const assetName = normalizedText(asset.name || asset.requested_name || "");
  const assetStem = normalizedText(String(asset.path || asset.resolved_path || asset.name || "").split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "");
  const tokenName = normalizedText(token.name || token.token_name || "");
  if (!tokenName) return false;
  if (assetName && assetName === tokenName) return true;
  if (assetStem && assetStem === tokenName) return true;
  if (assetPath) {
    const tail = assetPath.split("/").pop() || "";
    if (tail === tokenName || tail.replace(/\.[^.]+$/, "") === tokenName) return true;
  }
  return false;
}

function catalogMatchForToken(token = {}) {
  const tokenName = normalizedText(token.name || "");
  if (!tokenName) return null;
  const catalog = Array.isArray(state.loras) ? state.loras : [];
  return catalog.find((model) => {
    const key = modelKey(model);
    return [key.name, key.stem, key.path.split("/").pop() || ""].includes(tokenName);
  }) || null;
}

function promptFieldEntries() {
  return PROMPT_FIELDS.map((entry) => ({ ...entry, element: $(`#${entry.id}`) })).filter((entry) => entry.element);
}

function dispatchPromptMutation(element) {
  if (!element) return;
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function normalizePromptSpacing(text) {
  return String(text || "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+,/g, ",")
    .replace(/,\s*,+/g, ", ")
    .replace(/(^|\n)\s*,\s*/g, "$1")
    .replace(/\s*,\s*($|\n)/g, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function appendPromptFragment(text, fragment) {
  const base = String(text || "").trim();
  const extra = String(fragment || "").trim();
  if (!extra) return base;
  if (!base) return extra;
  if (base.includes("\n")) return `${base}\n${extra}`;
  return `${base.replace(/[\s,]+$/g, "")}, ${extra}`;
}

function parsePromptTokens() {
  const tokens = [];
  promptFieldEntries().forEach(({ id, polarity, element }) => {
    const text = String(element.value || "");
    LORA_TOKEN_RE.lastIndex = 0;
    let match;
    while ((match = LORA_TOKEN_RE.exec(text)) !== null) {
      tokens.push({
        fieldId: id,
        polarity,
        name: String(match[1] || "").trim(),
        weight: Number(match[2] ?? 1) || 1,
        tokenText: match[0],
        start: match.index,
        end: match.index + match[0].length,
      });
    }
  });
  return tokens;
}

function promptAssetsFromTokens(tokens = []) {
  const deduped = new Map();
  tokens.forEach((token) => {
    const catalog = catalogMatchForToken(token);
    const identity = [
      token.polarity,
      normalizedText(catalog?.asset_id || ""),
      normalizedPath(catalog?.path || ""),
      normalizedText(token.name || ""),
    ].join("|");
    if (deduped.has(identity)) {
      const existing = deduped.get(identity);
      existing.prompt_fields = Array.from(new Set([...(existing.prompt_fields || []), token.fieldId]));
      return;
    }
    const entry = {
      asset_type: "lora",
      asset_id: catalog?.asset_id || "",
      catalog_asset_id: catalog?.asset_id || "",
      name: catalog?.name || catalog?.display_name || token.name,
      requested_name: token.name,
      token_name: token.name,
      path: catalog?.path || "",
      resolved_path: catalog?.path || "",
      weight: Number(token.weight ?? catalog?.preferred_weight ?? 1),
      enabled: true,
      polarity: token.polarity,
      activation_text: catalog?.activation_text || "",
      model_family: catalog?.model_family || catalog?.detected_model_family || "",
      source_url: catalog?.source_url || "",
      preview_path: catalog?.preview_path || "",
      preview_url: catalog?.preview_url || "",
      notes: catalog?.notes || "",
      source: "inline_syntax",
      source_scope: "inline",
      original_source: "",
      prompt_field: token.fieldId,
      prompt_fields: [token.fieldId],
    };
    deduped.set(identity, entry);
  });
  return [...deduped.values()];
}

export function bindPromptLoraSync({ defaultAssetsController } = {}) {
  let syncing = false;

  const preservedAssets = (current = []) => current.filter((item) => {
    if (String(item?.asset_type || "") !== "lora") return true;
    const source = sourceKey(item.source || item.source_scope);
    return !["inline", "inline_syntax", "visual", "visual_selection"].includes(source);
  });

  const syncFromPrompts = () => {
    if (syncing) return defaultAssetsController?.activeAssets?.() || clone(state.activePromptAssets || []);
    syncing = true;
    try {
      const tokens = parsePromptTokens();
      const promptLoras = promptAssetsFromTokens(tokens);
      const current = defaultAssetsController?.activeAssets?.() || clone(state.activePromptAssets || []) || [];
      const next = [...preservedAssets(current), ...promptLoras];
      defaultAssetsController?.setActiveAssets?.(next);
      return clone(next);
    } finally {
      syncing = false;
    }
  };

  const matchingTokens = (asset = {}) => parsePromptTokens().filter((token) => (
    token.polarity === String(asset.polarity || "positive").trim().toLowerCase()
      && assetMatchesToken(asset, token)
  ));

  const updatePromptElement = (fieldId, mutator) => {
    const element = $(`#${fieldId}`);
    if (!element) return false;
    const original = String(element.value || "");
    const updated = mutator(original);
    if (updated === original) return false;
    element.value = normalizePromptSpacing(updated);
    dispatchPromptMutation(element);
    return true;
  };

  const ensureActivationText = (text, activationText) => {
    const extra = String(activationText || "").trim();
    if (!extra) return text;
    return normalizedText(text).includes(normalizedText(extra)) ? text : appendPromptFragment(text, extra);
  };

  const insertLora = (model = {}, { fieldId = "positivePrompt", includeActivationText = true } = {}) => {
    const name = String(model.name || model.display_name || "").trim();
    if (!name) {
      notify("This LoRA is missing a name, so it could not be added to the prompt.", "error");
      return false;
    }
    const weight = Number(model.weight ?? model.preferred_weight ?? model.metadata?.preferred_weight ?? 1);
    const syntax = `<lora:${name}:${weight.toFixed(2)}>`;
    const activationText = includeActivationText ? String(model.activation_text || "").trim() : "";
    const existing = matchingTokens({ ...model, name, polarity: fieldId.toLowerCase().includes("negative") ? "negative" : "positive" });
    if (existing.length) {
      const touched = new Set();
      existing.forEach((token) => {
        if (touched.has(token.fieldId)) return;
        touched.add(token.fieldId);
        updatePromptElement(token.fieldId, (text) => {
          let changed = text;
          const fieldTokens = matchingTokens({ ...model, name, polarity: token.polarity }).filter((item) => item.fieldId === token.fieldId);
          fieldTokens.slice().reverse().forEach((item) => {
            changed = `${changed.slice(0, item.start)}<lora:${item.name}:${weight.toFixed(2)}>${changed.slice(item.end)}`;
          });
          if (token.fieldId === fieldId && activationText) changed = ensureActivationText(changed, activationText);
          return changed;
        });
      });
      syncFromPrompts();
      return true;
    }
    const touched = updatePromptElement(fieldId, (text) => {
      let next = appendPromptFragment(text, syntax);
      if (activationText) next = appendPromptFragment(next, activationText);
      return next;
    });
    if (touched) syncFromPrompts();
    return touched;
  };

  const removeLora = (asset = {}) => {
    const activationText = String(asset.activation_text || "").trim();
    let changedAny = false;
    promptFieldEntries().forEach(({ id }) => {
      const tokens = matchingTokens(asset).filter((token) => token.fieldId === id);
      if (!tokens.length) return;
      const changed = updatePromptElement(id, (text) => {
        let next = text;
        tokens.slice().reverse().forEach((token) => {
          let removeEnd = token.end;
          if (activationText) {
            const remainder = next.slice(removeEnd);
            const escaped = activationText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
            const match = remainder.match(new RegExp(`^(\\s*,?\\s*)${escaped}(?=(\\s*,|\\s*$|\\n))`, "i"));
            if (match) removeEnd += match[0].length;
          }
          next = `${next.slice(0, token.start)}${next.slice(removeEnd)}`;
        });
        return next;
      });
      changedAny = changedAny || changed;
    });
    if (changedAny) syncFromPrompts();
    return changedAny;
  };

  const updateLoraWeight = (asset = {}, weight = 1) => {
    const parsedWeight = Math.max(-4, Math.min(4, Number(weight) || 0));
    let changedAny = false;
    promptFieldEntries().forEach(({ id }) => {
      const tokens = matchingTokens(asset).filter((token) => token.fieldId === id);
      if (!tokens.length) return;
      const changed = updatePromptElement(id, (text) => {
        let next = text;
        tokens.slice().reverse().forEach((token) => {
          next = `${next.slice(0, token.start)}<lora:${token.name}:${parsedWeight.toFixed(2)}>${next.slice(token.end)}`;
        });
        return next;
      });
      changedAny = changedAny || changed;
    });
    if (changedAny) syncFromPrompts();
    return changedAny;
  };

  const setEnabled = (asset = {}, enabled = true) => {
    if (enabled) return false;
    return removeLora(asset);
  };

  const onPromptChanged = () => syncFromPrompts();
  promptFieldEntries().forEach(({ element }) => {
    element.addEventListener("input", onPromptChanged);
    element.addEventListener("change", onPromptChanged);
  });
  window.addEventListener("image-gen-asset-catalog-refreshed", onPromptChanged);

  const controller = {
    syncFromPrompts,
    insertLora,
    removeLora,
    updateLoraWeight,
    setEnabled,
    matchingTokens,
  };
  window.imageGenPromptLoraSync = controller;
  syncFromPrompts();
  return controller;
}
