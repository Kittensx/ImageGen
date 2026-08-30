import { state } from "../../state.js";
import { $ } from "../../utils.js";

export function option(value, label, selected = false, disabled = false) {
  const node = document.createElement("option");
  node.value = String(value ?? "");
  node.textContent = label;
  node.selected = selected;
  node.disabled = disabled;
  return node;
}

export function slug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "") || "custom_profile";
}

export function safeParseJson(value, fallback = {}) {
  try {
    const parsed = JSON.parse(String(value || ""));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

export function parserById(value) {
  const token = String(value || "legacy").toLowerCase();
  return state.promptParsers.find((item) => String(item.parser_id || "").toLowerCase() === token) || null;
}

export function profileById(value) {
  const token = String(value || "").toLowerCase();
  return state.promptShortcutProfiles.find((item) => String(item.profile_id || "").toLowerCase() === token) || null;
}

export function presetById(value) {
  const token = String(value || "").toLowerCase();
  return state.promptParserPresets.find((item) => [item.preset_id, item.name].some((candidate) => String(candidate || "").toLowerCase() === token)) || null;
}

export function profileSnapshot(profile) {
  if (!profile) return {};
  const keys = [
    "contract_version", "profile_schema_version", "profile_id", "label", "version", "aliases", "parser_emitters",
    "semantic_modes", "preprocessing", "precedence", "reserved_syntax", "migrated_from_contract",
    "compatible_parsers", "escape_character", "builtin", "credit", "description", "source",
    "palette", "mapping_hash",
  ];
  return Object.fromEntries(keys.filter((key) => key in profile).map((key) => [key, profile[key]]));
}

export function setSnapshot(profile) {
  const snapshot = profileSnapshot(profile);
  state.promptConfiguration.shortcutProfileSnapshot = snapshot;
  const input = $("#promptShortcutProfileSnapshot");
  if (input) input.value = JSON.stringify(snapshot);
}

export function setParserKwargs(values = {}) {
  state.promptConfiguration.parserKwargs = { ...(values || {}) };
  const input = $("#promptParserKwargs");
  if (input) input.value = JSON.stringify(state.promptConfiguration.parserKwargs);
  document.dispatchEvent(new CustomEvent("prompt-parser-options-changed", {
    detail: { parserId: currentParserId(), options: { ...state.promptConfiguration.parserKwargs } },
  }));
}

export function currentParserId() {
  return $("#promptParserName")?.value || "legacy";
}

export function currentProfile() {
  return profileById($("#promptShortcutProfileName")?.value);
}

export function compatibleProfiles(parserId) {
  const normalizedParser = String(parserId || "legacy").toLowerCase();
  return state.promptShortcutProfiles.filter((profile) => {
    const compatible = (profile.compatible_parsers || []).map((item) => String(item).toLowerCase());
    const combinedCompatible = normalizedParser === "combined" && compatible.some((item) => ["combined", "legacy", "parser21"].includes(item));
    return profile.valid !== false && (compatible.includes(normalizedParser) || combinedCompatible);
  });
}

export function defaultProfileId(parserId) {
  const normalized = String(parserId || "legacy").toLowerCase();
  if (normalized === "parser21") return "parser21_native";
  if (normalized === "superhybrid") return "superhybrid_native";
  if (normalized === "combined") return "canonical";
  return "legacy_default";
}
