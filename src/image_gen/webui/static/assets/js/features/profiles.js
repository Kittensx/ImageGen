import { api } from "../api.js";
import { $, option, replaceOptions, notify } from "../utils.js";

let generationProfiles = [];

function profileFileIdentity(item) {
  return String(item?.file_name || "").trim();
}

function profileDisplayName(item) {
  const fileName = profileFileIdentity(item);
  const stem = fileName.replace(/\.json$/i, "");
  return String(item?.name || stem || "Untitled").trim() || "Untitled";
}

export function renderGenerationProfiles(values) {
  generationProfiles = [...(values || [])].sort((left, right) => {
    const nameOrder = profileDisplayName(left).localeCompare(profileDisplayName(right), undefined, { sensitivity: "base" });
    if (nameOrder) return nameOrder;
    return profileFileIdentity(left).localeCompare(profileFileIdentity(right), undefined, { sensitivity: "base" });
  });
  const displayCounts = new Map();
  generationProfiles.forEach((item) => {
    const key = profileDisplayName(item).toLocaleLowerCase();
    displayCounts.set(key, (displayCounts.get(key) || 0) + 1);
  });
  replaceOptions($("#generationProfile"), [
    option("", "Current / last session"),
    ...generationProfiles.map((item) => {
      const fileName = profileFileIdentity(item);
      const displayName = profileDisplayName(item);
      const duplicate = (displayCounts.get(displayName.toLocaleLowerCase()) || 0) > 1;
      const label = duplicate && fileName ? `${displayName} — ${fileName}` : displayName;
      return option(fileName || displayName, label);
    }),
  ]);
}

export async function reloadGenerationProfiles({ selectFileName = "" } = {}) {
  generationProfiles = await api.profiles("generation");
  renderGenerationProfiles(generationProfiles);
  if (selectFileName) $("#generationProfile").value = selectFileName;
  return generationProfiles;
}

export async function saveGenerationProfileValues(values, {
  namePrompt = "Name this generation profile:",
  defaultName = "",
  successMessage = "Generation profile saved.",
  selectSaved = true,
} = {}) {
  const name = window.prompt(namePrompt, defaultName);
  if (!name) return false;
  const saved = await api.saveProfile("generation", { name, values });
  await reloadGenerationProfiles({ selectFileName: selectSaved ? profileFileIdentity(saved) : "" });
  notify(successMessage);
  return true;
}

export function bindGenerationProfiles({ collect, apply }) {
  $("#generationProfile").addEventListener("change", () => {
    const selectedIdentity = $("#generationProfile").value;
    const selected = generationProfiles.find((item) => (profileFileIdentity(item) || profileDisplayName(item)) === selectedIdentity);
    if (selected) apply(selected.values || {});
  });

  $("#saveGenerationProfileButton").addEventListener("click", async () => {
    try {
      await saveGenerationProfileValues(collect());
    } catch (error) {
      notify(error.message, "error");
    }
  });
}
