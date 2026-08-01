import { api } from "../api.js";
import { $, option, replaceOptions, notify } from "../utils.js";

let generationProfiles = [];

export function renderGenerationProfiles(values) {
  generationProfiles = values || [];
  replaceOptions($("#generationProfile"), [
    option("", "Current / last session"),
    ...generationProfiles.map((item) => option(item.name, item.name)),
  ]);
}

export function bindGenerationProfiles({ collect, apply }) {
  $("#generationProfile").addEventListener("change", () => {
    const selected = generationProfiles.find((item) => item.name === $("#generationProfile").value);
    if (selected) apply(selected.values || {});
  });

  $("#saveGenerationProfileButton").addEventListener("click", async () => {
    const name = window.prompt("Name this generation profile:");
    if (!name) return;
    try {
      await api.saveProfile("generation", { name, values: collect() });
      generationProfiles = await api.profiles("generation");
      renderGenerationProfiles(generationProfiles);
      $("#generationProfile").value = name;
      notify("Generation profile saved.");
    } catch (error) {
      notify(error.message, "error");
    }
  });
}
