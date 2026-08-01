import { api } from "../api.js";
import { $, option, replaceOptions, notify } from "../utils.js";

let presets = [];

export function renderPromptPresets(values) {
  presets = values || [];
  replaceOptions($("#promptPreset"), [
    option("", "Choose a prompt preset…"),
    ...presets.map((item) => option(item.name, item.name)),
  ]);
}

export function bindPromptPresets(onPromptsChanged) {
  $("#savePromptPresetButton").addEventListener("click", async () => {
    const name = window.prompt("Name this positive + negative prompt preset:");
    if (!name) return;
    try {
      await api.savePromptPreset({
        name,
        positive_prompt: $("#positivePrompt").value,
        negative_prompt: $("#negativePrompt").value,
        notes: "",
        tags: [],
      });
      renderPromptPresets(await api.promptPresets());
      $("#promptPreset").value = name;
      notify("Prompt preset saved.");
    } catch (error) {
      notify(error.message, "error");
    }
  });

  $("#loadPromptPresetButton").addEventListener("click", () => {
    const selected = presets.find((item) => item.name === $("#promptPreset").value);
    if (!selected) return;
    $("#positivePrompt").value = selected.positive_prompt || "";
    $("#negativePrompt").value = selected.negative_prompt || "";
    onPromptsChanged();
  });
}
