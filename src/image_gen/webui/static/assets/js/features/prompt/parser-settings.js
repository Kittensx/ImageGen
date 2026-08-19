import { $ } from "../../utils.js";
import { saveSessionSoon } from "./runtime.js";
import { currentParserId, option, parserById, safeParseJson, setParserKwargs } from "./shared.js";

export function parserSettingsSchema(parserId) {
  return parserById(parserId)?.settings_schema || { properties: {} };
}

export function parserDefaultOptions(parserId) {
  const properties = parserSettingsSchema(parserId).properties || {};
  return Object.fromEntries(Object.entries(properties)
    .filter(([, spec]) => Object.prototype.hasOwnProperty.call(spec || {}, "default"))
    .map(([key, spec]) => [key, structuredClone(spec.default)]));
}

export function renderParserSettings(containerSelector, parserId, values = {}, onChange = () => {}) {
  const container = $(containerSelector);
  if (!container) return;
  container.replaceChildren();
  const parser = parserById(parserId);
  const properties = parser?.settings_schema?.properties || {};
  const effective = { ...parserDefaultOptions(parserId), ...(values || {}) };
  if (!Object.keys(properties).length) {
    const message = document.createElement("small");
    message.textContent = "This parser has no request-scoped advanced settings.";
    container.append(message);
    return;
  }
  Object.entries(properties).forEach(([key, specValue]) => {
    const spec = specValue || {};
    const label = document.createElement("label");
    label.className = `prompt-parser-setting${spec.type === "boolean" ? " is-boolean" : ""}`;
    const title = document.createElement("span");
    title.textContent = spec.title || key;
    let input;
    if (Array.isArray(spec.enum)) {
      input = document.createElement("select");
      spec.enum.forEach((value) => input.append(option(value, String(value), effective[key] === value)));
    } else {
      input = document.createElement("input");
      if (spec.type === "boolean") {
        input.type = "checkbox";
        input.checked = Boolean(effective[key]);
      } else if (spec.type === "integer" || spec.type === "number") {
        input.type = "number";
        if (spec.minimum !== undefined) input.min = String(spec.minimum);
        if (spec.maximum !== undefined) input.max = String(spec.maximum);
        input.step = spec.type === "integer" ? "1" : String(spec.multipleOf || "any");
        input.value = effective[key] ?? "";
        if (spec.x_nullable) input.placeholder = "Inherit generation value";
      } else {
        input.type = "text";
        input.value = effective[key] ?? "";
      }
    }
    input.dataset.parserSetting = key;
    input.setAttribute("aria-label", spec.title || key);
    const commit = () => {
      const next = { ...effective };
      if (spec.type === "boolean") next[key] = input.checked;
      else if (input.value === "" && spec.x_nullable) delete next[key];
      else if (spec.type === "integer") next[key] = Number.parseInt(input.value, 10);
      else if (spec.type === "number") next[key] = Number.parseFloat(input.value);
      else next[key] = input.value;
      onChange(next);
    };
    input.addEventListener("change", commit);
    input.addEventListener("input", commit);
    label.append(title, input);
    if (spec.description) {
      const description = document.createElement("small");
      description.textContent = spec.description;
      label.append(description);
    }
    container.append(label);
  });
}

export function renderBaseParserSettings() {
  const parser = parserById(currentParserId());
  const warning = $("#promptParserExperimentalWarning");
  if (warning) {
    warning.hidden = !parser?.experimental;
    warning.textContent = parser?.experimental
      ? `${parser.label || parser.parser_id} is experimental. Validate the prompt before queueing and preserve parser metadata for replay.`
      : "";
  }
  renderParserSettings(
    "#promptParserAdvancedContent",
    currentParserId(),
    safeParseJson($("#promptParserKwargs")?.value, {}),
    (next) => { setParserKwargs(next); saveSessionSoon(); },
  );
  const status = $("#promptParserSettingsStatus");
  if (status) {
    const processScoped = parser?.process_scoped_settings || [];
    status.textContent = processScoped.length
      ? `${processScoped.length} additional parser settings are process-scoped and are reported for diagnostics rather than changed per queue item.`
      : "Settings are request-scoped and saved with replay metadata.";
  }
}
