import { api } from "../api.js";
import { option, notify } from "../utils.js";

const editorViewState = new Map();

function stateKey(kind) {
  return String(kind || "advanced");
}

function getViewState(kind) {
  const key = stateKey(kind);
  if (!editorViewState.has(key)) {
    editorViewState.set(key, {
      search: "",
      modifiedOnly: false,
    });
  }
  return editorViewState.get(key);
}

function normalizedType(schema = {}) {
  if (Array.isArray(schema.type)) {
    const filtered = schema.type.filter((item) => item !== "null");
    return filtered[0] || "string";
  }
  if (schema.type) return schema.type;
  if (schema.enum) return "string";
  if (schema.properties) return "object";
  if (schema.items) return "array";
  return "string";
}

function valueAtPath(source, path, fallback = undefined) {
  const result = String(path || "").split(".").reduce((value, key) => (
    value && typeof value === "object" ? value[key] : undefined
  ), source);
  return result === undefined ? fallback : result;
}

function setValueAtPath(target, path, value) {
  const parts = String(path || "").split(".");
  let cursor = target;
  parts.forEach((part, index) => {
    if (index === parts.length - 1) {
      cursor[part] = value;
    } else {
      if (!cursor[part] || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) cursor[part] = {};
      cursor = cursor[part];
    }
  });
}

function deleteValueAtPath(target, path) {
  const parts = String(path || "").split(".");
  const leaf = parts.pop();
  if (!leaf) return;
  const parent = parts.reduce((value, key) => (
    value && typeof value === "object" ? value[key] : undefined
  ), target);
  if (parent && typeof parent === "object") delete parent[leaf];
}

function sameValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function linkedFieldMessage(kind, name) {
  if (name === "steps") {
    return kind === "scheduler"
      ? "This scheduler setting follows the main Generation Steps control and does not need a second editable copy."
      : "This sampler setting follows the main Generation Steps control and does not need a second editable copy.";
  }
  if (name === "device") {
    return "This value is supplied by the active runtime and is not user-editable in the WebUI.";
  }
  return "";
}

function configureInputMetadata(input, { path, name, type, schema, forceValue = false, extra = {} }) {
  input.dataset.schemaPath = path;
  input.dataset.schemaName = name;
  input.dataset.schemaType = type;
  input.dataset.schemaDefault = schema.default === undefined ? "" : JSON.stringify(schema.default);
  input.dataset.omitIfDefault = forceValue ? "0" : (schema.x_omit_if_default ? "1" : "0");
  Object.entries(extra).forEach(([key, value]) => {
    if (value === undefined || value === null) return;
    input.dataset[key] = String(value);
  });
}

function primitiveInput(name, path, schema, value, { forceValue = false } = {}) {
  if (schema.enum) {
    const select = document.createElement("select");
    select.replaceChildren(
      option("", "Default"),
      ...schema.enum.map((item) => option(item, String(item))),
    );
    select.value = value !== undefined ? String(value) : "";
    configureInputMetadata(select, { path, name, type: "string", schema, forceValue });
    return select;
  }

  const input = document.createElement("input");
  const type = normalizedType(schema);
  if (type === "boolean") {
    input.type = "checkbox";
    input.checked = value !== undefined ? Boolean(value) : Boolean(schema.default);
  } else {
    input.type = type === "string" && schema.format === "password" ? "password" : "text";
    input.value = value ?? schema.default ?? "";
  }
  configureInputMetadata(input, { path, name, type, schema, forceValue });
  return input;
}

function numericControl(name, path, schema, value, { forceValue = false } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "schema-number-control";

  const current = value ?? schema.default;
  const minimum = Number.isFinite(schema.minimum) ? Number(schema.minimum) : Number(current ?? 0);
  const maximum = Number.isFinite(schema.maximum) ? Number(schema.maximum) : Math.max(minimum + 1, Number(current ?? minimum + 1));
  const isInteger = normalizedType(schema) === "integer";
  const step = schema.multipleOf || (isInteger ? 1 : 0.01);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(minimum);
  slider.max = String(maximum);
  slider.step = String(step);
  slider.value = String(current ?? minimum);
  slider.dataset.mirror = path;

  const input = document.createElement("input");
  input.type = "number";
  input.min = String(minimum);
  input.max = String(maximum);
  input.step = String(step);
  input.value = current ?? "";
  configureInputMetadata(input, {
    path,
    name,
    type: normalizedType(schema),
    schema,
    forceValue,
    extra: {
      recommendedMinimum: schema.x_recommended_minimum,
      recommendedMaximum: schema.x_recommended_maximum,
      safetyOverrideField: schema.x_safety_override_field,
    },
  });

  slider.addEventListener("input", () => {
    input.value = slider.value;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  input.addEventListener("input", () => {
    if (input.value === "") return;
    slider.value = input.value;
  });

  wrap.append(slider, input);
  return wrap;
}

function arrayInput(name, path, schema, value, { forceValue = false } = {}) {
  if (Array.isArray(schema.prefixItems) && schema.prefixItems.every((item) => ["number", "integer"].includes(normalizedType(item)))) {
    const wrap = document.createElement("div");
    wrap.className = "schema-fixed-array";
    const defaults = Array.isArray(schema.default) ? schema.default : [];
    schema.prefixItems.forEach((itemSchema, index) => {
      const current = Array.isArray(value) ? value[index] : defaults[index];
      const control = numericControl(`${name}_${index}`, path, itemSchema, current, { forceValue: true });
      const input = control.querySelector("[data-schema-path]");
      input.dataset.schemaType = "array-fixed-number";
      input.dataset.schemaElementType = normalizedType(itemSchema);
      input.dataset.schemaIndex = String(index);
      input.dataset.schemaDefault = schema.default === undefined ? "" : JSON.stringify(schema.default);
      input.dataset.omitIfDefault = forceValue ? "0" : (schema.x_omit_if_default ? "1" : "0");
      wrap.append(control);
    });
    return wrap;
  }

  const input = document.createElement("select");
  input.multiple = true;
  input.size = Math.min(Math.max((schema.items?.enum || []).length, 3), 7);
  input.className = "schema-multi-select";
  const selected = new Set(Array.isArray(value) ? value : (schema.default || []));
  input.replaceChildren(...(schema.items?.enum || []).map((item) => {
    const entry = option(item, String(item));
    entry.selected = selected.has(item);
    return entry;
  }));
  configureInputMetadata(input, { path, name, type: "array", schema, forceValue });
  return input;
}

function buildSearchText(parts = []) {
  return parts
    .filter(Boolean)
    .join(" ")
    .replace(/[_.-]+/g, " ")
    .toLowerCase();
}

function wireControlChange(control, onChange) {
  control.querySelectorAll?.("[data-schema-path]").forEach((item) => item.addEventListener("change", onChange));
  if (control.matches?.("[data-schema-path]")) control.addEventListener("change", onChange);
}

function createFieldRow({ name, path, schema, value, kind, forceValue = false, onChange }) {
  const row = document.createElement("label");
  row.className = "schema-field schema-filter-target";
  row.dataset.searchText = buildSearchText([
    name,
    path,
    schema.title,
    schema.short_desc,
    schema.description,
    schema.long_desc,
    ...((schema.enum || []).map((item) => String(item))),
  ]);

  const label = document.createElement("span");
  label.textContent = schema.title || schema.short_desc || name.replaceAll("_", " ");
  row.append(label);

  const linkedMessage = linkedFieldMessage(kind, name);
  if (linkedMessage) {
    row.classList.add("schema-field-linked");
    const linked = document.createElement("output");
    linked.className = "linked-setting";
    linked.textContent = name === "steps" ? "Linked to Generation Steps" : "Linked to Runtime";
    row.append(linked);
    const small = document.createElement("small");
    small.textContent = linkedMessage;
    row.append(small);
    return row;
  }

  const type = normalizedType(schema);
  let control;
  if (type === "array") {
    control = arrayInput(name, path, schema, value, { forceValue });
  } else if (type === "number" || type === "integer") {
    control = numericControl(name, path, schema, value, { forceValue });
  } else {
    control = primitiveInput(name, path, schema, value, { forceValue });
  }
  wireControlChange(control, onChange);
  row.append(control);

  const help = schema.description || schema.long_desc;
  if (help) {
    const small = document.createElement("small");
    small.textContent = help;
    row.append(small);
  }
  return row;
}

function createObjectEditor({ name, path, schema, value, kind, onChange }) {
  const wrapper = document.createElement("div");
  wrapper.className = "schema-object schema-filter-target";
  wrapper.dataset.schemaObjectPath = path;
  wrapper.dataset.schemaDefault = schema.default === undefined ? "" : JSON.stringify(schema.default);
  wrapper.dataset.omitIfDefault = schema.x_omit_if_default ? "1" : "0";
  wrapper.dataset.searchText = buildSearchText([
    name,
    path,
    schema.title,
    schema.description,
    schema.long_desc,
  ]);

  const heading = document.createElement("div");
  heading.className = "schema-object-heading";
  const title = document.createElement("strong");
  title.textContent = schema.title || name.replaceAll("_", " ");
  heading.append(title);
  if (schema.description) {
    const help = document.createElement("small");
    help.textContent = schema.description;
    heading.append(help);
  }
  wrapper.append(heading);

  Object.entries(schema.properties || {}).forEach(([childName, childSchema]) => {
    const childPath = `${path}.${childName}`;
    const childValue = valueAtPath(value || {}, childName, childSchema.default);
    const childType = normalizedType(childSchema);
    if (childType === "object") {
      const details = document.createElement("details");
      details.className = "schema-nested-object schema-filter-target";
      details.dataset.searchText = buildSearchText([
        childName,
        childPath,
        childSchema.title,
        childSchema.description,
        childSchema.long_desc,
      ]);
      const summary = document.createElement("summary");
      summary.textContent = childSchema.title || childName.replaceAll("_", " ");
      details.append(summary, createObjectEditor({
        name: childName,
        path: childPath,
        schema: childSchema,
        value: childValue,
        kind,
        onChange,
      }));
      wrapper.append(details);
    } else {
      wrapper.append(createFieldRow({
        name: childName,
        path: childPath,
        schema: childSchema,
        value: childValue,
        kind,
        forceValue: true,
        onChange,
      }));
    }
  });
  return wrapper;
}

function readInputValue(input) {
  const type = input.dataset.schemaType === "array-fixed-number"
    ? input.dataset.schemaElementType
    : input.dataset.schemaType;
  if (type === "array") return [...input.selectedOptions].map((item) => item.value);
  if (input.type === "checkbox") return input.checked;
  if (input.value === "") return undefined;
  if (type === "number" || type === "integer") return Number(input.value);
  return input.value;
}

function readInputDefaultValue(input) {
  if (!input.dataset.schemaDefault) return undefined;
  try {
    const parsed = JSON.parse(input.dataset.schemaDefault);
    if (input.dataset.schemaType === "array-fixed-number") {
      const index = Number(input.dataset.schemaIndex || 0);
      return Array.isArray(parsed) ? parsed[index] : undefined;
    }
    return parsed;
  } catch {
    return undefined;
  }
}

function syncNumericMirror(input) {
  const slider = input.closest(".schema-number-control")?.querySelector("input[type=range]");
  if (slider && input.value !== "") slider.value = input.value;
}

function assignInputValue(input, value) {
  if (input.dataset.schemaType === "array") {
    const selected = new Set(Array.isArray(value) ? value : []);
    [...input.options].forEach((item) => { item.selected = selected.has(item.value); });
    return;
  }
  if (input.type === "checkbox") {
    input.checked = Boolean(value);
    return;
  }
  input.value = value ?? "";
  syncNumericMirror(input);
}

function inputIsModified(input) {
  const current = readInputValue(input);
  const defaultValue = readInputDefaultValue(input);
  return !sameValue(current, defaultValue);
}

function objectIsModified(objectNode) {
  const path = objectNode.dataset.schemaObjectPath;
  if (!path) return false;
  const currentValues = readAdvancedValues(objectNode);
  const current = valueAtPath(currentValues, path, undefined);
  let defaultValue;
  try {
    defaultValue = objectNode.dataset.schemaDefault ? JSON.parse(objectNode.dataset.schemaDefault) : undefined;
  } catch {
    defaultValue = undefined;
  }
  return !sameValue(current, defaultValue);
}

function entryIsModified(entry) {
  if (entry.matches(".schema-object")) return objectIsModified(entry);
  const inputs = [...entry.querySelectorAll("[data-schema-path]")];
  if (!inputs.length) return false;
  return inputs.some((input) => inputIsModified(input));
}

function entryMatchesSearch(entry, searchText) {
  if (!searchText) return true;
  const haystack = buildSearchText([
    entry.dataset.searchText,
    entry.textContent,
  ]);
  return haystack.includes(searchText);
}

function updateFilterStatus(container, visibleCount, totalCount, modifiedVisible, modifiedTotal) {
  const status = container.querySelector(".schema-filter-status");
  if (!status) return;
  status.textContent = `Showing ${visibleCount} of ${totalCount} setting${totalCount === 1 ? "" : "s"} · ${modifiedVisible} modified visible · ${modifiedTotal} modified total`;
}

function updateGroupRemainder(group, count, shouldOpen = false) {
  const details = group.querySelector(".schema-group-remainder");
  const summary = group.querySelector(".schema-group-remainder > summary");
  if (!details || !summary) return;
  details.hidden = count === 0;
  summary.textContent = count === 0
    ? "Show remaining default settings"
    : `Show remaining default settings (${count})`;
  if (count === 0) details.open = false;
  else if (shouldOpen) details.open = true;
}

function applyEditorFilters(container, kind) {
  const state = getViewState(kind);
  const searchText = String(state.search || "").trim().toLowerCase();
  let totalCount = 0;
  let visibleCount = 0;
  let modifiedTotal = 0;
  let modifiedVisible = 0;

  container.querySelectorAll(".schema-group").forEach((group) => {
    const primary = group.querySelector(".schema-group-primary");
    const remainder = group.querySelector(".schema-group-remainder-body");
    const empty = group.querySelector(".schema-group-empty-state");
    const entries = group.__schemaEntries || [];
    if (!primary || !remainder) return;

    primary.replaceChildren();
    remainder.replaceChildren();

    let groupVisible = 0;
    let groupRemainder = 0;
    let groupModified = 0;
    entries.forEach((entry) => {
      totalCount += 1;
      const modified = entryIsModified(entry);
      if (modified) modifiedTotal += 1;
      if (!entryMatchesSearch(entry, searchText)) return;
      if (modified) groupModified += 1;
      if (state.modifiedOnly && !modified) {
        remainder.append(entry);
        groupRemainder += 1;
        return;
      }
      primary.append(entry);
      groupVisible += 1;
      visibleCount += 1;
      if (modified) modifiedVisible += 1;
    });

    if (state.modifiedOnly) {
      visibleCount += groupRemainder;
      updateGroupRemainder(group, groupRemainder, Boolean(searchText) && groupVisible === 0 && groupRemainder > 0);
      if (empty) empty.hidden = !(groupVisible === 0 && groupRemainder > 0);
    } else {
      updateGroupRemainder(group, 0, false);
      if (empty) empty.hidden = true;
    }

    group.hidden = groupVisible === 0 && groupRemainder === 0;
  });

  container.querySelector(".schema-empty-results")?.classList.toggle("is-hidden", visibleCount > 0);
  updateFilterStatus(container, visibleCount, totalCount, modifiedVisible, modifiedTotal);
}

function resetSection(group) {
  const inputs = [...group.querySelectorAll("[data-schema-path]")];
  inputs.forEach((input) => assignInputValue(input, readInputDefaultValue(input)));
}

function updateSafetyOverrideState(container, kind) {
  if (kind !== "scheduler") return;
  const toggle = container.querySelector('[data-schema-path="allow_randomization_range_override"]');
  const enabled = Boolean(toggle?.checked);
  const banner = container.querySelector(".scheduler-safety-override-banner");
  if (banner) {
    banner.classList.toggle("is-active", enabled);
    banner.textContent = enabled
      ? "Randomization safety override is ON. Only min/max values you explicitly edit can extend beyond the recommended ranges; global randomization will remain inside those effective limits."
      : "Randomization safety limits are active. Global randomization is restricted to the recommended per-setting min/max ranges.";
  }

  container.querySelectorAll("input[data-safety-override-field]").forEach((input) => {
    if (enabled) {
      input.removeAttribute("min");
      input.removeAttribute("max");
    } else {
      const minimum = input.dataset.recommendedMinimum;
      const maximum = input.dataset.recommendedMaximum;
      if (minimum !== undefined && minimum !== "") input.min = minimum;
      if (maximum !== undefined && maximum !== "") input.max = maximum;
      const numericValue = Number(input.value);
      if (Number.isFinite(numericValue)) {
        const minimumValue = minimum !== undefined && minimum !== "" ? Number(minimum) : null;
        const maximumValue = maximum !== undefined && maximum !== "" ? Number(maximum) : null;
        if (minimumValue !== null && numericValue < minimumValue) input.value = String(minimumValue);
        if (maximumValue !== null && Number(input.value) > maximumValue) input.value = String(maximumValue);
        syncNumericMirror(input);
      }
    }
  });
}

export function readAdvancedValues(container) {
  const values = {};
  const fixedArrayGroups = new Map();
  container.querySelectorAll("[data-schema-path]").forEach((input) => {
    if (input.dataset.schemaType === "array-fixed-number") {
      const path = input.dataset.schemaPath;
      if (!fixedArrayGroups.has(path)) fixedArrayGroups.set(path, []);
      fixedArrayGroups.get(path).push(input);
      return;
    }
    const value = readInputValue(input);
    if (value === undefined) return;
    const omitIfDefault = input.dataset.omitIfDefault === "1";
    const defaultRaw = input.dataset.schemaDefault;
    const defaultValue = defaultRaw ? JSON.parse(defaultRaw) : undefined;
    if (omitIfDefault && defaultValue !== undefined && sameValue(value, defaultValue)) return;
    setValueAtPath(values, input.dataset.schemaPath, value);
  });

  fixedArrayGroups.forEach((inputs, path) => {
    const ordered = [...inputs].sort((left, right) => Number(left.dataset.schemaIndex || 0) - Number(right.dataset.schemaIndex || 0));
    const arrayValue = ordered.map((input) => readInputValue(input));
    if (arrayValue.every((item) => item === undefined)) return;
    const omitIfDefault = ordered[0]?.dataset.omitIfDefault === "1";
    const defaultRaw = ordered[0]?.dataset.schemaDefault;
    const defaultValue = defaultRaw ? JSON.parse(defaultRaw) : undefined;
    if (omitIfDefault && defaultValue !== undefined && sameValue(arrayValue, defaultValue)) return;
    setValueAtPath(values, path, arrayValue);
  });

  const objects = [...container.querySelectorAll("[data-schema-object-path]")]
    .sort((left, right) => right.dataset.schemaObjectPath.split(".").length - left.dataset.schemaObjectPath.split(".").length);
  objects.forEach((object) => {
    if (object.dataset.omitIfDefault !== "1" || !object.dataset.schemaDefault) return;
    const current = valueAtPath(values, object.dataset.schemaObjectPath, undefined);
    const defaultValue = JSON.parse(object.dataset.schemaDefault);
    if (current !== undefined && sameValue(current, defaultValue)) {
      deleteValueAtPath(values, object.dataset.schemaObjectPath);
    }
  });
  return values;
}

function applyProfileValues(container, values) {
  container.querySelectorAll("[data-schema-path]").forEach((input) => {
    const value = valueAtPath(values, input.dataset.schemaPath, undefined);
    if (value === undefined) return;
    if (input.dataset.schemaType === "array-fixed-number") {
      const index = Number(input.dataset.schemaIndex || 0);
      const component = Array.isArray(value) ? value[index] : undefined;
      if (component === undefined) return;
      assignInputValue(input, component);
      return;
    }
    assignInputValue(input, value);
  });
}

export async function renderAdvancedEditor({
  container,
  descriptor,
  kind,
  currentValues = {},
  onChange = () => {},
}) {
  container.replaceChildren();
  if (!descriptor) {
    container.textContent = `No ${kind} selected.`;
    return;
  }

  const viewState = getViewState(kind);
  const triggerChange = () => {
    updateSafetyOverrideState(container, kind);
    applyEditorFilters(container, kind);
    onChange();
  };

  const toolbar = document.createElement("div");
  toolbar.className = "profile-toolbar preset-toolbar";

  const selectorBlock = document.createElement("label");
  selectorBlock.className = "field-block preset-selector-block";
  const selectorLabel = document.createElement("span");
  selectorLabel.textContent = `${kind[0].toUpperCase()}${kind.slice(1)} profile`;
  const selectorRow = document.createElement("span");
  selectorRow.className = "input-action-row";
  const profileInput = document.createElement("input");
  profileInput.type = "text";
  profileInput.placeholder = `Type or select a ${kind} profile…`;
  profileInput.autocomplete = "off";
  const profileListId = `${kind}-advanced-profile-list`;
  profileInput.setAttribute("list", profileListId);
  const profileList = document.createElement("datalist");
  profileList.id = profileListId;
  const profileArrow = document.createElement("button");
  profileArrow.type = "button";
  profileArrow.className = "small-button";
  profileArrow.textContent = "⌄";
  profileArrow.title = `Show ${kind} profiles`;
  profileArrow.addEventListener("click", () => {
    profileInput.focus();
    try { profileInput.showPicker?.(); } catch { /* Browser may not expose a picker for datalists. */ }
  });
  selectorRow.append(profileInput, profileArrow);
  selectorBlock.append(selectorLabel, selectorRow, profileList);

  const profileActions = document.createElement("div");
  profileActions.className = "preset-actions";
  const loadButton = document.createElement("button");
  const saveButton = document.createElement("button");
  const saveAsButton = document.createElement("button");
  const deleteButton = document.createElement("button");
  const exportButton = document.createElement("button");
  const importButton = document.createElement("button");
  const importInput = document.createElement("input");
  [loadButton, saveButton, saveAsButton, deleteButton, exportButton, importButton].forEach((button) => {
    button.type = "button";
    button.className = "secondary-button compact-button";
  });
  deleteButton.className = "danger-button compact-button";
  loadButton.textContent = "Load";
  saveButton.textContent = "Save";
  saveAsButton.textContent = "Save as";
  deleteButton.textContent = "Delete";
  exportButton.textContent = "Save local";
  importButton.textContent = "Load local";
  importInput.type = "file";
  importInput.accept = ".json,application/json";
  importInput.hidden = true;
  profileActions.append(loadButton, saveButton, saveAsButton, deleteButton, exportButton, importButton, importInput);
  toolbar.append(selectorBlock, profileActions);
  container.append(toolbar);

  if (kind === "scheduler" && descriptor.config_schema?.properties?.allow_randomization_range_override) {
    const safetyBanner = document.createElement("div");
    safetyBanner.className = "scheduler-safety-override-banner";
    safetyBanner.setAttribute("role", "status");
    container.append(safetyBanner);
  }

  const filterToolbar = document.createElement("div");
  filterToolbar.className = "schema-filter-toolbar";

  const searchWrap = document.createElement("label");
  searchWrap.className = "schema-filter-search";
  const searchLabel = document.createElement("span");
  searchLabel.textContent = "Search settings";
  const searchInput = document.createElement("input");
  searchInput.type = "search";
  searchInput.placeholder = `Search ${kind} settings…`;
  searchInput.value = viewState.search || "";
  searchInput.autocomplete = "off";
  searchInput.addEventListener("input", () => {
    viewState.search = searchInput.value;
    applyEditorFilters(container, kind);
  });
  searchWrap.append(searchLabel, searchInput);

  const modifiedWrap = document.createElement("label");
  modifiedWrap.className = "schema-filter-toggle";
  const modifiedOnly = document.createElement("input");
  modifiedOnly.type = "checkbox";
  modifiedOnly.checked = Boolean(viewState.modifiedOnly);
  modifiedOnly.addEventListener("change", () => {
    viewState.modifiedOnly = modifiedOnly.checked;
    applyEditorFilters(container, kind);
  });
  const modifiedText = document.createElement("span");
  modifiedText.textContent = "Show modified only";
  modifiedWrap.append(modifiedOnly, modifiedText);

  const clearButton = document.createElement("button");
  clearButton.type = "button";
  clearButton.className = "small-button";
  clearButton.textContent = "Clear";
  clearButton.title = "Clear search and filters";
  clearButton.addEventListener("click", () => {
    viewState.search = "";
    viewState.modifiedOnly = false;
    searchInput.value = "";
    modifiedOnly.checked = false;
    applyEditorFilters(container, kind);
  });

  const filterStatus = document.createElement("small");
  filterStatus.className = "schema-filter-status";

  filterToolbar.append(searchWrap, modifiedWrap, clearButton, filterStatus);
  container.append(filterToolbar);

  const emptyResults = document.createElement("p");
  emptyResults.className = "schema-empty-results is-hidden";
  emptyResults.textContent = "No settings match the current search/filter selection.";
  container.append(emptyResults);

  const properties = descriptor.config_schema?.properties || {};
  const grouped = new Map();
  Object.entries(properties).forEach(([name, schema]) => {
    if (schema.x_hidden_in_advanced) return;
    const group = schema.x_group || "Advanced Settings";
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push([name, schema]);
  });

  [...grouped.entries()].forEach(([groupName, entries], groupIndex) => {
    const details = document.createElement("details");
    details.className = "schema-group";
    details.open = groupIndex < 3;
    const summary = document.createElement("summary");
    summary.className = "schema-group-summary";
    const summaryText = document.createElement("span");
    summaryText.textContent = groupName;
    const summaryActions = document.createElement("span");
    summaryActions.className = "schema-group-summary-actions";
    const resetButton = document.createElement("button");
    resetButton.type = "button";
    resetButton.className = "small-button schema-reset-section-button";
    resetButton.textContent = "Reset section";
    resetButton.title = `Reset the ${groupName} section to defaults`;
    summaryActions.append(resetButton);
    summary.append(summaryText, summaryActions);

    const body = document.createElement("div");
    body.className = "schema-group-content";

    const primary = document.createElement("div");
    primary.className = "schema-group-primary";
    const emptyState = document.createElement("p");
    emptyState.className = "schema-group-empty-state";
    emptyState.hidden = true;
    emptyState.textContent = "No modified settings match right now. Expand the remaining default settings below to browse the rest of this section.";
    const remainder = document.createElement("details");
    remainder.className = "schema-group-remainder";
    remainder.hidden = true;
    const remainderSummary = document.createElement("summary");
    remainderSummary.textContent = "Show remaining default settings";
    const remainderBody = document.createElement("div");
    remainderBody.className = "schema-group-remainder-body";
    remainder.append(remainderSummary, remainderBody);

    body.append(primary, emptyState, remainder);
    details.append(summary, body);
    container.append(details);

    const groupEntries = [];
    entries.forEach(([name, schema]) => {
      const value = valueAtPath(currentValues, name, schema.default);
      const node = normalizedType(schema) === "object"
        ? createObjectEditor({ name, path: name, schema, value, kind, onChange: triggerChange })
        : createFieldRow({ name, path: name, schema, value, kind, onChange: triggerChange });
      groupEntries.push(node);
    });
    details.__schemaEntries = groupEntries;

    resetButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      resetSection(details);
      triggerChange();
      notify(`${groupName} reset to defaults.`);
    });
  });

  const getProfiles = () => JSON.parse(profileInput.dataset.profiles || "[]");
  const findProfile = (name) => getProfiles().find((item) => String(item.name || "").trim() === String(name || "").trim());
  const syncProfileButtons = () => {
    const existing = findProfile(profileInput.value);
    loadButton.disabled = !existing;
    saveButton.disabled = !existing;
    deleteButton.disabled = !existing;
  };

  async function refreshProfiles(selectedName = "") {
    const profiles = await api.profiles(kind, descriptor.plugin_id);
    profileList.replaceChildren(...profiles.map((profile) => option(profile.name, profile.name)));
    profileInput.dataset.profiles = JSON.stringify(profiles);
    if (selectedName) profileInput.value = selectedName;
    syncProfileButtons();
  }

  function loadNamedProfile(name, { announce = true } = {}) {
    const profile = findProfile(name);
    if (!profile) return false;
    profileInput.value = profile.name;
    applyProfileValues(container, profile.values || {});
    triggerChange();
    syncProfileButtons();
    if (announce) notify(`${kind[0].toUpperCase()}${kind.slice(1)} profile loaded.`);
    return true;
  }

  async function saveNamedProfile(name, overwrite) {
    const trimmed = String(name || "").trim();
    if (!trimmed) {
      notify(`Enter a ${kind} profile name first.`, "error");
      return;
    }
    await api.saveProfile(kind, {
      name: trimmed,
      plugin_id: descriptor.plugin_id,
      values: readAdvancedValues(container),
      overwrite,
    });
    await refreshProfiles(trimmed);
    notify(`${kind[0].toUpperCase()}${kind.slice(1)} profile ${overwrite ? "saved" : "saved as new"}.`);
  }

  loadButton.addEventListener("click", () => {
    if (!loadNamedProfile(profileInput.value)) {
      notify(`Select a saved ${kind} profile to load.`, "error");
    }
  });

  saveButton.addEventListener("click", async () => {
    const existing = findProfile(profileInput.value);
    if (!existing) {
      notify(`Select an existing ${kind} profile name to save over, or use Save as.`, "error");
      return;
    }
    try {
      await saveNamedProfile(existing.name, true);
    } catch (error) {
      notify(error.message, "error");
    }
  });

  saveAsButton.addEventListener("click", async () => {
    try {
      await saveNamedProfile(profileInput.value, false);
    } catch (error) {
      notify(error.message, "error");
    }
  });

  deleteButton.addEventListener("click", async () => {
    const existing = findProfile(profileInput.value);
    if (!existing) return;
    if (!window.confirm(`Delete the ${kind} profile "${existing.name}"?`)) return;
    try {
      await api.deleteProfile(kind, existing.name, descriptor.plugin_id);
      await refreshProfiles("");
      profileInput.value = "";
      syncProfileButtons();
      notify(`${kind[0].toUpperCase()}${kind.slice(1)} profile deleted.`);
    } catch (error) {
      notify(error.message, "error");
    }
  });

  exportButton.addEventListener("click", () => {
    const payload = {
      export_version: 1,
      kind,
      plugin_id: descriptor.plugin_id,
      name: String(profileInput.value || `${kind}_profile`).trim() || `${kind}_profile`,
      values: readAdvancedValues(container),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stem = payload.name.replace(/[^a-z0-9._-]+/gi, "_");
    link.href = url;
    link.download = `${stem || kind}.image_gen_profile.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  });

  importButton.addEventListener("click", () => importInput.click());
  importInput.addEventListener("change", async () => {
    const [file] = importInput.files || [];
    importInput.value = "";
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      applyProfileValues(container, payload.values || {});
      triggerChange();
      profileInput.value = String(payload.name || file.name.replace(/\.json$/i, "")).trim();
      syncProfileButtons();
      notify(`${kind[0].toUpperCase()}${kind.slice(1)} settings loaded from local file.`);
    } catch (error) {
      notify(`Unable to load the local ${kind} profile: ${error.message}`, "error");
    }
  });

  profileInput.addEventListener("input", syncProfileButtons);
  profileInput.addEventListener("change", () => {
    if (findProfile(profileInput.value)) {
      loadNamedProfile(profileInput.value, { announce: false });
    }
    syncProfileButtons();
  });

  try {
    await refreshProfiles();
  } catch (error) {
    notify(error.message, "error");
  }

  updateSafetyOverrideState(container, kind);
  applyEditorFilters(container, kind);
}
