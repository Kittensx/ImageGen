import { api } from "../../api.js";
import { state } from "../../state.js";
import { $, notify } from "../../utils.js";
import { setActionIcon } from "../../components/action-icons.js?v=2";
import { currentParserId } from "./shared.js";
import { conciseRoute, conciseShadow, translationPayload } from "./translation.js";

export function renderMessageList(sectionSelector, listSelector, messages = []) {
  const section = $(sectionSelector);
  const list = $(listSelector);
  if (!section || !list) return;
  const grouped = new Map();
  (messages || []).forEach((item) => {
    const message = item?.message || String(item);
    const key = `${item?.code || "message"}::${message}`;
    if (!grouped.has(key)) grouped.set(key, { message, contexts: [], count: 0 });
    const entry = grouped.get(key);
    entry.count += 1;
    const context = [item?.pass_name, item?.prompt_role].filter(Boolean).join(" · ");
    if (context && !entry.contexts.includes(context)) entry.contexts.push(context);
  });
  const entries = [...grouped.values()];
  section.hidden = !entries.length;
  list.replaceChildren(...entries.map((entry) => {
    const node = document.createElement("li");
    node.className = "prompt-message-group";
    const copy = document.createElement("span");
    copy.textContent = entry.message;
    node.append(copy);
    if (entry.count > 1) {
      const count = document.createElement("strong");
      count.className = "prompt-message-count";
      count.textContent = `×${entry.count}`;
      count.title = `${entry.count} matching occurrences`;
      node.append(count);
    }
    if (entry.contexts.length) {
      const context = document.createElement("small");
      context.textContent = `Affected: ${entry.contexts.join(", ")}`;
      node.append(context);
    }
    return node;
  }));
}

export function formatRegionBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = bytes;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount.toFixed(index ? 2 : 0)} ${units[index]}`;
}

export function renderRegionTimeline(passData, timelineSelector, estimateSelector) {
  const timeline = $(timelineSelector);
  const estimateNode = $(estimateSelector);
  const regional = passData?.regional_prompting || {};
  const slots = regional.slots || [];
  const regions = slots.flatMap((slot, slotIndex) => (slot.regions || []).map((region, regionIndex) => ({
    slotIndex: Number(slot.slot_index ?? slotIndex),
    regionIndex: Number(region.region_index ?? regionIndex),
    prompt: String(region.prompt || "region"),
    start: Math.max(0, Math.min(1, Number(region.start ?? 0))),
    stop: Math.max(0, Math.min(1, Number(region.stop ?? 1))),
    curve: String(region.curve || "linear"),
  })));
  if (timeline) {
    timeline.replaceChildren();
    if (!regions.length) {
      const empty = document.createElement("div");
      empty.className = "region-timeline-empty";
      empty.textContent = "No native REGION branches detected.";
      timeline.append(empty);
    } else {
      regions.forEach((region) => {
        const row = document.createElement("div");
        row.className = "region-timeline-row";
        const label = document.createElement("div");
        label.className = "region-timeline-label";
        label.textContent = `S${region.slotIndex + 1} R${region.regionIndex + 1} · ${region.prompt}`;
        label.title = `${region.prompt} · ${region.start.toFixed(2)}–${region.stop.toFixed(2)} · ${region.curve}`;
        const track = document.createElement("div");
        track.className = "region-timeline-track";
        const windowNode = document.createElement("div");
        windowNode.className = "region-timeline-window";
        windowNode.style.left = `${region.start * 100}%`;
        windowNode.style.width = `${Math.max(0.5, (region.stop - region.start) * 100)}%`;
        windowNode.title = label.title;
        track.append(windowNode);
        row.append(label, track);
        timeline.append(row);
      });
    }
  }
  if (estimateNode) {
    const estimate = regional.runtime_estimate || {};
    const peak = estimate.estimated_incremental_peak_bytes || {};
    const masks = estimate.estimated_mask_cache_bytes || {};
    estimateNode.textContent = regions.length ? [
      `Backend: ${regional.backend || "image_gen_model_output"}`,
      `Overlap: ${regional.overlap_policy || "additive"}`,
      `Regions: ${estimate.region_count ?? regions.length}`,
      `Estimated extra UNet calls: ${estimate.extra_unet_calls ?? "unknown"}`,
      `Maximum active branches per step: ${estimate.max_active_regions_per_step ?? "unknown"}`,
      `Estimated FP16 mask cache: ${formatRegionBytes(masks.fp16)}`,
      `Estimated FP16 incremental peak: ${formatRegionBytes(peak.fp16)}`,
      "Estimate excludes model residency and allocator overhead.",
    ].join("\n") : "No REGION runtime overhead estimated.";
  }
}

export function promptStageText(role = {}, stage = "raw") {
  if (stage === "raw") return String(role.raw_prompt || "");
  if (stage === "parser") return String(role.parser_input || "");
  return String(role.parser_canonical_prompt || role.canonical_prompt || "");
}

export function normalizePromptSource(value) {
  return String(value || "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .join("\n")
    .trim();
}

export function parseCanonicalValue(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(String(value || ""));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export function canonicalStructureForRole(role = {}) {
  const direct = role.parser_canonical_structure || role.canonical_structure;
  if (direct && typeof direct === "object" && !Array.isArray(direct)) return direct;
  return parseCanonicalValue(promptStageText(role, "canonical"));
}

export function canonicalSourceForRole(role = {}) {
  const structure = canonicalStructureForRole(role);
  if (typeof structure.lossless_source === "string") return structure.lossless_source;
  return promptStageText(role, "parser");
}

export function canonicalTypeLabel(value) {
  const labels = {
    text: "text node",
    conjunction: "AND conjunction",
    scheduled_text: "scheduled text",
    alternate_text: "alternate text",
    weighted_text: "weighted text",
    attention_group: "attention group",
    deep_sequence: "deep sequence",
    sequence: "sequence",
    group: "group",
    prompt: "structured prompt",
    relation: "relation",
    owner_sequence: "owner sequence",
    weighted: "weighted node",
    literal: "literal",
    scheduled: "scheduled node",
    alternate: "alternate node",
    extension: "extension operator",
  };
  const token = String(value || "node");
  return labels[token] || token.replaceAll("_", " ");
}

export function canonicalSemanticNodes(structure = {}) {
  const root = structure?.semantic_ir?.root;
  if (!root || typeof root !== "object" || Array.isArray(root)) {
    return Array.isArray(structure.nodes) ? structure.nodes : [];
  }
  const nodes = [];
  const visit = (value) => {
    if (Array.isArray(value)) { value.forEach(visit); return; }
    if (!value || typeof value !== "object") return;
    if (typeof value.type === "string") nodes.push(value);
    Object.entries(value).forEach(([key, child]) => {
      if (["source_text", "value", "type"].includes(key)) return;
      visit(child);
    });
  };
  visit(root);
  return nodes;
}

export function canonicalNumericLabels(structure = {}) {
  const semantics = Array.isArray(structure?.semantic_ir?.numeric_semantics)
    ? structure.semantic_ir.numeric_semantics
    : [];
  return semantics.map((item) => {
    const value = item?.value;
    const scope = String(item?.scope || "");
    const context = String(item?.context || "");
    let label = String(item?.type || "number").replaceAll("_", " ");
    if (item?.type === "weight") {
      if (scope === "group_member") label = "Group member weight";
      else if (scope === "and_branch") label = "AND branch weight";
      else if (scope === "sequence_local") label = "Legacy sequence weight";
      else if (scope === "sequence_outer") label = "Legacy sequence outer weight";
      else if (scope === "structured_outer") label = "Structured outer weight";
      else label = "Weight";
    } else if (item?.type === "absolute_step") {
      label = item?.inferred ? "Legacy inferred step" : "Absolute step";
    } else if (item?.type === "fraction_boundary") label = "Fraction boundary";
    else if (item?.type === "percent_boundary") label = "Percent boundary";
    else if (item?.type === "quantity") label = "Quantity";
    const suffix = item?.type === "percent_boundary" ? "%" : "";
    const inference = item?.inferred && context ? ` · ${context}` : "";
    const invalid = item?.valid === false ? ` · INVALID${item?.message ? `: ${item.message}` : ""}` : "";
    return `${label}: ${value}${suffix}${inference}${invalid}`;
  });
}

export function canonicalStructureSummary(structure = {}) {
  const nodes = canonicalSemanticNodes(structure);
  const counts = new Map();
  nodes.forEach((node) => {
    const label = canonicalTypeLabel(node?.type);
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  return {
    contract: String(structure.contract || "canonical prompt"),
    parserNamespace: String(structure.parser_namespace || "unknown"),
    nodeCount: nodes.length,
    nodeLabels: [...counts.entries()].map(([label, count]) => `${count} ${label}${count === 1 ? "" : "s"}`),
    numericLabels: canonicalNumericLabels(structure),
  };
}

export function promptDiffTokens(value) {
  return String(value || "").split(/(\s+|[,;:{}()[\]|])/g).filter((item) => item !== "");
}

export function promptTokenDiff(before, after) {
  const left = promptDiffTokens(before);
  const right = promptDiffTokens(after);
  if (left.join("") === right.join("")) return [{ type: "equal", text: left.join("") }];
  if (left.length > 220 || right.length > 220) {
    return [
      ...(left.length ? [{ type: "remove", text: left.join("") }] : []),
      ...(right.length ? [{ type: "add", text: right.join("") }] : []),
    ];
  }
  const table = Array.from({ length: left.length + 1 }, () => new Uint16Array(right.length + 1));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      table[i][j] = left[i] === right[j]
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const ops = [];
  const push = (type, text) => {
    if (!text) return;
    const last = ops[ops.length - 1];
    if (last?.type === type) last.text += text;
    else ops.push({ type, text });
  };
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      push("equal", left[i]);
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      push("remove", left[i]);
      i += 1;
    } else {
      push("add", right[j]);
      j += 1;
    }
  }
  while (i < left.length) { push("remove", left[i]); i += 1; }
  while (j < right.length) { push("add", right[j]); j += 1; }
  return ops;
}

export function appendPromptDiff(target, before, after) {
  const diff = document.createElement("div");
  diff.className = "prompt-inline-diff";
  promptTokenDiff(before, after).forEach((part) => {
    const node = document.createElement("span");
    node.className = `prompt-diff-${part.type}`;
    node.textContent = part.text;
    diff.append(node);
  });
  target.append(diff);
}

export function makeCopyableTransformationBlock(label, text, copyLabel = "Copy recognized text") {
  const wrapper = document.createElement("div");
  wrapper.className = "prompt-transformation-copy-block";
  const header = document.createElement("div");
  header.className = "prompt-transformation-copy-header";
  const heading = document.createElement("span");
  heading.textContent = label;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "ui-action-button ui-icon-control ui-action-button--compact";
  setActionIcon(copy, "copy", { label: copyLabel, replace: true });
  const pre = document.createElement("pre");
  pre.textContent = String(text || "");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent || "");
      notify(`${label} copied.`);
    } catch (error) {
      notify(`Unable to copy ${label.toLowerCase()}: ${error.message}`, "error");
    }
  });
  header.append(heading, copy);
  wrapper.append(header, pre);
  return wrapper;
}

export function promptTransformationDetails(roleData = {}, before = "", after = "") {
  const container = document.createElement("div");
  container.className = "prompt-transformation-details";
  const substitutions = Array.isArray(roleData.substitutions) ? roleData.substitutions : [];
  if (substitutions.length) {
    substitutions.forEach((item, index) => {
      const card = document.createElement("section");
      card.className = "prompt-transformation-item";
      const title = document.createElement("strong");
      title.textContent = `Transformation ${index + 1}: ${item.canonical_operator || "shortcut"}`;
      const meta = document.createElement("small");
      meta.textContent = `Shortcut ${JSON.stringify(item.source || "")} → parser output ${JSON.stringify(item.parser_emission || "")}`;
      const start = Number(item.start);
      const end = Number(item.end);
      const exactSource = Number.isFinite(start) && Number.isFinite(end) && end >= start
        ? String(before || "").slice(start, end)
        : String(item.source || "");
      card.append(
        title,
        meta,
        makeCopyableTransformationBlock("Recognized source", exactSource, `Copy source for transformation ${index + 1}`),
        makeCopyableTransformationBlock("Parser output", item.parser_emission || "", `Copy parser output for transformation ${index + 1}`),
      );
      container.append(card);
    });
    return container;
  }

  const parts = promptTokenDiff(before, after);
  const removed = parts.filter((part) => part.type === "remove").map((part) => part.text).join("").trim();
  const added = parts.filter((part) => part.type === "add").map((part) => part.text).join("").trim();
  if (removed || added) {
    const card = document.createElement("section");
    card.className = "prompt-transformation-item";
    const title = document.createElement("strong");
    title.textContent = "Detected text transformation";
    card.append(title);
    if (removed) card.append(makeCopyableTransformationBlock("Recognized source", removed));
    if (added) card.append(makeCopyableTransformationBlock("Result", added, "Copy transformed result"));
    container.append(card);
  }
  return container;
}

export function promptTransitionRow(label, before, after, { roleData = null, showTransformationBlocks = false } = {}) {
  const row = document.createElement("div");
  row.className = "prompt-change-row";
  const heading = document.createElement("strong");
  heading.textContent = label;
  row.append(heading);
  if (String(before || "") === String(after || "")) {
    const same = document.createElement("span");
    same.className = "prompt-change-none";
    same.textContent = "No changes";
    row.append(same);
  } else {
    appendPromptDiff(row, before, after);
    if (showTransformationBlocks) row.append(promptTransformationDetails(roleData || {}, before, after));
  }
  return row;
}

export function promptCanonicalStructureRow(roleData = {}, inspectorTarget = "") {
  const row = document.createElement("div");
  row.className = "prompt-change-row prompt-canonical-structure-row";
  const heading = document.createElement("strong");
  heading.textContent = "Parser input → Canonical structure";
  row.append(heading);

  const parserInput = promptStageText(roleData, "parser");
  const structure = canonicalStructureForRole(roleData);
  const canonicalSource = canonicalSourceForRole(roleData);
  const normalizedParser = normalizePromptSource(parserInput);
  const sourceChanged = normalizedParser !== String(canonicalSource || "");

  const sourceStatus = document.createElement("p");
  sourceStatus.className = sourceChanged ? "prompt-canonical-source-warning" : "prompt-change-none";
  sourceStatus.textContent = sourceChanged
    ? "Canonicalization changed the normalized source text. Review the source diff below."
    : "No source-text changes. Canonicalization only describes the prompt in a machine-readable structure.";
  row.append(sourceStatus);
  if (sourceChanged) appendPromptDiff(row, normalizedParser, canonicalSource);

  const summary = canonicalStructureSummary(structure);
  const facts = document.createElement("ul");
  facts.className = "prompt-canonical-facts";
  [
    `Contract: ${summary.contract}`,
    `Parser namespace: ${summary.parserNamespace}`,
    `${summary.nodeCount} canonical node${summary.nodeCount === 1 ? "" : "s"}`,
    ...summary.nodeLabels,
    ...summary.numericLabels,
  ].forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    facts.append(item);
  });
  row.append(facts);

  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const compactSources = [];
  nodes.forEach((node, index) => {
    const source = typeof node?.source === "string" ? node.source : "";
    const start = Number(node?.start);
    const end = Number(node?.end);
    const exact = Number.isFinite(start) && Number.isFinite(end) && end > start
      ? canonicalSource.slice(start, end)
      : source;
    if (!exact || exact === canonicalSource) return;
    const key = `${node?.type || "node"}:${exact}`;
    if (compactSources.some((item) => item.key === key)) return;
    compactSources.push({ key, label: `${canonicalTypeLabel(node?.type)} ${index + 1}`, text: exact });
  });
  if (compactSources.length) {
    const details = document.createElement("details");
    details.className = "prompt-canonical-recognized";
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = `${compactSources.length} recognized canonical block${compactSources.length === 1 ? "" : "s"}`;
    details.append(detailsSummary);
    compactSources.forEach((item) => details.append(
      makeCopyableTransformationBlock(item.label, item.text, `Copy ${item.label}`),
    ));
    row.append(details);
  }

  if (inspectorTarget) {
    const inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "ui-action-button ui-icon-control prompt-canonical-inspect-button";
    inspect.dataset.promptInspectorTarget = inspectorTarget;
    inspect.dataset.promptInspectorTitle = "Canonical representation";
    setActionIcon(inspect, "maximize", { label: "Inspect canonical representation", replace: true });
    inspect.addEventListener("click", () => openPromptInspector(inspect));
    row.append(inspect);
  }
  return { row, sourceChanged };
}

export function semanticTransformationCount(roleData = {}) {
  const substitutions = Array.isArray(roleData.substitutions) ? roleData.substitutions : [];
  if (substitutions.length) return substitutions.length;
  return promptStageText(roleData, "raw") === promptStageText(roleData, "parser") ? 0 : 1;
}

export function renderRoleChanges(roleData, listSelector, countSelector, inspectorTarget = "") {
  const list = $(listSelector);
  const count = $(countSelector);
  if (!list || !count) return 0;
  const raw = promptStageText(roleData, "raw");
  const parser = promptStageText(roleData, "parser");
  const rows = [
    promptTransitionRow("Raw → Parser input", raw, parser, { roleData, showTransformationBlocks: true }),
  ];
  const canonical = promptCanonicalStructureRow(roleData, inspectorTarget);
  rows.push(canonical.row);
  list.replaceChildren(...rows);
  const semanticCount = semanticTransformationCount(roleData) + (canonical.sourceChanged ? 1 : 0);
  count.textContent = semanticCount
    ? `${semanticCount} semantic change${semanticCount === 1 ? "" : "s"}`
    : "No semantic changes";
  count.classList.toggle("has-changes", Boolean(semanticCount));
  return semanticCount;
}

export function regionBranchCount(passData = {}) {
  return (passData.regional_prompting?.slots || []).reduce((total, slot) => total + (slot.regions || []).length, 0);
}

export function canonicalSourceFromSerialized(value) {
  const structure = parseCanonicalValue(value);
  return typeof structure.lossless_source === "string" ? structure.lossless_source : String(value || "");
}

export function canonicalStructureSignature(value) {
  const structure = parseCanonicalValue(value);
  if (!Object.keys(structure).length) return String(value || "");
  return JSON.stringify({
    contract: structure.contract || "",
    parser_namespace: structure.parser_namespace || "",
    semantic_ir: structure.semantic_ir || {},
    nodes: Array.isArray(structure.nodes) ? structure.nodes : [],
  });
}

export function compactCanonicalDifferenceRow(label, beforeValue, afterValue) {
  const beforeSource = canonicalSourceFromSerialized(beforeValue);
  const afterSource = canonicalSourceFromSerialized(afterValue);
  if (beforeSource !== afterSource) {
    return promptTransitionRow(`${label} source`, beforeSource, afterSource);
  }
  const row = document.createElement("div");
  row.className = "prompt-change-row";
  const heading = document.createElement("strong");
  heading.textContent = `${label} structure`;
  const message = document.createElement("span");
  message.className = "prompt-change-none";
  message.textContent = canonicalStructureSignature(beforeValue) === canonicalStructureSignature(afterValue)
    ? "Canonical serialization differs, but source text and structural interpretation are equivalent."
    : "Source text is unchanged; only the canonical structural interpretation differs between passes.";
  row.append(heading, message);
  return row;
}

export function renderHiresChangeSummary(base, hires) {
  const list = $("#promptHiresChanges");
  const count = $("#promptHiresChangeCount");
  const summary = $("#promptHiresInterpretationSummary");
  if (!list || !count) return 0;
  const rows = [];
  if ((base.parser?.parser_id || "") !== (hires.parser?.parser_id || "")) {
    rows.push(promptTransitionRow("Parser", base.parser?.parser_id || "base", hires.parser?.parser_id || "hires"));
  }
  if ((base.shortcut_profile?.profile_id || "") !== (hires.shortcut_profile?.profile_id || "")) {
    rows.push(promptTransitionRow("Shortcut profile", base.shortcut_profile?.profile_id || "base", hires.shortcut_profile?.profile_id || "hires"));
  }
  ["positive", "negative"].forEach((role) => {
    const diff = hires.interpretation_diff?.[role] || {};
    if (!diff.different) return;
    const label = `${role[0].toUpperCase()}${role.slice(1)}`;
    if (String(diff.base_parser_input || "") !== String(diff.hires_parser_input || "")) {
      rows.push(promptTransitionRow(`${label} parser input`, diff.base_parser_input || "", diff.hires_parser_input || ""));
    }
    if (String(diff.base_canonical_prompt || "") !== String(diff.hires_canonical_prompt || "")) {
      rows.push(compactCanonicalDifferenceRow(`${label} canonical`, diff.base_canonical_prompt || "", diff.hires_canonical_prompt || ""));
    }
  });
  if (!rows.length) {
    const same = document.createElement("p");
    same.className = "prompt-change-none";
    same.textContent = "The hires pass uses the same prompt interpretation as the base pass.";
    list.replaceChildren(same);
    count.textContent = "Same as base";
    if (summary) summary.textContent = "Same as base · parser, shortcut profile, parser input, and canonical prompt are unchanged.";
    return 0;
  }
  list.replaceChildren(...rows);
  count.textContent = `${rows.length} difference${rows.length === 1 ? "" : "s"}`;
  count.classList.add("has-changes");
  if (summary) summary.textContent = `${rows.length} hires interpretation difference${rows.length === 1 ? "" : "s"} detected. Review Changes before queueing.`;
  return rows.length;
}

export function renderRegionChangeSummary(base, hires) {
  const baseCount = regionBranchCount(base);
  const hiresCount = regionBranchCount(hires);
  const list = $("#promptRegionChanges");
  const count = $("#promptRegionChangeCount");
  const overview = $("#promptPreflightRegionSummary");
  if (list && count) {
    if (!baseCount && !hiresCount) {
      const inactive = document.createElement("p");
      inactive.className = "prompt-change-none";
      inactive.textContent = "No native REGION branches detected; no REGION runtime overhead is estimated.";
      list.replaceChildren(inactive);
      count.textContent = "Inactive";
    } else {
      const active = document.createElement("p");
      active.textContent = `Base: ${baseCount} branch${baseCount === 1 ? "" : "es"} · Hires: ${hiresCount} branch${hiresCount === 1 ? "" : "es"}`;
      list.replaceChildren(active);
      count.textContent = `${baseCount + hiresCount} active`;
      count.classList.add("has-changes");
    }
  }
  if (overview) overview.textContent = baseCount || hiresCount ? `${baseCount + hiresCount} active` : "Inactive";
  const baseCard = $("#promptRegionBaseCard");
  const hiresCard = $("#promptRegionHiresCard");
  if (baseCard) baseCard.hidden = !baseCount;
  if (hiresCard) hiresCard.hidden = !hiresCount;
  return { baseCount, hiresCount };
}

export function renderPromptPreflightSummary(data, base, hires) {
  const positiveChanges = renderRoleChanges(base.positive || {}, "#promptPositiveChanges", "#promptPositiveChangeCount", "promptTranslationPositiveCanonical");
  const negativeChanges = renderRoleChanges(base.negative || {}, "#promptNegativeChanges", "#promptNegativeChangeCount", "promptTranslationNegativeCanonical");
  const hiresChanges = renderHiresChangeSummary(base, hires);
  renderRegionChangeSummary(base, hires);
  const setText = (selector, text) => { const node = $(selector); if (node) node.textContent = text; };
  setText("#promptPreflightValidity", data.valid ? "Valid" : "Blocked");
  setText("#promptPreflightPositiveSummary", positiveChanges ? `${positiveChanges} change${positiveChanges === 1 ? "" : "s"}` : "Unchanged");
  setText("#promptPreflightNegativeSummary", negativeChanges ? `${negativeChanges} change${negativeChanges === 1 ? "" : "s"}` : "Unchanged");
  setText("#promptPreflightHiresSummary", hiresChanges ? `${hiresChanges} difference${hiresChanges === 1 ? "" : "s"}` : "Same as base");
}

export function setPromptPreflightView(mode = "changes") {
  const changes = mode !== "pipeline";
  const changesView = $("#promptPreflightChangesView");
  const pipelineView = $("#promptPreflightPipelineView");
  const changesButton = $("#promptPreflightChangesTab");
  const pipelineButton = $("#promptPreflightPipelineTab");
  if (changesView) changesView.hidden = !changes;
  if (pipelineView) pipelineView.hidden = changes;
  if (changesButton) {
    changesButton.classList.toggle("is-active", changes);
    changesButton.setAttribute("aria-pressed", String(changes));
  }
  if (pipelineButton) {
    pipelineButton.classList.toggle("is-active", !changes);
    pipelineButton.setAttribute("aria-pressed", String(!changes));
  }
}

let promptInspectorDialog = null;

export function ensurePromptInspectorDialog() {
  if (promptInspectorDialog) return promptInspectorDialog;
  const dialog = document.createElement("dialog");
  dialog.className = "prompt-inspector-dialog";
  dialog.innerHTML = `
    <section class="prompt-inspector-dialog-shell">
      <header class="prompt-inspector-dialog-header">
        <div><small>Prompt pipeline inspector</small><h3 data-prompt-inspector-dialog-title>Prompt stage</h3></div>
        <div class="prompt-inspector-dialog-actions">
          <button type="button" data-prompt-inspector-dialog-compare></button>
          <button type="button" data-prompt-inspector-dialog-copy></button>
          <button type="button" data-prompt-inspector-dialog-close></button>
        </div>
      </header>
      <div class="prompt-inspector-dialog-body">
        <section><h4>Selected stage</h4><pre data-prompt-inspector-dialog-primary></pre></section>
        <section data-prompt-inspector-dialog-comparison hidden><h4>Next stage</h4><pre data-prompt-inspector-dialog-secondary></pre></section>
        <section data-prompt-inspector-dialog-diff hidden><h4>Highlighted changes</h4><div class="prompt-inspector-dialog-diff-content"></div></section>
      </div>
    </section>`;
  document.body.append(dialog);
  const compare = dialog.querySelector("[data-prompt-inspector-dialog-compare]");
  const copy = dialog.querySelector("[data-prompt-inspector-dialog-copy]");
  const close = dialog.querySelector("[data-prompt-inspector-dialog-close]");
  setActionIcon(compare, "compare", { label: "Compare with next prompt stage", replace: true });
  setActionIcon(copy, "copy", { label: "Copy prompt stage", replace: true });
  setActionIcon(close, "remove", { label: "Close prompt inspector", replace: true });
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  close.addEventListener("click", () => dialog.close());
  copy.addEventListener("click", async () => {
    const text = dialog.querySelector("[data-prompt-inspector-dialog-primary]")?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      notify("Prompt stage copied.");
    } catch (error) {
      notify(`Unable to copy prompt stage: ${error.message}`, "error");
    }
  });
  compare.addEventListener("click", () => {
    const comparison = dialog.querySelector("[data-prompt-inspector-dialog-comparison]");
    const diff = dialog.querySelector("[data-prompt-inspector-dialog-diff]");
    const visible = comparison?.hidden !== false;
    if (comparison) comparison.hidden = !visible;
    if (diff) diff.hidden = !visible;
    compare.setAttribute("aria-pressed", String(visible));
  });
  promptInspectorDialog = dialog;
  return dialog;
}

export function openPromptInspector(button) {
  const target = document.getElementById(button.dataset.promptInspectorTarget || "");
  if (!target) return;
  const compareTarget = document.getElementById(button.dataset.promptInspectorCompareTarget || "");
  const dialog = ensurePromptInspectorDialog();
  const title = button.dataset.promptInspectorTitle || "Prompt stage";
  dialog.querySelector("[data-prompt-inspector-dialog-title]").textContent = title;
  const primary = dialog.querySelector("[data-prompt-inspector-dialog-primary]");
  const secondary = dialog.querySelector("[data-prompt-inspector-dialog-secondary]");
  const comparison = dialog.querySelector("[data-prompt-inspector-dialog-comparison]");
  const diffSection = dialog.querySelector("[data-prompt-inspector-dialog-diff]");
  const diffContent = dialog.querySelector(".prompt-inspector-dialog-diff-content");
  const compareButton = dialog.querySelector("[data-prompt-inspector-dialog-compare]");
  primary.textContent = target.textContent || "";
  secondary.textContent = compareTarget?.textContent || "";
  if (comparison) comparison.hidden = true;
  if (diffSection) diffSection.hidden = true;
  if (compareButton) {
    compareButton.hidden = !compareTarget;
    compareButton.setAttribute("aria-pressed", "false");
  }
  diffContent.replaceChildren();
  if (compareTarget) appendPromptDiff(diffContent, primary.textContent, secondary.textContent);
  if (!dialog.open) dialog.showModal();
}

export function bindPromptPreflightInspectors() {
  $("#promptPreflightChangesTab")?.addEventListener("click", () => setPromptPreflightView("changes"));
  $("#promptPreflightPipelineTab")?.addEventListener("click", () => setPromptPreflightView("pipeline"));
  document.querySelectorAll("[data-prompt-inspector-target]").forEach((button) => {
    const title = button.dataset.promptInspectorTitle || "prompt stage";
    button.setAttribute("aria-label", `Open ${title} in a large inspector`);
    button.title = `Open ${title} in a large inspector`;
    button.addEventListener("click", () => openPromptInspector(button));
  });
  setPromptPreflightView("changes");
}

export function formatCanonicalForDisplay(role = {}) {
  const structure = canonicalStructureForRole(role);
  if (Object.keys(structure).length) return JSON.stringify(structure, null, 2);
  return promptStageText(role, "canonical");
}

export function formatSemanticInspection(role = {}) {
  const inspection = role?.semantic_inspection && typeof role.semantic_inspection === "object"
    ? role.semantic_inspection
    : {};
  if (!Object.keys(inspection).length) return "No semantic inspection data.";
  const lines = [];
  lines.push(`Contract: ${inspection.contract_version || "unknown"}`);
  lines.push(`Root: ${inspection.root_type || "text"}`);
  if (inspection.semantic_digest?.digest) lines.push(`Semantic digest: ${inspection.semantic_digest.digest}`);
  const groups = Array.isArray(inspection.groups) ? inspection.groups : [];
  groups.forEach((group, index) => {
    lines.push(`Group ${index + 1} · ${group.member_count || 0} members${group.fallback_used ? " · FALLBACK" : ""}`);
    (group.members || []).forEach((member) => {
      const pct = Number.isFinite(Number(member.normalized_weight))
        ? `${(Number(member.normalized_weight) * 100).toFixed(2)}%`
        : "n/a";
      lines.push(`  - ${member.source || "<empty>"} · local ${pct}${member.explicit_weight ? ` · raw weight ${member.raw_weight}` : ""}`);
    });
    if (group.fallback_reason) lines.push(`  Fallback: ${group.fallback_reason}`);
  });
  const experimentalGroups = Array.isArray(inspection.experimental_groups) ? inspection.experimental_groups : [];
  experimentalGroups.forEach((group, index) => {
    lines.push(`Experimental group ${index + 1} · ${group.algorithm || "unknown"} · ${group.member_count || 0} members${group.fallback_used ? " · FALLBACK" : ""}`);
    (group.members || []).forEach((member) => {
      const pct = Number.isFinite(Number(member.normalized_weight))
        ? `${(Number(member.normalized_weight) * 100).toFixed(2)}%`
        : "n/a";
      lines.push(`  - ${member.source || "<empty>"} · focus ${pct}${member.explicit_weight ? ` · raw weight ${member.raw_weight}` : ""}`);
      if (member.focus_encoder_text) lines.push(`    encoder: ${member.focus_encoder_text}`);
    });
    if (group.fallback_reason) lines.push(`  Fallback: ${group.fallback_reason}`);
  });
  const bindings = Array.isArray(inspection.bindings) ? inspection.bindings : [];
  bindings.forEach((binding, index) => {
    const scope = binding.scope === "subtree" ? "target + descendants" : "target only";
    lines.push(`Binding ${index + 1} · ${binding.modifier || "?"}${binding.operator || "^"}${binding.target || "?"} · ${scope}`);
    lines.push(`  Algorithm: ${binding.algorithm || "experimental"} · inheritance barrier=${binding.inheritance_barrier ? "yes" : "no"}`);
  });
  const relations = Array.isArray(inspection.relationships) ? inspection.relationships : [];
  relations.forEach((relation, index) => {
    lines.push(`Relationship ${index + 1} · ${relation.syntax_origin || "structured"}`);
    if (relation.owner) lines.push(`  Owner: ${relation.owner}`);
    if (relation.parent_scope) lines.push(`  Parent scope: ${relation.parent_scope}`);
    if (relation.owner_composition) lines.push(`  Owner composition: ${relation.owner_composition}`);
    (relation.relations || []).forEach((item, itemIndex) => {
      const weights = relation.normalized_weights || [];
      const pct = Number.isFinite(Number(weights[itemIndex])) ? `${(Number(weights[itemIndex]) * 100).toFixed(2)}%` : "";
      lines.push(`  - ${item}${pct ? ` · ${pct}` : ""}`);
    });
  });
  const schedules = Array.isArray(inspection.schedules) ? inspection.schedules : [];
  schedules.forEach((schedule) => {
    lines.push(`Schedule: ${schedule.source || schedule.encoder_text || ""}${schedule.active_until_step ? ` · through step ${schedule.active_until_step}` : ""}`);
  });
  const fallbacks = Array.isArray(inspection.fallbacks) ? inspection.fallbacks : [];
  lines.push(`Fallbacks: ${fallbacks.length ? fallbacks.join(" | ") : "none"}`);
  const warnings = Array.isArray(inspection.warnings) ? inspection.warnings : [];
  warnings.forEach((warning) => lines.push(`Warning [${warning.category || "parser"}]: ${warning.message || ""}`));
  const branches = Array.isArray(inspection.encoder_text_preview) ? inspection.encoder_text_preview : [];
  if (branches.length) {
    lines.push("Encoder text preview:");
    branches.forEach((branch) => {
      const effective = branch.effective_weight_dynamic
        ? "dynamic by step"
        : (Number.isFinite(Number(branch.effective_final_weight)) ? `${(Number(branch.effective_final_weight) * 100).toFixed(2)}% final` : "n/a");
      lines.push(`  ${branch.index}: ${branch.encoder_text || "<empty>"} · outer=${branch.outer_weight} · group=${branch.group_local_weight} · sequence=${branch.sequence_local_weight} · ${effective}`);
    });
  }
  return lines.join("\n");
}

export function renderTranslation(data, { revealPreview = false } = {}) {
  state.promptConfiguration.translationPreview = data;
  const set = (selector, value) => { const node = $(selector); if (node) node.textContent = value ?? ""; };
  const base = data.base || data;
  const hires = data.hires || {};
  set("#promptTranslationPositiveRaw", base.positive?.raw_prompt);
  set("#promptTranslationPositiveExpanded", base.positive?.parser_input);
  set("#promptTranslationPositiveCanonical", formatCanonicalForDisplay(base.positive || {}));
  set("#promptTranslationPositiveSemantics", formatSemanticInspection(base.positive || {}));
  set("#promptTranslationNegativeRaw", base.negative?.raw_prompt);
  set("#promptTranslationNegativeExpanded", base.negative?.parser_input);
  set("#promptTranslationNegativeCanonical", formatCanonicalForDisplay(base.negative || {}));
  set("#promptTranslationNegativeSemantics", formatSemanticInspection(base.negative || {}));
  set("#promptTranslationHiresPositiveRaw", hires.positive?.raw_prompt);
  set("#promptTranslationHiresPositiveExpanded", hires.positive?.parser_input);
  set("#promptTranslationHiresPositiveCanonical", formatCanonicalForDisplay(hires.positive || {}));
  set("#promptTranslationHiresPositiveSemantics", formatSemanticInspection(hires.positive || {}));
  set("#promptTranslationHiresNegativeRaw", hires.negative?.raw_prompt);
  set("#promptTranslationHiresNegativeExpanded", hires.negative?.parser_input);
  set("#promptTranslationHiresNegativeCanonical", formatCanonicalForDisplay(hires.negative || {}));
  set("#promptTranslationHiresNegativeSemantics", formatSemanticInspection(hires.negative || {}));
  const renderSlots = (passData) => JSON.stringify({
    scope: passData?.prompt_expansion_scope || passData?.expansion_scope || "per_batch",
    positive: passData?.expanded_prompts_by_slot?.positive || [],
    negative: passData?.expanded_prompts_by_slot?.negative || [],
    semantic_fingerprints: passData?.semantic_fingerprints_by_slot || {},
  }, null, 2);
  set("#promptTranslationBaseSlots", renderSlots(base));
  set("#promptTranslationHiresSlots", renderSlots(hires));
  renderRegionTimeline(base, "#promptRegionBaseTimeline", "#promptRegionBaseEstimate");
  renderRegionTimeline(hires, "#promptRegionHiresTimeline", "#promptRegionHiresEstimate");
  const routes = [base.positive?.route_plan, base.negative?.route_plan, hires.positive?.route_plan, hires.negative?.route_plan];
  const shadows = [base.positive?.shadow_comparison, base.negative?.shadow_comparison, hires.positive?.shadow_comparison, hires.negative?.shadow_comparison];
  set("#promptRouteBasePositive", conciseRoute(routes[0]));
  set("#promptRouteBaseNegative", conciseRoute(routes[1]));
  set("#promptRouteHiresPositive", conciseRoute(routes[2]));
  set("#promptRouteHiresNegative", conciseRoute(routes[3]));
  set("#promptShadowBasePositive", conciseShadow(shadows[0]));
  set("#promptShadowBaseNegative", conciseShadow(shadows[1]));
  set("#promptShadowHiresPositive", conciseShadow(shadows[2]));
  set("#promptShadowHiresNegative", conciseShadow(shadows[3]));
  const routeSection = $("#promptRouteSummarySection");
  if (routeSection) routeSection.hidden = !routes.some((item) => item && Object.keys(item).length) && !shadows.some((item) => item && Object.keys(item).length);
  const hiresRouteSection = $("#hiresPromptRouteSummarySection");
  if (hiresRouteSection) hiresRouteSection.hidden = !routes.slice(2).some((item) => item && Object.keys(item).length) && !shadows.slice(2).some((item) => item && Object.keys(item).length);
  renderMessageList("#promptPreflightBlockingSection", "#promptPreflightBlocking", data.blocking_errors || []);
  renderMessageList("#promptPreflightWarningSection", "#promptPreflightWarnings", data.behavior_warnings || []);
  renderMessageList("#promptPreflightNoticeSection", "#promptPreflightNotices", data.informational_notices || []);
  const summary = data.valid
    ? `Prompt preflight valid · base ${base.parser?.parser_id || currentParserId()} / ${base.shortcut_profile?.profile_id || "profile"} · hires ${hires.parser?.parser_id || "inherit"} / ${hires.shortcut_profile?.profile_id || "inherit"}`
    : "Prompt preflight contains blocking errors.";
  set("#promptTranslationWarnings", summary);
  renderPromptPreflightSummary(data, base, hires);
  const differs = Object.values(hires.interpretation_diff || {}).some((item) => item?.different);
  const diffWarning = $("#promptHiresInterpretationWarning");
  if (diffWarning) {
    diffWarning.hidden = !differs;
    diffWarning.textContent = differs
      ? "The hires pass resolves at least one prompt differently from the base pass. The differences are summarized above; expand the full hires pipeline only when you need the exact representations."
      : "";
  }
  const hiresPipeline = $("#promptHiresFullPipeline");
  if (hiresPipeline && !differs) hiresPipeline.open = false;
  const details = $("#promptTranslationPreview");
  if (details && revealPreview) {
    details.open = true;
    setPromptPreflightView("changes");
  }
}

export async function validateCurrentPrompt() {
  const button = $("#validateCurrentPromptButton");
  try {
    if (button) button.disabled = true;
    const report = await api.preflightPrompts(translationPayload());
    renderTranslation(report, { revealPreview: true });
    notify(report.valid ? "Prompt preflight completed successfully." : "Prompt preflight found blocking errors.", report.valid ? "info" : "error");
  } catch (error) {
    const warning = $("#promptTranslationWarnings");
    if (warning) warning.textContent = error.message;
    const details = $("#promptTranslationPreview");
    if (details) details.open = true;
    notify(error.message, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

export async function preflightCurrentPrompt(values = {}) {
  const report = await api.preflightPrompts(translationPayload(values));
  // Generation preflight updates the preview contents without changing the
  // user's collapsed/expanded state. Only the explicit Validate action reveals it.
  renderTranslation(report, { revealPreview: false });
  return report;
}

export function bindPromptPreflight() {
  bindPromptPreflightInspectors();
  $("#validateCurrentPromptButton")?.addEventListener("click", validateCurrentPrompt);
}
