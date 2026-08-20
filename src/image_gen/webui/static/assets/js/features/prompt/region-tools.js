import { $, notify } from "../../utils.js";
import { planHiresDimensions } from "../../components/hires-dimensions.js?v=0.1.79";
import { generationSpatialRequirements } from "../generation-capabilities.js";
import { saveSessionSoon } from "./runtime.js";

const REGION_BUILDER_TARGETS = {
  positive: "#positivePrompt",
  hires_positive: "#hiresPositivePrompt",
};
let regionBuilderDialog = null;
let regionBuilderFrame = null;
let regionBuilderTarget = "positive";
let regionBuilderBound = false;
let regionBuilderReady = false;

export function normalizedRegionBuilderTarget() {
  let target = String($("#promptSymbolTarget")?.value || "auto");
  // The REGION Builder defaults to the base positive prompt. The symbol
  // palette's auto target may remain on a hires field after focus moves back
  // to the base prompt, which can send the wrong pass dimensions.
  if (target === "auto") target = "positive";
  if (target === "negative") target = "positive";
  if (target === "hires_negative") target = "hires_positive";
  return REGION_BUILDER_TARGETS[target] ? target : "positive";
}

export function regionBuilderDimensions(target = regionBuilderTarget) {
  const baseWidth = Math.max(64, Math.round(Number($("#width")?.value || 512)));
  const baseHeight = Math.max(64, Math.round(Number($("#height")?.value || 512)));
  if (target !== "hires_positive") {
    return { width: baseWidth, height: baseHeight, pass: "base" };
  }

  const hiresEnabled = Boolean($("#hiresEnabled")?.checked);
  if (!hiresEnabled) {
    return { width: baseWidth, height: baseHeight, pass: "base" };
  }

  const spatial = generationSpatialRequirements();
  const plan = planHiresDimensions({
    baseWidth,
    baseHeight,
    mode: $("#hiresSizeMode")?.value || "scale_from_base",
    scale: $("#hiresScale")?.value || 1.5,
    targetWidth: $("#hiresWidth")?.value,
    targetHeight: $("#hiresHeight")?.value,
    dimensionMultiple: spatial.pixelAlignmentMultiple,
    baseDimensionMultiple: spatial.latentScaleFactor,
    enabled: true,
  });
  return {
    width: plan.final_width,
    height: plan.final_height,
    pass: "hires",
  };
}

export function findRegionBlockRange(text) {
  const source = String(text || "");
  const start = source.indexOf("REGION{");
  if (start < 0) return null;
  let depth = 1;
  let index = start + "REGION{".length;
  while (index < source.length && depth > 0) {
    if (source[index] === "\\" && index + 1 < source.length) {
      index += 2;
      continue;
    }
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    index += 1;
  }
  if (depth !== 0) return null;
  let end = index;
  let cursor = end;
  while (cursor < source.length && /\s/.test(source[cursor])) cursor += 1;
  if (source[cursor] === "[") {
    let bracketDepth = 1;
    cursor += 1;
    while (cursor < source.length && bracketDepth > 0) {
      if (source[cursor] === "\\" && cursor + 1 < source.length) {
        cursor += 2;
        continue;
      }
      if (source[cursor] === "[") bracketDepth += 1;
      if (source[cursor] === "]") bracketDepth -= 1;
      cursor += 1;
    }
    if (bracketDepth === 0) end = cursor;
  } else if (source[cursor] === ":") {
    cursor += 1;
    while (cursor < source.length && !/\s/.test(source[cursor])) cursor += 1;
    end = cursor;
  }
  return { start, end };
}

export function applyRegionBuilderPrompt(prompt) {
  const selector = REGION_BUILDER_TARGETS[regionBuilderTarget] || "#positivePrompt";
  const field = $(selector);
  if (!field) return;
  const replacement = String(prompt || "").trim();
  if (!replacement) return;
  const existing = String(field.value || "");
  const range = findRegionBlockRange(existing);
  if (range) {
    field.value = `${existing.slice(0, range.start)}${replacement}${existing.slice(range.end)}`.replace(/\s{2,}/g, " ").trim();
  } else {
    const start = Number.isInteger(field.selectionStart) ? field.selectionStart : existing.length;
    const end = Number.isInteger(field.selectionEnd) ? field.selectionEnd : start;
    const before = existing.slice(0, start).trimEnd();
    const after = existing.slice(end).trimStart();
    field.value = [before, replacement, after].filter(Boolean).join(" ");
  }
  field.dispatchEvent(new Event("input", { bubbles: true }));
  field.dispatchEvent(new Event("change", { bubbles: true }));
  field.focus();
  saveSessionSoon();
  notify("Applied the REGION plan to the prompt.");
}

export function regionBuilderView() {
  if (!regionBuilderDialog) regionBuilderDialog = $("#regionBuilderDialog");
  if (!regionBuilderFrame) regionBuilderFrame = $("#regionBuilderFrame");
  return {
    dialog: regionBuilderDialog,
    frame: regionBuilderFrame,
    closeButton: $("#regionBuilderCloseButton"),
    closeToolbar: $("#regionBuilderCloseToolbarButton"),
  };
}

export function closeRegionBuilder() {
  const view = regionBuilderView();
  if (view.dialog?.open) view.dialog.close();
}

export function sendRegionBuilderInit({ reason = "open" } = {}) {
  const view = regionBuilderView();
  const hostWindow = view.frame?.contentWindow;
  if (!hostWindow) return;
  const selector = REGION_BUILDER_TARGETS[regionBuilderTarget] || "#positivePrompt";
  const dimensions = regionBuilderDimensions(regionBuilderTarget);
  hostWindow.postMessage({
    type: "imagegen-region-builder-init",
    target: regionBuilderTarget,
    target_pass: dimensions.pass,
    prompt: $(selector)?.value || "",
    width: dimensions.width,
    height: dimensions.height,
    reason,
  }, window.location.origin);
}

export function openRegionBuilder() {
  const view = regionBuilderView();
  if (!view.dialog || !view.frame) {
    notify("REGION Builder UI is unavailable in this build.", "error");
    return;
  }
  regionBuilderTarget = normalizedRegionBuilderTarget();
  const dimensions = regionBuilderDimensions(regionBuilderTarget);
  const builderUrl = new URL("/region-builder.html", window.location.origin);
  builderUrl.searchParams.set("v", "0.1.66");
  builderUrl.searchParams.set("target", regionBuilderTarget);
  builderUrl.searchParams.set("pass", dimensions.pass);
  builderUrl.searchParams.set("width", String(dimensions.width));
  builderUrl.searchParams.set("height", String(dimensions.height));
  const nextUrl = builderUrl.toString();
  if (view.frame.dataset.loadedSrc !== nextUrl) {
    regionBuilderReady = false;
    view.frame.dataset.loadedSrc = nextUrl;
    view.frame.src = nextUrl;
  } else {
    window.setTimeout(() => sendRegionBuilderInit({ reason: "reopen" }), 80);
  }
  if (!view.dialog.open) view.dialog.showModal();
  window.setTimeout(() => {
    if (view.frame.contentWindow) sendRegionBuilderInit({ reason: regionBuilderReady ? "refresh" : "open" });
  }, 140);
}

export function bindRegionBuilderBridge() {
  if (regionBuilderBound) return;
  regionBuilderBound = true;
  const view = regionBuilderView();
  const close = () => closeRegionBuilder();
  view.closeButton?.addEventListener("click", close);
  view.closeToolbar?.addEventListener("click", close);
  view.dialog?.addEventListener("click", (event) => {
    if (event.target === view.dialog) closeRegionBuilder();
  });
  view.dialog?.addEventListener("close", () => {
    regionBuilderReady = false;
  });
  view.frame?.addEventListener("load", () => {
    regionBuilderReady = false;
    window.setTimeout(() => sendRegionBuilderInit({ reason: "frame-load" }), 180);
  });
  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const payload = event.data || {};
    if (payload.type === "imagegen-region-builder-ready") {
      regionBuilderReady = true;
      sendRegionBuilderInit({ reason: "ready" });
    } else if (payload.type === "imagegen-region-builder-apply") {
      if (REGION_BUILDER_TARGETS[payload.target]) regionBuilderTarget = payload.target;
      const dimensions = regionBuilderDimensions(regionBuilderTarget);
      const builderWidth = Math.round(Number(payload.width || 0));
      const builderHeight = Math.round(Number(payload.height || 0));
      const usesPixels = Boolean(payload.pixel_coordinates);
      if (usesPixels && (builderWidth !== dimensions.width || builderHeight !== dimensions.height)) {
        view.frame?.contentWindow?.postMessage({
          type: "imagegen-region-builder-resync",
          target: regionBuilderTarget,
          target_pass: dimensions.pass,
          width: dimensions.width,
          height: dimensions.height,
        }, window.location.origin);
        notify(
          `REGION Builder resolution ${builderWidth}x${builderHeight} did not match ${dimensions.pass} generation ${dimensions.width}x${dimensions.height}. The builder was resynchronized; review the layout and click Apply again.`,
          "warning",
        );
        return;
      }
      applyRegionBuilderPrompt(payload.prompt);
    }
  });
}

export function bindRegionTools() {
  bindRegionBuilderBridge();
  $("#openRegionBuilderButton")?.addEventListener("click", openRegionBuilder);
}
