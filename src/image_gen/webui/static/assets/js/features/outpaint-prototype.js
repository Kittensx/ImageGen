import { api } from "../api.js?v=0.1.82";
import { $, debounce, notify } from "../utils.js";

let sourceInfo = null;

function drawPlan(plan) {
  const canvas = $("#outpaintMaskPreview");
  if (!canvas || !plan) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const scale = Math.min(canvas.width / plan.target_width, canvas.height / plan.target_height);
  const drawWidth = plan.target_width * scale;
  const drawHeight = plan.target_height * scale;
  const ox = (canvas.width - drawWidth) / 2;
  const oy = (canvas.height - drawHeight) / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,0.82)";
  ctx.fillRect(ox, oy, drawWidth, drawHeight);

  const b = plan.source_bounds || {};
  const p = plan.protected_bounds || {};
  ctx.fillStyle = "rgba(90,90,90,0.62)";
  ctx.fillRect(
    ox + b.left * scale,
    oy + b.top * scale,
    (b.right - b.left) * scale,
    (b.bottom - b.top) * scale,
  );
  ctx.fillStyle = "rgba(20,20,20,0.88)";
  ctx.fillRect(
    ox + p.left * scale,
    oy + p.top * scale,
    Math.max(0, (p.right - p.left) * scale),
    Math.max(0, (p.bottom - p.top) * scale),
  );
  ctx.strokeStyle = "rgba(30,30,30,0.9)";
  ctx.lineWidth = 1;
  ctx.strokeRect(ox, oy, drawWidth, drawHeight);
}

function renderPlan(plan) {
  const status = $("#outpaintPlanStatus");
  if (!status) return;
  const b = plan.source_bounds || {};
  const p = plan.protected_bounds || {};
  status.textContent = [
    `Source ${plan.source_width}x${plan.source_height}`,
    `Target ${plan.target_width}x${plan.target_height}`,
    `placement x=${b.left}..${Math.max(b.left, b.right - 1)}, y=${b.top}..${Math.max(b.top, b.bottom - 1)}`,
    `expand L${plan.left_expansion} R${plan.right_expansion} T${plan.top_expansion} B${plan.bottom_expansion}`,
    `protected x=${p.left}..${Math.max(p.left, p.right - 1)}, y=${p.top}..${Math.max(p.top, p.bottom - 1)}`,
    `feather ${plan.feather_px}px`,
    `generated ${plan.generative_pixel_area} px`,
  ].join(" · ");
  status.className = "field-status";
  drawPlan(plan);
}

function renderPlanError(message) {
  const status = $("#outpaintPlanStatus");
  if (!status) return;
  status.textContent = message;
  status.className = "field-status error";
}


function renderContextSeedStatus() {
  const status = $("#outpaintContextSeedStatus");
  if (!status) return;
  const mode = $("#outpaintContextSeedMode")?.value || "edge_pad_v1";
  if (mode === "edge_pad_v1") {
    status.textContent = "Extend edge pixels: repeats the nearest source-edge pixels into the new canvas before diffusion begins.";
  } else if (mode === "reflect_pad_v1") {
    status.textContent = "Mirror edge pixels: mirrors source content outward before diffusion. This can duplicate subjects or objects near the edge.";
  } else if (mode === "neutral_gray_v1") {
    status.textContent = "Legacy neutral fill restored from replay/debug metadata. Extend edge pixels is preferred for new work.";
  } else {
    status.textContent = "Extend edge pixels is recommended. Mirroring remains available as an advanced option.";
  }
  status.className = "field-status subtle";
}

function renderPromptModeStatus() {
  const status = $("#outpaintPromptModeStatus");
  if (!status) return;
  const mode = $("#outpaintPromptMode")?.value || "overlay_only_v1";
  const overlay = String($("#outpaintOverlayPositivePrompt")?.value || "").trim();
  if (mode === "overlay_only_v1") {
    if (!overlay) {
      status.textContent = "Extension prompt only requires an extension prompt. The source image remains protected.";
      status.className = "field-status error";
      return;
    }
    status.textContent = "Extension prompt only: the extension prompt controls the generated canvas area while the source image remains protected.";
  } else if (mode === "source_plus_overlay_v1") {
    status.textContent = "Original + extension prompt: both texts are combined into one conditioning pair for the expansion pass.";
  } else {
    status.textContent = "Original prompt only: the main generation prompt is reused for the expansion pass.";
  }
  status.className = "field-status subtle";
}

async function refreshPlan() {
  if (!sourceInfo) return;
  try {
    const plan = await api.outpaintPrototypePlan({
      source_width: sourceInfo.width,
      source_height: sourceInfo.height,
      target_width: Number($("#width")?.value || 0),
      target_height: Number($("#height")?.value || 0),
      anchor: $("#outpaintAnchor")?.value || "center",
      feather_px: Number($("#outpaintFeatherPx")?.value || 24),
      source_x: -1,
      source_y: -1,
    });
    renderPlan(plan);
  } catch (error) {
    renderPlanError(error.message || "Unable to plan outpaint geometry.");
  }
}

const refreshPlanSoon = debounce(refreshPlan, 120);

function enforcePrototypeExclusivity() {
  const enabled = Boolean($("#outpaintPrototypeEnabled")?.checked);
  const hires = $("#hiresEnabled");
  const batch = $("#batchSize");
  if (enabled) {
    if (hires?.checked) {
      hires.checked = false;
      hires.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (batch && Number(batch.value || 1) !== 1) {
      batch.value = "1";
      batch.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }
  if (hires) hires.disabled = enabled;
  if (batch) batch.disabled = enabled;
}

function shapePlacement(baseWidth, baseHeight, targetWidth, targetHeight, anchor) {
  const maxX = targetWidth - baseWidth;
  const maxY = targetHeight - baseHeight;
  let x = Math.floor(maxX / 2);
  let y = Math.floor(maxY / 2);
  if (anchor === "left") x = 0;
  if (anchor === "right") x = maxX;
  if (anchor === "top") y = 0;
  if (anchor === "bottom") y = maxY;
  return { x, y, maxX, maxY };
}

function syncShapeSquareTarget() {
  if (($("#outpaintShapeTargetMode")?.value || "square") !== "square") return;
  const baseWidth = Number($("#width")?.value || 0);
  const baseHeight = Number($("#height")?.value || 0);
  const side = Math.max(baseWidth, baseHeight);
  if (side < 1) return;
  if ($("#outpaintShapeTargetWidth")) $("#outpaintShapeTargetWidth").value = String(side);
  if ($("#outpaintShapeTargetHeight")) $("#outpaintShapeTargetHeight").value = String(side);
}

function renderShapeExpansionStatus() {
  const status = $("#outpaintShapeStatus");
  if (!status) return;
  const enabled = Boolean($("#outpaintShapeExpansionEnabled")?.checked);
  if (!enabled) {
    status.textContent = "Post-generation expansion is off.";
    status.className = "field-status subtle";
    return;
  }
  const bw = Number($("#width")?.value || 0);
  const bh = Number($("#height")?.value || 0);
  const tw = Number($("#outpaintShapeTargetWidth")?.value || 0);
  const th = Number($("#outpaintShapeTargetHeight")?.value || 0);
  const anchor = $("#outpaintShapeAnchor")?.value || "center";
  if (tw < bw || th < bh || tw < 1 || th < 1) {
    status.textContent = `Target ${tw}x${th} must contain the full ${bw}x${bh} base generation.`;
    status.className = "field-status error";
    return;
  }
  const placement = shapePlacement(bw, bh, tw, th, anchor);
  const aligned = bw % 8 === 0 && bh % 8 === 0 && placement.x % 8 === 0 && placement.y % 8 === 0;
  const requested = $("#outpaintShapeSourceHandoff")?.value || "auto";
  let handoff;
  if (requested === "pixel_vae_reencode") {
    handoff = "Source reuse: re-encode image";
  } else if (requested === "live_latent") {
    handoff = aligned ? "Source reuse: live generation data" : "Live generation data requested, but this placement is not 8px-grid aligned";
  } else {
    handoff = aligned ? "Source reuse: live generation data" : "Source reuse: re-encode image (exact placement preserved)";
  }
  status.textContent = [
    `Base ${bw}x${bh}`,
    `Target ${tw}x${th}`,
    `source x=${placement.x}, y=${placement.y}`,
    `expand L${placement.x} R${placement.maxX - placement.x} T${placement.y} B${placement.maxY - placement.y}`,
    handoff,
    `edge init ${($("#outpaintShapeContextSeedMode")?.value || "edge_pad_v1") === "reflect_pad_v1" ? "Mirror edge pixels" : "Extend edge pixels"}`,
    `denoise ${$("#outpaintShapeDenoisingStrength")?.value || "0.40"}`,
  ].join(" · ");
  status.className = requested === "live_latent" && !aligned ? "field-status error" : "field-status";
}

function enforceShapeExpansionExclusivity() {
  const enabled = Boolean($("#outpaintShapeExpansionEnabled")?.checked);
  const hires = $("#hiresEnabled");
  const prototype = $("#outpaintPrototypeEnabled");
  const batch = $("#batchSize");
  if (enabled) {
    if (hires?.checked) {
      hires.checked = false;
      hires.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (prototype?.checked) {
      prototype.checked = false;
      prototype.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (batch && Number(batch.value || 1) !== 1) {
      batch.value = "1";
      batch.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }
  if (hires) hires.disabled = enabled || Boolean(prototype?.checked);
  if (prototype) prototype.disabled = enabled;
  if (batch) batch.disabled = enabled || Boolean(prototype?.checked);
}

export function bindOutpaintPrototype(saveSessionSoon = () => {}) {
  const enabled = $("#outpaintPrototypeEnabled");
  const fileInput = $("#outpaintSourceFile");
  const sourcePath = $("#outpaintSourceImage");
  if (!enabled || !fileInput || !sourcePath) return;

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const status = $("#outpaintSourceStatus");
    if (status) {
      status.textContent = "Reading source image…";
      status.className = "field-status subtle";
    }
    try {
      const result = await api.uploadOutpaintPrototypeSource(file);
      sourceInfo = {
        path: result.path,
        width: Number(result.width || 0),
        height: Number(result.height || 0),
      };
      sourcePath.value = result.path || "";
      if (status) {
        const metadata = result.metadata_available ? ` · metadata: ${result.metadata_source || "available"}` : "";
        status.textContent = `${result.filename || file.name}: ${sourceInfo.width}x${sourceInfo.height}${metadata}`;
        status.className = "field-status";
      }
      if (!String($("#positivePrompt")?.value || "").trim() && result.positive_prompt) {
        $("#positivePrompt").value = result.positive_prompt;
      }
      if (!String($("#negativePrompt")?.value || "").trim() && result.negative_prompt) {
        $("#negativePrompt").value = result.negative_prompt;
      }
      enabled.checked = true;
      enforcePrototypeExclusivity();
      await refreshPlan();
      saveSessionSoon();
    } catch (error) {
      sourceInfo = null;
      sourcePath.value = "";
      if (status) {
        status.textContent = error.message || "Unable to load source image.";
        status.className = "field-status error";
      }
      notify(`Outpaint source failed: ${error.message}`, "error");
    }
  });

  enabled.addEventListener("change", () => {
    if (enabled.checked && $("#outpaintShapeExpansionEnabled")?.checked) {
      $("#outpaintShapeExpansionEnabled").checked = false;
      $("#outpaintShapeExpansionEnabled").dispatchEvent(new Event("change", { bubbles: true }));
    }
    enforcePrototypeExclusivity();
    if (enabled.checked && !sourcePath.value) {
      renderPlanError("Choose a source image before expanding the canvas.");
    }
    refreshPlanSoon();
    renderPromptModeStatus();
    renderContextSeedStatus();
    saveSessionSoon();
  });

  $("#hiresEnabled")?.addEventListener("change", () => {
    if ($("#hiresEnabled")?.checked && enabled.checked) {
      enabled.checked = false;
      enforcePrototypeExclusivity();
      saveSessionSoon();
    }
    if ($("#hiresEnabled")?.checked && $("#outpaintShapeExpansionEnabled")?.checked) {
      $("#outpaintShapeExpansionEnabled").checked = false;
      enforceShapeExpansionExclusivity();
      renderShapeExpansionStatus();
      saveSessionSoon();
    }
  });

  ["#width", "#height", "#outpaintAnchor", "#outpaintFeatherPx", "#outpaintDenoisingStrength", "#outpaintContextSeedMode", "#outpaintLatentStrategy", "#outpaintPromptMode", "#outpaintDiagnosticArtifacts"].forEach((selector) => {
    $(selector)?.addEventListener("change", () => {
      refreshPlanSoon();
      renderPromptModeStatus();
      renderContextSeedStatus();
      saveSessionSoon();
    });
  });
  ["#width", "#height", "#outpaintFeatherPx", "#outpaintOverlayPositivePrompt", "#outpaintOverlayNegativePrompt"].forEach((selector) => {
    $(selector)?.addEventListener("input", refreshPlanSoon);
  });
  ["#outpaintOverlayPositivePrompt", "#outpaintOverlayNegativePrompt"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => {
      renderPromptModeStatus();
      saveSessionSoon();
    });
  });

  const shapeEnabled = $("#outpaintShapeExpansionEnabled");
  if (shapeEnabled) {
    shapeEnabled.addEventListener("change", () => {
      enforceShapeExpansionExclusivity();
      if (shapeEnabled.checked) syncShapeSquareTarget();
      renderShapeExpansionStatus();
      saveSessionSoon();
    });
    $("#outpaintShapeTargetMode")?.addEventListener("change", () => {
      syncShapeSquareTarget();
      renderShapeExpansionStatus();
      saveSessionSoon();
    });
    [
      "#width", "#height", "#outpaintShapeTargetWidth", "#outpaintShapeTargetHeight",
      "#outpaintShapeAnchor", "#outpaintShapeContextSeedMode", "#outpaintShapeSourceHandoff",
      "#outpaintShapePromptMode", "#outpaintShapeDenoisingStrength", "#outpaintShapeSaveBase",
    ].forEach((selector) => {
      $(selector)?.addEventListener("change", () => {
        if (selector === "#width" || selector === "#height") syncShapeSquareTarget();
        renderShapeExpansionStatus();
        saveSessionSoon();
      });
    });
    ["#outpaintShapeTargetWidth", "#outpaintShapeTargetHeight", "#outpaintShapeDenoisingStrength"].forEach((selector) => {
      $(selector)?.addEventListener("input", renderShapeExpansionStatus);
    });
    ["#outpaintShapeOverlayPositivePrompt", "#outpaintShapeOverlayNegativePrompt"].forEach((selector) => {
      $(selector)?.addEventListener("input", saveSessionSoon);
    });
  }

  enforcePrototypeExclusivity();
  enforceShapeExpansionExclusivity();
  renderPromptModeStatus();
  renderContextSeedStatus();
  renderShapeExpansionStatus();
  if (sourcePath.value) {
    const status = $("#outpaintSourceStatus");
    if (status) status.textContent = "A saved source image path is present. Re-select the image to refresh its geometry preview.";
  }
}
