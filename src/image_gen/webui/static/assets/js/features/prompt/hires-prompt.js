import { api } from "../../api.js";
import { state } from "../../state.js";
import { $, normalizeClampedNumberInput } from "../../utils.js";
import { clampHiresDimension, normalizeHiresSizeMode, planHiresDimensions } from "../../components/hires-dimensions.js?v=0.1.79";
import { updateHiresUpscalerPlanUI } from "../hires-upscalers.js?v=0.1.79";
import { saveSessionSoon } from "./runtime.js";
import { compatibleProfiles, currentParserId, defaultProfileId, option, profileById, profileSnapshot, safeParseJson } from "./shared.js";
import { parserDefaultOptions, renderParserSettings } from "./parser-settings.js";

let hiresPlannerParityTimer = null;
let hiresPlannerParitySequence = 0;

function normalizeHiresCfgRescaleInput() {
  return normalizeClampedNumberInput($("#hiresCfgRescale"), { minimum: 0, maximum: 1, decimals: 2 });
}

export function populateHiresParsers(selected = "") {
  const select = $("#hiresPromptParserName");
  if (!select) return;
  const available = state.promptParsers.filter((item) => item.available !== false);
  const fallback = available.some((item) => item.parser_id === currentParserId())
    ? currentParserId()
    : (available[0]?.parser_id || "legacy");
  const preferred = available.some((item) => item.parser_id === selected) ? selected : fallback;
  select.replaceChildren(...state.promptParsers.map((item) => option(
    item.parser_id,
    `${item.label || item.parser_id}${item.experimental ? " - Experimental" : ""}${item.available === false ? " - Unavailable" : ""}`,
    item.parser_id === preferred,
    item.available === false,
  )));
  select.value = preferred;
}

export function populateHiresProfiles(selected = "") {
  const select = $("#hiresShortcutProfileName");
  if (!select) return;
  const parserId = $("#hiresPromptParserName")?.value || currentParserId();
  const profiles = compatibleProfiles(parserId);
  const defaultId = defaultProfileId(parserId);
  const preferred = profiles.some((item) => item.profile_id === selected)
    ? selected
    : (profiles.some((item) => item.profile_id === defaultId) ? defaultId : (profiles[0]?.profile_id || ""));
  select.replaceChildren(...profiles.map((item) => option(
    item.profile_id,
    `${item.label || item.profile_id}${item.builtin ? "" : " - User"}`,
    item.profile_id === preferred,
  )));
  select.value = preferred;
  const profile = profileById(select.value);
  const snapshot = $("#hiresShortcutProfileSnapshot");
  if (snapshot) snapshot.value = JSON.stringify(profileSnapshot(profile));
}

export function renderHiresParserSettings() {
  const parserId = $("#hiresPromptParserName")?.value || currentParserId();
  renderParserSettings(
    "#hiresPromptParserAdvancedContent",
    parserId,
    safeParseJson($("#hiresPromptParserKwargs")?.value, {}),
    (next) => {
      const input = $("#hiresPromptParserKwargs");
      if (input) input.value = JSON.stringify(next);
      saveSessionSoon();
    },
  );
}

export function updateHiresRouting() {
  const parserMode = $("#hiresPromptParserMode")?.value || "same_as_base";
  const profileMode = $("#hiresShortcutProfileMode")?.value || "same_as_base";
  const parserSelect = $("#hiresPromptParserName");
  const profileSelect = $("#hiresShortcutProfileName");
  if (parserMode === "same_as_base" || parserMode === "canonical_only") {
    populateHiresParsers(currentParserId());
    if (parserSelect) parserSelect.disabled = true;
    const kwargs = $("#hiresPromptParserKwargs");
    if (kwargs) kwargs.value = $("#promptParserKwargs")?.value || "{}";
  } else {
    if (parserSelect) parserSelect.disabled = parserMode === "canonical_only";
  }
  if (profileMode === "same_as_base") {
    populateHiresProfiles($("#promptShortcutProfileName")?.value || "");
    if (profileSelect) profileSelect.disabled = true;
  } else if (profileMode === "canonical_only" || parserMode === "canonical_only") {
    populateHiresProfiles("canonical");
    if (profileSelect) {
      profileSelect.value = "canonical";
      profileSelect.disabled = true;
    }
  } else if (profileSelect) {
    profileSelect.disabled = false;
  }
  renderHiresParserSettings();
  const status = $("#hiresPromptRoutingStatus");
  if (status) {
    status.textContent = parserMode === "same_as_base" && profileMode === "same_as_base"
      ? "The second pass inherits the base parser, shortcut profile, and parser options."
      : `Second pass: ${parserMode.replaceAll("_", " ")} parser · ${profileMode.replaceAll("_", " ")} shortcut profile.`;
  }
}

export function normalizedHiresSizeMode(value, enabled = true) {
  return normalizeHiresSizeMode(value, enabled);
}

export function normalizedHiresDimension(value, fallback) {
  return clampHiresDimension(value, fallback);
}

export function updateHiresScheduleSummary() {
  const status = $("#hiresStepSummary");
  if (!status) return;
  const enabled = $("#hiresEnabled")?.checked === true;
  const requested = Math.max(1, Math.min(200, Math.round(Number($("#hiresSteps")?.value || 20))));
  const requestedStrength = Number($("#hiresDenoisingStrength")?.value ?? 0.4);
  const strength = Math.max(0.01, Math.min(1.0, Number.isFinite(requestedStrength) ? requestedStrength : 0.4));
  const policy = String($("#hiresStepPolicy")?.value || "a1111_fixed_steps_v1");
  if (!enabled) {
    status.textContent = "Hires schedule: disabled.";
    status.className = "field-status subtle hires-schedule-summary";
    return;
  }
  if (policy === "proportional_tail_v1") {
    const executed = Math.max(1, Math.min(requested, Math.round(requested * strength)));
    status.textContent = `Legacy proportional tail: ${requested} requested · ${requested} internal schedule transitions · ${executed} executed.`;
    status.className = "field-status warning hires-schedule-summary";
    return;
  }
  const safeStrength = Math.min(Math.max(strength, 0.01), 0.999);
  const internal = Math.max(requested, Math.floor(requested / safeStrength));
  status.textContent = `Fixed executed steps: ${requested} requested · ${internal} internal schedule transitions · ${requested} executed.`;
  status.className = "field-status ready hires-schedule-summary";
}

export function updateHiresPairStatus() {
  const status = $("#hiresPairStatus");
  if (!status) return;
  const enabled = $("#hiresEnabled")?.checked === true;
  if (!enabled) {
    status.textContent = "Hires generation is disabled.";
    status.className = "field-status subtle";
    return;
  }
  const explicitSampler = String($("#hiresSamplerName")?.value || "");
  const explicitScheduler = String($("#hiresSchedulerName")?.value || "");
  const baseSampler = String($("#samplerName")?.value || "");
  const baseScheduler = String($("#schedulerName")?.value || "");
  const sampler = explicitSampler || baseSampler;
  const scheduler = explicitScheduler || baseScheduler;
  const inheritSamplerOption = $("#hiresSamplerName")?.querySelector('option[value=""]');
  const inheritSchedulerOption = $("#hiresSchedulerName")?.querySelector('option[value=""]');
  if (inheritSamplerOption) inheritSamplerOption.textContent = `Inherit base sampler (${baseSampler || "current"})`;
  if (inheritSchedulerOption) inheritSchedulerOption.textContent = `Inherit base scheduler (${baseScheduler || "current"})`;
  const cfg = String($("#hiresCfgScale")?.value || "").trim();
  const cfgRescale = String($("#hiresCfgRescale")?.value || "").trim();
  const inherited = [
    explicitSampler ? "sampler explicit" : "sampler inherited",
    explicitScheduler ? "scheduler explicit" : "scheduler inherited",
    cfg ? `CFG ${cfg}` : "CFG inherited",
    cfgRescale ? `CFG rescale ${cfgRescale}` : "CFG rescale inherited",
  ].join(" · ");
  const experimental = sampler === "kes" || scheduler === "simple_kes";
  status.textContent = `Second pass: ${sampler || "sampler"} + ${scheduler || "scheduler"} · ${inherited}${experimental ? " · Experimental KES comparison path" : ""}.`;
  status.className = experimental ? "field-status warning" : "field-status subtle";
}

export function scheduleHiresPlannerParity(plan, enabled) {
  const toggle = $("#hiresPlannerParityDiagnostics");
  const status = $("#hiresPlannerParityStatus");
  if (hiresPlannerParityTimer) window.clearTimeout(hiresPlannerParityTimer);
  if (!toggle?.checked || !enabled) {
    if (status) {
      status.textContent = toggle?.checked ? "Planner parity diagnostic is waiting for hires to be enabled." : "Planner parity diagnostic is off.";
      status.className = "field-status subtle";
    }
    return;
  }
  const sequence = ++hiresPlannerParitySequence;
  if (status) {
    status.textContent = `Planner parity diagnostic: checking browser plan ${plan.contract_version || "unknown"} against server…`;
    status.className = "field-status subtle";
  }
  hiresPlannerParityTimer = window.setTimeout(async () => {
    try {
      const response = await api.hiresDimensionPlan({
        width: plan.base_width,
        height: plan.base_height,
        hires_size_mode: plan.mode,
        hires_scale: plan.requested_scale,
        hires_width: plan.requested_width,
        hires_height: plan.requested_height,
      });
      if (sequence !== hiresPlannerParitySequence) return;
      const server = response?.plan || {};
      const keys = [
        "contract_version", "mode", "base_width", "base_height", "requested_width",
        "requested_height", "internal_width", "internal_height", "final_width",
        "final_height", "axis_scale_width", "axis_scale_height", "uniform_scale",
        "aspect_ratio_changed", "alignment_applied", "dimension_multiple",
      ];
      const mismatches = keys.filter((key) => JSON.stringify(server[key]) !== JSON.stringify(plan[key]));
      if (status) {
        status.textContent = mismatches.length
          ? `Planner parity mismatch: ${mismatches.join(", ")} · browser ${plan.contract_version || "unknown"} / server ${response?.contract_version || server.contract_version || "unknown"}.`
          : `Planner parity: Match · ${plan.contract_version || "unknown"} · alignment multiple ${plan.dimension_multiple}.`;
        status.className = mismatches.length ? "field-status error" : "field-status ready";
      }
    } catch (error) {
      if (sequence !== hiresPlannerParitySequence) return;
      if (status) {
        status.textContent = `Planner parity diagnostic failed: ${error.message}`;
        status.className = "field-status error";
      }
    }
  }, 300);
}

export function updateHiresSizeControls({ source = "" } = {}) {
  const enabled = $("#hiresEnabled")?.checked === true;
  const sizeMode = $("#hiresSizeMode");
  let mode = normalizedHiresSizeMode(sizeMode?.value, enabled);
  const scaleField = $("#hiresScaleField");
  const widthField = $("#hiresWidthField");
  const heightField = $("#hiresHeightField");
  const scaleInput = $("#hiresScale");
  const widthInput = $("#hiresWidth");
  const heightInput = $("#hiresHeight");

  if (enabled && mode === "scale_from_base" && (source === "width" || source === "height")) {
    mode = "explicit_dimensions";
    if (sizeMode) sizeMode.value = mode;
  } else if (sizeMode && sizeMode.value !== mode) {
    sizeMode.value = mode;
  }

  [
    "#hiresPromptParserMode", "#hiresPromptParserName", "#hiresShortcutProfileMode",
    "#hiresShortcutProfileName", "#hiresPositivePrompt", "#hiresNegativePrompt",
    "#hiresSizeMode", "#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy",
    "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale",
    "#hiresUpscaler", "#hiresSaveLowres", "#hiresAspectPolicy", "#hiresPaddingMode",
    "#hiresBlurredEdgeMethod", "#hiresFinalSizeCorrectionFilter", "#hiresCorrectionFingerprintDiagnostics",
    "#hiresBlurredEdgeCompareDiagnostics",
  ].forEach((selector) => {
    const node = $(selector);
    if (node) node.disabled = !enabled;
  });

  if (scaleInput) scaleInput.disabled = !enabled || mode !== "scale_from_base";
  if (widthInput) widthInput.disabled = !enabled;
  if (heightInput) heightInput.disabled = !enabled;
  scaleField?.classList.toggle("is-disabled", !enabled || mode !== "scale_from_base");
  widthField?.classList.toggle("is-disabled", !enabled);
  heightField?.classList.toggle("is-disabled", !enabled);
  widthField?.classList.toggle("is-calculated", enabled && mode === "scale_from_base");
  heightField?.classList.toggle("is-calculated", enabled && mode === "scale_from_base");

  const plan = planHiresDimensions({
    baseWidth: $("#width")?.value,
    baseHeight: $("#height")?.value,
    mode,
    scale: scaleInput?.value || 1.5,
    targetWidth: widthInput?.value,
    targetHeight: heightInput?.value,
    enabled,
  });

  if (enabled && mode === "scale_from_base") {
    if (widthInput) widthInput.value = String(plan.requested_width);
    if (heightInput) heightInput.value = String(plan.requested_height);
  } else if (enabled && mode === "explicit_dimensions") {
    if (plan.uniform_scale !== null && scaleInput) {
      scaleInput.value = String(plan.uniform_scale);
    } else if (scaleInput) {
      scaleInput.value = "";
    }
  }

  const status = $("#hiresSizeStatus");
  if (status) status.textContent = enabled
    ? `Requested target: ${plan.requested_width} × ${plan.requested_height} · ${mode === "scale_from_base" ? "scale from base" : "exact target dimensions"}.`
    : "Hires generation is disabled.";

  const effectiveStatus = $("#hiresEffectiveScaleStatus");
  if (effectiveStatus) {
    if (!enabled) {
      effectiveStatus.textContent = "Effective scaling: disabled.";
      effectiveStatus.className = "field-status subtle";
    } else if (plan.is_uniform_scale) {
      effectiveStatus.textContent = `Effective scaling: Uniform ${Number(plan.uniform_scale).toFixed(6)}x.`;
      effectiveStatus.className = "field-status subtle";
    } else {
      effectiveStatus.textContent = `Effective scaling: Width ${plan.axis_scale_width.toFixed(6)}x / Height ${plan.axis_scale_height.toFixed(6)}x · Aspect ratio changed.`;
      effectiveStatus.className = "field-status warning";
    }
  }

  const alignmentStatus = $("#hiresAlignmentStatus");
  if (alignmentStatus) {
    alignmentStatus.textContent = enabled
      ? `Dimension plan: Requested ${plan.requested_width} × ${plan.requested_height} · Internal aligned ${plan.internal_width} × ${plan.internal_height} · Final target ${plan.final_width} × ${plan.final_height}${plan.alignment_applied ? " · alignment correction required" : " · no alignment correction"}.`
      : "Dimension plan: disabled.";
    alignmentStatus.className = plan.alignment_applied && enabled ? "field-status warning" : "field-status subtle";
  }

  updateHiresUpscalerPlanUI(plan, enabled);
  scheduleHiresPlannerParity(plan, enabled);
  const correctionFingerprintStatus = $("#hiresCorrectionFingerprintStatus");
  if (correctionFingerprintStatus) {
    const fingerprintEnabled = Boolean($("#hiresCorrectionFingerprintDiagnostics")?.checked);
    correctionFingerprintStatus.textContent = fingerprintEnabled
      ? "Correction fingerprint enabled: a deterministic metadata-only SHA-256 will be recorded for this hires correction contract."
      : "Correction fingerprint is off. Enabling it hashes only the deterministic correction contract; it does not add an image-processing pass.";
    correctionFingerprintStatus.className = fingerprintEnabled ? "field-status ready" : "field-status subtle";
  }
  const blurredEdgeCompareStatus = $("#hiresBlurredEdgeCompareStatus");
  if (blurredEdgeCompareStatus) {
    const compareEnabled = Boolean($("#hiresBlurredEdgeCompareDiagnostics")?.checked);
    blurredEdgeCompareStatus.textContent = compareEnabled
      ? "Blurred-edge comparison diagnostic enabled: the selected method and the alternate method will both run so diagnostics can record timing and quality-proxy deltas."
      : "Blurred-edge comparison diagnostic is off. Enable it only when you want side-by-side timing and quality-proxy metadata.";
    blurredEdgeCompareStatus.className = compareEnabled ? "field-status ready" : "field-status subtle";
  }
  updateHiresScheduleSummary();
  updateHiresPairStatus();
}

export function initializeHiresPrompt(current = {}) {
  if ($("#hiresEnabled")) $("#hiresEnabled").checked = Boolean(current.hires_enabled);
  if ($("#hiresPromptParserMode")) $("#hiresPromptParserMode").value = current.hires_prompt_parser_mode || "same_as_base";
  populateHiresParsers(current.hires_prompt_parser_name || current.prompt_parser_name || "legacy");
  if ($("#hiresPromptParserKwargs")) $("#hiresPromptParserKwargs").value = JSON.stringify(current.hires_prompt_parser_kwargs || current.prompt_parser_kwargs || {});
  if ($("#hiresShortcutProfileMode")) $("#hiresShortcutProfileMode").value = current.hires_shortcut_profile_mode || "same_as_base";
  populateHiresProfiles(current.hires_shortcut_profile_name || current.prompt_shortcut_profile_name || "");
  if ($("#hiresShortcutProfileSnapshot")) $("#hiresShortcutProfileSnapshot").value = JSON.stringify(current.hires_shortcut_profile_snapshot || {});
  if ($("#hiresPositivePrompt")) $("#hiresPositivePrompt").value = current.hires_positive_prompt || "";
  if ($("#hiresNegativePrompt")) $("#hiresNegativePrompt").value = current.hires_negative_prompt || "";
  if ($("#hiresSizeMode")) {
    $("#hiresSizeMode").value = normalizedHiresSizeMode(current.hires_size_mode, Boolean(current.hires_enabled));
  }
  if ($("#hiresScale")) $("#hiresScale").value = String(current.hires_scale || 1.5);
  if ($("#hiresWidth")) $("#hiresWidth").value = String(current.hires_width || Math.max(64, Number($("#width")?.value || 640) * 1.5));
  if ($("#hiresHeight")) $("#hiresHeight").value = String(current.hires_height || Math.max(64, Number($("#height")?.value || 960) * 1.5));
  if ($("#hiresSteps")) $("#hiresSteps").value = String(current.hires_steps || 20);
  if ($("#hiresDenoisingStrength")) $("#hiresDenoisingStrength").value = String(current.hires_denoising_strength ?? 0.4);
  if ($("#hiresStepPolicy")) $("#hiresStepPolicy").value = current.hires_step_policy || "a1111_fixed_steps_v1";
  if ($("#hiresSamplerName")) $("#hiresSamplerName").value = current.hires_sampler_name || "";
  if ($("#hiresSchedulerName")) $("#hiresSchedulerName").value = current.hires_scheduler_name || "";
  if ($("#hiresCfgScale")) $("#hiresCfgScale").value = current.hires_cfg_scale ?? "";
  if ($("#hiresCfgRescale")) $("#hiresCfgRescale").value = current.hires_cfg_rescale ?? "";
  normalizeHiresCfgRescaleInput();
  if ($("#hiresUpscaler")) $("#hiresUpscaler").value = current.hires_upscaler_id || current.hires_upscaler || "";
  if ($("#hiresAspectPolicy")) $("#hiresAspectPolicy").value = current.hires_aspect_policy || "stretch";
  if ($("#hiresPaddingMode")) $("#hiresPaddingMode").value = current.hires_padding_mode || "reflect";
  if ($("#hiresBlurredEdgeMethod")) $("#hiresBlurredEdgeMethod").value = current.hires_blurred_edge_method || "box";
  if ($("#hiresFinalSizeCorrectionFilter")) $("#hiresFinalSizeCorrectionFilter").value = current.hires_final_size_correction_filter || current.hires_exact_resize_filter || "auto";
  if ($("#hiresPlannerParityDiagnostics")) $("#hiresPlannerParityDiagnostics").checked = false;
  if ($("#hiresCorrectionFingerprintDiagnostics")) $("#hiresCorrectionFingerprintDiagnostics").checked = Boolean(current.hires_correction_fingerprint_enabled);
  if ($("#hiresBlurredEdgeCompareDiagnostics")) $("#hiresBlurredEdgeCompareDiagnostics").checked = Boolean(current.hires_blurred_edge_compare_diagnostics);
  if ($("#hiresSaveLowres")) $("#hiresSaveLowres").checked = Boolean(current.hires_save_lowres);
  updateHiresSizeControls();
  updateHiresRouting();
}

export function bindHiresPrompt() {
  $("#hiresPromptParserMode")?.addEventListener("change", () => { updateHiresRouting(); saveSessionSoon(); });
  $("#hiresPromptParserName")?.addEventListener("change", () => {
    const kwargs = $("#hiresPromptParserKwargs");
    if (kwargs) kwargs.value = JSON.stringify((parserDefaultOptions($("#hiresPromptParserName")?.value)));
    populateHiresProfiles(defaultProfileId($("#hiresPromptParserName")?.value));
    renderHiresParserSettings();
    saveSessionSoon();
  });
  $("#hiresShortcutProfileMode")?.addEventListener("change", () => { updateHiresRouting(); saveSessionSoon(); });
  $("#hiresShortcutProfileName")?.addEventListener("change", () => {
    const snapshot = $("#hiresShortcutProfileSnapshot");
    if (snapshot) snapshot.value = JSON.stringify(profileSnapshot(profileById($("#hiresShortcutProfileName")?.value)));
    saveSessionSoon();
  });
  ["#hiresPositivePrompt", "#hiresNegativePrompt"].forEach((selector) => {
    $(selector)?.addEventListener("input", saveSessionSoon);
  });
  const bindHiresDimensionEvent = (selector, source) => {
    $(selector)?.addEventListener("input", () => { updateHiresSizeControls({ source }); saveSessionSoon(); });
    $(selector)?.addEventListener("change", () => { updateHiresSizeControls({ source }); saveSessionSoon(); });
  };
  bindHiresDimensionEvent("#hiresScale", "scale");
  bindHiresDimensionEvent("#hiresWidth", "width");
  bindHiresDimensionEvent("#hiresHeight", "height");
  bindHiresDimensionEvent("#hiresSizeMode", "mode");
  bindHiresDimensionEvent("#hiresEnabled", "enabled");
  bindHiresDimensionEvent("#width", "base");
  bindHiresDimensionEvent("#height", "base");
  ["#hiresSteps", "#hiresDenoisingStrength", "#hiresStepPolicy", "#hiresSamplerName", "#hiresSchedulerName", "#hiresCfgScale", "#hiresCfgRescale", "#hiresUpscaler", "#hiresAspectPolicy", "#hiresPaddingMode", "#hiresBlurredEdgeMethod", "#hiresFinalSizeCorrectionFilter", "#hiresSaveLowres", "#samplerName", "#schedulerName", "#cfgScale"].forEach((selector) => {
    $(selector)?.addEventListener("input", () => { updateHiresSizeControls(); saveSessionSoon(); });
    $(selector)?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  });
  $("#hiresCfgRescale")?.addEventListener("change", () => { normalizeHiresCfgRescaleInput(); updateHiresSizeControls(); saveSessionSoon(); });
  $("#hiresCfgRescale")?.addEventListener("blur", () => { normalizeHiresCfgRescaleInput(); updateHiresSizeControls(); saveSessionSoon(); });
  $("#hiresPlannerParityDiagnostics")?.addEventListener("change", () => updateHiresSizeControls());
  $("#hiresCorrectionFingerprintDiagnostics")?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  $("#hiresBlurredEdgeCompareDiagnostics")?.addEventListener("change", () => { updateHiresSizeControls(); saveSessionSoon(); });
  window.addEventListener("image-gen-hires-upscaler-change", () => updateHiresSizeControls());
}
