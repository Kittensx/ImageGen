import { api } from "../api.js?v=theme-manager-tm02";
import { state } from "../state.js";
import { $, notify } from "../utils.js";
import { nearestColorName } from "../color-names.js";
import { renderRuntimeStartupStatus } from "./memory-status.js?v=0.1.62";

let runtimeOverridesInheritMode = false;

const RUNTIME_JOB_CONTROL_IDS = Object.freeze([
  "dialogMemoryPolicy",
  "dialogMemorySafetyMargin",
  "dialogAttentionSlicing",
  "dialogHiresMemoryProfile",
  "dialogPreHiresCleanup",
  "dialogPreviewPolicy",
  "dialogVaeTiling",
  "dialogVaeSlicing",
  "dialogVaeDevice",
  "dialogOomRetryProfile",
  "dialogOomRetryLimit",
  "dialogRetainCheckpoint",
  "dialogRetainVae",
  "dialogRetainTextEncoder",
]);

const FONT_FAMILY_STACKS = Object.freeze({
  "Inter": 'Inter, "Segoe UI", Arial, sans-serif',
  "System UI": 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  "Segoe UI": '"Segoe UI", Arial, sans-serif',
  "Arial": 'Arial, Helvetica, sans-serif',
  "Verdana": 'Verdana, Geneva, sans-serif',
  "Tahoma": 'Tahoma, Geneva, sans-serif',
  "Trebuchet MS": '"Trebuchet MS", Arial, sans-serif',
  "Georgia": 'Georgia, "Times New Roman", serif',
  "Monospace": 'Consolas, "SFMono-Regular", monospace',
});

const DEFAULT_THEME_PALETTE = Object.freeze({
  accent: Object.freeze({ name: "Sky Blue", color: "#179ee7" }),
  surface: Object.freeze({ name: "Charcoal", color: "#111d29" }),
  typography: Object.freeze({
    font_family: "Inter",
    primary_button_text: "#ffffff",
    secondary_button_text: "#d5f1ff",
  }),
  semantic: Object.freeze({
    surface_secondary: "#172431",
    component_surface: "#111d29",
    component_border: "#2b4358",
    component_accent: "#179ee7",
    text_primary: "#f4f9fd",
    text_secondary: "#9db2c4",
  }),
});

function clonePalette(palette) {
  return {
    accent: { ...DEFAULT_THEME_PALETTE.accent, ...(palette?.accent || {}) },
    surface: { ...DEFAULT_THEME_PALETTE.surface, ...(palette?.surface || {}) },
    typography: { ...DEFAULT_THEME_PALETTE.typography, ...(palette?.typography || {}) },
    semantic: { ...DEFAULT_THEME_PALETTE.semantic, ...(palette?.semantic || {}) },
  };
}

function normalizeHex(value, fallback) {
  const text = String(value || "").trim();
  const short = /^#?([0-9a-f]{3})$/i.exec(text);
  if (short) {
    return `#${short[1].split("").map((item) => item + item).join("")}`.toLowerCase();
  }
  const full = /^#?([0-9a-f]{6})$/i.exec(text);
  return full ? `#${full[1].toLowerCase()}` : fallback;
}

function normalizePalette(value = {}) {
  const output = clonePalette(DEFAULT_THEME_PALETTE);
  ["accent", "surface"].forEach((kind) => {
    const candidate = value?.[kind];
    if (!candidate || typeof candidate !== "object") return;
    output[kind].name = String(candidate.name || output[kind].name).trim().slice(0, 40) || output[kind].name;
    output[kind].color = normalizeHex(candidate.color, output[kind].color);
  });

  const typography = value?.typography;
  if (typography && typeof typography === "object") {
    const fontFamily = String(typography.font_family || output.typography.font_family);
    output.typography.font_family = Object.prototype.hasOwnProperty.call(FONT_FAMILY_STACKS, fontFamily)
      ? fontFamily
      : output.typography.font_family;
    output.typography.primary_button_text = normalizeHex(
      typography.primary_button_text,
      output.typography.primary_button_text,
    );
    output.typography.secondary_button_text = normalizeHex(
      typography.secondary_button_text,
      output.typography.secondary_button_text,
    );
  }
  const semantic = value?.semantic && typeof value.semantic === "object" ? value.semantic : {};
  const derivedSemantic = {
    surface_secondary: mixHex(output.surface.color, "#ffffff", 0.08),
    component_surface: output.surface.color,
    component_border: mixHex(output.surface.color, "#ffffff", 0.22),
    component_accent: output.accent.color,
    text_primary: DEFAULT_THEME_PALETTE.semantic.text_primary,
    text_secondary: DEFAULT_THEME_PALETTE.semantic.text_secondary,
  };
  Object.keys(derivedSemantic).forEach((key) => {
    output.semantic[key] = normalizeHex(semantic[key], derivedSemantic[key]);
  });
  return output;
}

function hexToRgb(hex) {
  const value = normalizeHex(hex, "#000000").slice(1);
  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16),
  };
}

function rgbToHex({ r, g, b }) {
  const channel = (value) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, "0");
  return `#${channel(r)}${channel(g)}${channel(b)}`;
}

function mixHex(source, target, targetWeight) {
  const left = hexToRgb(source);
  const right = hexToRgb(target);
  const weight = Math.max(0, Math.min(1, Number(targetWeight) || 0));
  return rgbToHex({
    r: left.r + ((right.r - left.r) * weight),
    g: left.g + ((right.g - left.g) * weight),
    b: left.b + ((right.b - left.b) * weight),
  });
}

function relativeLuminance(hex) {
  const channels = Object.values(hexToRgb(hex)).map((value) => {
    const normalized = value / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2]);
}

function contrastRatio(left, right) {
  const a = relativeLuminance(left);
  const b = relativeLuminance(right);
  const lighter = Math.max(a, b);
  const darker = Math.min(a, b);
  return (lighter + 0.05) / (darker + 0.05);
}

function applyThemeVariables(palette) {
  const root = document.documentElement.style;
  const accent = palette.accent.color;
  const surface = palette.surface.color;

  root.setProperty("--sky-50", mixHex(accent, "#ffffff", 0.92));
  root.setProperty("--sky-100", mixHex(accent, "#ffffff", 0.82));
  root.setProperty("--sky-200", mixHex(accent, "#ffffff", 0.62));
  root.setProperty("--sky-300", mixHex(accent, "#ffffff", 0.38));
  root.setProperty("--sky-400", mixHex(accent, "#ffffff", 0.18));
  root.setProperty("--sky-500", accent);
  root.setProperty("--sky-600", mixHex(accent, "#000000", 0.18));

  root.setProperty("--theme-surface-primary", surface);
  root.setProperty("--theme-surface-secondary", palette.semantic.surface_secondary);
  root.setProperty("--component-surface", palette.semantic.component_surface);
  root.setProperty("--component-border", palette.semantic.component_border);
  root.setProperty("--component-accent", palette.semantic.component_accent);
  root.setProperty("--charcoal-950", mixHex(surface, "#000000", 0.48));
  root.setProperty("--charcoal-900", mixHex(surface, "#000000", 0.24));
  root.setProperty("--charcoal-850", palette.semantic.component_surface);
  root.setProperty("--charcoal-800", palette.semantic.surface_secondary);
  root.setProperty("--charcoal-700", mixHex(palette.semantic.surface_secondary, "#ffffff", 0.12));
  root.setProperty("--line", palette.semantic.component_border);
  root.setProperty("--text", palette.semantic.text_primary);
  root.setProperty("--muted", palette.semantic.text_secondary);
  root.setProperty("--text-primary", palette.semantic.text_primary);
  root.setProperty("--text-secondary", palette.semantic.text_secondary);
  root.setProperty("--text-muted", palette.semantic.text_secondary);
  root.setProperty("--surface", surface);
  root.setProperty("--surface-1", palette.semantic.component_surface);
  root.setProperty("--surface-2", palette.semantic.surface_secondary);
  root.setProperty("--surface-3", mixHex(palette.semantic.surface_secondary, "#ffffff", 0.10));
  root.setProperty("--surface-raised", palette.semantic.surface_secondary);

  root.setProperty("--font-ui", FONT_FAMILY_STACKS[palette.typography.font_family]);
  root.setProperty("--primary-button-text", palette.typography.primary_button_text);
  root.setProperty("--secondary-button-text", palette.typography.secondary_button_text);
}

function updateThemeLabels(palette) {
  if ($("#accentThemeButton")) $("#accentThemeButton").textContent = palette.accent.name;
  if ($("#surfaceThemeButton")) $("#surfaceThemeButton").textContent = palette.surface.name;
  if ($("#accentThemePreview")) $("#accentThemePreview").style.background = palette.accent.color;
  if ($("#surfaceThemePreview")) $("#surfaceThemePreview").style.background = palette.surface.color;
}

const FORCED_LIVE_PREVIEW_MODE = "fast";

export function applyThemePalette(value, { persist = false } = {}) {
  const runtime = window.ImageGenThemeRuntime;
  const palette = runtime?.apply
    ? runtime.apply(value, { persist })
    : normalizePalette(value);
  if (!runtime?.apply) applyThemeVariables(palette);
  updateThemeLabels(palette);
  return palette;
}

export function applyUiScale(value) {
  const scale = Math.max(80, Math.min(150, Number(value) || 100));
  document.documentElement.style.setProperty("--ui-scale", String(scale / 100));
  if ($("#uiScale")) $("#uiScale").value = scale;
  if ($("#dialogUiScale")) $("#dialogUiScale").value = scale;
  if ($("#uiScaleOutput")) $("#uiScaleOutput").textContent = `${scale}%`;
}

function updateAccuratePreviewWarning() {
  const mode = $("#dialogLivePreviewMode");
  const warning = $("#accuratePreviewWarning");
  if (!mode || !warning) return;
  warning.classList.toggle("is-hidden", mode.value !== "accurate");
}

function enforceLivePreviewModeLock() {
  const mode = $("#dialogLivePreviewMode");
  const note = $("#livePreviewModeLockNote");
  if (!mode) return;
  mode.value = FORCED_LIVE_PREVIEW_MODE;
  Array.from(mode.options || []).forEach((option) => {
    option.disabled = option.value !== FORCED_LIVE_PREVIEW_MODE;
  });
  mode.setAttribute("aria-disabled", "true");
  mode.title = "Temporarily locked to A1111-style fast preview.";
  if (note) {
    note.textContent = "Preview mode is temporarily locked to the lower-overhead A1111-style fast preview path while preview performance is being qualified.";
  }
}

function setThemeEditorValues(palette) {
  ["accent", "surface"].forEach((kind) => {
    const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
    const entry = palette[kind];
    const name = $(`#${prefix}Name`);
    const color = $(`#${prefix}Color`);
    const hex = $(`#${prefix}Hex`);
    if (name) name.value = entry.name;
    if (color) color.value = entry.color;
    if (hex) hex.value = entry.color;
  });

  if ($("#themeFontFamily")) $("#themeFontFamily").value = palette.typography.font_family;
  if ($("#themePrimaryButtonTextColor")) $("#themePrimaryButtonTextColor").value = palette.typography.primary_button_text;
  if ($("#themePrimaryButtonTextHex")) $("#themePrimaryButtonTextHex").value = palette.typography.primary_button_text;
  if ($("#themeSecondaryButtonTextColor")) $("#themeSecondaryButtonTextColor").value = palette.typography.secondary_button_text;
  if ($("#themeSecondaryButtonTextHex")) $("#themeSecondaryButtonTextHex").value = palette.typography.secondary_button_text;
  const semanticControls = {
    surface_secondary: ["themeSurfaceSecondaryColor", "themeSurfaceSecondaryHex"],
    component_surface: ["themeComponentSurfaceColor", "themeComponentSurfaceHex"],
    component_border: ["themeComponentBorderColor", "themeComponentBorderHex"],
    component_accent: ["themeComponentAccentColor", "themeComponentAccentHex"],
    text_primary: ["themeTextPrimaryColor", "themeTextPrimaryHex"],
    text_secondary: ["themeTextSecondaryColor", "themeTextSecondaryHex"],
  };
  Object.entries(semanticControls).forEach(([key, ids]) => {
    ids.forEach((id) => { if ($(`#${id}`)) $(`#${id}`).value = palette.semantic[key]; });
  });
}

function readThemeEditorValues(fallback) {
  const palette = clonePalette(fallback);
  ["accent", "surface"].forEach((kind) => {
    const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
    palette[kind] = {
      name: String($(`#${prefix}Name`)?.value || palette[kind].name).trim().slice(0, 40) || palette[kind].name,
      color: normalizeHex($(`#${prefix}Hex`)?.value || $(`#${prefix}Color`)?.value, palette[kind].color),
    };
  });

  const selectedFont = String($("#themeFontFamily")?.value || palette.typography.font_family);
  palette.typography = {
    font_family: Object.prototype.hasOwnProperty.call(FONT_FAMILY_STACKS, selectedFont)
      ? selectedFont
      : palette.typography.font_family,
    primary_button_text: normalizeHex(
      $("#themePrimaryButtonTextHex")?.value || $("#themePrimaryButtonTextColor")?.value,
      palette.typography.primary_button_text,
    ),
    secondary_button_text: normalizeHex(
      $("#themeSecondaryButtonTextHex")?.value || $("#themeSecondaryButtonTextColor")?.value,
      palette.typography.secondary_button_text,
    ),
  };
  const semanticControls = {
    surface_secondary: ["themeSurfaceSecondaryHex", "themeSurfaceSecondaryColor"],
    component_surface: ["themeComponentSurfaceHex", "themeComponentSurfaceColor"],
    component_border: ["themeComponentBorderHex", "themeComponentBorderColor"],
    component_accent: ["themeComponentAccentHex", "themeComponentAccentColor"],
    text_primary: ["themeTextPrimaryHex", "themeTextPrimaryColor"],
    text_secondary: ["themeTextSecondaryHex", "themeTextSecondaryColor"],
  };
  palette.semantic = { ...palette.semantic };
  Object.entries(semanticControls).forEach(([key, ids]) => {
    palette.semantic[key] = normalizeHex($(`#${ids[0]}`)?.value || $(`#${ids[1]}`)?.value, palette.semantic[key]);
  });
  return palette;
}

function validateThemeContrast(palette) {
  const readability = [
    ["Primary text", palette.semantic.text_primary, "primary surface", palette.surface.color, 4.5],
    ["Primary text", palette.semantic.text_primary, "secondary surface", palette.semantic.surface_secondary, 4.5],
    ["Primary text", palette.semantic.text_primary, "component surface", palette.semantic.component_surface, 4.5],
    ["Secondary/muted text", palette.semantic.text_secondary, "primary surface", palette.surface.color, 4.5],
    ["Secondary/muted text", palette.semantic.text_secondary, "secondary surface", palette.semantic.surface_secondary, 4.5],
    ["Secondary/muted text", palette.semantic.text_secondary, "component surface", palette.semantic.component_surface, 4.5],
  ];
  const advisory = [
    ["Component border", palette.semantic.component_border, "component surface", palette.semantic.component_surface, 3.0],
    ["Component accent", palette.semantic.component_accent, "component surface", palette.semantic.component_surface, 3.0],
    ["Accent", palette.accent.color, "primary surface", palette.surface.color, 3.0],
    ["Primary button text", palette.typography.primary_button_text, "accent", palette.accent.color, 3.0],
    ["Secondary button text", palette.typography.secondary_button_text, "secondary surface", palette.semantic.surface_secondary, 3.0],
  ];
  const textWarnings = [];
  const advisoryWarnings = [];
  const checks = [];
  const evaluate = (entry, category) => {
    const [label, foreground, backgroundLabel, background, minimum] = entry;
    const ratio = contrastRatio(foreground, background);
    const identical = String(foreground).toLowerCase() === String(background).toLowerCase();
    const valid = !identical && ratio >= minimum;
    let message = `${label} vs ${backgroundLabel}: ${ratio.toFixed(2)}:1 (recommended ${minimum.toFixed(1)}:1).`;
    if (identical) message = `${label} and ${backgroundLabel} use the same color and may be unreadable.`;
    checks.push({ label, backgroundLabel, ratio, minimum, valid, category, message });
    if (!valid) (category === "readability" ? textWarnings : advisoryWarnings).push(message);
  };
  readability.forEach((entry) => evaluate(entry, "readability"));
  advisory.forEach((entry) => evaluate(entry, "advisory"));
  return { valid: true, textWarnings, advisoryWarnings, warnings: [...textWarnings, ...advisoryWarnings], checks };
}

function updateThemeWarning(palette) {
  const warning = $("#themePaletteWarning");
  const report = $("#themeContrastReport");
  const result = validateThemeContrast(palette);
  const messages = [...result.warnings];
  if (relativeLuminance(palette.surface.color) > 0.22) {
    messages.push("The primary surface is fairly bright; inspect all controls in the live preview before saving.");
  }
  if (relativeLuminance(palette.accent.color) < 0.035) {
    messages.push("The accent is very dark and may make highlighted controls difficult to distinguish.");
  }
  if (warning) {
    warning.textContent = messages.join(" ");
    warning.classList.toggle("is-hidden", messages.length === 0);
    warning.classList.remove("is-error");
  }
  if (report) {
    report.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = result.textWarnings.length ? "Low contrast warning" : "Text contrast looks readable";
    report.appendChild(heading);
    const summary = document.createElement("span");
    summary.textContent = result.textWarnings.length
      ? ` ${result.textWarnings[0]} Saving is allowed, but the text may be difficult or impossible to read.`
      : " Primary and secondary text meet the recommended 4.5:1 contrast target on all theme surfaces.";
    report.appendChild(summary);
    report.classList.remove("is-invalid");
    report.classList.toggle("is-warning", result.textWarnings.length > 0);
    report.classList.toggle("is-valid", result.textWarnings.length === 0);
  }
  const saveButton = $("#saveThemePaletteButton");
  if (saveButton) {
    saveButton.disabled = false;
    saveButton.title = result.textWarnings.length
      ? "Save is allowed; low-contrast colors may make text unreadable."
      : "Save the current theme";
  }
  return result;
}

function confirmLowContrastIfNeeded(contrast, warningsEnabled, actionLabel = "apply this theme") {
  if (!warningsEnabled || !contrast?.textWarnings?.length) return true;
  const first = contrast.textWarnings[0];
  const more = contrast.textWarnings.length > 1 ? `\n\nThere are ${contrast.textWarnings.length} text contrast warnings.` : "";
  return window.confirm(`This theme may make text unreadable.\n\n${first}${more}\n\nAre you sure you want to ${actionLabel}?`);
}

function bindThemePalette(settings) {
  const dialog = $("#themePaletteDialog");
  if (!dialog) return;

  let savedPalette = applyThemePalette(settings.theme_effective_palette || settings.theme_palette || DEFAULT_THEME_PALETTE, { persist: true });
  let workingPalette = clonePalette(savedPalette);
  let savedDuringOpen = false;
  let contrastWarningsEnabled = settings.theme_contrast_warnings_enabled !== false;
  const manualNameEdits = { accent: false, surface: false };
  const warningPreference = $("#themeContrastWarningsEnabled");
  if (warningPreference) warningPreference.checked = contrastWarningsEnabled;

  const renderThemeLibrary = (payload) => {
    const list = $("#themeLocalPackageList");
    const status = $("#themeLocalPackageStatus");
    if (!list) return;
    list.replaceChildren();
    const packages = Array.isArray(payload?.packages) ? payload.packages : [];
    packages.forEach((record) => {
      const row = document.createElement("article");
      row.className = "theme-package-row";
      const copy = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = String(record.name || record.packageId || "Theme package");
      const meta = document.createElement("small");
      const synthetic = record.packageId === "imagegen.local.legacy-palette";
      const packageContrastWarnings = Array.isArray(record.contrastWarnings) ? record.contrastWarnings : [];
      meta.textContent = synthetic
        ? "Built-in custom palette fallback"
        : `${record.installedVersion || ""} · ${record.type || "theme"} · ${record.verificationState || "unverified"}${packageContrastWarnings.length ? " · low-contrast warning" : ""}`;
      copy.append(title, meta);
      const actions = document.createElement("div");
      actions.className = "theme-package-actions";
      const enabled = record.enabledState === true;
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = enabled ? "secondary-button" : "primary-button";
      toggle.textContent = enabled ? "Active" : "Use";
      toggle.disabled = enabled;
      toggle.addEventListener("click", async () => {
        try {
          const packageContrastWarnings = Array.isArray(record.contrastWarnings) ? record.contrastWarnings : [];
          if (contrastWarningsEnabled && packageContrastWarnings.length) {
            const accepted = window.confirm(`This theme package may make text unreadable.\n\n${packageContrastWarnings[0]}\n\nAre you sure you want to apply this theme?`);
            if (!accepted) return;
          }
          toggle.disabled = true;
          const result = await api.enableThemePackage(record.packageId);
          savedPalette = applyThemePalette(result.effectivePalette || savedPalette, { persist: true });
          workingPalette = clonePalette(savedPalette);
          setThemeEditorValues(workingPalette);
          updateThemeWarning(workingPalette);
          renderThemeLibrary(result.library);
          notify(`Theme ${record.name || record.packageId} activated.`);
        } catch (error) {
          notify(error.message, "error");
          toggle.disabled = false;
        }
      });
      actions.appendChild(toggle);
      if (!synthetic) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary-button";
        remove.textContent = "Remove";
        remove.addEventListener("click", async () => {
          try {
            remove.disabled = true;
            const result = await api.removeThemePackage(record.packageId);
            savedPalette = applyThemePalette(result.effectivePalette || savedPalette, { persist: true });
            workingPalette = clonePalette(savedPalette);
            setThemeEditorValues(workingPalette);
            updateThemeWarning(workingPalette);
            renderThemeLibrary(result.library);
            notify(`Theme package ${record.name || record.packageId} removed.`);
          } catch (error) {
            notify(error.message, "error");
            remove.disabled = false;
          }
        });
        actions.appendChild(remove);
      }
      row.append(copy, actions);
      list.appendChild(row);
    });
    if (status) {
      const root = payload?.storage?.theme_library_root || payload?.storage?.themeLibraryRoot || "external theme library";
      status.textContent = `${packages.length} appearance source${packages.length === 1 ? "" : "s"} · ${root}`;
    }
  };

  const refreshThemeLibrary = async () => {
    try {
      renderThemeLibrary(await api.themeLibrary());
    } catch (error) {
      const status = $("#themeLocalPackageStatus");
      if (status) status.textContent = `Local theme library unavailable: ${error.message}`;
    }
  };

  const updateSuggestedNameHint = (kind, suggestedName = null) => {
    const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
    const hint = $(`#${prefix}NameHint`);
    if (!hint) return;
    const suggestion = suggestedName || nearestColorName($(`#${prefix}Hex`)?.value || workingPalette[kind].color);
    hint.textContent = manualNameEdits[kind]
      ? `Custom name retained. Closest named color: ${suggestion}.`
      : `Automatically matched to the closest named color: ${suggestion}.`;
  };

  const suggestThemeName = (kind, color, { force = false } = {}) => {
    if (manualNameEdits[kind] && !force) {
      updateSuggestedNameHint(kind);
      return;
    }
    const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
    const suggestion = nearestColorName(color);
    const nameInput = $(`#${prefix}Name`);
    if (nameInput) nameInput.value = suggestion;
    manualNameEdits[kind] = false;
    updateSuggestedNameHint(kind, suggestion);
  };

  const previewEditor = () => {
    workingPalette = readThemeEditorValues(workingPalette);
    applyThemePalette(workingPalette);
    updateThemeWarning(workingPalette);
  };

  const openEditor = (kind) => {
    savedDuringOpen = false;
    workingPalette = clonePalette(savedPalette);
    manualNameEdits.accent = false;
    manualNameEdits.surface = false;
    setThemeEditorValues(workingPalette);
    updateSuggestedNameHint("accent");
    updateSuggestedNameHint("surface");
    applyThemePalette(workingPalette);
    updateThemeWarning(workingPalette);
    void refreshThemeLibrary();
    if (!dialog.open) dialog.showModal();
    window.setTimeout(() => {
      const section = kind === "surface" ? $("#surfaceThemeSection") : $("#accentThemeSection");
      section?.scrollIntoView({ block: "center" });
      section?.querySelector("input[type=text]")?.focus();
    }, 0);
  };

  $("#accentThemeButton")?.addEventListener("click", () => openEditor("accent"));
  $("#surfaceThemeButton")?.addEventListener("click", () => openEditor("surface"));
  $("#openThemeManagerButton")?.addEventListener("click", () => openEditor("accent"));

  ["accent", "surface"].forEach((kind) => {
    const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
    const nameInput = $(`#${prefix}Name`);
    const colorInput = $(`#${prefix}Color`);
    const hexInput = $(`#${prefix}Hex`);

    nameInput?.addEventListener("input", () => {
      manualNameEdits[kind] = Boolean(nameInput.value.trim());
      if (!manualNameEdits[kind]) suggestThemeName(kind, hexInput?.value || colorInput?.value, { force: true });
      updateSuggestedNameHint(kind);
      previewEditor();
    });
    $(`#${prefix}SuggestName`)?.addEventListener("click", () => {
      suggestThemeName(kind, hexInput?.value || colorInput?.value, { force: true });
      previewEditor();
    });
    colorInput?.addEventListener("input", () => {
      if (hexInput) hexInput.value = colorInput.value;
      suggestThemeName(kind, colorInput.value);
      previewEditor();
    });
    hexInput?.addEventListener("input", () => {
      const normalized = normalizeHex(hexInput.value, "");
      if (!normalized) return;
      if (colorInput) colorInput.value = normalized;
      suggestThemeName(kind, normalized);
      previewEditor();
    });
    hexInput?.addEventListener("blur", () => {
      const normalized = normalizeHex(hexInput.value, workingPalette[kind].color);
      hexInput.value = normalized;
      if (colorInput) colorInput.value = normalized;
      suggestThemeName(kind, normalized);
      previewEditor();
    });
  });

  $("#themeFontFamily")?.addEventListener("change", previewEditor);

  const bindButtonTextColor = (colorId, hexId, fallbackKey) => {
    const colorInput = $(`#${colorId}`);
    const hexInput = $(`#${hexId}`);
    colorInput?.addEventListener("input", () => {
      if (hexInput) hexInput.value = colorInput.value;
      previewEditor();
    });
    hexInput?.addEventListener("input", () => {
      const normalized = normalizeHex(hexInput.value, "");
      if (!normalized) return;
      if (colorInput) colorInput.value = normalized;
      previewEditor();
    });
    hexInput?.addEventListener("blur", () => {
      const normalized = normalizeHex(hexInput.value, workingPalette.typography[fallbackKey]);
      hexInput.value = normalized;
      if (colorInput) colorInput.value = normalized;
      previewEditor();
    });
  };

  bindButtonTextColor("themePrimaryButtonTextColor", "themePrimaryButtonTextHex", "primary_button_text");
  bindButtonTextColor("themeSecondaryButtonTextColor", "themeSecondaryButtonTextHex", "secondary_button_text");

  const bindSemanticColor = (colorId, hexId, semanticKey) => {
    const colorInput = $(`#${colorId}`);
    const hexInput = $(`#${hexId}`);
    colorInput?.addEventListener("input", () => {
      if (hexInput) hexInput.value = colorInput.value;
      previewEditor();
    });
    hexInput?.addEventListener("input", () => {
      const normalized = normalizeHex(hexInput.value, "");
      if (!normalized) return;
      if (colorInput) colorInput.value = normalized;
      previewEditor();
    });
    hexInput?.addEventListener("blur", () => {
      const normalized = normalizeHex(hexInput.value, workingPalette.semantic[semanticKey]);
      hexInput.value = normalized;
      if (colorInput) colorInput.value = normalized;
      previewEditor();
    });
  };
  bindSemanticColor("themeSurfaceSecondaryColor", "themeSurfaceSecondaryHex", "surface_secondary");
  bindSemanticColor("themeComponentSurfaceColor", "themeComponentSurfaceHex", "component_surface");
  bindSemanticColor("themeComponentBorderColor", "themeComponentBorderHex", "component_border");
  bindSemanticColor("themeComponentAccentColor", "themeComponentAccentHex", "component_accent");
  bindSemanticColor("themeTextPrimaryColor", "themeTextPrimaryHex", "text_primary");
  bindSemanticColor("themeTextSecondaryColor", "themeTextSecondaryHex", "text_secondary");

  warningPreference?.addEventListener("change", async () => {
    contrastWarningsEnabled = Boolean(warningPreference.checked);
    settings.theme_contrast_warnings_enabled = contrastWarningsEnabled;
    try {
      await api.saveSettings({ theme_contrast_warnings_enabled: contrastWarningsEnabled });
      notify(contrastWarningsEnabled ? "Low-contrast theme warnings enabled." : "Low-contrast theme warnings disabled.");
    } catch (error) {
      contrastWarningsEnabled = !contrastWarningsEnabled;
      warningPreference.checked = contrastWarningsEnabled;
      settings.theme_contrast_warnings_enabled = contrastWarningsEnabled;
      notify(error.message, "error");
    }
  });

  $("#themeLocalPackageImportButton")?.addEventListener("click", async () => {
    const input = $("#themeLocalPackageFile");
    const file = input?.files?.[0];
    if (!file) {
      notify("Choose a local theme package first.", "warning");
      return;
    }
    const button = $("#themeLocalPackageImportButton");
    try {
      if (button) button.disabled = true;
      const result = await api.importThemePackage(file);
      renderThemeLibrary(result.library);
      if (input) input.value = "";
      notify(`Imported ${result.installed?.name || "theme package"}. It remains disabled until you choose Use.`);
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  });

  dialog.querySelectorAll(".theme-swatch").forEach((button) => {
    button.addEventListener("click", () => {
      const section = button.closest("[data-theme-kind]");
      const kind = section?.dataset.themeKind;
      if (!kind || !["accent", "surface"].includes(kind)) return;
      const prefix = kind === "accent" ? "accentTheme" : "surfaceTheme";
      const name = button.dataset.themeName || DEFAULT_THEME_PALETTE[kind].name;
      const color = normalizeHex(button.dataset.themeColor, DEFAULT_THEME_PALETTE[kind].color);
      if ($(`#${prefix}Name`)) $(`#${prefix}Name`).value = name;
      if ($(`#${prefix}Color`)) $(`#${prefix}Color`).value = color;
      if ($(`#${prefix}Hex`)) $(`#${prefix}Hex`).value = color;
      manualNameEdits[kind] = false;
      updateSuggestedNameHint(kind, name);
      previewEditor();
    });
  });

  $("#resetThemePaletteButton")?.addEventListener("click", () => {
    workingPalette = clonePalette(DEFAULT_THEME_PALETTE);
    manualNameEdits.accent = false;
    manualNameEdits.surface = false;
    setThemeEditorValues(workingPalette);
    updateSuggestedNameHint("accent", workingPalette.accent.name);
    updateSuggestedNameHint("surface", workingPalette.surface.name);
    previewEditor();
  });

  $("#cancelThemePaletteButton")?.addEventListener("click", () => {
    workingPalette = clonePalette(savedPalette);
    applyThemePalette(savedPalette, { persist: true });
    dialog.close("cancel");
  });

  $("#saveThemePaletteButton")?.addEventListener("click", async () => {
    const button = $("#saveThemePaletteButton");
    try {
      workingPalette = readThemeEditorValues(workingPalette);
      const contrast = updateThemeWarning(workingPalette);
      if (!confirmLowContrastIfNeeded(contrast, contrastWarningsEnabled, "save this theme")) return;
      if (button) button.disabled = true;
      const saved = await api.saveSettings({
        theme_palette: workingPalette,
        theme_contrast_warnings_enabled: contrastWarningsEnabled,
      });
      savedPalette = applyThemePalette(saved.theme_effective_palette || saved.theme_palette || workingPalette, { persist: true });
      settings.theme_palette = clonePalette(savedPalette);
      workingPalette = clonePalette(savedPalette);
      savedDuringOpen = true;
      dialog.close("saved");
      notify("Theme saved.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      updateThemeWarning(workingPalette);
    }
  });

  dialog.addEventListener("close", () => {
    if (!savedDuringOpen && dialog.returnValue !== "saved") {
      workingPalette = clonePalette(savedPalette);
      applyThemePalette(savedPalette, { persist: true });
    }
  });
}


function normalizeMslkSettings(settings = {}) {
  const source = settings?.mslk_fmha && typeof settings.mslk_fmha === "object"
    ? settings.mslk_fmha
    : {};
  return {
    policy: String(source.policy || "blackwell_safe"),
    debug: String(source.debug || ""),
    block_n: String(source.block_n || ""),
    block_m: String(source.block_m || ""),
    num_warps: String(source.num_warps || ""),
    num_stages: String(source.num_stages || ""),
    experimental_head_dims: String(source.experimental_head_dims || ""),
  };
}

function applyMslkEditorValues(settings = {}) {
  const values = normalizeMslkSettings(settings);
  $("#dialogMslkPolicy").value = values.policy;
  $("#dialogMslkDebug").checked = ["1", "true", "yes", "on"].includes(values.debug.toLowerCase());
  $("#dialogMslkBlockN").value = values.block_n;
  $("#dialogMslkBlockM").value = values.block_m;
  $("#dialogMslkNumWarps").value = values.num_warps;
  $("#dialogMslkNumStages").value = values.num_stages;
  $("#dialogMslkExperimentalHeadDims").value = values.experimental_head_dims;
}

function readMslkEditorValues() {
  return {
    policy: $("#dialogMslkPolicy").value || "blackwell_safe",
    debug: $("#dialogMslkDebug").checked ? "1" : "",
    block_n: $("#dialogMslkBlockN").value || "",
    block_m: $("#dialogMslkBlockM").value || "",
    num_warps: $("#dialogMslkNumWarps").value || "",
    num_stages: $("#dialogMslkNumStages").value || "",
    experimental_head_dims: $("#dialogMslkExperimentalHeadDims").value.trim(),
  };
}

function formatMslkValues(values = {}) {
  const shown = (value) => String(value ?? "").trim() || "auto";
  return [
    `policy=${shown(values.policy)}`,
    `BLOCK_N=${shown(values.block_n)}`,
    `BLOCK_M=${shown(values.block_m)}`,
    `warps=${shown(values.num_warps)}`,
    `stages=${shown(values.num_stages)}`,
    `head_dims=${shown(values.experimental_head_dims)}`,
  ].join(" · ");
}

function renderMslkRuntimeStatus(status = null) {
  const warning = $("#dialogMslkRestartStatus");
  const active = $("#dialogMslkActiveSummary");
  if (!status || typeof status !== "object") {
    warning?.classList.add("is-hidden");
    if (active) active.textContent = "Active process settings are unavailable.";
    return;
  }
  if (active) {
    active.textContent = `Active: ${formatMslkValues(status.active || {})} · fingerprint=${String(status.active_fingerprint || "unavailable").slice(0, 12)}`;
  }
  if (warning) {
    warning.textContent = status.message || "";
    warning.classList.toggle("is-hidden", !status.message);
  }
}

function explicitRuntimeJobOverrides(settings = {}) {
  return settings?.runtime_job_overrides && typeof settings.runtime_job_overrides === "object"
    ? settings.runtime_job_overrides
    : {};
}

function setRuntimeOverrideInheritance(inherit) {
  runtimeOverridesInheritMode = Boolean(inherit);
  const button = $("#inheritStartupRuntimeButton");
  const status = $("#runtimeInheritanceStatus");
  if (button) {
    button.setAttribute("aria-pressed", runtimeOverridesInheritMode ? "true" : "false");
    button.textContent = runtimeOverridesInheritMode
      ? "Inheriting Startup Profile"
      : "Inherit Startup Profile";
  }
  if (status) {
    status.textContent = runtimeOverridesInheritMode
      ? "No per-job runtime overrides are saved. Future jobs inherit the active startup profile."
      : "Per-job runtime controls are explicit and will override the startup profile for future jobs.";
  }
}

function markRuntimeJobOverridesExplicit() {
  setRuntimeOverrideInheritance(false);
}

function runtimeJobValues(settings = {}, status = null) {
  const explicit = explicitRuntimeJobOverrides(settings);
  const active = status?.runtime?.next_job_settings && typeof status.runtime.next_job_settings === "object"
    ? status.runtime.next_job_settings
    : {};
  return { ...active, ...explicit };
}

function applyRuntimeEditorValues(settings = {}, status = null) {
  const runtime = status?.runtime || {};
  const values = runtimeJobValues(settings, status);
  const allocator = settings?.allocator_options && typeof settings.allocator_options === "object"
    ? settings.allocator_options
    : {};
  $("#dialogAttentionBackend").value = String(settings.attention_backend || runtime.attention?.saved_next_restart || runtime.attention?.requested_backend || "auto");
  $("#dialogCudaAllocatorConfig").value = String(allocator.PYTORCH_CUDA_ALLOC_CONF ?? runtime.cuda_allocator?.saved_next_restart_config ?? "");
  $("#dialogMemoryPolicy").value = String(values.memory_policy || "auto");
  $("#dialogMemorySafetyMargin").value = String(values.memory_vram_safety_margin_mb ?? 1024);
  $("#dialogAttentionSlicing").value = String(values.attention_slicing || "off");
  $("#dialogHiresMemoryProfile").value = String(values.hires_memory_profile || "inherit");
  $("#dialogPreHiresCleanup").checked = Boolean(values.pre_hires_cleanup);
  $("#dialogPreviewPolicy").value = String(values.preview_policy || "normal");
  $("#dialogVaeTiling").checked = Boolean(values.vae_tiling);
  $("#dialogVaeSlicing").checked = Boolean(values.vae_slicing);
  $("#dialogVaeDevice").value = String(values.vae_device || "auto");
  $("#dialogOomRetryProfile").value = String(values.oom_retry_profile || "disabled");
  $("#dialogOomRetryLimit").value = String(values.oom_retry_limit ?? 1);
  $("#dialogRetainCheckpoint").checked = values.memory_retain_checkpoint_between_jobs !== false;
  $("#dialogRetainVae").checked = values.memory_retain_vae_between_jobs !== false;
  $("#dialogRetainTextEncoder").checked = values.model_runtime_retain_text_encoder_between_jobs !== false;
  setRuntimeOverrideInheritance(Object.keys(explicitRuntimeJobOverrides(settings)).length === 0);
}

function readRuntimeJobOverrides() {
  if (runtimeOverridesInheritMode) return {};
  return {
    memory_policy: $("#dialogMemoryPolicy").value || "auto",
    memory_vram_safety_margin_mb: Math.max(0, Number($("#dialogMemorySafetyMargin").value) || 0),
    memory_retain_checkpoint_between_jobs: $("#dialogRetainCheckpoint").checked,
    memory_retain_vae_between_jobs: $("#dialogRetainVae").checked,
    model_runtime_retain_text_encoder_between_jobs: $("#dialogRetainTextEncoder").checked,
    attention_slicing: $("#dialogAttentionSlicing").value || "off",
    vae_tiling: $("#dialogVaeTiling").checked,
    vae_slicing: $("#dialogVaeSlicing").checked,
    vae_device: $("#dialogVaeDevice").value || "auto",
    preview_policy: $("#dialogPreviewPolicy").value || "normal",
    hires_memory_profile: $("#dialogHiresMemoryProfile").value || "inherit",
    pre_hires_cleanup: $("#dialogPreHiresCleanup").checked,
    oom_retry_profile: $("#dialogOomRetryProfile").value || "disabled",
    oom_retry_limit: Math.max(0, Math.min(1, Number($("#dialogOomRetryLimit").value) || 0)),
  };
}

function renderRuntimeDialogStatus(status = null) {
  renderRuntimeStartupStatus(status);
  const runtime = status?.runtime || {};
  const attention = runtime.attention || {};
  const profile = runtime.runtime_profile || {};
  const provider = attention.provider_verified ? attention.verified_kernel_provider : "Unverified";
  if ($("#dialogRuntimeProfileStatus")) $("#dialogRuntimeProfileStatus").textContent = String(profile.profile_id || "default");
  if ($("#dialogRuntimeAttentionStatus")) $("#dialogRuntimeAttentionStatus").textContent = `${attention.requested_backend || "unavailable"} → ${attention.effective_backend || "unverified"}`;
  if ($("#dialogRuntimeKernelProviderStatus")) $("#dialogRuntimeKernelProviderStatus").textContent = String(provider || "Unverified");
  if ($("#dialogRuntimeMemoryStatus")) $("#dialogRuntimeMemoryStatus").textContent = `${runtime.memory?.policy || "auto"} · hires ${runtime.hires?.memory_profile || "inherit"} · OOM ${runtime.oom_retry?.profile || "disabled"}`;
  const warning = $("#dialogRuntimeRestartStatus");
  if (warning) {
    warning.textContent = status?.message || "";
    warning.classList.toggle("is-hidden", !(status?.restart_required || status?.pending_change_blocked));
  }
}

function updateStabilitySettingVisibility() {
  const cfgEnabled = $("#dialogCfgLabEnabled")?.checked === true;
  if ($("#dialogLivePreviewCfgVisualEnabled")) {
    $("#dialogLivePreviewCfgVisualEnabled").disabled = !cfgEnabled;
    if (!cfgEnabled) $("#dialogLivePreviewCfgVisualEnabled").checked = false;
  }
}

export function bindSettings(settings, { resetLayout = async () => {}, saveLayoutDefault = async () => {}, runtimeStartupStatus = null } = {}) {
  $("#restoreLastSession").checked = Boolean(settings.restore_last_session);
  $("#dialogRestoreLastSession").checked = Boolean(settings.restore_last_session);
  $("#dialogLivePreviewEnabled").checked = settings.live_preview_enabled !== false;
  $("#dialogCfgLabEnabled").checked = settings.cfg_lab_enabled === true;
  $("#dialogLivePreviewCfgVisualEnabled").checked = settings.cfg_lab_enabled === true && settings.live_preview_cfg_visual_enabled === true;
  $("#dialogDiagnosticsMode").value = settings.diagnostics_mode || "failures_only";
  $("#dialogDiagnosticDecodeEnabled").checked = settings.diagnostic_decode_enabled === true;
  $("#dialogLivePreviewMode").value = FORCED_LIVE_PREVIEW_MODE;
  $("#dialogLivePreviewInterval").value = String(settings.live_preview_interval || 10);
  $("#dialogLivePreviewWidth").value = String(settings.live_preview_width || 384);
  $("#dialogLivePreviewFormat").value = settings.live_preview_format || "webp";
  $("#dialogLivePreviewHistory").value = settings.live_preview_keep_history || "current_job";
  $("#dialogLivePreviewBatchIndex").value = Number(settings.live_preview_batch_index || 0);
  $("#dialogLivePreviewAdaptiveThrottle").checked = settings.live_preview_adaptive_throttle !== false;
  $("#dialogLivePreviewAdaptiveTarget").value = Number(settings.live_preview_adaptive_target_ratio || 0.75);
  $("#dialogLivePreviewAdaptiveMaxInterval").value = String(settings.live_preview_adaptive_max_interval || 8);
  applyRuntimeEditorValues(settings, runtimeStartupStatus || settings._runtime_startup_status || null);
  $("#dialogMemoryPinnedCpu").checked = Boolean(settings.memory_pinned_cpu_memory);
  $("#dialogMemoryTiledVaeFallback").checked = settings.memory_allow_tiled_vae_fallback !== false;
  $("#dialogMemoryPreviewSuspension").checked = settings.memory_allow_preview_suspension_on_oom !== false;
  applyMslkEditorValues(settings);
  renderMslkRuntimeStatus(runtimeStartupStatus || settings._runtime_startup_status || null);
  renderRuntimeDialogStatus(runtimeStartupStatus || settings._runtime_startup_status || null);
  $("#dialogRecentOutputsBackgroundRefreshEnabled").checked = settings.recent_outputs_background_refresh_enabled !== false;
  $("#dialogRecentOutputsRefreshMsActive").value = String(settings.recent_outputs_refresh_ms_active || 4000);
  $("#dialogRecentOutputsRefreshMsIdle").value = String(settings.recent_outputs_refresh_ms_idle || 12000);
  enforceLivePreviewModeLock();
  updateAccuratePreviewWarning();
  updateStabilitySettingVisibility();
  applyUiScale(settings.ui_scale || 100);
  bindThemePalette(settings);

  $("#uiScale").addEventListener("input", (event) => applyUiScale(event.target.value));
  $("#dialogUiScale").addEventListener("input", (event) => applyUiScale(event.target.value));
  $("#dialogLivePreviewMode").addEventListener("change", updateAccuratePreviewWarning);
  $("#dialogCfgLabEnabled")?.addEventListener("change", updateStabilitySettingVisibility);

  RUNTIME_JOB_CONTROL_IDS.forEach((id) => {
    const control = $(`#${id}`);
    control?.addEventListener("change", markRuntimeJobOverridesExplicit);
    if (control?.matches('input[type="number"]')) {
      control.addEventListener("input", markRuntimeJobOverridesExplicit);
    }
  });

  const openSettingsDialog = async () => {
    const dialog = $("#settingsDialog");
    if (!dialog.open) dialog.showModal();
    window.dispatchEvent(new CustomEvent("image-gen-settings-opened"));
    try {
      const status = await api.runtimeStartupStatus();
      renderMslkRuntimeStatus(status);
      renderRuntimeDialogStatus(status);
      applyRuntimeEditorValues(state.settings || settings, status);
    } catch (error) {
      console.warn("Unable to refresh MSLK startup status", error);
    }
  };
  $("#openSettingsButton")?.addEventListener("click", openSettingsDialog);
  window.addEventListener("image-gen-open-settings", openSettingsDialog);
  $("#settingsDialog")?.addEventListener("close", () => {
    window.dispatchEvent(new CustomEvent("image-gen-settings-closed"));
  });
  $("#inheritStartupRuntimeButton")?.addEventListener("click", async () => {
    const button = $("#inheritStartupRuntimeButton");
    try {
      if (button) button.disabled = true;
      const saved = await api.inheritRuntimeStartupProfile();
      const status = saved._runtime_startup_status || null;
      state.settings = { ...state.settings, ...saved, runtime_job_overrides: {} };
      applyRuntimeEditorValues(saved, status);
      renderRuntimeDialogStatus(status);
      renderMslkRuntimeStatus(status);
      notify("Per-job runtime overrides cleared. Future jobs now inherit the active startup profile.");
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  });

  $("#saveSettingsButton").addEventListener("click", async () => {
    try {
      const saved = await api.saveSettings({
        restore_last_session: $("#dialogRestoreLastSession").checked,
        ui_scale: Number($("#dialogUiScale").value),
        live_preview_enabled: $("#dialogLivePreviewEnabled").checked,
        cfg_lab_enabled: $("#dialogCfgLabEnabled").checked,
        live_preview_cfg_visual_enabled: $("#dialogCfgLabEnabled").checked && $("#dialogLivePreviewCfgVisualEnabled").checked,
        diagnostics_mode: $("#dialogDiagnosticsMode").value || "failures_only",
        diagnostic_decode_enabled: $("#dialogDiagnosticDecodeEnabled").checked,
        live_preview_mode: FORCED_LIVE_PREVIEW_MODE,
        live_preview_interval: Number($("#dialogLivePreviewInterval").value),
        live_preview_width: Number($("#dialogLivePreviewWidth").value),
        live_preview_format: $("#dialogLivePreviewFormat").value,
        live_preview_keep_history: $("#dialogLivePreviewHistory").value,
        live_preview_batch_index: Math.max(0, Number($("#dialogLivePreviewBatchIndex").value) || 0),
        live_preview_adaptive_throttle: $("#dialogLivePreviewAdaptiveThrottle").checked,
        live_preview_adaptive_target_ratio: Math.max(0.20, Math.min(1.50, Number($("#dialogLivePreviewAdaptiveTarget").value) || 0.75)),
        live_preview_adaptive_max_interval: Math.max(1, Number($("#dialogLivePreviewAdaptiveMaxInterval").value) || 8),
        attention_backend: $("#dialogAttentionBackend").value || "auto",
        allocator_options: { PYTORCH_CUDA_ALLOC_CONF: $("#dialogCudaAllocatorConfig").value.trim() },
        runtime_job_overrides: readRuntimeJobOverrides(),
        memory_pinned_cpu_memory: $("#dialogMemoryPinnedCpu").checked,
        memory_allow_tiled_vae_fallback: $("#dialogMemoryTiledVaeFallback").checked,
        memory_allow_preview_suspension_on_oom: $("#dialogMemoryPreviewSuspension").checked,
        mslk_fmha: readMslkEditorValues(),
        recent_outputs_background_refresh_enabled: $("#dialogRecentOutputsBackgroundRefreshEnabled").checked,
        recent_outputs_refresh_ms_active: Math.max(1000, Number($("#dialogRecentOutputsRefreshMsActive").value) || 4000),
        recent_outputs_refresh_ms_idle: Math.max(2000, Number($("#dialogRecentOutputsRefreshMsIdle").value) || 12000),
      });
      $("#restoreLastSession").checked = saved.restore_last_session;
      $("#dialogLivePreviewEnabled").checked = saved.live_preview_enabled !== false;
      $("#dialogCfgLabEnabled").checked = saved.cfg_lab_enabled === true;
      $("#dialogLivePreviewCfgVisualEnabled").checked = saved.cfg_lab_enabled === true && saved.live_preview_cfg_visual_enabled === true;
      $("#dialogDiagnosticsMode").value = saved.diagnostics_mode || "failures_only";
      $("#dialogDiagnosticDecodeEnabled").checked = saved.diagnostic_decode_enabled === true;
      $("#dialogLivePreviewMode").value = FORCED_LIVE_PREVIEW_MODE;
      $("#dialogLivePreviewInterval").value = String(saved.live_preview_interval || 10);
      $("#dialogLivePreviewWidth").value = String(saved.live_preview_width || 384);
      $("#dialogLivePreviewFormat").value = saved.live_preview_format || "webp";
      $("#dialogLivePreviewHistory").value = saved.live_preview_keep_history || "current_job";
      $("#dialogLivePreviewBatchIndex").value = Number(saved.live_preview_batch_index || 0);
      $("#dialogLivePreviewAdaptiveThrottle").checked = saved.live_preview_adaptive_throttle !== false;
      $("#dialogLivePreviewAdaptiveTarget").value = Number(saved.live_preview_adaptive_target_ratio || 0.75);
      $("#dialogLivePreviewAdaptiveMaxInterval").value = String(saved.live_preview_adaptive_max_interval || 8);
      applyRuntimeEditorValues(saved, saved._runtime_startup_status || null);
      $("#dialogMemoryPinnedCpu").checked = Boolean(saved.memory_pinned_cpu_memory);
      $("#dialogMemoryTiledVaeFallback").checked = saved.memory_allow_tiled_vae_fallback !== false;
      $("#dialogMemoryPreviewSuspension").checked = saved.memory_allow_preview_suspension_on_oom !== false;
      applyMslkEditorValues(saved);
      renderMslkRuntimeStatus(saved._runtime_startup_status || null);
      renderRuntimeDialogStatus(saved._runtime_startup_status || null);
      $("#dialogRecentOutputsBackgroundRefreshEnabled").checked = saved.recent_outputs_background_refresh_enabled !== false;
      $("#dialogRecentOutputsRefreshMsActive").value = String(saved.recent_outputs_refresh_ms_active || 4000);
      $("#dialogRecentOutputsRefreshMsIdle").value = String(saved.recent_outputs_refresh_ms_idle || 12000);
      enforceLivePreviewModeLock();
      updateAccuratePreviewWarning();
      updateStabilitySettingVisibility();
      applyUiScale(saved.ui_scale);
      state.settings = { ...state.settings, ...saved };
      settings.cfg_lab_enabled = saved.cfg_lab_enabled === true;
      settings.live_preview_cfg_visual_enabled = saved.cfg_lab_enabled === true && saved.live_preview_cfg_visual_enabled === true;
      window.dispatchEvent(new CustomEvent("live-preview-cfg-visual-setting-changed", {
        detail: { enabled: saved.cfg_lab_enabled === true && saved.live_preview_cfg_visual_enabled === true },
      }));
      window.dispatchEvent(new CustomEvent("recent-outputs-refresh-settings-changed", {
        detail: {
          enabled: saved.recent_outputs_background_refresh_enabled !== false,
          activeMs: saved.recent_outputs_refresh_ms_active || 4000,
          idleMs: saved.recent_outputs_refresh_ms_idle || 12000,
        },
      }));
      $("#settingsDialog").close();
      const restartMessage = saved._runtime_startup_status?.message;
      notify(restartMessage || "UI settings saved.", restartMessage ? "warning" : undefined);
    } catch (error) {
      notify(error.message, "error");
    }
  });

  $("#saveScaleLayoutDefaultButton")?.addEventListener("click", async () => {
    const button = $("#saveScaleLayoutDefaultButton");
    try {
      if (button) button.disabled = true;
      const scale = Number($("#dialogUiScale")?.value || settings.ui_scale || 100);
      await saveLayoutDefault(scale);
      notify(`Saved the current screen layout as the default for UI scale ${Math.round(scale)}%.`);
    } catch (error) {
      notify(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  });

  $("#resetLayoutButton").addEventListener("click", async () => {
    try {
      document.querySelectorAll(".panel.is-collapsed, .controls-panel.is-collapsed").forEach((item) => item.classList.remove("is-collapsed"));
      await resetLayout();
      notify("UI layout reset to the default layout for the current UI scale.");
    } catch (error) {
      notify(error.message, "error");
    }
  });
}
