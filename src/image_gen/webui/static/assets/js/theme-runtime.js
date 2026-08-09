(function initializeImageGenThemeRuntime(global) {
  "use strict";

  const STORAGE_KEY = "image-gen.theme-palette.v1";
  const DEFAULT_THEME_PALETTE = Object.freeze({
    accent: Object.freeze({ name: "Sky Blue", color: "#179ee7" }),
    surface: Object.freeze({ name: "Charcoal", color: "#111d29" }),
    typography: Object.freeze({
      font_family: "Inter",
      primary_button_text: "#ffffff",
      secondary_button_text: "#d5f1ff",
    }),
  });
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

  function normalizeHex(value, fallback) {
    const text = String(value || "").trim();
    const short = /^#?([0-9a-f]{3})$/i.exec(text);
    if (short) {
      return `#${short[1].split("").map((item) => item + item).join("")}`.toLowerCase();
    }
    const full = /^#?([0-9a-f]{6})$/i.exec(text);
    return full ? `#${full[1].toLowerCase()}` : fallback;
  }

  function clonePalette(value) {
    return {
      accent: { ...DEFAULT_THEME_PALETTE.accent, ...(value && value.accent ? value.accent : {}) },
      surface: { ...DEFAULT_THEME_PALETTE.surface, ...(value && value.surface ? value.surface : {}) },
      typography: { ...DEFAULT_THEME_PALETTE.typography, ...(value && value.typography ? value.typography : {}) },
    };
  }

  function normalizePalette(value) {
    const source = value && typeof value === "object" ? value : {};
    const output = clonePalette(DEFAULT_THEME_PALETTE);
    ["accent", "surface"].forEach((kind) => {
      const candidate = source[kind];
      if (!candidate || typeof candidate !== "object") return;
      output[kind].name = String(candidate.name || output[kind].name).trim().slice(0, 40) || output[kind].name;
      output[kind].color = normalizeHex(candidate.color, output[kind].color);
    });

    const typography = source.typography;
    if (typography && typeof typography === "object") {
      const family = String(typography.font_family || output.typography.font_family);
      output.typography.font_family = Object.prototype.hasOwnProperty.call(FONT_FAMILY_STACKS, family)
        ? family
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

  function rgbToHex(rgb) {
    const channel = (value) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, "0");
    return `#${channel(rgb.r)}${channel(rgb.g)}${channel(rgb.b)}`;
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

  function applyVariables(palette) {
    const root = global.document && global.document.documentElement
      ? global.document.documentElement.style
      : null;
    if (!root || typeof root.setProperty !== "function") return;

    const accent = palette.accent.color;
    const surface = palette.surface.color;
    root.setProperty("--sky-50", mixHex(accent, "#ffffff", 0.92));
    root.setProperty("--sky-100", mixHex(accent, "#ffffff", 0.82));
    root.setProperty("--sky-200", mixHex(accent, "#ffffff", 0.62));
    root.setProperty("--sky-300", mixHex(accent, "#ffffff", 0.38));
    root.setProperty("--sky-400", mixHex(accent, "#ffffff", 0.18));
    root.setProperty("--sky-500", accent);
    root.setProperty("--sky-600", mixHex(accent, "#000000", 0.18));

    root.setProperty("--charcoal-950", mixHex(surface, "#000000", 0.48));
    root.setProperty("--charcoal-900", mixHex(surface, "#000000", 0.24));
    root.setProperty("--charcoal-850", surface);
    root.setProperty("--charcoal-800", mixHex(surface, "#ffffff", 0.08));
    root.setProperty("--charcoal-700", mixHex(surface, "#ffffff", 0.20));
    root.setProperty("--line", mixHex(surface, "#ffffff", 0.22));

    root.setProperty("--font-ui", FONT_FAMILY_STACKS[palette.typography.font_family]);
    root.setProperty("--primary-button-text", palette.typography.primary_button_text);
    root.setProperty("--secondary-button-text", palette.typography.secondary_button_text);
  }

  function cachePalette(value) {
    const palette = normalizePalette(value);
    try {
      global.localStorage && global.localStorage.setItem(STORAGE_KEY, JSON.stringify(palette));
    } catch (_error) {
      // Theme caching is a paint optimization. Server-side settings remain authoritative.
    }
    return palette;
  }

  function readCachedPalette() {
    try {
      const raw = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      return normalizePalette(JSON.parse(raw));
    } catch (_error) {
      return null;
    }
  }

  function apply(value, options) {
    const palette = normalizePalette(value);
    applyVariables(palette);
    if (options && options.persist) cachePalette(palette);
    return palette;
  }

  function applyCached() {
    const palette = readCachedPalette();
    if (!palette) return null;
    applyVariables(palette);
    if (global.document && global.document.documentElement) {
      global.document.documentElement.dataset.themePrepaint = "cached";
    }
    return palette;
  }

  global.ImageGenThemeRuntime = Object.freeze({
    storageKey: STORAGE_KEY,
    defaultPalette: DEFAULT_THEME_PALETTE,
    fontFamilyStacks: FONT_FAMILY_STACKS,
    normalize: normalizePalette,
    apply,
    cache: cachePalette,
    readCached: readCachedPalette,
    applyCached,
  });
})(window);
