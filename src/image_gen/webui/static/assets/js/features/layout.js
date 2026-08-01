import { api } from "../api.js";
import { state } from "../state.js";
import { $, debounce } from "../utils.js";

const MIN_LEFT_COLUMN = 170;
const MIN_CENTER_COLUMN = 180;
const MIN_RIGHT_COLUMN = 190;
const MAX_STORED_COLUMN = Number.MAX_SAFE_INTEGER;
const MIN_GALLERY_HEIGHT = 96;
const MIN_OUTPUT_HEIGHT = 120;
const MAX_GALLERY_HEIGHT = Number.MAX_SAFE_INTEGER;
const MIN_LIVE_PREVIEW_HEIGHT = 240;
const MAX_LIVE_PREVIEW_HEIGHT = 450;
const PANEL_SCALE_MIN = 60;
const PANEL_SCALE_MAX = 240;
const PANEL_SCALE_STEP = 10;

export const DEFAULT_PANEL_SCALES = Object.freeze({
  controls: 100,
  output_viewer: 100,
  recent_outputs: 100,
  live_preview: 100,
  memory_status: 100,
  queue: 100,
  recent_runs: 100,
  prompt_presets: 100,
  model_refresh: 100,
  maintenance: 100,
});

export const DEFAULT_LAYOUT = Object.freeze({
  left_column_width: 330,
  right_column_width: 360,
  gallery_panel_height: 132,
  live_preview_panel_height: 360,
  live_preview_collapsed: false,
  follow_newest_output: false,
  panel_scales: DEFAULT_PANEL_SCALES,
});

const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, Number(value) || minimum));

function normalizePanelScales(value = {}) {
  const stored = value && typeof value === "object" ? value : {};
  return Object.fromEntries(
    Object.entries(DEFAULT_PANEL_SCALES).map(([key, fallback]) => [
      key,
      clamp(stored[key] ?? fallback, PANEL_SCALE_MIN, PANEL_SCALE_MAX),
    ]),
  );
}

function normalizeLayout(settings = {}) {
  const stored = settings.ui_layout && typeof settings.ui_layout === "object"
    ? settings.ui_layout
    : {};
  return {
    left_column_width: clamp(stored.left_column_width ?? DEFAULT_LAYOUT.left_column_width, MIN_LEFT_COLUMN, MAX_STORED_COLUMN),
    right_column_width: clamp(stored.right_column_width ?? DEFAULT_LAYOUT.right_column_width, MIN_RIGHT_COLUMN, MAX_STORED_COLUMN),
    gallery_panel_height: clamp(stored.gallery_panel_height ?? DEFAULT_LAYOUT.gallery_panel_height, MIN_GALLERY_HEIGHT, MAX_GALLERY_HEIGHT),
    live_preview_panel_height: clamp(stored.live_preview_panel_height ?? DEFAULT_LAYOUT.live_preview_panel_height, MIN_LIVE_PREVIEW_HEIGHT, MAX_LIVE_PREVIEW_HEIGHT),
    live_preview_collapsed: Boolean(stored.live_preview_collapsed),
    follow_newest_output: Boolean(stored.follow_newest_output),
    panel_scales: normalizePanelScales(stored.panel_scales),
  };
}

function cloneLayout(layout) {
  return { ...layout, panel_scales: { ...layout.panel_scales } };
}

function mergeLayoutValues(baseLayout = {}, overrideLayout = {}) {
  return {
    ...baseLayout,
    ...overrideLayout,
    panel_scales: {
      ...(baseLayout.panel_scales || {}),
      ...(overrideLayout.panel_scales || {}),
    },
  };
}

function resolveLayoutDefaultsForScale(settings = {}, scale = null) {
  const source = settings && typeof settings === "object" ? settings : {};
  const currentScale = Math.round(Number(scale ?? source.ui_scale ?? 100) || 100);
  const scaleLayouts = source.ui_scale_layout_defaults && typeof source.ui_scale_layout_defaults === "object"
    ? source.ui_scale_layout_defaults
    : {};
  const baseDefaults = source.ui_layout_defaults && typeof source.ui_layout_defaults === "object"
    ? source.ui_layout_defaults
    : DEFAULT_LAYOUT;
  const selectedScaleLayout = scaleLayouts[String(currentScale)] && typeof scaleLayouts[String(currentScale)] === "object"
    ? scaleLayouts[String(currentScale)]
    : null;
  return normalizeLayout({
    ui_layout: selectedScaleLayout
      ? mergeLayoutValues(baseDefaults, selectedScaleLayout)
      : baseDefaults,
  });
}

function workspaceHorizontalMetrics() {
  const workspace = $("#workspace");
  const styles = window.getComputedStyle(workspace);
  const gap = Number.parseFloat(styles.columnGap || styles.gap) || 0;
  const splitterWidth = [$("#leftColumnSplitter"), $("#rightColumnSplitter")]
    .reduce((total, item) => total + (item?.getBoundingClientRect().width || 0), 0);
  const fixedWidth = splitterWidth + (gap * 4);
  return {
    width: workspace.getBoundingClientRect().width,
    fixedWidth,
    usableWidth: Math.max(0, workspace.getBoundingClientRect().width - fixedWidth),
  };
}

function fitHorizontalLayout(layout, preference = "balanced") {
  const { usableWidth } = workspaceHorizontalMetrics();
  const maximumSideTotal = Math.max(
    MIN_LEFT_COLUMN + MIN_RIGHT_COLUMN,
    usableWidth - MIN_CENTER_COLUMN,
  );
  let left = clamp(layout.left_column_width, MIN_LEFT_COLUMN, MAX_STORED_COLUMN);
  let right = clamp(layout.right_column_width, MIN_RIGHT_COLUMN, MAX_STORED_COLUMN);

  if (left + right > maximumSideTotal) {
    if (preference === "left") {
      right = Math.max(MIN_RIGHT_COLUMN, maximumSideTotal - left);
      left = Math.max(MIN_LEFT_COLUMN, Math.min(left, maximumSideTotal - right));
    } else if (preference === "right") {
      left = Math.max(MIN_LEFT_COLUMN, maximumSideTotal - right);
      right = Math.max(MIN_RIGHT_COLUMN, Math.min(right, maximumSideTotal - left));
    } else {
      const leftCapacity = Math.max(0, left - MIN_LEFT_COLUMN);
      const rightCapacity = Math.max(0, right - MIN_RIGHT_COLUMN);
      const capacity = leftCapacity + rightCapacity;
      const overflow = left + right - maximumSideTotal;
      if (capacity > 0) {
        left -= overflow * (leftCapacity / capacity);
        right -= overflow * (rightCapacity / capacity);
      }
      left = Math.max(MIN_LEFT_COLUMN, left);
      right = Math.max(MIN_RIGHT_COLUMN, maximumSideTotal - left);
      if (left + right > maximumSideTotal) left = Math.max(MIN_LEFT_COLUMN, maximumSideTotal - right);
    }
  }

  return {
    left_column_width: Math.round(left),
    right_column_width: Math.round(right),
  };
}

function copyToState(layout, renderedHorizontal = layout, effectiveGalleryHeight = layout.gallery_panel_height) {
  state.layout = {
    leftColumnWidth: renderedHorizontal.left_column_width,
    rightColumnWidth: renderedHorizontal.right_column_width,
    galleryPanelHeight: effectiveGalleryHeight,
    livePreviewPanelHeight: layout.live_preview_panel_height,
    livePreviewCollapsed: layout.live_preview_collapsed,
    followNewestOutput: layout.follow_newest_output,
    panelScales: { ...layout.panel_scales },
  };
}

function createPanelScaleControls(panel, label, onChange) {
  const group = document.createElement("span");
  group.className = "panel-scale-controls";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", `${label} interface scale`);

  const decrease = document.createElement("button");
  decrease.type = "button";
  decrease.className = "panel-scale-button";
  decrease.textContent = "A−";
  decrease.title = `Make ${label.toLowerCase()} text and controls smaller`;
  decrease.setAttribute("aria-label", decrease.title);

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "panel-scale-reset";
  reset.title = `Reset ${label.toLowerCase()} scale to 100%`;
  reset.setAttribute("aria-label", reset.title);

  const increase = document.createElement("button");
  increase.type = "button";
  increase.className = "panel-scale-button";
  increase.textContent = "A+";
  increase.title = `Make ${label.toLowerCase()} text and controls larger`;
  increase.setAttribute("aria-label", increase.title);

  decrease.addEventListener("click", () => onChange(-PANEL_SCALE_STEP));
  increase.addEventListener("click", () => onChange(PANEL_SCALE_STEP));
  reset.addEventListener("click", () => onChange(0, true));
  group.append(decrease, reset, increase);
  panel._panelScaleOutput = reset;
  return group;
}

function installPanelScaleControls(onScaleChange) {
  document.querySelectorAll("[data-panel-scale-key]").forEach((panel) => {
    if (panel.querySelector(":scope > .panel-heading .panel-scale-controls")) return;
    const key = panel.dataset.panelScaleKey;
    const heading = panel.querySelector(":scope > .panel-heading");
    if (!key || !heading) return;
    const label = heading.querySelector("h2")?.textContent?.trim() || "Panel";
    const controls = createPanelScaleControls(panel, label, (delta, reset = false) => {
      onScaleChange(key, reset ? 100 : delta, reset);
    });

    let actions = heading.querySelector(
      ":scope > .viewer-actions, :scope > .gallery-selection-actions, :scope > .live-preview-heading-actions, :scope > .panel-heading-actions",
    );
    if (!actions) {
      actions = document.createElement("div");
      actions.className = "panel-heading-actions";
      [...heading.children]
        .filter((child) => child.matches("button, .panel-toggle"))
        .forEach((child) => actions.append(child));
      heading.append(actions);
    }
    actions.prepend(controls);
  });
}

function applyPanelScales(panelScales) {
  document.querySelectorAll("[data-panel-scale-key]").forEach((panel) => {
    const key = panel.dataset.panelScaleKey;
    const scale = clamp(panelScales[key] ?? 100, PANEL_SCALE_MIN, PANEL_SCALE_MAX);
    panel.style.setProperty("--panel-ui-scale", String(scale / 100));
    panel.dataset.panelScale = String(scale);
    if (panel._panelScaleOutput) {
      panel._panelScaleOutput.textContent = `${scale}%`;
      panel._panelScaleOutput.setAttribute("aria-label", `Reset panel scale to 100%. Current scale ${scale}%`);
    }
  });
}

export function bindWorkspaceLayout(settings = {}) {
  let sourceSettings = settings;
  let layout = normalizeLayout(settings);
  let renderedHorizontal = fitHorizontalLayout(layout);
  let effectiveGalleryHeight = layout.gallery_panel_height;
  let gallerySyncFrame = 0;

  const persistSoon = debounce(async () => {
    try {
      const saved = await api.saveSettings({ ui_layout: { ...layout, panel_scales: { ...layout.panel_scales } } });
      state.settings = saved;
      sourceSettings = saved;
    } catch (error) {
      console.error("Unable to save WebUI layout", error);
    }
  }, 250);

  const updateSeparatorValues = () => {
    const metrics = workspaceHorizontalMetrics();
    const maxLeft = Math.max(MIN_LEFT_COLUMN, metrics.usableWidth - MIN_CENTER_COLUMN - MIN_RIGHT_COLUMN);
    const maxRight = Math.max(MIN_RIGHT_COLUMN, metrics.usableWidth - MIN_CENTER_COLUMN - MIN_LEFT_COLUMN);
    const leftSplitter = $("#leftColumnSplitter");
    const rightSplitter = $("#rightColumnSplitter");
    leftSplitter?.setAttribute("aria-valuemin", String(MIN_LEFT_COLUMN));
    leftSplitter?.setAttribute("aria-valuemax", String(Math.round(maxLeft)));
    leftSplitter?.setAttribute("aria-valuenow", String(renderedHorizontal.left_column_width));
    rightSplitter?.setAttribute("aria-valuemin", String(MIN_RIGHT_COLUMN));
    rightSplitter?.setAttribute("aria-valuemax", String(Math.round(maxRight)));
    rightSplitter?.setAttribute("aria-valuenow", String(renderedHorizontal.right_column_width));
    $("#centerSplitter")?.setAttribute("aria-valuenow", String(Math.round(effectiveGalleryHeight)));
  };

  const syncGalleryStackHeight = () => {
    window.cancelAnimationFrame(gallerySyncFrame);
    gallerySyncFrame = window.requestAnimationFrame(() => {
      const workspace = $("#workspace");
      const browser = $("#outputBrowser");
      const splitter = $("#centerSplitter");
      if (!workspace || !browser) return;

      if (window.matchMedia("(max-width: 720px)").matches) {
        effectiveGalleryHeight = layout.gallery_panel_height;
      } else {
        const splitterHeight = splitter?.classList.contains("is-hidden")
          ? 0
          : (splitter?.getBoundingClientRect().height || 10);
        const maximum = Math.max(
          MIN_GALLERY_HEIGHT,
          browser.clientHeight - MIN_OUTPUT_HEIGHT - splitterHeight,
        );
        effectiveGalleryHeight = clamp(
          layout.gallery_panel_height,
          MIN_GALLERY_HEIGHT,
          Math.min(MAX_GALLERY_HEIGHT, maximum),
        );
      }

      workspace.style.setProperty("--gallery-panel-height", `${effectiveGalleryHeight}px`);
      copyToState(layout, renderedHorizontal, effectiveGalleryHeight);
      updateSeparatorValues();
    });
  };

  const applyLayout = (preference = "balanced") => {
    const workspace = $("#workspace");
    renderedHorizontal = fitHorizontalLayout(layout, preference);
    workspace.style.setProperty("--left-column-width", `${renderedHorizontal.left_column_width}px`);
    workspace.style.setProperty("--right-column-width", `${renderedHorizontal.right_column_width}px`);
    workspace.style.setProperty("--center-column-min-width", `${MIN_CENTER_COLUMN}px`);
    workspace.style.setProperty("--gallery-panel-height", `${layout.gallery_panel_height}px`);
    workspace.style.setProperty("--live-preview-panel-height", `${layout.live_preview_panel_height}px`);
    applyPanelScales(layout.panel_scales);

    const livePanel = $("#livePreviewPanel");
    const liveToggle = $("#livePreviewToggle");
    livePanel.classList.toggle("is-collapsed", layout.live_preview_collapsed);
    liveToggle.textContent = layout.live_preview_collapsed ? "⌄" : "⌃";
    liveToggle.setAttribute("aria-expanded", String(!layout.live_preview_collapsed));
    liveToggle.setAttribute(
      "aria-label",
      layout.live_preview_collapsed ? "Expand live preview" : "Collapse live preview",
    );

    const followNewest = $("#followNewestOutput");
    if (followNewest) followNewest.checked = layout.follow_newest_output;
    copyToState(layout, renderedHorizontal, effectiveGalleryHeight);
    updateSeparatorValues();
    syncGalleryStackHeight();
  };

  const update = (changes, { persist = true, preference = "balanced", useRenderedHorizontal = false } = {}) => {
    const merged = {
      ...layout,
      ...changes,
      panel_scales: {
        ...layout.panel_scales,
        ...(changes.panel_scales || {}),
      },
    };
    layout = normalizeLayout({ ui_layout: merged });
    applyLayout(preference);
    if (useRenderedHorizontal) {
      layout.left_column_width = renderedHorizontal.left_column_width;
      layout.right_column_width = renderedHorizontal.right_column_width;
    }
    if (persist) persistSoon();
  };

  installPanelScaleControls((key, value, reset = false) => {
    const current = layout.panel_scales[key] ?? 100;
    const next = reset ? 100 : clamp(current + value, PANEL_SCALE_MIN, PANEL_SCALE_MAX);
    update({ panel_scales: { [key]: next } });
  });
  applyLayout();

  const beginPointerResize = (event, onMove) => {
    if (window.matchMedia("(max-width: 720px)").matches) return;
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    document.body.classList.add("is-resizing");

    const move = (moveEvent) => onMove(moveEvent);
    const finish = () => {
      document.body.classList.remove("is-resizing");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", finish);
      handle.removeEventListener("pointercancel", finish);
      persistSoon();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
  };

  $("#leftColumnSplitter").addEventListener("pointerdown", (event) => {
    const bounds = $("#workspace").getBoundingClientRect();
    beginPointerResize(event, (moveEvent) => {
      update(
        { left_column_width: moveEvent.clientX - bounds.left },
        { persist: false, preference: "left", useRenderedHorizontal: true },
      );
    });
  });

  $("#rightColumnSplitter").addEventListener("pointerdown", (event) => {
    const bounds = $("#workspace").getBoundingClientRect();
    beginPointerResize(event, (moveEvent) => {
      update(
        { right_column_width: bounds.right - moveEvent.clientX },
        { persist: false, preference: "right", useRenderedHorizontal: true },
      );
    });
  });

  $("#centerSplitter").addEventListener("pointerdown", (event) => {
    const bounds = $("#outputBrowser").getBoundingClientRect();
    beginPointerResize(event, (moveEvent) => {
      const maximum = Math.min(MAX_GALLERY_HEIGHT, Math.max(MIN_GALLERY_HEIGHT, bounds.height - MIN_OUTPUT_HEIGHT));
      update(
        { gallery_panel_height: clamp(bounds.bottom - moveEvent.clientY, MIN_GALLERY_HEIGHT, maximum) },
        { persist: false },
      );
    });
  });

  $("#livePreviewSplitter").addEventListener("pointerdown", (event) => {
    const startY = event.clientY;
    const startHeight = layout.live_preview_panel_height;
    beginPointerResize(event, (moveEvent) => {
      const nextHeight = startHeight + (moveEvent.clientY - startY);
      update(
        { live_preview_panel_height: clamp(nextHeight, MIN_LIVE_PREVIEW_HEIGHT, MAX_LIVE_PREVIEW_HEIGHT) },
        { persist: false },
      );
    });
  });

  const keyboardStep = (event) => event.shiftKey ? 48 : 12;
  $("#leftColumnSplitter").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? keyboardStep(event) : -keyboardStep(event);
    update(
      { left_column_width: renderedHorizontal.left_column_width + delta },
      { preference: "left", useRenderedHorizontal: true },
    );
  });

  $("#rightColumnSplitter").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? keyboardStep(event) : -keyboardStep(event);
    update(
      { right_column_width: renderedHorizontal.right_column_width + delta },
      { preference: "right", useRenderedHorizontal: true },
    );
  });

  $("#centerSplitter").addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowUp" ? keyboardStep(event) : -keyboardStep(event);
    update({ gallery_panel_height: layout.gallery_panel_height + delta });
  });

  $("#livePreviewSplitter").addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? keyboardStep(event) : -keyboardStep(event);
    update({ live_preview_panel_height: layout.live_preview_panel_height + delta });
  });

  $("#livePreviewToggle").addEventListener("click", () => {
    update({ live_preview_collapsed: !layout.live_preview_collapsed });
  });

  const followNewest = $("#followNewestOutput");
  if (followNewest) {
    followNewest.addEventListener("change", (event) => {
      update({ follow_newest_output: event.target.checked });
    });
  }

  const workspaceResizeObserver = new ResizeObserver(() => applyLayout());
  workspaceResizeObserver.observe($("#workspace"));
  const galleryResizeObserver = new ResizeObserver(syncGalleryStackHeight);
  galleryResizeObserver.observe($("#outputBrowser"));

  return {
    reset: async () => {
      layout = resolveLayoutDefaultsForScale(sourceSettings, sourceSettings?.ui_scale ?? state.settings?.ui_scale ?? 100);
      applyLayout();
      const saved = await api.saveSettings({ ui_layout: { ...layout, panel_scales: { ...layout.panel_scales } } });
      state.settings = saved;
      sourceSettings = saved;
      return cloneLayout(layout);
    },
    saveCurrentScaleDefault: async (scale) => {
      const scaleKey = String(Math.round(Number(scale ?? sourceSettings?.ui_scale ?? state.settings?.ui_scale ?? 100) || 100));
      const payload = cloneLayout(layout);
      const saved = await api.saveSettings({
        ui_layout: payload,
        ui_scale_layout_defaults: { [scaleKey]: payload },
      });
      state.settings = saved;
      sourceSettings = saved;
      return { scale: scaleKey, layout: payload, settings: saved };
    },
    current: () => cloneLayout(layout),
  };
}
