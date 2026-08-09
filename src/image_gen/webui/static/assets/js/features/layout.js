import { api } from "../api.js";
import { state } from "../state.js";
import { $, debounce } from "../utils.js";
import { setActionIcon } from "../components/action-icons.js?v=0.1.0";

const MIN_LEFT_COLUMN = 170;
const MIN_CENTER_COLUMN = 240;
const PREFERRED_CENTER_COLUMN = 640;
const MIN_RIGHT_COLUMN = 190;
const MAX_STORED_COLUMN = Number.MAX_SAFE_INTEGER;
const MIN_GALLERY_HEIGHT = 96;
const MIN_OUTPUT_HEIGHT = 120;
const MAX_GALLERY_HEIGHT = Number.MAX_SAFE_INTEGER;
const MIN_LIVE_PREVIEW_HEIGHT = 240;
const MAX_LIVE_PREVIEW_HEIGHT = 450;
const MIN_STARTUP_DEFAULTS_WIDTH = 260;
const MAX_STARTUP_DEFAULTS_WIDTH = 520;
const PANEL_SCALE_MIN = 60;
const PANEL_SCALE_MAX = 240;
const PANEL_SCALE_STEP = 10;
const WORKSPACE_LAYOUT_VERSION = 1;
const VALID_ZONES = Object.freeze(["left", "center", "right"]);

export const DEFAULT_PANEL_SCALES = Object.freeze({
  controls: 100,
  output_viewer: 100,
  recent_outputs: 100,
  live_preview: 100,
  active_prompt_assets: 100,
  memory_status: 100,
  runtime_status: 100,
  queue: 100,
  recent_runs: 100,
  prompt_presets: 100,
  model_refresh: 100,
  maintenance: 100,
  startup_defaults: 100,
});

export const DEFAULT_PANEL_ZONES = Object.freeze({
  left: Object.freeze(["generation_controls"]),
  center: Object.freeze(["output_viewer", "recent_outputs"]),
  right: Object.freeze([
    "live_preview",
    "active_prompt_assets",
    "memory_status",
    "runtime_status",
    "queue",
    "recent_runs",
    "prompt_presets",
    "model_refresh",
    "maintenance",
  ]),
});

const PANEL_DEFAULT_ZONE = Object.freeze(
  Object.entries(DEFAULT_PANEL_ZONES).reduce((result, [zone, panelIds]) => {
    panelIds.forEach((panelId) => { result[panelId] = zone; });
    return result;
  }, {}),
);

const PROTECTED_PANEL_IDS = new Set(["output_viewer"]);

export const DEFAULT_LAYOUT = Object.freeze({
  workspace_layout_version: WORKSPACE_LAYOUT_VERSION,
  left_column_width: 330,
  right_column_width: 360,
  gallery_panel_height: 132,
  live_preview_panel_height: 360,
  live_preview_collapsed: false,
  follow_newest_output: false,
  startup_defaults_open: false,
  startup_defaults_pinned: false,
  startup_defaults_width: 300,
  panel_zones: DEFAULT_PANEL_ZONES,
  collapsed_panels: Object.freeze([]),
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

function normalizedPanelIdList(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  return value
    .map((item) => String(item || "").trim())
    .filter((item) => PANEL_DEFAULT_ZONE[item] && !seen.has(item) && seen.add(item));
}

function normalizePanelZones(value = {}) {
  const stored = value && typeof value === "object" ? value : {};
  const result = { left: [], center: [], right: [] };
  const assigned = new Set();

  VALID_ZONES.forEach((zone) => {
    normalizedPanelIdList(stored[zone]).forEach((panelId) => {
      if (assigned.has(panelId)) return;
      if (PROTECTED_PANEL_IDS.has(panelId) && zone !== "center") return;
      result[zone].push(panelId);
      assigned.add(panelId);
    });
  });

  VALID_ZONES.forEach((zone) => {
    DEFAULT_PANEL_ZONES[zone].forEach((panelId) => {
      if (assigned.has(panelId)) return;
      result[zone].push(panelId);
      assigned.add(panelId);
    });
  });

  result.center = ["output_viewer", ...result.center.filter((panelId) => panelId !== "output_viewer")];
  return result;
}

function normalizeCollapsedPanels(value, livePreviewCollapsed = false) {
  const collapsed = new Set(normalizedPanelIdList(value));
  if (livePreviewCollapsed) collapsed.add("live_preview");
  PROTECTED_PANEL_IDS.forEach((panelId) => collapsed.delete(panelId));
  return [...collapsed];
}

function normalizeLayout(settings = {}) {
  const stored = settings.ui_layout && typeof settings.ui_layout === "object"
    ? settings.ui_layout
    : {};
  const hasStructuredCollapsedPanels = Array.isArray(stored.collapsed_panels);
  const collapsedPanels = normalizeCollapsedPanels(
    stored.collapsed_panels,
    !hasStructuredCollapsedPanels && Boolean(stored.live_preview_collapsed),
  );
  return {
    workspace_layout_version: WORKSPACE_LAYOUT_VERSION,
    left_column_width: clamp(stored.left_column_width ?? DEFAULT_LAYOUT.left_column_width, MIN_LEFT_COLUMN, MAX_STORED_COLUMN),
    right_column_width: clamp(stored.right_column_width ?? DEFAULT_LAYOUT.right_column_width, MIN_RIGHT_COLUMN, MAX_STORED_COLUMN),
    gallery_panel_height: clamp(stored.gallery_panel_height ?? DEFAULT_LAYOUT.gallery_panel_height, MIN_GALLERY_HEIGHT, MAX_GALLERY_HEIGHT),
    live_preview_panel_height: clamp(stored.live_preview_panel_height ?? DEFAULT_LAYOUT.live_preview_panel_height, MIN_LIVE_PREVIEW_HEIGHT, MAX_LIVE_PREVIEW_HEIGHT),
    live_preview_collapsed: collapsedPanels.includes("live_preview"),
    follow_newest_output: Boolean(stored.follow_newest_output),
    startup_defaults_open: Boolean(stored.startup_defaults_open),
    startup_defaults_pinned: Boolean(stored.startup_defaults_pinned),
    startup_defaults_width: clamp(
      stored.startup_defaults_width ?? DEFAULT_LAYOUT.startup_defaults_width,
      MIN_STARTUP_DEFAULTS_WIDTH,
      MAX_STARTUP_DEFAULTS_WIDTH,
    ),
    panel_zones: normalizePanelZones(stored.panel_zones),
    collapsed_panels: collapsedPanels,
    panel_scales: normalizePanelScales(stored.panel_scales),
  };
}

function cloneLayout(layout) {
  return {
    ...layout,
    panel_zones: Object.fromEntries(VALID_ZONES.map((zone) => [zone, [...layout.panel_zones[zone]]])),
    collapsed_panels: [...layout.collapsed_panels],
    panel_scales: { ...layout.panel_scales },
  };
}

function mergeLayoutValues(baseLayout = {}, overrideLayout = {}) {
  return {
    ...baseLayout,
    ...overrideLayout,
    panel_zones: {
      ...(baseLayout.panel_zones || {}),
      ...(overrideLayout.panel_zones || {}),
    },
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

function workspaceHorizontalMetrics(layout) {
  const workspace = $("#workspace");
  const styles = window.getComputedStyle(workspace);
  const gap = Number.parseFloat(styles.columnGap || styles.gap) || 0;
  const splitterWidth = [$("#leftColumnSplitter"), $("#rightColumnSplitter")]
    .reduce((total, item) => total + (item?.getBoundingClientRect().width || 0), 0);
  const drawerWidth = layout.startup_defaults_open && layout.startup_defaults_pinned
    ? layout.startup_defaults_width
    : 0;
  const gapCount = drawerWidth > 0 ? 5 : 4;
  const fixedWidth = splitterWidth + drawerWidth + (gap * gapCount);
  const width = workspace.getBoundingClientRect().width;
  const usableWidth = Math.max(0, width - fixedWidth);
  const centerMinimum = Math.min(
    PREFERRED_CENTER_COLUMN,
    Math.max(MIN_CENTER_COLUMN, usableWidth - MIN_LEFT_COLUMN - MIN_RIGHT_COLUMN),
  );
  return { width, fixedWidth, usableWidth, centerMinimum };
}

function fitHorizontalLayout(layout, preference = "balanced") {
  const { usableWidth, centerMinimum } = workspaceHorizontalMetrics(layout);
  const maximumSideTotal = Math.max(
    MIN_LEFT_COLUMN + MIN_RIGHT_COLUMN,
    usableWidth - centerMinimum,
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
    center_column_min_width: Math.round(centerMinimum),
  };
}

function copyToState(layout, renderedHorizontal = layout, effectiveGalleryHeight = layout.gallery_panel_height) {
  state.layout = {
    leftColumnWidth: renderedHorizontal.left_column_width,
    rightColumnWidth: renderedHorizontal.right_column_width,
    centerColumnMinWidth: renderedHorizontal.center_column_min_width,
    galleryPanelHeight: effectiveGalleryHeight,
    livePreviewPanelHeight: layout.live_preview_panel_height,
    livePreviewCollapsed: layout.live_preview_collapsed,
    followNewestOutput: layout.follow_newest_output,
    startupDefaultsOpen: layout.startup_defaults_open,
    startupDefaultsPinned: layout.startup_defaults_pinned,
    panelZones: Object.fromEntries(VALID_ZONES.map((zone) => [zone, [...layout.panel_zones[zone]]])),
    collapsedPanels: [...layout.collapsed_panels],
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
  setActionIcon(decrease, "text-decrease", {
    label: `Make ${label.toLowerCase()} text and controls smaller`,
    title: `Make ${label.toLowerCase()} text and controls smaller`,
    replace: true,
  });

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "panel-scale-reset";
  reset.title = `Reset ${label.toLowerCase()} scale to 100%`;
  reset.setAttribute("aria-label", reset.title);

  const increase = document.createElement("button");
  increase.type = "button";
  increase.className = "panel-scale-button";
  setActionIcon(increase, "text-increase", {
    label: `Make ${label.toLowerCase()} text and controls larger`,
    title: `Make ${label.toLowerCase()} text and controls larger`,
    replace: true,
  });

  decrease.addEventListener("click", () => onChange(-PANEL_SCALE_STEP));
  increase.addEventListener("click", () => onChange(PANEL_SCALE_STEP));
  reset.addEventListener("click", () => onChange(0, true));
  group.dataset.actionGroup = "panel-scale";
  group.dataset.actionPriority = "28";
  group.append(decrease, reset, increase);
  panel._panelScaleOutput = reset;
  return group;
}

function headingActionsFor(panel) {
  const heading = panel.querySelector(":scope > .panel-heading");
  if (!heading) return null;
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
  return actions;
}

function installPanelScaleControls(onScaleChange) {
  document.querySelectorAll("[data-panel-scale-key]").forEach((panel) => {
    if (panel.querySelector(":scope > .panel-heading .panel-scale-controls")) return;
    const key = panel.dataset.panelScaleKey;
    const heading = panel.querySelector(":scope > .panel-heading");
    if (!key || !heading) return;
    heading.classList.add("panel-scale-exempt");
    const label = heading.querySelector("h2")?.textContent?.trim() || "Panel";
    const controls = createPanelScaleControls(panel, label, (delta, reset = false) => {
      onScaleChange(key, reset ? 100 : delta, reset);
    });
    headingActionsFor(panel)?.prepend(controls);
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

function zoneForPanel(layout, panelId) {
  return VALID_ZONES.find((zone) => layout.panel_zones[zone].includes(panelId)) || PANEL_DEFAULT_ZONE[panelId];
}

function panelElement(panelId) {
  return document.querySelector(`[data-workspace-panel="${panelId}"]`);
}

function zoneElement(zone) {
  return document.querySelector(`[data-workspace-zone="${zone}"]`);
}

function serializeLayout(layout) {
  const payload = cloneLayout(layout);
  payload.live_preview_collapsed = payload.collapsed_panels.includes("live_preview");
  return payload;
}

export function bindWorkspaceLayout(settings = {}) {
  let sourceSettings = settings;
  let layout = normalizeLayout(settings);
  let renderedHorizontal = fitHorizontalLayout(layout);
  let effectiveGalleryHeight = layout.gallery_panel_height;
  let gallerySyncFrame = 0;
  let draggedPanelId = "";
  let resizeActive = false;

  const persistSoon = debounce(async () => {
    try {
      const response = await api.saveWorkspaceLayout(serializeLayout(layout));
      const saved = {
        ...state.settings,
        ...(response.settings || {}),
        ui_layout: response.layout || response.settings?.ui_layout || serializeLayout(layout),
      };
      state.settings = saved;
      sourceSettings = saved;
    } catch (error) {
      console.error("Unable to save WebUI layout", error);
    }
  }, 250);

  const syncPanelToggle = (panelId) => {
    const panel = panelElement(panelId);
    if (!panel) return;
    const collapsed = layout.collapsed_panels.includes(panelId);
    panel.classList.toggle("is-collapsed", collapsed);
    panel.querySelectorAll(":scope > .panel-heading [data-workspace-collapse]").forEach((button) => {
      button.setAttribute("aria-expanded", String(!collapsed));
      const label = `${collapsed ? "Expand" : "Collapse"} ${panel.querySelector("h2")?.textContent || "panel"}`;
      button.setAttribute("aria-label", label);
      button.title = label;
    });
  };

  const refreshPanelControlStates = () => {
    document.querySelectorAll("[data-workspace-panel]").forEach((panel) => {
      const panelId = panel.dataset.workspacePanel;
      const currentZone = zoneForPanel(layout, panelId);
      panel.querySelectorAll("[data-workspace-move]").forEach((button) => {
        button.disabled = PROTECTED_PANEL_IDS.has(panelId) || button.dataset.workspaceMove === currentZone;
      });
      panel.querySelectorAll("[data-workspace-current-zone]").forEach((output) => {
        output.textContent = currentZone;
      });
      syncPanelToggle(panelId);
    });
  };

  const syncStructuralSplitters = () => {
    const center = zoneElement("center");
    const outputPanel = panelElement("output_viewer");
    const recentPanel = panelElement("recent_outputs");
    const centerSplitter = $("#centerSplitter");
    const recentInCenter = center && recentPanel?.parentElement === center && outputPanel?.parentElement === center;
    if (centerSplitter) {
      centerSplitter.classList.toggle("is-hidden", !recentInCenter || layout.collapsed_panels.includes("recent_outputs"));
      if (recentInCenter && centerSplitter.nextElementSibling !== recentPanel) {
        center.insertBefore(centerSplitter, recentPanel);
      }
    }

    const right = zoneElement("right");
    const livePanel = panelElement("live_preview");
    const liveSplitter = $("#livePreviewSplitter");
    const liveInRight = right && livePanel?.parentElement === right;
    if (liveSplitter) {
      liveSplitter.classList.toggle("is-hidden", !liveInRight);
      if (liveInRight && livePanel.nextElementSibling !== liveSplitter) {
        livePanel.after(liveSplitter);
      }
    }
  };

  const applyPanelPositions = () => {
    VALID_ZONES.forEach((zone) => {
      const container = zoneElement(zone);
      if (!container) return;
      layout.panel_zones[zone].forEach((panelId) => {
        const panel = panelElement(panelId);
        if (panel) container.append(panel);
      });
    });
    syncStructuralSplitters();
    refreshPanelControlStates();
  };

  const updateSeparatorValues = () => {
    const metrics = workspaceHorizontalMetrics(layout);
    const maxLeft = Math.max(MIN_LEFT_COLUMN, metrics.usableWidth - metrics.centerMinimum - MIN_RIGHT_COLUMN);
    const maxRight = Math.max(MIN_RIGHT_COLUMN, metrics.usableWidth - metrics.centerMinimum - MIN_LEFT_COLUMN);
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
      const recentPanel = panelElement("recent_outputs");
      const splitter = $("#centerSplitter");
      if (!workspace || !browser) return;

      const recentInCenter = recentPanel?.parentElement === browser;
      if (window.matchMedia("(max-width: 720px)").matches || !recentInCenter) {
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

  const applyStartupDefaultsDrawer = () => {
    const workspace = $("#workspace");
    const drawer = $("#startupDefaultsDrawer");
    const trigger = $("#startupDefaultsTrigger");
    const pin = $("#pinStartupDefaultsButton");
    if (!workspace || !drawer) return;
    const open = layout.startup_defaults_open;
    const pinned = open && layout.startup_defaults_pinned;
    workspace.classList.toggle("has-open-startup-defaults", open);
    workspace.classList.toggle("has-pinned-startup-defaults", pinned);
    drawer.hidden = !open;
    drawer.classList.toggle("is-pinned", pinned);
    drawer.classList.toggle("is-temporary", open && !pinned);
    workspace.style.setProperty("--startup-defaults-width", `${layout.startup_defaults_width}px`);
    trigger?.setAttribute("aria-expanded", String(open));
    trigger?.classList.toggle("is-active", open);
    if (pin) {
      pin.setAttribute("aria-pressed", String(pinned));
      const label = pinned ? "Unpin startup defaults" : "Pin startup defaults as a workspace column";
      pin.setAttribute("aria-label", label);
      pin.title = label;
    }
  };

  const applyLayout = (preference = "balanced", { structural = true } = {}) => {
    const workspace = $("#workspace");
    applyStartupDefaultsDrawer();
    renderedHorizontal = fitHorizontalLayout(layout, preference);
    workspace.style.setProperty("--left-column-width", `${renderedHorizontal.left_column_width}px`);
    workspace.style.setProperty("--right-column-width", `${renderedHorizontal.right_column_width}px`);
    workspace.style.setProperty("--center-column-min-width", `${renderedHorizontal.center_column_min_width}px`);
    workspace.style.setProperty("--gallery-panel-height", `${layout.gallery_panel_height}px`);
    workspace.style.setProperty("--live-preview-panel-height", `${layout.live_preview_panel_height}px`);
    if (structural) applyPanelPositions();
    applyPanelScales(layout.panel_scales);

    const livePanel = $("#livePreviewPanel");
    const liveToggle = $("#livePreviewToggle");
    const liveCollapsed = layout.collapsed_panels.includes("live_preview");
    layout.live_preview_collapsed = liveCollapsed;
    livePanel?.classList.toggle("is-collapsed", liveCollapsed);
    if (liveToggle) {
      liveToggle.setAttribute("aria-expanded", String(!liveCollapsed));
      const label = liveCollapsed ? "Expand live preview" : "Collapse live preview";
      liveToggle.setAttribute("aria-label", label);
      liveToggle.title = label;
    }

    const followNewest = $("#followNewestOutput");
    if (followNewest) followNewest.checked = layout.follow_newest_output;
    copyToState(layout, renderedHorizontal, effectiveGalleryHeight);
    updateSeparatorValues();
    syncGalleryStackHeight();
  };

  const update = (changes, {
    persist = true,
    preference = "balanced",
    useRenderedHorizontal = false,
    structural = true,
  } = {}) => {
    const merged = mergeLayoutValues(layout, changes);
    layout = normalizeLayout({ ui_layout: merged });
    applyLayout(preference, { structural });
    if (useRenderedHorizontal) {
      layout.left_column_width = renderedHorizontal.left_column_width;
      layout.right_column_width = renderedHorizontal.right_column_width;
    }
    if (persist) persistSoon();
  };

  const movePanel = (panelId, targetZone, { persist = true } = {}) => {
    if (!VALID_ZONES.includes(targetZone) || PROTECTED_PANEL_IDS.has(panelId)) return;
    const nextZones = Object.fromEntries(
      VALID_ZONES.map((zone) => [zone, layout.panel_zones[zone].filter((item) => item !== panelId)]),
    );
    nextZones[targetZone].push(panelId);
    update({ panel_zones: nextZones }, { persist });
  };

  const resetPanelPosition = (panelId) => {
    const defaultZone = PANEL_DEFAULT_ZONE[panelId];
    if (!defaultZone || PROTECTED_PANEL_IDS.has(panelId)) return;
    const nextZones = Object.fromEntries(
      VALID_ZONES.map((zone) => [zone, layout.panel_zones[zone].filter((item) => item !== panelId)]),
    );
    const defaultOrder = DEFAULT_PANEL_ZONES[defaultZone];
    const targetIndex = defaultOrder.indexOf(panelId);
    const insertionIndex = nextZones[defaultZone].findIndex((existingId) => {
      const existingDefaultIndex = defaultOrder.indexOf(existingId);
      return existingDefaultIndex >= 0 && existingDefaultIndex > targetIndex;
    });
    if (insertionIndex < 0) nextZones[defaultZone].push(panelId);
    else nextZones[defaultZone].splice(insertionIndex, 0, panelId);
    const collapsed = layout.collapsed_panels.filter((item) => item !== panelId);
    update({ panel_zones: nextZones, collapsed_panels: collapsed });
  };

  const togglePanelCollapsed = (panelId) => {
    if (PROTECTED_PANEL_IDS.has(panelId)) return;
    const collapsed = new Set(layout.collapsed_panels);
    if (collapsed.has(panelId)) collapsed.delete(panelId);
    else collapsed.add(panelId);
    update({ collapsed_panels: [...collapsed] });
  };

  const closeDockMenus = (except = null) => {
    document.querySelectorAll(".panel-dock-menu[open]").forEach((details) => {
      if (details !== except) details.removeAttribute("open");
    });
  };

  const installPanelDockControls = () => {
    document.querySelectorAll("[data-workspace-panel]").forEach((panel) => {
      if (panel.dataset.workspaceControlsInstalled === "true") return;
      const panelId = panel.dataset.workspacePanel;
      const protectedPanel = PROTECTED_PANEL_IDS.has(panelId) || panel.dataset.workspaceProtected === "true";
      const heading = panel.querySelector(":scope > .panel-heading");
      const actions = headingActionsFor(panel);
      if (!heading || !actions) return;
      panel.dataset.workspaceControlsInstalled = "true";

      if (!protectedPanel) {
        const dragHandle = document.createElement("button");
        dragHandle.type = "button";
        dragHandle.className = "panel-drag-handle";
        dragHandle.dataset.actionPinned = "true";
        setActionIcon(dragHandle, "drag", {
          label: "Drag panel to another workspace column",
          title: "Drag panel to another workspace column",
          replace: true,
        });
        dragHandle.draggable = true;
        dragHandle.addEventListener("dragstart", (event) => {
          draggedPanelId = panelId;
          panel.classList.add("is-dragging");
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", panelId);
        });
        dragHandle.addEventListener("dragend", () => {
          draggedPanelId = "";
          panel.classList.remove("is-dragging");
          document.querySelectorAll(".workspace-column.is-drop-target").forEach((zone) => zone.classList.remove("is-drop-target"));
        });
        actions.append(dragHandle);
      }

      if (!protectedPanel) {
        let collapse = panel.querySelector(":scope > .panel-heading .panel-toggle");
        if (!collapse) {
          collapse = document.createElement("button");
          collapse.type = "button";
          collapse.className = "workspace-panel-collapse";
          setActionIcon(collapse, "chevron-up", { label: "Collapse panel", title: "Collapse panel", replace: true });
          actions.append(collapse);
        }
        collapse.dataset.workspaceCollapse = panelId;
        collapse.dataset.layoutBound = "true";
        collapse.dataset.actionPinned = "true";
        collapse.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          togglePanelCollapsed(panelId);
        });
      }

      const menu = document.createElement("details");
      menu.className = "panel-dock-menu";
      menu.dataset.actionPinned = "true";
      const summary = document.createElement("summary");
      summary.className = "panel-dock-menu-trigger";
      const menuLabel = protectedPanel ? "Primary panel is locked to the center column" : "Panel layout options";
      setActionIcon(summary, protectedPanel ? "lock" : "more", { label: menuLabel, title: menuLabel, replace: true });
      menu.append(summary);

      const popup = document.createElement("div");
      popup.className = "panel-dock-menu-popup";
      const zoneLabel = document.createElement("small");
      zoneLabel.innerHTML = 'Current column: <strong data-workspace-current-zone></strong>';
      popup.append(zoneLabel);

      VALID_ZONES.forEach((zone) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.workspaceMove = zone;
        button.textContent = `Move to ${zone} column`;
        button.disabled = protectedPanel;
        button.addEventListener("click", () => {
          movePanel(panelId, zone);
          menu.removeAttribute("open");
        });
        popup.append(button);
      });

      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = protectedPanel ? "Primary center panel" : "Reset panel position";
      reset.disabled = protectedPanel;
      reset.addEventListener("click", () => {
        resetPanelPosition(panelId);
        menu.removeAttribute("open");
      });
      popup.append(reset);
      menu.append(popup);
      menu.addEventListener("toggle", () => {
        if (menu.open) closeDockMenus(menu);
      });
      actions.append(menu);
    });

    VALID_ZONES.forEach((zone) => {
      const container = zoneElement(zone);
      if (!container || container.dataset.workspaceDropInstalled === "true") return;
      container.dataset.workspaceDropInstalled = "true";
      container.addEventListener("dragover", (event) => {
        if (!draggedPanelId || PROTECTED_PANEL_IDS.has(draggedPanelId)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        container.classList.add("is-drop-target");
      });
      container.addEventListener("dragleave", (event) => {
        if (!container.contains(event.relatedTarget)) container.classList.remove("is-drop-target");
      });
      container.addEventListener("drop", (event) => {
        event.preventDefault();
        const panelId = draggedPanelId || event.dataTransfer.getData("text/plain");
        container.classList.remove("is-drop-target");
        if (panelId) movePanel(panelId, zone);
      });
    });
  };

  installPanelDockControls();
  installPanelScaleControls((key, value, reset = false) => {
    const current = layout.panel_scales[key] ?? 100;
    const next = reset ? 100 : clamp(current + value, PANEL_SCALE_MIN, PANEL_SCALE_MAX);
    update({ panel_scales: { [key]: next } });
  });
  applyLayout();

  const beginPointerResize = (event, onMove) => {
    if (window.matchMedia("(max-width: 720px)").matches) return false;
    event.preventDefault();
    const handle = event.currentTarget;
    const pointerId = event.pointerId;
    resizeActive = true;
    document.body.classList.add("is-resizing");
    try {
      handle.setPointerCapture(pointerId);
    } catch (_) {
      // Window-level listeners below still keep the resize stable if capture is unavailable.
    }

    const move = (moveEvent) => {
      if (moveEvent.pointerId !== pointerId) return;
      onMove(moveEvent);
    };
    const finish = (finishEvent) => {
      if (finishEvent?.pointerId != null && finishEvent.pointerId !== pointerId) return;
      resizeActive = false;
      document.body.classList.remove("is-resizing");
      window.removeEventListener("pointermove", move, true);
      window.removeEventListener("pointerup", finish, true);
      window.removeEventListener("pointercancel", finish, true);
      try {
        if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
      } catch (_) {
        // The element may have lost capture during a browser/layout transition.
      }
      syncGalleryStackHeight();
      persistSoon();
    };
    window.addEventListener("pointermove", move, true);
    window.addEventListener("pointerup", finish, true);
    window.addEventListener("pointercancel", finish, true);
    return true;
  };

  $("#leftColumnSplitter")?.addEventListener("pointerdown", (event) => {
    const bounds = $("#workspace").getBoundingClientRect();
    beginPointerResize(event, (moveEvent) => {
      update(
        { left_column_width: moveEvent.clientX - bounds.left },
        { persist: false, preference: "left", useRenderedHorizontal: true, structural: false },
      );
    });
  });

  $("#rightColumnSplitter")?.addEventListener("pointerdown", (event) => {
    const bounds = $("#workspace").getBoundingClientRect();
    beginPointerResize(event, (moveEvent) => {
      update(
        { right_column_width: bounds.right - moveEvent.clientX },
        { persist: false, preference: "right", useRenderedHorizontal: true, structural: false },
      );
    });
  });

  $("#centerSplitter")?.addEventListener("pointerdown", (event) => {
    const recentPanel = panelElement("recent_outputs");
    const browser = zoneElement("center");
    if (recentPanel?.parentElement !== browser) return;
    const splitter = event.currentTarget;
    const bounds = browser.getBoundingClientRect();
    const splitterHeight = splitter.getBoundingClientRect().height || 10;
    const maximum = Math.min(
      MAX_GALLERY_HEIGHT,
      Math.max(MIN_GALLERY_HEIGHT, bounds.height - MIN_OUTPUT_HEIGHT - splitterHeight),
    );
    const startY = event.clientY;
    const startHeight = clamp(
      recentPanel.getBoundingClientRect().height || effectiveGalleryHeight || layout.gallery_panel_height,
      MIN_GALLERY_HEIGHT,
      maximum,
    );
    beginPointerResize(event, (moveEvent) => {
      const deltaY = moveEvent.clientY - startY;
      update(
        { gallery_panel_height: clamp(startHeight - deltaY, MIN_GALLERY_HEIGHT, maximum) },
        { persist: false, structural: false },
      );
    });
  });

  $("#centerSplitter")?.addEventListener("dblclick", (event) => {
    event.preventDefault();
    update({ gallery_panel_height: DEFAULT_LAYOUT.gallery_panel_height });
  });

  $("#livePreviewSplitter")?.addEventListener("pointerdown", (event) => {
    if (panelElement("live_preview")?.parentElement !== zoneElement("right")) return;
    const startY = event.clientY;
    const startHeight = layout.live_preview_panel_height;
    beginPointerResize(event, (moveEvent) => {
      const nextHeight = startHeight + (moveEvent.clientY - startY);
      update(
        { live_preview_panel_height: clamp(nextHeight, MIN_LIVE_PREVIEW_HEIGHT, MAX_LIVE_PREVIEW_HEIGHT) },
        { persist: false, structural: false },
      );
    });
  });

  const keyboardStep = (event) => event.shiftKey ? 48 : 12;
  $("#leftColumnSplitter")?.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? keyboardStep(event) : -keyboardStep(event);
    update(
      { left_column_width: renderedHorizontal.left_column_width + delta },
      { preference: "left", useRenderedHorizontal: true },
    );
  });

  $("#rightColumnSplitter")?.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? keyboardStep(event) : -keyboardStep(event);
    update(
      { right_column_width: renderedHorizontal.right_column_width + delta },
      { preference: "right", useRenderedHorizontal: true },
    );
  });

  $("#centerSplitter")?.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowUp" ? keyboardStep(event) : -keyboardStep(event);
    update({ gallery_panel_height: layout.gallery_panel_height + delta });
  });

  $("#livePreviewSplitter")?.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? keyboardStep(event) : -keyboardStep(event);
    update({ live_preview_panel_height: layout.live_preview_panel_height + delta });
  });

  const followNewest = $("#followNewestOutput");
  followNewest?.addEventListener("change", (event) => {
    update({ follow_newest_output: event.target.checked });
  });

  const startupTrigger = $("#startupDefaultsTrigger");
  startupTrigger?.addEventListener("click", () => {
    if (layout.startup_defaults_open) {
      update({ startup_defaults_open: false, startup_defaults_pinned: false });
    } else {
      update({ startup_defaults_open: true, startup_defaults_pinned: false });
      window.setTimeout(() => $("#startupDefaultsDrawer")?.focus?.(), 0);
    }
  });
  $("#closeStartupDefaultsButton")?.addEventListener("click", () => {
    update({ startup_defaults_open: false, startup_defaults_pinned: false });
    startupTrigger?.focus();
  });
  $("#pinStartupDefaultsButton")?.addEventListener("click", () => {
    update({
      startup_defaults_open: true,
      startup_defaults_pinned: !layout.startup_defaults_pinned,
    });
  });

  document.addEventListener("pointerdown", (event) => {
    if (!layout.startup_defaults_open || layout.startup_defaults_pinned) return;
    const drawer = $("#startupDefaultsDrawer");
    if (drawer?.contains(event.target) || startupTrigger?.contains(event.target)) return;
    update({ startup_defaults_open: false }, { persist: true });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && layout.startup_defaults_open && !layout.startup_defaults_pinned) {
      update({ startup_defaults_open: false });
      startupTrigger?.focus();
    }
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".panel-dock-menu")) closeDockMenus();
  });

  const workspaceResizeObserver = new ResizeObserver(() => {
    if (resizeActive) {
      syncGalleryStackHeight();
      return;
    }
    applyLayout();
  });
  workspaceResizeObserver.observe($("#workspace"));
  const galleryResizeObserver = new ResizeObserver(syncGalleryStackHeight);
  galleryResizeObserver.observe($("#outputBrowser"));

  return {
    reset: async () => {
      layout = resolveLayoutDefaultsForScale(sourceSettings, sourceSettings?.ui_scale ?? state.settings?.ui_scale ?? 100);
      applyLayout();
      const response = await api.saveWorkspaceLayout(serializeLayout(layout));
      const saved = {
        ...state.settings,
        ...(response.settings || {}),
        ui_layout: response.layout || response.settings?.ui_layout || serializeLayout(layout),
      };
      state.settings = saved;
      sourceSettings = saved;
      return cloneLayout(layout);
    },
    saveCurrentScaleDefault: async (scale) => {
      const scaleKey = String(Math.round(Number(scale ?? sourceSettings?.ui_scale ?? state.settings?.ui_scale ?? 100) || 100));
      const payload = serializeLayout(layout);
      const saved = await api.saveSettings({
        ui_layout: payload,
        ui_scale_layout_defaults: { [scaleKey]: payload },
      });
      state.settings = saved;
      sourceSettings = saved;
      return { scale: scaleKey, layout: payload, settings: saved };
    },
    current: () => cloneLayout(layout),
    movePanel,
    togglePanelCollapsed,
  };
}
