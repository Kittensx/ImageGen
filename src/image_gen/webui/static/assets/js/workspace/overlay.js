import { setComponentShellState } from "../components/component-shell.js?v=content-capabilities2";

const controllers = new WeakMap();
const STORAGE_PREFIX = "image-gen.workspace-overlay.v1.";

function normalizeMode(value, modes, fallback) {
  const token = String(value || "").trim().toLowerCase();
  return modes.includes(token) ? token : fallback;
}

function clamp(value, minimum, maximum, fallback) {
  const number = Number(value);
  const resolved = Number.isFinite(number) ? number : fallback;
  return Math.max(minimum, Math.min(maximum, Math.round(resolved)));
}

function storageKey(componentId) {
  return `${STORAGE_PREFIX}${String(componentId || "").trim().toLowerCase()}`;
}

function readPreference(componentId) {
  try {
    const value = JSON.parse(window.localStorage.getItem(storageKey(componentId)) || "null");
    return value && typeof value === "object" ? value : {};
  } catch (_error) {
    return {};
  }
}

function writePreference(componentId, payload) {
  try { window.localStorage.setItem(storageKey(componentId), JSON.stringify(payload)); } catch (_error) {}
}

function button(label, className) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  node.textContent = label;
  return node;
}

function ensureCompanion(workspaceRoot, componentId, kind, label) {
  const selector = `[data-workspace-overlay-${kind}][data-workspace-overlay-for="${CSS.escape(componentId)}"]`;
  let node = document.querySelector(selector);
  if (!node) {
    if (kind === "backdrop") {
      node = document.createElement("div");
      node.className = "workspace-overlay-backdrop";
      node.dataset.workspaceOverlayBackdrop = "";
      node.setAttribute("aria-hidden", "true");
    } else if (kind === "close-button") {
      node = button("×", "workspace-overlay-floating-close");
      node.dataset.workspaceOverlayCloseButton = "";
      node.setAttribute("aria-label", `Close ${label || "overlay"}`);
      node.title = `Close ${label || "overlay"}`;
    } else {
      node = button(label || "Open", "workspace-overlay-edge-tab");
      node.dataset.workspaceOverlayEdgeTab = "";
      node.setAttribute("aria-label", `Open ${label || "details"}`);
      node.title = `Open ${label || "details"}`;
    }
    node.dataset.workspaceOverlayFor = componentId;
    node.hidden = true;
  }
  // Fixed overlay companions must live outside workspace overflow/transform
  // contexts. Moving existing declarative companions to body prevents a
  // drawer restore tab or focused close button from being clipped by the page.
  if (document.body && node.parentElement !== document.body) document.body.append(node);
  return node;
}

function ensureResizeHandle(root, label) {
  let handle = root.querySelector(":scope > [data-workspace-overlay-resize-handle]");
  if (handle) return handle;
  handle = document.createElement("div");
  handle.className = "workspace-overlay-resize-handle";
  handle.dataset.workspaceOverlayResizeHandle = "";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "vertical");
  handle.setAttribute("tabindex", "0");
  handle.setAttribute("aria-label", `Resize ${label}`);
  handle.title = `Drag to resize ${label}; use Left/Right arrows for small steps. Home resets width.`;
  root.prepend(handle);
  return handle;
}

export function bindWorkspaceComponentOverlay(root, descriptor, workspaceRoot) {
  const config = descriptor?.overlay && typeof descriptor.overlay === "object" ? descriptor.overlay : null;
  if (!root || !config) return null;
  if (controllers.has(root)) return controllers.get(root);

  const componentId = String(descriptor.componentId || root.dataset.workspaceComponent || "overlay");
  const modes = [...new Set((config.modes || ["drawer"]).map((item) => String(item).trim().toLowerCase()).filter(Boolean))];
  const defaultMode = normalizeMode(config.defaultMode || modes[0] || "drawer", modes, modes[0] || "drawer");
  const label = String(config.label || descriptor.title || "Details");
  const minWidth = Math.max(240, Number(config.minDrawerWidth || 360));
  const maxWidth = Math.max(minWidth, Number(config.maxDrawerWidth || 900));
  const defaultWidth = clamp(config.defaultDrawerWidth || 520, minWidth, maxWidth, 520);
  const preference = readPreference(componentId);
  const state = {
    mode: normalizeMode(preference.mode, modes, defaultMode),
    width: clamp(preference.width, minWidth, maxWidth, defaultWidth),
    open: false,
    available: config.defaultAvailable === true,
  };

  root.classList.add("workspace-overlay-surface");
  root.dataset.workspaceOverlay = "true";
  root.dataset.workspaceOverlayFor = componentId;

  const backdrop = ensureCompanion(workspaceRoot, componentId, "backdrop", label);
  const edge = config.edgeRestore === false ? null : ensureCompanion(workspaceRoot, componentId, "edge-tab", label);
  const floatingClose = config.floatingClose === false ? null : ensureCompanion(workspaceRoot, componentId, "close-button", label);
  if (edge) edge.textContent = String(config.edgeLabel || label);
  const resize = config.resizableDrawer === false ? null : ensureResizeHandle(root, label);

  const persist = () => writePreference(componentId, { mode: state.mode, width: state.width });

  const updateBounds = () => {
    if (!workspaceRoot) return;
    const rect = workspaceRoot.getBoundingClientRect();
    const top = Math.max(8, rect.top + 8);
    const left = Math.max(8, rect.left + 8);
    const right = Math.max(8, window.innerWidth - rect.right + 8);
    const bottom = Math.max(8, window.innerHeight - rect.bottom + 8);
    [root, backdrop, edge, floatingClose].filter(Boolean).forEach((node) => {
      node.style.setProperty("--workspace-overlay-top", `${Math.round(top)}px`);
      node.style.setProperty("--workspace-overlay-left", `${Math.round(left)}px`);
      node.style.setProperty("--workspace-overlay-right", `${Math.round(right)}px`);
      node.style.setProperty("--workspace-overlay-bottom", `${Math.round(bottom)}px`);
    });
  };

  const overlayControls = (attribute) => {
    const selector = `[${attribute}]`;
    const external = `[${attribute}][data-workspace-overlay-for="${CSS.escape(componentId)}"]`;
    return [...new Set([...root.querySelectorAll(selector), ...document.querySelectorAll(external)])];
  };

  const syncControls = () => {
    overlayControls("data-workspace-overlay-mode-button").forEach((control) => {
      const mode = String(control.dataset.workspaceOverlayModeButton || "").trim().toLowerCase();
      control.setAttribute("aria-pressed", String(mode === state.mode));
    });
    overlayControls("data-workspace-overlay-mode-select").forEach((control) => {
      control.value = state.mode;
    });
  };

  const apply = () => {
    updateBounds();
    root.style.setProperty("--workspace-overlay-drawer-width", `${state.width}px`);
    root.dataset.workspaceOverlayMode = state.mode;
    root.dataset.workspaceOverlayOpen = String(state.open);
    root.setAttribute("aria-hidden", String(!state.open));
    syncControls();
    const focusedOpen = state.open && state.mode === "focused";
    if (backdrop) {
      backdrop.hidden = !focusedOpen;
      backdrop.setAttribute("aria-hidden", String(backdrop.hidden));
    }
    if (floatingClose) {
      floatingClose.hidden = !focusedOpen;
      floatingClose.setAttribute("aria-hidden", String(floatingClose.hidden));
      floatingClose.dataset.workspaceOverlayCloseVisible = String(!floatingClose.hidden);
    }
    if (edge) {
      edge.hidden = !(state.available && !state.open);
      edge.dataset.workspaceOverlayRestoreVisible = String(!edge.hidden);
    }
    if (resize) resize.hidden = state.mode !== "drawer";
    root.dispatchEvent(new CustomEvent("workspace-overlay-state-change", {
      bubbles: true,
      detail: { componentId, mode: state.mode, open: state.open, available: state.available, width: state.width },
    }));
  };

  const open = () => {
    if (!state.available) return false;
    state.open = true;
    setComponentShellState(root, "expanded");
    apply();
    return true;
  };
  const collapse = () => {
    state.open = false;
    setComponentShellState(root, "expanded");
    apply();
    return true;
  };
  const setAvailable = (available) => {
    state.available = Boolean(available);
    if (!state.available) state.open = false;
    apply();
    return state.available;
  };
  const setMode = (value) => {
    state.mode = normalizeMode(value, modes, defaultMode);
    persist();
    apply();
    return state.mode;
  };
  const setWidth = (value, { save = true } = {}) => {
    const available = Math.max(minWidth, Number(workspaceRoot?.getBoundingClientRect?.().width || window.innerWidth) - 48);
    state.width = clamp(value, minWidth, Math.min(maxWidth, available), defaultWidth);
    if (save) persist();
    apply();
    return state.width;
  };

  edge?.addEventListener("click", open);
  backdrop?.addEventListener("click", collapse);
  floatingClose?.addEventListener("click", collapse);
  root.querySelectorAll("[data-workspace-overlay-close]").forEach((control) => control.addEventListener("click", collapse));
  overlayControls("data-workspace-overlay-mode-button").forEach((control) => control.addEventListener("click", () => setMode(control.dataset.workspaceOverlayModeButton)));
  overlayControls("data-workspace-overlay-mode-select").forEach((control) => control.addEventListener("change", () => setMode(control.value)));

  if (resize) {
    resize.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || state.mode !== "drawer") return;
      event.preventDefault();
      event.stopPropagation();
      const startX = event.clientX;
      const startWidth = state.width;
      resize.classList.add("is-dragging");
      resize.setPointerCapture?.(event.pointerId);
      const move = (moveEvent) => setWidth(startWidth + (startX - moveEvent.clientX), { save: false });
      const finish = (upEvent) => {
        resize.classList.remove("is-dragging");
        try { resize.releasePointerCapture?.(upEvent.pointerId); } catch (_error) {}
        resize.removeEventListener("pointermove", move);
        resize.removeEventListener("pointerup", finish);
        resize.removeEventListener("pointercancel", finish);
        persist();
      };
      resize.addEventListener("pointermove", move);
      resize.addEventListener("pointerup", finish);
      resize.addEventListener("pointercancel", finish);
    });
    resize.addEventListener("keydown", (event) => {
      if (state.mode !== "drawer") return;
      if (event.key === "ArrowLeft") { event.preventDefault(); setWidth(state.width + 24); }
      else if (event.key === "ArrowRight") { event.preventDefault(); setWidth(state.width - 24); }
      else if (event.key === "Home") { event.preventDefault(); setWidth(defaultWidth); }
    });
  }

  if (config.escapeCollapse !== false) {
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.open) collapse();
    });
  }
  if (config.clickOutsideCollapse !== false) {
    document.addEventListener("pointerdown", (event) => {
      if (!state.open) return;
      const target = event.target;
      if (root.contains(target) || edge?.contains(target) || floatingClose?.contains(target)) return;
      const isModeControl = overlayControls("data-workspace-overlay-mode-button").some((control) => control.contains(target))
        || overlayControls("data-workspace-overlay-mode-select").some((control) => control.contains(target));
      if (isModeControl) return;
      if (state.mode === "focused") {
        // Focused overlays must always close when the user clicks outside the
        // surface, even if the backdrop is obscured by another stacking context.
        collapse();
        return;
      }
      if (state.mode === "drawer") {
        const ignored = (config.ignoreOutsideSelectors || []).some((selector) => target?.closest?.(selector));
        if (!ignored) collapse();
      }
    }, true);
  }

  root.addEventListener("component-shell-state-change", (event) => {
    if (event.detail?.componentId !== componentId || event.detail?.state === "expanded") return;
    // Overlay-capable components never remain in a shell-only side/collapsed state.
    // Tuck/minimize means close the overlay and expose the independent edge tab.
    state.open = false;
    setComponentShellState(root, "expanded");
    apply();
  });
  window.addEventListener("resize", updateBounds);

  const controller = Object.freeze({
    root,
    descriptor,
    open,
    collapse,
    setAvailable,
    setMode,
    setWidth,
    refresh: apply,
    getState: () => ({ ...state }),
    edge,
    backdrop,
    floatingClose,
    resize,
  });
  controllers.set(root, controller);
  apply();
  return controller;
}

export function workspaceComponentOverlay(root) {
  return controllers.get(root) || null;
}
