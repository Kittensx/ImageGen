import {
  WORKSPACE_WIDTH_CLASSES,
  bindWorkspaceWidthObserver,
  responsiveGridSpan,
  responsivePresentationSpan,
  resolveResponsiveVariant,
} from "./responsive.js?v=workspace-responsive2";

const components = new Map();
const pages = new Map();
const LAYOUT_STORAGE_PREFIX = "image-gen.workspace-layout.v1.";
const responsiveOverrides = new WeakMap();
const responsiveControllers = new WeakMap();

function normalizedId(value, label = "component ID") {
  const id = String(value || "").trim();
  if (!id) throw new TypeError(`${label} is required.`);
  return id;
}

function normalizedSpan(value, fallback = 12) {
  const span = Number(value);
  if (!Number.isFinite(span)) return fallback;
  return Math.max(1, Math.min(12, Math.round(span)));
}

function canonicalVariant(value, fallback = "standard") {
  const token = String(value || fallback).trim().toLowerCase();
  return token === "featured" ? "feature" : (token || fallback);
}

function canonicalResponsiveVariants(value) {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(Object.entries(value).map(([widthClass, variant]) => [
    String(widthClass).trim().toLowerCase(),
    canonicalVariant(variant),
  ]));
}

function clearWorkspaceLayoutPrepaint(pageId) {
  const normalizedPageId = String(pageId || "").toLowerCase();
  const style = document.getElementById(`workspaceLayoutPrepaint-${normalizedPageId}`);
  style?.remove();
  document.querySelectorAll?.(`[data-workspace-page="${CSS.escape(normalizedPageId)}"] > [data-workspace-component]`).forEach((node) => {
    node.style.removeProperty("order");
  });
  if (normalizedPageId === "home") {
    window.__IMAGE_GEN_HOME_LAYOUT_PREPAINT_OBSERVER__?.disconnect?.();
    delete window.__IMAGE_GEN_HOME_LAYOUT_PREPAINT_OBSERVER__;
  }
  if (document.documentElement.dataset.workspaceLayoutPrepaint === normalizedPageId) {
    delete document.documentElement.dataset.workspaceLayoutPrepaint;
  }
}

export function registerWorkspaceComponent(descriptor) {
  const source = descriptor && typeof descriptor === "object" ? descriptor : {};
  const componentId = normalizedId(source.componentId);
  const key = componentId.toLocaleLowerCase();
  if (components.has(key)) throw new Error(`Workspace component '${componentId}' is already registered.`);
  const distribution = String(source.distribution || "bundled").trim().toLocaleLowerCase();
  if (!new Set(["bundled", "external"]).has(distribution)) {
    throw new Error(`Workspace component '${componentId}' has unsupported distribution '${distribution}'.`);
  }
  const install = source.install && typeof source.install === "object" ? Object.freeze({ ...source.install }) : null;
  if (distribution === "external" && (!install?.registry || !install?.packageRef)) {
    throw new Error(`External workspace component '${componentId}' requires canonical install metadata.`);
  }
  const registered = Object.freeze({
    componentId,
    packageId: String(source.packageId || "image_gen.core"),
    title: String(source.title || componentId),
    icon: String(source.icon || "info"),
    description: String(source.description || ""),
    category: String(source.category || "general"),
    distribution,
    install,
    links: Object.freeze({ ...(source.links || {}) }),
    allowedPages: Object.freeze([...(source.allowedPages || [])].map(String)),
    requiredCapabilities: Object.freeze([...(source.requiredCapabilities || [])].map(String)),
    defaultVariant: canonicalVariant(source.defaultVariant || "standard"),
    supportedVariants: Object.freeze([...new Set([...(source.supportedVariants || [source.defaultVariant || "standard"])].map((value) => canonicalVariant(value)))]),
    defaultGridSpan: normalizedSpan(source.defaultGridSpan),
    defaultVisible: source.defaultVisible !== false,
    minGridSpan: normalizedSpan(source.minGridSpan || 1),
    maxGridSpan: normalizedSpan(source.maxGridSpan || 12),
    shell: Object.freeze({ ...(source.shell || {}) }),
    settingsSchema: source.settingsSchema || { type: "object", properties: {}, additionalProperties: false },
    responsive: Object.freeze(canonicalResponsiveVariants(source.responsive)),
    minUsefulWidth: Number.isFinite(Number(source.minUsefulWidth)) ? Math.max(1, Math.round(Number(source.minUsefulWidth))) : null,
    minUsefulHeight: Number.isFinite(Number(source.minUsefulHeight)) ? Math.max(1, Math.round(Number(source.minUsefulHeight))) : null,
  });
  for (const [widthClass, variant] of Object.entries(registered.responsive)) {
    if (!WORKSPACE_WIDTH_CLASSES.includes(String(widthClass).toLowerCase())) {
      throw new Error(`Workspace component '${componentId}' has unknown responsive width class '${widthClass}'.`);
    }
    if (!registered.supportedVariants.includes(String(variant))) {
      throw new Error(`Workspace component '${componentId}' responsive variant '${variant}' is not supported.`);
    }
  }
  components.set(key, registered);
  return registered;
}

export function registerWorkspacePage(descriptor) {
  const source = descriptor && typeof descriptor === "object" ? descriptor : {};
  const pageId = normalizedId(source.pageId, "page ID");
  const key = pageId.toLocaleLowerCase();
  if (pages.has(key)) throw new Error(`Workspace page '${pageId}' is already registered.`);
  const allowedComponents = Object.freeze([...(source.allowedComponents || [])].map(String));
  const requiredComponents = Object.freeze([...(source.requiredComponents || [])].map(String));
  for (const id of [...allowedComponents, ...requiredComponents]) {
    if (!components.has(id.toLocaleLowerCase())) {
      throw new Error(`Workspace page '${pageId}' references unregistered component '${id}'.`);
    }
  }
  const registered = Object.freeze({
    pageId,
    title: String(source.title || pageId),
    layoutSchemaVersion: Number(source.layoutSchemaVersion || 1),
    allowedComponents,
    requiredComponents,
    defaultWorkspaceId: String(source.defaultWorkspaceId || `imagegen.${pageId}.default`),
    maximumInstances: Object.freeze({ ...(source.maximumInstances || {}) }),
    requiredCapabilities: Object.freeze([...(source.requiredCapabilities || [])].map(String)),
    defaultWorkspace: Object.freeze([...(source.defaultWorkspace || [])].map((entry) => Object.freeze({ ...entry }))),
  });
  pages.set(key, registered);
  return registered;
}

export function workspaceComponent(componentId) {
  return components.get(String(componentId || "").toLocaleLowerCase()) || null;
}

export function workspacePage(pageId) {
  return pages.get(String(pageId || "").toLocaleLowerCase()) || null;
}

function componentFailure(node, message) {
  node.dataset.workspaceComponentState = "error";
  node.classList.add("component-shell--error");
  const existing = node.querySelector(":scope > .component-shell__registry-error");
  if (existing) {
    existing.textContent = message;
    return;
  }
  const error = document.createElement("p");
  error.className = "component-shell__registry-error";
  error.setAttribute("role", "status");
  error.textContent = message;
  node.append(error);
}

function layoutStorageKey(pageId) {
  return `${LAYOUT_STORAGE_PREFIX}${normalizedId(pageId, "page ID").toLocaleLowerCase()}`;
}

export function readWorkspaceLayoutPreference(pageId) {
  try {
    const raw = window.localStorage.getItem(layoutStorageKey(pageId));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed?.components) ? parsed : null;
  } catch (_error) {
    return null;
  }
}

export function saveWorkspaceLayoutPreference(pageId, layout) {
  const payload = {
    pageId: normalizedId(pageId, "page ID"),
    components: Array.isArray(layout?.components) ? layout.components.map((entry) => ({ ...entry })) : [],
  };
  try {
    window.localStorage.setItem(layoutStorageKey(pageId), JSON.stringify(payload));
  } catch (_error) {
    return false;
  }
  return true;
}

export function clearWorkspaceLayoutPreference(pageId) {
  try {
    window.localStorage.removeItem(layoutStorageKey(pageId));
  } catch (_error) {
    return false;
  }
  return true;
}


function prepareWorkspaceDom(root, page, preference = null) {
  const nodes = [...root.children].filter((node) => node.matches?.("[data-workspace-component]"));
  const byId = new Map(nodes.map((node) => [String(node.dataset.workspaceComponent || "").toLocaleLowerCase(), node]));
  const source = preference?.components?.length ? preference.components : page.defaultWorkspace;
  const required = new Set(page.requiredComponents.map((id) => id.toLocaleLowerCase()));
  const ordered = [];
  source.forEach((entry) => {
    const id = String(entry.componentId || "").toLocaleLowerCase();
    const node = byId.get(id);
    if (!node) return;
    const descriptor = workspaceComponent(id);
    if (descriptor) {
      const span = Math.max(descriptor.minGridSpan, Math.min(descriptor.maxGridSpan, normalizedSpan(entry.span, descriptor.defaultGridSpan)));
      node.dataset.componentSpan = String(span);
      node.style.setProperty("--workspace-component-span", String(span));
      if (entry.variant) node.dataset.componentVariant = canonicalVariant(entry.variant);
      if (entry.shellState) node.dataset.componentShellState = String(entry.shellState);
      responsiveOverrides.set(node, canonicalResponsiveVariants(entry.responsive));
      const visible = required.has(id) ? true : entry.visible !== false;
      node.dataset.workspaceComponentVisible = String(visible);
      node.hidden = !visible;
    }
    ordered.push(node);
    byId.delete(id);
  });
  ordered.push(...byId.values());
  ordered.forEach((node) => root.append(node));
}

function applyWorkspaceResponsiveState(root, pageId, widthClass, applyResponsiveVariant = null) {
  const direct = [...root.children].filter((node) => node.matches?.("[data-workspace-component]"));
  direct.forEach((node) => {
    const componentId = String(node.dataset.workspaceComponent || "").trim();
    const descriptor = workspaceComponent(componentId);
    if (!descriptor) return;
    const baseSpan = normalizedSpan(node.dataset.componentSpan, descriptor.defaultGridSpan);
    const responsiveSpan = responsiveGridSpan(baseSpan, widthClass, descriptor);
    const shellState = String(node.dataset.componentShellState || "expanded");
    const effectiveSpan = responsivePresentationSpan(responsiveSpan, widthClass, shellState, descriptor);
    const baseVariant = canonicalVariant(node.dataset.componentVariant || descriptor.defaultVariant || "standard");
    const resolved = resolveResponsiveVariant(descriptor, widthClass, baseVariant, responsiveOverrides.get(node) || {});
    node.dataset.workspaceEffectiveSpan = String(effectiveSpan);
    node.dataset.workspaceResponsiveVariant = resolved.variant;
    node.dataset.workspaceResponsiveResolution = resolved.resolution;
    node.style.setProperty("--workspace-responsive-span", String(effectiveSpan));
    applyResponsiveVariant?.(node, resolved.variant, {
      pageId,
      widthClass,
      baseVariant,
      effectiveSpan,
      responsiveSpan,
      shellState,
      resolution: resolved.resolution,
    });
  });
}

export function refreshWorkspaceResponsiveState(scope, pageId) {
  const root = scope?.matches?.(`[data-workspace-page="${CSS.escape(pageId)}"]`)
    ? scope
    : scope?.querySelector?.(`[data-workspace-page="${CSS.escape(pageId)}"]`);
  const controller = root ? responsiveControllers.get(root) : null;
  return controller?.refresh?.() || null;
}

export function mountWorkspacePage(scope, pageId, { initComponent = null, applyResponsiveVariant = null } = {}) {
  const root = scope?.matches?.("[data-workspace-page]") ? scope : scope?.querySelector?.(`[data-workspace-page="${CSS.escape(pageId)}"]`);
  const page = workspacePage(pageId);
  if (!root || !page) return { page, root, mounted: [], errors: [] };
  const allowed = new Set(page.allowedComponents.map((id) => id.toLocaleLowerCase()));
  const mounted = [];
  const errors = [];
  prepareWorkspaceDom(root, page, readWorkspaceLayoutPreference(pageId));
  clearWorkspaceLayoutPrepaint(pageId);
  const direct = [...root.children].filter((node) => node.matches?.("[data-workspace-component]"));

  direct.forEach((node, order) => {
    const id = String(node.dataset.workspaceComponent || "").trim();
    const descriptor = workspaceComponent(id);
    if (!descriptor || (allowed.size && !allowed.has(id.toLocaleLowerCase()))) {
      const message = descriptor
        ? `Component '${id}' is not allowed on '${pageId}'.`
        : `Component '${id || "unknown"}' is not registered.`;
      errors.push({ componentId: id, message });
      componentFailure(node, message);
      return;
    }
    node.dataset.workspaceComponentOrder = String(order);
    node.dataset.workspaceComponentState = "ready";
    const preparedVisibility = String(node.dataset.workspaceComponentVisible || "").trim().toLowerCase();
    const explicitVisibility = String(node.dataset.workspaceDefaultVisible || "").trim().toLowerCase();
    const visible = preparedVisibility
      ? preparedVisibility !== "false"
      : (explicitVisibility ? !["false", "0", "no", "off"].includes(explicitVisibility) : descriptor.defaultVisible);
    node.dataset.workspaceComponentVisible = String(visible);
    node.hidden = !visible;
    try {
      initComponent?.(node, descriptor, page);
      mounted.push({ node, descriptor });
    } catch (error) {
      const message = `Unable to initialize ${descriptor.title}: ${error?.message || error}`;
      errors.push({ componentId: id, message });
      componentFailure(node, message);
    }
  });

  for (const required of page.requiredComponents) {
    if (!mounted.some(({ descriptor }) => descriptor.componentId === required)) {
      errors.push({ componentId: required, message: `Required component '${required}' is not mounted.` });
    }
  }
  root.dataset.workspaceRegistryMounted = "true";
  const widthObserver = bindWorkspaceWidthObserver(root, {
    onChange: ({ widthClass }) => applyWorkspaceResponsiveState(root, pageId, widthClass, applyResponsiveVariant),
  });
  const controller = {
    refresh: () => {
      const state = widthObserver?.refresh?.() || { widthClass: root.dataset.workspaceWidthClass || "wide" };
      applyWorkspaceResponsiveState(root, pageId, state.widthClass, applyResponsiveVariant);
      return state;
    },
    disconnect: () => widthObserver?.disconnect?.(),
  };
  responsiveControllers.set(root, controller);
  controller.refresh();
  root.addEventListener("component-shell-variant-change", (event) => {
    if (event.detail?.persist === false) return;
    controller.refresh();
  });
  root.addEventListener("component-shell-state-change", () => {
    controller.refresh();
  });
  return { page, root, mounted, errors, responsive: controller };
}

export function workspaceLayoutSnapshot(scope, pageId) {
  const root = scope?.matches?.("[data-workspace-page]") ? scope : scope?.querySelector?.(`[data-workspace-page="${CSS.escape(pageId)}"]`);
  if (!root) return { pageId, components: [] };
  const ordered = [...root.children].filter((node) => node.matches?.("[data-workspace-component]"));
  return {
    pageId,
    components: ordered.map((node, order) => ({
      componentId: String(node.dataset.workspaceComponent || ""),
      order,
      variant: canonicalVariant(node.dataset.componentVariant || "standard"),
      span: normalizedSpan(node.dataset.componentSpan),
      responsive: { ...(responsiveOverrides.get(node) || {}) },
      shellState: String(node.dataset.componentShellState || "expanded"),
      visible: node.dataset.workspaceComponentVisible !== "false" && !node.hidden,
    })),
  };
}

export function persistWorkspaceLayout(scope, pageId) {
  return saveWorkspaceLayoutPreference(pageId, workspaceLayoutSnapshot(scope, pageId));
}

export function setWorkspaceComponentVisibility(scope, componentId, visible) {
  const id = normalizedId(componentId);
  const root = scope?.querySelector?.(`[data-workspace-component="${CSS.escape(id)}"]`)
    || (scope?.matches?.(`[data-workspace-component="${CSS.escape(id)}"]`) ? scope : null);
  if (!root) return false;
  const next = Boolean(visible);
  root.hidden = !next;
  root.dataset.workspaceComponentVisible = String(next);
  root.dispatchEvent(new CustomEvent("workspace-component-visibility-change", {
    bubbles: true,
    detail: { componentId: id, visible: next },
  }));
  return next;
}

export function setWorkspaceComponentOrder(scope, pageId, componentId, targetIndex) {
  const page = workspacePage(pageId);
  const root = scope?.matches?.(`[data-workspace-page="${CSS.escape(pageId)}"]`)
    ? scope
    : scope?.querySelector?.(`[data-workspace-page="${CSS.escape(pageId)}"]`);
  if (!page || !root) return false;
  const nodes = [...root.children].filter((node) => node.matches?.("[data-workspace-component]"));
  const node = nodes.find((item) => String(item.dataset.workspaceComponent || "") === componentId);
  if (!node) return false;
  const others = nodes.filter((item) => item !== node);
  const index = Math.max(0, Math.min(others.length, Math.round(Number(targetIndex) || 0)));
  if (index >= others.length) root.append(node);
  else root.insertBefore(node, others[index]);
  return true;
}

export function setWorkspaceComponentSpan(scope, pageId, componentId, span) {
  const descriptor = workspaceComponent(componentId);
  const root = scope?.querySelector?.(`[data-workspace-page="${CSS.escape(pageId)}"] [data-workspace-component="${CSS.escape(componentId)}"]`);
  if (!descriptor || !root) return null;
  const next = Math.max(descriptor.minGridSpan, Math.min(descriptor.maxGridSpan, normalizedSpan(span, descriptor.defaultGridSpan)));
  root.dataset.componentSpan = String(next);
  root.style.setProperty("--workspace-component-span", String(next));
  const workspaceRoot = root.closest?.("[data-workspace-page]");
  responsiveControllers.get(workspaceRoot)?.refresh?.();
  return next;
}

export function compatibleWorkspaceComponents(pageId) {
  const page = workspacePage(pageId);
  if (!page) return [];
  const allowed = new Set(page.allowedComponents.map((id) => id.toLocaleLowerCase()));
  return [...components.values()].filter((component) => {
    const allowedByPage = !allowed.size || allowed.has(component.componentId.toLocaleLowerCase());
    const allowedByComponent = !component.allowedPages.length || component.allowedPages.includes(page.pageId);
    return allowedByPage && allowedByComponent;
  });
}

export function registeredWorkspaceComponents() {
  return [...components.values()];
}

export function registeredWorkspacePages() {
  return [...pages.values()];
}
