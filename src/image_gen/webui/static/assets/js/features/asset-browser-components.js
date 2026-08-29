import { initComponentShell, setComponentShellResponsiveVariant } from "../components/component-shell.js?v=content-capabilities2";
import {
  mountWorkspacePage,
  persistWorkspaceLayout,
  registerWorkspaceComponent,
  registerWorkspacePage,
  setWorkspaceComponentSpan,
  workspaceLayoutSnapshot,
  workspaceOverlayCapability,
} from "../workspace/registry.js?v=workspace-responsive4-overlay2";

const ASSET_BROWSER_COMPONENTS = [
  { componentId: "asset-browser.overview", title: "Asset Browser", icon: "models", category: "navigation", defaultVariant: "feature", defaultGridSpan: 12, minGridSpan: 6, resizable: true, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 520 },
  { componentId: "asset-browser.search", title: "Search and browse", icon: "filter", category: "discovery", defaultGridSpan: 12, minGridSpan: 6, resizable: true, minUsefulWidth: 520 },
  { componentId: "asset-browser.filters", title: "Filter current results", icon: "filter", category: "discovery", defaultGridSpan: 12, minGridSpan: 6, resizable: true, minUsefulWidth: 520 },
  { componentId: "asset-browser.downloads", title: "Downloads", icon: "queue-compose", category: "downloads", defaultGridSpan: 8, minGridSpan: 4, resizable: true, minUsefulWidth: 420 },
  { componentId: "asset-browser.saved", title: "Saved for later", icon: "save", category: "discovery", defaultGridSpan: 4, minGridSpan: 3, resizable: true, minUsefulWidth: 300 },
  { componentId: "asset-browser.results", title: "Search results", icon: "models", category: "discovery", defaultGridSpan: 12, minGridSpan: 4, resizable: true, minUsefulWidth: 420 },
  { componentId: "asset-browser.details", title: "Asset details", icon: "info", category: "discovery", defaultGridSpan: 4, minUsefulWidth: 320, overlay: { modes: ["drawer", "focused"], defaultMode: "drawer", defaultDrawerWidth: 520, minDrawerWidth: 360, maxDrawerWidth: 900, resizableDrawer: true, edgeRestore: true, edgeLabel: "Details", clickOutsideCollapse: true, escapeCollapse: true, ignoreOutsideSelectors: [".asset-browser-card"] } },
];

const B3_LAYOUT_MIGRATION_KEY = "imagegen.asset-browser.b3-layout-migrated.v1";
let registered = false;

function ensureRegistered() {
  if (registered) return;
  ASSET_BROWSER_COMPONENTS.forEach((descriptor) => registerWorkspaceComponent({
    packageId: "image_gen.core.asset_browser",
    distribution: "bundled",
    allowedPages: ["asset-browser"],
    requiredCapabilities: [],
    settingsSchema: { type: "object", properties: {}, additionalProperties: false },
    supportedVariants: ["standard", "feature", "horizontal"],
    minGridSpan: 2,
    maxGridSpan: 12,
    ...descriptor,
  }));
  registerWorkspacePage({
    pageId: "asset-browser",
    title: "Asset Browser",
    layoutSchemaVersion: 1,
    defaultWorkspaceId: "imagegen.asset-browser.default",
    allowedComponents: ASSET_BROWSER_COMPONENTS.map((item) => item.componentId),
    requiredComponents: ["asset-browser.search", "asset-browser.results", "asset-browser.details"],
    defaultWorkspace: ASSET_BROWSER_COMPONENTS.map((item, order) => ({
      componentId: item.componentId,
      order,
      span: item.defaultGridSpan,
      variant: item.defaultVariant || "standard",
      shellState: "expanded",
      visible: true,
    })),
  });
  registered = true;
}

function applyB3LayoutMigration(scope) {
  try {
    if (window.localStorage.getItem(B3_LAYOUT_MIGRATION_KEY) === "true") return false;
  } catch (_error) {}
  setWorkspaceComponentSpan(scope, "asset-browser", "asset-browser.results", 12);
  persistWorkspaceLayout(scope, "asset-browser");
  try { window.localStorage.setItem(B3_LAYOUT_MIGRATION_KEY, "true"); } catch (_error) {}
  return true;
}

export function bindAssetBrowserComponents(scope = document) {
  ensureRegistered();
  const result = mountWorkspacePage(scope, "asset-browser", {
    initComponent: (node, descriptor) => initComponentShell(node, descriptor),
    applyResponsiveVariant: (node, variant) => setComponentShellResponsiveVariant(node, variant),
  });
  applyB3LayoutMigration(scope);
  const snapshot = () => workspaceLayoutSnapshot(scope, "asset-browser");
  const persist = () => persistWorkspaceLayout(scope, "asset-browser");
  result.root?.addEventListener("component-shell-state-change", persist);
  result.root?.addEventListener("component-shell-variant-change", persist);
  window.imageGenAssetBrowserWorkspace = Object.freeze({ snapshot, persist });
  const detailsOverlay = workspaceOverlayCapability(scope, "asset-browser.details");
  return { ...result, snapshot, persist, detailsOverlay };
}
