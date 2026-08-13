import { initComponentShell, setComponentShellResponsiveVariant } from "../components/component-shell.js?v=content-capabilities2";
import {
  mountWorkspacePage,
  persistWorkspaceLayout,
  registerWorkspaceComponent,
  registerWorkspacePage,
  setWorkspaceComponentVisibility,
  workspaceLayoutSnapshot,
} from "../workspace/registry.js?v=workspace-responsive2";

const HOME_COMPONENTS = [
  { componentId: "home.welcome", title: "Welcome", icon: "home", category: "navigation", defaultVariant: "feature", defaultGridSpan: 12, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 520 },
  { componentId: "home.readiness", title: "Readiness", icon: "info", category: "status", defaultGridSpan: 4, minUsefulWidth: 300 },
  { componentId: "home.quick-launch", title: "Quick launch", icon: "generate", category: "navigation", defaultGridSpan: 5, defaultVisible: false, minUsefulWidth: 320 },
  { componentId: "home.profile", title: "Profile", icon: "home-bug-contribution", category: "profile", defaultVariant: "feature", defaultGridSpan: 4, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 320 },
  { componentId: "home.discord", title: "Discord", icon: "discord", category: "community", defaultVariant: "feature", defaultGridSpan: 4, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 320 },
  { componentId: "home.developer-updates", title: "Changelog", icon: "external-link", category: "updates", defaultVariant: "horizontal", defaultGridSpan: 12, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 420, requiredCapabilities: ["content.markdown"] },
  { componentId: "home.help-center", title: "Help Center", icon: "info", category: "help", defaultVariant: "horizontal", defaultGridSpan: 12, responsive: { compact: "standard", narrow: "standard" }, minUsefulWidth: 420, requiredCapabilities: ["content.markdown", "content.media"] },
];
let registered = false;

function ensureRegistered() {
  if (registered) return;
  HOME_COMPONENTS.forEach((descriptor) => registerWorkspaceComponent({
    packageId: "image_gen.core.home",
    distribution: "bundled",
    allowedPages: ["home"],
    requiredCapabilities: [],
    settingsSchema: { type: "object", properties: {}, additionalProperties: false },
    supportedVariants: ["standard", "feature", "horizontal"],
    minGridSpan: 2,
    maxGridSpan: 12,
    ...descriptor,
  }));
  registerWorkspacePage({
    pageId: "home",
    title: "Home",
    layoutSchemaVersion: 1,
    defaultWorkspaceId: "imagegen.home.default",
    allowedComponents: HOME_COMPONENTS.map((item) => item.componentId),
    requiredComponents: ["home.welcome", "home.readiness"],
    defaultWorkspace: HOME_COMPONENTS.map((item, order) => ({
      componentId: item.componentId,
      order,
      span: item.defaultGridSpan,
      variant: item.defaultVariant || "standard",
      shellState: "expanded",
      visible: item.defaultVisible !== false,
    })),
  });
  registered = true;
}

export function bindHomeComponents(scope = document) {
  ensureRegistered();
  const result = mountWorkspacePage(scope, "home", {
    initComponent: (node, descriptor) => initComponentShell(node, descriptor),
    applyResponsiveVariant: (node, variant) => setComponentShellResponsiveVariant(node, variant),
  });
  const snapshot = () => workspaceLayoutSnapshot(scope, "home");
  const persist = () => persistWorkspaceLayout(scope, "home");
  const setVisibility = (componentId, visible) => {
    const changed = setWorkspaceComponentVisibility(scope, componentId, visible);
    if (changed) persist();
    return changed;
  };
  result.root?.addEventListener("component-shell-state-change", persist);
  result.root?.addEventListener("component-shell-variant-change", persist);
  window.imageGenHomeWorkspace = Object.freeze({ snapshot, persist, setVisibility });
  return { ...result, snapshot, persist, setVisibility };
}
