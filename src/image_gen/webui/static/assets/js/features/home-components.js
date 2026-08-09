import { initComponentShell } from "../components/component-shell.js?v=featured-shell1";
import {
  mountWorkspacePage,
  persistWorkspaceLayout,
  registerWorkspaceComponent,
  registerWorkspacePage,
  setWorkspaceComponentVisibility,
  workspaceLayoutSnapshot,
} from "../workspace/registry.js?v=component-shell1";

const HOME_COMPONENTS = [
  { componentId: "home.welcome", title: "Welcome", icon: "home", category: "navigation", defaultVariant: "featured", defaultGridSpan: 12 },
  { componentId: "home.readiness", title: "Readiness", icon: "info", category: "status", defaultGridSpan: 4 },
  { componentId: "home.quick-launch", title: "Quick launch", icon: "generate", category: "navigation", defaultGridSpan: 5, defaultVisible: false },
  { componentId: "home.profile", title: "Profile", icon: "home-bug-contribution", category: "profile", defaultVariant: "feature", defaultGridSpan: 4 },
  { componentId: "home.discord", title: "Discord", icon: "discord", category: "community", defaultVariant: "feature", defaultGridSpan: 4 },
  { componentId: "home.developer-updates", title: "Changelog", icon: "external-link", category: "updates", defaultVariant: "horizontal", defaultGridSpan: 12 },
];
let registered = false;

function ensureRegistered() {
  if (registered) return;
  HOME_COMPONENTS.forEach((descriptor) => registerWorkspaceComponent({
    packageId: "image_gen.core.home",
    allowedPages: ["home"],
    supportedVariants: ["standard", "feature", "featured", "horizontal"],
    minGridSpan: 2,
    maxGridSpan: 12,
    ...descriptor,
  }));
  registerWorkspacePage({
    pageId: "home",
    title: "Home",
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
