const SIDEBAR_STORAGE_KEY = "image-gen.site-sidebar.state";
const MAIN_PATHS = new Set(["/", "/index.html"]);
const MAIN_WORKSPACES = new Set(["home", "generation", "checkpoints", "loras"]);
const MAIN_ROUTES = new Set([...MAIN_WORKSPACES, "settings"]);

function normalizePath(pathname = window.location.pathname) {
  const value = String(pathname || "/");
  return value.endsWith("/") && value !== "/" ? value.slice(0, -1) : value;
}

function storedSidebarState() {
  try {
    const value = window.localStorage.getItem(SIDEBAR_STORAGE_KEY);
    return value === "collapsed" || value === "expanded" ? value : "expanded";
  } catch (_error) {
    return "expanded";
  }
}

function applySidebarState(state) {
  const value = state === "collapsed" ? "collapsed" : "expanded";
  document.documentElement.dataset.siteSidebar = value;
  const toggle = document.getElementById("siteSidebarToggle");
  if (toggle) {
    const collapsed = value === "collapsed";
    toggle.setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
    toggle.title = collapsed ? "Expand navigation" : "Collapse navigation";
  }
  try {
    window.localStorage.setItem(SIDEBAR_STORAGE_KEY, value);
  } catch (_error) {
    // Sidebar state is a convenience only.
  }
}

function activeWorkspace() {
  const value = String(document.body.dataset.activeWorkspace || "home").trim().toLowerCase();
  return MAIN_WORKSPACES.has(value) ? value : "home";
}

function routeFromLocation() {
  const path = normalizePath();
  if (path === "/region-builder.html") return "region-builder";
  if (!MAIN_PATHS.has(path)) return "";

  const hash = window.location.hash.replace(/^#/, "").trim().toLowerCase();
  if (MAIN_ROUTES.has(hash)) return hash;
  return activeWorkspace();
}

function renderActiveRoute(route = routeFromLocation()) {
  document.querySelectorAll("[data-site-route]").forEach((item) => {
    const active = item.dataset.siteRoute === route;
    item.classList.toggle("is-active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
}

function waitForMainReady(timeoutMs = 15000) {
  if (window.__IMAGE_GEN_BOOT_READY__) return Promise.resolve(true);
  return new Promise((resolve) => {
    const started = performance.now();
    const check = () => {
      if (window.__IMAGE_GEN_BOOT_READY__) {
        resolve(true);
        return;
      }
      if (performance.now() - started >= timeoutMs) {
        resolve(false);
        return;
      }
      window.setTimeout(check, 80);
    };
    check();
  });
}

function workspaceUrl(route) {
  return route === "home" ? "/" : `/#${route}`;
}

function writeRoute(route, { replace = false } = {}) {
  if (!MAIN_ROUTES.has(route)) return;
  const url = workspaceUrl(route);
  const current = `${window.location.pathname}${window.location.hash}`;
  if (current === url || (route === "home" && current === "/index.html")) return;
  const method = replace ? "replaceState" : "pushState";
  window.history[method](null, "", url);
}

async function activateMainRoute(route, { updateHistory = false, replaceHistory = false } = {}) {
  if (!MAIN_PATHS.has(normalizePath()) || !MAIN_ROUTES.has(route)) return false;
  const ready = await waitForMainReady();
  if (!ready) return false;

  if (route === "settings") {
    window.dispatchEvent(new CustomEvent("image-gen-open-settings", { detail: { source: "sidebar" } }));
  } else {
    const settingsDialog = document.getElementById("settingsDialog");
    if (settingsDialog?.open) settingsDialog.close();
    window.dispatchEvent(new CustomEvent("image-gen-workspace-request", {
      detail: { workspace: route, source: "sidebar" },
    }));
  }

  if (updateHistory) writeRoute(route, { replace: replaceHistory });
  renderActiveRoute(route);
  return true;
}

function bindNavigation() {
  document.getElementById("siteSidebarToggle")?.addEventListener("click", () => {
    const current = document.documentElement.dataset.siteSidebar || "expanded";
    applySidebarState(current === "collapsed" ? "expanded" : "collapsed");
  });

  document.querySelectorAll("a[data-site-route]").forEach((link) => {
    link.addEventListener("click", (event) => {
      const route = link.dataset.siteRoute || "";
      if (!MAIN_PATHS.has(normalizePath()) || !MAIN_ROUTES.has(route)) return;
      event.preventDefault();
      activateMainRoute(route, { updateHistory: true });
    });
  });

  window.addEventListener("hashchange", () => {
    if (!MAIN_PATHS.has(normalizePath())) return;
    const route = routeFromLocation();
    renderActiveRoute(route);
    if (MAIN_ROUTES.has(route)) activateMainRoute(route);
  });

  window.addEventListener("popstate", () => {
    if (!MAIN_PATHS.has(normalizePath())) return;
    const route = routeFromLocation();
    renderActiveRoute(route);
    if (MAIN_ROUTES.has(route)) activateMainRoute(route);
  });

  window.addEventListener("image-gen-workspace-changed", (event) => {
    const workspace = String(event.detail?.workspace || "").trim().toLowerCase();
    if (!MAIN_WORKSPACES.has(workspace)) return;
    if (routeFromLocation() !== "settings") writeRoute(workspace, { replace: true });
    renderActiveRoute(routeFromLocation() === "settings" ? "settings" : workspace);
  });

  window.addEventListener("image-gen-settings-opened", () => {
    renderActiveRoute("settings");
  });

  window.addEventListener("image-gen-settings-closed", () => {
    const workspace = activeWorkspace();
    writeRoute(workspace, { replace: true });
    renderActiveRoute(workspace);
  });
}

async function loadSidebarFragment() {
  const placeholder = document.querySelector("[data-site-sidebar-fragment]");
  if (!placeholder) return null;
  const source = placeholder.dataset.siteSidebarFragment;
  const response = await fetch(source, {
    cache: "no-store",
    headers: { "Cache-Control": "no-cache" },
  });
  if (!response.ok) throw new Error(`Unable to load shared sidebar: ${source}`);
  const template = document.createElement("template");
  template.innerHTML = await response.text();
  const sidebar = template.content.firstElementChild;
  placeholder.replaceWith(template.content);
  return sidebar || document.getElementById("siteSidebar");
}

async function startSiteSidebar() {
  applySidebarState(storedSidebarState());
  const sidebar = await loadSidebarFragment();
  if (!sidebar && !document.getElementById("siteSidebar")) return;

  bindNavigation();
  document.documentElement.dataset.siteSidebarReady = "true";
  renderActiveRoute();

  if (MAIN_PATHS.has(normalizePath())) {
    const route = routeFromLocation();
    if (MAIN_ROUTES.has(route)) activateMainRoute(route);
  }
}

startSiteSidebar().catch((error) => {
  console.error("Shared sidebar startup failed:", error);
  document.documentElement.dataset.siteSidebarReady = "false";
});
