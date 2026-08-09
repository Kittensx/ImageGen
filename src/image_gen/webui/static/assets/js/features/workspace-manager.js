import { setActionIcon } from "../components/action-icons.js?v=0.1.1";
import { setComponentShellState, setComponentShellVariant } from "../components/component-shell.js?v=workspace-manager1";
import {
  clearWorkspaceLayoutPreference,
  compatibleWorkspaceComponents,
  registeredWorkspacePages,
  saveWorkspaceLayoutPreference,
  setWorkspaceComponentOrder,
  setWorkspaceComponentSpan,
  setWorkspaceComponentVisibility,
  workspaceLayoutSnapshot,
  workspacePage,
} from "../workspace/registry.js?v=workspace-manager1";

function byId(id) {
  return document.getElementById(id);
}

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = String(value);
  node.textContent = String(label);
  return node;
}

function componentNode(pageId, componentId) {
  try {
    return document.querySelector(`[data-workspace-page="${CSS.escape(pageId)}"] [data-workspace-component="${CSS.escape(componentId)}"]`);
  } catch (_error) {
    return null;
  }
}

function persist(pageId) {
  return saveWorkspaceLayoutPreference(pageId, workspaceLayoutSnapshot(document, pageId));
}

function requestWorkspace(pageId) {
  if (!pageId) return;
  window.history.pushState(null, "", pageId === "home" ? "/" : `/#${pageId}`);
  window.dispatchEvent(new CustomEvent("image-gen-workspace-request", {
    detail: { workspace: pageId, source: "workspace-manager" },
  }));
}

function stateOptions(hasSummary, current) {
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Component shell state");
  const states = [
    ["expanded", "Expanded"],
    ...(hasSummary ? [["summary", "Summary row"]] : []),
    ["collapsed", "Title bar"],
    ["side", "Tucked side"],
  ];
  states.forEach(([value, label]) => select.append(option(value, label)));
  select.value = states.some(([value]) => value === current) ? current : "expanded";
  return select;
}

function renderManager(pageId) {
  const list = byId("workspaceManagerComponentList");
  const summary = byId("workspaceManagerSummary");
  const page = workspacePage(pageId);
  if (!list || !summary) return;
  list.replaceChildren();
  if (!page) {
    summary.textContent = "No registered workspace is selected.";
    return;
  }

  const registered = compatibleWorkspaceComponents(pageId);
  const layout = workspaceLayoutSnapshot(document, pageId);
  const layoutById = new Map(layout.components.map((item) => [item.componentId, item]));
  const ordered = [...registered].sort((left, right) => {
    const leftOrder = layoutById.get(left.componentId)?.order ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = layoutById.get(right.componentId)?.order ?? Number.MAX_SAFE_INTEGER;
    return leftOrder - rightOrder || left.title.localeCompare(right.title);
  });
  const required = new Set(page.requiredComponents);
  summary.textContent = `${page.title}: ${registered.length} compatible registered component${registered.length === 1 ? "" : "s"}. Changes are saved locally for this workspace.`;

  if (!ordered.length) {
    const empty = document.createElement("div");
    empty.className = "workspace-manager-empty";
    empty.textContent = "This workspace does not have registered shell components yet.";
    list.append(empty);
    return;
  }

  ordered.forEach((descriptor, index) => {
    const placement = layoutById.get(descriptor.componentId) || {
      componentId: descriptor.componentId,
      order: index,
      variant: descriptor.defaultVariant,
      span: descriptor.defaultGridSpan,
      shellState: "expanded",
      visible: descriptor.defaultVisible,
    };
    const node = componentNode(pageId, descriptor.componentId);
    const visible = placement.visible !== false && !node?.hidden;
    const row = document.createElement("article");
    row.className = `workspace-manager-row${visible ? "" : " is-hidden-component"}`;
    row.dataset.workspaceManagerComponent = descriptor.componentId;

    const identity = document.createElement("div");
    identity.className = "workspace-manager-component-identity";
    const icon = document.createElement("span");
    icon.className = "ui-icon";
    icon.dataset.icon = descriptor.icon || "info";
    icon.setAttribute("aria-hidden", "true");
    const copy = document.createElement("div");
    copy.className = "workspace-manager-component-copy";
    const title = document.createElement("strong");
    title.textContent = descriptor.title;
    const meta = document.createElement("small");
    meta.textContent = `${descriptor.category} · ${descriptor.packageId}`;
    copy.append(title, meta);
    identity.append(icon, copy);

    const move = document.createElement("div");
    move.className = "workspace-manager-move-actions";
    const up = document.createElement("button");
    up.type = "button";
    setActionIcon(up, "chevron-up", { label: `Move ${descriptor.title} up`, title: `Move ${descriptor.title} up`, replace: true });
    up.disabled = index === 0;
    up.addEventListener("click", () => {
      setWorkspaceComponentOrder(document, pageId, descriptor.componentId, index - 1);
      persist(pageId);
      renderManager(pageId);
    });
    const down = document.createElement("button");
    down.type = "button";
    setActionIcon(down, "chevron-down", { label: `Move ${descriptor.title} down`, title: `Move ${descriptor.title} down`, replace: true });
    down.disabled = index === ordered.length - 1;
    down.addEventListener("click", () => {
      setWorkspaceComponentOrder(document, pageId, descriptor.componentId, index + 1);
      persist(pageId);
      renderManager(pageId);
    });
    move.append(up, down);

    const spanField = document.createElement("label");
    spanField.className = "workspace-manager-field workspace-manager-field--span";
    spanField.append(document.createTextNode("Width"));
    const span = document.createElement("select");
    span.setAttribute("aria-label", `${descriptor.title} width`);
    for (let value = descriptor.minGridSpan; value <= descriptor.maxGridSpan; value += 1) {
      span.append(option(value, `${value}/12`));
    }
    span.value = String(placement.span || descriptor.defaultGridSpan);
    span.addEventListener("change", () => {
      setWorkspaceComponentSpan(document, pageId, descriptor.componentId, span.value);
      persist(pageId);
      renderManager(pageId);
    });
    spanField.append(span);

    const variantField = document.createElement("label");
    variantField.className = "workspace-manager-field";
    variantField.append(document.createTextNode("Style"));
    const variant = document.createElement("select");
    variant.setAttribute("aria-label", `${descriptor.title} style`);
    descriptor.supportedVariants.forEach((value) => variant.append(option(value, value.replace(/(^.|-.)/g, (match) => match.replace("-", " ").toUpperCase()))));
    variant.value = placement.variant || descriptor.defaultVariant;
    variant.addEventListener("change", () => {
      if (node) setComponentShellVariant(node, variant.value);
      persist(pageId);
      renderManager(pageId);
    });
    variantField.append(variant);

    const stateField = document.createElement("label");
    stateField.className = "workspace-manager-field";
    stateField.append(document.createTextNode("Presentation"));
    const hasSummary = Boolean(node?.querySelector(":scope > [data-component-shell-summary]"));
    const shellState = stateOptions(hasSummary, placement.shellState || "expanded");
    shellState.addEventListener("change", () => {
      if (node) setComponentShellState(node, shellState.value);
      persist(pageId);
      renderManager(pageId);
    });
    stateField.append(shellState);

    const actions = document.createElement("div");
    actions.className = "workspace-manager-row-actions";
    const visibility = document.createElement("button");
    visibility.type = "button";
    const requiredComponent = required.has(descriptor.componentId);
    const unavailable = !node;
    setActionIcon(visibility, visible ? "remove" : "new", {
      label: unavailable
        ? `${descriptor.title} is registered but not mounted on ${page.title}`
        : (requiredComponent ? `${descriptor.title} is required` : (visible ? `Hide ${descriptor.title}` : `Add ${descriptor.title}`)),
      title: unavailable
        ? `${descriptor.title} cannot be added until its component mount is available on ${page.title}`
        : (requiredComponent ? `${descriptor.title} is required on ${page.title}` : (visible ? `Hide ${descriptor.title}` : `Add ${descriptor.title}`)),
      replace: true,
    });
    visibility.disabled = requiredComponent || unavailable;
    visibility.addEventListener("click", () => {
      setWorkspaceComponentVisibility(document, descriptor.componentId, !visible);
      persist(pageId);
      renderManager(pageId);
    });
    actions.append(visibility);

    row.append(identity, move, spanField, variantField, stateField, actions);
    list.append(row);
  });
}

export function bindWorkspaceManager() {
  const pageSelect = byId("workspaceManagerPageSelect");
  if (!pageSelect) return null;
  const pages = registeredWorkspacePages();
  pageSelect.replaceChildren();
  pages.forEach((page) => pageSelect.append(option(page.pageId, page.title)));
  if (!pages.length) {
    pageSelect.disabled = true;
    renderManager("");
    return null;
  }
  pageSelect.value = pages[0].pageId;
  pageSelect.addEventListener("change", () => renderManager(pageSelect.value));
  byId("workspaceManagerOpenPage")?.addEventListener("click", () => requestWorkspace(pageSelect.value));
  byId("workspaceManagerReset")?.addEventListener("click", () => {
    const pageId = pageSelect.value;
    clearWorkspaceLayoutPreference(pageId);
    window.location.href = pageId === "home" ? "/" : `/#${pageId}`;
  });
  renderManager(pageSelect.value);
  window.addEventListener("workspace-component-visibility-change", () => renderManager(pageSelect.value));
  window.addEventListener("component-shell-state-change", () => renderManager(pageSelect.value));
  window.addEventListener("component-shell-variant-change", () => renderManager(pageSelect.value));
  return { render: () => renderManager(pageSelect.value) };
}
