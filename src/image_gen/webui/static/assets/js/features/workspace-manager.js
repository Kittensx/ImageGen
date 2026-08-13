import { setActionIcon } from "../components/action-icons.js?v=0.1.1";
import { setComponentShellState, setComponentShellVariant } from "../components/component-shell.js?v=content-capabilities2";
import {
  clearWorkspaceLayoutPreference,
  compatibleWorkspaceComponents,
  registeredWorkspacePages,
  saveWorkspaceLayoutPreference,
  setWorkspaceComponentOrder,
  setWorkspaceComponentSpan,
  setWorkspaceComponentVisibility,
  workspaceComponent,
  workspaceLayoutSnapshot,
  workspacePage,
} from "../workspace/registry.js?v=workspace-responsive2";
import {
  WORKSPACE_REPRESENTATIVE_WIDTHS,
  WORKSPACE_WIDTH_CLASSES,
  bindWorkspaceWidthObserver,
  responsiveWorkspacePlacements,
} from "../workspace/responsive.js?v=workspace-responsive2";

let previewWidthClass = "wide";

function byId(id) {
  return document.getElementById(id);
}

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = String(value);
  node.textContent = String(label);
  return node;
}

function titleToken(value) {
  const token = String(value || "").trim();
  if (!token) return "";
  return token.replace(/[-_]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function previewPlan(pageId, layout = null) {
  const widthClass = WORKSPACE_WIDTH_CLASSES.includes(previewWidthClass) ? previewWidthClass : "wide";
  const representativeWidth = WORKSPACE_REPRESENTATIVE_WIDTHS[widthClass];
  const sourceLayout = layout || workspaceLayoutSnapshot(document, pageId);
  const placements = responsiveWorkspacePlacements(sourceLayout, workspaceComponent, widthClass);
  return {
    widthClass,
    widthLabel: titleToken(widthClass),
    representativeWidth,
    placements,
    byId: new Map(placements.map((placement) => [placement.componentId, placement])),
  };
}

function previewHint(kind, plan, placement) {
  const hint = document.createElement("small");
  hint.className = "workspace-manager-preview-value";
  if (!placement) {
    hint.textContent = `${plan.widthLabel} preview: unavailable`;
    return hint;
  }
  if (kind === "span") hint.textContent = `${plan.widthLabel} preview: ${placement.effectiveSpan}/12`;
  else if (kind === "variant") hint.textContent = `${plan.widthLabel} preview: ${titleToken(placement.effectiveVariant)}`;
  else hint.textContent = `${plan.widthLabel} preview: ${titleToken(placement.shellState || "expanded")}`;
  return hint;
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

function renderResponsivePreview(pageId, plan = null) {
  const canvas = byId("workspaceManagerPreviewCanvas");
  const label = byId("workspaceManagerPreviewLabel");
  if (!canvas || !label) return;
  const page = workspacePage(pageId);
  canvas.replaceChildren();
  if (!page) {
    label.textContent = "No workspace selected";
    canvas.style.removeProperty("width");
    return;
  }
  const activePlan = plan || previewPlan(pageId);
  label.textContent = `${activePlan.widthLabel} - ${activePlan.representativeWidth}px workspace`;
  canvas.dataset.previewWidthClass = activePlan.widthClass;
  canvas.style.width = `${activePlan.representativeWidth}px`;
  activePlan.placements.forEach((placement) => {
    const descriptor = workspaceComponent(placement.componentId);
    const card = document.createElement("div");
    card.className = `workspace-manager-preview-card${placement.visible === false ? " is-hidden-component" : ""}`;
    card.style.gridColumn = `span ${placement.effectiveSpan}`;
    card.dataset.previewComponent = placement.componentId;
    card.dataset.previewVariant = placement.effectiveVariant;
    card.dataset.previewShellState = placement.shellState || "expanded";
    const title = document.createElement("strong");
    title.textContent = descriptor?.title || placement.componentId;
    const meta = document.createElement("small");
    meta.textContent = `${placement.effectiveSpan}/12 · ${titleToken(placement.effectiveVariant)} · ${titleToken(placement.shellState || "expanded")}`;
    card.append(title, meta);
    canvas.append(card);
  });
}

function renderManager(pageId) {
  const list = byId("workspaceManagerComponentList");
  const summary = byId("workspaceManagerSummary");
  const page = workspacePage(pageId);
  if (!list || !summary) return;
  list.replaceChildren();
  if (!page) {
    summary.textContent = "No registered workspace is selected.";
    renderResponsivePreview("");
    return;
  }

  const registered = compatibleWorkspaceComponents(pageId);
  const layout = workspaceLayoutSnapshot(document, pageId);
  const activePreview = previewPlan(pageId, layout);
  const previewById = activePreview.byId;
  const layoutById = new Map(layout.components.map((item) => [item.componentId, item]));
  const ordered = [...registered].sort((left, right) => {
    const leftOrder = layoutById.get(left.componentId)?.order ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = layoutById.get(right.componentId)?.order ?? Number.MAX_SAFE_INTEGER;
    return leftOrder - rightOrder || left.title.localeCompare(right.title);
  });
  const required = new Set(page.requiredComponents);
  summary.textContent = `${page.title}: ${registered.length} compatible registered component${registered.length === 1 ? "" : "s"}. Base values are saved; the ${activePreview.widthLabel} preview values show the responsive result without overwriting the base layout.`;

  if (!ordered.length) {
    const empty = document.createElement("div");
    empty.className = "workspace-manager-empty";
    empty.textContent = "This workspace does not have registered shell components yet.";
    list.append(empty);
    renderResponsivePreview(pageId);
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
    const previewPlacement = previewById.get(descriptor.componentId);
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
    spanField.append(document.createTextNode("Base width"));
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
    spanField.append(span, previewHint("span", activePreview, previewPlacement));

    const variantField = document.createElement("label");
    variantField.className = "workspace-manager-field";
    variantField.append(document.createTextNode("Base style"));
    const variant = document.createElement("select");
    variant.setAttribute("aria-label", `${descriptor.title} style`);
    descriptor.supportedVariants.forEach((value) => variant.append(option(value, titleToken(value))));
    variant.value = placement.variant || descriptor.defaultVariant;
    variant.addEventListener("change", () => {
      if (node) setComponentShellVariant(node, variant.value);
      persist(pageId);
      renderManager(pageId);
    });
    variantField.append(variant, previewHint("variant", activePreview, previewPlacement));

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
    stateField.append(shellState, previewHint("state", activePreview, previewPlacement));

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
  renderResponsivePreview(pageId, activePreview);
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
  document.querySelectorAll("[data-workspace-preview-class]").forEach((button) => {
    button.addEventListener("click", () => {
      const requested = String(button.dataset.workspacePreviewClass || "").toLowerCase();
      if (!WORKSPACE_WIDTH_CLASSES.includes(requested)) return;
      previewWidthClass = requested;
      document.querySelectorAll("[data-workspace-preview-class]").forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      renderManager(pageSelect.value);
    });
  });
  bindWorkspaceWidthObserver(byId("workspaceManagerWorkspace"));
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
