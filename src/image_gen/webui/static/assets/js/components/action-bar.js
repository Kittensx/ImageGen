import { setActionBadge, setActionIcon } from "./action-icons.js?v=0.1.1";

const instances = new WeakMap();
let outsideHandlerInstalled = false;

function numericPriority(item) {
  if (item.dataset.actionPinned === "true") return Number.POSITIVE_INFINITY;
  const value = Number(item.dataset.actionPriority);
  return Number.isFinite(value) ? value : 50;
}

function groupKey(item, index) {
  return String(item.dataset.actionGroup || `__action_${index}`);
}

function installOutsideHandler() {
  if (outsideHandlerInstalled) return;
  outsideHandlerInstalled = true;
  document.addEventListener("pointerdown", (event) => {
    document.querySelectorAll(".ui-action-bar__overflow[open]").forEach((details) => {
      if (!details.contains(event.target)) details.removeAttribute("open");
    });
  });
}

function createOverflow(root) {
  const details = document.createElement("details");
  details.className = "ui-action-bar__overflow";
  details.hidden = true;

  const summary = document.createElement("summary");
  summary.className = "ui-action-bar__overflow-trigger";
  setActionIcon(summary, "more", {
    label: root.dataset.actionBarOverflowLabel || "More actions",
    title: root.dataset.actionBarOverflowLabel || "More actions",
    replace: true,
  });

  const menu = document.createElement("div");
  menu.className = "ui-action-bar__overflow-menu";
  menu.setAttribute("role", "group");
  menu.setAttribute("aria-label", root.dataset.actionBarOverflowLabel || "More actions");
  details.append(summary, menu);

  menu.addEventListener("click", (event) => {
    if (event.target.closest("button, a") && !event.target.closest("[data-action-bar-stay-open='true']")) {
      window.setTimeout(() => details.removeAttribute("open"), 0);
    }
  });

  if (root.dataset.actionBarHover === "true") {
    const finePointer = window.matchMedia?.("(hover: hover) and (pointer: fine)");
    let closeTimer = 0;
    details.addEventListener("mouseenter", () => {
      if (!finePointer?.matches || details.hidden) return;
      window.clearTimeout(closeTimer);
      details.open = true;
    });
    details.addEventListener("mouseleave", () => {
      if (!finePointer?.matches) return;
      closeTimer = window.setTimeout(() => {
        if (!details.matches(":focus-within")) details.removeAttribute("open");
      }, 180);
    });
  }

  return { details, summary, menu };
}

function collectItems(instance) {
  const known = new Set(instance.items);
  const candidates = [
    ...instance.root.children,
    ...instance.primary.children,
    ...instance.menu.children,
  ].filter((item) => item !== instance.primary && item !== instance.overflow);
  for (const item of candidates) {
    if (known.has(item)) continue;
    item.dataset.actionBarOrder = String(instance.nextOrder++);
    instance.items.push(item);
    known.add(item);
  }
  instance.items.sort((left, right) => Number(left.dataset.actionBarOrder) - Number(right.dataset.actionBarOrder));
}

function restorePrimary(instance) {
  collectItems(instance);
  for (const item of instance.items) instance.primary.append(item);
  instance.overflow.hidden = true;
  instance.overflow.removeAttribute("open");
  instance.root.dataset.actionBarOverflowing = "false";
  instance.root.style.setProperty("--ui-action-overflow-count", "0");
  setActionBadge(instance.summary, 0);
}

function groupedCandidates(instance) {
  const groups = new Map();
  instance.items.forEach((item, index) => {
    const key = groupKey(item, index);
    if (!groups.has(key)) groups.set(key, { key, items: [], priority: numericPriority(item), order: index, pinned: false });
    const group = groups.get(key);
    group.items.push(item);
    group.priority = Math.max(group.priority, numericPriority(item));
    group.pinned = group.pinned || item.dataset.actionPinned === "true";
  });
  return [...groups.values()]
    .filter((group) => !group.pinned)
    .sort((left, right) => left.priority - right.priority || right.order - left.order);
}

function primaryFits(instance) {
  return instance.primary.scrollWidth <= instance.primary.clientWidth + 1;
}

function layout(instance) {
  if (!instance.root.isConnected || instance.root.clientWidth <= 0) return;
  restorePrimary(instance);
  if (primaryFits(instance)) return;

  instance.overflow.hidden = false;
  const candidates = groupedCandidates(instance);
  let overflowed = 0;
  for (const group of candidates) {
    group.items.forEach((item) => instance.menu.prepend(item));
    /* prepend reverses groups, restore original order below. */
    [...instance.menu.children]
      .sort((left, right) => Number(left.dataset.actionBarOrder) - Number(right.dataset.actionBarOrder))
      .forEach((item) => instance.menu.append(item));
    overflowed += group.items.length;
    if (primaryFits(instance)) break;
  }

  instance.root.dataset.actionBarOverflowing = overflowed ? "true" : "false";
  instance.root.style.setProperty("--ui-action-overflow-count", String(overflowed));
  setActionBadge(instance.summary, overflowed);
  if (!overflowed) instance.overflow.hidden = true;
}

function schedule(instance) {
  if (instance.frame) cancelAnimationFrame(instance.frame);
  instance.frame = requestAnimationFrame(() => {
    instance.frame = 0;
    layout(instance);
  });
}

export function initActionBar(root) {
  if (!root || instances.has(root)) return instances.get(root) || null;
  installOutsideHandler();
  root.classList.add("ui-action-bar");

  const existing = [...root.children];
  const primary = document.createElement("div");
  primary.className = "ui-action-bar__primary";
  const { details: overflow, summary, menu } = createOverflow(root);
  root.replaceChildren(primary, overflow);

  const instance = {
    root,
    primary,
    overflow,
    summary,
    menu,
    items: [],
    nextOrder: 0,
    frame: 0,
    resizeObserver: null,
    mutationObserver: null,
  };
  instances.set(root, instance);

  existing.forEach((item) => {
    item.dataset.actionBarOrder = String(instance.nextOrder++);
    instance.items.push(item);
    primary.append(item);
  });

  if (typeof ResizeObserver === "function") {
    instance.resizeObserver = new ResizeObserver(() => schedule(instance));
    instance.resizeObserver.observe(root);
  } else {
    window.addEventListener("resize", () => schedule(instance), { passive: true });
  }
  if (typeof MutationObserver === "function") {
    instance.mutationObserver = new MutationObserver((records) => {
      if (records.some((record) => [...record.addedNodes, ...record.removedNodes].some((node) => node !== primary && node !== overflow))) {
        schedule(instance);
      }
    });
    instance.mutationObserver.observe(root, { childList: true });
  }

  root.addEventListener("ui-action-bar-refresh", () => schedule(instance));
  schedule(instance);
  return instance;
}

export function initResponsiveActionBars(scope = document) {
  const roots = [...scope.querySelectorAll("[data-action-bar]")];
  return roots.map(initActionBar).filter(Boolean);
}

export function refreshActionBar(root) {
  const instance = instances.get(root);
  if (instance) schedule(instance);
}
