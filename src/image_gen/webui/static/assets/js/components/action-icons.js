export function actionIconNode(icon) {
  const node = document.createElement("span");
  node.className = "ui-icon";
  node.dataset.icon = String(icon || "");
  node.setAttribute("aria-hidden", "true");
  return node;
}

export function setActionIcon(control, icon, { label = "", title = label, replace = false } = {}) {
  if (!control) return null;
  control.classList.add("ui-action-button", "ui-icon-control");
  let node = control.querySelector(":scope > .ui-icon");
  if (!node) {
    node = actionIconNode(icon);
    if (replace) control.replaceChildren(node);
    else control.prepend(node);
  } else {
    node.dataset.icon = String(icon || "");
  }
  if (label) control.setAttribute("aria-label", label);
  if (title) control.title = title;
  return node;
}

export function setActionBadge(control, value, { hiddenWhenZero = true } = {}) {
  if (!control) return null;
  let badge = control.querySelector(":scope > .ui-action-badge");
  if (!badge) {
    badge = document.createElement("span");
    badge.className = "ui-action-badge";
    badge.setAttribute("aria-hidden", "true");
    control.append(badge);
  }
  const text = String(value ?? "");
  badge.textContent = text;
  badge.hidden = hiddenWhenZero && (!text || text === "0");
  return badge;
}
