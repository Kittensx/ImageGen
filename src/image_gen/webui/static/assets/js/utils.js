export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function debounce(callback, delay = 500) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), delay);
  };
}

export function option(value, label, selected = false) {
  const element = document.createElement("option");
  element.value = value ?? "";
  element.textContent = label;
  element.selected = selected;
  return element;
}

export function replaceOptions(select, values, currentValue = "") {
  select.replaceChildren(...values);
  if ([...select.options].some((item) => item.value === String(currentValue ?? ""))) {
    select.value = String(currentValue ?? "");
  }
}

export function numberValue(input, fallback = null) {
  if (input.value.trim() === "") {
    return fallback;
  }
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

export function shortText(value, limit = 54) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

export function formatTime(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function notify(message, kind = "info") {
  const existing = $("#appNotice");
  if (existing) {
    existing.remove();
  }
  const notice = document.createElement("div");
  notice.id = "appNotice";
  notice.className = `app-notice ${kind}`;
  notice.textContent = message;
  document.body.append(notice);
  setTimeout(() => notice.remove(), 4200);
}
