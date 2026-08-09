import { initActionBar, refreshActionBar } from "./action-bar.js?v=responsive-action-bar1";
import { setActionIcon } from "./action-icons.js?v=0.1.1";
import { openMarkdownDocument } from "./markdown-reader.js?v=component-shell1";

const fieldRenderers = new Map();
const shellInstances = new WeakMap();
const VALID_STATES = new Set(["expanded", "summary", "collapsed", "side"]);
const FEATURED_VARIANTS = new Set(["featured"]);

function element(tag, className = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function appendText(container, value) {
  if (value === undefined || value === null || value === "") return;
  container.append(document.createTextNode(String(value)));
}

function renderTextField(value) {
  const paragraph = element("p", "component-shell-field component-shell-field--text");
  appendText(paragraph, typeof value === "object" ? value.text : value);
  return paragraph;
}

const IMAGE_FITS = new Set(["contain", "cover", "fill", "scale-down", "none"]);
const IMAGE_CONTRAST_MODES = new Set(["none", "soft", "outline", "plate"]);
const IMAGE_POSITIONS = new Set([
  "center", "top", "bottom", "left", "right",
  "top left", "top right", "bottom left", "bottom right",
]);

function boundedPixels(value, fallback, minimum, maximum) {
  const requested = Number(value);
  const candidate = Number.isFinite(requested) ? requested : Number(fallback);
  return Math.max(minimum, Math.min(maximum, Number.isFinite(candidate) ? candidate : minimum));
}

function configureMediaFrame(frame, value = {}) {
  const fit = String(value.fit || "contain").trim().toLowerCase();
  const position = String(value.position || "center").trim().toLowerCase();
  const requestedContrast = value.contrastAssist === true ? "soft" : String(value.contrastAssist || "none").trim().toLowerCase();
  const contrast = IMAGE_CONTRAST_MODES.has(requestedContrast) ? requestedContrast : "none";
  const padding = boundedPixels(value.padding, 8, 0, 64);
  const minHeight = boundedPixels(value.minHeight, 120, 48, 1200);
  const maxHeight = boundedPixels(value.maxHeight, 420, minHeight, 1600);
  const preferredHeight = boundedPixels(value.height, Math.min(220, maxHeight), minHeight, maxHeight);

  frame.classList.add("component-shell-media-frame");
  frame.style.setProperty("--component-shell-image-fit", IMAGE_FITS.has(fit) ? fit : "contain");
  frame.style.setProperty("--component-shell-image-position", IMAGE_POSITIONS.has(position) ? position : "center");
  frame.style.setProperty("--component-shell-image-padding", `${padding}px`);
  frame.style.setProperty("--component-shell-image-min-height", `${minHeight}px`);
  frame.style.setProperty("--component-shell-image-height", `${preferredHeight}px`);
  frame.style.setProperty("--component-shell-image-max-height", `${maxHeight}px`);
  frame.dataset.componentImageFit = IMAGE_FITS.has(fit) ? fit : "contain";
  frame.dataset.componentImageContrast = contrast;
  return frame;
}

function safeMediaHref(value) {
  const href = String(value || "").trim();
  if (!href) return "";
  if (/^https?:\/\//i.test(href)) return href;
  if (href.startsWith("/") || href.startsWith("./") || href.startsWith("../") || href.startsWith("#")) return href;
  return "";
}

function imageLinkTarget(value = {}, href = "") {
  const explicit = String(value.target || "").trim();
  if (explicit) return explicit;
  const isExternal = value.external === true
    || (value.external !== false && /^https?:\/\//i.test(href));
  return isExternal ? "_blank" : "";
}

function wrapMediaLink(media, value = {}) {
  const href = safeMediaHref(value.href);
  if (!href || !media) return media;
  const link = document.createElement("a");
  link.className = "component-shell-media-link";
  link.href = href;
  const label = String(value.linkLabel || value.alt || value.title || "Open image link").trim() || "Open image link";
  link.setAttribute("aria-label", label);
  if (value.title || value.linkLabel) link.title = String(value.title || value.linkLabel);
  const target = imageLinkTarget(value, href);
  if (target) link.target = target;
  if (target === "_blank") link.rel = "noopener noreferrer";
  link.append(media);
  return link;
}

function enhanceDeclarativeMediaLinks(root) {
  root?.querySelectorAll?.("[data-component-media-href]").forEach((frame) => {
    if (frame.querySelector(":scope > .component-shell-media-link")) return;
    const media = frame.querySelector(":scope > .component-shell-media-image");
    if (!media) return;
    const wrapped = wrapMediaLink(media, {
      href: frame.dataset.componentMediaHref,
      linkLabel: frame.dataset.componentMediaLinkLabel || media.alt || "Open image link",
      target: frame.dataset.componentMediaTarget || "",
      external: frame.dataset.componentMediaExternal === "false" ? false : undefined,
    });
    if (wrapped !== media) frame.append(wrapped);
  });
}

function renderImageField(value = {}) {
  const figure = configureMediaFrame(
    element("figure", "component-shell-field component-shell-field--image"),
    value,
  );
  const src = String(value.src || "").trim();
  if (src) {
    const image = document.createElement("img");
    image.className = "component-shell-media-image";
    image.src = src;
    image.alt = String(value.alt || "");
    image.loading = value.loading === "eager" ? "eager" : "lazy";
    image.decoding = "async";
    figure.append(wrapMediaLink(image, value));
  } else {
    const placeholder = element("div", "component-shell-image-placeholder component-shell-media-image");
    placeholder.setAttribute("role", "img");
    placeholder.setAttribute("aria-label", String(value.alt || value.placeholder || "Image placeholder"));
    const icon = element("span", "ui-icon");
    icon.dataset.icon = String(value.icon || "info");
    icon.setAttribute("aria-hidden", "true");
    placeholder.append(icon);
    figure.append(placeholder);
  }
  if (value.caption) {
    const caption = document.createElement("figcaption");
    caption.textContent = String(value.caption);
    figure.append(caption);
  }
  return figure;
}

function renderLinksField(value = [], context = {}) {
  const list = element("div", "component-shell-field component-shell-links");
  for (const item of Array.isArray(value) ? value : [value]) {
    const link = document.createElement("a");
    link.className = "component-shell-link ui-icon-control";
    link.textContent = String(item.label || item.title || item.href || "Open link");
    link.href = String(item.href || "#");
    if (item.icon) setActionIcon(link, item.icon, { label: link.textContent, title: link.textContent });
    if (item.markdown || item.kind === "markdown" || /\.md(?:$|[?#])/i.test(link.href)) {
      link.dataset.componentShellMarkdown = "true";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        if (typeof context.openMarkdown === "function") {
          context.openMarkdown(item, context);
          return;
        }
        openMarkdownDocument({
          title: item.title || item.label || "Markdown document",
          markdown: item.markdown ?? null,
          href: item.markdownHref || item.href || "",
          sourceHref: item.sourceHref || item.href || "",
          loader: item.markdownLoader || null,
          options: item.markdownOptions || {},
        });
      });
    } else if (item.external !== false) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    list.append(link);
  }
  return list;
}

function renderAttachmentsField(value = []) {
  const list = element("div", "component-shell-field component-shell-attachments");
  for (const item of Array.isArray(value) ? value : [value]) {
    const row = element("div", "component-shell-attachment");
    const icon = element("span", "ui-icon");
    icon.dataset.icon = String(item.icon || "folder-open");
    icon.setAttribute("aria-hidden", "true");
    const copy = element("span", "component-shell-attachment__copy");
    const name = element("strong");
    name.textContent = String(item.name || item.label || "Attachment");
    copy.append(name);
    if (item.metadata) {
      const meta = document.createElement("small");
      meta.textContent = String(item.metadata);
      copy.append(meta);
    }
    row.append(icon, copy);
    list.append(row);
  }
  return list;
}

function renderMetadataField(value = {}) {
  const list = document.createElement("dl");
  list.className = "component-shell-field component-shell-metadata";
  const entries = Array.isArray(value) ? value.map((item) => [item.label, item.value]) : Object.entries(value);
  for (const [label, itemValue] of entries) {
    const group = element("div", "component-shell-metadata__item");
    const term = document.createElement("dt");
    term.textContent = String(label || "Metadata");
    const description = document.createElement("dd");
    description.textContent = String(itemValue ?? "—");
    group.append(term, description);
    list.append(group);
  }
  return list;
}

function renderCardsField(value = []) {
  const grid = element("div", "component-shell-field component-shell-cards");
  for (const item of Array.isArray(value) ? value : [value]) {
    const card = element("article", "component-shell-card");
    if (item.image || item.imagePlaceholder) card.append(renderImageField(item.image || { placeholder: item.imagePlaceholder }));
    const body = element("div", "component-shell-card__body");
    const title = document.createElement("strong");
    title.textContent = String(item.title || item.name || "Card");
    body.append(title);
    if (item.text || item.description) body.append(renderTextField(item.text || item.description));
    if (item.metadata) body.append(renderMetadataField(item.metadata));
    card.append(body);
    grid.append(card);
  }
  return grid;
}

export function registerComponentShellField(type, renderer) {
  const key = String(type || "").trim().toLocaleLowerCase();
  if (!key || typeof renderer !== "function") throw new TypeError("Component shell fields require a type and renderer.");
  fieldRenderers.set(key, renderer);
}

registerComponentShellField("text", renderTextField);
registerComponentShellField("image", renderImageField);
registerComponentShellField("imageplaceholder", (value) => renderImageField({ ...(value || {}), src: "" }));
registerComponentShellField("links", renderLinksField);
registerComponentShellField("attachments", renderAttachmentsField);
registerComponentShellField("cards", renderCardsField);
registerComponentShellField("metadata", renderMetadataField);

export function renderComponentShellFields(container, fields = [], context = {}) {
  if (!container) return [];
  const rendered = [];
  for (const field of Array.isArray(fields) ? fields : []) {
    const type = String(field?.type || "text").toLocaleLowerCase();
    const renderer = fieldRenderers.get(type);
    if (!renderer) continue;
    const node = renderer(field.value ?? field, context);
    if (node) {
      node.dataset.componentShellField = type;
      container.append(node);
      rendered.push(node);
    }
  }
  return rendered;
}

function ensureHeader(root, descriptor) {
  let header = root.querySelector(":scope > [data-component-shell-header]");
  if (!header) {
    header = element("header", "component-shell__header");
    header.dataset.componentShellHeader = "";
    root.prepend(header);
  }
  header.classList.add("component-shell__header");

  let identity = header.querySelector(":scope > [data-component-shell-identity]");
  if (!identity) {
    identity = element("div", "component-shell__identity");
    identity.dataset.componentShellIdentity = "";
    const icon = element("span", "ui-icon component-shell__icon");
    icon.dataset.icon = descriptor.icon || "info";
    icon.setAttribute("aria-hidden", "true");
    const title = document.createElement("strong");
    title.className = "component-shell__title";
    title.dataset.componentShellTitle = "";
    title.textContent = descriptor.title;
    identity.append(icon, title);
    header.prepend(identity);
  }

  let actions = header.querySelector(":scope > [data-component-shell-actions]");
  if (!actions) {
    actions = element("div", "component-shell__actions");
    actions.dataset.componentShellActions = "";
    actions.dataset.actionBar = "";
    actions.dataset.actionBarOverflowLabel = `${descriptor.title} actions`;
    header.append(actions);
  }
  actions.classList.add("component-shell__actions", "ui-action-bar");
  actions.dataset.actionBar = "";
  return { header, identity, actions };
}

function actionButton(icon, label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.componentShellAction = action;
  button.dataset.actionPinned = "true";
  setActionIcon(button, icon, { label, title: label, replace: true });
  return button;
}

function ensureSummary(root) {
  const summary = root.querySelector(":scope > [data-component-shell-summary]");
  if (!summary) return null;
  summary.classList.add("component-shell__summary");
  return summary;
}

function syncSummary(instance) {
  if (!instance.summary) return;
  instance.summary.querySelectorAll("[data-component-summary-source]").forEach((target) => {
    const selector = String(target.dataset.componentSummarySource || "").trim();
    if (!selector) return;
    let source = null;
    try {
      source = instance.root.querySelector(selector);
    } catch (_error) {
      source = null;
    }
    const fallback = String(target.dataset.componentSummaryFallback || "—");
    target.textContent = String(source?.textContent || "").trim() || fallback;
  });
}

function ensureVariantBadge(identity) {
  let badge = identity.querySelector(":scope > [data-component-shell-variant-badge]");
  if (!badge) {
    badge = element("span", "component-shell__variant-badge");
    badge.dataset.componentShellVariantBadge = "";
    badge.hidden = true;
    identity.append(badge);
  }
  return badge;
}

function normalizeVariant(descriptor, requested) {
  const raw = String(requested || descriptor.defaultVariant || "standard").trim().toLocaleLowerCase();
  const variant = raw === "feature" ? "feature" : raw;
  const supported = new Set((descriptor.supportedVariants || [descriptor.defaultVariant || "standard"]).map((item) => String(item).toLocaleLowerCase()));
  return supported.has(variant) ? variant : String(descriptor.defaultVariant || "standard").toLocaleLowerCase();
}

function setVariant(instance, requested) {
  const variant = normalizeVariant(instance.descriptor, requested);
  instance.variant = variant;
  instance.root.dataset.componentVariant = variant;
  for (const className of [...instance.root.classList]) {
    if (className.startsWith("component-shell--variant-")) instance.root.classList.remove(className);
  }
  instance.root.classList.add(`component-shell--variant-${variant}`);
  const featured = FEATURED_VARIANTS.has(variant);
  instance.variantBadge.hidden = !featured;
  instance.variantBadge.textContent = featured ? String(instance.descriptor.shell?.featuredLabel || "Featured") : "";
  instance.root.setAttribute("aria-roledescription", featured ? "featured component" : "component");
  instance.root.dispatchEvent(new CustomEvent("component-shell-variant-change", {
    bubbles: true,
    detail: { componentId: instance.descriptor.componentId, variant },
  }));
  return variant;
}

function setState(instance, requested) {
  const state = VALID_STATES.has(requested) ? requested : "expanded";
  instance.state = state;
  instance.root.dataset.componentShellState = state;
  instance.root.classList.toggle("component-shell--summary", state === "summary");
  instance.root.classList.toggle("component-shell--collapsed", state === "collapsed");
  instance.root.classList.toggle("component-shell--side", state === "side");
  const expanded = state === "expanded";
  instance.body.hidden = !expanded;
  if (instance.summary) {
    syncSummary(instance);
    instance.summary.hidden = state !== "summary";
  }
  instance.collapse.setAttribute("aria-expanded", String(expanded));
  setActionIcon(instance.collapse, expanded ? "chevron-up" : "chevron-down", {
    label: expanded ? `Minimize ${instance.descriptor.title} to title bar` : `Expand ${instance.descriptor.title}`,
    title: expanded ? `Minimize ${instance.descriptor.title} to title bar` : `Expand ${instance.descriptor.title}`,
    replace: true,
  });
  if (instance.summaryToggle) {
    setActionIcon(instance.summaryToggle, state === "summary" ? "maximize" : "text-decrease", {
      label: state === "summary" ? `Expand ${instance.descriptor.title}` : `Compact ${instance.descriptor.title} to summary row`,
      title: state === "summary" ? `Expand ${instance.descriptor.title}` : `Compact ${instance.descriptor.title} to summary row`,
      replace: true,
    });
  }
  setActionIcon(instance.side, state === "side" ? "maximize" : "side-tuck", {
    label: state === "side" ? `Restore ${instance.descriptor.title}` : `Tuck ${instance.descriptor.title} to side`,
    title: state === "side" ? `Restore ${instance.descriptor.title}` : `Tuck ${instance.descriptor.title} to side`,
    replace: true,
  });
  instance.root.dispatchEvent(new CustomEvent("component-shell-state-change", {
    bubbles: true,
    detail: { componentId: instance.descriptor.componentId, state },
  }));
  refreshActionBar(instance.actions);
  return state;
}

export function initComponentShell(root, descriptor, options = {}) {
  if (!root) return null;
  if (shellInstances.has(root)) return shellInstances.get(root);
  root.classList.add("component-shell");
  root.dataset.componentShell = "";
  const { header, identity, actions } = ensureHeader(root, descriptor);
  const variantBadge = ensureVariantBadge(identity);
  const summary = ensureSummary(root);
  let body = root.querySelector(":scope > [data-component-shell-body]");
  if (!body) {
    body = element("div", "component-shell__body");
    body.dataset.componentShellBody = "";
    [...root.children].filter((node) => node !== header && !node.classList.contains("component-shell__registry-error")).forEach((node) => body.append(node));
    root.append(body);
  }
  body.classList.add("component-shell__body");
  enhanceDeclarativeMediaLinks(body);

  const summaryToggle = summary
    ? actionButton("text-decrease", `Compact ${descriptor.title} to summary row`, "summary")
    : null;
  const collapse = actionButton("chevron-up", `Minimize ${descriptor.title} to title bar`, "collapse");
  const side = actionButton("side-tuck", `Tuck ${descriptor.title} to side`, "side");
  if (summaryToggle) actions.append(summaryToggle);
  actions.append(collapse, side);
  const actionBar = initActionBar(actions);
  const instance = { root, descriptor, header, actions, body, summary, summaryToggle, collapse, side, actionBar, variantBadge, state: "expanded", variant: "standard" };
  shellInstances.set(root, instance);

  summaryToggle?.addEventListener("click", () => setState(instance, instance.state === "summary" ? "expanded" : "summary"));
  collapse.addEventListener("click", () => setState(instance, instance.state === "expanded" ? "collapsed" : "expanded"));
  side.addEventListener("click", () => setState(instance, instance.state === "side" ? "expanded" : "side"));
  if (summary) {
    const summaryObserver = new MutationObserver(() => {
      if (instance.state === "summary") syncSummary(instance);
    });
    summaryObserver.observe(body, { childList: true, characterData: true, subtree: true });
    instance.summaryObserver = summaryObserver;
  }
  setVariant(instance, String(root.dataset.componentVariant || options.variant || descriptor.defaultVariant || "standard"));
  setState(instance, String(root.dataset.componentShellState || options.initialState || "expanded"));
  return instance;
}

export function setComponentShellState(root, state) {
  const instance = shellInstances.get(root);
  return instance ? setState(instance, state) : null;
}

export function setComponentShellVariant(root, variant) {
  const instance = shellInstances.get(root);
  return instance ? setVariant(instance, variant) : null;
}

export function componentShellSnapshot(root) {
  const instance = shellInstances.get(root);
  return instance ? { componentId: instance.descriptor.componentId, state: instance.state, variant: instance.variant } : null;
}
