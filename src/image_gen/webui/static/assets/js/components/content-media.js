import { registerComponentCapability } from "./capabilities.js?v=content-capabilities2";

const IMAGE_FITS = new Set(["contain", "cover", "fill", "scale-down", "none"]);
const IMAGE_CONTRAST_MODES = new Set(["none", "soft", "outline", "plate"]);
const IMAGE_POSITIONS = new Set([
  "center", "top", "bottom", "left", "right",
  "top left", "top right", "bottom left", "bottom right",
]);
const VIDEO_EXTENSIONS = /\.(?:mp4|webm|ogg)(?:$|[?#])/i;

function element(tag, className = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function boundedPixels(value, fallback, minimum, maximum) {
  const requested = Number(value);
  const candidate = Number.isFinite(requested) ? requested : Number(fallback);
  return Math.max(minimum, Math.min(maximum, Number.isFinite(candidate) ? candidate : minimum));
}

export function safeMediaHref(value, { allowExternal = true } = {}) {
  const href = String(value || "").trim();
  if (!href) return "";
  if (/^https:\/\//i.test(href)) return allowExternal ? href : "";
  if (href.startsWith("/") || href.startsWith("./") || href.startsWith("../") || href.startsWith("#")) return href;
  return "";
}

export function configureMediaFrame(frame, value = {}) {
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

function linkTarget(value = {}, href = "") {
  const explicit = String(value.target || "").trim();
  if (explicit) return explicit;
  const isExternal = value.external === true || (value.external !== false && /^https:\/\//i.test(href));
  return isExternal ? "_blank" : "";
}

export function wrapMediaLink(media, value = {}) {
  const href = safeMediaHref(value.href);
  if (!href || !media) return media;
  const link = document.createElement("a");
  link.className = "component-shell-media-link";
  link.href = href;
  const label = String(value.linkLabel || value.alt || value.title || "Open media link").trim() || "Open media link";
  link.setAttribute("aria-label", label);
  if (value.title || value.linkLabel) link.title = String(value.title || value.linkLabel);
  const target = linkTarget(value, href);
  if (target) link.target = target;
  if (target === "_blank") link.rel = "noopener noreferrer";
  link.append(media);
  return link;
}

export function renderImage(value = {}) {
  const figure = configureMediaFrame(element("figure", "component-shell-field component-shell-field--image"), value);
  const src = safeMediaHref(value.src, { allowExternal: value.allowExternal === true });
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

export function renderVideo(value = {}) {
  const figure = configureMediaFrame(element("figure", "component-shell-field component-shell-field--video"), value);
  const src = safeMediaHref(value.src, { allowExternal: false });
  if (!src || !VIDEO_EXTENSIONS.test(src)) return renderImage({ placeholder: "Video unavailable", icon: "info", ...value, src: "" });
  const video = document.createElement("video");
  video.className = "component-shell-media-video";
  video.controls = true;
  video.preload = "metadata";
  video.src = src;
  const poster = safeMediaHref(value.poster, { allowExternal: false });
  if (poster) video.poster = poster;
  if (value.alt) video.setAttribute("aria-label", String(value.alt));
  figure.append(video);
  if (value.caption) {
    const caption = document.createElement("figcaption");
    caption.textContent = String(value.caption);
    figure.append(caption);
  }
  return figure;
}

export function renderMediaCollection(container, items = []) {
  if (!container) return [];
  const rendered = [];
  for (const item of Array.isArray(items) ? items : []) {
    const type = String(item?.type || "image").toLowerCase();
    const node = type === "video" ? renderVideo(item) : renderImage(item);
    if (node) {
      container.append(node);
      rendered.push(node);
    }
  }
  return rendered;
}

export function enhanceDeclarativeMediaLinks(root) {
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

export const CONTENT_MEDIA_CAPABILITY = registerComponentCapability("content.media", {
  version: 1,
  description: "Safe shared image/video rendering for workspace components and Help Center content.",
  safeHref: safeMediaHref,
  configureFrame: configureMediaFrame,
  renderImage,
  renderVideo,
  renderCollection: renderMediaCollection,
  enhanceDeclarativeLinks: enhanceDeclarativeMediaLinks,
});
