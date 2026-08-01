import { $, shortText } from "../utils.js";
import { state } from "../state.js";

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.2;
const VIEW_PADDING = 24;

let elements = null;
let openingControl = null;
let bound = false;

const drag = {
  active: false,
  moved: false,
  pointerId: null,
  startX: 0,
  startY: 0,
  originScrollLeft: 0,
  originScrollTop: 0,
  suppressBackdropClose: false,
};

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function refs() {
  if (elements) return elements;
  elements = {
    dialog: $("#imageLightbox"),
    viewport: $("#lightboxViewport"),
    canvas: $("#lightboxCanvas"),
    image: $("#lightboxImage"),
    title: $("#lightboxTitle"),
    sourceLabel: $("#lightboxSourceLabel"),
    meta: $("#lightboxMeta"),
    zoomStatus: $("#lightboxZoomStatus"),
    previous: $("#lightboxPreviousButton"),
    next: $("#lightboxNextButton"),
    fit: $("#lightboxFitButton"),
    actual: $("#lightboxActualButton"),
    pan: $("#lightboxPanButton"),
    zoomOut: $("#lightboxZoomOutButton"),
    zoomIn: $("#lightboxZoomInButton"),
    followLatest: $("#lightboxFollowLatestButton"),
    close: $("#lightboxCloseButton"),
    closeToolbar: $("#lightboxCloseToolbarButton"),
  };
  return elements;
}

function completedIndex() {
  if (state.lightbox.sourceType !== "completed") return -1;
  return state.recentOutputs.findIndex((item) => item.name === state.lightbox.sourceId);
}

function fitScale() {
  const { image, viewport } = refs();
  if (!image.naturalWidth || !image.naturalHeight) return 1;
  const availableWidth = Math.max(1, viewport.clientWidth - VIEW_PADDING);
  const availableHeight = Math.max(1, viewport.clientHeight - VIEW_PADDING);
  return Math.min(
    availableWidth / image.naturalWidth,
    availableHeight / image.naturalHeight,
    1,
  );
}

function effectiveScale() {
  return state.lightbox.fitMode ? fitScale() : clamp(state.lightbox.zoom, MIN_ZOOM, MAX_ZOOM);
}

function currentCenterRatio() {
  const { viewport } = refs();
  const width = Math.max(1, viewport.scrollWidth);
  const height = Math.max(1, viewport.scrollHeight);
  return {
    x: clamp((viewport.scrollLeft + viewport.clientWidth / 2) / width, 0, 1),
    y: clamp((viewport.scrollTop + viewport.clientHeight / 2) / height, 0, 1),
  };
}

function scrollToCenterRatio(ratio = { x: 0.5, y: 0.5 }) {
  const { viewport } = refs();
  viewport.scrollLeft = Math.max(0, ratio.x * viewport.scrollWidth - viewport.clientWidth / 2);
  viewport.scrollTop = Math.max(0, ratio.y * viewport.scrollHeight - viewport.clientHeight / 2);
  state.lightbox.panX = viewport.scrollLeft;
  state.lightbox.panY = viewport.scrollTop;
}

function updateControlState(scale, pannable) {
  const {
    previous,
    next,
    fit,
    actual,
    pan,
    followLatest,
    zoomStatus,
  } = refs();
  const completed = state.lightbox.sourceType === "completed";
  const index = completedIndex();
  const canNavigate = completed && index >= 0 && state.recentOutputs.length > 1;
  previous.disabled = !canNavigate;
  next.disabled = !canNavigate;
  fit.setAttribute("aria-pressed", String(state.lightbox.fitMode));
  actual.setAttribute("aria-pressed", String(!state.lightbox.fitMode && Math.abs(state.lightbox.zoom - 1) < 0.001));
  pan.setAttribute("aria-pressed", String(state.lightbox.panEnabled));
  pan.disabled = !pannable;
  pan.title = pannable
    ? "Click and drag the image to pan. Native scrollbars remain available."
    : "Panning becomes available when the image is larger than the viewer.";
  followLatest.hidden = state.lightbox.sourceType !== "live";
  followLatest.setAttribute("aria-pressed", String(state.lightbox.followLatest));
  followLatest.textContent = state.lightbox.followLatest ? "Following Latest" : "Follow Latest";
  const mode = state.lightbox.fitMode ? " · Fit" : Math.abs(scale - 1) < 0.001 ? " · Actual" : "";
  zoomStatus.textContent = `${Math.round(scale * 100)}%${mode}${pannable ? " · Pan/scroll" : ""}`;
}

function applyTransform({ center = false, centerRatio = null } = {}) {
  if (!state.lightbox.open) return;
  const { image, viewport, canvas } = refs();
  const preservedCenter = centerRatio || (center ? { x: 0.5, y: 0.5 } : currentCenterRatio());
  const scale = effectiveScale();

  if (!image.naturalWidth || !image.naturalHeight) {
    updateControlState(scale, false);
    return;
  }

  const scaledWidth = Math.max(1, image.naturalWidth * scale);
  const scaledHeight = Math.max(1, image.naturalHeight * scale);
  const canvasWidth = Math.max(viewport.clientWidth, Math.ceil(scaledWidth + VIEW_PADDING));
  const canvasHeight = Math.max(viewport.clientHeight, Math.ceil(scaledHeight + VIEW_PADDING));

  canvas.style.width = `${canvasWidth}px`;
  canvas.style.height = `${canvasHeight}px`;
  image.style.width = `${scaledWidth}px`;
  image.style.height = `${scaledHeight}px`;
  image.style.transform = "translate(-50%, -50%)";

  const pannable = canvasWidth > viewport.clientWidth + 1 || canvasHeight > viewport.clientHeight + 1;
  viewport.classList.toggle("is-fit", state.lightbox.fitMode);
  viewport.classList.toggle("is-zoomed", scale > fitScale() + 0.001);
  viewport.classList.toggle("is-pannable", pannable);
  viewport.classList.toggle("is-pan-enabled", state.lightbox.panEnabled);

  scrollToCenterRatio(preservedCenter);
  updateControlState(scale, pannable);
}

function resetView({ fit = true } = {}) {
  state.lightbox.fitMode = fit;
  state.lightbox.zoom = 1;
  state.lightbox.panX = 0;
  state.lightbox.panY = 0;
  state.lightbox.panEnabled = true;
  applyTransform({ center: true });
}

function completedAlt(item) {
  const parts = [`Completed output ${item?.name || "image"}`];
  if (item?.prompt) parts.push(`Prompt: ${shortText(item.prompt, 180)}`);
  if (item?.seed !== undefined && item?.seed !== null) parts.push(`Seed: ${item.seed}`);
  return parts.join(". ");
}

function liveAlt(item) {
  const parts = ["Live generation preview"];
  if (item?.prompt) parts.push(`Prompt: ${shortText(item.prompt, 180)}`);
  if (item?.seed !== undefined && item?.seed !== null) parts.push(`Seed: ${item.seed}`);
  if (item?.step && item?.totalSteps) parts.push(`Step ${item.step} of ${item.totalSteps}`);
  return parts.join(". ");
}

function setImageSource({ url, alt, title, sourceLabel, meta }) {
  const view = refs();
  view.title.textContent = title;
  view.sourceLabel.textContent = sourceLabel;
  view.meta.textContent = meta;
  view.image.alt = alt;
  if (view.image.src !== new URL(url, window.location.href).href) {
    view.image.src = url;
  } else {
    applyTransform({ center: true });
  }
}

function renderCompleted(item) {
  if (!item?.url) return;
  state.lightbox.sourceType = "completed";
  state.lightbox.sourceId = item.name;
  const metadata = [
    item.prompt ? shortText(item.prompt, 120) : "No prompt metadata",
    item.seed !== undefined && item.seed !== null ? `Seed ${item.seed}` : null,
    item.width && item.height ? `${item.width} × ${item.height}` : null,
  ].filter(Boolean).join(" · ");
  setImageSource({
    url: item.url,
    alt: completedAlt(item),
    title: item.name || "Completed output",
    sourceLabel: "Completed Output",
    meta: metadata,
  });
  updateControlState(effectiveScale(), refs().viewport.classList.contains("is-pannable"));
}

function renderLive(item) {
  if (!item?.frameUrl) return;
  state.lightbox.sourceType = "live";
  state.lightbox.sourceId = item.frameUrl;
  const metadata = [
    item.prompt ? shortText(item.prompt, 120) : "Live generation frame",
    item.seed !== undefined && item.seed !== null ? `Seed ${item.seed}` : null,
    item.step && item.totalSteps ? `Step ${item.step} / ${item.totalSteps}` : null,
    item.samplerName ? item.samplerName : null,
  ].filter(Boolean).join(" · ");
  setImageSource({
    url: item.frameUrl,
    alt: liveAlt(item),
    title: item.frameName || (item.step ? `Live frame · step ${item.step}` : "Live preview"),
    sourceLabel: "Live Preview",
    meta: metadata,
  });
  updateControlState(effectiveScale(), refs().viewport.classList.contains("is-pannable"));
}

function openDialog(opener) {
  const { dialog, close } = refs();
  openingControl = opener || document.activeElement;
  state.lightbox.open = true;
  if (!dialog.open) dialog.showModal();
  requestAnimationFrame(() => {
    applyTransform({ center: true });
    close.focus({ preventScroll: true });
  });
}

export function openCompletedLightbox(item = state.selectedOutput, options = {}) {
  if (!item?.url) return;
  resetView({ fit: true });
  renderCompleted(item);
  openDialog(options.opener);
}

export function openLiveLightbox(item = state.livePreview, options = {}) {
  if (!item?.frameUrl) return;
  state.lightbox.followLatest = item.followLatest !== false;
  resetView({ fit: true });
  renderLive(item);
  openDialog(options.opener);
}

export function syncLiveLightbox() {
  if (
    !state.lightbox.open
    || state.lightbox.sourceType !== "live"
    || !state.lightbox.followLatest
    || !state.livePreview?.frameUrl
  ) return;
  renderLive(state.livePreview);
}

function navigateCompleted(direction) {
  if (state.lightbox.sourceType !== "completed" || !state.recentOutputs.length) return;
  const index = completedIndex();
  if (index < 0) return;
  const nextIndex = (index + direction + state.recentOutputs.length) % state.recentOutputs.length;
  resetView({ fit: true });
  renderCompleted(state.recentOutputs[nextIndex]);
}

function setFit() {
  resetView({ fit: true });
}

function setActualSize() {
  state.lightbox.fitMode = false;
  state.lightbox.zoom = 1;
  state.lightbox.panX = 0;
  state.lightbox.panY = 0;
  state.lightbox.panEnabled = true;
  applyTransform({ center: true });
}

function changeZoom(factor) {
  const centerRatio = currentCenterRatio();
  const current = effectiveScale();
  state.lightbox.fitMode = false;
  state.lightbox.zoom = clamp(current * factor, MIN_ZOOM, MAX_ZOOM);
  applyTransform({ centerRatio });
}

function toggleFitAndActual() {
  if (state.lightbox.fitMode) setActualSize();
  else setFit();
}

function togglePan() {
  state.lightbox.panEnabled = !state.lightbox.panEnabled;
  applyTransform();
}

function closeLightbox() {
  const { dialog } = refs();
  if (dialog.open) dialog.close();
}

function beginDrag(event) {
  const { viewport } = refs();
  if (!state.lightbox.panEnabled || !viewport.classList.contains("is-pannable")) return;
  if (event.button !== undefined && event.button !== 0) return;
  drag.active = true;
  drag.moved = false;
  drag.pointerId = event.pointerId;
  drag.startX = event.clientX;
  drag.startY = event.clientY;
  drag.originScrollLeft = viewport.scrollLeft;
  drag.originScrollTop = viewport.scrollTop;
  viewport.classList.add("is-dragging");
  viewport.setPointerCapture?.(event.pointerId);
  event.preventDefault();
}

function moveDrag(event) {
  if (!drag.active || event.pointerId !== drag.pointerId) return;
  const { viewport } = refs();
  const deltaX = event.clientX - drag.startX;
  const deltaY = event.clientY - drag.startY;
  if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) drag.moved = true;
  viewport.scrollLeft = drag.originScrollLeft - deltaX;
  viewport.scrollTop = drag.originScrollTop - deltaY;
  state.lightbox.panX = viewport.scrollLeft;
  state.lightbox.panY = viewport.scrollTop;
}

function endDrag(event) {
  if (!drag.active || event.pointerId !== drag.pointerId) return;
  const { viewport } = refs();
  drag.active = false;
  drag.suppressBackdropClose = drag.moved;
  viewport.classList.remove("is-dragging");
  viewport.releasePointerCapture?.(event.pointerId);
  drag.pointerId = null;
  window.setTimeout(() => {
    drag.suppressBackdropClose = false;
  }, 0);
}

function handleKeyboard(event) {
  if (!state.lightbox.open) return;
  if (event.key === "ArrowLeft" && state.lightbox.sourceType === "completed") {
    event.preventDefault();
    navigateCompleted(-1);
  } else if (event.key === "ArrowRight" && state.lightbox.sourceType === "completed") {
    event.preventDefault();
    navigateCompleted(1);
  } else if (event.key === "0") {
    event.preventDefault();
    setFit();
  } else if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    changeZoom(ZOOM_STEP);
  } else if (event.key === "-") {
    event.preventDefault();
    changeZoom(1 / ZOOM_STEP);
  } else if (event.key.toLowerCase() === "p") {
    event.preventDefault();
    togglePan();
  }
}

export function bindLightbox() {
  if (bound) return;
  const view = refs();
  if (!view.dialog) return;
  bound = true;

  view.previous.addEventListener("click", () => navigateCompleted(-1));
  view.next.addEventListener("click", () => navigateCompleted(1));
  view.fit.addEventListener("click", setFit);
  view.actual.addEventListener("click", setActualSize);
  view.pan.addEventListener("click", togglePan);
  view.zoomOut.addEventListener("click", () => changeZoom(1 / ZOOM_STEP));
  view.zoomIn.addEventListener("click", () => changeZoom(ZOOM_STEP));
  view.followLatest.addEventListener("click", () => {
    state.lightbox.followLatest = !state.lightbox.followLatest;
    state.livePreview.followLatest = state.lightbox.followLatest;
    window.dispatchEvent(new CustomEvent("live-preview-follow-changed", {
      detail: { followLatest: state.lightbox.followLatest },
    }));
    if (state.lightbox.followLatest) syncLiveLightbox();
    updateControlState(effectiveScale(), view.viewport.classList.contains("is-pannable"));
  });
  view.close.addEventListener("click", closeLightbox);
  view.closeToolbar.addEventListener("click", closeLightbox);
  view.dialog.addEventListener("keydown", handleKeyboard);
  view.dialog.addEventListener("click", (event) => {
    if (event.target === view.dialog && !drag.suppressBackdropClose) closeLightbox();
  });
  view.dialog.addEventListener("close", () => {
    state.lightbox.open = false;
    drag.active = false;
    view.viewport.classList.remove("is-dragging");
    const restore = openingControl;
    openingControl = null;
    if (restore?.isConnected) restore.focus({ preventScroll: true });
  });
  view.image.addEventListener("load", () => applyTransform({ center: true }));
  view.image.addEventListener("dragstart", (event) => event.preventDefault());
  view.viewport.addEventListener("dblclick", (event) => {
    event.preventDefault();
    toggleFitAndActual();
  });
  view.viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    changeZoom(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
  }, { passive: false });
  view.viewport.addEventListener("pointerdown", beginDrag);
  view.viewport.addEventListener("pointermove", moveDrag);
  view.viewport.addEventListener("pointerup", endDrag);
  view.viewport.addEventListener("pointercancel", endDrag);
  view.viewport.addEventListener("scroll", () => {
    state.lightbox.panX = view.viewport.scrollLeft;
    state.lightbox.panY = view.viewport.scrollTop;
  }, { passive: true });
  window.addEventListener("resize", () => {
    if (state.lightbox.open) applyTransform();
  });

  updateControlState(1, false);
}
