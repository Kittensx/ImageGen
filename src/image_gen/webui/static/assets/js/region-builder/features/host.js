/* Clipboard and IMAGE_GEN host-window integration. */
function copyPrompt() {
  const ta = document.getElementById('promptText');
  ta.select();
  navigator.clipboard.writeText(ta.value).then(()=>toast('Copied!'));
}

function promptWithEmbeddedCanvas() {
  const ta = document.getElementById('promptText');
  const store = document.getElementById('canvasStore');
  let text = ta.value;
  if (store.value && text.includes('canvas=1')) {
    text = text.replace('canvas=1', 'canvas=' + store.value);
  }
  return text;
}

function copyWithCanvas() {
  const store = document.getElementById('canvasStore');
  navigator.clipboard.writeText(promptWithEmbeddedCanvas()).then(()=>toast(store.value ? 'Copied with canvas!' : 'Copied (no canvas)'));
}

function getHostWindow() {
  try {
    if (window.parent && window.parent !== window) return window.parent;
  } catch (_err) { /* ignore cross-origin parent */ }
  try {
    if (window.opener && !window.opener.closed) return window.opener;
  } catch (_err) { /* ignore closed opener */ }
  return null;
}

function postToHost(payload) {
  const host = getHostWindow();
  if (!host) return false;
  host.postMessage(payload, window.location.origin);
  return true;
}
function applyToImageGen() {
  const {w,h} = getRes();
  if (!postToHost({
    type: 'imagegen-region-builder-apply',
    target: imageGenTarget,
    prompt: promptWithEmbeddedCanvas(),
    width: w,
    height: h,
    pixel_coordinates: Boolean(document.getElementById('pixelMode').checked),
  })) {
    toast('IMAGE_GEN window is not available');
    return;
  }
  toast('Applied to IMAGE_GEN');
}

function applyBuilderDimensions(width, height, { rebuildPrompt = true } = {}) {
  const nextWidth = Number(width);
  const nextHeight = Number(height);
  if (Number.isFinite(nextWidth)) document.getElementById('resW').value = String(Math.max(64, Math.round(nextWidth)));
  if (Number.isFinite(nextHeight)) document.getElementById('resH').value = String(Math.max(64, Math.round(nextHeight)));
  onResChange();
  if (rebuildPrompt) rebuild();
}
