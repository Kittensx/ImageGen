/* Canvas setup, paint mode, brush input, and drawing behavior. */
function getCanvasCtx() {
  const c = document.getElementById('regionCanvas');
  if (!canvasCtx && c) { canvasCtx = c.getContext('2d'); }
  return canvasCtx;
}

function setupCanvas() {
  const c = document.getElementById('regionCanvas');
  if (!c) return;
  const wrap = document.getElementById('canvasWrap');
  const rect = wrap.getBoundingClientRect();
  const {w,h} = getRes();
  c.width = w;
  c.height = h;
  c.style.width = rect.width + 'px';
  c.style.height = rect.height + 'px';
  const ctx = c.getContext('2d');
  canvasCtx = ctx;
  canvasImgData = null;
}

function fillCanvasFromRects() {
  setupCanvas();
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const {w,h} = getRes();
  ctx.clearRect(0, 0, w, h);
  regions.forEach(r => {
    const color = SWATCH[r.c];
    const x1 = Math.round(r.x1 * w);
    const x2 = Math.round(r.x2 * w);
    const y1 = Math.round(r.y1 * h);
    const y2 = Math.round(r.y2 * h);
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1, Math.max(1, x2-x1), Math.max(1, y2-y1));
  });
  canvasImgData = ctx.getImageData(0, 0, w, h);
  canvasMaskDirty = false;
}

function hasCanvasMaskData() {
  return Boolean(canvasImgData && canvasImgData.data && canvasImgData.width > 0 && canvasImgData.height > 0);
}

function restoreCanvasFromImageData() {
  if (!hasCanvasMaskData()) return false;
  const canvas = document.getElementById('regionCanvas');
  const ctx = getCanvasCtx();
  const {w,h} = getRes();
  if (!canvas || !ctx) return false;
  if (canvasImgData.width !== w || canvasImgData.height !== h) return false;
  canvas.width = w;
  canvas.height = h;
  ctx.putImageData(canvasImgData, 0, 0);
  return true;
}

function syncRegionsFromCanvasMask() {
  if (!hasCanvasMaskData() || !regions.length) return { updated: 0, paintedRegions: 0, skippedSharedColorRegions: 0 };
  const {w,h} = getRes();
  const data = canvasImgData.data;
  const colorToBounds = new Map();
  const colorToRegionIndexes = new Map();
  const swatchToKey = {};
  const alphaThreshold = 24;

  regions.forEach((region, index) => {
    const key = String(region.c || '');
    if (!colorToRegionIndexes.has(key)) colorToRegionIndexes.set(key, []);
    colorToRegionIndexes.get(key).push(index);
    swatchToKey[SWATCH[region.c]] = key;
  });

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const pixelIndex = (y * w + x) * 4;
      const alpha = data[pixelIndex + 3];
      if (alpha < alphaThreshold) continue;
      const key = rgbToSwatchKey(data[pixelIndex], data[pixelIndex + 1], data[pixelIndex + 2], swatchToKey);
      if (!key) continue;
      let bounds = colorToBounds.get(key);
      if (!bounds) {
        bounds = { minX: x, maxX: x, minY: y, maxY: y };
        colorToBounds.set(key, bounds);
      } else {
        if (x < bounds.minX) bounds.minX = x;
        if (x > bounds.maxX) bounds.maxX = x;
        if (y < bounds.minY) bounds.minY = y;
        if (y > bounds.maxY) bounds.maxY = y;
      }
    }
  }

  let updated = 0;
  let paintedRegions = 0;
  let skippedSharedColorRegions = 0;
  colorToRegionIndexes.forEach((indexes, key) => {
    const bounds = colorToBounds.get(key);
    if (!bounds) return;
    paintedRegions += indexes.length;
    if (indexes.length !== 1) {
      skippedSharedColorRegions += indexes.length;
      return;
    }
    const region = regions[indexes[0]];
    if (!region) return;
    region.x1 = bounds.minX / w;
    region.x2 = (bounds.maxX + 1) / w;
    region.y1 = bounds.minY / h;
    region.y2 = (bounds.maxY + 1) / h;
    clampRegion(region);
    updated += 1;
  });
  return { updated, paintedRegions, skippedSharedColorRegions };
}

function rgbToSwatchKey(r, g, b, swatchToKey) {
  const tolerance = 36;
  let bestKey = '';
  let bestDistance = Number.POSITIVE_INFINITY;
  Object.entries(swatchToKey).forEach(([hex, key]) => {
    const rgb = hexToRgb(hex);
    const distance = Math.abs(rgb.r - r) + Math.abs(rgb.g - g) + Math.abs(rgb.b - b);
    if (distance < bestDistance && distance <= tolerance) {
      bestDistance = distance;
      bestKey = key;
    }
  });
  return bestKey;
}

function updateInteractionButtons() {
  const selectBtn = document.getElementById('selectModeBtn');
  const panBtn = document.getElementById('panModeBtn');
  const paintBtn = document.getElementById('paintBtn');
  const wrap = document.getElementById('canvasWrap');
  const canvas = document.getElementById('regionCanvas');
  const brushGroup = document.getElementById('brushSizeGroup');
  if (selectBtn) selectBtn.classList.toggle('active', interactionMode === 'select');
  if (panBtn) panBtn.classList.toggle('active', interactionMode === 'pan');
  if (paintBtn) paintBtn.classList.toggle('active', interactionMode === 'paint');
  if (paintBtn) paintBtn.textContent = interactionMode === 'paint' ? '🎨 Paint ON' : '🎨 Paint';
  if (wrap) {
    wrap.classList.toggle('paint-mode', interactionMode === 'paint');
    wrap.classList.toggle('pan-mode', interactionMode === 'pan');
  }
  if (canvas) canvas.classList.toggle('active', interactionMode === 'paint');
  if (brushGroup) brushGroup.style.display = interactionMode === 'paint' ? 'inline-flex' : 'none';
}

function setInteractionMode(mode, opts = {}) {
  const { toastMessage = true } = opts;
  const next = mode === 'pan' ? 'pan' : (mode === 'paint' ? 'paint' : 'select');
  const previousMode = interactionMode;
  const wasPaintMode = paintMode;
  const changed = interactionMode !== next;
  interactionMode = next;
  paintMode = interactionMode === 'paint';
  if (!paintMode && fillMode) {
    fillMode = false;
    const btn = document.getElementById('fillBtn');
    if (btn) { btn.style.background = '#444'; btn.style.borderColor = '#666'; btn.textContent = 'Fill'; }
  }
  if (!paintMode) {
    eraserMode = false;
    const eb = document.getElementById('eraserBtn');
    if (eb) { eb.style.background = '#444'; eb.textContent = 'Eraser'; }
  }
  if (interactionMode !== 'pan') {
    isPanning = false;
    const wrap = document.getElementById('canvasWrap');
    if (wrap) wrap.classList.remove('panning');
  }

  let maskSyncSummary = null;
  if (wasPaintMode && !paintMode && canvasMaskDirty) {
    maskSyncSummary = syncRegionsFromCanvasMask();
  }

  updateInteractionButtons();
  if (paintMode) {
    const restoredMask = canvasMaskDirty && restoreCanvasFromImageData();
    if (!restoredMask) fillCanvasFromRects();
    if (regions.length) selectRegion(Math.max(0, sel));
    else render();
  } else {
    render();
    rebuild();
    updateEditor();
    if (maskSyncSummary && maskSyncSummary.skippedSharedColorRegions > 0 && toastMessage) {
      toast(`Painted mask kept; ${maskSyncSummary.skippedSharedColorRegions} shared-color region${maskSyncSummary.skippedSharedColorRegions === 1 ? '' : 's'} kept their previous bounds`);
    }
  }
  if (changed && toastMessage) {
    const label = interactionMode === 'select' ? 'Mouse mode' : interactionMode === 'pan' ? 'Pan mode' : 'Paint mode';
    toast(previousMode === 'paint' && interactionMode !== 'paint' ? `${label} active · painted mask preserved` : `${label} active`);
  }
}

function togglePaint() {
  setInteractionMode(interactionMode === 'paint' ? 'select' : 'paint');
}

function toggleEraser() {
  eraserMode = !eraserMode;
  if (eraserMode && fillMode) { fillMode = false; const fb = document.getElementById('fillBtn'); fb.style.background = '#444'; fb.textContent = 'Fill'; }
  const btn = document.getElementById('eraserBtn');
  btn.style.background = eraserMode ? '#c92a2a' : '#444';
  btn.textContent = eraserMode ? 'Erase ON' : 'Eraser';
}

function onBrushSize() {
  const v = document.getElementById('brushSize').value;
  document.getElementById('brushSizeLabel').textContent = v;
}

function getCanvasPos(e) {
  const canvas = document.getElementById('regionCanvas');
  const rect = canvas.getBoundingClientRect();
  const {w,h} = getRes();
  return {
    x: Math.round(((e.clientX - rect.left) / rect.width) * w),
    y: Math.round(((e.clientY - rect.top) / rect.height) * h),
  };
}

function startDraw(e) {
  if (!paintMode || e.button !== 0) return;
  if (fillMode) {
    pushUndo();
    const pos = getCanvasPos(e);
    fillCanvasArea(pos.x, pos.y);
    return;
  }
  const ctx = getCanvasCtx();
  if (!ctx) return;
  isDrawing = true;
  pushUndo();
  const pos = getCanvasPos(e);
  const size = parseInt(document.getElementById('brushSize').value);
  const {w,h} = getRes();
  if (eraserMode) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    const color = sel >= 0 && sel < regions.length ? SWATCH[regions[sel].c] : '#3b5bdb';
    ctx.fillStyle = color;
  }
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, size / 2, 0, Math.PI * 2);
  ctx.fill();
  canvasImgData = ctx.getImageData(0, 0, w, h);
  canvasMaskDirty = true;
}

function draw(e) {
  if (!isDrawing || !paintMode || e.button !== 0) return;
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const pos = getCanvasPos(e);
  const size = parseInt(document.getElementById('brushSize').value);
  if (eraserMode) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    const color = sel >= 0 && sel < regions.length ? SWATCH[regions[sel].c] : '#3b5bdb';
    ctx.fillStyle = color;
  }
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, size / 2, 0, Math.PI * 2);
  ctx.fill();
}

function endDraw() {
  isDrawing = false;
  if (!paintMode) return;
  const ctx = getCanvasCtx();
  if (!ctx) return;
  ctx.globalCompositeOperation = 'source-over';
  const {w,h} = getRes();
  canvasImgData = ctx.getImageData(0, 0, w, h);
}

