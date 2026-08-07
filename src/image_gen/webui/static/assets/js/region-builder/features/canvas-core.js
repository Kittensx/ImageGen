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
  updateInteractionButtons();
  if (paintMode) {
    if (canvasImgData) pushUndo();
    fillCanvasFromRects();
    if (regions.length) selectRegion(Math.max(0, sel));
  } else {
    rebuild();
  }
  if (changed && toastMessage) {
    const label = interactionMode === 'select' ? 'Mouse mode' : interactionMode === 'pan' ? 'Pan mode' : 'Paint mode';
    toast(`${label} active`);
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

