/* Region collection, selection, keyboard movement, and clearing. */
function nextDefaultRect(k) {
  const w = 0.35, h = 0.35, step = 0.12;
  const maxCols = Math.max(1, Math.floor((1 - w) / step) + 1);
  const maxRows = Math.max(1, Math.floor((1 - h) / step) + 1);
  const col = k % maxCols;
  const row = Math.floor(k / maxCols) % maxRows;
  const x1 = Math.min(1 - w, col * step);
  const y1 = Math.min(1 - h, row * step);
  return { x1, x2: x1 + w, y1, y2: y1 + h };
}

function addRegion(text,x1,x2,y1,y2,w) {
  if (x1==null) {
    const L = regions.length;
    if (L===0) { x1=0; x2=1; y1=0; y2=1; }
    else if (L===1) { x1=0; x2=1; y1=0.5; y2=1; regions[0].y2=0.5; }
    else {
      const box = nextDefaultRect(L-2);
      x1=box.x1; x2=box.x2; y1=box.y1; y2=box.y2;
    }
  }
  const r = { id:id(), text:text||'', x1:+x1, x2:+x2, y1:+y1, y2:+y2, w:+(w||1), curve:'linear', c:COLORS[colorIdx%COLORS.length] };
  colorIdx++;
  regions.push(r);
  sel = regions.length-1;
  render();
}

function removeRegion(i) {
  if (i<0||i>=regions.length) return;
  regions.splice(i,1);
  if (sel===i) sel = -1;
  else if (sel>i) sel--;
  if (sel>=regions.length) sel = regions.length-1;
  render();
  rebuild();
  updateEditor();
  updateInteractionButtons();
}

function deleteSelectedRegion() {
  if (sel < 0 || sel >= regions.length) {
    toast('Select a region first');
    return;
  }
  removeRegion(sel);
}

function selectRegion(i) {
  sel = i;
  render();
  updateEditor();
}

function getKeyboardStepPx(evt = {}) {
  const base = Math.max(1, parseInt(document.getElementById('nudgePixels')?.value || '10', 10) || 10);
  if (evt.altKey) return 1;
  if (evt.shiftKey) return base * 10;
  return base;
}

function getStepNormalized(px) {
  const {w,h} = getRes();
  return { sx: Math.max(1, px) / Math.max(1, w), sy: Math.max(1, px) / Math.max(1, h) };
}

function clampRegion(region) {
  const minSize = 0.01;
  const width = Math.max(minSize, region.x2 - region.x1);
  const height = Math.max(minSize, region.y2 - region.y1);
  region.x1 = Math.max(0, Math.min(1 - width, region.x1));
  region.y1 = Math.max(0, Math.min(1 - height, region.y1));
  region.x2 = Math.min(1, Math.max(region.x1 + minSize, region.x1 + width));
  region.y2 = Math.min(1, Math.max(region.y1 + minSize, region.y1 + height));
  if (region.x2 > 1) { const overflow = region.x2 - 1; region.x1 = Math.max(0, region.x1 - overflow); region.x2 = 1; }
  if (region.y2 > 1) { const overflow = region.y2 - 1; region.y1 = Math.max(0, region.y1 - overflow); region.y2 = 1; }
}

function nudgeSelected(direction, evt = {}) {
  if (sel < 0 || sel >= regions.length) { toast('Select a region first'); return; }
  const px = getKeyboardStepPx(evt);
  const { sx, sy } = getStepNormalized(px);
  const r = regions[sel];
  if (direction === 'left') { r.x1 -= sx; r.x2 -= sx; }
  if (direction === 'right') { r.x1 += sx; r.x2 += sx; }
  if (direction === 'up') { r.y1 -= sy; r.y2 -= sy; }
  if (direction === 'down') { r.y1 += sy; r.y2 += sy; }
  clampRegion(r);
  render(); rebuild(); updateEditor();
}

function resizeSelectedByKey(direction, evt = {}) {
  if (sel < 0 || sel >= regions.length) { toast('Select a region first'); return; }
  const px = getKeyboardStepPx(evt);
  const { sx, sy } = getStepNormalized(px);
  const r = regions[sel];
  if (direction === 'left') r.x2 = Math.max(r.x1 + 0.01, r.x2 - sx);
  if (direction === 'right') r.x2 = Math.min(1, r.x2 + sx);
  if (direction === 'up') r.y2 = Math.max(r.y1 + 0.01, r.y2 - sy);
  if (direction === 'down') r.y2 = Math.min(1, r.y2 + sy);
  clampRegion(r);
  render(); rebuild(); updateEditor();
}

function centerSelectedOnCanvas() {
  if (sel < 0 || sel >= regions.length) { toast('Select a region first'); return; }
  const r = regions[sel];
  const w = r.x2 - r.x1;
  const h = r.y2 - r.y1;
  r.x1 = (1 - w) / 2; r.x2 = r.x1 + w;
  r.y1 = (1 - h) / 2; r.y2 = r.y1 + h;
  render(); rebuild(); updateEditor();
}

function selectAdjacentRegion(offset) {
  if (!regions.length) { return; }
  if (sel < 0) { selectRegion(0); return; }
  const next = (sel + offset + regions.length) % regions.length;
  selectRegion(next);
}
function clearAll() {
  if (regions.length&&!confirm('Clear all regions and mask paint?')) return;
  regions=[];
  sel=-1;
  colorIdx=0;
  undoStack=[];
  redoStack=[];
  const canvas = document.getElementById('regionCanvas');
  const ctx = getCanvasCtx();
  if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvasImgData = null;
  document.getElementById('canvasStore').value = '';
  render();
  rebuild();
  updateEditor();
  toast('All regions cleared');
}

function onRawEdit() { /* manual edit — no auto-sync back */ }
