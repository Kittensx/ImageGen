/* Flood-fill paint tool. */
function toggleFill() {
  if (!paintMode) { toast('Enter paint mode first'); return; }
  fillMode = !fillMode;
  if (fillMode) { eraserMode = false; const eb = document.getElementById('eraserBtn'); eb.style.background = '#444'; eb.textContent = 'Eraser'; }
  const btn = document.getElementById('fillBtn');
  btn.style.background = fillMode ? '#2b8a3e' : '#444';
  btn.style.borderColor = fillMode ? '#3baa3e' : '#666';
  btn.textContent = fillMode ? 'Fill ON' : 'Fill';
  if (fillMode) toast('Click canvas to fill with selected region color');
}

function fillCanvasArea(cx, cy) {
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const {w,h} = getRes();
  if (cx < 0 || cx >= w || cy < 0 || cy >= h) return;
  if (sel < 0 || sel >= regions.length) { toast('Select a region first'); fillMode = false; const fb = document.getElementById('fillBtn'); fb.style.background = '#444'; fb.style.borderColor = '#666'; fb.textContent = 'Fill'; return; }
  const fc = hexToRgb(SWATCH[regions[sel].c]);
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const idx = (cy * w + cx) * 4;
  const tR = d[idx], tG = d[idx+1], tB = d[idx+2], tA = d[idx+3];
  const TOL = 8;
  if (tA > 128 && Math.abs(tR - fc.r) < TOL && Math.abs(tG - fc.g) < TOL && Math.abs(tB - fc.b) < TOL) { toast('Already this color'); return; }
  function match(pi) {
    const r = d[pi], g = d[pi+1], b = d[pi+2], a = d[pi+3];
    if (a < 128 && tA < 128) return true;
    if (a < 128 || tA < 128) return false;
    return Math.abs(r - tR) < TOL && Math.abs(g - tG) < TOL && Math.abs(b - tB) < TOL;
  }
  const q = [[cx, cy]], vs = new Set(), px = [];
  while (q.length) {
    const [x, y] = q.shift(), key = y * w + x;
    if (vs.has(key)) continue;
    if (x < 0 || x >= w || y < 0 || y >= h) continue;
    const pi = (y * w + x) * 4;
    if (!match(pi)) continue;
    vs.add(key); px.push([x, y]);
    q.push([x+1, y], [x-1, y], [x, y+1], [x, y-1]);
  }
  if (!px.length) { toast('No matching pixels'); return; }
  px.forEach(([x, y]) => {
    const pi = (y * w + x) * 4;
    d[pi] = fc.r; d[pi+1] = fc.g; d[pi+2] = fc.b; d[pi+3] = 255;
  });
  ctx.putImageData(imgData, 0, 0);
  canvasImgData = ctx.getImageData(0, 0, w, h);
  toast(`Filled ${px.length} pixels`);
}

