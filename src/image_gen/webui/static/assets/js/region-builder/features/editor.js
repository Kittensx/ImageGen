/* Selected-region editor and global numeric controls. */
function updateEditor() {
  const r = sel>=0&&sel<regions.length ? regions[sel] : null;
  document.getElementById('noSelMsg').style.display = r ? 'none' : 'block';
  document.getElementById('editorFields').style.display = r ? 'flex' : 'none';
  if (!r) return;
  document.getElementById('propText').value = r.text;
  document.getElementById('propWeight').value = r.w.toFixed(2);
  document.getElementById('propWeightSlider').value = r.w;
  setCurveSelect(r.curve||'linear');
  updateCoordInputs(r);
}

// ─── Curve select handling ──────────────────────────────
function setCurveSelect(curve) {
  const selEl = document.getElementById('propCurve');
  const customEl = document.getElementById('propCurveCustom');
  const customMatch = /^cubic\([\d.,\s]+\)$/.test(curve);
  if (CURVE_PRESETS.has(curve)) {
    selEl.value = curve;
    customEl.style.display = 'none';
  } else if (customMatch) {
    selEl.value = '__custom__';
    customEl.value = curve;
    customEl.style.display = 'block';
  } else {
    selEl.value = 'linear';
    customEl.style.display = 'none';
  }
}

function onCurveCustom() {
  if (sel<0||!regions[sel]) return;
  const v = document.getElementById('propCurveCustom').value.trim();
  regions[sel].curve = /^cubic\([\d.,\s]+\)$/.test(v) ? v : 'linear';
  render(); rebuild();
}

function updateCoordInputs(r) {
  const px = document.getElementById('pixelMode').checked;
  const {w,h} = getRes();
  const mx = px ? w : 1, my = px ? h : 1;
  document.getElementById('coordUnitHint').textContent = px ? `(0–${w}×${h})` : '(0–1)';
  const fields = { coordX1:r.x1*mx, coordX2:r.x2*mx, coordY1:r.y1*my, coordY2:r.y2*my };
  Object.entries(fields).forEach(([elId,val]) => {
    const el = document.getElementById(elId);
    el.step = px ? '1' : '0.01';
    el.min = '0'; el.max = px ? (elId.startsWith('coordX') ? w : h) : '1';
    el.value = fmt(val);
  });
}

function onCoordInput() {
  if (sel<0||!regions[sel]) return;
  const px = document.getElementById('pixelMode').checked;
  const {w,h} = getRes();
  const mx = px ? w : 1, my = px ? h : 1;
  const x1 = parseFloat(document.getElementById('coordX1').value);
  const x2 = parseFloat(document.getElementById('coordX2').value);
  const y1 = parseFloat(document.getElementById('coordY1').value);
  const y2 = parseFloat(document.getElementById('coordY2').value);
  if ([x1,x2,y1,y2].some(v=>Number.isNaN(v))) return;
  const r = regions[sel];
  r.x1 = x1/mx; r.x2 = x2/mx; r.y1 = y1/my; r.y2 = y2/my;
  render(); rebuild();
}

function onCoordBlur() {
  if (sel<0||!regions[sel]) return;
  const r = regions[sel];
  if (r.x1 > r.x2) { const t=r.x1; r.x1=r.x2; r.x2=t; }
  if (r.y1 > r.y2) { const t=r.y1; r.y1=r.y2; r.y2=t; }
  r.x1 = Math.max(0, Math.min(1, r.x1));
  r.x2 = Math.max(0, Math.min(1, r.x2));
  r.y1 = Math.max(0, Math.min(1, r.y1));
  r.y2 = Math.max(0, Math.min(1, r.y2));
  render(); rebuild(); updateEditor();
}

function onEdit() {
  if (sel<0||!regions[sel]) return;
  regions[sel].text = document.getElementById('propText').value;
  const selEl = document.getElementById('propCurve');
  const customEl = document.getElementById('propCurveCustom');
  if (selEl.value === '__custom__') {
    const v = customEl.value.trim();
    regions[sel].curve = /^cubic\([\d.,\s]+\)$/.test(v) ? v : 'linear';
    customEl.style.display = 'block';
  } else {
    regions[sel].curve = selEl.value;
    customEl.style.display = 'none';
  }
  render(); rebuild();
}

function setPos(x1,x2,y1,y2) {
  if (sel<0||!regions[sel]) return;
  regions[sel].x1=x1; regions[sel].x2=x2; regions[sel].y1=y1; regions[sel].y2=y2;
  render(); rebuild(); updateEditor();
}

function onWeightSlider() {
  if (sel<0||!regions[sel]) return;
  const v = parseFloat(document.getElementById('propWeightSlider').value);
  regions[sel].w = v;
  document.getElementById('propWeight').value = v.toFixed(2);
  rebuild();
}

function onWeightText() {
  if (sel<0||!regions[sel]) return;
  const parsed = parseFloat(document.getElementById('propWeight').value);
  const v = Number.isNaN(parsed) ? 1 : Math.max(0, parsed);
  regions[sel].w = v;
  document.getElementById('propWeightSlider').value = Math.min(3, v);
  rebuild();
}



function onStartSlider() {
  const v = parseFloat(document.getElementById('startSlider').value);
  document.getElementById('startValue').value = v.toFixed(2);
  rebuild();
}

function onStartText() {
  const parsed = parseFloat(document.getElementById('startValue').value);
  const v = Number.isNaN(parsed) ? 0 : Math.max(0, Math.min(1, parsed));
  document.getElementById('startSlider').value = v;
  document.getElementById('startValue').value = v.toFixed(2);
  rebuild();
}

function onStopSlider() {
  const v = parseFloat(document.getElementById('stopSlider').value);
  document.getElementById('stopValue').value = v.toFixed(2);
  rebuild();
}

function onStopText() {
  const parsed = parseFloat(document.getElementById('stopValue').value);
  const v = Number.isNaN(parsed) ? 1 : Math.max(0, Math.min(1, parsed));
  document.getElementById('stopSlider').value = v;
  document.getElementById('stopValue').value = v.toFixed(2);
  rebuild();
}

function onBlurSlider() {
  const v = parseFloat(document.getElementById('blurSlider').value);
  document.getElementById('blurValue').value = v.toFixed(2);
  rebuild();
}

function onBlurText() {
  const parsed = parseFloat(document.getElementById('blurValue').value);
  const v = Number.isNaN(parsed) ? 0 : Math.max(0, Math.min(1, parsed));
  document.getElementById('blurSlider').value = v;
  document.getElementById('blurValue').value = v.toFixed(2);
  rebuild();
}

function onBaseRatioSlider() {
  const v = parseFloat(document.getElementById('baseRatioSlider').value);
  document.getElementById('baseRatioValue').value = v.toFixed(2);
  rebuild();
}

function onBaseRatioText() {
  const parsed = parseFloat(document.getElementById('baseRatioValue').value);
  const v = Number.isNaN(parsed) ? 0.2 : Math.max(0, Math.min(1, parsed));
  document.getElementById('baseRatioSlider').value = v;
  document.getElementById('baseRatioValue').value = v.toFixed(2);
  rebuild();
}
function extendSelectedEdge(edge) {
  if (sel < 0 || sel >= regions.length) return;
  const region = regions[sel];
  if (edge === 'left') region.x1 = 0;
  if (edge === 'right') region.x2 = 1;
  if (edge === 'top') region.y1 = 0;
  if (edge === 'bottom') region.y2 = 1;
  render();
  rebuild();
  updateEditor();
}
