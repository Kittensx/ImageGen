/* REGION prompt serialization and compact-grid detection. */
function rebuild() {
  const base = document.getElementById('baseText').value.trim();
  const mode = document.getElementById('modeSelect').value;
  const backend = document.getElementById('backendSelect').value;
  const start = document.getElementById('startSlider').value;
  const stop = document.getElementById('stopSlider').value;
  const blur = document.getElementById('blurSlider').value;
  const baseRatio = document.getElementById('baseRatioSlider').value;
  const px = document.getElementById('pixelMode').checked;
  const compact = document.getElementById('compactCheck').checked;
  const {w,h} = getRes();
  const directives = [];
  const branches = [];

  if (!regions.length) {
    document.getElementById('canvasStore').value = '';
    document.getElementById('promptText').value = '';
    return;
  }

  if (base) directives.push('*base='+base);
  if (mode!=='overlay') directives.push('mode='+mode);
  if (backend) directives.push('backend='+backend);
  if (+start>0) directives.push('start='+fmtWeight(+start));
  if (+stop<1) directives.push('stop='+fmtWeight(+stop));
  if (parseFloat(blur) > 0) directives.push('blur='+blur);
  if (parseFloat(baseRatio) !== 0.2) directives.push('base_ratio='+fmtWeight(parseFloat(baseRatio)));
  const _c = serializeCanvas();
  document.getElementById('canvasStore').value = _c || '';
  if (_c) directives.push('canvas=1');

  // Compact output: use [H:...|V:...] when regions form a clean grid
  const grid = compact ? detectGridCoords(regions) : null;
  if (grid) {
    regions.forEach(r => {
      let b = r.text || ' ';
      const tail = [];
      if (r.w!==1) tail.push(fmtWeight(r.w));
      if (r.curve && r.curve!=='linear') tail.push(r.curve);
      if (tail.length) b += '*' + tail.join('~');
      branches.push(b);
    });
    const suffix = '[H:' + grid.h.join(',') + ' | V:' + grid.v.join(',') + ']';
    document.getElementById('promptText').value = 'REGION{'+directives.concat(branches).join(' | ')+'}' + suffix;
  } else {
    regions.forEach(r => {
      const fx = v => px ? Math.round(v*w).toString() : fmt(v);
      const fy = v => px ? Math.round(v*h).toString() : fmt(v);
      let b = (r.text||' ') + '@' + fx(r.x1) + ',' + fx(r.x2) + ',' + fy(r.y1) + ',' + fy(r.y2);
      const tail = [];
      if (r.w!==1) tail.push(fmtWeight(r.w));
      if (r.curve && r.curve!=='linear') tail.push(r.curve);
      if (tail.length) b += '*' + tail.join('~');
      branches.push(b);
    });
    document.getElementById('promptText').value = 'REGION{'+directives.concat(branches).join(' | ')+'}';
  }
}

// ─── Compact grid detection ──────────────────────────────
function detectGridCoords(regs) {
  if (!regs || regs.length < 2) return null;
  const eps = 1e-4;
  const rx = v => Math.round(v / eps) * eps;
  let xs = new Set(), ys = new Set();
  regs.forEach(r => { xs.add(rx(r.x1)); xs.add(rx(r.x2)); ys.add(rx(r.y1)); ys.add(rx(r.y2)); });
  if (!xs.has(0) || !xs.has(1) || !ys.has(0) || !ys.has(1)) return null;
  const xa = [...xs].sort((a,b) => a-b);
  const ya = [...ys].sort((a,b) => a-b);
  const nCols = xa.length - 1, nRows = ya.length - 1;
  if (nCols < 1 || nRows < 1) return null;
  // Complete grid: every cell must be covered by exactly one single-cell region.
  if (nCols * nRows !== regs.length) return null;
  const seen = new Set();
  for (const r of regs) {
    const xi = xa.indexOf(rx(r.x1));
    const xi2 = xa.indexOf(rx(r.x2));
    const yi = ya.indexOf(rx(r.y1));
    const yi2 = ya.indexOf(rx(r.y2));
    // Each region must span exactly one cell (consecutive boundaries).
    if (xi2 !== xi + 1 || yi2 !== yi + 1) return null;
    const k = xi + ':' + yi;
    if (seen.has(k)) return null;  // duplicate cell
    seen.add(k);
  }
  if (seen.size !== nCols * nRows) return null;
  const h = [], v = [];
  for (let i = 0; i < nCols; i++) h.push(+(xa[i+1] - xa[i]).toFixed(4));
  for (let i = 0; i < nRows; i++) v.push(+(ya[i+1] - ya[i]).toFixed(4));
  return { h, v };
}
