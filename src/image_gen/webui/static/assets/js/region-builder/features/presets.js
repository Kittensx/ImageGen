/* Region layout presets and custom grid creation. */
function buildPresets() {
  const g = document.getElementById('tilesGrid');
  g.innerHTML = PRESETS.map(([l,r,c,type]) =>
    `<button class="tile-btn" onclick="autoTileMode('${type}',${r},${c})">${l}</button>`
  ).join('');
}

function autoTileMode(type, rows, cols) {
  const texts = [];
  const n = rows * cols;
  for (let i=0;i<n;i++) texts.push(prompt(`Text for region ${i+1}:`)||`R${i+1}`);
  const dw = 1/cols, dh = 1/rows;
  const nr = [];
  for (let r=0;r<rows;r++) for (let c=0;c<cols;c++) {
    nr.push({ id:id(), text:texts[r*cols+c], x1:c*dw, x2:(c+1)*dw, y1:r*dh, y2:(r+1)*dh, w:1, curve:'linear', c:COLORS[colorIdx%COLORS.length] });
    colorIdx++;
  }
  regions = nr; sel = -1; document.getElementById('compactCheck').checked = true; render(); rebuild(); updateEditor();
}

function autoTile(axis,n) {
  const texts = [];
  for (let i=0;i<n;i++) texts.push(prompt(`Text for region ${i+1}:`)||`R${i+1}`);
  const step = 1/n;
  const nr = [];
  for (let i=0;i<n;i++) {
    const r = { id:id(), text:texts[i], w:1, curve:'linear', c:COLORS[colorIdx%COLORS.length] };
    if (axis==='H') { r.x1=i*step; r.x2=(i+1)*step; r.y1=0; r.y2=1; }
    else { r.x1=0; r.x2=1; r.y1=i*step; r.y2=(i+1)*step; }
    nr.push(r); colorIdx++;
  }
  regions = nr; sel = -1; document.getElementById('compactCheck').checked = true; render(); rebuild(); updateEditor();
}

function autoTileMix(r,c) { autoTileMode('G',r,c); }

function applyCustomGrid() {
  const cols = parseInt(document.getElementById('gridCols').value) || 2;
  const rows = parseInt(document.getElementById('gridRows').value) || 2;
  document.getElementById('compactCheck').checked = true; autoTileMode('G', rows, cols);
}
