/* Region canvas/list/table rendering. */
function render() {
  const wrap = document.getElementById('canvasWrap');
  const vc = document.getElementById('viewContainer');
  const es = document.getElementById('emptyState');
  (vc||wrap).querySelectorAll('.region-block').forEach(e=>e.remove());
  const hasRegions = regions.length > 0;
  es.style.display = hasRegions ? 'none' : 'flex';
  const px = document.getElementById('pixelMode').checked;
  const {w,h} = getRes();
  const mx = px ? w : 1, my = px ? h : 1;

  if (hasRegions) regions.forEach((r,i) => {
    const el = document.createElement('div');
    el.className = `region-block ${r.c}${i===sel?' active':''}`;
    el.style.left = (r.x1*100)+'%';
    el.style.top = (r.y1*100)+'%';
    el.style.width = Math.max(.5, (r.x2-r.x1)*100)+'%';
    el.style.height = Math.max(.5, (r.y2-r.y1)*100)+'%';

    el.innerHTML =
      `<div class="r-label">R${i+1}</div>` +
      `<div class="r-text">${esc(r.text)||'&nbsp;'}</div>` +
      `<div class="r-coords">${fmt(r.x1*mx)},${fmt(r.x2*mx)},${fmt(r.y1*my)},${fmt(r.y2*my)}</div>` +
      `<button class="remove-btn" title="Delete region R${i+1}" aria-label="Delete region R${i+1}" onclick="event.stopPropagation();removeRegion(${i})">✕</button>` +
      `<div class="resize-handle resize-se" data-dir="se" data-idx="${i}"></div>` +
      `<div class="resize-handle resize-sw" data-dir="sw" data-idx="${i}"></div>` +
      `<div class="resize-handle resize-ne" data-dir="ne" data-idx="${i}"></div>` +
      `<div class="resize-handle resize-nw" data-dir="nw" data-idx="${i}"></div>`;

    el.onclick = (e) => { if (!e.target.closest('.remove-btn,.resize-handle')) selectRegion(i); };

    el.ondblclick = (e) => {
      if (paintMode) { selectRegion(i); toast(`Painting R${i+1} (${r.text||'empty'})`); }
    };

    const startRegionMove = (e) => {
      if (interactionMode !== 'select' || paintMode || (e.button ?? 0) !== 0 || e.target.closest('.remove-btn,.resize-handle')) return;
      e.preventDefault();
      e.stopPropagation();
      selectRegion(i);
      drag = { mode:'move', idx:i, startX:e.clientX, startY:e.clientY, x1:r.x1, x2:r.x2, y1:r.y1, y2:r.y2 };
      wrap.classList.add('dragging');
    };
    el.addEventListener('mousedown', startRegionMove);
    el.addEventListener('pointerdown', startRegionMove);

    (vc||wrap).appendChild(el);

    el.querySelectorAll('.resize-handle').forEach(h => {
      const startRegionResize = (e) => {
        if (interactionMode !== 'select' || paintMode || (e.button ?? 0) !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        drag = { mode:'resize', idx:i, dir:h.dataset.dir, startX:e.clientX, startY:e.clientY, x1:r.x1, x2:r.x2, y1:r.y1, y2:r.y2 };
      };
      h.addEventListener('mousedown', startRegionResize);
      h.addEventListener('pointerdown', startRegionResize);
    });
  });

  buildPresets();
  rebuildTable();
  renderList();

  // Freehand: click empty canvas → create a region at the click position.
  wrap.onclick = (e) => {
    if (paintMode || interactionMode !== 'select') return;
    if (e.target.closest('.region-block,.resize-handle,.remove-btn')) return;
    const rect = wrap.getBoundingClientRect();
    const cx = (e.clientX - rect.left) / rect.width;
    const cy = (e.clientY - rect.top) / rect.height;
    const hs = 0.15;
    addRegion('', Math.max(0, cx-hs), Math.min(1, cx+hs), Math.max(0, cy-hs), Math.min(1, cy+hs), 1);
    rebuild();
    updateEditor();
  };
}

function renderList() {
  const list = document.getElementById('regionList');
  document.getElementById('regionCount').textContent = regions.length;
  if (!regions.length) { list.innerHTML = '<div class="region-list-empty">No regions yet</div>'; return; }
  list.innerHTML = regions.map((r,i) => `
    <div class="region-list-item${i===sel?' active':''}" onclick="selectRegion(${i})">
      <span class="swatch" style="background:${SWATCH[r.c]}"></span>
      <span class="rl-num">R${i+1}</span>
      <span class="rl-text">${esc(r.text)||'(empty)'}</span>
      <button class="rl-del" title="Delete region R${i+1}" aria-label="Delete region R${i+1}" onclick="event.stopPropagation();removeRegion(${i})">✕</button>
    </div>
  `).join('');
}
function rebuildTable() {
  const tbody = document.getElementById('coordBody');
  if (!tbody) return;
  const px = document.getElementById('pixelMode').checked;
  const {w,h} = getRes();
  const mx = px ? w : 1, my = px ? h : 1;
  tbody.innerHTML = regions.map((r,i) => {
    const rowClass = i===sel ? 'coordinate-row is-active' : 'coordinate-row';
    return `<tr class="${rowClass}">
      <td class="coordinate-cell coordinate-cell-number">R${i+1}</td>
      <td class="coordinate-cell"><input class="coordinate-table-input" value="${esc(r.text)}" onchange="onTableCell(${i},'text',this.value)"></td>
      <td class="coordinate-cell"><input class="coordinate-table-input is-number" type="number" value="${fmt(r.x1*mx)}" step="any" onchange="onTableCell(${i},'x1',this.value,true)"></td>
      <td class="coordinate-cell"><input class="coordinate-table-input is-number" type="number" value="${fmt(r.x2*mx)}" step="any" onchange="onTableCell(${i},'x2',this.value,true)"></td>
      <td class="coordinate-cell"><input class="coordinate-table-input is-number" type="number" value="${fmt(r.y1*my)}" step="any" onchange="onTableCell(${i},'y1',this.value,true)"></td>
      <td class="coordinate-cell"><input class="coordinate-table-input is-number" type="number" value="${fmt(r.y2*my)}" step="any" onchange="onTableCell(${i},'y2',this.value,true)"></td>
      <td class="coordinate-cell"><input class="coordinate-table-input is-number" type="number" value="${fmtWeight(r.w)}" step="0.05" min="0" max="3" onchange="onTableCell(${i},'w',this.value)"></td>
      <td class="coordinate-cell"><button class="coordinate-table-delete" onclick="removeRegion(${i});renderTable();rebuild();updateEditor()">✕</button></td>
    </tr>`;
  }).join('');
}

function onTableCell(i, field, raw, divide) {
  if (i<0||!regions[i]) return;
  const px = document.getElementById('pixelMode').checked;
  const {w,h} = getRes();
  const mx = px ? w : 1, my = px ? h : 1;
  let v = parseFloat(raw);
  if (isNaN(v)) v = 0;
  if (divide) v /= (field[0]==='x' ? mx : my);
  if (field==='w') { if (v<0) v=0; if (v>3) v=3; }
  regions[i][field] = v;
  render(); rebuild(); updateEditor();
}

function renderTable() { rebuildTable(); }
