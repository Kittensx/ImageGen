/* Region drag/resize calculations. */
function updateDragFromClient(clientX, clientY) {
  if (!drag) return;
  const wrap = document.getElementById('canvasWrap');
  const dx = (clientX - drag.startX) / wrap.clientWidth;
  const dy = (clientY - drag.startY) / wrap.clientHeight;
  const r = regions[drag.idx];
  if (!r) return;

  if (drag.mode === 'move') {
    const w = drag.x2 - drag.x1, h = drag.y2 - drag.y1;
    let box = {
      x1: Math.max(0, Math.min(1 - w, drag.x1 + dx)),
      x2: 0,
      y1: Math.max(0, Math.min(1 - h, drag.y1 + dy)),
      y2: 0,
    };
    box.x2 = box.x1 + w;
    box.y2 = box.y1 + h;
    box = snapMovedBox(box, drag.idx);
    r.x1 = box.x1; r.x2 = box.x2; r.y1 = box.y1; r.y2 = box.y2;
  } else {
    let {x1,x2,y1,y2} = drag;
    const dir = drag.dir;
    if (dir.includes('e')) x2 = Math.max(Math.min(drag.x2 + dx, 1), x1 + 0.01);
    if (dir.includes('w')) x1 = Math.min(Math.max(drag.x1 + dx, 0), x2 - 0.01);
    if (dir.includes('s')) y2 = Math.max(Math.min(drag.y2 + dy, 1), y1 + 0.01);
    if (dir.includes('n')) y1 = Math.min(Math.max(drag.y1 + dy, 0), y2 - 0.01);
    let box = snapResizedBox({ x1, x2, y1, y2 }, drag.idx, dir);
    r.x1 = box.x1; r.x2 = box.x2; r.y1 = box.y1; r.y2 = box.y2;
  }
  render();
}
function finishDrag() {
  if (!drag) return;
  document.getElementById('canvasWrap').classList.remove('dragging');
  drag = null;
  document.querySelectorAll('.region-block').forEach((element) => clearRegionResizeHover(element));
  render();
  rebuild();
  updateEditor();
}
