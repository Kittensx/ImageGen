/* Canvas guides and snap calculations. */
function renderGuides() {
  const overlay = document.getElementById('guideOverlay');
  if (!overlay) return;
  const showGuides = Boolean(document.getElementById('showGuides')?.checked);
  const cols = Math.max(1, parseInt(document.getElementById('gridCols')?.value || '1', 10));
  const rows = Math.max(1, parseInt(document.getElementById('gridRows')?.value || '1', 10));
  overlay.classList.toggle('visible', showGuides);
  if (!showGuides) {
    overlay.style.backgroundImage = 'none';
    return;
  }
  overlay.style.backgroundImage = 'linear-gradient(to right, rgba(110,150,255,.16) 1px, transparent 1px), linear-gradient(to bottom, rgba(110,150,255,.16) 1px, transparent 1px)';
  overlay.style.backgroundSize = `${100 / cols}% 100%, 100% ${100 / rows}%`;
}

function getSnapConfig() {
  return {
    snapGrid: Boolean(document.getElementById('snapGrid')?.checked),
    snapEdge: Boolean(document.getElementById('snapEdge')?.checked),
  };
}

function getGridLines(axis) {
  const count = Math.max(1, parseInt(document.getElementById(axis === 'x' ? 'gridCols' : 'gridRows')?.value || '1', 10));
  const lines = [0, 1];
  for (let i = 1; i < count; i += 1) lines.push(i / count);
  return lines;
}

function snapThreshold(axis) {
  const wrap = document.getElementById('canvasWrap');
  const size = axis === 'x' ? Math.max(1, wrap?.clientWidth || 1) : Math.max(1, wrap?.clientHeight || 1);
  return 12 / size;
}

function findClosestSnapDelta(current, candidates, threshold) {
  let best = null;
  for (const candidate of candidates) {
    const delta = candidate - current;
    const abs = Math.abs(delta);
    if (abs <= threshold && (!best || abs < best.abs)) best = { delta, abs };
  }
  return best ? best.delta : null;
}

function getSnapCandidates(activeIdx, axis, edgeRole) {
  const cfg = getSnapConfig();
  const values = [];
  if (cfg.snapGrid) values.push(...getGridLines(axis));
  if (cfg.snapEdge) {
    values.push(0, 1);
    regions.forEach((region, idx) => {
      if (idx === activeIdx) return;
      if (axis === 'x') {
        if (edgeRole === 'start') {
          values.push(region.x1, region.x2);
        } else {
          values.push(region.x1, region.x2);
        }
      } else {
        if (edgeRole === 'start') {
          values.push(region.y1, region.y2);
        } else {
          values.push(region.y1, region.y2);
        }
      }
    });
  }
  return values;
}

function snapMovedBox(box, activeIdx) {
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;
  const leftDelta = findClosestSnapDelta(box.x1, getSnapCandidates(activeIdx, 'x', 'start'), snapThreshold('x'));
  const rightDelta = findClosestSnapDelta(box.x2, getSnapCandidates(activeIdx, 'x', 'end'), snapThreshold('x'));
  const dx = leftDelta != null && (rightDelta == null || Math.abs(leftDelta) <= Math.abs(rightDelta)) ? leftDelta : rightDelta;
  if (dx != null) {
    box.x1 = Math.max(0, Math.min(1 - width, box.x1 + dx));
    box.x2 = box.x1 + width;
  }
  const topDelta = findClosestSnapDelta(box.y1, getSnapCandidates(activeIdx, 'y', 'start'), snapThreshold('y'));
  const bottomDelta = findClosestSnapDelta(box.y2, getSnapCandidates(activeIdx, 'y', 'end'), snapThreshold('y'));
  const dy = topDelta != null && (bottomDelta == null || Math.abs(topDelta) <= Math.abs(bottomDelta)) ? topDelta : bottomDelta;
  if (dy != null) {
    box.y1 = Math.max(0, Math.min(1 - height, box.y1 + dy));
    box.y2 = box.y1 + height;
  }
  return box;
}

function snapResizedBox(box, activeIdx, dir) {
  if (dir.includes('e')) {
    const delta = findClosestSnapDelta(box.x2, getSnapCandidates(activeIdx, 'x', 'end'), snapThreshold('x'));
    if (delta != null) box.x2 = Math.max(box.x1 + 0.01, Math.min(1, box.x2 + delta));
  }
  if (dir.includes('w')) {
    const delta = findClosestSnapDelta(box.x1, getSnapCandidates(activeIdx, 'x', 'start'), snapThreshold('x'));
    if (delta != null) box.x1 = Math.max(0, Math.min(box.x2 - 0.01, box.x1 + delta));
  }
  if (dir.includes('s')) {
    const delta = findClosestSnapDelta(box.y2, getSnapCandidates(activeIdx, 'y', 'end'), snapThreshold('y'));
    if (delta != null) box.y2 = Math.max(box.y1 + 0.01, Math.min(1, box.y2 + delta));
  }
  if (dir.includes('n')) {
    const delta = findClosestSnapDelta(box.y1, getSnapCandidates(activeIdx, 'y', 'start'), snapThreshold('y'));
    if (delta != null) box.y1 = Math.max(0, Math.min(box.y2 - 0.01, box.y1 + delta));
  }
  return box;
}
