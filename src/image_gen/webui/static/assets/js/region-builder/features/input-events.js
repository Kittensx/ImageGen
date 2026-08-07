/* Canvas pointer, pan/zoom, drag, and keyboard bindings. */
function bindInputEvents() {
  document.addEventListener('mousemove', (event) => {
    if (drag) updateDragFromClient(event.clientX, event.clientY);
  });
  document.addEventListener('pointermove', (event) => {
    if (drag) updateDragFromClient(event.clientX, event.clientY);
  });
  document.addEventListener('mouseup', finishDrag);
  document.addEventListener('pointerup', finishDrag);

  const canvas = document.getElementById('regionCanvas');
  if (canvas) {
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);
  }

  const canvasWrap = document.getElementById('canvasWrap');
  const beginPan = (event) => {
    const leftPan = interactionMode === 'pan' && (event.button ?? 0) === 0;
    const rightPan = (event.button ?? 0) === 2;
    if (!leftPan && !rightPan) return;
    isPanning = true;
    panStartX = event.clientX - viewPanX;
    panStartY = event.clientY - viewPanY;
    canvasWrap.classList.add('panning');
    event.preventDefault();
    event.stopPropagation();
  };
  const updatePan = (event) => {
    if (!isPanning) return;
    viewPanX = event.clientX - panStartX;
    viewPanY = event.clientY - panStartY;
    updateViewTransform();
  };
  const endPan = () => {
    if (!isPanning) return;
    isPanning = false;
    canvasWrap?.classList.remove('panning');
  };

  if (canvasWrap) {
    canvasWrap.addEventListener('contextmenu', (event) => event.preventDefault());
    canvasWrap.addEventListener('mousedown', beginPan);
    canvasWrap.addEventListener('pointerdown', beginPan);
    canvasWrap.addEventListener('wheel', (event) => {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
      const nextZoom = Math.max(0.1, Math.min(5, viewZoom * zoomFactor));
      const rect = canvasWrap.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      viewPanX = mouseX - (mouseX - viewPanX) * (nextZoom / viewZoom);
      viewPanY = mouseY - (mouseY - viewPanY) * (nextZoom / viewZoom);
      viewZoom = nextZoom;
      updateViewTransform();
    }, { passive: false });
  }

  document.addEventListener('mousemove', updatePan);
  document.addEventListener('pointermove', updatePan);
  document.addEventListener('mouseup', endPan);
  document.addEventListener('pointerup', endPan);
  document.addEventListener('pointercancel', endPan);

  window.addEventListener('keydown', (event) => {
    const target = event.target;
    const typing = target && (
      target.isContentEditable ||
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.tagName === 'SELECT'
    );
    if (typing) return;

    const lower = event.key.toLowerCase();
    if (event.key === 'Delete' && sel >= 0 && sel < regions.length) {
      event.preventDefault();
      deleteSelectedRegion();
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      sel = -1;
      render();
      updateEditor();
      return;
    }
    if (event.key === '[') { event.preventDefault(); selectAdjacentRegion(-1); return; }
    if (event.key === ']') { event.preventDefault(); selectAdjacentRegion(1); return; }
    if (lower === 'm') { event.preventDefault(); setInteractionMode('select'); return; }
    if (lower === 'p') { event.preventDefault(); setInteractionMode('pan'); return; }
    if (lower === 'b') { event.preventDefault(); setInteractionMode('paint'); return; }
    if (event.key === '+' || event.key === '=') { event.preventDefault(); viewZoom = Math.min(5, viewZoom * 1.1); updateViewTransform(); return; }
    if (event.key === '-' || event.key === '_') { event.preventDefault(); viewZoom = Math.max(0.1, viewZoom / 1.1); updateViewTransform(); return; }
    if (event.key.startsWith('Arrow')) {
      event.preventDefault();
      const direction = event.key.replace('Arrow', '').toLowerCase();
      if (event.ctrlKey) resizeSelectedByKey(direction, event);
      else nudgeSelected(direction, event);
      return;
    }
    if (event.ctrlKey && lower === 'z' && !event.shiftKey) { event.preventDefault(); undoPaint(); return; }
    if ((event.ctrlKey && lower === 'y') || (event.ctrlKey && event.shiftKey && lower === 'z')) { event.preventDefault(); redoPaint(); return; }
    if ((event.ctrlKey && event.key === '0') || event.key === '0') { event.preventDefault(); resetView(); }
  });
}
