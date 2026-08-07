/* Paint undo/redo history and canvas serialization. */
function pushUndo() {
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const {w,h} = getRes();
  redoStack = [];
  undoStack.push(ctx.getImageData(0, 0, w, h));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
}

function undoPaint() {
  if (!undoStack.length || !paintMode) return;
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const {w,h} = getRes();
  const prev = undoStack[undoStack.length - 1];
  if (prev.width !== w || prev.height !== h) { undoStack = []; redoStack = []; toast('Resolution changed, undo cleared'); return; }
  redoStack.push(ctx.getImageData(0, 0, w, h));
  undoStack.pop();
  ctx.putImageData(prev, 0, 0);
  canvasImgData = ctx.getImageData(0, 0, w, h);
}

function redoPaint() {
  if (!redoStack.length || !paintMode) return;
  const ctx = getCanvasCtx();
  if (!ctx) return;
  const {w,h} = getRes();
  const next = redoStack[redoStack.length - 1];
  if (next.width !== w || next.height !== h) { undoStack = []; redoStack = []; toast('Resolution changed, redo cleared'); return; }
  undoStack.push(ctx.getImageData(0, 0, w, h));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack.pop();
  ctx.putImageData(next, 0, 0);
  canvasImgData = ctx.getImageData(0, 0, w, h);
}

function serializeCanvas() {
  const canvas = document.getElementById('regionCanvas');
  if (!canvas || !canvasImgData) return '';
  const ctx = getCanvasCtx();
  if (!ctx) return '';
  // restore from imgData (in case canvas was cleared)
  ctx.putImageData(canvasImgData, 0, 0);
  return canvas.toDataURL('image/png');
}
