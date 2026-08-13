/* Resolution changes and application of imported REGION prompts. */
function onResChange() {
  const {w,h} = getRes();
  const wrap = document.getElementById('canvasWrap');
  wrap.style.aspectRatio = `${w}/${h}`;
  const badge = document.getElementById('canvasDimensionBadge');
  if (badge) badge.textContent = `${w} × ${h} usable area`;
  undoStack = []; redoStack = [];
  rebuild(); render(); updateEditor(); renderGuides();
}

function onResPreset() {
  const val = document.getElementById('resPreset').value;
  const [w,h] = val.split(',').map(Number);
  document.getElementById('resW').value = w;
  document.getElementById('resH').value = h;
  onResChange();
}
function loadCanvasFromB64(b64, onLoaded) {
  const canvas = document.getElementById('regionCanvas');
  if (!canvas) { if (onLoaded) onLoaded(); return; }
  const img = new Image();
  img.onload = function() {
    const ctx = canvas.getContext('2d');
    const {w,h} = getRes();
    canvas.width = w; canvas.height = h;
    ctx.drawImage(img, 0, 0, w, h);
    canvasImgData = ctx.getImageData(0, 0, w, h);
    canvasMaskDirty = true;
    if (onLoaded) onLoaded();
  };
  img.onerror = function() {
    toast('Canvas image load failed');
    if (onLoaded) onLoaded();
  };
  img.src = b64;
}

function importPrompt() {
  const text = document.getElementById('promptText').value.trim();
  if (!text) return;
  const extracted = extractRegionBlock(text);
  if (!extracted) { toast('No REGION{...} block found'); return; }
  const { body, axis, ratios, gridH, gridV } = extracted;
  const rawBranches = splitPipes(body);
  let base='', mode='overlay', backend='', start=0, stop=1, blur='0', canvasB64='', baseRatio=0.2, reversedFixed=0;
  let px = document.getElementById('pixelMode').checked;
  const {w:rw,h:rh} = getRes();
  const coordBranches = [];
  const rawCoordBranches = [];
  const bareTexts = [];
  let sawPixelCoords = false;
  let sawNormalizedCoords = false;
  let coordModeAutoFixed = false;

  rawBranches.forEach(br => {
    br = br.trim(); if (!br) return;
    if (br.startsWith('*base=')) { base=br.slice(6).trim(); return; }
    if (br.startsWith('mode=')) { const _m=br.slice(5).trim(); if (_m==='forge'||_m==='monkey'||_m==='attention'||_m==='latent') { backend=_m; } else { mode=_m; } return; }
    if (br.startsWith('backend=')) { backend=br.slice(8).trim(); return; }
    if (br.startsWith('start=')) { start=Math.max(0,Math.min(1,parseFloat(br.slice(6).trim())||0)); return; }
    if (br.startsWith('stop=')) { const _s=parseFloat(br.slice(5).trim()); stop=isNaN(_s)?1:Math.max(0,Math.min(1,_s)); return; }
    if (br.startsWith('blur=')) { const _b=parseFloat(br.slice(5).trim()); blur=(isNaN(_b)?0:Math.max(0,Math.min(1,_b))).toFixed(2); return; }
    if (br.startsWith('base_ratio=')) { const _r=parseFloat(br.slice(11).trim()); baseRatio=isNaN(_r)?0.2:Math.max(0,Math.min(1,_r)); return; }
    if (br.startsWith('canvas=')) { const _cv=br.slice(7).trim(); if(_cv.length>2&&_cv!=='1') canvasB64=_cv; return; }
    // Match: text@x1,x2,y1,y2[*weight[~curve]]
    const bm = br.match(/^(.+?)@([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)(?:\*([-\d.]+(?:~[\w().,-]+)?))?\s*$/);
    if (!bm) { bareTexts.push(br); return; }
    const rawX1=+bm[2], rawX2=+bm[3], rawY1=+bm[4], rawY2=+bm[5];
    let w = 1, curve = 'linear';
    if (bm[6]) {
      const parts = bm[6].split('~');
      w = parts[0] ? +parts[0] : 1;
      curve = parts[1] || 'linear';
    }
    const maxAbsCoord = Math.max(Math.abs(rawX1), Math.abs(rawX2), Math.abs(rawY1), Math.abs(rawY2));
    if (maxAbsCoord > 1.000001) sawPixelCoords = true;
    else sawNormalizedCoords = true;
    rawCoordBranches.push({ text:(bm[1]||'').trim(), rawX1, rawX2, rawY1, rawY2, w, curve });
  });

  if (rawCoordBranches.length) {
    const importedMode = sawPixelCoords && !sawNormalizedCoords
      ? true
      : (!sawPixelCoords && sawNormalizedCoords ? false : px);
    if (px !== importedMode) {
      coordModeAutoFixed = true;
    }
    px = importedMode;
    document.getElementById('pixelMode').checked = px;

    rawCoordBranches.forEach((branch) => {
      let x1 = branch.rawX1;
      let x2 = branch.rawX2;
      let y1 = branch.rawY1;
      let y2 = branch.rawY2;
      if (px) { x1/=rw; x2/=rw; y1/=rh; y2/=rh; }
      if (x1>x2) { const t=x1; x1=x2; x2=t; reversedFixed++; }
      if (y1>y2) { const t=y1; y1=y2; y2=t; reversedFixed++; }
      coordBranches.push({ text:branch.text, x1, x2, y1, y2, w:branch.w, curve:branch.curve });
    });
  }

  // Helper: parse "text*weight~curve" from bare text
  function parseBareText(t) {
    const m = t.match(/^(.+?)\*([-\d.]+(?:~[\w().,-]+)?)$/);
    if (m) {
      const tailParts = m[2].split('~');
      return { text: m[1].trim(), w: +tailParts[0] || 1, curve: tailParts[1] || 'linear' };
    }
    return { text: t, w: 1, curve: 'linear' };
  }

  // Grid split (Matrix mode) for [H:...|V:...] suffix
  // Parser semantics (Fix 1, 2026-07-03): H = horizontal splits = columns (x),
  // V = vertical splits = rows (y). Row-major assignment of branch texts.
  if (bareTexts.length && gridH && gridV) {
    const totalH = gridH.reduce((a,b) => a+b, 0);
    const totalV = gridV.reduce((a,b) => a+b, 0);
    const nh = totalH > 0 ? gridH.map(v => v / totalH) : [1];
    const nv = totalV > 0 ? gridV.map(v => v / totalV) : [1];
    let rowPos = 0;
    for (let ri = 0; ri < nv.length; ri++) {
      const rowEnd = rowPos + nv[ri];
      let colPos = 0;
      for (let ci = 0; ci < nh.length; ci++) {
        const colEnd = colPos + nh[ci];
        const idx = ri * nh.length + ci;
        const raw = idx < bareTexts.length ? bareTexts[idx] : bareTexts[bareTexts.length - 1] || '';
        const parsed = parseBareText(raw);
        coordBranches.push({ text:parsed.text, x1:colPos, x2:colEnd, y1:rowPos, y2:rowEnd, w:parsed.w, curve:parsed.curve });
        colPos = colEnd;
      }
      rowPos = rowEnd;
    }
  } else if (bareTexts.length) {
    const n = bareTexts.length;
    if (ratios && ratios.length === n) {
      const total = ratios.reduce((a,b) => a+b, 0);
      const norm = ratios.map(v => v / total);
      let acc = 0;
      bareTexts.forEach((t,i) => {
        const s = acc, e = acc + norm[i];
        acc = e;
        const parsed = parseBareText(t);
        if (axis==='V') coordBranches.push({ text:parsed.text, x1:0, x2:1, y1:s, y2:e, w:parsed.w, curve:parsed.curve });
        else coordBranches.push({ text:parsed.text, x1:s, x2:e, y1:0, y2:1, w:parsed.w, curve:parsed.curve });
      });
    } else {
      const step = 1/n;
      bareTexts.forEach((t,i) => {
        const parsed = parseBareText(t);
        if (axis==='V') coordBranches.push({ text:parsed.text, x1:0, x2:1, y1:i*step, y2:(i+1)*step, w:parsed.w, curve:parsed.curve });
        else coordBranches.push({ text:parsed.text, x1:i*step, x2:(i+1)*step, y1:0, y2:1, w:parsed.w, curve:parsed.curve });
      });
    }
  }

  if (!coordBranches.length) { toast('No regions found'); return; }

  colorIdx = 0;
  regions = coordBranches.map(b => ({ id:id(), text:b.text, x1:b.x1, x2:b.x2, y1:b.y1, y2:b.y2, w:b.w, curve:b.curve||'linear', c:COLORS[colorIdx++%COLORS.length] }));
  document.getElementById('baseText').value = base;
  if (['overlay','common'].includes(mode)) document.getElementById('modeSelect').value = mode;
  if (['forge','monkey','latent'].includes(backend)) document.getElementById('backendSelect').value = backend;
  else if (backend === 'attention') document.getElementById('backendSelect').value = '';  // legacy → auto
  document.getElementById('stopSlider').value = stop;
  document.getElementById('stopValue').value = parseFloat(stop).toFixed(2);
  document.getElementById('blurSlider').value = blur;
  document.getElementById('blurValue').value = parseFloat(blur).toFixed(2);
  document.getElementById('startSlider').value = start;
  document.getElementById('startValue').value = parseFloat(start).toFixed(2);
  document.getElementById('baseRatioSlider').value = baseRatio;
  document.getElementById('baseRatioValue').value = parseFloat(baseRatio).toFixed(2);

  sel = -1;
  // Check for canvas data: 1) embedded B64 in prompt, 2) canvas=1 flag → use hidden store
  const storeEl = document.getElementById('canvasStore');
  if (!canvasB64) {
    const stored = storeEl ? storeEl.value : '';
    if (stored) { canvasB64 = stored; }
    else if (rawBranches.some(b => b.trim() === 'canvas=1')) {
      toast('Canvas data not found — use "Copy w/ Canvas" for portability');
    }
  }
  const hadGrid = gridH && gridV && gridH.length > 0 && gridV.length > 0;
  if (canvasB64) {
    // Store in hidden field for later use
    if (storeEl) storeEl.value = canvasB64;
    // Set up canvas with imported mask data — rebuild deferred to onload
    setupCanvas();
    if (!paintMode) togglePaint();
    loadCanvasFromB64(canvasB64, function() {
      if (hadGrid) document.getElementById('compactCheck').checked = true;
      rebuild();
      updateEditor();
    });
    render();
    toast(reversedFixed ? `Imported with canvas (${reversedFixed} regions reversed → auto-fixed)` : 'Imported with canvas');
    return;
  }
  if (hadGrid) document.getElementById('compactCheck').checked = true;
  render(); rebuild(); updateEditor();
  const importNotes = [];
  if (coordModeAutoFixed) importNotes.push(`auto-switched to ${px ? 'pixel' : 'normalized'} coords`);
  if (reversedFixed) importNotes.push(`fixed ${reversedFixed} reversed range(s)`);
  const summary = `Imported ${regions.length} region${regions.length>1?'s':''}`;
  toast(importNotes.length ? `${summary} (${importNotes.join(', ')})` : summary);
}
