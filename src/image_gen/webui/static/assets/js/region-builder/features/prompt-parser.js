/* REGION text parsing helpers. */
function extractRegionBlock(text) {
  const start = text.indexOf('REGION{');
  if (start===-1) return null;
  let i = start + 'REGION{'.length, depth = 1;
  const bodyStart = i;
  for (; i<text.length; i++) {
    if (text[i]==='{') depth++;
    else if (text[i]==='}') { depth--; if (depth===0) break; }
  }
  if (depth!==0) return null;
  let body = text.slice(bodyStart, i);
  let axis = 'H', ratios = null, gridH = null, gridV = null;
  let afterBody = text.slice(i+1);
  // Check for grid suffix [H:...|V:...]
  const gridMatch = afterBody.match(/^\s*\[(?:H:([^\|\]]*)(?:\s*\|\s*V:([^\]]*))?|V:([^\]]*))\s*\]/i);
  if (gridMatch) {
    const rawH = gridMatch[1], rawV = gridMatch[2] || gridMatch[3];
    if (rawH && rawH.trim()) {
      gridH = rawH.split(',').map(v => parseFloat(v.trim())).filter(v => isFinite(v) && v > 0);
      if (!gridH.length) gridH = null;
    }
    if (rawV && rawV.trim()) {
      gridV = rawV.split(',').map(v => parseFloat(v.trim())).filter(v => isFinite(v) && v > 0);
      if (!gridV.length) gridV = null;
    }
    // Default missing dimension
    if (!gridH && gridV) gridH = [1];
    if (!gridV && gridH) gridV = [1];
  } else {
    const ratioMatch = afterBody.match(/^\s*:([HV])\s*:\s*([\d.,\s]+)/i);
    if (ratioMatch) {
      axis = ratioMatch[1].toUpperCase();
      ratios = ratioMatch[2].split(',').map(v => parseFloat(v.trim())).filter(v => isFinite(v) && v > 0);
      if (!ratios.length) ratios = null;
    } else {
      const axisMatch = afterBody.match(/^\s*:([HV])/i);
      if (axisMatch) axis = axisMatch[1].toUpperCase();
    }
  }
  return { body, axis, ratios, gridH, gridV };
}
function splitPipes(body) {
  const parts = []; let buf='', d=0;
  for (const ch of body) {
    if (ch==='|'&&d===0) { parts.push(buf); buf=''; }
    else { if ('{(['.includes(ch)) d++; if('}])'.includes(ch)) d--; buf+=ch; }
  }
  if (buf.trim()) parts.push(buf);
  return parts;
}
