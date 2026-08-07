/* Small DOM/data helpers shared by Region Builder features. */
function id(){return Math.random().toString(36).slice(2,8)}
function hexToRgb(hex) {
  return { r: parseInt(hex.slice(1,3), 16), g: parseInt(hex.slice(3,5), 16), b: parseInt(hex.slice(5,7), 16) };
}

function onOpacityChange() {
  const v = document.getElementById('opacitySlider').value;
  const canvas = document.getElementById('regionCanvas');
  if (canvas) canvas.style.opacity = (v / 100).toString();
}

function getRes() {
  const w = parseInt(document.getElementById('resW').value)||512;
  const h = parseInt(document.getElementById('resH').value)||512;
  return { w, h };
}

function fmt(v){return Number.isInteger(v)?v.toString():parseFloat(v.toFixed(4)).toString()}
function fmtWeight(w){ return parseFloat(w.toFixed(2)).toString(); }
function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent=msg; t.classList.add('show');
  clearTimeout(t._to); t._to=setTimeout(()=>t.classList.remove('show'),1500);
}
