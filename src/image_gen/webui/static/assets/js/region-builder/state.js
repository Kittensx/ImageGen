/* Region Builder shared runtime state. Loaded first by region-builder-bootstrap.js. */
let regions = [];
let sel = -1;
let colorIdx = 0;
const COLORS = ['c0','c1','c2','c3','c4','c5','c6','c7'];
const SWATCH = { c0:'#3b5bdb', c1:'#2baa3e', c2:'#e63c3c', c3:'#dcaa32', c4:'#a050dc', c5:'#28bed2', c6:'#d26e32', c7:'#6ec86e' };
const PRESETS = [
  ['2→',1,2,'H'],['2↓',2,1,'H'],['3→',1,3,'H'],['3↓',3,1,'H'],
  ['4→',1,4,'H'],['4↓',4,1,'H'],['2×2',2,2,'G'],['3×2',3,2,'G'],
];
let drag = null; // { mode:'move'|'resize', idx, dir, startX, startY, x1, x2, y1, y2 }
let interactionMode = 'select';
let paintMode = false;
let eraserMode = false;
let isDrawing = false;
let canvasCtx = null;
let canvasImgData = null;
let canvasMaskDirty = false;
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 50;
let fillMode = false;
let viewZoom = 1.0, viewPanX = 0, viewPanY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
const CURVE_PRESETS = new Set(['linear','ease','ease-in','ease-out','ease-in-out','bezier','catmull','sine-in','sine-out','sine-in-out','quart-in','quart-out','quart-in-out','quint-in','quint-out','quint-in-out','expo-in','expo-out','expo-in-out','circ-in','circ-out','circ-in-out','back-in','back-out','back-in-out','bounce','cubic(0.25,0.1,0.75,0.9)']);
let imageGenTarget = 'positive';
