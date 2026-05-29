const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const resultEl = document.getElementById('result');

// Offscreen canvas drawn with a THIN pen — this is what the model sees.
// The original desktop app showed a 10px line but fed the model a 4px image
// (main.py); the CNN was trained on thin strokes, so a thick exported line is
// fattened by preprocessing and misclassified. We mirror that: display thick,
// classify thin. Same 800x256 coordinate space as the visible canvas.
const MODEL_PEN = 4;
const MODEL_ERASER = 12;
const model = document.createElement('canvas');
model.width = canvas.width;
model.height = canvas.height;
const mctx = model.getContext('2d');

// White background so exported PNG matches the model's expectations.
function clearBoard() {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  mctx.fillStyle = '#fff';
  mctx.fillRect(0, 0, model.width, model.height);
  resultEl.textContent = '';
  resultEl.className = 'result';
}
clearBoard();

let drawing = false;
let mode = 'pen';
let lastX = 0, lastY = 0;

function setMode(next) {
  mode = next;
  document.getElementById('pen').classList.toggle('active', next === 'pen');
  document.getElementById('eraser').classList.toggle('active', next === 'eraser');
}

function pos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = e.touches ? e.touches[0] : e;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

function start(e) {
  drawing = true;
  const p = pos(e);
  lastX = p.x; lastY = p.y;
  e.preventDefault();
}

function strokeOn(context, width) {
  context.strokeStyle = mode === 'pen' ? '#000' : '#fff';
  context.lineWidth = width;
  context.lineCap = 'round';
  context.beginPath();
  context.moveTo(lastX, lastY);
  context.lineTo(p.x, p.y);
  context.stroke();
}

let p;
function move(e) {
  if (!drawing) return;
  p = pos(e);
  strokeOn(ctx, mode === 'pen' ? 10 : 30);            // visible: thick, easy to draw
  strokeOn(mctx, mode === 'pen' ? MODEL_PEN : MODEL_ERASER);  // model: thin
  lastX = p.x; lastY = p.y;
  e.preventDefault();
}

function end() { drawing = false; }

canvas.addEventListener('mousedown', start);
canvas.addEventListener('mousemove', move);
canvas.addEventListener('mouseup', end);
canvas.addEventListener('mouseleave', end);
canvas.addEventListener('touchstart', start);
canvas.addEventListener('touchmove', move);
canvas.addEventListener('touchend', end);

document.getElementById('pen').addEventListener('click', () => setMode('pen'));
document.getElementById('eraser').addEventListener('click', () => setMode('eraser'));
document.getElementById('clear').addEventListener('click', clearBoard);

document.getElementById('evaluate').addEventListener('click', async () => {
  resultEl.textContent = 'Evaluating…';
  resultEl.className = 'result';
  try {
    const image = model.toDataURL('image/png');  // thin-stroke version
    const resp = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
    const body = await resp.json();
    if (!resp.ok) {
      resultEl.textContent = body.error || 'Something went wrong.';
      resultEl.className = 'result error';
      return;
    }
    resultEl.textContent = `${body.expression} = ${body.result}`;
  } catch (err) {
    resultEl.textContent = 'Network error.';
    resultEl.className = 'result error';
  }
});
