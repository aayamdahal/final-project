# Handwritten Expression Evaluator

> **Live demo:** https://archpc.tail8688ce.ts.net/

![screenshot placeholder](docs/screenshot.png)

Draw a handwritten math expression (digits and `+ - *`) on a canvas; a Keras CNN
recognizes the symbols and the app evaluates the result (e.g. `4+3 = 7`).

## Run locally (Docker)

```bash
cp .env.example .env        # adjust PORT if 8000 is taken
docker compose up --build
# open http://localhost:8000  (or your PORT)
```

The container pins Python 3.7 + TensorFlow 2.1 / Keras 2.3 to match the trained
model. The first build is slow (TensorFlow download); subsequent builds are cached.

## Run the desktop version (original)

```bash
pip install -r requirements.txt   # Windows; uses Tkinter
python main.py
```

## How it works

- `static/app.js` captures the drawing and POSTs a PNG to `/api/evaluate`.
- `app.py` (Flask) decodes the image and calls `inference.py`.
- `inference.py` segments characters with OpenCV contours, classifies each with
  the CNN (`model_final.json` / `model_final.h5`), maps classes to `0-9 + - *`,
  and evaluates the expression.

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Flask server: serves the page, `POST /api/evaluate` |
| `inference.py` | Recognition core (lazy-loads the model) |
| `templates/`, `static/` | Canvas UI |
| `Dockerfile`, `docker-compose.yml` | Containerized runtime |
| `main.py`, `utils.py` | Original desktop (Tkinter) app |
| `train.py` | Model training script |

## Tests

```bash
python -m pytest tests/ -v
```

(`tests/test_inference_mapping.py` runs anywhere; full inference is exercised by
the in-container smoke test.)
