FROM python:3.7-slim

# OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned ML deps first (best layer caching), then web deps.
# The original requirements.txt was frozen on Windows and pins Windows-only
# packages (pywin32, pywinpty) that don't exist on Linux — strip them here so
# the upstream file stays untouched.
COPY requirements.txt requirements-web.txt ./
RUN grep -ivE '^(pywin32|pywinpty)==' requirements.txt > requirements.linux.txt \
    && pip install --no-cache-dir -r requirements.linux.txt \
    && pip install --no-cache-dir -r requirements-web.txt

# App code + model weights.
COPY inference.py app.py ./
COPY model_final.json model_final.h5 ./
COPY templates/ ./templates/
COPY static/ ./static/

ENV PORT=8000
EXPOSE 8000

# Single worker: the model loads once and TF is memory-heavy.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 120 app:app
