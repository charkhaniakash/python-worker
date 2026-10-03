FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# System deps: media libs for LiveKit RTC, curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgstreamer1.0-0 \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-libav \
      gstreamer1.0-tools \
      libsndfile1 \
      ffmpeg \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer) — pinned tag for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Cache-friendly install: manifests first, source last.
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
COPY config ./config
RUN uv sync --no-dev

# Pre-download Silero VAD weights so the first call isn't cold. Ignored on failure
# (network sandbox at build time is fine — worker will fetch at prewarm time).
RUN uv run python -c "from livekit.plugins import silero; silero.VAD.load()" || true

# Google TTS credentials: fetched from Secret Manager at runtime.
# Auth via Workload Identity (GKE/Cloud Run) or GOOGLE_APPLICATION_CREDENTIALS.
# No SA keys are baked into this image.

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "worker", "dev"]
