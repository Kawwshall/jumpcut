# Single image, two roles (api / worker) selected at runtime via CMD override
# in fly.toml process groups — see fly.toml.
FROM python:3.12-slim-bookworm

# ffmpeg from Debian's repo is a full build (libass, libx264, etc. included),
# unlike Homebrew's minimal default formula. jumpcut's ffbin.py already falls
# back gracefully if a particular build is missing a filter, so this is safe
# either way — but Debian's build gives us libass (fast, single-pass burned
# captions) out of the box.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: fast, reproducible installs from the same lockfile used locally.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# Bake the default whisper model into the image so the first real request
# doesn't pay a cold Hugging Face download (and doesn't fail if the worker
# has no outbound network at request time). Bump/duplicate this line for
# other model sizes you plan to offer.
RUN uv run python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

ENV JUMPCUT_HOME=/data
ENV JUMPCUT_HOST=0.0.0.0
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8756

# Default: run the API. The worker process group in fly.toml overrides this
# with its own command (see fly.toml).
CMD ["uv", "run", "jumpcut-web"]
