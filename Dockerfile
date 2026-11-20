FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --upgrade pip && pip install .

FROM base AS dev
RUN pip install ".[dev]"
COPY tests/ tests/
CMD ["pytest", "-v"]

FROM base AS runtime
COPY . .
ENTRYPOINT ["psi-ecg"]