# Container image for the rag-eval-harness REST API (see README.md,
# "REST API / Container / AWS deployment example" section).
FROM python:3.11-slim

WORKDIR /app

# Copy only what's needed to install the package (pyproject.toml references
# README.md via `readme = "README.md"`, so it must be present at build time).
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir ".[api]"

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "harness.api:app", "--host", "0.0.0.0", "--port", "8000"]
