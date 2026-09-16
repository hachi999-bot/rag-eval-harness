"""REST API wrapper around the fixed RAG evaluation harness.

This module adds no new evaluation logic: it exposes the existing
`harness.runner` functions (load_corpus, load_eval_set,
validate_eval_set_against_corpus, build_retriever, run_eval) over HTTP so the
harness can be run as a service instead of only via the CLI. See the
top-level README for the demonstration/synthetic-data disclaimer that
applies to this repository as a whole.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from harness.runner import (
    DEFAULT_K_VALUES,
    build_retriever,
    load_corpus,
    load_eval_set,
    run_eval,
    validate_eval_set_against_corpus,
)

# Same relative defaults as harness.runner's CLI (`--corpus`, `--eval`):
# resolved against the current working directory, which is the repo root
# both for local `uv run uvicorn ...` and inside the Docker image.
DEFAULT_CORPUS_DIR = os.environ.get("HARNESS_CORPUS_DIR", "data/corpus")
DEFAULT_EVAL_SET_PATH = os.environ.get("HARNESS_EVAL_SET", "data/eval_set.json")

app = FastAPI(
    title="rag-eval-harness API",
    description=(
        "REST API wrapper around the fixed RAG evaluation harness "
        "(demonstration repository, synthetic data only)."
    ),
    version="0.1.0",
)


class EvaluateRequest(BaseModel):
    mode: str = Field(default="tfidf", description="Retriever mode: naive, tfidf, or openai")
    k: list[int] = Field(default_factory=lambda: list(DEFAULT_K_VALUES))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    corpus = load_corpus(Path(DEFAULT_CORPUS_DIR))
    eval_set = load_eval_set(Path(DEFAULT_EVAL_SET_PATH))
    validate_eval_set_against_corpus(eval_set, corpus)

    try:
        retriever = build_retriever(request.mode, corpus)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    per_query, summary = run_eval(corpus, eval_set, retriever, request.k)
    return {
        "mode": request.mode,
        "summary": summary,
        "per_query": [qm.as_dict() for qm in per_query],
    }
