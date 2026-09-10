"""Fixed evaluation-set runner: loads a corpus + eval set, retrieves ranked
document IDs per query with a chosen retriever, and scores them with
harness.metrics.

Two retrievers ship offline (no network, fully deterministic):

- "naive":  raw character-bigram overlap count, no term weighting. Stands in
  for a simplistic "before" baseline.
- "tfidf":  TF-IDF weighted cosine similarity over the same bigrams. Stands
  in for an "after" improvement.

A third, optional retriever ("openai") calls the real OpenAI embeddings API
and is only used if explicitly requested and an API key is configured; it is
never used by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from harness.metrics import QueryMetrics, aggregate, evaluate_query

DEFAULT_K_VALUES = (1, 3, 5)
_CORPUS_EXTENSIONS = (".md", ".txt")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Load every .md/.txt file in `corpus_dir` into {doc_id: text}."""
    path = Path(corpus_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {path}")

    corpus: dict[str, str] = {}
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() not in _CORPUS_EXTENSIONS:
            continue
        corpus[file_path.stem] = file_path.read_text(encoding="utf-8")

    if not corpus:
        raise ValueError(f"no {_CORPUS_EXTENSIONS} documents found in {path}")
    return corpus


def load_eval_set(path: str | Path) -> list[dict]:
    """Load and structurally validate the fixed evaluation set."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"eval set not found: {file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("eval set must contain a non-empty 'questions' list")

    seen_ids: set[str] = set()
    for item in questions:
        for required_field in ("id", "question", "relevant_doc_ids"):
            if required_field not in item:
                raise ValueError(f"eval set item missing required field '{required_field}': {item}")
        if not isinstance(item["relevant_doc_ids"], list):
            raise ValueError(f"relevant_doc_ids must be a list in item {item['id']!r}")
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate question id in eval set: {item['id']!r}")
        seen_ids.add(item["id"])
    return questions


def validate_eval_set_against_corpus(eval_set: list[dict], corpus: dict[str, str]) -> None:
    """Raise if any eval item references a document ID absent from the corpus."""
    unknown: dict[str, list[str]] = {}
    for item in eval_set:
        missing = [doc_id for doc_id in item["relevant_doc_ids"] if doc_id not in corpus]
        if missing:
            unknown[item["id"]] = missing
    if unknown:
        raise ValueError(f"eval set references unknown corpus document IDs: {unknown}")


# --------------------------------------------------------------------------
# Tokenization + TF-IDF (pure stdlib; works without a Japanese tokenizer
# because it operates on character bigrams rather than word boundaries)
# --------------------------------------------------------------------------


def bigrams(text: str) -> list[str]:
    """Character bigrams of the lowercased, whitespace-stripped text.

    Falls back to the single character itself for length-1 strings so that
    very short queries still produce at least one token instead of none.
    """
    cleaned = "".join(text.lower().split())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


class Retriever(Protocol):
    def rank(self, query: str) -> list[str]: ...


class NaiveOverlapRetriever:
    """Baseline: unweighted character-bigram overlap count."""

    def __init__(self, corpus: dict[str, str]) -> None:
        self._doc_bigram_sets = {doc_id: set(bigrams(text)) for doc_id, text in corpus.items()}

    def rank(self, query: str) -> list[str]:
        query_bigrams = set(bigrams(query))
        scores = {
            doc_id: len(query_bigrams & doc_bigrams)
            for doc_id, doc_bigrams in self._doc_bigram_sets.items()
        }
        return _sort_by_score(scores)


class TfidfRetriever:
    """TF-IDF weighted cosine similarity over character bigrams."""

    def __init__(self, corpus: dict[str, str]) -> None:
        self._doc_ids = list(corpus.keys())
        doc_tokens = {doc_id: bigrams(text) for doc_id, text in corpus.items()}

        n_docs = len(doc_tokens)
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            doc_freq.update(set(tokens))

        self._idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in doc_freq.items()
        }
        self._doc_vectors = {
            doc_id: self._vectorize(tokens) for doc_id, tokens in doc_tokens.items()
        }

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        term_freq = Counter(tokens)
        weights = {
            term: freq * self._idf.get(term, 0.0) for term, freq in term_freq.items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0.0:
            return {}
        return {term: w / norm for term, w in weights.items()}

    def rank(self, query: str) -> list[str]:
        query_vector = self._vectorize(bigrams(query))
        scores = {
            doc_id: _cosine_sparse(query_vector, doc_vector)
            for doc_id, doc_vector in self._doc_vectors.items()
        }
        return _sort_by_score(scores)


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return sum(weight * longer.get(term, 0.0) for term, weight in shorter.items())


def _sort_by_score(scores: dict[str, float]) -> list[str]:
    # Ties break on doc_id for full determinism across runs and platforms.
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class OpenAIEmbeddingRetriever:
    """Optional real-API retriever. Never used unless explicitly selected.

    Requires the `openai` package (install with `uv sync --extra real`) and
    an OPENAI_API_KEY environment variable. Imported lazily so the mock path
    never needs the dependency installed.
    """

    def __init__(self, corpus: dict[str, str], model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for --mode openai. "
                "Install it with: uv sync --extra real"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; required for --mode openai")

        self._model = model or os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self._client = OpenAI(api_key=api_key)
        self._doc_ids = list(corpus.keys())
        self._doc_vectors = self._embed(list(corpus.values()))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def rank(self, query: str) -> list[str]:
        query_vector = self._embed([query])[0]
        scores = {
            doc_id: _cosine_dense(query_vector, doc_vector)
            for doc_id, doc_vector in zip(self._doc_ids, self._doc_vectors)
        }
        return _sort_by_score(scores)


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_retriever(mode: str, corpus: dict[str, str]) -> Retriever:
    if mode == "naive":
        return NaiveOverlapRetriever(corpus)
    if mode == "tfidf":
        return TfidfRetriever(corpus)
    if mode == "openai":
        return OpenAIEmbeddingRetriever(corpus)
    raise ValueError(f"unknown retriever mode: {mode!r} (expected naive, tfidf, or openai)")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def run_eval(
    corpus: dict[str, str],
    eval_set: list[dict],
    retriever: Retriever,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> tuple[list[QueryMetrics], dict]:
    """Run every question through `retriever` and score the ranked results."""
    per_query: list[QueryMetrics] = []
    for item in eval_set:
        ranked_ids = retriever.rank(item["question"])
        per_query.append(
            evaluate_query(item["id"], ranked_ids, item["relevant_doc_ids"], k_values)
        )
    return per_query, aggregate(per_query, k_values)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed RAG evaluation set.")
    parser.add_argument("--eval", default="data/eval_set.json", help="Path to eval_set.json")
    parser.add_argument("--corpus", default="data/corpus", help="Path to the corpus directory")
    parser.add_argument(
        "--mode", choices=["naive", "tfidf", "openai"], default="tfidf", help="Retriever to use"
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="K values to evaluate"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    corpus = load_corpus(args.corpus)
    eval_set = load_eval_set(args.eval)
    validate_eval_set_against_corpus(eval_set, corpus)

    retriever = build_retriever(args.mode, corpus)
    per_query, summary = run_eval(corpus, eval_set, retriever, args.k)

    output = {
        "mode": args.mode,
        "summary": summary,
        "per_query": [qm.as_dict() for qm in per_query],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
