import json
from pathlib import Path

import pytest

from harness.runner import (
    DEFAULT_K_VALUES,
    NaiveOverlapRetriever,
    TfidfRetriever,
    bigrams,
    build_retriever,
    load_corpus,
    load_eval_set,
    run_eval,
    validate_eval_set_against_corpus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- load_corpus ------------------------------------------------------------


def test_load_corpus_reads_all_documents(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    assert set(corpus) == {"doc_apple", "doc_banana", "doc_carrot"}
    assert "リンゴ" in corpus["doc_apple"]


def test_load_corpus_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist")


def test_load_corpus_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        load_corpus(empty_dir)


def test_load_corpus_ignores_non_document_files(tiny_corpus_dir):
    (tiny_corpus_dir / "notes.json").write_text("{}", encoding="utf-8")
    corpus = load_corpus(tiny_corpus_dir)
    assert "notes" not in corpus


# --- load_eval_set -----------------------------------------------------------


def test_load_eval_set_reads_questions(tiny_eval_set_path):
    questions = load_eval_set(tiny_eval_set_path)
    assert len(questions) == 4
    assert questions[0]["id"] == "t1"


def test_load_eval_set_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "missing.json")


def test_load_eval_set_missing_required_field_raises(tmp_path):
    bad = {"questions": [{"id": "q1", "question": "no relevant_doc_ids field"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(path)


def test_load_eval_set_duplicate_id_raises(tmp_path):
    bad = {
        "questions": [
            {"id": "q1", "question": "a", "relevant_doc_ids": []},
            {"id": "q1", "question": "b", "relevant_doc_ids": []},
        ]
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(path)


def test_load_eval_set_empty_questions_list_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(path)


# --- validate_eval_set_against_corpus ---------------------------------------


def test_validate_eval_set_against_corpus_passes_for_known_docs(tiny_corpus_dir, tiny_eval_set_path):
    corpus = load_corpus(tiny_corpus_dir)
    eval_set = load_eval_set(tiny_eval_set_path)
    validate_eval_set_against_corpus(eval_set, corpus)  # should not raise


def test_validate_eval_set_against_corpus_rejects_unknown_doc_id(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    eval_set = [{"id": "q1", "question": "x", "relevant_doc_ids": ["doc_does_not_exist"]}]
    with pytest.raises(ValueError):
        validate_eval_set_against_corpus(eval_set, corpus)


# --- tokenization -------------------------------------------------------------


def test_bigrams_normal_text():
    assert bigrams("abc") == ["ab", "bc"]


def test_bigrams_single_character_boundary():
    assert bigrams("a") == ["a"]


def test_bigrams_empty_string_boundary():
    assert bigrams("") == []


def test_bigrams_strips_whitespace_and_lowercases():
    assert bigrams("A B") == bigrams("ab")


# --- retrievers ---------------------------------------------------------------


def test_naive_retriever_ranks_matching_doc_first(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    retriever = NaiveOverlapRetriever(corpus)
    ranked = retriever.rank("リンゴは赤い")
    assert ranked[0] == "doc_apple"


def test_tfidf_retriever_ranks_matching_doc_first(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    retriever = TfidfRetriever(corpus)
    ranked = retriever.rank("ベータカロテンを含む野菜")
    assert ranked[0] == "doc_carrot"


def test_tfidf_retriever_returns_all_doc_ids(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    retriever = TfidfRetriever(corpus)
    ranked = retriever.rank("何か")
    assert set(ranked) == set(corpus)


def test_build_retriever_rejects_unknown_mode(tiny_corpus_dir):
    corpus = load_corpus(tiny_corpus_dir)
    with pytest.raises(ValueError):
        build_retriever("not_a_real_mode", corpus)


# --- run_eval / determinism ----------------------------------------------------


def test_run_eval_is_deterministic(tiny_corpus_dir, tiny_eval_set_path):
    corpus = load_corpus(tiny_corpus_dir)
    eval_set = load_eval_set(tiny_eval_set_path)
    retriever_a = TfidfRetriever(corpus)
    retriever_b = TfidfRetriever(corpus)

    _, summary_a = run_eval(corpus, eval_set, retriever_a, DEFAULT_K_VALUES)
    _, summary_b = run_eval(corpus, eval_set, retriever_b, DEFAULT_K_VALUES)
    assert summary_a == summary_b


def test_run_eval_irrelevant_query_excluded_from_recall(tiny_corpus_dir, tiny_eval_set_path):
    corpus = load_corpus(tiny_corpus_dir)
    eval_set = load_eval_set(tiny_eval_set_path)
    retriever = TfidfRetriever(corpus)
    per_query, _ = run_eval(corpus, eval_set, retriever, k_values=[1])
    t4 = next(qm for qm in per_query if qm.query_id == "t4")
    assert t4.recall_at_k[1] is None


# --- integration against the real shipped data ---------------------------------


def test_shipped_eval_set_and_corpus_are_loadable_and_scoreable():
    corpus = load_corpus(REPO_ROOT / "data" / "corpus")
    eval_set = load_eval_set(REPO_ROOT / "data" / "eval_set.json")
    validate_eval_set_against_corpus(eval_set, corpus)

    retriever = TfidfRetriever(corpus)
    _, summary = run_eval(corpus, eval_set, retriever, DEFAULT_K_VALUES)

    assert summary["n_queries"] == len(eval_set)
    assert 0.0 <= summary["mrr"] <= 1.0
