import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc_apple.md").write_text(
        "リンゴは赤い果物です。甘みと酸味があります。", encoding="utf-8"
    )
    (corpus_dir / "doc_banana.md").write_text(
        "バナナは黄色い果物です。カリウムを多く含みます。", encoding="utf-8"
    )
    (corpus_dir / "doc_carrot.md").write_text(
        "ニンジンはオレンジ色の野菜です。ベータカロテンを含みます。", encoding="utf-8"
    )
    return corpus_dir


@pytest.fixture
def tiny_eval_set_path(tmp_path: Path) -> Path:
    eval_set = {
        "version": "1.0",
        "questions": [
            {"id": "t1", "question": "赤い果物は何ですか?", "relevant_doc_ids": ["doc_apple"]},
            {"id": "t2", "question": "黄色い果物は何ですか?", "relevant_doc_ids": ["doc_banana"]},
            {
                "id": "t3",
                "question": "ベータカロテンを含む野菜は?",
                "relevant_doc_ids": ["doc_carrot"],
            },
            {"id": "t4", "question": "今日の天気を教えてください", "relevant_doc_ids": []},
        ],
    }
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(eval_set, ensure_ascii=False), encoding="utf-8")
    return path
