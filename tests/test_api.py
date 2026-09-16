from fastapi.testclient import TestClient

from harness.api import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_summary_for_tfidf():
    response = client.post("/evaluate", json={"mode": "tfidf", "k": [1, 3, 5]})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "tfidf"
    assert "summary" in body
    assert body["summary"]["n_queries"] > 0
    assert 0.0 <= body["summary"]["mrr"] <= 1.0
    assert len(body["per_query"]) == body["summary"]["n_queries"]


def test_evaluate_uses_default_mode_and_k_when_body_omitted():
    response = client.post("/evaluate", json={})
    assert response.status_code == 200
    assert response.json()["mode"] == "tfidf"


def test_evaluate_rejects_unknown_mode():
    response = client.post("/evaluate", json={"mode": "not_a_real_mode"})
    assert response.status_code == 400
    assert "not_a_real_mode" in response.json()["detail"]
