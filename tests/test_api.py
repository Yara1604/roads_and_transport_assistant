# Testing the api endpoints & validation (7 tests thus far)
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert "indexed_chunks" in json_response

def test_search_rejects_before_ingestion():
    app.state.documents = None
    app.state.vector_store = None
    response = client.post(
        "/search",
        json={"query": "Define pavement", "mode": "hybrid", "top_k": 5},
    )        
    assert response.status_code == 409

def test_search_rejects_top_k_below_min():
    response = client.post(
        "/search",
        json={"query": "Define pavement", "mode": "hybrid", "top_k": 0},
    )
    assert response.status_code == 422


def test_search_rejects_top_k_above_max():
    response = client.post(
        "/search",
        json={"query": "Define pavement", "mode": "hybrid", "top_k": 21},
    )
    assert response.status_code == 422


def test_search_rejects_unsupported_mode():
    response = client.post(
        "/search",
        json={"query": "Define pavement", "mode": "IDK :)", "top_k": 5},
    )
    assert response.status_code == 422


def test_search_rejects_empty_query():
    response = client.post(
        "/search",
        json={"query": "", "mode": "hybrid", "top_k": 5},
    )
    assert response.status_code == 422


def test_chat_rejects_empty_query():
    response = client.post(
        "/chat",
        json={
            "thread_id": "test",
            "message": "",
            "mode": "hybrid",
            "top_k": 5,
        },
    )
    assert response.status_code == 422
