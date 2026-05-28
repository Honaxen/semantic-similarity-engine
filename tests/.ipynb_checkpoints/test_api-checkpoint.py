"""
test_api.py
-----------
Tests for the Semantic Similarity Engine API.
"""

from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Semantic Similarity Engine"


def test_health_check():
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check():
    """Test readiness endpoint returns ready status."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_get_corpus():
    """Test corpus endpoint returns list of sentences."""
    response = client.get("/search/corpus")
    assert response.status_code == 200
    assert "sentences" in response.json()
    assert len(response.json()["sentences"]) > 0


def test_semantic_search():
    """Test search endpoint returns relevant results."""
    response = client.post(
        "/search/query",
        json={"query": "machine learning", "top_n": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["score"] > 0


def test_empty_query():
    """Test that empty query returns 400 error."""
    response = client.post(
        "/search/query",
        json={"query": "", "top_n": 2}
    )
    assert response.status_code == 400


def test_add_sentences():
    """Test adding new sentences to corpus."""
    response = client.post(
        "/search/add",
        json={"sentences": ["Transformers revolutionized NLP"]}
    )
    assert response.status_code == 200
    assert response.json()["added"] == 1