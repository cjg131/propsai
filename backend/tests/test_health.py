from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://placeholder.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "placeholder-key")
    monkeypatch.setenv("SPORTSDATAIO_API_KEY", "placeholder-key")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-key")
    monkeypatch.setenv("APP_ENV", "test")

    from app.main import app

    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "propsai-backend"
    assert data["version"] == "0.1.0"
