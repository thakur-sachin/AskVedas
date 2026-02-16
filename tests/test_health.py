from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import health as health_module


class DummyClient:
    def get_collections(self):
        return {'collections': []}


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(health_module, 'get_qdrant_client', lambda: DummyClient())
    client = TestClient(app)
    resp = client.get('/api/v1/health')
    assert resp.status_code == 200
    body = resp.json()
    assert 'status' in body
    assert 'qdrant' in body
    assert 'llm' in body
    assert 'version' in body
