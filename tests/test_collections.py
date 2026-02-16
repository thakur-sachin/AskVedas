from qdrant_client.http import models as qm

from app.vector import collections


class DummyCollections:
    def __init__(self, names: list[str]) -> None:
        self.collections = [type('C', (), {'name': n}) for n in names]


class DummyClient:
    def __init__(self) -> None:
        self.created: list[str] = []

    def get_collections(self) -> DummyCollections:
        return DummyCollections(['scriptures_en'])

    def create_collection(self, collection_name: str, vectors_config: qm.VectorParams) -> None:
        self.created.append(collection_name)



def test_ensure_collections_creates_missing(monkeypatch) -> None:
    dummy = DummyClient()
    monkeypatch.setattr(collections, 'get_qdrant_client', lambda: dummy)
    collections.ensure_collections()
    assert 'scriptures_hi' in dummy.created
