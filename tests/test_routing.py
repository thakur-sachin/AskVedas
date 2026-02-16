from app.services.retriever import route_collection


def test_route_collection_auto_hi() -> None:
    detected, collection = route_collection('auto', 'धर्म')
    assert detected == 'hi'
    assert collection == 'scriptures_hi'


def test_route_collection_force_en() -> None:
    detected, collection = route_collection('en', 'धर्म')
    assert detected == 'en'
    assert collection == 'scriptures_en'
