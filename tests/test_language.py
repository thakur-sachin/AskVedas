from app.services.language import detect_language


def test_detect_language_english() -> None:
    assert detect_language('Tell me about dharma') == 'en'


def test_detect_language_devanagari() -> None:
    assert detect_language('धर्म क्या है') == 'hi'
