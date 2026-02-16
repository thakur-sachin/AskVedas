from __future__ import annotations

import re
from typing import Literal

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> Literal['hi', 'en']:
    if DEVANAGARI_RE.search(text):
        return 'hi'
    return 'en'


def resolve_language(lang_pref: str, query: str) -> Literal['hi', 'en']:
    if lang_pref == 'hi':
        return 'hi'
    if lang_pref == 'en':
        return 'en'
    return detect_language(query)
