from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text.replace('\x0c', ' ')
    text = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F]', ' ', text)
    text = re.sub(r'[_]{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
