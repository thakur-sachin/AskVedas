from __future__ import annotations

import re


def approx_tokens(text: str) -> int:
    return len(text.split())


def detect_text_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return 'mixed'
    lines = [ln.strip() for ln in stripped.split('\n') if ln.strip()]
    danda_lines = sum(1 for ln in lines if '।' in ln or '॥' in ln)
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 12)
    if lines and danda_lines >= max(2, len(lines) // 3):
        return 'verse'
    if lines and short_lines >= max(3, len(lines) // 2):
        return 'verse'
    if approx_tokens(stripped) > 140:
        return 'commentary'
    return 'mixed'


def chunk_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_tokens, len(words))
        out.append(' '.join(words[start:end]).strip())
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return out


def chunk_english(text: str, target_tokens: int = 800, overlap_tokens: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        tks = approx_tokens(para)
        if tks > target_tokens:
            if buffer:
                chunks.append('\n\n'.join(buffer))
                buffer = []
                current_tokens = 0
            chunks.extend(chunk_by_tokens(para, target_tokens, overlap_tokens))
            continue
        if current_tokens + tks > target_tokens and buffer:
            chunks.append('\n\n'.join(buffer))
            overlap_text = ' '.join(chunks[-1].split()[-overlap_tokens:])
            buffer = [overlap_text, para]
            current_tokens = approx_tokens(overlap_text) + tks
        else:
            buffer.append(para)
            current_tokens += tks

    if buffer:
        chunks.append('\n\n'.join(buffer))

    return [c.strip() for c in chunks if c.strip()]


def chunk_hindi_ocr(text: str, target_tokens: int = 500, overlap_tokens: int = 100) -> list[str]:
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return []

    verse_blocks: list[str] = []
    prose_lines: list[str] = []
    current_verse: list[str] = []

    def flush_verse() -> None:
        nonlocal current_verse
        if current_verse:
            verse_blocks.append('\n'.join(current_verse))
            current_verse = []

    for line in lines:
        is_verse_line = ('।' in line or '॥' in line) or len(line.split()) <= 10
        if is_verse_line:
            current_verse.append(line)
            if len(current_verse) >= 8:
                flush_verse()
        else:
            flush_verse()
            prose_lines.append(line)

    flush_verse()

    prose_text = '\n'.join(prose_lines)
    prose_chunks = chunk_by_tokens(prose_text, target_tokens, overlap_tokens) if prose_text else []

    chunks = verse_blocks + prose_chunks
    return [c.strip() for c in chunks if c.strip()]
