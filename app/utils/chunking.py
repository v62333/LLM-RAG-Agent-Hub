from typing import List


def simple_chunk(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    step = max_tokens - overlap
    if step <= 0:
        return [text]
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
    return chunks
