"""
text_splitter.py
----------------
Splits a long document into overlapping chunks that are small enough to
embed and retrieve effectively. Overlap helps preserve context across
chunk boundaries so we don't lose meaning at the edges.
"""

from typing import List


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Split text into overlapping chunks by character count.

    Args:
        text: the full document text.
        chunk_size: max number of characters per chunk.
        chunk_overlap: number of overlapping characters between consecutive
                       chunks (keeps context continuous).

    Returns:
        A list of text chunks (strings).
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace a little so chunk boundaries look cleaner.
    text = " ".join(text.split())

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        # Move the window forward, keeping `chunk_overlap` chars of overlap.
        start = end - chunk_overlap

    return chunks
