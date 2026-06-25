"""
vector_store.py
----------------
A thin wrapper around FAISS for storing chunk embeddings and performing
semantic similarity search.

Because embeddings are normalized (see embeddings.py), we use an
IndexFlatIP (inner product) index, which is equivalent to cosine
similarity for normalized vectors. FlatIP is simple, exact, and more
than fast enough for the size of documents an MVP needs to handle.
"""

from typing import List, Tuple
import numpy as np
import faiss


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[str] = []  # parallel list: index position -> chunk text

    def add(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        """Add a batch of embeddings + their corresponding text chunks."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Search for the top_k most similar chunks to the query embedding.

        Returns:
            List of (chunk_text, similarity_score) tuples, best first.
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
