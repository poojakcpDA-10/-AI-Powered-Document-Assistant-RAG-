from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    
    return SentenceTransformer(MODEL_NAME)
def embed_texts(texts: List[str]) -> np.ndarray:
   
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # so we can use cosine similarity via inner product
    )
    return embeddings.astype("float32")
def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])
