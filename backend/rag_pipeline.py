"""
rag_pipeline.py
----------------
Glues together pdf_loader -> text_splitter -> embeddings -> vector_store -> llm
into one simple RAG (Retrieval-Augmented Generation) pipeline class.

Both the FastAPI backend (main.py) and the Streamlit app (frontend/app.py)
use this same class, so the logic only lives in one place.
"""

from typing import List, Tuple

from utils.pdf_loader import extract_text_from_pdf, EmptyPDFError
from utils.text_splitter import split_text_into_chunks
from utils.embeddings import embed_texts, embed_query
from utils.vector_store import VectorStore
from utils.llm import generate_answer, MissingAPIKeyError, LLMRequestError, NOT_FOUND_MESSAGE


class DocumentAssistant:
    """
    Holds the state for ONE uploaded document: its chunks and FAISS index.
    Create a new instance whenever a new PDF is uploaded.
    """

    def __init__(self):
        self.vector_store: VectorStore = None
        self.num_chunks: int = 0

    def process_pdf(self, file_bytes: bytes, chunk_size: int = 800, chunk_overlap: int = 150) -> int:
        """
        Extract, split, and embed a PDF's text. Builds the FAISS index.

        Returns:
            The number of chunks created.

        Raises:
            EmptyPDFError, ValueError: propagated from pdf_loader for
            invalid / empty PDFs.
        """
        text = extract_text_from_pdf(file_bytes)  # may raise EmptyPDFError/ValueError
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

        if not chunks:
            raise EmptyPDFError("No usable text chunks could be created from this PDF.")

        embeddings = embed_texts(chunks)
        self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(embeddings, chunks)
        self.num_chunks = len(chunks)
        return self.num_chunks

    def answer_question(self, question: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks and generate an answer.

        Returns:
            (answer_text, list_of_retrieved_chunks)

        Raises:
            MissingAPIKeyError, LLMRequestError: propagated from llm.py.
        """
        if self.vector_store is None or self.vector_store.is_empty:
            return NOT_FOUND_MESSAGE, []

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_vec = embed_query(question)
        results = self.vector_store.search(query_vec, top_k=top_k)

        # Filter out very low-relevance matches (likely irrelevant to the question).
        SIMILARITY_THRESHOLD = 0.2
        relevant_chunks = [chunk for chunk, score in results if score >= SIMILARITY_THRESHOLD]

        if not relevant_chunks:
            return NOT_FOUND_MESSAGE, []

        answer = generate_answer(question, relevant_chunks)
        return answer, relevant_chunks
