"""
pdf_loader.py
--------------
Responsible for extracting raw text from an uploaded PDF file.

We use pypdf because it's lightweight, pure-Python, and has no system
dependencies (important for Streamlit Community Cloud, where you cannot
install OS-level packages).
"""

from io import BytesIO
from pypdf import PdfReader


class EmptyPDFError(Exception):
    """Raised when a PDF has no extractable text (e.g. a scanned image PDF)."""
    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF given as raw bytes.

    Args:
        file_bytes: the raw bytes of the uploaded PDF file.

    Returns:
        A single string containing the concatenated text of every page.

    Raises:
        EmptyPDFError: if no text could be extracted at all (likely a
                       scanned/image-only PDF with no OCR layer).
        ValueError: if the file is not a valid PDF.
    """
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read PDF file. Is it a valid PDF? ({e})")

    if len(reader.pages) == 0:
        raise EmptyPDFError("The PDF has no pages.")

    full_text = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            full_text.append(text)

    combined = "\n".join(full_text).strip()

    if not combined:
        raise EmptyPDFError(
            "No extractable text was found in this PDF. "
            "It might be a scanned/image-only document without OCR."
        )

    return combined
