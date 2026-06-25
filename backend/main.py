import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import DocumentAssistant
from utils.pdf_loader import EmptyPDFError
from utils.llm import MissingAPIKeyError, LLMRequestError

load_dotenv()

app = FastAPI(title="Document Assistant API")

# Allow the Streamlit frontend (any origin, for simplicity in this MVP) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: { session_id: DocumentAssistant }
# This is fine for an MVP / single-instance deployment. It resets on restart.
SESSIONS = {}


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF. Returns a session_id to use for /ask."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    assistant = DocumentAssistant()
    try:
        num_chunks = assistant.process_pdf(file_bytes)
    except EmptyPDFError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = assistant

    return {"session_id": session_id, "num_chunks": num_chunks}


@app.post("/ask")
async def ask_question(req: AskRequest):
    """Ask a question about a previously-uploaded document."""
    assistant = SESSIONS.get(req.session_id)
    if assistant is None:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = assistant.answer_question(req.question)
    except MissingAPIKeyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return {"answer": answer, "sources": sources}