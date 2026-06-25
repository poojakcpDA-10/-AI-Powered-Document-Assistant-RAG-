import os
import sys
import requests
import streamlit as st

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="centered")
from dotenv import load_dotenv

load_dotenv()


_secrets_paths = [
    os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
    os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
]
if any(os.path.exists(p) for p in _secrets_paths):
    try:
        for key, value in st.secrets.items():
            os.environ.setdefault(key, str(value))
    except Exception:
        pass  # secrets file existed but couldn't be read for some reason -> fine, ignore

# Make the backend/ folder importable so we can reuse rag_pipeline.py directly.
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

BACKEND_MODE = os.getenv("BACKEND_MODE", "direct")  # "direct" or "api"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}
if "assistant" not in st.session_state:
    st.session_state.assistant = None  # DocumentAssistant instance (direct mode)
if "session_id" not in st.session_state:
    st.session_state.session_id = None  # backend session id (api mode)
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ---------------------------------------------------------------------------
# Backend calls (abstracted so the UI code doesn't care which mode is used)
# ---------------------------------------------------------------------------
def process_uploaded_pdf(uploaded_file):
    """Process the uploaded PDF using either direct or API mode."""
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if BACKEND_MODE == "api":
        files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("detail", resp.text))
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        return data["num_chunks"]
    else:
        from rag_pipeline import DocumentAssistant
        assistant = DocumentAssistant()
        num_chunks = assistant.process_pdf(file_bytes)
        st.session_state.assistant = assistant
        return num_chunks


def ask_question(question: str):
    """Ask a question using either direct or API mode. Returns (answer, sources)."""
    if BACKEND_MODE == "api":
        payload = {"session_id": st.session_state.session_id, "question": question}
        resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("detail", resp.text))
        data = resp.json()
        return data["answer"], data.get("sources", [])
    else:
        return st.session_state.assistant.answer_question(question)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📄 AI-Powered Document Assistant")
st.caption(
    "Upload a PDF, then ask questions about it. Answers are generated "
    "**only** from the document's content using a Retrieval-Augmented "
    "Generation (RAG) pipeline + a free Hugging Face LLM."
)

# --- Check API key up front and warn early -------------------------------
if not os.getenv("HF_API_KEY"):
    st.warning(
        "⚠️ HF_API_KEY is not set. Add it to your `.env` file (local) or to "
        "**Settings → Secrets** (Streamlit Community Cloud) before asking "
        "questions. PDF upload/processing will still work without it.",
        icon="⚠️",
    )

# --- 1. Upload section -----------------------------------------------------
st.subheader("1️⃣ Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

process_clicked = st.button("📥 Process Document", use_container_width=True, disabled=uploaded_file is None)

if process_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF file first.")
    else:
        with st.spinner("Extracting text, splitting into chunks, and building embeddings..."):
            try:
                num_chunks = process_uploaded_pdf(uploaded_file)
                st.session_state.doc_ready = True
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []  # fresh conversation for a new doc
                st.success(f"✅ Processed **{uploaded_file.name}** into {num_chunks} chunks. Ready for questions!")
            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                # Covers EmptyPDFError and any unexpected backend error.
                st.error(f"❌ Could not process this PDF: {e}")

if st.session_state.doc_ready:
    st.info(f"📌 Currently loaded document: **{st.session_state.doc_name}**")

st.divider()

# --- 2. Question section ----------------------------------------------------
st.subheader("2️⃣ Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the termination clause in this contract?",
    disabled=not st.session_state.doc_ready,
)
ask_clicked = st.button("🔎 Ask", disabled=not st.session_state.doc_ready)

if ask_clicked:
    if not question or not question.strip():
        st.error("❌ Please enter a non-empty question.")
    else:
        with st.spinner("Searching the document and generating an answer..."):
            try:
                answer, sources = ask_question(question.strip())
                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                msg = str(e)
                if "HF_API_KEY" in msg or "API key" in msg:
                    st.error(
                        "❌ Missing or invalid Hugging Face API key. "
                        "Set HF_API_KEY in your .env file or Streamlit secrets."
                    )
                else:
                    st.error(f"❌ Something went wrong while generating the answer: {msg}")

st.divider()

# --- 3. Chat history --------------------------------------------------------
hist_col1, hist_col2 = st.columns([3, 1])
with hist_col1:
    st.subheader("💬 Chat History")
with hist_col2:
    clear_clicked = st.button("🗑️ Clear Chat", use_container_width=True)

if clear_clicked:
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

if not st.session_state.chat_history:
    st.caption("No questions asked yet. Upload a document and ask away!")
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                sources = msg.get("sources") or []
                if sources:
                    with st.expander(f"📚 View {len(sources)} source chunk(s) used"):
                        for i, chunk in enumerate(sources, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.write(chunk)
