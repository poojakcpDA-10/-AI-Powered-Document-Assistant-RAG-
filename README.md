# 📄 AI-Powered Document Assistant (RAG)

An MVP Retrieval-Augmented-Generation app: upload a PDF, ask questions about
it, and get answers grounded **only** in the document's content — powered by
free, open Hugging Face models.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key needed)
- **Vector search:** FAISS (`IndexFlatIP`, cosine similarity)
- **LLM:** Hugging Face Inference API (free tier), called via plain REST (`requests`) — e.g. `mistralai/Mistral-7B-Instruct-v0.3`
- **Backend:** FastAPI (REST API) — optional, for separating frontend/backend
- **Frontend:** Streamlit

---

## 1. Project Structure

```
document-assistant/
├── backend/
│   ├── main.py                # FastAPI REST API (optional separate service)
│   ├── rag_pipeline.py        # Shared RAG orchestration logic
│   ├── requirements.txt
│   ├── .env.example
│   └── utils/
│       ├── pdf_loader.py      # PDF text extraction (pypdf)
│       ├── text_splitter.py   # Chunking
│       ├── embeddings.py      # Sentence-Transformers embeddings
│       ├── vector_store.py    # FAISS wrapper
│       └── llm.py             # Hugging Face Inference API (REST)
├── frontend/
│   ├── app.py                 # Streamlit UI
│   └── requirements.txt
├── requirements.txt            # Root copy, used by Streamlit Community Cloud
├── .streamlit/secrets.toml.example
├── .gitignore
└── README.md
```

---

## 2. How it works (RAG pipeline)

1. **Upload** → PDF bytes are read.
2. **Extract** → `pdf_loader.py` pulls raw text out with `pypdf`.
3. **Split** → `text_splitter.py` breaks the text into ~800-character
   overlapping chunks.
4. **Embed** → `embeddings.py` encodes each chunk into a vector with
   `all-MiniLM-L6-v2` (local, free, no API key).
5. **Store** → `vector_store.py` puts the vectors into a FAISS index in memory.
6. **Ask** → user's question is embedded the same way, and FAISS returns the
   top-k most similar chunks (semantic search).
7. **Generate** → `llm.py` sends the question + retrieved chunks to a free
   Hugging Face LLM with a strict instruction: *answer only from this
   context, or say you don't know.*
8. **Fallback** → if no relevant chunk is found (similarity below a
   threshold) or the model says it doesn't know, the app shows:
   `"I couldn't find that information in the document."`

### Why two "backends"?
- `backend/main.py` is a real **FastAPI REST API** (`/upload`, `/ask`,
  `/health`) — useful if you want a proper client/server split, or to call
  this from something other than Streamlit.
- Streamlit Community Cloud can only run **one process** for free, so it
  can't host FastAPI + Streamlit together. By default, `frontend/app.py`
  therefore imports `backend/rag_pipeline.py` **directly** (in-process —
  no HTTP hop, `BACKEND_MODE=direct`). If you deploy the FastAPI backend
  separately (e.g. Render, Railway, a Hugging Face Space, your own VM),
  just set `BACKEND_MODE=api` and `BACKEND_URL=https://your-backend-url`
  and the same Streamlit UI will call it over REST instead. No UI code
  changes needed either way.

---

## 3. Local Setup

### Prerequisites
- Python 3.9+
- A free Hugging Face account + API token: https://huggingface.co/settings/tokens
  (Settings → Access Tokens → New token, "Read" role is enough)

### A) Run everything locally with Streamlit only (simplest)

```bash
git clone <your-repo-url>
cd document-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set your API key
cp backend/.env.example frontend/.env
# then edit frontend/.env and paste your real HF_API_KEY

streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

### B) Run backend (FastAPI) + frontend (Streamlit) separately

```bash
# Terminal 1 - backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your HF_API_KEY
uvicorn main:app --reload --port 8000

# Terminal 2 - frontend
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export BACKEND_MODE=api
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Test the API directly (optional):
```bash
curl -F "file=@/path/to/sample.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<id_from_upload>","question":"What is this document about?"}'
```

---

## 4. Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private repo connected to your Streamlit account).
2. Go to https://share.streamlit.io → **New app**.
3. Choose your repo/branch and set:
   - **Main file path:** `frontend/app.py`
4. Click **Advanced settings → Secrets**, and paste:
   ```toml
   HF_API_KEY = "your_huggingface_api_key_here"
   HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
   BACKEND_MODE = "direct"
   ```
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt` from
   the repo root automatically.
6. First load will be slower (downloading the embedding model, ~80MB) —
   subsequent runs are fast.

> If you instead want `BACKEND_MODE = "api"` pointing at a separately
> deployed FastAPI service, also add `BACKEND_URL = "https://your-api-url"`
> to the secrets above.

---

## 5. Error Handling Covered

| Case                          | Behavior                                                          |
|-------------------------------|---------------------------------------------------------------------|
| No PDF uploaded               | "Process Document" button disabled; clear message if forced        |
| Empty / scanned PDF (no text) | Friendly error: PDF has no extractable text                         |
| Invalid / corrupted PDF       | Friendly error: "Could not read PDF file..."                        |
| Empty question                | Validation error before calling the LLM                             |
| Missing/invalid HF API key    | Clear warning shown in UI before asking; clean error if it happens  |
| Answer not in document        | Returns: *"I couldn't find that information in the document."*      |
| HF API/network failure        | Caught and shown as a readable error, app doesn't crash             |

---

## 6. Tech Stack

Python · Streamlit · FastAPI · Hugging Face Inference API (free model) ·
Sentence-Transformers · FAISS · pypdf · python-dotenv · requests

---

## 7. Limitations (honest, MVP scope)

- Sessions are **in-memory** — restarting the app loses the index (re-upload the PDF).
- Scanned/image-only PDFs without OCR won't extract text (no OCR included).
- Free Hugging Face Inference API can be slow/rate-limited or occasionally
  "model loading" on cold start — the code uses `wait_for_model: true` to
  handle this gracefully, but expect some latency on the first question.
- Single PDF per session (by design, for simplicity).

---

## 8. Live App

_Add your published Streamlit Community Cloud link here, e.g.:_
`Live app: https://your-app-name.streamlit.app`
