import os
import requests

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

NOT_FOUND_MESSAGE = "I couldn't find that information in the document."

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided document context. "
    "Rules:\n"
    "1. Only use information found in the context below.\n"
    "2. If the answer is not contained in the context, reply exactly with: "
    f'"{NOT_FOUND_MESSAGE}"\n'
    "3. Do not make up facts or use outside knowledge.\n"
    "4. Keep answers concise and directly relevant to the question."
)


class MissingAPIKeyError(Exception):
    pass


class LLMRequestError(Exception):
    pass


def _build_messages(question: str, context_chunks: list) -> list:
    """Build an OpenAI-style messages list for the chat-completions API."""
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    user_content = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(question: str, context_chunks: list) -> str:
    
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "HF_API_KEY environment variable is not set. "
            "Add it to your .env file or Streamlit Cloud secrets."
        )

    if not context_chunks:
        # No relevant context was retrieved at all -> can't be in the doc.
        return NOT_FOUND_MESSAGE

    model = os.getenv("HF_MODEL", DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": _build_messages(question, context_chunks),
        "max_tokens": 300,
        "temperature": 0.2,
    }

    try:
        response = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise LLMRequestError(f"Network error while calling Hugging Face API: {e}")

    if response.status_code == 401:
        raise MissingAPIKeyError("Hugging Face API key is invalid or unauthorized.")

    if response.status_code == 400:
        raise LLMRequestError(
            f"Hugging Face rejected model '{model}' (HTTP 400): {response.text[:300]}\n"
            "This usually means the model isn't available for the chat-completions "
            "task on any provider. Try a different HF_MODEL, e.g. "
            "'Qwen/Qwen2.5-7B-Instruct' or 'meta-llama/Llama-3.2-3B-Instruct'."
        )

    if response.status_code != 200:
        raise LLMRequestError(
            f"Hugging Face API returned status {response.status_code}: {response.text[:300]}"
        )

    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise LLMRequestError(f"Hugging Face API error: {data['error']}")

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise LLMRequestError(f"Unexpected response format from Hugging Face API: {data}")

    if not answer:
        return NOT_FOUND_MESSAGE

    return answer