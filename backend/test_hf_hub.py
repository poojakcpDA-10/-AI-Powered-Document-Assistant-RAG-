from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="YOUR_HF_TOKEN"
)

try:
    result = client.text_generation(
        "Hello",
        max_new_tokens=20
    )
    print(result)
except Exception as e:
    print("ERROR:", e)