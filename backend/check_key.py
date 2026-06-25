import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("HF_API_KEY")

print("Key:", key[:10] + "..." if key else "None")