import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HUGGINGFACE_API_KEY")
r = requests.post(
    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"inputs": "tell me a joke", "parameters": {"max_new_tokens": 100}}
)
print("Status:", r.status_code)
print("Response:", r.text[:300])
