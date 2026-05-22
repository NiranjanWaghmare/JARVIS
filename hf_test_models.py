import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()
os.environ["LITELLM_LOG"] = "DEBUG"

models = [
    "huggingface/Qwen/Qwen2.5-72B-Instruct",
    "huggingface/CohereForAI/c4ai-command-r-plus",
    "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
]

for model in models:
    try:
        print(f"Testing {model}...")
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key=os.getenv("HUGGINGFACE_API_KEY")
        )
        print(f"SUCCESS: {model}")
        print(response.choices[0].message.content[:50])
    except Exception as e:
        print(f"FAILED {model}: {e}\n")
