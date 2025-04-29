from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
chat_response = client.chat.completions.create(
    model=os.getenv("MODEl_PATH"),
    messages=messages,
    extra_body = {
        "top_k":50,
        "repetition_penalty":1.05,
        # "stop_token_ids":
    }
    
)
print("Chat response:", chat_response)   