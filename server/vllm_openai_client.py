from openai import OpenAI
from dotenv import load_dotenv
from test import tools
import vllm
import logging
import os
import openai
load_dotenv("/mnt/nlp/gaoqiang/project/training/server/.env")



# Set OpenAI's API key and API base to use vLLM's API server.
model_path = os.getenv("MODEl_PATH")
openai_api_key = os.getenv("API_KEY")
openai_api_base = os.getenv("API_BASE")

logging.basicConfig(level=logging.INFO)

logging.info(f"model_path: {model_path}")
logging.info(f"openai_api_key: {openai_api_key}")
logging.info(f"openai_api_base: {openai_api_base}")



client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

while True:
    query = input("Please enter your query: ")
    if query == "exit":
        break
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

    # chat_response = openai.ChatCompletion.create   no longer supported
    chat_response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        extra_body = {
            "top_k":50,
            "repetition_penalty":1.05,
            # "stop_token_ids":
            "temperature":0.5
        },
        # stream=False,
        tools=tools
    )

    # for chunk in chat_response:
    #     print(chunk.choices[0].delta.content, end='', flush=True)
    """
    chat_response.choices[0].message.content
    chat_response.choices[0].finish_reason
    chat_response.choices[0].message.model_extra['reasoning_content']
    chat_response.usage.total_tokens
    chat_response.usage.prompt_tokens
    chat_response.usage.completion_tokens 
    chat_response.id
    chat_response.model
    """
    """ import requests
    response = requests.post(
        url=f"{openai_api_base}/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_path,
            "messages": messages,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "temperature": 0.5
        }
    )
    chat_response = response.json() """

    print("Chat response:", chat_response)

    # print("Chat response:", dir(chat_response))
    print("chat_response.choices[0].message.content:", chat_response.choices[0].message.content)
    # print("chat_response.choices[0].finish_reason:", chat_response.choices[0].finish_reason)
    print("chat_response.choices[0].message.model_extra.reasoning_content:", chat_response.choices[0].message.model_extra['reasoning_content'])
    # print("chat_response.usage.total_tokens:", chat_response.usage.total_tokens)
    # print("chat_response.usage.prompt_tokens:", chat_response.usage.prompt_tokens)
    # print("chat_response.usage.completion_tokens:", chat_response.usage.completion_tokens)
    # print("chat_response.id:", chat_response.id)
    # print("chat_response.model:", chat_response.model) 
    