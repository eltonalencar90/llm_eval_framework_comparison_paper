import os
import requests

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


OLLAMA_URL = os.getenv('OLLAMA_URL')

PROMPT_TEMPLATES = {
    "qna": (
        "Considering the {section} and page(s) {pages} of the user manual of "
        "{market_name} {product_type} by {manufacture}, answer the following question:\n"
        "{question}"
    ),
    "translation": (
        "Translate the following {source_language} sentence into {target_language}:\n"
        "{source}"
    )
}

def generate_response_ollama(model_name, prompt):
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response_json = response.json()
    if "response" in response_json:
        return response_json["response"]
    else:
        raise ValueError(f"API Ollama: {response_json}")

def generate_response_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model= model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content