import os
from adapter import OllamaAdapter, OpenAIAdapter

SYSTEM_PROMPTS = ["01.jinja", "02.jinja"]

MODEL_ADAPTERS = [
    (OllamaAdapter, {"host": "http://localhost:11434", "model_name": "gemma3:4b"}),
    (
        OpenAIAdapter,
        {"api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "gpt-4-turbo"},
    ),
]
