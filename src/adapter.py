from openai import OpenAI
import httpx


class OllamaAdapter:
    def __init__(self, host, model_name):
        self.model_name = model_name
        self.host = host

    async def query(self, messages):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                },
            )

            if response.status_code == 200:
                result = response.json()
                gcp_command = result["message"]["content"].strip()
                return gcp_command
            else:
                raise Exception(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
        return response


class ClaudeAPI:
    pass


class OpenAIAdapter:
    def __init__(self, api_key, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    async def query(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=0
        )
        return response.choices[0].message.content


class ModelAdapter:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def query(self, messages):
        return self.model_adapter.query(messages)
