from openai import OpenAI

class OpenAIAdapter:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def query(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

class ModelAdapter:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def query(self, messages):
        return self.model_adapter.query(messages)
