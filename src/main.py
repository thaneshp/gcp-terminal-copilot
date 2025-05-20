from dotenv import load_dotenv
from client import MCPClient
import httpx
import os
import asyncio
from adapter import ModelAdapter, OllamaAdapter, OpenAIAdapter


async def main():
    load_dotenv()
    server_script_path = os.getenv("SERVER_SCRIPT_PATH")
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_model = os.getenv("OLLAMA_MODEL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    system_prompt_template = os.getenv("SYSTEM_PROMPT_TEMPLATE")
    model_provider = os.getenv("MODEL_PROVIDER", "ollama").lower()

    if not server_script_path:
        print("ERROR: SERVER_SCRIPT_PATH environment variable not set")
        return

    ollama_available = False
    if model_provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ollama_host}/api/version")
                if response.status_code == 200:
                    ollama_available = True
        except Exception:
            pass

    client = MCPClient(system_prompt_template)

    try:
        await client.connect_to_server(server_script_path)

        if ollama_available:
            print("\n✓ Ollama is available for natural language processing")
            print(f"   Using model: {ollama_model}")
            model = ModelAdapter(OllamaAdapter(ollama_host, ollama_model))
        elif model_provider == "openai":
            print("✓ Using OpenAI for natural language processing")
            print(f"   Model: {openai_model} or as configured")
            model = ModelAdapter(OpenAIAdapter(openai_api_key, openai_model))

        while True:
            print("\n" + "=" * 50)
            print(
                "Enter your Google Cloud request in natural language (or 'exit' to quit)"
            )
            print(
                "Example: 'List all GCP projects I have access to' or 'What's my current billing status?'"
            )

            user_input = input("> ")

            if user_input.lower() in ("exit", "quit", "q"):
                break

            if not user_input.strip():
                continue

            result = await client.send_command(user_input, model)

            print("\n🔹 Response:")
            MCPClient.print_text_content(result)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
