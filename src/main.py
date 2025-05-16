from dotenv import load_dotenv
from client import MCPClient
import httpx
import os
import asyncio

async def main(): 
    load_dotenv()
    server_script_path = os.getenv("SERVER_SCRIPT_PATH")
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_model = os.getenv("OLLAMA_MODEL")

    if not server_script_path:
        print("ERROR: SERVER_SCRIPT_PATH environment variable not set")
        return

    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_host}/api/version")
            if response.status_code == 200:
                ollama_available = True
    except Exception:
        pass
    
    client = MCPClient()

    try:
        await client.connect_to_server(server_script_path)

        if ollama_available:
            print("\n✓ Ollama is available for natural language processing")
            print(f"   Using model: {ollama_model}")
        
        while True:
            print("\n" + "=" * 50)
            if ollama_available:
                print(
                    "Enter your Google Cloud request in natural language (or 'exit' to quit)"
                )
                print(
                    "Example: 'List all GCP projects I have access to' or 'What's my current billing status?'"
                )
            else:
                print("Enter gcloud CLI command (or 'exit' to quit)")
                print("Example: 'gcloud projects list' or 'gcloud storage buckets list'")

            user_input = input("> ")

            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            if not user_input.strip():
                continue
            
            result = await client.send_command(user_input, ollama_host, ollama_model)
            
            print("\n🔹 Response:")
            MCPClient.print_text_content(result)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
