import os
import asyncio
import httpx
from typing import Optional
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

from rich.console import Console

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.console = Console()

    async def cleanup(self):
        if self.exit_stack:
            await self.exit_stack.aclose()
            print("Resources cleaned up")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # is_python = server_script_path.endswith('.py')
        # is_js = server_script_path.endswith('.js')
        # if not (is_python or is_js):
        #     raise ValueError("Server script must be a .py or .js file")

        # command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        self.available_tools = tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def send_command(
        self, command: str, ollama_host: str, ollama_model: str
    ) -> Union[Dict[str, Any], List[Any], str]:
    
        print(f"Processing: {command}")
    
        gcp_command = await self.translate_to_gcpmcp_command(
            command, ollama_host, ollama_model
        )
        if gcp_command != command:
            print(f"Translated to: {gcp_command}")
        
        result_metadata = {
            "original_command": command,
            "gcp_command": gcp_command,
        }

        response = await self.session.call_tool(
            gcp_command
        )

        print(response)
    
    async def translate_to_gcpmcp_command(self, natural_language_query: str, ollama_host: str, ollama_model: str) -> str:
        available_commands = []

        if self.available_tools:
            available_commands.extend(
                [
                    tool.name
                    for tool in self.available_tools
                    if hasattr(tool, "name")
                ]
            )

            available_commands = list(set(available_commands))
            command_list = "\n".join([f"- {cmd}" for cmd in available_commands])

            system_prompt = f"""
                You are an gcloud CLI expert. Translate the user's 
                natural language query into the appropriate
                gcloud CLI command based on the available commands.
        
                Available commands:
                {command_list}
                
                Only return the suggested command and nothing else. Do not include any Markdown, formatting or backticks.
                """
        
        try:
            print(f"Calling Ollama to translate '{natural_language_query}'")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": natural_language_query},
                        ],
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    gcp_command = result["message"]["content"].strip()
                    print(
                        f"Translated '{natural_language_query}' to '{gcp_command}'"
                    )
                    return gcp_command
                else:
                    print(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )
                    return natural_language_query
        except Exception as e:
            print(f"Failed to translate query: {str(e)}")
            return natural_language_query


async def main(): 
    load_dotenv()
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_model = os.getenv("OLLAMA_MODEL")

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
        await client.connect_to_server("/Users/thaneshp/Documents/git/gcp-mcp")

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
                print("Enter Azure CLI command (or 'exit' to quit)")
                print("Example: 'group list' or 'storage account list'")

            user_input = input("> ")

            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            if not user_input.strip():
                continue
            
            result = await client.send_command(user_input, ollama_host, ollama_model)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
