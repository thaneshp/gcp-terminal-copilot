import os
import asyncio
import httpx
import logging
from typing import Optional
from contextlib import AsyncExitStack
from typing import Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from rich.console import Console

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("gcp-terminal-copilot")

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
            server_script_path: Path to the server script
        """
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        self.available_tools = tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def send_command(
        self, command: str, ollama_host: str, ollama_model: str
    ):
        """Send command to Ollama
        
        Args:
            command: Natural Language command passed in by user
            ollama_host: Host URL of Ollama
            ollama_model: Model used for translation
        """

        print(f"Processing: {command}")

        try:
            gcp_command = await self.translate_to_gcpmcp_command(
                command, ollama_host, ollama_model
            )

            if gcp_command != command: print(f"Translated to: {gcp_command}")
            gcp_command = gcp_command.strip().split()
            tool_name = gcp_command[0]
            tool_args = {}

            i = 1
            while i < len(gcp_command):
                if gcp_command[i].startswith("--"):
                    param_name = gcp_command[i][2:]
                    param_value = gcp_command[i + 1]
                    tool_args.update({param_name: param_value})
                    i += 2
            
            response = await self.session.call_tool(
                name=tool_name, arguments=tool_args
            )
            return response
        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            return {"error": f"Command execution failed: {str(e)}"}
    
    async def translate_to_gcpmcp_command(
        self, natural_language_query: str, ollama_host: str, ollama_model: str
    ) -> str:
        """Translate the command into a GCP MCP command

        Args:
            natural_language_query: Natural Language Query given by user
            ollama_host: Host URL of Ollama
            ollama_model: Model used for translation
        """
        
        command_details = {}
        for tool in self.available_tools:
            name = getattr(tool, "name", None)
            inputSchema = getattr(tool, "inputSchema", None)
            if name:
                properties = inputSchema.get("properties")
                param_info = []
                if properties:
                    for param_name, param_details in properties.items():
                        param_type = param_details.get("type")
                        param_description = param_details.get("description")

                        param_info.append({
                            "name": param_name,
                            "type": param_type,
                            "description": param_description
                        })

                    command_details[name] = param_info
                else:
                    command_details[name] = []

        command_list = []
        for command, params in command_details.items():
            command_list.append(f"- {command}")
            if params:
                for param in params:
                    command_list.append(
                        f"  --{param['name']} ({param['type']}): {param['description']}"
                    )

        command_list_str = "\n".join(command_list)

        system_prompt = f"""
            You are an gcloud CLI expert. Translate the user's 
            natural language query into the appropriate
            gcloud CLI command based on the available commands.
    
            Available commands:
            {command_list_str}
            
            Only return the suggested command from the available commands and nothing else. 
            Do not include any Markdown, formatting or backticks.
            """
        print(f"System prompt: {system_prompt}")
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
            logger.error(f"Failed to translate query: {str(e)}")
            return natural_language_query


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
            print(result)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
