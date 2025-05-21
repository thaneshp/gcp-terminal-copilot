import logging
from typing import Optional
from contextlib import AsyncExitStack
from typing import Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
import json
from adapter import ModelAdapter, OllamaAdapter
from utilities import process_template

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("gcp-terminal-copilot")


class MCPClient:
    def __init__(self, system_prompt_template: str):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.console = Console()
        self.system_prompt_template = system_prompt_template

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
            command="npx", args=["-y", server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        self.available_tools = tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def send_command(self, command: str, model: ModelAdapter):
        """Send command to Ollama

        Args:
            command: Natural Language command passed in by user
            ollama_host: Host URL of Ollama
            ollama_model: Model used for translation
        """

        print(f"Processing: {command}")

        try:
            command_list = self._get_command_list()
            print(command_list)
            gcp_command = await self.translate_to_gcpmcp_command(
                command, command_list, model
            )

            if gcp_command != command:
                print(f"Translated to: {gcp_command}")
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

            response = await self.session.call_tool(name=tool_name, arguments=tool_args)
            return response
        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            return {"error": f"Command execution failed: {str(e)}"}

    async def translate_to_gcpmcp_command(
        self,
        natural_language_query: str,
        available_commands: list,
        model: ModelAdapter,
    ) -> str:
        """Translate the command into a GCP MCP command

        Args:
            natural_language_query: Natural Language Query given by user
            available_commands: List of available commands
            model: Model used for translation
        """
        command_list = "\n".join(available_commands)

        system_prompt = process_template(
            self.system_prompt_template, {"command_list": command_list}
        )

        try:
            print(f"Calling LLM to translate '{natural_language_query}'")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": natural_language_query},
            ]

            response = await model.query(messages)
            return response
        except Exception as e:
            logger.error(f"Failed to translate query: {str(e)}")
            return natural_language_query

    @staticmethod
    def print_text_content(response):
        content = response.content[0]

        if hasattr(content, "text"):
            data = content.text
            try:
                parsed = json.loads(data)
                print(json.dumps(parsed, indent=2))
            except:
                print(data)

    def _get_command_list(self):
        """Get the list of available commands and their parameters"""
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

                        param_info.append(
                            {
                                "name": param_name,
                                "type": param_type,
                                "description": param_description,
                            }
                        )

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

        return command_list
