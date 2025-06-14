import asyncio
import json
import logging
import os
import shutil
import json5
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# from mcp.server.lowlevel.server import Server
from mcp.server import FastMCP,Server

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class MCPServer(FastMCP):
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(self,name)
        # self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        

    

    async def initialize(self) -> None:
        """Initialize the server connection."""
        print(self.config)
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            
        
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")
    
    async def list_tools(self) -> list[Any]:
        tools_response = await self.session.list_tools()
        return tools_response.tools
    

#     async def list_tools(self) -> list[Any]:
#         """List available tools from the server.

#         Returns:
#             A list of available tools.

#         Raises:
#             RuntimeError: If the server is not initialized.
#         """
#         if not self.session:
#             raise RuntimeError(f"Server {self.name} not initialized")

#         tools_response = await self.session.list_tools()
#         tools = []

#         for item in tools_response:
#             if isinstance(item, tuple) and item[0] == "tools":
#                 tools.extend(
#                     Tool(tool.name, tool.description, tool.inputSchema)
#                     for tool in item[1]
#                 )

#         return tools
#     async def execute_tool(
#         self,
#         tool_name: str,
#         arguments: dict[str, Any],
#         retries: int = 2,
#         delay: float = 1.0,
#     ) -> Any:
#         """Execute a tool with retry mechanism.

#         Args:
#             tool_name: Name of the tool to execute.
#             arguments: Tool arguments.
#             retries: Number of retry attempts.
#             delay: Delay between retries in seconds.

#         Returns:
#             Tool execution result.

#         Raises:
#             RuntimeError: If server is not initialized.
#             Exception: If tool execution fails after all retries.
#         """
#         if not self.session:
#             raise RuntimeError(f"Server {self.name} not initialized")

#         attempt = 0
#         while attempt < retries:
#             try:
#                 logging.info(f"Executing {tool_name}...")
#                 result = await self.session.call_tool(tool_name, arguments)

#                 return result

#             except Exception as e:
#                 attempt += 1
#                 logging.warning(
#                     f"Error executing tool: {e}. Attempt {attempt} of {retries}."
#                 )
#                 if attempt < retries:
#                     logging.info(f"Retrying in {delay} seconds...")
#                     await asyncio.sleep(delay)
#                 else:
#                     logging.error("Max retries reached. Failing.")
#                     raise
# class Tool:
#     """Represents a tool with its properties and formatting."""

#     def __init__(
#         self, name: str, description: str, input_schema: dict[str, Any]
#     ) -> None:
#         self.name: str = name
#         self.description: str = description
#         self.input_schema: dict[str, Any] = input_schema

#     def format_for_llm(self) -> str:
#         """Format tool information for LLM.

#         Returns:
#             A formatted string describing the tool.
#         """
#         args_desc = []
#         if "properties" in self.input_schema:
#             for param_name, param_info in self.input_schema["properties"].items():
#                 arg_desc = (
#                     f"- {param_name}: {param_info.get('description', 'No description')}"
#                 )
#                 if param_name in self.input_schema.get("required", []):
#                     arg_desc += " (required)"
#                 args_desc.append(arg_desc)

#         return f"""
# Tool: {self.name}
# Description: {self.description}
# Arguments:
# {chr(10).join(args_desc)}
# """



# async def connect_to_server(self, server_script_path: str):
#     """Connect to an MCP server

#     Args:
#         server_script_path: Path to the server script (.py or .js)
#     """
#     is_python = server_script_path.endswith('.py')
#     is_js = server_script_path.endswith('.js')
#     if not (is_python or is_js):
#         raise ValueError("Server script must be a .py or .js file")

#     command = "python" if is_python else "node"
#     server_params = StdioServerParameters(
#         command=command,
#         args=[server_script_path],
#         env=None
#     )
    

#     stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#     self.stdio, self.write = stdio_transport
#     self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

#     await self.session.initialize()