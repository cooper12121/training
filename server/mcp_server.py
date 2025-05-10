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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        

    

    async def initialize(self) -> None:
        """Initialize the server connection."""
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