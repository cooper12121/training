from agents import Agent, Runner
from agents.mcp.server import MCPServerStdio
import os
import asyncio

# os.environ["OPENAI_API_KEY"] = ""

# agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
# print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
async def get_samples_dir():
    async with MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/qianggao/project/intern/training/agent"],
        }
    ) as server:
        tools = await server.list_tools()
        print(tools)
        return tools

# Wrap the call in an async function and run it
async def main():
    response = await get_samples_dir()
    print(response)

# Run the main function using asyncio
asyncio.run(main())