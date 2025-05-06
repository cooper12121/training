from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

import vllm
import logging
import os
import openai
load_dotenv("/mnt/nlp/gaoqiang/project/training/server/.env")



# Set OpenAI's API key and API base to use vLLM's API server.
model_path = os.getenv("MODEl_PATH")
openai_api_key = os.getenv("API_KEY")
openai_api_base = os.getenv("API_BASE")

logging.basicConfig(level=logging.INFO)

logging.info(f"model_path: {model_path}")
logging.info(f"openai_api_key: {openai_api_key}")
logging.info(f"openai_api_base: {openai_api_base}")



client =
while True:
    query = input("Please enter your query: ")
    if query == "exit":
        break
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

    # chat_response = openai.ChatCompletion.create   no longer supported
    chat_response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        extra_body = {
            "top_k":50,
            "repetition_penalty":1.05,
            # "stop_token_ids":
            "temperature":0.5
        },
        # stream=False,
        
    )

    print("Chat response:", chat_response)

    # print("Chat response:", dir(chat_response))
    print("chat_response.choices[0].message.content:", chat_response.choices[0].message.content)
    # print("chat_response.choices[0].finish_reason:", chat_response.choices[0].finish_reason)
    print("chat_response.choices[0].message.model_extra.reasoning_content:", chat_response.choices[0].message.model_extra['reasoning_content'])
    # print("chat_response.usage.total_tokens:", chat_response.usage.total_tokens)
    # print("chat_response.usage.prompt_tokens:", chat_response.usage.prompt_tokens)
    # print("chat_response.usage.completion_tokens:", chat_response.usage.completion_tokens)
    # print("chat_response.id:", chat_response.id)
    # print("chat_response.model:", chat_response.model) 


import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Union

load_dotenv("/mnt/nlp/gaoqiang/project/training/server/.env")



# Set OpenAI's API key and API base to use vLLM's API server.
model_path = os.getenv("MODEl_PATH")
openai_api_key = os.getenv("API_KEY")
openai_api_base = os.getenv("API_BASE")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

logging.basicConfig(level=logging.INFO)

logging.info(f"model_path: {model_path}")
logging.info(f"openai_api_key: {openai_api_key}")
logging.info(f"openai_api_base: {openai_api_base}")


class MCPClient:
    def __init__(self,server_name:Union[str]="OpenAI"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools_list = None
        self.available_tools = None
        self.server_name = server_name
        if server_name == "OpenAI":
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        elif server_name == "Anthropic":
            self.client = Anthropic(
                api_key=anthropic_api_key,
            )
        else:
            raise ValueError("Server name must be either 'OpenAI' or 'Anthropic'")


    # methods will go here
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        self.tools_list = await self.session.list_tools()
        self.available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in self.tools_list.tools]
        logging.info(f"Available tools: {self.available_tools}")

    async def call_llm(self, messages):

        # call llm
        response = self.client.messages.create(
            model=model_path,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )
        
        
        # process response and handle tool calls
        final_text = []

        assistant_message_content = []
        if self.server_name == "OpenAI":
            message = response.choices[0].message
            if message.content is None and message.tool_calls==[]:
                # need to call tools, tool_calls=[], not sure whether content is None
                tool_name = 
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    assistant_message_content.append(content)
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message_content
                    })
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
       

        
        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())