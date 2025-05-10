import os
import logging
import asyncio
import traceback
import json
import pdb
import json5
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv
from typing import Union,Optional,Any


from mcp_server import Server
logging.basicConfig(level=logging.INFO)

def load_config(cls, config_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(config_path, "r") as f:
            return json5.load(f)

class MCPClient:
    def __init__(self,servers:list[Server],server_api:Union[str]="OpenAI",env_path:str="./.env") -> None:

        # Initialize configuration with environment variables.
        self.load_env(env_path)

        # Initialize session and client objects
        self.servers =   servers     # list of server objects
        self.available_tools = []

        self.initialize_tools()

        self.server_api = server_api
        if server_api == "OpenAI":
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
            )
            pass
            
        elif server_api == "Anthropic":
            self.client = Anthropic(
                api_key=self.anthropic_api_key,
            )
        else:
            raise ValueError("Server name must be either 'OpenAI' or 'Anthropic'")

    @staticmethod
    def load_env(self,env_path:str) -> None:
        """Load environment variables from .env file."""
        load_dotenv(env_path)
        # Set OpenAI's API key and API base to use vLLM's API server.
        self.self.model_path = os.getenv("self.model_path")
        self.openai_api_key = os.getenv("API_KEY")
        self.openai_api_base = os.getenv("API_BASE")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        logging.info(f"self.model_path: {self.self.model_path}")
        logging.info(f"openai_api_key: {self.openai_api_key}")
        logging.info(f"openai_api_base: {self.openai_api_base}")
        logging.info(f"anthropic_api_key: {self.anthropic_api_key}")
    
    # methods will go here
    async def initialize_tools(self):
        """Initialize tools and resources for the server.
        """
        # List available tools
        for server in self.servers:
            tools = await server.list_tools()
            for tool in tools:
                # tools setting for openai api
                self.available_tools.append({
                    "type":"function",
                    "function":{
                        "name": tool.name,
                        "description": tool.description, 
                        "parameters": tool.inputSchema  
                    }}
                )
                # tools setting for anthropic
                # self.available_tools.append({
                #     "name": tool.name,    
                #     "description": tool.description,
                #     "input_schema": tool.inputSchema
                # })
        logging.info(f"Available tools: {self.available_tools}")

    async def call_llm(self, messages:list[dict])-> Union[ChatCompletion, Message]:

        # call llm
        logging.info(f"Calling LLM with messages:\n {messages}")
        if self.server_api == "OpenAI":
            response = self.client.chat.completions.create(
                model=self.model_path,
                max_tokens=1000,
                messages=messages,
                tools=self.available_tools,
                extra_body = {
                    "top_k":50,
                    "repetition_penalty":1.05,
                    # "stop_token_ids":
                    "temperature":0.5
                },
                tool_choice="auto"
            )
        elif self.server_api == "Anthropic":
            # For Anthropic, use the appropriate method to call the model
            response = self.client.messages.create(
                model=self.model_path,
                max_tokens=1000,
                messages=messages,
                tools=self.available_tools,
                
            )
        return response
    async def process_response(self, response:Union[Message,ChatCompletion], messages:list[dict]):
        """
        Process a response from the server and handle tool calls
        
        if need to call tools, append the tool calls to the messages list
        and call the tools, then append the result to the messages list,then call the llm again
        and return the final text
        if not need to call tools, just return the final text
        """
        
        
        # process response and handle tool calls
        if self.server_api == "OpenAI":
            logging.info(f"response: \n{response}")
            # warning whether the response finish reason is "tool_calls"
            logging.warning(f"response finish reason: {response.choices[0].finish_reason}")
            message = response.choices[0].message
           
            if message.tool_calls!=[]:
                logging.info(f"need to call tools")
                
                # or
                # messages.append(message)
               
            
                # use response.choices[0].message.model_dump() transfer to dict，but still need to transfer further. because the whole arguments args is a str 
                tool_calls = message.model_dump()['tool_calls']
                # for tool in tool_calls:
                #     tool['function']['arguments'] = json.loads(tool['function']['arguments'])
                messages.append(
                   {"role":"assistant", "content": message.content ,"tool_calls": tool_calls}
                )# whether content has impact on the result:message.content.strip()
                    
                # execute tool calls functions
                for tool_call in message.tool_calls:

                    logging.info(f"Tool type: {tool_call.type}")

                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments # str

                    # print(type(tool_args))
                    # pdb.set_trace()
                    
                    # Execute tool call
                    logging.info(f"Calling tool {tool_name} with args {tool_args}")

                    tool_result = await self.session.call_tool(tool_name, json.loads(tool_args))

                    logging.info(f"Tool: {tool_name} \nresult: {tool_result}")

                    # Append tool result to the message
                    messages.append(
                        {   
                            "role": "tool",
                            "content": tool_result.content[0].text,
                            "tool_call_id": tool_call.id
                        }
                    )
                response = await self.call_llm(messages)

            return response.choices[0].message.model_dump()
            # return response.choices[0].content
        elif self.server_api == "Anthropic":
            pass
    

    async def process_query(self, query: str,messages:list[dict]=[]) -> str:
        """Process a query using Claude and available tools"""
        if messages == []:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
        else:
            messages.append({"role": "user", "content": query})

        response = await self.call_llm(messages)
        final_response = await self.process_response(response, messages)
        messages.append(final_response)
        # return final_response['content']
        return messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        messages = []
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                
                if query.startswith("new"):
                    messages = []
                    query = query[4:].strip()
                    print(f"New conversation started. Query: {query}")

                messages = await self.process_query(query,messages)
                print("\n" + messages[-1]['content'])

            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    
    # Load mcp servers 
    config_path = "/mnt/nlp/gaoqiang/project/training/server/mcp_server_config.json5"
    server_config = load_config("./mcp_server_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    # Initialize client
    client = MCPClient(servers=servers, server_api="OpenAI")
    try:

        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())

    # uv run mcp_vllm_openai_client.py  /path/server.py