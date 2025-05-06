import os
import logging
import asyncio

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Union,Optional

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
logging.info(f"anthropic_api_key: {anthropic_api_key}")


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
        logging.info(f"Calling LLM with messages:\n {messages}")
        response = self.client.messages.create(
            model=model_path,
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools
        )
        return response
    async def process_response(self, response, messages):
        """
        Process a response from the server and handle tool calls
        
        if need to call tools, append the tool calls to the messages list
        and call the tools, then append the result to the messages list,then call the llm again
        and return the final text
        if not need to call tools, just return the final text
        """
        
        
        # process response and handle tool calls
        if self.server_name == "OpenAI":
            logging.info(f"response: \n{message}")
            # warning whether the response finish reason is "tool_calls"
            logging.warning(f"response finish reason: {response.choices[0].finish_reason}")
            message = response.choices[0].message
           
            if message.tool_calls!=[]:
                logging.info(f"need to call tools")
                
                # or
                messages.append(message)
               
            
                # use response.choices[0].message.model_dump() transfer to dict
                # tool_calls = message.dump()['tool_calls']
                # messages.append(
                   # {"role":"assistant", "content": None ,"tool_calls": tool_calls}
                #)# whether content has impact on the result:message.content.strip()
                    
                # execute tool calls functions
                for tool_call in message.tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    
                    # Execute tool call
                    logging.info(f"Calling tool {tool_name} with args {tool_args}")

                    tool_result = await self.session.call_tool(tool_name, tool_args)

                    logging.info(f"Tool {tool_name} result: {tool_result}")

                    # Append tool result to the message
                    messages.append(
                        {   
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call.id
                        }
                    )
                response = self.call_llm(messages)

            return response.choices[0].message.model_dump()
            # return response.choices[0].content
        elif self.server_name == "Anthropic":
            pass
    

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        response = self.call_llm(messages)
        return self.process_response(response, messages)['content']

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