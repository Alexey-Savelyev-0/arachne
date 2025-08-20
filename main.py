
"""
Welcome to arachne - a personalised financial advisor. 
For self: to run - please use uv run main.py "-m alpha_vantage_mcp.server" in this directory.
If this does not work, please run  uv run main.py "C:/CS/arachne/alpha-vantage-mcp/src/alpha_vantage_mcp/server.py" instead.

This program relies on the calling of an external LLM - this can either be done via an API call to a closed source model,
or to a locally hosted open source model. At this point in time, this model was developed with the usage of an open source model in mind. 
"""



# Using Redis/RabbitMQ or lighter options like Python's asyncio queues
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Callable
import json
from abc import ABC, abstractmethod
import asyncio
from typing import Optional
import requests
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import os


env_exists = load_dotenv()  
#if not(env_exists):
    #print("Warning: no environment variables are being set. Please ensure the appropriate API keys are set in your windows environment.")
# Not critical, can be removed.
env=os.environ.copy()
print(f"Debug: API key from environment: {os.environ.get('ALPHA_VANTAGE_API_KEY', 'NOT FOUND')}")

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen3"

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if server_script_path.startswith('-m '):
            # Module execution: -m module.name
            module_name = server_script_path[3:]  # Remove '-m ' prefix
            command = "python"
            args = ["-m", module_name]
        else:
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            args = [server_script_path]
        env=os.environ.copy()
        server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        try:
            await self.session.initialize()
        except Exception as e:
            print(f"Debug: Failed at: {e}")
            raise

            # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def call_ollama(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Call Ollama API with messages and optional tools"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        # Add tools if provided (for models that support function calling)
        if tools:
            # Convert MCP tool format to Ollama/OpenAI format
            ollama_tools = []
            for tool in tools:
                ollama_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                ollama_tools.append(ollama_tool)
            payload["tools"] = ollama_tools

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            raise

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""
        messages = [{"role": "user", "content": query}]

        # Get available tools from MCP server
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Add system message optimized for Qwen's tool usage
        system_message = {
            "role": "system", 
            "content": f"""You are a helpful financial assistant with access to Alpha Vantage stock market tools. 
            Available tools: {[tool['name'] for tool in available_tools]}
            
            When users ask about stocks, stock prices, company information, or financial data, use the appropriate tools.
            Always use the exact tool names and required parameters as specified."""
        }
        messages.insert(0, system_message)

        # Initial Ollama API call
        ollama_response = self.call_ollama(messages, available_tools)
        
        final_text = []
        
        # Check if Ollama wants to use tools
        message_content = ollama_response.get("message", {})
        
        if "tool_calls" in message_content:
            # Handle tool calls (if the model supports function calling)
            for tool_call in message_content["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                print(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Execute tool call via MCP
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Tool {tool_name} result: {result.content}]")
                
                # Add tool result to conversation
                messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "1"),
                    "content": str(result.content)
                })
                
                # Get final response from Ollama
                final_response = self.call_ollama(messages)
                final_text.append(final_response["message"]["content"])
        else:
            # No native tool calls, check if we can parse them from text
            response_text = message_content.get("content", "")
            final_text.append(response_text)

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