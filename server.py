"""
We'll make it give us financial advice
Instead of attempting to predict stock prices or telling us where to invest,
instead we'll aim for a semantic approach - an analysis of articles/public sentiment
about a set of given companies.
For now just analyze trending companies and tech that you might want to be aware of.


1. Stock Data API
2. Knowledge base (RAG)
3. News Info (API)

# First this agent should tell me what the current hot news are in regards to finance.

"""




from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def getData(company: str) -> str:
    return ""


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

